"""Evaluate CoTracker joint tracking, TFA, and motion disentanglement on frozen 73 videos.

The experiment is deliberately isolated from the production ``innovation_repro`` files.
Each stage is retained only when frozen-truth RR R2 improves without increasing RR MAE.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import paper_repro_rr as rr
from evaluate_paper73_adaptive_roi import config_from_baseline


POINT_NAMES = (
    "left_center",
    "left_axis_pos",
    "left_axis_neg",
    "left_normal_pos",
    "left_normal_neg",
    "right_center",
    "right_axis_pos",
    "right_axis_neg",
    "right_normal_pos",
    "right_normal_neg",
)
SIDE_POINT_INDICES = {"left": np.arange(0, 5), "right": np.arange(5, 10)}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root)
    parser.add_argument(
        "--baseline-summary", type=Path, default=root / "innovation_repro_summary.csv"
    )
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=repo
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
    )
    parser.add_argument(
        "--cotracker-repo",
        type=Path,
        default=repo / ".codex_tmp" / "torch" / "hub" / "facebookresearch_co-tracker_main",
    )
    parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=repo / ".codex_tmp" / "torch" / "hub" / "checkpoints" / "scaled_offline.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--run-name", default="innovation_cotracker_tfa")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--window-overlap", type=int, default=12)
    parser.add_argument("--anchor-search-frames", type=int, default=8)
    parser.add_argument("--roi-radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    return parser.parse_args()


def numeric(table: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(
        table.get(column, pd.Series(np.nan, index=table.index)), errors="coerce"
    ).to_numpy(dtype=float)


def valid_pair(table: pd.DataFrame, minimum_confidence: float = 0.5) -> np.ndarray:
    coords = np.column_stack(
        [
            numeric(table, "left_x"),
            numeric(table, "left_y"),
            numeric(table, "right_x"),
            numeric(table, "right_y"),
        ]
    )
    left_conf = numeric(table, "left_conf")
    right_conf = numeric(table, "right_conf")
    return (
        np.isfinite(coords).all(axis=1)
        & (left_conf >= minimum_confidence)
        & (right_conf >= minimum_confidence)
    )


def finite_plausible_pair(table: pd.DataFrame) -> np.ndarray:
    coords = np.column_stack(
        [
            numeric(table, "left_x"),
            numeric(table, "left_y"),
            numeric(table, "right_x"),
            numeric(table, "right_y"),
        ]
    )
    spacing = np.linalg.norm(coords[:, 2:] - coords[:, :2], axis=1)
    return np.isfinite(coords).all(axis=1) & (spacing >= 20.0) & (spacing <= 500.0)


def choose_reference_pair(table: pd.DataFrame) -> tuple[int, np.ndarray, np.ndarray]:
    valid = valid_pair(table)
    if not valid.any():
        # Some legacy rows retain a geometrically stable second keypoint whose
        # YOLO confidence is very low. It may initialize tracking, but is never
        # relabeled as a direct high-confidence detection.
        valid = finite_plausible_pair(table)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError("No frame contains a valid bilateral nostril pair")
    pairs = np.column_stack(
        [
            numeric(table, "left_x"),
            numeric(table, "left_y"),
            numeric(table, "right_x"),
            numeric(table, "right_y"),
        ]
    )[indices]
    centers = (pairs[:, :2] + pairs[:, 2:]) / 2.0
    vectors = pairs[:, 2:] - pairs[:, :2]
    spacing = np.linalg.norm(vectors, axis=1)
    angle = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    features = np.column_stack(
        [centers[:, 0], centers[:, 1], spacing, angle * np.nanmedian(spacing)]
    )
    scale = np.nanmedian(np.abs(features - np.nanmedian(features, axis=0)), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    score = np.sum(np.abs(features - np.nanmedian(features, axis=0)) / scale, axis=1)
    reference_index = int(indices[int(np.argmin(score))])
    row = table.iloc[reference_index]
    return (
        reference_index,
        np.asarray([float(row.left_x), float(row.left_y)], dtype=float),
        np.asarray([float(row.right_x), float(row.right_y)], dtype=float),
    )


def face_constellation(left: np.ndarray, right: np.ndarray, ratio: float = 0.045) -> np.ndarray:
    vector = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    spacing = float(np.linalg.norm(vector))
    if not math.isfinite(spacing) or spacing < 1.0:
        raise ValueError("Invalid inter-nostril spacing")
    axis = vector / spacing
    normal = np.asarray([-axis[1], axis[0]])
    offset = float(np.clip(ratio * spacing, 4.0, 10.0))
    points: list[np.ndarray] = []
    for center in (np.asarray(left, dtype=float), np.asarray(right, dtype=float)):
        points.extend(
            [
                center,
                center + offset * axis,
                center - offset * axis,
                center + offset * normal,
                center - offset * normal,
            ]
        )
    return np.asarray(points, dtype=np.float32)


def chunk_ranges(length: int, window_length: int, overlap: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    window_length = max(2, int(window_length))
    overlap = int(np.clip(overlap, 0, window_length - 1))
    stride = window_length - overlap
    starts = list(range(0, length, stride))
    if starts and starts[-1] + window_length < length:
        starts.append(length - window_length)
    ranges: list[tuple[int, int]] = []
    for start in starts:
        end = min(length, start + window_length)
        if ranges and start >= ranges[-1][1]:
            start = max(0, ranges[-1][1] - overlap)
        item = (int(start), int(end))
        if item not in ranges:
            ranges.append(item)
        if end == length:
            break
    return ranges


def choose_chunk_anchor(table: pd.DataFrame, start: int, end: int, search_frames: int) -> int:
    search_end = min(end, start + max(1, int(search_frames)))
    candidates = np.flatnonzero(valid_pair(table.iloc[start:search_end])) + start
    if len(candidates):
        return int(candidates[0])
    candidates = np.flatnonzero(valid_pair(table.iloc[start:end])) + start
    if len(candidates):
        return int(candidates[0])
    weak = np.flatnonzero(finite_plausible_pair(table.iloc[start:search_end])) + start
    if len(weak):
        return int(weak[0])
    weak = np.flatnonzero(finite_plausible_pair(table.iloc[start:end])) + start
    if len(weak):
        return int(weak[0])
    raise ValueError(f"No bilateral anchor in frames {start}:{end}")


def load_cotracker(repo: Path, checkpoint: Path, device: str):
    import torch

    if not repo.exists():
        raise FileNotFoundError(f"CoTracker repository not found: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"CoTracker checkpoint not found: {checkpoint}")
    model = torch.hub.load(str(repo), "cotracker3_offline", source="local", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state)
    return model.eval().to(device)


def load_resized_video(
    paths: list[Path], output_height: int, output_width: int, device: str
):
    import torch

    frames = []
    original_shape: tuple[int, int] | None = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read frame: {path}")
        height, width = image.shape[:2]
        if original_shape is None:
            original_shape = (height, width)
        elif original_shape != (height, width):
            raise ValueError(f"Frame dimensions changed inside {path.parent}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (output_width, output_height), interpolation=cv2.INTER_AREA))
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)[None].float().to(device)
    assert original_shape is not None
    return tensor, original_shape


def joint_cotracker_tracks(
    table: pd.DataFrame,
    frame_paths: list[Path],
    model,
    device: str,
    window_length: int,
    overlap: int,
    anchor_search_frames: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    length = len(table)
    point_sum = np.zeros((length, len(POINT_NAMES), 2), dtype=float)
    point_weight = np.zeros((length, len(POINT_NAMES)), dtype=float)
    point_votes = np.zeros((length, len(POINT_NAMES)), dtype=int)
    anchor_count = np.zeros(length, dtype=int)
    interp_h, interp_w = map(int, model.interp_shape)

    for start, end in chunk_ranges(length, window_length, overlap):
        try:
            anchor = choose_chunk_anchor(table, start, end, anchor_search_frames)
            row = table.iloc[anchor]
            anchor_points = face_constellation(
                np.asarray([row.left_x, row.left_y], dtype=float),
                np.asarray([row.right_x, row.right_y], dtype=float),
            )
        except ValueError:
            # A detector-free tail is initialized from the preceding chunk's
            # overlap. This is a genuine temporal continuation, not a fixed ROI.
            overlap_end = min(end, start + max(1, int(overlap)))
            candidates = [
                index
                for index in range(start, overlap_end)
                if int(np.sum(point_weight[index] > 0)) == len(POINT_NAMES)
            ]
            if not candidates:
                raise
            anchor = max(candidates, key=lambda index: float(np.sum(point_weight[index])))
            anchor_points = (
                point_sum[anchor] / point_weight[anchor, :, None]
            ).astype(np.float32)
        video, (height, width) = load_resized_video(
            frame_paths[start:end], interp_h, interp_w, device
        )
        resized_points = anchor_points.copy()
        resized_points[:, 0] *= (interp_w - 1) / (width - 1)
        resized_points[:, 1] *= (interp_h - 1) / (height - 1)
        query_time = anchor - start
        query = np.column_stack(
            [np.full(len(resized_points), query_time, dtype=np.float32), resized_points]
        )
        queries = torch.from_numpy(query)[None].to(device)
        with torch.inference_mode():
            tracks, visibility = model(
                video,
                queries=queries,
                backward_tracking=bool(query_time > 0),
            )
        tracks_np = tracks[0].detach().cpu().numpy()
        visibility_np = visibility[0].detach().cpu().numpy().astype(bool)
        tracks_np[:, :, 0] *= (width - 1) / (interp_w - 1)
        tracks_np[:, :, 1] *= (height - 1) / (interp_h - 1)
        del video, queries, tracks, visibility
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        for local_index, global_index in enumerate(range(start, end)):
            edge_distance = min(local_index, end - start - 1 - local_index)
            edge_weight = 0.35 + 0.65 * min(1.0, max(0, edge_distance) / max(1, overlap))
            anchor_weight = math.exp(-abs(global_index - anchor) / max(1.0, window_length))
            weight = edge_weight * (0.75 + 0.25 * anchor_weight)
            for point_index in range(len(POINT_NAMES)):
                if not visibility_np[local_index, point_index]:
                    continue
                point = tracks_np[local_index, point_index]
                if not np.isfinite(point).all():
                    continue
                point_sum[global_index, point_index] += weight * point
                point_weight[global_index, point_index] += weight
                point_votes[global_index, point_index] += 1
            anchor_count[global_index] += int(global_index == anchor)

    points = np.full_like(point_sum, np.nan)
    available = point_weight > 0
    points[available] = point_sum[available] / point_weight[available, None]
    detector = np.column_stack(
        [
            numeric(table, "left_x"),
            numeric(table, "left_y"),
            numeric(table, "right_x"),
            numeric(table, "right_y"),
        ]
    ).reshape(length, 2, 2)
    detector_valid = np.column_stack(
        [
            np.isfinite(detector[:, 0]).all(axis=1) & (numeric(table, "left_conf") >= 0.5),
            np.isfinite(detector[:, 1]).all(axis=1) & (numeric(table, "right_conf") >= 0.5),
        ]
    )
    fused_centers = np.full((length, 2, 2), np.nan, dtype=float)
    rejected = np.zeros((length, 2), dtype=bool)
    visible_fraction = np.zeros((length, 2), dtype=float)

    for frame_index in range(length):
        spacing = float(np.linalg.norm(detector[frame_index, 1] - detector[frame_index, 0]))
        gate = max(35.0, 0.35 * spacing) if math.isfinite(spacing) else 45.0
        for side_index, side in enumerate(("left", "right")):
            indices = SIDE_POINT_INDICES[side]
            valid_points = np.isfinite(points[frame_index, indices]).all(axis=1)
            visible_fraction[frame_index, side_index] = float(valid_points.mean())
            tracker_center = (
                np.median(points[frame_index, indices][valid_points], axis=0)
                if int(valid_points.sum()) >= 2
                else np.asarray([np.nan, np.nan])
            )
            direct = detector[frame_index, side_index]
            direct_ok = detector_valid[frame_index, side_index] and np.isfinite(direct).all()
            tracker_ok = np.isfinite(tracker_center).all()
            if direct_ok and tracker_ok:
                displacement = float(np.linalg.norm(tracker_center - direct))
                if displacement <= gate:
                    fused_centers[frame_index, side_index] = 0.8 * tracker_center + 0.2 * direct
                else:
                    fused_centers[frame_index, side_index] = direct
                    rejected[frame_index, side_index] = True
            elif tracker_ok:
                fused_centers[frame_index, side_index] = tracker_center
            elif direct_ok:
                fused_centers[frame_index, side_index] = direct

    rows: list[dict[str, object]] = []
    for frame_index, frame_name in enumerate(table["frame_name"].astype(str)):
        output: dict[str, object] = {
            "frame_name": frame_name,
            "left_x": fused_centers[frame_index, 0, 0],
            "left_y": fused_centers[frame_index, 0, 1],
            "right_x": fused_centers[frame_index, 1, 0],
            "right_y": fused_centers[frame_index, 1, 1],
            "left_visible_fraction": visible_fraction[frame_index, 0],
            "right_visible_fraction": visible_fraction[frame_index, 1],
            "left_tracker_rejected": rejected[frame_index, 0],
            "right_tracker_rejected": rejected[frame_index, 1],
            "anchor_count": anchor_count[frame_index],
        }
        for point_index, name in enumerate(POINT_NAMES):
            output[f"{name}_x"] = points[frame_index, point_index, 0]
            output[f"{name}_y"] = points[frame_index, point_index, 1]
            output[f"{name}_votes"] = int(point_votes[frame_index, point_index])
        rows.append(output)
    return pd.DataFrame(rows), points, point_votes > 0, fused_centers


def pair_similarity_matrix(
    left: np.ndarray, right: np.ndarray, reference_left: np.ndarray, reference_right: np.ndarray
) -> np.ndarray | None:
    source_vector = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    target_vector = np.asarray(reference_right, dtype=float) - np.asarray(reference_left, dtype=float)
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1.0 or target_length < 1.0:
        return None
    scale = target_length / source_length
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    cosine, sine = math.cos(angle), math.sin(angle)
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    source_center = (np.asarray(left, dtype=float) + np.asarray(right, dtype=float)) / 2.0
    target_center = (
        np.asarray(reference_left, dtype=float) + np.asarray(reference_right, dtype=float)
    ) / 2.0
    translation = target_center - linear @ source_center
    return np.column_stack([linear, translation]).astype(np.float32)


def robust_tfa_matrix(
    tracked_points: np.ndarray,
    point_visible: np.ndarray,
    canonical_points: np.ndarray,
    centers: np.ndarray,
    reference_left: np.ndarray,
    reference_right: np.ndarray,
) -> tuple[np.ndarray | None, str, float]:
    valid = point_visible & np.isfinite(tracked_points).all(axis=1)
    matrix: np.ndarray | None = None
    source = "pair_fallback"
    residual = math.nan
    if int(valid.sum()) >= 4:
        estimate, inliers = cv2.estimateAffinePartial2D(
            tracked_points[valid].astype(np.float32),
            canonical_points[valid].astype(np.float32),
            method=cv2.LMEDS,
        )
        if estimate is not None:
            linear = estimate[:, :2]
            scale = math.sqrt(max(0.0, float(abs(np.linalg.det(linear)))))
            transformed = cv2.transform(
                tracked_points[valid][None].astype(np.float32), estimate.astype(np.float32)
            )[0]
            residual = float(np.median(np.linalg.norm(transformed - canonical_points[valid], axis=1)))
            if 0.65 <= scale <= 1.55 and residual <= 10.0:
                matrix = estimate.astype(np.float32)
                source = "joint_points"
    if matrix is None and np.isfinite(centers).all():
        matrix = pair_similarity_matrix(
            centers[0], centers[1], reference_left, reference_right
        )
    return matrix, source, residual


def circle_pixels(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray | None:
    if image is None or not np.isfinite([x, y]).all():
        return None
    height, width = image.shape[:2]
    center_x, center_y = int(round(x)), int(round(y))
    if center_x < 0 or center_y < 0 or center_x >= width or center_y >= height:
        return None
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    pixels = image[y0:y1, x0:x1][mask]
    return pixels.reshape(-1, 3) if len(pixels) else None


def predict_pixel_tasks(
    model, tasks: list[np.ndarray | None], min_temp: float, batch_pixels: int = 150_000
) -> list[float]:
    result = [math.nan] * len(tasks)
    valid = [(index, pixels) for index, pixels in enumerate(tasks) if pixels is not None and len(pixels)]
    for batch_start in range(0, len(valid), 64):
        batch = valid[batch_start : batch_start + 64]
        offset = 0
        while offset < len(batch):
            selected: list[tuple[int, np.ndarray]] = []
            pixels_total = 0
            while offset < len(batch):
                index, pixels = batch[offset]
                if selected and pixels_total + len(pixels) > batch_pixels:
                    break
                selected.append((index, pixels))
                pixels_total += len(pixels)
                offset += 1
            merged = np.concatenate([pixels for _, pixels in selected], axis=0)
            predictions = np.asarray(model.predict(merged.reshape(-1, 3)), dtype=float)
            cursor = 0
            for index, pixels in selected:
                values = predictions[cursor : cursor + len(pixels)]
                cursor += len(pixels)
                values = values[values > min_temp]
                result[index] = float(values.mean()) if len(values) else math.nan
    return result


def extract_candidate_temperatures(
    source: pd.DataFrame,
    frame_paths: list[Path],
    tracks: pd.DataFrame,
    tracked_points: np.ndarray,
    point_visible: np.ndarray,
    centers: np.ndarray,
    temp_model,
    roi_radius: int,
    min_temp: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, reference_left, reference_right = choose_reference_pair(source)
    canonical_points = face_constellation(reference_left, reference_right)
    radius_column = (
        "roi_radius"
        if "roi_radius" in source.columns
        else "adaptive_roi_radius"
        if "adaptive_roi_radius" in source.columns
        else None
    )
    if radius_column is None:
        radius_values = np.full(len(source), int(roi_radius), dtype=int)
    else:
        radius_values = (
            pd.to_numeric(source[radius_column], errors="coerce")
            .fillna(int(roi_radius))
            .round()
            .clip(lower=1)
            .to_numpy(dtype=int)
        )
    tracker_tasks: list[np.ndarray | None] = []
    tfa_tasks: list[np.ndarray | None] = []
    diagnostics: list[dict[str, object]] = []

    for index, path in enumerate(frame_paths):
        frame_radius = int(radius_values[index])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            tracker_tasks.extend([None, None])
            tfa_tasks.extend([None, None])
            diagnostics.append({"registration_applied": False, "registration_source": "read_failed"})
            continue
        tracker_tasks.extend(
            [
                circle_pixels(image, *centers[index, 0], frame_radius),
                circle_pixels(image, *centers[index, 1], frame_radius),
            ]
        )
        matrix, matrix_source, residual = robust_tfa_matrix(
            tracked_points[index],
            point_visible[index],
            canonical_points,
            centers[index],
            reference_left,
            reference_right,
        )
        if matrix is None:
            tfa_tasks.extend([None, None])
            diagnostics.append(
                {
                    "registration_applied": False,
                    "registration_source": "unavailable",
                    "registration_residual_px": residual,
                }
            )
            continue
        for reference in (reference_left, reference_right):
            patch_matrix = matrix.copy()
            patch_matrix[0, 2] += frame_radius - float(reference[0])
            patch_matrix[1, 2] += frame_radius - float(reference[1])
            patch = cv2.warpAffine(
                image,
                patch_matrix,
                (2 * frame_radius + 1, 2 * frame_radius + 1),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            tfa_tasks.append(
                circle_pixels(patch, frame_radius, frame_radius, frame_radius)
            )
        diagnostics.append(
            {
                "registration_applied": True,
                "registration_source": matrix_source,
                "registration_residual_px": residual,
            }
        )

    tracker_values = predict_pixel_tasks(temp_model, tracker_tasks, min_temp)
    tfa_values = predict_pixel_tasks(temp_model, tfa_tasks, min_temp)
    tracker_table = source.copy()
    tfa_table = source.copy()
    diagnostic_table = pd.DataFrame(diagnostics)
    for table, values, source_name in (
        (tracker_table, tracker_values, "cotracker"),
        (tfa_table, tfa_values, "cotracker_tfa"),
    ):
        table["left_temp"] = values[0::2]
        table["right_temp"] = values[1::2]
        table["left_x"] = centers[:, 0, 0]
        table["left_y"] = centers[:, 0, 1]
        table["right_x"] = centers[:, 1, 0]
        table["right_y"] = centers[:, 1, 1]
        table["nostril_spacing"] = np.linalg.norm(centers[:, 1] - centers[:, 0], axis=1)
        table["roi_radius"] = radius_values
        table["left_conf"] = tracks["left_visible_fraction"].to_numpy(dtype=float)
        table["right_conf"] = tracks["right_visible_fraction"].to_numpy(dtype=float)
        for side in ("left", "right"):
            original_source = source.get(
                f"{side}_source", pd.Series("", index=source.index)
            ).astype(str)
            original_confidence = pd.to_numeric(
                source.get(f"{side}_conf", pd.Series(np.nan, index=source.index)),
                errors="coerce",
            )
            direct = original_source.eq("detected") & original_confidence.ge(0.5)
            table[f"{side}_source"] = np.where(
                table[f"{side}_temp"].isna(),
                "missing",
                np.where(direct, "detected", "tracked"),
            )
        table["status"] = np.where(
            table["left_temp"].notna() | table["right_temp"].notna(), "ok", "no_valid_nostril"
        )
        table["temperature_method"] = source_name
    tfa_table["registration_applied"] = diagnostic_table["registration_applied"]
    tfa_table["registration_source"] = diagnostic_table["registration_source"]
    tfa_table["registration_residual_px"] = diagnostic_table.get("registration_residual_px")
    return tracker_table, tfa_table, diagnostic_table


def fill_numeric(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    if series.notna().sum() < 2:
        return np.full(len(series), np.nan, dtype=float)
    return series.interpolate(limit_direction="both").to_numpy(dtype=float)


def safe_bandpass(values: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = fill_numeric(np.asarray(values, dtype=float))
    if len(values) < 18 or not np.isfinite(values).all():
        return np.zeros(len(values), dtype=float)
    sos = butter(2, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return np.zeros(len(values), dtype=float)


def cross_fitted_ridge_prediction(features: np.ndarray, target: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    length = len(target)
    prediction = np.zeros(length, dtype=float)
    folds = np.array_split(np.arange(length), min(5, max(2, length // 20)))
    identity = np.eye(features.shape[1], dtype=float)
    for test in folds:
        train_mask = np.ones(length, dtype=bool)
        train_mask[test] = False
        train = np.flatnonzero(train_mask)
        if len(train) <= features.shape[1]:
            continue
        beta = np.linalg.solve(
            features[train].T @ features[train] + alpha * identity,
            features[train].T @ target[train],
        )
        prediction[test] = features[test] @ beta
    return prediction


def disentangle_motion_component(
    table: pd.DataFrame,
    fps: float = 8.7,
    low_hz: float = 0.55,
    high_hz: float = 1.7,
    minimum_explained: float = 0.15,
    maximum_gain: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = table.copy()
    left = np.column_stack([numeric(table, "left_x"), numeric(table, "left_y")])
    right = np.column_stack([numeric(table, "right_x"), numeric(table, "right_y")])
    center = (left + right) / 2.0
    vector = right - left
    spacing = np.linalg.norm(vector, axis=1)
    angle = np.unwrap(np.arctan2(vector[:, 1], vector[:, 0]))
    geometry = np.column_stack(
        [center[:, 0], center[:, 1], np.log(np.maximum(spacing, 1.0)), angle]
    )
    geometry = np.column_stack(
        [geometry, np.vstack([np.zeros((1, 4)), np.diff(geometry, axis=0)])]
    )
    filtered_features = np.column_stack(
        [safe_bandpass(geometry[:, col], fps, low_hz, high_hz) for col in range(geometry.shape[1])]
    )
    feature_scale = np.std(filtered_features, axis=0)
    usable = np.isfinite(feature_scale) & (feature_scale > 1e-8)
    diagnostics: dict[str, float] = {}
    if not usable.any():
        for side in ("left", "right"):
            diagnostics[f"{side}_motion_explained_cv"] = 0.0
            diagnostics[f"{side}_disentangle_gain"] = 0.0
        return result, diagnostics
    features = filtered_features[:, usable] / feature_scale[usable]
    features -= np.mean(features, axis=0)

    for side in ("left", "right"):
        raw = numeric(table, f"{side}_temp")
        filled = fill_numeric(raw)
        if not np.isfinite(filled).all():
            diagnostics[f"{side}_motion_explained_cv"] = 0.0
            diagnostics[f"{side}_disentangle_gain"] = 0.0
            continue
        target = safe_bandpass(filled, fps, low_hz, high_hz)
        target -= float(np.mean(target))
        variance = float(np.var(target))
        if variance < 1e-10:
            diagnostics[f"{side}_motion_explained_cv"] = 0.0
            diagnostics[f"{side}_disentangle_gain"] = 0.0
            continue
        cv_prediction = cross_fitted_ridge_prediction(features, target)
        explained = float(1.0 - np.var(target - cv_prediction) / variance)
        explained = float(np.clip(explained, 0.0, 1.0))
        gain = (
            float(np.clip((explained - minimum_explained) / 0.50, 0.0, maximum_gain))
            if explained >= minimum_explained
            else 0.0
        )
        if gain > 0:
            identity = np.eye(features.shape[1], dtype=float)
            beta = np.linalg.solve(
                features.T @ features + 10.0 * identity,
                features.T @ target,
            )
            nuisance = features @ beta
            corrected = filled - gain * nuisance
            corrected[~np.isfinite(raw)] = np.nan
            result[f"{side}_temp"] = corrected
        diagnostics[f"{side}_motion_explained_cv"] = explained
        diagnostics[f"{side}_disentangle_gain"] = gain
    result["temperature_method"] = "cotracker_tfa_motion_disentangled"
    return result, diagnostics


def bool_values(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy()


def short_internal_repair_mask(
    direct: np.ndarray, eligible: np.ndarray, max_gap: int = 3
) -> np.ndarray:
    """Allow only short non-direct runs bounded by direct detections."""
    direct = np.asarray(direct, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    selected = np.zeros(len(direct), dtype=bool)
    for start, end in rr.missing_runs(pd.Series(~direct)):
        if start == 0 or end == len(direct) - 1 or end - start + 1 > int(max_gap):
            continue
        if direct[start - 1] and direct[end + 1] and bool(eligible[start : end + 1].all()):
            selected[start : end + 1] = True
    return selected


def conservative_selective_candidates(
    source: pd.DataFrame,
    tracks: pd.DataFrame,
    joint: pd.DataFrame,
    tfa: pd.DataFrame,
    disentangled: pd.DataFrame,
    diagnostics: pd.DataFrame,
    baseline_row: pd.Series,
    *,
    motion_config: rr.ReproConfig | None = None,
    maximum_repair_gap: int = 3,
    maximum_tfa_residual_px: float = 2.0,
    minimum_visible_fraction: float = 0.8,
    tfa_blend: float = 0.25,
    disentangle_blend: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    tables = (tracks, joint, tfa, disentangled, diagnostics)
    if any(len(table) != len(source) for table in tables):
        raise ValueError("Selective candidates require frame-aligned cached tables")

    if motion_config is None:
        config_row = baseline_row.copy()
        if str(config_row.get("selected_fusion_mode", "")) == "quality":
            config_row["selected_fusion_mode"] = "mean"
        motion_config = config_from_baseline(config_row, "selective_gate")
    motion = rr.nostril_motion_features(source, motion_config)
    motion_mask = bool_values(motion["motion_artifact"])
    residual = pd.to_numeric(diagnostics["registration_residual_px"], errors="coerce").to_numpy()
    registration_reliable = (
        bool_values(diagnostics["registration_applied"])
        & diagnostics["registration_source"].astype(str).eq("joint_points").to_numpy()
        & np.isfinite(residual)
        & (residual <= float(maximum_tfa_residual_px))
    )

    repair = source.copy()
    motion_tfa = source.copy()
    selective_disentangled = source.copy()
    stats: dict[str, int] = {
        "motion_frames": int(motion_mask.sum()),
        "reliable_registration_frames": int(registration_reliable.sum()),
    }
    side_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for side in ("left", "right"):
        baseline_temp = pd.to_numeric(source[f"{side}_temp"], errors="coerce").to_numpy(dtype=float)
        joint_temp = pd.to_numeric(joint[f"{side}_temp"], errors="coerce").to_numpy(dtype=float)
        tfa_temp = pd.to_numeric(tfa[f"{side}_temp"], errors="coerce").to_numpy(dtype=float)
        disentangled_temp = pd.to_numeric(
            disentangled[f"{side}_temp"], errors="coerce"
        ).to_numpy(dtype=float)
        confidence = pd.to_numeric(source[f"{side}_conf"], errors="coerce").to_numpy(dtype=float)
        source_kind = source.get(
            f"{side}_source", pd.Series("", index=source.index)
        ).astype(str).to_numpy()
        direct = np.isfinite(baseline_temp) & (confidence >= 0.5) & (source_kind == "detected")
        visible = pd.to_numeric(
            tracks.get(f"{side}_visible_fraction", pd.Series(0.0, index=source.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        rejected = bool_values(
            tracks.get(f"{side}_tracker_rejected", pd.Series(False, index=source.index))
        )
        tracker_reliable = (visible >= float(minimum_visible_fraction)) & ~rejected
        repair_eligible = tracker_reliable & np.isfinite(joint_temp)
        repair_mask = short_internal_repair_mask(
            direct, repair_eligible, max_gap=maximum_repair_gap
        )
        repaired_values = baseline_temp.copy()
        repaired_values[repair_mask] = joint_temp[repair_mask]
        repair[f"{side}_temp"] = repaired_values
        repair.loc[repair_mask, f"{side}_source"] = "cotracker_short_repair"
        repair[f"{side}_selective_repair"] = repair_mask

        tfa_mask = (
            motion_mask
            & registration_reliable
            & tracker_reliable
            & direct
            & np.isfinite(tfa_temp)
        )
        motion_values = repaired_values.copy()
        motion_values[tfa_mask] = (
            (1.0 - float(tfa_blend)) * motion_values[tfa_mask]
            + float(tfa_blend) * tfa_temp[tfa_mask]
        )
        motion_tfa[f"{side}_temp"] = motion_values
        motion_tfa.loc[repair_mask, f"{side}_source"] = "cotracker_short_repair"
        motion_tfa[f"{side}_selective_repair"] = repair_mask
        motion_tfa[f"{side}_motion_tfa_blended"] = tfa_mask

        disentangle_mask = tfa_mask & np.isfinite(disentangled_temp)
        final_values = motion_values.copy()
        nuisance_delta = disentangled_temp - tfa_temp
        final_values[disentangle_mask] += float(disentangle_blend) * nuisance_delta[
            disentangle_mask
        ]
        selective_disentangled[f"{side}_temp"] = final_values
        selective_disentangled.loc[repair_mask, f"{side}_source"] = "cotracker_short_repair"
        selective_disentangled[f"{side}_selective_repair"] = repair_mask
        selective_disentangled[f"{side}_motion_tfa_blended"] = tfa_mask
        selective_disentangled[f"{side}_motion_disentangle_applied"] = disentangle_mask
        side_masks[side] = (repair_mask, tfa_mask, disentangle_mask)
        stats[f"{side}_repair_frames"] = int(repair_mask.sum())
        stats[f"{side}_tfa_frames"] = int(tfa_mask.sum())
        stats[f"{side}_disentangle_frames"] = int(disentangle_mask.sum())

    repair["temperature_method"] = "cotracker_selective_short_repair"
    motion_tfa["temperature_method"] = "cotracker_selective_repair_motion_tfa"
    selective_disentangled[
        "temperature_method"
    ] = "cotracker_selective_repair_motion_tfa_disentangled"
    return repair, motion_tfa, selective_disentangled, stats


def frozen_truth_row(baseline_row: pd.Series) -> pd.Series:
    duration = baseline_row.get("truth_duration_seconds", baseline_row.get("duration_seconds"))
    return pd.Series(
        {
            "video_id": str(baseline_row.video_id),
            "breath_count": int(float(baseline_row.truth_count)),
            "duration_seconds": float(duration),
            "rr": float(baseline_row.truth_rr),
        }
    )


def evaluate_table(table: pd.DataFrame, baseline_row: pd.Series, stage: str) -> dict[str, object]:
    analysis_limit = int(float(baseline_row.analysis_frame_limit))
    table = table.iloc[:analysis_limit].copy()
    truth = frozen_truth_row(baseline_row)
    config_row = baseline_row.copy()
    # One frozen row used the retired static quality-weighted mode. Its saved
    # bilateral shares are 0.502/0.498, so ``mean`` is its executable replay.
    if str(config_row.get("selected_fusion_mode", "")) == "quality":
        config_row["selected_fusion_mode"] = "mean"
    config = config_from_baseline(config_row, stage)
    _, summary = rr.fuse_temperature_curve(table, config, truth)
    return {
        "video_id": str(baseline_row.video_id),
        "stage": stage,
        "predicted_count": int(summary["peaks"]),
        "predicted_rr_bpm": float(summary["rr_bpm"]),
        "truth_count": int(truth.breath_count),
        "truth_rr": float(truth.rr),
        "count_error": int(summary["peaks"] - truth.breath_count),
        "selected_fusion_mode": str(summary["selected_fusion_mode"]),
    }


def stage_metrics(predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> dict[str, object]:
    truth = predictions["truth_rr"].to_numpy(dtype=float)
    predicted = predictions["predicted_rr_bpm"].to_numpy(dtype=float)
    count_error = predictions["count_error"].to_numpy(dtype=float)
    truth_count = predictions["truth_count"].to_numpy(dtype=float)
    rr_error = predicted - truth
    joined = predictions.merge(
        baseline_predictions[["video_id", "predicted_count", "predicted_rr_bpm"]],
        on="video_id",
        suffixes=("", "_baseline"),
    )
    baseline_abs = np.abs(joined["predicted_count_baseline"] - joined["truth_count"])
    candidate_abs = np.abs(joined["predicted_count"] - joined["truth_count"])
    return {
        "stage": str(predictions["stage"].iloc[0]),
        "videos": int(len(predictions)),
        "rr_r2": rr.regression_r2(pd.Series(truth), pd.Series(predicted)),
        "rr_pearson_r2": rr.pearson_r2(pd.Series(truth), pd.Series(predicted)),
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "count_mae": float(np.mean(np.abs(count_error))),
        "mean_count_accuracy_pct": float(
            100.0
            * np.mean(
                np.clip(
                    1.0 - np.abs(count_error) / np.maximum(truth_count, 1.0),
                    0.0,
                    1.0,
                )
            )
        ),
        "exact_count": int(np.sum(np.abs(count_error) == 0)),
        "within_one_count": int(np.sum(np.abs(count_error) <= 1)),
        "changed_videos": int(np.sum(joined["predicted_count"] != joined["predicted_count_baseline"])),
        "improved_videos": int(np.sum(candidate_abs < baseline_abs)),
        "worsened_videos": int(np.sum(candidate_abs > baseline_abs)),
    }


def baseline_prediction_table(baseline: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": baseline["video_id"].astype(str),
            "stage": "frozen_innovation_repro",
            "predicted_count": pd.to_numeric(baseline["peaks"]).astype(int),
            "predicted_rr_bpm": pd.to_numeric(baseline["rr_bpm"]).astype(float),
            "truth_count": pd.to_numeric(baseline["truth_count"]).astype(int),
            "truth_rr": pd.to_numeric(baseline["truth_rr"]).astype(float),
            "count_error": pd.to_numeric(baseline["peaks"]).astype(int)
            - pd.to_numeric(baseline["truth_count"]).astype(int),
            "selected_fusion_mode": baseline["selected_fusion_mode"].astype(str),
        }
    )


def cache_paths(video_dir: Path, run_name: str) -> dict[str, Path]:
    return {
        "tracks": video_dir / f"{run_name}_tracks.csv",
        "joint": video_dir / f"{run_name}_joint_temperatures.csv",
        "tfa": video_dir / f"{run_name}_temperatures.csv",
        "disentangled": video_dir / f"{run_name}_disentangled_temperatures.csv",
        "diagnostics": video_dir / f"{run_name}_diagnostics.csv",
        "selective_repair": video_dir / f"{run_name}_selective_repair_temperatures.csv",
        "selective_tfa": video_dir / f"{run_name}_selective_tfa_temperatures.csv",
        "selective_disentangled": video_dir
        / f"{run_name}_selective_disentangled_temperatures.csv",
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    baseline = pd.read_csv(args.baseline_summary, dtype={"video_id": str})
    if len(baseline) != 73 or baseline["video_id"].nunique() != 73:
        raise ValueError("Frozen baseline must contain exactly 73 unique video IDs")
    if args.video_id:
        missing = sorted(set(args.video_id) - set(baseline["video_id"]))
        if missing:
            raise ValueError(f"Unknown video IDs: {missing}")
        baseline = baseline[baseline["video_id"].isin(args.video_id)].copy()
    if args.max_videos is not None:
        baseline = baseline.head(max(1, int(args.max_videos))).copy()
    baseline = baseline.reset_index(drop=True)
    baseline_predictions = baseline_prediction_table(baseline)

    replay_rows: list[dict[str, object]] = []
    for row in baseline.itertuples(index=False):
        source = pd.read_csv(args.input_root / str(row.video_id) / "paper_repro_temperatures.csv")
        replay_rows.append(evaluate_table(source, pd.Series(row._asdict()), "baseline_replay"))
    replay = pd.DataFrame(replay_rows)
    replay_matches = int(
        np.sum(replay["predicted_count"].to_numpy() == baseline_predictions["predicted_count"].to_numpy())
    )
    if replay_matches != len(baseline):
        mismatches = replay.loc[
            replay["predicted_count"].to_numpy() != baseline_predictions["predicted_count"].to_numpy(),
            "video_id",
        ].tolist()
        raise RuntimeError(f"Frozen baseline replay failed for {mismatches}")
    if args.replay_only:
        print(f"baseline_replay={replay_matches}/{len(baseline)}")
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CoTracker experiment requires CUDA")
    device = "cuda"
    tracker = load_cotracker(args.cotracker_repo, args.cotracker_checkpoint, device)
    temp_model = joblib.load(args.temp_model)
    stage_rows: dict[str, list[dict[str, object]]] = {
        "cotracker_joint": [],
        "cotracker_tfa": [],
        "cotracker_tfa_disentangled": [],
        "cotracker_selective_repair": [],
        "cotracker_selective_tfa": [],
        "cotracker_selective_tfa_disentangled": [],
    }
    diagnostic_rows: list[dict[str, object]] = []

    for position, baseline_row in baseline.iterrows():
        video_id = str(baseline_row.video_id)
        video_dir = args.input_root / video_id
        paths = cache_paths(video_dir, args.run_name)
        source = pd.read_csv(video_dir / "paper_repro_temperatures.csv")
        limit = int(float(baseline_row.analysis_frame_limit))
        source = source.iloc[:limit].copy().reset_index(drop=True)
        image_by_name = {path.name: path for path in rr.frame_image_paths(video_dir)}
        frame_paths = [image_by_name[str(name)] for name in source["frame_name"]]
        core_cache_names = ("tracks", "joint", "tfa", "disentangled", "diagnostics")
        can_reuse = args.reuse_cache and all(paths[name].exists() for name in core_cache_names)
        if can_reuse:
            tracks = pd.read_csv(paths["tracks"])
            joint_table = pd.read_csv(paths["joint"])
            tfa_table = pd.read_csv(paths["tfa"])
            disentangled = pd.read_csv(paths["disentangled"])
            diagnostics = pd.read_csv(paths["diagnostics"])
            if not (
                len(tracks)
                == len(joint_table)
                == len(tfa_table)
                == len(disentangled)
                == len(source)
            ):
                can_reuse = False
        if not can_reuse:
            print(f"[{position + 1}/{len(baseline)}] {video_id}: CoTracker")
            try:
                tracks, tracked_points, point_visible, centers = joint_cotracker_tracks(
                    source,
                    frame_paths,
                    tracker,
                    device,
                    args.window_length,
                    args.window_overlap,
                    args.anchor_search_frames,
                )
                joint_table, tfa_table, diagnostics = extract_candidate_temperatures(
                    source,
                    frame_paths,
                    tracks,
                    tracked_points,
                    point_visible,
                    centers,
                    temp_model,
                    args.roi_radius,
                    args.min_temp,
                )
                disentangled, disentangle_diagnostics = disentangle_motion_component(
                    tfa_table, fps=args.fps
                )
            except ValueError as error:
                if "bilateral" not in str(error):
                    raise
                tracks = source[["frame_name", "left_x", "left_y", "right_x", "right_y"]].copy()
                tracks["tracking_status"] = "no_anchor_fallback"
                joint_table = source.copy()
                tfa_table = source.copy()
                disentangled = source.copy()
                for table in (joint_table, tfa_table, disentangled):
                    table["temperature_method"] = "no_anchor_fallback"
                diagnostics = pd.DataFrame(
                    {
                        "registration_applied": np.zeros(len(source), dtype=bool),
                        "registration_source": ["no_anchor_fallback"] * len(source),
                        "registration_residual_px": np.full(len(source), np.nan),
                    }
                )
                disentangle_diagnostics = {
                    "left_motion_explained_cv": 0.0,
                    "right_motion_explained_cv": 0.0,
                    "left_disentangle_gain": 0.0,
                    "right_disentangle_gain": 0.0,
                }
            tracks.to_csv(paths["tracks"], index=False)
            joint_table.to_csv(paths["joint"], index=False)
            tfa_table.to_csv(paths["tfa"], index=False)
            disentangled.to_csv(paths["disentangled"], index=False)
            diagnostics = diagnostics.assign(**disentangle_diagnostics)
            diagnostics.to_csv(paths["diagnostics"], index=False)
        selective_repair, selective_tfa, selective_disentangled, selective_stats = (
            conservative_selective_candidates(
                source,
                tracks,
                joint_table,
                tfa_table,
                disentangled,
                diagnostics,
                baseline_row,
            )
        )
        selective_repair.to_csv(paths["selective_repair"], index=False)
        selective_tfa.to_csv(paths["selective_tfa"], index=False)
        selective_disentangled.to_csv(paths["selective_disentangled"], index=False)
        print(f"[{position + 1}/{len(baseline)}] {video_id}: evaluate")
        for stage, table in (
            ("cotracker_joint", joint_table),
            ("cotracker_tfa", tfa_table),
            ("cotracker_tfa_disentangled", disentangled),
            ("cotracker_selective_repair", selective_repair),
            ("cotracker_selective_tfa", selective_tfa),
            ("cotracker_selective_tfa_disentangled", selective_disentangled),
        ):
            stage_rows[stage].append(evaluate_table(table, baseline_row, stage))
        diagnostic_rows.append(
            {
                "video_id": video_id,
                "frames": len(source),
                "registration_coverage": float(
                    pd.Series(diagnostics["registration_applied"]).astype(bool).mean()
                ),
                "joint_registration_fraction": float(
                    pd.Series(diagnostics["registration_source"]).eq("joint_points").mean()
                ),
                "median_registration_residual_px": float(
                    pd.to_numeric(diagnostics.get("registration_residual_px"), errors="coerce").median()
                ),
                "left_motion_explained_cv": float(
                    pd.to_numeric(diagnostics.get("left_motion_explained_cv"), errors="coerce").median()
                ),
                "right_motion_explained_cv": float(
                    pd.to_numeric(diagnostics.get("right_motion_explained_cv"), errors="coerce").median()
                ),
                "left_disentangle_gain": float(
                    pd.to_numeric(diagnostics.get("left_disentangle_gain"), errors="coerce").median()
                ),
                "right_disentangle_gain": float(
                    pd.to_numeric(diagnostics.get("right_disentangle_gain"), errors="coerce").median()
                ),
                **selective_stats,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = [baseline_predictions]
    metrics = [stage_metrics(baseline_predictions, baseline_predictions)]
    for stage in stage_rows:
        table = pd.DataFrame(stage_rows[stage])
        table.to_csv(args.output_dir / f"{args.run_name}_{stage}_predictions.csv", index=False)
        all_predictions.append(table)
        metrics.append(stage_metrics(table, baseline_predictions))
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    prediction_table.to_csv(args.output_dir / f"{args.run_name}_all_predictions.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / f"{args.run_name}_video_diagnostics.csv", index=False
    )
    metric_table = pd.DataFrame(metrics)

    retained = metric_table.iloc[0]
    decisions: list[dict[str, object]] = []
    for index in range(1, len(metric_table)):
        candidate = metric_table.iloc[index]
        keep = bool(
            float(candidate.rr_r2) > float(retained.rr_r2) + 1e-12
            and float(candidate.rr_mae_bpm) <= float(retained.rr_mae_bpm) + 1e-12
        )
        decisions.append(
            {
                "candidate_stage": candidate.stage,
                "compared_with": retained.stage,
                "candidate_rr_r2": candidate.rr_r2,
                "candidate_rr_mae_bpm": candidate.rr_mae_bpm,
                "retained": keep,
                "decision": "keep" if keep else "rollback",
            }
        )
        if keep:
            retained = candidate
    metric_table["final_retained_stage"] = str(retained.stage)
    metric_table["elapsed_seconds"] = time.perf_counter() - started
    metric_table.to_csv(args.output_dir / f"{args.run_name}_metrics.csv", index=False)
    pd.DataFrame(decisions).to_csv(
        args.output_dir / f"{args.run_name}_rollback_decisions.csv", index=False
    )
    print(metric_table.to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))
    print(f"final_retained_stage={retained.stage}")


if __name__ == "__main__":
    main()
