from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from paper_repro_rr import circle_temperature, frame_image_paths, missing_runs


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Repair missing nostril temperatures using bidirectional Lucas-Kanade "
            "tracking with forward-backward and trajectory-agreement gates."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument("--source-prefix", default="external_repro_single_reference")
    parser.add_argument("--target-prefix", default="external_repro_optical_flow")
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=(
            repo_root
            / "temperature_extraction"
            / "getRandomForestRegress"
            / "clf_model_RGB_20240906.pkl"
        ),
    )
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--max-gap-frames", type=int, default=45)
    parser.add_argument("--lk-window", type=int, default=31)
    parser.add_argument("--lk-levels", type=int, default=3)
    parser.add_argument("--fb-error-threshold", type=float, default=1.5)
    parser.add_argument("--lk-error-threshold", type=float, default=30.0)
    parser.add_argument("--trajectory-disagreement-threshold", type=float, default=18.0)
    parser.add_argument("--template-half-size", type=int, default=24)
    parser.add_argument("--template-search-radius", type=int, default=100)
    parser.add_argument("--template-correlation-threshold", type=float, default=0.55)
    parser.add_argument("--template-fb-error-threshold", type=float, default=8.0)
    parser.add_argument("--video-id", action="append", default=[])
    return parser.parse_args()


def numeric(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else math.nan


def load_gray_frames(paths: list[Path]) -> list[np.ndarray | None]:
    return [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]


def lk_step(
    source: np.ndarray,
    target: np.ndarray,
    point: np.ndarray,
    *,
    window: int,
    levels: int,
    fb_threshold: float,
    error_threshold: float,
) -> tuple[np.ndarray | None, float, float]:
    initial = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        source,
        target,
        initial,
        None,
        winSize=(int(window), int(window)),
        maxLevel=int(levels),
        criteria=criteria,
        flags=0,
        minEigThreshold=1e-4,
    )
    if tracked is None or status is None or int(status.ravel()[0]) != 1:
        return None, math.nan, math.nan
    reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        target,
        source,
        tracked,
        None,
        winSize=(int(window), int(window)),
        maxLevel=int(levels),
        criteria=criteria,
        flags=0,
        minEigThreshold=1e-4,
    )
    if reverse is None or reverse_status is None or int(reverse_status.ravel()[0]) != 1:
        return None, math.nan, math.nan
    fb_error = float(np.linalg.norm(reverse.reshape(2) - initial.reshape(2)))
    lk_error = float(error.ravel()[0]) if error is not None else math.nan
    candidate = tracked.reshape(2).astype(float)
    height, width = target.shape[:2]
    inside = 0 <= candidate[0] < width and 0 <= candidate[1] < height
    if (
        not inside
        or not math.isfinite(fb_error)
        or fb_error > float(fb_threshold)
        or (math.isfinite(lk_error) and lk_error > float(error_threshold))
    ):
        return None, fb_error, lk_error
    return candidate, fb_error, lk_error


def template_match_once(
    source: np.ndarray,
    target: np.ndarray,
    point: np.ndarray,
    *,
    half_size: int,
    search_radius: int,
) -> tuple[np.ndarray | None, float]:
    x, y = np.asarray(point, dtype=float)
    h, w = source.shape[:2]
    half = int(half_size)
    cx, cy = int(round(x)), int(round(y))
    if cx - half < 0 or cy - half < 0 or cx + half + 1 > w or cy + half + 1 > h:
        return None, math.nan
    template = source[cy - half : cy + half + 1, cx - half : cx + half + 1]
    target_h, target_w = target.shape[:2]
    radius = int(search_radius)
    left = max(0, cx - radius - half)
    top = max(0, cy - radius - half)
    right = min(target_w, cx + radius + half + 1)
    bottom = min(target_h, cy + radius + half + 1)
    search = target[top:bottom, left:right]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, math.nan
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(response)
    candidate = np.array(
        [left + location[0] + half, top + location[1] + half],
        dtype=float,
    )
    return candidate, float(maximum)


def correlation_step(
    source: np.ndarray,
    target: np.ndarray,
    point: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float, float]:
    candidate, correlation = template_match_once(
        source,
        target,
        point,
        half_size=args.template_half_size,
        search_radius=args.template_search_radius,
    )
    if candidate is None or correlation < args.template_correlation_threshold:
        return None, math.nan, correlation
    reverse, reverse_correlation = template_match_once(
        target,
        source,
        candidate,
        half_size=args.template_half_size,
        search_radius=args.template_search_radius,
    )
    if reverse is None or reverse_correlation < args.template_correlation_threshold:
        return None, math.nan, min(correlation, reverse_correlation)
    fb_error = float(np.linalg.norm(reverse - np.asarray(point, dtype=float)))
    if fb_error > args.template_fb_error_threshold:
        return None, fb_error, min(correlation, reverse_correlation)
    return candidate, fb_error, min(correlation, reverse_correlation)


def track_direction(
    gray: list[np.ndarray | None],
    anchor_index: int,
    anchor_point: np.ndarray,
    targets: list[int],
    direction: int,
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[float, float, str]]]:
    points: dict[int, np.ndarray] = {}
    quality: dict[int, tuple[float, float, str]] = {}
    current_index = int(anchor_index)
    current_point = np.asarray(anchor_point, dtype=float)
    for target_index in targets:
        if int(target_index) - current_index != int(direction):
            break
        source_gray = gray[current_index]
        target_gray = gray[int(target_index)]
        if source_gray is None or target_gray is None:
            break
        candidate, fb_error, lk_error = lk_step(
            source_gray,
            target_gray,
            current_point,
            window=args.lk_window,
            levels=args.lk_levels,
            fb_threshold=args.fb_error_threshold,
            error_threshold=args.lk_error_threshold,
        )
        method = "lucas_kanade"
        if candidate is None:
            candidate, fb_error, correlation = correlation_step(
                source_gray,
                target_gray,
                current_point,
                args,
            )
            lk_error = -float(correlation) if math.isfinite(correlation) else math.nan
            method = "correlation_fallback"
        if candidate is None:
            break
        points[int(target_index)] = candidate
        quality[int(target_index)] = (fb_error, lk_error, method)
        current_index = int(target_index)
        current_point = candidate
    return points, quality


def nearest_anchor(
    table: pd.DataFrame,
    side: str,
    start: int,
    direction: int,
) -> int | None:
    x = pd.to_numeric(table[f"{side}_x"], errors="coerce")
    y = pd.to_numeric(table[f"{side}_y"], errors="coerce")
    source = table[f"{side}_source"].fillna("").astype(str)
    reliable = x.notna() & y.notna() & source.eq("detected")
    if direction < 0:
        indices = np.flatnonzero(reliable.to_numpy()[:start])
        return int(indices[-1]) if len(indices) else None
    indices = np.flatnonzero(reliable.to_numpy()[start + 1 :])
    return int(start + 1 + indices[0]) if len(indices) else None


def repair_side(
    table: pd.DataFrame,
    paths: list[Path],
    gray: list[np.ndarray | None],
    temp_model,
    side: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    temp_column = f"{side}_temp"
    x_column = f"{side}_x"
    y_column = f"{side}_y"
    source_column = f"{side}_source"
    report_rows: list[dict[str, object]] = []
    for run_index, (start, end) in enumerate(missing_runs(table[temp_column].isna()), start=1):
        length = end - start + 1
        before = nearest_anchor(table, side, start, -1)
        after = nearest_anchor(table, side, end, 1)
        forward_points: dict[int, np.ndarray] = {}
        backward_points: dict[int, np.ndarray] = {}
        forward_quality: dict[int, tuple[float, float, str]] = {}
        backward_quality: dict[int, tuple[float, float, str]] = {}
        if before is not None and start - before <= args.max_gap_frames:
            anchor = np.array(
                [numeric(table.at[before, x_column]), numeric(table.at[before, y_column])]
            )
            targets = list(range(before + 1, min(end, before + args.max_gap_frames) + 1))
            forward_points, forward_quality = track_direction(
                gray, before, anchor, targets, 1, args
            )
        if after is not None and after - end <= args.max_gap_frames:
            anchor = np.array(
                [numeric(table.at[after, x_column]), numeric(table.at[after, y_column])]
            )
            targets = list(range(after - 1, max(start, after - args.max_gap_frames) - 1, -1))
            backward_points, backward_quality = track_direction(
                gray, after, anchor, targets, -1, args
            )

        attempted = 0
        repaired = 0
        rejected_disagreement = 0
        fb_errors: list[float] = []
        lk_errors: list[float] = []
        correlation_steps = 0
        lucas_kanade_steps = 0
        for row_index in range(start, end + 1):
            forward = forward_points.get(row_index)
            backward = backward_points.get(row_index)
            if forward is None and backward is None:
                continue
            attempted += 1
            disagreement = math.nan
            if forward is not None and backward is not None:
                disagreement = float(np.linalg.norm(forward - backward))
                if disagreement > args.trajectory_disagreement_threshold:
                    rejected_disagreement += 1
                    continue
                fraction = (row_index - start + 1) / (length + 1)
                point = (1.0 - fraction) * forward + fraction * backward
                fb_errors.extend(
                    [forward_quality[row_index][0], backward_quality[row_index][0]]
                )
                lk_errors.extend(
                    [forward_quality[row_index][1], backward_quality[row_index][1]]
                )
                methods = [forward_quality[row_index][2], backward_quality[row_index][2]]
                source_name = "optical_flow_bidirectional"
            elif forward is not None:
                point = forward
                fb_errors.append(forward_quality[row_index][0])
                lk_errors.append(forward_quality[row_index][1])
                methods = [forward_quality[row_index][2]]
                source_name = "optical_flow_forward"
            else:
                point = backward
                fb_errors.append(backward_quality[row_index][0])
                lk_errors.append(backward_quality[row_index][1])
                methods = [backward_quality[row_index][2]]
                source_name = "optical_flow_backward"
            correlation_steps += sum(method == "correlation_fallback" for method in methods)
            lucas_kanade_steps += sum(method == "lucas_kanade" for method in methods)
            image = cv2.imread(str(paths[row_index]))
            if image is None:
                continue
            temperature = circle_temperature(
                image,
                temp_model,
                float(point[0]),
                float(point[1]),
                args.radius,
                args.min_temp,
            )
            if not math.isfinite(temperature):
                continue
            table.at[row_index, temp_column] = temperature
            table.at[row_index, x_column] = float(point[0])
            table.at[row_index, y_column] = float(point[1])
            table.at[row_index, source_column] = source_name
            table.at[row_index, "status"] = "optical_flow_repair"
            repaired += 1
        report_rows.append(
            {
                "side": side,
                "run_index": run_index,
                "run_start": start,
                "run_end": end,
                "run_length": length,
                "anchor_before": before if before is not None else "",
                "anchor_after": after if after is not None else "",
                "attempted_rows": attempted,
                "repaired_rows": repaired,
                "rejected_trajectory_disagreement": rejected_disagreement,
                "median_fb_error": float(np.median(fb_errors)) if fb_errors else math.nan,
                "median_lk_error": float(np.median(lk_errors)) if lk_errors else math.nan,
                "lucas_kanade_steps": lucas_kanade_steps,
                "correlation_fallback_steps": correlation_steps,
            }
        )
    return table, report_rows


def process_video(
    video_dir: Path,
    temp_model,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_path = video_dir / f"{args.source_prefix}_temperatures.csv"
    target_path = video_dir / f"{args.target_prefix}_temperatures.csv"
    if not source_path.exists():
        return (
            {
                "video_id": video_dir.name,
                "status": "source_temperature_missing",
                "source_missing_left": "",
                "source_missing_right": "",
                "target_missing_left": "",
                "target_missing_right": "",
                "repaired_left": 0,
                "repaired_right": 0,
                "target_temperature_csv": str(target_path),
            },
            [],
        )
    table = pd.read_csv(source_path)
    paths = frame_image_paths(video_dir)
    if len(paths) != len(table):
        return (
            {
                "video_id": video_dir.name,
                "status": "frame_temperature_length_mismatch",
                "source_missing_left": int(table["left_temp"].isna().sum()),
                "source_missing_right": int(table["right_temp"].isna().sum()),
                "target_missing_left": "",
                "target_missing_right": "",
                "repaired_left": 0,
                "repaired_right": 0,
                "target_temperature_csv": str(target_path),
            },
            [],
        )
    source_missing_left = int(table["left_temp"].isna().sum())
    source_missing_right = int(table["right_temp"].isna().sum())
    gray = load_gray_frames(paths)
    details: list[dict[str, object]] = []
    for side in ["left", "right"]:
        table, side_rows = repair_side(table, paths, gray, temp_model, side, args)
        details.extend(
            {"video_id": video_dir.name, **row} for row in side_rows
        )
    target_missing_left = int(table["left_temp"].isna().sum())
    target_missing_right = int(table["right_temp"].isna().sum())
    table.to_csv(target_path, index=False)
    return (
        {
            "video_id": video_dir.name,
            "status": "PASS",
            "source_missing_left": source_missing_left,
            "source_missing_right": source_missing_right,
            "target_missing_left": target_missing_left,
            "target_missing_right": target_missing_right,
            "repaired_left": source_missing_left - target_missing_left,
            "repaired_right": source_missing_right - target_missing_right,
            "target_temperature_csv": str(target_path),
        },
        details,
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    temp_model = joblib.load(args.temp_model.resolve())
    selected = set(args.video_id)
    video_dirs = sorted(
        [
            path
            for path in input_root.iterdir()
            if path.is_dir() and (not selected or path.name in selected)
        ],
        key=lambda path: path.name,
    )
    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for index, video_dir in enumerate(video_dirs, start=1):
        print(f"[{index}/{len(video_dirs)}] {video_dir.name}")
        summary, rows = process_video(video_dir, temp_model, args)
        summaries.append(summary)
        details.extend(rows)
    summary_table = pd.DataFrame(summaries)
    detail_table = pd.DataFrame(details)
    summary_path = input_root / f"{args.target_prefix}_optical_flow_repair_summary.csv"
    detail_path = input_root / f"{args.target_prefix}_optical_flow_repair_runs.csv"
    summary_table.to_csv(summary_path, index=False)
    detail_table.to_csv(detail_path, index=False)
    passed = int(summary_table["status"].eq("PASS").sum()) if not summary_table.empty else 0
    repaired_left = pd.to_numeric(summary_table.get("repaired_left"), errors="coerce").sum()
    repaired_right = pd.to_numeric(summary_table.get("repaired_right"), errors="coerce").sum()
    print(f"Saved optical-flow summary: {summary_path}")
    print(f"Saved optical-flow run audit: {detail_path}")
    print(
        f"videos_pass={passed}/{len(summary_table)}; "
        f"repaired_left={int(repaired_left)}; repaired_right={int(repaired_right)}"
    )


if __name__ == "__main__":
    main()
