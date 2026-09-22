"""Evaluate nostril-spacing adaptive ROI radii without rerunning YOLO."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import paper_repro_rr as rr
from run_lindian_robust_rr import robust_config


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    assets = data_root / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=data_root / "lindian_anchored30_frames"
    )
    parser.add_argument(
        "--truth", type=Path, default=assets / "lindian_anchored30_manual_truth.csv"
    )
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=repo_root
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
    )
    parser.add_argument("--temperature-prefix", default="lindian_anchored30_repro")
    parser.add_argument("--radius-table-prefix", default="lindian_motion_roi_radius")
    parser.add_argument("--min-radius", type=int, default=14)
    parser.add_argument("--max-radius", type=int, default=28)
    parser.add_argument("--base-radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--batch-frames", type=int, default=24)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--reuse-radius-tables", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=assets)
    return parser.parse_args()


def roi_pixels(image: np.ndarray, x: float, y: float, radius: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return np.empty((0, 3), dtype=np.uint8), np.empty(0, dtype=np.int32)
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = distance_sq <= radius * radius
    return image[y0:y1, x0:x1][mask], distance_sq[mask].astype(np.int32)


def fill_batch(
    rows: list[tuple[int, str, np.ndarray, np.ndarray]],
    model,
    output: dict[str, np.ndarray],
    radii: list[int],
    min_temp: float | None,
) -> None:
    valid = [row for row in rows if len(row[2])]
    if not valid:
        return
    pixels = np.concatenate([row[2] for row in valid], axis=0)
    predictions = np.asarray(model.predict(pixels.reshape(-1, 3)), dtype=float)
    offset = 0
    for frame_index, side, block, distance_sq in valid:
        values = predictions[offset : offset + len(block)]
        offset += len(block)
        temperature_valid = np.ones(len(values), dtype=bool)
        if min_temp is not None:
            temperature_valid &= values > float(min_temp)
        for radius in radii:
            selected = temperature_valid & (distance_sq <= radius * radius)
            if selected.any():
                output[f"{side}_temp_r{radius}"][frame_index] = float(
                    np.mean(values[selected])
                )


def extract_radius_table(
    video_dir: Path,
    temperatures: pd.DataFrame,
    model,
    *,
    radii: list[int],
    min_temp: float | None,
    batch_frames: int,
) -> pd.DataFrame:
    output = {
        f"{side}_temp_r{radius}": np.full(len(temperatures), np.nan, dtype=float)
        for side in ["left", "right"]
        for radius in radii
    }
    pending: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for start in range(0, len(temperatures), max(1, int(batch_frames))):
        stop = min(len(temperatures), start + max(1, int(batch_frames)))
        for frame_index in range(start, stop):
            row = temperatures.iloc[frame_index]
            image = cv2.imread(str(video_dir / str(row["frame_name"])))
            if image is None:
                continue
            for side in ["left", "right"]:
                x = pd.to_numeric(pd.Series([row.get(f"{side}_x")]), errors="coerce").iloc[0]
                y = pd.to_numeric(pd.Series([row.get(f"{side}_y")]), errors="coerce").iloc[0]
                if pd.isna(x) or pd.isna(y):
                    continue
                pixels, distance_sq = roi_pixels(image, float(x), float(y), max(radii))
                pending.append((frame_index, side, pixels, distance_sq))
        fill_batch(pending, model, output, radii, min_temp)
        pending.clear()
    result = temperatures.copy()
    for column, values in output.items():
        result[column] = values
    return result


def radius_policy_table(
    radius_table: pd.DataFrame,
    *,
    base_radius: int,
    exponent: float,
    clip_low: int,
    clip_high: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    left_x = pd.to_numeric(radius_table["left_x"], errors="coerce").to_numpy(dtype=float)
    left_y = pd.to_numeric(radius_table["left_y"], errors="coerce").to_numpy(dtype=float)
    right_x = pd.to_numeric(radius_table["right_x"], errors="coerce").to_numpy(dtype=float)
    right_y = pd.to_numeric(radius_table["right_y"], errors="coerce").to_numpy(dtype=float)
    spacing = np.hypot(left_x - right_x, left_y - right_y)
    finite = spacing[np.isfinite(spacing) & (spacing > 1e-6)]
    reference = float(np.median(finite)) if len(finite) else math.nan
    if not math.isfinite(reference) or reference <= 0:
        selected_radius = np.full(len(radius_table), int(base_radius), dtype=int)
    else:
        ratio = np.divide(
            spacing,
            reference,
            out=np.ones(len(radius_table), dtype=float),
            where=np.isfinite(spacing) & (spacing > 1e-6),
        )
        selected_radius = np.rint(base_radius * np.power(ratio, float(exponent))).astype(int)
        selected_radius = np.clip(selected_radius, int(clip_low), int(clip_high))

    result = radius_table[
        [
            "frame_name",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "left_conf",
            "right_conf",
            "left_source",
            "right_source",
            "status",
        ]
    ].copy()
    for side in ["left", "right"]:
        values = np.full(len(result), np.nan, dtype=float)
        for radius in np.unique(selected_radius):
            column = f"{side}_temp_r{int(radius)}"
            use = selected_radius == radius
            values[use] = pd.to_numeric(radius_table.loc[use, column], errors="coerce")
        result[f"{side}_temp"] = values
    result["adaptive_roi_radius"] = selected_radius
    result["nostril_spacing_px"] = spacing
    result["reference_nostril_spacing_px"] = reference
    return result, selected_radius


def policies(base_radius: int) -> list[dict[str, object]]:
    result = [
        {
            "policy_id": "fixed_radius20_reextracted",
            "exponent": 0.0,
            "clip_low": base_radius,
            "clip_high": base_radius,
        },
        {
            "policy_id": "adaptive_roi_sqrt_clip16_24",
            "exponent": 0.5,
            "clip_low": 16,
            "clip_high": 24,
        },
        {
            "policy_id": "adaptive_roi_linear_clip16_24",
            "exponent": 1.0,
            "clip_low": 16,
            "clip_high": 24,
        },
        {
            "policy_id": "adaptive_roi_sqrt_clip14_26",
            "exponent": 0.5,
            "clip_low": 14,
            "clip_high": 26,
        },
        {
            "policy_id": "adaptive_roi_linear_clip14_26",
            "exponent": 1.0,
            "clip_low": 14,
            "clip_high": 26,
        },
        {
            "policy_id": "adaptive_roi_linear_clip14_28",
            "exponent": 1.0,
            "clip_low": 14,
            "clip_high": 28,
        },
    ]
    for exponent in [0.6, 0.7, 0.8, 0.9]:
        for clip_high in [22, 23, 24]:
            result.append(
                {
                    "policy_id": f"adaptive_roi_exp{exponent:g}_clip16_{clip_high}",
                    "exponent": exponent,
                    "clip_low": 16,
                    "clip_high": clip_high,
                }
            )
    for spacing_cv_threshold in [0.04, 0.05, 0.06, 0.07]:
        result.append(
            {
                "policy_id": f"adaptive_roi_damped_cv{spacing_cv_threshold:g}",
                "exponent": 1.0,
                "clip_low": 16,
                "clip_high": 24,
                "spacing_cv_threshold": spacing_cv_threshold,
                "damped_exponent": 0.5,
            }
        )
    return result


def nostril_spacing_cv(radius_table: pd.DataFrame) -> float:
    left_x = pd.to_numeric(radius_table["left_x"], errors="coerce").to_numpy(dtype=float)
    left_y = pd.to_numeric(radius_table["left_y"], errors="coerce").to_numpy(dtype=float)
    right_x = pd.to_numeric(radius_table["right_x"], errors="coerce").to_numpy(dtype=float)
    right_y = pd.to_numeric(radius_table["right_y"], errors="coerce").to_numpy(dtype=float)
    spacing = np.hypot(left_x - right_x, left_y - right_y)
    finite = spacing[np.isfinite(spacing) & (spacing > 1e-6)]
    if len(finite) < 2 or float(np.mean(finite)) <= 0:
        return math.nan
    return float(np.std(finite) / np.mean(finite))


def metrics(group: pd.DataFrame, policy: dict[str, object], subset: str) -> dict[str, object]:
    truth_rr = group["manual_rr_bpm"].to_numpy(dtype=float)
    predicted_rr = group["predicted_rr_bpm"].to_numpy(dtype=float)
    error = group["count_error"].to_numpy(dtype=float)
    abs_error = np.abs(error)
    return {
        **policy,
        "analysis_set": subset,
        "videos": len(group),
        "rr_r2": rr.regression_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_mae_bpm": float(np.mean(np.abs(predicted_rr - truth_rr))),
        "rr_rmse_bpm": float(np.sqrt(np.mean((predicted_rr - truth_rr) ** 2))),
        "count_mae": float(np.mean(abs_error)),
        "exact_count": int(np.sum(abs_error == 0)),
        "within_one_count": int(np.sum(abs_error <= 1)),
        "count_error_ge_two": int(np.sum(abs_error >= 2)),
        "radius_mean": float(group["radius_mean"].mean()),
        "radius_sd_mean": float(group["radius_sd"].mean()),
        "radius_changed_fraction_mean": float(group["radius_changed_fraction"].mean()),
        "spacing_cv_mean": float(group["spacing_cv"].mean()),
        "damped_video_count": int(group["roi_exponent_damped"].sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    truth = truth.loc[
        truth["include_sensitivity_analysis"].astype(str).str.lower().eq("true")
    ].copy()
    if args.video_ids:
        truth = truth.loc[truth["video_id"].isin(args.video_ids)].copy()
    radii = list(range(int(args.min_radius), int(args.max_radius) + 1))
    if args.base_radius not in radii:
        raise ValueError("base radius must be inside the extracted radius range")

    model = joblib.load(args.temp_model)
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1
    radius_tables: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict[str, object]] = []
    for row in truth.itertuples(index=False):
        video_id = str(row.video_id)
        video_dir = args.input_root / video_id
        cached_path = video_dir / f"{args.temperature_prefix}_temperatures.csv"
        table_path = video_dir / f"{args.radius_table_prefix}_temperatures.csv"
        cached = pd.read_csv(cached_path)
        if args.reuse_radius_tables and table_path.exists():
            table = pd.read_csv(table_path)
        else:
            print(f"Extracting radius table: {video_id}")
            table = extract_radius_table(
                video_dir,
                cached,
                model,
                radii=radii,
                min_temp=args.min_temp,
                batch_frames=args.batch_frames,
            )
            table.to_csv(table_path, index=False, encoding="utf-8-sig")
        radius_tables[video_id] = table
        for side in ["left", "right"]:
            old = pd.to_numeric(cached[f"{side}_temp"], errors="coerce")
            new = pd.to_numeric(table[f"{side}_temp_r{args.base_radius}"], errors="coerce")
            valid = old.notna() & new.notna()
            validation_rows.append(
                {
                    "video_id": video_id,
                    "side": side,
                    "paired_frames": int(valid.sum()),
                    "fixed_radius20_mae": float(np.mean(np.abs(old[valid] - new[valid])))
                    if valid.any()
                    else math.nan,
                    "fixed_radius20_max_abs": float(np.max(np.abs(old[valid] - new[valid])))
                    if valid.any()
                    else math.nan,
                }
            )

    config = robust_config("adaptive_roi_candidate")
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for policy in policies(args.base_radius):
        current_rows: list[dict[str, object]] = []
        for row in truth.itertuples(index=False):
            video_id = str(row.video_id)
            spacing_cv = nostril_spacing_cv(radius_tables[video_id])
            exponent = float(policy["exponent"])
            damping_threshold = policy.get("spacing_cv_threshold")
            damped = bool(
                damping_threshold is not None
                and math.isfinite(spacing_cv)
                and spacing_cv > float(damping_threshold)
            )
            if damped:
                exponent = float(policy["damped_exponent"])
            temp_df, selected_radius = radius_policy_table(
                radius_tables[video_id],
                base_radius=args.base_radius,
                exponent=exponent,
                clip_low=int(policy["clip_low"]),
                clip_high=int(policy["clip_high"]),
            )
            _, summary = rr.fuse_temperature_curve(
                temp_df, config, pd.Series(row._asdict())
            )
            predicted_count = int(summary["peaks"])
            predicted_rr = predicted_count / (float(row.duration_seconds) / 60.0)
            current_rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "video_id": video_id,
                    "manual_breath_count": int(row.breath_count),
                    "manual_rr_bpm": float(row.rr),
                    "include_primary_analysis": bool(row.include_primary_analysis),
                    "truth_reliability": str(row.truth_reliability),
                    "predicted_breath_count": predicted_count,
                    "predicted_rr_bpm": predicted_rr,
                    "count_error": predicted_count - int(row.breath_count),
                    "selected_fusion_mode": summary["selected_fusion_mode"],
                    "spacing_cv": spacing_cv,
                    "effective_roi_exponent": exponent,
                    "roi_exponent_damped": damped,
                    "radius_mean": float(np.mean(selected_radius)),
                    "radius_sd": float(np.std(selected_radius)),
                    "radius_min": int(np.min(selected_radius)),
                    "radius_max": int(np.max(selected_radius)),
                    "radius_changed_fraction": float(
                        np.mean(selected_radius != int(args.base_radius))
                    ),
                }
            )
        current = pd.DataFrame(current_rows)
        prediction_rows.extend(current_rows)
        metric_rows.append(
            metrics(
                current.loc[current["include_primary_analysis"]],
                policy,
                "primary_completed",
            )
        )
        metric_rows.append(metrics(current, policy, "sensitivity_all_numeric"))

    predictions = pd.DataFrame(prediction_rows)
    metric_table = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(validation_rows)
    predictions.to_csv(args.output_dir / "lindian_adaptive_roi_predictions.csv", index=False)
    metric_table.to_csv(args.output_dir / "lindian_adaptive_roi_metrics.csv", index=False)
    validation.to_csv(args.output_dir / "lindian_adaptive_roi_radius20_validation.csv", index=False)
    print(
        metric_table.loc[metric_table["analysis_set"].eq("primary_completed")]
        .sort_values(["rr_r2", "rr_mae_bpm"], ascending=[False, True])
        .to_string(index=False)
    )
    print(validation.describe(include="all").to_string())


if __name__ == "__main__":
    main()
