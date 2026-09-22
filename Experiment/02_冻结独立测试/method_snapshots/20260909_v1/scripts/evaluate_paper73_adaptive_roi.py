"""Apply the frozen adaptive-ROI policy to the authoritative 73-video set."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import paper_repro_rr as rr
from evaluate_lindian_adaptive_roi import (
    extract_radius_table,
    nostril_spacing_cv,
    radius_policy_table,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    input_root = data_root / "al_images"
    assets = input_root / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument(
        "--truth", type=Path, default=data_root / "video" / "temperature_curves.csv"
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=input_root / "paper_repro_summary.csv"
    )
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=repo_root
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
    )
    parser.add_argument("--temperature-prefix", default="paper_repro")
    parser.add_argument("--radius-table-prefix", default="paper73_motion_roi_radius")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--min-radius", type=int, default=14)
    parser.add_argument("--max-radius", type=int, default=28)
    parser.add_argument("--base-radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--batch-frames", type=int, default=24)
    parser.add_argument("--reuse-radius-tables", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def config_from_baseline(
    row: pd.Series, output_prefix: str, *, reselect_fusion: bool = False
) -> rr.ReproConfig:
    edge_completion = str(row.get("edge_peak_completion", "none"))
    if int(float(row.get("edge_peak_added", 0))) == 0:
        edge_completion = "none"
    return replace(
        rr.fast_fusion_quality_config(as_bool(row.get("repair_missing", True))),
        fusion_mode="adaptive" if reselect_fusion else str(row["selected_fusion_mode"]),
        smooth_window=int(float(row["smooth_window"])),
        peak_distance=int(float(row["peak_distance"])),
        peak_prominence=float(row["selected_peak_prominence"]),
        adaptive_peak_prominence=False,
        min_rr_bpm=float(row.get("min_rr_bpm", 40.0)),
        max_rr_bpm=float(row.get("max_rr_bpm", 90.0)),
        merge_shallow_peaks=as_bool(row.get("merge_shallow_peaks", True)),
        merge_peak_gap=int(float(row.get("merge_peak_gap", 6))),
        merge_valley_relief=float(row.get("merge_valley_relief", 0.10)),
        adaptive_peak_retuning=False,
        short_sparse_edge_completion=False,
        edge_peak_completion=edge_completion,
        edge_peak_min_gap=int(float(row.get("edge_peak_min_gap", 5))),
        edge_peak_min_relief=float(row.get("edge_peak_min_relief", 0.5)),
        source_peak_gate=False,
        use_truth_duration_for_rr=True,
        limit_to_truth_duration=False,
        output_prefix=output_prefix,
        overwrite=True,
    )


def policy_rows() -> list[dict[str, object]]:
    result = [
        {
            "policy_id": "fixed_radius20_reextracted",
            "exponent": 0.0,
            "clip_low": 20,
            "clip_high": 20,
            "reselect_fusion": False,
        },
        {
            "policy_id": "adaptive_roi_linear_clip16_24",
            "exponent": 1.0,
            "clip_low": 16,
            "clip_high": 24,
            "reselect_fusion": False,
        },
        {
            "policy_id": "adaptive_roi_damped_cv0.06",
            "exponent": 1.0,
            "clip_low": 16,
            "clip_high": 24,
            "spacing_cv_threshold": 0.06,
            "damped_exponent": 0.5,
            "reselect_fusion": False,
        },
        {
            "policy_id": "fixed_radius20_adaptive_fusion_control",
            "exponent": 0.0,
            "clip_low": 20,
            "clip_high": 20,
            "reselect_fusion": True,
        },
        {
            "policy_id": "adaptive_roi_linear_clip16_24_adaptive_fusion",
            "exponent": 1.0,
            "clip_low": 16,
            "clip_high": 24,
            "reselect_fusion": True,
        },
        {
            "policy_id": "adaptive_roi_damped_cv0.06_adaptive_fusion",
            "exponent": 1.0,
            "clip_low": 16,
            "clip_high": 24,
            "spacing_cv_threshold": 0.06,
            "damped_exponent": 0.5,
            "reselect_fusion": True,
        },
    ]
    for radius in range(14, 29):
        if radius == 20:
            continue
        result.append(
            {
                "policy_id": f"fixed_radius{radius}_global",
                "exponent": 0.0,
                "clip_low": radius,
                "clip_high": radius,
                "reselect_fusion": False,
            }
        )
    return result


def metric_row(table: pd.DataFrame, policy: dict[str, object]) -> dict[str, object]:
    truth_rr = table["truth_rr"].to_numpy(dtype=float)
    predicted_rr = table["predicted_rr_bpm"].to_numpy(dtype=float)
    count_error = table["count_error"].to_numpy(dtype=float)
    rr_error = predicted_rr - truth_rr
    abs_count_error = np.abs(count_error)
    return {
        **policy,
        "videos": int(len(table)),
        "rr_r2": rr.regression_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_pearson_r2": rr.pearson_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "count_mae": float(np.mean(abs_count_error)),
        "exact_count": int(np.sum(abs_count_error == 0)),
        "within_one_count": int(np.sum(abs_count_error <= 1)),
        "count_error_ge_two": int(np.sum(abs_count_error >= 2)),
        "changed_videos": int(table["prediction_changed"].sum()),
        "improved_videos": int((table["abs_count_error_change"] < 0).sum()),
        "worsened_videos": int((table["abs_count_error_change"] > 0).sum()),
        "damped_videos": int(table["roi_exponent_damped"].sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    baseline = pd.read_csv(args.baseline_summary, dtype={"video_id": str})
    if len(truth) != 73 or truth["video_id"].nunique() != 73:
        raise ValueError("Truth table must contain exactly 73 unique video IDs")
    if set(truth["video_id"]) != set(baseline["video_id"]):
        raise ValueError("Baseline summary IDs do not match the 73-video truth table")
    truth_by_id = truth.set_index("video_id", drop=False)
    baseline_by_id = baseline.set_index("video_id", drop=False)

    replay_rows: list[dict[str, object]] = []
    for video_id in truth["video_id"]:
        temp_path = args.input_root / video_id / f"{args.temperature_prefix}_temperatures.csv"
        temp_df = pd.read_csv(temp_path)
        truth_row = truth_by_id.loc[video_id]
        baseline_row = baseline_by_id.loc[video_id]
        analysis_limit = int(float(baseline_row["analysis_frame_limit"]))
        temp_df = temp_df.iloc[:analysis_limit].copy()
        config = config_from_baseline(baseline_row, "paper73_replay")
        _, summary = rr.fuse_temperature_curve(temp_df, config, truth_row)
        adaptive_config = config_from_baseline(
            baseline_row, "paper73_replay_adaptive_fusion", reselect_fusion=True
        )
        _, adaptive_summary = rr.fuse_temperature_curve(
            temp_df, adaptive_config, truth_row
        )
        replay_rows.append(
            {
                "video_id": video_id,
                "baseline_count": int(float(baseline_row["peaks"])),
                "replayed_count": int(summary["peaks"]),
                "count_match": int(float(baseline_row["peaks"])) == int(summary["peaks"]),
                "baseline_fusion_mode": str(baseline_row["selected_fusion_mode"]),
                "replayed_fusion_mode": str(summary["selected_fusion_mode"]),
                "baseline_peak_retune_rule": str(baseline_row["peak_retune_rule"]),
                "adaptive_replay_count": int(adaptive_summary["peaks"]),
                "adaptive_replay_count_match": int(float(baseline_row["peaks"]))
                == int(adaptive_summary["peaks"]),
                "adaptive_replay_fusion_mode": str(
                    adaptive_summary["selected_fusion_mode"]
                ),
            }
        )
    replay = pd.DataFrame(replay_rows)
    replay_path = args.output_dir / "paper73_adaptive_roi_baseline_replay.csv"
    replay.to_csv(replay_path, index=False)
    mismatches = replay.loc[~replay["count_match"]]
    print(f"Baseline replay matches: {int(replay['count_match'].sum())}/73")
    print(
        "Adaptive-fusion replay matches: "
        f"{int(replay['adaptive_replay_count_match'].sum())}/73"
    )
    if not mismatches.empty:
        print(mismatches.to_string(index=False))
        raise ValueError("Frozen baseline replay did not reproduce all 73 counts")
    if args.replay_only:
        return

    radii = list(range(args.min_radius, args.max_radius + 1))
    model = joblib.load(args.temp_model)
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1
    radius_tables: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict[str, object]] = []
    for video_id in truth["video_id"]:
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

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for policy in policy_rows():
        rows: list[dict[str, object]] = []
        for video_id in truth["video_id"]:
            truth_row = truth_by_id.loc[video_id]
            baseline_row = baseline_by_id.loc[video_id]
            analysis_limit = int(float(baseline_row["analysis_frame_limit"]))
            analysis_radius_table = radius_tables[video_id].iloc[:analysis_limit].copy()
            spacing_cv = nostril_spacing_cv(analysis_radius_table)
            exponent = float(policy["exponent"])
            threshold = policy.get("spacing_cv_threshold")
            damped = bool(
                threshold is not None
                and math.isfinite(spacing_cv)
                and spacing_cv > float(threshold)
            )
            if damped:
                exponent = float(policy["damped_exponent"])
            temp_df, radius = radius_policy_table(
                analysis_radius_table,
                base_radius=args.base_radius,
                exponent=exponent,
                clip_low=int(policy["clip_low"]),
                clip_high=int(policy["clip_high"]),
            )
            config = config_from_baseline(
                baseline_row,
                "paper73_adaptive_roi_candidate",
                reselect_fusion=bool(policy.get("reselect_fusion", False)),
            )
            _, summary = rr.fuse_temperature_curve(temp_df, config, truth_row)
            predicted_count = int(summary["peaks"])
            baseline_count = int(float(baseline_row["peaks"]))
            truth_count = int(float(truth_row["breath_count"]))
            baseline_abs_error = abs(baseline_count - truth_count)
            candidate_abs_error = abs(predicted_count - truth_count)
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "video_id": video_id,
                    "truth_count": truth_count,
                    "truth_rr": float(truth_row["rr"]),
                    "duration_seconds": float(truth_row["duration_seconds"]),
                    "baseline_count": baseline_count,
                    "baseline_rr_bpm": float(baseline_row["rr_bpm"]),
                    "baseline_count_error": baseline_count - truth_count,
                    "predicted_count": predicted_count,
                    "predicted_rr_bpm": float(summary["rr_bpm"]),
                    "count_error": predicted_count - truth_count,
                    "prediction_changed": predicted_count != baseline_count,
                    "abs_count_error_change": candidate_abs_error - baseline_abs_error,
                    "selected_fusion_mode": summary["selected_fusion_mode"],
                    "spacing_cv": spacing_cv,
                    "effective_roi_exponent": exponent,
                    "roi_exponent_damped": damped,
                    "radius_mean": float(np.mean(radius)),
                    "radius_sd": float(np.std(radius)),
                    "radius_min": int(np.min(radius)),
                    "radius_max": int(np.max(radius)),
                }
            )
        table = pd.DataFrame(rows)
        prediction_rows.extend(rows)
        metric_rows.append(metric_row(table, policy))

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["worsened_videos", "rr_r2", "rr_mae_bpm"], ascending=[True, False, True]
    )
    validation = pd.DataFrame(validation_rows)
    predictions.to_csv(args.output_dir / "paper73_adaptive_roi_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "paper73_adaptive_roi_metrics.csv", index=False)
    validation.to_csv(args.output_dir / "paper73_adaptive_roi_radius20_validation.csv", index=False)
    print(metrics.to_string(index=False))
    print(validation.describe().to_string())


if __name__ == "__main__":
    main()
