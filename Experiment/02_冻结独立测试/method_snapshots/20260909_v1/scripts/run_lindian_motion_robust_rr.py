"""Run the frozen motion-robust adaptive-ROI RR policy from radius tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import paper_repro_rr as rr
from evaluate_lindian_adaptive_roi import nostril_spacing_cv, radius_policy_table
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
    parser.add_argument("--radius-table-prefix", default="lindian_motion_roi_radius")
    parser.add_argument("--output-prefix", default="lindian_anchored30_motion_robust")
    parser.add_argument("--base-radius", type=int, default=20)
    parser.add_argument("--min-radius", type=int, default=16)
    parser.add_argument("--max-radius", type=int, default=24)
    parser.add_argument("--spacing-cv-threshold", type=float, default=0.06)
    parser.add_argument("--linear-exponent", type=float, default=1.0)
    parser.add_argument("--damped-exponent", type=float, default=0.5)
    parser.add_argument("--unreliable-spacing-cv", type=float, default=0.20)
    parser.add_argument("--min-motion-stable-fraction", type=float, default=0.75)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    if args.video_ids:
        truth = truth.loc[truth["video_id"].isin(args.video_ids)].copy()
        missing = sorted(set(args.video_ids) - set(truth["video_id"]))
        if missing:
            raise ValueError(f"Missing truth rows: {missing}")
    config = robust_config(args.output_prefix)
    summaries: list[dict[str, object]] = []
    for truth_row in truth.itertuples(index=False):
        video_id = str(truth_row.video_id)
        video_dir = args.input_root / video_id
        radius_table_path = video_dir / f"{args.radius_table_prefix}_temperatures.csv"
        if not radius_table_path.exists():
            raise FileNotFoundError(radius_table_path)
        print(f"Processing {video_id} ...")
        radius_table = pd.read_csv(radius_table_path)
        spacing_cv = nostril_spacing_cv(radius_table)
        damped = bool(
            math.isfinite(spacing_cv) and spacing_cv > float(args.spacing_cv_threshold)
        )
        exponent = float(args.damped_exponent if damped else args.linear_exponent)
        temp_df, selected_radius = radius_policy_table(
            radius_table,
            base_radius=args.base_radius,
            exponent=exponent,
            clip_low=args.min_radius,
            clip_high=args.max_radius,
        )
        truth_series = pd.Series(truth_row._asdict())
        curve, summary = rr.fuse_temperature_curve(temp_df, config, truth_series)
        temperature_csv = video_dir / f"{args.output_prefix}_temperatures.csv"
        curve_csv = video_dir / f"{args.output_prefix}_curve.csv"
        curve_png = video_dir / f"{args.output_prefix}_curve.png"
        review_png = video_dir / f"{args.output_prefix}_peak_review.png"
        temp_df.to_csv(temperature_csv, index=False, encoding="utf-8-sig")
        curve["raw_frames"] = len(temp_df)
        curve["analysis_frame_limit"] = len(temp_df)
        curve["analysis_window_source"] = "adaptive_roi_radius_table"
        curve["raw_duration_seconds"] = float(len(temp_df) / config.fps)
        curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
        rr.plot_outputs(curve, video_id, summary, curve_png)
        rr.plot_peak_review(curve, video_id, summary, review_png)
        stable_fraction = float(summary["motion_stable_fraction"])
        motion_reliable = bool(
            (not math.isfinite(spacing_cv) or spacing_cv <= args.unreliable_spacing_cv)
            and stable_fraction >= args.min_motion_stable_fraction
        )
        summaries.append(
            {
                "video_id": video_id,
                **summary,
                "adaptive_roi_base_radius": int(args.base_radius),
                "adaptive_roi_min_radius": int(args.min_radius),
                "adaptive_roi_max_radius": int(args.max_radius),
                "adaptive_roi_spacing_cv": spacing_cv,
                "adaptive_roi_spacing_cv_threshold": float(args.spacing_cv_threshold),
                "adaptive_roi_exponent": exponent,
                "adaptive_roi_damped": str(damped),
                "adaptive_roi_radius_mean": float(np.mean(selected_radius)),
                "adaptive_roi_radius_sd": float(np.std(selected_radius)),
                "adaptive_roi_radius_min_used": int(np.min(selected_radius)),
                "adaptive_roi_radius_max_used": int(np.max(selected_radius)),
                "motion_quality_reliable": str(motion_reliable),
                "motion_quality_status": (
                    "motion_acceptable_damped"
                    if motion_reliable and damped
                    else "motion_acceptable_linear"
                    if motion_reliable
                    else "motion_unreliable_review_required"
                ),
                "raw_frames": len(temp_df),
                "analysis_frame_limit": len(temp_df),
                "analysis_window_source": "adaptive_roi_radius_table",
                "raw_duration_seconds": float(len(temp_df) / config.fps),
                "radius_table_csv": str(radius_table_path.resolve()),
                "temperature_csv": str(temperature_csv.resolve()),
                "curve_csv": str(curve_csv.resolve()),
                "curve_png": str(curve_png.resolve()),
                "review_png": str(review_png.resolve()),
                "status": "motion_robust_adaptive_roi_guarded_fusion_source_gate",
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = args.input_root / f"{args.output_prefix}_summary.csv"
    metrics_path = args.input_root / f"{args.output_prefix}_metrics.csv"
    error_path = args.input_root / f"{args.output_prefix}_error_report.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    metrics = rr.write_metrics_report(summary_df, metrics_path, args.output_prefix)
    rr.write_error_report(summary_df, error_path)
    print(f"Saved summary: {summary_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved error report: {error_path.resolve()}")
    if not metrics.empty:
        print(f"RR R^2: {float(metrics.iloc[0]['rr_r2']):.6f}")


if __name__ == "__main__":
    main()
