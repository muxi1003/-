"""Run the frozen robust Lindian RR policy from cached temperature CSVs."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

import paper_repro_rr as rr


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
    parser.add_argument("--temperature-prefix", default="lindian_anchored30_repro")
    parser.add_argument("--output-prefix", default="lindian_anchored30_robust")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def robust_config(output_prefix: str) -> rr.ReproConfig:
    return replace(
        rr.fast_fusion_quality_config(True),
        fusion_mode="adaptive",
        fusion_interval_cv_weight=0.5,
        fusion_rescue_interval_cv_weight=1.1,
        fusion_rescue_outlier_threshold=3,
        source_peak_gate=True,
        source_peak_support_radius=3,
        min_rr_bpm=20.0,
        max_rr_bpm=100.0,
        limit_to_truth_duration=False,
        adaptive_peak_retuning=False,
        short_sparse_edge_completion=False,
        infer_global_offset=False,
        edge_peak_completion="none",
        output_prefix=output_prefix,
        overwrite=True,
    )


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    if args.video_ids:
        truth = truth.loc[truth["video_id"].isin(args.video_ids)].copy()
        missing = sorted(set(args.video_ids) - set(truth["video_id"]))
        if missing:
            raise ValueError(f"Missing truth rows: {missing}")
    if truth["video_id"].duplicated().any():
        raise ValueError("Truth table contains duplicate video IDs")

    config = robust_config(args.output_prefix)
    summaries: list[dict[str, object]] = []
    for truth_row in truth.itertuples(index=False):
        video_id = str(truth_row.video_id)
        video_dir = args.input_root / video_id
        temperature_csv = video_dir / f"{args.temperature_prefix}_temperatures.csv"
        if not temperature_csv.exists():
            raise FileNotFoundError(temperature_csv)
        print(f"Processing {video_id} ...")
        temp_df = pd.read_csv(temperature_csv)
        truth_series = pd.Series(truth_row._asdict())
        curve, summary = rr.fuse_temperature_curve(temp_df, config, truth_series)
        curve_csv = video_dir / f"{args.output_prefix}_curve.csv"
        curve_png = video_dir / f"{args.output_prefix}_curve.png"
        review_png = video_dir / f"{args.output_prefix}_peak_review.png"
        curve["raw_frames"] = len(temp_df)
        curve["analysis_frame_limit"] = len(temp_df)
        curve["analysis_window_source"] = "full_cached_temperature_csv"
        curve["raw_duration_seconds"] = (
            float(len(temp_df) / config.fps) if config.fps > 0 else math.nan
        )
        curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
        rr.plot_outputs(curve, video_id, summary, curve_png)
        rr.plot_peak_review(curve, video_id, summary, review_png)
        summaries.append(
            {
                "video_id": video_id,
                **summary,
                "raw_frames": len(temp_df),
                "analysis_frame_limit": len(temp_df),
                "analysis_window_source": "full_cached_temperature_csv",
                "raw_duration_seconds": float(len(temp_df) / config.fps),
                "temperature_csv": str(temperature_csv.resolve()),
                "curve_csv": str(curve_csv.resolve()),
                "curve_png": str(curve_png.resolve()),
                "review_png": str(review_png.resolve()),
                "status": "robust_guarded_fusion_source_gate",
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
