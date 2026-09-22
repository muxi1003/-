"""Evaluate FFT/Butterworth/peak counting on 49 Lindian fixed 30-second clips."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import paper_repro_rr as rr
from evaluate_paper73_butterworth_spectral import (
    PAPER_LIKE_CUTOFF_HZ,
    PAPER_LIKE_ORDER,
    SPECTRAL_MAX_HZ,
    SPECTRAL_MIN_HZ,
    config_id,
    high_frequency_energy_ratio,
    leave_one_out_selection,
    make_three_panel_figure,
    metric_row,
    parse_float_grid,
    parse_int_grid,
    spectral_features,
)
from compare_lindian_robust_rr import bootstrap_delta
from run_lindian_robust_rr import robust_config


TRANSFER_ORDER = 2
TRANSFER_CUTOFF_HZ = 1.75


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    input_root = data_root / "lindian_anchored30_frames"
    assets = data_root / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument(
        "--truth", type=Path, default=assets / "lindian_anchored30_manual_truth.csv"
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=input_root / "lindian_anchored30_motion_robust_summary.csv",
    )
    parser.add_argument(
        "--temperature-prefix", default="lindian_anchored30_motion_robust"
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--orders", default="2,4,6")
    parser.add_argument("--cutoffs-hz", default="1.5,1.75,2.0,2.25")
    parser.add_argument("--spectral-min-hz", type=float, default=SPECTRAL_MIN_HZ)
    parser.add_argument("--spectral-max-hz", type=float, default=SPECTRAL_MAX_HZ)
    parser.add_argument("--figure-video-id", default="zs197000")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def frozen_config(row: pd.Series, output_prefix: str, fps: float) -> rr.ReproConfig:
    edge_completion = str(row.get("edge_peak_completion", "none"))
    if int(float(row.get("edge_peak_added", 0))) == 0:
        edge_completion = "none"
    return replace(
        robust_config(output_prefix),
        fps=fps,
        fusion_mode=str(row["selected_fusion_mode"]),
        smooth_window=int(float(row["smooth_window"])),
        smoothing_method="moving_average",
        peak_distance=int(float(row["peak_distance"])),
        peak_prominence=float(row["selected_peak_prominence"]),
        adaptive_peak_prominence=False,
        merge_shallow_peaks=as_bool(row.get("merge_shallow_peaks", True)),
        merge_peak_gap=int(float(row.get("merge_peak_gap", 6))),
        merge_valley_relief=float(row.get("merge_valley_relief", 0.10)),
        adaptive_peak_retuning=False,
        edge_peak_completion=edge_completion,
        edge_peak_min_gap=int(float(row.get("edge_peak_min_gap", 5))),
        edge_peak_min_relief=float(row.get("edge_peak_min_relief", 0.5)),
        source_peak_gate=as_bool(row.get("source_peak_gate", True)),
        source_peak_support_radius=int(float(row.get("source_peak_support_radius", 3))),
        use_truth_duration_for_rr=True,
        limit_to_truth_duration=False,
        overwrite=True,
    )


def add_metric_rows(
    rows: list[dict[str, object]],
    table: pd.DataFrame,
    method: str,
    prediction_rr_column: str,
    prediction_count_column: str,
    *,
    evaluation: str,
    order: float = math.nan,
    cutoff_hz: float = math.nan,
) -> None:
    analyses = {
        "primary_completed": table.loc[table["include_primary_analysis"]].copy(),
        "sensitivity_all_numeric": table.copy(),
    }
    for analysis_set, subset in analyses.items():
        result = metric_row(
            method,
            subset["truth_rr"].to_numpy(float),
            subset[prediction_rr_column].to_numpy(float),
            subset["truth_count"].to_numpy(float),
            subset[prediction_count_column].to_numpy(float),
            order=order,
            cutoff_hz=cutoff_hz,
            evaluation=evaluation,
        )
        result["analysis_set"] = analysis_set
        rows.append(result)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    loo: pd.DataFrame,
    paired: pd.DataFrame,
    fps: float,
) -> None:
    fixed_id = config_id(PAPER_LIKE_ORDER, PAPER_LIKE_CUTOFF_HZ)
    transfer_id = config_id(TRANSFER_ORDER, TRANSFER_CUTOFF_HZ)
    lines = [
        "# Lindian49 Butterworth and Spectral Analysis",
        "",
        "Status: `internal_fixed_window_validation_not_independent_external_validation`",
        "",
        "The current motion-robust adaptive-ROI temperature tables are reused. Fusion side, "
        "peak distance, prominence, source support gate, endpoint completion, and manual "
        "duration are frozen per clip; only the smoother is replaced.",
        "",
        "The order-4, 2 Hz row is the pre-specified paper-like engineering replication. "
        "The order-2, 1.75 Hz row was selected on the separate 73-video sensitivity analysis "
        "before this 49-clip run. The remaining grid is same-cohort sensitivity analysis.",
        "",
    ]
    for analysis_set in ("primary_completed", "sensitivity_all_numeric"):
        group = metrics.loc[metrics["analysis_set"].eq(analysis_set)]
        baseline = group.loc[group["method"].eq("motion_robust_moving_average_baseline")].iloc[0]
        fixed = group.loc[group["method"].eq(fixed_id)].iloc[0]
        transfer = group.loc[group["method"].eq(transfer_id)].iloc[0]
        spectral = group.loc[group["method"].eq("direct_fft_after_fixed_butterworth")].iloc[0]
        loo_metric = group.loc[
            group["method"].eq("leave_one_video_out_butterworth_selector")
        ].iloc[0]
        lines.extend(
            [
                f"## {analysis_set}",
                "",
                "| method | R2 | MAE (bpm) | RMSE (bpm) | exact | within one |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| motion-robust moving-average baseline | {baseline.rr_r2:.6f} | {baseline.rr_mae_bpm:.6f} | {baseline.rr_rmse_bpm:.6f} | {int(baseline.exact_count)}/{int(baseline.videos)} | {int(baseline.within_one_count)}/{int(baseline.videos)} |",
                f"| fixed Butterworth order 4, 2 Hz | {fixed.rr_r2:.6f} | {fixed.rr_mae_bpm:.6f} | {fixed.rr_rmse_bpm:.6f} | {int(fixed.exact_count)}/{int(fixed.videos)} | {int(fixed.within_one_count)}/{int(fixed.videos)} |",
                f"| transferred Butterworth order 2, 1.75 Hz | {transfer.rr_r2:.6f} | {transfer.rr_mae_bpm:.6f} | {transfer.rr_rmse_bpm:.6f} | {int(transfer.exact_count)}/{int(transfer.videos)} | {int(transfer.within_one_count)}/{int(transfer.videos)} |",
                f"| direct FFT dominant frequency | {spectral.rr_r2:.6f} | {spectral.rr_mae_bpm:.6f} | {spectral.rr_rmse_bpm:.6f} | {int(spectral.exact_count)}/{int(spectral.videos)} | {int(spectral.within_one_count)}/{int(spectral.videos)} |",
                f"| LOO-selected Butterworth | {loo_metric.rr_r2:.6f} | {loo_metric.rr_mae_bpm:.6f} | {loo_metric.rr_rmse_bpm:.6f} | {int(loo_metric.exact_count)}/{int(loo_metric.videos)} | {int(loo_metric.within_one_count)}/{int(loo_metric.videos)} |",
                "",
            ]
        )
        grid = group.loc[group["evaluation"].eq("same_cohort_sensitivity_grid")].sort_values(
            ["rr_rmse_bpm", "rr_mae_bpm", "method"]
        )
        if not grid.empty:
            best = grid.iloc[0]
            lines.extend(
                [
                    f"Best same-cohort grid row: `{best.method}` (R2={best.rr_r2:.6f}, "
                    f"RMSE={best.rr_rmse_bpm:.6f}); diagnostic only.",
                    "",
                ]
            )
        for candidate_id, label in (
            (fixed_id, "fixed order 4, 2 Hz"),
            (config_id(2, 1.5), "order 2, 1.5 Hz"),
        ):
            delta = paired.loc[
                paired["analysis_set"].eq(analysis_set)
                & paired["candidate_id"].eq(candidate_id)
                & paired["metric"].eq("rr_r2")
            ].iloc[0]
            lines.append(
                f"Paired bootstrap R2 delta for {label}: "
                f"{delta['delta_robust_minus_baseline']:+.6f} "
                f"(95% CI {delta['delta_ci_lower']:+.6f} to "
                f"{delta['delta_ci_upper']:+.6f})."
            )
        lines.append("")

    fixed = predictions.loc[predictions["config_id"].eq(fixed_id)]
    changed = int(fixed["prediction_changed"].sum())
    improved = int((fixed["abs_count_error_change"] < 0).sum())
    worsened = int((fixed["abs_count_error_change"] > 0).sum())
    resolution = fixed["rayleigh_rr_resolution_bpm"].to_numpy(float)
    lines.extend(
        [
            "## Diagnostics",
            "",
            f"The fixed order-4, 2 Hz filter changed {changed}/49 counts: {improved} improved "
            f"and {worsened} worsened by absolute count error.",
            f"Median Rayleigh RR resolution is {float(np.median(resolution)):.3f} bpm "
            f"(range {float(np.min(resolution)):.3f}-{float(np.max(resolution)):.3f} bpm).",
            "",
            "LOO selection counts: "
            + "; ".join(
                f"{analysis_set}/{name}={count}"
                for (analysis_set, name), count in loo.groupby(
                    ["analysis_set", "selected_config_id"]
                ).size().sort_index().items()
            )
            + ".",
            "",
            "## Claim boundary",
            "",
            "The primary set excludes uncertain manual counts. The all-numeric set retains all "
            "49 clips as sensitivity analysis. These windows come from the same Lindian farm and "
            "overlap the development domain, so they are not independent external validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders = parse_int_grid(args.orders)
    cutoffs = parse_float_grid(args.cutoffs_hz)
    for order in (PAPER_LIKE_ORDER, TRANSFER_ORDER):
        if order not in orders:
            orders.append(order)
    for cutoff in (PAPER_LIKE_CUTOFF_HZ, TRANSFER_CUTOFF_HZ):
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    orders.sort()
    cutoffs.sort()
    if any(cutoff >= args.fps / 2.0 for cutoff in cutoffs):
        raise ValueError("Every cutoff must be below the Nyquist frequency")

    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    baseline = pd.read_csv(args.baseline_summary, dtype={"video_id": str})
    if len(truth) != 49 or truth["video_id"].nunique() != 49:
        raise ValueError("Truth table must contain exactly 49 unique video IDs")
    if set(truth["video_id"]) != set(baseline["video_id"]):
        raise ValueError("Baseline summary IDs do not match the 49-video truth table")
    truth["include_primary_analysis"] = truth["include_primary_analysis"].map(as_bool)
    truth_by_id = truth.set_index("video_id", drop=False)
    baseline_by_id = baseline.set_index("video_id", drop=False)

    replay_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    figure_payload: dict[str, object] | None = None
    for index, video_id in enumerate(truth["video_id"], start=1):
        print(f"[{index}/49] {video_id}")
        truth_row = truth_by_id.loc[video_id]
        baseline_row = baseline_by_id.loc[video_id]
        temperature_path = (
            args.input_root / video_id / f"{args.temperature_prefix}_temperatures.csv"
        )
        temp_df = pd.read_csv(temperature_path)
        base_config = frozen_config(
            baseline_row, "lindian49_butterworth_spectral_replay", args.fps
        )
        _, replay_summary = rr.fuse_temperature_curve(
            temp_df.copy(), base_config, truth_row
        )
        baseline_count = int(float(baseline_row["peaks"]))
        replay_count = int(replay_summary["peaks"])
        replay_rows.append(
            {
                "video_id": video_id,
                "baseline_count": baseline_count,
                "replayed_count": replay_count,
                "count_match": baseline_count == replay_count,
            }
        )
        if replay_count != baseline_count:
            raise ValueError(
                f"Baseline replay mismatch for {video_id}: {replay_count} != {baseline_count}"
            )

        for order in orders:
            for cutoff_hz in cutoffs:
                candidate_config = replace(
                    base_config,
                    smoothing_method="butterworth",
                    butterworth_order=order,
                    butterworth_cutoff_hz=cutoff_hz,
                )
                curve, summary = rr.fuse_temperature_curve(
                    temp_df.copy(), candidate_config, truth_row
                )
                raw = curve["fused_norm"].to_numpy(float)
                filtered = curve["smoothed_norm"].dropna().to_numpy(float)
                raw_spectral = spectral_features(
                    raw, args.fps, args.spectral_min_hz, args.spectral_max_hz
                )
                filtered_spectral = spectral_features(
                    filtered, args.fps, args.spectral_min_hz, args.spectral_max_hz
                )
                truth_count = int(float(truth_row["breath_count"]))
                predicted_count = int(summary["peaks"])
                candidate_rows.append(
                    {
                        "config_id": config_id(order, cutoff_hz),
                        "video_id": video_id,
                        "butterworth_order": order,
                        "butterworth_cutoff_hz": cutoff_hz,
                        "frames": len(raw),
                        "duration_seconds": float(truth_row["duration_seconds"]),
                        "truth_count": truth_count,
                        "truth_rr": float(truth_row["manual_rr_bpm"]),
                        "truth_reliability": str(truth_row["truth_reliability"]),
                        "include_primary_analysis": bool(
                            truth_row["include_primary_analysis"]
                        ),
                        "baseline_count": baseline_count,
                        "baseline_rr_bpm": float(baseline_row["rr_bpm"]),
                        "predicted_count": predicted_count,
                        "predicted_rr_bpm": float(summary["rr_bpm"]),
                        "count_error": predicted_count - truth_count,
                        "prediction_changed": predicted_count != baseline_count,
                        "abs_count_error_change": abs(predicted_count - truth_count)
                        - abs(baseline_count - truth_count),
                        "selected_fusion_mode": str(summary["selected_fusion_mode"]),
                        "peak_distance": int(summary["peak_distance"]),
                        "selected_peak_prominence": float(
                            summary["selected_peak_prominence"]
                        ),
                        "raw_dominant_frequency_hz": raw_spectral[
                            "dominant_frequency_hz"
                        ],
                        "raw_spectral_rr_bpm": raw_spectral["spectral_rr_bpm"],
                        "raw_spectral_concentration": raw_spectral[
                            "spectral_concentration"
                        ],
                        "filtered_dominant_frequency_hz": filtered_spectral[
                            "dominant_frequency_hz"
                        ],
                        "filtered_spectral_rr_bpm": filtered_spectral[
                            "spectral_rr_bpm"
                        ],
                        "filtered_spectral_concentration": filtered_spectral[
                            "spectral_concentration"
                        ],
                        "raw_high_frequency_energy_ratio": high_frequency_energy_ratio(
                            raw, args.fps, cutoff_hz
                        ),
                        "filtered_high_frequency_energy_ratio": high_frequency_energy_ratio(
                            filtered, args.fps, cutoff_hz
                        ),
                        "rayleigh_rr_resolution_bpm": 60.0 * args.fps / len(raw),
                    }
                )
                if (
                    video_id == args.figure_video_id
                    and order == PAPER_LIKE_ORDER
                    and math.isclose(cutoff_hz, PAPER_LIKE_CUTOFF_HZ)
                ):
                    figure_payload = {
                        "raw": raw,
                        "filtered": filtered,
                        "selected_peaks": np.flatnonzero(
                            curve["is_peak"].to_numpy(bool)
                        ),
                        "peak_distance": int(summary["peak_distance"]),
                        "peak_prominence": float(summary["selected_peak_prominence"]),
                    }

    replay = pd.DataFrame(replay_rows)
    predictions = pd.DataFrame(candidate_rows)
    metrics_rows: list[dict[str, object]] = []
    fixed_id = config_id(PAPER_LIKE_ORDER, PAPER_LIKE_CUTOFF_HZ)
    fixed = predictions.loc[predictions["config_id"].eq(fixed_id)].copy()
    add_metric_rows(
        metrics_rows,
        fixed,
        "motion_robust_moving_average_baseline",
        "baseline_rr_bpm",
        "baseline_count",
        evaluation="current_frozen_lindian49_baseline",
    )
    for candidate_id, group in predictions.groupby("config_id", sort=True):
        if candidate_id == fixed_id:
            evaluation = "pre_specified_paper_like_internal_probe"
        elif candidate_id == config_id(TRANSFER_ORDER, TRANSFER_CUTOFF_HZ):
            evaluation = "transferred_from_paper73_sensitivity"
        else:
            evaluation = "same_cohort_sensitivity_grid"
        add_metric_rows(
            metrics_rows,
            group,
            str(candidate_id),
            "predicted_rr_bpm",
            "predicted_count",
            evaluation=evaluation,
            order=float(group["butterworth_order"].iloc[0]),
            cutoff_hz=float(group["butterworth_cutoff_hz"].iloc[0]),
        )

    fixed = fixed.copy()
    fixed["spectral_predicted_count"] = np.rint(
        fixed["filtered_spectral_rr_bpm"].to_numpy(float)
        * fixed["duration_seconds"].to_numpy(float)
        / 60.0
    )
    add_metric_rows(
        metrics_rows,
        fixed,
        "direct_fft_after_fixed_butterworth",
        "filtered_spectral_rr_bpm",
        "spectral_predicted_count",
        evaluation="secondary_frequency_domain_diagnostic",
        order=PAPER_LIKE_ORDER,
        cutoff_hz=PAPER_LIKE_CUTOFF_HZ,
    )

    loo_tables: list[pd.DataFrame] = []
    for analysis_set, subset in (
        ("primary_completed", predictions.loc[predictions["include_primary_analysis"]]),
        ("sensitivity_all_numeric", predictions),
    ):
        selected = leave_one_out_selection(subset).copy()
        selected["analysis_set"] = analysis_set
        loo_tables.append(selected)
        result = metric_row(
            "leave_one_video_out_butterworth_selector",
            selected["truth_rr"].to_numpy(float),
            selected["predicted_rr_bpm"].to_numpy(float),
            selected["truth_count"].to_numpy(float),
            selected["predicted_count"].to_numpy(float),
            evaluation="internal_leave_one_video_out_parameter_selection",
        )
        result["analysis_set"] = analysis_set
        metrics_rows.append(result)
    loo = pd.concat(loo_tables, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)

    paired_tables: list[pd.DataFrame] = []
    paired_candidates = [
        fixed_id,
        config_id(TRANSFER_ORDER, TRANSFER_CUTOFF_HZ),
        config_id(2, 1.5),
    ]
    for candidate_index, candidate_id in enumerate(paired_candidates):
        candidate = predictions.loc[predictions["config_id"].eq(candidate_id)].copy()
        candidate["manual_rr_bpm"] = candidate["truth_rr"]
        candidate["baseline_count_error"] = (
            candidate["baseline_count"] - candidate["truth_count"]
        )
        candidate["robust_rr_bpm"] = candidate["predicted_rr_bpm"]
        candidate["robust_count_error"] = (
            candidate["predicted_count"] - candidate["truth_count"]
        )
        for analysis_index, (analysis_set, subset) in enumerate(
            (
                (
                    "primary_completed",
                    candidate.loc[candidate["include_primary_analysis"]],
                ),
                ("sensitivity_all_numeric", candidate),
            )
        ):
            comparison = bootstrap_delta(
                subset,
                analysis_set=analysis_set,
                iterations=args.bootstrap_iterations,
                seed=args.seed + candidate_index * 10 + analysis_index,
            )
            comparison.insert(1, "candidate_id", candidate_id)
            paired_tables.append(comparison)
    paired = pd.concat(paired_tables, ignore_index=True)

    replay_path = args.output_dir / "lindian49_butterworth_spectral_baseline_replay.csv"
    predictions_path = args.output_dir / "lindian49_butterworth_spectral_predictions.csv"
    metrics_path = args.output_dir / "lindian49_butterworth_spectral_metrics.csv"
    loo_path = args.output_dir / "lindian49_butterworth_spectral_loo_predictions.csv"
    paired_path = args.output_dir / "lindian49_butterworth_spectral_paired_bootstrap.csv"
    report_path = args.output_dir / "lindian49_butterworth_spectral_report.md"
    figure_path = (
        args.output_dir
        / f"lindian49_butterworth_spectral_{args.figure_video_id}_three_panel.png"
    )
    replay.to_csv(replay_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    loo.to_csv(loo_path, index=False)
    paired.to_csv(paired_path, index=False)
    write_report(report_path, metrics, predictions, loo, paired, args.fps)
    if figure_payload is None:
        raise ValueError(f"Figure video was not found: {args.figure_video_id}")
    make_three_panel_figure(
        figure_path,
        args.figure_video_id,
        np.asarray(figure_payload["raw"], dtype=float),
        np.asarray(figure_payload["filtered"], dtype=float),
        np.asarray(figure_payload["selected_peaks"], dtype=int),
        args.fps,
        int(figure_payload["peak_distance"]),
        float(figure_payload["peak_prominence"]),
        PAPER_LIKE_ORDER,
        PAPER_LIKE_CUTOFF_HZ,
    )

    print(f"Baseline replay matches: {int(replay['count_match'].sum())}/49")
    print(metrics.to_string(index=False))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
