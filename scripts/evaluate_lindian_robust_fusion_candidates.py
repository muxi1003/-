"""Evaluate truth-independent robust fusion policies on cached Lindian temperatures."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import paper_repro_rr as rr


FUSION_MODES = ("mean", "max", "min", "left", "right")
AMPLITUDE_WEIGHTS = (1.5, 2.0)
INTERVAL_WEIGHTS = (0.5, 0.75, 1.0, 1.1, 1.25, 1.5)
CONSENSUS_WEIGHTS = (0.0, 0.05, 0.1, 0.2)
SOURCE_GATE_OPTIONS = (False, True)


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
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--source-radius", type=int, default=3)
    return parser.parse_args()


def frozen_config(mode: str) -> rr.ReproConfig:
    return replace(
        rr.fast_fusion_quality_config(True),
        fusion_mode=mode,
        min_rr_bpm=20.0,
        max_rr_bpm=100.0,
        limit_to_truth_duration=False,
        adaptive_peak_retuning=False,
        short_sparse_edge_completion=False,
        infer_global_offset=False,
        edge_peak_completion="none",
    )


def frame_peaks(curve: pd.DataFrame) -> list[int]:
    return curve.loc[curve["is_peak"], "frame_index"].astype(int).tolist()


def any_direct_detection(temp_df: pd.DataFrame) -> np.ndarray:
    left = temp_df.get("left_source", pd.Series("", index=temp_df.index)).astype(str)
    right = temp_df.get("right_source", pd.Series("", index=temp_df.index)).astype(str)
    return (left.eq("detected") | right.eq("detected")).to_numpy(dtype=bool)


def source_supported_peaks(
    peaks: list[int], direct: np.ndarray, radius: int
) -> tuple[list[int], list[int]]:
    kept: list[int] = []
    rejected: list[int] = []
    for peak in peaks:
        start = max(0, peak - radius)
        end = min(len(direct), peak + radius + 1)
        if bool(direct[start:end].any()):
            kept.append(peak)
        else:
            rejected.append(peak)
    return kept, rejected


def coefficient_of_determination(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return 1.0 - float(np.sum((prediction - truth) ** 2)) / denominator


def aggregate_metrics(table: pd.DataFrame) -> dict[str, float | int]:
    truth_rr = table["manual_rr_bpm"].to_numpy(dtype=float)
    predicted_rr = table["predicted_rr_bpm"].to_numpy(dtype=float)
    count_error = table["count_error"].to_numpy(dtype=float)
    rr_error = predicted_rr - truth_rr
    pearson = float(np.corrcoef(truth_rr, predicted_rr)[0, 1])
    return {
        "videos": len(table),
        "rr_r2": coefficient_of_determination(truth_rr, predicted_rr),
        "rr_pearson_r2": pearson**2,
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "rr_bias_bpm": float(np.mean(rr_error)),
        "count_mae": float(np.mean(np.abs(count_error))),
        "exact_count": int(np.sum(count_error == 0)),
        "within_one_count": int(np.sum(np.abs(count_error) <= 1)),
        "count_error_ge_two": int(np.sum(np.abs(count_error) >= 2)),
        "max_abs_count_error": int(np.max(np.abs(count_error))),
    }


def build_candidate_table(
    input_root: Path,
    truth: pd.DataFrame,
    source_radius: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for truth_row in truth.itertuples(index=False):
        video_id = str(truth_row.video_id)
        temp_path = input_root / video_id / "lindian_anchored30_repro_temperatures.csv"
        temp_df = pd.read_csv(temp_path)
        direct = any_direct_detection(temp_df)
        truth_series = pd.Series(truth_row._asdict())
        for mode in FUSION_MODES:
            curve, summary = rr.fuse_temperature_curve(
                temp_df, frozen_config(mode), truth_row=truth_series
            )
            peaks = frame_peaks(curve)
            gated_peaks, rejected = source_supported_peaks(peaks, direct, source_radius)
            rows.append(
                {
                    "video_id": video_id,
                    "fusion_mode": mode,
                    "manual_breath_count": int(truth_row.breath_count),
                    "duration_seconds": float(truth_row.duration_seconds),
                    "manual_rr_bpm": float(truth_row.manual_rr_bpm),
                    "include_primary_analysis": bool(truth_row.include_primary_analysis),
                    "truth_reliability": str(truth_row.truth_reliability),
                    "candidate_count": len(peaks),
                    "source_gated_count": len(gated_peaks),
                    "peak_frames": ";".join(map(str, peaks)),
                    "source_gated_peak_frames": ";".join(map(str, gated_peaks)),
                    "source_rejected_peak_frames": ";".join(map(str, rejected)),
                    "median_prominence": float(summary["selection_median_prominence"]),
                    "amplitude": float(summary["selection_amplitude"]),
                    "interval_cv": float(summary["selection_interval_cv"]),
                    "missing_rate": float(summary["selection_missing_rate"]),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates["consensus_count"] = candidates.groupby("video_id")[
        "source_gated_count"
    ].transform("median")
    return candidates


def policy_id(
    amplitude_weight: float,
    interval_weight: float,
    consensus_weight: float,
    source_gate: bool,
) -> str:
    gate = "gate" if source_gate else "no_gate"
    return (
        f"amp{amplitude_weight:g}_cv{interval_weight:g}_"
        f"cons{consensus_weight:g}_{gate}"
    )


def apply_policy(
    candidates: pd.DataFrame,
    *,
    amplitude_weight: float,
    interval_weight: float,
    consensus_weight: float,
    source_gate: bool,
) -> pd.DataFrame:
    table = candidates.copy()
    count_column = "source_gated_count" if source_gate else "candidate_count"
    table["policy_score"] = (
        table["median_prominence"]
        + amplitude_weight * table["amplitude"]
        - interval_weight * table["interval_cv"]
        - 2.0 * table["missing_rate"]
        - consensus_weight * (table[count_column] - table["consensus_count"]).abs()
    )
    selected = (
        table.sort_values(
            ["video_id", "policy_score", "interval_cv", "fusion_mode"],
            ascending=[True, False, True, True],
        )
        .groupby("video_id", as_index=False)
        .first()
    )
    selected["predicted_breath_count"] = selected[count_column].astype(int)
    selected["predicted_rr_bpm"] = (
        selected["predicted_breath_count"] * 60.0 / selected["duration_seconds"]
    )
    selected["count_error"] = (
        selected["predicted_breath_count"] - selected["manual_breath_count"]
    )
    selected["rr_error_bpm"] = (
        selected["predicted_rr_bpm"] - selected["manual_rr_bpm"]
    )
    selected["policy_id"] = policy_id(
        amplitude_weight, interval_weight, consensus_weight, source_gate
    )
    selected["amplitude_weight"] = amplitude_weight
    selected["interval_weight"] = interval_weight
    selected["consensus_weight"] = consensus_weight
    selected["source_gate"] = source_gate
    return selected


def score_candidates(
    candidates: pd.DataFrame,
    *,
    amplitude_weight: float,
    interval_weight: float,
) -> pd.DataFrame:
    table = candidates.copy()
    table["policy_score"] = (
        table["median_prominence"]
        + amplitude_weight * table["amplitude"]
        - interval_weight * table["interval_cv"]
        - 2.0 * table["missing_rate"]
    )
    return table


def select_best_scored(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.sort_values(
            ["video_id", "policy_score", "interval_cv", "fusion_mode"],
            ascending=[True, False, True, True],
        )
        .groupby("video_id", as_index=False)
        .first()
    )


def apply_guarded_policy(
    candidates: pd.DataFrame,
    *,
    outlier_threshold: int,
    source_gate: bool = True,
) -> pd.DataFrame:
    legacy = select_best_scored(
        score_candidates(candidates, amplitude_weight=2.0, interval_weight=0.5)
    )
    robust = select_best_scored(
        score_candidates(candidates, amplitude_weight=2.0, interval_weight=1.1)
    )
    count_column = "source_gated_count" if source_gate else "candidate_count"
    legacy = legacy.set_index("video_id", drop=False)
    robust = robust.set_index("video_id", drop=False)
    selected_rows: list[pd.Series] = []
    for video_id in legacy.index:
        baseline_row = legacy.loc[video_id].copy()
        robust_row = robust.loc[video_id].copy()
        median_count = float(baseline_row["consensus_count"])
        baseline_distance = abs(float(baseline_row[count_column]) - median_count)
        robust_distance = abs(float(robust_row[count_column]) - median_count)
        use_rescue = (
            baseline_distance >= outlier_threshold
            and robust_distance < baseline_distance
        )
        chosen = robust_row if use_rescue else baseline_row
        chosen["legacy_fusion_mode"] = baseline_row["fusion_mode"]
        chosen["legacy_candidate_count"] = int(baseline_row[count_column])
        chosen["fusion_rescue_applied"] = bool(use_rescue)
        chosen["legacy_consensus_distance"] = float(baseline_distance)
        chosen["robust_consensus_distance"] = float(robust_distance)
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["predicted_breath_count"] = selected[count_column].astype(int)
    selected["predicted_rr_bpm"] = (
        selected["predicted_breath_count"] * 60.0 / selected["duration_seconds"]
    )
    selected["count_error"] = (
        selected["predicted_breath_count"] - selected["manual_breath_count"]
    )
    selected["rr_error_bpm"] = (
        selected["predicted_rr_bpm"] - selected["manual_rr_bpm"]
    )
    selected["policy_id"] = (
        f"guarded_amp2_cv1.1_outlier{outlier_threshold}_"
        f"{'gate' if source_gate else 'no_gate'}"
    )
    selected["amplitude_weight"] = 2.0
    selected["interval_weight"] = 1.1
    selected["consensus_weight"] = 0.0
    selected["source_gate"] = source_gate
    selected["outlier_threshold"] = outlier_threshold
    return selected


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    truth["include_primary_analysis"] = (
        truth["include_primary_analysis"].astype(str).str.lower().eq("true")
    )
    candidates = build_candidate_table(args.input_root, truth, args.source_radius)

    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    for amplitude_weight in AMPLITUDE_WEIGHTS:
        for interval_weight in INTERVAL_WEIGHTS:
            for consensus_weight in CONSENSUS_WEIGHTS:
                for source_gate in SOURCE_GATE_OPTIONS:
                    predictions = apply_policy(
                        candidates,
                        amplitude_weight=amplitude_weight,
                        interval_weight=interval_weight,
                        consensus_weight=consensus_weight,
                        source_gate=source_gate,
                    )
                    prediction_tables.append(predictions)
                    for analysis_set, selected in (
                        (
                            "primary_completed",
                            predictions.loc[predictions["include_primary_analysis"]],
                        ),
                        ("sensitivity_all_numeric", predictions),
                    ):
                        metric_rows.append(
                            {
                                "policy_id": predictions["policy_id"].iloc[0],
                                "analysis_set": analysis_set,
                                "amplitude_weight": amplitude_weight,
                                "interval_weight": interval_weight,
                                "consensus_weight": consensus_weight,
                                "source_gate": source_gate,
                                **aggregate_metrics(selected),
                            }
                        )

    for outlier_threshold in (2, 3, 4):
        for source_gate in SOURCE_GATE_OPTIONS:
            policy_predictions = apply_guarded_policy(
                candidates,
                outlier_threshold=outlier_threshold,
                source_gate=source_gate,
            )
            prediction_tables.append(policy_predictions)
            for analysis_set, selected in (
                (
                    "primary_completed",
                    policy_predictions.loc[
                        policy_predictions["include_primary_analysis"]
                    ],
                ),
                ("sensitivity_all_numeric", policy_predictions),
            ):
                metric_rows.append(
                    {
                        "policy_id": policy_predictions["policy_id"].iloc[0],
                        "analysis_set": analysis_set,
                        "amplitude_weight": 2.0,
                        "interval_weight": 1.1,
                        "consensus_weight": 0.0,
                        "source_gate": source_gate,
                        "outlier_threshold": outlier_threshold,
                        **aggregate_metrics(selected),
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["analysis_set", "rr_r2", "rr_mae_bpm"], ascending=[True, False, True]
    )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "lindian_robust_fusion_candidates.csv"
    metrics_path = args.output_dir / "lindian_robust_fusion_policy_grid.csv"
    predictions_path = args.output_dir / "lindian_robust_fusion_policy_predictions.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidates: {candidates_path.resolve()}")
    print(f"Saved policy grid: {metrics_path.resolve()}")
    print(f"Saved policy predictions: {predictions_path.resolve()}")
    print("Top primary policies:")
    print(metrics.loc[metrics["analysis_set"].eq("primary_completed")].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
