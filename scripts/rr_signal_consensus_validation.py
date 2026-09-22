from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_group_validation import video_prefix
from rr_quality_residual_validation import metric_dict


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a conservative signal-consensus supplement for RR peak-count "
            "correction. The supplement preserves the existing quality residual "
            "correction and adds a correction only when spectral/FFT/autocorrelation "
            "signals agree with the residual model direction."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--residual-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--consensus-prefix", default="paper_repro_signal_consensus")
    parser.add_argument("--supplement-confidence", type=float, default=0.50)
    parser.add_argument("--supplement-margin", type=float, default=0.10)
    parser.add_argument("--min-signal-votes", type=int, default=2)
    parser.add_argument(
        "--min-supplement-interval-cv",
        type=float,
        default=0.09,
        help=(
            "Do not add a signal-only supplement on very regular peak trains. "
            "This keeps the rule conservative for rows that already look stable."
        ),
    )
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fft_count_estimate(values: np.ndarray) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8 or float(np.nanstd(y)) <= 1e-9:
        return math.nan, math.nan
    centered = y - float(np.nanmean(y))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(n))) ** 2
    if len(spectrum) <= 3:
        return math.nan, math.nan
    spectrum[0] = 0.0
    min_cycles = 3
    max_cycles = min(18, n // 2)
    if max_cycles < min_cycles:
        return math.nan, math.nan
    scores = spectrum.copy()
    mask = np.zeros_like(scores, dtype=bool)
    mask[min_cycles : max_cycles + 1] = True
    scores[~mask] = 0.0
    index = int(np.argmax(scores))
    if scores[index] <= 0:
        return math.nan, math.nan
    strength = float(scores[index] / (np.sum(spectrum) + 1e-9))
    return float(index), strength


def autocorr_count_estimate(values: np.ndarray) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8 or float(np.nanstd(y)) <= 1e-9:
        return math.nan, math.nan
    centered = y - float(np.nanmean(y))
    corr = np.correlate(centered, centered, mode="full")[n - 1 :]
    corr = corr / (corr[0] + 1e-9)
    min_lag = 3
    max_lag = max(min_lag + 1, min(30, n // 2))
    segment = corr[min_lag : max_lag + 1]
    if len(segment) == 0:
        return math.nan, math.nan
    lag = int(np.argmax(segment) + min_lag)
    strength = float(corr[lag])
    if strength <= 0:
        return math.nan, strength
    return float(int(round(n / lag))), strength


def build_signal_candidates(
    input_root: Path, output_prefix: str, video_ids: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for video_id in video_ids.astype(str).drop_duplicates().sort_values():
        curve_path = input_root / video_id / f"{output_prefix}_curve.csv"
        row: dict[str, object] = {
            "video_id": video_id,
            "fft_count_estimate": math.nan,
            "fft_strength": math.nan,
            "autocorr_count_estimate": math.nan,
            "autocorr_strength": math.nan,
        }
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            if "smoothed_norm" in curve.columns:
                smoothed = pd.to_numeric(curve["smoothed_norm"], errors="coerce").to_numpy(
                    dtype=float
                )
                fft_count, fft_strength = fft_count_estimate(smoothed)
                autocorr_count, autocorr_strength = autocorr_count_estimate(smoothed)
                row.update(
                    {
                        "fft_count_estimate": fft_count,
                        "fft_strength": fft_strength,
                        "autocorr_count_estimate": autocorr_count,
                        "autocorr_strength": autocorr_strength,
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def count_adjust(candidate_count: object, peak_count: object) -> int:
    candidate = finite_float(candidate_count)
    peaks = finite_float(peak_count)
    if math.isnan(candidate) or math.isnan(peaks):
        return 0
    delta = int(round(candidate - peaks))
    return int(np.clip(delta, -1, 1))


def signal_vote_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["spectral_signal_adjust"] = [
        count_adjust(candidate, peaks)
        for candidate, peaks in zip(
            result.get("spectral_count_estimate", pd.Series([math.nan] * len(result))),
            result["peaks"],
        )
    ]
    result["fft_signal_adjust"] = [
        count_adjust(candidate, peaks)
        for candidate, peaks in zip(result["fft_count_estimate"], result["peaks"])
    ]
    result["autocorr_signal_adjust"] = [
        count_adjust(candidate, peaks)
        for candidate, peaks in zip(result["autocorr_count_estimate"], result["peaks"])
    ]
    return result


def mode_columns(mode: str) -> dict[str, str]:
    mapping = {
        "fixed_oof": {
            "predicted": "predicted_adjust",
            "applied": "applied_adjust",
            "confidence": "residual_confidence",
            "margin": "residual_margin",
            "existing_peaks": "corrected_peaks",
            "existing_rr": "corrected_rr_bpm",
            "output_peaks": "signal_consensus_peaks",
            "output_rr": "signal_consensus_rr_bpm",
            "adjust": "signal_consensus_adjust",
        },
        "nested": {
            "predicted": "nested_predicted_adjust",
            "applied": "nested_applied_adjust",
            "confidence": "nested_residual_confidence",
            "margin": "nested_residual_margin",
            "existing_peaks": "nested_corrected_peaks",
            "existing_rr": "nested_corrected_rr_bpm",
            "output_peaks": "nested_signal_consensus_peaks",
            "output_rr": "nested_signal_consensus_rr_bpm",
            "adjust": "nested_signal_consensus_adjust",
        },
        "group_fixed": {
            "predicted": "group_fixed_predicted_adjust",
            "applied": "group_fixed_applied_adjust",
            "confidence": "group_fixed_confidence",
            "margin": "group_fixed_margin",
            "existing_peaks": "group_fixed_corrected_peaks",
            "existing_rr": "group_fixed_corrected_rr_bpm",
            "output_peaks": "group_fixed_signal_consensus_peaks",
            "output_rr": "group_fixed_signal_consensus_rr_bpm",
            "adjust": "group_fixed_signal_consensus_adjust",
        },
        "group_selected": {
            "predicted": "group_selected_predicted_adjust",
            "applied": "group_selected_applied_adjust",
            "confidence": "group_selected_confidence",
            "margin": "group_selected_margin",
            "existing_peaks": "group_selected_corrected_peaks",
            "existing_rr": "group_selected_corrected_rr_bpm",
            "output_peaks": "group_selected_signal_consensus_peaks",
            "output_rr": "group_selected_signal_consensus_rr_bpm",
            "adjust": "group_selected_signal_consensus_adjust",
        },
    }
    if mode not in mapping:
        raise ValueError(f"Unknown consensus mode: {mode}")
    return mapping[mode]


def apply_signal_consensus(
    predictions: pd.DataFrame,
    signal_candidates: pd.DataFrame,
    mode: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    columns = mode_columns(mode)
    if {"fft_count_estimate", "autocorr_count_estimate"}.issubset(predictions.columns):
        data = predictions.copy()
    else:
        data = predictions.merge(signal_candidates, on="video_id", how="left")
    data = signal_vote_columns(data)
    adjustments: list[int] = []
    reasons: list[str] = []
    votes_for_direction: list[int] = []
    for row in data.itertuples(index=False):
        row_dict = row._asdict()
        applied = int(finite_float(row_dict[columns["applied"]]))
        predicted = int(finite_float(row_dict[columns["predicted"]]))
        confidence = finite_float(row_dict[columns["confidence"]])
        margin = finite_float(row_dict[columns["margin"]])
        interval_cv = finite_float(row_dict.get("selection_interval_cv", math.nan))
        signal_adjusts = [
            int(row_dict["spectral_signal_adjust"]),
            int(row_dict["fft_signal_adjust"]),
            int(row_dict["autocorr_signal_adjust"]),
        ]

        def votes(direction: int) -> int:
            return int(sum(1 for value in signal_adjusts if value == direction))

        adjustment = applied
        vote_count = votes(applied) if applied != 0 else votes(predicted)
        reason = "existing_residual_adjustment" if applied != 0 else "no_adjustment"
        if (
            adjustment == 0
            and predicted != 0
            and confidence >= float(args.supplement_confidence)
            and margin >= float(args.supplement_margin)
            and votes(predicted) >= int(args.min_signal_votes)
            and math.isfinite(interval_cv)
            and interval_cv >= float(args.min_supplement_interval_cv)
        ):
            adjustment = predicted
            vote_count = votes(predicted)
            reason = "signal_consensus_supplement"
        adjustments.append(int(adjustment))
        reasons.append(reason)
        votes_for_direction.append(int(vote_count))

    data[columns["adjust"]] = adjustments
    data[f"{columns['adjust']}_reason"] = reasons
    data[f"{columns['adjust']}_signal_votes"] = votes_for_direction
    peaks = pd.to_numeric(data["peaks"], errors="coerce")
    duration = pd.to_numeric(data["duration_seconds"], errors="coerce")
    data[columns["output_peaks"]] = (peaks + data[columns["adjust"]]).clip(lower=0).astype(int)
    data[columns["output_rr"]] = data[columns["output_peaks"]] / duration * 60.0
    return data


def metrics_from_columns(
    label: str,
    data: pd.DataFrame,
    rr_column: str,
    count_column: str,
    note: str,
) -> dict[str, object]:
    return metric_dict(
        label,
        pd.to_numeric(data["truth_rr"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(data[rr_column], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(data["truth_count"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(data[count_column], errors="coerce").to_numpy(dtype=float),
        evaluation_note=note,
    )


def fixed_metrics(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            metrics_from_columns(
                "baseline_default",
                data,
                "rr_bpm",
                "peaks",
                "existing_default_outputs",
            ),
            metrics_from_columns(
                "quality_residual_fixed_threshold",
                data,
                "corrected_rr_bpm",
                "corrected_peaks",
                "existing_quality_residual_fixed_threshold",
            ),
            metrics_from_columns(
                "signal_consensus_supplement_fixed_threshold",
                data,
                "signal_consensus_rr_bpm",
                "signal_consensus_peaks",
                "quality_residual_plus_fft_autocorr_spectral_supplement",
            ),
        ]
    )


def nested_metrics(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            metrics_from_columns(
                "nested_baseline_default",
                data,
                "rr_bpm",
                "peaks",
                "same_rows_baseline",
            ),
            metrics_from_columns(
                "nested_quality_residual",
                data,
                "nested_corrected_rr_bpm",
                "nested_corrected_peaks",
                "existing_nested_threshold_cv",
            ),
            metrics_from_columns(
                "nested_signal_consensus_supplement",
                data,
                "nested_signal_consensus_rr_bpm",
                "nested_signal_consensus_peaks",
                "nested_quality_residual_plus_signal_supplement",
            ),
        ]
    )


def group_metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            metrics_from_columns(
                "group_holdout_baseline_default",
                data,
                "rr_bpm",
                "peaks",
                "same_rows_baseline",
            ),
            metrics_from_columns(
                "group_holdout_fixed_threshold",
                data,
                "group_fixed_corrected_rr_bpm",
                "group_fixed_corrected_peaks",
                "existing_leave_one_prefix_group_out_fixed_threshold",
            ),
            metrics_from_columns(
                "group_holdout_fixed_signal_consensus_supplement",
                data,
                "group_fixed_signal_consensus_rr_bpm",
                "group_fixed_signal_consensus_peaks",
                "fixed_group_holdout_plus_signal_supplement",
            ),
            metrics_from_columns(
                "group_holdout_train_selected_threshold",
                data,
                "group_selected_corrected_rr_bpm",
                "group_selected_corrected_peaks",
                "existing_leave_one_prefix_group_out_train_selected_threshold",
            ),
            metrics_from_columns(
                "group_holdout_train_selected_signal_consensus_supplement",
                data,
                "group_selected_signal_consensus_rr_bpm",
                "group_selected_signal_consensus_peaks",
                "train_selected_group_holdout_plus_signal_supplement",
            ),
        ]
    )
    rows: list[dict[str, object]] = []
    for group_name, group_df in data.groupby("heldout_prefix_group", sort=True):
        for label, rr_column, count_column, note in [
            ("baseline_default", "rr_bpm", "peaks", "same_rows_baseline"),
            (
                "fixed_threshold",
                "group_fixed_corrected_rr_bpm",
                "group_fixed_corrected_peaks",
                "existing_leave_one_prefix_group_out_fixed_threshold",
            ),
            (
                "fixed_signal_consensus_supplement",
                "group_fixed_signal_consensus_rr_bpm",
                "group_fixed_signal_consensus_peaks",
                "fixed_group_holdout_plus_signal_supplement",
            ),
            (
                "train_selected_threshold",
                "group_selected_corrected_rr_bpm",
                "group_selected_corrected_peaks",
                "existing_leave_one_prefix_group_out_train_selected_threshold",
            ),
            (
                "train_selected_signal_consensus_supplement",
                "group_selected_signal_consensus_rr_bpm",
                "group_selected_signal_consensus_peaks",
                "train_selected_group_holdout_plus_signal_supplement",
            ),
        ]:
            row = metrics_from_columns(
                f"{group_name}_{label}",
                group_df,
                rr_column,
                count_column,
                note,
            )
            row["heldout_prefix_group"] = group_name
            rows.append(row)
    return overall, pd.DataFrame(rows)


def add_prefix_group_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "heldout_prefix_group" not in result.columns:
        result["heldout_prefix_group"] = result["video_id"].map(video_prefix)
    return result


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    residual_predictions_path = input_root / f"{args.residual_prefix}_predictions.csv"
    nested_predictions_path = input_root / f"{args.residual_prefix}_nested_predictions.csv"
    group_predictions_path = input_root / f"{args.residual_prefix}_group_predictions.csv"

    residual_predictions = pd.read_csv(residual_predictions_path)
    signal_candidates = build_signal_candidates(
        input_root, args.output_prefix, residual_predictions["video_id"]
    )
    fixed_predictions = apply_signal_consensus(
        residual_predictions,
        signal_candidates,
        "fixed_oof",
        args,
    )
    fixed_metric_table = fixed_metrics(fixed_predictions)

    nested_predictions = pd.read_csv(nested_predictions_path)
    nested_predictions = apply_signal_consensus(
        nested_predictions,
        signal_candidates,
        "nested",
        args,
    )
    nested_metric_table = nested_metrics(nested_predictions)

    group_predictions = pd.read_csv(group_predictions_path)
    group_predictions = add_prefix_group_if_missing(group_predictions)
    group_predictions = apply_signal_consensus(
        group_predictions,
        signal_candidates,
        "group_fixed",
        args,
    )
    group_predictions = apply_signal_consensus(
        group_predictions,
        signal_candidates,
        "group_selected",
        args,
    )
    group_metric_table, group_by_prefix = group_metrics(group_predictions)

    comparison = pd.concat(
        [
            fixed_metric_table.assign(validation_family="fixed_oof"),
            nested_metric_table.assign(validation_family="nested_cv"),
            group_metric_table.assign(validation_family="prefix_group_holdout"),
        ],
        ignore_index=True,
    )

    signal_candidates.to_csv(input_root / f"{args.consensus_prefix}_candidates.csv", index=False)
    fixed_predictions.to_csv(input_root / f"{args.consensus_prefix}_predictions.csv", index=False)
    fixed_metric_table.to_csv(input_root / f"{args.consensus_prefix}_metrics.csv", index=False)
    nested_predictions.to_csv(
        input_root / f"{args.consensus_prefix}_nested_predictions.csv", index=False
    )
    nested_metric_table.to_csv(
        input_root / f"{args.consensus_prefix}_nested_metrics.csv", index=False
    )
    group_predictions.to_csv(
        input_root / f"{args.consensus_prefix}_group_predictions.csv", index=False
    )
    group_metric_table.to_csv(
        input_root / f"{args.consensus_prefix}_group_metrics.csv", index=False
    )
    group_by_prefix.to_csv(
        input_root / f"{args.consensus_prefix}_group_metrics_by_prefix.csv", index=False
    )
    comparison.to_csv(input_root / f"{args.consensus_prefix}_comparison_summary.csv", index=False)

    print(f"Saved signal candidates: {input_root / f'{args.consensus_prefix}_candidates.csv'}")
    print(f"Saved fixed predictions: {input_root / f'{args.consensus_prefix}_predictions.csv'}")
    print(f"Saved fixed metrics: {input_root / f'{args.consensus_prefix}_metrics.csv'}")
    print(f"Saved nested metrics: {input_root / f'{args.consensus_prefix}_nested_metrics.csv'}")
    print(f"Saved group metrics: {input_root / f'{args.consensus_prefix}_group_metrics.csv'}")
    print("\nComparison summary:")
    print(
        comparison[
            [
                "validation_family",
                "label",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "within_one_count",
                "evaluation_note",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
