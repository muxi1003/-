from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rr_quality_residual_corrector import (
    apply_thresholds,
    build_feature_frame,
    metrics_row,
    out_of_fold_predictions,
    safe_adjust_target,
)


FeaturePredicate = Callable[[str], bool]


@dataclass(frozen=True)
class AblationSpec:
    label: str
    description: str
    selector: Callable[[list[str]], list[str]]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Run feature-block ablations for the quality-aware RR residual corrector."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--confidence-threshold", type=float, default=0.54)
    parser.add_argument("--margin-threshold", type=float, default=0.26)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    return parser.parse_args()


def has_any(column: str, tokens: tuple[str, ...]) -> bool:
    lower = column.lower()
    return any(token in lower for token in tokens)


def is_peak_count_feature(column: str) -> bool:
    lower = column.lower()
    return (
        lower == "peaks"
        or "peak_count" in lower
        or "candidate_peaks" in lower
        or "target_peaks" in lower
        or "original_peaks" in lower
        or "count_estimate" in lower
    )


def is_endpoint_gap_feature(column: str) -> bool:
    return has_any(column, ("gap", "edge"))


def is_prominence_feature(column: str) -> bool:
    return "prominence" in column.lower()


def is_spectral_feature(column: str) -> bool:
    return "spectral" in column.lower()


def is_signal_strength_feature(column: str) -> bool:
    return has_any(column, ("amplitude", "std"))


def is_missing_or_fusion_feature(column: str) -> bool:
    return has_any(
        column,
        (
            "missing",
            "repair",
            "repaired",
            "track",
            "tracked",
            "infer",
            "inferred",
            "fusion",
            "left_",
            "right_",
        ),
    )


def remove_where(predicate: FeaturePredicate) -> Callable[[list[str]], list[str]]:
    return lambda columns: [column for column in columns if not predicate(column)]


def keep_where(predicate: FeaturePredicate) -> Callable[[list[str]], list[str]]:
    return lambda columns: [column for column in columns if predicate(column)]


def ablation_specs() -> list[AblationSpec]:
    return [
        AblationSpec(
            "all_features",
            "Full quality-aware residual corrector feature set.",
            lambda columns: list(columns),
        ),
        AblationSpec(
            "summary_only",
            "Only video-level summary and pipeline-state features; curve-derived features removed.",
            remove_where(lambda column: column.startswith("curve_")),
        ),
        AblationSpec(
            "curve_only",
            "Only curve-derived quality features from per-video respiratory curves.",
            keep_where(lambda column: column.startswith("curve_")),
        ),
        AblationSpec(
            "without_peak_count_context",
            "Remove detected/candidate/retuned peak-count context features.",
            remove_where(is_peak_count_feature),
        ),
        AblationSpec(
            "without_endpoint_gap_context",
            "Remove endpoint and boundary-gap features.",
            remove_where(is_endpoint_gap_feature),
        ),
        AblationSpec(
            "without_prominence_context",
            "Remove peak-prominence and weak-peak salience features.",
            remove_where(is_prominence_feature),
        ),
        AblationSpec(
            "without_spectral_context",
            "Remove frequency-domain and spectral-temporal consistency features.",
            remove_where(is_spectral_feature),
        ),
        AblationSpec(
            "without_signal_strength_context",
            "Remove amplitude and standard-deviation signal-strength features.",
            remove_where(is_signal_strength_feature),
        ),
        AblationSpec(
            "without_missing_fusion_context",
            "Remove missingness, nostril-channel, tracking, repair, and fusion-state features.",
            remove_where(is_missing_or_fusion_feature),
        ),
    ]


def split_columns(
    selected_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[list[str], list[str]]:
    selected = set(selected_columns)
    return (
        [column for column in numeric_columns if column in selected],
        [column for column in categorical_columns if column in selected],
    )


def corrected_predictions(
    summary: pd.DataFrame,
    target: pd.Series,
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    proba, fold_ids = out_of_fold_predictions(
        features, target, numeric_columns, categorical_columns, args
    )
    predicted_adjust, applied_adjust, margin = apply_thresholds(
        proba,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
    )
    confidence = np.max(proba, axis=1)

    predictions = summary[
        ["video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"]
    ].copy()
    predictions["ablation_label"] = label
    predictions["target_adjust"] = target.to_numpy(dtype=int)
    predictions["cv_fold"] = fold_ids
    predictions["prob_adjust_minus1"] = proba[:, 0]
    predictions["prob_adjust_0"] = proba[:, 1]
    predictions["prob_adjust_plus1"] = proba[:, 2]
    predictions["predicted_adjust"] = predicted_adjust
    predictions["applied_adjust"] = applied_adjust
    predictions["residual_confidence"] = confidence
    predictions["residual_margin"] = margin
    peaks = pd.to_numeric(predictions["peaks"], errors="coerce")
    duration = pd.to_numeric(predictions["duration_seconds"], errors="coerce")
    predictions["corrected_peaks"] = (peaks + applied_adjust).clip(lower=0)
    predictions["corrected_rr_bpm"] = predictions["corrected_peaks"] / duration * 60.0
    predictions["corrected_count_error"] = predictions["corrected_peaks"] - pd.to_numeric(
        predictions["truth_count"], errors="coerce"
    )
    predictions["corrected_abs_count_error"] = predictions["corrected_count_error"].abs()
    predictions["corrected_rr_error"] = predictions["corrected_rr_bpm"] - pd.to_numeric(
        predictions["truth_rr"], errors="coerce"
    )
    predictions["corrected_abs_rr_error"] = predictions["corrected_rr_error"].abs()

    metric = metrics_row(
        label,
        predictions,
        rr_column="corrected_rr_bpm",
        count_column="corrected_peaks",
        evaluation_note=(
            "feature_block_ablation_out_of_fold_thresholded_confidence_"
            f"{float(args.confidence_threshold):.3f}_margin_"
            f"{float(args.margin_threshold):.3f}"
        ),
    )
    metric["applied_corrections"] = int(np.sum(applied_adjust != 0))
    return predictions, metric


def baseline_metric(summary: pd.DataFrame) -> dict[str, object]:
    metric = metrics_row(
        "baseline_default",
        summary,
        rr_column="rr_bpm",
        count_column="peaks",
        evaluation_note="existing_default_outputs",
    )
    metric["applied_corrections"] = 0
    return metric


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    baseline = table[table["label"] == "baseline_default"].iloc[0]
    full = table[table["label"] == "all_features"].iloc[0]
    for metric in ["rr_r2", "rr_mae", "rr_rmse", "exact_count", "count_mae"]:
        table[f"delta_vs_baseline_{metric}"] = table[metric] - baseline[metric]
        table[f"delta_vs_full_{metric}"] = table[metric] - full[metric]
    return table


def feature_group_membership(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in features.columns:
        rows.append(
            {
                "feature": column,
                "is_curve_feature": bool(column.startswith("curve_")),
                "is_summary_feature": bool(not column.startswith("curve_")),
                "is_peak_count_context": bool(is_peak_count_feature(column)),
                "is_endpoint_gap_context": bool(is_endpoint_gap_feature(column)),
                "is_prominence_context": bool(is_prominence_feature(column)),
                "is_spectral_context": bool(is_spectral_feature(column)),
                "is_signal_strength_context": bool(is_signal_strength_feature(column)),
                "is_missing_fusion_context": bool(is_missing_or_fusion_feature(column)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    metrics_csv = input_root / f"{args.corrected_prefix}_ablation_metrics.csv"
    predictions_csv = input_root / f"{args.corrected_prefix}_ablation_predictions.csv"
    groups_csv = input_root / f"{args.corrected_prefix}_ablation_feature_groups.csv"

    summary = pd.read_csv(summary_csv)
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")
    valid = summary[list(required)].notna().all(axis=1)
    if not valid.all():
        summary = summary.loc[valid].reset_index(drop=True)

    target = safe_adjust_target(summary)
    features, numeric_columns, categorical_columns = build_feature_frame(
        summary, input_root, args.output_prefix
    )

    metric_rows = [baseline_metric(summary)]
    prediction_tables: list[pd.DataFrame] = []
    all_columns = list(features.columns)
    for spec in ablation_specs():
        selected_columns = spec.selector(all_columns)
        selected_columns = [column for column in selected_columns if column in features.columns]
        if not selected_columns:
            metric_rows.append(
                {
                    "label": spec.label,
                    "videos": int(len(summary)),
                    "rr_valid_videos": 0,
                    "rr_r2": math.nan,
                    "rr_pearson_r2": math.nan,
                    "rr_mae": math.nan,
                    "rr_rmse": math.nan,
                    "count_valid_videos": 0,
                    "count_mae": math.nan,
                    "exact_count": 0,
                    "within_one_count": 0,
                    "abs_count_error_ge2": 0,
                    "evaluation_note": "feature_block_ablation_skipped_no_features",
                    "applied_corrections": 0,
                    "description": spec.description,
                    "retained_feature_count": 0,
                    "dropped_feature_count": int(len(all_columns)),
                }
            )
            continue
        selected_numeric, selected_categorical = split_columns(
            selected_columns, numeric_columns, categorical_columns
        )
        selected_features = features[selected_numeric + selected_categorical].copy()
        predictions, metric = corrected_predictions(
            summary,
            target,
            selected_features,
            selected_numeric,
            selected_categorical,
            args,
            spec.label,
        )
        metric["description"] = spec.description
        metric["retained_feature_count"] = int(len(selected_features.columns))
        metric["dropped_feature_count"] = int(len(all_columns) - len(selected_features.columns))
        metric_rows.append(metric)
        prediction_tables.append(predictions)

    metrics = add_deltas(pd.DataFrame(metric_rows))
    if prediction_tables:
        predictions = pd.concat(prediction_tables, ignore_index=True)
    else:
        predictions = pd.DataFrame()
    groups = feature_group_membership(features)

    metrics.to_csv(metrics_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)
    groups.to_csv(groups_csv, index=False)

    print(f"Saved ablation metrics: {metrics_csv}")
    print(f"Saved ablation predictions: {predictions_csv}")
    print(f"Saved ablation feature groups: {groups_csv}")
    print(
        metrics[
            [
                "label",
                "retained_feature_count",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "applied_corrections",
                "delta_vs_full_rr_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
