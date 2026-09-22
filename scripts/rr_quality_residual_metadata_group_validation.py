from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_corrector import (
    aligned_predict_proba,
    build_feature_frame,
    make_model,
    safe_adjust_target,
)
from rr_quality_residual_group_validation import (
    metrics_from_columns,
    out_of_fold_train_predictions,
    video_prefix,
)
from rr_quality_residual_validation import apply_threshold, select_threshold


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Run metadata-driven grouped validation for the quality-aware RR residual "
            "corrector. In the current dataset cow_id is a numeric video label; "
            "animal-level interpretation requires a separate true-cow identity map. "
            "Use this after filling paper_metadata_template.csv with video labels, "
            "date, scene, camera, or other real grouping variables."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--group-columns", nargs="+", default=["cow_id"])
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--confidence-threshold", type=float, default=0.54)
    parser.add_argument("--margin-threshold", type=float, default=0.26)
    parser.add_argument("--confidence-min", type=float, default=0.50)
    parser.add_argument("--confidence-max", type=float, default=0.72)
    parser.add_argument("--confidence-step", type=float, default=0.01)
    parser.add_argument("--margin-min", type=float, default=0.10)
    parser.add_argument("--margin-max", type=float, default=0.36)
    parser.add_argument("--margin-step", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument(
        "--allow-prefix-fallback",
        action="store_true",
        help=(
            "If metadata groups are not ready, run the old video-prefix fallback for "
            "pipeline testing. Do not report fallback results as metadata GroupKFold."
        ),
    )
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.isin(["", "nan", "NaN", "None", "<NA>"]))


def default_metadata_csv(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets" / "paper_metadata_template.csv"


def write_readiness(
    metadata: pd.DataFrame,
    summary_video_ids: pd.Series,
    group_columns: list[str],
    output_csv: Path,
) -> pd.DataFrame:
    rows = []
    metadata_ids = set(metadata["video_id"].astype(str)) if "video_id" in metadata.columns else set()
    summary_ids = set(summary_video_ids.astype(str))
    for column in group_columns:
        if column in metadata.columns:
            values = normalize_text(metadata[column])
            nonempty = int(values.notna().sum())
            unique = int(values.dropna().nunique())
            missing_for_summary = int(
                metadata.loc[metadata["video_id"].astype(str).isin(summary_ids), column]
                .pipe(normalize_text)
                .isna()
                .sum()
            )
        else:
            nonempty = 0
            unique = 0
            missing_for_summary = len(summary_ids)
        rows.append(
            {
                "group_column": column,
                "total_summary_videos": int(len(summary_ids)),
                "metadata_rows": int(len(metadata)),
                "metadata_video_id_matches": int(len(summary_ids & metadata_ids)),
                "nonempty_values": nonempty,
                "unique_groups": unique,
                "missing_values_for_summary_videos": missing_for_summary,
                "ready_for_group_validation": bool(
                    column in metadata.columns and missing_for_summary == 0 and unique >= 2
                ),
            }
        )
    readiness = pd.DataFrame(rows)
    readiness.to_csv(output_csv, index=False)
    return readiness


def load_metadata_groups(
    summary: pd.DataFrame,
    metadata_csv: Path,
    group_columns: list[str],
    readiness_csv: Path,
    allow_prefix_fallback: bool,
) -> tuple[pd.DataFrame, str]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    if "video_id" not in metadata.columns:
        raise ValueError(f"Metadata CSV is missing required column: video_id")
    metadata["video_id"] = metadata["video_id"].astype(str)
    readiness = write_readiness(metadata, summary["video_id"].astype(str), group_columns, readiness_csv)

    missing_columns = [column for column in group_columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing group columns: {missing_columns}")

    merged = summary.merge(
        metadata,
        on="video_id",
        how="left",
        suffixes=("", "_metadata"),
        validate="one_to_one",
    )
    missing_metadata = merged[group_columns].apply(normalize_text).isna().any(axis=1)
    unique_groups = merged[group_columns].apply(normalize_text).dropna().drop_duplicates()
    if missing_metadata.any() or len(unique_groups) < 2:
        if allow_prefix_fallback:
            merged["metadata_group"] = merged["video_id"].map(video_prefix)
            merged["metadata_group_source"] = "video_prefix_fallback_not_real_metadata"
            return merged, "video_prefix_fallback_not_real_metadata"
        readiness_text = readiness.to_string(index=False)
        raise ValueError(
            "Metadata groups are not ready for grouped validation. Fill all requested "
            f"group columns for every summary video and provide at least 2 groups.\n"
            f"Readiness report: {readiness_csv}\n{readiness_text}"
        )

    normalized = merged[group_columns].apply(normalize_text)
    merged["metadata_group"] = normalized.apply(
        lambda row: "|".join(f"{column}={row[column]}" for column in group_columns),
        axis=1,
    )
    merged["metadata_group_source"] = "+".join(group_columns)
    return merged, "+".join(group_columns)


def enough_target_classes(target: pd.Series) -> bool:
    counts = target.value_counts()
    return len(counts) >= 2 and int(counts.min()) >= 2


def build_metadata_group_predictions(
    summary_with_metadata: pd.DataFrame,
    input_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = safe_adjust_target(summary_with_metadata)
    features, numeric_columns, categorical_columns = build_feature_frame(
        summary_with_metadata,
        input_root,
        args.output_prefix,
    )

    prediction_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    for group_name in sorted(summary_with_metadata["metadata_group"].unique()):
        test_mask = summary_with_metadata["metadata_group"] == group_name
        train_mask = ~test_mask
        train_target = target.loc[train_mask].reset_index(drop=True)
        if not enough_target_classes(train_target):
            raise ValueError(
                f"Training partition for held-out group {group_name!r} does not have "
                "enough residual classes for inner CV threshold selection."
            )

        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        train_summary = summary_with_metadata.loc[train_mask].reset_index(drop=True)
        test_summary = summary_with_metadata.loc[test_mask].copy()

        inner_proba = out_of_fold_train_predictions(
            train_features,
            train_target,
            numeric_columns,
            categorical_columns,
            args,
        )
        inner_predictions = train_summary.copy()
        inner_predictions["prob_adjust_minus1"] = inner_proba[:, 0]
        inner_predictions["prob_adjust_0"] = inner_proba[:, 1]
        inner_predictions["prob_adjust_plus1"] = inner_proba[:, 2]
        selected_confidence, selected_margin, best_row = select_threshold(inner_predictions, args)

        model = make_model(
            numeric_columns,
            categorical_columns,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.model_random_state),
        )
        model.fit(train_features, train_target)
        test_proba = aligned_predict_proba(model, test_features)
        fixed_pred, fixed_adjust, fixed_confidence, fixed_margin = apply_threshold(
            test_proba,
            confidence_threshold=float(args.confidence_threshold),
            margin_threshold=float(args.margin_threshold),
        )
        selected_pred, selected_adjust, selected_confidence_values, selected_margin_values = (
            apply_threshold(
                test_proba,
                confidence_threshold=selected_confidence,
                margin_threshold=selected_margin,
            )
        )

        peaks = pd.to_numeric(test_summary["peaks"], errors="coerce").to_numpy(dtype=float)
        duration = pd.to_numeric(test_summary["duration_seconds"], errors="coerce").to_numpy(dtype=float)
        fixed_peaks = np.clip(peaks + fixed_adjust, 0, None)
        selected_peaks = np.clip(peaks + selected_adjust, 0, None)

        test_summary["heldout_metadata_group"] = group_name
        test_summary["prob_adjust_minus1"] = test_proba[:, 0]
        test_summary["prob_adjust_0"] = test_proba[:, 1]
        test_summary["prob_adjust_plus1"] = test_proba[:, 2]
        test_summary["metadata_fixed_predicted_adjust"] = fixed_pred
        test_summary["metadata_fixed_applied_adjust"] = fixed_adjust
        test_summary["metadata_fixed_confidence"] = fixed_confidence
        test_summary["metadata_fixed_margin"] = fixed_margin
        test_summary["metadata_fixed_corrected_peaks"] = fixed_peaks.astype(int)
        test_summary["metadata_fixed_corrected_rr_bpm"] = fixed_peaks / duration * 60.0
        test_summary["metadata_selected_confidence_threshold"] = selected_confidence
        test_summary["metadata_selected_margin_threshold"] = selected_margin
        test_summary["metadata_selected_predicted_adjust"] = selected_pred
        test_summary["metadata_selected_applied_adjust"] = selected_adjust
        test_summary["metadata_selected_confidence"] = selected_confidence_values
        test_summary["metadata_selected_margin"] = selected_margin_values
        test_summary["metadata_selected_corrected_peaks"] = selected_peaks.astype(int)
        test_summary["metadata_selected_corrected_rr_bpm"] = selected_peaks / duration * 60.0
        prediction_rows.append(test_summary)

        best_row.update(
            {
                "heldout_metadata_group": group_name,
                "group_source": str(test_summary["metadata_group_source"].iloc[0]),
                "train_videos": int(train_mask.sum()),
                "test_videos": int(test_mask.sum()),
                "selected_confidence_threshold": selected_confidence,
                "selected_margin_threshold": selected_margin,
            }
        )
        threshold_rows.append(best_row)

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("video_id")
    thresholds = pd.DataFrame(threshold_rows).sort_values("heldout_metadata_group")
    return predictions, thresholds


def metrics_from_columns_local(
    label: str,
    data: pd.DataFrame,
    rr_column: str,
    count_column: str,
    note: str,
) -> dict[str, object]:
    row = metrics_from_columns(label, data, rr_column, count_column, note)
    row["group_source"] = str(data["metadata_group_source"].iloc[0])
    return row


def build_metadata_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            metrics_from_columns_local(
                "metadata_group_baseline_default",
                predictions,
                "rr_bpm",
                "peaks",
                "same_rows_baseline",
            ),
            metrics_from_columns_local(
                "metadata_group_fixed_threshold",
                predictions,
                "metadata_fixed_corrected_rr_bpm",
                "metadata_fixed_corrected_peaks",
                "metadata_group_holdout_fixed_threshold",
            ),
            metrics_from_columns_local(
                "metadata_group_train_selected_threshold",
                predictions,
                "metadata_selected_corrected_rr_bpm",
                "metadata_selected_corrected_peaks",
                "metadata_group_holdout_threshold_selected_on_training_groups",
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    for group_name, group_df in predictions.groupby("heldout_metadata_group", sort=True):
        for label, rr_column, count_column, note in [
            ("baseline_default", "rr_bpm", "peaks", "same_rows_baseline"),
            (
                "fixed_threshold",
                "metadata_fixed_corrected_rr_bpm",
                "metadata_fixed_corrected_peaks",
                "metadata_group_holdout_fixed_threshold",
            ),
            (
                "train_selected_threshold",
                "metadata_selected_corrected_rr_bpm",
                "metadata_selected_corrected_peaks",
                "metadata_group_holdout_threshold_selected_on_training_groups",
            ),
        ]:
            row = metrics_from_columns_local(
                f"{group_name}_{label}",
                group_df,
                rr_column,
                count_column,
                note,
            )
            row["heldout_metadata_group"] = group_name
            rows.append(row)
    by_group = pd.DataFrame(rows)
    return overall, by_group


def validate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")
    valid = summary[list(required)].notna().all(axis=1)
    return summary.loc[valid].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    metadata_csv = args.metadata_csv or default_metadata_csv(input_root, args.corrected_prefix)

    readiness_csv = input_root / f"{args.corrected_prefix}_metadata_group_readiness.csv"
    summary = validate_summary(pd.read_csv(summary_csv))
    summary_with_metadata, group_source = load_metadata_groups(
        summary,
        metadata_csv=metadata_csv,
        group_columns=list(args.group_columns),
        readiness_csv=readiness_csv,
        allow_prefix_fallback=bool(args.allow_prefix_fallback),
    )
    output_suffix = (
        "metadata_group_prefix_fallback"
        if group_source == "video_prefix_fallback_not_real_metadata"
        else "metadata_group"
    )
    predictions_csv = input_root / f"{args.corrected_prefix}_{output_suffix}_predictions.csv"
    thresholds_csv = input_root / f"{args.corrected_prefix}_{output_suffix}_thresholds.csv"
    metrics_csv = input_root / f"{args.corrected_prefix}_{output_suffix}_metrics.csv"
    by_group_csv = input_root / f"{args.corrected_prefix}_{output_suffix}_metrics_by_group.csv"

    predictions, thresholds = build_metadata_group_predictions(summary_with_metadata, input_root, args)
    overall_metrics, by_group_metrics = build_metadata_metrics(predictions)

    predictions.to_csv(predictions_csv, index=False)
    thresholds.to_csv(thresholds_csv, index=False)
    overall_metrics.to_csv(metrics_csv, index=False)
    by_group_metrics.to_csv(by_group_csv, index=False)

    print(f"Metadata group source: {group_source}")
    print(f"Saved readiness report: {readiness_csv}")
    print(f"Saved metadata group predictions: {predictions_csv}")
    print(f"Saved metadata group thresholds: {thresholds_csv}")
    print(f"Saved metadata group metrics: {metrics_csv}")
    print(f"Saved metadata group metrics by group: {by_group_csv}")
    print("\nOverall metrics:")
    print(overall_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
