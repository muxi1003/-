from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from rr_quality_residual_corrector import (
    TARGET_CLASSES,
    aligned_predict_proba,
    build_feature_frame,
    make_model,
    safe_adjust_target,
)
from rr_quality_residual_validation import apply_threshold, metric_dict, select_threshold


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the quality residual corrector with leave-one-video-prefix-group-out "
            "validation. Prefix groups are heuristic dataset domains, not proven cow IDs."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=None)
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
    return parser.parse_args()


def video_prefix(video_id: object) -> str:
    text = str(video_id)
    match = re.match(r"^[A-Za-z]+", text)
    return match.group(0) if match else "numeric"


def class_count_safe_folds(target: pd.Series, requested_folds: int) -> int:
    counts = target.value_counts()
    if len(counts) < 2:
        return 0
    return max(2, min(int(requested_folds), int(counts.min())))


def out_of_fold_train_predictions(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    folds = class_count_safe_folds(train_target, int(args.folds))
    if folds < 2:
        raise ValueError("Training partition does not have enough target classes for inner CV.")
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(args.cv_random_state),
    )
    proba = np.zeros((len(train_features), len(TARGET_CLASSES)), dtype=float)
    for train_index, valid_index in cv.split(train_features, train_target):
        model = make_model(
            numeric_columns,
            categorical_columns,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.model_random_state),
        )
        model.fit(train_features.iloc[train_index], train_target.iloc[train_index])
        proba[valid_index] = aligned_predict_proba(model, train_features.iloc[valid_index])
    return proba


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


def build_group_predictions(
    summary: pd.DataFrame,
    input_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    summary["prefix_group"] = summary["video_id"].map(video_prefix)
    target = safe_adjust_target(summary)
    features, numeric_columns, categorical_columns = build_feature_frame(
        summary, input_root, args.output_prefix
    )

    prediction_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    for group_name in sorted(summary["prefix_group"].unique()):
        test_mask = summary["prefix_group"] == group_name
        train_mask = ~test_mask
        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        train_target = target.loc[train_mask].reset_index(drop=True)
        train_summary = summary.loc[train_mask].reset_index(drop=True)
        test_summary = summary.loc[test_mask].copy()

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

        test_summary["heldout_prefix_group"] = group_name
        test_summary["prob_adjust_minus1"] = test_proba[:, 0]
        test_summary["prob_adjust_0"] = test_proba[:, 1]
        test_summary["prob_adjust_plus1"] = test_proba[:, 2]
        test_summary["group_fixed_predicted_adjust"] = fixed_pred
        test_summary["group_fixed_applied_adjust"] = fixed_adjust
        test_summary["group_fixed_confidence"] = fixed_confidence
        test_summary["group_fixed_margin"] = fixed_margin
        test_summary["group_fixed_corrected_peaks"] = fixed_peaks.astype(int)
        test_summary["group_fixed_corrected_rr_bpm"] = fixed_peaks / duration * 60.0
        test_summary["group_selected_confidence_threshold"] = selected_confidence
        test_summary["group_selected_margin_threshold"] = selected_margin
        test_summary["group_selected_predicted_adjust"] = selected_pred
        test_summary["group_selected_applied_adjust"] = selected_adjust
        test_summary["group_selected_confidence"] = selected_confidence_values
        test_summary["group_selected_margin"] = selected_margin_values
        test_summary["group_selected_corrected_peaks"] = selected_peaks.astype(int)
        test_summary["group_selected_corrected_rr_bpm"] = selected_peaks / duration * 60.0
        prediction_rows.append(test_summary)

        best_row.update(
            {
                "heldout_prefix_group": group_name,
                "train_videos": int(train_mask.sum()),
                "test_videos": int(test_mask.sum()),
                "selected_confidence_threshold": selected_confidence,
                "selected_margin_threshold": selected_margin,
            }
        )
        threshold_rows.append(best_row)

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("video_id")
    thresholds = pd.DataFrame(threshold_rows).sort_values("heldout_prefix_group")
    return predictions, thresholds


def build_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            metrics_from_columns(
                "group_holdout_baseline_default",
                predictions,
                "rr_bpm",
                "peaks",
                "same_rows_baseline",
            ),
            metrics_from_columns(
                "group_holdout_fixed_threshold",
                predictions,
                "group_fixed_corrected_rr_bpm",
                "group_fixed_corrected_peaks",
                "leave_one_prefix_group_out_fixed_threshold",
            ),
            metrics_from_columns(
                "group_holdout_train_selected_threshold",
                predictions,
                "group_selected_corrected_rr_bpm",
                "group_selected_corrected_peaks",
                "leave_one_prefix_group_out_threshold_selected_on_training_groups",
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    for group_name, group_df in predictions.groupby("heldout_prefix_group", sort=True):
        for label, rr_column, count_column, note in [
            ("baseline_default", "rr_bpm", "peaks", "same_rows_baseline"),
            (
                "fixed_threshold",
                "group_fixed_corrected_rr_bpm",
                "group_fixed_corrected_peaks",
                "leave_one_prefix_group_out_fixed_threshold",
            ),
            (
                "train_selected_threshold",
                "group_selected_corrected_rr_bpm",
                "group_selected_corrected_peaks",
                "leave_one_prefix_group_out_threshold_selected_on_training_groups",
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
    by_group = pd.DataFrame(rows)
    return overall, by_group


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    predictions_csv = input_root / f"{args.corrected_prefix}_group_predictions.csv"
    thresholds_csv = input_root / f"{args.corrected_prefix}_group_thresholds.csv"
    metrics_csv = input_root / f"{args.corrected_prefix}_group_metrics.csv"
    by_group_csv = input_root / f"{args.corrected_prefix}_group_metrics_by_prefix.csv"

    summary = pd.read_csv(summary_csv)
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")
    valid = summary[list(required)].notna().all(axis=1)
    summary = summary.loc[valid].reset_index(drop=True)

    predictions, thresholds = build_group_predictions(summary, input_root, args)
    overall_metrics, by_group_metrics = build_metrics(predictions)

    predictions.to_csv(predictions_csv, index=False)
    thresholds.to_csv(thresholds_csv, index=False)
    overall_metrics.to_csv(metrics_csv, index=False)
    by_group_metrics.to_csv(by_group_csv, index=False)

    print(f"Saved group predictions: {predictions_csv}")
    print(f"Saved group thresholds: {thresholds_csv}")
    print(f"Saved group metrics: {metrics_csv}")
    print(f"Saved group metrics by prefix: {by_group_csv}")
    print("\nPrefix groups:")
    print(
        predictions.groupby("heldout_prefix_group")
        .size()
        .reset_index(name="videos")
        .to_string(index=False)
    )
    print("\nOverall metrics:")
    print(overall_metrics.to_string(index=False))
    print("\nSelected thresholds by held-out group:")
    print(
        thresholds[
            [
                "heldout_prefix_group",
                "train_videos",
                "test_videos",
                "selected_confidence_threshold",
                "selected_margin_threshold",
                "rr_r2",
                "rr_mae",
                "exact_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
