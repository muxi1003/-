from __future__ import annotations

import argparse
import math
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Validate the quality-aware RR residual corrector with threshold "
            "sensitivity, bootstrap CIs, and nested threshold selection."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
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
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-random-state", type=int, default=20260704)
    return parser.parse_args()


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Threshold step must be positive.")
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def apply_threshold(
    proba: np.ndarray, confidence_threshold: float, margin_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(proba, axis=1)
    best_index = order[:, -1]
    second_index = order[:, -2]
    confidence = proba[np.arange(len(proba)), best_index]
    margin = confidence - proba[np.arange(len(proba)), second_index]
    predicted_adjust = TARGET_CLASSES[best_index].astype(int)
    applied_adjust = np.where(
        (predicted_adjust != 0)
        & (confidence >= confidence_threshold)
        & (margin >= margin_threshold),
        predicted_adjust,
        0,
    )
    return predicted_adjust, applied_adjust.astype(int), confidence, margin


def regression_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(pred)
    if int(valid.sum()) < 2:
        return math.nan
    y = truth[valid].astype(float)
    p = pred[valid].astype(float)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 0:
        return math.nan
    return float(1.0 - np.sum((p - y) ** 2) / denominator)


def pearson_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(pred)
    if int(valid.sum()) < 2:
        return math.nan
    corr = np.corrcoef(truth[valid].astype(float), pred[valid].astype(float))[0, 1]
    return float(corr * corr) if math.isfinite(float(corr)) else math.nan


def metric_dict(
    label: str,
    truth_rr: np.ndarray,
    pred_rr: np.ndarray,
    truth_count: np.ndarray,
    pred_count: np.ndarray,
    *,
    evaluation_note: str,
) -> dict[str, object]:
    rr_valid = np.isfinite(truth_rr) & np.isfinite(pred_rr)
    count_valid = np.isfinite(truth_count) & np.isfinite(pred_count)
    rr_error = pred_rr[rr_valid] - truth_rr[rr_valid]
    count_error = pred_count[count_valid] - truth_count[count_valid]
    return {
        "label": label,
        "videos": int(len(truth_rr)),
        "rr_valid_videos": int(rr_valid.sum()),
        "rr_r2": regression_r2(truth_rr, pred_rr),
        "rr_pearson_r2": pearson_r2(truth_rr, pred_rr),
        "rr_mae": float(np.mean(np.abs(rr_error))) if len(rr_error) else math.nan,
        "rr_rmse": float(np.sqrt(np.mean(rr_error**2))) if len(rr_error) else math.nan,
        "count_valid_videos": int(count_valid.sum()),
        "count_mae": float(np.mean(np.abs(count_error))) if len(count_error) else math.nan,
        "exact_count": int((np.abs(count_error) == 0).sum()) if len(count_error) else 0,
        "within_one_count": int((np.abs(count_error) <= 1).sum()) if len(count_error) else 0,
        "abs_count_error_ge2": int((np.abs(count_error) >= 2).sum()) if len(count_error) else 0,
        "evaluation_note": evaluation_note,
    }


def metrics_for_threshold(
    predictions: pd.DataFrame, confidence_threshold: float, margin_threshold: float
) -> dict[str, object]:
    proba = predictions[
        ["prob_adjust_minus1", "prob_adjust_0", "prob_adjust_plus1"]
    ].to_numpy(dtype=float)
    _, applied_adjust, _, _ = apply_threshold(
        proba,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    peaks = pd.to_numeric(predictions["peaks"], errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(predictions["duration_seconds"], errors="coerce").to_numpy(dtype=float)
    corrected_peaks = np.clip(peaks + applied_adjust, 0, None)
    corrected_rr = corrected_peaks / duration * 60.0
    row = metric_dict(
        "threshold_candidate",
        pd.to_numeric(predictions["truth_rr"], errors="coerce").to_numpy(dtype=float),
        corrected_rr,
        pd.to_numeric(predictions["truth_count"], errors="coerce").to_numpy(dtype=float),
        corrected_peaks,
        evaluation_note="existing_out_of_fold_predictions_threshold_scan",
    )
    row.update(
        {
            "confidence_threshold": float(confidence_threshold),
            "margin_threshold": float(margin_threshold),
            "applied_corrections": int(np.sum(applied_adjust != 0)),
        }
    )
    return row


def threshold_grid(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for confidence_threshold in threshold_values(
        float(args.confidence_min), float(args.confidence_max), float(args.confidence_step)
    ):
        for margin_threshold in threshold_values(
            float(args.margin_min), float(args.margin_max), float(args.margin_step)
        ):
            rows.append(metrics_for_threshold(predictions, confidence_threshold, margin_threshold))
    return pd.DataFrame(rows).sort_values(
        ["rr_r2", "rr_mae", "rr_rmse", "exact_count"],
        ascending=[False, True, True, False],
    )


def select_threshold(
    predictions: pd.DataFrame, args: argparse.Namespace
) -> tuple[float, float, dict[str, object]]:
    grid = threshold_grid(predictions, args)
    best = grid.iloc[0].to_dict()
    return float(best["confidence_threshold"]), float(best["margin_threshold"]), best


def bootstrap_ci(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    truth_rr = pd.to_numeric(predictions["truth_rr"], errors="coerce").to_numpy(dtype=float)
    base_rr = pd.to_numeric(predictions["rr_bpm"], errors="coerce").to_numpy(dtype=float)
    corr_rr = pd.to_numeric(predictions["corrected_rr_bpm"], errors="coerce").to_numpy(dtype=float)
    truth_count = pd.to_numeric(predictions["truth_count"], errors="coerce").to_numpy(dtype=float)
    base_count = pd.to_numeric(predictions["peaks"], errors="coerce").to_numpy(dtype=float)
    corr_count = pd.to_numeric(predictions["corrected_peaks"], errors="coerce").to_numpy(dtype=float)

    def values(index: np.ndarray) -> dict[str, float]:
        base = metric_dict(
            "baseline",
            truth_rr[index],
            base_rr[index],
            truth_count[index],
            base_count[index],
            evaluation_note="bootstrap",
        )
        corr = metric_dict(
            "corrected",
            truth_rr[index],
            corr_rr[index],
            truth_count[index],
            corr_count[index],
            evaluation_note="bootstrap",
        )
        return {
            "baseline_rr_r2": float(base["rr_r2"]),
            "corrected_rr_r2": float(corr["rr_r2"]),
            "delta_rr_r2": float(corr["rr_r2"]) - float(base["rr_r2"]),
            "baseline_rr_mae": float(base["rr_mae"]),
            "corrected_rr_mae": float(corr["rr_mae"]),
            "delta_rr_mae": float(corr["rr_mae"]) - float(base["rr_mae"]),
            "baseline_rr_rmse": float(base["rr_rmse"]),
            "corrected_rr_rmse": float(corr["rr_rmse"]),
            "delta_rr_rmse": float(corr["rr_rmse"]) - float(base["rr_rmse"]),
            "baseline_exact_count": float(base["exact_count"]),
            "corrected_exact_count": float(corr["exact_count"]),
            "delta_exact_count": float(corr["exact_count"]) - float(base["exact_count"]),
        }

    rng = np.random.default_rng(int(args.bootstrap_random_state))
    estimates = values(np.arange(len(predictions)))
    samples = {key: [] for key in estimates}
    for _ in range(int(args.bootstrap_samples)):
        index = rng.integers(0, len(predictions), size=len(predictions))
        sample_values = values(index)
        for key, value in sample_values.items():
            if math.isfinite(value):
                samples[key].append(value)

    rows = []
    for key, estimate in estimates.items():
        values_array = np.array(samples[key], dtype=float)
        rows.append(
            {
                "metric": key,
                "estimate": float(estimate),
                "ci_low_2_5": float(np.quantile(values_array, 0.025)),
                "ci_high_97_5": float(np.quantile(values_array, 0.975)),
                "bootstrap_samples": int(args.bootstrap_samples),
                "note": "paired_video_level_bootstrap",
            }
        )
    return pd.DataFrame(rows)


def nested_threshold_cv(
    summary: pd.DataFrame, input_root: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = safe_adjust_target(summary)
    features, numeric_columns, categorical_columns = build_feature_frame(
        summary, input_root, args.output_prefix
    )
    folds = max(2, min(int(args.folds), int(target.value_counts().min())))
    outer_cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(args.cv_random_state),
    )
    nested_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(features, target), start=1):
        train_features = features.iloc[train_index].reset_index(drop=True)
        train_target = target.iloc[train_index].reset_index(drop=True)
        train_summary = summary.iloc[train_index].reset_index(drop=True)

        inner_folds = max(2, min(int(args.folds), int(train_target.value_counts().min())))
        inner_cv = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=int(args.cv_random_state) + outer_fold,
        )
        inner_proba = np.zeros((len(train_features), len(TARGET_CLASSES)), dtype=float)
        for inner_train, inner_valid in inner_cv.split(train_features, train_target):
            model = make_model(
                numeric_columns,
                categorical_columns,
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                min_samples_leaf=int(args.min_samples_leaf),
                random_state=int(args.model_random_state),
            )
            model.fit(train_features.iloc[inner_train], train_target.iloc[inner_train])
            inner_proba[inner_valid] = aligned_predict_proba(
                model, train_features.iloc[inner_valid]
            )

        inner_predictions = train_summary.copy()
        inner_predictions["prob_adjust_minus1"] = inner_proba[:, 0]
        inner_predictions["prob_adjust_0"] = inner_proba[:, 1]
        inner_predictions["prob_adjust_plus1"] = inner_proba[:, 2]
        confidence_threshold, margin_threshold, best_row = select_threshold(inner_predictions, args)
        best_row["outer_fold"] = int(outer_fold)
        threshold_rows.append(best_row)

        final_model = make_model(
            numeric_columns,
            categorical_columns,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.model_random_state),
        )
        final_model.fit(features.iloc[train_index], target.iloc[train_index])
        test_proba = aligned_predict_proba(final_model, features.iloc[test_index])
        predicted_adjust, applied_adjust, confidence, margin = apply_threshold(
            test_proba, confidence_threshold, margin_threshold
        )
        test_rows = summary.iloc[test_index].copy()
        peaks = pd.to_numeric(test_rows["peaks"], errors="coerce").to_numpy(dtype=float)
        duration = pd.to_numeric(test_rows["duration_seconds"], errors="coerce").to_numpy(dtype=float)
        corrected_peaks = np.clip(peaks + applied_adjust, 0, None)
        test_rows["nested_outer_fold"] = int(outer_fold)
        test_rows["nested_confidence_threshold"] = float(confidence_threshold)
        test_rows["nested_margin_threshold"] = float(margin_threshold)
        test_rows["prob_adjust_minus1"] = test_proba[:, 0]
        test_rows["prob_adjust_0"] = test_proba[:, 1]
        test_rows["prob_adjust_plus1"] = test_proba[:, 2]
        test_rows["nested_predicted_adjust"] = predicted_adjust
        test_rows["nested_applied_adjust"] = applied_adjust
        test_rows["nested_residual_confidence"] = confidence
        test_rows["nested_residual_margin"] = margin
        test_rows["nested_corrected_peaks"] = corrected_peaks.astype(int)
        test_rows["nested_corrected_rr_bpm"] = corrected_peaks / duration * 60.0
        nested_rows.append(test_rows)

    nested_predictions = pd.concat(nested_rows, ignore_index=True).sort_values("video_id")
    threshold_details = pd.DataFrame(threshold_rows)
    return nested_predictions, threshold_details


def metrics_from_predictions(predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    truth_rr = pd.to_numeric(predictions["truth_rr"], errors="coerce").to_numpy(dtype=float)
    truth_count = pd.to_numeric(predictions["truth_count"], errors="coerce").to_numpy(dtype=float)
    baseline = metric_dict(
        f"{prefix}_baseline_default",
        truth_rr,
        pd.to_numeric(predictions["rr_bpm"], errors="coerce").to_numpy(dtype=float),
        truth_count,
        pd.to_numeric(predictions["peaks"], errors="coerce").to_numpy(dtype=float),
        evaluation_note="same_rows_baseline",
    )
    corrected = metric_dict(
        f"{prefix}_quality_residual",
        truth_rr,
        pd.to_numeric(
            predictions[
                "nested_corrected_rr_bpm"
                if "nested_corrected_rr_bpm" in predictions.columns
                else "corrected_rr_bpm"
            ],
            errors="coerce",
        ).to_numpy(dtype=float),
        truth_count,
        pd.to_numeric(
            predictions[
                "nested_corrected_peaks"
                if "nested_corrected_peaks" in predictions.columns
                else "corrected_peaks"
            ],
            errors="coerce",
        ).to_numpy(dtype=float),
        evaluation_note=prefix,
    )
    return pd.DataFrame([baseline, corrected])


def baseline_metric_from_predictions(predictions: pd.DataFrame, label: str) -> dict[str, object]:
    return metric_dict(
        label,
        pd.to_numeric(predictions["truth_rr"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(predictions["rr_bpm"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(predictions["truth_count"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(predictions["peaks"], errors="coerce").to_numpy(dtype=float),
        evaluation_note="existing_default_outputs",
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    predictions_csv = (
        args.predictions_csv
        or input_root / f"{args.corrected_prefix}_predictions.csv"
    )

    threshold_grid_csv = input_root / f"{args.corrected_prefix}_threshold_grid.csv"
    bootstrap_ci_csv = input_root / f"{args.corrected_prefix}_bootstrap_ci.csv"
    nested_predictions_csv = input_root / f"{args.corrected_prefix}_nested_predictions.csv"
    nested_thresholds_csv = input_root / f"{args.corrected_prefix}_nested_thresholds.csv"
    nested_metrics_csv = input_root / f"{args.corrected_prefix}_nested_metrics.csv"
    validation_summary_csv = input_root / f"{args.corrected_prefix}_validation_summary.csv"

    predictions = pd.read_csv(predictions_csv)
    grid = threshold_grid(predictions, args)
    grid.to_csv(threshold_grid_csv, index=False)

    ci = bootstrap_ci(predictions, args)
    ci.to_csv(bootstrap_ci_csv, index=False)

    summary = pd.read_csv(summary_csv)
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"}
    valid = summary[list(required)].notna().all(axis=1)
    summary = summary.loc[valid].reset_index(drop=True)
    nested_predictions, nested_thresholds = nested_threshold_cv(summary, input_root, args)
    nested_metrics = metrics_from_predictions(nested_predictions, "nested_threshold_cv")
    nested_predictions.to_csv(nested_predictions_csv, index=False)
    nested_thresholds.to_csv(nested_thresholds_csv, index=False)
    nested_metrics.to_csv(nested_metrics_csv, index=False)

    fixed_threshold = metrics_for_threshold(
        predictions,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
    )
    fixed_threshold["label"] = "fixed_default_threshold_quality_residual"
    fixed_threshold["evaluation_note"] = (
        "fixed_threshold_existing_out_of_fold_predictions_"
        f"confidence_{float(args.confidence_threshold):.3f}_margin_"
        f"{float(args.margin_threshold):.3f}"
    )
    best_grid = grid.iloc[0].to_dict()
    best_grid["label"] = "best_grid_threshold_quality_residual"
    best_grid["evaluation_note"] = "threshold_selected_on_existing_out_of_fold_predictions"
    nested_corrected = nested_metrics.iloc[1].to_dict()
    nested_corrected["label"] = "nested_threshold_cv_quality_residual"
    validation_summary = pd.DataFrame(
        [
            baseline_metric_from_predictions(predictions, "baseline_default"),
            fixed_threshold,
            best_grid,
            nested_corrected,
        ]
    )
    validation_summary.to_csv(validation_summary_csv, index=False)

    print(f"Saved threshold grid: {threshold_grid_csv}")
    print(f"Saved bootstrap CIs: {bootstrap_ci_csv}")
    print(f"Saved nested predictions: {nested_predictions_csv}")
    print(f"Saved nested thresholds: {nested_thresholds_csv}")
    print(f"Saved nested metrics: {nested_metrics_csv}")
    print(f"Saved validation summary: {validation_summary_csv}")
    print("\nBest fixed-threshold candidates:")
    print(
        grid[
            [
                "confidence_threshold",
                "margin_threshold",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "applied_corrections",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\nBootstrap CI:")
    print(ci.to_string(index=False))
    print("\nNested threshold CV metrics:")
    print(nested_metrics.to_string(index=False))
    print("\nValidation summary:")
    print(
        validation_summary[
            [
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
