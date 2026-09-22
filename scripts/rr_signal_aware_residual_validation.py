from __future__ import annotations

import argparse
import copy
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rr_quality_residual_validation import metric_dict


TARGET_CLASSES = np.array([-1, 0, 1], dtype=int)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a signal-aware second-stage residual correction after the "
            "signal-consensus RR estimator. The model is evaluated with "
            "out-of-fold predictions and never uses the held-out video's truth "
            "during prediction."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-name", default="signal_aware_residual")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=13)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-random-state", type=int, default=20260706)
    return parser.parse_args()


def one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def simple_imputer(strategy: str) -> SimpleImputer:
    kwargs = {"strategy": strategy}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        kwargs["keep_empty_features"] = True
    return SimpleImputer(**kwargs)


def video_prefix(video_id: object) -> str:
    text = str(video_id)
    match = re.match(r"[A-Za-z]+", text)
    if match:
        return match.group(0)
    return text[:2]


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_signal_aware_target(predictions: pd.DataFrame) -> pd.Series:
    target = numeric(predictions, "truth_count") - numeric(
        predictions, "signal_consensus_peaks"
    )
    return target.clip(lower=-1, upper=1).astype(int)


def is_leakage_column(column: str) -> bool:
    lower = column.lower()
    leakage_tokens = [
        "truth",
        "error",
        "abs_",
        "target_adjust",
        "cv_fold",
        "evaluation_note",
    ]
    if any(token in lower for token in leakage_tokens):
        return True
    if column in {
        "video_id",
        "frame_name",
        "temperature_csv",
        "curve_csv",
        "curve_png",
        "review_png",
    }:
        return True
    if column.startswith("prob_adjust"):
        return True
    if column in {
        "predicted_adjust",
        "applied_adjust",
        "corrected_count_error",
        "corrected_abs_count_error",
        "corrected_rr_error",
        "corrected_abs_rr_error",
        "selective_count_error",
        "selective_abs_count_error",
        "selective_rr_error",
        "selective_abs_rr_error",
    }:
        return True
    return False


def build_feature_frame(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_columns = [
        column for column in predictions.columns if not is_leakage_column(column)
    ]
    features = predictions[feature_columns].copy()
    for base in ["peaks", "corrected_peaks", "signal_consensus_peaks"]:
        for estimate in [
            "spectral_count_estimate",
            "fft_count_estimate",
            "autocorr_count_estimate",
        ]:
            if base in predictions.columns and estimate in predictions.columns:
                features[f"{estimate}_minus_{base}"] = numeric(
                    predictions, estimate
                ) - numeric(predictions, base)

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(features[column]) or int(converted.notna().sum()) >= int(
            0.8 * len(features)
        ):
            if int(converted.notna().sum()) == 0:
                continue
            features[column] = converted
            numeric_columns.append(column)
        else:
            features[column] = features[column].astype(str)
            categorical_columns.append(column)
    return features[numeric_columns + categorical_columns], numeric_columns, categorical_columns


def make_classifier(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    model_random_state: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", simple_imputer("median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", simple_imputer("most_frequent")),
                        ("one_hot", one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    classifier = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        random_state=model_random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def full_probability_matrix(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    result = np.zeros((probabilities.shape[0], len(TARGET_CLASSES)), dtype=float)
    for index, label in enumerate(classes):
        target_index = int(np.where(TARGET_CLASSES == int(label))[0][0])
        result[:, target_index] = probabilities[:, index]
    return result


def out_of_fold_probabilities(
    features: pd.DataFrame,
    target: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    mode: str,
    folds: int,
    random_state: int,
    model_random_state: int,
    video_ids: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(features), len(TARGET_CLASSES)), dtype=float)
    fold_ids = np.zeros(len(features), dtype=int)
    if mode == "stratified":
        splitter = StratifiedKFold(
            n_splits=int(folds), shuffle=True, random_state=int(random_state)
        )
        splits = splitter.split(features, target)
    elif mode == "prefix_group":
        groups = video_ids.map(video_prefix).to_numpy()
        n_splits = min(int(folds), len(set(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(features, target, groups)
    else:
        raise ValueError(f"Unknown OOF mode: {mode}")

    base_model = make_classifier(
        numeric_columns,
        categorical_columns,
        model_random_state=int(model_random_state),
    )
    for fold_index, (train_index, test_index) in enumerate(splits, start=1):
        model = copy.deepcopy(base_model)
        model.fit(features.iloc[train_index], target.iloc[train_index])
        fold_probabilities = model.predict_proba(features.iloc[test_index])
        classes = model.named_steps["classifier"].classes_
        probabilities[test_index] = full_probability_matrix(fold_probabilities, classes)
        fold_ids[test_index] = fold_index
    return probabilities, fold_ids


def apply_threshold(
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = TARGET_CLASSES[np.argmax(probabilities, axis=1)]
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = sorted_probabilities[:, -1]
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    applied = np.where(
        (confidence >= float(confidence_threshold))
        & (margin >= float(margin_threshold)),
        predicted,
        0,
    )
    return predicted, applied, confidence, margin


def add_prediction_columns(
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    mode: str,
) -> pd.DataFrame:
    result = predictions.copy()
    predicted, applied, confidence, margin = apply_threshold(
        probabilities,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    base_count = numeric(result, "signal_consensus_peaks").to_numpy(dtype=float)
    duration = numeric(result, "rr_duration_seconds").to_numpy(dtype=float)
    final_count = np.clip(base_count + applied, 0, None)
    final_rr = final_count / duration * 60.0
    truth_count = numeric(result, "truth_count").to_numpy(dtype=float)
    truth_rr = numeric(result, "truth_rr").to_numpy(dtype=float)

    result[f"{mode}_fold"] = fold_ids
    result[f"{mode}_prob_adjust_minus1"] = probabilities[:, 0]
    result[f"{mode}_prob_adjust_0"] = probabilities[:, 1]
    result[f"{mode}_prob_adjust_plus1"] = probabilities[:, 2]
    result[f"{mode}_predicted_adjust"] = predicted
    result[f"{mode}_applied_adjust"] = applied
    result[f"{mode}_confidence"] = confidence
    result[f"{mode}_margin"] = margin
    result[f"{mode}_final_peaks"] = final_count
    result[f"{mode}_final_rr_bpm"] = final_rr
    result[f"{mode}_count_error"] = final_count - truth_count
    result[f"{mode}_abs_count_error"] = np.abs(final_count - truth_count)
    result[f"{mode}_rr_error"] = final_rr - truth_rr
    result[f"{mode}_abs_rr_error"] = np.abs(final_rr - truth_rr)
    result[f"{mode}_evaluation_note"] = (
        f"{mode}_out_of_fold_signal_aware_residual_correction_"
        f"confidence_{confidence_threshold:.2f}_margin_{margin_threshold:.2f}"
    )
    return result


def metrics_for_counts(
    label: str,
    predictions: pd.DataFrame,
    pred_count: np.ndarray,
    *,
    evaluation_note: str,
) -> dict[str, object]:
    duration = numeric(predictions, "rr_duration_seconds").to_numpy(dtype=float)
    pred_rr = pred_count / duration * 60.0
    row = metric_dict(
        label,
        numeric(predictions, "truth_rr").to_numpy(dtype=float),
        pred_rr,
        numeric(predictions, "truth_count").to_numpy(dtype=float),
        pred_count,
        evaluation_note=evaluation_note,
    )
    return row


def threshold_grid(
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    label_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_count = numeric(predictions, "signal_consensus_peaks").to_numpy(dtype=float)
    for confidence_threshold in [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for margin_threshold in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            _, applied, _, _ = apply_threshold(
                probabilities,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            )
            final_count = np.clip(base_count + applied, 0, None)
            row = metrics_for_counts(
                f"{label_prefix}_threshold_candidate",
                predictions,
                final_count,
                evaluation_note=f"{label_prefix}_threshold_grid",
            )
            row.update(
                {
                    "mode": label_prefix,
                    "confidence_threshold": float(confidence_threshold),
                    "margin_threshold": float(margin_threshold),
                    "applied_corrections": int(np.sum(applied != 0)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_metrics(
    predictions: pd.DataFrame,
    stratified_probabilities: np.ndarray,
    group_probabilities: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_count = numeric(predictions, "signal_consensus_peaks").to_numpy(dtype=float)
    rows.append(
        metrics_for_counts(
            "signal_consensus_baseline",
            predictions,
            base_count,
            evaluation_note="existing_signal_consensus_predictions",
        )
    )
    for mode, probabilities, label in [
        ("stratified", stratified_probabilities, "signal_aware_residual_fixed_oof"),
        (
            "prefix_group",
            group_probabilities,
            "signal_aware_residual_prefix_group_fixed_threshold",
        ),
    ]:
        _, applied, _, _ = apply_threshold(
            probabilities,
            confidence_threshold=float(args.confidence_threshold),
            margin_threshold=float(args.margin_threshold),
        )
        final_count = np.clip(base_count + applied, 0, None)
        row = metrics_for_counts(
            label,
            predictions,
            final_count,
            evaluation_note=(
                f"{mode}_out_of_fold_signal_aware_residual_correction_"
                f"confidence_{args.confidence_threshold:.2f}_"
                f"margin_{args.margin_threshold:.2f}"
            ),
        )
        row.update(
            {
                "mode": mode,
                "confidence_threshold": float(args.confidence_threshold),
                "margin_threshold": float(args.margin_threshold),
                "applied_corrections": int(np.sum(applied != 0)),
            }
        )
        rows.append(row)
    grid = threshold_grid(
        predictions,
        stratified_probabilities,
        label_prefix="stratified",
    )
    best = grid.sort_values(
        ["exact_count", "rr_mae", "applied_corrections"],
        ascending=[False, True, True],
    ).iloc[0]
    best_row = best.to_dict()
    best_row["label"] = "signal_aware_residual_best_threshold_sensitivity"
    best_row["evaluation_note"] = "threshold_selected_on_existing_oof_predictions"
    rows.append(best_row)
    return pd.DataFrame(rows)


def paired_metric_values(
    predictions: pd.DataFrame,
    baseline_count: np.ndarray,
    candidate_count: np.ndarray,
    index: np.ndarray,
) -> dict[str, float]:
    truth_rr = numeric(predictions, "truth_rr").to_numpy(dtype=float)[index]
    truth_count = numeric(predictions, "truth_count").to_numpy(dtype=float)[index]
    duration = numeric(predictions, "rr_duration_seconds").to_numpy(dtype=float)[index]
    baseline_count_sample = baseline_count[index]
    candidate_count_sample = candidate_count[index]
    baseline_rr = baseline_count_sample / duration * 60.0
    candidate_rr = candidate_count_sample / duration * 60.0
    baseline = metric_dict(
        "baseline",
        truth_rr,
        baseline_rr,
        truth_count,
        baseline_count_sample,
        evaluation_note="bootstrap",
    )
    candidate = metric_dict(
        "candidate",
        truth_rr,
        candidate_rr,
        truth_count,
        candidate_count_sample,
        evaluation_note="bootstrap",
    )
    return {
        "delta_rr_r2": float(candidate["rr_r2"]) - float(baseline["rr_r2"]),
        "delta_rr_mae": float(candidate["rr_mae"]) - float(baseline["rr_mae"]),
        "delta_rr_rmse": float(candidate["rr_rmse"]) - float(baseline["rr_rmse"]),
        "delta_exact_count": float(candidate["exact_count"])
        - float(baseline["exact_count"]),
    }


def format_ci(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.4f} ({low:.4f}, {high:.4f})"


def signal_aware_bootstrap_ci(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    candidate_count = numeric(predictions, "signal_aware_final_peaks").to_numpy(dtype=float)
    comparisons = [
        (
            "signal_aware_vs_signal_consensus",
            numeric(predictions, "signal_consensus_peaks").to_numpy(dtype=float),
        ),
        (
            "signal_aware_vs_quality_residual",
            numeric(predictions, "corrected_peaks").to_numpy(dtype=float),
        ),
        (
            "signal_aware_vs_default",
            numeric(predictions, "peaks").to_numpy(dtype=float),
        ),
    ]
    display_names = {
        "delta_rr_r2": "Delta RR R2",
        "delta_rr_mae": "Delta MAE (bpm)",
        "delta_rr_rmse": "Delta RMSE (bpm)",
        "delta_exact_count": "Delta exact count",
    }
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.bootstrap_random_state))
    full_index = np.arange(len(predictions))
    for comparison, baseline_count in comparisons:
        estimates = paired_metric_values(
            predictions,
            baseline_count,
            candidate_count,
            full_index,
        )
        samples = {key: [] for key in estimates}
        for _ in range(int(args.bootstrap_samples)):
            index = rng.integers(0, len(predictions), size=len(predictions))
            sample_values = paired_metric_values(
                predictions,
                baseline_count,
                candidate_count,
                index,
            )
            for key, value in sample_values.items():
                if math.isfinite(float(value)):
                    samples[key].append(float(value))
        for key, estimate in estimates.items():
            values = np.asarray(samples[key], dtype=float)
            low = float(np.quantile(values, 0.025))
            high = float(np.quantile(values, 0.975))
            rows.append(
                {
                    "comparison": comparison,
                    "metric": key,
                    "display_metric": display_names[key],
                    "estimate": float(estimate),
                    "ci_low_2_5": low,
                    "ci_high_97_5": high,
                    "estimate_with_ci": format_ci(float(estimate), low, high),
                    "bootstrap_samples": int(args.bootstrap_samples),
                    "note": "paired_video_level_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    metrics: pd.DataFrame,
    threshold_grid_table: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fixed = metrics[metrics["label"] == "signal_aware_residual_fixed_oof"].iloc[0]
    group = metrics[
        metrics["label"] == "signal_aware_residual_prefix_group_fixed_threshold"
    ].iloc[0]
    base = metrics[metrics["label"] == "signal_consensus_baseline"].iloc[0]
    ci_match = bootstrap_ci[
        (bootstrap_ci["comparison"] == "signal_aware_vs_signal_consensus")
        & (bootstrap_ci["metric"] == "delta_rr_r2")
    ]
    ci_text = (
        str(ci_match.iloc[0]["estimate_with_ci"])
        if not ci_match.empty
        else "not available"
    )
    text = f"""# Signal-Aware Residual Correction Report

This experiment trains a fixed GradientBoosting second-stage classifier on
non-truth signal-consensus features to predict whether the current peak count
should be shifted by -1, 0, or +1. Predictions are out-of-fold, so a video's
reference count is not used to predict that same video.

## Fixed Threshold

- Confidence threshold: {args.confidence_threshold:.2f}
- Margin threshold: {args.margin_threshold:.2f}
- Baseline signal consensus: RR R2={base['rr_r2']:.6f}, MAE={base['rr_mae']:.4f} bpm, exact={int(base['exact_count'])}/{int(base['count_valid_videos'])}
- Stratified OOF signal-aware residual: RR R2={fixed['rr_r2']:.6f}, MAE={fixed['rr_mae']:.4f} bpm, exact={int(fixed['exact_count'])}/{int(fixed['count_valid_videos'])}, applied={int(fixed['applied_corrections'])}
- Prefix-group OOF signal-aware residual: RR R2={group['rr_r2']:.6f}, MAE={group['rr_mae']:.4f} bpm, exact={int(group['exact_count'])}/{int(group['count_valid_videos'])}, applied={int(group['applied_corrections'])}
- Paired bootstrap Delta RR R2 versus signal-consensus baseline: {ci_text}

## Claim Boundary

The stratified OOF result can be used as an internal high-precision candidate.
The prefix-group result is weaker, so this experiment should not replace the
method-frozen main result until real cow_id/session/external validation confirms
the improvement.

## Threshold Grid

The threshold grid is written to `{threshold_grid_table.name}` for sensitivity
analysis. Do not report threshold-selected rows as the primary method estimate.

Bootstrap confidence intervals are written to
`{args.output_prefix}_{args.output_name}_bootstrap_ci.csv`.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = args.input_file or (
        args.input_root / f"{args.output_prefix}_signal_consensus_predictions.csv"
    )
    predictions = pd.read_csv(input_file)
    target = build_signal_aware_target(predictions)
    features, numeric_columns, categorical_columns = build_feature_frame(predictions)

    stratified_probabilities, stratified_folds = out_of_fold_probabilities(
        features,
        target,
        numeric_columns,
        categorical_columns,
        mode="stratified",
        folds=int(args.folds),
        random_state=int(args.random_state),
        model_random_state=int(args.model_random_state),
        video_ids=predictions["video_id"],
    )
    group_probabilities, group_folds = out_of_fold_probabilities(
        features,
        target,
        numeric_columns,
        categorical_columns,
        mode="prefix_group",
        folds=int(args.folds),
        random_state=int(args.random_state),
        model_random_state=int(args.model_random_state),
        video_ids=predictions["video_id"],
    )

    stratified_predictions = add_prediction_columns(
        predictions,
        stratified_probabilities,
        stratified_folds,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
        mode="signal_aware",
    )
    group_predictions = add_prediction_columns(
        predictions,
        group_probabilities,
        group_folds,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
        mode="signal_aware_group",
    )
    metrics = build_metrics(
        predictions,
        stratified_probabilities,
        group_probabilities,
        args,
    )
    stratified_grid = threshold_grid(
        predictions,
        stratified_probabilities,
        label_prefix="stratified",
    )
    group_grid = threshold_grid(
        predictions,
        group_probabilities,
        label_prefix="prefix_group",
    )
    grid = pd.concat([stratified_grid, group_grid], ignore_index=True)
    bootstrap_ci = signal_aware_bootstrap_ci(stratified_predictions, args)

    output_stem = f"{args.output_prefix}_{args.output_name}"
    predictions_path = args.input_root / f"{output_stem}_predictions.csv"
    group_predictions_path = args.input_root / f"{output_stem}_group_predictions.csv"
    metrics_path = args.input_root / f"{output_stem}_metrics.csv"
    grid_path = args.input_root / f"{output_stem}_threshold_grid.csv"
    bootstrap_path = args.input_root / f"{output_stem}_bootstrap_ci.csv"
    features_path = args.input_root / f"{output_stem}_feature_columns.csv"
    report_path = args.input_root / f"{output_stem}_report.md"

    stratified_predictions.to_csv(predictions_path, index=False)
    group_predictions.to_csv(group_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    grid.to_csv(grid_path, index=False)
    bootstrap_ci.to_csv(bootstrap_path, index=False)
    pd.DataFrame(
        {
            "feature_column": numeric_columns + categorical_columns,
            "feature_type": ["numeric"] * len(numeric_columns)
            + ["categorical"] * len(categorical_columns),
        }
    ).to_csv(features_path, index=False)
    write_report(report_path, metrics, grid_path, bootstrap_ci, args)

    print(f"Saved signal-aware predictions: {predictions_path}")
    print(f"Saved signal-aware group predictions: {group_predictions_path}")
    print(f"Saved signal-aware metrics: {metrics_path}")
    print(f"Saved signal-aware threshold grid: {grid_path}")
    print(f"Saved signal-aware bootstrap CI: {bootstrap_path}")
    print(f"Saved signal-aware feature columns: {features_path}")
    print(f"Saved signal-aware report: {report_path}")
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nBootstrap CI:")
    print(bootstrap_ci.to_string(index=False))


if __name__ == "__main__":
    main()
