from __future__ import annotations

import argparse
import inspect
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import peak_prominences
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_CLASSES = np.array([-1, 0, 1], dtype=int)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a quality-aware residual peak-count corrector with "
            "out-of-fold predictions."
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


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def empty_curve_features() -> dict[str, float | int]:
    return {
        "curve_smoothed_frames": 0,
        "curve_peak_count": 0,
        "curve_first_gap": math.nan,
        "curve_last_gap": math.nan,
        "curve_edge_gap_ratio": math.nan,
        "curve_median_interval": math.nan,
        "curve_mean_interval": math.nan,
        "curve_interval_cv": math.nan,
        "curve_min_interval_ratio": math.nan,
        "curve_max_interval_ratio": math.nan,
        "curve_mean_prominence": math.nan,
        "curve_median_prominence": math.nan,
        "curve_min_prominence_ratio": math.nan,
        "curve_first_prominence_ratio": math.nan,
        "curve_last_prominence_ratio": math.nan,
        "curve_amplitude": math.nan,
        "curve_std": math.nan,
        "curve_fused_repaired_rate": math.nan,
        "curve_left_missing_rate": math.nan,
        "curve_right_missing_rate": math.nan,
        "curve_fusion_source_switches": math.nan,
    }


def curve_quality_features(curve: pd.DataFrame) -> dict[str, float | int]:
    features = empty_curve_features()
    if "smoothed_norm" not in curve.columns or "is_peak" not in curve.columns:
        return features

    smoothed_mask = curve["smoothed_norm"].notna().to_numpy()
    if not smoothed_mask.any():
        return features

    smoothed = curve.loc[smoothed_mask, "smoothed_norm"].to_numpy(dtype=float)
    first_smoothed_index = int(np.where(smoothed_mask)[0][0])
    peak_frames = curve.index[curve["is_peak"].astype(bool)].to_numpy(dtype=int)
    peaks = peak_frames - first_smoothed_index
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed))]
    intervals = np.diff(peaks) if len(peaks) > 1 else np.array([], dtype=float)

    if len(peaks) > 0:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prominences = peak_prominences(smoothed, peaks)[0]
    else:
        prominences = np.array([], dtype=float)

    median_interval = float(np.median(intervals)) if len(intervals) else math.nan
    mean_interval = float(np.mean(intervals)) if len(intervals) else math.nan
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals)
        else math.nan
    )
    min_interval_ratio = (
        float(np.min(intervals) / (median_interval + 1e-9))
        if len(intervals) and not math.isnan(median_interval)
        else math.nan
    )
    max_interval_ratio = (
        float(np.max(intervals) / (median_interval + 1e-9))
        if len(intervals) and not math.isnan(median_interval)
        else math.nan
    )
    median_prominence = float(np.median(prominences)) if len(prominences) else math.nan
    mean_prominence = float(np.mean(prominences)) if len(prominences) else math.nan
    min_prominence_ratio = (
        float(np.min(prominences) / (median_prominence + 1e-9))
        if len(prominences) and not math.isnan(median_prominence)
        else math.nan
    )
    first_prominence_ratio = (
        float(prominences[0] / (median_prominence + 1e-9))
        if len(prominences) and not math.isnan(median_prominence)
        else math.nan
    )
    last_prominence_ratio = (
        float(prominences[-1] / (median_prominence + 1e-9))
        if len(prominences) and not math.isnan(median_prominence)
        else math.nan
    )

    if len(peaks):
        first_gap = int(peaks[0])
        last_gap = int((len(smoothed) - 1) - peaks[-1])
    else:
        first_gap = math.nan
        last_gap = math.nan
    edge_gap_ratio = (
        float(max(first_gap, last_gap) / (median_interval + 1e-9))
        if not math.isnan(finite_float(first_gap))
        and not math.isnan(finite_float(last_gap))
        and not math.isnan(median_interval)
        else math.nan
    )

    features.update(
        {
            "curve_smoothed_frames": int(len(smoothed)),
            "curve_peak_count": int(len(peaks)),
            "curve_first_gap": first_gap,
            "curve_last_gap": last_gap,
            "curve_edge_gap_ratio": edge_gap_ratio,
            "curve_median_interval": median_interval,
            "curve_mean_interval": mean_interval,
            "curve_interval_cv": interval_cv,
            "curve_min_interval_ratio": min_interval_ratio,
            "curve_max_interval_ratio": max_interval_ratio,
            "curve_mean_prominence": mean_prominence,
            "curve_median_prominence": median_prominence,
            "curve_min_prominence_ratio": min_prominence_ratio,
            "curve_first_prominence_ratio": first_prominence_ratio,
            "curve_last_prominence_ratio": last_prominence_ratio,
            "curve_amplitude": float(np.nanmax(smoothed) - np.nanmin(smoothed)),
            "curve_std": float(np.nanstd(smoothed)),
        }
    )

    for column, feature_name in [
        ("fused_repaired", "curve_fused_repaired_rate"),
        ("left_missing_raw", "curve_left_missing_rate"),
        ("right_missing_raw", "curve_right_missing_rate"),
    ]:
        if column in curve.columns:
            features[feature_name] = float(pd.Series(curve[column]).astype(bool).mean())

    if "fusion_source" in curve.columns:
        source = curve["fusion_source"].astype(str).to_numpy()
        features["curve_fusion_source_switches"] = int(np.sum(source[1:] != source[:-1]))

    return features


def safe_adjust_target(summary: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(summary["truth_count"], errors="coerce") - pd.to_numeric(
        summary["peaks"], errors="coerce"
    )
    target = target.clip(lower=-1, upper=1)
    return target.astype(int)


def build_feature_frame(
    summary: pd.DataFrame, input_root: Path, output_prefix: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    excluded = {
        "video_id",
        "truth_count",
        "truth_rr",
        "truth_duration_seconds",
        "count_error",
        "abs_count_error",
        "rr_error",
        "abs_rr_error",
        "temperature_csv",
        "curve_csv",
        "curve_png",
        "review_png",
        "status",
    }
    summary_features = summary.copy()
    for column in list(summary_features.columns):
        lower = column.lower()
        if column in excluded or "error" in lower or lower.startswith("truth_"):
            summary_features = summary_features.drop(columns=[column])

    curve_feature_rows: list[dict[str, float | int]] = []
    for row in summary.itertuples(index=False):
        video_id = str(getattr(row, "video_id"))
        curve_csv = input_root / video_id / f"{output_prefix}_curve.csv"
        if curve_csv.exists():
            curve = pd.read_csv(curve_csv)
            curve_feature_rows.append(curve_quality_features(curve))
        else:
            curve_feature_rows.append(empty_curve_features())

    curve_features = pd.DataFrame(curve_feature_rows)
    features = pd.concat(
        [
            summary_features.reset_index(drop=True),
            curve_features.reset_index(drop=True),
        ],
        axis=1,
    )
    features = features.dropna(axis=1, how="all")
    min_non_missing = max(5, int(math.ceil(0.05 * len(features))))
    features = features.loc[:, features.notna().sum(axis=0) >= min_non_missing]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        if column == "video_id":
            continue
        series = features[column]
        if pd.api.types.is_bool_dtype(series):
            features[column] = series.astype(int)
            numeric_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
            features[column] = series.fillna("missing").astype(str)

    return features[numeric_columns + categorical_columns], numeric_columns, categorical_columns


def make_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def aligned_predict_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    classifier = model.named_steps["classifier"]
    aligned = np.zeros((len(features), len(TARGET_CLASSES)), dtype=float)
    for source_index, class_value in enumerate(classifier.classes_):
        target_index = int(np.where(TARGET_CLASSES == int(class_value))[0][0])
        aligned[:, target_index] = proba[:, source_index]
    return aligned


def out_of_fold_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    min_class_count = int(target.value_counts().min())
    folds = max(2, min(int(args.folds), min_class_count))
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(args.cv_random_state),
    )
    proba = np.zeros((len(features), len(TARGET_CLASSES)), dtype=float)
    fold_ids = np.zeros(len(features), dtype=int)

    for fold_id, (train_index, test_index) in enumerate(cv.split(features, target), start=1):
        model = make_model(
            numeric_columns,
            categorical_columns,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.model_random_state),
        )
        model.fit(features.iloc[train_index], target.iloc[train_index])
        proba[test_index] = aligned_predict_proba(model, features.iloc[test_index])
        fold_ids[test_index] = fold_id

    return proba, fold_ids


def apply_thresholds(
    proba: np.ndarray, confidence_threshold: float, margin_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return predicted_adjust, applied_adjust.astype(int), margin


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = truth.notna() & pred.notna()
    if int(valid.sum()) < 2:
        return math.nan
    truth_values = truth[valid].to_numpy(dtype=float)
    pred_values = pred[valid].to_numpy(dtype=float)
    denominator = float(np.sum((truth_values - np.mean(truth_values)) ** 2))
    if denominator <= 0:
        return math.nan
    return float(1.0 - np.sum((pred_values - truth_values) ** 2) / denominator)


def pearson_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = truth.notna() & pred.notna()
    if int(valid.sum()) < 2:
        return math.nan
    corr = np.corrcoef(truth[valid].to_numpy(dtype=float), pred[valid].to_numpy(dtype=float))[0, 1]
    return float(corr * corr) if math.isfinite(float(corr)) else math.nan


def metrics_row(
    label: str,
    data: pd.DataFrame,
    *,
    rr_column: str,
    count_column: str,
    evaluation_note: str,
) -> dict[str, object]:
    truth_rr = pd.to_numeric(data["truth_rr"], errors="coerce")
    pred_rr = pd.to_numeric(data[rr_column], errors="coerce")
    truth_count = pd.to_numeric(data["truth_count"], errors="coerce")
    pred_count = pd.to_numeric(data[count_column], errors="coerce")
    rr_valid = truth_rr.notna() & pred_rr.notna()
    count_valid = truth_count.notna() & pred_count.notna()
    rr_error = pred_rr[rr_valid] - truth_rr[rr_valid]
    count_error = pred_count[count_valid] - truth_count[count_valid]

    return {
        "label": label,
        "videos": int(len(data)),
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


def feature_importance(
    model: Pipeline, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(numeric_columns + categorical_columns, dtype=object)
    importances = classifier.feature_importances_
    rows = []
    for feature_name, importance in zip(feature_names, importances):
        rows.append(
            {
                "feature": str(feature_name),
                "importance": float(importance),
                "note": "trained_on_all_rows_for_interpretation_only",
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    predictions_csv = input_root / f"{args.corrected_prefix}_predictions.csv"
    metrics_csv = input_root / f"{args.corrected_prefix}_metrics.csv"
    importance_csv = input_root / f"{args.corrected_prefix}_feature_importance.csv"

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
    proba, fold_ids = out_of_fold_predictions(
        features, target, numeric_columns, categorical_columns, args
    )
    predicted_adjust, applied_adjust, margin = apply_thresholds(
        proba,
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
    )
    confidence = np.max(proba, axis=1)

    predictions = summary.copy()
    predictions["target_adjust"] = target.to_numpy(dtype=int)
    predictions["cv_fold"] = fold_ids
    predictions["prob_adjust_minus1"] = proba[:, 0]
    predictions["prob_adjust_0"] = proba[:, 1]
    predictions["prob_adjust_plus1"] = proba[:, 2]
    predictions["predicted_adjust"] = predicted_adjust
    predictions["applied_adjust"] = applied_adjust
    predictions["residual_confidence"] = confidence
    predictions["residual_margin"] = margin
    predictions["corrected_peaks"] = (
        pd.to_numeric(predictions["peaks"], errors="coerce").astype(int) + applied_adjust
    ).clip(lower=0)
    duration = pd.to_numeric(predictions["duration_seconds"], errors="coerce")
    predictions["corrected_rr_bpm"] = predictions["corrected_peaks"] / duration * 60.0
    predictions["corrected_count_error"] = (
        predictions["corrected_peaks"] - pd.to_numeric(predictions["truth_count"], errors="coerce")
    )
    predictions["corrected_abs_count_error"] = predictions["corrected_count_error"].abs()
    predictions["corrected_rr_error"] = (
        predictions["corrected_rr_bpm"] - pd.to_numeric(predictions["truth_rr"], errors="coerce")
    )
    predictions["corrected_abs_rr_error"] = predictions["corrected_rr_error"].abs()
    predictions["evaluation_note"] = (
        "5fold_cross_validated_residual_corrector_no_same_video_training"
    )

    metrics = pd.DataFrame(
        [
            metrics_row(
                "baseline_default",
                predictions,
                rr_column="rr_bpm",
                count_column="peaks",
                evaluation_note="existing_default_outputs",
            ),
            metrics_row(
                "quality_residual_corrected_cv",
                predictions,
                rr_column="corrected_rr_bpm",
                count_column="corrected_peaks",
                evaluation_note=(
                    "out_of_fold_predictions_thresholded_confidence_"
                    f"{float(args.confidence_threshold):.3f}_margin_"
                    f"{float(args.margin_threshold):.3f}"
                ),
            ),
        ]
    )

    full_model = make_model(
        numeric_columns,
        categorical_columns,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.model_random_state),
    )
    full_model.fit(features, target)
    importance = feature_importance(full_model, numeric_columns, categorical_columns)

    predictions.to_csv(predictions_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    importance.to_csv(importance_csv, index=False)

    print(f"Saved predictions: {predictions_csv}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved feature importance: {importance_csv}")
    print(metrics.to_string(index=False))
    changed = predictions[predictions["applied_adjust"] != 0]
    if not changed.empty:
        print("\nApplied corrections:")
        print(
            changed[
                [
                    "video_id",
                    "target_adjust",
                    "peaks",
                    "truth_count",
                    "predicted_adjust",
                    "applied_adjust",
                    "residual_confidence",
                    "residual_margin",
                    "corrected_peaks",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
