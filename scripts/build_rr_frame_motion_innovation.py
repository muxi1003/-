from __future__ import annotations

import argparse
import inspect
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rr_quality_residual_validation import metric_dict


TARGET_CLASSES = np.array([-1, 0, 1], dtype=int)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Compute frame-level motion/stability features and evaluate whether "
            "they create a non-truth innovation path for motion-aware RR residual "
            "correction."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-sampled-frames", type=int, default=80)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260708)
    parser.add_argument("--model-random-state", type=int, default=20260708)
    parser.add_argument("--confidence-threshold", type=float, default=0.54)
    parser.add_argument("--margin-threshold", type=float, default=0.26)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def sorted_frame_paths(video_dir: Path) -> list[Path]:
    paths = sorted(video_dir.glob("*_frame_*.jpg"))
    if not paths:
        paths = sorted(video_dir.glob("*.jpg"))
    return paths


def sample_indices(count: int, max_samples: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    samples = min(int(count), max(2, int(max_samples)))
    return np.unique(np.linspace(0, count - 1, samples).round().astype(int))


def read_gray(path: Path, resize: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if resize > 0:
        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def safe_stat(values: list[float] | np.ndarray, fn: str) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "std":
        return float(np.std(arr))
    if fn == "p90":
        return float(np.percentile(arr, 90))
    if fn == "p95":
        return float(np.percentile(arr, 95))
    if fn == "max":
        return float(np.max(arr))
    raise ValueError(fn)


def compute_video_motion(video_dir: Path, *, max_samples: int, resize: int) -> dict[str, object]:
    frames = sorted_frame_paths(video_dir)
    indices = sample_indices(len(frames), max_samples)
    sampled_frames: list[np.ndarray] = []
    sampled_paths: list[Path] = []
    for index in indices:
        gray = read_gray(frames[int(index)], resize)
        if gray is not None:
            sampled_frames.append(gray)
            sampled_paths.append(frames[int(index)])

    diffs: list[float] = []
    shifts: list[float] = []
    shift_x: list[float] = []
    shift_y: list[float] = []
    phase_response: list[float] = []
    sharpness: list[float] = []
    brightness: list[float] = []
    contrast: list[float] = []
    for frame in sampled_frames:
        sharpness.append(float(cv2.Laplacian(frame, cv2.CV_32F).var()))
        brightness.append(float(np.mean(frame)))
        contrast.append(float(np.std(frame)))
    hann = None
    for previous, current in zip(sampled_frames[:-1], sampled_frames[1:]):
        diffs.append(float(np.mean(np.abs(current - previous))))
        if hann is None:
            hann = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
        try:
            (dx, dy), response = cv2.phaseCorrelate(previous, current, hann)
            shift_x.append(float(dx))
            shift_y.append(float(dy))
            shifts.append(float(math.hypot(dx, dy)))
            phase_response.append(float(response))
        except cv2.error:
            shift_x.append(math.nan)
            shift_y.append(math.nan)
            shifts.append(math.nan)
            phase_response.append(math.nan)

    return {
        "video_id": video_dir.name,
        "raw_frame_count": int(len(frames)),
        "sampled_frame_count": int(len(sampled_frames)),
        "first_sampled_frame": sampled_paths[0].name if sampled_paths else "",
        "last_sampled_frame": sampled_paths[-1].name if sampled_paths else "",
        "motion_mean_absdiff": safe_stat(diffs, "mean"),
        "motion_p95_absdiff": safe_stat(diffs, "p95"),
        "motion_max_absdiff": safe_stat(diffs, "max"),
        "motion_shift_mean_px": safe_stat(shifts, "mean"),
        "motion_shift_median_px": safe_stat(shifts, "median"),
        "motion_shift_p95_px": safe_stat(shifts, "p95"),
        "motion_shift_max_px": safe_stat(shifts, "max"),
        "motion_shift_x_std_px": safe_stat(shift_x, "std"),
        "motion_shift_y_std_px": safe_stat(shift_y, "std"),
        "motion_phase_response_mean": safe_stat(phase_response, "mean"),
        "motion_phase_response_p90": safe_stat(phase_response, "p90"),
        "frame_sharpness_mean": safe_stat(sharpness, "mean"),
        "frame_sharpness_std": safe_stat(sharpness, "std"),
        "frame_brightness_mean": safe_stat(brightness, "mean"),
        "frame_brightness_std": safe_stat(brightness, "std"),
        "frame_contrast_mean": safe_stat(contrast, "mean"),
        "frame_contrast_std": safe_stat(contrast, "std"),
    }


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    low = float(values.min(skipna=True))
    high = float(values.max(skipna=True))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return pd.Series(0.0, index=series.index)
    return (values - low) / (high - low)


def add_motion_risk(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    components = [
        minmax(result["motion_p95_absdiff"]),
        minmax(result["motion_shift_p95_px"]),
        minmax(result["motion_shift_x_std_px"] + result["motion_shift_y_std_px"]),
        minmax(result["frame_brightness_std"]),
        minmax(result["frame_sharpness_std"]),
    ]
    result["motion_risk_score"] = pd.concat(components, axis=1).mean(axis=1)
    if result["motion_risk_score"].notna().sum() >= 3:
        result["motion_risk_tier"] = pd.qcut(
            result["motion_risk_score"].rank(method="first"),
            q=3,
            labels=["low_motion", "mid_motion", "high_motion"],
        ).astype(str)
    else:
        result["motion_risk_tier"] = "unknown"
    return result


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def bool_rate(series: pd.Series) -> float:
    text = series.astype(str).str.lower()
    if len(text) == 0:
        return math.nan
    return float(text.isin({"true", "1", "yes"}).mean())


def build_tier_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tier, group in joined.groupby("motion_risk_tier", dropna=False):
        rows.append(
            {
                "motion_risk_tier": str(tier),
                "videos": int(len(group)),
                "motion_risk_score_mean": float(group["motion_risk_score"].mean()),
                "motion_shift_p95_px_mean": float(group["motion_shift_p95_px"].mean()),
                "motion_p95_absdiff_mean": float(group["motion_p95_absdiff"].mean()),
                "default_exact_rate": bool_rate(group["default_exact"]),
                "default_abs_count_error_mean": float(numeric(group, "default_abs_count_error").mean()),
                "default_abs_rr_error_mean": float(numeric(group, "default_abs_rr_error").mean()),
                "safe_gate_exact_rate": bool_rate(group["signal_aware_safe_fixed_oof_exact"]),
                "safe_gate_abs_count_error_mean": float(
                    numeric(group, "signal_aware_safe_fixed_oof_abs_count_error").mean()
                ),
                "safe_gate_abs_rr_error_mean": float(
                    numeric(group, "signal_aware_safe_fixed_oof_abs_rr_error").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("motion_risk_tier")


def corr_pair(joined: pd.DataFrame, feature: str, outcome: str, method: str) -> dict[str, object]:
    view = joined[[feature, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
    value = math.nan
    if len(view) >= 3:
        value = float(view[feature].corr(view[outcome], method=method))
    return {
        "feature": feature,
        "outcome": outcome,
        "correlation_method": method,
        "n": int(len(view)),
        "correlation": value,
        "interpretation": (
            "positive means larger motion/stability feature tracks larger RR/count error"
            if math.isfinite(value)
            else "not enough valid rows"
        ),
    }


def build_association(joined: pd.DataFrame) -> pd.DataFrame:
    features = [
        "motion_risk_score",
        "motion_p95_absdiff",
        "motion_shift_p95_px",
        "motion_shift_max_px",
        "frame_sharpness_std",
        "frame_brightness_std",
    ]
    outcomes = [
        "default_abs_count_error",
        "default_abs_rr_error",
        "signal_aware_safe_fixed_oof_abs_count_error",
        "signal_aware_safe_fixed_oof_abs_rr_error",
    ]
    rows = []
    for feature in features:
        for outcome in outcomes:
            for method in ["pearson", "spearman"]:
                rows.append(corr_pair(joined, feature, outcome, method))
    return pd.DataFrame(rows)


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


def is_leakage_column(column: str) -> bool:
    lower = column.lower()
    if any(token in lower for token in ["truth", "error", "abs_", "target_adjust", "evaluation_note"]):
        return True
    if column in {
        "video_id",
        "frame_name",
        "temperature_csv",
        "curve_csv",
        "curve_png",
        "review_png",
        "status",
        "first_sampled_frame",
        "last_sampled_frame",
    }:
        return True
    if column.startswith("prob_adjust") or column.startswith("motion_probe_prob"):
        return True
    if any(
        token in lower
        for token in [
            "predicted_adjust",
            "applied_adjust",
            "corrected_",
            "selective_",
            "signal_aware_",
        ]
    ):
        return True
    return False


def build_feature_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_columns = [column for column in table.columns if not is_leakage_column(column)]
    features = table[feature_columns].copy()
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(features[column]) or int(converted.notna().sum()) >= int(0.8 * len(features)):
            if int(converted.notna().sum()) == 0:
                continue
            features[column] = converted
            numeric_columns.append(column)
        else:
            features[column] = features[column].astype(str)
            categorical_columns.append(column)
    return features[numeric_columns + categorical_columns], numeric_columns, categorical_columns


def make_model(numeric_columns: list[str], categorical_columns: list[str], random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", simple_imputer("median"))]), numeric_columns),
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
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def aligned_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["classifier"].classes_
    result = np.zeros((len(features), len(TARGET_CLASSES)), dtype=float)
    for index, label in enumerate(classes):
        target_index = int(np.where(TARGET_CLASSES == int(label))[0][0])
        result[:, target_index] = probabilities[:, index]
    return result


def out_of_fold_probe(
    table: pd.DataFrame,
    *,
    folds: int,
    random_state: int,
    model_random_state: int,
) -> pd.DataFrame:
    target = (
        numeric(table, "truth_count") - numeric(table, "peaks")
    ).clip(lower=-1, upper=1).astype(int)
    features, numeric_columns, categorical_columns = build_feature_frame(table)
    counts = target.value_counts()
    actual_folds = min(int(folds), int(counts.min())) if not counts.empty else 0
    if actual_folds < 2:
        raise ValueError("Not enough target-class observations for stratified OOF probe.")
    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
    probabilities = np.zeros((len(features), len(TARGET_CLASSES)), dtype=float)
    fold_ids = np.zeros(len(features), dtype=int)
    for fold_id, (train_index, test_index) in enumerate(splitter.split(features, target), start=1):
        model = make_model(numeric_columns, categorical_columns, model_random_state + fold_id)
        model.fit(features.iloc[train_index], target.iloc[train_index])
        probabilities[test_index] = aligned_proba(model, features.iloc[test_index])
        fold_ids[test_index] = fold_id
    result = table.copy()
    result["motion_probe_target_adjust"] = target.to_numpy(dtype=int)
    result["motion_probe_fold"] = fold_ids
    result["motion_probe_prob_adjust_minus1"] = probabilities[:, 0]
    result["motion_probe_prob_adjust_0"] = probabilities[:, 1]
    result["motion_probe_prob_adjust_plus1"] = probabilities[:, 2]
    return result


def apply_threshold(proba: np.ndarray, confidence: float, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(proba, axis=1)
    best_index = order[:, -1]
    second_index = order[:, -2]
    best_confidence = proba[np.arange(len(proba)), best_index]
    best_margin = best_confidence - proba[np.arange(len(proba)), second_index]
    predicted_adjust = TARGET_CLASSES[best_index].astype(int)
    applied_adjust = np.where(
        (predicted_adjust != 0) & (best_confidence >= confidence) & (best_margin >= margin),
        predicted_adjust,
        0,
    ).astype(int)
    return predicted_adjust, applied_adjust, best_confidence, best_margin


def evaluate_probe(table: pd.DataFrame, confidence: float, margin: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    proba = table[
        [
            "motion_probe_prob_adjust_minus1",
            "motion_probe_prob_adjust_0",
            "motion_probe_prob_adjust_plus1",
        ]
    ].to_numpy(dtype=float)
    predicted, applied, best_confidence, best_margin = apply_threshold(proba, confidence, margin)
    peaks = numeric(table, "peaks").to_numpy(dtype=float)
    duration = numeric(table, "duration_seconds").to_numpy(dtype=float)
    corrected_count = np.clip(peaks + applied, 0, None)
    corrected_rr = corrected_count / duration * 60.0
    result = table.copy()
    result["motion_probe_predicted_adjust"] = predicted
    result["motion_probe_applied_adjust"] = applied
    result["motion_probe_confidence"] = best_confidence
    result["motion_probe_margin"] = best_margin
    result["motion_probe_count"] = corrected_count
    result["motion_probe_rr"] = corrected_rr
    result["motion_probe_count_error"] = corrected_count - numeric(table, "truth_count").to_numpy(dtype=float)
    result["motion_probe_abs_count_error"] = np.abs(result["motion_probe_count_error"])
    result["motion_probe_rr_error"] = corrected_rr - numeric(table, "truth_rr").to_numpy(dtype=float)
    result["motion_probe_abs_rr_error"] = np.abs(result["motion_probe_rr_error"])
    metrics = [
        metric_dict(
            "default_thermal_rr_pipeline",
            numeric(table, "truth_rr").to_numpy(dtype=float),
            numeric(table, "rr_bpm").to_numpy(dtype=float),
            numeric(table, "truth_count").to_numpy(dtype=float),
            numeric(table, "peaks").to_numpy(dtype=float),
            evaluation_note="existing_default_outputs",
        ),
        metric_dict(
            "existing_quality_residual_fixed_oof",
            numeric(table, "truth_rr").to_numpy(dtype=float),
            numeric(table, "corrected_rr_bpm").to_numpy(dtype=float),
            numeric(table, "truth_count").to_numpy(dtype=float),
            numeric(table, "corrected_peaks").to_numpy(dtype=float),
            evaluation_note="existing_quality_residual_oof_reference",
        ),
        metric_dict(
            "motion_augmented_residual_probe_fixed_threshold",
            numeric(table, "truth_rr").to_numpy(dtype=float),
            corrected_rr,
            numeric(table, "truth_count").to_numpy(dtype=float),
            corrected_count,
            evaluation_note=(
                "internal_oof_probe_with_frame_motion_features; "
                "not_frozen_primary_method"
            ),
        ),
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table["confidence_threshold"] = float(confidence)
    metrics_table["margin_threshold"] = float(margin)
    metrics_table["applied_corrections"] = [
        0,
        int((numeric(table, "applied_adjust") != 0).sum()) if "applied_adjust" in table.columns else math.nan,
        int(np.sum(applied != 0)),
    ]
    return result, metrics_table


def write_report(
    path: Path,
    features: pd.DataFrame,
    association: pd.DataFrame,
    tiers: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    best_corr = association.sort_values("correlation", key=lambda s: s.abs(), ascending=False).head(6)
    probe = metrics[metrics["label"].eq("motion_augmented_residual_probe_fixed_threshold")]
    default = metrics[metrics["label"].eq("default_thermal_rr_pipeline")]
    quality = metrics[metrics["label"].eq("existing_quality_residual_fixed_oof")]
    probe_row = probe.iloc[0] if not probe.empty else pd.Series(dtype=object)
    default_row = default.iloc[0] if not default.empty else pd.Series(dtype=object)
    quality_row = quality.iloc[0] if not quality.empty else pd.Series(dtype=object)

    def line_for(row: pd.Series) -> str:
        if row.empty:
            return "metrics unavailable"
        return (
            f"RR R2={float(row['rr_r2']):.4f}, MAE={float(row['rr_mae']):.3f}, "
            f"RMSE={float(row['rr_rmse']):.3f}, exact={int(row['exact_count'])}/"
            f"{int(row['count_valid_videos'])}, within-one={int(row['within_one_count'])}/"
            f"{int(row['count_valid_videos'])}"
        )

    def markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No rows available._"
        view = table.fillna("").astype(str)
        lines = [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join(["---"] * len(view.columns)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append(
                "| "
                + " | ".join(str(value).replace("\n", " ") for value in row)
                + " |"
            )
        return "\n".join(lines)

    text = f"""# Frame-Motion-Aware RR Innovation Probe

This report adds a non-truth video-motion layer to the thermal RR package. It
was motivated by head-movement scenarios in recent infrared thermography RR
work and by the practical need to distinguish signal failure from genuine
respiratory periodicity. The current implementation uses only image frames and
pipeline state available at prediction time; manual head-motion, occlusion, and
nostril-visibility scores are still unavailable.

Frame motion was computed for {len(features)} video folders by sampling up to
{int(features['sampled_frame_count'].max()) if not features.empty else 0} frames
per video. The feature set includes frame-to-frame absolute difference,
phase-correlation displacement, sharpness variability, brightness variability,
and a normalized motion-risk score. These features are not yet a validated
manual quality label, but they provide an objective engineering proxy for
whether head motion or video instability could explain residual RR errors.

The fixed-threshold internal motion-augmented residual probe reached
{line_for(probe_row)}. For comparison, the default pipeline was
{line_for(default_row)}, and the existing quality-aware residual corrector was
{line_for(quality_row)}. This probe should be interpreted as an internal
innovation screen only. It should not replace the current conservative
signal-aware safe gate unless it is frozen and passes nested, prefix-group, and
external validation.

## Strongest Motion/Error Associations

{markdown_table(best_corr)}

## Motion-Risk Tier Metrics

{markdown_table(tiers)}

## Claim Boundary

The safe claim is that objective frame-motion features have been added as a
candidate non-truth quality-control layer for future motion-aware RR correction.
The unsafe claim is that head-motion robustness has been externally validated or
manually quality-stratified; those claims still require independent external
videos and manual or protocol-defined quality labels.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = [
        path
        for path in sorted(args.input_root.iterdir())
        if path.is_dir() and sorted_frame_paths(path)
    ]
    rows = [
        compute_video_motion(
            video_dir,
            max_samples=args.max_sampled_frames,
            resize=args.resize,
        )
        for video_dir in video_dirs
    ]
    motion_features = add_motion_risk(pd.DataFrame(rows))

    paired_path = args.input_root / f"{args.output_prefix}_rr_method_paired_predictions.csv"
    quality_path = args.input_root / f"{args.corrected_prefix}_predictions.csv"
    paired = pd.read_csv(paired_path)
    quality = pd.read_csv(quality_path)
    motion_features["video_id"] = motion_features["video_id"].astype(str)
    paired["video_id"] = paired["video_id"].astype(str)
    quality["video_id"] = quality["video_id"].astype(str)

    joined = paired.merge(motion_features, on="video_id", how="left")
    association = build_association(joined)
    tiers = build_tier_metrics(joined)

    probe_input = quality.merge(motion_features, on="video_id", how="left")
    probe_oof = out_of_fold_probe(
        probe_input,
        folds=args.folds,
        random_state=args.random_state,
        model_random_state=args.model_random_state,
    )
    probe_predictions, probe_metrics = evaluate_probe(
        probe_oof,
        confidence=args.confidence_threshold,
        margin=args.margin_threshold,
    )

    features_path = output_dir / "paper_frame_motion_features.csv"
    association_path = output_dir / "paper_frame_motion_error_association.csv"
    tiers_path = output_dir / "paper_frame_motion_tier_metrics.csv"
    probe_predictions_path = output_dir / "paper_frame_motion_augmented_residual_probe_predictions.csv"
    probe_metrics_path = output_dir / "paper_frame_motion_augmented_residual_probe_metrics.csv"
    report_path = output_dir / "paper_frame_motion_innovation_report.md"

    motion_features.to_csv(features_path, index=False)
    association.to_csv(association_path, index=False)
    tiers.to_csv(tiers_path, index=False)
    probe_predictions.to_csv(probe_predictions_path, index=False)
    probe_metrics.to_csv(probe_metrics_path, index=False)
    write_report(report_path, motion_features, association, tiers, probe_metrics)

    print(f"Saved frame motion features: {features_path}")
    print(f"Saved motion/error association: {association_path}")
    print(f"Saved motion tier metrics: {tiers_path}")
    print(f"Saved motion-augmented probe predictions: {probe_predictions_path}")
    print(f"Saved motion-augmented probe metrics: {probe_metrics_path}")
    print(f"Saved motion innovation report: {report_path}")
    print(probe_metrics[["label", "rr_r2", "rr_mae", "rr_rmse", "exact_count", "within_one_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
