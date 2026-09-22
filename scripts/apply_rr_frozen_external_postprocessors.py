from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_corrector import (
    aligned_predict_proba,
    apply_thresholds as apply_quality_thresholds,
    curve_quality_features,
    empty_curve_features,
    make_model as make_quality_model,
    safe_adjust_target,
)
from rr_quality_residual_validation import metric_dict
from rr_signal_aware_residual_validation import (
    build_feature_frame as build_signal_feature_frame,
    build_signal_aware_target,
    full_probability_matrix,
    make_classifier as make_signal_model,
)
from rr_signal_aware_safe_policy import add_safe_policy_columns
from rr_signal_consensus_validation import apply_signal_consensus, build_signal_candidates


QUALITY_EXCLUDED_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_internal_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_external_root = repo_root / "Dataset_new" / "72video" / "external_al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen internal RR post-processors and apply them to an "
            "independent external default RR summary without using external truth "
            "for model selection or threshold tuning."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_internal_root)
    parser.add_argument("--external-input-root", type=Path, default=default_external_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--external-output-prefix", default="external_repro")
    parser.add_argument("--external-curve-prefix", default=None)
    parser.add_argument("--external-summary-csv", type=Path, default=None)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--quality-confidence-threshold", type=float, default=0.54)
    parser.add_argument("--quality-margin-threshold", type=float, default=0.26)
    parser.add_argument("--quality-n-estimators", type=int, default=200)
    parser.add_argument("--quality-max-depth", type=int, default=3)
    parser.add_argument("--quality-min-samples-leaf", type=int, default=4)
    parser.add_argument("--quality-model-random-state", type=int, default=4)
    parser.add_argument("--signal-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--signal-margin-threshold", type=float, default=0.20)
    parser.add_argument("--signal-model-random-state", type=int, default=13)
    parser.add_argument("--supplement-confidence", type=float, default=0.50)
    parser.add_argument("--supplement-margin", type=float, default=0.10)
    parser.add_argument("--min-signal-votes", type=int, default=2)
    parser.add_argument("--min-supplement-interval-cv", type=float, default=0.09)
    parser.add_argument("--safe-max-autocorr-delta", type=float, default=2.0)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def require_columns(data: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_summary(path: Path, *, label: str, require_truth: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    data = pd.read_csv(path)
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds"}
    if require_truth:
        required.update({"truth_count", "truth_rr"})
    require_columns(data, required, label)
    valid = data[["video_id", "peaks", "rr_bpm", "duration_seconds"]].notna().all(axis=1)
    if require_truth:
        valid = valid & data[["truth_count", "truth_rr"]].notna().all(axis=1)
    data = data.loc[valid].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"{label} has no usable rows after required-column filtering: {path}")
    data["video_id"] = data["video_id"].astype(str)
    return data


def quality_raw_feature_frame(
    summary: pd.DataFrame, input_root: Path, curve_prefix: str
) -> pd.DataFrame:
    summary_features = summary.copy()
    for column in list(summary_features.columns):
        lower = column.lower()
        if column in QUALITY_EXCLUDED_COLUMNS or "error" in lower or lower.startswith("truth_"):
            summary_features = summary_features.drop(columns=[column])

    curve_rows: list[dict[str, float | int]] = []
    for video_id in summary["video_id"].astype(str):
        curve_csv = input_root / video_id / f"{curve_prefix}_curve.csv"
        if curve_csv.exists():
            curve_rows.append(curve_quality_features(pd.read_csv(curve_csv)))
        else:
            curve_rows.append(empty_curve_features())
    curve_features = pd.DataFrame(curve_rows)
    return pd.concat(
        [summary_features.reset_index(drop=True), curve_features.reset_index(drop=True)],
        axis=1,
    )


def infer_quality_schema(raw_features: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = raw_features.dropna(axis=1, how="all").copy()
    min_non_missing = max(5, int(math.ceil(0.05 * len(features))))
    features = features.loc[:, features.notna().sum(axis=0) >= min_non_missing]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        if column == "video_id":
            continue
        series = features[column]
        converted = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        elif int(converted.notna().sum()) >= int(0.8 * len(series)):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def align_feature_frame(
    raw_features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    columns: dict[str, object] = {}
    for column in numeric_columns:
        if column in raw_features.columns:
            columns[column] = pd.to_numeric(raw_features[column], errors="coerce")
        else:
            columns[column] = np.nan
    for column in categorical_columns:
        if column in raw_features.columns:
            columns[column] = raw_features[column].fillna("missing").astype(str)
        else:
            columns[column] = "missing"
    aligned = pd.DataFrame(columns, index=raw_features.index)
    return aligned[numeric_columns + categorical_columns]


def rr_duration(data: pd.DataFrame) -> np.ndarray:
    rr_duration_seconds = numeric(data, "rr_duration_seconds")
    if rr_duration_seconds.notna().any():
        return rr_duration_seconds.to_numpy(dtype=float)
    return numeric(data, "duration_seconds").to_numpy(dtype=float)


def rr_from_count(data: pd.DataFrame, count: np.ndarray) -> np.ndarray:
    duration = rr_duration(data)
    return np.divide(
        count * 60.0,
        duration,
        out=np.full(len(count), np.nan, dtype=float),
        where=np.isfinite(duration) & (duration > 0),
    )


def add_quality_prediction_columns(
    summary: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    predicted, applied, margin = apply_quality_thresholds(
        probabilities,
        confidence_threshold=float(confidence_threshold),
        margin_threshold=float(margin_threshold),
    )
    out = summary.copy()
    peaks = numeric(out, "peaks").to_numpy(dtype=float)
    corrected_peaks = np.clip(peaks + applied, 0, None)
    corrected_rr = rr_from_count(out, corrected_peaks)
    out["prob_adjust_minus1"] = probabilities[:, 0]
    out["prob_adjust_0"] = probabilities[:, 1]
    out["prob_adjust_plus1"] = probabilities[:, 2]
    out["predicted_adjust"] = predicted
    out["applied_adjust"] = applied
    out["residual_confidence"] = np.max(probabilities, axis=1)
    out["residual_margin"] = margin
    out["corrected_peaks"] = corrected_peaks
    out["corrected_rr_bpm"] = corrected_rr
    truth_count = numeric(out, "truth_count").to_numpy(dtype=float)
    truth_rr = numeric(out, "truth_rr").to_numpy(dtype=float)
    out["corrected_count_error"] = corrected_peaks - truth_count
    out["corrected_abs_count_error"] = np.abs(corrected_peaks - truth_count)
    out["corrected_rr_error"] = corrected_rr - truth_rr
    out["corrected_abs_rr_error"] = np.abs(corrected_rr - truth_rr)
    out["evaluation_note"] = (
        "frozen_internal_quality_residual_model_applied_to_external_features"
    )
    return out


def apply_signal_threshold(
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_classes = np.array([-1, 0, 1], dtype=int)
    predicted = target_classes[np.argmax(probabilities, axis=1)]
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = sorted_probabilities[:, -1]
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    applied = np.where(
        (confidence >= float(confidence_threshold))
        & (margin >= float(margin_threshold)),
        predicted,
        0,
    )
    return predicted, applied.astype(int), confidence, margin


def add_signal_aware_prediction_columns(
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    out = predictions.copy()
    predicted, applied, confidence, margin = apply_signal_threshold(
        probabilities,
        confidence_threshold=float(confidence_threshold),
        margin_threshold=float(margin_threshold),
    )
    base_count = numeric(out, "signal_consensus_peaks").to_numpy(dtype=float)
    final_count = np.clip(base_count + applied, 0, None)
    final_rr = rr_from_count(out, final_count)
    truth_count = numeric(out, "truth_count").to_numpy(dtype=float)
    truth_rr = numeric(out, "truth_rr").to_numpy(dtype=float)
    out["signal_aware_fold"] = "trained_internal_full_fit"
    out["signal_aware_prob_adjust_minus1"] = probabilities[:, 0]
    out["signal_aware_prob_adjust_0"] = probabilities[:, 1]
    out["signal_aware_prob_adjust_plus1"] = probabilities[:, 2]
    out["signal_aware_predicted_adjust"] = predicted
    out["signal_aware_applied_adjust"] = applied
    out["signal_aware_confidence"] = confidence
    out["signal_aware_margin"] = margin
    out["signal_aware_final_peaks"] = final_count
    out["signal_aware_final_rr_bpm"] = final_rr
    out["signal_aware_count_error"] = final_count - truth_count
    out["signal_aware_abs_count_error"] = np.abs(final_count - truth_count)
    out["signal_aware_rr_error"] = final_rr - truth_rr
    out["signal_aware_abs_rr_error"] = np.abs(final_rr - truth_rr)
    out["signal_aware_evaluation_note"] = (
        "frozen_internal_signal_aware_model_applied_to_external_features"
    )
    return out


def align_signal_features(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    columns: dict[str, object] = {}
    for column in numeric_columns:
        if column in features.columns:
            columns[column] = pd.to_numeric(features[column], errors="coerce")
        else:
            columns[column] = np.nan
    for column in categorical_columns:
        if column in features.columns:
            columns[column] = features[column].fillna("missing").astype(str)
        else:
            columns[column] = "missing"
    aligned = pd.DataFrame(columns, index=features.index)
    return aligned[numeric_columns + categorical_columns]


def truth_available(data: pd.DataFrame) -> bool:
    if not {"truth_rr", "truth_count"}.issubset(data.columns):
        return False
    return bool(numeric(data, "truth_rr").notna().any() and numeric(data, "truth_count").notna().any())


def metric_row(
    label: str,
    data: pd.DataFrame,
    *,
    rr_column: str,
    count_column: str,
    note: str,
) -> dict[str, object]:
    return metric_dict(
        label,
        numeric(data, "truth_rr").to_numpy(dtype=float),
        numeric(data, rr_column).to_numpy(dtype=float),
        numeric(data, "truth_count").to_numpy(dtype=float),
        numeric(data, count_column).to_numpy(dtype=float),
        evaluation_note=note,
    )


def build_metrics(
    summary: pd.DataFrame,
    quality: pd.DataFrame,
    consensus: pd.DataFrame,
    signal_aware: pd.DataFrame,
    safe: pd.DataFrame,
) -> pd.DataFrame:
    if not truth_available(summary):
        return pd.DataFrame(
            columns=[
                "label",
                "videos",
                "rr_valid_videos",
                "rr_r2",
                "rr_pearson_r2",
                "rr_mae",
                "rr_rmse",
                "count_valid_videos",
                "count_mae",
                "exact_count",
                "within_one_count",
                "abs_count_error_ge2",
                "evaluation_note",
            ]
        )
    return pd.DataFrame(
        [
            metric_row(
                "external_default_pipeline",
                summary,
                rr_column="rr_bpm",
                count_column="peaks",
                note="external_default_outputs",
            ),
            metric_row(
                "external_quality_residual_frozen",
                quality,
                rr_column="corrected_rr_bpm",
                count_column="corrected_peaks",
                note="frozen_internal_quality_residual_full_fit",
            ),
            metric_row(
                "external_signal_consensus_frozen",
                consensus,
                rr_column="signal_consensus_rr_bpm",
                count_column="signal_consensus_peaks",
                note="frozen_quality_residual_plus_signal_consensus_rule",
            ),
            metric_row(
                "external_signal_aware_residual_frozen",
                signal_aware,
                rr_column="signal_aware_final_rr_bpm",
                count_column="signal_aware_final_peaks",
                note="frozen_internal_signal_aware_full_fit",
            ),
            metric_row(
                "external_signal_aware_safe_gate_frozen",
                safe,
                rr_column="signal_aware_safe_final_rr_bpm",
                count_column="signal_aware_safe_final_peaks",
                note="frozen_internal_signal_aware_safe_gate",
            ),
        ]
    )


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    internal_summary: Path,
    external_summary: Path,
    external_root: Path,
) -> None:
    lines = [
        "# Frozen External RR Postprocessor Application",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"Internal training summary: `{internal_summary}`",
        f"External default summary: `{external_summary}`",
        f"External output root: `{external_root}`",
        "",
        "## Leakage Boundary",
        "",
        "The fitted postprocessors use the current internal development set for model fitting and fixed thresholds. External `truth_count` and `truth_rr`, when present, are used only for the optional metrics table after predictions are already generated.",
        "",
        "## Metrics",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were written because external truth columns are absent or empty.")
    else:
        lines.append("| label | videos | rr_r2 | rr_mae | rr_rmse | exact_count | within_one_count |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in metrics.iterrows():
            lines.append(
                "| {label} | {videos} | {rr_r2:.6f} | {rr_mae:.4f} | {rr_rmse:.4f} | {exact} | {within_one} |".format(
                    label=row["label"],
                    videos=int(row["videos"]),
                    rr_r2=float(row["rr_r2"]) if np.isfinite(float(row["rr_r2"])) else math.nan,
                    rr_mae=float(row["rr_mae"]) if np.isfinite(float(row["rr_mae"])) else math.nan,
                    rr_rmse=float(row["rr_rmse"]) if np.isfinite(float(row["rr_rmse"])) else math.nan,
                    exact=int(row["exact_count"]),
                    within_one=int(row["within_one_count"]),
                )
            )
    lines.extend(["", "## Output Manifest", ""])
    lines.append("| artifact | rows | columns | purpose |")
    lines.append("|---|---:|---:|---|")
    for _, row in manifest.iterrows():
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | {row['columns']} | {row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Boundary",
            "",
            "Use these outputs as frozen external predictions only after the external videos are independent and the method-freeze CSV is passed into `rr_external_split_validation.py`. Do not tune thresholds on the external rows.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_row(path: Path, data: pd.DataFrame, purpose: str) -> dict[str, object]:
    return {
        "artifact": str(path),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "purpose": purpose,
    }


def namespace_for_consensus(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        supplement_confidence=float(args.supplement_confidence),
        supplement_margin=float(args.supplement_margin),
        min_signal_votes=int(args.min_signal_votes),
        min_supplement_interval_cv=float(args.min_supplement_interval_cv),
    )


def main() -> None:
    args = parse_args()
    internal_root = args.input_root.resolve()
    external_root = args.external_input_root.resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    external_curve_prefix = args.external_curve_prefix or args.external_output_prefix

    internal_summary_path = internal_root / f"{args.output_prefix}_summary.csv"
    external_summary_path = (
        args.external_summary_csv.resolve()
        if args.external_summary_csv is not None
        else external_root / f"{args.external_output_prefix}_summary.csv"
    )
    internal_summary = read_summary(
        internal_summary_path,
        label="internal summary",
        require_truth=True,
    )
    external_summary = read_summary(
        external_summary_path,
        label="external summary",
        require_truth=False,
    )

    quality_raw_train = quality_raw_feature_frame(
        internal_summary, internal_root, args.output_prefix
    )
    quality_numeric, quality_categorical = infer_quality_schema(quality_raw_train)
    quality_train = align_feature_frame(
        quality_raw_train, quality_numeric, quality_categorical
    )
    quality_target = safe_adjust_target(internal_summary)
    quality_model = make_quality_model(
        quality_numeric,
        quality_categorical,
        n_estimators=int(args.quality_n_estimators),
        max_depth=int(args.quality_max_depth),
        min_samples_leaf=int(args.quality_min_samples_leaf),
        random_state=int(args.quality_model_random_state),
    )
    quality_model.fit(quality_train, quality_target)
    quality_external_raw = quality_raw_feature_frame(
        external_summary, external_root, external_curve_prefix
    )
    quality_external = align_feature_frame(
        quality_external_raw, quality_numeric, quality_categorical
    )
    quality_proba = aligned_predict_proba(quality_model, quality_external)
    quality_predictions = add_quality_prediction_columns(
        external_summary,
        quality_proba,
        confidence_threshold=float(args.quality_confidence_threshold),
        margin_threshold=float(args.quality_margin_threshold),
    )

    signal_candidates = build_signal_candidates(
        external_root, external_curve_prefix, quality_predictions["video_id"]
    )
    consensus_predictions = apply_signal_consensus(
        quality_predictions,
        signal_candidates,
        "fixed_oof",
        namespace_for_consensus(args),
    )

    internal_signal_path = internal_root / f"{args.output_prefix}_signal_consensus_predictions.csv"
    if not internal_signal_path.exists():
        raise FileNotFoundError(
            f"Missing internal signal-consensus predictions: {internal_signal_path}"
        )
    internal_signal = pd.read_csv(internal_signal_path)
    signal_target = build_signal_aware_target(internal_signal)
    signal_features, signal_numeric, signal_categorical = build_signal_feature_frame(
        internal_signal
    )
    signal_model = make_signal_model(
        signal_numeric,
        signal_categorical,
        model_random_state=int(args.signal_model_random_state),
    )
    signal_model.fit(signal_features, signal_target)
    external_signal_features, _, _ = build_signal_feature_frame(consensus_predictions)
    external_signal_features = align_signal_features(
        external_signal_features, signal_numeric, signal_categorical
    )
    signal_proba_raw = signal_model.predict_proba(external_signal_features)
    signal_classes = signal_model.named_steps["classifier"].classes_
    signal_proba = full_probability_matrix(signal_proba_raw, signal_classes)
    signal_aware_predictions = add_signal_aware_prediction_columns(
        consensus_predictions,
        signal_proba,
        confidence_threshold=float(args.signal_confidence_threshold),
        margin_threshold=float(args.signal_margin_threshold),
    )
    safe_predictions = add_safe_policy_columns(
        signal_aware_predictions,
        source_prefix="signal_aware",
        output_prefix="signal_aware_safe",
        max_autocorr_delta=float(args.safe_max_autocorr_delta),
    )

    quality_path = external_root / f"{args.corrected_prefix}_predictions.csv"
    candidates_path = external_root / f"{args.external_output_prefix}_signal_consensus_candidates.csv"
    consensus_path = external_root / f"{args.external_output_prefix}_signal_consensus_predictions.csv"
    signal_path = external_root / f"{args.external_output_prefix}_signal_aware_residual_predictions.csv"
    safe_path = external_root / f"{args.external_output_prefix}_signal_aware_safe_policy_predictions.csv"
    metrics_path = external_root / f"{args.external_output_prefix}_frozen_postprocessor_metrics.csv"
    manifest_path = external_root / f"{args.external_output_prefix}_frozen_postprocessor_manifest.csv"
    json_path = external_root / f"{args.external_output_prefix}_frozen_postprocessor_parameters.json"
    report_path = external_root / f"{args.external_output_prefix}_frozen_postprocessor_report.md"

    metrics = build_metrics(
        external_summary,
        quality_predictions,
        consensus_predictions,
        signal_aware_predictions,
        safe_predictions,
    )
    manifest = pd.DataFrame(
        [
            artifact_row(quality_path, quality_predictions, "quality residual external predictions"),
            artifact_row(candidates_path, signal_candidates, "external FFT/autocorrelation candidates"),
            artifact_row(consensus_path, consensus_predictions, "signal-consensus external predictions"),
            artifact_row(signal_path, signal_aware_predictions, "signal-aware residual external predictions"),
            artifact_row(safe_path, safe_predictions, "safe-gated signal-aware external predictions"),
            artifact_row(metrics_path, metrics, "optional external postprocessor metrics"),
        ]
    )

    quality_predictions.to_csv(quality_path, index=False)
    signal_candidates.to_csv(candidates_path, index=False)
    consensus_predictions.to_csv(consensus_path, index=False)
    signal_aware_predictions.to_csv(signal_path, index=False)
    safe_predictions.to_csv(safe_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "internal_summary": str(internal_summary_path),
                "external_summary": str(external_summary_path),
                "external_root": str(external_root),
                "quality_numeric_features": quality_numeric,
                "quality_categorical_features": quality_categorical,
                "signal_numeric_features": signal_numeric,
                "signal_categorical_features": signal_categorical,
                "quality_confidence_threshold": float(args.quality_confidence_threshold),
                "quality_margin_threshold": float(args.quality_margin_threshold),
                "signal_confidence_threshold": float(args.signal_confidence_threshold),
                "signal_margin_threshold": float(args.signal_margin_threshold),
                "safe_max_autocorr_delta": float(args.safe_max_autocorr_delta),
                "external_truth_used_for_training_or_thresholds": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        report_path,
        metrics,
        manifest,
        internal_summary=internal_summary_path,
        external_summary=external_summary_path,
        external_root=external_root,
    )

    print(f"Saved quality residual external predictions: {quality_path}")
    print(f"Saved signal candidates: {candidates_path}")
    print(f"Saved signal-consensus external predictions: {consensus_path}")
    print(f"Saved signal-aware external predictions: {signal_path}")
    print(f"Saved safe-gate external predictions: {safe_path}")
    print(f"Saved postprocessor metrics: {metrics_path}")
    print(f"Saved postprocessor manifest: {manifest_path}")
    print(f"Saved postprocessor report: {report_path}")
    if metrics.empty:
        print("No postprocessor metrics produced because external truth is absent.")
    else:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
