from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_corrector import (
    aligned_predict_proba,
    build_feature_frame,
    make_model,
    out_of_fold_predictions,
    safe_adjust_target,
)
from rr_quality_residual_validation import apply_threshold, metric_dict


DIAGNOSTIC_COLUMNS = [
    "raw_curve_duration_seconds",
    "fixed_window_duration_seconds",
    "fixed_window_start_frame",
    "fixed_window_quality_score",
    "adaptive_fixed_window_duration_seconds",
    "adaptive_fixed_window_start_frame",
    "adaptive_fixed_window_quality_score",
    "left_weight",
    "right_weight",
    "left_quality_score",
    "right_quality_score",
    "left_missing_fraction",
    "right_missing_fraction",
    "adaptive_selected_fusion_mode",
    "adaptive_fusion_full_clip_raw_peaks",
    "adaptive_fusion_full_clip_raw_selected_prominence",
    "adaptive_fusion_full_clip_double_suppressed_peaks",
    "adaptive_fusion_full_clip_double_suppressed_double_peaks_removed",
    "adaptive_fusion_full_clip_periodic_edge_peaks",
    "adaptive_fusion_full_clip_periodic_edge_edge_peaks_added",
    "adaptive_fusion_full_clip_full_peaks",
    "adaptive_fusion_full_clip_full_double_peaks_removed",
    "adaptive_fusion_full_clip_full_edge_peaks_added",
    "adaptive_fusion_fixed_window_peaks",
    "adaptive_fusion_fixed_window_double_suppressed_peaks",
    "adaptive_fusion_fixed_window_double_suppressed_double_peaks_removed",
    "adaptive_fusion_fixed_window_periodic_edge_peaks",
    "adaptive_fusion_fixed_window_periodic_edge_edge_peaks_added",
    "adaptive_fusion_fixed_window_full_peaks",
    "adaptive_fusion_fixed_window_full_double_peaks_removed",
    "adaptive_fusion_fixed_window_full_edge_peaks_added",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "Dataset_new" / "72video" / "al_images"
    assets = root / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a quality-residual corrector augmented with fixed-window, edge, "
            "double-peak, and bilateral-fusion diagnostics. Diagnostic extraction is "
            "label-free; labels are used only to train and score out-of-fold corrections."
        )
    )
    parser.add_argument("--input-root", type=Path, default=root)
    parser.add_argument("--summary-csv", type=Path, default=root / "paper_repro_summary.csv")
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=assets / "paper_bounded_bilateral_peak_fusion_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--curve-prefix", default="paper_repro")
    parser.add_argument("--confidence-threshold", type=float, default=0.54)
    parser.add_argument("--margin-threshold", type=float, default=0.26)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument("--model-random-state", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser.parse_args()


def numeric(values: pd.Series | object) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def video_prefix(video_id: object) -> str:
    match = re.match(r"^[A-Za-z]+", str(video_id))
    return match.group(0) if match else "numeric"


def diagnostic_features(path: Path, video_ids: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = pd.read_csv(path, dtype={"video_id": str})
    if "video_id" not in data.columns:
        raise ValueError(f"Diagnostic table lacks video_id: {path}")
    missing = sorted(set(DIAGNOSTIC_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Diagnostic table lacks required non-truth columns: {missing}")
    selected = data[["video_id", *DIAGNOSTIC_COLUMNS]].copy()
    selected = selected.rename(columns={column: f"diagnostic_{column}" for column in DIAGNOSTIC_COLUMNS})
    merged = pd.DataFrame({"video_id": video_ids.astype(str)}).merge(
        selected, on="video_id", how="left", validate="one_to_one"
    )
    if merged.drop(columns=["video_id"]).isna().all(axis=None):
        raise ValueError("No diagnostic features aligned to the requested summary videos.")
    features = merged.drop(columns=["video_id"])
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
            features[column] = features[column].fillna("missing").astype(str)
    return features, numeric_columns, categorical_columns


def merged_features(
    summary: pd.DataFrame,
    input_root: Path,
    curve_prefix: str,
    diagnostic_csv: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    base, base_numeric, base_categorical = build_feature_frame(summary, input_root, curve_prefix)
    diagnostics, diagnostic_numeric, diagnostic_categorical = diagnostic_features(
        diagnostic_csv, summary["video_id"]
    )
    features = pd.concat([base.reset_index(drop=True), diagnostics.reset_index(drop=True)], axis=1)
    return (
        features,
        [*base_numeric, *diagnostic_numeric],
        [*base_categorical, *diagnostic_categorical],
    )


def corrected_frame(
    summary: pd.DataFrame,
    probabilities: np.ndarray,
    fold_label: pd.Series | np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    predicted, applied, confidence, margin = apply_threshold(
        probabilities,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    result = summary[["video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"]].copy()
    result["fold_label"] = np.asarray(fold_label)
    result["prob_adjust_minus1"] = probabilities[:, 0]
    result["prob_adjust_0"] = probabilities[:, 1]
    result["prob_adjust_plus1"] = probabilities[:, 2]
    result["diagnostic_predicted_adjust"] = predicted
    result["diagnostic_applied_adjust"] = applied
    result["diagnostic_confidence"] = confidence
    result["diagnostic_margin"] = margin
    result["diagnostic_final_peaks"] = np.clip(
        numeric(result["peaks"]).to_numpy(float) + applied, 0, None
    ).astype(int)
    duration = numeric(result["duration_seconds"]).to_numpy(float)
    result["diagnostic_final_rr_bpm"] = np.divide(
        result["diagnostic_final_peaks"].to_numpy(float) * 60.0,
        duration,
        out=np.full(len(result), np.nan),
        where=np.isfinite(duration) & (duration > 0),
    )
    return result


def metrics_row(label: str, frame: pd.DataFrame, *, corrected: bool, note: str) -> dict[str, object]:
    count_column = "diagnostic_final_peaks" if corrected else "peaks"
    rr_column = "diagnostic_final_rr_bpm" if corrected else "rr_bpm"
    return metric_dict(
        label,
        numeric(frame["truth_rr"]).to_numpy(float),
        numeric(frame[rr_column]).to_numpy(float),
        numeric(frame["truth_count"]).to_numpy(float),
        numeric(frame[count_column]).to_numpy(float),
        evaluation_note=note,
    )


def group_predictions(
    summary: pd.DataFrame,
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    target: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    groups = summary["video_id"].map(video_prefix)
    rows: list[pd.DataFrame] = []
    for group in sorted(groups.unique()):
        test_mask = groups.eq(group)
        train_mask = ~test_mask
        model = make_model(
            numeric_columns,
            categorical_columns,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.model_random_state),
        )
        model.fit(features.loc[train_mask], target.loc[train_mask])
        probabilities = aligned_predict_proba(model, features.loc[test_mask])
        fold = np.repeat(f"prefix_{group}", int(test_mask.sum()))
        rows.append(
            corrected_frame(
                summary.loc[test_mask].reset_index(drop=True),
                probabilities,
                fold,
                float(args.confidence_threshold),
                float(args.margin_threshold),
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values("video_id", kind="stable")


def bootstrap_delta(
    frame: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    truth_rr = numeric(frame["truth_rr"]).to_numpy(float)
    truth_count = numeric(frame["truth_count"]).to_numpy(float)
    baseline_rr = numeric(frame["rr_bpm"]).to_numpy(float)
    baseline_count = numeric(frame["peaks"]).to_numpy(float)
    corrected_rr = numeric(frame["diagnostic_final_rr_bpm"]).to_numpy(float)
    corrected_count = numeric(frame["diagnostic_final_peaks"]).to_numpy(float)
    valid = np.isfinite(truth_rr) & np.isfinite(truth_count) & np.isfinite(baseline_rr) & np.isfinite(corrected_rr)
    indices = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    metrics = {"rr_r2": [], "rr_mae": [], "rr_rmse": [], "exact_count": []}
    for _ in range(resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        baseline = metric_dict("baseline", truth_rr[sample], baseline_rr[sample], truth_count[sample], baseline_count[sample], evaluation_note="bootstrap")
        corrected = metric_dict("corrected", truth_rr[sample], corrected_rr[sample], truth_count[sample], corrected_count[sample], evaluation_note="bootstrap")
        for metric in metrics:
            metrics[metric].append(float(corrected[metric]) - float(baseline[metric]))
    baseline = metric_dict("baseline", truth_rr[indices], baseline_rr[indices], truth_count[indices], baseline_count[indices], evaluation_note="point")
    corrected = metric_dict("corrected", truth_rr[indices], corrected_rr[indices], truth_count[indices], corrected_count[indices], evaluation_note="point")
    rows = []
    for metric, draws in metrics.items():
        values = np.asarray(draws, dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "metric": metric,
                "estimate": float(corrected[metric]) - float(baseline[metric]),
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else math.nan,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else math.nan,
                "bootstrap_resamples": int(resamples),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    rows = table[columns].fillna("").astype(str)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows.to_numpy():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, metrics: pd.DataFrame, bootstrap: pd.DataFrame, feature_count: int) -> None:
    r2 = bootstrap[bootstrap["metric"].eq("rr_r2")]
    r2_text = "not computed"
    if not r2.empty:
        row = r2.iloc[0]
        r2_text = f"delta R2={float(row['estimate']):.6f} (95% CI {float(row['ci_low']):.6f} to {float(row['ci_high']):.6f})"
    text = f"""# Structured Peak-Diagnostic Residual Corrector

Status: `exploratory_out_of_fold_internal_candidate`

The predictor starts from the frozen peak count and uses a residual classifier only
to choose a confidence-gated adjustment in `{{-1, 0, +1}}`. It adds {feature_count}
features derived without manual RR: bounded-window diagnostics, periodic-edge support,
double-peak suppression support, and bilateral nostril quality/fusion diagnostics.
Manual counts are used only as the supervised target inside training folds and for
held-out scoring.

## Metrics

{markdown_table(metrics, ['validation', 'label', 'rr_r2', 'rr_mae', 'rr_rmse', 'exact_count', 'within_one_count', 'abs_count_error_ge2', 'applied_corrections'])}

For stratified out-of-fold correction versus the same frozen baseline: {r2_text}.
The prefix-group result is a heuristic domain stress test, not cow-level validation.
This candidate must not replace the manuscript main method unless both evaluation
rows are non-inferior and an independent frozen cohort confirms the gain.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, dtype={"video_id": str})
    required = {"video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")
    features, numeric_columns, categorical_columns = merged_features(
        summary, input_root, args.curve_prefix, args.diagnostic_csv.resolve()
    )
    target = safe_adjust_target(summary)
    probabilities, fold_ids = out_of_fold_predictions(
        features, target, numeric_columns, categorical_columns, args
    )
    oof = corrected_frame(
        summary,
        probabilities,
        fold_ids,
        float(args.confidence_threshold),
        float(args.margin_threshold),
    )
    grouped = group_predictions(summary, features, numeric_columns, categorical_columns, target, args)
    metrics_rows = []
    for validation, frame, note in [
        ("stratified_oof", oof, "stratified_oof_fixed_threshold"),
        ("prefix_group", grouped, "leave_one_prefix_group_out_fixed_threshold"),
    ]:
        baseline = metrics_row("frozen_baseline", frame, corrected=False, note="same_rows_baseline")
        corrected = metrics_row("structured_peak_diagnostic_residual", frame, corrected=True, note=note)
        baseline["validation"] = validation
        baseline["applied_corrections"] = 0
        corrected["validation"] = validation
        corrected["applied_corrections"] = int((frame["diagnostic_applied_adjust"] != 0).sum())
        metrics_rows.extend([baseline, corrected])
    metrics = pd.DataFrame(metrics_rows)
    bootstrap = bootstrap_delta(oof, int(args.bootstrap_resamples), int(args.bootstrap_seed))
    columns = pd.DataFrame(
        {
            "feature": features.columns,
            "source": ["structured_peak_diagnostic" if name.startswith("diagnostic_") else "existing_quality_residual" for name in features.columns],
            "type": ["numeric" if name in numeric_columns else "categorical" for name in features.columns],
        }
    )
    oof_path = output_dir / "paper_structured_peak_diagnostic_residual_oof_predictions.csv"
    group_path = output_dir / "paper_structured_peak_diagnostic_residual_prefix_group_predictions.csv"
    metrics_path = output_dir / "paper_structured_peak_diagnostic_residual_metrics.csv"
    bootstrap_path = output_dir / "paper_structured_peak_diagnostic_residual_bootstrap_ci.csv"
    columns_path = output_dir / "paper_structured_peak_diagnostic_residual_feature_columns.csv"
    report_path = output_dir / "paper_structured_peak_diagnostic_residual_report.md"
    oof.to_csv(oof_path, index=False)
    grouped.to_csv(group_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    columns.to_csv(columns_path, index=False)
    write_report(report_path, metrics, bootstrap, len(features.columns))
    print(f"Saved OOF predictions: {oof_path}")
    print(f"Saved prefix-group predictions: {group_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved bootstrap CIs: {bootstrap_path}")
    print(f"Saved feature columns: {columns_path}")
    print(f"Saved report: {report_path}")
    print(metrics[["validation", "label", "rr_r2", "rr_mae", "rr_rmse", "exact_count", "applied_corrections"]].to_string(index=False))


if __name__ == "__main__":
    main()
