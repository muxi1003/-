from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    {
        "method": "default_pipeline",
        "label": "Default thermal RR pipeline",
        "rr_col": "default_rr",
        "abs_error_col": "default_abs_rr_error",
        "paper_use": "baseline_internal_uncertainty",
    },
    {
        "method": "quality_residual_fixed_oof",
        "label": "Quality-aware residual correction (fixed threshold, out-of-fold)",
        "rr_col": "quality_residual_fixed_oof_rr",
        "abs_error_col": "quality_residual_fixed_oof_abs_rr_error",
        "paper_use": "main_internal_uncertainty",
    },
    {
        "method": "signal_consensus_fixed_oof",
        "label": "Signal-consensus supplement (fixed threshold, out-of-fold)",
        "rr_col": "signal_consensus_fixed_oof_rr",
        "abs_error_col": "signal_consensus_fixed_oof_abs_rr_error",
        "paper_use": "candidate_internal_uncertainty",
    },
    {
        "method": "signal_aware_safe_fixed_oof",
        "label": "Conservative signal-aware safe gate (fixed threshold, out-of-fold)",
        "rr_col": "signal_aware_safe_fixed_oof_rr",
        "abs_error_col": "signal_aware_safe_fixed_oof_abs_rr_error",
        "paper_use": "highest_precision_internal_uncertainty",
    },
    {
        "method": "signal_aware_safe_prefix_group_fixed",
        "label": "Conservative signal-aware safe gate (leave-one-prefix-group-out, fixed threshold)",
        "rr_col": "signal_aware_safe_prefix_group_fixed_rr",
        "abs_error_col": "signal_aware_safe_prefix_group_fixed_abs_rr_error",
        "paper_use": "prefix_group_stress_uncertainty",
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build leave-one-video-out conformal RR prediction intervals from the "
            "current paired method predictions. This adds an uncertainty-reporting "
            "layer without using truth-calibrated predictions as a deployable method."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--target-coverages",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.95],
        help="Conformal target coverage levels to evaluate.",
    )
    parser.add_argument(
        "--risk-auto-threshold",
        type=float,
        default=None,
        help=(
            "Optional low-risk automatic-report threshold. By default the script "
            "uses the deployment-decision recommendation if available, else 0.30."
        ),
    )
    return parser.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def conformal_quantile(scores: pd.Series, target_coverage: float) -> float:
    valid = finite_series(scores).dropna().to_numpy(dtype=float)
    if len(valid) == 0:
        return math.nan
    rank = math.ceil((len(valid) + 1) * float(target_coverage))
    rank = max(1, min(rank, len(valid)))
    return float(np.sort(valid)[rank - 1])


def load_risk_threshold(input_root: Path, output_prefix: str, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    recommendation = read_optional(input_root / f"{output_prefix}_deployment_decision_recommendation.csv")
    if not recommendation.empty and "operating_point" in recommendation.columns:
        operating_point = str(recommendation.iloc[0].get("operating_point", ""))
        if operating_point.startswith("risk_le_"):
            try:
                return float(operating_point.replace("risk_le_", ""))
            except ValueError:
                pass
    return 0.30


def load_inputs(input_root: Path, output_prefix: str) -> pd.DataFrame:
    paired = read_required(input_root / f"{output_prefix}_rr_method_paired_predictions.csv")
    quality_path = input_root / f"{output_prefix}_algorithmic_quality_predictions.csv"
    quality = read_optional(quality_path)
    if not quality.empty and {
        "video_id",
        "algorithmic_review_risk_score",
    }.issubset(quality.columns):
        paired = paired.merge(
            quality[["video_id", "algorithmic_review_risk_score"]],
            on="video_id",
            how="left",
            validate="one_to_one",
        )
    else:
        paired["algorithmic_review_risk_score"] = 0.0
    paired["algorithmic_review_risk_score"] = (
        finite_series(paired["algorithmic_review_risk_score"]).fillna(0.0).clip(lower=0.0)
    )
    return paired


def available_methods(data: pd.DataFrame) -> list[dict[str, str]]:
    methods = []
    for method in METHODS:
        if method["rr_col"] in data.columns and method["abs_error_col"] in data.columns:
            methods.append(method)
    return methods


def risk_scale(data: pd.DataFrame, variant: str) -> pd.Series:
    if variant == "uniform_abs_error":
        return pd.Series(1.0, index=data.index, dtype=float)
    if variant == "risk_adaptive_fixed_scale":
        return 1.0 + finite_series(data["algorithmic_review_risk_score"]).fillna(0.0).clip(lower=0.0)
    raise ValueError(f"Unknown interval variant: {variant}")


def build_intervals(
    data: pd.DataFrame,
    target_coverages: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = available_methods(data)
    variants = ["uniform_abs_error", "risk_adaptive_fixed_scale"]
    truth_rr = finite_series(data["truth_rr"])

    for method in methods:
        predicted = finite_series(data[method["rr_col"]])
        abs_error = finite_series(data[method["abs_error_col"]])
        for target_coverage in target_coverages:
            for variant in variants:
                scales = risk_scale(data, variant)
                normalized_errors = abs_error / scales
                for idx, row in data.iterrows():
                    calibration_mask = data.index != idx
                    q_value = conformal_quantile(
                        normalized_errors.loc[calibration_mask],
                        target_coverage,
                    )
                    if not np.isfinite(q_value):
                        half_width = math.nan
                    else:
                        half_width = float(q_value * scales.loc[idx])
                    prediction = float(predicted.loc[idx])
                    truth = float(truth_rr.loc[idx])
                    lower = prediction - half_width if np.isfinite(half_width) else math.nan
                    upper = prediction + half_width if np.isfinite(half_width) else math.nan
                    covered = bool(lower <= truth <= upper) if np.isfinite(lower + upper + truth) else False
                    rows.append(
                        {
                            "video_id": row["video_id"],
                            "method": method["method"],
                            "method_label": method["label"],
                            "paper_use": method["paper_use"],
                            "target_coverage": float(target_coverage),
                            "alpha": float(1.0 - target_coverage),
                            "interval_variant": variant,
                            "calibration_scheme": "leave_one_video_out_internal_conformal",
                            "predicted_rr": prediction,
                            "truth_rr": truth,
                            "abs_rr_error": float(abs_error.loc[idx]),
                            "algorithmic_review_risk_score": float(
                                data.loc[idx, "algorithmic_review_risk_score"]
                            ),
                            "risk_scale": float(scales.loc[idx]),
                            "nonconformity_quantile": q_value,
                            "interval_half_width_bpm": half_width,
                            "interval_width_bpm": half_width * 2.0
                            if np.isfinite(half_width)
                            else math.nan,
                            "lower_rr_bpm": lower,
                            "upper_rr_bpm": upper,
                            "covered": covered,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_group(group: pd.DataFrame, risk_threshold: float) -> dict[str, object]:
    covered = group["covered"].astype(bool)
    widths = finite_series(group["interval_width_bpm"])
    low_risk = finite_series(group["algorithmic_review_risk_score"]) <= risk_threshold
    high_risk = ~low_risk
    return {
        "videos": int(len(group)),
        "coverage": float(covered.mean()) if len(group) else math.nan,
        "covered_videos": int(covered.sum()),
        "mean_width_bpm": float(widths.mean()) if widths.notna().any() else math.nan,
        "median_width_bpm": float(widths.median()) if widths.notna().any() else math.nan,
        "max_width_bpm": float(widths.max()) if widths.notna().any() else math.nan,
        "mean_abs_rr_error": float(finite_series(group["abs_rr_error"]).mean()),
        "low_risk_threshold": float(risk_threshold),
        "low_risk_videos": int(low_risk.sum()),
        "low_risk_coverage": float(covered[low_risk].mean()) if low_risk.any() else math.nan,
        "low_risk_mean_width_bpm": float(widths[low_risk].mean()) if low_risk.any() else math.nan,
        "high_risk_videos": int(high_risk.sum()),
        "high_risk_coverage": float(covered[high_risk].mean()) if high_risk.any() else math.nan,
        "high_risk_mean_width_bpm": float(widths[high_risk].mean()) if high_risk.any() else math.nan,
    }


def build_metrics(intervals: pd.DataFrame, risk_threshold: float) -> pd.DataFrame:
    rows = []
    group_columns = [
        "method",
        "method_label",
        "paper_use",
        "target_coverage",
        "interval_variant",
        "calibration_scheme",
    ]
    for keys, group in intervals.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row.update(summarize_group(group, risk_threshold))
        row["coverage_minus_target"] = row["coverage"] - float(row["target_coverage"])
        row["internal_status"] = (
            "meets_target_internal"
            if row["coverage"] >= float(row["target_coverage"])
            else "below_target_internal"
        )
        row["claim_boundary"] = (
            "Internal leave-one-video-out conformal uncertainty only; freeze on "
            "training data and verify on an independent external set before "
            "prospective reliability claims."
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_recommendation(metrics: pd.DataFrame) -> pd.DataFrame:
    preferred = metrics[
        (metrics["method"] == "signal_aware_safe_fixed_oof")
        & (metrics["target_coverage"].round(6) == 0.90)
    ].copy()
    if preferred.empty:
        preferred = metrics[
            (metrics["method"] == "quality_residual_fixed_oof")
            & (metrics["target_coverage"].round(6) == 0.90)
        ].copy()
    if preferred.empty:
        preferred = metrics.copy()
    adaptive = preferred[
        preferred["interval_variant"].astype(str) == "risk_adaptive_fixed_scale"
    ]
    if not adaptive.empty:
        row = adaptive.iloc[0].copy()
    else:
        row = preferred.iloc[0].copy()
    row["recommendation_role"] = "internal_reliability_supplement"
    row["selection_rule"] = (
        "Prespecified 90% internal conformal interval for the strongest "
        "non-truth safety-gated RR candidate; use as reliability supplement, "
        "not as external validation."
    )
    return pd.DataFrame([row])


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    table = df[columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: fmt_float(value))
    table = table.fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    recommendation: pd.DataFrame,
    risk_threshold: float,
) -> None:
    primary = metrics[
        (metrics["target_coverage"].round(6) == 0.90)
        & metrics["method"].isin(
            [
                "default_pipeline",
                "quality_residual_fixed_oof",
                "signal_consensus_fixed_oof",
                "signal_aware_safe_fixed_oof",
            ]
        )
    ].copy()
    lines = [
        "# Internal Conformal RR Uncertainty Report",
        "",
        "This report adds prediction intervals to the current thermal RR point estimates. It uses leave-one-video-out conformal calibration on the current 73-video internal dataset. It does not use truth-calibrated predictions as a deployable method and does not create external validation evidence.",
        "",
        f"Low-risk threshold used for subset summaries: `{risk_threshold:.2f}`.",
        "",
        "## Recommended Internal Reliability Supplement",
        "",
        markdown_table(
            recommendation,
            [
                "method_label",
                "target_coverage",
                "interval_variant",
                "coverage",
                "mean_width_bpm",
                "median_width_bpm",
                "low_risk_videos",
                "low_risk_coverage",
                "low_risk_mean_width_bpm",
                "internal_status",
            ],
        ),
        "",
        "## 90% Internal Coverage Summary",
        "",
        markdown_table(
            primary,
            [
                "method_label",
                "interval_variant",
                "coverage",
                "coverage_minus_target",
                "mean_width_bpm",
                "median_width_bpm",
                "low_risk_videos",
                "low_risk_coverage",
                "high_risk_coverage",
                "internal_status",
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "- These intervals are an internal uncertainty and deployment-reporting supplement.",
        "- Use the risk-adaptive interval only as a non-truth algorithmic quality layer; it is not a manual quality score.",
        "- Freeze the interval method and risk threshold on development data before any external/prospective validation.",
        "- Report coverage and interval width together; do not report interval coverage as point-estimate accuracy.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    risk_threshold = load_risk_threshold(
        input_root,
        args.output_prefix,
        args.risk_auto_threshold,
    )
    data = load_inputs(input_root, args.output_prefix)
    intervals = build_intervals(data, list(args.target_coverages))
    metrics = build_metrics(intervals, risk_threshold)
    recommendation = build_recommendation(metrics)

    intervals_path = input_root / f"{args.output_prefix}_conformal_rr_intervals.csv"
    metrics_path = input_root / f"{args.output_prefix}_conformal_rr_metrics.csv"
    recommendation_path = input_root / f"{args.output_prefix}_conformal_rr_recommendation.csv"
    report_path = input_root / f"{args.output_prefix}_conformal_rr_report.md"

    intervals.to_csv(intervals_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    recommendation.to_csv(recommendation_path, index=False)
    write_report(report_path, metrics, recommendation, risk_threshold)

    print(f"Saved conformal RR intervals: {intervals_path}")
    print(f"Saved conformal RR metrics: {metrics_path}")
    print(f"Saved conformal RR recommendation: {recommendation_path}")
    print(f"Saved conformal RR report: {report_path}")
    print("\nRecommendation:")
    print(
        recommendation[
            [
                "method",
                "target_coverage",
                "interval_variant",
                "coverage",
                "mean_width_bpm",
                "low_risk_videos",
                "low_risk_coverage",
                "internal_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
