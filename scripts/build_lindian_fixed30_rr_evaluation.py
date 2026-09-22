"""Evaluate fixed-window Lindian RR predictions against manual breath counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BOOTSTRAP_METRICS = (
    "rr_r2",
    "rr_pearson_r2",
    "rr_mae_bpm",
    "rr_rmse_bpm",
    "rr_bias_bpm",
    "count_mae",
    "exact_count_rate",
    "within_one_count_rate",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    assets = data_root / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=data_root
        / "lindian_anchored30_frames"
        / "lindian_anchored30_repro_summary.csv",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=assets / "lindian_anchored30_manual_truth.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--output-prefix", default="lindian_anchored30_rr")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def parse_bool(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    values = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    unknown = ~normalized.isin(values)
    if unknown.any():
        raise ValueError(f"Unknown boolean values in {column}: {sorted(set(normalized[unknown]))}")
    return normalized.map(values).astype(bool)


def require_unique_ids(table: pd.DataFrame, name: str) -> None:
    if "video_id" not in table.columns:
        raise ValueError(f"{name} is missing video_id")
    duplicated = table["video_id"].duplicated(keep=False)
    if duplicated.any():
        ids = sorted(set(table.loc[duplicated, "video_id"]))
        raise ValueError(f"{name} contains duplicate video IDs: {ids}")


def coefficient_of_determination(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return 1.0 - float(np.sum((prediction - truth) ** 2)) / denominator


def metric_values(table: pd.DataFrame) -> dict[str, float | int]:
    truth_rr = table["manual_rr_bpm"].to_numpy(dtype=float)
    predicted_rr = table["predicted_rr_bpm"].to_numpy(dtype=float)
    truth_count = table["manual_breath_count"].to_numpy(dtype=float)
    predicted_count = table["predicted_breath_count"].to_numpy(dtype=float)
    rr_error = predicted_rr - truth_rr
    count_error = predicted_count - truth_count

    pearson_r = np.nan
    if len(table) >= 2 and np.std(truth_rr) > 0 and np.std(predicted_rr) > 0:
        pearson_r = float(np.corrcoef(truth_rr, predicted_rr)[0, 1])
    slope = np.nan
    intercept = np.nan
    if len(table) >= 2 and np.std(truth_rr) > 0:
        slope, intercept = np.polyfit(truth_rr, predicted_rr, 1)
    error_sd = float(np.std(rr_error, ddof=1)) if len(table) > 1 else np.nan
    shapiro_p = float(stats.shapiro(rr_error).pvalue) if 3 <= len(table) <= 5000 else np.nan

    exact = np.abs(count_error) < 0.5
    within_one = np.abs(count_error) <= 1.0
    error_ge_two = np.abs(count_error) >= 2.0
    return {
        "videos": len(table),
        "truth_rr_mean_bpm": float(np.mean(truth_rr)),
        "predicted_rr_mean_bpm": float(np.mean(predicted_rr)),
        "rr_r2": coefficient_of_determination(truth_rr, predicted_rr),
        "rr_pearson_r": pearson_r,
        "rr_pearson_r2": pearson_r**2 if np.isfinite(pearson_r) else np.nan,
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "rr_bias_bpm": float(np.mean(rr_error)),
        "rr_error_sd_bpm": error_sd,
        "rr_loa_lower_bpm": float(np.mean(rr_error) - 1.96 * error_sd),
        "rr_loa_upper_bpm": float(np.mean(rr_error) + 1.96 * error_sd),
        "rr_regression_slope": float(slope),
        "rr_regression_intercept": float(intercept),
        "residual_shapiro_p": shapiro_p,
        "count_mae": float(np.mean(np.abs(count_error))),
        "exact_count": int(np.sum(exact)),
        "exact_count_rate": float(np.mean(exact)),
        "within_one_count": int(np.sum(within_one)),
        "within_one_count_rate": float(np.mean(within_one)),
        "count_error_ge_two": int(np.sum(error_ge_two)),
        "count_error_ge_two_rate": float(np.mean(error_ge_two)),
    }


def bootstrap_intervals(
    table: pd.DataFrame,
    *,
    analysis_set: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(iterations):
        sample = table.iloc[rng.integers(0, len(table), size=len(table))]
        sample_metrics = metric_values(sample)
        for metric in BOOTSTRAP_METRICS:
            value = float(sample_metrics[metric])
            if np.isfinite(value):
                values[metric].append(value)

    point = metric_values(table)
    rows: list[dict[str, object]] = []
    for metric in BOOTSTRAP_METRICS:
        samples = np.asarray(values[metric], dtype=float)
        if samples.size == 0:
            lower = upper = np.nan
        else:
            lower, upper = np.percentile(samples, [2.5, 97.5])
        rows.append(
            {
                "analysis_set": analysis_set,
                "metric": metric,
                "point_estimate": point[metric],
                "ci_method": "percentile_bootstrap",
                "ci_level": 0.95,
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_iterations_requested": iterations,
                "bootstrap_iterations_valid": int(samples.size),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def build_report(metrics: pd.DataFrame, intervals: pd.DataFrame, predictions: pd.DataFrame) -> str:
    lines = [
        "# Lindian Anchored 30-Second RR Evaluation",
        "",
        "All predictions were generated with one fixed configuration before statistical stratification. Manual counts were used only for evaluation, not per-video peak tuning.",
        "",
    ]
    labels = {
        "primary_completed": "Primary analysis (completed manual annotations)",
        "sensitivity_all_numeric": "Sensitivity analysis (completed plus uncertain annotations)",
    }
    for analysis_set, label in labels.items():
        row = metrics.loc[metrics["analysis_set"].eq(analysis_set)].iloc[0]
        ci = intervals.loc[
            intervals["analysis_set"].eq(analysis_set) & intervals["metric"].eq("rr_r2")
        ].iloc[0]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Videos: {int(row['videos'])}",
                f"- RR R-squared: {fmt(row['rr_r2'])} (bootstrap 95% CI {fmt(ci['ci_lower'])} to {fmt(ci['ci_upper'])})",
                f"- Pearson r-squared: {fmt(row['rr_pearson_r2'])}",
                f"- MAE: {fmt(row['rr_mae_bpm'])} bpm",
                f"- RMSE: {fmt(row['rr_rmse_bpm'])} bpm",
                f"- Bias: {fmt(row['rr_bias_bpm'])} bpm",
                f"- Exact count: {int(row['exact_count'])}/{int(row['videos'])}",
                f"- Within one breath: {int(row['within_one_count'])}/{int(row['videos'])}",
                "",
            ]
        )
    uncertain = int((predictions["truth_reliability"] == "uncertain").sum())
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            f"The primary result excludes {uncertain} uncertain single-annotator counts. The sensitivity result retains them to show robustness to possible counting error.",
            "These are retrospectively selected, quality-gated windows from the Lindian farm and therefore constitute internal fixed-window validation, not independent external validation.",
            "Percentile bootstrap intervals avoid relying on normally distributed RR residuals; the Shapiro-Wilk p-value is retained only as a residual diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary, dtype={"video_id": str})
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    require_unique_ids(summary, "prediction summary")
    require_unique_ids(truth, "manual truth")
    required_summary = {"video_id", "peaks", "rr_bpm"}
    required_truth = {
        "video_id",
        "breath_count",
        "duration_seconds",
        "manual_rr_bpm",
        "truth_reliability",
        "include_primary_analysis",
        "include_sensitivity_analysis",
    }
    if missing := required_summary - set(summary.columns):
        raise ValueError(f"Prediction summary is missing columns: {sorted(missing)}")
    if missing := required_truth - set(truth.columns):
        raise ValueError(f"Manual truth is missing columns: {sorted(missing)}")
    if set(summary["video_id"]) != set(truth["video_id"]):
        only_summary = sorted(set(summary["video_id"]) - set(truth["video_id"]))
        only_truth = sorted(set(truth["video_id"]) - set(summary["video_id"]))
        raise ValueError(f"Video ID mismatch; only summary={only_summary}, only truth={only_truth}")

    truth = truth.copy()
    truth["include_primary_analysis"] = parse_bool(
        truth["include_primary_analysis"], "include_primary_analysis"
    )
    truth["include_sensitivity_analysis"] = parse_bool(
        truth["include_sensitivity_analysis"], "include_sensitivity_analysis"
    )
    truth_columns = [
        "video_id",
        "window_id",
        "breath_count",
        "duration_seconds",
        "manual_rr_bpm",
        "annotation_status",
        "truth_reliability",
        "include_primary_analysis",
        "include_sensitivity_analysis",
        "bilateral_valid_fraction_1hz",
        "review_notes",
        "raw_source_path",
        "window_start_seconds",
    ]
    truth_columns = [column for column in truth_columns if column in truth.columns]
    predictions = summary.merge(
        truth[truth_columns], on="video_id", how="inner", validate="one_to_one", suffixes=("", "_manual")
    )
    predictions = predictions.rename(
        columns={
            "peaks": "predicted_breath_count",
            "rr_bpm": "predicted_rr_bpm",
            "breath_count": "manual_breath_count",
            "duration_seconds_manual": "manual_window_seconds",
        }
    )
    predictions["count_error"] = (
        predictions["predicted_breath_count"] - predictions["manual_breath_count"]
    )
    predictions["abs_count_error"] = predictions["count_error"].abs()
    predictions["rr_error_bpm"] = predictions["predicted_rr_bpm"] - predictions["manual_rr_bpm"]
    predictions["abs_rr_error_bpm"] = predictions["rr_error_bpm"].abs()
    predictions["analysis_membership"] = np.where(
        predictions["include_primary_analysis"], "primary_and_sensitivity", "sensitivity_only"
    )
    predictions = predictions.sort_values("video_id").reset_index(drop=True)

    analyses = {
        "primary_completed": predictions.loc[predictions["include_primary_analysis"]].copy(),
        "sensitivity_all_numeric": predictions.loc[
            predictions["include_sensitivity_analysis"]
        ].copy(),
    }
    if len(analyses["primary_completed"]) == 0:
        raise ValueError("Primary analysis contains no rows")
    metrics_rows = []
    interval_tables = []
    for index, (analysis_set, table) in enumerate(analyses.items()):
        metrics_rows.append({"analysis_set": analysis_set, **metric_values(table)})
        interval_tables.append(
            bootstrap_intervals(
                table,
                analysis_set=analysis_set,
                iterations=args.bootstrap_iterations,
                seed=args.seed + index,
            )
        )
    metrics = pd.DataFrame(metrics_rows)
    intervals = pd.concat(interval_tables, ignore_index=True)
    for analysis_set in analyses:
        selected = intervals["analysis_set"].eq(analysis_set)
        for metric in BOOTSTRAP_METRICS:
            ci = intervals.loc[selected & intervals["metric"].eq(metric)].iloc[0]
            metrics.loc[metrics["analysis_set"].eq(analysis_set), f"{metric}_ci_lower"] = ci["ci_lower"]
            metrics.loc[metrics["analysis_set"].eq(analysis_set), f"{metric}_ci_upper"] = ci["ci_upper"]

    high_error = predictions.sort_values(
        ["abs_rr_error_bpm", "video_id"], ascending=[False, True]
    ).reset_index(drop=True)
    high_error.insert(0, "error_rank", np.arange(1, len(high_error) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.output_prefix}_predictions.csv"
    metrics_path = args.output_dir / f"{args.output_prefix}_metrics.csv"
    intervals_path = args.output_dir / f"{args.output_prefix}_bootstrap_ci.csv"
    high_error_path = args.output_dir / f"{args.output_prefix}_high_error_cases.csv"
    report_path = args.output_dir / f"{args.output_prefix}_evaluation.md"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    intervals.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    high_error.to_csv(high_error_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(metrics, intervals, predictions), encoding="utf-8")

    print(f"Saved predictions: {predictions_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved bootstrap intervals: {intervals_path.resolve()}")
    print(f"Saved high-error cases: {high_error_path.resolve()}")
    print(f"Saved report: {report_path.resolve()}")
    for _, row in metrics.iterrows():
        print(
            f"{row['analysis_set']}: n={int(row['videos'])}, "
            f"R2={row['rr_r2']:.6f}, MAE={row['rr_mae_bpm']:.3f}, RMSE={row['rr_rmse_bpm']:.3f}"
        )


if __name__ == "__main__":
    main()
