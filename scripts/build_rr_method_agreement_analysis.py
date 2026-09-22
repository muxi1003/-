from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    rr_col: str
    count_col: str
    paper_use: str
    include_plot: bool = False


METHODS = [
    MethodSpec(
        "default",
        "Default thermal RR pipeline",
        "default_rr",
        "default_count",
        "baseline",
        include_plot=True,
    ),
    MethodSpec(
        "quality_residual_fixed_oof",
        "Quality-aware residual correction",
        "quality_residual_fixed_oof_rr",
        "quality_residual_fixed_oof_count",
        "main_internal_innovation",
        include_plot=True,
    ),
    MethodSpec(
        "signal_consensus_fixed_oof",
        "Signal-consensus supplement",
        "signal_consensus_fixed_oof_rr",
        "signal_consensus_fixed_oof_count",
        "candidate_precision_extension",
    ),
    MethodSpec(
        "signal_aware_safe_fixed_oof",
        "Conservative signal-aware safe gate",
        "signal_aware_safe_fixed_oof_rr",
        "signal_aware_safe_fixed_oof_count",
        "highest_precision_internal_candidate",
        include_plot=True,
    ),
    MethodSpec(
        "signal_aware_safe_prefix_group_fixed",
        "Safe gate, leave-one-prefix-group-out",
        "signal_aware_safe_prefix_group_fixed_rr",
        "signal_aware_safe_prefix_group_fixed_count",
        "internal_group_stress_test",
    ),
]

TRUTH_RR_BINS = [
    ("low_rr_lt_50", -math.inf, 50.0),
    ("mid_rr_50_to_70", 50.0, 70.0),
    ("high_rr_ge_70", 70.0, math.inf),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build Bland-Altman and agreement summaries for RR methods against "
            "manual reference RR."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paired-csv", type=Path, default=None)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_paired_csv(args: argparse.Namespace) -> Path:
    return args.input_root / f"{args.output_prefix}_rr_method_paired_predictions.csv"


def numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def regression_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(pred)
    if int(valid.sum()) < 2:
        return math.nan
    truth_values = truth[valid]
    pred_values = pred[valid]
    sst = float(np.sum((truth_values - np.mean(truth_values)) ** 2))
    if math.isclose(sst, 0.0):
        return math.nan
    sse = float(np.sum((truth_values - pred_values) ** 2))
    return 1.0 - sse / sst


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def agreement_row(data: pd.DataFrame, spec: MethodSpec) -> dict[str, object]:
    truth_rr = numeric(data, "truth_rr")
    pred_rr = numeric(data, spec.rr_col)
    truth_count = numeric(data, "truth_count")
    pred_count = numeric(data, spec.count_col)
    valid_rr = truth_rr.notna() & pred_rr.notna()
    valid_count = truth_count.notna() & pred_count.notna()
    rr_error = (pred_rr[valid_rr] - truth_rr[valid_rr]).to_numpy(dtype=float)
    truth_rr_values = truth_rr[valid_rr].to_numpy(dtype=float)
    pred_rr_values = pred_rr[valid_rr].to_numpy(dtype=float)
    mean_rr = (truth_rr_values + pred_rr_values) / 2.0
    count_error = (pred_count[valid_count] - truth_count[valid_count]).to_numpy(dtype=float)
    bias = float(np.mean(rr_error)) if len(rr_error) else math.nan
    sd = float(np.std(rr_error, ddof=1)) if len(rr_error) > 1 else math.nan
    loa_lower = bias - 1.96 * sd if math.isfinite(sd) else math.nan
    loa_upper = bias + 1.96 * sd if math.isfinite(sd) else math.nan
    slope = math.nan
    slope_p = math.nan
    if len(rr_error) >= 3 and np.std(mean_rr) > 0:
        reg = stats.linregress(mean_rr, rr_error)
        slope = float(reg.slope)
        slope_p = float(reg.pvalue)
    abs_rr = np.abs(rr_error)
    abs_count = np.abs(count_error)
    return {
        "method_id": spec.method_id,
        "display_name": spec.display_name,
        "paper_use": spec.paper_use,
        "n": int(valid_rr.sum()),
        "rr_r2": regression_r2(truth_rr_values, pred_rr_values),
        "rr_bias_bpm": bias,
        "rr_error_sd_bpm": sd,
        "rr_loa_lower_bpm": loa_lower,
        "rr_loa_upper_bpm": loa_upper,
        "rr_loa_width_bpm": loa_upper - loa_lower if math.isfinite(loa_upper) and math.isfinite(loa_lower) else math.nan,
        "rr_mae_bpm": float(np.mean(abs_rr)) if len(abs_rr) else math.nan,
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))) if len(rr_error) else math.nan,
        "rr_median_abs_error_bpm": float(np.median(abs_rr)) if len(abs_rr) else math.nan,
        "rr_abs_error_p75_bpm": float(np.percentile(abs_rr, 75)) if len(abs_rr) else math.nan,
        "rr_abs_error_p95_bpm": float(np.percentile(abs_rr, 95)) if len(abs_rr) else math.nan,
        "rr_abs_error_le_1_bpm": int(np.sum(abs_rr <= 1.0)) if len(abs_rr) else 0,
        "rr_abs_error_le_2_bpm": int(np.sum(abs_rr <= 2.0)) if len(abs_rr) else 0,
        "count_exact": int(np.sum(abs_count == 0)) if len(abs_count) else 0,
        "count_within_one": int(np.sum(abs_count <= 1)) if len(abs_count) else 0,
        "count_error_ge_2": int(np.sum(abs_count >= 2)) if len(abs_count) else 0,
        "count_overestimates": int(np.sum(count_error > 0)) if len(count_error) else 0,
        "count_underestimates": int(np.sum(count_error < 0)) if len(count_error) else 0,
        "proportional_bias_slope": slope,
        "proportional_bias_p": slope_p,
    }


def strata_rows(data: pd.DataFrame, spec: MethodSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    truth_rr = numeric(data, "truth_rr")
    pred_rr = numeric(data, spec.rr_col)
    truth_count = numeric(data, "truth_count")
    pred_count = numeric(data, spec.count_col)
    for label, low, high in TRUTH_RR_BINS:
        mask = truth_rr.notna() & pred_rr.notna() & (truth_rr >= low) & (truth_rr < high)
        count_mask = truth_count.notna() & pred_count.notna() & (truth_rr >= low) & (truth_rr < high)
        rr_error = (pred_rr[mask] - truth_rr[mask]).to_numpy(dtype=float)
        count_error = (pred_count[count_mask] - truth_count[count_mask]).to_numpy(dtype=float)
        abs_rr = np.abs(rr_error)
        abs_count = np.abs(count_error)
        rows.append(
            {
                "method_id": spec.method_id,
                "display_name": spec.display_name,
                "truth_rr_stratum": label,
                "n": int(mask.sum()),
                "rr_bias_bpm": float(np.mean(rr_error)) if len(rr_error) else math.nan,
                "rr_mae_bpm": float(np.mean(abs_rr)) if len(abs_rr) else math.nan,
                "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))) if len(rr_error) else math.nan,
                "count_exact": int(np.sum(abs_count == 0)) if len(abs_count) else 0,
                "count_within_one": int(np.sum(abs_count <= 1)) if len(abs_count) else 0,
                "count_error_ge_2": int(np.sum(abs_count >= 2)) if len(abs_count) else 0,
            }
        )
    return rows


def comparison_rows(summary: pd.DataFrame, baseline_id: str = "default") -> pd.DataFrame:
    baseline = summary[summary["method_id"] == baseline_id]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        if row["method_id"] == baseline_id:
            continue
        rows.append(
            {
                "baseline_method_id": baseline_id,
                "method_id": row["method_id"],
                "delta_rr_r2": float(row["rr_r2"]) - float(base["rr_r2"]),
                "delta_rr_mae_bpm": float(row["rr_mae_bpm"]) - float(base["rr_mae_bpm"]),
                "delta_rr_rmse_bpm": float(row["rr_rmse_bpm"]) - float(base["rr_rmse_bpm"]),
                "delta_rr_loa_width_bpm": float(row["rr_loa_width_bpm"]) - float(base["rr_loa_width_bpm"]),
                "delta_count_exact": int(row["count_exact"]) - int(base["count_exact"]),
                "delta_count_error_ge_2": int(row["count_error_ge_2"]) - int(base["count_error_ge_2"]),
                "interpretation": (
                    "negative LoA width/MAE/RMSE deltas indicate tighter agreement than baseline"
                ),
            }
        )
    return pd.DataFrame(rows)


def bland_altman_plot(data: pd.DataFrame, spec: MethodSpec, output_path: Path) -> None:
    truth = numeric(data, "truth_rr")
    pred = numeric(data, spec.rr_col)
    mask = truth.notna() & pred.notna()
    mean_rr = ((truth[mask] + pred[mask]) / 2.0).to_numpy(dtype=float)
    error = (pred[mask] - truth[mask]).to_numpy(dtype=float)
    if len(error) < 2:
        return
    bias = float(np.mean(error))
    sd = float(np.std(error, ddof=1))
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.scatter(mean_rr, error, s=28, alpha=0.78, color="#2f6f7e", edgecolors="white", linewidths=0.4)
    ax.axhline(bias, color="#1f2937", linewidth=1.4, label=f"bias {bias:.2f}")
    ax.axhline(loa_lower, color="#b45309", linestyle="--", linewidth=1.2, label=f"-1.96 SD {loa_lower:.2f}")
    ax.axhline(loa_upper, color="#b45309", linestyle="--", linewidth=1.2, label=f"+1.96 SD {loa_upper:.2f}")
    ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.set_title(spec.display_name)
    ax.set_xlabel("Mean of manual and predicted RR (bpm)")
    ax.set_ylabel("Predicted - manual RR (bpm)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if data.empty:
        return "_No rows available._"
    table = data[columns].head(max_rows).copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: fmt(value, 3))
    table = table.fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for values in table.to_numpy():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    best = summary.sort_values(["rr_mae_bpm", "rr_loa_width_bpm"], ascending=[True, True]).iloc[0]
    text = f"""# RR Method Agreement Analysis

This report adds Bland-Altman style agreement evidence to the RR method table.
It complements R2/MAE/RMSE and is intended for manuscript Results or
Supplementary Methods.

Best internal agreement by MAE: `{best['display_name']}` with MAE
`{fmt(best['rr_mae_bpm'])}` bpm and LoA width `{fmt(best['rr_loa_width_bpm'])}`
bpm.

## Agreement Summary

{markdown_table(summary, ['method_id', 'paper_use', 'n', 'rr_r2', 'rr_bias_bpm', 'rr_loa_lower_bpm', 'rr_loa_upper_bpm', 'rr_loa_width_bpm', 'rr_mae_bpm', 'rr_rmse_bpm', 'count_exact', 'count_error_ge_2'])}

## Delta Versus Default

{markdown_table(comparison, ['method_id', 'delta_rr_r2', 'delta_rr_mae_bpm', 'delta_rr_rmse_bpm', 'delta_rr_loa_width_bpm', 'delta_count_exact', 'delta_count_error_ge_2'])}

## Truth-RR Strata

{markdown_table(strata, ['method_id', 'truth_rr_stratum', 'n', 'rr_bias_bpm', 'rr_mae_bpm', 'rr_rmse_bpm', 'count_exact', 'count_error_ge_2'], max_rows=60)}

## Figures

{chr(10).join(f'- `{path}`' for path in figure_paths)}

## Claim Boundary

These are internal agreement results on the current 73-video development set.
They should not be described as external validation. After the 119 `split_all_use`
clips receive blinded A/B consensus labels, rerun this analysis on the external
prediction table or extend the script to the external paired predictions.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = (
        args.paired_csv.resolve()
        if args.paired_csv is not None
        else default_paired_csv(args).resolve()
    )
    if not paired_csv.exists():
        raise FileNotFoundError(f"Missing paired predictions CSV: {paired_csv}")
    data = pd.read_csv(paired_csv)
    missing_base = {"video_id", "truth_count", "truth_rr"} - set(data.columns)
    if missing_base:
        raise ValueError(f"{paired_csv} is missing required columns: {sorted(missing_base)}")
    available = [
        spec
        for spec in METHODS
        if spec.rr_col in data.columns and spec.count_col in data.columns
    ]
    if not available:
        raise ValueError("No configured method columns were found in paired predictions CSV")

    summary = pd.DataFrame([agreement_row(data, spec) for spec in available])
    strata = pd.DataFrame([row for spec in available for row in strata_rows(data, spec)])
    comparison = comparison_rows(summary)
    figure_dir = output_dir / "paper_rr_method_agreement_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for spec in available:
        if not spec.include_plot:
            continue
        fig_path = figure_dir / f"paper_rr_method_agreement_bland_altman_{spec.method_id}.png"
        bland_altman_plot(data, spec, fig_path)
        if fig_path.exists():
            figure_paths.append(fig_path)

    summary_path = output_dir / "paper_rr_method_agreement_summary.csv"
    comparison_path = output_dir / "paper_rr_method_agreement_delta_vs_default.csv"
    strata_path = output_dir / "paper_rr_method_agreement_truth_rr_strata.csv"
    report_path = output_dir / "paper_rr_method_agreement_analysis.md"
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    strata.to_csv(strata_path, index=False)
    write_report(report_path, summary, comparison, strata, figure_paths)

    print(f"Saved RR method agreement summary: {summary_path}")
    print(f"Saved RR method agreement deltas: {comparison_path}")
    print(f"Saved RR method agreement strata: {strata_path}")
    print(f"Saved RR method agreement report: {report_path}")
    print(summary[["method_id", "rr_bias_bpm", "rr_loa_width_bpm", "rr_mae_bpm", "count_exact"]].to_string(index=False))


if __name__ == "__main__":
    main()
