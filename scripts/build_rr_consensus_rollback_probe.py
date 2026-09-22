from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_validation import metric_dict


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a narrow non-truth consensus rollback guard for harmful "
            "quality-residual -1 corrections. This is an exploratory probe, not "
            "the frozen primary method."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--amplitude-max", type=float, default=0.60)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-random-state", type=int, default=20260708)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def metric_row(
    label: str,
    table: pd.DataFrame,
    count_column: str,
    rr_column: str,
    *,
    evaluation_note: str,
) -> dict[str, object]:
    return metric_dict(
        label,
        numeric(table, "truth_rr").to_numpy(dtype=float),
        numeric(table, rr_column).to_numpy(dtype=float),
        numeric(table, "truth_count").to_numpy(dtype=float),
        numeric(table, count_column).to_numpy(dtype=float),
        evaluation_note=evaluation_note,
    )


def metric_snapshot(
    table: pd.DataFrame,
    count_column: str,
    rr_column: str,
) -> dict[str, object]:
    return metric_row(
        "snapshot",
        table,
        count_column,
        rr_column,
        evaluation_note="paired_snapshot",
    )


def binomial_upper_tail(k: int, n: int) -> float:
    if n <= 0:
        return math.nan
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n))


def binomial_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return math.nan
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return float(min(1.0, 2.0 * min(lower, upper)))


def mcnemar_exact_pvalues(
    baseline_correct: np.ndarray,
    method_correct: np.ndarray,
) -> dict[str, float | int]:
    baseline_only = int(np.sum(baseline_correct & ~method_correct))
    method_only = int(np.sum(~baseline_correct & method_correct))
    discordant = baseline_only + method_only
    return {
        "mcnemar_baseline_only_correct": baseline_only,
        "mcnemar_method_only_correct": method_only,
        "mcnemar_discordant_pairs": discordant,
        "mcnemar_exact_p_two_sided": binomial_two_sided_pvalue(
            min(baseline_only, method_only),
            discordant,
        ),
        "mcnemar_exact_p_method_better": binomial_upper_tail(
            method_only,
            discordant,
        ),
    }


def bootstrap_delta_tests(
    table: pd.DataFrame,
    *,
    baseline_count_column: str,
    baseline_rr_column: str,
    method_count_column: str,
    method_rr_column: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    valid = (
        numeric(table, "truth_rr").notna()
        & numeric(table, "truth_count").notna()
        & numeric(table, baseline_count_column).notna()
        & numeric(table, baseline_rr_column).notna()
        & numeric(table, method_count_column).notna()
        & numeric(table, method_rr_column).notna()
    )
    df = table.loc[valid].reset_index(drop=True)
    n = len(df)
    base_metrics = metric_snapshot(df, baseline_count_column, baseline_rr_column)
    method_metrics = metric_snapshot(df, method_count_column, method_rr_column)
    point_deltas: dict[str, object] = {
        "videos": int(n),
        "baseline_rr_r2": base_metrics["rr_r2"],
        "method_rr_r2": method_metrics["rr_r2"],
        "delta_rr_r2": float(method_metrics["rr_r2"]) - float(base_metrics["rr_r2"]),
        "baseline_rr_mae": base_metrics["rr_mae"],
        "method_rr_mae": method_metrics["rr_mae"],
        "delta_rr_mae": float(method_metrics["rr_mae"]) - float(base_metrics["rr_mae"]),
        "baseline_rr_rmse": base_metrics["rr_rmse"],
        "method_rr_rmse": method_metrics["rr_rmse"],
        "delta_rr_rmse": float(method_metrics["rr_rmse"])
        - float(base_metrics["rr_rmse"]),
        "baseline_exact_count": base_metrics["exact_count"],
        "method_exact_count": method_metrics["exact_count"],
        "delta_exact_count": int(method_metrics["exact_count"])
        - int(base_metrics["exact_count"]),
        "baseline_abs_count_error_ge2": base_metrics["abs_count_error_ge2"],
        "method_abs_count_error_ge2": method_metrics["abs_count_error_ge2"],
        "delta_abs_count_error_ge2": int(method_metrics["abs_count_error_ge2"])
        - int(base_metrics["abs_count_error_ge2"]),
    }
    baseline_correct = np.isclose(
        numeric(df, baseline_count_column).to_numpy(dtype=float),
        numeric(df, "truth_count").to_numpy(dtype=float),
    )
    method_correct = np.isclose(
        numeric(df, method_count_column).to_numpy(dtype=float),
        numeric(df, "truth_count").to_numpy(dtype=float),
    )
    point_deltas.update(mcnemar_exact_pvalues(baseline_correct, method_correct))
    if n < 2 or samples <= 0:
        return point_deltas

    deltas: dict[str, list[float]] = {
        "delta_rr_r2": [],
        "delta_rr_mae": [],
        "delta_rr_rmse": [],
        "delta_exact_count": [],
        "delta_abs_count_error_ge2": [],
    }
    for _ in range(samples):
        sampled = df.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        base = metric_snapshot(sampled, baseline_count_column, baseline_rr_column)
        method = metric_snapshot(sampled, method_count_column, method_rr_column)
        deltas["delta_rr_r2"].append(float(method["rr_r2"]) - float(base["rr_r2"]))
        deltas["delta_rr_mae"].append(float(method["rr_mae"]) - float(base["rr_mae"]))
        deltas["delta_rr_rmse"].append(float(method["rr_rmse"]) - float(base["rr_rmse"]))
        deltas["delta_exact_count"].append(
            float(method["exact_count"]) - float(base["exact_count"])
        )
        deltas["delta_abs_count_error_ge2"].append(
            float(method["abs_count_error_ge2"])
            - float(base["abs_count_error_ge2"])
        )

    positive_improvement = {"delta_rr_r2", "delta_exact_count"}
    for key, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            point_deltas[f"{key}_ci_low"] = math.nan
            point_deltas[f"{key}_ci_high"] = math.nan
            point_deltas[f"{key}_bootstrap_prob_improvement"] = math.nan
            continue
        point_deltas[f"{key}_ci_low"] = float(np.percentile(arr, 2.5))
        point_deltas[f"{key}_ci_high"] = float(np.percentile(arr, 97.5))
        if key in positive_improvement:
            point_deltas[f"{key}_bootstrap_prob_improvement"] = float(np.mean(arr > 0))
        else:
            point_deltas[f"{key}_bootstrap_prob_improvement"] = float(np.mean(arr < 0))
    return point_deltas


def apply_rollback_guard(
    table: pd.DataFrame,
    *,
    safe_count_column: str,
    safe_rr_column: str,
    amplitude_max: float,
) -> pd.DataFrame:
    result = table.copy()
    peaks = numeric(result, "peaks")
    duration = numeric(result, "duration_seconds")
    corrected = numeric(result, "corrected_peaks")
    safe_count = numeric(result, safe_count_column)
    residual_applied = numeric(result, "applied_adjust")
    fft = numeric(result, "fft_count_estimate")
    spectral = numeric(result, "spectral_count_estimate")
    autocorr = numeric(result, "autocorr_count_estimate")
    amplitude = numeric(result, "selection_amplitude")

    rollback = (
        (residual_applied == -1)
        & (fft == peaks)
        & (spectral == peaks)
        & (autocorr == corrected)
        & (amplitude < float(amplitude_max))
    )
    final_count = safe_count.where(~rollback, peaks)
    final_rr = final_count / duration * 60.0
    result["consensus_rollback_guard"] = rollback
    result["consensus_rollback_reason"] = np.where(
        rollback,
        (
            "rollback_minus1_when_fft_and_spectral_support_original_count_"
            "and_autocorr_supports_corrected_count_with_low_amplitude"
        ),
        "no_rollback",
    )
    result["consensus_rollback_count"] = final_count
    result["consensus_rollback_rr"] = final_rr
    result["consensus_rollback_count_error"] = final_count - numeric(result, "truth_count")
    result["consensus_rollback_abs_count_error"] = result[
        "consensus_rollback_count_error"
    ].abs()
    result["consensus_rollback_rr_error"] = final_rr - numeric(result, "truth_rr")
    result["consensus_rollback_abs_rr_error"] = result[
        "consensus_rollback_rr_error"
    ].abs()
    result["consensus_rollback_safe_count_column"] = safe_count_column
    result["consensus_rollback_safe_rr_column"] = safe_rr_column
    result["consensus_rollback_amplitude_max"] = float(amplitude_max)
    return result


def evaluate_mode(
    input_root: Path,
    output_prefix: str,
    *,
    mode: str,
    amplitude_max: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "fixed":
        path = input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv"
        safe_count_column = "signal_aware_safe_final_peaks"
        safe_rr_column = "signal_aware_safe_final_rr_bpm"
        safe_label = "signal_aware_safe_fixed_oof"
    elif mode == "prefix_group":
        path = input_root / f"{output_prefix}_signal_aware_safe_policy_group_predictions.csv"
        safe_count_column = "signal_aware_safe_group_final_peaks"
        safe_rr_column = "signal_aware_safe_group_final_rr_bpm"
        safe_label = "signal_aware_safe_prefix_group"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    table = pd.read_csv(path)
    predictions = apply_rollback_guard(
        table,
        safe_count_column=safe_count_column,
        safe_rr_column=safe_rr_column,
        amplitude_max=amplitude_max,
    )
    metrics = pd.DataFrame(
        [
            metric_row(
                f"default_{mode}",
                predictions,
                "peaks",
                "rr_bpm",
                evaluation_note=f"{mode}_default_reference",
            ),
            metric_row(
                safe_label,
                predictions,
                safe_count_column,
                safe_rr_column,
                evaluation_note=f"{mode}_safe_gate_reference",
            ),
            metric_row(
                f"consensus_rollback_probe_{mode}",
                predictions,
                "consensus_rollback_count",
                "consensus_rollback_rr",
                evaluation_note=(
                    f"{mode}_narrow_consensus_rollback_probe_not_primary"
                ),
            ),
        ]
    )
    metrics["validation_mode"] = mode
    metrics["amplitude_max"] = float(amplitude_max)
    metrics["rollback_count"] = [
        0,
        0,
        int(predictions["consensus_rollback_guard"].sum()),
    ]
    cases = predictions[predictions["consensus_rollback_guard"]].copy()
    case_columns = [
        "video_id",
        "truth_count",
        "peaks",
        "corrected_peaks",
        safe_count_column,
        "fft_count_estimate",
        "autocorr_count_estimate",
        "spectral_count_estimate",
        "selection_amplitude",
        "consensus_rollback_count",
        "consensus_rollback_count_error",
        "consensus_rollback_reason",
    ]
    cases = cases[[column for column in case_columns if column in cases.columns]]
    cases["validation_mode"] = mode
    tests = {
        "validation_mode": mode,
        "comparison": f"{safe_label}_vs_consensus_rollback_probe_{mode}",
        "baseline_label": safe_label,
        "method_label": f"consensus_rollback_probe_{mode}",
        "bootstrap_samples": int(bootstrap_samples),
        "amplitude_max": float(amplitude_max),
        **bootstrap_delta_tests(
            predictions,
            baseline_count_column=safe_count_column,
            baseline_rr_column=safe_rr_column,
            method_count_column="consensus_rollback_count",
            method_rr_column="consensus_rollback_rr",
            samples=bootstrap_samples,
            rng=rng,
        ),
    }
    return predictions, metrics, cases, pd.DataFrame([tests])


def metric_text(row: pd.Series) -> str:
    return (
        f"RR R2={float(row['rr_r2']):.4f}, "
        f"MAE={float(row['rr_mae']):.3f}, "
        f"RMSE={float(row['rr_rmse']):.3f}, "
        f"exact={int(row['exact_count'])}/{int(row['count_valid_videos'])}, "
        f"within-one={int(row['within_one_count'])}/{int(row['count_valid_videos'])}"
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
            "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        )
    return "\n".join(lines)


def statistical_text(row: pd.Series) -> str:
    return (
        f"delta RR R2={float(row['delta_rr_r2']):.4f} "
        f"(95% bootstrap CI {float(row['delta_rr_r2_ci_low']):.4f} to "
        f"{float(row['delta_rr_r2_ci_high']):.4f}), "
        f"delta MAE={float(row['delta_rr_mae']):.3f} bpm "
        f"(95% CI {float(row['delta_rr_mae_ci_low']):.3f} to "
        f"{float(row['delta_rr_mae_ci_high']):.3f}), "
        f"delta exact={int(row['delta_exact_count'])}, "
        f"McNemar one-sided p={float(row['mcnemar_exact_p_method_better']):.3f}"
    )


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    cases: pd.DataFrame,
    statistical_tests: pd.DataFrame,
) -> None:
    fixed = metrics[metrics["validation_mode"].eq("fixed")]
    group = metrics[metrics["validation_mode"].eq("prefix_group")]
    fixed_probe = fixed[fixed["label"].str.contains("consensus_rollback_probe")].iloc[0]
    fixed_safe = fixed[fixed["label"].str.contains("signal_aware_safe")].iloc[0]
    group_probe = group[group["label"].str.contains("consensus_rollback_probe")].iloc[0]
    group_safe = group[group["label"].str.contains("signal_aware_safe")].iloc[0]
    fixed_stats = statistical_tests[
        statistical_tests["validation_mode"].astype(str).eq("fixed")
    ].iloc[0]
    group_stats = statistical_tests[
        statistical_tests["validation_mode"].astype(str).eq("prefix_group")
    ].iloc[0]
    text = f"""# Consensus Rollback Probe

This report evaluates a narrow non-truth rollback guard for one specific risk
pattern: the residual model applies a -1 adjustment, while FFT and spectral
count estimates support the original count, autocorrelation supports the
corrected count, and the selected thermal signal amplitude is below the fixed
threshold. The rule does not use reference RR or reference count at prediction
time, but it was discovered during internal error analysis and must be treated
as exploratory until frozen and externally validated.

In fixed out-of-fold validation, the safe gate was {metric_text(fixed_safe)}.
The consensus rollback probe was {metric_text(fixed_probe)}. In prefix-group
stress validation, the safe gate was {metric_text(group_safe)}, and the rollback
probe was {metric_text(group_probe)}.

Paired statistical evidence was weak because the rule only changed one video in
each validation view. Fixed out-of-fold comparison: {statistical_text(fixed_stats)}.
Prefix-group comparison: {statistical_text(group_stats)}. The point estimates
improved, but the McNemar exact test and bootstrap intervals do not support a
confirmatory claim from the current internal dataset alone.

The safe interpretation is that this guard identifies a small, auditable failure
mode of the residual correction. The unsafe interpretation is to claim a new
general method from a one-video internal rescue. It can be used as a candidate
external validation rule only if it is frozen before scoring.

## Rollback Cases

{markdown_table(cases)}

## Paired Statistical Tests

{markdown_table(statistical_tests)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_random_state)

    all_metrics: list[pd.DataFrame] = []
    all_cases: list[pd.DataFrame] = []
    all_tests: list[pd.DataFrame] = []
    for mode in ["fixed", "prefix_group"]:
        predictions, metrics, cases, tests = evaluate_mode(
            args.input_root,
            args.output_prefix,
            mode=mode,
            amplitude_max=args.amplitude_max,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
        )
        predictions.to_csv(
            output_dir / f"paper_consensus_rollback_probe_{mode}_predictions.csv",
            index=False,
        )
        all_metrics.append(metrics)
        all_cases.append(cases)
        all_tests.append(tests)

    metrics_table = pd.concat(all_metrics, ignore_index=True)
    cases_table = pd.concat(all_cases, ignore_index=True)
    statistical_tests = pd.concat(all_tests, ignore_index=True)
    metrics_path = output_dir / "paper_consensus_rollback_probe_metrics.csv"
    cases_path = output_dir / "paper_consensus_rollback_probe_cases.csv"
    tests_path = output_dir / "paper_consensus_rollback_probe_statistical_tests.csv"
    report_path = output_dir / "paper_consensus_rollback_probe_report.md"
    metrics_table.to_csv(metrics_path, index=False)
    cases_table.to_csv(cases_path, index=False)
    statistical_tests.to_csv(tests_path, index=False)
    write_report(report_path, metrics_table, cases_table, statistical_tests)

    print(f"Saved consensus rollback metrics: {metrics_path}")
    print(f"Saved consensus rollback cases: {cases_path}")
    print(f"Saved consensus rollback statistical tests: {tests_path}")
    print(f"Saved consensus rollback report: {report_path}")
    print(
        metrics_table[
            [
                "validation_mode",
                "label",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "within_one_count",
                "rollback_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
