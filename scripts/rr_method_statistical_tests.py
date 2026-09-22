from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    csv_name: str
    count_col: str
    rr_col: str
    validation_setting: str
    paper_use: str
    include_in_pairwise_tests: bool = True


METHODS = [
    MethodSpec(
        method_id="default",
        display_name="Default thermal RR pipeline",
        csv_name="{prefix}_summary.csv",
        count_col="peaks",
        rr_col="rr_bpm",
        validation_setting="existing_default_outputs",
        paper_use="main_or_supplement",
    ),
    MethodSpec(
        method_id="quality_residual_fixed_oof",
        display_name="Quality-aware residual correction (fixed threshold, out-of-fold)",
        csv_name="{prefix}_quality_residual_predictions.csv",
        count_col="corrected_peaks",
        rr_col="corrected_rr_bpm",
        validation_setting="fixed_threshold_out_of_fold",
        paper_use="main_or_supplement",
    ),
    MethodSpec(
        method_id="signal_consensus_fixed_oof",
        display_name="Signal-consensus supplement (fixed threshold, out-of-fold)",
        csv_name="{prefix}_signal_consensus_predictions.csv",
        count_col="signal_consensus_peaks",
        rr_col="signal_consensus_rr_bpm",
        validation_setting="quality_residual_plus_fft_autocorr_spectral_supplement",
        paper_use="candidate_internal_extension",
    ),
    MethodSpec(
        method_id="signal_aware_fixed_oof",
        display_name="Signal-aware residual correction (fixed threshold, out-of-fold)",
        csv_name="{prefix}_signal_aware_residual_predictions.csv",
        count_col="signal_aware_final_peaks",
        rr_col="signal_aware_final_rr_bpm",
        validation_setting="stratified_out_of_fold_signal_aware_residual_correction",
        paper_use="candidate_internal_precision",
    ),
    MethodSpec(
        method_id="signal_aware_prefix_group_fixed",
        display_name="Signal-aware residual correction (leave-one-prefix-group-out, fixed threshold)",
        csv_name="{prefix}_signal_aware_residual_group_predictions.csv",
        count_col="signal_aware_group_final_peaks",
        rr_col="signal_aware_group_final_rr_bpm",
        validation_setting="prefix_group_out_of_fold_signal_aware_residual_correction",
        paper_use="candidate_group_validation_failed",
    ),
    MethodSpec(
        method_id="signal_aware_safe_fixed_oof",
        display_name="Conservative signal-aware safe gate (fixed threshold, out-of-fold)",
        csv_name="{prefix}_signal_aware_safe_policy_predictions.csv",
        count_col="signal_aware_safe_final_peaks",
        rr_col="signal_aware_safe_final_rr_bpm",
        validation_setting="safe_gate_stratified_oof_negative_adjust_only_autocorr_delta_le_2.0",
        paper_use="candidate_internal_precision_safety_gated",
    ),
    MethodSpec(
        method_id="signal_aware_safe_prefix_group_fixed",
        display_name="Conservative signal-aware safe gate (leave-one-prefix-group-out, fixed threshold)",
        csv_name="{prefix}_signal_aware_safe_policy_group_predictions.csv",
        count_col="signal_aware_safe_group_final_peaks",
        rr_col="signal_aware_safe_group_final_rr_bpm",
        validation_setting="safe_gate_prefix_group_oof_negative_adjust_only_autocorr_delta_le_2.0",
        paper_use="candidate_prefix_group_stress_passed_internal_only",
    ),
    MethodSpec(
        method_id="truth_calibrated_upper_bound",
        display_name="Truth-calibrated upper bound",
        csv_name="{prefix}_truth_calibrated_summary.csv",
        count_col="peaks",
        rr_col="rr_bpm",
        validation_setting="truth_assisted_upper_bound_not_main_method",
        paper_use="upper_bound_only",
        include_in_pairwise_tests=False,
    ),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build paired statistical tests for RR method comparisons. The script "
            "uses per-video predictions and reports paired bootstrap CIs plus "
            "non-parametric paired tests for manuscript reporting."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--baseline-method", default="default")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=20260706)
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def regression_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(pred)
    if int(valid.sum()) < 2:
        return math.nan
    truth_values = truth[valid].astype(float)
    pred_values = pred[valid].astype(float)
    total_sum_squares = float(np.sum((truth_values - np.mean(truth_values)) ** 2))
    if math.isclose(total_sum_squares, 0.0):
        return math.nan
    residual_sum_squares = float(np.sum((truth_values - pred_values) ** 2))
    return float(1.0 - residual_sum_squares / total_sum_squares)


def metric_snapshot(
    truth_rr: np.ndarray,
    pred_rr: np.ndarray,
    truth_count: np.ndarray,
    pred_count: np.ndarray,
) -> dict[str, float | int]:
    rr_valid = np.isfinite(truth_rr) & np.isfinite(pred_rr)
    count_valid = np.isfinite(truth_count) & np.isfinite(pred_count)
    rr_error = pred_rr[rr_valid] - truth_rr[rr_valid]
    count_error = pred_count[count_valid] - truth_count[count_valid]
    return {
        "videos": int(len(truth_rr)),
        "rr_valid_videos": int(rr_valid.sum()),
        "rr_r2": regression_r2(truth_rr, pred_rr),
        "rr_mae": float(np.mean(np.abs(rr_error))) if int(rr_valid.sum()) else math.nan,
        "rr_rmse": float(np.sqrt(np.mean(rr_error**2))) if int(rr_valid.sum()) else math.nan,
        "count_valid_videos": int(count_valid.sum()),
        "count_mae": float(np.mean(np.abs(count_error))) if int(count_valid.sum()) else math.nan,
        "exact_count": int(np.sum(np.abs(count_error) == 0)) if int(count_valid.sum()) else 0,
        "within_one_count": int(np.sum(np.abs(count_error) <= 1)) if int(count_valid.sum()) else 0,
        "abs_count_error_ge2": int(np.sum(np.abs(count_error) >= 2)) if int(count_valid.sum()) else 0,
    }


def read_method_predictions(input_root: Path, prefix: str, spec: MethodSpec) -> pd.DataFrame | None:
    path = input_root / spec.csv_name.format(prefix=prefix)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"video_id", "truth_count", "truth_rr", spec.count_col, spec.rr_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "video_id": df["video_id"].astype(str),
            "truth_count": numeric(df, "truth_count"),
            "truth_rr": numeric(df, "truth_rr"),
            f"{spec.method_id}_count": numeric(df, spec.count_col),
            f"{spec.method_id}_rr": numeric(df, spec.rr_col),
        }
    )
    return out


def build_paired_table(input_root: Path, prefix: str) -> tuple[pd.DataFrame, list[MethodSpec]]:
    merged: pd.DataFrame | None = None
    available: list[MethodSpec] = []
    for spec in METHODS:
        method_df = read_method_predictions(input_root, prefix, spec)
        if method_df is None:
            continue
        available.append(spec)
        if merged is None:
            merged = method_df
            continue
        method_only = method_df.drop(columns=["truth_count", "truth_rr"])
        merged = merged.merge(method_only, on="video_id", how="outer")
    if merged is None:
        raise FileNotFoundError(f"No method prediction files found under {input_root}")
    for spec in available:
        count_col = f"{spec.method_id}_count"
        rr_col = f"{spec.method_id}_rr"
        merged[f"{spec.method_id}_count_error"] = merged[count_col] - merged["truth_count"]
        merged[f"{spec.method_id}_abs_count_error"] = merged[
            f"{spec.method_id}_count_error"
        ].abs()
        merged[f"{spec.method_id}_rr_error"] = merged[rr_col] - merged["truth_rr"]
        merged[f"{spec.method_id}_abs_rr_error"] = merged[f"{spec.method_id}_rr_error"].abs()
        merged[f"{spec.method_id}_exact"] = (
            merged[f"{spec.method_id}_abs_count_error"] == 0
        )
    return merged.sort_values("video_id").reset_index(drop=True), available


def bootstrap_delta_ci(
    paired: pd.DataFrame,
    baseline_id: str,
    method_id: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    valid = (
        paired["truth_rr"].notna()
        & paired["truth_count"].notna()
        & paired[f"{baseline_id}_rr"].notna()
        & paired[f"{method_id}_rr"].notna()
        & paired[f"{baseline_id}_count"].notna()
        & paired[f"{method_id}_count"].notna()
    )
    df = paired.loc[valid].reset_index(drop=True)
    n = len(df)
    if n < 2:
        return {}
    truth_rr = df["truth_rr"].to_numpy(dtype=float)
    truth_count = df["truth_count"].to_numpy(dtype=float)
    base_rr = df[f"{baseline_id}_rr"].to_numpy(dtype=float)
    base_count = df[f"{baseline_id}_count"].to_numpy(dtype=float)
    method_rr = df[f"{method_id}_rr"].to_numpy(dtype=float)
    method_count = df[f"{method_id}_count"].to_numpy(dtype=float)
    deltas = {
        "delta_rr_r2": [],
        "delta_rr_mae": [],
        "delta_rr_rmse": [],
        "delta_count_mae": [],
        "delta_exact_count": [],
        "delta_within_one_count": [],
    }
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        base_metrics = metric_snapshot(
            truth_rr[idx], base_rr[idx], truth_count[idx], base_count[idx]
        )
        method_metrics = metric_snapshot(
            truth_rr[idx], method_rr[idx], truth_count[idx], method_count[idx]
        )
        for key in deltas:
            metric_name = key.removeprefix("delta_")
            deltas[key].append(float(method_metrics[metric_name]) - float(base_metrics[metric_name]))
    out: dict[str, float] = {}
    for key, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            out[f"{key}_ci_low"] = math.nan
            out[f"{key}_ci_high"] = math.nan
        else:
            out[f"{key}_ci_low"] = float(np.percentile(arr, 2.5))
            out[f"{key}_ci_high"] = float(np.percentile(arr, 97.5))
    return out


def sign_flip_pvalue(
    values: np.ndarray,
    observed: float,
    samples: int,
    rng: np.random.Generator,
    direction: str,
) -> float:
    clean = values[np.isfinite(values)]
    clean = clean[~np.isclose(clean, 0.0)]
    if len(clean) == 0 or not np.isfinite(observed):
        return math.nan
    draws = []
    batch_size = 2000
    remaining = samples
    while remaining > 0:
        batch = min(batch_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(clean)))
        draws.append(np.mean(signs * clean, axis=1))
        remaining -= batch
    null_stats = np.concatenate(draws)
    if direction == "less":
        return float((np.sum(null_stats <= observed) + 1) / (len(null_stats) + 1))
    if direction == "greater":
        return float((np.sum(null_stats >= observed) + 1) / (len(null_stats) + 1))
    return float((np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (len(null_stats) + 1))


def safe_wilcoxon(values: np.ndarray, alternative: str) -> float:
    clean = values[np.isfinite(values)]
    clean = clean[~np.isclose(clean, 0.0)]
    if len(clean) == 0:
        return math.nan
    try:
        return float(stats.wilcoxon(clean, alternative=alternative).pvalue)
    except ValueError:
        return math.nan


def mcnemar_exact_pvalues(baseline_correct: np.ndarray, method_correct: np.ndarray) -> dict[str, float | int]:
    b = int(np.sum(baseline_correct & ~method_correct))
    c = int(np.sum(~baseline_correct & method_correct))
    discordant = b + c
    if discordant == 0:
        two_sided = math.nan
        one_sided_method_better = math.nan
    else:
        two_sided = float(stats.binomtest(min(b, c), n=discordant, p=0.5).pvalue)
        one_sided_method_better = float(
            stats.binomtest(c, n=discordant, p=0.5, alternative="greater").pvalue
        )
    return {
        "mcnemar_baseline_only_correct": b,
        "mcnemar_method_only_correct": c,
        "mcnemar_discordant_pairs": discordant,
        "mcnemar_exact_p_two_sided": two_sided,
        "mcnemar_exact_p_method_better": one_sided_method_better,
    }


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if int(valid.sum()) == 0:
        return pd.Series(adjusted, index=p_values.index)
    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return pd.Series(np.clip(adjusted, 0, 1), index=p_values.index)


def build_method_metrics(paired: pd.DataFrame, methods: list[MethodSpec]) -> pd.DataFrame:
    rows = []
    for spec in methods:
        valid = (
            paired["truth_rr"].notna()
            & paired["truth_count"].notna()
            & paired[f"{spec.method_id}_rr"].notna()
            & paired[f"{spec.method_id}_count"].notna()
        )
        df = paired.loc[valid]
        metrics = metric_snapshot(
            df["truth_rr"].to_numpy(dtype=float),
            df[f"{spec.method_id}_rr"].to_numpy(dtype=float),
            df["truth_count"].to_numpy(dtype=float),
            df[f"{spec.method_id}_count"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method_id": spec.method_id,
                "method": spec.display_name,
                "validation_setting": spec.validation_setting,
                **metrics,
                "exact_fraction": (
                    float(metrics["exact_count"]) / float(metrics["count_valid_videos"])
                    if metrics["count_valid_videos"]
                    else math.nan
                ),
                "paper_use": spec.paper_use,
            }
        )
    return pd.DataFrame(rows)


def compare_methods(
    paired: pd.DataFrame,
    methods: list[MethodSpec],
    baseline_id: str,
    bootstrap_samples: int,
    permutation_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    method_by_id = {spec.method_id: spec for spec in methods}
    if baseline_id not in method_by_id:
        raise ValueError(f"Baseline method not found: {baseline_id}")
    comparisons: list[tuple[str, str, str]] = [
        (baseline_id, spec.method_id, "vs_default")
        for spec in methods
        if spec.method_id != baseline_id and spec.include_in_pairwise_tests
    ]
    if "signal_consensus_fixed_oof" in method_by_id and "signal_aware_fixed_oof" in method_by_id:
        comparisons.append(
            ("signal_consensus_fixed_oof", "signal_aware_fixed_oof", "vs_signal_consensus")
        )
    if "signal_consensus_fixed_oof" in method_by_id and "signal_aware_safe_fixed_oof" in method_by_id:
        comparisons.append(
            (
                "signal_consensus_fixed_oof",
                "signal_aware_safe_fixed_oof",
                "vs_signal_consensus",
            )
        )
    if "signal_aware_fixed_oof" in method_by_id and "signal_aware_safe_fixed_oof" in method_by_id:
        comparisons.append(
            (
                "signal_aware_fixed_oof",
                "signal_aware_safe_fixed_oof",
                "vs_ungated_signal_aware",
            )
        )
    if (
        "signal_aware_prefix_group_fixed" in method_by_id
        and "signal_aware_safe_prefix_group_fixed" in method_by_id
    ):
        comparisons.append(
            (
                "signal_aware_prefix_group_fixed",
                "signal_aware_safe_prefix_group_fixed",
                "vs_ungated_prefix_group_signal_aware",
            )
        )
    rows = []
    for base_id, method_id, family in comparisons:
        valid = (
            paired["truth_rr"].notna()
            & paired["truth_count"].notna()
            & paired[f"{base_id}_rr"].notna()
            & paired[f"{method_id}_rr"].notna()
            & paired[f"{base_id}_count"].notna()
            & paired[f"{method_id}_count"].notna()
        )
        df = paired.loc[valid].reset_index(drop=True)
        if len(df) < 2:
            continue
        truth_rr = df["truth_rr"].to_numpy(dtype=float)
        truth_count = df["truth_count"].to_numpy(dtype=float)
        base_rr = df[f"{base_id}_rr"].to_numpy(dtype=float)
        method_rr = df[f"{method_id}_rr"].to_numpy(dtype=float)
        base_count = df[f"{base_id}_count"].to_numpy(dtype=float)
        method_count = df[f"{method_id}_count"].to_numpy(dtype=float)
        base_metrics = metric_snapshot(truth_rr, base_rr, truth_count, base_count)
        method_metrics = metric_snapshot(truth_rr, method_rr, truth_count, method_count)
        base_rr_error = base_rr - truth_rr
        method_rr_error = method_rr - truth_rr
        abs_diff = np.abs(method_rr_error) - np.abs(base_rr_error)
        squared_error_improvement = base_rr_error**2 - method_rr_error**2
        rr_total_sum_squares = float(np.sum((truth_rr - np.mean(truth_rr)) ** 2))
        observed_delta_r2 = (
            float(np.sum(squared_error_improvement) / rr_total_sum_squares)
            if not math.isclose(rr_total_sum_squares, 0.0)
            else math.nan
        )
        observed_mae_diff = float(np.mean(abs_diff))
        improved = int(np.sum(abs_diff < 0))
        worsened = int(np.sum(abs_diff > 0))
        tied = int(np.sum(np.isclose(abs_diff, 0.0)))
        baseline_correct = np.isclose(base_count, truth_count)
        method_correct = np.isclose(method_count, truth_count)
        row = {
            "comparison_family": family,
            "baseline_method_id": base_id,
            "baseline_method": method_by_id[base_id].display_name,
            "method_id": method_id,
            "method": method_by_id[method_id].display_name,
            "videos": int(len(df)),
            "baseline_rr_r2": base_metrics["rr_r2"],
            "method_rr_r2": method_metrics["rr_r2"],
            "delta_rr_r2": float(method_metrics["rr_r2"]) - float(base_metrics["rr_r2"]),
            "baseline_rr_mae": base_metrics["rr_mae"],
            "method_rr_mae": method_metrics["rr_mae"],
            "delta_rr_mae": float(method_metrics["rr_mae"]) - float(base_metrics["rr_mae"]),
            "baseline_rr_rmse": base_metrics["rr_rmse"],
            "method_rr_rmse": method_metrics["rr_rmse"],
            "delta_rr_rmse": float(method_metrics["rr_rmse"]) - float(base_metrics["rr_rmse"]),
            "baseline_count_mae": base_metrics["count_mae"],
            "method_count_mae": method_metrics["count_mae"],
            "delta_count_mae": float(method_metrics["count_mae"]) - float(base_metrics["count_mae"]),
            "baseline_exact_count": base_metrics["exact_count"],
            "method_exact_count": method_metrics["exact_count"],
            "delta_exact_count": int(method_metrics["exact_count"]) - int(base_metrics["exact_count"]),
            "baseline_within_one_count": base_metrics["within_one_count"],
            "method_within_one_count": method_metrics["within_one_count"],
            "delta_within_one_count": int(method_metrics["within_one_count"])
            - int(base_metrics["within_one_count"]),
            "abs_rr_error_improved_videos": improved,
            "abs_rr_error_worsened_videos": worsened,
            "abs_rr_error_tied_videos": tied,
            "abs_rr_error_improved_fraction": improved / len(df),
            "paired_standardized_mae_effect": (
                float(np.mean(abs_diff) / np.std(abs_diff, ddof=1))
                if len(abs_diff) > 1 and not math.isclose(float(np.std(abs_diff, ddof=1)), 0.0)
                else math.nan
            ),
            "rr_r2_permutation_p_two_sided": sign_flip_pvalue(
                squared_error_improvement,
                float(np.mean(squared_error_improvement)),
                permutation_samples,
                rng,
                direction="two-sided",
            ),
            "rr_r2_permutation_p_method_better": sign_flip_pvalue(
                squared_error_improvement,
                float(np.mean(squared_error_improvement)),
                permutation_samples,
                rng,
                direction="greater",
            ),
            "mae_wilcoxon_p_two_sided": safe_wilcoxon(abs_diff, alternative="two-sided"),
            "mae_wilcoxon_p_method_better": safe_wilcoxon(abs_diff, alternative="less"),
            "mae_permutation_p_two_sided": sign_flip_pvalue(
                abs_diff, observed_mae_diff, permutation_samples, rng, direction="two-sided"
            ),
            "mae_permutation_p_method_better": sign_flip_pvalue(
                abs_diff, observed_mae_diff, permutation_samples, rng, direction="less"
            ),
            **mcnemar_exact_pvalues(baseline_correct, method_correct),
            "paper_use": method_by_id[method_id].paper_use,
        }
        row.update(
            bootstrap_delta_ci(
                paired=df,
                baseline_id=base_id,
                method_id=method_id,
                samples=bootstrap_samples,
                rng=rng,
            )
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    for column in [
        "rr_r2_permutation_p_two_sided",
        "rr_r2_permutation_p_method_better",
        "mae_wilcoxon_p_two_sided",
        "mae_wilcoxon_p_method_better",
        "mae_permutation_p_two_sided",
        "mae_permutation_p_method_better",
        "mcnemar_exact_p_two_sided",
        "mcnemar_exact_p_method_better",
    ]:
        out[f"{column}_bh"] = bh_adjust(out[column])
    out["reporting_interpretation"] = out.apply(reporting_interpretation, axis=1)
    return out


def ci_excludes_zero(row: pd.Series, metric: str, improvement_direction: str) -> bool:
    low = row.get(f"delta_{metric}_ci_low", math.nan)
    high = row.get(f"delta_{metric}_ci_high", math.nan)
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    if improvement_direction == "positive":
        return low > 0
    return high < 0


def reporting_interpretation(row: pd.Series) -> str:
    if row["paper_use"] == "candidate_group_validation_failed":
        return (
            "Do not use as a primary generalization claim; prefix-group validation "
            "does not support the signal-aware method."
        )
    mae_support = (
        row.get("mae_permutation_p_method_better", math.nan) < 0.05
        or row.get("mae_wilcoxon_p_method_better", math.nan) < 0.05
        or ci_excludes_zero(row, "rr_mae", "negative")
    )
    exact_support = (
        row.get("mcnemar_exact_p_method_better", math.nan) < 0.05
        or ci_excludes_zero(row, "exact_count", "positive")
    )
    r2_support = ci_excludes_zero(row, "rr_r2", "positive")
    if mae_support and exact_support and r2_support:
        return "Strong internal paired support; still requires metadata/external validation for Q2+ generalization."
    if mae_support or exact_support:
        return "Report as internal paired improvement with caution; R2 or generalization evidence is not fully conclusive."
    return "Exploratory only; paired evidence is not strong enough for a primary claim."


def write_summary(
    path: Path,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    bootstrap_samples: int,
    permutation_samples: int,
) -> None:
    lines = [
        "# RR Method Paired Statistical Tests",
        "",
        f"- Bootstrap samples: {bootstrap_samples}",
        f"- Sign-flip permutation samples: {permutation_samples}",
        "- Primary interpretation uses paired evidence across the same videos; truth-calibrated upper bound is excluded from pairwise tests.",
        "",
        "## Method Metrics",
        "",
        "| method | rr_r2 | rr_mae | rr_rmse | exact_count | paper_use |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in metrics.iterrows():
        exact = f"{int(row['exact_count'])}/{int(row['count_valid_videos'])}"
        lines.append(
            "| "
            f"{row['method']} | {row['rr_r2']:.4f} | {row['rr_mae']:.3f} | "
            f"{row['rr_rmse']:.3f} | {exact} | {row['paper_use']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| comparison | delta_rr_r2 | delta_rr_mae | delta_exact | MAE one-sided p | McNemar one-sided p | interpretation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in tests.iterrows():
        lines.append(
            "| "
            f"{row['method']} vs {row['baseline_method']} | "
            f"{row['delta_rr_r2']:.4f} | {row['delta_rr_mae']:.3f} | "
            f"{int(row['delta_exact_count'])} | "
            f"{row['mae_permutation_p_method_better']:.4f} | "
            f"{row['mcnemar_exact_p_method_better']:.4f} | "
            f"{row['reporting_interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Note",
            "",
            "Use the default, quality-aware, and fixed out-of-fold signal-aware results as internal-validation evidence only. "
            "Do not promote the truth-calibrated upper bound or the failed prefix-group signal-aware result to the main deployable claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    rng = np.random.default_rng(args.random_state)
    paired, methods = build_paired_table(input_root, args.output_prefix)
    metrics = build_method_metrics(paired, methods)
    tests = compare_methods(
        paired,
        methods,
        baseline_id=args.baseline_method,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        rng=rng,
    )
    paired_path = input_root / f"{args.output_prefix}_rr_method_paired_predictions.csv"
    metrics_path = input_root / f"{args.output_prefix}_rr_method_statistics.csv"
    tests_path = input_root / f"{args.output_prefix}_rr_method_statistical_tests.csv"
    summary_path = input_root / f"{args.output_prefix}_rr_method_statistical_tests_summary.md"
    paired.to_csv(paired_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    tests.to_csv(tests_path, index=False)
    write_summary(summary_path, metrics, tests, args.bootstrap_samples, args.permutation_samples)
    print(f"Saved paired predictions: {paired_path}")
    print(f"Saved method statistics: {metrics_path}")
    print(f"Saved paired statistical tests: {tests_path}")
    print(f"Saved paired statistical summary: {summary_path}")
    if not tests.empty:
        print(
            tests[
                [
                    "method_id",
                    "delta_rr_r2",
                    "delta_rr_mae",
                    "delta_exact_count",
                    "mae_permutation_p_method_better",
                    "mcnemar_exact_p_method_better",
                    "paper_use",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
