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
            "Evaluate a conservative, non-truth signal-aware residual gate. "
            "The policy starts from the signal-consensus RR estimate and only "
            "accepts an existing out-of-fold signal-aware -1 correction when "
            "the autocorrelation count estimate is close to the consensus count."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--source-name", default="signal_aware_residual")
    parser.add_argument("--output-name", default="signal_aware_safe_policy")
    parser.add_argument("--max-autocorr-delta", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-random-state", type=int, default=20260707)
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def rr_duration(predictions: pd.DataFrame) -> np.ndarray:
    duration = numeric(predictions, "rr_duration_seconds")
    if duration.notna().any():
        return duration.to_numpy(dtype=float)
    return numeric(predictions, "duration_seconds").to_numpy(dtype=float)


def rr_from_count(predictions: pd.DataFrame, count: np.ndarray) -> np.ndarray:
    duration = rr_duration(predictions)
    return np.divide(
        count * 60.0,
        duration,
        out=np.full(len(count), np.nan, dtype=float),
        where=np.isfinite(duration) & (duration > 0),
    )


def add_safe_policy_columns(
    predictions: pd.DataFrame,
    *,
    source_prefix: str,
    output_prefix: str,
    max_autocorr_delta: float,
) -> pd.DataFrame:
    required = {
        "signal_consensus_peaks",
        "autocorr_count_estimate",
        f"{source_prefix}_applied_adjust",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns for {source_prefix}: {missing}")

    out = predictions.copy()
    source_adjust = numeric(out, f"{source_prefix}_applied_adjust").fillna(0).astype(int)
    consensus_count = numeric(out, "signal_consensus_peaks").to_numpy(dtype=float)
    autocorr_count = numeric(out, "autocorr_count_estimate").to_numpy(dtype=float)
    autocorr_delta = np.abs(autocorr_count - consensus_count)
    safe_mask = (source_adjust.to_numpy(dtype=int) == -1) & (
        np.isfinite(autocorr_delta) & (autocorr_delta <= float(max_autocorr_delta))
    )
    safe_adjust = np.where(safe_mask, -1, 0)
    final_count = np.clip(consensus_count + safe_adjust, 0, None)
    final_rr = rr_from_count(out, final_count)

    source_applied = source_adjust.to_numpy(dtype=int) != 0
    guard_reason = np.where(
        safe_mask,
        "accepted_negative_adjust_autocorr_consistent",
        np.where(
            ~source_applied,
            "no_source_adjustment",
            np.where(
                source_adjust.to_numpy(dtype=int) != -1,
                "blocked_positive_adjustment",
                "blocked_autocorr_discordance",
            ),
        ),
    )

    truth_count = numeric(out, "truth_count").to_numpy(dtype=float)
    truth_rr = numeric(out, "truth_rr").to_numpy(dtype=float)
    count_error = final_count - truth_count
    rr_error = final_rr - truth_rr

    out[f"{output_prefix}_autocorr_delta"] = autocorr_delta
    if "fft_count_estimate" in out.columns:
        out[f"{output_prefix}_fft_delta"] = np.abs(
            numeric(out, "fft_count_estimate").to_numpy(dtype=float) - consensus_count
        )
    out[f"{output_prefix}_source_applied_adjust"] = source_adjust
    out[f"{output_prefix}_applied_adjust"] = safe_adjust
    out[f"{output_prefix}_guard_reason"] = guard_reason
    out[f"{output_prefix}_final_peaks"] = final_count
    out[f"{output_prefix}_final_rr_bpm"] = final_rr
    out[f"{output_prefix}_count_error"] = count_error
    out[f"{output_prefix}_abs_count_error"] = np.abs(count_error)
    out[f"{output_prefix}_rr_error"] = rr_error
    out[f"{output_prefix}_abs_rr_error"] = np.abs(rr_error)
    out[f"{output_prefix}_evaluation_note"] = (
        "safe_gate_existing_oof_predictions_negative_adjust_only_"
        f"autocorr_delta_le_{float(max_autocorr_delta):.1f}"
    )
    return out


def metric_row(
    predictions: pd.DataFrame,
    *,
    label: str,
    pred_count_column: str,
    pred_rr_column: str | None,
    evaluation_note: str,
) -> dict[str, object]:
    pred_count = numeric(predictions, pred_count_column).to_numpy(dtype=float)
    pred_rr = (
        numeric(predictions, pred_rr_column).to_numpy(dtype=float)
        if pred_rr_column is not None
        else rr_from_count(predictions, pred_count)
    )
    return metric_dict(
        label,
        numeric(predictions, "truth_rr").to_numpy(dtype=float),
        pred_rr,
        numeric(predictions, "truth_count").to_numpy(dtype=float),
        pred_count,
        evaluation_note=evaluation_note,
    )


def build_metrics(
    fixed: pd.DataFrame,
    group: pd.DataFrame,
    *,
    max_autocorr_delta: float,
) -> pd.DataFrame:
    rows = [
        metric_row(
            fixed,
            label="signal_consensus_baseline",
            pred_count_column="signal_consensus_peaks",
            pred_rr_column="signal_consensus_rr_bpm",
            evaluation_note="existing_signal_consensus_predictions",
        ),
        metric_row(
            fixed,
            label="signal_aware_residual_fixed_oof",
            pred_count_column="signal_aware_final_peaks",
            pred_rr_column="signal_aware_final_rr_bpm",
            evaluation_note="existing_signal_aware_residual_fixed_oof",
        ),
        metric_row(
            fixed,
            label="signal_aware_safe_policy_fixed_oof",
            pred_count_column="signal_aware_safe_final_peaks",
            pred_rr_column="signal_aware_safe_final_rr_bpm",
            evaluation_note=(
                "safe_gate_stratified_oof_negative_adjust_only_"
                f"autocorr_delta_le_{float(max_autocorr_delta):.1f}"
            ),
        ),
        metric_row(
            group,
            label="signal_aware_residual_prefix_group_fixed_threshold",
            pred_count_column="signal_aware_group_final_peaks",
            pred_rr_column="signal_aware_group_final_rr_bpm",
            evaluation_note="existing_prefix_group_signal_aware_residual_fixed_threshold",
        ),
        metric_row(
            group,
            label="signal_aware_safe_policy_prefix_group_fixed_threshold",
            pred_count_column="signal_aware_safe_group_final_peaks",
            pred_rr_column="signal_aware_safe_group_final_rr_bpm",
            evaluation_note=(
                "safe_gate_prefix_group_oof_negative_adjust_only_"
                f"autocorr_delta_le_{float(max_autocorr_delta):.1f}"
            ),
        ),
    ]
    metrics = pd.DataFrame(rows)
    applied = {
        "signal_aware_residual_fixed_oof": int(
            (numeric(fixed, "signal_aware_applied_adjust") != 0).sum()
        ),
        "signal_aware_safe_policy_fixed_oof": int(
            (numeric(fixed, "signal_aware_safe_applied_adjust") != 0).sum()
        ),
        "signal_aware_residual_prefix_group_fixed_threshold": int(
            (numeric(group, "signal_aware_group_applied_adjust") != 0).sum()
        ),
        "signal_aware_safe_policy_prefix_group_fixed_threshold": int(
            (numeric(group, "signal_aware_safe_group_applied_adjust") != 0).sum()
        ),
    }
    metrics["max_autocorr_delta"] = float(max_autocorr_delta)
    metrics["applied_corrections"] = metrics["label"].map(applied)
    metrics["safe_policy_rule"] = (
        "accept source applied_adjust == -1 only when "
        f"abs(autocorr_count_estimate - signal_consensus_peaks) <= {float(max_autocorr_delta):.1f}"
    )
    return metrics


def paired_values(
    predictions: pd.DataFrame,
    *,
    baseline_count: np.ndarray,
    candidate_count: np.ndarray,
    index: np.ndarray,
) -> dict[str, float]:
    truth_rr = numeric(predictions, "truth_rr").to_numpy(dtype=float)
    truth_count = numeric(predictions, "truth_count").to_numpy(dtype=float)
    baseline_rr = rr_from_count(predictions, baseline_count)
    candidate_rr = rr_from_count(predictions, candidate_count)
    baseline = metric_dict(
        "baseline",
        truth_rr[index],
        baseline_rr[index],
        truth_count[index],
        baseline_count[index],
        evaluation_note="bootstrap",
    )
    candidate = metric_dict(
        "candidate",
        truth_rr[index],
        candidate_rr[index],
        truth_count[index],
        candidate_count[index],
        evaluation_note="bootstrap",
    )
    return {
        "delta_rr_r2": float(candidate["rr_r2"]) - float(baseline["rr_r2"]),
        "delta_rr_mae": float(candidate["rr_mae"]) - float(baseline["rr_mae"]),
        "delta_rr_rmse": float(candidate["rr_rmse"]) - float(baseline["rr_rmse"]),
        "delta_exact_count": float(candidate["exact_count"]) - float(baseline["exact_count"]),
        "delta_abs_count_error_ge2": float(candidate["abs_count_error_ge2"])
        - float(baseline["abs_count_error_ge2"]),
    }


def format_ci(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.4f} ({low:.4f}, {high:.4f})"


def bootstrap_ci(
    fixed: pd.DataFrame,
    group: pd.DataFrame,
    *,
    samples: int,
    random_state: int,
) -> pd.DataFrame:
    comparisons = [
        (
            "fixed",
            "safe_policy_vs_signal_consensus",
            fixed,
            numeric(fixed, "signal_consensus_peaks").to_numpy(dtype=float),
            numeric(fixed, "signal_aware_safe_final_peaks").to_numpy(dtype=float),
        ),
        (
            "fixed",
            "safe_policy_vs_original_signal_aware",
            fixed,
            numeric(fixed, "signal_aware_final_peaks").to_numpy(dtype=float),
            numeric(fixed, "signal_aware_safe_final_peaks").to_numpy(dtype=float),
        ),
        (
            "prefix_group",
            "safe_policy_vs_signal_consensus",
            group,
            numeric(group, "signal_consensus_peaks").to_numpy(dtype=float),
            numeric(group, "signal_aware_safe_group_final_peaks").to_numpy(dtype=float),
        ),
        (
            "prefix_group",
            "safe_policy_vs_original_signal_aware",
            group,
            numeric(group, "signal_aware_group_final_peaks").to_numpy(dtype=float),
            numeric(group, "signal_aware_safe_group_final_peaks").to_numpy(dtype=float),
        ),
    ]
    display_names = {
        "delta_rr_r2": "Delta RR R2",
        "delta_rr_mae": "Delta MAE (bpm)",
        "delta_rr_rmse": "Delta RMSE (bpm)",
        "delta_exact_count": "Delta exact count",
        "delta_abs_count_error_ge2": "Delta count errors >=2",
    }
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, object]] = []
    for mode, comparison, predictions, baseline_count, candidate_count in comparisons:
        full_index = np.arange(len(predictions))
        estimates = paired_values(
            predictions,
            baseline_count=baseline_count,
            candidate_count=candidate_count,
            index=full_index,
        )
        values_by_metric = {key: [] for key in estimates}
        for _ in range(int(samples)):
            index = rng.integers(0, len(predictions), size=len(predictions))
            sampled = paired_values(
                predictions,
                baseline_count=baseline_count,
                candidate_count=candidate_count,
                index=index,
            )
            for key, value in sampled.items():
                if math.isfinite(float(value)):
                    values_by_metric[key].append(float(value))
        for key, estimate in estimates.items():
            values = np.asarray(values_by_metric[key], dtype=float)
            low = float(np.quantile(values, 0.025))
            high = float(np.quantile(values, 0.975))
            rows.append(
                {
                    "validation_mode": mode,
                    "comparison": comparison,
                    "metric": key,
                    "display_metric": display_names[key],
                    "estimate": float(estimate),
                    "ci_low_2_5": low,
                    "ci_high_97_5": high,
                    "estimate_with_ci": format_ci(float(estimate), low, high),
                    "bootstrap_samples": int(samples),
                    "note": "paired_video_level_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def build_case_table(fixed: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    fixed_cols = [
        "video_id",
        "truth_count",
        "signal_consensus_peaks",
        "signal_aware_applied_adjust",
        "signal_aware_final_peaks",
        "signal_aware_abs_count_error",
        "signal_aware_safe_applied_adjust",
        "signal_aware_safe_guard_reason",
        "signal_aware_safe_final_peaks",
        "signal_aware_safe_abs_count_error",
        "signal_aware_safe_autocorr_delta",
    ]
    group_cols = [
        "video_id",
        "signal_aware_group_applied_adjust",
        "signal_aware_group_final_peaks",
        "signal_aware_group_abs_count_error",
        "signal_aware_safe_group_applied_adjust",
        "signal_aware_safe_group_guard_reason",
        "signal_aware_safe_group_final_peaks",
        "signal_aware_safe_group_abs_count_error",
        "signal_aware_safe_group_autocorr_delta",
    ]
    available_fixed = [col for col in fixed_cols if col in fixed.columns]
    available_group = [col for col in group_cols if col in group.columns]
    merged = fixed[available_fixed].merge(group[available_group], on="video_id", how="outer")
    changed = (
        (numeric(merged, "signal_aware_applied_adjust") != numeric(merged, "signal_aware_safe_applied_adjust"))
        | (
            numeric(merged, "signal_aware_group_applied_adjust")
            != numeric(merged, "signal_aware_safe_group_applied_adjust")
        )
    )
    return merged.loc[changed].sort_values("video_id").reset_index(drop=True)


def write_report(
    output_path: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    max_autocorr_delta: float,
) -> None:
    def metric(label: str) -> pd.Series:
        match = metrics[metrics["label"] == label]
        if match.empty:
            raise ValueError(f"Missing metric row: {label}")
        return match.iloc[0]

    base = metric("signal_consensus_baseline")
    fixed = metric("signal_aware_safe_policy_fixed_oof")
    group = metric("signal_aware_safe_policy_prefix_group_fixed_threshold")
    original_group = metric("signal_aware_residual_prefix_group_fixed_threshold")
    ci_match = bootstrap[
        (bootstrap["validation_mode"] == "prefix_group")
        & (bootstrap["comparison"] == "safe_policy_vs_original_signal_aware")
        & (bootstrap["metric"] == "delta_abs_count_error_ge2")
    ]
    ci_text = str(ci_match.iloc[0]["estimate_with_ci"]) if not ci_match.empty else "not available"
    text = f"""# Signal-Aware Safe Policy Report

This report evaluates a conservative gate on top of the existing out-of-fold
signal-aware residual predictions. The policy starts from the signal-consensus
count and only accepts a second-stage correction when both conditions hold:

1. the source signal-aware model already applied a -1 correction;
2. `abs(autocorr_count_estimate - signal_consensus_peaks) <= {float(max_autocorr_delta):.1f}`.

The gate uses only features available at prediction time and does not inspect
`truth_count`, `truth_rr`, or residual errors.

## Results

- Signal-consensus baseline: RR R2={base['rr_r2']:.6f}, MAE={base['rr_mae']:.4f} bpm, exact={int(base['exact_count'])}/{int(base['count_valid_videos'])}, count errors >=2={int(base['abs_count_error_ge2'])}
- Safe policy fixed OOF: RR R2={fixed['rr_r2']:.6f}, MAE={fixed['rr_mae']:.4f} bpm, exact={int(fixed['exact_count'])}/{int(fixed['count_valid_videos'])}, applied={int(fixed['applied_corrections'])}, count errors >=2={int(fixed['abs_count_error_ge2'])}
- Original prefix-group signal-aware: RR R2={original_group['rr_r2']:.6f}, MAE={original_group['rr_mae']:.4f} bpm, exact={int(original_group['exact_count'])}/{int(original_group['count_valid_videos'])}, applied={int(original_group['applied_corrections'])}, count errors >=2={int(original_group['abs_count_error_ge2'])}
- Safe policy prefix-group: RR R2={group['rr_r2']:.6f}, MAE={group['rr_mae']:.4f} bpm, exact={int(group['exact_count'])}/{int(group['count_valid_videos'])}, applied={int(group['applied_corrections'])}, count errors >=2={int(group['abs_count_error_ge2'])}
- Prefix-group delta in count errors >=2 versus original signal-aware: {ci_text}

## Claim Boundary

This is a method-development result. It is stronger than the original
signal-aware candidate under prefix-group stress testing, but the grouping is
still a filename-prefix proxy rather than real cow/session/external validation.
Use it as an internal candidate innovation, not as an external generalization
claim.

Changed or blocked cases are listed in `{output_path.with_name(output_path.stem.replace('_report', '_cases') + '.csv').name}`.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    source_stem = f"{args.output_prefix}_{args.source_name}"
    output_stem = f"{args.output_prefix}_{args.output_name}"
    fixed_path = input_root / f"{source_stem}_predictions.csv"
    group_path = input_root / f"{source_stem}_group_predictions.csv"
    if not fixed_path.exists():
        raise FileNotFoundError(f"Missing fixed source predictions: {fixed_path}")
    if not group_path.exists():
        raise FileNotFoundError(f"Missing group source predictions: {group_path}")

    fixed = pd.read_csv(fixed_path)
    group = pd.read_csv(group_path)
    fixed_safe = add_safe_policy_columns(
        fixed,
        source_prefix="signal_aware",
        output_prefix="signal_aware_safe",
        max_autocorr_delta=float(args.max_autocorr_delta),
    )
    group_safe = add_safe_policy_columns(
        group,
        source_prefix="signal_aware_group",
        output_prefix="signal_aware_safe_group",
        max_autocorr_delta=float(args.max_autocorr_delta),
    )
    metrics = build_metrics(
        fixed_safe,
        group_safe,
        max_autocorr_delta=float(args.max_autocorr_delta),
    )
    ci = bootstrap_ci(
        fixed_safe,
        group_safe,
        samples=int(args.bootstrap_samples),
        random_state=int(args.bootstrap_random_state),
    )
    cases = build_case_table(fixed_safe, group_safe)

    predictions_path = input_root / f"{output_stem}_predictions.csv"
    group_predictions_path = input_root / f"{output_stem}_group_predictions.csv"
    metrics_path = input_root / f"{output_stem}_metrics.csv"
    bootstrap_path = input_root / f"{output_stem}_bootstrap_ci.csv"
    cases_path = input_root / f"{output_stem}_cases.csv"
    report_path = input_root / f"{output_stem}_report.md"

    fixed_safe.to_csv(predictions_path, index=False)
    group_safe.to_csv(group_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    ci.to_csv(bootstrap_path, index=False)
    cases.to_csv(cases_path, index=False)
    write_report(
        report_path,
        metrics,
        ci,
        cases,
        max_autocorr_delta=float(args.max_autocorr_delta),
    )

    print(f"Saved safe policy predictions: {predictions_path}")
    print(f"Saved safe policy group predictions: {group_predictions_path}")
    print(f"Saved safe policy metrics: {metrics_path}")
    print(f"Saved safe policy bootstrap CI: {bootstrap_path}")
    print(f"Saved safe policy cases: {cases_path}")
    print(f"Saved safe policy report: {report_path}")
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nChanged cases:")
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()
