from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "default_pipeline": ("rr_bpm", "peaks"),
    "quality_residual": ("corrected_rr_bpm", "corrected_peaks"),
    "signal_consensus": ("signal_consensus_rr_bpm", "signal_consensus_peaks"),
    "signal_aware_safe_gate": (
        "signal_aware_safe_final_rr_bpm",
        "signal_aware_safe_final_peaks",
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build non-truth algorithmic quality/risk strata for RR predictions. "
            "The score is derived from curve quality, signal agreement, model "
            "margin, and safe-gate state; reference RR is used only for reporting "
            "stratum-level performance."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Defaults to <input-root>/<output-prefix>_signal_aware_safe_policy_predictions.csv.",
    )
    parser.add_argument(
        "--quality-prefix",
        default="paper_repro_algorithmic_quality",
        help="Prefix for output CSV/MD files.",
    )
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def robust_norm(
    series: pd.Series,
    lo: float | None = None,
    hi: float | None = None,
    fill: float = 0.0,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(fill, index=values.index, dtype=float)
    low = float(np.nanpercentile(finite, 5)) if lo is None else float(lo)
    high = float(np.nanpercentile(finite, 95)) if hi is None else float(hi)
    if high <= low:
        return pd.Series(fill, index=values.index, dtype=float)
    return ((values - low) / (high - low)).clip(0.0, 1.0).fillna(fill)


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    if len(valid) < 2:
        return math.nan
    truth = valid["truth"].to_numpy(dtype=float)
    pred = valid["pred"].to_numpy(dtype=float)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if ss_tot == 0.0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def pearson_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    if len(valid) < 2:
        return math.nan
    corr = float(np.corrcoef(valid["truth"], valid["pred"])[0, 1])
    return corr * corr


def count_vote_agreement(data: pd.DataFrame, final_count: pd.Series) -> tuple[pd.Series, pd.Series]:
    vote_columns = [
        "spectral_count_estimate",
        "fft_count_estimate",
        "autocorr_count_estimate",
    ]
    rounded_final = final_count.round()
    votes = pd.Series(np.zeros(len(data), dtype=float), index=data.index)
    available = pd.Series(np.zeros(len(data), dtype=float), index=data.index)
    for column in vote_columns:
        estimate = numeric(data, column).round()
        present = estimate.notna()
        available += present.astype(float)
        votes += (present & (estimate == rounded_final)).astype(float)
    agreement = (votes / available.replace(0.0, np.nan)).fillna(0.5)
    return votes.astype(int), agreement.clip(0.0, 1.0)


def max_signal_delta(data: pd.DataFrame, final_count: pd.Series) -> pd.Series:
    deltas = []
    for column in [
        "spectral_count_estimate",
        "fft_count_estimate",
        "autocorr_count_estimate",
    ]:
        deltas.append((numeric(data, column).round() - final_count.round()).abs())
    return pd.concat(deltas, axis=1).max(axis=1).fillna(0.0)


def guard_score(data: pd.DataFrame) -> pd.Series:
    reason = data.get(
        "signal_aware_safe_guard_reason",
        pd.Series("", index=data.index),
    ).astype(str)
    scores = pd.Series(np.zeros(len(data), dtype=float), index=data.index)
    scores[reason.str.contains("blocked_autocorr_discordance", case=False, regex=False)] = 1.0
    scores[reason.str.contains("blocked_positive_adjustment", case=False, regex=False)] = 0.8
    scores[reason.str.contains("no_source_adjustment", case=False, regex=False)] = 0.1
    scores[reason.str.contains("accepted", case=False, regex=False)] = 0.0
    return scores.clip(0.0, 1.0)


def add_quality_scores(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    final_count = numeric(scored, "signal_aware_safe_final_peaks")
    if final_count.isna().all():
        final_count = numeric(scored, "signal_consensus_peaks")
    if final_count.isna().all():
        final_count = numeric(scored, "corrected_peaks")
    if final_count.isna().all():
        final_count = numeric(scored, "peaks")

    votes, agreement = count_vote_agreement(scored, final_count)
    scored["algorithmic_signal_vote_count"] = votes
    scored["algorithmic_signal_agreement"] = agreement
    scored["algorithmic_signal_disagreement_score"] = 1.0 - agreement
    scored["algorithmic_signal_delta_score"] = robust_norm(
        max_signal_delta(scored, final_count), 0.0, 6.0
    )
    scored["algorithmic_interval_instability_score"] = robust_norm(
        numeric(scored, "selection_interval_cv"), 0.05, 0.50
    )
    residual_margin = numeric(scored, "residual_margin")
    signal_margin = numeric(scored, "signal_aware_margin")
    best_margin = pd.concat([residual_margin, signal_margin], axis=1).max(axis=1)
    scored["algorithmic_model_ambiguity_score"] = (
        1.0 - robust_norm(best_margin, 0.05, 0.90, fill=0.5)
    ).clip(0.0, 1.0)
    strength = pd.concat(
        [
            robust_norm(numeric(scored, "spectral_strength"), fill=0.5),
            robust_norm(numeric(scored, "fft_strength"), fill=0.5),
            robust_norm(numeric(scored, "autocorr_strength"), fill=0.5),
        ],
        axis=1,
    ).median(axis=1)
    scored["algorithmic_periodicity_weakness_score"] = (1.0 - strength).clip(0.0, 1.0)
    scored["algorithmic_missingness_score"] = robust_norm(
        numeric(scored, "selection_missing_rate"), 0.0, 0.25
    )
    scored["algorithmic_guard_discordance_score"] = guard_score(scored)
    adjustment_cols = [
        numeric(scored, "applied_adjust").abs() > 0,
        numeric(scored, "signal_consensus_adjust").abs() > 0,
        numeric(scored, "signal_aware_safe_applied_adjust").abs() > 0,
    ]
    scored["algorithmic_correction_activity_score"] = (
        pd.concat(adjustment_cols, axis=1).any(axis=1).astype(float)
    )
    scored["algorithmic_review_risk_score"] = (
        0.22 * scored["algorithmic_signal_disagreement_score"]
        + 0.18 * scored["algorithmic_signal_delta_score"]
        + 0.16 * scored["algorithmic_interval_instability_score"]
        + 0.14 * scored["algorithmic_model_ambiguity_score"]
        + 0.12 * scored["algorithmic_periodicity_weakness_score"]
        + 0.08 * scored["algorithmic_missingness_score"]
        + 0.06 * scored["algorithmic_guard_discordance_score"]
        + 0.04 * scored["algorithmic_correction_activity_score"]
    ).clip(0.0, 1.0)
    scored["algorithmic_quality_score"] = 1.0 - scored["algorithmic_review_risk_score"]
    scored["algorithmic_risk_tier_fixed"] = pd.cut(
        scored["algorithmic_review_risk_score"],
        bins=[-0.001, 0.25, 0.45, 1.001],
        labels=["low_risk_auto_candidate", "moderate_risk", "high_risk_review_priority"],
    ).astype(str)
    scored["algorithmic_risk_tertile"] = pd.qcut(
        scored["algorithmic_review_risk_score"].rank(method="first"),
        q=3,
        labels=["lowest_third", "middle_third", "highest_third"],
    ).astype(str)
    scored["algorithmic_quality_score_uses_truth"] = False
    return scored


def metric_row(
    method: str,
    tier_type: str,
    tier: str,
    subset: pd.DataFrame,
    total_videos: int,
    rr_col: str,
    count_col: str,
    note: str,
) -> dict[str, object]:
    truth_rr = numeric(subset, "truth_rr")
    pred_rr = numeric(subset, rr_col)
    truth_count = numeric(subset, "truth_count")
    pred_count = numeric(subset, count_col)
    rr_error = pred_rr - truth_rr
    count_error = pred_count - truth_count
    valid_rr = pd.DataFrame({"truth": truth_rr, "pred": pred_rr}).dropna()
    valid_count = pd.DataFrame({"truth": truth_count, "pred": pred_count}).dropna()
    abs_count = count_error.abs()
    return {
        "method": method,
        "tier_type": tier_type,
        "tier": tier,
        "videos": int(len(subset)),
        "coverage": float(len(subset) / total_videos) if total_videos else math.nan,
        "rr_valid_videos": int(len(valid_rr)),
        "rr_r2": regression_r2(truth_rr, pred_rr),
        "rr_pearson_r2": pearson_r2(truth_rr, pred_rr),
        "rr_mae": float(rr_error.abs().mean()) if len(valid_rr) else math.nan,
        "rr_rmse": float(np.sqrt(np.mean(np.square(rr_error.dropna())))) if len(valid_rr) else math.nan,
        "count_valid_videos": int(len(valid_count)),
        "count_mae": float(abs_count.mean()) if len(valid_count) else math.nan,
        "exact_count": int((abs_count == 0).sum()),
        "exact_rate": float((abs_count == 0).mean()) if len(valid_count) else math.nan,
        "within_one_count": int((abs_count <= 1).sum()),
        "abs_count_error_ge1": int((abs_count >= 1).sum()),
        "abs_count_error_ge2": int((abs_count >= 2).sum()),
        "evaluation_note": note,
    }


def build_tier_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(scored)
    for method, (rr_col, count_col) in METHODS.items():
        if rr_col not in scored.columns or count_col not in scored.columns:
            continue
        rows.append(
            metric_row(
                method,
                "overall",
                "all_videos",
                scored,
                total,
                rr_col,
                count_col,
                "all videos; algorithmic score not used for filtering",
            )
        )
        for tier_type, column in [
            ("fixed_risk_tier", "algorithmic_risk_tier_fixed"),
            ("descriptive_tertile", "algorithmic_risk_tertile"),
        ]:
            for tier, subset in scored.groupby(column, sort=True):
                rows.append(
                    metric_row(
                        method,
                        tier_type,
                        str(tier),
                        subset,
                        total,
                        rr_col,
                        count_col,
                        (
                            "risk tier computed from non-truth signal-quality "
                            "features; truth used only for evaluation"
                        ),
                    )
                )
    return pd.DataFrame(rows)


def build_feature_summary(scored: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "algorithmic_review_risk_score",
        "algorithmic_quality_score",
        "algorithmic_signal_agreement",
        "algorithmic_signal_delta_score",
        "algorithmic_interval_instability_score",
        "algorithmic_model_ambiguity_score",
        "algorithmic_periodicity_weakness_score",
        "algorithmic_missingness_score",
        "algorithmic_guard_discordance_score",
        "algorithmic_correction_activity_score",
        "selection_interval_cv",
        "selection_missing_rate",
        "residual_margin",
        "signal_aware_margin",
        "signal_aware_safe_autocorr_delta",
        "signal_aware_safe_fft_delta",
    ]
    rows: list[dict[str, object]] = []
    for tier_type, column in [
        ("fixed_risk_tier", "algorithmic_risk_tier_fixed"),
        ("descriptive_tertile", "algorithmic_risk_tertile"),
    ]:
        for tier, subset in scored.groupby(column, sort=True):
            row: dict[str, object] = {
                "tier_type": tier_type,
                "tier": str(tier),
                "videos": int(len(subset)),
            }
            for feature in feature_columns:
                if feature in subset.columns:
                    row[f"{feature}_mean"] = float(numeric(subset, feature).mean())
                    row[f"{feature}_median"] = float(numeric(subset, feature).median())
            rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_grid(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(scored)
    rr_col, count_col = METHODS["signal_aware_safe_gate"]
    for threshold in np.arange(0.20, 0.71, 0.05):
        mask = scored["algorithmic_review_risk_score"] <= threshold
        rows.append(
            metric_row(
                "signal_aware_safe_gate",
                "risk_score_threshold",
                f"risk_le_{threshold:.2f}",
                scored.loc[mask],
                total,
                rr_col,
                count_col,
                "automatic-report candidate subset by non-truth risk score",
            )
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    scored: pd.DataFrame,
    tier_metrics: pd.DataFrame,
    threshold_grid: pd.DataFrame,
) -> None:
    prefix = path.name.removesuffix("_report.md")
    safe_overall = tier_metrics[
        (tier_metrics["method"] == "signal_aware_safe_gate")
        & (tier_metrics["tier_type"] == "overall")
    ].iloc[0]
    fixed = tier_metrics[
        (tier_metrics["method"] == "signal_aware_safe_gate")
        & (tier_metrics["tier_type"] == "fixed_risk_tier")
    ].copy()
    low = fixed[fixed["tier"] == "low_risk_auto_candidate"]
    high = fixed[fixed["tier"] == "high_risk_review_priority"]

    def compact(row: pd.Series | None) -> str:
        if row is None or row.empty:
            return "not available"
        return (
            f"n={int(row['videos'])}, R2={float(row['rr_r2']):.4f}, "
            f"MAE={float(row['rr_mae']):.3f}, exact={int(row['exact_count'])}/"
            f"{int(row['count_valid_videos'])}"
        )

    low_text = compact(low.iloc[0] if not low.empty else None)
    high_text = compact(high.iloc[0] if not high.empty else None)
    lines = [
        "# Algorithmic Quality Context For Thermal RR",
        "",
        "This report builds a non-truth quality/risk score from signal agreement, "
        "peak-interval stability, model margins, missingness, periodicity strength, "
        "and safe-gate state. The score is not a manual head-motion, occlusion, "
        "or nostril-visibility label. Reference RR is used only after scoring to "
        "describe stratum-level performance.",
        "",
        "## Summary",
        "",
        f"- Videos scored: `{len(scored)}`.",
        (
            "- Overall safe-gate performance: "
            f"R2=`{float(safe_overall['rr_r2']):.4f}`, "
            f"MAE=`{float(safe_overall['rr_mae']):.3f}` bpm, "
            f"exact=`{int(safe_overall['exact_count'])}/"
            f"{int(safe_overall['count_valid_videos'])}`."
        ),
        f"- Low-risk automatic-candidate tier: {low_text}.",
        f"- High-risk review-priority tier: {high_text}.",
        "",
        "## Claim Boundary",
        "",
        "- Manuscript-safe use: deployment triage, algorithmic curve-quality context, "
        "and prioritization for manual review.",
        "- Avoid: claiming these are visual head-motion/occlusion labels or an "
        "externally validated automatic-release policy.",
        "- Next proof needed: freeze the risk threshold before testing on a true "
        "external set and compare false automatic accepts.",
        "",
        "## Files",
        "",
        f"- `{prefix}_predictions.csv`: per-video quality scores.",
        f"- `{prefix}_tier_metrics.csv`: method performance by risk tier.",
        f"- `{prefix}_feature_summary.csv`: feature summaries by tier.",
        f"- `{prefix}_threshold_grid.csv`: safe-gate performance by risk threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    input_file = (
        args.input_file.resolve()
        if args.input_file is not None
        else input_root / f"{args.output_prefix}_signal_aware_safe_policy_predictions.csv"
    )
    predictions = pd.read_csv(input_file)
    scored = add_quality_scores(predictions)
    tier_metrics = build_tier_metrics(scored)
    feature_summary = build_feature_summary(scored)
    threshold_grid = build_threshold_grid(scored)

    pred_path = input_root / f"{args.quality_prefix}_predictions.csv"
    tier_path = input_root / f"{args.quality_prefix}_tier_metrics.csv"
    feature_path = input_root / f"{args.quality_prefix}_feature_summary.csv"
    threshold_path = input_root / f"{args.quality_prefix}_threshold_grid.csv"
    report_path = input_root / f"{args.quality_prefix}_report.md"
    scored.to_csv(pred_path, index=False)
    tier_metrics.to_csv(tier_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    threshold_grid.to_csv(threshold_path, index=False)
    write_report(report_path, scored, tier_metrics, threshold_grid)

    print(f"Saved algorithmic quality predictions: {pred_path}")
    print(f"Saved algorithmic quality tier metrics: {tier_path}")
    print(f"Saved algorithmic quality feature summary: {feature_path}")
    print(f"Saved algorithmic quality threshold grid: {threshold_path}")
    print(f"Saved algorithmic quality report: {report_path}")
    print("\nSafe-gate tier metrics:")
    safe = tier_metrics[tier_metrics["method"] == "signal_aware_safe_gate"]
    print(
        safe[
            [
                "tier_type",
                "tier",
                "videos",
                "coverage",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "within_one_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
