from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build an uncertainty-aware selective RR reporting analysis from "
            "signal-consensus predictions."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro_selective_rr")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--strict-votes", type=int, default=3)
    parser.add_argument("--strict-margin", type=float, default=0.10)
    parser.add_argument("--strict-interval-cv", type=float, default=0.20)
    parser.add_argument("--strict-missing-rate", type=float, default=0.05)
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def robust_norm(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.zeros(len(values)), index=values.index)
    low = float(np.nanpercentile(finite, 5)) if lo is None else float(lo)
    high = float(np.nanpercentile(finite, 95)) if hi is None else float(hi)
    if high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values - low) / (high - low)).clip(0.0, 1.0).fillna(0.0)


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


def metric_row(
    label: str,
    predictions: pd.DataFrame,
    mask: pd.Series,
    rr_col: str,
    count_col: str,
    total_videos: int,
    note: str,
) -> dict[str, object]:
    subset = predictions.loc[mask].copy()
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
        "label": label,
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


def signal_vote_agreement(predictions: pd.DataFrame, final_count: pd.Series) -> pd.Series:
    vote_columns = ["spectral_count_estimate", "fft_count_estimate", "autocorr_count_estimate"]
    votes = pd.Series(np.zeros(len(predictions), dtype=int), index=predictions.index)
    rounded_final = final_count.round()
    for column in vote_columns:
        estimate = numeric(predictions, column).round()
        votes += (estimate.notna() & (estimate == rounded_final)).astype(int)
    return votes


def build_selective_predictions(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    triage = predictions.copy()
    final_count = numeric(triage, "signal_consensus_peaks")
    if final_count.isna().all():
        final_count = numeric(triage, "corrected_peaks")
        triage["signal_consensus_peaks"] = final_count
    final_rr = numeric(triage, "signal_consensus_rr_bpm")
    if final_rr.isna().all():
        final_rr = numeric(triage, "corrected_rr_bpm")
        triage["signal_consensus_rr_bpm"] = final_rr

    triage["signal_count_vote_agreement"] = signal_vote_agreement(triage, final_count)
    triage["signal_disagreement_fraction"] = 1.0 - triage["signal_count_vote_agreement"] / 3.0
    triage["interval_instability_score"] = robust_norm(numeric(triage, "selection_interval_cv"), 0.05, 0.50)
    triage["model_ambiguity_score"] = (
        1.0 - robust_norm(numeric(triage, "residual_margin"), 0.05, 0.55)
    ).clip(0.0, 1.0)
    strength = pd.concat(
        [
            robust_norm(numeric(triage, "spectral_strength")),
            robust_norm(numeric(triage, "fft_strength")),
            robust_norm(numeric(triage, "autocorr_strength")),
        ],
        axis=1,
    ).median(axis=1)
    triage["periodicity_weakness_score"] = (1.0 - strength).clip(0.0, 1.0)
    triage["missingness_score"] = robust_norm(numeric(triage, "selection_missing_rate"), 0.0, 0.25)
    triage["correction_applied_score"] = (
        (numeric(triage, "signal_consensus_adjust").abs() > 0)
        | (numeric(triage, "applied_adjust").abs() > 0)
    ).astype(float)
    triage["selective_review_score"] = (
        0.28 * triage["signal_disagreement_fraction"]
        + 0.22 * triage["interval_instability_score"]
        + 0.18 * triage["model_ambiguity_score"]
        + 0.18 * triage["periodicity_weakness_score"]
        + 0.08 * triage["missingness_score"]
        + 0.06 * triage["correction_applied_score"]
    ).clip(0.0, 1.0)

    strict_mask = (
        (triage["signal_count_vote_agreement"] >= int(args.strict_votes))
        & (numeric(triage, "residual_margin") >= float(args.strict_margin))
        & (numeric(triage, "selection_interval_cv") <= float(args.strict_interval_cv))
        & (numeric(triage, "selection_missing_rate").fillna(0.0) <= float(args.strict_missing_rate))
    )
    triage["strict_auto_accept"] = strict_mask
    triage["score_auto_accept"] = triage["selective_review_score"] <= float(args.score_threshold)
    triage["selective_action"] = np.where(strict_mask, "auto_report_strict", "manual_review_required")

    fail_reasons: list[str] = []
    for _, row in triage.iterrows():
        reasons = []
        if int(row["signal_count_vote_agreement"]) < int(args.strict_votes):
            reasons.append("signal_count_disagreement")
        if pd.isna(row.get("residual_margin")) or float(row.get("residual_margin", np.nan)) < float(
            args.strict_margin
        ):
            reasons.append("low_residual_margin")
        if pd.isna(row.get("selection_interval_cv")) or float(row.get("selection_interval_cv", np.nan)) > float(
            args.strict_interval_cv
        ):
            reasons.append("unstable_peak_interval")
        if float(row.get("selection_missing_rate", 0.0) or 0.0) > float(args.strict_missing_rate):
            reasons.append("high_missing_rate")
        fail_reasons.append("auto_accepted" if not reasons else ";".join(reasons))
    triage["selective_triage_reason"] = fail_reasons
    triage["selective_count_error"] = final_count - numeric(triage, "truth_count")
    triage["selective_abs_count_error"] = triage["selective_count_error"].abs()
    triage["selective_rr_error"] = final_rr - numeric(triage, "truth_rr")
    triage["selective_abs_rr_error"] = triage["selective_rr_error"].abs()
    return triage


def build_metrics(triage: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    total = len(triage)
    rows = [
        metric_row(
            "all_signal_consensus_predictions",
            triage,
            pd.Series(True, index=triage.index),
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
            total,
            "all_predictions_before_selective_triage",
        ),
        metric_row(
            "strict_auto_report_subset",
            triage,
            triage["strict_auto_accept"].astype(bool),
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
            total,
            (
                f"votes>={args.strict_votes}; residual_margin>={args.strict_margin:.2f}; "
                f"interval_cv<={args.strict_interval_cv:.2f}; missing_rate<={args.strict_missing_rate:.2f}"
            ),
        ),
        metric_row(
            "strict_manual_review_subset",
            triage,
            ~triage["strict_auto_accept"].astype(bool),
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
            total,
            "videos_not_meeting_strict_auto_report_rule",
        ),
        metric_row(
            f"score_auto_report_subset_{args.score_threshold:.2f}",
            triage,
            triage["score_auto_accept"].astype(bool),
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
            total,
            f"selective_review_score<={args.score_threshold:.2f}",
        ),
        metric_row(
            f"score_manual_review_subset_{args.score_threshold:.2f}",
            triage,
            ~triage["score_auto_accept"].astype(bool),
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
            total,
            f"selective_review_score>{args.score_threshold:.2f}",
        ),
    ]
    return pd.DataFrame(rows)


def build_threshold_grid(triage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(triage)
    for threshold in np.arange(0.30, 0.71, 0.05):
        mask = triage["selective_review_score"] <= threshold
        rows.append(
            metric_row(
                f"score_threshold_{threshold:.2f}",
                triage,
                mask,
                "signal_consensus_rr_bpm",
                "signal_consensus_peaks",
                total,
                f"selective_review_score<={threshold:.2f}",
            )
        )
    return pd.DataFrame(rows)


def write_summary(
    output_path: Path,
    metrics: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    strict = metrics[metrics["label"] == "strict_auto_report_subset"].iloc[0]
    all_row = metrics[metrics["label"] == "all_signal_consensus_predictions"].iloc[0]
    score_row = metrics[metrics["label"] == f"score_auto_report_subset_{args.score_threshold:.2f}"].iloc[0]
    lines = [
        "# Selective RR Reporting Analysis",
        "",
        "This internal analysis evaluates an uncertainty-aware deployment layer for the signal-consensus RR candidate.",
        "The rule does not change the RR prediction. It separates videos into an automatic-report subset and a manual-review subset using non-truth signal quality fields.",
        "",
        "## Fixed Strict Rule",
        "",
        (
            f"The strict rule requires signal count vote agreement >= {args.strict_votes}, "
            f"residual-model margin >= {args.strict_margin:.2f}, peak-interval CV <= {args.strict_interval_cv:.2f}, "
            f"and missing rate <= {args.strict_missing_rate:.2f}."
        ),
        "",
        (
            f"All signal-consensus predictions: n={int(all_row['videos'])}, "
            f"RR R2={float(all_row['rr_r2']):.4f}, MAE={float(all_row['rr_mae']):.4f} bpm, "
            f"exact={int(all_row['exact_count'])}/{int(all_row['count_valid_videos'])}."
        ),
        (
            f"Strict automatic-report subset: n={int(strict['videos'])} "
            f"({float(strict['coverage']) * 100:.1f}% coverage), RR R2={float(strict['rr_r2']):.4f}, "
            f"MAE={float(strict['rr_mae']):.4f} bpm, exact={int(strict['exact_count'])}/{int(strict['count_valid_videos'])}."
        ),
        (
            f"Score-threshold automatic subset at {args.score_threshold:.2f}: n={int(score_row['videos'])} "
            f"({float(score_row['coverage']) * 100:.1f}% coverage), RR R2={float(score_row['rr_r2']):.4f}, "
            f"MAE={float(score_row['rr_mae']):.4f} bpm, exact={int(score_row['exact_count'])}/{int(score_row['count_valid_videos'])}."
        ),
        "",
        "## Interpretation Boundary",
        "",
        "This is a selective-reporting and review-prioritization analysis, not an external validation result. The thresholds are internal candidates and should be refit or frozen on training data before prospective deployment.",
        "",
        "## Score Threshold Grid",
        "",
        "| threshold | coverage | rr_r2 | rr_mae | exact | exact_rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in threshold_grid.iterrows():
        threshold = str(row["label"]).rsplit("_", 1)[-1]
        lines.append(
            f"| {threshold} | {float(row['coverage']) * 100:.1f}% | "
            f"{float(row['rr_r2']):.4f} | {float(row['rr_mae']):.4f} | "
            f"{int(row['exact_count'])}/{int(row['count_valid_videos'])} | "
            f"{float(row['exact_rate']) * 100:.1f}% |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = args.input_file or args.input_root / "paper_repro_signal_consensus_predictions.csv"
    predictions = pd.read_csv(input_file)
    triage = build_selective_predictions(predictions, args)
    metrics = build_metrics(triage, args)
    threshold_grid = build_threshold_grid(triage)

    predictions_path = args.input_root / f"{args.output_prefix}_predictions.csv"
    metrics_path = args.input_root / f"{args.output_prefix}_metrics.csv"
    threshold_path = args.input_root / f"{args.output_prefix}_threshold_grid.csv"
    summary_path = args.input_root / f"{args.output_prefix}_summary.md"

    triage.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    threshold_grid.to_csv(threshold_path, index=False)
    write_summary(summary_path, metrics, threshold_grid, args)

    print(f"Saved selective RR predictions: {predictions_path}")
    print(f"Saved selective RR metrics: {metrics_path}")
    print(f"Saved selective RR threshold grid: {threshold_path}")
    print(f"Saved selective RR summary: {summary_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
