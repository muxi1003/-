from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a bilateral nostril physiological-consistency gate for the "
            "thermal RR package. The gate uses only prediction-time left/right "
            "nostril temperature-curve agreement and the existing RR-only "
            "auto-report flag; truth labels are used only for post-hoc internal "
            "evaluation."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--min-bilateral-corr", type=float, default=0.20)
    parser.add_argument("--max-fft-count-delta", type=float, default=1.0)
    parser.add_argument("--min-amplitude-ratio", type=float, default=0.10)
    parser.add_argument("--max-correlation-lag", type=int, default=8)
    return parser.parse_args()


def output_dir_for(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, dtype={"video_id": str}, keep_default_na=False)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normalize_curve(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 8:
        return None
    filled = values.copy()
    median = float(np.nanmedian(filled))
    filled[~finite] = median
    low, high = [float(v) for v in np.nanpercentile(filled, [5, 95])]
    scale = high - low
    if not np.isfinite(scale) or scale <= 1e-9:
        return None
    return np.clip((filled - low) / scale, -2.0, 3.0)


def best_lagged_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_lag: int,
) -> tuple[float, int, float]:
    left_norm = normalize_curve(left)
    right_norm = normalize_curve(right)
    if left_norm is None or right_norm is None:
        return math.nan, 0, math.nan
    n = min(len(left_norm), len(right_norm))
    left_norm = left_norm[:n]
    right_norm = right_norm[:n]

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 8:
            return math.nan
        if float(np.nanstd(x)) <= 1e-9 or float(np.nanstd(y)) <= 1e-9:
            return math.nan
        return float(np.corrcoef(x, y)[0, 1])

    zero_corr = corr(left_norm, right_norm)
    best_corr = -2.0
    best_lag = 0
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            x = left_norm[-lag:]
            y = right_norm[: n + lag]
        elif lag > 0:
            x = left_norm[: n - lag]
            y = right_norm[lag:]
        else:
            x = left_norm
            y = right_norm
        value = corr(x, y)
        if np.isfinite(value) and value > best_corr:
            best_corr = value
            best_lag = lag
    if best_corr <= -2.0:
        return math.nan, 0, zero_corr
    return best_corr, best_lag, zero_corr


def fft_count_estimate(values: np.ndarray) -> tuple[float, float]:
    norm = normalize_curve(values)
    if norm is None or len(norm) < 16:
        return math.nan, math.nan
    n = len(norm)
    centered = norm - float(np.nanmean(norm))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(n))) ** 2
    if len(spectrum) < 5:
        return math.nan, math.nan
    spectrum[0] = 0.0
    min_cycles = 3
    max_cycles = min(18, n // 2)
    if max_cycles < min_cycles:
        return math.nan, math.nan
    scores = spectrum.copy()
    mask = np.zeros_like(scores, dtype=bool)
    mask[min_cycles : max_cycles + 1] = True
    scores[~mask] = 0.0
    index = int(np.argmax(scores))
    if scores[index] <= 0:
        return math.nan, math.nan
    strength = float(scores[index] / (np.sum(spectrum) + 1e-9))
    return float(index), strength


def amplitude(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 8:
        return math.nan
    return float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5))


def build_features(
    input_root: Path,
    video_ids: pd.Series,
    output_prefix: str,
    *,
    max_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for video_id in video_ids.astype(str).drop_duplicates().sort_values():
        curve_path = input_root / video_id / f"{output_prefix}_curve.csv"
        row: dict[str, object] = {
            "video_id": video_id,
            "curve_csv": str(curve_path),
            "bilateral_feature_available": False,
            "bilateral_best_corr": math.nan,
            "bilateral_best_lag_frames": math.nan,
            "bilateral_abs_lag_frames": math.nan,
            "bilateral_zero_lag_corr": math.nan,
            "left_fft_count_estimate": math.nan,
            "right_fft_count_estimate": math.nan,
            "fused_fft_count_estimate": math.nan,
            "left_fft_strength": math.nan,
            "right_fft_strength": math.nan,
            "fused_fft_strength": math.nan,
            "bilateral_fft_count_delta": math.nan,
            "bilateral_amplitude_ratio": math.nan,
        }
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            left = pd.to_numeric(curve.get("left_norm"), errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(curve.get("right_norm"), errors="coerce").to_numpy(dtype=float)
            fused_source = curve.get("smoothed_norm")
            if fused_source is None:
                fused_source = curve.get("fused_norm")
            fused = pd.to_numeric(fused_source, errors="coerce").to_numpy(dtype=float)
            if np.isfinite(fused).sum() < 8 and "fused_norm" in curve.columns:
                fused = pd.to_numeric(curve["fused_norm"], errors="coerce").to_numpy(dtype=float)

            best_corr, best_lag, zero_corr = best_lagged_correlation(
                left,
                right,
                max_lag=max_lag,
            )
            left_count, left_strength = fft_count_estimate(left)
            right_count, right_strength = fft_count_estimate(right)
            fused_count, fused_strength = fft_count_estimate(fused)
            left_amp = amplitude(left)
            right_amp = amplitude(right)
            if np.isfinite(left_amp) and np.isfinite(right_amp) and max(left_amp, right_amp) > 1e-9:
                amp_ratio = min(left_amp, right_amp) / max(left_amp, right_amp)
            else:
                amp_ratio = math.nan
            if np.isfinite(left_count) and np.isfinite(right_count):
                count_delta = abs(left_count - right_count)
            else:
                count_delta = math.nan
            row.update(
                {
                    "bilateral_feature_available": bool(np.isfinite(best_corr)),
                    "bilateral_best_corr": best_corr,
                    "bilateral_best_lag_frames": best_lag,
                    "bilateral_abs_lag_frames": abs(best_lag),
                    "bilateral_zero_lag_corr": zero_corr,
                    "left_fft_count_estimate": left_count,
                    "right_fft_count_estimate": right_count,
                    "fused_fft_count_estimate": fused_count,
                    "left_fft_strength": left_strength,
                    "right_fft_strength": right_strength,
                    "fused_fft_strength": fused_strength,
                    "bilateral_fft_count_delta": count_delta,
                    "bilateral_amplitude_ratio": amp_ratio,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    if len(valid) < 2:
        return math.nan
    truth = valid["truth"].to_numpy(dtype=float)
    pred = valid["pred"].to_numpy(dtype=float)
    sse = float(np.sum((truth - pred) ** 2))
    sst = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if sst == 0.0:
        return math.nan
    return 1.0 - sse / sst


def metric_row(
    data: pd.DataFrame,
    mask: pd.Series,
    label: str,
    selection_rule: str,
) -> dict[str, object]:
    subset = data.loc[mask].copy()
    truth_rr = numeric(subset, "truth_rr")
    pred_rr = numeric(subset, "bilateral_gate_predicted_rr_bpm")
    truth_count = numeric(subset, "truth_count")
    pred_count = numeric(subset, "bilateral_gate_predicted_count")
    valid = truth_rr.notna() & pred_rr.notna()
    exact = int((truth_count.round() == pred_count.round()).sum()) if len(subset) else 0
    n = int(valid.sum())
    return {
        "subset": label,
        "selection_rule": selection_rule,
        "uses_truth_for_decision": False,
        "videos": int(len(subset)),
        "valid_rr_pairs": n,
        "coverage": int(len(subset)) / max(len(data), 1),
        "rr_r2": regression_r2(truth_rr, pred_rr),
        "rr_mae_bpm": float(np.nanmean(np.abs(truth_rr - pred_rr))) if n else math.nan,
        "rr_rmse_bpm": float(np.sqrt(np.nanmean((truth_rr - pred_rr) ** 2))) if n else math.nan,
        "exact_count": f"{exact}/{len(subset)}",
        "exact_rate": exact / max(len(subset), 1),
    }


def build_gate(
    safe: pd.DataFrame,
    triage: pd.DataFrame,
    features: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    triage_cols = [
        column
        for column in [
            "video_id",
            "triage_auto_report_candidate",
            "triage_rr_band",
            "triage_recommended_action",
            "algorithmic_review_risk_score",
        ]
        if column in triage.columns
    ]
    data = safe.merge(features, on="video_id", how="left", validate="one_to_one")
    if "video_id" in triage_cols:
        data = data.merge(triage[triage_cols], on="video_id", how="left", validate="one_to_one")
    data["bilateral_gate_predicted_count"] = numeric(data, "signal_aware_safe_final_peaks")
    data["bilateral_gate_predicted_rr_bpm"] = numeric(data, "signal_aware_safe_final_rr_bpm")
    data["triage_auto_report_candidate"] = (
        data.get("triage_auto_report_candidate", pd.Series(False, index=data.index))
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    data["bilateral_corr_pass"] = numeric(data, "bilateral_best_corr") >= float(args.min_bilateral_corr)
    data["bilateral_count_delta_pass"] = numeric(data, "bilateral_fft_count_delta") <= float(args.max_fft_count_delta)
    data["bilateral_amplitude_pass"] = numeric(data, "bilateral_amplitude_ratio") >= float(args.min_amplitude_ratio)
    data["bilateral_consistency_pass"] = (
        data["bilateral_corr_pass"]
        & data["bilateral_count_delta_pass"]
        & data["bilateral_amplitude_pass"]
    )
    data["bilateral_auto_report_candidate"] = (
        data["triage_auto_report_candidate"] & data["bilateral_consistency_pass"]
    )
    data["bilateral_gate_uses_truth_for_decision"] = False
    data["bilateral_gate_rule"] = (
        "triage_auto_report_candidate AND "
        f"bilateral_best_corr >= {float(args.min_bilateral_corr):.2f} AND "
        f"bilateral_fft_count_delta <= {float(args.max_fft_count_delta):.1f} AND "
        f"bilateral_amplitude_ratio >= {float(args.min_amplitude_ratio):.2f}"
    )
    action = pd.Series("manual_review_not_rr_auto_report", index=data.index, dtype=object)
    action[data["triage_auto_report_candidate"]] = "manual_review_bilateral_consistency_failed"
    action[data["bilateral_auto_report_candidate"]] = "auto_report_bilateral_consistent_rr"
    high_rr = data.get("triage_rr_band", pd.Series("", index=data.index)).astype(str).str.contains(
        "upper_quartile|top_decile", case=False, regex=True
    )
    action[data["bilateral_auto_report_candidate"] & high_rr] = (
        "auto_report_bilateral_consistent_high_rr_context_review"
    )
    data["bilateral_gate_recommended_action"] = action
    data["bilateral_claim_boundary"] = (
        "Internal RR reliability screen only. Bilateral consistency does not diagnose "
        "heat stress, disease, welfare state, or animal identity; thresholds must be "
        "frozen before external evaluation."
    )
    keep = [
        "video_id",
        "bilateral_gate_predicted_count",
        "bilateral_gate_predicted_rr_bpm",
        "bilateral_feature_available",
        "bilateral_best_corr",
        "bilateral_best_lag_frames",
        "bilateral_abs_lag_frames",
        "bilateral_zero_lag_corr",
        "left_fft_count_estimate",
        "right_fft_count_estimate",
        "fused_fft_count_estimate",
        "left_fft_strength",
        "right_fft_strength",
        "fused_fft_strength",
        "bilateral_fft_count_delta",
        "bilateral_amplitude_ratio",
        "triage_auto_report_candidate",
        "bilateral_corr_pass",
        "bilateral_count_delta_pass",
        "bilateral_amplitude_pass",
        "bilateral_consistency_pass",
        "bilateral_auto_report_candidate",
        "bilateral_gate_rule",
        "bilateral_gate_recommended_action",
        "bilateral_gate_uses_truth_for_decision",
        "triage_rr_band",
        "algorithmic_review_risk_score",
        "truth_count",
        "truth_rr",
        "signal_aware_safe_abs_count_error",
        "signal_aware_safe_abs_rr_error",
        "bilateral_claim_boundary",
    ]
    existing = [column for column in keep if column in data.columns]
    return data[existing].copy()


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        metric_row(
            predictions,
            pd.Series(True, index=predictions.index),
            "all_safe_gate_predictions",
            "all conservative signal-aware safe-gate predictions",
        ),
        metric_row(
            predictions,
            predictions["triage_auto_report_candidate"].astype(bool),
            "rr_only_auto_report_subset",
            "existing RR-only triage auto-report candidate",
        ),
        metric_row(
            predictions,
            predictions["bilateral_consistency_pass"].astype(bool),
            "bilateral_consistent_all_predictions",
            "bilateral signal-consistency gate only",
        ),
        metric_row(
            predictions,
            predictions["bilateral_auto_report_candidate"].astype(bool),
            "bilateral_consistent_auto_report_subset",
            "existing RR-only auto-report AND bilateral signal-consistency gate",
        ),
        metric_row(
            predictions,
            (
                predictions["triage_auto_report_candidate"].astype(bool)
                & ~predictions["bilateral_consistency_pass"].astype(bool)
            ),
            "auto_report_rejected_by_bilateral_gate",
            "existing RR-only auto-report candidate but bilateral consistency failed",
        ),
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("\n", " ").replace("|", "/") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    best = summary[summary["subset"].eq("bilateral_consistent_auto_report_subset")]
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    text = [
        "# Bilateral Nostril Consistency Gate",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        (
            "This internal reliability layer asks whether the left and right nostril "
            "temperature curves carry compatible respiratory periodicity before a "
            "low-risk RR estimate is auto-reported. It is designed to improve the "
            "physiological interpretability of selective RR reporting, not to infer "
            "heat stress or animal health status."
        ),
        "",
        "## Frozen Candidate Rule",
        "",
        (
            "- Decision rule: existing RR-only auto-report candidate AND "
            f"`bilateral_best_corr >= {float(args.min_bilateral_corr):.2f}` AND "
            f"`bilateral_fft_count_delta <= {float(args.max_fft_count_delta):.1f}` AND "
            f"`bilateral_amplitude_ratio >= {float(args.min_amplitude_ratio):.2f}`."
        ),
        "- The rule uses only prediction-time curve features and existing non-truth triage output.",
        "- Manual reference RR is used only to evaluate internal performance after the decision is made.",
        "",
        "## Internal Metrics",
        "",
        markdown_table(
            summary,
            [
                "subset",
                "videos",
                "coverage",
                "rr_r2",
                "rr_mae_bpm",
                "rr_rmse_bpm",
                "exact_count",
                "exact_rate",
            ],
        ),
        "",
        "## Manuscript-Safe Result",
        "",
        (
            "The bilateral-consistent auto-report subset reached "
            f"n={best_row.get('videos', 'NA')}, coverage={float(best_row.get('coverage', math.nan)):.3f}, "
            f"RR R2={float(best_row.get('rr_r2', math.nan)):.4f}, "
            f"MAE={float(best_row.get('rr_mae_bpm', math.nan)):.3f} breaths/min, "
            f"RMSE={float(best_row.get('rr_rmse_bpm', math.nan)):.3f} breaths/min, "
            f"and exact={best_row.get('exact_count', 'NA')} on the current internal set."
        ),
        "",
        "## Claim Boundary",
        "",
        (
            "Report this as an internal physiological-consistency reliability screen. "
            "Do not describe it as external validation, heat-stress diagnosis, welfare "
            "classification, disease detection, or true animal-identity evidence. The "
            "rule should be frozen before any independent external scoring."
        ),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = output_dir_for(input_root, args.corrected_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    safe = read_required(input_root / f"{args.output_prefix}_signal_aware_safe_policy_predictions.csv")
    triage = read_required(output_dir / "paper_rr_physiological_triage_predictions.csv")
    features = build_features(
        input_root,
        safe["video_id"].astype(str),
        args.output_prefix,
        max_lag=int(args.max_correlation_lag),
    )
    predictions = build_gate(safe, triage, features, args)
    summary = build_summary(predictions)

    features_path = output_dir / "paper_rr_bilateral_consistency_features.csv"
    predictions_path = output_dir / "paper_rr_bilateral_consistency_predictions.csv"
    summary_path = output_dir / "paper_rr_bilateral_consistency_summary.csv"
    report_path = output_dir / "paper_rr_bilateral_consistency_report.md"
    docs_report_path = docs_dir / "thermal_rr_bilateral_consistency_gate.md"
    features.to_csv(features_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, summary, args)
    docs_report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved bilateral consistency features: {features_path}")
    print(f"Saved bilateral consistency predictions: {predictions_path}")
    print(f"Saved bilateral consistency summary: {summary_path}")
    print(f"Saved bilateral consistency report: {report_path}")
    print(f"Saved docs report: {docs_report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
