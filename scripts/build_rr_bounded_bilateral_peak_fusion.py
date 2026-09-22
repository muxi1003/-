from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from rr_quality_residual_validation import metric_dict


METHODS = [
    "adaptive_fusion_full_clip_raw",
    "adaptive_fusion_full_clip_double_suppressed",
    "adaptive_fusion_full_clip_periodic_edge",
    "adaptive_fusion_full_clip_full",
    "adaptive_fusion_fixed_window",
    "adaptive_fusion_fixed_window_double_suppressed",
    "adaptive_fusion_fixed_window_periodic_edge",
    "adaptive_fusion_fixed_window_full",
    "bilateral_quality_full_clip",
    "bilateral_quality_fixed_window",
    "bilateral_quality_fixed_window_double_suppressed",
    "bilateral_quality_fixed_window_periodic_edge",
    "bilateral_quality_fixed_window_full",
]


@dataclass(frozen=True)
class PeakFusionConfig:
    fps: float
    fixed_window_seconds: float
    window_stride_seconds: float
    peak_distance_frames: int = 5
    base_prominence: float = 0.035
    min_rr_bpm: float = 40.0
    max_rr_bpm: float = 90.0
    double_peak_interval_fraction: float = 0.55
    double_peak_relief_fraction: float = 0.10
    double_peak_minor_relief_fraction: float = 0.18
    double_peak_minor_prominence_ratio: float = 0.65
    edge_interval_min_fraction: float = 0.55
    edge_interval_max_fraction: float = 1.45
    edge_relief_fraction: float = 0.15


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_root = repo / "Dataset_new" / "72video" / "al_images"
    default_assets = default_root / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fixed, non-truth bilateral nostril RR peak-fusion ablation. "
            "Prediction uses temperature curves and frame duration only; manual RR is "
            "read only after prediction for internal scoring."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=default_root / "paper_repro_summary.csv")
    parser.add_argument("--curve-prefix", default="paper_repro")
    parser.add_argument("--output-dir", type=Path, default=default_assets)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser.parse_args()


def numeric(values: pd.Series | object) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def bool_series(values: pd.Series | object, size: int) -> np.ndarray:
    if not isinstance(values, pd.Series):
        return np.zeros(size, dtype=bool)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    ).to_numpy(dtype=bool)


def derive_config(summary: pd.DataFrame, fps: float) -> PeakFusionConfig:
    duration = numeric(summary.get("raw_duration_seconds", summary.get("duration_seconds")))
    duration = duration[np.isfinite(duration) & (duration > 0)]
    if duration.empty:
        raise ValueError("Summary lacks a positive raw_duration_seconds or duration_seconds column.")
    window_seconds = float(math.ceil(float(duration.median())))
    return PeakFusionConfig(
        fps=float(fps),
        fixed_window_seconds=window_seconds,
        window_stride_seconds=1.0,
    )


def repair(values: pd.Series) -> np.ndarray:
    return (
        numeric(values)
        .interpolate(method="linear", limit_direction="both")
        .fillna(0.5)
        .to_numpy(dtype=float)
    )


def peak_properties(values: np.ndarray, config: PeakFusionConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    peaks, properties = find_peaks(
        values,
        distance=config.peak_distance_frames,
        prominence=config.base_prominence,
        width=1,
        plateau_size=1,
    )
    return peaks.astype(int), properties


def side_quality(values: np.ndarray, missing: np.ndarray, config: PeakFusionConfig) -> dict[str, float]:
    peaks, properties = peak_properties(values, config)
    intervals = np.diff(peaks.astype(float))
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals) >= 2
        else 1.0
    )
    prominence = properties.get("prominences", np.array([], dtype=float))
    median_prominence = float(np.median(prominence)) if len(prominence) else 0.0
    amplitude = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    missing_fraction = float(np.mean(missing)) if len(missing) else 1.0
    score = median_prominence + 2.0 * amplitude - 0.5 * interval_cv - 2.0 * missing_fraction
    weight = max(1e-6, (0.25 + median_prominence + amplitude) / (1.0 + interval_cv + missing_fraction))
    return {
        "score": float(score),
        "weight": float(weight),
        "candidate_peaks": float(len(peaks)),
        "interval_cv": interval_cv,
        "median_prominence": median_prominence,
        "amplitude": amplitude,
        "missing_fraction": missing_fraction,
    }


def bilateral_curve(
    table: pd.DataFrame,
    config: PeakFusionConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    required = {"left_norm", "right_norm"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Curve table is missing {missing}")
    left = repair(table["left_norm"])
    right = repair(table["right_norm"])
    left_missing = bool_series(table.get("left_missing_raw"), len(table))
    right_missing = bool_series(table.get("right_missing_raw"), len(table))
    left_quality = side_quality(left, left_missing, config)
    right_quality = side_quality(right, right_missing, config)
    total_weight = left_quality["weight"] + right_quality["weight"]
    left_weight = left_quality["weight"] / total_weight
    right_weight = right_quality["weight"] / total_weight
    curve = left_weight * left + right_weight * right
    return curve, {
        "left_weight": float(left_weight),
        "right_weight": float(right_weight),
        "left_quality_score": left_quality["score"],
        "right_quality_score": right_quality["score"],
        "left_missing_fraction": left_quality["missing_fraction"],
        "right_missing_fraction": right_quality["missing_fraction"],
    }


def saved_adaptive_curve(table: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "fused_norm" not in table.columns:
        raise ValueError("Curve table is missing fused_norm from the existing adaptive fusion stage.")
    selected = "unknown"
    if "selected_fusion_mode" in table.columns:
        values = table["selected_fusion_mode"].dropna().astype(str)
        if not values.empty:
            selected = values.iloc[0]
    return repair(table["fused_norm"]), selected


def adaptive_peaks(
    values: np.ndarray,
    config: PeakFusionConfig,
) -> tuple[np.ndarray, float]:
    duration = len(values) / config.fps

    def run(prominence: float) -> tuple[np.ndarray, float]:
        peaks, _ = find_peaks(
            values,
            distance=config.peak_distance_frames,
            prominence=prominence,
            width=1,
            plateau_size=1,
        )
        rr = len(peaks) * 60.0 / duration if duration > 0 else math.nan
        return peaks.astype(int), float(rr)

    prominence = float(config.base_prominence)
    peaks, rr = run(prominence)
    if math.isfinite(rr) and rr < config.min_rr_bpm:
        for candidate in [0.025, 0.015, 0.010, 0.005, 0.002]:
            peaks, rr = run(candidate)
            prominence = float(candidate)
            if rr >= config.min_rr_bpm:
                break
    elif math.isfinite(rr) and rr > config.max_rr_bpm:
        for candidate in [0.050, 0.070, 0.090, 0.120, 0.160, 0.200]:
            peaks, rr = run(candidate)
            prominence = float(candidate)
            if rr <= config.max_rr_bpm:
                break
    return peaks, prominence


def suppress_double_peaks(
    peaks: np.ndarray,
    values: np.ndarray,
    config: PeakFusionConfig,
) -> tuple[np.ndarray, list[int]]:
    """Suppress only shape-supported, unusually short double peaks without reference labels."""
    active = [int(peak) for peak in np.sort(peaks)]
    removed: list[int] = []
    if len(active) < 3:
        return np.asarray(active, dtype=int), removed
    amplitude = max(float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)), 1e-6)

    while len(active) >= 3:
        gaps = np.diff(np.asarray(active, dtype=float))
        median_gap = float(np.median(gaps))
        max_gap = max(config.peak_distance_frames + 1, int(math.floor(median_gap * config.double_peak_interval_fraction)))
        prominences = peak_prominences(values, np.asarray(active, dtype=int))[0]
        median_prominence = max(float(np.median(prominences)), 1e-6)
        remove_index: int | None = None
        for index, gap in enumerate(gaps):
            if gap > max_gap:
                continue
            left_index = active[index]
            right_index = active[index + 1]
            lower_peak = min(float(values[left_index]), float(values[right_index]))
            valley = float(np.nanmin(values[left_index : right_index + 1]))
            relief = (lower_peak - valley) / amplitude
            smaller_prominence = min(float(prominences[index]), float(prominences[index + 1]))
            shallow_double = relief <= config.double_peak_relief_fraction
            weak_secondary = (
                relief <= config.double_peak_minor_relief_fraction
                and smaller_prominence <= config.double_peak_minor_prominence_ratio * median_prominence
            )
            if shallow_double or weak_secondary:
                remove_index = index if values[left_index] < values[right_index] else index + 1
                break
        if remove_index is None:
            break
        removed.append(active.pop(remove_index))
    return np.asarray(active, dtype=int), removed


def periodic_edge_completion(
    peaks: np.ndarray,
    values: np.ndarray,
    config: PeakFusionConfig,
) -> tuple[np.ndarray, list[int]]:
    """Add a boundary maximum only when its spacing agrees with the detected respiratory period."""
    if len(peaks) < 3:
        return peaks, []
    active = {int(peak) for peak in peaks}
    ordered = np.asarray(sorted(active), dtype=int)
    median_gap = float(np.median(np.diff(ordered.astype(float))))
    if not math.isfinite(median_gap) or median_gap <= 0:
        return ordered, []
    amplitude = max(float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)), 1e-6)
    added: list[int] = []

    def valid_gap(gap: int) -> bool:
        ratio = gap / median_gap
        return config.edge_interval_min_fraction <= ratio <= config.edge_interval_max_fraction

    first = int(ordered[0])
    start_end = min(first, int(math.ceil(config.edge_interval_max_fraction * median_gap)))
    if start_end > 0:
        candidate = int(np.argmax(values[:start_end]))
        gap = first - candidate
        if candidate not in active and gap >= config.peak_distance_frames and valid_gap(gap):
            trough = float(np.nanmin(values[candidate : first + 1]))
            relief = (float(values[candidate]) - trough) / amplitude
            if relief >= config.edge_relief_fraction:
                active.add(candidate)
                added.append(candidate)

    ordered = np.asarray(sorted(active), dtype=int)
    last = int(ordered[-1])
    max_gap = int(math.ceil(config.edge_interval_max_fraction * median_gap))
    end_start = last + 1
    end_stop = min(len(values), last + max_gap + 1)
    if end_start < end_stop:
        candidate = end_start + int(np.argmax(values[end_start:end_stop]))
        gap = candidate - last
        if candidate not in active and gap >= config.peak_distance_frames and valid_gap(gap):
            trough = float(np.nanmin(values[last : candidate + 1]))
            relief = (float(values[candidate]) - trough) / amplitude
            if relief >= config.edge_relief_fraction:
                active.add(candidate)
                added.append(candidate)
    return np.asarray(sorted(active), dtype=int), added


def window_starts(frames: int, window_frames: int, stride_frames: int) -> list[int]:
    if frames <= window_frames:
        return [0]
    starts = list(range(0, frames - window_frames + 1, max(1, stride_frames)))
    final_start = frames - window_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def window_quality(values: np.ndarray, config: PeakFusionConfig) -> float:
    peaks, properties = peak_properties(values, config)
    intervals = np.diff(peaks.astype(float))
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals) >= 2
        else 1.0
    )
    prominence = properties.get("prominences", np.array([], dtype=float))
    median_prominence = float(np.median(prominence)) if len(prominence) else 0.0
    amplitude = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    return float(median_prominence + 2.0 * amplitude - 0.5 * interval_cv)


def best_fixed_window(values: np.ndarray, config: PeakFusionConfig) -> tuple[np.ndarray, int, float]:
    window_frames = max(8, int(round(config.fixed_window_seconds * config.fps)))
    stride_frames = max(1, int(round(config.window_stride_seconds * config.fps)))
    best: tuple[float, int, np.ndarray] | None = None
    for start in window_starts(len(values), window_frames, stride_frames):
        segment = values[start : min(len(values), start + window_frames)]
        score = window_quality(segment, config)
        candidate = (score, -start, segment)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
            best_start = start
    if best is None:
        return values, 0, math.nan
    return best[2], best_start, float(best[0])


def method_prediction(
    values: np.ndarray,
    config: PeakFusionConfig,
    *,
    double_suppression: bool,
    edge_completion: bool,
) -> dict[str, object]:
    peaks, prominence = adaptive_peaks(values, config)
    raw_count = len(peaks)
    removed: list[int] = []
    added: list[int] = []
    if double_suppression:
        peaks, removed = suppress_double_peaks(peaks, values, config)
    if edge_completion:
        peaks, added = periodic_edge_completion(peaks, values, config)
    duration = len(values) / config.fps
    rr = len(peaks) * 60.0 / duration if duration > 0 else math.nan
    return {
        "peaks": int(len(peaks)),
        "rr_bpm": float(rr),
        "raw_peak_count": int(raw_count),
        "selected_prominence": float(prominence),
        "double_peaks_removed": int(len(removed)),
        "double_peak_frames": ";".join(str(value) for value in removed),
        "edge_peaks_added": int(len(added)),
        "edge_peak_frames": ";".join(str(value) for value in added),
    }


def predict_video(curve_path: Path, config: PeakFusionConfig) -> dict[str, object]:
    table = pd.read_csv(curve_path)
    curve, quality = bilateral_curve(table, config)
    adaptive_curve, selected_fusion_mode = saved_adaptive_curve(table)
    adaptive_fixed, adaptive_start, adaptive_fixed_quality = best_fixed_window(adaptive_curve, config)
    fixed, start, fixed_quality = best_fixed_window(curve, config)
    adaptive_full_raw = method_prediction(
        adaptive_curve, config, double_suppression=False, edge_completion=False
    )
    adaptive_full_double = method_prediction(
        adaptive_curve, config, double_suppression=True, edge_completion=False
    )
    adaptive_full_edge = method_prediction(
        adaptive_curve, config, double_suppression=False, edge_completion=True
    )
    adaptive_full = method_prediction(
        adaptive_curve, config, double_suppression=True, edge_completion=True
    )
    adaptive_window = method_prediction(
        adaptive_fixed, config, double_suppression=False, edge_completion=False
    )
    adaptive_double = method_prediction(
        adaptive_fixed, config, double_suppression=True, edge_completion=False
    )
    adaptive_edge = method_prediction(
        adaptive_fixed, config, double_suppression=False, edge_completion=True
    )
    adaptive_fused = method_prediction(
        adaptive_fixed, config, double_suppression=True, edge_completion=True
    )
    full = method_prediction(curve, config, double_suppression=False, edge_completion=False)
    window = method_prediction(fixed, config, double_suppression=False, edge_completion=False)
    double = method_prediction(fixed, config, double_suppression=True, edge_completion=False)
    edge = method_prediction(fixed, config, double_suppression=False, edge_completion=True)
    fused = method_prediction(fixed, config, double_suppression=True, edge_completion=True)
    row: dict[str, object] = {
        "raw_curve_duration_seconds": len(curve) / config.fps,
        "fixed_window_duration_seconds": len(fixed) / config.fps,
        "fixed_window_start_frame": int(start),
        "fixed_window_quality_score": fixed_quality,
        "adaptive_selected_fusion_mode": selected_fusion_mode,
        "adaptive_fixed_window_duration_seconds": len(adaptive_fixed) / config.fps,
        "adaptive_fixed_window_start_frame": int(adaptive_start),
        "adaptive_fixed_window_quality_score": adaptive_fixed_quality,
        "prediction_origin": "fixed_signal_formula_without_manual_rr_read",
        **quality,
    }
    for method, prediction in [
        ("adaptive_fusion_full_clip_raw", adaptive_full_raw),
        ("adaptive_fusion_full_clip_double_suppressed", adaptive_full_double),
        ("adaptive_fusion_full_clip_periodic_edge", adaptive_full_edge),
        ("adaptive_fusion_full_clip_full", adaptive_full),
        ("adaptive_fusion_fixed_window", adaptive_window),
        ("adaptive_fusion_fixed_window_double_suppressed", adaptive_double),
        ("adaptive_fusion_fixed_window_periodic_edge", adaptive_edge),
        ("adaptive_fusion_fixed_window_full", adaptive_fused),
        ("bilateral_quality_full_clip", full),
        ("bilateral_quality_fixed_window", window),
        ("bilateral_quality_fixed_window_double_suppressed", double),
        ("bilateral_quality_fixed_window_periodic_edge", edge),
        ("bilateral_quality_fixed_window_full", fused),
    ]:
        for key, value in prediction.items():
            row[f"{method}_{key}"] = value
    return row


def baseline_predictions(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["video_id", "truth_rr", "truth_count", "rr_bpm", "peaks"]
    missing = sorted(set(columns) - set(summary.columns))
    if missing:
        raise ValueError(f"Summary missing baseline/evaluation columns: {missing}")
    return summary[columns].copy()


def score_methods(predictions: pd.DataFrame) -> pd.DataFrame:
    truth_rr = numeric(predictions["truth_rr"]).to_numpy(float)
    truth_count = numeric(predictions["truth_count"]).to_numpy(float)
    rows = [
        metric_dict(
            "current_frozen_baseline",
            truth_rr,
            numeric(predictions["rr_bpm"]).to_numpy(float),
            truth_count,
            numeric(predictions["peaks"]).to_numpy(float),
            evaluation_note="existing_frozen_summary_for_ablation_comparator",
        )
    ]
    for method in METHODS:
        rows.append(
            metric_dict(
                method,
                truth_rr,
                numeric(predictions[f"{method}_rr_bpm"]).to_numpy(float),
                truth_count,
                numeric(predictions[f"{method}_peaks"]).to_numpy(float),
                evaluation_note="fixed_formula_non_truth_signal_ablation",
            )
        )
    return pd.DataFrame(rows)


def bootstrap_delta(
    predictions: pd.DataFrame,
    candidate: str,
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    truth_rr = numeric(predictions["truth_rr"]).to_numpy(float)
    truth_count = numeric(predictions["truth_count"]).to_numpy(float)
    baseline_rr = numeric(predictions["rr_bpm"]).to_numpy(float)
    baseline_count = numeric(predictions["peaks"]).to_numpy(float)
    candidate_rr = numeric(predictions[f"{candidate}_rr_bpm"]).to_numpy(float)
    candidate_count = numeric(predictions[f"{candidate}_peaks"]).to_numpy(float)
    valid = np.isfinite(truth_rr) & np.isfinite(truth_count) & np.isfinite(baseline_rr) & np.isfinite(candidate_rr)
    indices = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"rr_r2": [], "rr_mae": [], "rr_rmse": [], "exact_count": []}
    for _ in range(resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        baseline = metric_dict(
            "baseline", truth_rr[sample], baseline_rr[sample], truth_count[sample], baseline_count[sample], evaluation_note="bootstrap"
        )
        candidate_row = metric_dict(
            "candidate", truth_rr[sample], candidate_rr[sample], truth_count[sample], candidate_count[sample], evaluation_note="bootstrap"
        )
        for metric in values:
            values[metric].append(float(candidate_row[metric]) - float(baseline[metric]))
    baseline = metric_dict("baseline", truth_rr[indices], baseline_rr[indices], truth_count[indices], baseline_count[indices], evaluation_note="point")
    candidate_row = metric_dict("candidate", truth_rr[indices], candidate_rr[indices], truth_count[indices], candidate_count[indices], evaluation_note="point")
    rows: list[dict[str, object]] = []
    for metric, draws in values.items():
        finite = np.asarray(draws, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "candidate": candidate,
                "baseline": "current_frozen_baseline",
                "metric": metric,
                "estimate": float(candidate_row[metric]) - float(baseline[metric]),
                "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else math.nan,
                "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                "bootstrap_resamples": int(resamples),
            }
        )
    return rows


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    if table.empty:
        return "_No rows generated._"
    rows = table[columns].fillna("").astype(str)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows.to_numpy():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    config: PeakFusionConfig,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    full = metrics[metrics["label"].eq("adaptive_fusion_fixed_window_full")].iloc[0]
    delta = bootstrap[
        (bootstrap["candidate"].eq("adaptive_fusion_fixed_window_full"))
        & (bootstrap["metric"].eq("rr_r2"))
    ]
    delta_text = "not computed"
    if not delta.empty:
        row = delta.iloc[0]
        delta_text = (
            f"delta R2={float(row['estimate']):.6f} "
            f"(95% CI {float(row['ci_low']):.6f} to {float(row['ci_high']):.6f})"
        )
    report = f"""# Bounded Bilateral Peak-Fusion Ablation

Status: `exploratory_fixed_non_truth_internal_ablation`

## Prediction Boundary

The prediction functions read only the saved left/right nostril temperature curves,
frame count, and fixed physiological constants. The window length is
`ceil(median(raw_duration_seconds))={config.fixed_window_seconds:.0f}` seconds.
Manual breath count and RR are merged only after all candidate predictions are written.
The best-quality fixed window is an offline analysis mode and must not be called
real-time.

## Mechanisms

- The current adaptive per-video nostril quality fusion is retained as the primary
  fusion comparator. A separate continuous bilateral-weighting ablation is kept as
  a negative-control alternative.
- Candidate windows use the fixed duration and a one-second stride, selecting only by
  signal quality.
- Double-peak suppression removes only unusually short, shallow or weak-secondary
  adjacent maxima.
- Edge completion adds a boundary maximum only when its gap matches the detected
  median respiratory interval and its trough relief is sufficient.

## Metrics

{markdown_table(metrics, ['label', 'videos', 'rr_r2', 'rr_mae', 'rr_rmse', 'exact_count', 'within_one_count', 'abs_count_error_ge2'])}

For the full fixed rule versus the current frozen baseline: {delta_text}.
Do not replace the manuscript main method from this table alone: the ablation is
internally evaluated on the same 73 videos and needs a frozen independent test before
promotion.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, dtype={"video_id": str})
    config = derive_config(summary, args.fps)
    baseline = baseline_predictions(summary)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(baseline.itertuples(index=False), start=1):
        video_id = str(source.video_id)
        curve_path = input_root / video_id / f"{args.curve_prefix}_curve.csv"
        if not curve_path.exists():
            raise FileNotFoundError(f"Missing curve for {video_id}: {curve_path}")
        print(f"[{index}/{len(baseline)}] {video_id}")
        rows.append({"video_id": video_id, **predict_video(curve_path, config)})
    predictions = baseline.merge(pd.DataFrame(rows), on="video_id", how="inner", validate="one_to_one")
    metrics = score_methods(predictions)
    bootstrap_rows: list[dict[str, object]] = []
    for index, method in enumerate(METHODS):
        bootstrap_rows.extend(
            bootstrap_delta(
                predictions,
                method,
                args.bootstrap_resamples,
                args.bootstrap_seed + index,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    config_path = output_dir / "paper_bounded_bilateral_peak_fusion_config.csv"
    predictions_path = output_dir / "paper_bounded_bilateral_peak_fusion_predictions.csv"
    metrics_path = output_dir / "paper_bounded_bilateral_peak_fusion_metrics.csv"
    bootstrap_path = output_dir / "paper_bounded_bilateral_peak_fusion_bootstrap_ci.csv"
    report_path = output_dir / "paper_bounded_bilateral_peak_fusion_report.md"
    pd.DataFrame([asdict(config)]).to_csv(config_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    write_report(report_path, config, metrics, bootstrap)
    print(f"Saved config: {config_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved bootstrap CIs: {bootstrap_path}")
    print(f"Saved report: {report_path}")
    print(metrics[["label", "rr_r2", "rr_mae", "rr_rmse", "exact_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
