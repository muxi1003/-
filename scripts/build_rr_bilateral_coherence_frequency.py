from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_rr_calibration_free_thermal_index import SIGNALS, extract_video_signals
from build_rr_duration_normalized_windowed_innovation import (
    WindowConfig,
    derive_config,
    external_truth,
    internal_truth,
    metric_dict,
    window_starts,
)
from paper_repro_rr import normalize_series, repair_series


METHOD_ID = "fixed_bilateral_coherence_frequency"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = (
        repo
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Estimate RR from fixed bilateral nostril spectral coherence across relative "
            "thermal-color signals. Frequency bounds and windowing come only from the "
            "internal cohort; external references are optional scoring inputs."
        )
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "al_images",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument(
        "--external-reference-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
        help="Optional single-annotator reference used only to score the post-hoc probe.",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--overwrite-signals", action="store_true")
    return parser.parse_args()


def numeric(values: pd.Series | object) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def source_curves(table: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Repair and normalize each nostril independently so color calibration cancels."""
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for signal in SIGNALS:
        left = normalize_series(repair_series(table[f"left_{signal}"], True))
        right = normalize_series(repair_series(table[f"right_{signal}"], True))
        left_values = left.interpolate(limit_direction="both").fillna(0.5).to_numpy(float)
        right_values = right.interpolate(limit_direction="both").fillna(0.5).to_numpy(float)
        curves[signal] = (left_values, right_values)
    return curves


def detrended(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.zeros(len(values), dtype=float)
    index = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(index, values, deg=1)
    centered = values - (slope * index + intercept)
    return centered * np.hanning(len(centered))


def coherence_score(
    left: np.ndarray,
    right: np.ndarray,
    min_cycles: int,
    max_cycles: int,
) -> np.ndarray:
    """Return normalized, phase-aware bilateral spectral support for each cycle count."""
    left_fft = np.fft.rfft(detrended(left))
    right_fft = np.fft.rfft(detrended(right))
    left_power = np.abs(left_fft) ** 2
    right_power = np.abs(right_fft) ** 2
    usable = slice(min_cycles, max_cycles + 1)
    left_total = float(np.sum(left_power[usable]))
    right_total = float(np.sum(right_power[usable]))
    if left_total <= 1e-12 or right_total <= 1e-12:
        return np.zeros(max_cycles - min_cycles + 1, dtype=float)
    relative_power = np.sqrt(
        (left_power[usable] / left_total) * (right_power[usable] / right_total)
    )
    cross_phase = np.angle(left_fft[usable] * np.conj(right_fft[usable]))
    phase_agreement = 0.5 * (1.0 + np.cos(cross_phase))
    return relative_power * phase_agreement


def estimate_window(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    start: int,
    end: int,
    config: WindowConfig,
) -> dict[str, float]:
    frames = int(end - start)
    duration = frames / config.fps
    min_cycles = max(1, int(math.ceil(config.min_rr_bpm * duration / 60.0)))
    max_cycles = min(
        frames // 2,
        int(math.floor(config.max_rr_bpm * duration / 60.0)),
    )
    if frames < 8 or max_cycles < min_cycles:
        return {
            "window_rr_bpm": math.nan,
            "window_cycle_count": math.nan,
            "window_coherence_score": math.nan,
            "window_frequency_margin": math.nan,
            "window_signal_support": math.nan,
        }

    per_signal: list[np.ndarray] = []
    for left, right in curves.values():
        score = coherence_score(left[start:end], right[start:end], min_cycles, max_cycles)
        if np.isfinite(score).any():
            per_signal.append(score)
    if not per_signal:
        return {
            "window_rr_bpm": math.nan,
            "window_cycle_count": math.nan,
            "window_coherence_score": math.nan,
            "window_frequency_margin": math.nan,
            "window_signal_support": 0.0,
        }

    consensus = np.nanmedian(np.vstack(per_signal), axis=0)
    best_index = int(np.nanargmax(consensus))
    cycles = min_cycles + best_index
    best_score = float(consensus[best_index])
    ordered = np.sort(consensus[np.isfinite(consensus)])
    runner_up = float(ordered[-2]) if len(ordered) >= 2 else 0.0
    return {
        "window_rr_bpm": float(cycles * 60.0 / duration),
        "window_cycle_count": float(cycles),
        "window_coherence_score": best_score,
        "window_frequency_margin": float(best_score - runner_up),
        "window_signal_support": float(len(per_signal)),
    }


def predict_video(
    table: pd.DataFrame,
    baseline_rr: float,
    config: WindowConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    curves = source_curves(table)
    frames = len(table)
    duration = frames / config.fps
    window_frames = max(8, int(round(config.window_seconds * config.fps)))
    stride_frames = max(1, int(round(config.stride_seconds * config.fps)))
    if frames < max(8, int(round(0.7 * window_frames))):
        starts = [0]
    else:
        starts = window_starts(frames, window_frames, stride_frames)

    windows: list[dict[str, object]] = []
    for window_index, start in enumerate(starts):
        end = min(frames, start + window_frames)
        if end - start < max(8, int(round(0.7 * window_frames))):
            continue
        row = estimate_window(curves, start, end, config)
        windows.append(
            {
                "window_index": window_index,
                "window_start_frame": start,
                "window_end_frame_exclusive": end,
                "window_duration_seconds": (end - start) / config.fps,
                **row,
            }
        )

    details = pd.DataFrame(windows)
    valid = details["window_rr_bpm"].notna() if not details.empty else pd.Series([], dtype=bool)
    if details.empty or not bool(valid.any()):
        candidate_rr = math.nan
        confidence = math.nan
        margin = math.nan
        support = 0.0
    else:
        active = details.loc[valid]
        weights = np.maximum(numeric(active["window_coherence_score"]).to_numpy(float), 1e-9)
        candidate_rr = float(np.average(numeric(active["window_rr_bpm"]).to_numpy(float), weights=weights))
        confidence = float(np.mean(numeric(active["window_coherence_score"])))
        margin = float(np.mean(numeric(active["window_frequency_margin"])))
        support = float(np.mean(numeric(active["window_signal_support"])))

    activate = bool(
        np.isfinite(candidate_rr) and duration >= float(config.long_clip_threshold_seconds)
    )
    return (
        {
            "raw_duration_seconds": duration,
            "baseline_rr_bpm": baseline_rr,
            "bilateral_coherence_rr_bpm": candidate_rr,
            "duration_gated_rr_bpm": candidate_rr if activate else baseline_rr,
            "duration_gate_activated": activate,
            "windows": int(len(details)),
            "mean_coherence_score": confidence,
            "mean_frequency_margin": margin,
            "mean_signal_support": support,
        },
        windows,
    )


def cohort_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: WindowConfig,
    radius: int,
    overwrite_signals: bool,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    for index, summary_row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(summary_row.video_id)
        print(f"[{cohort} {index}/{len(summary)}] {video_id}")
        table = extract_video_signals(root / video_id, prefix, radius, overwrite_signals)
        baseline = float(numeric(pd.Series([getattr(summary_row, "rr_bpm", math.nan)])).iloc[0])
        prediction, details = predict_video(table, baseline, config)
        rows.append({"cohort": cohort, "video_id": video_id, **prediction})
        for detail in details:
            windows.append({"cohort": cohort, "video_id": video_id, **detail})
    return pd.DataFrame(rows), pd.DataFrame(windows)


def score_predictions(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    cohort: str,
    note: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = predictions.merge(truth, on="video_id", how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for method, column in [
        ("frozen_whole_clip_baseline", "baseline_rr_bpm"),
        (METHOD_ID, "bilateral_coherence_rr_bpm"),
        (f"duration_gated_{METHOD_ID}", "duration_gated_rr_bpm"),
    ]:
        rows.append(
            metric_dict(
                cohort,
                method,
                numeric(data["truth_rr"]).to_numpy(float),
                numeric(data[column]).to_numpy(float),
                numeric(data["truth_count"]).to_numpy(float),
                numeric(data["truth_duration_seconds"]).to_numpy(float),
                note,
            )
        )
    return pd.DataFrame(rows), data


def write_report(
    path: Path,
    config: WindowConfig,
    metrics: pd.DataFrame,
    external_scored: bool,
) -> None:
    lines = [
        "# Bilateral Coherence Frequency RR Probe",
        "",
        "Status: `fixed_formula_posthoc_external_development_probe`",
        "",
        "The estimator independently normalizes left/right nostril color traces, then "
        "uses the median phase-aware spectral support across all predefined relative "
        "color signals to choose a respiratory cycle count. It has no learned weights, "
        "and it does not use an external RR label at prediction time.",
        "",
        "## Fixed Configuration Origin",
        "",
        f"- Window length: `{config.window_seconds:.3f}` s, from the internal duration distribution.",
        f"- Frequency range: `{config.min_rr_bpm:.1f}` to `{config.max_rr_bpm:.1f}` bpm, with the lower bound derived from internal reference RR only.",
        f"- Long-clip activation: `{config.long_clip_threshold_seconds:.3f}` s, from internal duration only.",
        f"- Signals: `{'; '.join(SIGNALS)}`; aggregation: fixed cross-signal median.",
        "",
        "## Metrics",
        "",
        "| cohort | method | R2 | MAE (bpm) | RMSE (bpm) | valid videos |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.method} | {row.rr_r2:.6f} | {row.rr_mae:.6f} | "
            f"{row.rr_rmse:.6f} | {row.rr_valid_videos} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The external rows are a single-annotator, already-inspected development analysis. "
            "They are not confirmatory external validation and cannot replace the frozen P2g "
            "primary method. A candidate selected from this work must be frozen and scored "
            "against independent blinded A/B consensus counts or an untouched external cohort.",
        ]
    )
    if not external_scored:
        lines.append(
            "No external reference CSV was available; this run reports only the internal diagnostic."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    internal_root = args.internal_root.resolve()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    internal_summary["video_id"] = internal_summary["video_id"].astype(str)
    config = derive_config(internal_summary, args.fps)
    internal_predictions, internal_windows = cohort_predictions(
        internal_root,
        args.internal_prefix,
        internal_summary,
        config,
        args.radius,
        args.overwrite_signals,
        "internal_development",
    )
    internal_metrics, internal_scored = score_predictions(
        internal_predictions,
        internal_truth(internal_summary),
        "internal_development",
        "fixed_formula_frequency_bounds_and_windowing_derived_from_internal_cohort",
    )

    external_summary = pd.read_csv(external_root / f"{args.external_prefix}_summary.csv")
    external_summary["video_id"] = external_summary["video_id"].astype(str)
    external_predictions, external_windows = cohort_predictions(
        external_root,
        args.external_prefix,
        external_summary,
        config,
        args.radius,
        args.overwrite_signals,
        "external_provisional",
    )
    external_metrics = pd.DataFrame()
    external_scored = pd.DataFrame()
    reference_path = args.external_reference_csv.resolve()
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        external_metrics, external_scored = score_predictions(
            external_predictions,
            external_truth(reference),
            "external_provisional",
            "single_annotator_posthoc_development_score_not_confirmatory_external_validation",
        )

    metrics = pd.concat([internal_metrics, external_metrics], ignore_index=True)
    prediction_path = output_dir / "paper_bilateral_coherence_frequency_predictions.csv"
    window_path = output_dir / "paper_bilateral_coherence_frequency_windows.csv"
    metrics_path = output_dir / "paper_bilateral_coherence_frequency_metrics.csv"
    config_path = output_dir / "paper_bilateral_coherence_frequency_config.csv"
    report_path = output_dir / "paper_bilateral_coherence_frequency_report.md"

    combined = pd.concat([internal_predictions, external_predictions], ignore_index=True)
    if not internal_scored.empty:
        scored_columns = ["video_id", "truth_rr", "truth_count", "truth_duration_seconds"]
        combined = combined.merge(internal_scored[scored_columns], on="video_id", how="left")
    if not external_scored.empty:
        scored_columns = ["video_id", "truth_rr", "truth_count", "truth_duration_seconds"]
        external_truth_rows = external_scored[scored_columns].copy()
        external_truth_rows["cohort"] = "external_provisional"
        combined = combined.merge(
            external_truth_rows,
            on=["cohort", "video_id"],
            how="left",
            suffixes=("", "_external"),
        )
        for column in ["truth_rr", "truth_count", "truth_duration_seconds"]:
            external_column = f"{column}_external"
            if external_column in combined.columns:
                combined[column] = combined[column].combine_first(combined[external_column])
                combined = combined.drop(columns=external_column)
    combined.to_csv(prediction_path, index=False)
    pd.concat([internal_windows, external_windows], ignore_index=True).to_csv(window_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame([asdict(config)]).to_csv(config_path, index=False)
    write_report(report_path, config, metrics, not external_metrics.empty)

    print(f"Saved predictions: {prediction_path}")
    print(f"Saved windows: {window_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
