from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class WindowConfig:
    fps: float
    window_seconds: float
    stride_seconds: float
    long_clip_threshold_seconds: float
    base_prominence: float
    peak_distance_frames: int
    min_rr_bpm: float
    max_rr_bpm: float
    session_shrinkage: float
    session_rolling_clips: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build a duration-normalized, quality-weighted windowed RR estimator "
            "with fixed sequential fusion for long external clips."
        )
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "al_images",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument(
        "--external-reference-audit",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
    )
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def derive_config(internal_summary: pd.DataFrame, fps: float) -> WindowConfig:
    duration = numeric(internal_summary.get("raw_duration_seconds", internal_summary["duration_seconds"]))
    duration = duration[np.isfinite(duration) & (duration > 0)]
    truth_rr = numeric(internal_summary["truth_rr"])
    truth_rr = truth_rr[np.isfinite(truth_rr) & (truth_rr > 0)]
    if duration.empty or truth_rr.empty:
        raise ValueError("Internal summary lacks valid duration or truth RR values")
    window_seconds = float(math.ceil(float(duration.median())))
    long_clip_threshold = max(
        window_seconds * 1.5,
        float(math.ceil(float(duration.max()))),
    )
    min_rr = max(10.0, math.floor(float(truth_rr.min()) / 5.0) * 5.0 - 5.0)
    return WindowConfig(
        fps=float(fps),
        window_seconds=window_seconds,
        stride_seconds=window_seconds / 2.0,
        long_clip_threshold_seconds=long_clip_threshold,
        base_prominence=0.035,
        peak_distance_frames=5,
        min_rr_bpm=min_rr,
        max_rr_bpm=90.0,
        session_shrinkage=0.5,
        session_rolling_clips=3,
    )


def interpolate_curve(values: pd.Series) -> np.ndarray:
    return (
        numeric(values)
        .interpolate(method="linear", limit_direction="both")
        .fillna(0.5)
        .to_numpy(dtype=float)
    )


def detect_window(
    curve: np.ndarray,
    missing_fraction: float,
    config: WindowConfig,
) -> dict[str, object]:
    values = np.asarray(curve, dtype=float)
    duration = len(values) / config.fps

    def run(prominence: float) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
        peaks, properties = find_peaks(
            values,
            distance=config.peak_distance_frames,
            prominence=float(prominence),
            width=1,
            plateau_size=1,
        )
        rr = len(peaks) * 60.0 / duration if duration > 0 else math.nan
        return peaks, properties, float(rr)

    selected_prominence = config.base_prominence
    peaks, properties, rr_bpm = run(selected_prominence)
    if rr_bpm < config.min_rr_bpm:
        for prominence in [0.025, 0.015, 0.010, 0.005, 0.002]:
            peaks, properties, rr_bpm = run(prominence)
            selected_prominence = prominence
            if rr_bpm >= config.min_rr_bpm:
                break
    elif rr_bpm > config.max_rr_bpm:
        for prominence in [0.05, 0.07, 0.09, 0.12, 0.16, 0.20]:
            peaks, properties, rr_bpm = run(prominence)
            selected_prominence = prominence
            if rr_bpm <= config.max_rr_bpm:
                break

    intervals = np.diff(peaks.astype(float))
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals) >= 2
        else 1.0
    )
    amplitude = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    prominences = properties.get("prominences", np.array([], dtype=float))
    median_prominence = float(np.median(prominences)) if len(prominences) else 0.0
    quality_score = (
        median_prominence
        + 2.0 * amplitude
        - 0.5 * interval_cv
        - 2.0 * float(missing_fraction)
    )
    return {
        "window_rr_bpm": rr_bpm,
        "window_peak_count": len(peaks),
        "window_duration_seconds": duration,
        "window_quality_score": quality_score,
        "window_interval_cv": interval_cv,
        "window_amplitude": amplitude,
        "window_median_prominence": median_prominence,
        "window_missing_fraction": float(missing_fraction),
        "window_selected_prominence": selected_prominence,
    }


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, max(1, stride)))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_video(
    curve_path: Path,
    baseline_rr: float,
    config: WindowConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    curve_table = pd.read_csv(curve_path)
    if "smoothed_norm" not in curve_table.columns:
        raise ValueError(f"Missing smoothed_norm: {curve_path}")
    available = numeric(curve_table["smoothed_norm"]).notna()
    curve = interpolate_curve(curve_table.loc[available, "smoothed_norm"])
    duration = len(curve) / config.fps
    if duration < config.long_clip_threshold_seconds:
        return (
            {
                "raw_curve_duration_seconds": duration,
                "windowed_rr_bpm": baseline_rr,
                "windowed_windows": 0,
                "windowed_activation": False,
                "windowed_reason": "short_clip_keep_frozen_baseline",
                "windowed_quality_mean": math.nan,
                "windowed_rr_std": math.nan,
            },
            [],
        )

    window_frames = max(8, int(round(config.window_seconds * config.fps)))
    stride_frames = max(1, int(round(config.stride_seconds * config.fps)))
    starts = window_starts(len(curve), window_frames, stride_frames)
    detail_rows: list[dict[str, object]] = []
    missing_both = (
        curve_table.get("left_missing_raw", pd.Series(False, index=curve_table.index)).astype(bool)
        & curve_table.get("right_missing_raw", pd.Series(False, index=curve_table.index)).astype(bool)
    )
    available_indices = np.flatnonzero(available.to_numpy())
    for window_index, start in enumerate(starts):
        end = min(len(curve), start + window_frames)
        segment = curve[start:end]
        if len(segment) < max(8, int(round(0.7 * window_frames))):
            continue
        source_indices = available_indices[start:end]
        missing_fraction = (
            float(missing_both.iloc[source_indices].mean()) if len(source_indices) else 1.0
        )
        detail = detect_window(segment, missing_fraction, config)
        detail_rows.append(
            {
                "window_index": window_index,
                "window_start_frame": start,
                "window_end_frame_exclusive": end,
                **detail,
            }
        )
    if not detail_rows:
        return (
            {
                "raw_curve_duration_seconds": duration,
                "windowed_rr_bpm": baseline_rr,
                "windowed_windows": 0,
                "windowed_activation": False,
                "windowed_reason": "no_valid_windows_keep_frozen_baseline",
                "windowed_quality_mean": math.nan,
                "windowed_rr_std": math.nan,
            },
            [],
        )
    details = pd.DataFrame(detail_rows)
    weights = np.exp(
        np.clip(numeric(details["window_quality_score"]).to_numpy(dtype=float), -3.0, 3.0)
    )
    rr_values = numeric(details["window_rr_bpm"]).to_numpy(dtype=float)
    windowed_rr = float(np.average(rr_values, weights=weights))
    return (
        {
            "raw_curve_duration_seconds": duration,
            "windowed_rr_bpm": windowed_rr,
            "windowed_windows": len(details),
            "windowed_activation": True,
            "windowed_reason": "duration_normalized_quality_weighted_windows",
            "windowed_quality_mean": float(np.mean(numeric(details["window_quality_score"]))),
            "windowed_rr_std": float(np.std(rr_values)),
        },
        detail_rows,
    )


def clip_order(video_id: object) -> int:
    match = re.search(r"_clip(\d+)$", str(video_id))
    return int(match.group(1)) if match else 0


def predict_cohort(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    session_map: pd.DataFrame,
    config: WindowConfig,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    session_lookup = dict(
        zip(session_map["video_id"].astype(str), session_map["source_session_id"].astype(str))
    )
    for _, row in summary.iterrows():
        video_id = str(row["video_id"])
        curve_path = root / video_id / f"{prefix}_curve.csv"
        prediction, detail_rows = predict_video(
            curve_path,
            baseline_rr=float(row["rr_bpm"]),
            config=config,
        )
        predictions.append(
            {
                "cohort": cohort,
                "video_id": video_id,
                "source_session_id": session_lookup.get(video_id, video_id),
                "clip_order": clip_order(video_id),
                "baseline_rr_bpm": float(row["rr_bpm"]),
                **prediction,
            }
        )
        windows.extend(
            {"cohort": cohort, "video_id": video_id, **detail} for detail in detail_rows
        )
    output = pd.DataFrame(predictions).sort_values(
        ["source_session_id", "clip_order", "video_id"]
    )
    centered = output.groupby("source_session_id", sort=False)["windowed_rr_bpm"].transform(
        lambda values: values.rolling(
            config.session_rolling_clips,
            center=True,
            min_periods=1,
        ).median()
    )
    causal = output.groupby("source_session_id", sort=False)["windowed_rr_bpm"].transform(
        lambda values: values.rolling(
            config.session_rolling_clips,
            center=False,
            min_periods=1,
        ).median()
    )
    active = output["windowed_activation"].astype(bool)
    output["session_centered_median_rr_bpm"] = centered
    output["session_causal_median_rr_bpm"] = causal
    output["windowed_offline_context_rr_bpm"] = output["baseline_rr_bpm"]
    output.loc[active, "windowed_offline_context_rr_bpm"] = (
        config.session_shrinkage * output.loc[active, "windowed_rr_bpm"]
        + (1.0 - config.session_shrinkage)
        * output.loc[active, "session_centered_median_rr_bpm"]
    )
    output["windowed_causal_rr_bpm"] = output["baseline_rr_bpm"]
    output.loc[active, "windowed_causal_rr_bpm"] = (
        config.session_shrinkage * output.loc[active, "windowed_rr_bpm"]
        + (1.0 - config.session_shrinkage)
        * output.loc[active, "session_causal_median_rr_bpm"]
    )
    # Compatibility column now has causal semantics; offline estimates have an explicit name.
    output["windowed_sequential_rr_bpm"] = output["windowed_causal_rr_bpm"]
    return output.reset_index(drop=True), pd.DataFrame(windows)


def metric_dict(
    cohort: str,
    method: str,
    truth_rr: np.ndarray,
    predicted_rr: np.ndarray,
    truth_count: np.ndarray,
    duration_seconds: np.ndarray,
    note: str,
) -> dict[str, object]:
    rr_valid = np.isfinite(truth_rr) & np.isfinite(predicted_rr)
    truth = truth_rr[rr_valid]
    predicted = predicted_rr[rr_valid]
    error = predicted - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    predicted_count = np.rint(predicted_rr * duration_seconds / 60.0)
    count_valid = np.isfinite(truth_count) & np.isfinite(predicted_count)
    count_error = np.abs(predicted_count[count_valid] - truth_count[count_valid])
    return {
        "cohort": cohort,
        "method": method,
        "videos": int(len(truth_rr)),
        "rr_valid_videos": int(rr_valid.sum()),
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_pearson_r2": (
            float(np.corrcoef(truth, predicted)[0, 1] ** 2)
            if len(truth) >= 2 and np.std(truth) > 0 and np.std(predicted) > 0
            else math.nan
        ),
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
        "count_valid_videos": int(count_valid.sum()),
        "count_mae": float(np.mean(count_error)),
        "exact_count": int(np.isclose(count_error, 0).sum()),
        "within_one_count": int((count_error <= 1 + 1e-9).sum()),
        "abs_count_error_ge2": int((count_error >= 2 - 1e-9).sum()),
        "evaluation_note": note,
    }


def evaluate(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    cohort: str,
    note: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = predictions.merge(truth, on="video_id", how="inner", validate="one_to_one")
    truth_rr = numeric(data["truth_rr"]).to_numpy(dtype=float)
    truth_count = numeric(data["truth_count"]).to_numpy(dtype=float)
    duration = numeric(data["truth_duration_seconds"]).to_numpy(dtype=float)
    rows = []
    for method, column in [
        ("frozen_whole_clip_baseline", "baseline_rr_bpm"),
        ("duration_normalized_windowed", "windowed_rr_bpm"),
        ("duration_normalized_windowed_offline_context", "windowed_offline_context_rr_bpm"),
        ("duration_normalized_windowed_causal", "windowed_causal_rr_bpm"),
        ("duration_normalized_windowed_sequential", "windowed_sequential_rr_bpm"),
    ]:
        rows.append(
            metric_dict(
                cohort,
                method,
                truth_rr,
                numeric(data[column]).to_numpy(dtype=float),
                truth_count,
                duration,
                note,
            )
        )
    return pd.DataFrame(rows), data


def rr_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    y = truth[valid]
    p = prediction[valid]
    error = p - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
    }


def cluster_bootstrap_improvement(
    external_scored: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    data = external_scored.reset_index(drop=True)
    groups = data["source_session_id"].astype(str).unique()
    indices_by_group = {
        group: np.flatnonzero(data["source_session_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    truth = numeric(data["truth_rr"]).to_numpy(dtype=float)
    baseline = numeric(data["baseline_rr_bpm"]).to_numpy(dtype=float)
    baseline_point = rr_metrics(truth, baseline)
    rng = np.random.default_rng(seed)
    candidates = {
        "duration_normalized_offline_context_minus_frozen_baseline": numeric(
            data["windowed_offline_context_rr_bpm"]
        ).to_numpy(dtype=float),
        "duration_normalized_causal_minus_frozen_baseline": numeric(
            data["windowed_causal_rr_bpm"]
        ).to_numpy(dtype=float),
    }
    samples = {
        comparison: {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
        for comparison in candidates
    }
    for _ in range(int(resamples)):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        base_metrics = rr_metrics(truth[indices], baseline[indices])
        for comparison, candidate in candidates.items():
            candidate_metrics = rr_metrics(truth[indices], candidate[indices])
            for metric in samples[comparison]:
                samples[comparison][metric].append(
                    candidate_metrics[metric] - base_metrics[metric]
                )
    rows: list[dict[str, object]] = []
    for comparison, candidate in candidates.items():
        candidate_point = rr_metrics(truth, candidate)
        for metric, values in samples[comparison].items():
            rows.append(
                {
                    "comparison": comparison,
                    "cluster_unit": "source_session_id",
                    "clusters": len(groups),
                    "videos": len(data),
                    "bootstrap_resamples": int(resamples),
                    "metric": metric,
                    "baseline_point": baseline_point[metric],
                    "candidate_point": candidate_point[metric],
                    "delta_candidate_minus_baseline": (
                        candidate_point[metric] - baseline_point[metric]
                    ),
                    "delta_ci_2_5": float(np.percentile(values, 2.5)),
                    "delta_ci_97_5": float(np.percentile(values, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def internal_truth(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": summary["video_id"].astype(str),
            "truth_rr": numeric(summary["truth_rr"]),
            "truth_count": numeric(summary["truth_count"]),
            "truth_duration_seconds": numeric(summary["duration_seconds"]),
        }
    )


def external_truth(reference: pd.DataFrame) -> pd.DataFrame:
    active = reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    data = reference.loc[active].copy()
    return pd.DataFrame(
        {
            "video_id": data["external_video_id"].astype(str),
            "truth_rr": numeric(data["manual_rr_bpm_point"]),
            "truth_count": numeric(data["manual_breath_count"]),
            "truth_duration_seconds": numeric(data["manual_duration_seconds"]),
        }
    )


def write_report(
    path: Path,
    config: WindowConfig,
    metrics: pd.DataFrame,
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    def row(cohort: str, method: str) -> pd.Series:
        return metrics[
            metrics["cohort"].eq(cohort) & metrics["method"].eq(method)
        ].iloc[0]

    internal_base = row("internal_development", "frozen_whole_clip_baseline")
    internal_offline = row(
        "internal_development", "duration_normalized_windowed_offline_context"
    )
    internal_causal = row(
        "internal_development", "duration_normalized_windowed_causal"
    )
    external_base = row("external_provisional", "frozen_whole_clip_baseline")
    external_offline = row(
        "external_provisional", "duration_normalized_windowed_offline_context"
    )
    external_causal = row(
        "external_provisional", "duration_normalized_windowed_causal"
    )
    offline_delta_r2 = bootstrap[
        bootstrap["comparison"].eq(
            "duration_normalized_offline_context_minus_frozen_baseline"
        )
        & bootstrap["metric"].eq("rr_r2")
    ].iloc[0]
    causal_delta_r2 = bootstrap[
        bootstrap["comparison"].eq(
            "duration_normalized_causal_minus_frozen_baseline"
        )
        & bootstrap["metric"].eq("rr_r2")
    ].iloc[0]
    lines = [
        "# Duration-Normalized Quality-Weighted Context RR Probe",
        "",
        "Status: `development_candidate_external_diagnostic_only`",
        "",
        "## Fixed Configuration Origin",
        "",
        f"Window length `{config.window_seconds:g}` s was derived by rounding up the "
        "median internal raw duration. Long-clip activation begins at "
        f"`{config.long_clip_threshold_seconds:g}` s. The lower adaptive RR bound "
        f"`{config.min_rr_bpm:g}` bpm was derived from the minimum internal reference "
        "RR with a 5-bpm tolerance. External reference labels are not read by the "
        "prediction functions.",
        "",
        "## Results",
        "",
        "| cohort | method | R2 | MAE (bpm) | RMSE (bpm) |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| internal development | frozen baseline | {internal_base['rr_r2']:.6f} | "
        f"{internal_base['rr_mae']:.6f} | {internal_base['rr_rmse']:.6f} |",
        f"| internal development | offline bidirectional context | {internal_offline['rr_r2']:.6f} | "
        f"{internal_offline['rr_mae']:.6f} | {internal_offline['rr_rmse']:.6f} |",
        f"| internal development | causal context | {internal_causal['rr_r2']:.6f} | "
        f"{internal_causal['rr_mae']:.6f} | {internal_causal['rr_rmse']:.6f} |",
        f"| external provisional | frozen baseline | {external_base['rr_r2']:.6f} | "
        f"{external_base['rr_mae']:.6f} | {external_base['rr_rmse']:.6f} |",
        f"| external provisional | offline bidirectional context | {external_offline['rr_r2']:.6f} | "
        f"{external_offline['rr_mae']:.6f} | {external_offline['rr_rmse']:.6f} |",
        f"| external provisional | causal context | {external_causal['rr_r2']:.6f} | "
        f"{external_causal['rr_mae']:.6f} | {external_causal['rr_rmse']:.6f} |",
        "",
        f"The rule activated on `{int(internal_predictions['windowed_activation'].sum())}`/"
        f"`{len(internal_predictions)}` internal videos and "
        f"`{int(external_predictions['windowed_activation'].sum())}`/"
        f"`{len(external_predictions)}` external videos.",
        "",
        f"For offline bidirectional context, source-session clustered paired bootstrap "
        f"gave Delta R2 `{float(offline_delta_r2['delta_candidate_minus_baseline']):.6f}` "
        f"(95% CI `{float(offline_delta_r2['delta_ci_2_5']):.6f}` to "
        f"`{float(offline_delta_r2['delta_ci_97_5']):.6f}`).",
        f"For causal context, Delta R2 was "
        f"`{float(causal_delta_r2['delta_candidate_minus_baseline']):.6f}` "
        f"(95% CI `{float(causal_delta_r2['delta_ci_2_5']):.6f}` to "
        f"`{float(causal_delta_r2['delta_ci_97_5']):.6f}`).",
        "",
        "## Interpretation",
        "",
        "The offline estimator uses both preceding and following clips from the same "
        "recorded source video and must not be described as causal or real-time. The causal "
        "estimator uses only the current and preceding clips. This probe tests acquisition-"
        "duration normalization and temporal robustness, not "
        "a truth-calibrated correction. Improvement on the already inspected Jiufu cohort "
        "is development evidence only. It cannot unlock an external-performance claim and "
        "must be frozen before evaluation on new or untouched source sessions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    internal_root = args.internal_root.resolve()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_summary = pd.read_csv(
        internal_root / f"{args.internal_prefix}_summary.csv"
    )
    external_summary = pd.read_csv(
        external_root / f"{args.external_prefix}_summary.csv"
    )
    reference = pd.read_csv(args.external_reference_audit.resolve())
    active_ids = set(
        reference.loc[
            reference["primary_analysis_include"].astype(str).str.lower().eq("true"),
            "external_video_id",
        ].astype(str)
    )
    external_summary = external_summary[
        external_summary["video_id"].astype(str).isin(active_ids)
    ].reset_index(drop=True)
    config = derive_config(internal_summary, args.fps)
    internal_sessions = pd.DataFrame(
        {
            "video_id": internal_summary["video_id"].astype(str),
            "source_session_id": internal_summary["video_id"].astype(str),
        }
    )
    external_sessions = reference.rename(
        columns={"external_video_id": "video_id"}
    )[["video_id", "source_session_id"]]
    internal_predictions, internal_windows = predict_cohort(
        internal_root,
        args.internal_prefix,
        internal_summary,
        internal_sessions,
        config,
        "internal_development",
    )
    external_predictions, external_windows = predict_cohort(
        external_root,
        args.external_prefix,
        external_summary,
        external_sessions,
        config,
        "external_provisional",
    )
    internal_metrics, internal_scored = evaluate(
        internal_predictions,
        internal_truth(internal_summary),
        "internal_development",
        "configuration_derived_from_internal_distribution",
    )
    external_metrics, external_scored = evaluate(
        external_predictions,
        external_truth(reference),
        "external_provisional",
        "single_annotator_external_development_diagnostic",
    )
    metrics = pd.concat([internal_metrics, external_metrics], ignore_index=True)
    predictions = pd.concat([internal_scored, external_scored], ignore_index=True)
    windows = pd.concat([internal_windows, external_windows], ignore_index=True)
    config_table = pd.DataFrame(
        [{**asdict(config), "configuration_origin": "internal_development_distribution_only"}]
    )
    bootstrap = cluster_bootstrap_improvement(
        external_scored,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    predictions_path = output_dir / "paper_duration_normalized_windowed_predictions.csv"
    windows_path = output_dir / "paper_duration_normalized_windowed_windows.csv"
    metrics_path = output_dir / "paper_duration_normalized_windowed_metrics.csv"
    config_path = output_dir / "paper_duration_normalized_windowed_config.csv"
    bootstrap_path = output_dir / "paper_duration_normalized_windowed_bootstrap_ci.csv"
    report_path = output_dir / "paper_duration_normalized_windowed_report.md"
    predictions.to_csv(predictions_path, index=False)
    windows.to_csv(windows_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    config_table.to_csv(config_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    write_report(
        report_path,
        config,
        metrics,
        internal_predictions,
        external_predictions,
        bootstrap,
    )
    print(f"Saved duration-normalized predictions: {predictions_path}")
    print(f"Saved duration-normalized metrics: {metrics_path}")
    print(f"Saved duration-normalized report: {report_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
