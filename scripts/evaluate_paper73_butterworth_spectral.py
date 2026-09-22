"""Evaluate the paper-described FFT/Butterworth/peak workflow on 73 videos."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

import paper_repro_rr as rr
from evaluate_paper73_adaptive_roi import config_from_baseline


PAPER_LIKE_ORDER = 4
PAPER_LIKE_CUTOFF_HZ = 2.0
SPECTRAL_MIN_HZ = 0.2
SPECTRAL_MAX_HZ = 1.8


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    input_root = data_root / "al_images"
    assets = input_root / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument(
        "--truth", type=Path, default=data_root / "video" / "temperature_curves.csv"
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=input_root / "paper_repro_summary.csv"
    )
    parser.add_argument("--temperature-prefix", default="paper_repro")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--orders", default="2,4,6")
    parser.add_argument("--cutoffs-hz", default="1.5,1.75,2.0,2.25")
    parser.add_argument("--spectral-min-hz", type=float, default=SPECTRAL_MIN_HZ)
    parser.add_argument("--spectral-max-hz", type=float, default=SPECTRAL_MAX_HZ)
    parser.add_argument("--figure-video-id", default="zs197000")
    return parser.parse_args()


def parse_int_grid(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 1 for item in result):
        raise ValueError("Filter orders must contain positive integers")
    return result


def parse_float_grid(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0 for item in result):
        raise ValueError("Filter cutoffs must contain positive values")
    return result


def config_id(order: int, cutoff_hz: float) -> str:
    cutoff = f"{cutoff_hz:.2f}".replace(".", "p")
    return f"butterworth_o{order}_c{cutoff}hz"


def one_sided_spectrum(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=float)
    if len(signal) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    signal = signal - float(np.mean(signal))
    window = np.hanning(len(signal)) if len(signal) > 2 else np.ones(len(signal))
    n_fft = max(256, 1 << int(math.ceil(math.log2(max(2, len(signal) * 8)))))
    spectrum = np.fft.rfft(signal * window, n=n_fft)
    scale = max(float(np.sum(window)), 1e-12)
    amplitude = 2.0 * np.abs(spectrum) / scale
    frequency = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    return frequency, amplitude


def spectral_features(
    values: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
) -> dict[str, float]:
    frequency, amplitude = one_sided_spectrum(values, fps)
    band = (frequency >= min_hz) & (frequency <= max_hz)
    if not bool(np.any(band)):
        return {
            "dominant_frequency_hz": math.nan,
            "spectral_rr_bpm": math.nan,
            "spectral_concentration": math.nan,
        }
    band_frequency = frequency[band]
    band_power = amplitude[band] ** 2
    best = int(np.argmax(band_power))
    total_power = float(np.sum(band_power))
    return {
        "dominant_frequency_hz": float(band_frequency[best]),
        "spectral_rr_bpm": float(60.0 * band_frequency[best]),
        "spectral_concentration": float(band_power[best] / total_power)
        if total_power > 0
        else math.nan,
    }


def high_frequency_energy_ratio(
    values: np.ndarray, fps: float, cutoff_hz: float
) -> float:
    frequency, amplitude = one_sided_spectrum(values, fps)
    power = amplitude**2
    non_dc = frequency > 0
    total = float(np.sum(power[non_dc]))
    if total <= 0:
        return math.nan
    return float(np.sum(power[frequency > cutoff_hz]) / total)


def metric_row(
    method: str,
    truth_rr: np.ndarray,
    predicted_rr: np.ndarray,
    truth_count: np.ndarray,
    predicted_count: np.ndarray,
    *,
    order: float = math.nan,
    cutoff_hz: float = math.nan,
    evaluation: str,
) -> dict[str, object]:
    rr_error = predicted_rr - truth_rr
    count_error = predicted_count - truth_count
    abs_count_error = np.abs(count_error)
    return {
        "method": method,
        "butterworth_order": order,
        "butterworth_cutoff_hz": cutoff_hz,
        "evaluation": evaluation,
        "videos": int(len(truth_rr)),
        "rr_r2": rr.regression_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_pearson_r2": rr.pearson_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "count_mae": float(np.mean(abs_count_error)),
        "exact_count": int(np.sum(abs_count_error == 0)),
        "within_one_count": int(np.sum(abs_count_error <= 1)),
        "count_error_ge_two": int(np.sum(abs_count_error >= 2)),
    }


def make_three_panel_figure(
    output_path: Path,
    video_id: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    selected_peaks: np.ndarray,
    fps: float,
    peak_distance: int,
    peak_prominence: float,
    order: int,
    cutoff_hz: float,
) -> None:
    raw_frequency, raw_amplitude = one_sided_spectrum(raw, fps)
    filtered_frequency, filtered_amplitude = one_sided_spectrum(filtered, fps)
    trial_prominence = max(0.002, peak_prominence * 0.20)
    candidate_peaks, _ = find_peaks(
        filtered,
        distance=max(1, peak_distance),
        prominence=trial_prominence,
    )
    selected_set = {int(value) for value in selected_peaks}
    unselected = np.asarray(
        [int(value) for value in candidate_peaks if int(value) not in selected_set],
        dtype=int,
    )
    prominences = (
        peak_prominences(filtered, selected_peaks)[0]
        if len(selected_peaks)
        else np.array([], dtype=float)
    )

    fig = plt.figure(figsize=(10.5, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.36, wspace=0.28)
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_filtered = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, :])

    ax_raw.plot(raw_frequency, raw_amplitude, color="#1769aa", linewidth=1.1)
    ax_raw.axvline(cutoff_hz, color="#b23a2b", linestyle="--", linewidth=1.0)
    ax_raw.set_title("(a) Spectrum before filtering")
    ax_raw.set_xlabel("Frequency (Hz)")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.set_xlim(0, fps / 2.0)
    ax_raw.grid(alpha=0.25)

    ax_filtered.plot(
        filtered_frequency, filtered_amplitude, color="#1769aa", linewidth=1.1
    )
    ax_filtered.axvline(cutoff_hz, color="#b23a2b", linestyle="--", linewidth=1.0)
    ax_filtered.set_title(f"(b) Butterworth spectrum (order={order}, cutoff={cutoff_hz:g} Hz)")
    ax_filtered.set_xlabel("Frequency (Hz)")
    ax_filtered.set_ylabel("Amplitude")
    ax_filtered.set_xlim(0, fps / 2.0)
    ax_filtered.grid(alpha=0.25)

    time = np.arange(len(filtered), dtype=float) / fps
    ax_curve.plot(time, filtered, color="#1769aa", linewidth=1.6, label="Filtered curve")
    if len(unselected):
        ax_curve.scatter(
            unselected / fps,
            filtered[unselected],
            marker="x",
            s=58,
            linewidth=2.0,
            color="#f39c12",
            label="Unselected candidate",
            zorder=4,
        )
    if len(selected_peaks):
        ax_curve.scatter(
            selected_peaks / fps,
            filtered[selected_peaks],
            s=46,
            color="#f39c12",
            edgecolor="white",
            linewidth=0.6,
            label="Selected peak",
            zorder=5,
        )
        for peak, prominence in zip(selected_peaks, prominences):
            ax_curve.vlines(
                peak / fps,
                filtered[peak] - prominence,
                filtered[peak],
                color="#d62728",
                linewidth=1.1,
                alpha=0.85,
            )
    ax_curve.set_title(f"(c) Filtered curve and prominence peaks: {video_id}")
    ax_curve.set_xlabel("Time (s)")
    ax_curve.set_ylabel("Normalized nostril temperature")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def leave_one_out_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    video_ids = predictions["video_id"].drop_duplicates().tolist()
    for held_video in video_ids:
        training = predictions.loc[predictions["video_id"] != held_video]
        scores: list[tuple[tuple[float, float, float, float, int], str]] = []
        for candidate_id, group in training.groupby("config_id", sort=True):
            errors = group["predicted_rr_bpm"].to_numpy(float) - group["truth_rr"].to_numpy(float)
            rmse = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(np.abs(errors)))
            order = int(group["butterworth_order"].iloc[0])
            cutoff = float(group["butterworth_cutoff_hz"].iloc[0])
            tie_break = (
                round(rmse, 12),
                round(mae, 12),
                abs(cutoff - PAPER_LIKE_CUTOFF_HZ),
                abs(order - PAPER_LIKE_ORDER),
                order,
            )
            scores.append((tie_break, str(candidate_id)))
        selected_id = min(scores)[1]
        held = predictions.loc[
            (predictions["video_id"] == held_video)
            & (predictions["config_id"] == selected_id)
        ].iloc[0]
        row = held.to_dict()
        row["selected_config_id"] = selected_id
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    loo: pd.DataFrame,
    fps: float,
) -> None:
    fixed_id = config_id(PAPER_LIKE_ORDER, PAPER_LIKE_CUTOFF_HZ)
    baseline = metrics.loc[metrics["method"] == "validated_moving_average_baseline"].iloc[0]
    fixed = metrics.loc[metrics["method"] == fixed_id].iloc[0]
    spectral = metrics.loc[metrics["method"] == "direct_fft_after_fixed_butterworth"].iloc[0]
    loo_metric = metrics.loc[metrics["method"] == "leave_one_video_out_butterworth_selector"].iloc[0]
    grid = metrics.loc[metrics["evaluation"] == "same_cohort_sensitivity_grid"].sort_values(
        ["rr_rmse_bpm", "rr_mae_bpm", "method"]
    )
    best = grid.iloc[0]
    fixed_predictions = predictions.loc[predictions["config_id"] == fixed_id]
    improved = int((fixed_predictions["abs_count_error_change"] < 0).sum())
    worsened = int((fixed_predictions["abs_count_error_change"] > 0).sum())
    changed = int(fixed_predictions["prediction_changed"].sum())
    selected_counts = loo["selected_config_id"].value_counts().sort_index()
    resolution = 60.0 * fps / fixed_predictions["frames"].to_numpy(float)

    lines = [
        "# Paper73 Butterworth and Spectral Analysis",
        "",
        "Status: `internal_same_cohort_method_probe_not_external_confirmation`",
        "",
        "## Method provenance",
        "",
        "The source paper explicitly states the sequence Fourier spectrum analysis -> "
        "Butterworth low-pass filtering -> prominence-based sliding peak counting. It does "
        "not report the Butterworth order, cutoff frequency, or numerical prominence P.",
        "",
        f"The pre-specified paper-like engineering replication uses a zero-phase order-{PAPER_LIKE_ORDER} "
        f"Butterworth low-pass filter with a {PAPER_LIKE_CUTOFF_HZ:g} Hz cutoff at {fps:g} fps. "
        "The 2 Hz cutoff follows the visible post-filter spectrum boundary in Figure 7 rather "
        "than a disclosed paper parameter. Existing per-video fusion, prominence, distance, "
        "endpoint completion, and duration settings are held fixed to isolate the smoother.",
        "",
        "## Main comparison",
        "",
        "| method | R2 | MAE (bpm) | RMSE (bpm) | exact count | within one |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| validated moving-average baseline | {baseline.rr_r2:.6f} | {baseline.rr_mae_bpm:.6f} | {baseline.rr_rmse_bpm:.6f} | {int(baseline.exact_count)}/73 | {int(baseline.within_one_count)}/73 |",
        f"| fixed Butterworth order 4, 2 Hz | {fixed.rr_r2:.6f} | {fixed.rr_mae_bpm:.6f} | {fixed.rr_rmse_bpm:.6f} | {int(fixed.exact_count)}/73 | {int(fixed.within_one_count)}/73 |",
        f"| direct FFT dominant frequency | {spectral.rr_r2:.6f} | {spectral.rr_mae_bpm:.6f} | {spectral.rr_rmse_bpm:.6f} | {int(spectral.exact_count)}/73 | {int(spectral.within_one_count)}/73 |",
        f"| LOO-selected Butterworth | {loo_metric.rr_r2:.6f} | {loo_metric.rr_mae_bpm:.6f} | {loo_metric.rr_rmse_bpm:.6f} | {int(loo_metric.exact_count)}/73 | {int(loo_metric.within_one_count)}/73 |",
        "",
        f"The fixed paper-like filter changed {changed}/73 video counts versus baseline: "
        f"{improved} improved and {worsened} worsened.",
        "",
        "## Sensitivity grid",
        "",
        f"The best same-cohort grid row was `{best.method}` (R2={best.rr_r2:.6f}, "
        f"RMSE={best.rr_rmse_bpm:.6f}). This is diagnostic and cannot be reported as an "
        "independent test result.",
        "",
        "LOO selection counts: "
        + "; ".join(f"{name}={count}" for name, count in selected_counts.items())
        + ".",
        "",
        "## Spectral limitation",
        "",
        f"The median whole-clip Rayleigh RR resolution is {float(np.median(resolution)):.3f} bpm "
        f"(range {float(np.min(resolution)):.3f}-{float(np.max(resolution)):.3f} bpm). Zero-padding "
        "smooths the displayed spectrum but does not create additional independent frequency "
        "information, so direct FFT RR is secondary for these short clips.",
        "",
        "## Claim boundary",
        "",
        "Use the fixed order-4, 2 Hz result as the faithful engineering probe. Treat grid and "
        "LOO rows as internal sensitivity analyses. A change to the paper primary method requires "
        "freezing the filter and testing it on the independent Jiufu farm cohort.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders = parse_int_grid(args.orders)
    cutoffs = parse_float_grid(args.cutoffs_hz)
    if PAPER_LIKE_ORDER not in orders:
        orders.append(PAPER_LIKE_ORDER)
        orders.sort()
    if PAPER_LIKE_CUTOFF_HZ not in cutoffs:
        cutoffs.append(PAPER_LIKE_CUTOFF_HZ)
        cutoffs.sort()
    if any(cutoff >= args.fps / 2.0 for cutoff in cutoffs):
        raise ValueError("Every cutoff must be below the Nyquist frequency")
    if not 0 < args.spectral_min_hz < args.spectral_max_hz < args.fps / 2.0:
        raise ValueError("Spectral band must be positive, ordered, and below Nyquist")

    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    baseline = pd.read_csv(args.baseline_summary, dtype={"video_id": str})
    if len(truth) != 73 or truth["video_id"].nunique() != 73:
        raise ValueError("Truth table must contain exactly 73 unique video IDs")
    if set(truth["video_id"]) != set(baseline["video_id"]):
        raise ValueError("Baseline summary IDs do not match the 73-video truth table")
    truth_by_id = truth.set_index("video_id", drop=False)
    baseline_by_id = baseline.set_index("video_id", drop=False)

    candidate_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    figure_payload: dict[str, object] | None = None
    for index, video_id in enumerate(truth["video_id"], start=1):
        print(f"[{index}/73] {video_id}")
        truth_row = truth_by_id.loc[video_id]
        baseline_row = baseline_by_id.loc[video_id]
        temp_path = args.input_root / video_id / f"{args.temperature_prefix}_temperatures.csv"
        temp_df = pd.read_csv(temp_path)
        analysis_limit = int(float(baseline_row["analysis_frame_limit"]))
        temp_df = temp_df.iloc[:analysis_limit].copy()

        baseline_config = config_from_baseline(baseline_row, "paper73_spectral_replay")
        baseline_config = replace(baseline_config, fps=args.fps)
        replay_curve, replay_summary = rr.fuse_temperature_curve(
            temp_df.copy(), baseline_config, truth_row
        )
        baseline_count = int(float(baseline_row["peaks"]))
        replay_count = int(replay_summary["peaks"])
        replay_rows.append(
            {
                "video_id": video_id,
                "baseline_count": baseline_count,
                "replayed_count": replay_count,
                "count_match": baseline_count == replay_count,
            }
        )
        if replay_count != baseline_count:
            raise ValueError(
                f"Baseline replay mismatch for {video_id}: {replay_count} != {baseline_count}"
            )

        for order in orders:
            for cutoff_hz in cutoffs:
                candidate_config = replace(
                    baseline_config,
                    smoothing_method="butterworth",
                    butterworth_order=order,
                    butterworth_cutoff_hz=cutoff_hz,
                    adaptive_peak_retuning=False,
                )
                curve, summary = rr.fuse_temperature_curve(
                    temp_df.copy(), candidate_config, truth_row
                )
                candidate_count = int(summary["peaks"])
                truth_count = int(float(truth_row["breath_count"]))
                filtered = curve["smoothed_norm"].dropna().to_numpy(float)
                raw = curve["fused_norm"].to_numpy(float)
                raw_spectral = spectral_features(
                    raw, args.fps, args.spectral_min_hz, args.spectral_max_hz
                )
                filtered_spectral = spectral_features(
                    filtered, args.fps, args.spectral_min_hz, args.spectral_max_hz
                )
                candidate_rows.append(
                    {
                        "config_id": config_id(order, cutoff_hz),
                        "video_id": video_id,
                        "butterworth_order": order,
                        "butterworth_cutoff_hz": cutoff_hz,
                        "frames": int(len(raw)),
                        "duration_seconds": float(truth_row["duration_seconds"]),
                        "truth_count": truth_count,
                        "truth_rr": float(truth_row["rr"]),
                        "baseline_count": baseline_count,
                        "baseline_rr_bpm": float(baseline_row["rr_bpm"]),
                        "predicted_count": candidate_count,
                        "predicted_rr_bpm": float(summary["rr_bpm"]),
                        "count_error": candidate_count - truth_count,
                        "prediction_changed": candidate_count != baseline_count,
                        "abs_count_error_change": abs(candidate_count - truth_count)
                        - abs(baseline_count - truth_count),
                        "selected_fusion_mode": str(summary["selected_fusion_mode"]),
                        "peak_distance": int(summary["peak_distance"]),
                        "selected_peak_prominence": float(
                            summary["selected_peak_prominence"]
                        ),
                        "raw_dominant_frequency_hz": raw_spectral[
                            "dominant_frequency_hz"
                        ],
                        "raw_spectral_rr_bpm": raw_spectral["spectral_rr_bpm"],
                        "raw_spectral_concentration": raw_spectral[
                            "spectral_concentration"
                        ],
                        "filtered_dominant_frequency_hz": filtered_spectral[
                            "dominant_frequency_hz"
                        ],
                        "filtered_spectral_rr_bpm": filtered_spectral[
                            "spectral_rr_bpm"
                        ],
                        "filtered_spectral_concentration": filtered_spectral[
                            "spectral_concentration"
                        ],
                        "raw_high_frequency_energy_ratio": high_frequency_energy_ratio(
                            raw, args.fps, cutoff_hz
                        ),
                        "filtered_high_frequency_energy_ratio": high_frequency_energy_ratio(
                            filtered, args.fps, cutoff_hz
                        ),
                        "rayleigh_rr_resolution_bpm": 60.0 * args.fps / len(raw),
                    }
                )
                if (
                    video_id == args.figure_video_id
                    and order == PAPER_LIKE_ORDER
                    and math.isclose(cutoff_hz, PAPER_LIKE_CUTOFF_HZ)
                ):
                    selected_peaks = np.flatnonzero(curve["is_peak"].to_numpy(bool))
                    figure_payload = {
                        "raw": raw,
                        "filtered": filtered,
                        "selected_peaks": selected_peaks,
                        "peak_distance": int(summary["peak_distance"]),
                        "peak_prominence": float(summary["selected_peak_prominence"]),
                    }

    replay = pd.DataFrame(replay_rows)
    predictions = pd.DataFrame(candidate_rows)
    fixed_id = config_id(PAPER_LIKE_ORDER, PAPER_LIKE_CUTOFF_HZ)
    fixed = predictions.loc[predictions["config_id"] == fixed_id].copy()
    truth_rr = fixed["truth_rr"].to_numpy(float)
    truth_count = fixed["truth_count"].to_numpy(float)
    metric_rows = [
        metric_row(
            "validated_moving_average_baseline",
            truth_rr,
            fixed["baseline_rr_bpm"].to_numpy(float),
            truth_count,
            fixed["baseline_count"].to_numpy(float),
            evaluation="authoritative_post_truth_correction_baseline",
        )
    ]
    for candidate_id, group in predictions.groupby("config_id", sort=True):
        metric_rows.append(
            metric_row(
                str(candidate_id),
                group["truth_rr"].to_numpy(float),
                group["predicted_rr_bpm"].to_numpy(float),
                group["truth_count"].to_numpy(float),
                group["predicted_count"].to_numpy(float),
                order=float(group["butterworth_order"].iloc[0]),
                cutoff_hz=float(group["butterworth_cutoff_hz"].iloc[0]),
                evaluation=(
                    "pre_specified_paper_like_internal_probe"
                    if candidate_id == fixed_id
                    else "same_cohort_sensitivity_grid"
                ),
            )
        )

    spectral_rr = fixed["filtered_spectral_rr_bpm"].to_numpy(float)
    spectral_count = np.rint(
        spectral_rr * fixed["duration_seconds"].to_numpy(float) / 60.0
    )
    metric_rows.append(
        metric_row(
            "direct_fft_after_fixed_butterworth",
            truth_rr,
            spectral_rr,
            truth_count,
            spectral_count,
            order=PAPER_LIKE_ORDER,
            cutoff_hz=PAPER_LIKE_CUTOFF_HZ,
            evaluation="secondary_frequency_domain_diagnostic",
        )
    )

    loo = leave_one_out_selection(predictions)
    metric_rows.append(
        metric_row(
            "leave_one_video_out_butterworth_selector",
            loo["truth_rr"].to_numpy(float),
            loo["predicted_rr_bpm"].to_numpy(float),
            loo["truth_count"].to_numpy(float),
            loo["predicted_count"].to_numpy(float),
            evaluation="internal_leave_one_video_out_parameter_selection",
        )
    )
    metrics = pd.DataFrame(metric_rows)

    replay_path = args.output_dir / "paper73_butterworth_spectral_baseline_replay.csv"
    predictions_path = args.output_dir / "paper73_butterworth_spectral_predictions.csv"
    metrics_path = args.output_dir / "paper73_butterworth_spectral_metrics.csv"
    loo_path = args.output_dir / "paper73_butterworth_spectral_loo_predictions.csv"
    report_path = args.output_dir / "paper73_butterworth_spectral_report.md"
    figure_path = (
        args.output_dir
        / f"paper73_butterworth_spectral_{args.figure_video_id}_three_panel.png"
    )
    replay.to_csv(replay_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    loo.to_csv(loo_path, index=False)
    write_report(report_path, metrics, predictions, loo, args.fps)
    if figure_payload is not None:
        make_three_panel_figure(
            figure_path,
            args.figure_video_id,
            np.asarray(figure_payload["raw"], dtype=float),
            np.asarray(figure_payload["filtered"], dtype=float),
            np.asarray(figure_payload["selected_peaks"], dtype=int),
            args.fps,
            int(figure_payload["peak_distance"]),
            float(figure_payload["peak_prominence"]),
            PAPER_LIKE_ORDER,
            PAPER_LIKE_CUTOFF_HZ,
        )
    else:
        raise ValueError(f"Figure video was not found: {args.figure_video_id}")

    print(f"Baseline replay matches: {int(replay['count_match'].sum())}/73")
    print(metrics.to_string(index=False))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
