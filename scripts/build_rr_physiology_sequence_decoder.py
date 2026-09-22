from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedKFold

from rr_quality_residual_validation import metric_dict


@dataclass(frozen=True)
class DecoderConfig:
    candidate_prominence: float
    interval_weight: float
    signal_weight: float
    change_penalty: float
    decision_margin: float

    @property
    def config_id(self) -> str:
        return (
            f"prom_{self.candidate_prominence:.3f}_iw_{self.interval_weight:.2f}_"
            f"sw_{self.signal_weight:.2f}_cp_{self.change_penalty:.2f}_"
            f"margin_{self.decision_margin:.2f}"
        )


FIXED_CONFIG = DecoderConfig(0.020, 1.00, 1.00, 0.35, 0.15)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a physiology-constrained peak-sequence decoder after the "
            "conservative signal-aware safe gate. The decoder uses only curve "
            "morphology and independent signal-count estimates at prediction time."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--random-state", type=int, default=20260710)
    return parser.parse_args()


def numeric(data: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def video_prefix(video_id: object) -> str:
    match = re.match(r"^[A-Za-z]+", str(video_id))
    return match.group(0).lower() if match else "numeric"


def duration_values(data: pd.DataFrame) -> np.ndarray:
    for column in ["rr_duration_seconds", "duration_seconds", "raw_duration_seconds"]:
        values = numeric(data, column).to_numpy(dtype=float)
        if np.isfinite(values).any():
            return values
    return np.full(len(data), np.nan, dtype=float)


def rr_from_count(data: pd.DataFrame, count: np.ndarray) -> np.ndarray:
    duration = duration_values(data)
    return np.divide(
        count * 60.0,
        duration,
        out=np.full(len(count), np.nan, dtype=float),
        where=np.isfinite(duration) & (duration > 0),
    )


def load_predictions(input_root: Path, output_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_path = input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv"
    group_path = input_root / f"{output_prefix}_signal_aware_safe_policy_group_predictions.csv"
    if not fixed_path.exists() or not group_path.exists():
        raise FileNotFoundError(
            "Missing safe-policy predictions. Run scripts/rr_signal_aware_safe_policy.py first."
        )
    fixed = pd.read_csv(fixed_path)
    group = pd.read_csv(group_path)
    fixed["video_id"] = fixed["video_id"].astype(str)
    group["video_id"] = group["video_id"].astype(str)
    fixed["video_prefix_group"] = fixed["video_id"].map(video_prefix)
    group["video_prefix_group"] = group["video_id"].map(video_prefix)
    return fixed, group


def interpolate_curve(values: pd.Series) -> np.ndarray:
    series = pd.to_numeric(values, errors="coerce")
    series = series.interpolate(limit_direction="both")
    return series.to_numpy(dtype=float)


def signal_items(row: pd.Series, baseline_count: int) -> list[tuple[float, float, str]]:
    definitions = [
        ("spectral_count_estimate", "spectral_strength", "spectral"),
        ("fft_count_estimate", "fft_strength", "fft"),
        ("autocorr_count_estimate", "autocorr_strength", "autocorr"),
    ]
    items: list[tuple[float, float, str]] = []
    for count_column, strength_column, label in definitions:
        count = pd.to_numeric(pd.Series([row.get(count_column, math.nan)]), errors="coerce").iloc[0]
        strength = pd.to_numeric(
            pd.Series([row.get(strength_column, math.nan)]), errors="coerce"
        ).iloc[0]
        if pd.isna(count) or abs(float(count) - baseline_count) > 4:
            continue
        weight = 0.25 if pd.isna(strength) else float(np.clip(float(strength), 0.05, 1.0))
        items.append((float(count), weight, label))
    return items


def expected_period_frames(
    n_frames: int,
    baseline_count: int,
    signals: list[tuple[float, float, str]],
) -> float:
    if signals:
        counts = np.asarray([item[0] for item in signals], dtype=float)
        weights = np.asarray([item[1] for item in signals], dtype=float)
        order = np.argsort(counts)
        counts = counts[order]
        weights = weights[order]
        midpoint = 0.5 * float(np.sum(weights))
        signal_count = float(counts[np.searchsorted(np.cumsum(weights), midpoint)])
    else:
        signal_count = float(baseline_count)
    signal_count = max(signal_count, 1.0)
    return max(float(n_frames) / signal_count, 2.0)


def candidate_peaks(
    curve: np.ndarray,
    expected_period: float,
    prominence_threshold: float,
    existing_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(curve) < 3 or not np.isfinite(curve).any():
        return np.array([], dtype=int), np.array([], dtype=float)
    distance = max(1, int(round(expected_period * 0.25)))
    detected, _ = find_peaks(
        curve,
        distance=distance,
        prominence=float(prominence_threshold),
        width=1,
    )
    edge_width = max(2, min(len(curve), int(math.ceil(expected_period))))
    edge_candidates = np.array(
        [int(np.argmax(curve[:edge_width])), len(curve) - edge_width + int(np.argmax(curve[-edge_width:]))],
        dtype=int,
    )
    frames = np.unique(
        np.concatenate(
            [
                detected.astype(int),
                existing_frames.astype(int),
                edge_candidates,
            ]
        )
    )
    frames = frames[(frames >= 0) & (frames < len(curve))]
    reliefs: list[float] = []
    half_window = max(2, int(round(expected_period * 0.55)))
    for frame in frames:
        start = max(0, int(frame) - half_window)
        end = min(len(curve), int(frame) + half_window + 1)
        local = curve[start:end]
        relief = float(curve[int(frame)] - np.nanmin(local)) if len(local) else 0.0
        reliefs.append(max(relief, 0.0))
    return frames.astype(int), np.asarray(reliefs, dtype=float)


def interval_penalty(interval: np.ndarray, expected_period: float) -> np.ndarray:
    ratio = np.clip(interval / max(expected_period, 1e-6), 1e-3, None)
    penalty = np.square(np.log(ratio))
    penalty += 4.0 * np.square(np.clip(0.55 - ratio, 0.0, None))
    penalty += 2.0 * np.square(np.clip(ratio - 1.65, 0.0, None))
    return penalty


def best_sequence(
    frames: np.ndarray,
    reliefs: np.ndarray,
    target_count: int,
    expected_period: float,
    n_frames: int,
    interval_weight: float,
) -> tuple[float, list[int], float, float, float]:
    m = len(frames)
    k = int(target_count)
    if k <= 0 or m < k:
        return -math.inf, [], math.nan, math.nan, math.nan
    scale = float(np.nanquantile(reliefs, 0.75)) if len(reliefs) else 0.0
    scale = max(scale, 1e-6)
    node_score = np.clip(reliefs / scale, 0.0, 3.0)
    dp = np.full((k, m), -math.inf, dtype=float)
    predecessor = np.full((k, m), -1, dtype=int)
    dp[0, :] = node_score
    for selected in range(1, k):
        for current in range(selected, m):
            previous = np.arange(selected - 1, current, dtype=int)
            valid = np.isfinite(dp[selected - 1, previous])
            if not valid.any():
                continue
            previous = previous[valid]
            intervals = frames[current] - frames[previous]
            scores = (
                dp[selected - 1, previous]
                + node_score[current]
                - float(interval_weight) * interval_penalty(intervals, expected_period)
            )
            best_index = int(np.argmax(scores))
            dp[selected, current] = float(scores[best_index])
            predecessor[selected, current] = int(previous[best_index])
    end = int(np.argmax(dp[k - 1, :]))
    if not np.isfinite(dp[k - 1, end]):
        return -math.inf, [], math.nan, math.nan, math.nan
    selected_frames = [int(frames[end])]
    current = end
    for selected in range(k - 1, 0, -1):
        current = int(predecessor[selected, current])
        if current < 0:
            return -math.inf, [], math.nan, math.nan, math.nan
        selected_frames.append(int(frames[current]))
    selected_frames.reverse()
    intervals = np.diff(np.asarray(selected_frames, dtype=float))
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals) >= 2
        else math.nan
    )
    first_gap_ratio = float(selected_frames[0] / max(expected_period, 1e-6))
    last_gap_ratio = float((n_frames - 1 - selected_frames[-1]) / max(expected_period, 1e-6))
    edge_penalty = max(first_gap_ratio - 1.25, 0.0) ** 2 + max(last_gap_ratio - 1.25, 0.0) ** 2
    score = float(dp[k - 1, end] / k - float(interval_weight) * edge_penalty)
    return score, selected_frames, interval_cv, first_gap_ratio, last_gap_ratio


def signal_score(
    target_count: int,
    signals: list[tuple[float, float, str]],
) -> float:
    if not signals:
        return 0.0
    scores = [weight * math.exp(-abs(count - target_count)) for count, weight, _ in signals]
    weights = [weight for _, weight, _ in signals]
    return float(sum(scores) / (sum(weights) + 1e-9))


def config_grid() -> list[DecoderConfig]:
    return [
        DecoderConfig(prominence, interval_weight, signal_weight, change_penalty, margin)
        for prominence in [0.010, 0.020]
        for interval_weight in [0.50, 1.00]
        for signal_weight in [0.50, 1.00, 1.50]
        for change_penalty in [0.20, 0.35]
        for margin in [0.05, 0.15]
    ]


def build_curve_context(
    input_root: Path,
    output_prefix: str,
    predictions: pd.DataFrame,
    baseline_column: str,
) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    for _, row in predictions.iterrows():
        video_id = str(row["video_id"])
        baseline_value = pd.to_numeric(pd.Series([row.get(baseline_column)]), errors="coerce").iloc[0]
        if pd.isna(baseline_value):
            continue
        baseline_count = int(round(float(baseline_value)))
        curve_path = input_root / video_id / f"{output_prefix}_curve.csv"
        if not curve_path.exists():
            continue
        curve_table = pd.read_csv(curve_path)
        if "smoothed_norm" not in curve_table.columns:
            continue
        curve = interpolate_curve(curve_table["smoothed_norm"])
        finite = np.isfinite(curve)
        if not finite.any():
            continue
        if not finite.all():
            curve = pd.Series(curve).interpolate(limit_direction="both").to_numpy(dtype=float)
        existing_frames = (
            pd.to_numeric(
                curve_table.loc[
                    curve_table.get("is_peak", pd.Series(False, index=curve_table.index)).astype(str).str.lower().isin(["true", "1"]),
                    "frame_index",
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=int)
            if "frame_index" in curve_table.columns
            else np.array([], dtype=int)
        )
        signals = signal_items(row, baseline_count)
        expected_period = expected_period_frames(len(curve), baseline_count, signals)
        contexts[video_id] = {
            "curve": curve,
            "existing_frames": existing_frames,
            "signals": signals,
            "expected_period": expected_period,
            "baseline_count": baseline_count,
        }
    return contexts


def decode_config(
    predictions: pd.DataFrame,
    contexts: dict[str, dict[str, object]],
    config: DecoderConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, source in predictions.iterrows():
        video_id = str(source["video_id"])
        context = contexts.get(video_id)
        if context is None:
            baseline_count = int(round(float(source.get("signal_aware_safe_final_peaks", 0))))
            rows.append(
                {
                    "video_id": video_id,
                    "decoder_count": baseline_count,
                    "decoder_adjust": 0,
                    "decoder_reason": "curve_context_missing",
                    "winner_margin_vs_baseline": math.nan,
                    "decoder_selected_frames": "",
                }
            )
            continue
        curve = np.asarray(context["curve"], dtype=float)
        baseline_count = int(context["baseline_count"])
        expected_period = float(context["expected_period"])
        signals = list(context["signals"])
        frames, reliefs = candidate_peaks(
            curve,
            expected_period,
            config.candidate_prominence,
            np.asarray(context["existing_frames"], dtype=int),
        )
        candidate_rows: list[dict[str, object]] = []
        for target_count in sorted({max(1, baseline_count - 1), baseline_count, baseline_count + 1}):
            sequence_score, selected, interval_cv, first_gap, last_gap = best_sequence(
                frames,
                reliefs,
                target_count,
                expected_period,
                len(curve),
                config.interval_weight,
            )
            total_score = (
                sequence_score
                + config.signal_weight * signal_score(target_count, signals)
                - config.change_penalty * abs(target_count - baseline_count)
            )
            candidate_rows.append(
                {
                    "target_count": target_count,
                    "sequence_score": sequence_score,
                    "signal_score": signal_score(target_count, signals),
                    "total_score": total_score,
                    "selected_frames": selected,
                    "interval_cv": interval_cv,
                    "first_gap_ratio": first_gap,
                    "last_gap_ratio": last_gap,
                }
            )
        candidate_table = pd.DataFrame(candidate_rows)
        baseline_row = candidate_table[candidate_table["target_count"].eq(baseline_count)].iloc[0]
        winner = candidate_table.sort_values(
            ["total_score", "target_count"], ascending=[False, True]
        ).iloc[0]
        baseline_score = float(baseline_row["total_score"])
        winner_score = float(winner["total_score"])
        winner_margin = winner_score - baseline_score
        if not np.isfinite(baseline_score):
            winner = baseline_row
            winner_margin = math.nan
            reason = "baseline_sequence_not_observable_keep_safe_gate"
        elif not np.isfinite(winner_score):
            winner = baseline_row
            winner_margin = math.nan
            reason = "candidate_sequences_not_observable_keep_safe_gate"
        elif int(winner["target_count"]) != baseline_count and winner_margin < config.decision_margin:
            winner = baseline_row
            reason = "change_blocked_by_margin"
        elif int(winner["target_count"]) == baseline_count:
            reason = "baseline_sequence_preferred"
        else:
            reason = "physiology_sequence_adjustment"
        decoder_count = int(winner["target_count"])
        rows.append(
            {
                "video_id": video_id,
                "decoder_count": decoder_count,
                "decoder_adjust": decoder_count - baseline_count,
                "decoder_reason": reason,
                "winner_margin_vs_baseline": winner_margin,
                "decoder_sequence_score": float(winner["sequence_score"]),
                "decoder_signal_score": float(winner["signal_score"]),
                "decoder_total_score": float(winner["total_score"]),
                "decoder_interval_cv": winner["interval_cv"],
                "decoder_first_gap_ratio": winner["first_gap_ratio"],
                "decoder_last_gap_ratio": winner["last_gap_ratio"],
                "decoder_expected_period_frames": expected_period,
                "decoder_candidate_peak_count": len(frames),
                "decoder_selected_frames": ";".join(str(value) for value in winner["selected_frames"]),
                "decoder_signal_counts": ";".join(
                    f"{label}:{count:.0f}:{weight:.3f}" for count, weight, label in signals
                ),
            }
        )
    return pd.DataFrame(rows)


def metrics_for_count(
    data: pd.DataFrame,
    count: np.ndarray,
    label: str,
    evaluation_note: str,
) -> dict[str, object]:
    rr = rr_from_count(data, count)
    return metric_dict(
        label,
        numeric(data, "truth_rr").to_numpy(dtype=float),
        rr,
        numeric(data, "truth_count").to_numpy(dtype=float),
        count,
        evaluation_note=evaluation_note,
    )


def score_config_on_indices(
    data: pd.DataFrame,
    count: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, ...]:
    truth_count = numeric(data, "truth_count").to_numpy(dtype=float)[indices]
    truth_rr = numeric(data, "truth_rr").to_numpy(dtype=float)[indices]
    predicted_count = count[indices]
    predicted_rr = rr_from_count(data, count)[indices]
    valid_count = np.isfinite(truth_count) & np.isfinite(predicted_count)
    abs_error = np.abs(predicted_count[valid_count] - truth_count[valid_count])
    ge2 = int(np.sum(abs_error >= 2))
    exact = int(np.sum(abs_error == 0))
    rr_valid = np.isfinite(truth_rr) & np.isfinite(predicted_rr)
    mae = float(np.mean(np.abs(predicted_rr[rr_valid] - truth_rr[rr_valid])))
    sse = float(np.sum((predicted_rr[rr_valid] - truth_rr[rr_valid]) ** 2))
    centered = truth_rr[rr_valid] - float(np.mean(truth_rr[rr_valid]))
    sst = float(np.sum(centered**2))
    r2 = 1.0 - sse / sst if sst > 0 else -math.inf
    changes = int(np.sum(count[indices] != numeric(data, "signal_aware_safe_final_peaks").to_numpy(dtype=float)[indices]))
    return float(ge2), float(-exact), mae, float(-r2), float(changes)


def select_config(
    data: pd.DataFrame,
    config_counts: dict[DecoderConfig, np.ndarray],
    train_indices: np.ndarray,
) -> DecoderConfig:
    return min(
        config_counts,
        key=lambda config: score_config_on_indices(
            data, config_counts[config], train_indices
        ),
    )


def build_config_predictions(
    data: pd.DataFrame,
    contexts: dict[str, dict[str, object]],
    configs: list[DecoderConfig],
) -> tuple[dict[DecoderConfig, np.ndarray], dict[DecoderConfig, pd.DataFrame]]:
    counts: dict[DecoderConfig, np.ndarray] = {}
    details: dict[DecoderConfig, pd.DataFrame] = {}
    for config in configs:
        decoded = decode_config(data, contexts, config)
        aligned = data[["video_id"]].merge(decoded, on="video_id", how="left", validate="one_to_one")
        fallback = numeric(data, "signal_aware_safe_final_peaks")
        count = pd.to_numeric(aligned["decoder_count"], errors="coerce").fillna(fallback).to_numpy(dtype=float)
        counts[config] = count
        details[config] = aligned
    return counts, details


def stratified_nested_predictions(
    data: pd.DataFrame,
    config_counts: dict[DecoderConfig, np.ndarray],
    random_state: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    baseline = numeric(data, "signal_aware_safe_final_peaks").to_numpy(dtype=float)
    truth = numeric(data, "truth_count").to_numpy(dtype=float)
    target = np.clip(np.rint(truth - baseline), -1, 1).astype(int)
    min_class = int(pd.Series(target).value_counts().min())
    n_splits = max(2, min(5, min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = np.full(len(data), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(data)), target), start=1):
        config = select_config(data, config_counts, train)
        output[test] = config_counts[config][test]
        rows.append(
            {
                "validation_mode": "nested_stratified_config_selection",
                "fold": fold,
                "train_videos": len(train),
                "test_videos": len(test),
                **asdict(config),
                "config_id": config.config_id,
            }
        )
    return output, rows


def prefix_group_predictions(
    data: pd.DataFrame,
    config_counts: dict[DecoderConfig, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    groups = data["video_prefix_group"].astype(str).to_numpy()
    output = np.full(len(data), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold, held_out in enumerate(sorted(np.unique(groups)), start=1):
        test = np.flatnonzero(groups == held_out)
        train = np.flatnonzero(groups != held_out)
        config = select_config(data, config_counts, train)
        output[test] = config_counts[config][test]
        rows.append(
            {
                "validation_mode": "leave_one_prefix_group_out_config_selection",
                "fold": fold,
                "held_out_group": held_out,
                "train_videos": len(train),
                "test_videos": len(test),
                **asdict(config),
                "config_id": config.config_id,
            }
        )
    return output, rows


def metric_grid(
    data: pd.DataFrame,
    config_counts: dict[DecoderConfig, np.ndarray],
) -> pd.DataFrame:
    baseline = numeric(data, "signal_aware_safe_final_peaks").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for config, count in config_counts.items():
        metrics = metrics_for_count(
            data,
            count,
            config.config_id,
            "full_internal_config_sensitivity_not_primary",
        )
        metrics.update(asdict(config))
        metrics["config_id"] = config.config_id
        metrics["changed_videos"] = int(np.sum(count != baseline))
        rows.append(metrics)
    grid = pd.DataFrame(rows)
    return grid.sort_values(
        ["abs_count_error_ge2", "exact_count", "rr_mae", "changed_videos"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def assemble_predictions(
    fixed: pd.DataFrame,
    fixed_details: pd.DataFrame,
    fixed_count: np.ndarray,
    nested_count: np.ndarray,
    group: pd.DataFrame,
    group_count: np.ndarray,
) -> pd.DataFrame:
    output = fixed.copy()
    output = output.merge(
        fixed_details.add_prefix("physiology_fixed_").rename(
            columns={"physiology_fixed_video_id": "video_id"}
        ),
        on="video_id",
        how="left",
        validate="one_to_one",
    )
    output["physiology_fixed_count"] = fixed_count
    output["physiology_fixed_rr_bpm"] = rr_from_count(fixed, fixed_count)
    output["physiology_nested_count"] = nested_count
    output["physiology_nested_rr_bpm"] = rr_from_count(fixed, nested_count)
    group_values = pd.DataFrame(
        {
            "video_id": group["video_id"].astype(str),
            "physiology_prefix_group_count": group_count,
            "physiology_prefix_group_rr_bpm": rr_from_count(group, group_count),
        }
    )
    output = output.merge(group_values, on="video_id", how="left", validate="one_to_one")
    return output


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    grid: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    def metric_line(label: str) -> str:
        row = metrics[metrics["label"].astype(str).eq(label)].iloc[0]
        return (
            f"RR R2={float(row['rr_r2']):.4f}, MAE={float(row['rr_mae']):.3f}, "
            f"RMSE={float(row['rr_rmse']):.3f}, exact={int(row['exact_count'])}/"
            f"{int(row['count_valid_videos'])}, >=2 errors={int(row['abs_count_error_ge2'])}"
        )

    fixed = metrics[metrics["label"].astype(str).eq("physiology_decoder_fixed")].iloc[0]
    nested = metrics[metrics["label"].astype(str).eq("physiology_decoder_nested")].iloc[0]
    grouped = metrics[metrics["label"].astype(str).eq("physiology_decoder_prefix_group")].iloc[0]
    stable = bool(
        int(fixed["abs_count_error_ge2"]) == 0
        and int(nested["abs_count_error_ge2"]) == 0
        and int(grouped["abs_count_error_ge2"]) == 0
        and float(nested["rr_mae"]) <= float(
            metrics[metrics["label"].astype(str).eq("safe_gate_fixed_baseline")].iloc[0]["rr_mae"]
        )
        and float(grouped["rr_mae"]) <= float(
            metrics[metrics["label"].astype(str).eq("safe_gate_prefix_group_baseline")].iloc[0]["rr_mae"]
        )
    )
    best = grid.iloc[0]
    lines = [
        "# Physiology-Constrained Peak-Sequence Decoder",
        "",
        "This probe evaluates a dynamic-programming decoder that compares only the "
        "safe-gate count and its +/-1 neighbors. Candidate sequences are scored from "
        "peak relief, interval regularity, boundary coverage, and independent spectral/"
        "FFT/autocorrelation count support. No reference count or RR is used in a "
        "per-video decision.",
        "",
        "## Results",
        "",
        f"Safe-gate fixed baseline: {metric_line('safe_gate_fixed_baseline')}.",
        f"Fixed physiology decoder: {metric_line('physiology_decoder_fixed')}.",
        f"Nested configuration selection: {metric_line('physiology_decoder_nested')}.",
        f"Prefix-group configuration selection: {metric_line('physiology_decoder_prefix_group')}.",
        f"Prefix-group safe baseline: {metric_line('safe_gate_prefix_group_baseline')}.",
        "",
        f"Internal stability gate: `{'PASS' if stable else 'FAIL'}`.",
        "",
        "The stability gate requires fixed, nested, and prefix-group evaluations to "
        "avoid >=2-breath errors, while nested and grouped MAE must not exceed their "
        "corresponding safe-gate baselines. A failed gate keeps this experiment as a "
        "negative or exploratory result rather than a manuscript primary method.",
        "",
        "## Full-Data Sensitivity Upper Candidate",
        "",
        f"Best point-selected configuration: `{best['config_id']}`; RR R2="
        f"{float(best['rr_r2']):.4f}, MAE={float(best['rr_mae']):.3f}, exact="
        f"{int(best['exact_count'])}/{int(best['count_valid_videos'])}, >=2 errors="
        f"{int(best['abs_count_error_ge2'])}. This row is sensitivity analysis only.",
        "",
        "## Selection Audit",
        "",
        f"Configuration-selection rows: `{len(selections)}`. Fixed configuration: "
        f"`{FIXED_CONFIG.config_id}`.",
        "",
        "## Claim Boundary",
        "",
        "This internal probe may support a physiology-constrained decoding claim only "
        "if the stability gate passes and the frozen rule is subsequently evaluated on "
        "the independent external cohort. Truth-calibrated optimization and full-data "
        "best-grid rows are not deployable performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = input_root / f"{args.corrected_prefix}_paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed, group = load_predictions(input_root, args.output_prefix)
    configs = config_grid()
    if FIXED_CONFIG not in configs:
        configs.append(FIXED_CONFIG)

    fixed_context = build_curve_context(
        input_root,
        args.output_prefix,
        fixed,
        "signal_aware_safe_final_peaks",
    )
    group_context = build_curve_context(
        input_root,
        args.output_prefix,
        group,
        "signal_aware_safe_group_final_peaks",
    )
    fixed_counts, fixed_details = build_config_predictions(fixed, fixed_context, configs)

    group_for_decoder = group.copy()
    group_for_decoder["signal_aware_safe_final_peaks"] = numeric(
        group, "signal_aware_safe_group_final_peaks"
    )
    group_for_decoder["signal_aware_safe_final_rr_bpm"] = numeric(
        group, "signal_aware_safe_group_final_rr_bpm"
    )
    group_counts, _ = build_config_predictions(group_for_decoder, group_context, configs)

    fixed_count = fixed_counts[FIXED_CONFIG]
    nested_count, nested_selections = stratified_nested_predictions(
        fixed,
        fixed_counts,
        args.random_state,
    )
    group_count, group_selections = prefix_group_predictions(
        group_for_decoder,
        group_counts,
    )
    grid = metric_grid(fixed, fixed_counts)
    metrics = pd.DataFrame(
        [
            metrics_for_count(
                fixed,
                numeric(fixed, "signal_aware_safe_final_peaks").to_numpy(dtype=float),
                "safe_gate_fixed_baseline",
                "existing_safe_gate_fixed_oof",
            ),
            metrics_for_count(
                fixed,
                fixed_count,
                "physiology_decoder_fixed",
                f"fixed_non_truth_decoder_{FIXED_CONFIG.config_id}",
            ),
            metrics_for_count(
                fixed,
                nested_count,
                "physiology_decoder_nested",
                "outer_stratified_fold_configuration_selection",
            ),
            metrics_for_count(
                group_for_decoder,
                numeric(group_for_decoder, "signal_aware_safe_final_peaks").to_numpy(dtype=float),
                "safe_gate_prefix_group_baseline",
                "existing_safe_gate_prefix_group_oof",
            ),
            metrics_for_count(
                group_for_decoder,
                group_count,
                "physiology_decoder_prefix_group",
                "leave_one_prefix_group_out_configuration_selection",
            ),
        ]
    )
    selections = pd.DataFrame(nested_selections + group_selections)
    predictions = assemble_predictions(
        fixed,
        fixed_details[FIXED_CONFIG],
        fixed_count,
        nested_count,
        group_for_decoder,
        group_count,
    )

    features_path = output_dir / "paper_physiology_sequence_decoder_predictions.csv"
    metrics_path = output_dir / "paper_physiology_sequence_decoder_metrics.csv"
    grid_path = output_dir / "paper_physiology_sequence_decoder_config_grid.csv"
    selections_path = output_dir / "paper_physiology_sequence_decoder_fold_selections.csv"
    report_path = output_dir / "paper_physiology_sequence_decoder_report.md"
    predictions.to_csv(features_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    grid.to_csv(grid_path, index=False)
    selections.to_csv(selections_path, index=False)
    write_report(report_path, metrics, grid, selections)
    print(f"Saved physiology decoder predictions: {features_path}")
    print(f"Saved physiology decoder metrics: {metrics_path}")
    print(f"Saved physiology decoder config grid: {grid_path}")
    print(f"Saved physiology decoder fold selections: {selections_path}")
    print(f"Saved physiology decoder report: {report_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
