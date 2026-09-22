from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_rr_duration_normalized_windowed_innovation import (
    WindowConfig,
    derive_config,
    detect_window,
    external_truth,
    internal_truth,
    metric_dict,
    window_starts,
)
from paper_repro_rr import (
    fast_fusion_quality_config,
    frame_image_paths,
    normalize_series,
    repair_series,
    select_fused_series,
    smooth_curve,
)


SIGNALS = ["gray", "blue", "green", "red", "hue", "lab_a", "lab_b", "red_minus_blue"]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description="Evaluate calibration-free relative thermal-color signals selected on internal data only."
    )
    parser.add_argument("--internal-root", type=Path, default=repo / "Dataset_new" / "72video" / "al_images")
    parser.add_argument("--external-root", type=Path, default=repo / "Dataset_new" / "72video" / "external_al_images_single_reference")
    parser.add_argument("--reference-csv", type=Path, default=assets / "paper_external_validation_single_reference_rows.csv")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--overwrite-signals", action="store_true")
    return parser.parse_args()


def roi_means(image: np.ndarray, x: object, y: object, radius: int) -> dict[str, float]:
    x_value = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    y_value = pd.to_numeric(pd.Series([y]), errors="coerce").iloc[0]
    if pd.isna(x_value) or pd.isna(y_value):
        return {signal: math.nan for signal in SIGNALS}
    cx, cy = int(round(float(x_value))), int(round(float(y_value)))
    height, width = image.shape[:2]
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return {signal: math.nan for signal in SIGNALS}
    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    if not np.any(mask):
        return {signal: math.nan for signal in SIGNALS}
    pixels = patch[mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    blue, green, red = np.mean(pixels.astype(float), axis=0)
    return {
        "gray": float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[mask])),
        "blue": float(blue),
        "green": float(green),
        "red": float(red),
        "hue": float(np.median(hsv[:, 0].astype(float))),
        "lab_a": float(np.mean(lab[:, 1].astype(float))),
        "lab_b": float(np.mean(lab[:, 2].astype(float))),
        "red_minus_blue": float(red - blue),
    }


def extract_video_signals(video_dir: Path, prefix: str, radius: int, overwrite: bool) -> pd.DataFrame:
    cache = video_dir / f"{prefix}_calibration_free_signals.csv"
    if cache.exists() and not overwrite:
        return pd.read_csv(cache)
    temperature_path = video_dir / f"{prefix}_temperatures.csv"
    temperature = pd.read_csv(temperature_path)
    paths = frame_image_paths(video_dir)
    path_by_name = {path.name: path for path in paths}
    rows: list[dict[str, object]] = []
    for row in temperature.itertuples(index=False):
        path = path_by_name.get(str(row.frame_name))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path is not None else None
        output: dict[str, object] = {"frame_name": row.frame_name}
        if image is None:
            for side in ["left", "right"]:
                for signal in SIGNALS:
                    output[f"{side}_{signal}"] = math.nan
        else:
            for side in ["left", "right"]:
                values = roi_means(
                    image,
                    getattr(row, f"{side}_x"),
                    getattr(row, f"{side}_y"),
                    radius,
                )
                for signal, value in values.items():
                    output[f"{side}_{signal}"] = value
        rows.append(output)
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False)
    return table


def fused_curve(table: pd.DataFrame, signal: str, invert: bool, config: WindowConfig) -> np.ndarray:
    left = repair_series(table[f"left_{signal}"], True)
    right = repair_series(table[f"right_{signal}"], True)
    left_norm = normalize_series(left)
    right_norm = normalize_series(right)
    if invert:
        left_norm = 1.0 - left_norm
        right_norm = 1.0 - right_norm
    fusion_table = pd.DataFrame({"left_temp": left, "right_temp": right})
    quality_config = replace(
        fast_fusion_quality_config(True),
        min_rr_bpm=config.min_rr_bpm,
        max_rr_bpm=config.max_rr_bpm,
    )
    _, fused, _ = select_fused_series(
        fusion_table, left_norm, right_norm, "adaptive", quality_config
    )
    values = fused.interpolate(method="linear", limit_direction="both").fillna(0.5).to_numpy(float)
    smoothed, _ = smooth_curve(values, 3)
    return smoothed


def estimate_curve(curve: np.ndarray, config: WindowConfig) -> tuple[float, int, float]:
    window_frames = max(8, int(round(config.window_seconds * config.fps)))
    stride_frames = max(1, int(round(config.stride_seconds * config.fps)))
    if 8 <= len(curve) < int(round(0.7 * window_frames)):
        detail = detect_window(curve, 0.0, config)
        return (
            float(detail["window_rr_bpm"]),
            1,
            float(detail["window_quality_score"]),
        )
    starts = window_starts(len(curve), window_frames, stride_frames)
    details = []
    for start in starts:
        segment = curve[start : min(len(curve), start + window_frames)]
        if len(segment) < max(8, int(round(0.7 * window_frames))):
            continue
        details.append(detect_window(segment, 0.0, config))
    if not details:
        return math.nan, 0, math.nan
    rr = np.asarray([float(row["window_rr_bpm"]) for row in details])
    quality = np.asarray([float(row["window_quality_score"]) for row in details])
    weights = np.exp(np.clip(quality, -3.0, 3.0))
    return float(np.average(rr, weights=weights)), len(details), float(np.mean(quality))


def cohort_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: WindowConfig,
    overwrite: bool,
    radius: int,
    cohort: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, summary_row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(summary_row.video_id)
        print(f"[{cohort} {index}/{len(summary)}] {video_id}")
        signals = extract_video_signals(root / video_id, prefix, radius, overwrite)
        for signal in SIGNALS:
            for invert in [False, True]:
                curve = fused_curve(signals, signal, invert, config)
                rr, windows, quality = estimate_curve(curve, config)
                rows.append(
                    {
                        "cohort": cohort,
                        "video_id": video_id,
                        "signal": signal,
                        "polarity": "inverted" if invert else "direct",
                        "rr_bpm": rr,
                        "windows": windows,
                        "quality_mean": quality,
                    }
                )
    return pd.DataFrame(rows)


def score_candidates(predictions: pd.DataFrame, truth: pd.DataFrame, cohort: str) -> pd.DataFrame:
    data = predictions.merge(truth, on="video_id", how="inner", validate="many_to_one")
    rows = []
    for (signal, polarity), group in data.groupby(["signal", "polarity"]):
        row = metric_dict(
            cohort,
            f"calibration_free_{signal}_{polarity}",
            group["truth_rr"].to_numpy(float),
            group["rr_bpm"].to_numpy(float),
            group["truth_count"].to_numpy(float),
            group["truth_duration_seconds"].to_numpy(float),
            "signal_and_polarity_selected_on_internal_development_only",
        )
        row.update({"signal": signal, "polarity": polarity})
        rows.append(row)
    return pd.DataFrame(rows), data


def score_ranked_ensembles(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ranking: list[tuple[str, str]],
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = predictions.pivot_table(
        index="video_id",
        columns=["signal", "polarity"],
        values="rr_bpm",
        aggfunc="first",
    )
    reference = truth.set_index("video_id").loc[wide.index].reset_index()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for size in range(1, len(ranking) + 1):
        members = ranking[:size]
        estimate = wide[members].mean(axis=1)
        row = metric_dict(
            cohort,
            f"calibration_free_internal_ranked_top_{size}_ensemble",
            reference["truth_rr"].to_numpy(float),
            estimate.to_numpy(float),
            reference["truth_count"].to_numpy(float),
            reference["truth_duration_seconds"].to_numpy(float),
            "ensemble_size_and_members_ranked_on_internal_development_only",
        )
        row.update(
            {
                "signal": "internal_ranked_ensemble",
                "polarity": f"top_{size}",
                "selected_members": ";".join(f"{signal}:{polarity}" for signal, polarity in members),
            }
        )
        metric_rows.append(row)
        for video_id, rr in estimate.items():
            prediction_rows.append(
                {
                    "cohort": cohort,
                    "video_id": video_id,
                    "signal": "internal_ranked_ensemble",
                    "polarity": f"top_{size}",
                    "rr_bpm": float(rr),
                    "selected_members": row["selected_members"],
                }
            )
    scored = pd.DataFrame(prediction_rows).merge(
        truth, on="video_id", how="left", validate="many_to_one"
    )
    return pd.DataFrame(metric_rows), scored


def selected_member_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: WindowConfig,
    members: list[tuple[str, str]],
    radius: int,
    cohort: str,
) -> pd.DataFrame:
    rows = []
    member_text = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    for summary_row in summary.itertuples(index=False):
        video_id = str(summary_row.video_id)
        table = extract_video_signals(root / video_id, prefix, radius, False)
        estimates = []
        for signal, polarity in members:
            curve = fused_curve(table, signal, polarity == "inverted", config)
            estimates.append(estimate_curve(curve, config)[0])
        rows.append(
            {
                "cohort": cohort,
                "video_id": video_id,
                "signal": "fixed_internal_ranked_ensemble",
                "polarity": f"top_{len(members)}",
                "rr_bpm": float(np.mean(estimates)),
                "selected_members": member_text,
                "selected_prominence": config.base_prominence,
            }
        )
    return pd.DataFrame(rows)


def score_fixed_ensemble(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    cohort: str,
    method: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    data = predictions.merge(truth, on="video_id", how="inner", validate="one_to_one")
    row = metric_dict(
        cohort,
        method,
        data["truth_rr"].to_numpy(float),
        data["rr_bpm"].to_numpy(float),
        data["truth_count"].to_numpy(float),
        data["truth_duration_seconds"].to_numpy(float),
        "channels_and_prominence_selected_on_internal_development_only",
    )
    row.update(
        {
            "signal": "fixed_internal_ranked_ensemble",
            "polarity": str(data["polarity"].iloc[0]),
            "selected_members": str(data["selected_members"].iloc[0]),
            "selected_prominence": float(data["selected_prominence"].iloc[0]),
        }
    )
    return row, data


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    selected: pd.Series,
    bootstrap: pd.DataFrame,
) -> None:
    internal = metrics[
        metrics["cohort"].eq("internal_development")
        & ~metrics["signal"].astype(str).eq(
            "duration_gated_internal_ranked_ensemble"
        )
    ].sort_values("rr_rmse")
    internal_selected = metrics[
        metrics["cohort"].eq("internal_development")
        & metrics["method"].eq(
            f"calibration_free_fixed_members_prominence_{float(selected['selected_prominence']):g}"
        )
    ].iloc[0]
    external = metrics[
        metrics["cohort"].eq("external_provisional")
        & metrics["method"].eq(
            "calibration_free_fixed_members_internal_selected_prominence"
        )
    ].iloc[0]
    internal_gated = metrics[
        metrics["cohort"].eq("internal_development")
        & metrics["method"].eq(
            "duration_gated_calibration_free_internal_ranked_ensemble"
        )
    ].iloc[0]
    external_gated = metrics[
        metrics["cohort"].eq("external_provisional")
        & metrics["method"].eq(
            "duration_gated_calibration_free_internal_ranked_ensemble"
        )
    ].iloc[0]
    lines = [
        "# Calibration-Free Relative Thermal-Color Signal Probe",
        "",
        "Status: `internal_selected_external_zero_shot_development_probe`",
        "",
        "The color signal and polarity are selected by internal-development RMSE only. "
        "External manual RR is used only after selection for zero-shot scoring.",
        "",
        f"Selected signal: `{selected['signal']}` with `{selected['polarity']}` polarity.",
        f"Selected base prominence: `{float(selected['selected_prominence']):g}`.",
        f"Internal R2 `{internal_selected['rr_r2']:.6f}`, MAE `{internal_selected['rr_mae']:.6f}`, RMSE `{internal_selected['rr_rmse']:.6f}` bpm.",
        f"Provisional external R2 `{external['rr_r2']:.6f}`, MAE `{external['rr_mae']:.6f}`, RMSE `{external['rr_rmse']:.6f}` bpm.",
        "",
        "## Duration-Gated Hybrid",
        "",
        f"Internal R2 `{internal_gated['rr_r2']:.6f}`, MAE `{internal_gated['rr_mae']:.6f}`, RMSE `{internal_gated['rr_rmse']:.6f}` bpm.",
        f"Provisional external R2 `{external_gated['rr_r2']:.6f}`, MAE `{external_gated['rr_mae']:.6f}`, RMSE `{external_gated['rr_rmse']:.6f}` bpm.",
        "The activation threshold is inherited from the internal acquisition-duration "
        "distribution; no external RR label is read by the gate.",
        "",
        "## Source-Session Cluster Bootstrap",
        "",
        "| comparison | metric | estimate | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.estimate:.6f} | "
            f"[{row.ci_low:.6f}, {row.ci_high:.6f}] |"
        )
    lines.extend([
        "",
        "## Internal Ranking",
        "",
        "| signal | polarity | R2 | MAE | RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in internal.itertuples(index=False):
        lines.append(
            f"| {row.signal} | {row.polarity} | {row.rr_r2:.6f} | {row.rr_mae:.6f} | {row.rr_rmse:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a relative pseudo-color signal ablation, not absolute thermometry. "
            "It is promoted only if internal selection transfers to the external cohort "
            "without degrading the frozen duration-normalized comparator.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def duration_gated_predictions(
    ensemble_scored: pd.DataFrame,
    summary: pd.DataFrame,
    selected_polarity: str,
    threshold_seconds: float,
    cohort: str,
) -> pd.DataFrame:
    selected = ensemble_scored[
        ensemble_scored["polarity"].astype(str).eq(selected_polarity)
    ].copy()
    summary_columns = summary[["video_id", "rr_bpm", "raw_duration_seconds"]].copy()
    summary_columns["video_id"] = summary_columns["video_id"].astype(str)
    selected = selected.merge(
        summary_columns,
        on="video_id",
        how="left",
        validate="one_to_one",
    )
    activate = pd.to_numeric(
        selected["raw_duration_seconds"], errors="coerce"
    ).ge(float(threshold_seconds))
    selected["duration_gated_rr_bpm"] = pd.to_numeric(
        selected["rr_bpm_y"], errors="coerce"
    )
    selected.loc[activate, "duration_gated_rr_bpm"] = pd.to_numeric(
        selected.loc[activate, "rr_bpm_x"], errors="coerce"
    )
    selected["duration_gate_activated"] = activate
    selected["cohort"] = cohort
    return selected


def gated_metric(data: pd.DataFrame, cohort: str) -> dict[str, object]:
    row = metric_dict(
        cohort,
        "duration_gated_calibration_free_internal_ranked_ensemble",
        data["truth_rr"].to_numpy(float),
        data["duration_gated_rr_bpm"].to_numpy(float),
        data["truth_count"].to_numpy(float),
        data["truth_duration_seconds"].to_numpy(float),
        "duration_threshold_derived_from_internal_distribution_and_color_ensemble_selected_on_internal_only",
    )
    row.update(
        {
            "signal": "duration_gated_internal_ranked_ensemble",
            "polarity": str(data["polarity"].iloc[0]),
            "selected_members": str(data["selected_members"].iloc[0]),
        }
    )
    return row


def bootstrap_gated_improvement(
    gated: pd.DataFrame,
    reference: pd.DataFrame,
    context_predictions: pd.DataFrame,
    resamples: int = 2000,
    seed: int = 20260711,
) -> pd.DataFrame:
    session_map = reference.rename(columns={"external_video_id": "video_id"})[
        ["video_id", "source_session_id"]
    ]
    data = gated.merge(session_map, on="video_id", how="left", validate="one_to_one")
    context = context_predictions[
        context_predictions["cohort"].astype(str).eq("external_provisional")
    ][
        [
            "video_id",
            "baseline_rr_bpm",
            "windowed_offline_context_rr_bpm",
            "windowed_causal_rr_bpm",
        ]
    ]
    data = data.merge(context, on="video_id", how="left", validate="one_to_one")
    comparisons = {
        "gated_color_minus_frozen_baseline": "baseline_rr_bpm",
        "gated_color_minus_offline_temperature_context": "windowed_offline_context_rr_bpm",
        "gated_color_minus_causal_temperature_context": "windowed_causal_rr_bpm",
    }

    def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
        truth = frame["truth_rr"].to_numpy(float)
        prediction = frame[column].to_numpy(float)
        error = prediction - truth
        denominator = float(np.sum((truth - np.mean(truth)) ** 2))
        return {
            "rr_r2": float(1.0 - np.sum(error**2) / denominator),
            "rr_mae": float(np.mean(np.abs(error))),
            "rr_rmse": float(np.sqrt(np.mean(error**2))),
        }

    groups = sorted(data["source_session_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    samples = {
        comparison: {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
        for comparison in comparisons
    }
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat(
            [data[data["source_session_id"].astype(str).eq(group)] for group in sampled],
            ignore_index=True,
        )
        candidate = metrics(replicate, "duration_gated_rr_bpm")
        for comparison, reference_column in comparisons.items():
            baseline = metrics(replicate, reference_column)
            for metric in samples[comparison]:
                samples[comparison][metric].append(candidate[metric] - baseline[metric])
    candidate_point = metrics(data, "duration_gated_rr_bpm")
    rows = []
    for comparison, reference_column in comparisons.items():
        baseline_point = metrics(data, reference_column)
        for metric, values in samples[comparison].items():
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": candidate_point[metric] - baseline_point[metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "resamples": int(resamples),
                    "cluster_unit": "source_session_id",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    internal_root = args.internal_root.resolve()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    reference = pd.read_csv(args.reference_csv.resolve())
    active = reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    active_ids = set(reference.loc[active, "external_video_id"].astype(str))
    external_summary = pd.read_csv(external_root / f"{args.external_prefix}_summary.csv")
    external_summary = external_summary[external_summary["video_id"].astype(str).isin(active_ids)]
    config = derive_config(internal_summary, args.fps)
    internal_predictions = cohort_predictions(
        internal_root, args.internal_prefix, internal_summary, config,
        args.overwrite_signals, args.radius, "internal_development"
    )
    external_predictions = cohort_predictions(
        external_root, args.external_prefix, external_summary, config,
        args.overwrite_signals, args.radius, "external_provisional"
    )
    internal_metrics, internal_scored = score_candidates(
        internal_predictions, internal_truth(internal_summary), "internal_development"
    )
    ranking = list(
        internal_metrics.sort_values(["rr_rmse", "rr_mae", "signal", "polarity"])[
            ["signal", "polarity"]
        ].itertuples(index=False, name=None)
    )
    external_metrics, external_scored = score_candidates(
        external_predictions, external_truth(reference), "external_provisional"
    )
    internal_ensemble_metrics, internal_ensemble_scored = score_ranked_ensembles(
        internal_predictions,
        internal_truth(internal_summary),
        ranking,
        "internal_development",
    )
    external_ensemble_metrics, external_ensemble_scored = score_ranked_ensembles(
        external_predictions,
        external_truth(reference),
        ranking,
        "external_provisional",
    )
    selected = internal_ensemble_metrics.sort_values(
        ["rr_rmse", "rr_mae", "polarity"]
    ).iloc[0]
    selected_members = [
        tuple(member.split(":")) for member in str(selected["selected_members"]).split(";")
    ]
    prominence_rows = []
    prominence_scored: dict[float, pd.DataFrame] = {}
    for prominence in [0.02, 0.035, 0.05, 0.07]:
        candidate_config = replace(config, base_prominence=float(prominence))
        candidate_predictions = selected_member_predictions(
            internal_root,
            args.internal_prefix,
            internal_summary,
            candidate_config,
            selected_members,
            args.radius,
            "internal_development",
        )
        candidate_metric, candidate_scored = score_fixed_ensemble(
            candidate_predictions,
            internal_truth(internal_summary),
            "internal_development",
            f"calibration_free_fixed_members_prominence_{prominence:g}",
        )
        prominence_rows.append(candidate_metric)
        prominence_scored[float(prominence)] = candidate_scored
    prominence_grid = pd.DataFrame(prominence_rows).sort_values(
        ["rr_rmse", "rr_mae", "selected_prominence"]
    )
    selected_prominence = float(prominence_grid.iloc[0]["selected_prominence"])
    selected_config = replace(config, base_prominence=selected_prominence)
    tuned_internal_scored = prominence_scored[selected_prominence]
    tuned_internal_metric = prominence_grid.iloc[0].to_dict()
    tuned_external_predictions = selected_member_predictions(
        external_root,
        args.external_prefix,
        external_summary,
        selected_config,
        selected_members,
        args.radius,
        "external_provisional",
    )
    tuned_external_metric, tuned_external_scored = score_fixed_ensemble(
        tuned_external_predictions,
        external_truth(reference),
        "external_provisional",
        "calibration_free_fixed_members_internal_selected_prominence",
    )
    selected = selected.copy()
    selected["selected_prominence"] = selected_prominence
    internal_gated = duration_gated_predictions(
        tuned_internal_scored,
        internal_summary,
        str(selected["polarity"]),
        config.long_clip_threshold_seconds,
        "internal_development",
    )
    external_gated = duration_gated_predictions(
        tuned_external_scored,
        external_summary,
        str(selected["polarity"]),
        config.long_clip_threshold_seconds,
        "external_provisional",
    )
    gated_metrics = pd.DataFrame(
        [
            gated_metric(internal_gated, "internal_development"),
            gated_metric(external_gated, "external_provisional"),
        ]
    )
    context_predictions = pd.read_csv(
        output_dir / "paper_duration_normalized_windowed_predictions.csv"
    )
    bootstrap = bootstrap_gated_improvement(
        external_gated, reference, context_predictions
    )
    metrics = pd.concat(
        [
            internal_metrics,
            internal_ensemble_metrics,
            external_metrics,
            external_ensemble_metrics,
            prominence_grid,
            pd.DataFrame([tuned_external_metric]),
            gated_metrics,
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            internal_scored,
            internal_ensemble_scored,
            external_scored,
            external_ensemble_scored,
        ],
        ignore_index=True,
        sort=False,
    )
    metrics.to_csv(output_dir / "paper_calibration_free_thermal_index_metrics.csv", index=False)
    predictions.to_csv(output_dir / "paper_calibration_free_thermal_index_predictions.csv", index=False)
    pd.DataFrame([selected]).to_csv(
        output_dir / "paper_calibration_free_thermal_index_selection.csv", index=False
    )
    prominence_grid.to_csv(
        output_dir / "paper_calibration_free_thermal_index_peak_grid.csv",
        index=False,
    )
    pd.concat([internal_gated, external_gated], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_thermal_index_gated_predictions.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "paper_calibration_free_thermal_index_bootstrap_ci.csv",
        index=False,
    )
    write_report(
        output_dir / "paper_calibration_free_thermal_index_report.md",
        metrics,
        selected,
        bootstrap,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
