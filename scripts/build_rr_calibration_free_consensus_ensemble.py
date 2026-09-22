from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_rr_calibration_free_thermal_index import (
    estimate_curve,
    extract_video_signals,
    fused_curve,
)
from build_rr_duration_normalized_windowed_innovation import (
    derive_config,
    external_truth,
    internal_truth,
    metric_dict,
)


AGGREGATIONS = [
    "mean",
    "median",
    "trimmed_mean_1",
    "trimmed_mean_2",
    "quality_weighted_mean",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate internal-selected robust aggregation of calibration-free relative "
            "thermal-color RR channels."
        )
    )
    parser.add_argument("--internal-root", type=Path, default=repo / "Dataset_new" / "72video" / "al_images")
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
    )
    parser.add_argument(
        "--p2g-selection-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_selection.csv",
    )
    parser.add_argument(
        "--p2g-gated-predictions-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_gated_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    return parser.parse_args()


def selected_members(selection: pd.DataFrame) -> tuple[list[tuple[str, str]], float]:
    if selection.empty:
        raise ValueError("P2g selection table is empty")
    row = selection.iloc[0]
    members = []
    for item in str(row["selected_members"]).split(";"):
        signal, polarity = item.split(":", maxsplit=1)
        members.append((signal, polarity))
    prominence = float(pd.to_numeric(pd.Series([row["selected_prominence"]]), errors="coerce").iloc[0])
    if not members or not np.isfinite(prominence):
        raise ValueError("P2g selection lacks channel members or peak prominence")
    return members, prominence


def member_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: object,
    members: list[tuple[str, str]],
    radius: int,
    cohort: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        print(f"[{cohort} {index}/{len(summary)}] {video_id}")
        signals = extract_video_signals(root / video_id, prefix, radius, False)
        for signal, polarity in members:
            curve = fused_curve(signals, signal, polarity == "inverted", config)
            rr_bpm, windows, quality = estimate_curve(curve, config)
            rows.append(
                {
                    "cohort": cohort,
                    "video_id": video_id,
                    "member": f"{signal}:{polarity}",
                    "rr_bpm": rr_bpm,
                    "windows": windows,
                    "quality_mean": quality,
                }
            )
    return pd.DataFrame(rows)


def aggregation_value(values: np.ndarray, qualities: np.ndarray, aggregation: str) -> float:
    valid = np.isfinite(values)
    values = values[valid]
    qualities = qualities[valid]
    if not len(values):
        return math.nan
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "median":
        return float(np.median(values))
    if aggregation.startswith("trimmed_mean_"):
        trim = int(aggregation.rsplit("_", maxsplit=1)[1])
        if len(values) <= 2 * trim:
            return float(np.median(values))
        center = float(np.median(values))
        keep = np.argsort(np.abs(values - center))[: len(values) - trim]
        return float(np.mean(values[keep]))
    if aggregation == "quality_weighted_mean":
        finite_quality = np.isfinite(qualities)
        if finite_quality.sum() < 2:
            return float(np.mean(values))
        center = float(np.median(qualities[finite_quality]))
        scale = float(np.median(np.abs(qualities[finite_quality] - center)))
        if scale <= 1e-8:
            return float(np.mean(values))
        weights = np.exp(np.clip((qualities - center) / scale, -2.0, 2.0))
        weights[~np.isfinite(weights)] = 1.0
        return float(np.average(values, weights=weights))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def aggregate_members(
    predictions: pd.DataFrame,
    members: list[tuple[str, str]],
    aggregation: str,
) -> pd.DataFrame:
    member_names = [f"{signal}:{polarity}" for signal, polarity in members]
    rr_wide = predictions.pivot(index="video_id", columns="member", values="rr_bpm").reindex(columns=member_names)
    quality_wide = (
        predictions.pivot(index="video_id", columns="member", values="quality_mean")
        .reindex(columns=member_names)
        .reindex(rr_wide.index)
    )
    rows = []
    for video_id in rr_wide.index:
        values = rr_wide.loc[video_id].to_numpy(float)
        qualities = quality_wide.loc[video_id].to_numpy(float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "video_id": str(video_id),
                "aggregation": aggregation,
                "color_rr_bpm": aggregation_value(values, qualities, aggregation),
                "valid_members": int(np.isfinite(values).sum()),
                "channel_rr_median": float(np.median(finite)) if len(finite) else math.nan,
                "channel_rr_iqr": (
                    float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
                    if len(finite) >= 2
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_metrics(
    aggregated: pd.DataFrame,
    truth: pd.DataFrame,
    cohort: str,
    members_text: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = aggregated.merge(truth, on="video_id", how="inner", validate="one_to_one")
    rows = []
    for aggregation, group in data.groupby("aggregation", sort=False):
        row = metric_dict(
            cohort,
            f"calibration_free_consensus_{aggregation}",
            group["truth_rr"].to_numpy(float),
            group["color_rr_bpm"].to_numpy(float),
            group["truth_count"].to_numpy(float),
            group["truth_duration_seconds"].to_numpy(float),
            "aggregation_selected_by_internal_color_only_rmse",
        )
        row.update({"aggregation": aggregation, "selected_members": members_text})
        rows.append(row)
    return pd.DataFrame(rows), data


def duration_gated(
    aggregated: pd.DataFrame,
    summary: pd.DataFrame,
    threshold_seconds: float,
) -> pd.DataFrame:
    duration_columns = [
        column
        for column in ["raw_duration_seconds", "duration_seconds", "rr_duration_seconds"]
        if column in summary.columns
    ]
    if not duration_columns:
        raise ValueError("Summary lacks every duration column required for duration gating")
    base = summary[["video_id", "rr_bpm", *duration_columns]].copy()
    base["video_id"] = base["video_id"].astype(str)
    base = base.rename(columns={"rr_bpm": "baseline_rr_bpm"})
    raw_duration = pd.to_numeric(
        base.get("raw_duration_seconds", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    )
    frame_duration = pd.to_numeric(
        base.get("duration_seconds", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    )
    rr_duration = pd.to_numeric(
        base.get("rr_duration_seconds", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    )
    base["duration_gate_seconds"] = raw_duration.combine_first(frame_duration).combine_first(
        rr_duration
    )
    base["duration_gate_source"] = np.select(
        [raw_duration.notna(), frame_duration.notna(), rr_duration.notna()],
        ["raw_duration_seconds", "duration_seconds", "rr_duration_seconds"],
        default="missing",
    )
    data = aggregated.merge(base, on="video_id", how="left", validate="one_to_one")
    duration = pd.to_numeric(data["duration_gate_seconds"], errors="coerce")
    use_color = duration.ge(float(threshold_seconds))
    data["duration_gate_activated"] = use_color
    data["duration_gated_rr_bpm"] = data["baseline_rr_bpm"]
    data.loc[use_color, "duration_gated_rr_bpm"] = data.loc[use_color, "color_rr_bpm"]
    return data


def gated_metric(
    data: pd.DataFrame,
    truth: pd.DataFrame,
    cohort: str,
    aggregation: str,
    members_text: str,
    threshold_seconds: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    scored = data.merge(truth, on="video_id", how="inner", validate="one_to_one")
    row = metric_dict(
        cohort,
        "duration_gated_calibration_free_consensus_ensemble",
        scored["truth_rr"].to_numpy(float),
        scored["duration_gated_rr_bpm"].to_numpy(float),
        scored["truth_count"].to_numpy(float),
        scored["truth_duration_seconds"].to_numpy(float),
        "aggregation_selected_on_internal_color_only_rmse_and_duration_gate_inherited_from_internal_distribution",
    )
    row.update(
        {
            "aggregation": aggregation,
            "selected_members": members_text,
            "duration_threshold_seconds": float(threshold_seconds),
        }
    )
    return row, scored


def metric_values(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    truth = frame["truth_rr"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
    }


def cluster_bootstrap_vs_p2g(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
    p2g_gated: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    p2g = p2g_gated[p2g_gated["cohort"].astype(str).eq("external_provisional")][
        ["video_id", "duration_gated_rr_bpm"]
    ].rename(columns={"duration_gated_rr_bpm": "mean_p2g_rr_bpm"})
    session_map = reference.rename(columns={"external_video_id": "video_id"})[
        ["video_id", "source_session_id"]
    ]
    data = selected.merge(p2g, on="video_id", how="inner", validate="one_to_one")
    data = data.merge(session_map, on="video_id", how="inner", validate="one_to_one")
    groups = sorted(data["source_session_id"].astype(str).unique())
    if not groups:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat(
            [data[data["source_session_id"].astype(str).eq(group)] for group in sampled],
            ignore_index=True,
        )
        candidate = metric_values(replicate, "duration_gated_rr_bpm")
        baseline = metric_values(replicate, "mean_p2g_rr_bpm")
        for metric in samples:
            samples[metric].append(candidate[metric] - baseline[metric])
    candidate_point = metric_values(data, "duration_gated_rr_bpm")
    baseline_point = metric_values(data, "mean_p2g_rr_bpm")
    return pd.DataFrame(
        [
            {
                "comparison": "robust_consensus_minus_mean_p2g",
                "metric": metric,
                "estimate": candidate_point[metric] - baseline_point[metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "resamples": int(resamples),
                "cluster_unit": "source_session_id",
            }
            for metric, values in samples.items()
        ]
    )


def write_report(
    path: Path,
    selection: pd.Series,
    gated_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    internal = gated_metrics[gated_metrics["cohort"].eq("internal_development")].iloc[0]
    external = gated_metrics[gated_metrics["cohort"].eq("external_provisional")].iloc[0]
    lines = [
        "# Calibration-Free Relative Thermal-Color Consensus Probe",
        "",
        "Status: `internal_selected_external_zero_shot_development_probe`",
        "",
        "The aggregation rule is selected by internal color-only RMSE. External manual RR "
        "is read only after the aggregation is frozen for zero-shot scoring.",
        "",
        f"Selected aggregation: `{selection['aggregation']}`.",
        f"Selected members: `{selection['selected_members']}`.",
        f"Internal duration-gated R2 `{internal['rr_r2']:.6f}`, MAE `{internal['rr_mae']:.6f}`, "
        f"RMSE `{internal['rr_rmse']:.6f}` bpm.",
        f"Provisional external duration-gated R2 `{external['rr_r2']:.6f}`, "
        f"MAE `{external['rr_mae']:.6f}`, RMSE `{external['rr_rmse']:.6f}` bpm.",
        "",
        "## Source-Session Cluster Bootstrap Against Mean P2g",
        "",
        "| metric | estimate | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.estimate:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "This probe tests robust cross-channel consensus for relative pseudo-color signals. "
            "It is not confirmatory external validation because the current external reference "
            "is single-annotator and was inspected during development.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    internal_root = args.internal_root.resolve()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_table = pd.read_csv(args.p2g_selection_csv.resolve())
    members, prominence = selected_members(selection_table)
    members_text = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    reference = pd.read_csv(args.reference_csv.resolve())
    active = reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    active_ids = set(reference.loc[active, "external_video_id"].astype(str))
    external_summary = pd.read_csv(external_root / f"{args.external_prefix}_summary.csv")
    external_summary = external_summary[external_summary["video_id"].astype(str).isin(active_ids)].copy()
    config = derive_config(internal_summary, args.fps)
    config = config.__class__(**{**config.__dict__, "base_prominence": prominence})
    internal_members = member_predictions(
        internal_root, args.internal_prefix, internal_summary, config, members, args.radius, "internal_development"
    )
    external_members = member_predictions(
        external_root, args.external_prefix, external_summary, config, members, args.radius, "external_provisional"
    )
    internal_truth_table = internal_truth(internal_summary)
    external_truth_table = external_truth(reference)
    metric_frames = []
    prediction_frames = [internal_members, external_members]
    candidate_data: dict[tuple[str, str], pd.DataFrame] = {}
    for cohort, raw, truth in [
        ("internal_development", internal_members, internal_truth_table),
        ("external_provisional", external_members, external_truth_table),
    ]:
        for aggregation in AGGREGATIONS:
            aggregated = aggregate_members(raw, members, aggregation)
            metrics, scored = candidate_metrics(aggregated, truth, cohort, members_text)
            metric_frames.append(metrics)
            prediction_frames.append(scored.assign(prediction_kind="color_only"))
            candidate_data[(cohort, aggregation)] = aggregated
    candidate_metric_table = pd.concat(metric_frames, ignore_index=True)
    internal_candidates = candidate_metric_table[
        candidate_metric_table["cohort"].eq("internal_development")
    ].sort_values(["rr_rmse", "rr_mae", "aggregation"])
    selection = internal_candidates.iloc[0].copy()
    aggregation = str(selection["aggregation"])
    internal_gated = duration_gated(
        candidate_data[("internal_development", aggregation)],
        internal_summary,
        config.long_clip_threshold_seconds,
    )
    external_gated = duration_gated(
        candidate_data[("external_provisional", aggregation)],
        external_summary,
        config.long_clip_threshold_seconds,
    )
    internal_gated_metric, internal_gated_scored = gated_metric(
        internal_gated,
        internal_truth_table,
        "internal_development",
        aggregation,
        members_text,
        config.long_clip_threshold_seconds,
    )
    external_gated_metric, external_gated_scored = gated_metric(
        external_gated,
        external_truth_table,
        "external_provisional",
        aggregation,
        members_text,
        config.long_clip_threshold_seconds,
    )
    gated_metrics = pd.DataFrame([internal_gated_metric, external_gated_metric])
    p2g_gated = pd.read_csv(args.p2g_gated_predictions_csv.resolve())
    bootstrap = cluster_bootstrap_vs_p2g(
        external_gated_scored,
        reference,
        p2g_gated,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    selection.to_frame().T.to_csv(
        output_dir / "paper_calibration_free_consensus_ensemble_selection.csv", index=False
    )
    pd.concat([candidate_metric_table, gated_metrics], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_consensus_ensemble_metrics.csv", index=False
    )
    pd.concat(
        [*prediction_frames, internal_gated_scored.assign(prediction_kind="duration_gated_selected"), external_gated_scored.assign(prediction_kind="duration_gated_selected")],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "paper_calibration_free_consensus_ensemble_predictions.csv", index=False)
    bootstrap.to_csv(
        output_dir / "paper_calibration_free_consensus_ensemble_bootstrap_ci.csv", index=False
    )
    write_report(
        output_dir / "paper_calibration_free_consensus_ensemble_report.md",
        selection,
        gated_metrics,
        bootstrap,
    )
    print("Selected aggregation:")
    print(selection.to_string())
    print("\nGated metrics:")
    print(gated_metrics.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
