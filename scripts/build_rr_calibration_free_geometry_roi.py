from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_rr_calibration_free_consensus_ensemble import (
    aggregate_members,
    candidate_metrics,
    cluster_bootstrap_vs_p2g,
    duration_gated,
    gated_metric,
    selected_members,
)
from build_rr_calibration_free_thermal_index import (
    SIGNALS,
    estimate_curve,
    extract_video_signals,
    fused_curve,
    roi_means,
)
from build_rr_duration_normalized_windowed_innovation import (
    derive_config,
    external_truth,
    internal_truth,
)
from paper_repro_rr import frame_image_paths


ROI_STRATEGIES = {
    "fixed_20px": None,
    "inter_nostril_0.12": 0.12,
    "inter_nostril_0.16": 0.16,
    "inter_nostril_0.20": 0.20,
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate internal-selected inter-nostril geometry-normalized circular ROIs "
            "for calibration-free relative thermal-color RR."
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
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--minimum-radius", type=int, default=8)
    parser.add_argument("--maximum-radius", type=int, default=48)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    return parser.parse_args()


def numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table.get(column, pd.Series(np.nan, index=table.index)), errors="coerce")


def median_inter_nostril_distance(temperature: pd.DataFrame) -> float:
    left_x, left_y = numeric(temperature, "left_x"), numeric(temperature, "left_y")
    right_x, right_y = numeric(temperature, "right_x"), numeric(temperature, "right_y")
    distance = np.hypot(left_x - right_x, left_y - right_y)
    valid = distance[np.isfinite(distance) & (distance > 1.0)]
    return float(np.median(valid)) if len(valid) else math.nan


def strategy_radius(
    strategy: str,
    median_distance: float,
    minimum_radius: int,
    maximum_radius: int,
) -> int:
    ratio = ROI_STRATEGIES[strategy]
    if ratio is None or not np.isfinite(median_distance):
        return 20
    return int(np.clip(round(float(ratio) * median_distance), minimum_radius, maximum_radius))


def all_strategy_signals(
    video_dir: Path,
    prefix: str,
    strategies: list[str],
    minimum_radius: int,
    maximum_radius: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    temperature = pd.read_csv(video_dir / f"{prefix}_temperatures.csv")
    median_distance = median_inter_nostril_distance(temperature)
    radii = {
        strategy: strategy_radius(strategy, median_distance, minimum_radius, maximum_radius)
        for strategy in strategies
    }
    tables: dict[str, pd.DataFrame] = {}
    if "fixed_20px" in strategies:
        tables["fixed_20px"] = extract_video_signals(video_dir, prefix, 20, False)
    scaled_strategies = [strategy for strategy in strategies if strategy != "fixed_20px"]
    rows = {strategy: [] for strategy in scaled_strategies}
    if scaled_strategies:
        paths = {path.name: path for path in frame_image_paths(video_dir)}
        for row in temperature.itertuples(index=False):
            image_path = paths.get(str(row.frame_name))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path is not None else None
            for strategy in scaled_strategies:
                radius = radii[strategy]
                output: dict[str, object] = {
                    "frame_name": row.frame_name,
                    "roi_radius_px": radius,
                }
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
                rows[strategy].append(output)
    geometry = {
        "video_id": video_dir.name,
        "median_inter_nostril_distance_px": median_distance,
        **{f"{strategy}_radius_px": radius for strategy, radius in radii.items()},
    }
    tables.update({strategy: pd.DataFrame(values) for strategy, values in rows.items()})
    return tables, geometry


def strategy_member_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: object,
    members: list[tuple[str, str]],
    cohort: str,
    strategies: list[str],
    minimum_radius: int,
    maximum_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        print(f"[{cohort} {index}/{len(summary)}] {video_id}")
        tables, geometry = all_strategy_signals(
            root / video_id, prefix, strategies, minimum_radius, maximum_radius
        )
        geometry["cohort"] = cohort
        geometry_rows.append(geometry)
        for strategy in strategies:
            for signal, polarity in members:
                curve = fused_curve(tables[strategy], signal, polarity == "inverted", config)
                rr_bpm, windows, quality = estimate_curve(curve, config)
                prediction_rows.append(
                    {
                        "cohort": cohort,
                        "video_id": video_id,
                        "strategy": strategy,
                        "member": f"{signal}:{polarity}",
                        "rr_bpm": rr_bpm,
                        "windows": windows,
                        "quality_mean": quality,
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(geometry_rows)


def write_report(
    path: Path,
    selection: pd.Series,
    gated_metrics: pd.DataFrame,
    geometry: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    internal = gated_metrics[gated_metrics["cohort"].eq("internal_development")].iloc[0]
    external = gated_metrics[gated_metrics["cohort"].eq("external_provisional")].iloc[0]
    internal_geometry = geometry[geometry["cohort"].eq("internal_development")]
    external_geometry = geometry[geometry["cohort"].eq("external_provisional")]
    selected_radius_column = f"{selection['strategy']}_radius_px"
    lines = [
        "# Geometry-Normalized Relative Thermal-Color ROI Probe",
        "",
        "Status: `internal_selected_external_zero_shot_development_probe`",
        "",
        "The selected ROI radius is either the frozen 20 px baseline or a bounded fraction "
        "of the within-video median inter-nostril distance. Selection uses internal color-only "
        "RMSE; external manual RR is read only after the ROI strategy is frozen.",
        "",
        f"Selected ROI strategy: `{selection['strategy']}`.",
        f"Internal selected radius mean: `{internal_geometry[selected_radius_column].mean():.3f}` px.",
        f"External selected radius mean: `{external_geometry[selected_radius_column].mean():.3f}` px.",
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
            "This is a relative pseudo-color ROI-scale experiment, not confirmatory external "
            "validation. The Jiufu reference remains single-annotator and development-inspected.",
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
    members, prominence = selected_members(pd.read_csv(args.p2g_selection_csv.resolve()))
    members_text = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    reference = pd.read_csv(args.reference_csv.resolve())
    active = reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    active_ids = set(reference.loc[active, "external_video_id"].astype(str))
    external_summary = pd.read_csv(external_root / f"{args.external_prefix}_summary.csv")
    external_summary = external_summary[external_summary["video_id"].astype(str).isin(active_ids)].copy()
    config = derive_config(internal_summary, args.fps)
    config = config.__class__(**{**config.__dict__, "base_prominence": prominence})
    internal_truth_table = internal_truth(internal_summary)
    external_truth_table = external_truth(reference)
    strategies = list(ROI_STRATEGIES)
    internal_members, internal_geometry = strategy_member_predictions(
        internal_root,
        args.internal_prefix,
        internal_summary,
        config,
        members,
        "internal_development",
        strategies,
        args.minimum_radius,
        args.maximum_radius,
    )
    metric_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = [internal_members]
    internal_aggregated: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        aggregated = aggregate_members(
            internal_members[internal_members["strategy"].eq(strategy)], members, "mean"
        )
        metrics, scored = candidate_metrics(
            aggregated, internal_truth_table, "internal_development", members_text
        )
        metrics["strategy"] = strategy
        metric_rows.append(metrics)
        prediction_frames.append(scored.assign(prediction_kind="color_only", strategy=strategy))
        internal_aggregated[strategy] = aggregated
    candidates = pd.concat(metric_rows, ignore_index=True).sort_values(
        ["rr_rmse", "rr_mae", "strategy"]
    )
    selection = candidates.iloc[0].copy()
    strategy = str(selection["strategy"])
    external_members, external_geometry = strategy_member_predictions(
        external_root,
        args.external_prefix,
        external_summary,
        config,
        members,
        "external_provisional",
        [strategy],
        args.minimum_radius,
        args.maximum_radius,
    )
    external_aggregated = aggregate_members(external_members, members, "mean")
    external_metrics, external_scored_color = candidate_metrics(
        external_aggregated, external_truth_table, "external_provisional", members_text
    )
    external_metrics["strategy"] = strategy
    internal_gated = duration_gated(
        internal_aggregated[strategy], internal_summary, config.long_clip_threshold_seconds
    )
    external_gated = duration_gated(
        external_aggregated, external_summary, config.long_clip_threshold_seconds
    )
    internal_gated_metric, internal_gated_scored = gated_metric(
        internal_gated,
        internal_truth_table,
        "internal_development",
        "mean",
        members_text,
        config.long_clip_threshold_seconds,
    )
    external_gated_metric, external_gated_scored = gated_metric(
        external_gated,
        external_truth_table,
        "external_provisional",
        "mean",
        members_text,
        config.long_clip_threshold_seconds,
    )
    for row in [internal_gated_metric, external_gated_metric]:
        row["strategy"] = strategy
    gated_metrics = pd.DataFrame([internal_gated_metric, external_gated_metric])
    bootstrap = cluster_bootstrap_vs_p2g(
        external_gated_scored,
        reference,
        pd.read_csv(args.p2g_gated_predictions_csv.resolve()),
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    pd.concat([candidates, external_metrics, gated_metrics], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_geometry_roi_metrics.csv", index=False
    )
    selection.to_frame().T.to_csv(
        output_dir / "paper_calibration_free_geometry_roi_selection.csv", index=False
    )
    pd.concat(
        [
            *prediction_frames,
            external_members,
            external_scored_color.assign(prediction_kind="color_only", strategy=strategy),
            internal_gated_scored.assign(prediction_kind="duration_gated_selected", strategy=strategy),
            external_gated_scored.assign(prediction_kind="duration_gated_selected", strategy=strategy),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "paper_calibration_free_geometry_roi_predictions.csv", index=False)
    pd.concat([internal_geometry, external_geometry], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_geometry_roi_summary.csv", index=False
    )
    bootstrap.to_csv(
        output_dir / "paper_calibration_free_geometry_roi_bootstrap_ci.csv", index=False
    )
    write_report(
        output_dir / "paper_calibration_free_geometry_roi_report.md",
        selection,
        gated_metrics,
        pd.concat([internal_geometry, external_geometry], ignore_index=True),
        bootstrap,
    )
    print("Selected strategy:")
    print(selection.to_string())
    print("\nGated metrics:")
    print(gated_metrics.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
