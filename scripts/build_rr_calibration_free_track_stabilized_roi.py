from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_rr_calibration_free_consensus_ensemble import (
    aggregate_members,
    cluster_bootstrap_vs_p2g,
    duration_gated,
    gated_metric,
    selected_members,
)
from build_rr_calibration_free_thermal_index import SIGNALS, estimate_curve, fused_curve, roi_means
from build_rr_duration_normalized_windowed_innovation import derive_config, external_truth, internal_truth
from paper_repro_rr import frame_image_paths


STRATEGIES: dict[str, tuple[int, float | None]] = {
    "raw_fixed20": (1, None),
    "bilateral_median3_fixed20": (3, None),
    "bilateral_median5_fixed20": (5, None),
    "bilateral_median5_dynamic015": (5, 0.15),
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Select an internal-only bilateral trajectory-stabilized ROI for calibration-free "
            "thermal-color RR, then make a zero-shot external development probe."
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


def smooth_coordinate(values: pd.Series, window: int) -> pd.Series:
    value = pd.to_numeric(values, errors="coerce")
    if window <= 1:
        return value
    # Only bridge detector gaps that are no longer than the local stabilization window.
    bridged = value.interpolate(limit=window, limit_direction="both")
    return bridged.rolling(window=window, center=True, min_periods=1).median()


def bilateral_track(temperature: pd.DataFrame, window: int) -> pd.DataFrame:
    left_x, left_y = numeric(temperature, "left_x"), numeric(temperature, "left_y")
    right_x, right_y = numeric(temperature, "right_x"), numeric(temperature, "right_y")
    both = left_x.notna() & left_y.notna() & right_x.notna() & right_y.notna()
    median_dx = float((right_x[both] - left_x[both]).median()) if both.any() else math.nan
    median_dy = float((right_y[both] - left_y[both]).median()) if both.any() else math.nan
    if not np.isfinite(median_dx) or not np.isfinite(median_dy):
        median_dx, median_dy = 0.0, 0.0

    center_x = pd.Series(np.nan, index=temperature.index, dtype=float)
    center_y = pd.Series(np.nan, index=temperature.index, dtype=float)
    center_x.loc[both] = (left_x.loc[both] + right_x.loc[both]) / 2.0
    center_y.loc[both] = (left_y.loc[both] + right_y.loc[both]) / 2.0
    left_only = left_x.notna() & left_y.notna() & ~both
    right_only = right_x.notna() & right_y.notna() & ~both
    center_x.loc[left_only] = left_x.loc[left_only] + median_dx / 2.0
    center_y.loc[left_only] = left_y.loc[left_only] + median_dy / 2.0
    center_x.loc[right_only] = right_x.loc[right_only] - median_dx / 2.0
    center_y.loc[right_only] = right_y.loc[right_only] - median_dy / 2.0

    vector_x = (right_x - left_x).where(both)
    vector_y = (right_y - left_y).where(both)
    stable_center_x = smooth_coordinate(center_x, window)
    stable_center_y = smooth_coordinate(center_y, window)
    stable_vector_x = smooth_coordinate(vector_x, window).fillna(median_dx)
    stable_vector_y = smooth_coordinate(vector_y, window).fillna(median_dy)
    valid = stable_center_x.notna() & stable_center_y.notna()
    distance = np.hypot(stable_vector_x, stable_vector_y)
    return pd.DataFrame(
        {
            "left_x": (stable_center_x - stable_vector_x / 2.0).where(valid),
            "left_y": (stable_center_y - stable_vector_y / 2.0).where(valid),
            "right_x": (stable_center_x + stable_vector_x / 2.0).where(valid),
            "right_y": (stable_center_y + stable_vector_y / 2.0).where(valid),
            "inter_nostril_distance": distance.where(valid),
            "raw_pair_available": both,
            "stable_pair_available": valid,
        }
    )


def jitter(values: pd.Series) -> float:
    value = pd.to_numeric(values, errors="coerce").to_numpy(float)
    value = value[np.isfinite(value)]
    return float(np.median(np.abs(np.diff(value)))) if len(value) >= 2 else math.nan


def signals_for_strategies(
    video_dir: Path,
    prefix: str,
    strategies: list[str],
    minimum_radius: int,
    maximum_radius: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    temperature = pd.read_csv(video_dir / f"{prefix}_temperatures.csv")
    paths = {path.name: path for path in frame_image_paths(video_dir)}
    tracks = {
        strategy: bilateral_track(temperature, STRATEGIES[strategy][0]) for strategy in strategies
    }
    rows = {strategy: [] for strategy in strategies}
    for index, row in enumerate(temperature.itertuples(index=False)):
        image_path = paths.get(str(row.frame_name))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path is not None else None
        for strategy in strategies:
            window, dynamic_ratio = STRATEGIES[strategy]
            track = tracks[strategy].iloc[index]
            distance = float(track.inter_nostril_distance)
            radius = 20 if dynamic_ratio is None or not np.isfinite(distance) else int(
                np.clip(round(dynamic_ratio * distance), minimum_radius, maximum_radius)
            )
            output: dict[str, object] = {"frame_name": row.frame_name, "roi_radius_px": radius}
            if image is None:
                for side in ["left", "right"]:
                    for signal in SIGNALS:
                        output[f"{side}_{signal}"] = math.nan
            else:
                for side in ["left", "right"]:
                    values = roi_means(image, track[f"{side}_x"], track[f"{side}_y"], radius)
                    for signal, value in values.items():
                        output[f"{side}_{signal}"] = value
            rows[strategy].append(output)

    raw_track = bilateral_track(temperature, 1)
    diagnostics = pd.DataFrame(
        [
            {
                "video_id": video_dir.name,
                "raw_pair_available_fraction": float(raw_track["raw_pair_available"].mean()),
                "raw_pair_coordinate_jitter_px": float(
                    np.nanmean(
                        [
                            jitter(numeric(temperature, "left_x")),
                            jitter(numeric(temperature, "left_y")),
                            jitter(numeric(temperature, "right_x")),
                            jitter(numeric(temperature, "right_y")),
                        ]
                    )
                ),
                **{
                    f"{strategy}_stable_pair_available_fraction": float(
                        tracks[strategy]["stable_pair_available"].mean()
                    )
                    for strategy in strategies
                },
                **{
                    f"{strategy}_median_radius_px": float(
                        pd.DataFrame(rows[strategy])["roi_radius_px"].median()
                    )
                    for strategy in strategies
                },
            }
        ]
    )
    return {strategy: pd.DataFrame(values) for strategy, values in rows.items()}, diagnostics


def member_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: object,
    members: list[tuple[str, str]],
    strategies: list[str],
    minimum_radius: int,
    maximum_radius: int,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    diagnostics: list[pd.DataFrame] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        print(f"[{cohort} {index}/{len(summary)}] {video_id}")
        signal_tables, table_diagnostics = signals_for_strategies(
            root / video_id,
            prefix,
            strategies,
            minimum_radius,
            maximum_radius,
        )
        table_diagnostics["cohort"] = cohort
        diagnostics.append(table_diagnostics)
        for strategy, table in signal_tables.items():
            for signal, polarity in members:
                curve = fused_curve(table, signal, polarity == "inverted", config)
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
    return pd.DataFrame(prediction_rows), pd.concat(diagnostics, ignore_index=True)


def select_internal_strategy(
    member_table: pd.DataFrame,
    internal_summary: pd.DataFrame,
    truth: pd.DataFrame,
    members: list[tuple[str, str]],
    duration_threshold_seconds: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    member_text = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    rows: list[dict[str, object]] = []
    aggregates: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        aggregate = aggregate_members(
            member_table[member_table["strategy"].eq(strategy)], members, "mean"
        )
        aggregates[strategy] = aggregate
        gated = duration_gated(aggregate, internal_summary, duration_threshold_seconds)
        metric, _ = gated_metric(
            gated,
            truth,
            "internal_development",
            "mean",
            member_text,
            duration_threshold_seconds,
        )
        metric["strategy"] = strategy
        metric["selection_boundary"] = "internal_development_only"
        rows.append(metric)
    return pd.DataFrame(rows).sort_values(["rr_rmse", "rr_mae", "strategy"]), aggregates


def write_report(
    path: Path,
    candidates: pd.DataFrame,
    selected: pd.Series,
    external_metric: dict[str, object],
    bootstrap: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# Bilateral Trajectory-Stabilized Relative Thermal-Color ROI Probe",
        "",
        "Status: `internal_selected_zero_shot_external_development_probe`",
        "",
        "The candidate reconstructs each nostril from a temporally median-filtered bilateral "
        "midpoint and inter-nostril vector before sampling the same internally selected, "
        "relative thermal-color channels. The trajectory window and dynamic-radius option are "
        "selected only on the 73-video internal development cohort.",
        "",
        f"Selected strategy: `{selected['strategy']}`.",
        f"Internal selected RR R2: `{selected['rr_r2']:.6f}`; MAE: `{selected['rr_mae']:.6f}` bpm; "
        f"RMSE: `{selected['rr_rmse']:.6f}` bpm.",
        f"Provisional external RR R2: `{external_metric['rr_r2']:.6f}`; "
        f"MAE: `{external_metric['rr_mae']:.6f}` bpm; RMSE: `{external_metric['rr_rmse']:.6f}` bpm.",
        "",
        "## Internal Candidate Selection",
        "",
        "| strategy | RR R2 | MAE | RMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in candidates.itertuples(index=False):
        lines.append(f"| {row.strategy} | {row.rr_r2:.6f} | {row.rr_mae:.6f} | {row.rr_rmse:.6f} |")
    lines.extend(
        [
            "",
            "## Track Diagnostics",
            "",
            f"Internal raw bilateral-pair availability: `{diagnostics[diagnostics['cohort'].eq('internal_development')]['raw_pair_available_fraction'].mean():.4f}`.",
            f"External raw bilateral-pair availability: `{diagnostics[diagnostics['cohort'].eq('external_provisional')]['raw_pair_available_fraction'].mean():.4f}`.",
            "",
            "## Source-Session Bootstrap Versus Frozen P2g",
            "",
            "| metric | estimate | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.estimate:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |")
    lines.extend(
        [
            "",
            "The Jiufu labels are single-annotator and have already been inspected during "
            "development. These external values are exploratory only. Do not replace the frozen "
            "P2g primary prediction or report this as confirmatory external validation; a newly "
            "frozen prediction file must be scored against blinded A/B consensus or an untouched "
            "external holdout before a publication claim.",
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
    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    reference = pd.read_csv(args.reference_csv.resolve())
    external_ids = set(
        reference.loc[
            reference["primary_analysis_include"].astype(str).str.lower().eq("true"),
            "external_video_id",
        ].astype(str)
    )
    external_summary = pd.read_csv(external_root / f"{args.external_prefix}_summary.csv")
    external_summary = external_summary[external_summary["video_id"].astype(str).isin(external_ids)].copy()
    config = derive_config(internal_summary, args.fps)
    config = config.__class__(**{**config.__dict__, "base_prominence": prominence})
    internal_truth_table = internal_truth(internal_summary)
    external_truth_table = external_truth(reference)

    internal_members, internal_diagnostics = member_predictions(
        internal_root,
        args.internal_prefix,
        internal_summary,
        config,
        members,
        list(STRATEGIES),
        args.minimum_radius,
        args.maximum_radius,
        "internal_development",
    )
    candidates, aggregates = select_internal_strategy(
        internal_members,
        internal_summary,
        internal_truth_table,
        members,
        config.long_clip_threshold_seconds,
    )
    selected = candidates.iloc[0]
    strategy = str(selected["strategy"])

    external_members, external_diagnostics = member_predictions(
        external_root,
        args.external_prefix,
        external_summary,
        config,
        members,
        [strategy],
        args.minimum_radius,
        args.maximum_radius,
        "external_provisional",
    )
    external_aggregate = aggregate_members(external_members, members, "mean")
    external_gated = duration_gated(
        external_aggregate, external_summary, config.long_clip_threshold_seconds
    )
    external_metric, external_scored = gated_metric(
        external_gated,
        external_truth_table,
        "external_provisional",
        "mean",
        ";".join(f"{signal}:{polarity}" for signal, polarity in members),
        config.long_clip_threshold_seconds,
    )
    external_metric["strategy"] = strategy
    external_metric["evaluation_status"] = "provisional_single_annotator_development_probe"
    external_metric["confirmatory_status"] = "not_confirmatory_new_candidate"
    baseline = pd.read_csv(args.p2g_gated_predictions_csv.resolve())
    bootstrap = cluster_bootstrap_vs_p2g(
        external_scored,
        reference,
        baseline,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    bootstrap["comparison"] = "bilateral_trajectory_stabilized_minus_frozen_p2g"
    diagnostics = pd.concat([internal_diagnostics, external_diagnostics], ignore_index=True)

    candidates.to_csv(output_dir / "paper_calibration_free_track_stabilized_roi_selection.csv", index=False)
    pd.DataFrame([external_metric]).to_csv(
        output_dir / "paper_calibration_free_track_stabilized_roi_metrics.csv", index=False
    )
    pd.concat(
        [
            internal_members.assign(prediction_kind="internal_candidate_member"),
            external_members.assign(prediction_kind="external_selected_member"),
            external_scored.assign(prediction_kind="external_selected_duration_gated"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "paper_calibration_free_track_stabilized_roi_predictions.csv", index=False)
    diagnostics.to_csv(output_dir / "paper_calibration_free_track_stabilized_roi_diagnostics.csv", index=False)
    bootstrap.to_csv(output_dir / "paper_calibration_free_track_stabilized_roi_bootstrap_ci.csv", index=False)
    selected.to_frame().T.to_csv(
        output_dir / "paper_calibration_free_track_stabilized_roi_selected_strategy.csv", index=False
    )
    write_report(
        output_dir / "paper_calibration_free_track_stabilized_roi_report.md",
        candidates,
        selected,
        external_metric,
        bootstrap,
        diagnostics,
    )
    print(f"selected_strategy={strategy}")
    print(f"internal_rr_r2={selected['rr_r2']:.6f}")
    print(f"external_provisional_rr_r2={external_metric['rr_r2']:.6f}")


if __name__ == "__main__":
    main()
