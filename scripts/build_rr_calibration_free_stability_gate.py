from __future__ import annotations

import argparse
import math
from pathlib import Path

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
    estimate_curve,
    extract_video_signals,
    fused_curve,
)
from build_rr_duration_normalized_windowed_innovation import (
    derive_config,
    external_truth,
    internal_truth,
)


STABILITY_RULES = {
    "no_kinematic_filter": (math.inf, math.inf),
    "relative_strict": (0.010, 0.010),
    "relative_moderate": (0.025, 0.025),
    "relative_relaxed": (0.050, 0.050),
    "relative_permissive": (0.100, 0.100),
    "relative_robust_3mad": (math.nan, math.nan),
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an internal-selected kinematic frame-stability gate for the "
            "calibration-free relative thermal-color RR branch."
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
    parser.add_argument("--minimum-confidence", type=float, default=0.50)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    return parser.parse_args()


def numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table.get(column, pd.Series(np.nan, index=table.index)), errors="coerce")


def robust_limit(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < 3:
        return math.inf
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    return center + 3.0 * max(mad, 1e-5)


def frame_stability(
    temperature: pd.DataFrame,
    rule: str,
    minimum_confidence: float,
) -> pd.DataFrame:
    if rule not in STABILITY_RULES:
        raise ValueError(f"Unknown kinematic rule: {rule}")
    left_x, left_y = numeric(temperature, "left_x"), numeric(temperature, "left_y")
    right_x, right_y = numeric(temperature, "right_x"), numeric(temperature, "right_y")
    left_conf, right_conf = numeric(temperature, "left_conf"), numeric(temperature, "right_conf")
    midpoint_x = (left_x + right_x) / 2.0
    midpoint_y = (left_y + right_y) / 2.0
    nostril_distance = np.hypot(left_x - right_x, left_y - right_y)
    midpoint_speed = np.hypot(midpoint_x.diff(), midpoint_y.diff())
    distance_change = nostril_distance.diff().abs()
    scale = nostril_distance.rolling(5, min_periods=1, center=True).median().clip(lower=1.0)
    translation_ratio = midpoint_speed / scale
    distance_ratio = distance_change / scale
    translation_limit, distance_limit = STABILITY_RULES[rule]
    if rule == "relative_robust_3mad":
        translation_limit = robust_limit(translation_ratio.to_numpy(float))
        distance_limit = robust_limit(distance_ratio.to_numpy(float))
    if rule == "no_kinematic_filter":
        stable = pd.Series(True, index=temperature.index)
    else:
        coordinates_valid = (
            left_x.notna()
            & left_y.notna()
            & right_x.notna()
            & right_y.notna()
            & (nostril_distance > 1.0)
        )
        confidence_valid = left_conf.ge(float(minimum_confidence)) & right_conf.ge(
            float(minimum_confidence)
        )
        motion_valid = translation_ratio.le(translation_limit) & distance_ratio.le(distance_limit)
        motion_valid.iloc[0] = True
        stable = coordinates_valid & confidence_valid & motion_valid
        status = temperature.get("status", pd.Series("ok", index=temperature.index)).astype(str).str.lower()
        stable &= status.isin({"ok", "detected", "tracked"})
    return pd.DataFrame(
        {
            "frame_name": temperature["frame_name"].astype(str),
            "stable": stable.to_numpy(bool),
            "translation_ratio": translation_ratio.to_numpy(float),
            "distance_ratio": distance_ratio.to_numpy(float),
            "translation_limit": float(translation_limit),
            "distance_limit": float(distance_limit),
        }
    )


def filtered_signals(signals: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    result = signals.merge(stability[["frame_name", "stable"]], on="frame_name", how="left", validate="one_to_one")
    stable = result["stable"].fillna(False).astype(bool)
    columns = [column for column in result.columns if column.startswith("left_") or column.startswith("right_")]
    columns = [column for column in columns if column != "stable"]
    result.loc[~stable, columns] = math.nan
    return result.drop(columns="stable")


def member_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: object,
    members: list[tuple[str, str]],
    radius: int,
    cohort: str,
    rule: str,
    minimum_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        print(f"[{cohort} {rule} {index}/{len(summary)}] {video_id}")
        temperature = pd.read_csv(root / video_id / f"{prefix}_temperatures.csv")
        stability = frame_stability(temperature, rule, minimum_confidence)
        signals = filtered_signals(
            extract_video_signals(root / video_id, prefix, radius, False), stability
        )
        stability_rows.append(
            {
                "cohort": cohort,
                "video_id": video_id,
                "rule": rule,
                "frames": int(len(stability)),
                "stable_frames": int(stability["stable"].sum()),
                "stable_fraction": float(stability["stable"].mean()) if len(stability) else math.nan,
                "translation_ratio_p95": float(np.nanquantile(stability["translation_ratio"], 0.95)),
                "distance_ratio_p95": float(np.nanquantile(stability["distance_ratio"], 0.95)),
                "translation_limit": float(stability["translation_limit"].iloc[0]),
                "distance_limit": float(stability["distance_limit"].iloc[0]),
            }
        )
        for signal, polarity in members:
            curve = fused_curve(signals, signal, polarity == "inverted", config)
            rr_bpm, windows, quality = estimate_curve(curve, config)
            rows.append(
                {
                    "cohort": cohort,
                    "video_id": video_id,
                    "rule": rule,
                    "member": f"{signal}:{polarity}",
                    "rr_bpm": rr_bpm,
                    "windows": windows,
                    "quality_mean": quality,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(stability_rows)


def write_report(
    path: Path,
    selection: pd.Series,
    gated_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    internal = gated_metrics[gated_metrics["cohort"].eq("internal_development")].iloc[0]
    external = gated_metrics[gated_metrics["cohort"].eq("external_provisional")].iloc[0]
    internal_stability = stability[stability["cohort"].eq("internal_development")]
    external_stability = stability[stability["cohort"].eq("external_provisional")]
    lines = [
        "# Kinematic Stability-Gated Relative Thermal-Color Probe",
        "",
        "Status: `internal_selected_external_zero_shot_development_probe`",
        "",
        "Frame inclusion uses only pose confidence, normalized midpoint displacement, "
        "and normalized nostril-distance changes. The rule is selected by internal "
        "color-only RMSE; external manual RR is used only after the rule is frozen.",
        "",
        f"Selected rule: `{selection['rule']}`.",
        f"Internal stable-frame fraction: `{internal_stability['stable_fraction'].mean():.4f}`.",
        f"External stable-frame fraction: `{external_stability['stable_fraction'].mean():.4f}`.",
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
            "This is a non-truth kinematic quality experiment. It remains exploratory "
            "until the external reference is independently blinded and adjudicated.",
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
    metric_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    candidate_data: dict[str, pd.DataFrame] = {}
    for rule in STABILITY_RULES:
        member_table, stability = member_predictions(
            internal_root,
            args.internal_prefix,
            internal_summary,
            config,
            members,
            args.radius,
            "internal_development",
            rule,
            args.minimum_confidence,
        )
        aggregated = aggregate_members(member_table, members, "mean")
        metrics, scored = candidate_metrics(
            aggregated,
            internal_truth_table,
            "internal_development",
            members_text,
        )
        metrics["rule"] = rule
        metrics["stable_fraction_mean"] = float(stability["stable_fraction"].mean())
        metric_rows.append(metrics)
        prediction_frames.extend([member_table, scored.assign(prediction_kind="color_only")])
        stability_frames.append(stability)
        candidate_data[rule] = aggregated
    candidates = pd.concat(metric_rows, ignore_index=True).sort_values(
        ["rr_rmse", "rr_mae", "rule"]
    )
    selection = candidates.iloc[0].copy()
    rule = str(selection["rule"])
    external_members, external_stability = member_predictions(
        external_root,
        args.external_prefix,
        external_summary,
        config,
        members,
        args.radius,
        "external_provisional",
        rule,
        args.minimum_confidence,
    )
    external_aggregated = aggregate_members(external_members, members, "mean")
    external_metrics, external_scored_color = candidate_metrics(
        external_aggregated,
        external_truth_table,
        "external_provisional",
        members_text,
    )
    external_metrics["rule"] = rule
    external_metrics["stable_fraction_mean"] = float(external_stability["stable_fraction"].mean())
    internal_gated = duration_gated(
        candidate_data[rule], internal_summary, config.long_clip_threshold_seconds
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
        row["rule"] = rule
    gated_metrics = pd.DataFrame([internal_gated_metric, external_gated_metric])
    bootstrap = cluster_bootstrap_vs_p2g(
        external_gated_scored,
        reference,
        pd.read_csv(args.p2g_gated_predictions_csv.resolve()),
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    pd.concat([candidates, external_metrics, gated_metrics], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_stability_gate_metrics.csv", index=False
    )
    selection.to_frame().T.to_csv(
        output_dir / "paper_calibration_free_stability_gate_selection.csv", index=False
    )
    pd.concat(
        [
            *prediction_frames,
            external_members,
            external_scored_color.assign(prediction_kind="color_only"),
            internal_gated_scored.assign(prediction_kind="duration_gated_selected"),
            external_gated_scored.assign(prediction_kind="duration_gated_selected"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "paper_calibration_free_stability_gate_predictions.csv", index=False)
    pd.concat([*stability_frames, external_stability], ignore_index=True).to_csv(
        output_dir / "paper_calibration_free_stability_gate_frame_summary.csv", index=False
    )
    bootstrap.to_csv(
        output_dir / "paper_calibration_free_stability_gate_bootstrap_ci.csv", index=False
    )
    write_report(
        output_dir / "paper_calibration_free_stability_gate_report.md",
        selection,
        gated_metrics,
        pd.concat([*stability_frames, external_stability], ignore_index=True),
        bootstrap,
    )
    print("Selected rule:")
    print(selection.to_string())
    print("\nGated metrics:")
    print(gated_metrics.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
