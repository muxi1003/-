"""Evaluate CoTracker, TFA, and motion disentanglement on 49 Lindian clips."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import paper_repro_rr as rr
from evaluate_innovation73_cotracker_tfa_disentangle import (
    cache_paths,
    conservative_selective_candidates,
    disentangle_motion_component,
    extract_candidate_temperatures,
    joint_cotracker_tracks,
    load_cotracker,
    stage_metrics,
)
from evaluate_lindian49_butterworth_spectral import as_bool, frozen_config


STAGES = (
    "cotracker_joint",
    "cotracker_tfa",
    "cotracker_tfa_disentangled",
    "cotracker_selective_repair",
    "cotracker_selective_tfa",
    "cotracker_selective_tfa_disentangled",
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "Dataset_new" / "72video"
    input_root = data_root / "lindian_anchored30_frames"
    assets = data_root / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument(
        "--truth", type=Path, default=assets / "lindian_anchored30_manual_truth.csv"
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=input_root / "lindian_anchored30_motion_robust_summary.csv",
    )
    parser.add_argument("--temperature-prefix", default="lindian_anchored30_motion_robust")
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=repo
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
    )
    parser.add_argument(
        "--cotracker-repo",
        type=Path,
        default=repo / ".codex_tmp" / "torch" / "hub" / "facebookresearch_co-tracker_main",
    )
    parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=repo / ".codex_tmp" / "torch" / "hub" / "checkpoints" / "scaled_offline.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--run-name", default="lindian49_cotracker_tfa")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--window-overlap", type=int, default=12)
    parser.add_argument("--anchor-search-frames", type=int, default=8)
    parser.add_argument("--roi-radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    return parser.parse_args()


def truth_rr_column(truth: pd.DataFrame) -> str:
    return "manual_rr_bpm" if "manual_rr_bpm" in truth.columns else "rr"


def baseline_predictions(baseline: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_index = truth.set_index("video_id")
    rr_column = truth_rr_column(truth)
    rows = []
    for row in baseline.itertuples(index=False):
        video_id = str(row.video_id)
        truth_row = truth_index.loc[video_id]
        truth_count = int(float(truth_row["breath_count"]))
        rows.append(
            {
                "video_id": video_id,
                "stage": "frozen_lindian49_motion_robust",
                "predicted_count": int(float(row.peaks)),
                "predicted_rr_bpm": float(row.rr_bpm),
                "truth_count": truth_count,
                "truth_rr": float(truth_row[rr_column]),
                "count_error": int(float(row.peaks)) - truth_count,
                "selected_fusion_mode": str(row.selected_fusion_mode),
                "include_primary_analysis": as_bool(truth_row["include_primary_analysis"]),
                "truth_reliability": str(truth_row.get("truth_reliability", "")),
            }
        )
    return pd.DataFrame(rows)


def evaluate_table(
    table: pd.DataFrame,
    baseline_row: pd.Series,
    truth_row: pd.Series,
    stage: str,
    fps: float,
) -> dict[str, object]:
    limit = int(float(baseline_row.get("analysis_frame_limit", len(table))))
    table = table.iloc[:limit].copy()
    config = frozen_config(baseline_row, stage, fps)
    _, summary = rr.fuse_temperature_curve(table, config, truth_row)
    truth_count = int(float(truth_row["breath_count"]))
    rr_column = "manual_rr_bpm" if "manual_rr_bpm" in truth_row else "rr"
    return {
        "video_id": str(baseline_row.video_id),
        "stage": stage,
        "predicted_count": int(summary["peaks"]),
        "predicted_rr_bpm": float(summary["rr_bpm"]),
        "truth_count": truth_count,
        "truth_rr": float(truth_row[rr_column]),
        "count_error": int(summary["peaks"] - truth_count),
        "selected_fusion_mode": str(summary["selected_fusion_mode"]),
        "include_primary_analysis": as_bool(truth_row["include_primary_analysis"]),
        "truth_reliability": str(truth_row.get("truth_reliability", "")),
    }


def metric_rows(
    predictions_by_stage: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_ids = set(
        baseline.loc[baseline["include_primary_analysis"], "video_id"].astype(str)
    )
    for stage, predictions in predictions_by_stage.items():
        for analysis_set, ids in (
            ("sensitivity_all_numeric", set(baseline["video_id"].astype(str))),
            ("primary_completed", primary_ids),
        ):
            candidate_subset = predictions[predictions["video_id"].isin(ids)].copy()
            baseline_subset = baseline[baseline["video_id"].isin(ids)].copy()
            metric = stage_metrics(candidate_subset, baseline_subset)
            metric["analysis_set"] = analysis_set
            rows.append(metric)
    return pd.DataFrame(rows)


def unmatched_events(source: list[int], target: list[int], tolerance: int = 2) -> list[int]:
    return [
        event
        for event in source
        if not target or min(abs(event - candidate) for candidate in target) > tolerance
    ]


def write_selective_repair_evidence(
    args: argparse.Namespace,
    baseline: pd.DataFrame,
    truth: pd.DataFrame,
    baseline_prediction: pd.DataFrame,
    candidate_prediction: pd.DataFrame,
) -> pd.DataFrame:
    comparison = candidate_prediction.merge(
        baseline_prediction[["video_id", "predicted_count"]],
        on="video_id",
        suffixes=("", "_baseline"),
    )
    changed = comparison[
        comparison["predicted_count"].ne(comparison["predicted_count_baseline"])
    ]
    baseline_by_id = baseline.set_index("video_id", drop=False)
    rows: list[dict[str, object]] = []
    for changed_row in changed.itertuples(index=False):
        video_id = str(changed_row.video_id)
        baseline_row = baseline_by_id.loc[video_id]
        truth_row = truth.loc[video_id]
        video_dir = args.input_root / video_id
        paths = cache_paths(video_dir, args.run_name)
        source = pd.read_csv(video_dir / f"{args.temperature_prefix}_temperatures.csv")
        candidate = pd.read_csv(paths["selective_repair"])
        config = frozen_config(baseline_row, "lindian49_selective_repair_evidence", args.fps)
        baseline_curve, baseline_summary = rr.fuse_temperature_curve(
            source.copy(), config, truth_row
        )
        candidate_curve, candidate_summary = rr.fuse_temperature_curve(
            candidate.copy(), config, truth_row
        )
        baseline_peaks = np.flatnonzero(baseline_curve["is_peak"].to_numpy(bool)).tolist()
        candidate_peaks = np.flatnonzero(candidate_curve["is_peak"].to_numpy(bool)).tolist()
        left_repair = np.flatnonzero(
            candidate.get("left_selective_repair", pd.Series(False, index=candidate.index))
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy()
        ).tolist()
        right_repair = np.flatnonzero(
            candidate.get("right_selective_repair", pd.Series(False, index=candidate.index))
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy()
        ).tolist()
        baseline_curve.to_csv(
            video_dir / f"{args.run_name}_selective_repair_baseline_curve.csv", index=False
        )
        candidate_curve.to_csv(
            video_dir / f"{args.run_name}_selective_repair_curve.csv", index=False
        )
        rr.plot_peak_review(
            candidate_curve,
            video_id,
            candidate_summary,
            video_dir / f"{args.run_name}_selective_repair_peak_review.png",
        )
        rows.append(
            {
                "video_id": video_id,
                "truth_count": int(float(truth_row["breath_count"])),
                "truth_reliability": str(truth_row.get("truth_reliability", "")),
                "include_primary_analysis": as_bool(truth_row["include_primary_analysis"]),
                "baseline_count": int(baseline_summary["peaks"]),
                "candidate_count": int(candidate_summary["peaks"]),
                "baseline_peak_frames": ";".join(map(str, baseline_peaks)),
                "candidate_peak_frames": ";".join(map(str, candidate_peaks)),
                "removed_peak_frames": ";".join(
                    map(str, unmatched_events(baseline_peaks, candidate_peaks))
                ),
                "added_peak_frames": ";".join(
                    map(str, unmatched_events(candidate_peaks, baseline_peaks))
                ),
                "left_repaired_frames": ";".join(map(str, left_repair)),
                "right_repaired_frames": ";".join(map(str, right_repair)),
                "selected_fusion_mode": str(candidate_summary["selected_fusion_mode"]),
                "max_abs_left_temperature_change": float(
                    np.nanmax(
                        np.abs(
                            pd.to_numeric(candidate["left_temp"], errors="coerce")
                            - pd.to_numeric(source["left_temp"], errors="coerce")
                        )
                    )
                ),
                "max_abs_right_temperature_change": float(
                    np.nanmax(
                        np.abs(
                            pd.to_numeric(candidate["right_temp"], errors="coerce")
                            - pd.to_numeric(source["right_temp"], errors="coerce")
                        )
                    )
                ),
            }
        )
    evidence = pd.DataFrame(rows)
    evidence.to_csv(
        args.output_dir / f"{args.run_name}_selective_repair_changed_cases.csv", index=False
    )
    return evidence


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    baseline = pd.read_csv(args.baseline_summary, dtype={"video_id": str})
    if len(truth) != 49 or truth["video_id"].nunique() != 49:
        raise ValueError("Truth table must contain exactly 49 unique video IDs")
    if set(truth["video_id"]) != set(baseline["video_id"]):
        raise ValueError("Frozen baseline IDs do not match the Lindian49 truth table")
    if args.video_id:
        missing = sorted(set(args.video_id) - set(truth["video_id"]))
        if missing:
            raise ValueError(f"Unknown video IDs: {missing}")
        truth = truth[truth["video_id"].isin(args.video_id)].copy()
        baseline = baseline[baseline["video_id"].isin(args.video_id)].copy()
    order = {video_id: index for index, video_id in enumerate(truth["video_id"])}
    baseline = baseline.sort_values("video_id", key=lambda values: values.map(order)).reset_index(
        drop=True
    )
    truth = truth.set_index("video_id", drop=False)
    baseline_prediction = baseline_predictions(baseline, truth.reset_index(drop=True))

    replay_rows = []
    for _, baseline_row in baseline.iterrows():
        video_id = str(baseline_row.video_id)
        source = pd.read_csv(
            args.input_root / video_id / f"{args.temperature_prefix}_temperatures.csv"
        )
        replay_rows.append(
            evaluate_table(source, baseline_row, truth.loc[video_id], "baseline_replay", args.fps)
        )
    replay = pd.DataFrame(replay_rows)
    replay_matches = int(
        np.sum(
            replay["predicted_count"].to_numpy()
            == baseline_prediction["predicted_count"].to_numpy()
        )
    )
    if replay_matches != len(baseline):
        failed = replay.loc[
            replay["predicted_count"].to_numpy()
            != baseline_prediction["predicted_count"].to_numpy(),
            "video_id",
        ].tolist()
        raise RuntimeError(f"Frozen Lindian49 replay failed for {failed}")
    if args.replay_only:
        print(f"baseline_replay={replay_matches}/{len(baseline)}")
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CoTracker experiment requires CUDA")
    device = "cuda"
    tracker = load_cotracker(args.cotracker_repo, args.cotracker_checkpoint, device)
    temp_model = joblib.load(args.temp_model)
    stage_rows: dict[str, list[dict[str, object]]] = {stage: [] for stage in STAGES}
    diagnostic_rows: list[dict[str, object]] = []

    for position, baseline_row in baseline.iterrows():
        video_id = str(baseline_row.video_id)
        truth_row = truth.loc[video_id]
        video_dir = args.input_root / video_id
        paths = cache_paths(video_dir, args.run_name)
        source = pd.read_csv(video_dir / f"{args.temperature_prefix}_temperatures.csv")
        limit = int(float(baseline_row.get("analysis_frame_limit", len(source))))
        source = source.iloc[:limit].copy().reset_index(drop=True)
        image_by_name = {path.name: path for path in rr.frame_image_paths(video_dir)}
        frame_paths = [image_by_name[str(name)] for name in source["frame_name"]]
        core_names = ("tracks", "joint", "tfa", "disentangled", "diagnostics")
        can_reuse = args.reuse_cache and all(paths[name].exists() for name in core_names)
        if can_reuse:
            tracks = pd.read_csv(paths["tracks"])
            joint_table = pd.read_csv(paths["joint"])
            tfa_table = pd.read_csv(paths["tfa"])
            disentangled = pd.read_csv(paths["disentangled"])
            diagnostics = pd.read_csv(paths["diagnostics"])
            if not (
                len(tracks)
                == len(joint_table)
                == len(tfa_table)
                == len(disentangled)
                == len(source)
            ):
                can_reuse = False
        if not can_reuse:
            print(f"[{position + 1}/{len(baseline)}] {video_id}: CoTracker", flush=True)
            try:
                tracks, points, visible, centers = joint_cotracker_tracks(
                    source,
                    frame_paths,
                    tracker,
                    device,
                    args.window_length,
                    args.window_overlap,
                    args.anchor_search_frames,
                )
                joint_table, tfa_table, diagnostics = extract_candidate_temperatures(
                    source,
                    frame_paths,
                    tracks,
                    points,
                    visible,
                    centers,
                    temp_model,
                    args.roi_radius,
                    args.min_temp,
                )
                disentangled, disentangle_diagnostics = disentangle_motion_component(
                    tfa_table, fps=args.fps
                )
            except ValueError as error:
                if "bilateral" not in str(error):
                    raise
                tracks = source[["frame_name", "left_x", "left_y", "right_x", "right_y"]].copy()
                tracks["tracking_status"] = "no_anchor_fallback"
                joint_table = source.copy()
                tfa_table = source.copy()
                disentangled = source.copy()
                diagnostics = pd.DataFrame(
                    {
                        "registration_applied": np.zeros(len(source), dtype=bool),
                        "registration_source": ["no_anchor_fallback"] * len(source),
                        "registration_residual_px": np.full(len(source), np.nan),
                    }
                )
                disentangle_diagnostics = {
                    "left_motion_explained_cv": 0.0,
                    "right_motion_explained_cv": 0.0,
                    "left_disentangle_gain": 0.0,
                    "right_disentangle_gain": 0.0,
                }
            tracks.to_csv(paths["tracks"], index=False)
            joint_table.to_csv(paths["joint"], index=False)
            tfa_table.to_csv(paths["tfa"], index=False)
            disentangled.to_csv(paths["disentangled"], index=False)
            diagnostics = diagnostics.assign(**disentangle_diagnostics)
            diagnostics.to_csv(paths["diagnostics"], index=False)

        motion_config = frozen_config(baseline_row, "lindian49_selective_gate", args.fps)
        selective_repair, selective_tfa, selective_disentangled, selective_stats = (
            conservative_selective_candidates(
                source,
                tracks,
                joint_table,
                tfa_table,
                disentangled,
                diagnostics,
                baseline_row,
                motion_config=motion_config,
            )
        )
        selective_repair.to_csv(paths["selective_repair"], index=False)
        selective_tfa.to_csv(paths["selective_tfa"], index=False)
        selective_disentangled.to_csv(paths["selective_disentangled"], index=False)

        print(f"[{position + 1}/{len(baseline)}] {video_id}: evaluate", flush=True)
        for stage, table in (
            ("cotracker_joint", joint_table),
            ("cotracker_tfa", tfa_table),
            ("cotracker_tfa_disentangled", disentangled),
            ("cotracker_selective_repair", selective_repair),
            ("cotracker_selective_tfa", selective_tfa),
            ("cotracker_selective_tfa_disentangled", selective_disentangled),
        ):
            stage_rows[stage].append(
                evaluate_table(table, baseline_row, truth_row, stage, args.fps)
            )
        diagnostic_rows.append(
            {
                "video_id": video_id,
                "frames": len(source),
                "include_primary_analysis": as_bool(truth_row["include_primary_analysis"]),
                "registration_coverage": float(
                    pd.Series(diagnostics["registration_applied"]).astype(bool).mean()
                ),
                "joint_registration_fraction": float(
                    diagnostics["registration_source"].astype(str).eq("joint_points").mean()
                ),
                "median_registration_residual_px": float(
                    pd.to_numeric(diagnostics["registration_residual_px"], errors="coerce").median()
                ),
                "left_disentangle_gain": float(
                    pd.to_numeric(diagnostics["left_disentangle_gain"], errors="coerce").median()
                ),
                "right_disentangle_gain": float(
                    pd.to_numeric(diagnostics["right_disentangle_gain"], errors="coerce").median()
                ),
                **selective_stats,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_stage = {"frozen_lindian49_motion_robust": baseline_prediction}
    for stage, rows in stage_rows.items():
        table = pd.DataFrame(rows)
        predictions_by_stage[stage] = table
        table.to_csv(args.output_dir / f"{args.run_name}_{stage}_predictions.csv", index=False)
    all_predictions = pd.concat(predictions_by_stage.values(), ignore_index=True)
    all_predictions.to_csv(args.output_dir / f"{args.run_name}_all_predictions.csv", index=False)
    write_selective_repair_evidence(
        args,
        baseline,
        truth,
        baseline_prediction,
        predictions_by_stage["cotracker_selective_repair"],
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / f"{args.run_name}_video_diagnostics.csv", index=False
    )
    metrics = metric_rows(predictions_by_stage, baseline_prediction)

    all49 = metrics[metrics["analysis_set"].eq("sensitivity_all_numeric")].reset_index(drop=True)
    retained = all49.iloc[0]
    decisions = []
    for index in range(1, len(all49)):
        candidate = all49.iloc[index]
        keep = bool(
            float(candidate.rr_r2) > float(retained.rr_r2) + 1e-12
            and float(candidate.rr_mae_bpm) <= float(retained.rr_mae_bpm) + 1e-12
        )
        decisions.append(
            {
                "candidate_stage": candidate.stage,
                "compared_with": retained.stage,
                "candidate_rr_r2": candidate.rr_r2,
                "candidate_rr_mae_bpm": candidate.rr_mae_bpm,
                "retained": keep,
                "decision": "keep" if keep else "rollback",
            }
        )
        if keep:
            retained = candidate
    metrics["final_retained_stage"] = str(retained.stage)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics.to_csv(args.output_dir / f"{args.run_name}_metrics.csv", index=False)
    pd.DataFrame(decisions).to_csv(
        args.output_dir / f"{args.run_name}_rollback_decisions.csv", index=False
    )
    print(metrics.to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))
    print(f"final_retained_stage={retained.stage}")


if __name__ == "__main__":
    main()
