"""Evaluate truth-free frame-motion artifact policies on cached Lindian curves."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import paper_repro_rr as rr
from run_lindian_robust_rr import robust_config


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Dataset_new" / "72video"
    assets = data_root / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=data_root / "lindian_anchored30_frames"
    )
    parser.add_argument(
        "--truth", type=Path, default=assets / "lindian_anchored30_manual_truth.csv"
    )
    parser.add_argument("--temperature-prefix", default="lindian_anchored30_repro")
    parser.add_argument("--output-dir", type=Path, default=assets)
    return parser.parse_args()


def policies() -> list[dict[str, object]]:
    common = {
        "suppression": False,
        "peak_gate": False,
        "gate_radius": 0,
        "interval_regularization": False,
        "short_ratio": 0.60,
        "min_improvement": 0.08,
        "reconstruct": False,
        "long_ratio": 1.65,
        "cycle_tolerance": 0.30,
    }
    result: list[dict[str, object]] = [
        {
            **common,
            "policy_id": "robust_no_motion",
            "center": 0.20,
            "scale": 0.12,
            "angle": 0.18,
            "mad": 6.0,
            "mask_radius": 0,
            "fusion_motion_weight": 0.0,
        }
    ]
    profiles = {
        "balanced": (0.20, 0.12, 0.18),
        "conservative": (0.28, 0.16, 0.25),
    }
    for profile, (center, scale, angle) in profiles.items():
        for mad in [6.0, 8.0]:
            for fusion_motion_weight in [0.05, 0.10, 0.20, 0.40]:
                result.append(
                    {
                        **common,
                        "policy_id": f"motion_fusion_{profile}_mad{mad:g}_w{fusion_motion_weight:g}",
                        "center": center,
                        "scale": scale,
                        "angle": angle,
                        "mad": mad,
                        "mask_radius": 0,
                        "fusion_motion_weight": fusion_motion_weight,
                    }
                )
    return result


def metric_row(group: pd.DataFrame, policy: dict[str, object], subset: str) -> dict[str, object]:
    truth_rr = group["manual_rr_bpm"].to_numpy(dtype=float)
    predicted_rr = group["predicted_rr_bpm"].to_numpy(dtype=float)
    count_error = group["count_error"].to_numpy(dtype=float)
    abs_error = np.abs(count_error)
    baseline_abs = group["baseline_abs_count_error"].to_numpy(dtype=float)
    return {
        **policy,
        "analysis_set": subset,
        "videos": int(len(group)),
        "rr_r2": rr.regression_r2(pd.Series(truth_rr), pd.Series(predicted_rr)),
        "rr_mae_bpm": float(np.mean(np.abs(predicted_rr - truth_rr))),
        "rr_rmse_bpm": float(np.sqrt(np.mean((predicted_rr - truth_rr) ** 2))),
        "count_mae": float(np.mean(abs_error)),
        "exact_count": int(np.sum(abs_error == 0)),
        "within_one_count": int(np.sum(abs_error <= 1)),
        "count_error_ge_two": int(np.sum(abs_error >= 2)),
        "changed_videos": int(group["prediction_changed"].sum()),
        "improved_videos": int(np.sum(abs_error < baseline_abs)),
        "worsened_videos": int(np.sum(abs_error > baseline_abs)),
        "motion_artifact_frames": int(group["motion_artifact_frames"].sum()),
        "motion_rejected_peaks": int(group["motion_rejected_peak_count"].sum()),
        "motion_interval_rejected_peaks": int(
            group["motion_interval_rejected_peak_count"].sum()
        ),
        "motion_reconstructed_peaks": int(
            group["motion_reconstructed_peak_count"].sum()
        ),
        "motion_stable_fraction_mean": float(group["motion_stable_fraction"].mean()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    truth = truth.loc[truth["include_sensitivity_analysis"].astype(str).str.lower().eq("true")].copy()
    temperatures: dict[str, pd.DataFrame] = {}
    for video_id in truth["video_id"]:
        path = args.input_root / video_id / f"{args.temperature_prefix}_temperatures.csv"
        temperatures[video_id] = pd.read_csv(path)

    base_config = robust_config("motion_candidate")
    baseline_counts: dict[str, int] = {}
    baseline_abs_errors: dict[str, float] = {}
    for row in truth.itertuples(index=False):
        curve, summary = rr.fuse_temperature_curve(
            temperatures[str(row.video_id)], base_config, pd.Series(row._asdict())
        )
        del curve
        baseline_counts[str(row.video_id)] = int(summary["peaks"])
        baseline_abs_errors[str(row.video_id)] = abs(
            int(summary["peaks"]) - int(row.breath_count)
        )

    prediction_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    all_policies = policies()
    for index, policy in enumerate(all_policies, start=1):
        print(f"Policy {index}/{len(all_policies)}: {policy['policy_id']}")
        config = replace(
            base_config,
            motion_artifact_suppression=bool(policy["suppression"]),
            motion_peak_gate=bool(policy["peak_gate"]),
            motion_center_step_threshold=float(policy["center"]),
            motion_scale_step_threshold=float(policy["scale"]),
            motion_angle_step_threshold=float(policy["angle"]),
            motion_mad_multiplier=float(policy["mad"]),
            motion_mask_radius=int(policy["mask_radius"]),
            motion_peak_gate_radius=int(policy["gate_radius"]),
            motion_interval_regularization=bool(policy["interval_regularization"]),
            motion_short_interval_ratio=float(policy["short_ratio"]),
            motion_interval_min_improvement=float(policy["min_improvement"]),
            motion_reconstruct_missing_cycles=bool(policy["reconstruct"]),
            motion_long_interval_ratio=float(policy["long_ratio"]),
            motion_cycle_tolerance=float(policy["cycle_tolerance"]),
            fusion_motion_penalty_weight=float(policy["fusion_motion_weight"]),
        )
        current_rows: list[dict[str, object]] = []
        for row in truth.itertuples(index=False):
            video_id = str(row.video_id)
            _, summary = rr.fuse_temperature_curve(
                temperatures[video_id], config, pd.Series(row._asdict())
            )
            predicted_count = int(summary["peaks"])
            duration = float(row.duration_seconds)
            predicted_rr = predicted_count / (duration / 60.0)
            current_rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "video_id": video_id,
                    "manual_breath_count": int(row.breath_count),
                    "manual_rr_bpm": float(row.rr),
                    "include_primary_analysis": bool(row.include_primary_analysis),
                    "truth_reliability": str(row.truth_reliability),
                    "baseline_breath_count": baseline_counts[video_id],
                    "baseline_abs_count_error": baseline_abs_errors[video_id],
                    "predicted_breath_count": predicted_count,
                    "predicted_rr_bpm": predicted_rr,
                    "count_error": predicted_count - int(row.breath_count),
                    "prediction_changed": predicted_count != baseline_counts[video_id],
                    "selected_fusion_mode": summary["selected_fusion_mode"],
                    "selection_motion_coupling": summary["selection_motion_coupling"],
                    "motion_artifact_frames": summary["motion_artifact_frames"],
                    "motion_artifact_fraction": summary["motion_artifact_fraction"],
                    "motion_stable_fraction": summary["motion_stable_fraction"],
                    "motion_score_p95": summary["motion_score_p95"],
                    "motion_score_max": summary["motion_score_max"],
                    "motion_rejected_peak_count": summary["motion_rejected_peak_count"],
                    "motion_rejected_peak_frames": summary["motion_rejected_peak_frames"],
                    "motion_interval_rejected_peak_count": summary[
                        "motion_interval_rejected_peak_count"
                    ],
                    "motion_interval_rejected_peak_frames": summary[
                        "motion_interval_rejected_peak_frames"
                    ],
                    "motion_reconstructed_peak_count": summary[
                        "motion_reconstructed_peak_count"
                    ],
                    "motion_reconstructed_peak_frames": summary[
                        "motion_reconstructed_peak_frames"
                    ],
                }
            )
        current = pd.DataFrame(current_rows)
        prediction_rows.extend(current_rows)
        primary = current.loc[current["include_primary_analysis"]].copy()
        policy_rows.append(metric_row(primary, policy, "primary_completed"))
        policy_rows.append(metric_row(current, policy, "sensitivity_all_numeric"))

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(policy_rows)
    primary_metrics = metrics.loc[metrics["analysis_set"].eq("primary_completed")].copy()
    primary_metrics = primary_metrics.sort_values(
        ["worsened_videos", "rr_r2", "rr_mae_bpm"],
        ascending=[True, False, True],
    )
    predictions.to_csv(
        args.output_dir / "lindian_motion_candidate_predictions.csv", index=False
    )
    metrics.to_csv(args.output_dir / "lindian_motion_candidate_metrics.csv", index=False)
    primary_metrics.to_csv(
        args.output_dir / "lindian_motion_candidate_primary_ranking.csv", index=False
    )
    print(primary_metrics.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
