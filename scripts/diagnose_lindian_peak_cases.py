"""Build frame-level diagnostics for selected Lindian fixed-window RR cases."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

import paper_repro_rr as rr


VIDEO_IDS = ("170333", "16170075", "zs197000")
FUSION_MODES = ("adaptive", "left", "right", "mean", "min", "max")


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
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def frozen_config(mode: str) -> rr.ReproConfig:
    return replace(
        rr.fast_fusion_quality_config(True),
        fusion_mode=mode,
        min_rr_bpm=20.0,
        max_rr_bpm=100.0,
        limit_to_truth_duration=False,
        adaptive_peak_retuning=False,
        short_sparse_edge_completion=False,
        infer_global_offset=False,
        edge_peak_completion="none",
    )


def peak_frames(curve: pd.DataFrame) -> list[int]:
    return curve.loc[curve["is_peak"], "frame_index"].astype(int).tolist()


def direct_detection_mask(temp_df: pd.DataFrame, selected_mode: str) -> np.ndarray:
    left = temp_df.get("left_source", pd.Series("", index=temp_df.index)).astype(str).eq("detected")
    right = temp_df.get("right_source", pd.Series("", index=temp_df.index)).astype(str).eq("detected")
    if selected_mode == "left":
        return left.to_numpy(dtype=bool)
    if selected_mode == "right":
        return right.to_numpy(dtype=bool)
    return (left | right).to_numpy(dtype=bool)


def is_directly_supported(mask: np.ndarray, frame: int, radius: int = 3) -> bool:
    start = max(0, frame - radius)
    end = min(len(mask), frame + radius + 1)
    return bool(mask[start:end].any())


def tail_without_direct_detection(temp_df: pd.DataFrame) -> tuple[int, int]:
    left = temp_df.get("left_source", pd.Series("", index=temp_df.index)).astype(str).eq("detected")
    right = temp_df.get("right_source", pd.Series("", index=temp_df.index)).astype(str).eq("detected")
    direct = (left | right).to_numpy(dtype=bool)
    locations = np.flatnonzero(direct)
    if locations.size == 0:
        return 0, len(temp_df)
    start = int(locations[-1] + 1)
    return start, int(len(temp_df) - start)


def analyze_mode(
    video_id: str,
    temp_df: pd.DataFrame,
    truth_row: pd.Series,
    mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = frozen_config(mode)
    curve, summary = rr.fuse_temperature_curve(temp_df, config, truth_row)
    frames = peak_frames(curve)
    selected_mode = str(summary["selected_fusion_mode"])
    direct_mask = direct_detection_mask(temp_df, selected_mode)
    unsupported = [frame for frame in frames if not is_directly_supported(direct_mask, frame)]
    early = [frame for frame in frames if frame <= 75]
    intervals = np.diff(frames)
    comparison = {
        "video_id": video_id,
        "requested_fusion_mode": mode,
        "selected_fusion_mode": selected_mode,
        "manual_breath_count": int(truth_row["breath_count"]),
        "predicted_breath_count": len(frames),
        "count_error": len(frames) - int(truth_row["breath_count"]),
        "early_peak_count_frame_0_75": len(early),
        "early_peak_frames": ";".join(map(str, early)),
        "peak_frames": ";".join(map(str, frames)),
        "median_peak_interval_frames": float(np.median(intervals)) if intervals.size else np.nan,
        "interval_cv": float(np.std(intervals) / np.mean(intervals)) if intervals.size else np.nan,
        "selected_peak_prominence": float(summary["selected_peak_prominence"]),
        "mean_prominence": float(summary["mean_prominence"]),
        "selection_score": float(summary["selection_score"]),
        "selection_candidate_median_prominence": float(
            summary["selection_median_prominence"]
        ),
        "selection_candidate_amplitude": float(summary["selection_amplitude"]),
        "selection_candidate_missing_rate": float(summary["selection_missing_rate"]),
        "unsupported_peak_count_radius3": len(unsupported),
        "unsupported_peak_frames_radius3": ";".join(map(str, unsupported)),
    }

    smoothed = pd.to_numeric(curve["smoothed_norm"], errors="coerce").dropna().to_numpy(dtype=float)
    smooth_indices = curve.index[curve["smoothed_norm"].notna()].to_numpy(dtype=int)
    candidates, _ = find_peaks(
        smoothed,
        distance=config.peak_distance,
        prominence=0.002,
        width=1,
        plateau_size=1,
    )
    candidate_prominences = peak_prominences(smoothed, candidates)[0] if len(candidates) else np.array([])
    final_set = set(frames)
    details = []
    for candidate, prominence in zip(candidates, candidate_prominences):
        frame = int(smooth_indices[int(candidate)])
        details.append(
            {
                "video_id": video_id,
                "requested_fusion_mode": mode,
                "selected_fusion_mode": selected_mode,
                "frame_index": frame,
                "prominence": float(prominence),
                "selected_as_final_peak": frame in final_set,
                "within_frame_0_75": frame <= 75,
                "direct_detection_support_radius3": is_directly_supported(direct_mask, frame),
            }
        )
    return comparison, details


def truncate_tail_counterfactual(
    video_id: str, temp_df: pd.DataFrame, truth_row: pd.Series
) -> dict[str, object]:
    tail_start, tail_length = tail_without_direct_detection(temp_df)
    if tail_length == 0:
        truncated = temp_df
    else:
        truncated = temp_df.iloc[:tail_start].copy()
    curve, summary = rr.fuse_temperature_curve(truncated, frozen_config("adaptive"), truth_row)
    frames = peak_frames(curve)
    return {
        "video_id": video_id,
        "last_direct_detection_frame": tail_start - 1,
        "terminal_frames_without_direct_detection": tail_length,
        "truncated_frames": len(truncated),
        "selected_fusion_mode_after_truncation": summary["selected_fusion_mode"],
        "predicted_breath_count_after_truncation": len(frames),
        "count_error_after_truncation": len(frames) - int(truth_row["breath_count"]),
        "peak_frames_after_truncation": ";".join(map(str, frames)),
    }


def build_report(
    comparisons: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    truth: pd.DataFrame,
) -> str:
    lines = [
        "# Lindian Peak Case Diagnosis",
        "",
        "This is a diagnostic replay. Alternative fusion modes and truth comparisons are not deployable paper results.",
        "",
    ]
    for video_id in comparisons["video_id"].drop_duplicates():
        rows = comparisons.loc[comparisons["video_id"].eq(video_id)]
        baseline = rows.loc[rows["requested_fusion_mode"].eq("adaptive")].iloc[0]
        best = rows.iloc[(rows["count_error"].abs()).argmin()]
        tail = counterfactuals.loc[counterfactuals["video_id"].eq(video_id)].iloc[0]
        truth_row = truth.loc[truth["video_id"].eq(video_id)].iloc[0]
        lines.extend(
            [
                f"## {video_id}",
                "",
                f"- Manual count: {int(truth_row['breath_count'])}",
                f"- Frozen adaptive result: {int(baseline['predicted_breath_count'])} peaks at {baseline['peak_frames']}",
                f"- Early peaks through frame 75: {int(baseline['early_peak_count_frame_0_75'])} at {baseline['early_peak_frames']}",
                f"- Peaks without direct-detection support within +/-3 frames: {baseline['unsupported_peak_frames_radius3'] or 'none'}",
                f"- Last direct-detection frame: {int(tail['last_direct_detection_frame'])}; trailing unsupported frames: {int(tail['terminal_frames_without_direct_detection'])}",
                f"- Truncating the unsupported tail gives {int(tail['predicted_breath_count_after_truncation'])} peaks at {tail['peak_frames_after_truncation']}",
                f"- Closest truth-assisted fusion diagnostic: {best['requested_fusion_mode']} -> {int(best['predicted_breath_count'])} peaks",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    video_ids = args.video_ids or list(VIDEO_IDS)
    truth = pd.read_csv(args.truth, dtype={"video_id": str})
    truth = truth.loc[truth["video_id"].isin(video_ids)].copy()
    if set(truth["video_id"]) != set(video_ids):
        missing = sorted(set(video_ids) - set(truth["video_id"]))
        raise ValueError(f"Missing truth rows: {missing}")

    comparisons: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    counterfactuals: list[dict[str, object]] = []
    for video_id in video_ids:
        temp_path = (
            args.input_root
            / video_id
            / "lindian_anchored30_repro_temperatures.csv"
        )
        temp_df = pd.read_csv(temp_path)
        truth_row = truth.loc[truth["video_id"].eq(video_id)].iloc[0]
        for mode in FUSION_MODES:
            comparison, mode_details = analyze_mode(video_id, temp_df, truth_row, mode)
            comparisons.append(comparison)
            details.extend(mode_details)
        counterfactuals.append(truncate_tail_counterfactual(video_id, temp_df, truth_row))

    comparison_df = pd.DataFrame(comparisons)
    detail_df = pd.DataFrame(details)
    counterfactual_df = pd.DataFrame(counterfactuals)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "lindian_peak_case_mode_comparison.csv"
    detail_path = args.output_dir / "lindian_peak_case_peak_detail.csv"
    counterfactual_path = args.output_dir / "lindian_peak_case_tail_counterfactual.csv"
    report_path = args.output_dir / "lindian_peak_case_diagnosis.md"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    counterfactual_df.to_csv(counterfactual_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(comparison_df, counterfactual_df, truth), encoding="utf-8"
    )
    print(f"Saved mode comparison: {comparison_path.resolve()}")
    print(f"Saved peak details: {detail_path.resolve()}")
    print(f"Saved tail counterfactuals: {counterfactual_path.resolve()}")
    print(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
