"""Screen anchored Lindian fixed windows by non-truth YOLO bilateral visibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=assets / "lindian_anchored_fixed30_manifest.csv")
    parser.add_argument("--yolo-model", type=Path, default=repo_root / "models" / "best.pt")
    parser.add_argument("--output-csv", type=Path, default=assets / "lindian_anchored30_quality_screen.csv")
    parser.add_argument("--output-report", type=Path, default=assets / "lindian_anchored30_quality_screen.md")
    parser.add_argument(
        "--annotation-template",
        type=Path,
        default=assets / "lindian_anchored30_quality_gated_annotation_template.csv",
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--min-bilateral-fraction", type=float, default=0.7)
    return parser.parse_args()


def select_detection(result) -> int | None:
    if result.keypoints is None or result.keypoints.data is None:
        return None
    keypoints = result.keypoints.data
    if len(keypoints) == 0:
        return None
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == len(keypoints):
        return int(result.boxes.conf.detach().cpu().numpy().argmax())
    return 0


def keypoint_confidence(kpt: np.ndarray) -> float:
    return float(kpt[2]) if len(kpt) >= 3 else 1.0


def screen_video(row: pd.Series, model: YOLO, args: argparse.Namespace) -> dict[str, object]:
    video_id = str(row["video_id"])
    raw_path = Path(str(row["raw_source_path"]))
    start_seconds = float(row["anchored_window_start_seconds"])
    duration_seconds = float(row["anchored_window_seconds"])
    base = {
        "video_id": video_id,
        "raw_source_path": str(raw_path),
        "window_start_seconds": start_seconds,
        "window_seconds": duration_seconds,
        "sample_hz": float(args.sample_hz),
        "keypoint_confidence_threshold": float(args.conf),
        "min_bilateral_fraction": float(args.min_bilateral_fraction),
    }
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        return {**base, "status": "FAIL", "issue": "could not open raw video"}
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return {**base, "status": "FAIL", "issue": "invalid raw fps"}
        start_frame = max(0, int(round(start_seconds * fps)))
        window_frames = max(1, int(round(duration_seconds * fps)))
        sample_count = max(1, int(round(duration_seconds * args.sample_hz)))
        sample_offsets = set(np.linspace(0, window_frames - 1, sample_count).round().astype(int).tolist())
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        sampled = left_valid = right_valid = bilateral_valid = 0
        left_confidences: list[float] = []
        right_confidences: list[float] = []
        for offset in range(window_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if offset not in sample_offsets:
                continue
            sampled += 1
            result = model(frame, verbose=False)[0]
            detection_index = select_detection(result)
            if detection_index is None:
                continue
            keypoints = result.keypoints.data[detection_index].detach().cpu().numpy()
            if len(keypoints) < 2:
                continue
            left_conf = keypoint_confidence(keypoints[0])
            right_conf = keypoint_confidence(keypoints[1])
            left_is_valid = left_conf >= args.conf
            right_is_valid = right_conf >= args.conf
            if left_is_valid:
                left_valid += 1
                left_confidences.append(left_conf)
            if right_is_valid:
                right_valid += 1
                right_confidences.append(right_conf)
            if left_is_valid and right_is_valid:
                bilateral_valid += 1
    finally:
        capture.release()
    bilateral_fraction = bilateral_valid / sampled if sampled else 0.0
    return {
        **base,
        "status": "PASS",
        "issue": "",
        "source_fps": fps,
        "sampled_frames": sampled,
        "left_valid_samples": left_valid,
        "right_valid_samples": right_valid,
        "bilateral_valid_samples": bilateral_valid,
        "left_valid_fraction": left_valid / sampled if sampled else 0.0,
        "right_valid_fraction": right_valid / sampled if sampled else 0.0,
        "bilateral_valid_fraction": bilateral_fraction,
        "mean_left_confidence": float(np.mean(left_confidences)) if left_confidences else np.nan,
        "mean_right_confidence": float(np.mean(right_confidences)) if right_confidences else np.nan,
        "quality_gate": "pass" if bilateral_fraction >= args.min_bilateral_fraction else "fail",
    }


def main() -> None:
    args = parse_args()
    if args.sample_hz <= 0:
        raise ValueError("--sample-hz must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.min_bilateral_fraction <= 1:
        raise ValueError("confidence and bilateral fraction thresholds must be in [0, 1]")
    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    required = {"video_id", "raw_source_path", "anchored_window_start_seconds", "anchored_window_seconds", "anchor_status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[manifest["anchor_status"].eq("ready_for_anchored_fixed_window")].copy()
    if args.video_id:
        selected = selected.loc[selected["video_id"].isin(set(args.video_id))]
    if selected.empty:
        raise ValueError("No anchored windows selected")
    model = YOLO(str(args.yolo_model))
    results = pd.DataFrame(
        [screen_video(row, model, args) for _, row in selected.sort_values("video_id").iterrows()]
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    passed = results.loc[results["quality_gate"].eq("pass")]
    passed_with_manifest = passed.merge(
        manifest[["video_id", "reference_rr_bpm"]], on="video_id", how="left", validate="one_to_one"
    )
    template = pd.DataFrame(
        {
            "window_id": [
                f"{video_id}_anchored_{int(round(window_seconds))}s"
                for video_id, window_seconds in zip(
                    passed_with_manifest["video_id"], passed_with_manifest["window_seconds"]
                )
            ],
            "video_id": passed_with_manifest["video_id"],
            "raw_source_path": passed_with_manifest["raw_source_path"],
            "window_start_seconds": passed_with_manifest["window_start_seconds"],
            "window_seconds": passed_with_manifest["window_seconds"],
            "bilateral_valid_fraction_1hz": passed_with_manifest["bilateral_valid_fraction"],
            "short_clip_reference_rr_bpm_not_window_truth": passed_with_manifest["reference_rr_bpm"],
            "manual_breath_count": pd.NA,
            "manual_rr_bpm": pd.NA,
            "annotation_status": "pending",
            "labeler_id": "",
            "review_notes": "",
            "include_in_fixed_window_evaluation": "",
        }
    )
    template.to_csv(args.annotation_template, index=False, encoding="utf-8-sig")
    report = [
        "# Lindian Anchored 30-Second YOLO Quality Screen",
        "",
        "The screen uses source raw frames and YOLO keypoint confidence only. It does not use temperature, breath count, RR, or prediction error.",
        "",
        f"- Windows screened: {len(results)}",
        f"- Windows passing the bilateral gate: {len(passed)}",
        f"- Sampling rate: {args.sample_hz:.3f} Hz",
        f"- Keypoint confidence threshold: {args.conf:.3f}",
        f"- Bilateral-valid fraction threshold: {args.min_bilateral_fraction:.3f}",
    ]
    if not results.empty:
        report.extend(
            [
                "",
                f"- Median bilateral-valid fraction: {results['bilateral_valid_fraction'].median():.3f}",
                f"- Minimum bilateral-valid fraction: {results['bilateral_valid_fraction'].min():.3f}",
                f"- Maximum bilateral-valid fraction: {results['bilateral_valid_fraction'].max():.3f}",
            ]
        )
    args.output_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved quality screen: {args.output_csv.resolve()}")
    print(f"Saved quality screen report: {args.output_report.resolve()}")
    print(f"Saved quality-gated annotation template: {args.annotation_template.resolve()}")
    print(results["quality_gate"].value_counts().to_string())


if __name__ == "__main__":
    main()
