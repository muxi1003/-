"""Decode every released first-30s window without loading human references or models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import read_csv, sha256, write_csv, write_json, write_text


def inspect_window(row):
    path = Path(row.source_path)
    if sha256(path) != row.source_sha256:
        raise ValueError(f"Frozen raw video changed: {row.window_id}")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    times, shapes, reached_end = [], set(), False
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start, end = float(row.start_seconds), float(row.start_seconds) + float(row.duration_seconds)
    if start != 0:
        cap.release()
        raise ValueError("Only the predeclared first-window release is supported")
    try:
        # Sequential full decoding, including one boundary frame, avoids approximate seeks.
        for index in range(20000):
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            shapes.add(tuple(frame.shape))
            times.append(pts)
            if pts >= end:
                reached_end = True
                break
    finally:
        cap.release()
    arr = np.asarray(times)
    steps = np.diff(arr)
    monotonic = bool(len(arr) > 1 and np.isfinite(arr).all() and np.all(steps > 0))
    inside = np.flatnonzero((arr >= start) & (arr < end))
    issues = []
    if not reached_end:
        issues.append("decode_did_not_reach_window_boundary")
    if not monotonic:
        issues.append("invalid_or_nonmonotonic_backend_timestamps")
    if not len(arr) or abs(arr[0] - start) > 1e-6:
        issues.append("first_timestamp_not_zero_requires_origin_policy")
    if len(shapes) != 1:
        issues.append("changing_or_absent_frame_geometry")
    if len(steps) and max(steps) > .5:
        issues.append("timestamp_gap_over_0.5_seconds_requires_review")
    nominal_error = float(np.max(abs(arr[inside] - inside / 8.7))) if len(inside) else None
    record = {"window_id": row.window_id, "source_path": str(path), "source_sha256": row.source_sha256,
              "decode_status": "PASS" if not issues else "REVIEW", "decoded_frames_in_window": len(inside),
              "metadata_fps": fps, "metadata_total_frames": total_frames,
              "frame_shapes": json.dumps(sorted(shapes)), "first_timestamp_seconds": float(arr[0]) if len(arr) else None,
              "last_in_window_timestamp_seconds": float(arr[inside[-1]]) if len(inside) else None,
              "boundary_timestamp_seconds": float(arr[-1]) if reached_end else None,
              "monotonic_backend_timestamps": monotonic,
              "max_timestamp_step_seconds": float(max(steps)) if len(steps) else None,
              "max_error_if_index_divided_by_8_7_seconds": nominal_error,
              "issues": ";".join(issues), "calibration_compatibility": "NOT_VERIFIED",
              "visual_modality_review": "NOT_PERFORMED", "runner_uses_reference_counts": False}
    frame_rows = [{"window_id": row.window_id, "decoded_frame_index": int(i),
                   "opencv_ffmpeg_timestamp_seconds": float(t), "inside_half_open_window": start <= t < end}
                  for i, t in enumerate(arr)]
    return record, frame_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((args.release_dir / "test_release.json").read_text(encoding="utf-8"))
    manifest = args.release_dir / "test_windows.csv"
    if sha256(manifest) != release["test_windows_sha256"]:
        raise ValueError("Release manifest hash mismatch")
    windows = read_csv(manifest)
    if len(windows) != release["n_windows"] or windows.window_id.duplicated().any():
        raise ValueError("Changed or duplicate release membership")
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start = time.monotonic()
    write_json(args.out / "run_config.json", {"runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__)),
               "manifest_sha256": sha256(manifest), "opencv_version": cv2.__version__,
               "reference_files_read": [], "model_inference": False, "timestamp_source": "OpenCV FFMPEG CAP_PROP_POS_MSEC",
               "limitation": "Backend presentation timestamps; not an independent ffprobe cross-check or calibration validation"})
    records, timestamps = [], []
    for index, row in enumerate(windows.itertuples()):
        record, frames = inspect_window(row)
        records.append(record)
        timestamps.extend(frames)
        if (index + 1) % 10 == 0 or index == 0:
            print(f"Decoded {index+1}/{len(windows)} windows; elapsed {time.monotonic()-start:.1f}s", flush=True)
    write_csv(args.out / "video_decode_checks.csv", records)
    write_csv(args.out / "decoded_frame_timestamps.csv", timestamps)
    passed = sum(x["decode_status"] == "PASS" for x in records)
    summary = {"status": "DECODE_CHECKED_END_TO_END_NOT_VALIDATED", "windows": len(records),
               "decode_timing_pass": passed, "decode_timing_review": len(records)-passed,
               "elapsed_seconds": time.monotonic()-start, "reference_counts_read": False,
               "calibration_compatibility": "PENDING", "spatial_input_adapter": "PENDING",
               "frozen_prediction_seal_created": False,
               "not_a_validated_timestamp_report_for_sealing": True}
    write_json(args.out / "preflight_summary.json", summary)
    write_text(args.out / "README.md", "# 冻结视频技术检查\n\n逐个核验271原视频SHA256，顺序完整解码首30秒并读到边界帧；未读取人工标注、未运行模型。\n\n"
               f"解码/后端时间戳规则通过{passed}/{len(records)}。原始逐帧时间见decoded_frame_timestamps.csv。\n\n"
               "该检查不验证图像是正确热红外模态、鼻孔可见性、色盘/温标兼容或几何缩放；不能作为端到端已验证证明。"
               "时间为OpenCV FFMPEG后端报告值，非第二解码器独立交叉核验；用frame_index/8.7替代实际时间的误差单列。\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
