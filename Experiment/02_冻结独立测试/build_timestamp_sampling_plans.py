"""Build holdout frame plans from timestamps only, without RR inference or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *
from timestamp_adapter import POLICY, sample_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release_dir = HOLDOUT / "release_20260909_v1"
    preflight = HOLDOUT / "preflight/20260911_v1"
    release = json.loads((release_dir / "test_release.json").read_text(encoding="utf-8"))
    if sha256(release_dir / "test_windows.csv") != release["test_windows_sha256"]:
        raise ValueError("Frozen population changed")
    windows = read_csv(release_dir / "test_windows.csv")
    timestamps = read_csv(preflight / "decoded_frame_timestamps.csv")
    if set(timestamps.window_id) != set(windows.window_id):
        raise ValueError("Timestamp population mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent / "timestamp_adapter.py"
    with (args.out / "adapter_source_snapshot.py").open("xb") as handle:
        handle.write(source.read_bytes())
    write_json(args.out / "plan_config_before_processing.json", {
        "policy": POLICY, "adapter_sha256": sha256(source), "runner_sha256": sha256(Path(__file__)),
        "frozen_window_manifest_sha256": sha256(release_dir / "test_windows.csv"),
        "timestamp_source_sha256": sha256(preflight / "decoded_frame_timestamps.csv"),
        "human_reference_files_read": [], "model_inference": False,
        "scope": "candidate_input_plan_only_not_prediction_seal_or_algorithm_output_coverage"})
    rows = []
    for row in windows.itertuples():
        t = timestamps[timestamps.window_id.eq(row.window_id)].opencv_ffmpeg_timestamp_seconds.to_numpy(float)
        mapping, status = sample_map(t, float(row.duration_seconds))
        if mapping is not None:
            write_csv(args.out / "sample_maps" / f"{row.window_id}.csv", mapping)
        rows.append({"window_id": row.window_id, "source_path": row.source_path,
                     "source_sha256": row.source_sha256, "sampling_plan_available": mapping is not None,
                     "status": status["prediction_status"], "reason": status["reason"],
                     "source_frames": status.get("source_frames", ""), "target_frames": status.get("target_frames", ""),
                     "reused_source_frames": status.get("reused_source_frames", ""),
                     "inference_performed": False, "predicted_count": ""})
    write_csv(args.out / "all271_sampling_plan_status.csv", rows)
    ready = sum(r["sampling_plan_available"] for r in rows)
    reasons = pd.Series([r["reason"] for r in rows]).value_counts().to_dict()
    write_json(args.out / "summary.json", {"released_windows": len(rows), "timestamp_plan_ready": ready,
               "timestamp_plan_unavailable": len(rows)-ready, "reason_counts": reasons,
               "is_external_rr_evaluation": False, "default_or_release_changed": False})
    write_text(args.out / "README.md", f"# 久福候选采样方案\n\n保留271窗，仅根据实际后端时间戳生成候选输入计划：{ready}窗可建立采样映射，{len(rows)-ready}窗按候选规则拒绝生成完整窗口计数输入。\n\n"
               "本程序未读取人工标注、未运行YOLO/RF/RR，也未生成预测次数或封存正式测试结果。采样方案通过率不是算法输出率或准确率。\n\n"
               "长间隔>0.5秒的整窗不补造呼吸、不压缩时间；短间隔使用现有帧最近邻，重复帧在映射表中明示。时间采样候选在内部49窗验证后保留供进一步验证，空间输入和温标仍需核验。\n")
    print(json.dumps({"ready": ready, "unavailable": len(rows)-ready, "reasons": reasons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
