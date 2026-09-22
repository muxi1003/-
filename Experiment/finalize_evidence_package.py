"""Verify delivered evidence, record runtime versions, and generate file-purpose indexes."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from experiment_common import *


def purpose(path):
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    if "/curves/" in rel:
        return "消融逐帧曲线与峰位置；内部数据，不是外测"
    if "/inputs/" in rel:
        return "冻结的同口径r20温度输入"
    if "/historical_sources/" in rel:
        return "历史标注/结果快照；不改原件"
    if "/weights/" in rel:
        return "冻结权重/RF文件，不表示本轮重新训练"
    if "/method_snapshots/" in rel:
        return "既有信号方法溯源快照；端到端适配尚未完成"
    if "/annotations/" in rel:
        return "人工事件参考模板或配套溯源；初始空白不是零次真值"
    if "curated_20260909_v3/" in rel:
        return "最新用户确认范围、候选成员和推理前核验表"
    if "curated_" in rel or "/inventories/" in rel:
        return "数据清查/确认过程版本，最终以release成员清单为准"
    if "/release_" in rel:
        return "数据成员与信号策略冻结记录；不含实际外测成绩"
    if "/runs/" in rel:
        return "实际内部消融结果、统计、图表或运行配置"
    if path.suffix == ".py":
        return "可执行脚本；具体使用与边界见所在目录README"
    if path.suffix == ".md":
        return "使用说明、结果解释或预先定义协议"
    return "审计、接口模板、字段字典或交付状态"


def main():
    inputs = ABLATION / "input_snapshots" / "20260909_v1"
    run = ABLATION / "runs" / "20260909_v1"
    source_checks = []
    for row in read_csv(inputs / "input_hashes.csv").itertuples():
        source_checks.append({"source": row.source, "snapshot": row.snapshot,
                              "source_unchanged": sha256(Path(row.source)) == row.source_sha256,
                              "snapshot_unchanged": sha256(Path(row.snapshot)) == row.sha256})
    if not all(row["source_unchanged"] and row["snapshot_unchanged"] for row in source_checks):
        raise ValueError("Original data or input snapshot changed during this task")
    pred = read_csv(run / "paired_predictions.csv")
    metrics = read_csv(run / "metrics.csv")
    replay = read_csv(run / "historical_replay_audit.csv")
    assert len(pred) == 976 and not pred.duplicated(["cohort", "video_id", "variant"]).any()
    assert len(metrics) == 24 and len(replay) == 122 and replay.matches.map(is_true).all()
    assert len(list((run / "curves").rglob("*.csv"))) == 976
    release_dir = HOLDOUT / "release_20260909_v1"
    release = json.loads((release_dir / "test_release.json").read_text(encoding="utf-8"))
    windows = read_csv(release_dir / "test_windows.csv")
    assert sha256(release_dir / "test_windows.csv") == release["test_windows_sha256"]
    assert len(windows) == 271 and windows.verified_cow_id.nunique() == 168
    assert not (release_dir / "prediction_seal.json").exists(), "Update delivery status if actual inference has begun"
    for name, expected in (("20260909_v1", 49), ("jiufu271_frozen_v1", 271)):
        forms = REFERENCE / "annotations" / name
        data = read_csv(forms / "annotation_windows.csv")
        assert len(data) == expected and data.annotation_status.eq("pending").all()
        assert data.manual_breath_count.eq("").all() and read_csv(forms / "reference_events.csv").empty
        assert data.video_path.map(lambda p: Path(p).is_file()).all()
    write_csv(ROOT / "source_integrity_checks.csv", source_checks)
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "opencv-python", "ultralytics", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_available_under_this_distribution_name"
    write_json(ROOT / "runtime_versions.json", {"python": sys.version, "executable": sys.executable,
                                               "platform": platform.platform(), "packages": packages})
    write_csv(ABLATION / "network_comparison_plan.csv", [
        {"comparison": "YOLOv8n_pose_vs_YOLO11n_pose", "status": "NOT_RUN", "checkpoint_available": "yes_both",
         "required_before_claim": "same training/validation split and annotation version; same training budget; no cow/session leakage",
         "metrics": "box/keypoint metrics; localization failure; end_to_end_RR; measured runtime"},
        {"comparison": "YOLO11n_pose_vs_P3_attention", "status": "NOT_RUN_IN_THIS_PACKAGE", "checkpoint_available": "not_audited",
         "required_before_claim": "matched training setup and seeds; freeze before independent test; reject test-guided tuning",
         "metrics": "keypoint metrics; end_to_end_RR; parameters; FLOPs; hardware_latency"},
        {"comparison": "fixed20_vs_adaptive_ROI", "status": "HISTORICAL_RESULTS_NOT_MATCHED_SINGLE_FACTOR_HERE", "checkpoint_available": "same_detector_possible",
         "required_before_claim": "same keypoints, frame geometry, sampling windows, postprocessing and reference; change ROI rule only",
         "metrics": "paired_RR_error; motion_strata; event_FP_FN; coverage"}])
    write_csv(REFERENCE / "prediction_events_template.csv", [], columns=["window_id", "event_id", "event_time_seconds"])
    write_csv(REFERENCE / "prediction_windows_template.csv", [], columns=["window_id", "prediction_status", "duration_seconds", "timebase", "timebase_verified"])
    fields = [
        ("annotation_windows.csv", "manual_breath_count", "人工确认事件数；complete时为全窗次数，partial时不能当全窗真值", "blank_until_reviewed"),
        ("annotation_windows.csv", "annotation_status", "pending/complete/partial/unobservable", "pending"),
        ("annotation_windows.csv", "predictions_hidden", "本轮未展示算法结果填yes，不能冒称历史从未接触", "blank"),
        ("annotation_windows.csv", "annotator", "实际标注人或固定匿名编号", "blank"),
        ("annotation_windows.csv", "annotation_round", "R1首次；R2另一次独立复核，不覆盖R1", "R1"),
        ("reference_events.csv", "event_time_seconds", "相对本观看窗口起点的真实秒数，范围[0,T)", "blank"),
        ("reference_events.csv", "event_id", "同一窗口同轮次唯一，例如e001", "blank"),
        ("reference_events.csv", "event_type", "统一呼气相代表时点expiration_peak，不能确认则记不确定", "expiration_peak"),
        ("reference_events.csv", "confidence", "confirmed或uncertain；uncertain不能进入complete主参考", "blank"),
        ("reference_events.csv", "event_start_seconds/event_end_seconds", "可选相位不确定范围，必须包含代表时点；不自动扩大评分容差", "blank"),
        ("unobservable_intervals.csv", "start_seconds/end_seconds", "真正不可观察时段，起点小于终点，区间不得重叠", "blank"),
        ("unobservable_intervals.csv", "reason", "鼻孔出视野/遮挡/运动模糊/相位无法分辨等实际原因", "blank"),
        ("prediction_windows_template.csv", "timebase", "必须是annotation_video_seconds，且timebase_verified=yes才可事件评分", "blank"),
    ]
    write_csv(REFERENCE / "field_dictionary.csv", [{"file": f, "field": n, "meaning": m, "initial_value": d} for f, n, m, d in fields])
    write_json(ROOT / "DELIVERY_STATUS.json", {
        "date": "2026-09-09", "ablation": {"status": "RUN_INTERNAL_ONLY", "paired_predictions": 976, "analyses": 24, "default_changed": False},
        "independent_test": {"status": "DATA_AND_SIGNAL_POLICY_FROZEN_END_TO_END_PENDING", "windows": 271,
                             "filename_cow_labels": 168, "external_metrics": None,
                             "exposure": "user_confirmed_no_training_no_RR_tuning_no_RR_result_viewing"},
        "event_reference": {"status": "FORMS_AND_SCORER_READY_AWAIT_HUMAN_EVENTS", "internal_windows": 49, "external_windows": 271, "completed": 0},
        "integrity": {"all_original_sources_unchanged": True, "historical_replay_matches": "122/122", "curve_csv_files": 976,
                      "review_scope": "deterministic_checks_and_local_self_review_not_external_reviewer"},
        "not_done": ["YOLO_retraining", "full_end_to_end_external_inference", "human_event_annotation", "RGB_synchronization", "double_blind_annotation"]})
    manifest_rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "FILE_MANIFEST.csv":
            continue
        manifest_rows.append({"relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": sha256(path), "purpose": purpose(path)})
    write_csv(ROOT / "FILE_MANIFEST.csv", manifest_rows)
    print(f"Verified 976 predictions, 24 metric rows, 122 historical replays, 271 frozen windows; indexed {len(manifest_rows)} files")


if __name__ == "__main__":
    main()
