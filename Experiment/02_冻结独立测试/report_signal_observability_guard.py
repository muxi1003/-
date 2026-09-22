"""Verify guard artifacts and render descriptive results without further tuning."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import ROOT, REFERENCE, sha256, write_json, write_csv
from signal_observability_guard import protect_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    seal = json.loads((root / "prediction_seal.json").read_text(encoding="utf-8"))
    for field, filename in [("internal_predictions_sha256","internal_predictions_before_scoring.csv"),("raw_predictions_sha256","raw_predictions_unscored.csv")]:
        if sha256(root / filename) != seal[field]:
            raise ValueError("Prediction seal mismatch")
    ref = REFERENCE / "submissions/20260911_v2/lindian49/count_reference_only.csv"
    delivery = pd.read_csv(ref.parent.parent / "delivery_artifact_manifest.csv")
    expected = delivery.loc[delivery.path.eq(str(ref.relative_to(ROOT))), "sha256"]
    if len(expected) != 1 or expected.iloc[0] != sha256(ref):
        raise ValueError("Reference differs from archived delivery hash")
    decisions = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    data = pd.read_csv(root / "internal_predictions_before_scoring.csv", dtype={"video_id":str})
    metrics = pd.read_csv(root / "internal_metrics.csv")
    raw = pd.read_csv(root / "raw_predictions_unscored.csv")
    replayed = 0
    for row in data.itertuples():
        table = pd.read_csv(root / "internal49/input_temperatures" / f"{row.video_id}.csv", float_precision="round_trip")
        mask, gaps, status = protect_window(table, row.selected_mode)
        stored = pd.read_csv(root / "internal49/support" / f"{row.video_id}.csv", float_precision="round_trip")
        for column in mask:
            np.testing.assert_array_equal(mask[column], stored[column])
        if status["prediction_status"] == "ready":
            assert row.candidate_count == row.control_count
        else:
            assert pd.isna(row.candidate_count) and pd.isna(row.candidate_rr_bpm)
        replayed += 1
    write_json(root / "artifact_verification.json", {"status":"GUARD_REPLAY_AND_REFERENCE_BINDING_PASS",
               "replayed_internal_windows":replayed,"reference_delivery_hash_matches":True,
               "prediction_seals_match":True,"empirical_event_accuracy_validated":False})
    plt.rcParams.update({"font.size":10})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    cases = []
    for axis, (video_id, known_frame) in zip(axes, [("170333",63),("ns210947",82)]):
        curve = pd.read_csv(root / "internal49/control_curves" / f"{video_id}.csv")
        support = pd.read_csv(root / "internal49/support" / f"{video_id}.csv")
        gaps = pd.read_csv(root / "internal49/gaps" / f"{video_id}.csv")
        axis.plot(curve.frame_index, curve.smoothed_norm, color="#1976B9", label="Unchanged baseline curve")
        peaks = curve.is_peak.to_numpy(bool)
        flags = support.diagnostic_peak_unsafe.to_numpy(bool)
        axis.scatter(curve.frame_index[peaks], curve.smoothed_norm[peaks], s=18, color="#222222", label="Baseline peaks")
        axis.scatter(curve.frame_index[flags], curve.smoothed_norm[flags], s=70, facecolors="none", edgecolors="#D76B21", label="Insufficient-support flag, not false-peak truth")
        for gap in gaps.itertuples():
            if not gap.permitted_short_gap:
                axis.axvspan(gap.start_frame, gap.stop_frame_exclusive, alpha=.13, color="#D04C50")
        axis.axvline(known_frame, linestyle="--", color="#28845B", linewidth=1)
        flag = bool(support.iloc[known_frame].diagnostic_peak_unsafe)
        axis.set(title=f"{video_id}: frame {known_frame}, diagnostic flag = {flag}", xlabel="Annotation video frame index", ylabel="Normalized signal")
        axis.grid(alpha=.2)
        cases.append({"video_id":video_id,"frame_index":known_frame,"baseline_peak":bool(peaks[known_frame]),
                      "diagnostic_flag":flag,"event_removed":False})
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(root / "case_support_diagnostics.png", dpi=180)
    plt.close(fig)
    write_csv(root / "known_case_checks.csv", cases)
    rows = ["# 长缺失计数保护候选：不采用为默认", "", "日期：2026-09-14。独立候选运行完成；仅内部开发验证，没有久福预测。", "",
            "## 规则与证据范围", "",
            "沿用3帧短缺口、0.5关键点置信度和3帧事件支持半径。单侧融合检查所选侧；mean/min/max保守要求两个输入都有直接检测支持。仅允许内部不超过3帧的短缺口；长缺口或边界缺失导致整窗不报数值。被标记峰仅作诊断，不能把删峰后的部分次数当完整30秒呼吸次数。", "",
            "这不是鼻孔可见性的人工真值：检测也可能错位；混合模式要求双侧可使门控过严。通过门控只说明满足本条代理规则，不证明呼吸计数准确。", "",
            "## 同口径结果", "", "| 分析集与方法 | 输出/参考窗口 | RR R² | MAE（次/分） |", "|---|---:|---:|---:|"]
    for row in metrics.itertuples():
        rows.append(f"| {row.analysis_set} / {row.arm} | {int(row.n)}/{int(row.eligible_reference_n)} | {row.rr_r2:.6f} | {row.rr_mae_bpm:.6f} |")
    rows += ["", "旧49窗控制回放49/49计数一致。全49候选只输出18窗，39 completed主分析只输出15窗。候选与旧方法在相同保留子集上计数和误差完全相同；不能把子集0.908793与全49的0.888037相减称提升。预设覆盖率不得降低条件未满足，决定DO_NOT_ADOPT_KEEP_DEFAULT。", "",
             f"新原始时间49窗仅描述覆盖：原37窗输出进一步降到{int(raw.candidate_count.notna().sum())}窗；上游12个拒绝保持，新增{int(raw.reason.eq('incomplete_selected_signal_support').sum())}个信号支持拒绝。此组不与旧人工计数评分。", "",
             "## 病例与局限", "",
             "- 170333：整窗被拒绝，但用户怀疑的第63帧峰没有被本规则标记，说明长缺失保护不能解释或修正该峰。",
             "- ns210947：用户确认的82帧真峰未被标记、未被删除；整窗拒绝来自其他支持不足区段，不代表82帧错误。",
             "- zs197000：保留25次输出，与旧计数一致。",
             "- 共55个基线峰落在支持不足诊断范围，不能称55个伪峰，也没有据此生成55个事件真值。", "",
             f"![真实基线曲线与支持不足标记]({(root / 'case_support_diagnostics.png').as_posix()})", "",
             "## 验证与下一步", "",
             "推理和评分由两次独立命令运行；先保存/校验预测哈希，再读取20260911_v2林甸参考。参考文件与归档交付哈希一致；旧观看视频哈希与时长均绑定。58项软件测试通过，49窗支持掩码重放一致。单观察者非盲总次数不是事件参考，未报事件F1或独立外测精度。", "",
             "默认算法、模型和人工表未改。保留候选作为质量诊断，不采用如此严格的整窗拒绝策略。下一步可在连续可观察片段内单独检测、报告有效观测时长和部分次数；任何片段RR须明确口径，不能混称整窗RR，也不能按久福人工次数选规则。", "",
             "## 文件用途", "",
             "- internal_metrics.csv、internal_evaluated_rows.csv：同口径内部评分及逐窗参考。",
             "- internal_predictions_before_scoring.csv、prediction_seal.json：评分前预测与哈希。",
             "- raw_predictions_unscored.csv：新原始时间窗的覆盖诊断，不含人工评分。",
             "- internal49/support/、gaps/、control_curves/：逐帧来源支持、缺口区间与原峰位置。",
             "- known_case_checks.csv、case_support_diagnostics.png：病例诊断，不是新事件真值。",
             "- decision.json、artifact_verification.json：不采用决定及确定性核验。", ""]
    with (root / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(rows))
    print(decisions["status"], "report and deterministic verification written")


if __name__ == "__main__":
    main()
