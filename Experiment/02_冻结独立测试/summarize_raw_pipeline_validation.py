"""Render descriptive pipeline evidence, without accuracy scoring or human labels."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    fixed = pd.read_csv(root / "predictions_unscored.csv", dtype={"video_id": str})
    adaptive_dir = root / "adaptive_roi_20260914_v2"
    adaptive = pd.read_csv(adaptive_dir / "predictions_unscored.csv", dtype={"video_id": str})
    boundaries = pd.read_csv(root / "reference_window_boundary_comparison.csv")
    compared = fixed[["video_id", "predicted_breath_count"]].merge(adaptive[["video_id", "predicted_breath_count"]], on="video_id", suffixes=("_fixed", "_adaptive"), validate="one_to_one")
    valid = compared.dropna()
    changed = valid[valid.predicted_breath_count_fixed.ne(valid.predicted_breath_count_adaptive)]
    reasons = fixed.loc[fixed.predicted_breath_count.isna(), "reason"].value_counts()
    long_signal = adaptive[adaptive.base_longest_missing_frames.gt(3)]
    long_direct = adaptive[adaptive.longest_no_direct_detection_frames.gt(3)]
    verification = json.loads((root / "artifact_verification.json").read_text())
    adaptive_verification = json.loads((adaptive_dir / "artifact_verification.json").read_text())
    lines = ["# 林甸49锚点原始时间输入链验证", "", "日期：2026-09-12原始运行，2026-09-14修复与复核。状态：内部工程诊断，不采用为默认，不是独立外测或准确率验证。", "",
        "## 本次做了什么", "",
        "顺序解码原始长视频，以既有锚点帧的解码时间为零点取[0,30)秒；按8.7 Hz选择最近已有帧、JPEG95编码，再运行冻结YOLO和信号代码。原始序列、采样索引、帧哈希、关键点与温度代理信号、峰事件均保存。缺失画面不靠时间压缩补齐。", "",
        "固定半径分支是20像素；adaptive_roi_20260914_v2另行调用冻结半径表实现，范围16–24像素，根据鼻孔间距变异系数选择原已规定的一次或平方根尺度规则。两分支用同一批新原始时间窗口；未按人工次数选择半径。旧adaptive_roi_v1因中文路径读取失败无效，保留但不用于任何覆盖/性能结论。", "",
        "## 输出与拒绝", "", "| 分支 | 请求窗口 | 数值输出 | 拒绝 |", "|---|---:|---:|---:|",
        f"| 固定半径20诊断 | 49 | {fixed.predicted_breath_count.notna().sum()} | {fixed.predicted_breath_count.isna().sum()} |",
        f"| 冻结尺度自适应ROI | 49 | {adaptive.predicted_breath_count.notna().sum()} | {adaptive.predicted_breath_count.isna().sum()} |", "",
        "这只是内部数值输出覆盖，不是检测成功率；输出含插值/推断，不能当作鼻孔连续可观察。", "",
        "固定半径拒绝原因：" + "；".join(f"{k}：{v}窗" for k, v in reasons.items()) + "。拒绝值为空，不填0。", "",
        f"双方均有输出的{len(valid)}窗中，{len(changed)}窗计数不同；没有人工评分，不能称其中任何改变为改善。", "",
        "## 信号可观察性风险", "",
        f"有{len(long_signal)}窗的基础温度表存在超过3个连续网格样本的双侧温度缺失；有{len(long_direct)}窗存在超过3个连续样本没有任一侧直接检测。这是描述性检查，不是本轮新采用的拒绝阈值。", "",
        "旧repair_missing可能跨较长缺失插值；半径表也可能在已有低置信度或推断坐标处取值。温度数值存在不等于真实鼻孔可见。因此暂不放行正式外测，下一步需单独版本化信号可观察性/缺失保护，不能根据久福人工次数调门限。", "",
        "## 为什么不计算新R²", "",
        f"与旧49窗原始帧范围相比，{int(boundaries.frame_content_scope.eq('same_index_range').sum())}窗索引范围相同，{int(boundaries.frame_content_scope.ne('same_index_range').sum())}窗边界或起点不一致。参见reference_window_boundary_comparison.csv。旧人工计数针对CFR观看窗口，不能静默迁移到新物理时间窗口。", "",
        f"其中{int((~boundaries.same_origin_timestamp).sum())}窗的顺序解码起始时间戳与先前按帧号seek的记录不一致。这是后端时间/定位一致性问题，仅凭该差值不能断言画面一定发生位移；本轮不将先前seek时间作为绝对时钟真值。", "",
        "本轮不要求重填全部旧计数表，先按边界对照决定必要复核。原49全量R²=0.888037及既有候选结果仍保留原适用范围；这里没有新准确率、事件F1或久福R²。", "",
        "## 审查与验证", "",
        f"固定分支：{verification['status']}，核验{verification['verified_prediction_frames']}个预测网格帧及其文件哈希、索引、曲线峰和次数一致性。",
        f"自适应分支：{adaptive_verification['status']}，重放{adaptive_verification['replayed_predicted_windows']}窗半径选择和计数。", "",
        "49项合成软件测试通过，不是49个动物实验。审查记录见REVIEW_AND_CORRECTIONS.md。执行版读取过含内部计数列的成员表但未使用这些列预测，原协议truth_read=false已被更正说明；未来runner改为仅解析ID/cohort。本轮没有读取久福人工次数或运行久福预测。", "",
        "温标未知仍保留未知：模型输出是固定RF映射代理信号，绝对摄氏温度准确性未验证；没有统一加3℃。依赖版本另记于adaptive_roi_20260914_v2/runtime_environment.json，不声称完整跨环境运行冻结。", "",
        "## 文件用途", "",
        "- predictions_unscored.csv：固定半径数值预测/拒绝原因，不含人工真值。",
        "- sampled_frames/、timestamps/、maps/、frame_hashes/：原始时间、网格映射和YOLO输入证据。",
        "- temperatures/、curves/、algorithm_events/：固定半径信号和算法事件，不是人工事件。",
        "- adaptive_roi_20260914_v2/：有效的独立自适应半径表、曲线、预测和缺失描述。",
        "- reference_window_boundary_comparison.csv：只比较窗口帧范围，不迁移标签。",
        "- artifact_verification.json、adaptive_roi_20260914_v2/artifact_verification.json：确定性一致性检查。", ""]
    with (root / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print("\n".join(lines[8:23]))


if __name__ == "__main__":
    main()
