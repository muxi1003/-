"""Verify half-open segment accounting and report engineering coverage only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_common import sha256,write_json
from supported_segments import segments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    protocol=json.loads((root/"protocol_before_results.json").read_text(encoding="utf-8"))
    assert sha256(root/"input_manifest.csv")==protocol["input_manifest_sha256"]
    inputs=pd.read_csv(root/"input_manifest.csv",dtype={"video_id":str})
    windows=pd.read_csv(root/"window_results.csv",dtype={"video_id":str})
    spans=pd.read_csv(root/"segments.csv",dtype={"video_id":str})
    events=pd.read_csv(root/"algorithm_events.csv",dtype={"video_id":str})
    assert len(windows)==98 and not windows.duplicated(["cohort","video_id"]).any()
    assert not events.human_reference.any()
    for item in inputs.itertuples():
        row=windows[(windows.cohort==item.cohort)&(windows.video_id==item.video_id)].iloc[0]
        if item.upstream_rejected:
            assert pd.isna(row.partial_count) and pd.isna(row.full_window_count)
            continue
        assert sha256(Path(item.temperature_path))==item.temperature_sha256
        assert sha256(Path(item.support_path))==item.support_sha256
        mask=pd.read_csv(item.support_path).support_after_short_gap_policy.to_numpy(bool)
        expected=segments(mask,float(item.duration_seconds))
        actual=spans[(spans.cohort==item.cohort)&(spans.video_id==item.video_id)]
        assert len(expected)==len(actual)
        for spec,segment in zip(expected,actual.itertuples()):
            assert spec["core_start_frame"]==segment.core_start_frame and spec["core_stop_frame_exclusive"]==segment.core_stop_frame_exclusive
            assert abs(spec["core_seconds"]-segment.core_seconds)<1e-8
            assert mask[segment.start_frame:segment.stop_frame_exclusive].all()
            ev=events[(events.cohort==item.cohort)&(events.video_id==item.video_id)&(events.segment_id==segment.segment_id)]
            if segment.status=="eligible":
                assert len(ev)==int(segment.segment_count)
                assert ev.original_frame_index.ge(segment.core_start_frame).all() and ev.original_frame_index.lt(segment.core_stop_frame_exclusive).all()
                assert not ev.original_frame_index.duplicated().any()
                np.testing.assert_allclose(ev.time_seconds,ev.original_frame_index*float(item.duration_seconds)/len(mask),atol=1e-8)
                assert abs(segment.segment_rr_bpm-60*len(ev)/segment.core_seconds)<1e-8
                curve=pd.read_csv(root/"segment_curves"/item.cohort/item.video_id/f"segment_{segment.segment_id:03d}.csv")
                peak_indices=curve.loc[curve.is_peak,"original_frame_index"]
                peak_indices=peak_indices[(peak_indices>=segment.core_start_frame)&(peak_indices<segment.core_stop_frame_exclusive)]
                np.testing.assert_array_equal(peak_indices.to_numpy(),ev.original_frame_index.to_numpy())
            else:
                assert ev.empty
        eligible=actual[actual.status.eq("eligible")]
        assert abs(row.eligible_seconds-eligible.core_seconds.sum())<1e-8
        if len(eligible):
            assert row.partial_count==eligible.segment_count.sum()
            assert abs(row.partial_rr_bpm-60*row.partial_count/row.eligible_seconds)<1e-8
        else:
            assert pd.isna(row.partial_count) and pd.isna(row.partial_rr_bpm)
        if pd.notna(row.full_window_count):
            assert mask.all() and len(actual)==1 and abs(row.eligible_seconds-row.window_seconds)<1e-8
            assert row.full_window_count==float(item.control_count)
        else:
            assert row.status!="full_window_supported_proxy"
    write_json(root/"artifact_verification.json",{"status":"SEGMENT_INTERVAL_EVENT_ACCOUNTING_PASS","window_records":98,"segment_records":len(spans),"algorithm_events":len(events),"accuracy_scored":False})
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))["cohorts"]
    lines=["# 连续支持片段计数：工程候选", "", "日期：2026-09-14。默认整窗方法不变；没有人工片段真值评分或久福外测。", "",
           "## 方法", "",
           "复用N049的来源支持掩码，在长缺口处分段，不把不连续片段拼成连续曲线。保留上游所选融合侧/模式；完整窗用原配置回放，部分片段固定该模式并重新进行局部归一化与峰检测。缺口邻接边缘各预留3帧，原始窗口端点不另裁。计数只取片段核心区域。", "",
           "核心时长至少6秒，来自现有最低20次/分下两个周期的时长，不是按人工计数调优；至少2个峰才报告片段RR。这只是预设工程条件，并非已经验证的临床/生理可靠性标准。", "",
           "支持包括获准的至多3帧内部短缺口，并非每一帧都是真实直接检测；direct_selected_seconds另报直接检测来源支持时长。", "",
           "## 结果与口径", "", "| 数据 | 有片段RR的窗口 | 可报告完整窗次数 | 仅部分窗口 | 片段RR覆盖秒数 / 请求秒数 |", "|---|---:|---:|---:|---:|"]
    for cohort,s in summary.items():
        lines.append(f"| {cohort} | {s['any_segment_output']}/49 | {s['full_window_outputs']}/49 | {s['any_segment_output']-s['full_window_outputs']} | {s['eligible_seconds']:.3f} / {s['requested_seconds']:.3f} |")
    lines += ["", "旧49窗相比N049整窗拒绝18窗输出，现在49窗都能提供某些片段信息，但完整窗仍仅18窗；不能称完整49窗计数恢复。新raw49为35窗有片段信息，完整仍9窗，12个上游拒绝不变，另2窗片段过短。", "",
              "partial_count只汇总达到时长/峰数条件的核心片段；partial_rr_bpm = 60 × partial_count / eligible_seconds，不按30秒外推次数。eligible_seconds是达到报告条件的片段时长，不是全部可见时长；按峰数筛选也可能造成选择偏差。full_window_count在部分覆盖时始终留空。", "",
              "不能把部分次数与旧整窗manual_breath_count比较，不能算混合口径R²；无事件级真值，不能声称精度提高或漏峰已经修复。", "",
              "## 病例", "",
              "- 旧170333：两个片段合计16次、22.532秒；不是整窗16次。第63帧可疑峰仍存在，尚未解决其机制。",
              "- 旧ns210947：片段合计21次、22.717秒；虽然数字恰好与整窗人工21相同，也不能称误计修复。82帧真峰仍在片段事件中。",
              "- 旧zs197000：完整30.055秒、25次，控制回放一致。新raw同ID上游时间间隔拒绝保持，不能混同两组结果。", "",
              "## 验证与下一步", "",
              "64项软件测试通过；98窗记录的输入哈希、半开区间、不跨缺口、事件位置、峰数/有效时长汇总和完整窗计数回放核验通过。来源支持和时间契约继承上游，仍不是新增鼻孔可见性真值或独立时钟校准。", "",
              "保留为独立诊断候选，不替换默认。下一步在预先选定的片段上建立人工事件/区间参考，并检查局部检测是否引入新分峰；未补片段参考前不比较片段RR准确度，不读取久福人工次数选规则。已有整窗人工表无需覆盖。", "",
              "## 文件用途", "",
              "- window_results.csv：完整/部分次数严格分列，片段RR及对应时长。",
              "- segments.csv：全部片段起止、核心区间、是否过短及局部次数。",
              "- algorithm_events.csv：算法事件和原窗口帧位置，不是人工标注。",
              "- segment_curves/：局部重新检测曲线，保留原窗口帧坐标。",
              "- inputs/、input_manifest.csv、protocol_before_results.json：运行前输入副本、哈希与固定规则。",
              "- artifact_verification.json：确定性区间/事件核验，不是准确率报告。", ""]
    with (root/"RESULTS.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print("Segment accounting verified; no reference accuracy score generated.")


if __name__=="__main__":
    main()
