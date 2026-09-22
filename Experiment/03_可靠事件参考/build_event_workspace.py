"""Create a count-free, prediction-free offline event annotation workspace."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, REFERENCE, sha256, write_csv, write_json, write_text


def main():
    out = REFERENCE / "event_workspace/20260914_v3"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    viewing = pd.read_csv(REFERENCE / "event_workspace/browser_media_v1/viewing_copy_manifest.csv", dtype=str).set_index("video_id")
    # Only identifiers and viewing-window metadata are parsed; no count columns.
    for cohort, folder in [("lindian49", "20260909_v1"), ("jiufu271", "jiufu271_frozen_v1")]:
        path = REFERENCE / "annotations" / folder / "annotation_windows.csv"
        columns = ["window_id", "video_id", "video_path", "source_path", "source_start_seconds", "duration_seconds"]
        frame = pd.read_csv(path, encoding="gb18030", usecols=columns, dtype=str, keep_default_na=False)
        for item in frame.to_dict("records"):
            item.update(cohort=cohort, annotation_round="R2", annotator="", annotation_status="pending", manual_breath_count="",
                        event_definition="expiration_peak", predictions_hidden="", reference_notes="",
                        prior_algorithm_exposure="yes" if cohort == "lindian49" else "not_reported_at_R1; must_reconfirm_R2")
            item["browser_video_path"] = viewing.loc[item["video_id"], "browser_video_path"] if cohort == "lindian49" else item["video_path"]
            item["view_start_seconds"] = "0"
            rows.append(item)
    if len(rows) != 320 or len({r["window_id"] for r in rows}) != 320:
        raise ValueError("Unexpected annotation inventory")
    write_csv(out / "annotation_windows.csv", rows)
    for name in ["reference_events.csv", "unobservable_intervals.csv"]:
        source = REFERENCE / "annotations/20260909_v1" / name
        write_csv(out / name, pd.read_csv(source).iloc[:0])
    template = Path(__file__).with_name("event_workspace_template.html").read_text(encoding="utf-8")
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    write_text(out / "index.html", template.replace("__WINDOW_DATA__", payload))
    write_json(out / "workspace_manifest.json", {"windows":320, "reference_events":0, "unobservable_intervals":0,
               "prior_counts_copied":False, "algorithm_outputs_read":False, "annotation_round":"R2",
               "scope":"human_event_entry_tool_not_generated_truth; R1_total_counts_unchanged",
               "generator_sha256":sha256(Path(__file__)), "template_sha256":sha256(Path(__file__).with_name("event_workspace_template.html"))})
    write_text(out / "使用说明.md", """# 人工事件参考工作区

打开index.html。先选择窗口，核对视频路径、起点和时长，再填写标注者及本轮是否隐藏预测。

视频可直接加载；浏览器不允许文件地址时，用“选择对应视频”选择该行原文件，不要选择另外裁剪的片段。久福窗口只看原视频前30秒；林甸使用原49个观看MP4及原表时长。

每次可确认的呼气峰按“记录呼气峰”。不确定事件设为uncertain；看不清的区间分别记录起止并写原因。暂停、进退由播放器控制，已有事件可跳转复核或删除。不要把系统峰或均匀时间点抄作人工事件。

完成一窗后确认状态；界面总数仅统计你本轮确认的人工事件。有不确定事件或不可观察区间时不能标complete。待做窗口保持pending，不能当作0。

分别导出窗口表、事件表、不可观察区间表，三个文件放同一个新目录，再使用reference_tools.py验证。另导出JSON备份，换浏览器前保留备份；本地缓存不是可靠备份。原R1计数和任何算法文件均不被覆盖。

本轮记为R2。林甸已有算法接触历史不能通过隐藏页面“变成盲法”。久福R1对算法不可见；如果现在已经看过预测，R2请如实标为未隐藏并说明。非盲材料只能用于相应诊断，不能冒称独立盲参考。事件F1须等待真实人工事件和时间轴验证。
""")


if __name__ == "__main__":
    main()
