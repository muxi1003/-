"""Archive edited references and rescore existing internal predictions, without tuning."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

from experiment_common import *

sys.path.insert(0, str(REFERENCE))
from reference_tools import validate_tables


def decode_table(data):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = data.decode(encoding, errors="strict")
            return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("CSV is neither UTF-8 nor GB18030")


def valid_counts(values):
    n = pd.to_numeric(values, errors="coerce")
    return n.notna() & np.isfinite(n) & n.ge(0) & n.mod(1).eq(0)


def comparable_path(value):
    return str(value).replace("\\", "/").casefold()


def declaration_for(cohort, declarations, source_hash):
    item = declarations[cohort]
    if item.get("annotation_windows_sha256") != source_hash:
        raise ValueError("Blinding declaration must bind this exact submitted window table")
    if not isinstance(item.get("predictions_hidden"), bool) or not item.get("user_statement", "").strip():
        raise ValueError("Explicit user blinding declaration required")
    return item


def verify_delivery_file(path):
    index = read_csv(ROOT / "FILE_MANIFEST.csv").set_index("relative_path")
    if sha256(path) != index.loc[str(path.relative_to(ROOT)), "sha256"]:
        raise ValueError(f"Frozen delivery input changed: {path}")


def check_identity(windows, expected, external):
    if windows.window_id.duplicated().any() or len(windows) != len(expected):
        raise ValueError("Duplicate or changed window population")
    key = "window_id" if external else "video_id"
    if set(windows[key]) != set(expected[key]):
        raise ValueError("Window membership differs from frozen population")
    joined = windows.merge(expected, on=key, suffixes=("_submitted", "_frozen"), validate="one_to_one")
    if not np.allclose(joined.duration_seconds_submitted.astype(float), joined.duration_seconds_frozen.astype(float), rtol=0, atol=1e-8):
        raise ValueError("Submitted duration differs from frozen duration")
    if external:
        if not np.allclose(joined.source_start_seconds.astype(float), joined.start_seconds.astype(float), rtol=0, atol=1e-8):
            raise ValueError("Submitted source start differs from release")
        if any(comparable_path(a) != comparable_path(b) for a, b in zip(joined.source_path_submitted, joined.source_path_frozen)):
            raise ValueError("Submitted source path differs from release")
    else:
        if not windows.window_id.eq(windows.video_id + "_anchored_30s").all():
            raise ValueError("Unexpected internal window identifier")
        if not np.allclose(joined.source_start_seconds.astype(float), joined.start_seconds.astype(float), rtol=0, atol=1e-8):
            raise ValueError("Submitted internal source start differs from snapshot")
        if any(comparable_path(a) != comparable_path(b) for a, b in zip(joined.source_path, joined.raw_source_path)):
            raise ValueError("Submitted internal source path differs from snapshot")
    return joined


def archive_group(source, target):
    tables, manifest = {}, []
    for name in ("annotation_windows.csv", "reference_events.csv", "unobservable_intervals.csv"):
        p = source / name
        data = p.read_bytes()
        table, encoding = decode_table(data)
        raw = target / "original_bytes" / name
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("xb") as handle:
            handle.write(data)
        digest = hashlib.sha256(data).hexdigest()
        if sha256(p) != digest:
            raise ValueError("Human reference changed during snapshot; retry in a new directory")
        tables[name] = table
        write_csv(target / "utf8_preserved_columns" / name, table)
        manifest.append({"source": str(p), "snapshot": str(raw), "sha256": digest, "encoding": encoding,
                         "rows": len(table), "source_modified": False})
    write_json(target / "manifest.json", manifest)
    return tables, manifest


def rescore49(windows, matched, out, reference_manifest, declaration_path, matched_path):
    reference = windows[["video_id", "manual_breath_count", "annotation_status"]].copy()
    reference["truth_count"] = reference.manual_breath_count.astype(int)
    ref = matched.merge(reference, on="video_id", validate="one_to_one", suffixes=("_old", ""))
    changed = ref.truth_count_old.astype(int).ne(ref.truth_count)
    write_csv(out / "label_comparison_all49.csv", ref[["video_id", "truth_count_old", "truth_count", "reference_reliability", "annotation_status", "duration_seconds"]])
    pred_path = ABLATION / "runs/20260909_v1/paired_predictions.csv"
    verify_delivery_file(pred_path)
    variants = read_csv(pred_path).query("cohort == 'anchored49'")
    if len(variants) != 392 or variants.duplicated(["video_id", "variant"]).any():
        raise ValueError("Incomplete or duplicate frozen factorial predictions")
    records = variants[["video_id", "variant", "predicted_count", "duration_seconds", "cluster_id"]].copy()
    historical = matched[["video_id", "historical_predicted_count", "duration_seconds", "cluster_id"]].rename(columns={"historical_predicted_count": "predicted_count"})
    historical["variant"] = "historical_motion_robust"
    records = pd.concat([records, historical], ignore_index=True).merge(reference, on="video_id", validate="many_to_one")
    frozen_duration = records.video_id.map(matched.set_index("video_id").duration_seconds).astype(float)
    if not np.allclose(records.duration_seconds.astype(float), frozen_duration, rtol=0, atol=1e-8):
        raise ValueError("Prediction durations differ from frozen windows")
    for col in ("predicted_count", "duration_seconds"):
        records[col] = records[col].astype(float)
    records["truth_rr_bpm"] = 60 * records.truth_count / records.duration_seconds
    records["predicted_rr_bpm"] = 60 * records.predicted_count / records.duration_seconds
    records["annotation_blinding"] = "not_hidden_user_confirmed_20260911"
    records["evaluation_role"] = "internal_count_only_not_event_validation"
    metrics, contrasts = [], []
    for name, data in (("all49_sensitivity", records), ("completed39_count_reference", records[records.annotation_status.eq("completed")])):
        for variant, group in data.groupby("variant"):
            if len(group) != (49 if name.startswith("all") else 39):
                raise ValueError("Unexpected count analysis population")
            metrics.append({"analysis_set": name, "variant": variant, **measurements(group)})
        for index, factor in enumerate("FGP"):
            for background in itertools.product((0, 1), repeat=2):
                off = list(background)
                off.insert(index, 0)
                on = off.copy()
                on[index] = 1
                a, b = [f"F{x[0]}G{x[1]}P{x[2]}" for x in (off, on)]
                ga = data[data.variant.eq(a)].set_index("video_id")
                gb = data[data.variant.eq(b)].copy()
                pair = gb.merge(ga[["predicted_rr_bpm"]], left_on="video_id", right_index=True, suffixes=("", "_off"), validate="one_to_one")
                pair["paired_abs_error_delta"] = abs(pair.predicted_rr_bpm - pair.truth_rr_bpm) - abs(pair.predicted_rr_bpm_off - pair.truth_rr_bpm)
                contrasts.append({"analysis_set": name, "factor": factor, "off": a, "on": b, **paired_cluster_bootstrap(pair)})
    write_csv(out / "predictions_rescored.csv", records)
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "factorial_contrasts.csv", contrasts)
    write_json(out / "provenance.json", {"count_changes": int(changed.sum()), "prediction_source_sha256": sha256(pred_path),
               "historical_prediction_and_duration_source": str(matched_path), "historical_source_sha256": sha256(matched_path),
               "submitted_reference_manifest": str(reference_manifest), "submitted_reference_manifest_sha256": sha256(reference_manifest),
               "blinding_declarations_sha256": sha256(declaration_path), "processor_sha256": sha256(Path(__file__)),
               "common_metrics_source_sha256": sha256(ROOT / "experiment_common.py"),
               "inference_rerun": False, "parameter_tuning": False, "default_changed": False,
               "scope": "49 internal windows; 39 completed count labels, not confirmed event references",
               "ablation_scope": "8 fixed-radius20 arms; historical baseline has a different ROI policy; do not attribute their difference to one factor"})
    return metrics, int(changed.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--declarations", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("Unsafe run id")
    out = REFERENCE / "submissions" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    declarations = json.loads(args.declarations.read_text(encoding="utf-8"))
    write_json(out / "annotation_declarations.json", declarations)
    write_json(out / "run_provenance.json", {"processor_sha256": sha256(Path(__file__)),
               "source_declarations": str(args.declarations), "source_declarations_sha256": sha256(args.declarations),
               "created_at": stamp(), "command_run_id": args.run_id})
    matched_path = ABLATION / "input_snapshots/20260909_v1/matched_windows.csv"
    verify_delivery_file(matched_path)
    matched = read_csv(matched_path).query("cohort == 'anchored49'")
    old_path = matched_path.parent / "historical_sources/lindian_anchored30_quality_gated_annotation_template.csv"
    verify_delivery_file(old_path)
    old, _ = decode_table(old_path.read_bytes())
    matched = matched.merge(old[["video_id", "window_start_seconds"]].rename(columns={"window_start_seconds": "start_seconds"}), on="video_id", validate="one_to_one")
    release = HOLDOUT / "release_20260909_v1"
    released = json.loads((release / "test_release.json").read_text(encoding="utf-8"))
    if sha256(release / "test_windows.csv") != released["test_windows_sha256"]:
        raise ValueError("Frozen holdout manifest changed")
    expected_test = read_csv(release / "test_windows.csv")
    summaries, actions, metrics = {}, [], []
    for cohort, folder in (("lindian49", "20260909_v1"), ("jiufu271", "jiufu271_frozen_v1")):
        target = out / cohort
        tables, manifests = archive_group(REFERENCE / "annotations" / folder, target)
        windows, events, intervals = [tables[n] for n in ("annotation_windows.csv", "reference_events.csv", "unobservable_intervals.csv")]
        check_identity(windows, expected_test if cohort == "jiufu271" else matched, cohort == "jiufu271")
        valid = valid_counts(windows.manual_breath_count)
        filled = windows.manual_breath_count.str.strip().ne("")
        if (filled & ~valid).any():
            raise ValueError("Nonempty invalid count; do not coerce to zero")
        count_status = windows.annotation_status.isin(["complete", "completed", "uncertain"])
        if (count_status & ~valid).any() or (windows.annotation_status.eq("unobservable") & filled).any():
            raise ValueError("Count status conflicts with explicit value or missing-reference policy")
        errors = validate_tables(windows, events, intervals)
        write_json(target / "strict_event_reference_validation.json", {
            "status": "NOT_READY_FOR_EVENT_SCORING", "errors": errors,
            "event_rows": len(events), "interval_rows": len(intervals), "event_metrics": None,
            "interpretation": "Valid total counts do not supply event timestamps; do not fabricate events from predictions"})
        derived = windows.copy()
        derived["source_annotation_status"] = windows.annotation_status
        declaration = declaration_for(cohort, declarations, manifests[0]["sha256"])
        hidden = declaration["predictions_hidden"]
        derived["predictions_hidden"] = "true" if hidden else "false"
        derived["blinding_source"] = declaration["user_statement"]
        derived["count_reference_role"] = np.where(valid, np.where(windows.annotation_status.eq("uncertain"), "uncertain_sensitivity_only", "count_only"), "no_reference_not_zero")
        derived["event_reference_ready"] = "false"
        write_csv(target / "count_reference_only.csv", derived)
        for row in derived.to_dict("records"):
            actions.append({"cohort": cohort, "window_id": row["window_id"], "annotation_status": row["source_annotation_status"],
                            "count_available": row["count_reference_role"] != "no_reference_not_zero",
                            "event_reference_state": "no_human_event_timestamps" if len(events) == 0 else "needs_validation",
                            "reason_from_user": row.get("备注", "") or row.get("reference_notes", ""),
                            "action": "do_not_force_count; retain_in_coverage_denominator" if row["count_reference_role"] == "no_reference_not_zero" else "count_usable_with_cohort_blinding_disclosure; events_required_only_for_event_metrics"})
        summaries[cohort] = {"windows": len(windows), "numeric_counts": int(valid.sum()),
                             "status_counts": windows.annotation_status.value_counts().to_dict(),
                             "event_rows": len(events), "interval_rows": len(intervals),
                             "predictions_hidden_user_confirmed": hidden, "window_identity_check": "PASS",
                             "count_coverage": float(valid.mean()), "source_hashes": manifests}
        if cohort == "lindian49":
            metrics, changes = rescore49(windows, matched, ABLATION / "reference_updates" / args.run_id,
                                        target / "manifest.json", out / "annotation_declarations.json", matched_path)
            summaries[cohort]["count_changes_from_20260909"] = changes
        else:
            notes = windows["备注"] if "备注" in windows else windows.reference_notes
            write_csv(target / "user_reason_counts.csv", notes.value_counts().rename_axis("user_note").reset_index(name="windows"))
            write_csv(target / "unobservable_reference_windows.csv", derived[derived.annotation_status.eq("unobservable")])
    write_csv(out / "reference_followup_by_window.csv", actions)
    write_json(out / "submission_summary.json", summaries)
    lines = ["# 本次标注提交处理结果", "", "## 已完成", "",
             "- 原始字节、编码、哈希及新增中文备注均已归档；未覆盖任何人工表。",
             "- 林甸49窗均有次数，与20260909快照相比次数、时长和可靠性分组不变；39 completed + 10 uncertain。",
             "- 久福271窗：157 complete有计数，114 unobservable未填写次数。人工可计数覆盖率57.93%，不是算法覆盖率或准确率。",
             "- 用户最新更正：林甸未隐藏预测；久福此前没有已有计数和预测标注，人工计数未参考算法结果。久福可描述为单观察者、对算法预测不可见的计数参考，不等同于双观察者/双盲事件验证。原表空白predictions_hidden未改，派生表按用户更正登记。",
             "- 两组逐事件表及不可观察区间表均为空。目前不计算事件Precision/Recall/F1，不用模型峰填补人工事件。", "",
             "## 林甸已有预测复算（未重新推理）", "", "| 分析集 | 方法 | n | RR R² | MAE (次/min) | 次数平均准确度 |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['analysis_set']} | {row['variant']} | {row['n']} | {row['rr_r2']:.6f} | {row['rr_mae_bpm']:.6f} | {row['mean_count_accuracy_percent']:.4f}% |")
    lines += ["", "R²=1-SSE/SST；RR=60×次数/冻结实际时长。未选择新最优规则、未修改默认方法。8组消融采用同一r20输入；历史motion-robust基线ROI策略不同，不作为单因素因果对照。",
              "平均计数准确度=100×mean(max(0,1-|预测次数-人工次数|/人工次数))，仅对人工次数>0计算；不是完全计数正确率。本次49条人工次数均大于0，分母分别为49/39；零参考不能默默混入该分母。",
              "", "## 接下来", "", "1. 久福推理前仍须验证原视频解码、时间戳、图像几何与温标/标定兼容，不能将157个有计数窗口当作全部测试集。",
              "2. 无法辨认的114窗不要求勉强补总数；保留在271窗的参考覆盖率、算法输出率及不可判定分析中。",
              "3. 要声称解决伪峰/漏峰，需在可辨认窗口另填reference_events.csv中的逐次时间；不等于重填总数，不必伪造不能观察的相位。现有非盲参考需如实披露。",
              "4. 久福尚无已封存算法预测，本次不报告其RR R²。冻结方法的后续实现不得读取人工总数用于规则选择。", ""]
    write_text(out / "处理结果与下一步.md", "\n".join(lines))
    print(json.dumps({k: {x: v for x, v in d.items() if x != "source_hashes"} for k, d in summaries.items()}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
