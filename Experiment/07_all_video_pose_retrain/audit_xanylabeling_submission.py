"""Audit a partial, human-reviewed X-AnyLabeling submission against fixed frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Experiment" / "04_跨牧场机制补强"))
from strict_pose_label_converter import convert_pose, serialize_pose

CORPUS = HERE / "corpus_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def folder_name(row: object) -> str:
    return row.segment_id.removeprefix(f"{row.cohort}_") if row.cohort != "anchored49" else row.segment_id.removeprefix("A49_")


def audit(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Preserve earlier intake snapshots: {destination}")
    frames = pd.read_csv(CORPUS / "frames.csv", dtype=str).fillna("")
    active = frames[frames.cohort.ne("R3_jiufu")].copy()
    pending = active[active.label_status.eq("pending_manual_review")]
    expected = {}
    for row in pending.itertuples(index=False):
        key = (folder_name(row), Path(row.image_path).stem)
        if key in expected:
            raise ValueError(f"Ambiguous canonical frame mapping: {key}")
        expected[key] = row
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    submitted = {}
    unmatched = []
    for path in files:
        relative = path.relative_to(source)
        key = (relative.parts[0], path.stem) if len(relative.parts) == 2 else None
        if key not in expected:
            unmatched.append({"json_path": str(path), "reason": "Outside active frame manifest; not imported"})
        else:
            submitted.setdefault(key, []).append(path)
    destination.mkdir(parents=True)
    records = []
    for number, (key, row) in enumerate(expected.items(), 1):
        record = {"segment_id": row.segment_id, "cohort": row.cohort,
                  "cow_group": row.cow_group, "split": row.split,
                  "source_frame": int(row.source_frame), "image_path": row.image_path,
                  "expected_json_path": str(source / key[0] / f"{key[1]}.json"),
                  "json_path": "", "json_sha256": "", "status": "missing_json",
                  "reason": "", "instances": 0, "visible_keypoints": 0,
                  "checked_field": "", "converted_label_path": ""}
        candidates = submitted.get(key, [])
        if len(candidates) > 1:
            record.update(status="conflict", reason="Multiple JSON files map to one canonical frame")
        elif candidates:
            path = candidates[0]
            record["json_path"] = str(path)
            try:
                raw = path.read_bytes()
                record["json_sha256"] = hashlib.sha256(raw).hexdigest()
                data = json.loads(raw.decode("utf-8-sig"))
                record["checked_field"] = data.get("checked", "absent")
                image_name = str(data.get("imagePath", "")).replace("\\", "/").rsplit("/", 1)[-1]
                if image_name != Path(row.image_path).name:
                    raise ValueError(f"imagePath mismatch: {image_name}")
                with Image.open(row.image_path) as image:
                    size = image.size
                if size != (data.get("imageWidth"), data.get("imageHeight")):
                    raise ValueError(f"JSON/image size mismatch: {size}")
                converted = convert_pose(data)
                label = destination / "converted_labels" / row.segment_id / f"source_{int(row.source_frame):06d}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(serialize_pose(converted), encoding="ascii")
                record.update(status="validated_json", instances=len(converted),
                              visible_keypoints=sum(int(r[7] == 2) + int(r[10] == 2) for r in converted),
                              converted_label_path=str(label))
            except (OSError, ValueError, TypeError, KeyError) as error:
                record.update(status="conflict", reason=str(error))
        records.append(record)
        if number % 5000 == 0:
            print(f"Audited {number}/{len(expected)} canonical frames", flush=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(destination / "annotation_manifest.csv", index=False, encoding="utf-8-sig")
    manifest[manifest.status.eq("missing_json")].to_csv(destination / "missing_frames.csv", index=False, encoding="utf-8-sig")
    manifest[manifest.status.eq("conflict")].to_csv(destination / "annotation_issues.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unmatched, columns=["json_path", "reason"]).to_csv(destination / "unmatched_json.csv", index=False, encoding="utf-8-sig")
    progress = manifest.groupby(["segment_id", "cohort", "split", "cow_group"], sort=False).agg(
        expected=("status", "size"), present=("json_path", lambda x: int(x.ne("").sum())),
        validated=("status", lambda x: int(x.eq("validated_json").sum())),
        missing=("status", lambda x: int(x.eq("missing_json").sum())),
        conflicts=("status", lambda x: int(x.eq("conflict").sum()))).reset_index()
    progress["complete_and_valid"] = progress.validated.eq(progress.expected)
    progress.sort_values(["complete_and_valid", "cohort", "segment_id"]).to_csv(
        destination / "segment_progress.csv", index=False, encoding="utf-8-sig")
    active.to_csv(destination / "active_frames_lindian.csv", index=False, encoding="utf-8-sig")
    scope = {"scope": "lindian_without_R3_jiufu", "excluded_cohorts": ["R3_jiufu"],
             "active_segments": int(active.segment_id.nunique()), "active_frames": len(active),
             "existing_73_label_frames": int(active.label_status.eq("existing_73_label").sum()),
             "required_reviewed_json_frames": len(pending), "split_frames": active.groupby("split").size().to_dict(),
             "frames_manifest_sha256": digest(CORPUS / "frames.csv"),
             "segments_manifest_sha256": digest(CORPUS / "segments.csv"),
             "user_statement": "X-AnyLabeling-assisted labels followed by per-frame human correction; partial submission; R3 Jiufu deferred",
             "training_started": False, "status": "PROGRESS_AUDIT_ONLY"}
    (destination / "run_scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {**scope, "audited_at": datetime.now().astimezone().isoformat(),
               "submission_directory": str(source), "submitted_json_files": len(files),
               "matched_json_files": sum(len(x) for x in submitted.values()),
               "unmatched_json_files": len(unmatched), "frame_status_counts": manifest.status.value_counts().to_dict(),
               "complete_valid_segments": int(progress.complete_and_valid.sum()),
               "required_segments": len(progress),
               "empty_valid_json_frames": int((manifest.status.eq("validated_json") & manifest.instances.eq(0)).sum()),
               "checked_field_counts": manifest[manifest.json_path.ne("")].checked_field.astype(str).value_counts().to_dict(),
               "annotation_manifest_sha256": digest(destination / "annotation_manifest.csv"),
               "converter_sha256": digest(ROOT / "Experiment" / "04_跨牧场机制补强" / "strict_pose_label_converter.py"),
               "image_association": "folder, exact image basename, and image dimensions; external pixels not independently traced to annotation session",
               "approval": "User-described human review is recorded; software checked field is not a review decision; canonical review table untouched"}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = int(manifest.status.eq("missing_json").sum())
    conflicts = int(manifest.status.eq("conflict").sum())
    text = ("# X-AnyLabeling 标注进度核验\n\n"
            f"本轮按用户安排暂缓R3久福，保留原牛号划分：{len(active)}帧、{active.segment_id.nunique()}片段；"
            f"其中{len(pending)}帧需要新增人工复核标注，7488帧保留原73标签来源。\n\n"
            f"提交JSON {len(files)}份，匹配{summary['matched_json_files']}份，清单外{len(unmatched)}份；"
            f"结构校验通过{int(manifest.status.eq('validated_json').sum())}帧，缺JSON {missing}帧，冲突{conflicts}帧。"
            f"完整且结构通过的片段{int(progress.complete_and_valid.sum())}/{len(progress)}。\n\n"
            "`segment_progress.csv`列出各片段进度；`missing_frames.csv`逐帧列出缺少的JSON；"
            "`annotation_issues.csv`保留转换冲突，不自动删除或替换人工点；`unmatched_json.csv`列出范围外文件。\n\n"
            "`converted_labels/`是本次结构核验转换副本，尚未进入训练。空TXT只对应已提交的明确空shapes JSON，"
            "缺JSON不生成空标签。`checked=false`不否定用户的人工复核说明。待林甸标注全部完成后重新冻结导出，"
            "采用同一牛号划分训练YOLO11n-Pose，随后按固定温度映射、窗口与四组规则重跑呼吸曲线和RR评价。\n")
    (destination / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "biaozhu")
    parser.add_argument("--output", type=Path, default=HERE / "annotation_intake_20261014_v1")
    args = parser.parse_args()
    audit(args.source.resolve(), args.output.resolve())
