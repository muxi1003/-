"""Freeze reviewed JSON, register explicit negatives, and import only a complete scope."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from audit_xanylabeling_submission import convert_pose, folder_name, serialize_pose
from review_gate import CORPUS, FRAMES, HERE, REVIEWS, SEGMENTS, frame_key, pending_rows, sha256


def bounded_boxes(data: dict, tolerance: float = 1.0) -> tuple[dict, list[dict]]:
    """Clip subpixel rectangle overshoot only; never move keypoints or drop instances."""
    result = copy.deepcopy(data)
    changes = []
    width, height = data["imageWidth"], data["imageHeight"]
    for index, shape in enumerate(result.get("shapes", [])):
        if shape.get("label") != "nose" or shape.get("shape_type") != "rectangle":
            continue
        for point in shape["points"]:
            for axis, limit in enumerate((width, height)):
                old = point[axis]
                new = min(max(old, 0), limit)
                if new != old:
                    if abs(new - old) > tolerance:
                        raise ValueError(f"Nose box overshoot exceeds {tolerance} pixel")
                    changes.append({"shape_index": index, "axis": axis, "old": old, "new": new})
                    point[axis] = new
    return result, changes


def negative_keys(declaration: dict) -> set[str]:
    keys = set()
    for span in declaration["confirmed_negative_ranges"]:
        for number in range(span["first_source_frame"], span["stop_source_frame_exclusive"]):
            key = f"{span['segment_id']}:{number:06d}"
            if key in keys:
                raise ValueError(f"Duplicate negative declaration: {key}")
            keys.add(key)
    return keys


def run(source: Path, output: Path, declaration_path: Path, apply: bool) -> None:
    if output.exists():
        raise FileExistsError(f"Preserve earlier imports: {output}")
    declaration_raw = declaration_path.read_bytes()
    declaration = json.loads(declaration_raw.decode("utf-8-sig"))
    if declaration["scope"] != "lindian" or not declaration["human_review_declaration"]:
        raise ValueError("Explicit user-confirmed review and scope are required")
    expected = pending_rows()
    actual = pd.read_csv(REVIEWS, dtype=str).fillna("")
    fixed = ["frame_key", "cohort", "cow_group", "split", "image_path", "manual_label_path"]
    if not actual[fixed].equals(expected[fixed]):
        raise ValueError("Existing review table mapping changed; preserve it for reconciliation")
    meta = json.loads((CORPUS / "manual_review_status.meta.json").read_text(encoding="utf-8"))
    if sha256(FRAMES) != meta["frames_sha256"] or sha256(SEGMENTS) != meta["segments_sha256"]:
        raise ValueError("Frozen corpus manifests changed")
    frames = pd.read_csv(FRAMES, dtype=str).fillna("")
    active = frames[frames.cohort.ne("R3_jiufu")]
    pending = active[active.label_status.eq("pending_manual_review")]
    keys = {frame_key(row) for row in pending.itertuples(index=False)}
    negatives = negative_keys(declaration)
    if not negatives.issubset(keys):
        raise ValueError("Negative declaration contains frames outside active scope")
    if not actual.loc[actual.frame_key.isin(keys), "review_status"].isin(["pending", "approved"]).all():
        raise ValueError("Existing non-pending review decisions require reconciliation")
    output.mkdir(parents=True)
    (output / "user_declaration.json").write_bytes(declaration_raw)
    shutil.copy2(REVIEWS, output / "manual_review_status.before.csv")
    records, issues, changes = [], [], []
    with zipfile.ZipFile(output / "original_json.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for number, row in enumerate(pending.itertuples(index=False), 1):
            key = frame_key(row)
            relative = Path(folder_name(row)) / (Path(row.image_path).stem + ".json")
            path = source / relative
            record = {"frame_key": key, "segment_id": row.segment_id, "cohort": row.cohort,
                      "split": row.split, "cow_group": row.cow_group, "image_path": row.image_path,
                      "json_path": str(path), "json_sha256": "", "origin": "reviewed_json",
                      "instances": 0, "visible_keypoints": 0, "box_clips": 0,
                      "label_path": "", "label_sha256": "", "status": "invalid", "reason": ""}
            try:
                with Image.open(row.image_path) as image:
                    width, height = image.size
                if path.is_file():
                    raw = path.read_bytes()
                    record["json_sha256"] = hashlib.sha256(raw).hexdigest()
                    archive.writestr(relative.as_posix(), raw)
                    data = json.loads(raw.decode("utf-8-sig"))
                    if (str(data.get("imagePath", "")).replace("\\", "/").rsplit("/", 1)[-1]
                            != Path(row.image_path).name
                            or (data.get("imageWidth"), data.get("imageHeight")) != (width, height)):
                        raise ValueError("JSON/image name or dimensions differ")
                    if key in negatives and data.get("shapes"):
                        raise ValueError("Positive JSON conflicts with explicit negative declaration")
                elif key in negatives:
                    data = {"imageWidth": width, "imageHeight": height, "shapes": []}
                    record.update(origin="user_confirmed_negative", json_path="")
                else:
                    raise FileNotFoundError(f"Missing JSON without a user negative declaration: {path}")
                converted_data, clipped = bounded_boxes(data)
                converted = convert_pose(converted_data)
                text = serialize_pose(converted).encode("ascii")
                label = output / "converted_labels" / row.segment_id / f"source_{int(row.source_frame):06d}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_bytes(text)
                record.update(status="validated", instances=len(converted), box_clips=len(clipped),
                              visible_keypoints=sum(int(r[7] == 2) + int(r[10] == 2) for r in converted),
                              label_path=str(label), label_sha256=hashlib.sha256(text).hexdigest())
                changes.extend({"frame_key": key, **change} for change in clipped)
            except (OSError, ValueError, TypeError, KeyError) as error:
                record["reason"] = str(error)
                issues.append(record.copy())
            records.append(record)
            if number % 5000 == 0:
                print(f"Frozen and validated {number}/{len(pending)} reviewed frames", flush=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(output / "annotation_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(issues, columns=manifest.columns).to_csv(output / "issues.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes, columns=["frame_key", "shape_index", "axis", "old", "new"]).to_csv(
        output / "box_boundary_clips.csv", index=False, encoding="utf-8-sig")
    active.to_csv(output / "active_frames.csv", index=False, encoding="utf-8-sig")
    manifest.groupby(["cohort", "status", "origin"]).size().rename("frames").to_csv(
        output / "cohort_summary.csv", encoding="utf-8-sig")
    summary = {"scope": "lindian", "active_frames": len(active), "new_reviewed_frames": len(pending),
               "segments": int(active.segment_id.nunique()), "split_frames": active.groupby("split").size().to_dict(),
               "valid": int(manifest.status.eq("validated").sum()), "issues": len(issues),
               "explicit_empty_labels": int((manifest.status.eq("validated") & manifest.instances.eq(0)).sum()),
               "user_confirmed_negatives_without_json": int(manifest.origin.eq("user_confirmed_negative").sum()),
               "clipped_frames": int(manifest.box_clips.gt(0).sum()), "maximum_box_clip_px":
               max((abs(change["old"] - change["new"]) for change in changes), default=0),
               "frames_sha256": sha256(FRAMES), "segments_sha256": sha256(SEGMENTS),
               "review_table_before_sha256": sha256(output / "manual_review_status.before.csv"),
               "annotation_manifest_sha256": sha256(output / "annotation_manifest.csv"),
               "json_archive_sha256": sha256(output / "original_json.zip"),
               "declaration_sha256": sha256(output / "user_declaration.json"),
               "source_json_modified": False, "review_time_semantics": "User confirmation date; actual per-frame review timestamps unknown",
               "canonical_imported": False, "training_started": False}
    if not issues and apply:
        targets = actual.set_index("frame_key").manual_label_path.to_dict()
        for row in manifest.itertuples(index=False):
            target = Path(targets[row.frame_key])
            if target.exists() and sha256(target) != row.label_sha256:
                raise FileExistsError(f"Existing manual label differs, not overwritten: {target}")
        if sha256(REVIEWS) != summary["review_table_before_sha256"]:
            raise ValueError("Review table changed during import; original edits preserved")
        for row in manifest.itertuples(index=False):
            target = Path(targets[row.frame_key])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(row.label_path, target)
        selected = actual.frame_key.isin(keys)
        actual.loc[selected, "review_status"] = "approved"
        actual.loc[selected & actual.reviewer.eq(""), "reviewer"] = "user (assisted labels manually reviewed)"
        actual.loc[selected & actual.reviewed_at.eq(""), "reviewed_at"] = declaration["confirmation_date"]
        note = f"Import {output.name}; user-confirmed review; reviewed_at records confirmation date, not per-frame review time"
        actual.loc[selected, "notes"] = actual.loc[selected, "notes"].map(lambda s: s + "; " + note if s else note)
        revised = output / "manual_review_status.after.csv"
        actual.to_csv(revised, index=False, encoding="utf-8-sig")
        temp = REVIEWS.with_suffix(".importing.csv")
        if temp.exists():
            raise FileExistsError(temp)
        shutil.copy2(revised, temp)
        os.replace(temp, REVIEWS)
        summary.update(canonical_imported=True, reviews_sha256=sha256(REVIEWS))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if issues:
        raise SystemExit("Import blocked: see issues.csv; canonical labels/review table untouched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "biaozhu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.declaration.resolve(), args.apply)
