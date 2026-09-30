"""Require explicit per-frame human review before building a new YOLO-Pose set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus_v1"
FRAMES = CORPUS / "frames.csv"
SEGMENTS = CORPUS / "segments.csv"
REVIEWS = CORPUS / "manual_review_status.csv"
REVIEW_META = CORPUS / "manual_review_status.meta.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def frame_key(row: object) -> str:
    return f"{row.segment_id}:{int(row.source_frame):06d}"


def pending_rows() -> pd.DataFrame:
    frames = pd.read_csv(FRAMES, dtype=str).fillna("")
    pending = frames[frames.label_status.eq("pending_manual_review")]
    rows = []
    for row in pending.itertuples(index=False):
        rows.append({
            "frame_key": frame_key(row),
            "cohort": row.cohort,
            "cow_group": row.cow_group,
            "split": row.split,
            "image_path": row.image_path,
            "manual_label_path": str(CORPUS / "manual_labels" / row.segment_id /
                                     f"source_{int(row.source_frame):06d}.txt"),
            "review_status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "notes": "",
        })
    expected = pd.DataFrame(rows)
    if expected.frame_key.duplicated().any():
        raise ValueError("Duplicate frame key in corpus")
    return expected


def init() -> None:
    if REVIEWS.exists() or REVIEW_META.exists():
        raise FileExistsError("Review file already exists; edits must be preserved")
    expected = pending_rows()
    expected.to_csv(REVIEWS, index=False, encoding="utf-8-sig")
    REVIEW_META.write_text(json.dumps({
        "frames_sha256": sha256(FRAMES),
        "segments_sha256": sha256(SEGMENTS),
        "review_rows": len(expected),
        "semantics": "approved means each frame was manually inspected, including empty labels",
    }, indent=2), encoding="utf-8")
    print(f"Created {REVIEWS} with {len(expected)} pending rows", flush=True)


def parse_pose_label(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 11:
            raise ValueError(f"{path}:{number}: expected class, box, and two x/y/v keypoints")
        values = [float(part) for part in parts]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite value")
        if values[0] != 0 or not all(0 <= value <= 1 for value in values[1:5]):
            raise ValueError(f"{path}:{number}: invalid class or normalized box")
        if values[3] <= 0 or values[4] <= 0:
            raise ValueError(f"{path}:{number}: zero-size box")
        x, y, width, height = values[1:5]
        if min(x - width / 2, y - height / 2) < -0.001 or max(
            x + width / 2, y + height / 2
        ) > 1.001:
            raise ValueError(f"{path}:{number}: box extends outside image")
        for offset in (5, 8):
            if not 0 <= values[offset] <= 1 or not 0 <= values[offset + 1] <= 1:
                raise ValueError(f"{path}:{number}: keypoint coordinate outside [0,1]")
            if values[offset + 2] not in (0, 1, 2):
                raise ValueError(f"{path}:{number}: invalid keypoint visibility")


def validated_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not REVIEWS.is_file() or not REVIEW_META.is_file():
        raise FileNotFoundError("Run init after inventory, then review all new frames")
    meta = json.loads(REVIEW_META.read_text(encoding="utf-8"))
    if sha256(FRAMES) != meta["frames_sha256"] or sha256(SEGMENTS) != meta["segments_sha256"]:
        raise ValueError("Frozen corpus manifest changed after review initialization")
    expected = pending_rows()
    actual = pd.read_csv(REVIEWS, dtype=str).fillna("")
    if len(actual) != len(expected) or actual.frame_key.duplicated().any():
        raise ValueError("Review row count or unique frame keys changed")
    fixed = ["frame_key", "cohort", "cow_group", "split", "image_path", "manual_label_path"]
    if not actual[fixed].equals(expected[fixed]):
        raise ValueError("Review source/frame/split/label mapping changed")
    frames = pd.read_csv(FRAMES, dtype=str).fillna("")
    if not (CORPUS / "extraction_complete.json").is_file():
        raise ValueError("Frame extraction has not completed")
    pending = actual[~actual.review_status.eq("approved")]
    missing_signoff = actual[actual.review_status.eq("approved") &
                             (actual.reviewer.eq("") | actual.reviewed_at.eq(""))]
    missing_labels = actual[actual.review_status.eq("approved") &
                            ~actual.manual_label_path.map(lambda name: Path(name).is_file())]
    result = {"total_new_frames": len(actual),
              "approved": int(actual.review_status.eq("approved").sum()),
              "pending_or_other": len(pending),
              "approved_without_reviewer_or_time": len(missing_signoff),
              "approved_without_label_file": len(missing_labels)}
    if any(result[key] for key in ("pending_or_other", "approved_without_reviewer_or_time",
                                   "approved_without_label_file")):
        return frames, actual, result
    for row in frames.itertuples(index=False):
        if not Path(row.image_path).is_file():
            raise FileNotFoundError(row.image_path)
        if row.label_status == "existing_73_label":
            parse_pose_label(Path(row.label_path))
    for name in actual.manual_label_path:
        parse_pose_label(Path(name))
    result["all_labels_valid"] = True
    return frames, actual, result


def check() -> None:
    _, _, result = validated_inputs()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result.get("all_labels_valid"):
        raise SystemExit("Training set blocked: manual review incomplete")


def materialize(name: str) -> None:
    frames, reviews, result = validated_inputs()
    if not result.get("all_labels_valid"):
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit("Training set blocked: manual review incomplete")
    if not name.startswith("yolo_dataset_") or "/" in name or "\\" in name:
        raise ValueError("Dataset name must start with yolo_dataset_ and contain no separators")
    destination = HERE / name
    if destination.exists():
        raise FileExistsError(f"Never overwrite a prepared training set: {destination}")
    manual = dict(zip(reviews.frame_key, reviews.manual_label_path))
    destination.mkdir()
    label_hashes = []
    for row in frames.itertuples(index=False):
        split = row.split
        key = frame_key(row)
        source_image = Path(row.image_path)
        source_label = Path(row.label_path if row.label_status == "existing_73_label" else manual[key])
        image_dest = destination / "images" / split / f"{row.segment_id}__{int(row.source_frame):06d}{source_image.suffix}"
        label_dest = destination / "labels" / split / f"{row.segment_id}__{int(row.source_frame):06d}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_image, image_dest)
        os.link(source_label, label_dest)
        label_hashes.append({"frame_key": key, "label_sha256": sha256(source_label)})
    label_manifest = destination / "label_hashes.csv"
    pd.DataFrame(label_hashes).to_csv(label_manifest, index=False, encoding="utf-8-sig")
    yaml_text = (f"path: {destination.as_posix()}\ntrain: images/train\nval: images/val\n"
                 "nc: 1\nnames: [nose]\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n")
    (destination / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (destination / "READY.json").write_text(json.dumps({
        "frames_sha256": sha256(FRAMES), "segments_sha256": sha256(SEGMENTS),
        "reviews_sha256": sha256(REVIEWS), "label_manifest_sha256": sha256(label_manifest),
        "frames": len(frames),
        "train": int(frames.split.eq("train").sum()),
        "val": int(frames.split.eq("val").sum()),
        "source_labels": "7488 existing labels plus all new manually approved labels",
    }, indent=2), encoding="utf-8")
    print(f"Prepared {destination}; new labels were all manually reviewed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "check", "materialize"))
    parser.add_argument("--name", default="yolo_dataset_v1")
    args = parser.parse_args()
    {"init": init, "check": check,
     "materialize": lambda: materialize(args.name)}[args.stage]()
