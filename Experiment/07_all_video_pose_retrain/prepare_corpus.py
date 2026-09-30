"""Inventory and losslessly extract the existing cow-RR video cohorts.

The split is by farm and filename-derived cow number, never by adjacent frame.
Only the original 73-video images currently have YOLO pose labels. New frames
remain in a manual-review queue and are not silently used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "Dataset_new" / "72video"
EXP6 = ROOT / "Experiment" / "06_林甸扩展试验"
EXP4 = ROOT / "Experiment" / "04_跨牧场机制补强"
OUT = HERE / "corpus_v1"
SEED = 20260929
IMG_RE = re.compile(r"^(?P<video>.+)_frame_(?P<frame>\d+)$")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def cow_group(farm: str, label: str) -> str:
    match = re.search(r"\d+", label)
    if match is None:
        raise ValueError(f"Cannot identify cow number: {farm} {label}")
    return f"{farm}:{int(match.group())}"


def frame_row(segment: dict, frame: int, image: Path, label: Path | None,
              pts: float | None = None, playback: float | None = None) -> dict:
    return {
        "segment_id": segment["segment_id"], "cohort": segment["cohort"],
        "cow_group": segment["cow_group"], "source_frame": frame,
        "source_pts_seconds": pts, "playback_seconds": playback,
        "image_path": str(image), "label_path": str(label) if label else "",
        "label_status": "existing_73_label" if label else "pending_manual_review",
    }


def choose_group_split(segments: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    weights = segments.groupby(["cow_group", "farm"], as_index=False).expected_frames.sum()
    labeled = frames[frames.label_status.eq("existing_73_label")].groupby("cow_group").size()
    weights["labeled_frames"] = weights.cow_group.map(labeled).fillna(0).astype(int)
    farms = {farm: weights.index[weights.farm.eq(farm)].to_numpy() for farm in weights.farm.unique()}
    if set(farms) != {"lindian", "jiufu"}:
        raise ValueError(f"Unexpected farms: {set(farms)}")
    rng = np.random.default_rng(SEED)
    total = int(weights.expected_frames.sum())
    labeled_total = int(weights.labeled_frames.sum())
    count_by_farm = {farm: max(1, round(len(indices) * 0.2)) for farm, indices in farms.items()}
    best_score = float("inf")
    best: set[int] | None = None
    for _ in range(10000):
        selected = set()
        for farm, indices in farms.items():
            selected.update(int(x) for x in rng.choice(indices, count_by_farm[farm], replace=False))
        val_total = int(weights.loc[list(selected), "expected_frames"].sum())
        val_labeled = int(weights.loc[list(selected), "labeled_frames"].sum())
        score = abs(val_total / total - 0.2) + abs(val_labeled / labeled_total - 0.2)
        if score < best_score:
            best_score, best = score, selected
            if score == 0:
                break
    assert best is not None
    weights["split"] = np.where(weights.index.isin(best), "val", "train")
    if weights.cow_group.duplicated().any():
        raise ValueError("Duplicate cow group")
    return weights.sort_values(["farm", "cow_group"]).reset_index(drop=True)


def inventory() -> None:
    if OUT.exists():
        raise FileExistsError(f"Corpus output already exists: {OUT}")
    segments: list[dict] = []
    frames: list[dict] = []
    source_hashes: dict[Path, str] = {}

    def add_segment(segment: dict, expected_hash: str | None = None) -> None:
        source = Path(segment["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if source not in source_hashes:
            source_hashes[source] = digest(source)
        if expected_hash and source_hashes[source].lower() != expected_hash.lower():
            raise ValueError(f"Frozen source hash mismatch: {source}")
        segment["source_sha256"] = source_hashes[source]
        segments.append(segment)

    short_images: dict[str, list[tuple[int, Path, Path]]] = defaultdict(list)
    for old_split in ("train", "val"):
        for image in sorted((DATA / "images" / old_split).glob("*.jpg")):
            match = IMG_RE.fullmatch(image.stem)
            if match is None:
                raise ValueError(f"Unexpected labeled image name: {image}")
            label = DATA / "labels" / old_split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            short_images[match.group("video")].append((int(match.group("frame")), image, label))
    videos = sorted((DATA / "video").glob("*.MP4"))
    if len(videos) != 73 or set(short_images) != {video.stem for video in videos}:
        raise ValueError("Labeled 73-video membership changed")
    for video in videos:
        items = sorted(short_images[video.stem])
        if [item[0] for item in items] != list(range(len(items))):
            raise ValueError(f"Short-video frames not contiguous: {video.stem}")
        segment = {"segment_id": f"S73_{video.stem}", "cohort": "short73",
                   "farm": "lindian", "cow_group": cow_group("lindian", video.stem),
                   "source_path": str(video), "first_source_frame": 0,
                   "stop_source_frame_exclusive": len(items), "expected_frames": len(items),
                   "extract_required": False, "reference_rounds": "legacy73"}
        add_segment(segment)
        frames.extend(frame_row(segment, number, image, label) for number, image, label in items)

    r3_windows = pd.read_csv(EXP4 / "20260921_v1" / "review24" / "annotation_windows.csv", dtype=str)
    r3_lindian = {str(row.window_id).split("_anchored_30s")[0]
                  for row in r3_windows.itertuples(index=False) if not str(row.window_id).startswith("jiufu_")}
    anchored_videos = sorted((DATA / "lindian_anchored30_videos").glob("*_anchored30s.mp4"))
    if len(anchored_videos) != 49:
        raise ValueError("49 anchored videos changed")
    for video in anchored_videos:
        stem = video.stem.removesuffix("_anchored30s")
        folder = DATA / "lindian_anchored30_frames" / stem
        images = sorted(folder.glob("frame_*.jpg"))
        frame_numbers = [int(image.stem.removeprefix("frame_")) for image in images]
        if not images or frame_numbers != list(range(len(images))):
            raise ValueError(f"Anchored frames missing/non-contiguous: {stem}")
        segment = {"segment_id": f"A49_{stem}", "cohort": "anchored49",
                   "farm": "lindian", "cow_group": cow_group("lindian", stem),
                   "source_path": str(video), "first_source_frame": 0,
                   "stop_source_frame_exclusive": len(images), "expected_frames": len(images),
                   "extract_required": False,
                   "reference_rounds": "anchored49;R3" if stem in r3_lindian else "anchored49"}
        add_segment(segment)
        frames.extend(frame_row(segment, number, image, None) for number, image in zip(frame_numbers, images))
    if not r3_lindian.issubset({video.stem.removesuffix("_anchored30s") for video in anchored_videos}):
        raise ValueError("R3 Lindian reference missing from anchored49")

    r1_ids = set(pd.read_csv(EXP6 / "20260928_r1_reference_audit_v1" / "window_comparison.csv",
                             dtype=str).video_id)
    batches = (
        ("P1", EXP6 / "20260924_expansion_v2" / "annotation_batch_v2"),
        ("P2", EXP6 / "20260928_expansion_p2_v1" / "annotation_batch_v1"),
        ("P3", EXP6 / "20260928_expansion_p3_v2" / "annotation_batch_A"),
        ("P3", EXP6 / "20260928_expansion_p3_v2" / "annotation_batch_B"),
    )
    seen_ids: set[str] = set()
    for cohort, folder in batches:
        batch_segments = pd.read_csv(folder / "segments.csv", dtype=str)
        mapping = pd.read_csv(folder / "frame_time_map.csv", dtype=str)
        for seg in batch_segments.itertuples(index=False):
            video_id = str(seg.video_id)
            if video_id in seen_ids:
                raise ValueError(f"Repeated P1/P2/P3 video: {video_id}")
            seen_ids.add(video_id)
            first, stop = int(seg.first_source_frame), int(seg.stop_source_frame_exclusive)
            mapped = mapping[mapping.video_id.eq(video_id)].sort_values("output_frame", key=lambda x: x.astype(int))
            source_frames = mapped.source_frame.astype(int).tolist()
            if source_frames != list(range(first, stop)) or len(mapped) != int(seg.source_frame_count):
                raise ValueError(f"Frame map changed: {video_id}")
            segment = {"segment_id": f"{cohort}_{video_id}", "cohort": cohort,
                       "farm": "lindian", "cow_group": cow_group("lindian", str(seg.cow_id_from_filename)),
                       "source_path": str(seg.source_path), "first_source_frame": first,
                       "stop_source_frame_exclusive": stop, "expected_frames": len(mapped),
                       "extract_required": True,
                       "reference_rounds": f"P2;R1" if video_id in r1_ids else cohort}
            add_segment(segment, str(seg.source_sha256))
            for row in mapped.itertuples(index=False):
                source_frame = int(row.source_frame)
                image = OUT / "frames" / cohort / video_id / f"source_{source_frame:06d}.png"
                frames.append(frame_row(segment, source_frame, image, None,
                                        float(row.source_pts_seconds), float(row.playback_seconds)))
    if len(seen_ids) != 77 or not r1_ids.issubset(seen_ids):
        raise ValueError("P1/P2/P3 or R1 membership changed")

    points = json.loads((EXP4 / "20260921_v1" / "review24" / "decoded_frame_pts.json").read_text(encoding="utf-8"))
    jiufu = r3_windows[r3_windows.window_id.str.startswith("jiufu_")]
    if len(jiufu) != 12:
        raise ValueError("R3 Jiufu membership changed")
    for window in jiufu.itertuples(index=False):
        source = Path(window.source_path)
        cow = source.stem.rsplit("-", 1)[-1]
        pts = points[window.window_id]
        if not pts or pts[0] != 0 or pts[-1] >= 30:
            raise ValueError(f"Invalid R3 first-30 PTS: {window.window_id}")
        segment = {"segment_id": f"R3_{window.window_id}", "cohort": "R3_jiufu",
                   "farm": "jiufu", "cow_group": cow_group("jiufu", cow),
                   "source_path": str(source), "first_source_frame": 0,
                   "stop_source_frame_exclusive": len(pts), "expected_frames": len(pts),
                   "extract_required": True, "reference_rounds": "R3"}
        add_segment(segment)
        for number, second in enumerate(pts):
            image = OUT / "frames" / "R3_jiufu" / window.window_id / f"source_{number:06d}.png"
            frames.append(frame_row(segment, number, image, None, float(second), float(second)))

    segments_df = pd.DataFrame(segments)
    frames_df = pd.DataFrame(frames)
    if len(segments_df) != 211 or segments_df.segment_id.duplicated().any():
        raise ValueError(f"Expected 211 unique physical clips, got {len(segments_df)}")
    if len(frames_df) != int(segments_df.expected_frames.sum()) or frames_df.image_path.duplicated().any():
        raise ValueError("Frame manifest count/path mismatch")
    groups = choose_group_split(segments_df, frames_df)
    assignment = groups.set_index("cow_group").split
    segments_df["split"] = segments_df.cow_group.map(assignment)
    frames_df["split"] = frames_df.cow_group.map(assignment)
    if frames_df.split.isna().any() or segments_df.split.isna().any():
        raise ValueError("Unassigned group")
    OUT.mkdir(parents=True)
    segments_df.to_csv(OUT / "segments.csv", index=False, encoding="utf-8-sig")
    frames_df.to_csv(OUT / "frames.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "groups_80_20.csv", index=False, encoding="utf-8-sig")
    frames_df[frames_df.label_status.eq("pending_manual_review")].to_csv(
        OUT / "annotation_queue.csv", index=False, encoding="utf-8-sig")
    summary = {"segment_counts": segments_df.groupby("cohort").size().to_dict(),
               "frame_counts": frames_df.groupby("cohort").size().to_dict(),
               "group_counts": groups.groupby(["farm", "split"]).size().to_dict(),
               "split_frames": frames_df.groupby("split").size().to_dict(),
               "split_labeled_frames": frames_df[frames_df.label_status.eq("existing_73_label")]
               .groupby("split").size().to_dict(),
               "pending_review_frames": int(frames_df.label_status.eq("pending_manual_review").sum()),
               "seed": SEED, "split_unit": "farm + numeric filename cow ID",
               "status": "inventory_only; no model training; new frames not yet extracted"}
    summary["group_counts"] = {f"{farm}/{split}": int(value) for (farm, split), value in summary["group_counts"].items()}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def extract() -> None:
    segments = pd.read_csv(OUT / "segments.csv", dtype=str)
    frames = pd.read_csv(OUT / "frames.csv", dtype=str)
    needed = segments[segments.extract_required.eq("True")]
    for seg in needed.itertuples(index=False):
        expected = frames[frames.segment_id.eq(seg.segment_id)].sort_values("source_frame", key=lambda x: x.astype(int))
        if len(expected) != int(seg.expected_frames):
            raise ValueError(f"Frame manifest changed: {seg.segment_id}")
        done = OUT / "done" / f"{seg.segment_id}.json"
        if done.exists():
            if not all(Path(path).is_file() for path in expected.image_path):
                raise ValueError(f"Completed segment has missing frames: {seg.segment_id}")
            continue
        source = Path(seg.source_path)
        if digest(source) != seg.source_sha256:
            raise ValueError(f"Original source changed: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode source: {source}")
        first, stop = int(seg.first_source_frame), int(seg.stop_source_frame_exclusive)
        expected_paths = dict(zip(expected.source_frame.astype(int), expected.image_path))
        count = 0
        try:
            for frame_index in range(stop):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Source decode failed: {source} frame={frame_index}")
                if frame_index < first:
                    continue
                target = Path(expected_paths[frame_index])
                if not target.resolve().is_relative_to((OUT / "frames").resolve()):
                    raise ValueError(f"Output escaped corpus: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError(f"PNG write failed: {target}")
                count += 1
        finally:
            capture.release()
        if count != int(seg.expected_frames):
            raise ValueError(f"Extracted frame count changed: {seg.segment_id}")
        for frame_index in (first, (first + stop - 1) // 2, stop - 1):
            target = Path(expected_paths[frame_index])
            if cv2.imread(str(target)) is None:
                raise RuntimeError(f"Unreadable extracted PNG: {target}")
        done.parent.mkdir(exist_ok=True)
        done.write_text(json.dumps({"segment_id": seg.segment_id, "source_sha256": seg.source_sha256,
                                    "frames": count}, indent=2), encoding="utf-8")
        print(f"{seg.segment_id}: {count} frames", flush=True)
    (OUT / "extraction_complete.json").write_text(json.dumps({
        "segments": len(needed), "frames": int(needed.expected_frames.astype(int).sum()),
        "status": "all planned PNG frames extracted; YOLO labels still pending manual review"
    }, indent=2), encoding="utf-8")
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pending = int(frames.label_status.eq("pending_manual_review").sum())
    summary["status"] = f"extracted; {pending} unlabeled frames pending manual review; no model training"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def verify() -> None:
    segments = pd.read_csv(OUT / "segments.csv", dtype=str)
    frames = pd.read_csv(OUT / "frames.csv", dtype=str)
    groups = pd.read_csv(OUT / "groups_80_20.csv", dtype=str)
    if len(frames) != segments.expected_frames.astype(int).sum():
        raise ValueError("Frame count changed")
    if not all(Path(path).is_file() for path in frames.image_path):
        raise ValueError("Missing frame image")
    labeled = frames[frames.label_status.eq("existing_73_label")]
    if not all(Path(path).is_file() for path in labeled.label_path):
        raise ValueError("Missing original YOLO label")
    extracted = segments[segments.extract_required.eq("True")]
    for seg in extracted.itertuples(index=False):
        done = OUT / "done" / f"{seg.segment_id}.json"
        if not done.is_file():
            raise ValueError(f"Extraction not completed: {seg.segment_id}")
        record = json.loads(done.read_text(encoding="utf-8"))
        if record["source_sha256"] != seg.source_sha256 or record["frames"] != int(seg.expected_frames):
            raise ValueError(f"Extraction record mismatch: {seg.segment_id}")
    mapping = groups.set_index("cow_group").split
    if not frames.cow_group.map(mapping).equals(frames.split):
        raise ValueError("Train/val cow leakage or changed assignment")
    if (segments.groupby("source_sha256").split.nunique() > 1).any():
        raise ValueError("Identical source-video bytes cross train/val")
    if len(segments) != 211 or len(labeled) != 7488:
        raise ValueError("Cohort membership changed")
    aliases_path = OUT / "reference_aliases.csv"
    if aliases_path.is_file():
        alias_rows = pd.read_csv(aliases_path, dtype=str)
        if len(alias_rows) != 18 or not set(alias_rows.included_segment_id).issubset(set(segments.segment_id)):
            raise ValueError("Reference alias map changed")
    print(f"Verified {len(segments)} clips, {len(frames)} frames, {len(labeled)} existing labels; "
          f"{len(groups)} cow/farm groups, no split overlap", flush=True)


def aliases() -> None:
    path = OUT / "reference_aliases.csv"
    if path.exists():
        raise FileExistsError(path)
    segments = set(pd.read_csv(OUT / "segments.csv", dtype=str).segment_id)
    rows = []
    r1 = pd.read_csv(EXP6 / "20260928_r1_reference_audit_v1" / "window_comparison.csv", dtype=str)
    for video_id in sorted(set(r1.video_id)):
        target = f"P2_{video_id}"
        if target not in segments:
            raise ValueError(f"R1 alias target missing: {target}")
        rows.append({"reference_round": "R1", "reference_window_id": video_id,
                     "included_segment_id": target, "reason": "P2 reannotation, same source frames"})
    r3 = pd.read_csv(EXP4 / "20260921_v1" / "review24" / "annotation_windows.csv", dtype=str)
    for window_id in sorted(r3.window_id[~r3.window_id.str.startswith("jiufu_")]):
        target = f"A49_{window_id.split('_anchored_30s')[0]}"
        if target not in segments:
            raise ValueError(f"R3 alias target missing: {target}")
        rows.append({"reference_round": "R3_lindian", "reference_window_id": window_id,
                     "included_segment_id": target, "reason": "Already in anchored49"})
    if len(rows) != 18:
        raise ValueError(f"Expected 18 reference aliases, got {len(rows)}")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} alias rows to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "extract", "verify", "aliases"))
    args = parser.parse_args()
    {"inventory": inventory, "extract": extract, "verify": verify,
     "aliases": aliases}[args.stage]()
