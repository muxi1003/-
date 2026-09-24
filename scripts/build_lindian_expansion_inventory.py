"""Inventory Lindian originals and freeze a small new-cow annotation batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("E:/real/use_code/林甸红外视频")
MATCHED = ROOT / "Experiment/01_同口径消融/input_snapshots/20260909_v1/matched_windows.csv"
DATE_DIRS = ("0805", "0806", "0807", "0808", "0809", "0810")
NAME = re.compile(r"^202308\d{2}T\d{6}(?:n(.+))?$", re.IGNORECASE)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty inventory")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-day", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.per_day < 1:
        raise ValueError("Output must be new and per-day positive")

    with MATCHED.open(newline="", encoding="utf-8-sig") as stream:
        matched = list(csv.DictReader(stream))
    old_paths = {str(Path(row["raw_source_path"]).resolve()).casefold() for row in matched
                 if row["raw_source_path"]}
    old_cows = set()
    for path in old_paths:
        match = NAME.fullmatch(Path(path).stem)
        if match and match.group(1):
            old_cows.add(match.group(1).casefold())

    inventory = []
    for day in DATE_DIRS:
        for path in sorted((RAW / day).rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue
            match = NAME.fullmatch(path.stem)
            if not match:
                continue
            cow_id = match.group(1) or ""
            capture = cv2.VideoCapture(str(path))
            try:
                opened = capture.isOpened()
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
                fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            finally:
                capture.release()
            duration = frames / fps if fps > 0 else 0.0
            existing = str(path.resolve()).casefold() in old_paths
            known_cow = cow_id.casefold() in old_cows if cow_id else False
            if existing:
                disposition = "known_source"
            elif "无法分辨" in str(path):
                disposition = "source_identity_uncertain"
            elif not cow_id:
                disposition = "cow_id_unknown"
            elif known_cow:
                disposition = "known_cow"
            elif not opened or width <= 0 or height <= 0 or duration < 30.25:
                disposition = "metadata_short_or_invalid"
            else:
                disposition = "candidate_new_cow"
            inventory.append({"date": day, "cow_id": cow_id, "source_path": str(path.resolve()),
                              "file_bytes": path.stat().st_size, "metadata_frames": frames,
                              "metadata_fps": fps, "metadata_duration_seconds": duration,
                              "width": width, "height": height,
                              "known_source_path": existing, "known_cow_id": known_cow,
                              "disposition": disposition})

    # Path hash is only a deterministic sampling order; it is not a content hash.
    selected = []
    seen_cows = set()
    for day in DATE_DIRS:
        choices = [row for row in inventory if row["date"] == day
                   and row["disposition"] == "candidate_new_cow"]
        choices.sort(key=lambda row: hashlib.sha256(
            str(Path(row["source_path"]).relative_to(RAW)).encode("utf-8")).hexdigest())
        for row in choices:
            if row["cow_id"].casefold() in seen_cows:
                continue
            selected.append({"batch_order": len(selected) + 1, **row,
                             "source_sha256": file_hash(Path(row["source_path"])),
                             "selection_reason": "two_per_day_new_cow_path_hash_order"})
            seen_cows.add(row["cow_id"].casefold())
            if sum(item["date"] == day for item in selected) == args.per_day:
                break

    args.output.mkdir(parents=True)
    write_csv(args.output / "source_inventory.csv", inventory)
    write_csv(args.output / "selected_sources.csv", selected)
    (args.output / "freeze.json").write_text(json.dumps({
        "date_dirs": DATE_DIRS, "inventory_count": len(inventory), "selection_count": len(selected),
        "selection": "up to two per date; unknown/known cow, uncertain-identity directory, and old source excluded; deterministic path hash; metadata duration >=30.25 s",
        "limits": ["Metadata duration is not verified PTS continuity", "Known source list covers the matched 73+49 input snapshot only",
                   "Cow IDs are parsed from filename n-suffix, not independently verified",
                   "YOLO training provenance has not been audited; this is not a pristine holdout",
                   "File-size equality is not used as proof of duplicate content"],
        "input_hashes": {"matched_windows": file_hash(MATCHED), "generator": file_hash(Path(__file__))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Inventory: {len(inventory)} sources; selected: {len(selected)}")
    for day in DATE_DIRS:
        print(f"{day}: {sum(row['date'] == day and row['disposition'] == 'candidate_new_cow' for row in inventory)} new-cow candidates; "
              f"{sum(row['date'] == day for row in selected)} selected")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
