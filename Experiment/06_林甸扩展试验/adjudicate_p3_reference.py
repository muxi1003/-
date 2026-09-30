"""Create a new P3 export version for the user-adjudicated partial window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

TARGET = "20230808T164507n211023_000_030_source30s"
FILES = (
    "lindian_expansion_annotation_windows.csv",
    "lindian_expansion_reference_events.csv",
    "lindian_expansion_unobservable_intervals.csv",
    "lindian_expansion_event_backup.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing reference version: {output}")
    for batch in ("A", "B"):
        for name in FILES:
            if not (source / f"raw_{batch}" / name).is_file():
                raise FileNotFoundError(source / f"raw_{batch}" / name)

    a = source / "raw_A"
    with (a / FILES[0]).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields, windows = reader.fieldnames, list(reader)
    backup = json.loads((a / FILES[3]).read_text(encoding="utf-8"))
    csv_target = [row for row in windows if row["window_id"] == TARGET]
    json_target = [row for row in backup["windows"] if row["window_id"] == TARGET]
    confirmed = sum(event["window_id"] == TARGET and event["confidence"] == "confirmed"
                    for event in backup["events"])
    spans = sum(interval["window_id"] == TARGET for interval in backup["intervals"])
    if (len(csv_target) != 1 or len(json_target) != 1 or confirmed != 10 or spans != 2
            or csv_target[0]["annotation_status"] != "pending"
            or json_target[0]["annotation_status"] != "pending"
            or csv_target[0]["manual_breath_count"] != ""
            or json_target[0]["manual_breath_count"] != ""):
        raise ValueError("Adjudication preconditions changed; do not silently rewrite")
    for row in (csv_target[0], json_target[0]):
        row["annotation_status"] = "partial"
        row["manual_breath_count"] = str(confirmed)

    output.mkdir(parents=True)
    target_a, target_b = output / "raw_A", output / "raw_B"
    target_a.mkdir()
    target_b.mkdir()
    with (target_a / FILES[0]).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(windows)
    (target_a / FILES[3]).write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in FILES[1:3]:
        shutil.copy2(a / name, target_a / name)
    for name in FILES:
        shutil.copy2(source / "raw_B" / name, target_b / name)
    manifest = {
        "source": str(source),
        "user_decision": "2026-09-29: set 20230808T164507n211023_000_030 to partial",
        "change": {"window_id": TARGET, "annotation_status": ["pending", "partial"],
                   "manual_breath_count": ["", "10"],
                   "reason": "10 confirmed visible events and two unobservable intervals; not full 30 s truth"},
        "other_annotation_rows_changed": False,
        "source_files_sha256": {f"{batch}/{name}": digest(source / f"raw_{batch}" / name)
                                for batch in ("A", "B") for name in FILES},
        "output_files_sha256": {f"{batch}/{name}": digest(output / f"raw_{batch}" / name)
                                for batch in ("A", "B") for name in FILES},
        "note": "Structured re-export generated from frozen snapshot; not a new human peak annotation.",
    }
    (output / "adjudication.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created adjudicated reference copy: {output}")


if __name__ == "__main__":
    main()
