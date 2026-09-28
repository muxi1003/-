"""Freeze all remaining filename-cow groups for the P3 same-farm annotation queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = Path("E:/real/use_code/林甸红外视频")
POOL = HERE / "20260928_confirmation_gate_v1/p3_candidate_pool_unselected.csv"
P1 = HERE / "20260924_expansion_v2/selected_sources.csv"
P2 = HERE / "20260928_expansion_p2_v1/selected_sources.csv"
PROTOCOL = HERE / "20260928_p3_pre_score_protocol_v2.md"
MODEL = ROOT / "models/best.pt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, items: list[dict[str, str]]) -> None:
    if not items:
        raise ValueError("Empty P3 selection")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "20260928_expansion_p3_v2")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving prior selection: {output}")

    pool = rows(POOL)
    old = rows(P1) + rows(P2)
    old_ids = {row["cow_id"].casefold() for row in old}
    old_hashes = {row["source_sha256"] for row in old}
    if len(pool) != 82 or len({row["filename_cow_id"].casefold() for row in pool}) != 59:
        raise ValueError("P3 candidate pool changed")
    eligible = [row for row in pool if re.fullmatch(r"[0-9]+", row["filename_cow_id"])]
    if len(eligible) != 76 or len({row["filename_cow_id"] for row in eligible}) != 53:
        raise ValueError("P3 numeric-ID eligibility changed")
    grouped = {}
    for row in eligible:
        source = Path(row["source_path"]).resolve(strict=True)
        cow = row["filename_cow_id"].casefold()
        if (cow in old_ids or row["current_train_group_match"] != "false"
                or row["status"] != "candidate_only_not_selected_or_scored"
                or not source.is_relative_to(RAW)):
            raise ValueError(f"Source not eligible by frozen pool: {source}")
        relative = source.relative_to(RAW).as_posix()
        key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        if cow not in grouped or key < grouped[cow][0]:
            grouped[cow] = (key, row, source)

    selected = []
    seen_hashes = set()
    for cow in sorted(grouped, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()):
        _, row, source = grouped[cow]
        source_hash = digest(source)
        if source_hash in old_hashes or source_hash in seen_hashes:
            raise ValueError(f"Duplicate source content; do not silently replace: {source}")
        seen_hashes.add(source_hash)
        selected.append({
            "batch_order": str(len(selected) + 1),
            "annotation_batch": "A" if len(selected) < 27 else "B",
            "date": row["capture_day_from_path"],
            "cow_id": row["filename_cow_id"],
            "source_path": str(source),
            "source_sha256": source_hash,
            "metadata_duration_seconds": row["metadata_duration_seconds"],
            "selection_reason": "minimum_relative_path_sha256_per_filename_cow",
        })
    if len(selected) != 53 or sum(row["annotation_batch"] == "A" for row in selected) != 27:
        raise ValueError("Unexpected P3 group count")

    output.mkdir(parents=True)
    write_csv(output / "selected_sources.csv", selected)
    write_csv(output / "selected_sources_A.csv", [row for row in selected if row["annotation_batch"] == "A"])
    write_csv(output / "selected_sources_B.csv", [row for row in selected if row["annotation_batch"] == "B"])
    (output / "freeze.json").write_text(json.dumps({
        "scope": "P3 same-farm existing-archive annotation queue; no RR predictions or truth consulted",
        "selected_sources": 53,
        "unique_filename_cows": 53,
        "annotation_batches": {"A": 27, "B": 26},
        "identity_filter": "filename n-suffix must be digits only; six ambiguous IDs excluded",
        "selection": "per cow minimum SHA256 of UTF-8 relative raw path; A/B by SHA256 of casefold cow ID",
        "window": "first PTS-contiguous 30 s, no visibility or error based replacement",
        "inputs_sha256": {str(path): digest(path) for path in (POOL, P1, P2, PROTOCOL, MODEL,
                                                              Path(__file__))},
        "selected_sources_sha256": digest(output / "selected_sources.csv"),
        "training_limit": "current directory IDs nonoverlap, historical per-image frozen membership unavailable",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"P3 v2 locked: {len(selected)} numeric filename cows; A=27, B=26; no RR scored")
    print(output)


if __name__ == "__main__":
    main()
