"""Audit model/source provenance and prepare a count-blind P2 reference recheck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_lindian_expansion_annotation_batch import make_page

P1 = HERE / "20260924_expansion_v2/selected_sources.csv"
P2 = HERE / "20260928_expansion_p2_v1/selected_sources.csv"
P2_INVENTORY = HERE / "20260928_expansion_p2_v1/source_inventory.csv"
P2_PENDING = HERE / "20260928_expansion_p2_v1/annotation_batch_v1/annotation_windows_pending.csv"
P2_SEGMENTS = HERE / "20260928_expansion_p2_v1/annotation_batch_v1/segments.csv"
P2_DISPOSITION = HERE / "20260928_lx_p2_v1/reference_disposition.csv"
MODEL = ROOT / "models/best.pt"
RUN = ROOT / "runs/pose/runs/pose/cow_nose_pose"
OTHER_MODEL = ROOT / "runs/pose/YOLO11n_cow_pose/weights/YOLO11n-best.pt"
TRAIN_IMAGES = ROOT / "Dataset_new/72video/images/train"
VAL_IMAGES = ROOT / "Dataset_new/72video/images/val"
ROUND = "LX-20260928-R1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_groups(directory: Path) -> tuple[int, set[str]]:
    paths = list(directory.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No training images: {directory}")
    groups = set()
    for path in paths:
        stem, separator, frame = path.stem.rpartition("_frame_")
        if not separator or not frame.isdigit():
            raise ValueError(f"Unexpected training image name: {path.name}")
        groups.add(stem.casefold())
    return len(paths), groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "20260928_confirmation_gate_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Preserving existing audit: {out}")

    p1, p2 = read_csv(P1), read_csv(P2)
    selected = [("P1", row) for row in p1] + [("P2", row) for row in p2]
    if len(selected) != 24 or len({row["cow_id"].casefold() for _, row in selected}) != 24:
        raise ValueError("P1/P2 source membership or filename cow grouping changed")
    for _, row in selected:
        if digest(Path(row["source_path"])) != row["source_sha256"]:
            raise ValueError(f"Source content changed: {row['source_path']}")

    train_count, train_groups = image_groups(TRAIN_IMAGES)
    val_count, val_groups = image_groups(VAL_IMAGES)
    selected_ids = {row["cow_id"].casefold() for _, row in selected}
    direct_matches = selected_ids & (train_groups | val_groups)
    with (RUN / "args.yaml").open(encoding="utf-8") as stream:
        train_args = yaml.safe_load(stream)
    if not isinstance(train_args, dict):
        raise ValueError("Training args are not a mapping")
    model_hash = digest(MODEL)
    run_hash = digest(RUN / "weights/best.pt")
    if model_hash != run_hash:
        raise ValueError("Scored checkpoint no longer matches the candidate training run")

    identity = []
    for batch, row in selected:
        source = Path(row["source_path"])
        identity.append({
            "batch": batch,
            "source_path": str(source),
            "source_sha256": row["source_sha256"],
            "capture_day_from_path": row["date"],
            "path_context": str(source.parent),
            "filename_cow_id": row["cow_id"],
            "cow_id_rule": "user_confirms_n_suffix_is_cow_number",
            "farm_record_cow_id": "",
            "farm_record_or_ear_tag_evidence": "",
            "verified_session_id": "",
            "reviewer": "",
            "verification_status": "filename_rule_only",
        })

    pending = {row["window_id"]: row for row in read_csv(P2_PENDING)}
    segments = {row["window_id"]: row for row in read_csv(P2_SEGMENTS)}
    complete = [row for row in read_csv(P2_DISPOSITION) if row["analysis_status"] == "complete"]
    if len(complete) != 6 or len({row["window_id"] for row in complete}) != 6:
        raise ValueError("Expected six P2 complete-reference windows")
    complete.sort(key=lambda row: hashlib.sha256(row["window_id"].encode("utf-8")).hexdigest())
    recheck_windows, recheck_queue = [], []
    for row in complete:
        window_id = row["window_id"]
        original = pending[window_id]
        segment = segments[window_id]
        clip = Path(original["browser_video_path"])
        if digest(clip) != segment["clip_sha256"]:
            raise ValueError(f"Viewing clip content changed: {clip}")
        new = dict(original)
        new.update(annotation_round=ROUND, annotator="", annotation_status="pending",
                   manual_breath_count="", predictions_hidden="", reference_notes="",
                   prior_algorithm_exposure="P2 reference previously annotated; hide prior export and all predictions",
                   full_window_basis="")
        recheck_windows.append(new)
        recheck_queue.append({
            "recheck_order": str(len(recheck_queue) + 1),
            "window_id": window_id,
            "video_id": row["video_id"],
            "clip_path": str(clip),
            "clip_sha256": segment["clip_sha256"],
            "source_sha256": segment["source_sha256"],
            "annotation_round": ROUND,
        })

    pool = []
    for row in read_csv(P2_INVENTORY):
        cow_id = row["cow_id"].casefold()
        if row["disposition"] != "candidate_new_cow" or cow_id in selected_ids:
            continue
        pool.append({
            "capture_day_from_path": row["date"],
            "filename_cow_id": row["cow_id"],
            "source_path": row["source_path"],
            "metadata_duration_seconds": row["metadata_duration_seconds"],
            "current_train_group_match": str(cow_id in train_groups | val_groups).lower(),
            "historical_train_membership": "unknown",
            "farm_record_identity": "unknown",
            "status": "candidate_only_not_selected_or_scored",
        })
    pool.sort(key=lambda row: (row["capture_day_from_path"],
                               hashlib.sha256(row["source_path"].encode("utf-8")).hexdigest()))

    audit = {
        "scored_model": {"path": str(MODEL), "sha256": model_hash},
        "matching_training_run_checkpoint": {"path": str(RUN / "weights/best.pt"), "sha256": run_hash},
        "different_legacy_checkpoint": {"path": str(OTHER_MODEL), "sha256": digest(OTHER_MODEL)},
        "matching_run_args_path": str(RUN / "args.yaml"),
        "matching_run_args_sha256": digest(RUN / "args.yaml"),
        "matching_run_data_argument": str(train_args.get("data")),
        "matching_run_data_argument_exists_now": Path(str(train_args.get("data"))).is_file(),
        "current_local_training_images": {"train_count": train_count, "val_count": val_count,
                                          "train_group_count": len(train_groups),
                                          "val_group_count": len(val_groups),
                                          "train_val_group_overlap_count": len(train_groups & val_groups)},
        "p1_p2_filename_cow_group_direct_matches_current_layout": sorted(direct_matches),
        "historical_per_image_training_membership": "not_proven_by_checkpoint_or_current_directory",
        "farm_record_identity": "not_supplied; user confirms n suffix naming rule only",
        "p2_complete_windows_for_count_blind_repeat": len(recheck_queue),
        "remaining_p3_candidate_sources_not_yet_selected": len(pool),
        "input_sha256": {str(p): digest(p) for p in
                         (P1, P2, P2_INVENTORY, P2_PENDING, P2_SEGMENTS, P2_DISPOSITION)},
    }

    out.mkdir(parents=True)
    write_csv(out / "identity_review.csv", identity)
    write_csv(out / "p2_reference_recheck_queue.csv", recheck_queue)
    write_csv(out / "p3_candidate_pool_unselected.csv", pool)
    write_csv(out / "p2_reference_recheck_pending.csv", recheck_windows)
    (out / "p2_reference_recheck.html").write_text(
        make_page(recheck_windows, ROUND, strict_full_window=True), encoding="utf-8")
    (out / "provenance_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Checkpoint match: {model_hash == run_hash}; historical train list: unknown")
    print(f"P1/P2 filename cow overlap with current train/val groups: {len(direct_matches)}")
    print(f"P2 count-blind recheck: {len(recheck_queue)}; unselected P3 source pool: {len(pool)}")
    print(out)


if __name__ == "__main__":
    main()
