"""Curate an exposure audit and freeze eligible windows without using reference counts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def curate(package_path: Path, out: Path):
    package = json.loads(package_path.read_text(encoding="utf-8"))
    source = Path(package["inventory"])
    table = read_csv(source / "raw_video_inventory.csv")
    table["media_status"] = ["apple_sidecar_not_video" if Path(p).name.startswith("._") else
                             "metadata_readable_not_full_decode_audited" if is_true(o) else
                             "unreadable_requires_manual_check" for p, o in zip(table.path, table.opens)]
    development_fingerprints = set(table.loc[table.known_development_session.map(is_true), "sampled_fingerprint"])
    table["sampled_match_to_development"] = table.sampled_fingerprint.isin(development_fingerprints)
    write_csv(out / "media_quality_inventory.csv", table)
    # Only Jiufu is proposed for a cross-farm test. Lindian stays development.
    candidates = table[(table.farm == "jiufu") & (table.modality_hint == "thermal_name_hint") &
                       table.opens.map(is_true) & ~table.path.map(lambda p: Path(p).name.startswith("._")) &
                       ~table.known_development_session.map(is_true) & ~table.known_development_cow.map(is_true) &
                       ~table.sampled_match_to_development &
                       (pd.to_numeric(table.duration_seconds_metadata, errors="coerce") >= 30)].copy()
    candidates = candidates.sort_values("path").drop_duplicates("sampled_fingerprint")
    rows, windows = [], []
    for row in candidates.itertuples():
        rows.append({"file_id": row.file_id, "source_path": row.path,
                     "date_from_name": row.date_from_name, "cow_token_unverified": row.cow_token,
                     "session_token": row.session_token, "include_in_test": "",
                     "yolo_training_exposure": "no", "yolo_exposure_evidence": "user_confirmed_20260909",
                     "rr_tuning_exposure": "", "rr_result_viewing_exposure": "",
                     "cow_identity_verified": "", "verified_cow_id": "", "modality_verified": "",
                     "duplicate_identity_reviewed": "", "reviewer": "", "review_date": "", "notes": ""})
        windows.append({"window_id": row.file_id + "_first30", "file_id": row.file_id,
                        "farm": row.farm, "source_path": row.path, "source_session": row.session_token,
                        "start_seconds": 0., "duration_seconds": 30.,
                        "sampling_rule": "first_30_seconds_predeclared_not_best_segment",
                        "sampled_fingerprint": row.sampled_fingerprint})
    write_csv(out / "independence_review.csv", rows)
    write_csv(out / "candidate_windows.csv", windows)
    pairs = read_csv(source / "rgb_thermal_pair_candidates.csv")
    pair_ids = set(candidates.file_id)
    pairs = pairs[pairs.thermal_file_id.isin(pair_ids) &
                  ~pairs.rgb_path.map(lambda p: Path(p).name.startswith("._"))]
    write_csv(out / "rgb_pair_candidates.csv", pairs)
    write_json(out / "curation_manifest.json", {
        "source_inventory": str(source), "source_inventory_sha256": sha256(source / "raw_video_inventory.csv"),
        "candidate_manifest_sha256": sha256(out / "candidate_windows.csv"),
        "method_snapshot": package["method_snapshot"],
        "method_manifest_sha256": sha256(Path(package["method_snapshot"]) / "freeze_manifest.json"),
        "created_at": stamp(), "status": "AWAIT_HUMAN_EXPOSURE_REVIEW_NOT_A_TEST_SET",
        "counts_by_farm_media_status": table.groupby(["farm", "media_status"]).size().rename("n").reset_index().to_dict("records"),
        "candidate_windows": len(windows), "verified_test_windows": 0,
        "deduplication": "sampled fingerprint screening only; selected sources get full SHA256 at release",
        "scope": "Jiufu only; excludes known development sessions/cows and sampled matches; other files retained in audit"})
    print(f"Curated {len(windows)} pending Jiufu windows; 0 released test windows", flush=True)


def review_blockers(row):
    checks = {"yolo_training_exposure": "no", "rr_tuning_exposure": "no",
              "rr_result_viewing_exposure": "no", "cow_identity_verified": "yes",
              "modality_verified": "thermal", "duplicate_identity_reviewed": "yes"}
    issues = [f"{key}_must_be_{value}" for key, value in checks.items()
              if str(row.get(key, "")).strip().lower() != value]
    for key in ("verified_cow_id", "reviewer", "review_date"):
        if not str(row.get(key, "")).strip():
            issues.append(f"missing_{key}")
    return issues


def verify_method(method: Path, expected_hash: str):
    path = method / "freeze_manifest.json"
    if sha256(path) != expected_hash:
        raise ValueError("Frozen method manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(method / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError(f"Frozen method file changed: {item['snapshot_relative_path']}")


def audit(curated: Path, out: Path, release: bool):
    manifest = json.loads((curated / "curation_manifest.json").read_text(encoding="utf-8"))
    candidate_path = curated / "candidate_windows.csv"
    if sha256(candidate_path) != manifest["candidate_manifest_sha256"]:
        raise ValueError("Candidate windows changed after curation")
    source = Path(manifest["source_inventory"]) / "raw_video_inventory.csv"
    if sha256(source) != manifest["source_inventory_sha256"]:
        raise ValueError("Original exposure inventory changed")
    verify_method(Path(manifest["method_snapshot"]), manifest["method_manifest_sha256"])
    review = read_csv(curated / "independence_review.csv")
    windows = read_csv(candidate_path).set_index("file_id")
    inventory = read_csv(source)
    known_cows = set(inventory.loc[(inventory.farm == "jiufu") &
                                  (inventory.known_development_session.map(is_true) |
                                   inventory.known_development_cow.map(is_true)), "cow_token"]) - {""}
    if review.file_id.duplicated().any() or set(review.file_id) != set(windows.index):
        raise ValueError("Review IDs must match the frozen candidate list exactly")
    blockers, selected, hashes = [], [], set()
    for row in review.to_dict("records"):
        if not is_true(row["include_in_test"]):
            continue
        issues = review_blockers(row)
        original = windows.loc[row["file_id"]]
        if row["source_path"] != original.source_path:
            issues.append("source_path_changed")
        if row["verified_cow_id"] in known_cows:
            issues.append("verified_cow_overlaps_known_development")
        if issues:
            blockers.append({"file_id": row["file_id"], "blockers": ";".join(issues)})
        else:
            selected.append({**original.to_dict(), "file_id": row["file_id"],
                             "verified_cow_id": row["verified_cow_id"]})
    if not selected:
        blockers.append({"file_id": "ALL", "blockers": "no_reviewed_eligible_windows"})
    write_csv(out / "release_blockers.csv", blockers, columns=["file_id", "blockers"])
    status = "BLOCKED_PENDING_REVIEW" if blockers else "ELIGIBLE_PENDING_RELEASE"
    if release and not blockers:
        for row in selected:
            digest = sha256(Path(row["source_path"]))
            if digest in hashes:
                raise ValueError("Duplicate complete files in proposed test")
            hashes.add(digest)
            row["source_sha256"] = digest
        write_csv(out / "test_windows.csv", selected)
        write_json(out / "test_release.json", {
            "status": "DATA_AND_SIGNAL_POLICY_FROZEN_END_TO_END_PENDING", "released_at": stamp(),
            "method_snapshot": manifest["method_snapshot"], "method_manifest_sha256": manifest["method_manifest_sha256"],
            "test_windows_sha256": sha256(out / "test_windows.csv"),
            "exposure_review_sha256": sha256(curated / "independence_review.csv"),
            "reference_used_for_selection": False, "n_windows": len(selected),
            "inference_adapter_status": "requires_validated_timestamp_and_modality_adapter_before_execution"})
        status = "DATA_AND_SIGNAL_POLICY_FROZEN_END_TO_END_PENDING"
    write_json(out / "audit_status.json", {"status": status, "eligible_reviewed_windows": len(selected),
                                          "blocked_rows": len(blockers), "external_rr_metrics": None})
    print(f"{status}: {len(selected)} eligible windows, {len(blockers)} blockers")
    if release and blockers:
        raise SystemExit("Test release refused; complete the exposure/identity review first")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cur = sub.add_parser("curate")
    cur.add_argument("--package", type=Path, required=True)
    cur.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--curated", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--release", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.command == "curate":
        curate(args.package, args.out)
    else:
        audit(args.curated, args.out, args.release)


if __name__ == "__main__":
    main()
