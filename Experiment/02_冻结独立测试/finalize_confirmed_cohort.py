"""Freeze the user-confirmed Jiufu identity/exposure cohort without using outcomes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / "curation_manifest.json").read_text(encoding="utf-8"))
    review = read_csv(args.source / "independence_review.csv")
    windows = read_csv(args.source / "candidate_windows.csv")
    full = read_csv(args.source / "candidate_full_hashes.csv")
    inventory = read_csv(Path(manifest["source_inventory"]) / "raw_video_inventory.csv").set_index("file_id")
    if len(windows) != 271 or sha256(args.source / "candidate_windows.csv") != manifest["candidate_manifest_sha256"]:
        raise ValueError("Wrong confirmation scope")
    if sha256(args.source / "candidate_full_hashes.csv") != manifest["candidate_full_hashes_sha256"]:
        raise ValueError("Full hashes changed")
    if full.sha256_full.duplicated().any() or windows.source_session.duplicated().any():
        raise ValueError("Repeated complete file or session in proposed cohort")
    for row in review.itertuples():
        if row.yolo_training_exposure != "no" or row.rr_tuning_exposure != "no" or row.rr_result_viewing_exposure != "no":
            raise ValueError("Exposure confirmation incomplete")
        media = inventory.loc[row.file_id]
        if media.modality_hint != "thermal_name_hint" or (media.width, media.height) != ("1080", "1440"):
            raise ValueError("Media inconsistent with the selected thermal filename/geometry stratum")
    args.out.mkdir(parents=True, exist_ok=False)
    review["include_in_test"] = "yes"
    review["cow_identity_verified"] = "yes"
    review["verified_cow_id"] = review.cow_token_unverified
    review["modality_verified"] = "thermal"
    review["duplicate_identity_reviewed"] = "yes"
    review["reviewer"] = "user_naming_and_exposure_confirmation_plus_automated_manifest_audit"
    review["review_date"] = "2026-09-09"
    review["modality_verification_basis"] = "provided_thermal_dataset_filename_pattern_and_1080x1440_metadata_not_visual_review"
    review["notes"] = ("Cow labels confirmed by user; thermal filename/geometry stratum; full-file hashes unique; "
                       "known development cow/session exclusion; no framewise identity or semantic-near-duplicate guarantee")
    write_csv(args.out / "independence_review.csv", review)
    for name in ("candidate_windows.csv", "candidate_full_hashes.csv", "full_hash_duplicate_candidates.csv", "rgb_pair_candidates.csv"):
        write_csv(args.out / name, read_csv(args.source / name))
    manifest.update(created_at=stamp(), prior_curation=str(args.source.resolve()),
                    status="USER_EXPOSURE_AND_FILENAME_IDENTITY_CONFIRMED_PENDING_RELEASE_AUDIT",
                    candidate_manifest_sha256=sha256(args.out / "candidate_windows.csv"),
                    candidate_full_hashes_sha256=sha256(args.out / "candidate_full_hashes.csv"),
                    unique_cow_labels=int(review.verified_cow_id.nunique()),
                    identity_basis="user-confirmed filename cow labels, not framewise biometric verification",
                    modality_basis="dataset label, thermal filename pattern and consistent 1080x1440 metadata; visual/calibration audit pending")
    write_json(args.out / "curation_manifest.json", manifest)
    write_json(args.out / "USER_CONFIRMATION.json", {
        "date": "2026-09-09", "exposure": json.loads((args.source / "USER_CONFIRMATION.json").read_text(encoding="utf-8")),
        "filename_example": "20240729T084330-1479.MP4", "user_answer": "1479是牛号，084330是拍摄时间",
        "rule": "YYYYMMDD T HHMMSS - cow_id", "grouping": "farm plus cow_id across recording dates",
        "sampling": "all 271 eligible candidate recordings, first 30 seconds; no outcome-based selection"})
    write_csv(args.out / "preinference_media_review.csv", [{
        "file_id": row.file_id, "source_path": row.source_path,
        "visual_modality_checked": "", "first30_full_decode_checked": "", "pts_monotonic_verified": "",
        "thermal_scale_mode": "", "palette": "", "temperature_min_c": "", "temperature_max_c": "",
        "folder_hint_not_measurement": "18-36" if "18-36" in row.source_path else "",
        "bgr_rf_compatibility": "", "reviewer": "", "notes": ""
    } for row in windows.itertuples()])
    print(f"Confirmed {len(windows)} windows, {review.verified_cow_id.nunique()} user-defined cow labels")


if __name__ == "__main__":
    main()
