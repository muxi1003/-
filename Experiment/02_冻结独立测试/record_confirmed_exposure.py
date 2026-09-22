"""Record the user's 2026-09-09 confirmation for the 271 already-curated candidates."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / "curation_manifest.json").read_text(encoding="utf-8"))
    review = read_csv(args.source / "independence_review.csv")
    candidates = read_csv(args.source / "candidate_windows.csv")
    if len(candidates) != 271 or sha256(args.source / "candidate_windows.csv") != manifest["candidate_manifest_sha256"]:
        raise ValueError("This confirmation applies only to the original 271-candidate manifest")
    args.out.mkdir(parents=True, exist_ok=False)
    review["rr_tuning_exposure"] = "no"
    review["rr_result_viewing_exposure"] = "no"
    review["exposure_evidence"] = "user_explicit_reply_20260909_all_no_for_these_271_candidates"
    review["notes"] = "Exposure confirmed by user; cow identity/modality/duplicate relationship still separate review"
    write_csv(args.out / "independence_review.csv", review)
    write_csv(args.out / "candidate_windows.csv", candidates)
    write_csv(args.out / "rgb_pair_candidates.csv", read_csv(args.source / "rgb_pair_candidates.csv"))
    hashes = []
    for i, row in enumerate(candidates.itertuples()):
        path = Path(row.source_path)
        hashes.append({"file_id": row.file_id, "source_path": str(path), "bytes": path.stat().st_size,
                       "sha256_full": sha256(path), "scope": "full_file_identity_not_visual_near_duplicate_detection"})
        if (i + 1) % 50 == 0:
            print(f"Full-file hashing: {i + 1}/{len(candidates)}", flush=True)
    write_csv(args.out / "candidate_full_hashes.csv", hashes)
    duplicates = pd.DataFrame(hashes)
    write_csv(args.out / "full_hash_duplicate_candidates.csv", duplicates[duplicates.sha256_full.duplicated(keep=False)])
    manifest.update(created_at=stamp(), status="EXPOSURE_CONFIRMED_IDENTITY_AND_RELEASE_PENDING",
                    prior_curation=str(args.source.resolve()), candidate_manifest_sha256=sha256(args.out / "candidate_windows.csv"),
                    exposure_statement="User confirmed no YOLO training, RR tuning, or RR result viewing for the remaining 271 candidates",
                    candidate_full_hashes_sha256=sha256(args.out / "candidate_full_hashes.csv"),
                    verified_test_windows=0)
    write_json(args.out / "curation_manifest.json", manifest)
    write_json(args.out / "USER_CONFIRMATION.json", {
        "date": "2026-09-09", "evidence_type": "user_statement_not_forensic_proof",
        "first_answer": "都没有用于YOLO训练", "followup_scope": "排除已知开发数据后剩余271个候选窗口的呼吸率调参和预测结果查看经历",
        "followup_answer": "都没有", "does_not_confirm": ["cow_id_filename_semantics", "event_reference", "RGB_synchronization", "environment_measurements"],
        "candidate_ids_sha256": sha256(args.out / "candidate_windows.csv")})
    print(f"Exposure updated, {len(hashes)} full hashes, {int(duplicates.sha256_full.duplicated().sum())} duplicate complete files")


if __name__ == "__main__":
    main()
