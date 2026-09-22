"""Deterministic verification of the 20260911 submission and saved-prediction rescore."""
from __future__ import annotations

import json
from pathlib import Path

from experiment_common import *


def main():
    submission = REFERENCE / "submissions/20260911_v2"
    rescore = ABLATION / "reference_updates/20260911_v2"
    checks, sources = [], []
    index = read_csv(ROOT / "FILE_MANIFEST.csv").set_index("relative_path")
    frozen_paths = [ABLATION / "input_snapshots/20260909_v1/matched_windows.csv",
                    ABLATION / "input_snapshots/20260909_v1/historical_sources/lindian_anchored30_quality_gated_annotation_template.csv",
                    ABLATION / "runs/20260909_v1/paired_predictions.csv"]
    for p in frozen_paths:
        key = str(p.relative_to(ROOT))
        if sha256(p) != index.loc[key, "sha256"]:
            raise ValueError(f"Original package input changed: {p}")
        sources.append({"path": str(p), "sha256": sha256(p), "checked_against": str(ROOT / "FILE_MANIFEST.csv")})
    checks.append("historical_predictions_and_reference_snapshot_hashes_match_original_delivery_index")
    for cohort in ("lindian49", "jiufu271"):
        for entry in json.loads((submission / cohort / "manifest.json").read_text(encoding="utf-8")):
            if sha256(Path(entry["source"])) != entry["sha256"] or sha256(Path(entry["snapshot"])) != entry["sha256"]:
                raise ValueError("Original human submission or snapshot changed")
        checks.append(f"{cohort}_all_three_original_tables_unchanged_and_byte_exact_snapshot")
    old_path = frozen_paths[1]
    old = pd.read_csv(old_path, encoding="gb18030", dtype=str, keep_default_na=False).set_index("video_id")
    new = read_csv(submission / "lindian49/count_reference_only.csv").set_index("video_id")
    for video, row in new.iterrows():
        expected = old.loc[video]
        if row.window_id != expected.window_id or abs(float(row.source_start_seconds)-float(expected.window_start_seconds)) > 1e-8:
            raise ValueError("Lindian start or window identity changed")
        if row.source_path.replace("\\", "/").casefold() != expected.raw_source_path.replace("\\", "/").casefold():
            raise ValueError("Lindian raw source path changed")
        if abs(float(row.duration_seconds) - float(expected.window_seconds)) > 1e-8:
            raise ValueError("Lindian duration changed")
    checks.append("all49_raw_source_paths_starts_durations_and_window_ids_match_frozen_historical_annotation")
    matched = read_csv(frozen_paths[0]).query("cohort == 'anchored49'").set_index("video_id")
    original = read_csv(frozen_paths[2]).query("cohort == 'anchored49'").set_index(["video_id", "variant"])
    scored = read_csv(rescore / "predictions_rescored.csv")
    if len(scored) != 441 or scored.duplicated(["video_id", "variant"]).any():
        raise ValueError("Rescored prediction population mismatch")
    for row in scored.itertuples():
        expected_count = matched.loc[row.video_id, "historical_predicted_count"] if row.variant == "historical_motion_robust" else original.loc[(row.video_id, row.variant), "predicted_count"]
        if float(row.predicted_count) != float(expected_count) or float(row.truth_count) != float(new.loc[row.video_id, "manual_breath_count"]):
            raise ValueError("Prediction or submitted reference mismatch")
        if abs(float(row.duration_seconds)-float(matched.loc[row.video_id, "duration_seconds"])) > 1e-8:
            raise ValueError("Prediction duration mismatch")
    checks.append("all441_rescored_predictions_references_and_durations_match_inputs")
    for row in read_csv(rescore / "metrics.csv").itertuples():
        data = scored[scored.variant.eq(row.variant)]
        if row.analysis_set == "completed39_count_reference":
            data = data[data.annotation_status.eq("completed")]
        duration = data.duration_seconds.to_numpy(float)
        truth, pred = data.truth_count.to_numpy(float), data.predicted_count.to_numpy(float)
        y, p = truth * 60 / duration, pred * 60 / duration
        expected = {"rr_r2": 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2), "rr_mae_bpm": np.mean(abs(p-y)),
                    "rr_rmse_bpm": np.sqrt(np.mean((p-y)**2)),
                    "mean_count_accuracy_percent": 100*np.mean(np.maximum(0, 1-abs(pred[truth>0]-truth[truth>0])/truth[truth>0]))}
        if len(data) != int(row.n) or any(abs(float(getattr(row, key))-value) > 1e-9 for key, value in expected.items()):
            raise ValueError("Independent metric recomputation mismatch")
    checks.append("all18_metric_groups_match_independent_formula_recomputation")
    freeze = HOLDOUT / "method_snapshots/20260909_v1"
    manifest = json.loads((freeze / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(freeze / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method altered")
    checks.append("all_frozen_method_snapshot_files_unchanged")
    write_json(submission / "artifact_verification.json", {"status": "PASS_DETERMINISTIC_CHECKS", "checks": checks,
               "historical_input_provenance": sources, "verifier_sha256": sha256(Path(__file__)),
               "independent_scientific_review": "separate_same_family_review_not_implied_by_deterministic_pass"})
    artifacts = []
    for directory in [submission, rescore, HOLDOUT / "preflight/20260911_v1"]:
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                artifacts.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    for p in [ROOT / "process_annotation_submission.py", ROOT / "test_annotation_submission.py", Path(__file__),
              HOLDOUT / "preflight_frozen_videos.py", HOLDOUT / "summarize_timestamp_gaps.py"]:
        artifacts.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    write_csv(submission / "delivery_artifact_manifest.csv", artifacts)
    print(json.dumps({"status": "PASS", "checks": checks, "manifest_files": len(artifacts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
