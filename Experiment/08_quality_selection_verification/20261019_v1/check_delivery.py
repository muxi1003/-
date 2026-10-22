"""Bounded delivery checks; no prediction, parameter search or source modification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import run_verification as run

HERE = Path(__file__).resolve().parent


def main():
    run.verify_frozen()
    predictions = pd.read_csv(HERE / "predictions.csv")
    candidates = pd.read_csv(HERE / "candidate_quality.csv")
    registry = pd.read_csv(HERE / "inputs/all_candidate_reference_audit.csv")
    quality_records, selection_checks = [], 0
    for policy in run.POLICIES:
        for segment, group in candidates[candidates.policy.eq(policy)].groupby("segment_id"):
            indexed = group.set_index("mode").reindex(run.MODES)
            assert indexed.index.is_unique and len(indexed) == 5
            for term in (None, *run.TERMS) if policy == "LINEAR" else (None,):
                scores = indexed.apply(lambda row: run.score_value(row, term), axis=1)
                chosen = scores.idxmax()
                arm = f"SELECT_{policy}" + (f"_NO_{term.upper()}" if term else "")
                row = predictions[predictions.segment_id.eq(segment) & predictions.arm.eq(arm)].iloc[0]
                assert row.selected_mode == chosen
                if term:
                    baseline = predictions[predictions.segment_id.eq(segment) & predictions.arm.eq("SELECT_LINEAR")].iloc[0]
                    quality_records.append({"segment_id": segment, "cohort": row.cohort, "split": row.split,
                        "removed_term": term, "baseline_mode": baseline.selected_mode, "new_mode": chosen,
                        "baseline_full_score": float(indexed.loc[baseline.selected_mode, "score"]),
                        "new_mode_full_score": float(indexed.loc[chosen, "score"]),
                        "old_mode_reduced_score": float(scores.loc[baseline.selected_mode]),
                        "new_mode_reduced_score": float(scores.loc[chosen]),
                        "selection_changed": chosen != baseline.selected_mode,
                        "baseline_rr_error_bpm": baseline.absolute_rr_error_bpm,
                        "new_rr_error_bpm": row.absolute_rr_error_bpm,
                        "delta_rr_error_bpm": row.absolute_rr_error_bpm - baseline.absolute_rr_error_bpm})
                selection_checks += 1
    details = pd.DataFrame(quality_records)
    run.save_csv(HERE / "score_selection_diagnostics.csv", details)
    summary = []
    for scope, split in (("primary47", "all"), ("all82", "all"), ("all82", "train"), ("all82", "val")):
        group = details if scope == "all82" else details[details.cohort.eq("anchored49")]
        if split != "all":
            group = group[group.split.eq(split)]
        for term, pool in group.groupby("removed_term", sort=False):
            delta = pool.delta_rr_error_bpm
            unchanged = ~pool.selection_changed
            assert np.allclose(delta[unchanged], 0, atol=1e-12)
            summary.append({"scope": scope, "split": split, "removed_term": term, "windows": len(pool),
                "selection_changed": int(pool.selection_changed.sum()),
                "error_improved": int(delta.lt(-1e-10).sum()), "error_worsened": int(delta.gt(1e-10).sum()),
                "error_equal": int(delta.abs().le(1e-10).sum()), "delta_mae_bpm": float(delta.mean())})
    run.save_csv(HERE / "score_selection_summary.csv", summary)
    coverage = []
    for cohort, pool in registry.groupby("cohort", sort=False):
        n = int(pool.eligible_rr.sum())
        coverage.append({"cohort": cohort, "candidate_windows": len(pool), "complete_reference_windows": n,
                         "excluded_incomplete_reference": len(pool) - n, "reference_coverage": n / len(pool)})
    run.save_csv(HERE / "reference_coverage.csv", coverage)
    hashes = []
    historical = json.loads((run.SOURCE / "freeze.json").read_text(encoding="utf-8"))
    for entry in historical["files"]:
        path = Path(entry["path"])
        current = run.sha256(path) if path.is_file() else None
        hashes.append({"path": str(path), "historical_sha256": entry["sha256"], "current_sha256": current,
                       "unchanged": current == entry["sha256"]})
    run.save_csv(HERE / "upstream_provenance_check.csv", hashes)
    assert all(entry["unchanged"] for entry in hashes), "Upstream frozen source changed; investigate before completion"
    images = []
    for path in sorted((HERE / "figures").rglob("*.png")):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert rgb.shape[0] >= 700 and rgb.shape[1] >= 1000
        colored = (rgb.max(axis=2) - rgb.min(axis=2)) > 20
        assert float(colored.mean()) > .005 and float(rgb.std()) > 10
        images.append({"path": str(path.relative_to(HERE)), "width": rgb.shape[1], "height": rgb.shape[0],
                       "color_pixel_fraction": float(colored.mean()), "pixel_std": float(rgb.std())})
    run.write_json(HERE / "supplemental_validation.json", {"status": "passed", "selection_checks": selection_checks,
        "score_removal_rows": len(details), "unchanged_selection_has_unchanged_error": True,
        "upstream_frozen_files_verified": len(hashes), "reference_candidates": len(registry),
        "complete_references": int(registry.eligible_rr.sum()), "all_arms_output_coverage": "82/82; no rejection rule",
        "output_coverage_is_not_successful_observation": True, "figures": images,
        "visual_inspection": ["arm_mae_matrix.png", "continuous_attribution.png", "cases/A49_190588.png",
                              "cases/P3_20230810T071310n211109_000_030.png"],
        "workbook_visual_office_render": "not run; structural and numerical snapshot checks only"})
    print(json.dumps({"status": "passed", "selection_checks": selection_checks, "upstream_files": len(hashes),
                      "figure_pixel_checks": len(images)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
