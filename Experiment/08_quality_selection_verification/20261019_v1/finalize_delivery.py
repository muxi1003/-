"""Verify the bounded delivery and seal it without altering scientific outputs."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import run_verification as run

HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE / "COMPLETED.json").exists(), "Delivery already sealed"
    frozen = run.verify_frozen()
    required = ["README.md", "REPORT.md", "PROTOCOL.md", "WRITING_DRAFT.md", "CLAIMS_FROM_RESULTS.md",
                "DECISION.json", "metric_comparison.xlsx", "literature_comparison.csv", "LITERATURE_COMPARISON.md",
                "predictions.csv", "candidate_quality.csv", "predicted_events.csv", "event_matches.csv",
                "metrics.csv", "paired_intervals.csv", "replay_acceptance.json", "numeric_validation.json",
                "supplemental_validation.json", "score_selection_diagnostics.csv", "score_selection_summary.csv",
                "reference_coverage.csv", "upstream_provenance_check.csv", "CASES.md", "freeze.json"]
    for name in required:
        path = HERE / name
        assert path.is_file() and path.stat().st_size > 0, name
    acceptance = json.loads((HERE / "replay_acceptance.json").read_text(encoding="utf-8"))
    numeric = json.loads((HERE / "numeric_validation.json").read_text(encoding="utf-8"))
    supplemental = json.loads((HERE / "supplemental_validation.json").read_text(encoding="utf-8"))
    decision = json.loads((HERE / "DECISION.json").read_text(encoding="utf-8"))
    assert numeric["status"] == supplemental["status"] == "passed"
    assert decision["approved_plan_route"] == "uncertain" and not decision["default_changed"]
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
                            "-p", "test_verification.py", "-v"], capture_output=True, text=True, encoding="utf-8")
    assert tests.returncode == 0, tests.stdout + tests.stderr
    (HERE / "unit_test_output.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    pixel_checks = []
    for path in sorted((HERE / "figures").rglob("*.png")):
        array = np.asarray(Image.open(path).convert("RGB"))
        assert array.shape[0] >= 700 and array.shape[1] >= 1000 and array.std() > 10
        colored = (array.max(axis=2) - array.min(axis=2)) > 20
        assert colored.mean() > .005
        pixel_checks.append({"path": str(path.relative_to(HERE)), "size": [array.shape[1], array.shape[0]],
                             "std": float(array.std()), "color_fraction": float(colored.mean())})
    assert len(pixel_checks) == 9
    cases = (HERE / "CASES.md").read_text(encoding="utf-8")
    assert "全部82窗" in cases and "不是主要47窗" in cases
    log = run.REPO / "docs/cow_rr_project_log.md"
    assert "### N100 " in log.read_text(encoding="utf-8")
    for skill in ("novelty-check", "result-to-claim"):
        trace = run.REPO / f".aris/traces/{skill}/20261019_quality_selection"
        for name in ("prompt.txt", "raw_response.txt", "tool_result.json", "verdict.json"):
            assert (trace / name).is_file() and (trace / name).stat().st_size > 0
    run.verify_frozen()
    run.write_json(HERE / "FINAL_VALIDATION.json", {"status": "passed", "client_date": "2026-10-19",
        "utc_completed": datetime.now(timezone.utc).isoformat(), "required_artifacts": required,
        "original_sources_and_core_hashes_unchanged": True, "source_hash_count": len(frozen["sources"]),
        "upstream_hash_count": supplemental["upstream_frozen_files_verified"], "unit_tests_passed": 8,
        "numeric_validation": numeric, "historical_replay": acceptance, "final_figure_checks": pixel_checks,
        "cases_scope_label_fixed": True, "project_node": "N100", "review_status": "same-family/provisional/not_formally_audited",
        "new_annotation": False, "new_yolo_training_or_inference": False, "new_algorithm_modules": False,
        "default_model_or_rr_changed": False, "workbook_office_render": "not run"})
    excluded = {"delivery_manifest.json", "COMPLETED.json"}
    manifest = [{"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": run.sha256(path)}
                for path in sorted(HERE.rglob("*")) if path.is_file() and path.name not in excluded
                and "__pycache__" not in path.parts and "runtime" not in path.parts]
    run.write_json(HERE / "delivery_manifest.json", {"files": manifest, "excludes": sorted(excluded),
        "project_log": {"path": str(log), "sha256": run.sha256(log), "node": "N100"},
        "policy": "Frozen scientific inputs and original outputs preserved; no automatic next iteration"})
    run.write_json(HERE / "COMPLETED.json", {"client_date": "2026-10-19", "status": "complete",
        "completion_scope": "approved bounded verification, not independent confirmation",
        "windows": 82, "arms": 16, "predictions": 1312, "decision": "uncertain",
        "validation": "FINAL_VALIDATION.json", "manifest_sha256": run.sha256(HERE / "delivery_manifest.json"),
        "default_changed": False, "automatic_next_iteration": False})
    print(json.dumps({"status": "complete", "decision": "uncertain", "manifest_files": len(manifest),
                      "unit_tests": 8, "project_node": "N100"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
