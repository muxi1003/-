"""Seal externally produced frozen predictions, then evaluate against human event references.

This is an evaluation gate, NOT an unvalidated raw-video inference adapter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *
from holdout_tools import verify_method


def released_windows(release_dir):
    release_path = release_dir / "test_release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if sha256(release_dir / "test_windows.csv") != release["test_windows_sha256"]:
        raise ValueError("Released windows changed")
    verify_method(Path(release["method_snapshot"]), release["method_manifest_sha256"])
    return release, read_csv(release_dir / "test_windows.csv")


def make_forms(release_dir, out):
    release, windows = released_windows(release_dir)
    rows = []
    for row in windows.itertuples():
        rows.append({"window_id": row.window_id, "video_id": row.file_id, "video_path": row.source_path,
                     "source_path": row.source_path, "source_start_seconds": row.start_seconds,
                     "duration_seconds": row.duration_seconds, "annotation_round": "R1", "annotator": "",
                     "annotation_status": "pending", "manual_breath_count": "", "event_definition": "expiration_peak",
                     "predictions_hidden": "", "reference_notes": "", "analysis_role": "frozen_external_test"})
    write_csv(out / "annotation_windows.csv", rows)
    template = REFERENCE / "annotations" / "20260909_v1"
    for name in ("reference_events.csv", "unobservable_intervals.csv"):
        write_csv(out / name, read_csv(template / name).iloc[:0])
    write_json(out / "reference_source.json", {"release_dir": str(release_dir.resolve()),
                                               "release_sha256": sha256(release_dir / "test_release.json")})


def validate_predictions(pred, windows, method_hash):
    if pred.window_id.duplicated().any() or set(pred.window_id) != set(windows.window_id):
        raise ValueError("Every released window must occur exactly once, including abstentions")
    lookup = windows.set_index("window_id")
    for row in pred.itertuples():
        if row.method_manifest_sha256 != method_hash:
            raise ValueError("Mixed or incorrect method snapshot")
        if abs(float(row.duration_seconds) - float(lookup.loc[row.window_id].duration_seconds)) > 1e-6:
            raise ValueError("Window duration mismatch")
        if row.prediction_status not in {"ok", "abstain"}:
            raise ValueError("Status must be ok or abstain; failures must not disappear")
        if row.prediction_status == "abstain":
            if str(row.predicted_count).strip():
                raise ValueError("An abstention is missing, not a zero-breath prediction")
        else:
            count = float(row.predicted_count)
            if not np.isfinite(count) or count < 0 or not count.is_integer():
                raise ValueError("Explicit nonnegative integer breath count required")


def seal(release_dir, predictions_path, provenance_path):
    release, windows = released_windows(release_dir)
    pred = read_csv(predictions_path)
    validate_predictions(pred, windows, release["method_manifest_sha256"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for key in ("runner_path", "runner_sha256", "operator", "timestamp_validation_report", "started_at", "finished_at"):
        if not provenance.get(key):
            raise ValueError(f"Inference provenance missing: {key}")
    if provenance.get("reference_counts_used") is not False or provenance.get("test_outcomes_used_for_tuning") is not False:
        raise ValueError("Truth-assisted or test-tuned runs are not frozen external predictions")
    if sha256(Path(provenance["runner_path"])) != provenance["runner_sha256"]:
        raise ValueError("Inference runner differs from its declared hash")
    timing_report = Path(provenance["timestamp_validation_report"])
    timing = json.loads(timing_report.read_text(encoding="utf-8"))
    if timing.get("status") != "VALIDATED" or set(timing.get("window_ids", [])) != set(windows.window_id):
        raise ValueError("Complete first-30-second timestamp validation required")
    for row in windows.itertuples():
        if sha256(Path(row.source_path)) != row.source_sha256:
            raise ValueError("Released source file changed")
    # Exclusive creation locks one prediction set to this release. Never tune then replace it.
    write_json(release_dir / "prediction_seal.json", {
        "sealed_at": stamp(), "predictions_path": str(predictions_path.resolve()), "predictions_sha256": sha256(predictions_path),
        "release_sha256": sha256(release_dir / "test_release.json"),
        "provenance_path": str(provenance_path.resolve()), "provenance_sha256": sha256(provenance_path),
        "timestamp_report_sha256": sha256(timing_report),
        "statement_scope": "hashes and operator declaration; not independent proof of blinding"})


def score(release_dir, annotations, out, annotation_round):
    sys.path.insert(0, str(REFERENCE))
    from reference_tools import load_reference, validate_tables
    release, windows = released_windows(release_dir)
    sealed = json.loads((release_dir / "prediction_seal.json").read_text(encoding="utf-8"))
    if sha256(release_dir / "test_release.json") != sealed["release_sha256"]:
        raise ValueError("Release changed since prediction seal")
    pred_path = Path(sealed["predictions_path"])
    if sha256(pred_path) != sealed["predictions_sha256"]:
        raise ValueError("Sealed predictions changed")
    pred = read_csv(pred_path)
    validate_predictions(pred, windows, release["method_manifest_sha256"])
    labels, events, intervals = load_reference(annotations)
    errors = validate_tables(labels, events, intervals)
    if errors:
        raise ValueError("Invalid human event reference: " + "; ".join(errors[:10]))
    labels = labels[labels.annotation_round == annotation_round]
    if set(labels.window_id) != set(windows.window_id):
        raise ValueError("Reference must list all released test windows")
    merged = pred.merge(labels[["window_id", "manual_breath_count", "annotation_status", "duration_seconds"]],
                        on="window_id", validate="one_to_one", suffixes=("", "_reference"))
    if not np.allclose(merged.duration_seconds.astype(float), merged.duration_seconds_reference.astype(float), atol=1e-6, rtol=0):
        raise ValueError("Reference duration differs from frozen window")
    primary = merged[merged.annotation_status.eq("complete") & merged.prediction_status.eq("ok")].copy()
    result = None
    if len(primary):
        primary["truth_count"] = primary.manual_breath_count.astype(float)
        primary["predicted_count"] = primary.predicted_count.astype(float)
        primary["truth_rr_bpm"] = primary.truth_count * 60 / primary.duration_seconds.astype(float)
        primary["predicted_rr_bpm"] = primary.predicted_count * 60 / primary.duration_seconds.astype(float)
        result = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in measurements(primary).items()}
    write_csv(out / "all_released_windows.csv", merged)
    write_csv(out / "primary_paired_predictions.csv", primary)
    write_json(out / "frozen_rr_metrics.json", {
        "status": "SCORED" if result else "AWAIT_RELIABLE_REFERENCE_OR_USABLE_PREDICTIONS",
        "n_released": len(windows), "n_complete_reference": int(merged.annotation_status.eq("complete").sum()),
        "n_algorithm_ok": int(merged.prediction_status.eq("ok").sum()), "n_primary": len(primary),
        "output_coverage": float(merged.prediction_status.eq("ok").mean()), "primary_metrics": result,
        "analysis_policy": "complete human event reference AND ok predictions; all exclusions/abstentions retained",
        "reference_hashes": {name: sha256(annotations / name) for name in ("annotation_windows.csv", "reference_events.csv", "unobservable_intervals.csv")},
        "uncertainty_limit": "Point estimates only; cluster confidence intervals require a reviewed cow mapping"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["make-forms", "seal", "score"])
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--round", default="R1")
    args = parser.parse_args()
    if args.command == "seal":
        if not args.predictions or not args.provenance:
            parser.error("seal requires --predictions and --provenance")
        seal(args.release_dir, args.predictions, args.provenance)
        return
    if not args.out:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.command == "make-forms":
        make_forms(args.release_dir, args.out)
    else:
        if not args.annotations:
            parser.error("score requires --annotations")
        score(args.release_dir, args.annotations, args.out, args.round)


if __name__ == "__main__":
    main()
