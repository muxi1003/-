"""Separate prediction and scoring commands for a predeclared internal-only guard."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, REFERENCE, ABLATION, read_csv, write_csv, write_json, sha256, measurements
from signal_observability_guard import POLICY, protect_window


def infer(out):
    out.mkdir(parents=True, exist_ok=False)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    for item in json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    config = rr.ReproConfig(**json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))["signal_config"])
    assert config.max_track_gap == POLICY["max_short_gap_frames"] and config.conf == POLICY["confidence_minimum"]
    if config.truth_csv is not None or config.optimize_peaks or config.adaptive_peak_retuning:
        raise ValueError("Truth-dependent configuration forbidden")
    source = HOLDOUT / "adapter_validation/20260912_v2"
    manifest = pd.read_csv(source / "source_manifest.csv", dtype={"video_id": str})
    if len(manifest) != 49 or manifest.video_id.nunique() != 49:
        raise ValueError("Invalid internal membership")
    for path in [Path(__file__), Path(__file__).with_name("signal_observability_guard.py")]:
        (out / path.name).write_bytes(path.read_bytes())
    write_json(out / "protocol_before_predictions.json", {
        "policy": POLICY, "reference_counts_used_in_inference": False,
        "input_scope": "internal49_annotation_CFR_existing_temperature_snapshot; plus_unscored_raw49",
        "source_manifest_sha256": sha256(source / "source_manifest.csv"),
        "method_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "adoption_rule": "keep_default_if_coverage_drops_or_either_all49_or_completed39_R2_MAE_regresses",
        "partial_peak_count": "diagnostic_only_never_reported_as_full_window_count",
        "no_parameter_search": True, "external_inference": False})
    sources, predictions = [], []

    def process(video_id, table, duration, cohort):
        curve, summary = rr.fuse_temperature_curve(table, config, truth_row=None)
        support, gaps, status = protect_window(table, str(summary["selected_fusion_mode"]))
        count = int(summary["peaks"])
        if count != int(curve.is_peak.sum()):
            raise ValueError("Peak/count mismatch")
        support["baseline_is_peak"] = curve.is_peak.to_numpy(bool)
        support["diagnostic_peak_unsafe"] = support.baseline_is_peak & support.unsafe_event_context
        support["time_seconds"] = np.arange(len(table)) * duration / len(table)
        write_csv(out / cohort / "support" / f"{video_id}.csv", support)
        write_csv(out / cohort / "gaps" / f"{video_id}.csv", gaps,
                  columns=["start_frame", "stop_frame_exclusive", "length_frames", "boundary", "permitted_short_gap"])
        write_csv(out / cohort / "control_curves" / f"{video_id}.csv", curve)
        candidate = count if status["prediction_status"] == "ready" else None
        return {"video_id":video_id,"duration_seconds":duration,"frames":len(table),
                "control_count":count,"candidate_count":candidate,
                "candidate_rr_bpm":60*candidate/duration if candidate is not None else None,
                "diagnostic_unsafe_peaks":int(support.diagnostic_peak_unsafe.sum()), **status}

    for row in manifest.itertuples():
        path = source / "temperature_input_snapshots" / f"{row.video_id}.csv"
        if sha256(path) != row.temperature_sha256:
            raise ValueError("Internal temperature snapshot changed")
        target = out / "internal49" / "input_temperatures" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        table = pd.read_csv(path, float_precision="round_trip")
        if len(table) != int(row.frames):
            raise ValueError("Internal frame count mismatch")
        predictions.append(process(row.video_id, table, float(row.frozen_duration), "internal49"))
        sources.append({"video_id":row.video_id,"temperature_sha256":sha256(path),"duration_seconds":float(row.frozen_duration),
                        "video_path":row.video_path,"video_sha256":row.video_sha256})
    write_csv(out / "internal_predictions_before_scoring.csv", predictions)
    write_csv(out / "internal_sources.csv", sources)
    raw = HOLDOUT / "raw_pipeline_validation/20260912_v1/adaptive_roi_20260914_v2"
    raw_table = pd.read_csv(raw / "predictions_unscored.csv", dtype={"video_id":str}, keep_default_na=False)
    if set(raw_table.video_id) != set(manifest.video_id) or len(raw_table) != 49:
        raise ValueError("Raw internal cohort mismatch")
    raw_predictions = []
    for row in raw_table.itertuples():
        if row.prediction_status == "abstain":
            raw_predictions.append({"video_id":row.video_id,"duration_seconds":30.,"control_count":None,
                                    "candidate_count":None,"prediction_status":"abstain","reason":"upstream_"+row.reason})
            continue
        path = raw / "temperatures" / f"{row.video_id}.csv"
        table = pd.read_csv(path, float_precision="round_trip")
        item = process(row.video_id, table, 30., "raw49_unscored")
        if item["control_count"] != int(float(row.predicted_breath_count)):
            raise ValueError("Raw candidate baseline replay drift")
        item["input_temperature_sha256"] = sha256(path)
        raw_predictions.append(item)
    write_csv(out / "raw_predictions_unscored.csv", raw_predictions)
    write_json(out / "prediction_seal.json", {
        "internal_predictions_sha256":sha256(out / "internal_predictions_before_scoring.csv"),
        "raw_predictions_sha256":sha256(out / "raw_predictions_unscored.csv"),
        "counts_reference_read":False,"default_modified":False})
    print("Predictions saved before scoring; 49 aligned internal and 49 unscored raw windows.")


def score(out):
    seal = json.loads((out / "prediction_seal.json").read_text(encoding="utf-8"))
    if sha256(out / "internal_predictions_before_scoring.csv") != seal["internal_predictions_sha256"]:
        raise ValueError("Prediction seal changed")
    reference_path = REFERENCE / "submissions/20260911_v2/lindian49/count_reference_only.csv"
    reference = pd.read_csv(reference_path, dtype={"video_id":str})
    predicted = pd.read_csv(out / "internal_predictions_before_scoring.csv", dtype={"video_id":str})
    source = pd.read_csv(out / "internal_sources.csv", dtype={"video_id":str})
    if len(reference) != 49 or set(reference.video_id) != set(predicted.video_id):
        raise ValueError("Reference membership mismatch")
    reference_index = reference.set_index("video_id")
    for row in source.itertuples():
        r = reference_index.loc[row.video_id]
        if abs(float(row.duration_seconds)-float(r.duration_seconds)) > 1e-8 or Path(row.video_path) != Path(r.video_path):
            raise ValueError("Reference duration/video mismatch")
        if sha256(Path(row.video_path)) != row.video_sha256:
            raise ValueError("Human-viewed video changed")
    historical = pd.read_csv(ABLATION / "input_snapshots/20260909_v1/historical_sources/lindian_anchored30_motion_robust_rr_predictions.csv", dtype={"video_id":str})
    merged = predicted.merge(reference[["video_id","manual_breath_count","annotation_status"]], on="video_id", validate="one_to_one")
    old_counts = historical.set_index("video_id").predicted_breath_count
    if not all(int(r.control_count) == int(old_counts.loc[r.video_id]) for r in merged.itertuples()):
        raise ValueError("49-window historical count replay drift")
    merged["truth_count"] = merged.manual_breath_count
    merged["truth_rr_bpm"] = 60*merged.truth_count/merged.duration_seconds
    metrics = []
    adopt = True
    for name, group in [("all49_sensitivity",merged),("completed39_primary",merged[merged.annotation_status.eq("completed")])]:
        pair = group[group.candidate_count.notna()]
        values = {}
        for arm, data, field in [("control_full",group,"control_count"),("control_same_output_subset",pair,"control_count"),("guard_output_subset",pair,"candidate_count")]:
            data = data.copy()
            data["predicted_count"] = data[field]
            data["predicted_rr_bpm"] = 60*data.predicted_count/data.duration_seconds
            value = measurements(data) if len(data) else {"n":0,"rr_r2":np.nan,"rr_mae_bpm":np.nan}
            values[arm] = value
            metrics.append({"analysis_set":name,"arm":arm,"eligible_reference_n":len(group),"coverage":len(data)/len(group),**value})
        adopt = adopt and len(pair)==len(group) and values["guard_output_subset"]["rr_r2"]>=values["control_full"]["rr_r2"]-1e-12 and values["guard_output_subset"]["rr_mae_bpm"]<=values["control_full"]["rr_mae_bpm"]+1e-12
    write_csv(out / "internal_metrics.csv", metrics)
    write_csv(out / "internal_evaluated_rows.csv", merged)
    write_json(out / "decision.json", {"status":"RETAIN_CANDIDATE_NOT_DEPLOYED" if adopt else "DO_NOT_ADOPT_KEEP_DEFAULT",
               "reference_sha256":sha256(reference_path),"control_replay_matches":49,
               "candidate_outputs":int(merged.candidate_count.notna().sum()),
               "reason":"coverage_and_same_analysis_set_predeclared_rule; subset_metrics_not_full_cohort_improvement",
               "default_modified":False,"external_evaluation":False,
               "reference_limit":"internal_single_observer_nonblind_total_counts_not_event_reference"})
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["infer","score"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (infer if args.phase == "infer" else score)(args.out)
