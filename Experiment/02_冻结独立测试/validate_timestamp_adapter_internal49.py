"""Evaluate one predeclared temporal adapter on internal49, not on holdout counts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *
from timestamp_adapter import POLICY, sample_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for source, name in ((Path(__file__), "runner_source_snapshot.py"), (Path(__file__).parent / "timestamp_adapter.py", "adapter_source_snapshot.py")):
        with (args.out / name).open("xb") as handle:
            handle.write(source.read_bytes())
    cv2.setNumThreads(1)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    manifest = json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    config_data = json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))["signal_config"]
    config = rr.ReproConfig(**config_data)
    if config.truth_csv is not None or config.optimize_peaks or config.adaptive_peak_retuning:
        raise ValueError("Truth-assisted config is not allowed")
    ref_dir = REFERENCE / "submissions/20260911_v2/lindian49"
    reference_path = ref_dir / "count_reference_only.csv"
    windows = read_csv(reference_path)
    source_pred = ABLATION / "input_snapshots/20260909_v1/historical_sources/lindian_anchored30_motion_robust_rr_predictions.csv"
    historical = read_csv(source_pred).set_index("video_id")
    if set(windows.video_id) != set(historical.index) or len(windows) != 49:
        raise ValueError("Internal49 population mismatch")
    write_json(args.out / "candidate_config_before_results.json", {
        "policy": POLICY, "signal_config": asdict(config), "runner_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(__file__).parent / "timestamp_adapter.py"),
        "method_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "reference_sha256": sha256(reference_path), "historical_prediction_sha256": sha256(source_pred),
        "adoption_rule": "do_not_adopt_if_any_internal_primary_or_all49_RR_R2_or_MAE_regresses_or_coverage_decreases",
        "reference_role": "internal_development_only; count_fields_used_only_after_predictions_written",
        "validation_limit": "human-viewed CFR MP4s, not raw-video VFR validation",
        "v2_engineering_correction": "normal terminal frame support is based on observed frame interval, not assumed target fps; not selected by count errors"})
    predictions, sources, events = [], [], []
    for index, row in enumerate(windows.itertuples()):
        path = Path(historical.loc[row.video_id, "temperature_csv"])
        source_bytes = path.read_bytes()
        snapshot = args.out / "temperature_input_snapshots" / f"{row.video_id}.csv"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as handle:
            handle.write(source_bytes)
        table = pd.read_csv(snapshot)
        cap = cv2.VideoCapture(row.video_path, cv2.CAP_FFMPEG)
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = []
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        finally:
            cap.release()
        if len(table) != len(times) or n != len(times):
            raise ValueError(f"Temperature/frame mapping mismatch: {row.video_id}")
        duration = float(row.duration_seconds)
        if abs(len(times)/fps-duration) > .02:
            raise ValueError(f"Human MP4 duration incompatible with reference: {row.video_id}")
        sources.append({"video_id": row.video_id, "temperature_path": str(path), "temperature_sha256": sha256(snapshot),
                        "video_path": row.video_path, "video_sha256": sha256(Path(row.video_path)),
                        "video_fps": fps, "frames": len(times), "frozen_duration": duration,
                        "encoded_duration": len(times)/fps, "timebase": "annotation_MP4_CFR_not_raw_source_VFR"})
        write_csv(args.out / "decoded_timestamps" / f"{row.video_id}.csv", pd.DataFrame({"frame_index": np.arange(len(times)), "annotation_video_time_seconds": times}))
        old_curve, old_summary = rr.fuse_temperature_curve(table, config, truth_row=None)
        historical_count = int(historical.loc[row.video_id, "predicted_breath_count"])
        mapping, status = sample_map(times, duration)
        candidate_count = None
        if mapping is not None:
            sampled = table.iloc[mapping.source_frame_index.to_numpy(int)].reset_index(drop=True)
            curve, summary = rr.fuse_temperature_curve(sampled, config, truth_row=None)
            candidate_count = int(summary["peaks"])
            curve["target_time_seconds"] = mapping.target_time_seconds
            write_csv(args.out / "candidate_curves" / f"{row.video_id}.csv", curve)
            write_csv(args.out / "sample_maps" / f"{row.video_id}.csv", mapping)
            for k in np.flatnonzero(curve.is_peak.to_numpy(bool)):
                events.append({"video_id": row.video_id, "event_id": f"p{k}", "grid_time_seconds": float(mapping.iloc[k].target_time_seconds),
                               "source_frame_time_seconds": float(mapping.iloc[k].source_time_seconds),
                               "scope": "algorithm_event_not_human_reference"})
        predictions.append({"video_id": row.video_id, "historical_count": historical_count,
                            "control_count": int(old_summary["peaks"]), "candidate_count": candidate_count,
                            "historical_replay_matches": int(old_summary["peaks"]) == historical_count,
                            "duration_seconds": duration, **status})
        if (index+1) % 10 == 0:
            print(f"Internal temporal candidate {index+1}/49", flush=True)
    write_csv(args.out / "predictions_before_scoring.csv", predictions)
    write_csv(args.out / "source_manifest.csv", sources)
    write_csv(args.out / "candidate_algorithm_events.csv", events)
    # Counts only enter after all candidate and control predictions have been saved.
    evaluated = pd.DataFrame(predictions).merge(windows[["video_id", "manual_breath_count", "annotation_status"]], on="video_id", validate="one_to_one")
    evaluated["truth_count"] = evaluated.manual_breath_count.astype(int)
    evaluated["truth_rr_bpm"] = 60 * evaluated.truth_count / evaluated.duration_seconds
    metric_rows = []
    adopted = evaluated.historical_replay_matches.all()
    for name, data in (("all49_sensitivity", evaluated), ("completed39_count_reference", evaluated[evaluated.annotation_status.eq("completed")])):
        groups = {}
        for arm, field in (("control", "control_count"), ("timestamp_candidate", "candidate_count")):
            g = data[data[field].notna()].copy()
            if len(g):
                g["predicted_count"] = g[field]
                g["predicted_rr_bpm"] = 60*g.predicted_count/g.duration_seconds
                values = measurements(g)
            else:
                values = {"n": 0, "rr_r2": None, "rr_mae_bpm": None}
            groups[arm] = values
            metric_rows.append({"analysis_set": name, "arm": arm, "eligible_reference_n": len(data), "output_coverage": len(g)/len(data), **values})
        candidate, control = groups["timestamp_candidate"], groups["control"]
        adopted = adopted and candidate["n"] == control["n"] and candidate["rr_r2"] >= control["rr_r2"]-1e-12 and candidate["rr_mae_bpm"] <= control["rr_mae_bpm"]+1e-12
    write_csv(args.out / "internal_metrics.csv", metric_rows)
    write_csv(args.out / "count_differences.csv", evaluated)
    write_json(args.out / "decision.json", {"status": "ELIGIBLE_FOR_FURTHER_VALIDATION_NOT_DEPLOYED" if adopted else "DO_NOT_ADOPT_KEEP_BASELINE",
               "historical_replay_matches": int(evaluated.historical_replay_matches.sum()),
               "candidate_count_changes": int(evaluated.candidate_count.ne(evaluated.control_count).sum()),
               "candidate_output_windows": int(evaluated.candidate_count.notna().sum()),
               "default_modified": False, "external_inference_run": False,
               "scope_limit": "internal_CFR_MP4_and_existing_temperatures; not raw_video_VFR_or_end_to_end_external_validation"})
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print("Eligibility for further validation:", bool(adopted), flush=True)


if __name__ == "__main__":
    main()
