"""Internal raw-time validation; only membership columns parsed, counts not used."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import ABLATION, HOLDOUT, read_csv, sha256, write_csv, write_json
from timestamp_adapter import POLICY, sample_map


def frames_from_anchor(path, first, duration=30.0):
    """Decode sequentially to avoid approximate frame seeking; include boundary frame."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open {path}")
    origin = None
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if index >= first:
                if origin is None:
                    origin = timestamp
                relative = timestamp - origin
                yield index - first, relative, timestamp, frame
                if relative >= duration:
                    break
            index += 1
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    for item in json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    from ultralytics import YOLO
    config = rr.ReproConfig(**json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))["signal_config"])
    if config.truth_csv is not None or config.optimize_peaks or config.adaptive_peak_retuning:
        raise ValueError("Truth-assisted config forbidden")
    source_path = HOLDOUT / "raw_input_validation/20260912_v1/raw_window_audit.csv"
    sources = read_csv(source_path)
    cohort = pd.read_csv(ABLATION / "input_snapshots/20260909_v1/matched_windows.csv", usecols=["video_id", "cohort"], dtype=str)
    ids = set(cohort.loc[cohort.cohort.eq("anchored49"), "video_id"])
    if len(sources) != 49 or sources.video_id.duplicated().any() or set(sources.video_id) != ids:
        raise ValueError("Internal cohort mismatch")
    for p in (Path(__file__), Path(__file__).with_name("timestamp_adapter.py")):
        (args.out / p.name).write_bytes(p.read_bytes())
    write_json(args.out / "protocol_before_predictions.json", {
        "scope": "internal49_raw_time_new_windows_no_reference_scoring",
        "duration_seconds": 30.0, "origin": "timestamp_of_existing_audited_anchor_frame_sequentially_decoded",
        "not_claimed": "original_requested_wall_clock_anchor_accuracy_or_old_CFR_reference_equivalence",
        "time_policy": POLICY, "signal_config": json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))["signal_config"],
        "source_manifest_sha256": sha256(source_path), "method_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "input": "raw_decode_to_JPEG95_on_timestamp_grid_then_frozen_extractor",
        "temperature_scope": "frozen_RF_proxy_signal_absolute_Celsius_not_validated",
        "truth_read": False, "holdout_inference": False, "default_modified": False,
        "acceptance": "engineering_validation_only_no_adoption_or_accuracy_claim"})
    detector = YOLO(str(frozen / "weights/YOLO11n-best.pt"))
    model = joblib.load(frozen / "weights/clf_model_RGB_20240906.pkl")
    results = []
    for ordinal, row in enumerate(sources.itertuples(), 1):
        path, first = Path(row.raw_source_path), int(row.requested_start_frame)
        if sha256(path) != row.raw_source_sha256:
            raise ValueError(f"Source changed: {row.video_id}")
        times, absolute = [], []
        for _, relative, timestamp, _ in frames_from_anchor(path, first):
            times.append(relative)
            absolute.append(timestamp)
        write_csv(args.out / "timestamps" / f"{row.video_id}.csv", pd.DataFrame({"local_frame_index": range(len(times)), "relative_seconds": times, "absolute_seconds": absolute}))
        mapping, status = sample_map(times, 30.0)
        result = {"video_id": row.video_id, "raw_source_path": str(path), "raw_source_sha256": row.raw_source_sha256,
                  "origin_frame_index": first, "origin_timestamp_seconds": absolute[0] if absolute else None,
                  "duration_seconds": 30.0, "predicted_breath_count": None, "predicted_rr_bpm": None,
                  "reference_status": "new_raw_time_window_not_scored", **status}
        if mapping is not None:
            frame_dir = args.out / "sampled_frames" / row.video_id
            frame_dir.mkdir(parents=True)
            wanted = mapping.groupby("source_frame_index").target_index.apply(list).to_dict()
            written = []
            for index, relative, timestamp, frame in frames_from_anchor(path, first):
                if index not in wanted:
                    continue
                if abs(relative-times[index]) > 1e-6 or abs(timestamp-absolute[index]) > 1e-6:
                    raise ValueError("Decode replay timestamp mismatch")
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("JPEG encode failed")
                for target in wanted[index]:
                    image_path = frame_dir / f"frame_{target:06d}.jpg"
                    image_path.write_bytes(encoded.tobytes())
                    written.append({"target_index": target, "source_frame_index": index, "jpeg_sha256": sha256(image_path)})
            if len(written) != len(mapping):
                raise ValueError("Sampling materialization incomplete")
            write_csv(args.out / "maps" / f"{row.video_id}.csv", mapping)
            write_csv(args.out / "frame_hashes" / f"{row.video_id}.csv", sorted(written, key=lambda x: x["target_index"]))
            temperatures = rr.extract_temperatures(frame_dir, detector, model, config)
            if len(temperatures) != len(mapping):
                raise ValueError("Extraction length mismatch")
            write_csv(args.out / "temperatures" / f"{row.video_id}.csv", temperatures)
            valid = temperatures[["left_temp", "right_temp"]].notna().any(axis=1)
            result["valid_signal_frames"] = int(valid.sum())
            result["grid_frames"] = len(mapping)
            if not valid.any():
                result.update(prediction_status="abstain", reason="no_valid_temperature_signal")
            else:
                curve, summary = rr.fuse_temperature_curve(temperatures, config, truth_row=None)
                curve["target_time_seconds"] = mapping.target_time_seconds
                write_csv(args.out / "curves" / f"{row.video_id}.csv", curve)
                peaks = np.flatnonzero(curve.is_peak.to_numpy(bool))
                if len(peaks) != int(summary["peaks"]):
                    raise ValueError("Event/count mismatch")
                write_csv(args.out / "algorithm_events" / f"{row.video_id}.csv", [{"grid_index": int(k), "time_seconds": float(mapping.iloc[k].target_time_seconds), "reference": False} for k in peaks], columns=["grid_index", "time_seconds", "reference"])
                result.update(prediction_status="predicted_internal_only", predicted_breath_count=len(peaks), predicted_rr_bpm=2*len(peaks))
        results.append(result)
        write_json(args.out / "window_status" / f"{row.video_id}.json", result)
        print(f"{ordinal}/49 {row.video_id}: {result['prediction_status']} {result.get('reason', '')}", flush=True)
    write_csv(args.out / "predictions_unscored.csv", results)
    table = pd.DataFrame(results)
    write_json(args.out / "summary.json", {"windows": len(table), "predicted_windows": int(table.predicted_breath_count.notna().sum()),
               "abstention_reasons": table.loc[table.predicted_breath_count.isna(), "reason"].value_counts().to_dict(),
               "human_reference_scored": False, "external_predictions": 0, "adopted": False})
    if detector.predictor is not None:
        write_json(args.out / "actual_predictor_arguments.json", {k: str(v) if isinstance(v,Path) else v for k,v in vars(detector.predictor.args).items()})


if __name__ == "__main__":
    main()
