"""Locked count-independent raw-video transfer run; never imports human references."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment_common import HOLDOUT, read_csv, sha256, stamp, write_csv, write_json
from timestamp_adapter import POLICY, sample_map
from validate_raw_pipeline_internal49 import frames_from_anchor
from apply_frozen_roi_to_raw_validation import read_bgr_strict
from frozen_rr_evaluation import released_windows, validate_predictions


DEPENDENCIES = ["run_frozen_external.py", "timestamp_adapter.py",
                "validate_raw_pipeline_internal49.py", "apply_frozen_roi_to_raw_validation.py",
                "frozen_rr_evaluation.py", "../experiment_common.py", "../holdout_tools.py"]


def prepare(out):
    release_dir = HOLDOUT / "release_20260909_v1"
    release, windows = released_windows(release_dir)
    if len(windows) != 271 or not windows.start_seconds.astype(float).eq(0).all():
        raise ValueError("Unexpected cohort/origin")
    if not windows.duration_seconds.astype(float).eq(30).all():
        raise ValueError("Unexpected duration")
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for i, name in enumerate(DEPENDENCIES):
        source = (HERE / name).resolve()
        target = out / "code_snapshots" / f"{i:02d}_{source.name}"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        files.append({"path": str(source), "sha256": sha256(source),
                      "snapshot": str(target.resolve())})
    for name in ["test_windows.csv", "test_release.json"]:
        (out / name).write_bytes((release_dir / name).read_bytes())
    write_json(out / "protocol_lock.json", {
        "locked_at": stamp(), "method_name": "frozen_20260909_adaptive_ROI_raw_timestamp_transfer_v1",
        "release_dir": str(release_dir), "release_sha256": sha256(release_dir / "test_release.json"),
        "test_windows_sha256": sha256(out / "test_windows.csv"),
        "method_snapshot": release["method_snapshot"], "method_manifest_sha256": release["method_manifest_sha256"],
        "code": files, "time_policy": POLICY,
        "spatial_policy": "OpenCV_FFMPEG_display_orientation_native_BGR_full_frame; no_rescale_crop_or_channel_swap; YOLO_keypoints_in_native_coordinates; JPEG95",
        "spatial_limit": "IRT_source_directory_and_user_provenance; no_independent_anatomical_visibility_or_overlay_mask_validation",
        "signal_policy": "original_frozen_ROI_and_signal_parameters; no_new_source_guard_or_partial_segment_substitution",
        "calibration_scope": "RF_mapped_pseudocolor_proxy; absolute_Celsius_and_fixed_palette_not_verified; no_offset_correction",
        "abstentions": "timestamp_policy_rejection_or_no_finite_bilateral_signal; software_or_decode_errors_stop_not_silently_drop",
        "outcome_policy": "single_frozen_count_only_external_transfer_evaluation; complete_blinded_manual_counts_AND_ok_predictions; report_all271_and_coverage; cow_cluster_CI; no_event_F1",
        "reference_counts_used": False, "test_outcomes_used_for_tuning": False,
        "training_scope": "YOLO_and_RR_heldout_videos; target_farm_separately_captured_RF_calibration_available_not_zero_target_calibration",
        "internal_evidence": ["raw_pipeline_validation/20260912_v1", "raw_pipeline_validation/20260912_v1/adaptive_roi_20260914_v2"],
        "acceptance_scope": "engineering_checked_transfer_evaluation_not_a_claim_of_good_accuracy_or_radiometric_validity",
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "packages": {k: version(k) for k in ["ultralytics", "torch", "numpy", "pandas", "scikit-learn", "scipy", "opencv-python", "joblib"]}}
    })


def verify_lock(out):
    lock = json.loads((out / "protocol_lock.json").read_text(encoding="utf-8"))
    for item in lock["code"]:
        if sha256(Path(item["path"])) != item["sha256"] or sha256(Path(item["snapshot"])) != item["sha256"]:
            raise ValueError("Locked runner/dependency changed")
    release, windows = released_windows(Path(lock["release_dir"]))
    if (sha256(out / "test_windows.csv") != lock["test_windows_sha256"]
            or sha256(Path(lock["release_dir"]) / "test_release.json") != lock["release_sha256"]
            or release["method_manifest_sha256"] != lock["method_manifest_sha256"]):
        raise ValueError("Release changed")
    return lock, windows


def verify_checkpoint(directory):
    seal = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    for name, digest in seal["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Checkpoint changed: {directory}/{name}")
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def run(out):
    lock, windows = verify_lock(out)
    if (out / "prediction_seal.json").exists():
        raise ValueError("Already sealed; no repeated holdout run")
    frozen = Path(lock["method_snapshot"])
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    import evaluate_lindian_adaptive_roi as roi
    from ultralytics import YOLO
    config_data = json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))
    config, policy = rr.ReproConfig(**config_data["signal_config"]), config_data["roi_policy"]
    if config.truth_csv is not None or config.optimize_peaks or config.adaptive_peak_retuning:
        raise ValueError("Truth-assisted config forbidden")
    roi.cv2 = SimpleNamespace(imread=read_bgr_strict)
    cv2.setNumThreads(1)
    detector = YOLO(str(frozen / "weights/YOLO11n-best.pt"))
    model = joblib.load(frozen / "weights/clf_model_RGB_20240906.pkl")
    results = []
    for ordinal, row in enumerate(windows.itertuples(), 1):
        directory = out / "windows" / row.window_id
        if (directory / "checkpoint.json").exists():
            result = verify_checkpoint(directory)
            if result["window_id"] != row.window_id or result["source_sha256"] != row.source_sha256:
                raise ValueError("Checkpoint identity mismatch")
            results.append(result)
            continue
        # An interrupted unsealed window is not silently overwritten or salvaged.
        directory.mkdir(parents=True, exist_ok=False)
        source = Path(row.source_path)
        if sha256(source) != row.source_sha256:
            raise ValueError("Raw source changed")
        times, absolute, shapes = [], [], set()
        for _, relative, timestamp, frame in frames_from_anchor(source, 0):
            times.append(relative)
            absolute.append(timestamp)
            shapes.add(tuple(frame.shape))
        if not absolute or abs(absolute[0]) > 1e-6 or len(shapes) != 1:
            raise ValueError("Source origin/geometry differs from preflight")
        write_csv(directory / "timestamps.csv", pd.DataFrame({"relative_seconds": times, "absolute_seconds": absolute}))
        mapping, timing = sample_map(times, 30.0)
        result = {"window_id": row.window_id, "prediction_status": "abstain", "predicted_count": None,
                  "duration_seconds": 30.0, "method_manifest_sha256": lock["method_manifest_sha256"],
                  "source_sha256": row.source_sha256, "reason": timing.get("reason", ""),
                  "frame_shape": list(next(iter(shapes))), "timestamp_policy_status": timing["prediction_status"]}
        write_json(directory / "timing_check.json", timing)
        if mapping is not None:
            frame_dir = directory / "frames"
            frame_dir.mkdir()
            wanted = mapping.groupby("source_frame_index").target_index.apply(list).to_dict()
            written = []
            for index, relative, timestamp, frame in frames_from_anchor(source, 0):
                if index not in wanted:
                    continue
                if abs(relative-times[index]) > 1e-6 or abs(timestamp-absolute[index]) > 1e-6:
                    raise ValueError("Decode timestamp replay failed")
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("JPEG encode failed")
                for target in wanted[index]:
                    image = frame_dir / f"frame_{target:06d}.jpg"
                    image.write_bytes(encoded.tobytes())
                    decoded = read_bgr_strict(image)
                    if tuple(decoded.shape) not in shapes:
                        raise ValueError("Spatial shape changed")
                    written.append({"target_index": target, "source_frame_index": index, "sha256": sha256(image)})
            if len(written) != len(mapping):
                raise ValueError("Incomplete sampling")
            write_csv(directory / "map.csv", mapping)
            write_csv(directory / "frame_hashes.csv", sorted(written, key=lambda x: x["target_index"]))
            base = rr.extract_temperatures(frame_dir, detector, model, config)
            names = [f"frame_{k:06d}.jpg" for k in mapping.target_index]
            if base.frame_name.tolist() != names or base.status.eq("read_failed").any():
                raise ValueError("Extraction frame order/read failed")
            write_csv(directory / "base_temperatures.csv", base)
            table = roi.extract_radius_table(frame_dir, base, model, radii=list(range(policy["min_radius"], policy["max_radius"]+1)), min_temp=config.min_temp, batch_frames=24)
            spacing_cv = roi.nostril_spacing_cv(table)
            exponent = policy["damped_exponent"] if math.isfinite(spacing_cv) and spacing_cv > policy["spacing_cv_threshold"] else policy["linear_exponent"]
            selected, radii = roi.radius_policy_table(table, base_radius=policy["base_radius"], exponent=exponent, clip_low=policy["min_radius"], clip_high=policy["max_radius"])
            write_csv(directory / "radius_table.csv", table)
            write_csv(directory / "temperatures.csv", selected)
            result.update(grid_frames=len(mapping), radius_min=int(radii.min()), radius_max=int(radii.max()), exponent=exponent)
            if not np.isfinite(selected[["left_temp", "right_temp"]].to_numpy(float)).any():
                result["reason"] = "no_valid_temperature_signal"
            else:
                curve, summary = rr.fuse_temperature_curve(selected, config, truth_row=None)
                curve["target_time_seconds"] = mapping.target_time_seconds
                count = int(curve.is_peak.sum())
                if count != int(summary["peaks"]):
                    raise ValueError("Peak/count mismatch")
                write_csv(directory / "curve.csv", curve)
                peaks = np.flatnonzero(curve.is_peak.to_numpy(bool))
                write_csv(directory / "algorithm_events.csv", [{"grid_index": int(k), "time_seconds": float(mapping.iloc[k].target_time_seconds), "is_human_reference": False} for k in peaks], columns=["grid_index", "time_seconds", "is_human_reference"])
                result.update(prediction_status="ok", predicted_count=count, reason="")
        write_json(directory / "result.json", result)
        write_json(directory / "checkpoint.json", {"files": {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}})
        results.append(result)
        print(f"{ordinal}/271 completed {row.window_id} status={result['prediction_status']}", flush=True)
    predictions = pd.DataFrame(results)
    prediction_path = out / "prediction_windows.csv"
    write_csv(prediction_path, predictions)
    validate_predictions(read_csv(prediction_path), windows, lock["method_manifest_sha256"])
    verify_lock(out)
    write_json(out / "predictor_arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(detector.predictor.args).items()} if detector.predictor else {})
    write_json(out / "prediction_seal.json", {
        "sealed_at": stamp(), "n_windows": len(results), "predictions_sha256": sha256(prediction_path),
        "protocol_sha256": sha256(out / "protocol_lock.json"), "reference_counts_used": False,
        "test_outcomes_used_for_tuning": False, "scope": "locked_proxy_signal_transfer_count_evaluation_not_radiometric_validation",
        "checkpoints": {row.window_id: sha256(out / "windows" / row.window_id / "checkpoint.json") for row in windows.itertuples()}
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args.out)
