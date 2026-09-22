"""Apply the separate frozen adaptive-radius policy to internal raw-window artifacts."""
import argparse
import json
import math
import platform
from importlib.metadata import version
from types import SimpleNamespace
from pathlib import Path
import sys

import joblib
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, read_csv, sha256, write_csv, write_json
from timestamp_adapter import POLICY, sample_map


def read_bgr_strict(path):
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Image decode failed, not a missing nostril: {path}")
    return image


def longest_missing(mask):
    best = current = 0
    for missing in mask:
        current = current + 1 if missing else 0
        best = max(best, current)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    for item in json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    import evaluate_lindian_adaptive_roi as roi
    # The frozen helper only uses cv2.imread; isolate Unicode-safe I/O without changing ROI math.
    roi.cv2 = SimpleNamespace(imread=read_bgr_strict)
    configuration = json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))
    config, policy = rr.ReproConfig(**configuration["signal_config"]), configuration["roi_policy"]
    model = joblib.load(frozen / "weights/clf_model_RGB_20240906.pkl")
    source_predictions = args.source / "predictions_unscored.csv"
    rows = pd.read_csv(source_predictions, dtype={"video_id": str}, keep_default_na=False)
    internal_manifest_path = HOLDOUT / "raw_input_validation/20260912_v1/raw_window_audit.csv"
    internal = read_csv(internal_manifest_path).set_index("video_id")
    source_protocol = json.loads((args.source / "protocol_before_predictions.json").read_text(encoding="utf-8"))
    if len(rows) != 49 or rows.video_id.nunique() != 49 or set(rows.video_id) != set(internal.index):
        raise ValueError("Incomplete internal source")
    if (source_protocol["scope"] != "internal49_raw_time_new_windows_no_reference_scoring"
            or source_protocol["time_policy"] != POLICY or source_protocol["duration_seconds"] != 30.0
            or source_protocol["source_manifest_sha256"] != sha256(internal_manifest_path)
            or source_protocol["method_manifest_sha256"] != sha256(frozen / "freeze_manifest.json")
            or not rows.duration_seconds.astype(float).eq(30).all()):
        raise ValueError("Source protocol is not the intended internal raw-time run")
    for row in rows.itertuples():
        if (row.raw_source_sha256 != internal.loc[row.video_id, "raw_source_sha256"]
                or Path(row.raw_source_path) != Path(internal.loc[row.video_id, "raw_source_path"])
                or int(row.origin_frame_index) != int(internal.loc[row.video_id, "requested_start_frame"])):
            raise ValueError("Internal source identity mismatch")
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(args.out / "runtime_environment.json", {
        "stage": "adaptive_ROI_stage_runtime_not_retroactive_v1_process_capture",
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: version(name) for name in ["ultralytics", "torch", "numpy", "pandas", "scikit-learn", "scipy", "opencv-python", "joblib"]},
        "loaded_rr_module": rr.__file__, "loaded_roi_module": roi.__file__,
        "rf_sha256": sha256(frozen / "weights/clf_model_RGB_20240906.pkl")})
    write_json(args.out / "protocol_before_predictions.json", {
        "scope": "internal_raw30_adaptive_ROI_unscored_not_adopted",
        "source_predictions_sha256": sha256(source_predictions), "roi_policy": policy,
        "method_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "source_reference_exposure": "v1_loaded_internal_membership_table_containing_counts_but_did_not_use_count_columns; corrected_in_future_runner",
        "this_stage_reads_human_counts": False, "holdout_run": False,
        "image_reader": "strict_numpy_fromfile_cv2_imdecode_BGR; Unicode_paths_supported; decode_error_aborts",
        "signal_observability": "descriptive_only_no_new_gate_selected_from_results",
        "frozen_limit": "coordinate-based_radius_table_can_extract_at_low_confidence_or_inferred_coordinates; does_not_prove_visible_nostril"})
    results = []
    for ordinal, row in enumerate(rows.itertuples(), 1):
        result = {"video_id": row.video_id, "duration_seconds": 30.0,
                  "prediction_status": row.prediction_status, "reason": row.reason,
                  "predicted_breath_count": None, "predicted_rr_bpm": None,
                  "reference_status": "new_raw_window_no_accuracy_scoring"}
        source_temp = args.source / "temperatures" / f"{row.video_id}.csv"
        eligible = row.prediction_status == "predicted_internal_only" or (row.prediction_status == "abstain" and row.reason == "no_valid_temperature_signal")
        if not eligible and row.prediction_status != "abstain":
            raise ValueError("Unknown upstream status")
        if not eligible and source_temp.exists():
            raise ValueError("Unexpected temperature artifact for timestamp abstention")
        if eligible:
            if not source_temp.exists():
                raise ValueError("Missing required upstream temperature artifact")
            base = pd.read_csv(source_temp)
            frame_dir = args.source / "sampled_frames" / row.video_id
            hashes = pd.read_csv(args.source / "frame_hashes" / f"{row.video_id}.csv")
            mapping = pd.read_csv(args.source / "maps" / f"{row.video_id}.csv", float_precision="round_trip")
            timestamps = pd.read_csv(args.source / "timestamps" / f"{row.video_id}.csv", float_precision="round_trip")
            expected_map, _ = sample_map(timestamps.relative_seconds.to_numpy(), 30.0)
            if expected_map is None:
                raise ValueError("Cannot process an invalid upstream time map")
            pd.testing.assert_frame_equal(mapping, expected_map, check_dtype=False, atol=1e-10, rtol=0)
            expected_names = [f"frame_{int(k):06d}.jpg" for k in mapping.target_index]
            if (len(base) != len(mapping) or len(hashes) != len(mapping)
                    or base.frame_name.tolist() != expected_names
                    or not np.array_equal(hashes.target_index, mapping.target_index)
                    or not np.array_equal(hashes.source_frame_index, mapping.source_frame_index)):
                raise ValueError("Temperature/frame/hash/map alignment mismatch")
            for entry in hashes.itertuples():
                if sha256(frame_dir / f"frame_{entry.target_index:06d}.jpg") != entry.jpeg_sha256:
                    raise ValueError("Source frame changed")
            table = roi.extract_radius_table(frame_dir, base, model, radii=list(range(policy["min_radius"], policy["max_radius"]+1)), min_temp=config.min_temp, batch_frames=24)
            cv = roi.nostril_spacing_cv(table)
            exponent = policy["damped_exponent"] if math.isfinite(cv) and cv > policy["spacing_cv_threshold"] else policy["linear_exponent"]
            selected, radii = roi.radius_policy_table(table, base_radius=policy["base_radius"], exponent=exponent, clip_low=policy["min_radius"], clip_high=policy["max_radius"])
            write_csv(args.out / "radius_tables" / f"{row.video_id}.csv", table)
            write_csv(args.out / "temperatures" / f"{row.video_id}.csv", selected)
            missing = ~np.isfinite(base[["left_temp", "right_temp"]].to_numpy(float)).any(axis=1)
            direct = base.left_source.eq("detected") | base.right_source.eq("detected")
            usable = np.isfinite(selected[["left_temp", "right_temp"]].to_numpy(float)).any(axis=1)
            result.update(base_temperature_sha256=sha256(source_temp), grid_frames=len(base),
                          radius_min=int(radii.min()), radius_max=int(radii.max()), exponent=exponent,
                          spacing_cv=float(cv) if math.isfinite(cv) else None,
                          base_both_missing_frames=int(missing.sum()), base_longest_missing_frames=longest_missing(missing),
                          base_direct_detection_frames=int(direct.sum()), longest_no_direct_detection_frames=longest_missing(~direct.to_numpy()),
                          adaptive_valid_signal_frames=int(usable.sum()),
                          no_direct_detection_support=not bool(direct.any()),
                          exceeds_existing_short_repair_limit=longest_missing(missing) > config.max_track_gap,
                          signal_quality_status="not_validated; interpolation_or_coordinate_inference_may_bridge_missing_observations")
            if not usable.any():
                result.update(prediction_status="abstain", reason="no_valid_temperature_signal")
            else:
                curve, summary = rr.fuse_temperature_curve(selected, config, truth_row=None)
                curve["target_time_seconds"] = mapping.target_time_seconds
                write_csv(args.out / "curves" / f"{row.video_id}.csv", curve)
                count = int(curve.is_peak.sum())
                if count != int(summary["peaks"]):
                    raise ValueError("Event count mismatch")
                result.update(prediction_status="predicted_internal_only", reason="", predicted_breath_count=count, predicted_rr_bpm=2*count)
        results.append(result)
        write_json(args.out / "window_status" / f"{row.video_id}.json", result)
        print(f"Adaptive ROI {ordinal}/49 {row.video_id}", flush=True)
    write_csv(args.out / "predictions_unscored.csv", results)
    output = pd.DataFrame(results)
    write_json(args.out / "summary.json", {"windows":49,"predicted_windows":int(output.predicted_breath_count.notna().sum()),
               "abstention_reasons":output.loc[output.predicted_breath_count.isna(), "reason"].value_counts().to_dict(),
               "accuracy_scored":False,"adopted":False,"external_run":False})


if __name__ == "__main__":
    main()
