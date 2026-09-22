"""Internal raw/JPEG/pose/ROI parity and paired still-image RF checks, without RR labels."""
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
from experiment_common import *
from raw_frame_adapter import FRAME_POLICY, canonical_bgr


def stats(y, p):
    error = p-y
    sst = np.sum((y-y.mean())**2)
    return {"temperature_r2": float(1-np.sum(error**2)/sst) if sst else None,
            "mae_celsius": float(np.mean(abs(error))), "rmse_celsius": float(np.sqrt(np.mean(error**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    manifest = json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    for path in (Path(__file__), Path(__file__).parent / "raw_frame_adapter.py"):
        with (args.out / path.name).open("xb") as handle:
            handle.write(path.read_bytes())
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    from ultralytics import YOLO
    yolo_path = frozen / "weights/YOLO11n-best.pt"
    rf_path = frozen / "weights/clf_model_RGB_20240906.pkl"
    detector = YOLO(str(yolo_path))
    model = joblib.load(rf_path)
    source_manifest = ASSETS / "lindian_anchored_fixed30_manifest.csv"
    all_windows = read_csv(source_manifest).set_index("video_id")
    selected = read_csv(ABLATION / "input_snapshots/20260909_v1/matched_windows.csv").query("cohort == 'anchored49'")
    if len(selected) != 49 or selected.video_id.duplicated().any():
        raise ValueError("Incorrect internal cohort")
    write_json(args.out / "config_before_results.json", {"frame_policy": FRAME_POLICY, "pose_task": detector.task,
               "detector_sha256": sha256(yolo_path), "rf_sha256": sha256(rf_path),
               "raw_source_manifest_sha256": sha256(source_manifest), "opencv_version": cv2.__version__,
               "frames_per_window": "first_middle_last_predeclared", "roi_radius_pixels": 20,
               "min_temperature_celsius": 20, "rr_labels_used": False, "test_predictions_generated": False,
               "acceptance": "exact_canonical_pixel_pose_ROI_parity_on_all_sampled_internal_frames; calibration_results_descriptive_only"})
    rows, roi_rows, window_rows = [], [], []
    for ordinal, item in enumerate(selected.itertuples()):
        meta = all_windows.loc[item.video_id]
        raw_path = Path(meta.raw_source_path)
        current_hash = sha256(raw_path)
        if current_hash != meta.raw_source_sha256:
            raise ValueError(f"Raw source changed: {item.video_id}")
        cap = cv2.VideoCapture(str(raw_path), cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        advertised = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        first = int(round(float(meta.anchored_window_start_seconds)*fps))
        n = int(item.frames)
        safety = min(2, first) if first+n >= advertised else 0
        first -= safety
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        checkpoints = {0, n//2, n-1}
        timestamps = []
        try:
            for i in range(n):
                ok, raw = cap.read()
                if not ok:
                    raise ValueError(f"Raw window decode ended early: {item.video_id}/{i}")
                t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                timestamps.append(t)
                if i not in checkpoints:
                    continue
                path = Path(item.image_dir)/f"frame_{i:06d}.jpg"
                saved = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if saved is None or saved.shape != raw.shape:
                    raise ValueError("Stored/raw geometry mismatch")
                canonical = canonical_bgr(raw)
                exact = np.array_equal(canonical, saved)
                # Identical inference entry points expose any preprocessing/geometry drift.
                a = detector(saved, verbose=False)[0]
                b = detector(canonical, verbose=False)[0]
                ia, ib = rr.select_detection(a), rr.select_detection(b)
                ka = a.keypoints.data[ia].detach().cpu().numpy() if ia is not None else np.empty((0,3))
                kb = b.keypoints.data[ib].detach().cpu().numpy() if ib is not None else np.empty((0,3))
                pose_equal = ka.shape == kb.shape and np.array_equal(ka, kb)
                delta = float(np.max(abs(ka-kb))) if ka.size and ka.shape == kb.shape else None
                rows.append({"video_id": item.video_id, "frame_index": i, "raw_frame_index_requested": first+i,
                             "source_timestamp_seconds": t, "saved_image": str(path), "saved_image_sha256": sha256(path),
                             "height": raw.shape[0], "width": raw.shape[1], "canonical_pixels_exact": exact,
                             "canonical_pixel_mae": float(np.mean(abs(canonical.astype(float)-saved.astype(float)))),
                             "direct_raw_pixel_mae": float(np.mean(abs(raw.astype(float)-saved.astype(float)))),
                             "pose_exact": bool(pose_equal), "pose_max_abs_delta": delta,
                             "left_right_order": "model_keypoint_0_then_1_not_verified_anatomical_names"})
                for side, k in (("left",0),("right",1)):
                    if len(ka) <= k:
                        continue
                    x,y,confidence = rr.keypoint_xy_conf(ka[k])
                    if confidence < .5:
                        continue
                    ts = rr.circle_temperature(saved, model, x, y, 20, 20.)
                    tc = rr.circle_temperature(canonical, model, x, y, 20, 20.)
                    tr = rr.circle_temperature(raw, model, x, y, 20, 20.)
                    equivalent = (np.isnan(ts) and np.isnan(tc)) or np.isclose(ts,tc,rtol=0,atol=1e-10)
                    roi_rows.append({"video_id": item.video_id, "frame_index": i, "side": side, "x_native": x,
                                     "y_native": y, "confidence": confidence, "roi_pixels": 20,
                                     "saved_jpeg_temperature": ts, "canonical_temperature": tc, "raw_direct_temperature": tr,
                                     "canonical_matches": bool(equivalent), "raw_direct_minus_jpeg": tr-ts})
        finally:
            cap.release()
        times = np.asarray(timestamps)
        write_csv(args.out/"raw_window_timestamps"/f"{item.video_id}.csv", pd.DataFrame({"local_frame_index": np.arange(n), "raw_timestamp_seconds": times,
                     "raw_relative_to_first_frame_seconds": times-times[0], "legacy_nominal_seconds": np.arange(n)/fps}))
        window_rows.append({"video_id": item.video_id, "raw_source_path": str(raw_path), "raw_source_sha256": current_hash,
                            "requested_start_frame": first, "end_safety_shift_frames": safety, "frames_decoded": n,
                            "raw_max_interval_seconds": float(np.max(np.diff(times))),
                            "raw_timestamp_monotonic": bool(np.all(np.diff(times)>0)),
                            "relative_nominal_max_time_error_seconds": float(np.max(abs((times-times[0])-np.arange(n)/fps)))})
        if (ordinal+1)%5 == 0:
            print(f"Internal raw/JPEG/pose/ROI {ordinal+1}/49", flush=True)
    write_csv(args.out/"frame_parity.csv", rows)
    write_csv(args.out/"roi_temperature_parity.csv", roi_rows)
    write_csv(args.out/"raw_window_audit.csv", window_rows)
    predictor = vars(detector.predictor.args)
    write_json(args.out/"actual_yolo_predictor_arguments.json", {k: str(v) if isinstance(v,Path) else v for k,v in predictor.items()})
    calibration_dir = REPO/"temperature_extraction/getRandomForestRegress"
    calibration_rows = []
    for image_name, csv_name, farm, scope in (("1.png","1.csv","jiufu2024","training_image_fit_not_independent_accuracy"),
                                             ("20230810T152315.JPG","20230810T152315.csv","lindian2023","paired_still_transfer_not_video_temperature_validation")):
        image_path, csv_path = calibration_dir/image_name, calibration_dir/csv_name
        image = cv2.imread(str(image_path))
        target = pd.read_csv(csv_path,header=None).to_numpy(float)
        if image.shape[:2] != target.shape or not np.isfinite(target).all():
            raise ValueError("Calibration image and matrix are not pixel-aligned in shape")
        pixels, inverse = np.unique(image.reshape(-1,3),axis=0,return_inverse=True)
        estimate = model.predict(pixels)[inverse]
        y = target.ravel()
        calibration_rows.append({"image": str(image_path), "matrix": str(csv_path), "farm_user_declared": farm,
                                 "image_sha256":sha256(image_path), "matrix_sha256":sha256(csv_path), "pixels":len(y),
                                 "unique_bgr_colors":len(pixels), "true_min_celsius":float(y.min()),"true_max_celsius":float(y.max()),
                                 "scope":scope, **stats(y,estimate)})
    write_csv(args.out/"paired_calibration_still_checks.csv",calibration_rows)
    exact_frames = sum(r["canonical_pixels_exact"] for r in rows)
    pose_frames = sum(r["pose_exact"] for r in rows)
    roi_match = sum(r["canonical_matches"] for r in roi_rows)
    result = {"status":"SAMPLED_INTERNAL_FRAME_PARITY_PASS" if rows and roi_rows and exact_frames==len(rows) and pose_frames==len(rows) and roi_match==len(roi_rows) else "PARITY_REVIEW_REQUIRED",
              "internal_windows":49,"sampled_frames":len(rows),"exact_canonical_frames":exact_frames,
              "exact_pose_frames":pose_frames,"roi_comparisons":len(roi_rows),"roi_matches":roi_match,
              "roi_covered_frames":len({(r["video_id"], r["frame_index"]) for r in roi_rows}),
              "roi_covered_windows":len({r["video_id"] for r in roi_rows}),
              "external_videos_validated":0,"external_rr_predictions":0,
              "remaining":"full_raw_runner_and_target_video_palette_scale_compatibility; keypoint anatomical side naming not visually verified"}
    write_json(args.out/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
