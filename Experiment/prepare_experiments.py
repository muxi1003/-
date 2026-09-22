"""Inventory raw data, preserve evaluation inputs, and prepare human reference files."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from experiment_common import *

SESSION = re.compile(r"(?P<date>20\d{6})T(?P<time>\d{6})[-n]?(?P<cow>[A-Za-z]*\d+)", re.I)
PHONE = re.compile(r"VID_(20\d{6})_(\d{6})", re.I)


def parsed_name(path: Path):
    match = SESSION.search(path.stem)
    if match:
        d = match.groupdict()
        return d["date"], d["time"], d["cow"], f'{d["date"]}T{d["time"]}_{d["cow"]}', "thermal_name_hint"
    match = PHONE.search(path.stem)
    if match:
        return match[1], match[2], "", path.stem, "rgb_name_hint"
    # Resolve renamed clips using their parent recording folder when possible.
    for parent in path.parents:
        match = SESSION.fullmatch(parent.name)
        if match:
            d = match.groupdict()
            return d["date"], d["time"], d["cow"], f'{d["date"]}T{d["time"]}_{d["cow"]}', "derived_clip_hint"
    return "", "", "", path.stem, "unknown"


def quick_fingerprint(path: Path) -> str:
    """A sampled duplicate-screening fingerprint, never a full-file identity claim."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - 32768), max(0, size - 65536)}):
            handle.seek(offset)
            digest.update(handle.read(65536))
    return digest.hexdigest()


def inventory(out: Path):
    prior_paths = []
    prior_table = ASSETS / "paper_external_validation_split_all_use_fieldwork_template.csv"
    for value in read_csv(prior_table)["raw_video_path"]:
        match = SESSION.search(value)
        if match:
            d = match.groupdict()
            prior_paths.append((d["date"], d["time"], d["cow"]))
    known_jiufu_sessions = {f"{d}T{t}_{c}" for d, t, c in prior_paths}
    known_jiufu_cows = {c for _, _, c in prior_paths}
    anchor = read_csv(ASSETS / "lindian_anchored_fixed30_manifest.csv")
    known_lindian_paths = {str(Path(p).resolve()).casefold() for p in anchor["raw_source_path"] if p}
    rows = []
    for farm, root in RAW_ROOTS.items():
        paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS), key=lambda p: str(p).casefold())
        print(f"Inventory {farm}: {len(paths)} video files", flush=True)
        for i, path in enumerate(paths):
            date, time, cow, session, kind = parsed_name(path)
            known_session = (farm == "jiufu" and session in known_jiufu_sessions) or str(path.resolve()).casefold() in known_lindian_paths
            known_cow = farm == "jiufu" and cow in known_jiufu_cows
            role = "exclude_known_development_session" if known_session else "exclude_development_cow_overlap" if known_cow else "hold_pending_exposure_and_identity"
            if kind != "thermal_name_hint":
                role = "reference_or_renamed_media_review"
            capture = cv2.VideoCapture(str(path))
            opened = capture.isOpened()
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            capture.release()
            rows.append({"file_id": farm + "_" + hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:14],
                         "farm": farm, "path": str(path.resolve()), "relative_path": str(path.relative_to(root)),
                         "bytes": path.stat().st_size, "sampled_fingerprint": quick_fingerprint(path),
                         "sha256_full": "", "date_from_name": date, "time_from_name": time,
                         "cow_token": cow, "cow_identity_verified": "", "session_token": session,
                         "modality_hint": kind, "modality_verified": "", "opens": opened,
                         "fps_metadata": fps, "frames_metadata": count,
                         "duration_seconds_metadata": count / fps if fps > 0 else "",
                         "width": width, "height": height,
                         "known_development_session": known_session, "known_development_cow": known_cow,
                         "yolo_training_exposure": "no_user_confirmed_20260909" if farm == "jiufu" else "unknown",
                         "rr_tuning_exposure_review": "yes" if known_session else "",
                         "rr_result_viewing_exposure_review": "", "candidate_role": role,
                         "eligible_independent_test": False})
            if (i + 1) % 100 == 0:
                print(f"  {farm}: {i + 1}/{len(paths)} metadata inspected", flush=True)
    table = pd.DataFrame(rows)
    table["sampled_duplicate_group_size"] = table.groupby("sampled_fingerprint")["file_id"].transform("size")
    write_csv(out / "raw_video_inventory.csv", table)
    write_csv(out / "sampled_duplicate_candidates.csv", table[table.sampled_duplicate_group_size > 1])
    write_csv(out / "known_development_sessions.csv", [{"farm": "jiufu", "session_token": s, "evidence": str(prior_table)} for s in sorted(known_jiufu_sessions)])
    candidates = table[(table.modality_hint == "thermal_name_hint") & (~table.known_development_session) & (~table.known_development_cow) & table.opens].copy()
    candidates["reviewer"] = ""
    candidates["review_date"] = ""
    candidates["review_notes"] = ""
    write_csv(out / "independent_candidate_review.csv", candidates)
    windows = []
    for row in candidates.itertuples():
        if float(row.duration_seconds_metadata) < 30:
            continue
        windows.append({"window_id": row.file_id + "_first30", "file_id": row.file_id,
                        "farm": row.farm, "source_path": row.path, "source_session": row.session_token,
                        "cow_token": row.cow_token, "start_seconds": 0, "duration_seconds": 30,
                        "sampling_rule": "first_30_seconds_before_viewing_outcomes",
                        "role": "candidate_not_test", "reference_status": "pending",
                        "timestamp_validation": "metadata_only_not_PTS_audited"})
    write_csv(out / "candidate_windows30.csv", windows)
    pair_rows = []
    rgb = table[table.modality_hint == "rgb_name_hint"]
    for row in table[table.modality_hint == "thermal_name_hint"].itertuples():
        options = rgb[(rgb.farm == row.farm) & (rgb.date_from_name == row.date_from_name)]
        if options.empty or not row.time_from_name:
            continue
        thermal_time = datetime.strptime(row.time_from_name, "%H%M%S")
        for item in options.itertuples():
            seconds = (datetime.strptime(item.time_from_name, "%H%M%S") - thermal_time).total_seconds()
            if abs(seconds) <= 90:
                pair_rows.append({"thermal_file_id": row.file_id, "thermal_path": row.path,
                                  "rgb_file_id": item.file_id, "rgb_path": item.path,
                                  "filename_clock_difference_seconds": seconds,
                                  "same_cow_verified": "", "synchronization_offset_seconds": "",
                                  "synchronization_status": "candidate_only_not_verified"})
    write_csv(out / "rgb_thermal_pair_candidates.csv", pair_rows)
    summary = {"created_at": datetime.now().astimezone().isoformat(), "source_roots": {k: str(v) for k, v in RAW_ROOTS.items()},
               "video_file_count": len(table), "farm_file_counts": table.groupby("farm").size().to_dict(),
               "modality_hints": table.groupby(["farm", "modality_hint"]).size().rename("n").reset_index().to_dict("records"),
               "candidate_review_count": len(candidates), "candidate_window_count": len(windows),
               "confirmed_independent_test_count": 0, "training_statement": "User confirmed Jiufu raw dataset not used for YOLO training only",
               "independence_status": "Await RR tuning/viewing exposure and cow identity review; known split_all_use sessions/cows excluded",
               "fingerprint_scope": "sampled fingerprints screen duplicates; full SHA256 required for test release"}
    write_json(out / "inventory_summary.json", summary)
    return table, summary


def preserve_inputs(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    inputs.mkdir()
    s73 = pd.read_csv(DATA / "al_images" / "paper_repro_summary.csv", dtype={"video_id": str})
    t73 = read_csv(DATA / "video" / "temperature_curves.csv").set_index("video_id")
    p49 = pd.read_csv(ASSETS / "lindian_anchored30_motion_robust_rr_predictions.csv", dtype={"video_id": str})
    anchor = read_csv(ASSETS / "lindian_anchored_fixed30_manifest.csv").set_index("video_id")
    rows, files = [], []
    for cohort, table in (("internal73", s73), ("anchored49", p49)):
        for row in table.itertuples():
            video_id = str(row.video_id)
            record = row._asdict()
            temperature = Path(record["temperature_csv"]) if cohort == "internal73" else DATA / "lindian_anchored30_frames" / video_id / "lindian_motion_roi_radius_temperatures.csv"
            source = pd.read_csv(temperature)
            limit = int(record["analysis_frame_limit"])
            source = source.iloc[:limit].copy()
            if cohort == "anchored49":
                source["left_temp"] = source["left_temp_r20"]
                source["right_temp"] = source["right_temp_r20"]
            source["roi_radius"] = 20
            dest = inputs / cohort / f"{video_id}.csv"
            write_csv(dest, source)
            files.append({"role": "matched_input_temperature", "source": str(temperature), "source_sha256": sha256(temperature), "snapshot": str(dest), "sha256": sha256(dest)})
            if cohort == "internal73":
                reference = t73.loc[video_id]
                count, duration = int(reference.breath_count), float(reference.duration_seconds)
                primary, reliability, old_rr = True, "legacy_manual", float(reference.rr)
                expected = int(record["peaks"])
                image_dir = DATA / "al_images" / video_id
            else:
                count, duration = int(record["manual_breath_count"]), float(record["manual_window_seconds"])
                primary, reliability, old_rr = is_true(record["include_primary_analysis"]), record["truth_reliability"], float(record["manual_rr_bpm"])
                expected = int(record["predicted_breath_count"])
                image_dir = DATA / "lindian_anchored30_frames" / video_id
            raw = str(anchor.loc[video_id, "raw_source_path"]) if video_id in anchor.index else ""
            cluster = "source_" + hashlib.sha256(raw.encode()).hexdigest()[:12] if raw else "unverified_video_" + video_id
            rows.append({"cohort": cohort, "video_id": video_id, "input_temperature_csv": str(dest),
                         "input_sha256": sha256(dest), "image_dir": str(image_dir), "frames": len(source),
                         "duration_seconds": duration, "duration_source": "frozen_window_metadata",
                         "truth_count": count, "truth_rr_bpm": 60 * count / duration, "legacy_truth_rr_bpm": old_rr,
                         "include_primary": primary, "reference_reliability": reliability,
                         "historical_predicted_count": expected, "cluster_id": cluster,
                         "cluster_basis": "raw_video_not_verified_cow", "raw_source_path": raw,
                         "exposure": "development", "roi_radius": 20})
    write_csv(out / "matched_windows.csv", rows)
    for path in [DATA / "al_images" / "paper_repro_summary.csv", DATA / "al_images" / "paper_repro_metrics.csv",
                 ASSETS / "lindian_anchored30_motion_robust_rr_predictions.csv", ASSETS / "lindian_anchored30_quality_gated_annotation_template.csv",
                 DATA / "video" / "temperature_curves.csv"]:
        dest = out / "historical_sources" / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        files.append({"role": "historical_snapshot", "source": str(path), "source_sha256": sha256(path), "snapshot": str(dest), "sha256": sha256(dest)})
    write_csv(out / "input_hashes.csv", files)
    return pd.DataFrame(rows)


def reference_forms(out: Path, inventory_path: Path):
    out.mkdir(parents=True, exist_ok=True)
    source_path = ASSETS / "lindian_anchored30_quality_gated_annotation_template.csv"
    try:
        source = pd.read_csv(source_path, encoding="utf-8-sig", dtype={"video_id": str})
    except UnicodeDecodeError:
        source = pd.read_csv(source_path, encoding="gb18030", dtype={"video_id": str})
    p49 = pd.read_csv(ASSETS / "lindian_anchored30_motion_robust_rr_predictions.csv", dtype={"video_id": str}).set_index("video_id")
    rows = []
    for row in source.itertuples():
        p = p49.loc[row.video_id]
        rows.append({"window_id": row.video_id + "_anchored_30s", "video_id": row.video_id,
                     "video_path": str(DATA / "lindian_anchored30_videos" / f"{row.video_id}_anchored30s.mp4"),
                     "source_path": p.raw_source_path, "source_start_seconds": p.window_start_seconds,
                     "duration_seconds": p.manual_window_seconds, "annotation_round": "R1",
                     "annotator": "", "annotation_status": "pending", "manual_breath_count": "",
                     "event_definition": "expiration_peak", "predictions_hidden": "", "reference_notes": "",
                     "analysis_role": "internal_event_validation_not_external_test"})
    write_csv(out / "annotation_windows.csv", rows)
    write_csv(out / "reference_events.csv", [], columns=["window_id", "annotation_round", "event_id", "event_time_seconds", "event_start_seconds", "event_end_seconds", "event_type", "confidence", "annotator", "notes"])
    write_csv(out / "unobservable_intervals.csv", [], columns=["window_id", "annotation_round", "start_seconds", "end_seconds", "reason", "annotator"])
    shutil.copy2(inventory_path / "rgb_thermal_pair_candidates.csv", out / "rgb_thermal_pair_candidates.csv")
    known = [
        {"video_id": "170333", "frame_index_zero_based": 63, "user_evidence": "suspected_false_peak", "whole_window_count_user": 19, "status": "diagnostic_not_reference_event_annotation"},
        {"video_id": "ns210947", "frame_index_zero_based": 82, "user_evidence": "true_breath_keep", "whole_window_count_user": 21, "status": "diagnostic_not_reference_event_annotation"},
        {"video_id": "16170075", "frame_index_zero_based": "", "user_evidence": "reviewed_correct", "whole_window_count_user": 27, "status": "diagnostic_not_reference_event_annotation"},
        {"video_id": "zs197000", "frame_index_zero_based": "", "user_evidence": "reviewed_correct", "whole_window_count_user": 25, "status": "diagnostic_not_reference_event_annotation"},
    ]
    write_csv(out / "historical_case_notes_not_blind.csv", known)


def method_snapshot(out: Path):
    sys.path.insert(0, str(REPO / "scripts"))
    from dataclasses import asdict
    from run_lindian_robust_rr import robust_config
    config = asdict(robust_config("frozen30_candidate"))
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    # Duration metadata is supplied by the video window; reference counts are never inference inputs.
    config.update(truth_csv=None, use_truth_duration_for_rr=False, limit_to_truth_duration=False, overwrite=False)
    configs = {"snapshot_role": "existing_30s_method_candidate_not_selected_by_new_test", "signal_config": config,
               "roi_policy": {"base_radius": 20, "min_radius": 16, "max_radius": 24, "spacing_cv_threshold": .06,
                              "linear_exponent": 1., "damped_exponent": .5},
               "test_window_seconds": 30., "sampling_rule": "first_30_seconds", "reference_input_allowed": False,
               "endpoint_rule": "no_completion_in_current_30s_policy",
               "eligibility": "requires_independent_candidate_review_and_full_hashes_before_release"}
    write_json(out / "method_config.json", configs)
    files = []
    source_paths = [REPO / "scripts" / n for n in ("paper_repro_rr.py", "run_lindian_robust_rr.py", "run_lindian_motion_robust_rr.py", "evaluate_lindian_adaptive_roi.py", "evaluate_paper73_adaptive_roi.py", "evaluate_lindian49_butterworth_spectral.py")]
    source_paths += [REPO / "temperature_extraction" / "getRandomForestRegress" / "clf_model_RGB_20240906.pkl", REPO / "runs" / "pose" / "YOLO11n_cow_pose" / "weights" / "YOLO11n-best.pt", REPO / "runs" / "pose" / "YOLOv8n_cow_pose-21" / "weights" / "YOLO8n-best.pt", REPO / "runs" / "pose" / "YOLO11n_cow_pose" / "args.yaml", REPO / "dataset.yaml"]
    for path in source_paths:
        dest = out / ("scripts" if path.suffix == ".py" else "weights" if path.suffix in {".pt", ".pkl"} else "training_provenance") / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files.append({"source": str(path), "snapshot_relative_path": str(dest.relative_to(out)), "bytes": dest.stat().st_size, "sha256": sha256(dest)})
    files.append({"source": "generated_explicit_config", "snapshot_relative_path": "method_config.json", "bytes": (out / "method_config.json").stat().st_size, "sha256": sha256(out / "method_config.json")})
    write_json(out / "freeze_manifest.json", {"created_at": datetime.now().astimezone().isoformat(), "status": "METHOD_SNAPSHOT_ONLY_TEST_NOT_RELEASED", "files": files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=stamp())
    parser.add_argument("--resume-after-inputs", action="store_true",
                        help="Reuse an existing completed inventory/input snapshot; never overwrite it")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unsafe run id")
    out = HOLDOUT / "inventories" / args.run_id
    inputs = ABLATION / "input_snapshots" / args.run_id
    if args.resume_after_inputs:
        import json
        summary = json.loads((out / "inventory_summary.json").read_text(encoding="utf-8"))
        matched = read_csv(inputs / "matched_windows.csv")
        for item in read_csv(inputs / "input_hashes.csv").itertuples():
            if sha256(Path(item.snapshot)) != item.sha256:
                raise ValueError("Existing input snapshot changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        table, summary = inventory(out)
        matched = preserve_inputs(inputs)
    forms = REFERENCE / "annotations" / args.run_id
    reference_forms(forms, out)
    method = HOLDOUT / "method_snapshots" / args.run_id
    method_snapshot(method)
    write_json(ROOT / f"package_{args.run_id}.json", {"inventory": str(out), "matched_inputs": str(inputs), "reference_forms": str(forms), "method_snapshot": str(method), "created_at": datetime.now().astimezone().isoformat()})
    print(summary, flush=True)
    print(f"Matched windows: {len(matched)}; package id: {args.run_id}", flush=True)


if __name__ == "__main__":
    main()
