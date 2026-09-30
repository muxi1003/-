"""Score the frozen P3 four arms using separately audited RR and event references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from score_lindian_expansion import ARMS, MODEL, RF, digest, extract, match_events, run_one, save_csv

HERE = Path(__file__).resolve().parent
P3 = HERE / "20260928_expansion_p3_v2"
PROTOCOL = HERE / "20260928_p3_pre_score_protocol_v2.md"
REFERENCE_NAMES = {
    "windows": "lindian_expansion_annotation_windows.csv",
    "events": "lindian_expansion_reference_events.csv",
    "intervals": "lindian_expansion_unobservable_intervals.csv",
    "backup": "lindian_expansion_event_backup.json",
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"video_id": str, "window_id": str})


def validate(reference: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[Path]]:
    audit_file = reference / "audit" / "window_audit.csv"
    summary_file = reference / "audit" / "summary.json"
    claims_file = reference / "audit" / "count_only_claims.csv"
    adjudication_file = reference / "adjudication.json"
    for path in (audit_file, summary_file, claims_file, adjudication_file, PROTOCOL, MODEL, RF):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = read(audit_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    claims = read(claims_file)
    adjudication = json.loads(adjudication_file.read_text(encoding="utf-8"))
    if (len(audit) != 53 or audit.window_id.duplicated().any()
            or audit.analysis_disposition.value_counts().to_dict() != {
                "complete_primary": 24, "not_primary": 27, "count_only_claim_excluded_primary": 2}
            or summary["export_status_counts"] != {"complete": 25, "unobservable": 10, "partial": 18}
            or len(claims) != 2 or not claims.user_reported_30s_count.eq(36).all()
            or not claims.frozen_primary_rr.eq("excluded").all()):
        raise ValueError("P3 reference eligibility or count-only claims changed")
    if adjudication["change"]["window_id"] != "20230808T164507n211023_000_030_source30s":
        raise ValueError("P3 adjudication identity changed")
    for entry in summary["raw_export_sha256"]:
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Audited reference changed: {entry['path']}")
    for key, expected in adjudication["output_files_sha256"].items():
        batch, name = key.split("/", 1)
        if digest(reference / f"raw_{batch}" / name) != expected:
            raise ValueError(f"Adjudicated export changed: {key}")

    windows_all, records, paths = [], {}, [audit_file, summary_file, claims_file, adjudication_file,
                                              PROTOCOL, MODEL, RF, Path(__file__),
                                              HERE / "score_lindian_expansion.py",
                                              HERE.parent / "05_方法核证" / "run.py",
                                              HERE.parent / "03_可靠事件参考" / "reference_tools.py"]
    for batch, expected_size in (("A", 27), ("B", 26)):
        raw = reference / f"raw_{batch}"
        batch_dir = P3 / f"annotation_batch_{batch}"
        files = {key: raw / name for key, name in REFERENCE_NAMES.items()}
        files.update(segments=batch_dir / "segments.csv", mapping=batch_dir / "frame_time_map.csv",
                     selection=P3 / f"selected_sources_{batch}.csv", manifest=batch_dir / "manifest.json")
        if any(not path.is_file() for path in files.values()):
            raise FileNotFoundError([str(path) for path in files.values() if not path.is_file()])
        manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
        if (manifest["selection_hash"] != digest(files["selection"])
                or manifest["protocol_hash"] != digest(PROTOCOL) or manifest["clips"] != expected_size):
            raise ValueError(f"P3 {batch} batch freeze changed")
        windows, events, intervals = (read(files[key]) for key in ("windows", "events", "intervals"))
        segments, mapping = read(files["segments"]), read(files["mapping"])
        if (len(windows) != expected_size or len(segments) != expected_size
                or windows.window_id.duplicated().any() or segments.window_id.duplicated().any()
                or set(windows.window_id) != set(segments.window_id)
                or set(events.window_id) - set(windows.window_id)
                or set(intervals.window_id) - set(windows.window_id)
                or not windows.annotation_round.eq(f"LX-20260928-P3{batch}").all()
                or not windows.predictions_hidden.astype(str).str.lower().eq("true").all()):
            raise ValueError(f"P3 {batch} export membership/round changed")
        windows = windows.merge(audit[["window_id", "analysis_disposition"]],
                                on="window_id", validate="one_to_one")
        segment_by_id = segments.set_index("window_id")
        for win in windows.itertuples(index=False):
            seg = segment_by_id.loc[win.window_id]
            rows = mapping[mapping.video_id.eq(win.video_id)].sort_values("output_frame")
            expected_frames = np.arange(int(seg.first_source_frame), int(seg.stop_source_frame_exclusive))
            if (win.video_id != seg.video_id or Path(win.source_path) != Path(seg.source_path)
                    or float(win.source_start_seconds) != float(seg.source_start_seconds)
                    or float(win.duration_seconds) != 30 or len(rows) != int(seg.source_frame_count)
                    or not np.array_equal(rows.output_frame.to_numpy(int), np.arange(len(rows)))
                    or not np.array_equal(rows.source_frame.to_numpy(int), expected_frames)
                    or not np.all(np.diff(rows.playback_seconds.to_numpy(float)) > 0)
                    or not np.all(np.diff(rows.source_pts_seconds.to_numpy(float)) > 0)
                    or rows.playback_seconds.iloc[0] < 0 or rows.playback_seconds.iloc[-1] >= 30):
                raise ValueError(f"P3 frame/time binding changed: {win.window_id}")
            refs = events[events.window_id.eq(win.window_id)].sort_values("event_time_seconds")
            spans = intervals[intervals.window_id.eq(win.window_id)]
            if win.analysis_disposition == "complete_primary":
                if (pd.isna(win.manual_breath_count) or len(refs) != int(win.manual_breath_count)
                        or not refs.confidence.eq("confirmed").all() or not spans.empty):
                    raise ValueError(f"Invalid P3 complete reference: {win.window_id}")
            elif win.analysis_disposition == "count_only_claim_excluded_primary":
                if win.video_id not in set(claims.video_id):
                    raise ValueError(f"Unknown count-only claim: {win.video_id}")
            source = Path(seg.source_path)
            if not source.is_file() or digest(source) != str(seg.source_sha256):
                raise ValueError(f"P3 source missing or changed: {source}")
            records[win.window_id] = (seg, rows, refs)
        windows_all.append(windows)
        paths.extend(files.values())
        paths.extend(Path(value) for value in segments.source_path)
    windows = pd.concat(windows_all, ignore_index=True)
    if len(windows) != 53 or windows.window_id.duplicated().any():
        raise ValueError("P3 combined membership changed")
    return windows, records, claims, paths


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = predictions[predictions.analysis_disposition.eq("complete_primary")]
    count_only = predictions[predictions.analysis_disposition.eq("count_only_claim_excluded_primary")]
    summaries = []
    for arm in ARMS:
        group = main[main.arm.eq(arm)]
        truth, predicted = group.truth_rr_bpm.to_numpy(float), group.predicted_rr_bpm.to_numpy(float)
        denominator = np.sum((truth - truth.mean()) ** 2)
        tp, fp, fn = (int(group[key].sum()) for key in ("tp", "fp", "fn"))
        summaries.append({"arm": arm, "selected_sources": 53, "main_complete_windows": len(group),
                          "count_only_not_main": 2, "reference_not_main_other": 27,
                          "main_rr_mae_bpm": float(group.absolute_rr_error_bpm.mean()),
                          "main_rr_r2": 1 - np.sum((truth - predicted) ** 2) / denominator if denominator > 0 else np.nan,
                          "main_exact_count": int(group.exact_count.sum()),
                          "event_tp": tp, "event_fp": fp, "event_fn": fn,
                          "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    base = main[main.arm.eq("C0P0")].set_index("video_id")
    paired = []
    for arm in ARMS[1:]:
        other = main[main.arm.eq(arm)].set_index("video_id").loc[base.index]
        differences = (other.absolute_rr_error_bpm - base.absolute_rr_error_bpm).to_numpy(float)
        rng = np.random.default_rng(20260928)
        draws = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        paired.append({"candidate": arm, "control": "C0P0", "main_complete_windows": len(differences),
                       "cluster_unit": "one source per filename cow; farm identity not independently verified",
                       "delta_mae_bpm": float(differences.mean()), "ci95_low": float(low),
                       "ci95_high": float(high), "bootstrap_draws": 10000, "seed": 20260928})
    return pd.DataFrame(summaries), pd.DataFrame(paired), count_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    reference = args.reference.resolve(strict=True)
    windows, records, claims, paths = validate(reference)
    print(windows.analysis_disposition.value_counts().to_string(), flush=True)
    if args.validate_only:
        return
    if args.output is None or args.output.exists():
        raise FileExistsError("Specify a fresh immutable --output directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "INCOMPLETE.txt").write_text("P3 scoring in progress.\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "scope": "P3 same-farm fixed-source evaluation; 24 frozen complete references",
        "arms": ARMS, "duration_seconds": 30, "roi_radius": 20, "keypoint_confidence": 0.5,
        "event_tolerance_seconds": 0.30, "count_only_36": "post-hoc sensitivity, excluded from main/F1",
        "files": [{"path": str(path), "sha256": digest(path)} for path in dict.fromkeys(paths)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output / "reference_disposition.csv", windows)
    detector, rf_model = YOLO(str(MODEL)), joblib.load(RF)
    predictions, predicted_events, matches = [], [], []
    claims_by_video = claims.set_index("video_id")
    for win in windows.itertuples(index=False):
        seg, mapping, refs = records[win.window_id]
        table = extract(Path(seg.source_path), seg, mapping, detector, rf_model)
        save_csv(output / "temperatures" / f"{win.video_id}.csv", table)
        times = table.playback_seconds.to_numpy(float)
        complete = win.analysis_disposition == "complete_primary"
        count_only = win.analysis_disposition == "count_only_claim_excluded_primary"
        truth_count = int(win.manual_breath_count) if complete else (
            int(claims_by_video.loc[win.video_id].user_reported_30s_count) if count_only else None)
        truth_times = refs.event_time_seconds.to_numpy(float) if complete else None
        for arm in ARMS:
            curve, peaks, fusion, prominence = run_one(table, arm)
            save_csv(output / "curves" / arm / f"{win.video_id}.csv", curve)
            predicted_count = len(peaks)
            row = {"video_id": win.video_id, "window_id": win.window_id, "arm": arm,
                   "export_status": win.annotation_status, "analysis_disposition": win.analysis_disposition,
                   "truth_count": truth_count if truth_count is not None else np.nan,
                   "predicted_count": predicted_count,
                   "truth_rr_bpm": 2 * truth_count if truth_count is not None else np.nan,
                   "predicted_rr_bpm": 2 * predicted_count,
                   "absolute_rr_error_bpm": 2 * abs(predicted_count - truth_count)
                   if truth_count is not None else np.nan,
                   "exact_count": predicted_count == truth_count if truth_count is not None else pd.NA,
                   "tp": np.nan, "fp": np.nan, "fn": np.nan, "event_f1": np.nan,
                   "fusion": fusion, "prominence": prominence,
                   "left_valid": int(table.left_temp.notna().sum()),
                   "right_valid": int(table.right_temp.notna().sum())}
            for peak in peaks:
                predicted_events.append({"video_id": win.video_id, "arm": arm,
                                         "frame_index": int(peak),
                                         "source_frame": int(table.iloc[peak].source_frame),
                                         "playback_seconds": float(times[peak])})
            if complete:
                hits, fp, fn = match_events(times[peaks], truth_times, 0.30)
                tp = len(hits)
                row.update(tp=tp, fp=len(fp), fn=len(fn),
                           event_f1=2 * tp / (2 * tp + len(fp) + len(fn)))
                for i, j, delta in hits:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "TP",
                                    "predicted_time": float(times[peaks[i]]),
                                    "reference_time": float(truth_times[j]),
                                    "reference_event_id": refs.iloc[j].event_id,
                                    "time_error_seconds": delta})
                for i in fp:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FP",
                                    "predicted_time": float(times[peaks[i]]), "reference_time": np.nan,
                                    "reference_event_id": "", "time_error_seconds": np.nan})
                for j in fn:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FN",
                                    "predicted_time": np.nan, "reference_time": float(truth_times[j]),
                                    "reference_event_id": refs.iloc[j].event_id,
                                    "time_error_seconds": np.nan})
            predictions.append(row)
        print(f"Scored {win.video_id}: {len(table)} original frames; {win.analysis_disposition}", flush=True)
    frame = pd.DataFrame(predictions)
    save_csv(output / "predictions.csv", frame)
    save_csv(output / "predicted_events.csv", predicted_events)
    save_csv(output / "event_matches.csv", matches)
    summary, paired, count_only = summarize(frame)
    save_csv(output / "summary_main.csv", summary)
    save_csv(output / "paired_intervals_main.csv", paired)
    save_csv(output / "count_only_sensitivity.csv", count_only)
    (output / "INCOMPLETE.txt").unlink()
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
