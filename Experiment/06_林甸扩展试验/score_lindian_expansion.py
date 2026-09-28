"""Score a frozen Lindian annotation export on decoded original video frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "Experiment" / "05_方法核证"))
sys.path.insert(0, str(ROOT / "Experiment" / "03_可靠事件参考"))
import paper_repro_rr as rr
from reference_tools import match_events
from run import ARMS, run_one

HERE = Path(__file__).resolve().parent
BATCH = HERE / "20260924_expansion_v2" / "annotation_batch_v2"
MODEL = ROOT / "models" / "best.pt"
RF = ROOT / "temperature_extraction" / "getRandomForestRegress" / "clf_model_RGB_20240906.pkl"
REFERENCE_NAMES = {
    "windows": "lindian_expansion_annotation_windows.csv",
    "events": "lindian_expansion_reference_events.csv",
    "intervals": "lindian_expansion_unobservable_intervals.csv",
    "backup": "lindian_expansion_event_backup.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def validate(annotations: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[Path]]:
    files = {key: annotations / name for key, name in REFERENCE_NAMES.items()}
    files.update(segments=BATCH / "segments.csv", frame_map=BATCH / "frame_time_map.csv",
                 model=MODEL, rf=RF, script=Path(__file__), rules=ROOT / "Experiment" / "05_方法核证" / "run.py",
                 extractor=ROOT / "scripts" / "paper_repro_rr.py",
                 matcher=ROOT / "Experiment" / "03_可靠事件参考" / "reference_tools.py")
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    windows = pd.read_csv(files["windows"], encoding="utf-8-sig", dtype={"video_id": str})
    events = pd.read_csv(files["events"], encoding="utf-8-sig")
    intervals = pd.read_csv(files["intervals"], encoding="utf-8-sig")
    segments = pd.read_csv(files["segments"], encoding="utf-8-sig", dtype={"video_id": str})
    frame_map = pd.read_csv(files["frame_map"], encoding="utf-8-sig", dtype={"video_id": str})
    if (len(windows) != 8 or len(segments) != 8 or windows.window_id.duplicated().any()
            or segments.window_id.duplicated().any()
            or set(windows.window_id) != set(segments.window_id)
            or events.duplicated(["window_id", "event_id"]).any()
            or frame_map.duplicated(["video_id", "source_frame"]).any()):
        raise ValueError("Unexpected or duplicate window, event, or frame identifiers")
    if set(events.window_id) - set(windows.window_id) or set(intervals.window_id) - set(windows.window_id):
        raise ValueError("Reference records contain an unknown window")
    if not windows.annotation_status.isin(["complete", "partial", "unobservable"]).all():
        raise ValueError("Unknown annotation status")
    if not windows.predictions_hidden.astype(str).str.lower().eq("true").all():
        raise ValueError("Annotation export does not declare predictions hidden")
    if not windows.event_definition.eq("expiration_peak").all():
        raise ValueError("Event definition changed")
    records = {}
    for win in windows.itertuples(index=False):
        seg = segments.set_index("window_id").loc[win.window_id]
        mapping = frame_map[frame_map.video_id.eq(win.video_id)].sort_values("output_frame")
        expected = np.arange(int(seg.first_source_frame), int(seg.stop_source_frame_exclusive))
        if (win.video_id != seg.video_id or Path(win.source_path) != Path(seg.source_path)
                or float(win.source_start_seconds) != float(seg.source_start_seconds)
                or float(win.duration_seconds) != 30.0 or len(mapping) != int(seg.source_frame_count)
                or not np.array_equal(mapping.source_frame.to_numpy(int), expected)
                or not np.array_equal(mapping.output_frame.to_numpy(int), np.arange(len(mapping)))
                or not np.isfinite(mapping.playback_seconds).all()
                or not np.all(np.diff(mapping.playback_seconds) > 0)
                or not np.all(np.diff(mapping.source_pts_seconds) > 0)
                or mapping.playback_seconds.iloc[0] < 0 or mapping.playback_seconds.iloc[-1] >= 30):
            raise ValueError(f"Invalid source-to-playback frame binding: {win.window_id}")
        source = Path(seg.source_path)
        if not source.is_file() or digest(source) != str(seg.source_sha256):
            raise ValueError(f"Source file missing or changed: {source}")
        selected = events[events.window_id.eq(win.window_id)].sort_values("event_time_seconds")
        if (not selected.event_time_seconds.between(0, 30, inclusive="left").all()
                or selected.event_time_seconds.duplicated().any()):
            raise ValueError(f"Invalid reference event times: {win.window_id}")
        unavailable = intervals[intervals.window_id.eq(win.window_id)]
        if win.annotation_status == "complete":
            if (pd.isna(win.manual_breath_count) or len(selected) != int(win.manual_breath_count)
                    or not selected.confidence.eq("confirmed").all() or not unavailable.empty):
                raise ValueError(f"Invalid complete reference: {win.window_id}")
        elif pd.notna(win.manual_breath_count):
            raise ValueError(f"Incomplete window has a count: {win.window_id}")
        records[win.window_id] = (seg, mapping, selected)
    return windows, segments, records, list(files.values()) + [Path(p) for p in segments.source_path]


def extract(source: Path, seg: pd.Series, mapping: pd.DataFrame, detector: YOLO, rf_model: object) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {source}")
    first, stop = int(seg.first_source_frame), int(seg.stop_source_frame_exclusive)
    rows = []
    try:
        for frame_id in range(stop):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source decode failed at frame {frame_id}: {source}")
            if frame_id < first:
                continue
            binding = mapping.iloc[frame_id - first]
            result = detector(frame, verbose=False)[0]
            detection = rr.select_detection(result)
            values = {f"{side}_{key}": np.nan for side in ("left", "right")
                      for key in ("temp", "x", "y", "conf")}
            values.update(left_source="missing", right_source="missing")
            if detection is not None:
                points = result.keypoints.data[detection].detach().cpu().numpy()
                for index, side in enumerate(("left", "right")):
                    if len(points) <= index:
                        continue
                    x, y, conf = rr.keypoint_xy_conf(points[index])
                    values.update({f"{side}_x": x, f"{side}_y": y, f"{side}_conf": conf})
                    if conf >= 0.5:
                        temp = rr.circle_temperature(frame, rf_model, x, y, 20, 20.0)
                        values[f"{side}_temp"] = temp
                        values[f"{side}_source"] = "detected" if np.isfinite(temp) else "missing"
            rows.append({"frame_name": f"source_{frame_id:06d}", "source_frame": frame_id,
                         "source_pts_seconds": binding.source_pts_seconds,
                         "playback_seconds": binding.playback_seconds, **values})
    finally:
        capture.release()
    table = pd.DataFrame(rows)
    if len(table) != len(mapping) or not np.array_equal(table.source_frame, mapping.source_frame):
        raise ValueError(f"Decoded frame binding changed: {source}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    windows, segments, records, files = validate(args.annotations)
    print(windows.annotation_status.value_counts().to_string(), flush=True)
    print(f"Confirmed complete events: {sum(len(records[w.window_id][2]) for w in windows.itertuples() if w.annotation_status == 'complete')}", flush=True)
    if args.validate_only:
        return
    if args.output is None or args.output.exists():
        raise FileExistsError("Specify a fresh --output directory")
    args.output.mkdir(parents=True)
    (args.output / "INCOMPLETE.txt").write_text("Scoring has not completed.\n", encoding="utf-8")
    manifest = {"scope": "Lindian same-farm development batch; not independent validation",
                "round": "LX-20260924-P1", "arms": ARMS, "roi_radius": 20,
                "min_temperature": 20.0, "keypoint_confidence": 0.5,
                "event_tolerance_seconds": 0.30, "duration_seconds": 30,
                "input": "original MP4 BGR frames mapped to annotation playback time",
                "files": [{"path": str(p), "sha256": digest(p)} for p in files]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(args.output / "reference_disposition.csv", windows)
    detector = YOLO(str(MODEL))
    rf_model = joblib.load(RF)
    predictions, pred_events, matches = [], [], []
    for win in windows.itertuples(index=False):
        seg, mapping, actual = records[win.window_id]
        table = extract(Path(seg.source_path), seg, mapping, detector, rf_model)
        save_csv(args.output / "temperatures" / f"{win.video_id}.csv", table)
        times = table.playback_seconds.to_numpy(float)
        truth = actual.event_time_seconds.to_numpy(float) if win.annotation_status == "complete" else None
        for arm in ARMS:
            curve, peaks, fusion, prominence = run_one(table, arm)
            save_csv(args.output / "curves" / arm / f"{win.video_id}.csv", curve)
            count = len(peaks)
            row = {"video_id": win.video_id, "window_id": win.window_id, "arm": arm,
                   "reference_status": win.annotation_status, "truth_count": int(win.manual_breath_count) if truth is not None else np.nan,
                   "predicted_count": count, "truth_rr_bpm": 2 * int(win.manual_breath_count) if truth is not None else np.nan,
                   "predicted_rr_bpm": 2 * count, "absolute_rr_error_bpm": np.nan,
                   "exact_count": pd.NA, "tp": np.nan, "fp": np.nan, "fn": np.nan,
                   "event_f1": np.nan, "fusion": fusion, "prominence": prominence,
                   "left_valid": int(table.left_temp.notna().sum()),
                   "right_valid": int(table.right_temp.notna().sum())}
            for peak in peaks:
                pred_events.append({"video_id": win.video_id, "arm": arm, "frame_index": int(peak),
                                    "source_frame": int(table.iloc[peak].source_frame),
                                    "playback_seconds": float(times[peak])})
            if truth is not None:
                paired, false_pos, false_neg = match_events(times[peaks], truth, 0.30)
                tp, fp, fn = len(paired), len(false_pos), len(false_neg)
                row.update(absolute_rr_error_bpm=2 * abs(count - len(truth)), exact_count=count == len(truth),
                           tp=tp, fp=fp, fn=fn, event_f1=2 * tp / (2 * tp + fp + fn))
                for i, j, delta in paired:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "TP",
                                    "predicted_time": float(times[peaks[i]]), "reference_time": float(truth[j]),
                                    "time_error_seconds": delta})
                for i in false_pos:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FP",
                                    "predicted_time": float(times[peaks[i]]), "reference_time": np.nan,
                                    "time_error_seconds": np.nan})
                for j in false_neg:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FN",
                                    "predicted_time": np.nan, "reference_time": float(truth[j]),
                                    "time_error_seconds": np.nan})
            predictions.append(row)
        print(f"Scored {win.video_id}: {len(table)} original frames; {win.annotation_status}", flush=True)
    save_csv(args.output / "predictions.csv", predictions)
    save_csv(args.output / "predicted_events.csv", pred_events)
    save_csv(args.output / "event_matches.csv", matches)
    frame = pd.DataFrame(predictions)
    summary = []
    for arm in ARMS:
        group = frame[frame.arm.eq(arm) & frame.reference_status.eq("complete")]
        tp, fp, fn = (int(group[key].sum()) for key in ("tp", "fp", "fn"))
        observed, predicted = group.truth_rr_bpm.to_numpy(float), group.predicted_rr_bpm.to_numpy(float)
        denominator = np.sum((observed - observed.mean()) ** 2)
        summary.append({"arm": arm, "selected_windows": len(windows), "reference_complete": len(group),
                        "reference_unobservable_or_partial": len(windows) - len(group),
                        "mae_bpm": group.absolute_rr_error_bpm.mean(),
                        "rr_r2": 1 - np.sum((predicted - observed) ** 2) / denominator if denominator > 0 else np.nan,
                        "exact_count": int(group.exact_count.sum()), "tp": tp, "fp": fp, "fn": fn,
                        "event_precision": tp / (tp + fp) if tp + fp else 0,
                        "event_recall": tp / (tp + fn) if tp + fn else 0,
                        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    save_csv(args.output / "summary.csv", summary)
    (args.output / "INCOMPLETE.txt").unlink()
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
