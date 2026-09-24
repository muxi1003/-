"""Score three annotated 190588 source-time clips without changing frozen runs."""
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

SELECTED = ("000_030", "030_060", "180_210")
PILOT = ROOT / "Experiment" / "06_林甸扩展试验" / "190588_source30_v1"
MODEL = ROOT / "models" / "best.pt"
RF = ROOT / "temperature_extraction" / "getRandomForestRegress" / "clf_model_RGB_20240906.pkl"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    reference_files = {
        "windows": args.annotations / "lindian190588_annotation_windows.csv",
        "events": args.annotations / "lindian190588_reference_events.csv",
        "intervals": args.annotations / "lindian190588_unobservable_intervals.csv",
    }
    windows = pd.read_csv(reference_files["windows"], encoding="utf-8-sig", dtype={"video_id": str})
    events = pd.read_csv(reference_files["events"], encoding="utf-8-sig")
    intervals = pd.read_csv(reference_files["intervals"], encoding="utf-8-sig")
    segments = pd.read_csv(PILOT / "segments.csv")
    frame_map = pd.read_csv(PILOT / "frame_time_map.csv")
    selected = segments[segments.segment_id.isin(SELECTED)].set_index("segment_id").loc[list(SELECTED)]
    source = Path(selected.iloc[0].source_path)
    if selected.source_path.nunique() != 1 or any(selected.status.ne("CUT")):
        raise ValueError("Selected source windows changed")
    if digest(source) != str(selected.iloc[0].source_sha256):
        raise ValueError("Source video digest changed")
    if not windows.window_id.is_unique or events.duplicated(["window_id", "event_id"]).any():
        raise ValueError("Duplicate reference identifiers")
    records = {}
    for segment in SELECTED:
        ref = windows.set_index("video_id").loc[f"190588_{segment}"]
        matched = events[events.window_id.eq(ref.window_id)].sort_values("event_time_seconds")
        unavailable = intervals[intervals.window_id.eq(ref.window_id)]
        if (ref.annotation_status != "complete" or not bool(ref.predictions_hidden)
                or ref.event_definition != "expiration_peak" or not unavailable.empty
                or len(matched) != int(ref.manual_breath_count)
                or not matched.confidence.eq("confirmed").all()
                or not matched.event_time_seconds.between(0, 30, inclusive="left").all()
                or float(ref.duration_seconds) != 30):
            raise ValueError(f"Invalid complete reference: {segment}")
        mapping = frame_map[frame_map.segment_id.eq(segment)].sort_values("output_frame")
        seg = selected.loc[segment]
        expected = np.arange(int(seg.first_source_frame), int(seg.stop_source_frame_exclusive))
        if not np.array_equal(mapping.source_frame.to_numpy(), expected):
            raise ValueError(f"Noncontiguous frame binding: {segment}")
        records[segment] = (ref, matched, mapping)

    args.output.mkdir(parents=True)
    manifest_paths = [*reference_files.values(), PILOT / "segments.csv", PILOT / "frame_time_map.csv",
                      source, MODEL, RF, Path(__file__), ROOT / "Experiment" / "05_方法核证" / "run.py",
                      ROOT / "scripts" / "paper_repro_rr.py"]
    (args.output / "manifest.json").write_text(json.dumps({
        "scope": "190588 same-source development pilot; three windows are not independent cows",
        "selected": SELECTED, "arms": ARMS, "radius": 20, "min_temp": 20.0,
        "keypoint_confidence": 0.5, "event_tolerance_seconds": 0.30,
        "duration_seconds": 30, "input": "decoded original MP4 BGR frames; no viewing-copy pixels",
        "files": [{"path": str(p), "sha256": digest(p)} for p in manifest_paths],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    detector = YOLO(str(MODEL))
    temperature_model = joblib.load(RF)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {source}")
    wanted = set().union(*(set(rec[2].source_frame.astype(int)) for rec in records.values()))
    last = max(wanted)
    rows = {segment: [] for segment in SELECTED}
    indexed = {int(row.source_frame): (segment, row)
               for segment in SELECTED for row in records[segment][2].itertuples(index=False)}
    try:
        for frame_id in range(last + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Original source decode failed at frame {frame_id}")
            if frame_id not in wanted:
                continue
            segment, binding = indexed[frame_id]
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
                        temp = rr.circle_temperature(frame, temperature_model, x, y, 20, 20.0)
                        values[f"{side}_temp"] = temp
                        values[f"{side}_source"] = "detected" if np.isfinite(temp) else "missing"
            rows[segment].append({"frame_name": f"source_{frame_id:06d}",
                                  "source_frame": frame_id, "source_pts_seconds": binding.source_pts_seconds,
                                  "playback_seconds": binding.playback_seconds, **values})
            if len(rows[segment]) % 50 == 0:
                print(f"{segment}: {len(rows[segment])}/{len(records[segment][2])} frames", flush=True)
    finally:
        capture.release()

    predictions, pred_events, matches = [], [], []
    for segment in SELECTED:
        ref, actual, mapping = records[segment]
        video_id = f"190588_{segment}"
        table = pd.DataFrame(rows[segment])
        if len(table) != len(mapping) or not np.array_equal(table.source_frame, mapping.source_frame):
            raise ValueError(f"Decoded frame binding mismatch: {segment}")
        save_csv(args.output / f"temperature_{segment}.csv", table)
        times = table.playback_seconds.to_numpy(float)
        truth = actual.event_time_seconds.to_numpy(float)
        for arm in ARMS:
            curve, peaks, fusion, prominence = run_one(table, arm)
            save_csv(args.output / f"curve_{segment}_{arm}.csv", curve)
            matched, false_pos, false_neg = match_events(times[peaks], truth, 0.30)
            count = len(peaks)
            tp, fp, fn = len(matched), len(false_pos), len(false_neg)
            predictions.append({"video_id": video_id, "arm": arm, "truth_count": int(ref.manual_breath_count),
                                "predicted_count": count, "truth_rr_bpm": 2 * int(ref.manual_breath_count),
                                "predicted_rr_bpm": 2 * count, "absolute_rr_error_bpm": 2 * abs(count - int(ref.manual_breath_count)),
                                "exact_count": count == int(ref.manual_breath_count), "tp": tp, "fp": fp, "fn": fn,
                                "event_f1": 2 * tp / (2 * tp + fp + fn), "fusion": fusion,
                                "prominence": prominence, "left_valid": int(table.left_temp.notna().sum()),
                                "right_valid": int(table.right_temp.notna().sum())})
            for peak in peaks:
                pred_events.append({"video_id": video_id, "arm": arm, "frame_index": int(peak),
                                    "source_frame": int(table.iloc[peak].source_frame), "time_seconds": float(times[peak])})
            for i, j, delta in matched:
                matches.append({"video_id": video_id, "arm": arm, "kind": "TP",
                                "predicted_time": float(times[peaks[i]]), "reference_time": float(truth[j]),
                                "time_error_seconds": delta})
            for i in false_pos:
                matches.append({"video_id": video_id, "arm": arm, "kind": "FP",
                                "predicted_time": float(times[peaks[i]]), "reference_time": np.nan, "time_error_seconds": np.nan})
            for j in false_neg:
                matches.append({"video_id": video_id, "arm": arm, "kind": "FN",
                                "predicted_time": np.nan, "reference_time": float(truth[j]), "time_error_seconds": np.nan})
    save_csv(args.output / "predictions.csv", predictions)
    save_csv(args.output / "predicted_events.csv", pred_events)
    save_csv(args.output / "event_matches.csv", matches)
    frame = pd.DataFrame(predictions)
    summary = []
    for arm, group in frame.groupby("arm", sort=False):
        tp, fp, fn = (int(group[k].sum()) for k in ("tp", "fp", "fn"))
        summary.append({"arm": arm, "windows": len(group), "mae_bpm": group.absolute_rr_error_bpm.mean(),
                        "exact_count": int(group.exact_count.sum()), "tp": tp, "fp": fp, "fn": fn,
                        "event_precision": tp / (tp + fp) if tp + fp else 0,
                        "event_recall": tp / (tp + fn) if tp + fn else 0,
                        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    save_csv(args.output / "summary.csv", summary)
    print(frame.to_string(index=False), flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
