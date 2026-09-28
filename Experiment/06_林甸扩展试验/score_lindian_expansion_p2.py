"""Score the P2 batch after the frozen visibility-reference review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from score_lindian_expansion import ARMS, MODEL, RF, digest, extract, match_events, run_one, save_csv
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
P2 = HERE / "20260928_expansion_p2_v1"
BATCH = P2 / "annotation_batch_v1"
PROTOCOL = HERE / "20260928_p2_rr_mae_protocol.md"
EXCLUSIONS = P2 / "visibility_exclusions.json"
REFERENCE_NAMES = {
    "windows": "lindian_expansion_annotation_windows.csv",
    "events": "lindian_expansion_reference_events.csv",
    "intervals": "lindian_expansion_unobservable_intervals.csv",
    "backup": "lindian_expansion_event_backup.json",
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((key, str(value)) for key, value in row.items())) for row in rows)


def validate(annotations: Path) -> tuple[pd.DataFrame, dict, list[Path]]:
    files = {key: annotations / name for key, name in REFERENCE_NAMES.items()}
    files.update(segments=BATCH / "segments.csv", frame_map=BATCH / "frame_time_map.csv",
                 selection=P2 / "selected_sources.csv", inventory_freeze=P2 / "freeze.json",
                 batch_freeze=BATCH / "manifest.json", protocol=PROTOCOL, exclusions=EXCLUSIONS,
                 model=MODEL, rf=RF, script=Path(__file__), extractor=HERE / "score_lindian_expansion.py",
                 rules=HERE.parent / "05_方法核证" / "run.py",
                 matcher=HERE.parent / "03_可靠事件参考" / "reference_tools.py")
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    backup = json.loads(files["backup"].read_text(encoding="utf-8"))
    for key, csv_key in (("windows", "windows"), ("events", "events"), ("intervals", "intervals")):
        if normalized(backup[key]) != normalized(csv_rows(files[csv_key])):
            raise ValueError(f"JSON backup and CSV differ: {key}")
    inventory_freeze = json.loads(files["inventory_freeze"].read_text(encoding="utf-8"))
    batch_freeze = json.loads(files["batch_freeze"].read_text(encoding="utf-8"))
    if (inventory_freeze["input_hashes"]["protocol"] != digest(PROTOCOL)
            or batch_freeze["protocol_hash"] != digest(PROTOCOL)
            or batch_freeze["selection_hash"] != digest(files["selection"])):
        raise ValueError("P2 protocol or selected-source freeze changed")
    windows = pd.read_csv(files["windows"], encoding="utf-8-sig", dtype={"video_id": str})
    events = pd.read_csv(files["events"], encoding="utf-8-sig")
    intervals = pd.read_csv(files["intervals"], encoding="utf-8-sig")
    segments = pd.read_csv(files["segments"], encoding="utf-8-sig", dtype={"video_id": str})
    frame_map = pd.read_csv(files["frame_map"], encoding="utf-8-sig", dtype={"video_id": str})
    selected = pd.read_csv(files["selection"], encoding="utf-8-sig")
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    excluded = set(exclusions["window_ids"])
    if (len(windows) != 16 or len(segments) != 16 or len(selected) != 16
            or windows.window_id.duplicated().any() or segments.window_id.duplicated().any()
            or set(windows.window_id) != set(segments.window_id)
            or len(excluded) != 4 or not excluded <= set(windows.window_id)
            or events.duplicated(["window_id", "event_id"]).any()
            or frame_map.duplicated(["video_id", "source_frame"]).any()
            or set(events.window_id) - set(windows.window_id)
            or set(intervals.window_id) - set(windows.window_id)):
        raise ValueError("P2 source, reference, or exclusion membership changed")
    if (not windows.annotation_round.eq("LX-20260928-P2").all()
            or not windows.event_definition.eq("expiration_peak").all()
            or not windows.predictions_hidden.astype(str).str.lower().eq("true").all()
            or not events.annotation_round.eq("LX-20260928-P2").all()):
        raise ValueError("P2 annotation round, event definition, or prediction-hidden flag changed")
    records = {}
    statuses = []
    segment_by_id = segments.set_index("window_id")
    for win in windows.itertuples(index=False):
        seg = segment_by_id.loc[win.window_id]
        mapping = frame_map[frame_map.video_id.eq(win.video_id)].sort_values("output_frame")
        expected = np.arange(int(seg.first_source_frame), int(seg.stop_source_frame_exclusive))
        if (win.video_id != seg.video_id or Path(win.source_path) != Path(seg.source_path)
                or float(win.source_start_seconds) != float(seg.source_start_seconds)
                or float(win.duration_seconds) != 30.0 or len(mapping) != int(seg.source_frame_count)
                or not np.array_equal(mapping.output_frame.to_numpy(int), np.arange(len(mapping)))
                or not np.array_equal(mapping.source_frame.to_numpy(int), expected)
                or not np.isfinite(mapping.playback_seconds).all()
                or not np.all(np.diff(mapping.playback_seconds) > 0)
                or not np.all(np.diff(mapping.source_pts_seconds) > 0)
                or mapping.playback_seconds.iloc[0] < 0 or mapping.playback_seconds.iloc[-1] >= 30):
            raise ValueError(f"Invalid P2 source-frame time binding: {win.window_id}")
        source = Path(seg.source_path)
        if not source.is_file() or digest(source) != str(seg.source_sha256):
            raise ValueError(f"P2 source file missing or changed: {source}")
        refs = events[events.window_id.eq(win.window_id)].sort_values("event_time_seconds")
        if (not refs.event_time_seconds.between(0, 30, inclusive="left").all()
                or refs.event_time_seconds.duplicated().any()):
            raise ValueError(f"Invalid P2 event times: {win.window_id}")
        spans = intervals[intervals.window_id.eq(win.window_id)]
        if win.window_id in excluded:
            if win.annotation_status != "complete" or not str(win.reference_notes).strip():
                raise ValueError(f"Visibility exclusion no longer matches export: {win.window_id}")
            analysis_status = "partial_visibility_override"
        elif win.annotation_status == "complete":
            if (pd.isna(win.manual_breath_count) or len(refs) != int(win.manual_breath_count)
                    or not refs.confidence.eq("confirmed").all() or not spans.empty
                    or pd.notna(win.reference_notes)):
                raise ValueError(f"Invalid full-window complete reference: {win.window_id}")
            analysis_status = "complete"
        elif win.annotation_status == "unobservable" and pd.isna(win.manual_breath_count):
            analysis_status = "unobservable"
        else:
            raise ValueError(f"Unexpected P2 reference disposition: {win.window_id}")
        records[win.window_id] = (seg, mapping, refs)
        statuses.append(analysis_status)
    windows["analysis_status"] = statuses
    if windows.analysis_status.value_counts().to_dict() != {
            "complete": 6, "partial_visibility_override": 4, "unobservable": 6}:
        raise ValueError("P2 analysis denominator changed")
    return windows, records, list(files.values()) + [Path(p) for p in segments.source_path]


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = frame[frame.analysis_status.eq("complete")]
    summary = []
    for arm in ARMS:
        group = complete[complete.arm.eq(arm)]
        truth = group.truth_rr_bpm.to_numpy(float)
        pred = group.predicted_rr_bpm.to_numpy(float)
        tp, fp, fn = (int(group[key].sum()) for key in ("tp", "fp", "fn"))
        denominator = np.sum((truth - truth.mean()) ** 2)
        summary.append({"arm": arm, "selected_windows": 16, "complete_reference": len(group),
                        "partial_visibility": 4, "unobservable": 6,
                        "rr_mae_bpm": float(group.absolute_rr_error_bpm.mean()),
                        "rr_r2": 1 - np.sum((pred - truth) ** 2) / denominator if denominator > 0 else np.nan,
                        "exact_count": int(group.exact_count.sum()), "tp": tp, "fp": fp, "fn": fn,
                        "event_precision": tp / (tp + fp) if tp + fp else 0,
                        "event_recall": tp / (tp + fn) if tp + fn else 0,
                        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    base = complete[complete.arm.eq("C0P0")].set_index("video_id")
    paired = []
    for candidate in ARMS[1:]:
        other = complete[complete.arm.eq(candidate)].set_index("video_id").loc[base.index]
        differences = (other.absolute_rr_error_bpm - base.absolute_rr_error_bpm).to_numpy(float)
        rng = np.random.default_rng(20260928)
        draws = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        paired.append({"candidate": candidate, "control": "C0P0", "complete_windows": len(differences),
                       "cluster_unit": "one source per distinct filename cow ID; unverified farm ID",
                       "delta_mae_bpm": float(differences.mean()), "ci95_low": float(low),
                       "ci95_high": float(high), "bootstrap_draws": 10000, "seed": 20260928})
    return pd.DataFrame(summary), pd.DataFrame(paired)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    windows, records, files = validate(args.annotations)
    print(windows.analysis_status.value_counts().to_string(), flush=True)
    print("Complete reference events:", sum(len(records[row.window_id][2])
                                         for row in windows.itertuples() if row.analysis_status == "complete"), flush=True)
    if args.validate_only:
        return
    if args.output is None or args.output.exists():
        raise FileExistsError("Specify a fresh immutable --output directory")
    args.output.mkdir(parents=True)
    (args.output / "INCOMPLETE.txt").write_text("Scoring has not completed.\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({
        "scope": "P2 same-farm development; six full-window references after pre-inference visibility review",
        "arms": ARMS, "duration_seconds": 30, "roi_radius": 20, "min_temperature": 20.0,
        "keypoint_confidence": 0.5, "event_tolerance_seconds": 0.30,
        "files": [{"path": str(path), "sha256": digest(path)} for path in files],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(args.output / "reference_disposition.csv", windows)
    detector = YOLO(str(MODEL))
    rf_model = joblib.load(RF)
    predictions, predicted_events, matches = [], [], []
    for win in windows.itertuples(index=False):
        seg, mapping, refs = records[win.window_id]
        table = extract(Path(seg.source_path), seg, mapping, detector, rf_model)
        save_csv(args.output / "temperatures" / f"{win.video_id}.csv", table)
        times = table.playback_seconds.to_numpy(float)
        truth = refs.event_time_seconds.to_numpy(float) if win.analysis_status == "complete" else None
        for arm in ARMS:
            curve, peaks, fusion, prominence = run_one(table, arm)
            save_csv(args.output / "curves" / arm / f"{win.video_id}.csv", curve)
            count = len(peaks)
            row = {"video_id": win.video_id, "window_id": win.window_id, "arm": arm,
                   "export_status": win.annotation_status, "analysis_status": win.analysis_status,
                   "truth_count": int(win.manual_breath_count) if truth is not None else np.nan,
                   "predicted_count": count, "truth_rr_bpm": 2 * len(truth) if truth is not None else np.nan,
                   "predicted_rr_bpm": 2 * count, "absolute_rr_error_bpm": np.nan,
                   "exact_count": pd.NA, "tp": np.nan, "fp": np.nan, "fn": np.nan,
                   "event_f1": np.nan, "fusion": fusion, "prominence": prominence,
                   "left_valid": int(table.left_temp.notna().sum()),
                   "right_valid": int(table.right_temp.notna().sum())}
            for peak in peaks:
                predicted_events.append({"video_id": win.video_id, "arm": arm,
                                         "frame_index": int(peak), "source_frame": int(table.iloc[peak].source_frame),
                                         "playback_seconds": float(times[peak])})
            if truth is not None:
                hits, fp, fn = match_events(times[peaks], truth, 0.30)
                tp = len(hits)
                row.update(absolute_rr_error_bpm=2 * abs(count - len(truth)),
                           exact_count=count == len(truth), tp=tp, fp=len(fp), fn=len(fn),
                           event_f1=2 * tp / (2 * tp + len(fp) + len(fn)))
                for i, j, delta in hits:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "TP",
                                    "predicted_time": float(times[peaks[i]]), "reference_time": float(truth[j]),
                                    "reference_event_id": refs.iloc[j].event_id, "time_error_seconds": delta})
                for i in fp:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FP",
                                    "predicted_time": float(times[peaks[i]]), "reference_time": np.nan,
                                    "reference_event_id": "", "time_error_seconds": np.nan})
                for j in fn:
                    matches.append({"video_id": win.video_id, "arm": arm, "kind": "FN",
                                    "predicted_time": np.nan, "reference_time": float(truth[j]),
                                    "reference_event_id": refs.iloc[j].event_id, "time_error_seconds": np.nan})
            predictions.append(row)
        print(f"Scored {win.video_id}: {len(table)} original frames; {win.analysis_status}", flush=True)
    save_csv(args.output / "predictions.csv", predictions)
    save_csv(args.output / "predicted_events.csv", predicted_events)
    save_csv(args.output / "event_matches.csv", matches)
    summary, paired = summarize(pd.DataFrame(predictions))
    save_csv(args.output / "summary.csv", summary)
    save_csv(args.output / "paired_intervals.csv", paired)
    (args.output / "INCOMPLETE.txt").unlink()
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
