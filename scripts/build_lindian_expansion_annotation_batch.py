"""Cut one PTS-contiguous 30 s viewing window per frozen Lindian source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cut_lindian_contiguous_30s import cut_windows, read_timestamps, window_rows

ROUND = "LX-20260924-P1"
COHORT = "lindian_expansion"
TEMPLATE = ROOT / "Experiment/03_可靠事件参考/event_workspace_template.html"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty batch")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"Expected {count} template occurrences of {old!r}")
    return text.replace(old, new)


def make_page(windows: list[dict]) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    page = replace_exact(page, "R2", ROUND, 8)
    page = replace_exact(page, "20260915.2", "20260924")
    page = replace_exact(page, '<option value="lindian49">林甸49窗</option><option value="jiufu271">久福271窗</option>',
                         f'<option value="{COHORT}">林甸扩展批次</option>')
    page = replace_exact(page,
                         "x.cohort==='lindian49'?'林甸：存在历史算法接触，不能宣称原始盲法。':'久福：本轮接触情况请如实登记。'",
                         "'按原视频逐窗标注；同一牛或同一来源的窗口不能当作独立牛。算法输出未嵌入本页。'")
    page = replace_exact(page, "cow-rr-event-LX-20260924-P1-20260914-v1",
                         "cow-rr-event-LX-20260924-P1-batch-v1")
    page = replace_exact(page, "'annotation_windows.csv'", "'lindian_expansion_annotation_windows.csv'")
    page = replace_exact(page, "'reference_events.csv'", "'lindian_expansion_reference_events.csv'")
    page = replace_exact(page, "'unobservable_intervals.csv'", "'lindian_expansion_unobservable_intervals.csv'")
    page = replace_exact(page, "event_reference_LX-20260924-P1_backup.json",
                         "lindian_expansion_event_backup.json")
    payload = json.dumps(windows, ensure_ascii=False).replace("<", "\\u003c")
    return replace_exact(page, "__WINDOW_DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.selection = args.selection.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    with args.selection.open(newline="", encoding="utf-8-sig") as stream:
        selected = list(csv.DictReader(stream))
    if not selected or len({row["cow_id"] for row in selected}) != len(selected):
        raise ValueError("Selected sources must have unique cow IDs")

    planned, excluded = [], []
    for item in selected:
        source = Path(item["source_path"]).resolve(strict=True)
        if digest(source) != item["source_sha256"]:
            raise ValueError(f"Source hash changed: {source}")
        try:
            timestamps, fps, size = read_timestamps(source)
            all_windows = window_rows(timestamps, 30, 0.5, 0.25)
            eligible = next((row for row in all_windows if row["status"] == "CUT"), None)
        except (RuntimeError, ValueError) as exc:
            excluded.append({"cow_id": item["cow_id"], "source_path": str(source),
                             "reason": f"timestamp_check_failed:{exc}"})
            continue
        if eligible is None:
            excluded.append({"cow_id": item["cow_id"], "source_path": str(source),
                             "reason": "no_pts_contiguous_30s_window"})
            continue
        planned.append((item, source, timestamps, fps, size, eligible))
    if not planned:
        raise ValueError("No eligible 30-second clips")

    args.output.mkdir(parents=True)
    media = args.output / "media"
    media.mkdir()
    segments, frame_map, windows = [], [], []
    for item, source, timestamps, fps, size, segment in planned:
        mapped = cut_windows(source, media, [segment], timestamps, size, browser_webm=True)
        clip = Path(segment["clip_path"])
        capture = cv2.VideoCapture(str(clip))
        try:
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
        finally:
            capture.release()
        if count != len(mapped) or count != int(segment["source_frame_count"]):
            raise ValueError(f"Encoded/decode frame count mismatch: {clip}")
        uid = f"{source.stem}_{segment['segment_id']}"
        segments.append({"window_id": uid + "_source30s", "video_id": uid, "date": item["date"],
                         "cow_id_from_filename": item["cow_id"], "source_path": str(source),
                         "source_sha256": item["source_sha256"], "source_start_seconds": segment["source_start_seconds"],
                         "source_end_seconds": segment["source_end_seconds"],
                         "first_source_frame": segment["first_source_frame"],
                         "stop_source_frame_exclusive": segment["stop_source_frame_exclusive"],
                         "source_frame_count": count, "first_source_pts_seconds": segment["first_source_pts_seconds"],
                         "last_source_pts_seconds": segment["last_source_pts_seconds"],
                         "max_interframe_gap_seconds": segment["max_interframe_gap_seconds"],
                         "source_fps_metadata": fps, "clip_path": str(clip), "clip_sha256": digest(clip)})
        frame_map.extend({"video_id": uid, **row} for row in mapped)
        windows.append({"window_id": uid + "_source30s", "video_id": uid, "video_path": str(clip),
                        "source_path": str(source), "source_start_seconds": str(segment["source_start_seconds"]),
                        "duration_seconds": "30", "cohort": COHORT, "annotation_round": ROUND,
                        "annotator": "", "annotation_status": "pending", "manual_breath_count": "",
                        "event_definition": "expiration_peak", "predictions_hidden": "", "reference_notes": "",
                        "prior_algorithm_exposure": "not_verified; annotator must report", "browser_video_path": str(clip),
                        "view_start_seconds": "0"})
        print(f"{uid}: {count} source frames, first valid PTS bin {segment['segment_id']}", flush=True)

    write_csv(args.output / "segments.csv", segments)
    write_csv(args.output / "frame_time_map.csv", frame_map)
    write_csv(args.output / "annotation_windows_pending.csv", windows)
    if excluded:
        write_csv(args.output / "technical_exclusions.csv", excluded)
    (args.output / "index.html").write_text(make_page(windows), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({
        "annotation_round": ROUND, "cohort": COHORT, "source_candidates": len(selected),
        "clips": len(windows), "technical_exclusions": excluded,
        "selection_hash": digest(args.selection), "generator_hash": digest(Path(__file__)),
        "template_hash": digest(TEMPLATE), "source_time_window_seconds": 30,
        "max_interframe_gap_seconds": 0.5, "max_edge_distance_seconds": 0.25,
        "algorithm_predictions_read": False, "manual_truth_present": False,
        "warning": "Cows parsed from filenames only; YOLO training provenance not checked; not pristine holdout",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(windows)} clips, {len(excluded)} technical exclusions: {args.output}")


if __name__ == "__main__":
    main()
