"""Compare the P2 first pass with the count-hidden R1 repeat, without rescoring RR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "03_可靠事件参考"))
from reference_tools import match_events

P2 = Path("C:/Users/muxi/Desktop/实验/呼吸事件标注 LX-20260928-P2")
R1 = Path("C:/Users/muxi/Desktop/实验/呼吸事件标注 LX-20260928-R1")
BATCH = HERE / "20260928_expansion_p2_v1/annotation_batch_v1"
GATE = HERE / "20260928_confirmation_gate_v1"
P2_DISPOSITION = HERE / "20260928_lx_p2_v1/reference_disposition.csv"
FILENAMES = {
    "windows": "lindian_expansion_annotation_windows.csv",
    "events": "lindian_expansion_reference_events.csv",
    "intervals": "lindian_expansion_unobservable_intervals.csv",
    "backup": "lindian_expansion_event_backup.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((key, str(value)) for key, value in row.items())) for row in rows)


def load_reference(directory: Path) -> dict[str, list[dict[str, str]]]:
    files = {key: directory / filename for key, filename in FILENAMES.items()}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError(f"Missing files in {directory}")
    backup = json.loads(files["backup"].read_text(encoding="utf-8"))
    tables = {key: read_csv(files[key]) for key in ("windows", "events", "intervals")}
    for key, rows in tables.items():
        if normalized(rows) != normalized(backup[key]):
            raise ValueError(f"JSON/CSV mismatch: {directory}, {key}")
    return tables


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "20260928_r1_reference_audit_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")

    p2, r1 = load_reference(P2), load_reference(R1)
    queue = read_csv(GATE / "p2_reference_recheck_queue.csv")
    segments = {row["window_id"]: row for row in read_csv(BATCH / "segments.csv")}
    disposition = {row["window_id"]: row for row in read_csv(P2_DISPOSITION)}
    p2_windows = {row["window_id"]: row for row in p2["windows"]}
    r1_windows = {row["window_id"]: row for row in r1["windows"]}
    expected = {row["window_id"] for row in queue}
    if len(expected) != 6 or set(r1_windows) != expected:
        raise ValueError("R1 membership differs from six queued windows")
    if any(row["window_id"] not in expected for row in r1["events"] + r1["intervals"]):
        raise ValueError("R1 contains events or intervals outside the queue")

    event_rows = defaultdict(lambda: {"P2": [], "R1": []})
    for round_name, reference in (("P2", p2), ("R1", r1)):
        seen = set()
        for row in reference["events"]:
            if row["window_id"] not in expected:
                continue
            key = (row["window_id"], row["event_id"])
            if key in seen or row["event_type"] != "expiration_peak":
                raise ValueError(f"Duplicate or wrong event: {key}")
            seen.add(key)
            t = float(row["event_time_seconds"])
            if not 0 <= t < 30:
                raise ValueError(f"Event outside 30 s: {key}")
            event_rows[row["window_id"]][round_name].append(row)

    comparisons, match_rows = [], []
    for item in queue:
        window_id = item["window_id"]
        old, new = p2_windows[window_id], r1_windows[window_id]
        segment = segments[window_id]
        clip = Path(item["clip_path"])
        if (disposition[window_id]["analysis_status"] != "complete"
                or old["annotation_status"] != "complete"
                or old["annotation_round"] != "LX-20260928-P2"
                or new["annotation_round"] != "LX-20260928-R1"
                or old["window_id"] != new["window_id"]
                or old["video_id"] != new["video_id"]
                or old["source_path"] != new["source_path"]
                or old["source_start_seconds"] != new["source_start_seconds"]
                or float(new["duration_seconds"]) != 30
                or item["clip_sha256"] != segment["clip_sha256"]
                or digest(clip) != item["clip_sha256"]):
            raise ValueError(f"Window identity or clip changed: {window_id}")
        if (new["annotation_status"] not in {"complete", "partial", "unobservable"}
                or new["predictions_hidden"].lower() != "true"
                or not new["annotator"].strip()):
            raise ValueError(f"R1 status/exposure/annotator invalid: {window_id}")

        p_events = sorted(event_rows[window_id]["P2"], key=lambda row: float(row["event_time_seconds"]))
        r_events = sorted(event_rows[window_id]["R1"], key=lambda row: float(row["event_time_seconds"]))
        p_confirmed = [row for row in p_events if row["confidence"] == "confirmed"]
        r_confirmed = [row for row in r_events if row["confidence"] == "confirmed"]
        old_count = int(float(old["manual_breath_count"]))
        if old_count != len(p_confirmed):
            raise ValueError(f"P2 count/event mismatch: {window_id}")
        if new["annotation_status"] in {"complete", "partial"}:
            if not new["manual_breath_count"] or int(float(new["manual_breath_count"])) != len(r_confirmed):
                raise ValueError(f"R1 count/event mismatch: {window_id}")
        spans = [row for row in r1["intervals"] if row["window_id"] == window_id]
        if new["annotation_status"] == "complete":
            if (new["full_window_basis"] not in {"nose_visible_throughout", "independent_body_signal"}
                    or spans or len(r_confirmed) != len(r_events)
                    or (new["full_window_basis"] == "independent_body_signal"
                        and not new["reference_notes"].strip())):
                raise ValueError(f"R1 complete lacks full-window support: {window_id}")

        p_times = [float(row["event_time_seconds"]) for row in p_confirmed]
        r_times = [float(row["event_time_seconds"]) for row in r_confirmed]
        hits, p_only, r_only = match_events(p_times, r_times, .30)
        for i, j, error in hits:
            match_rows.append({"window_id": window_id, "p2_event_id": p_confirmed[i]["event_id"],
                               "r1_event_id": r_confirmed[j]["event_id"],
                               "p2_seconds": p_times[i], "r1_seconds": r_times[j],
                               "absolute_time_difference_seconds": error})
        comparisons.append({
            "window_id": window_id,
            "video_id": new["video_id"],
            "p2_status": "complete",
            "p2_count": old_count,
            "r1_status": new["annotation_status"],
            "r1_count_visible": len(r_confirmed),
            "r1_full_window_basis": new["full_window_basis"],
            "r1_unobservable_intervals": len(spans),
            "r1_count_minus_p2": len(r_confirmed) - old_count,
            "time_matched_at_0p30s": len(hits),
            "p2_only_times_seconds": ";".join(f"{p_times[i]:.3f}" for i in p_only),
            "r1_only_times_seconds": ";".join(f"{r_times[i]:.3f}" for i in r_only),
            "event_time_match_scope": "paired_full_window" if new["annotation_status"] == "complete"
                                      else "partial_diagnostic_only",
            "reference_issue": "missing_unobservable_interval_bounds" if new["annotation_status"] == "partial"
                               and not spans else "",
        })

    complete = [row for row in comparisons if row["r1_status"] == "complete"]
    summary = {
        "p2_complete_queued": len(comparisons),
        "r1_complete": len(complete),
        "r1_partial": sum(row["r1_status"] == "partial" for row in comparisons),
        "paired_complete_count_exact": sum(row["r1_count_minus_p2"] == 0 for row in complete),
        "paired_complete_count_disagreements": [row["video_id"] for row in complete
                                                if row["r1_count_minus_p2"] != 0],
        "same_day_repeat": P2.joinpath(FILENAMES["backup"]).stat().st_mtime_ns // (86400 * 10**9)
                           == R1.joinpath(FILENAMES["backup"]).stat().st_mtime_ns // (86400 * 10**9),
        "r1_not_independent_blind_reference": True,
        "p2_frozen_score_unchanged": True,
        "rr_rescored": False,
        "input_sha256": {str(directory / filename): digest(directory / filename)
                         for directory in (P2, R1) for filename in FILENAMES.values()},
        "queue_sha256": digest(GATE / "p2_reference_recheck_queue.csv"),
        "matcher_sha256": digest(HERE.parent / "03_可靠事件参考/reference_tools.py"),
    }
    output.mkdir(parents=True)
    write_csv(output / "window_comparison.csv", comparisons, list(comparisons[0]))
    write_csv(output / "event_time_matches.csv", match_rows,
              ["window_id", "p2_event_id", "r1_event_id", "p2_seconds", "r1_seconds",
               "absolute_time_difference_seconds"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "input_sha256"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
