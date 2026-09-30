"""Audit the frozen P3 annotation exports without reading RR predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE / "20260928_expansion_p3_v2"
PROTOCOL = HERE / "20260928_p3_pre_score_protocol_v2.md"
REFERENCE_FILES = {
    "windows": "lindian_expansion_annotation_windows.csv",
    "events": "lindian_expansion_reference_events.csv",
    "intervals": "lindian_expansion_unobservable_intervals.csv",
    "backup": "lindian_expansion_event_backup.json",
}
COUNT_ONLY_CLAIMS = {
    "20230809T084030n170020_030_060": 36,
    "20230809T073104n200792_060_090": 36,
}
IMMUTABLE_FIELDS = (
    "window_id", "video_id", "video_path", "source_path",
    "source_start_seconds", "duration_seconds", "cohort", "annotation_round",
    "event_definition", "browser_video_path", "view_start_seconds",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((key, str(value)) for key, value in row.items())) for row in rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(snapshot: Path, output: Path) -> None:
    raw_hashes: list[dict[str, str]] = []
    audited: list[dict] = []
    claims: list[dict] = []
    all_ids: set[str] = set()
    for batch, expected_size in (("A", 27), ("B", 26)):
        raw = snapshot / f"raw_{batch}"
        files = {key: raw / name for key, name in REFERENCE_FILES.items()}
        for file in files.values():
            if not file.is_file():
                raise FileNotFoundError(file)
            raw_hashes.append({"path": str(file), "sha256": digest(file)})
        backup = json.loads(files["backup"].read_text(encoding="utf-8"))
        exports = {key: csv_rows(files[key]) for key in ("windows", "events", "intervals")}
        for key in exports:
            if normalized(backup[key]) != normalized(exports[key]):
                raise ValueError(f"{batch} JSON/CSV mismatch: {key}")

        batch_dir = P3 / f"annotation_batch_{batch}"
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        selected = P3 / f"selected_sources_{batch}.csv"
        if (manifest["clips"] != expected_size or len(exports["windows"]) != expected_size
                or manifest["selection_hash"] != digest(selected)
                or manifest["protocol_hash"] != digest(PROTOCOL)
                or manifest["technical_exclusions"]):
            raise ValueError(f"{batch} selection/annotation freeze mismatch")
        planned = {row["window_id"]: row for row in csv_rows(batch_dir / "annotation_windows_pending.csv")}
        segments = {row["window_id"]: row for row in csv_rows(batch_dir / "segments.csv")}
        if len(planned) != expected_size or len(segments) != expected_size:
            raise ValueError(f"{batch} planned source list mismatch")
        events: dict[str, list[dict]] = defaultdict(list)
        intervals: dict[str, list[dict]] = defaultdict(list)
        event_ids: set[str] = set()
        for event in exports["events"]:
            wid = event["window_id"]
            seconds = float(event["event_time_seconds"])
            if (wid not in planned or event["annotation_round"] != f"LX-20260928-P3{batch}"
                    or event["event_id"] in event_ids or not 0 <= seconds < 30
                    or event["confidence"] not in {"confirmed", "uncertain"}):
                raise ValueError(f"Invalid {batch} event: {event}")
            event_ids.add(event["event_id"])
            events[wid].append(event)
        for interval in exports["intervals"]:
            wid = interval["window_id"]
            start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
            if (wid not in planned or interval["annotation_round"] != f"LX-20260928-P3{batch}"
                    or not 0 <= start < end <= 30):
                raise ValueError(f"Invalid {batch} interval: {interval}")
            intervals[wid].append(interval)

        for win in exports["windows"]:
            wid, video = win["window_id"], win["video_id"]
            if wid in all_ids or wid not in planned or wid not in segments:
                raise ValueError(f"Duplicate/unknown P3 window: {wid}")
            all_ids.add(wid)
            if any(win[key] != planned[wid][key] for key in IMMUTABLE_FIELDS):
                raise ValueError(f"P3 source/time binding changed: {wid}")
            seg = segments[wid]
            if (seg["video_id"] != video or seg["source_path"] != win["source_path"]
                    or float(seg["source_start_seconds"]) != float(win["source_start_seconds"])):
                raise ValueError(f"P3 segment binding changed: {wid}")
            refs, spans = events[wid], intervals[wid]
            confirmed = sum(item["confidence"] == "confirmed" for item in refs)
            uncertain = len(refs) - confirmed
            count = win["manual_breath_count"].strip()
            issues = []
            if count and (not count.isdecimal() or int(count) != confirmed):
                issues.append("manual_count_vs_confirmed_events")
            if win["annotation_status"] == "pending":
                issues.append("pending_status")
            if win["annotation_status"] == "partial" and not spans and win["full_window_basis"] == "none":
                issues.append("partial_missing_unobservable_interval")
            if (win["annotation_status"] == "unobservable" and
                    win["full_window_basis"] == "nose_visible_throughout"):
                issues.append("unobservable_basis_conflict")
            complete_valid = (
                win["annotation_status"] == "complete" and count.isdecimal()
                and int(count) == confirmed and uncertain == 0 and not spans
                and win["predictions_hidden"].lower() == "true"
                and win["full_window_basis"] in {"nose_visible_throughout", "independent_body_signal"}
                and (win["full_window_basis"] != "independent_body_signal" or win["reference_notes"].strip())
            )
            if win["annotation_status"] == "complete" and not complete_valid:
                issues.append("complete_fails_frozen_reference_rule")
            if video in COUNT_ONLY_CLAIMS:
                issues.append("user_36_count_event_times_unreliable")
                if win["annotation_status"] == "complete":
                    issues.append("export_complete_zero_contradicts_user_count")
                disposition = "count_only_claim_excluded_primary"
                claims.append({
                    "batch": batch, "window_id": wid, "video_id": video,
                    "user_reported_30s_count": COUNT_ONLY_CLAIMS[video],
                    "reported_rr_bpm": 2 * COUNT_ONLY_CLAIMS[video],
                    "source": "user_message_2026-09-29",
                    "event_timing": "unreliable_not_supplied",
                    "frozen_primary_rr": "excluded", "event_f1": "not_scored",
                    "export_status": win["annotation_status"],
                    "export_confirmed_events": confirmed,
                })
            else:
                disposition = "complete_primary" if complete_valid else "not_primary"
            audited.append({
                "batch": batch, "window_id": wid, "video_id": video,
                "export_status": win["annotation_status"],
                "analysis_disposition": disposition,
                "export_manual_count": count,
                "confirmed_events": confirmed, "uncertain_events": uncertain,
                "unobservable_intervals": len(spans),
                "full_window_basis": win["full_window_basis"],
                "predictions_hidden": win["predictions_hidden"],
                "issue_flags": ";".join(issues),
            })
    if len(audited) != 53 or len(claims) != 2:
        raise ValueError("P3 denominator or count-only claim membership changed")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "window_audit.csv", audited)
    write_csv(output / "count_only_claims.csv", claims)
    summary = {
        "scope": "pre-prediction P3 annotation audit; no RR model outputs read",
        "raw_export_sha256": raw_hashes,
        "export_status_counts": dict(Counter(row["export_status"] for row in audited)),
        "analysis_disposition_counts": dict(Counter(row["analysis_disposition"] for row in audited)),
        "issue_counts": dict(Counter(flag for row in audited for flag in row["issue_flags"].split(";") if flag)),
        "frozen_primary_eligible": sum(row["analysis_disposition"] == "complete_primary" for row in audited),
        "selected_sources": 53,
        "count_only_user_claims_not_primary": 2,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "raw_export_sha256"},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.snapshot.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
