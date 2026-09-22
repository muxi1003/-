from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FIELDWORK_COLUMNS = [
    "external_video_id",
    "raw_video_path",
    "source_session_id",
    "cow_id",
    "collection_date",
    "collection_time",
    "collection_location_country",
    "collection_location_province",
    "collection_location_county",
    "collection_site",
    "camera_id",
    "scene_id",
    "external_test_split",
    "manual_breath_count",
    "manual_duration_seconds",
    "manual_rr_bpm",
    "reference_rr_annotator",
    "include_in_external_validation",
    "ambient_temperature_c",
    "relative_humidity_percent",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
    "posture",
    "athi",
    "reference_protocol_notes",
    "error_driven_planning_stratum",
    "matched_internal_prototype_video_id",
    "external_collection_notes",
]

SESSION_RE = re.compile(
    r"^(?P<date>\d{8})T(?P<time>\d{6})-(?P<cow_id>.+?)_output_clips$"
)
CLIP_RE = re.compile(r"^clip_(?P<clip_index>\d+)\.mp4$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Scan the split_all_use external video batch and build a fieldwork "
            "worksheet plus clip/session inventory for frozen RR validation."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\real\use_code\split_all_use"),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "al_images",
    )
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--external-test-split", default="external")
    parser.add_argument("--nominal-duration-seconds", type=float, default=30.0)
    parser.add_argument("--min-included-duration-seconds", type=float, default=5.0)
    parser.add_argument("--include-in-external-validation", default="yes")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def format_date(value: str) -> str:
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def format_time(value: str) -> str:
    return f"{value[0:2]}:{value[2:4]}:{value[4:6]}"


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def video_duration_seconds(path: Path) -> float | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frames <= 0:
            return None
        return frames / fps
    finally:
        capture.release()


def iter_clip_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    source_root = args.source_root.resolve()
    rows: list[dict[str, object]] = []
    if not source_root.exists():
        raise FileNotFoundError(f"Missing split_all_use source root: {source_root}")
    for session_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        match = SESSION_RE.match(session_dir.name)
        if not match:
            rows.append(
                {
                    "parse_status": "unparsed_session_name",
                    "session_folder": str(session_dir),
                    "source_session_id": session_dir.name,
                    "session_name": session_dir.name,
                }
            )
            continue
        date_raw = match.group("date")
        time_raw = match.group("time")
        cow_id = match.group("cow_id")
        collection_date = format_date(date_raw)
        collection_time = format_time(time_raw)
        source_session_id = f"{date_raw}T{time_raw}-{cow_id}"
        scene_id = (
            "CN_InnerMongolia_Hulunbuir_JiufuRanch_"
            f"{date_raw}T{time_raw}_cow{safe_id(cow_id)}"
        )
        clip_paths = sorted(
            path
            for path in session_dir.iterdir()
            if path.is_file() and CLIP_RE.match(path.name)
        )
        for clip_path in clip_paths:
            clip_match = CLIP_RE.match(clip_path.name)
            assert clip_match is not None
            clip_index_text = clip_match.group("clip_index")
            clip_index = int(clip_index_text)
            external_video_id = (
                "NM_Hulunbuir_Jiufu_"
                f"{date_raw}T{time_raw}_cow{safe_id(cow_id)}_clip{clip_index_text}"
            )
            duration = video_duration_seconds(clip_path)
            duration_source = "video_metadata" if duration is not None else "nominal_30s_from_user"
            if duration is None:
                duration = float(args.nominal_duration_seconds)
            duration_is_scoreable = float(duration) >= float(args.min_included_duration_seconds)
            rows.append(
                {
                    "parse_status": "parsed",
                    "external_video_id": external_video_id,
                    "source_session_id": source_session_id,
                    "session_name": session_dir.name,
                    "session_folder": str(session_dir),
                    "raw_video_path": str(clip_path),
                    "clip_file": clip_path.name,
                    "clip_index": clip_index,
                    "clip_duration_seconds": round(float(duration), 6),
                    "clip_duration_source": duration_source,
                    "clip_duration_is_scoreable": duration_is_scoreable,
                    "clip_duration_screen_reason": (
                        "ok"
                        if duration_is_scoreable
                        else f"duration_below_{args.min_included_duration_seconds:g}s"
                    ),
                    "cow_id": cow_id,
                    "collection_date": collection_date,
                    "collection_time": collection_time,
                    "collection_location_country": "China",
                    "collection_location_province": "Inner Mongolia",
                    "collection_location_county": "Hulunbuir City",
                    "collection_site": "Jiufu Ranch",
                    "camera_id": "",
                    "scene_id": scene_id,
                    "external_test_split": str(args.external_test_split),
                }
            )
    return rows


def build_fieldwork(inventory: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    parsed = inventory[inventory["parse_status"].astype(str).eq("parsed")].copy()
    rows: list[dict[str, object]] = []
    for _, row in parsed.iterrows():
        duration_is_scoreable = bool(row.get("clip_duration_is_scoreable", True))
        include_value = (
            str(args.include_in_external_validation)
            if duration_is_scoreable
            else "no"
        )
        reference_note = (
            "Manual breath count still required before scoring."
            if duration_is_scoreable
            else (
                "Excluded from external scoring by default because video metadata "
                f"duration is below {args.min_included_duration_seconds:g} seconds."
            )
        )
        notes = (
            "Candidate external validation clip from split_all_use. "
            "Parent folder is one long video/session split into 30-second clips; "
            "do not treat clips from the same source_session_id as independent "
            "cow/session units in grouped claims."
        )
        item = {
            "external_video_id": row["external_video_id"],
            "raw_video_path": row["raw_video_path"],
            "source_session_id": row["source_session_id"],
            "cow_id": row["cow_id"],
            "collection_date": row["collection_date"],
            "collection_time": row["collection_time"],
            "collection_location_country": row["collection_location_country"],
            "collection_location_province": row["collection_location_province"],
            "collection_location_county": row["collection_location_county"],
            "collection_site": row["collection_site"],
            "camera_id": row["camera_id"],
            "scene_id": row["scene_id"],
            "external_test_split": row["external_test_split"],
            "manual_breath_count": "",
            "manual_duration_seconds": row["clip_duration_seconds"],
            "manual_rr_bpm": "",
            "reference_rr_annotator": "",
            "include_in_external_validation": include_value,
            "ambient_temperature_c": "",
            "relative_humidity_percent": "",
            "head_motion_score_0_3": "",
            "occlusion_score_0_3": "",
            "nostril_visibility_score_0_3": "",
            "posture": "",
            "athi": "",
            "reference_protocol_notes": reference_note,
            "error_driven_planning_stratum": "",
            "matched_internal_prototype_video_id": "",
            "external_collection_notes": notes,
        }
        rows.append(item)
    return pd.DataFrame(rows, columns=FIELDWORK_COLUMNS)


def build_session_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    parsed = inventory[inventory["parse_status"].astype(str).eq("parsed")].copy()
    if parsed.empty:
        return pd.DataFrame()
    grouped = (
        parsed.groupby(
            [
                "source_session_id",
                "session_name",
                "cow_id",
                "collection_date",
                "collection_time",
                "scene_id",
            ],
            dropna=False,
        )
        .agg(
            clip_count=("external_video_id", "count"),
            first_clip=("external_video_id", "first"),
            last_clip=("external_video_id", "last"),
            total_clip_duration_seconds=("clip_duration_seconds", "sum"),
        )
        .reset_index()
    )
    return grouped


def build_readiness_summary(inventory: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    parsed = inventory[inventory["parse_status"].astype(str).eq("parsed")].copy()
    clip_count = int(len(parsed))
    scoreable = parsed[
        parsed.get("clip_duration_is_scoreable", pd.Series(False, index=parsed.index)).astype(bool)
    ].copy()
    scoreable_clip_count = int(len(scoreable))
    short_clip_count = clip_count - scoreable_clip_count
    session_count = int(len(sessions))
    cow_count = int(scoreable["cow_id"].nunique()) if not scoreable.empty else 0
    date_count = int(scoreable["collection_date"].nunique()) if not scoreable.empty else 0
    rows = [
        {
            "check": "clip-level external rows available",
            "status": "PASS" if clip_count > 0 else "FAIL",
            "evidence": f"clips={clip_count}",
            "interpretation": "All discovered clip rows, including short final fragments retained for traceability.",
        },
        {
            "check": "default included clips after duration screen",
            "status": "PASS" if scoreable_clip_count > 0 else "FAIL",
            "evidence": f"included_clips={scoreable_clip_count}; short_excluded={short_clip_count}",
            "interpretation": "Short final fragments are kept in inventory but excluded from default RR scoring.",
        },
        {
            "check": "source long-video/session rows available",
            "status": "PASS" if session_count > 0 else "FAIL",
            "evidence": f"sessions={session_count}",
            "interpretation": "Use source_session_id for leakage checks because clips from one session are correlated.",
        },
        {
            "check": "unique cow_id labels available",
            "status": "PASS" if cow_count >= 8 else "WARN",
            "evidence": f"unique_cows={cow_count}",
            "interpretation": "Q2 algorithmic external tier asks for at least 8 cow labels.",
        },
        {
            "check": "collection dates available",
            "status": "PASS" if date_count >= 4 else "WARN",
            "evidence": f"unique_dates={date_count}",
            "interpretation": "Q2 algorithmic external tier asks for at least 4 dates.",
        },
        {
            "check": "minimum 50-video external smoke tier by clips",
            "status": "PASS" if scoreable_clip_count >= 50 else "FAIL",
            "evidence": f"included_clips={scoreable_clip_count}; target=50",
            "interpretation": "Passes only after manual RR labels and frozen pipeline outputs exist.",
        },
        {
            "check": "104-video Q2 algorithmic external tier by clips",
            "status": "PASS" if scoreable_clip_count >= 104 else "FAIL",
            "evidence": f"included_clips={scoreable_clip_count}; target=104",
            "interpretation": "Clip-level target is feasible only if grouped/session leakage is handled.",
        },
        {
            "check": "manual RR labels present",
            "status": "FAIL",
            "evidence": "manual_breath_count=0 filled by this manifest",
            "interpretation": "Fill manual_breath_count for each included clip before external scoring.",
        },
        {
            "check": "environment and manual quality labels present",
            "status": "FAIL",
            "evidence": "ambient_temperature_c, relative_humidity_percent, head_motion, occlusion, nostril_visibility are blank",
            "interpretation": "Current batch can support algorithmic external validation first, not heat-stress/manual-quality claims.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(data: pd.DataFrame, max_rows: int = 30) -> str:
    if data.empty:
        return "_No rows._"
    table = data.head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    sessions: pd.DataFrame,
    readiness: pd.DataFrame,
    fieldwork_path: Path,
) -> None:
    parsed = inventory[inventory["parse_status"].astype(str).eq("parsed")].copy()
    dates = ", ".join(sorted(parsed["collection_date"].dropna().astype(str).unique()))
    lines = [
        "# split_all_use External Batch Manifest",
        "",
        f"Parsed clips: `{len(parsed)}`",
        f"Source long-video/session folders: `{len(sessions)}`",
        f"Unique cow_id labels: `{parsed['cow_id'].nunique() if not parsed.empty else 0}`",
        f"Collection dates: `{dates}`",
        "",
        "## Batch Context",
        "",
        "Location was prefilled from the user-provided context: China, Inner Mongolia, Hulunbuir City, Jiufu Ranch. Folder names are parsed as `YYYYMMDDTHHMMSS-cow_id_output_clips`; `clip_0001.mp4`, `clip_0002.mp4`, etc. are 30-second segments from one long source video.",
        "",
        "## Readiness",
        "",
        markdown_table(readiness),
        "",
        "## Session Summary Preview",
        "",
        markdown_table(sessions),
        "",
        "## Next Fill File",
        "",
        f"Fill manual reference RR in `{fieldwork_path}`. Keep `source_session_id` in mind when designing grouped validation; clips from the same parent folder are not independent cow/session evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.DataFrame(iter_clip_rows(args))
    fieldwork = build_fieldwork(inventory, args)
    sessions = build_session_summary(inventory)
    readiness = build_readiness_summary(inventory, sessions)

    inventory_path = output_dir / "paper_external_validation_split_all_use_inventory.csv"
    fieldwork_path = output_dir / "paper_external_validation_split_all_use_fieldwork_template.csv"
    sessions_path = output_dir / "paper_external_validation_split_all_use_session_summary.csv"
    readiness_path = output_dir / "paper_external_validation_split_all_use_readiness.csv"
    report_path = output_dir / "paper_external_validation_split_all_use_manifest.md"

    inventory.to_csv(inventory_path, index=False)
    fieldwork.to_csv(fieldwork_path, index=False)
    sessions.to_csv(sessions_path, index=False)
    readiness.to_csv(readiness_path, index=False)
    write_report(report_path, inventory, sessions, readiness, fieldwork_path)

    print(f"Saved split_all_use inventory: {inventory_path}")
    print(f"Saved split_all_use fieldwork template: {fieldwork_path}")
    print(f"Saved split_all_use session summary: {sessions_path}")
    print(f"Saved split_all_use readiness: {readiness_path}")
    print(f"Saved split_all_use report: {report_path}")
    parsed = inventory[inventory["parse_status"].astype(str).eq("parsed")]
    print(
        f"Parsed clips={len(parsed)}, sessions={len(sessions)}, "
        f"unique_cows={parsed['cow_id'].nunique() if not parsed.empty else 0}, "
        f"dates={parsed['collection_date'].nunique() if not parsed.empty else 0}"
    )


if __name__ == "__main__":
    main()
