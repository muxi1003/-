"""Create a reproducible mapping from labelled short clips to source videos."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd


RAW_NAME_PATTERN = re.compile(
    r"^(?P<date>20\d{6})T(?P<time>\d{6})n(?P<animal_token>.+)$", re.IGNORECASE
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    finally:
        capture.release()
    duration_seconds = frames / fps if fps > 0 else float("nan")
    return {
        "raw_fps": fps,
        "raw_frames": frames,
        "raw_duration_seconds": duration_seconds,
    }


def raw_siblings(short_source: Path) -> list[Path]:
    siblings = []
    for candidate in short_source.parent.glob("*.MP4"):
        if candidate == short_source:
            continue
        if RAW_NAME_PATTERN.match(candidate.stem):
            siblings.append(candidate)
    return sorted(siblings, key=lambda path: str(path).lower())


def select_raw_source(short_source: Path, candidates: list[Path]) -> Path | None:
    if len(candidates) == 1:
        return candidates[0]
    directory_named_raw = [
        candidate
        for candidate in candidates
        if candidate.stem.lower() == short_source.parent.name.lower()
    ]
    if len(directory_named_raw) == 1:
        return directory_named_raw[0]
    return None


def build_manifest(data_root: Path, window_seconds: float) -> pd.DataFrame:
    labelled_dir = data_root / "all_can_use"
    truth_path = labelled_dir / "temperature_curves.csv"
    truth = pd.read_csv(truth_path, dtype={"video_id": str})
    required = {"video_id", "rr"}
    if not required.issubset(truth.columns):
        raise ValueError(f"{truth_path} must contain {sorted(required)}")

    source_index: dict[int, list[Path]] = defaultdict(list)
    for candidate in data_root.rglob("*.MP4"):
        if candidate.parent == labelled_dir:
            continue
        source_index[candidate.stat().st_size].append(candidate)

    rows: list[dict[str, object]] = []
    for record in truth.sort_values("video_id").itertuples(index=False):
        video_id = str(record.video_id)
        labelled_clip = labelled_dir / f"{video_id}.MP4"
        if not labelled_clip.exists():
            rows.append(
                {
                    "video_id": video_id,
                    "reference_rr_bpm": record.rr,
                    "mapping_status": "labelled_clip_missing",
                }
            )
            continue

        labelled_hash = sha256(labelled_clip)
        same_size = source_index.get(labelled_clip.stat().st_size, [])
        same_content = [path for path in same_size if sha256(path) == labelled_hash]
        raw_candidates = sorted(
            {raw for source in same_content for raw in raw_siblings(source)},
            key=lambda path: str(path).lower(),
        )
        row: dict[str, object] = {
            "video_id": video_id,
            "reference_rr_bpm": record.rr,
            "labelled_clip_path": str(labelled_clip),
            "labelled_clip_sha256": labelled_hash,
            "short_source_match_count": len(same_content),
            "short_source_paths": " | ".join(str(path) for path in same_content),
            "raw_candidate_count": len(raw_candidates),
            "raw_candidate_paths": " | ".join(str(path) for path in raw_candidates),
            "fixed_window_seconds": window_seconds,
        }
        selected_raw = select_raw_source(same_content[0], raw_candidates) if same_content else None
        if not same_content:
            row["mapping_status"] = "short_source_not_found"
        elif selected_raw is None:
            row["mapping_status"] = "raw_source_ambiguous_or_missing"
        else:
            raw_source = selected_raw
            name_match = RAW_NAME_PATTERN.match(raw_source.stem)
            assert name_match is not None
            probe = probe_video(raw_source)
            duration = float(probe["raw_duration_seconds"])
            window_ready = bool(duration >= window_seconds)
            row.update(
                {
                    "mapping_status": "ready_for_fixed_window" if window_ready else "raw_too_short",
                    "raw_source_path": str(raw_source),
                    "raw_source_sha256": sha256(raw_source),
                    "capture_date": name_match.group("date"),
                    "capture_time": name_match.group("time"),
                    "raw_animal_token": name_match.group("animal_token"),
                    **probe,
                    "window_ready": window_ready,
                    "center_window_start_seconds": max((duration - window_seconds) / 2.0, 0.0),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def write_report(manifest: pd.DataFrame, output_path: Path) -> None:
    status_counts = manifest["mapping_status"].value_counts(dropna=False)
    ready = manifest.loc[manifest["mapping_status"] == "ready_for_fixed_window"]
    report = [
        "# Lindian Long-Video Source Manifest",
        "",
        "This manifest maps each labelled short clip to a sibling timestamped raw video by SHA-256 content matching. It does not copy, edit, or process raw videos.",
        "",
        f"- Labelled clips: {len(manifest)}",
        f"- Ready for a fixed 30-second window: {len(ready)}",
        "",
        "## Mapping Status",
        "",
        "| status | videos |",
        "| --- | ---: |",
    ]
    report.extend(f"| {status} | {count} |" for status, count in status_counts.items())
    if not ready.empty:
        durations = ready["raw_duration_seconds"]
        report.extend(
            [
                "",
                "## Raw Duration Among Window-Ready Videos",
                "",
                f"- Minimum: {durations.min():.3f} s",
                f"- Median: {durations.median():.3f} s",
                f"- Maximum: {durations.max():.3f} s",
            ]
        )
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def write_annotation_template(manifest: pd.DataFrame, output_path: Path) -> None:
    ready = manifest.loc[manifest["mapping_status"].eq("ready_for_fixed_window")].copy()
    template = pd.DataFrame(
        {
            "window_id": [
                f"{video_id}_center_{int(round(window_seconds))}s"
                for video_id, window_seconds in zip(ready["video_id"], ready["fixed_window_seconds"])
            ],
            "video_id": ready["video_id"],
            "raw_source_path": ready["raw_source_path"],
            "capture_date": ready["capture_date"],
            "capture_time": ready["capture_time"],
            "window_start_seconds": ready["center_window_start_seconds"],
            "window_seconds": ready["fixed_window_seconds"],
            "short_clip_reference_rr_bpm_not_window_truth": ready["reference_rr_bpm"],
            "manual_breath_count": pd.NA,
            "manual_rr_bpm": pd.NA,
            "annotation_status": "pending",
            "labeler_id": "",
            "review_notes": "",
            "include_in_fixed_window_evaluation": "",
        }
    )
    template.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\real\use_code\林甸红外视频"),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Dataset_new/72video/al_images/"
            "paper_repro_quality_residual_paper_assets"
        ),
    )
    args = parser.parse_args()
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_root, args.window_seconds)
    csv_path = args.output_dir / "lindian_long_video_manifest.csv"
    report_path = args.output_dir / "lindian_long_video_manifest.md"
    template_path = args.output_dir / "lindian_fixed30_window_annotation_template.csv"
    manifest.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_report(manifest, report_path)
    write_annotation_template(manifest, template_path)
    print(f"Saved manifest: {csv_path.resolve()}")
    print(f"Saved report: {report_path.resolve()}")
    print(f"Saved annotation template: {template_path.resolve()}")
    print(manifest["mapping_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
