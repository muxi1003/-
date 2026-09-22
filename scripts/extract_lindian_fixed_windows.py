"""Export centre fixed-duration frame windows from the Lindian raw-video manifest."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=assets / "lindian_long_video_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "lindian_fixed30_frames",
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument(
        "--window-start-column",
        default="center_window_start_seconds",
        help="Manifest column containing the non-truth-derived start time in seconds.",
    )
    parser.add_argument(
        "--include-csv",
        type=Path,
        default=None,
        help="Optional CSV whose video_id rows restrict which manifest records are exported.",
    )
    parser.add_argument("--include-column", default="quality_gate")
    parser.add_argument("--include-value", default="pass")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def clean_existing_frames(output_dir: Path) -> None:
    for path in output_dir.glob("frame_*.jpg"):
        path.unlink()


def export_window(
    row: pd.Series,
    output_root: Path,
    overwrite: bool,
    jpeg_quality: int,
    window_start_column: str,
) -> dict[str, object]:
    video_id = str(row["video_id"])
    raw_path = Path(str(row["raw_source_path"]))
    output_dir = output_root / video_id
    base = {
        "video_id": video_id,
        "raw_source_path": str(raw_path),
        "requested_window_start_seconds": float(row[window_start_column]),
        "requested_window_seconds": float(row["fixed_window_seconds"]),
        "frame_dir": str(output_dir),
    }
    existing = sorted(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
    if existing and not overwrite:
        return {**base, "status": "SKIP", "frames_written": len(existing), "issue": "existing frames retained"}
    if existing:
        clean_existing_frames(output_dir)
    if not raw_path.is_file():
        return {**base, "status": "FAIL", "frames_written": 0, "issue": "raw source is missing"}

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        return {**base, "status": "FAIL", "frames_written": 0, "issue": "could not open raw source"}
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return {**base, "status": "FAIL", "frames_written": 0, "issue": "raw source has invalid fps"}
        start_frame = max(0, int(round(float(row[window_start_column]) * fps)))
        expected_frames = max(1, int(round(float(row["fixed_window_seconds"]) * fps)))
        advertised_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        end_safety_shift = 0
        if advertised_frames and start_frame + expected_frames >= advertised_frames:
            end_safety_shift = min(2, start_frame)
            start_frame -= end_safety_shift
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        for frame_number in range(expected_frames):
            ok, frame = capture.read()
            if not ok:
                break
            path = output_dir / f"frame_{frame_number:06d}.jpg"
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                return {
                    **base,
                    "status": "FAIL",
                    "frames_written": frames_written,
                    "source_fps": fps,
                    "source_start_frame": start_frame,
                    "end_safety_shift_frames": end_safety_shift,
                    "expected_frames": expected_frames,
                    "issue": f"could not write {path}",
                }
            frames_written += 1
    finally:
        capture.release()

    actual_duration = frames_written / fps
    status = "PASS" if frames_written == expected_frames else "FAIL"
    return {
        **base,
        "status": status,
        "frames_written": frames_written,
        "source_fps": fps,
        "source_start_frame": start_frame,
        "end_safety_shift_frames": end_safety_shift,
        "expected_frames": expected_frames,
        "actual_window_start_seconds": start_frame / fps,
        "actual_window_seconds": actual_duration,
        "issue": "" if status == "PASS" else "source ended before the requested window",
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    required = {
        "video_id",
        "mapping_status",
        "raw_source_path",
        "fixed_window_seconds",
    }
    required.add(args.window_start_column)
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[manifest["mapping_status"].eq("ready_for_fixed_window")].copy()
    if args.video_id:
        selected = selected.loc[selected["video_id"].isin(set(args.video_id))]
    if args.include_csv is not None:
        include = pd.read_csv(args.include_csv, dtype={"video_id": str})
        include_required = {"video_id", args.include_column}
        include_missing = include_required - set(include.columns)
        if include_missing:
            raise ValueError(f"Include CSV is missing columns: {sorted(include_missing)}")
        allowed = set(
            include.loc[include[args.include_column].astype(str).eq(args.include_value), "video_id"].astype(str)
        )
        selected = selected.loc[selected["video_id"].isin(allowed)]
    if selected.empty:
        raise ValueError("No ready fixed-window records selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        export_window(
            row,
            args.output_root,
            bool(args.overwrite),
            int(args.jpeg_quality),
            args.window_start_column,
        )
        for _, row in selected.sort_values("video_id").iterrows()
    ]
    report = pd.DataFrame(rows)
    report_path = args.output_root / "lindian_fixed_window_frame_extraction_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"Saved fixed-window extraction report: {report_path.resolve()}")
    print(f"Rows: {len(report)}; status_counts={report['status'].value_counts().to_dict()}")
    if report["status"].eq("FAIL").any():
        failed = report.loc[report["status"].eq("FAIL"), "video_id"].tolist()
        raise RuntimeError(f"Fixed-window extraction failed for: {failed}")


if __name__ == "__main__":
    main()
