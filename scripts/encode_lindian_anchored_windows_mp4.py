"""Encode quality-gated Lindian anchored frame folders as directly playable MP4 files."""

from __future__ import annotations

import argparse
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
        "--frames-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "lindian_anchored30_frames",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=assets / "lindian_anchored_fixed30_manifest.csv",
    )
    parser.add_argument(
        "--quality-screen",
        type=Path,
        default=assets / "lindian_anchored30_quality_screen.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "lindian_anchored30_videos",
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def probe_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)) if opened else 0
    finally:
        capture.release()
    return {
        "output_readable": opened,
        "encoded_fps": fps,
        "encoded_frames": frames,
        "encoded_duration_seconds": frames / fps if fps > 0 else 0.0,
    }


def encode_video(
    video_id: str,
    frame_dir: Path,
    output_path: Path,
    fps: float,
    codec: str,
    overwrite: bool,
) -> dict[str, object]:
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    base = {
        "video_id": video_id,
        "frame_dir": str(frame_dir),
        "output_video": str(output_path),
        "source_frames": len(frame_paths),
        "source_fps": fps,
        "expected_duration_seconds": len(frame_paths) / fps if fps > 0 else 0.0,
        "codec": codec,
    }
    if not frame_paths:
        return {**base, "status": "FAIL", "issue": "no extracted frames"}
    if fps <= 0:
        return {**base, "status": "FAIL", "issue": "invalid source fps"}
    if len(codec) != 4:
        return {**base, "status": "FAIL", "issue": "codec must contain four characters"}
    if output_path.exists() and not overwrite:
        probe = probe_video(output_path)
        valid = bool(probe["output_readable"]) and int(probe["encoded_frames"]) == len(frame_paths)
        return {
            **base,
            **probe,
            "output_bytes": output_path.stat().st_size,
            "status": "SKIP" if valid else "FAIL",
            "issue": "existing valid MP4 retained" if valid else "existing MP4 failed validation",
        }

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return {**base, "status": "FAIL", "issue": f"could not read {frame_paths[0]}"}
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        return {**base, "status": "FAIL", "issue": f"VideoWriter could not open codec {codec}"}

    written = 0
    issue = ""
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                issue = f"could not read {frame_path}"
                break
            if frame.shape[:2] != (height, width):
                issue = f"frame dimensions changed at {frame_path}"
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    probe = probe_video(output_path)
    valid = (
        not issue
        and written == len(frame_paths)
        and bool(probe["output_readable"])
        and int(probe["encoded_frames"]) == len(frame_paths)
    )
    return {
        **base,
        **probe,
        "frames_written": written,
        "width": width,
        "height": height,
        "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "status": "PASS" if valid else "FAIL",
        "issue": "" if valid else issue or "encoded MP4 failed frame-count validation",
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    quality = pd.read_csv(args.quality_screen, dtype={"video_id": str})
    required_manifest = {"video_id", "raw_fps", "anchored_window_frames"}
    required_quality = {"video_id", "quality_gate"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if missing := required_quality - set(quality.columns):
        raise ValueError(f"Quality screen is missing columns: {sorted(missing)}")

    passed_ids = set(quality.loc[quality["quality_gate"].eq("pass"), "video_id"])
    selected = manifest.loc[manifest["video_id"].isin(passed_ids)].copy()
    if args.video_id:
        selected = selected.loc[selected["video_id"].isin(set(args.video_id))]
    if selected.empty:
        raise ValueError("No quality-gated windows selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in selected.sort_values("video_id").iterrows():
        video_id = str(row["video_id"])
        result = encode_video(
            video_id,
            args.frames_root / video_id,
            args.output_root / f"{video_id}_anchored30s.mp4",
            float(row["raw_fps"]),
            str(args.codec),
            bool(args.overwrite),
        )
        rows.append(result)
        print(f"{video_id}: {result['status']}", flush=True)

    report = pd.DataFrame(rows)
    report_path = args.output_root / "lindian_anchored30_video_encoding_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"Saved encoding report: {report_path.resolve()}")
    print(report["status"].value_counts().to_string())
    if report["status"].eq("FAIL").any():
        failed = report.loc[report["status"].eq("FAIL"), "video_id"].tolist()
        raise RuntimeError(f"MP4 encoding failed for: {failed}")


if __name__ == "__main__":
    main()
