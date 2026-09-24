"""Cut nonoverlapping source-time windows while rejecting timestamp gaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
from bisect import bisect_left
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--max-edge-seconds", type=float, default=0.25)
    parser.add_argument("--browser-webm", action="store_true", help="Encode VP8 WebM viewing copies")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_timestamps(path: Path) -> tuple[list[float], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {path}")
    timestamps: list[float] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        capture.release()
    if fps <= 0 or not timestamps or min(size) <= 0:
        raise RuntimeError("Source has invalid video metadata")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError("Source timestamps are not strictly increasing")
    return timestamps, fps, size


def window_rows(
    timestamps: list[float], duration: float, max_gap: float, max_edge: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number in range(int(timestamps[-1] // duration)):
        start = number * duration
        end = start + duration
        first = bisect_left(timestamps, start)
        stop = bisect_left(timestamps, end)
        selected = timestamps[first:stop]
        observed_gap = max((b - a for a, b in zip(selected, selected[1:])), default=0.0)
        start_offset = selected[0] - start if selected else duration
        end_margin = end - selected[-1] if selected else duration
        reasons = []
        if len(selected) < 2:
            reasons.append("too_few_frames")
        if start_offset > max_edge or end_margin > max_edge:
            reasons.append("edge_not_covered")
        if observed_gap > max_gap:
            reasons.append("timestamp_gap")
        rows.append(
            {
                "segment_id": f"{int(start):03d}_{int(end):03d}",
                "source_start_seconds": start,
                "source_end_seconds": end,
                "first_source_frame": first,
                "stop_source_frame_exclusive": stop,
                "source_frame_count": len(selected),
                "first_source_pts_seconds": selected[0] if selected else "",
                "last_source_pts_seconds": selected[-1] if selected else "",
                "max_interframe_gap_seconds": observed_gap,
                "status": "CUT" if not reasons else "SKIP",
                "reason": ";".join(reasons),
                "playback_fps": len(selected) / duration if not reasons else "",
                "clip_path": "",
            }
        )
    return rows


def cut_windows(
    source: Path,
    output_root: Path,
    rows: list[dict[str, object]],
    timestamps: list[float],
    size: tuple[int, int],
    browser_webm: bool,
) -> list[dict[str, object]]:
    selected = [row for row in rows if row["status"] == "CUT"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen source: {source}")
    writer = None
    frame_map: list[dict[str, object]] = []
    row_index = 0
    try:
        for source_index, expected_pts in enumerate(timestamps):
            ok, frame = capture.read()
            if not ok or frame.shape[1::-1] != size:
                raise RuntimeError(f"Source decode failed at frame {source_index}")
            decoded_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if abs(decoded_pts - expected_pts) > 0.02:
                raise RuntimeError(f"Sequential timestamp mismatch at frame {source_index}")
            if row_index >= len(selected):
                continue
            row = selected[row_index]
            first = int(row["first_source_frame"])
            stop = int(row["stop_source_frame_exclusive"])
            if source_index == first:
                suffix = ".webm" if browser_webm else ".mp4"
                codec = "VP80" if browser_webm else "mp4v"
                output = output_root / f"{source.stem}_{row['segment_id']}{suffix}"
                fps = float(row["playback_fps"])
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output: {output}")
                row["clip_path"] = str(output)
            if writer is None:
                continue
            fps = float(row["playback_fps"])
            writer.write(frame)
            frame_map.append(
                {
                    "segment_id": row["segment_id"],
                    "output_frame": source_index - first,
                    "playback_seconds": (source_index - first) / fps,
                    "source_frame": source_index,
                    "source_pts_seconds": expected_pts,
                }
            )
            if source_index + 1 == stop:
                writer.release()
                writer = None
                row_index += 1
        if row_index != len(selected):
            raise RuntimeError("Source ended before all selected windows were encoded")
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    return frame_map


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0 or args.max_gap_seconds <= 0 or args.max_edge_seconds <= 0:
        raise ValueError("Duration and gap limits must be positive")
    source = args.source.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    timestamps, source_fps, size = read_timestamps(source)
    rows = window_rows(
        timestamps, args.window_seconds, args.max_gap_seconds, args.max_edge_seconds
    )
    output_root.mkdir(parents=True, exist_ok=True)
    frame_map = cut_windows(source, output_root, rows, timestamps, size, args.browser_webm)
    source_digest = sha256(source)
    for row in rows:
        row["source_path"] = str(source)
        row["source_sha256"] = source_digest
        row["source_fps_metadata"] = source_fps
        row["view_codec"] = "VP8/WebM" if args.browser_webm else "MPEG-4 Part 2/MP4"
    write_csv(output_root / "segments.csv", rows)
    write_csv(output_root / "frame_time_map.csv", frame_map)
    print(f"Source: {source}")
    print(f"Windows: {sum(row['status'] == 'CUT' for row in rows)} cut, {sum(row['status'] == 'SKIP' for row in rows)} skipped")
    for row in rows:
        print(f"{row['segment_id']}: {row['status']} {row['reason']}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
