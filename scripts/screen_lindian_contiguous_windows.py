"""Screen cut windows for visible nostrils and a near-white palette proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=20)
    return parser.parse_args()


def white_roi_fraction(frame: np.ndarray, x: float, y: float, radius: int) -> float:
    height, width = frame.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    left, right = max(0, cx - radius), min(width, cx + radius + 1)
    top, bottom = max(0, cy - radius), min(height, cy + radius + 1)
    if left >= right or top >= bottom:
        return float("nan")
    yy, xx = np.ogrid[top:bottom, left:right]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    pixels = frame[top:bottom, left:right][mask]
    return float(np.mean(np.all(pixels >= 245, axis=1)))


def screen_clip(path: Path, model: YOLO, args: argparse.Namespace) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open clip: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = set(np.linspace(0, count - 1, min(count, args.samples)).round().astype(int))
    sampled = left_valid = right_valid = bilateral = 0
    white_fractions: list[float] = []
    try:
        for frame_index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Decode failed in {path} at frame {frame_index}")
            if frame_index not in sample_indices:
                continue
            sampled += 1
            result = model(frame, verbose=False)[0]
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            if result.boxes is not None and len(result.boxes.conf) == len(result.keypoints.data):
                selected = int(result.boxes.conf.argmax().item())
            else:
                selected = 0
            keypoints = result.keypoints.data[selected].detach().cpu().numpy()
            if len(keypoints) < 2:
                continue
            visible = []
            for index in (0, 1):
                point = keypoints[index]
                valid = len(point) < 3 or point[2] >= args.confidence
                visible.append(valid)
                if valid:
                    white_fractions.append(
                        white_roi_fraction(frame, float(point[0]), float(point[1]), args.radius)
                    )
            left_valid += int(visible[0])
            right_valid += int(visible[1])
            bilateral += int(all(visible))
    finally:
        capture.release()
    finite = np.asarray([value for value in white_fractions if np.isfinite(value)])
    return {
        "sampled_frames": sampled,
        "left_visible_fraction": left_valid / sampled,
        "right_visible_fraction": right_valid / sampled,
        "bilateral_visible_fraction": bilateral / sampled,
        "nostril_roi_white_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "nostril_roi_white_p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "nostril_roi_white_samples": int(finite.size),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.radius <= 0 or not 0 <= args.confidence <= 1:
        raise ValueError("Invalid screening parameters")
    segments = pd.read_csv(args.segments)
    selected = segments.loc[segments["status"].eq("CUT")].copy()
    model = YOLO(str(args.model))
    rows = []
    for row in selected.itertuples(index=False):
        metrics = screen_clip(Path(row.clip_path), model, args)
        rows.append({"segment_id": row.segment_id, "clip_path": row.clip_path, **metrics})
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.drop(columns=["clip_path"]).round(3).to_string(index=False))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
