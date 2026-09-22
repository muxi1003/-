"""Anchor fixed windows around content-matched labelled clips inside Lindian raw videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
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
    parser.add_argument("--source-manifest", type=Path, default=assets / "lindian_long_video_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--max-anchor-mse", type=float, default=100.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows in the output manifest and add the selected video IDs.",
    )
    return parser.parse_args()


def thumbnail(frame: np.ndarray, size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def read_reference_samples(video_path: Path, size: int) -> tuple[dict[int, np.ndarray], int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        if frame_count <= 0:
            raise RuntimeError("labelled clip has no readable frames")
        indices = sorted({0, frame_count // 2, frame_count - 1})
        references: dict[int, np.ndarray] = {}
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in indices:
                references[index] = thumbnail(frame, size)
            index += 1
    finally:
        capture.release()
    if len(references) != len(indices):
        raise RuntimeError("could not read all labelled clip anchor frames")
    return references, frame_count


def match_short_clip(
    raw_video: Path,
    labelled_clip: Path,
    thumbnail_size: int,
    candidate_count: int,
) -> dict[str, object]:
    references, short_frames = read_reference_samples(labelled_clip, thumbnail_size)
    first_reference = references[0]
    capture = cv2.VideoCapture(str(raw_video))
    try:
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        raw_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        if raw_fps <= 0 or raw_frames < short_frames:
            raise RuntimeError("raw video cannot contain the labelled clip")
        raw_thumbnails: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            raw_thumbnails.append(thumbnail(frame, thumbnail_size))
        if len(raw_thumbnails) < short_frames:
            raise RuntimeError("decoded raw video cannot contain the labelled clip")
        raw_array = np.stack(raw_thumbnails, axis=0)
        valid_starts = len(raw_array) - short_frames + 1
        first_errors = np.mean((raw_array[:valid_starts] - first_reference) ** 2, axis=(1, 2))
        candidates = np.argsort(first_errors)[: min(int(candidate_count), len(first_errors))]
        best: dict[str, object] | None = None
        for start_frame in candidates:
            sample_errors: list[float] = []
            for short_index, reference in references.items():
                raw_thumb = raw_array[int(start_frame + short_index)]
                sample_errors.append(float(np.mean((raw_thumb - reference) ** 2)))
            if not sample_errors:
                continue
            candidate = {
                "anchor_start_raw_frame": int(start_frame),
                "anchor_mse_mean": float(np.mean(sample_errors)),
                "anchor_mse_max": float(np.max(sample_errors)),
                "anchor_sample_errors": ";".join(f"{value:.6f}" for value in sample_errors),
                "raw_fps": raw_fps,
                "raw_frames": raw_frames,
                "labelled_short_frames": short_frames,
            }
            if best is None or candidate["anchor_mse_mean"] < best["anchor_mse_mean"]:
                best = candidate
    finally:
        capture.release()
    if best is None:
        raise RuntimeError("could not evaluate an anchor candidate")
    return best


def build_rows(source: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    source = source.loc[source["mapping_status"].eq("ready_for_fixed_window")].copy()
    if args.video_id:
        source = source.loc[source["video_id"].astype(str).isin(set(args.video_id))]
    rows: list[dict[str, object]] = []
    for _, item in source.sort_values("video_id").iterrows():
        row = item.to_dict()
        raw_video = Path(str(item["raw_source_path"]))
        labelled_clip = Path(str(item["labelled_clip_path"]))
        try:
            alignment = match_short_clip(
                raw_video,
                labelled_clip,
                int(args.thumbnail_size),
                int(args.candidate_count),
            )
            window_frames = max(1, int(round(float(item["fixed_window_seconds"]) * alignment["raw_fps"])))
            anchor_midpoint = alignment["anchor_start_raw_frame"] + (alignment["labelled_short_frames"] - 1) / 2.0
            start_frame = int(round(anchor_midpoint - (window_frames - 1) / 2.0))
            start_frame = min(max(0, start_frame), max(0, int(alignment["raw_frames"]) - window_frames))
            row.update(alignment)
            row.update(
                {
                    "anchored_window_start_frame": start_frame,
                    "anchored_window_start_seconds": start_frame / alignment["raw_fps"],
                    "anchored_window_seconds": window_frames / alignment["raw_fps"],
                    "anchored_window_frames": window_frames,
                    "anchor_status": (
                        "ready_for_anchored_fixed_window"
                        if alignment["anchor_mse_mean"] <= float(args.max_anchor_mse)
                        else "anchor_similarity_failed"
                    ),
                }
            )
        except Exception as error:
            row.update({"anchor_status": "anchor_processing_failed", "anchor_issue": str(error)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_annotation_template(manifest: pd.DataFrame, output_path: Path) -> None:
    ready = manifest.loc[manifest["anchor_status"].eq("ready_for_anchored_fixed_window")].copy()
    template = pd.DataFrame(
        {
            "window_id": [
                f"{video_id}_anchored_{int(round(window_seconds))}s"
                for video_id, window_seconds in zip(ready["video_id"], ready["anchored_window_seconds"])
            ],
            "video_id": ready["video_id"],
            "raw_source_path": ready["raw_source_path"],
            "window_start_seconds": ready["anchored_window_start_seconds"],
            "window_seconds": ready["anchored_window_seconds"],
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
    args = parse_args()
    if args.thumbnail_size < 8:
        raise ValueError("--thumbnail-size must be at least 8")
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be positive")
    source = pd.read_csv(args.source_manifest, dtype={"video_id": str})
    required = {
        "video_id",
        "mapping_status",
        "labelled_clip_path",
        "raw_source_path",
        "fixed_window_seconds",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "lindian_anchored_fixed30_manifest.csv"
    report_path = args.output_dir / "lindian_anchored_fixed30_manifest.md"
    template_path = args.output_dir / "lindian_anchored_fixed30_window_annotation_template.csv"
    existing = pd.DataFrame()
    if args.resume and manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype={"video_id": str})
        source = source.loc[~source["video_id"].isin(set(existing["video_id"]))].copy()
    new_rows = build_rows(source, args)
    result = pd.concat([existing, new_rows], ignore_index=True)
    if not result.empty:
        result = result.drop_duplicates("video_id", keep="last").sort_values("video_id").reset_index(drop=True)
    result.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    write_annotation_template(result, template_path)
    status_counts = result["anchor_status"].value_counts(dropna=False)
    report = [
        "# Lindian Anchored Fixed-Window Manifest",
        "",
        "Each window is centred on the content-matched short clip within its source raw video. Matching uses three downsampled image anchors and does not use RR values.",
        "",
        "| status | videos |",
        "| --- | ---: |",
        *[f"| {status} | {count} |" for status, count in status_counts.items()],
    ]
    ready = result.loc[result["anchor_status"].eq("ready_for_anchored_fixed_window")]
    if not ready.empty:
        report.extend(
            [
                "",
                f"- Maximum accepted anchor mean MSE: {float(args.max_anchor_mse):.3f}",
                f"- Observed median anchor mean MSE: {ready['anchor_mse_mean'].median():.3f}",
                f"- Observed maximum anchor mean MSE: {ready['anchor_mse_mean'].max():.3f}",
            ]
        )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved anchored manifest: {manifest_path.resolve()}")
    print(f"Saved anchored report: {report_path.resolve()}")
    print(f"Saved anchored annotation template: {template_path.resolve()}")
    print(status_counts.to_string())


if __name__ == "__main__":
    main()
