from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Extract external mp4 clips listed in the fieldwork worksheet into "
            "one frame-folder per external_video_id for paper_repro_rr.py."
        )
    )
    parser.add_argument(
        "--fieldwork-csv",
        type=Path,
        default=default_assets / "paper_external_validation_split_all_use_fieldwork_template.csv",
    )
    parser.add_argument(
        "--external-input-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-only",
        action="store_true",
        help="Extract only rows with include_in_external_validation=yes.",
    )
    parser.add_argument(
        "--frame-prefix",
        default="frame",
        help="Output frame filename prefix.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def yes(value: object) -> bool:
    return normalize_text(value).lower() in {"yes", "y", "true", "1", "include", "included"}


def extract_clip(raw_video: Path, output_dir: Path, frame_prefix: str, overwrite: bool) -> dict[str, object]:
    if not raw_video.exists():
        return {
            "status": "FAIL",
            "frames_written": 0,
            "fps": "",
            "duration_seconds": "",
            "issue": f"missing raw video: {raw_video}",
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{frame_prefix}_*.jpg"))
    if existing and not overwrite:
        return {
            "status": "SKIP",
            "frames_written": len(existing),
            "fps": "",
            "duration_seconds": "",
            "issue": "frame folder already exists; pass --overwrite to regenerate",
        }
    if existing and overwrite:
        for path in existing:
            path.unlink()

    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        return {
            "status": "FAIL",
            "frames_written": 0,
            "fps": "",
            "duration_seconds": "",
            "issue": f"could not open raw video: {raw_video}",
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = source_frames / fps if fps > 0 and source_frames > 0 else ""
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_path = output_dir / f"{frame_prefix}_{frames_written:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                return {
                    "status": "FAIL",
                    "frames_written": frames_written,
                    "fps": fps,
                    "duration_seconds": duration,
                    "issue": f"failed to write frame: {frame_path}",
                }
            frames_written += 1
    finally:
        capture.release()
    return {
        "status": "PASS" if frames_written > 0 else "FAIL",
        "frames_written": frames_written,
        "fps": fps,
        "duration_seconds": duration,
        "issue": "" if frames_written > 0 else "no frames decoded",
    }


def main() -> None:
    args = parse_args()
    fieldwork = pd.read_csv(args.fieldwork_csv, dtype=str, keep_default_na=False)
    required = {"external_video_id", "raw_video_path"}
    missing = required - set(fieldwork.columns)
    if missing:
        raise ValueError(f"Fieldwork CSV missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, item in fieldwork.iterrows():
        if args.include_only and not yes(item.get("include_in_external_validation", "")):
            continue
        external_video_id = normalize_text(item.get("external_video_id", ""))
        raw_video_path = normalize_text(item.get("raw_video_path", ""))
        if not external_video_id or not raw_video_path:
            continue
        output_dir = args.external_input_root / external_video_id
        result = extract_clip(
            Path(raw_video_path),
            output_dir,
            frame_prefix=str(args.frame_prefix),
            overwrite=bool(args.overwrite),
        )
        rows.append(
            {
                "external_video_id": external_video_id,
                "raw_video_path": raw_video_path,
                "frame_dir": str(output_dir),
                **result,
            }
        )
    report = pd.DataFrame(rows)
    args.external_input_root.mkdir(parents=True, exist_ok=True)
    report_path = args.external_input_root / "external_clip_frame_extraction_report.csv"
    report.to_csv(report_path, index=False)
    status_counts = report["status"].value_counts().to_dict() if not report.empty else {}
    print(f"Saved frame extraction report: {report_path}")
    print(f"Rows: {len(report)}; status_counts={status_counts}")


if __name__ == "__main__":
    main()
