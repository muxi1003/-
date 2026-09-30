"""Train a fresh YOLO11n-Pose run only from a fully reviewed grouped corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from review_gate import HERE, REVIEWS, FRAMES, SEGMENTS, frame_key, sha256, validated_inputs


ROOT = HERE.parents[1]
PRETRAINED = ROOT / "yolo11n-pose.pt"
PRETRAINED_SHA256 = "869e83fcdfdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"


def preflight(dataset: Path, name: str) -> None:
    frames, _, result = validated_inputs()
    if not result.get("all_labels_valid"):
        raise RuntimeError(f"Manual labels not fully reviewed: {result}")
    ready_path = dataset / "READY.json"
    yaml_path = dataset / "dataset.yaml"
    if not ready_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError("Materialize a reviewed YOLO dataset before training")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    for key, path in (("frames_sha256", FRAMES), ("segments_sha256", SEGMENTS),
                      ("reviews_sha256", REVIEWS),
                      ("label_manifest_sha256", dataset / "label_hashes.csv")):
        if sha256(path) != ready[key]:
            raise ValueError(f"Prepared dataset metadata changed: {key}")
    if ready["frames"] != len(frames):
        raise ValueError("Prepared dataset frame count changed")
    labels = pd.read_csv(dataset / "label_hashes.csv", dtype=str)
    if labels.frame_key.tolist() != [frame_key(row) for row in frames.itertuples(index=False)]:
        raise ValueError("Prepared label member order changed")
    for row, label_hash in zip(frames.itertuples(index=False), labels.label_sha256):
        basename = f"{row.segment_id}__{int(row.source_frame):06d}"
        image = dataset / "images" / row.split / f"{basename}{Path(row.image_path).suffix}"
        label = dataset / "labels" / row.split / f"{basename}.txt"
        if not image.is_file() or not label.is_file() or sha256(label) != label_hash:
            raise ValueError(f"Prepared image/label missing or changed: {basename}")
    if not PRETRAINED.is_file() or sha256(PRETRAINED) != PRETRAINED_SHA256:
        raise ValueError("Pretrained YOLO11n-Pose weight is missing or changed")
    if not name or "/" in name or "\\" in name:
        raise ValueError("Run name must be one folder name")
    if (HERE / "training_runs" / name).exists():
        raise FileExistsError("Training run already exists; never overwrite prior weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="yolo_dataset_v1")
    parser.add_argument("--name", default="yolo11n_pose_grouped_v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    dataset = HERE / args.dataset
    preflight(dataset, args.name)
    print(f"Reviewed dataset ready: {dataset} ({args.name})", flush=True)
    if args.preflight_only:
        return

    os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics_config")
    from ultralytics import YOLO

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(dataset / "dataset.yaml"), epochs=args.epochs, batch=args.batch,
        imgsz=640, device=args.device, patience=50, lr0=0.001,
        weight_decay=0.0005, seed=20260929, workers=4,
        project=str(HERE / "training_runs"), name=args.name,
        exist_ok=False, save=True, plots=True,
    )


if __name__ == "__main__":
    main()
