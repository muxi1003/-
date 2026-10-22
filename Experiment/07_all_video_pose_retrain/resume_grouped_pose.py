"""Resume only a matching unfinished grouped run, after preserving its checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from review_gate import HERE, sha256
from train_grouped_pose import preflight

os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics_config")
os.environ["WANDB_MODE"] = "disabled"


def check_checkpoint(checkpoint: dict, run: Path, dataset: Path, completed_epochs: list[int]) -> int:
    epoch = checkpoint.get("epoch", -1)
    args = checkpoint.get("train_args", {})
    if epoch < 0 or not checkpoint.get("optimizer") or checkpoint.get("ema") is None:
        raise ValueError("Checkpoint lacks an unfinished epoch, optimizer or EMA; do not start fresh silently")
    if completed_epochs != list(range(1, epoch + 2)):
        raise ValueError("results.csv and checkpoint epochs differ; preserve for reconciliation")
    if Path(args.get("data", "")).resolve() != (dataset / "dataset.yaml").resolve():
        raise ValueError("Resume dataset differs from frozen checkpoint")
    if Path(args.get("save_dir", "")).resolve() != run.resolve() or args.get("name") != run.name:
        raise ValueError("Resume save directory differs from the original run")
    for key, expected in {"epochs": 200, "batch": 8, "imgsz": 640, "seed": 20260929, "patience": 50}.items():
        if args.get(key) != expected:
            raise ValueError(f"Frozen training argument changed: {key}")
    if epoch + 1 >= args["epochs"]:
        raise ValueError("Training already reached its epoch limit")
    return epoch + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    dataset, run = HERE / args.dataset, HERE / "training_runs" / args.name
    preflight(dataset, args.name, allow_existing_run=True)
    if args.checkpoint.resolve() != (run / "weights" / "last.pt").resolve():
        raise ValueError("Resume only the named run's last.pt")
    import torch
    from ultralytics import YOLO

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    completed = pd.read_csv(run / "results.csv").epoch.to_numpy(int).tolist()
    count = check_checkpoint(checkpoint, run, dataset, completed)
    print(f"RESUME_READY completed_epochs={count}; next_epoch={count + 1}; total_epochs=200", flush=True)
    if args.preflight_only:
        return
    args.backup.mkdir(parents=True, exist_ok=False)
    sources = [run / "weights/last.pt", run / "weights/best.pt", run / "results.csv", run / "args.yaml"]
    frozen = []
    for source in sources:
        target = args.backup / source.name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise ValueError(f"Recovery backup differs: {source}")
        frozen.append({"source": str(source), "backup": str(target), "sha256": sha256(target)})
    (args.backup / "recovery.json").write_text(json.dumps({"completed_epochs": count, "next_epoch": count + 1,
        "files": frozen, "restored_by_ultralytics": "optimizer, EMA, AMP scaler, epoch, scheduler position, best_fitness",
        "limitation": "The unfinished epoch is replayed; random-number and batch trajectories are not promised bitwise identical."}, indent=2), encoding="utf-8")
    del checkpoint
    model = YOLO(str(args.backup / "last.pt"))
    model.train(resume=True)


if __name__ == "__main__":
    main()
