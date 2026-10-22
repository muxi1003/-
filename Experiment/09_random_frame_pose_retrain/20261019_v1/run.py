"""Isolated frame-random YOLO training and unchanged-rule RR replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "Experiment/07_all_video_pose_retrain"
DATA_SOURCE = OLD / "yolo_dataset_lindian_20261016_v1"
BASELINE = OLD / "replay_lindian_20261016_v1"
DATASET = HERE / "dataset"
REPLAY = HERE / "rr_replay"
NAME = "yolo11n_pose_frame_random_20261019_v1"
SEED = 20260929
sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD))
os.environ["YOLO_CONFIG_DIR"] = str(HERE / "runtime/ultralytics")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        pd.DataFrame(frame).to_csv(stream, index=False)


def assignments(keys, seed=SEED):
    ordered = sorted(keys)
    if len(set(ordered)) != len(ordered) or len(ordered) < 2:
        raise ValueError("Frame keys must be unique and nonempty")
    n_val = math.ceil(.2 * len(ordered))
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    val = {ordered[index] for index in permutation[:n_val]}
    return {key: "val" if key in val else "train" for key in ordered}


def verify_sources():
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["sources"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['path']}")
    for item in frozen["generated"]:
        if sha(HERE / item["path"]) != item["sha256"]:
            raise ValueError(f"New frozen input changed: {item['path']}")
    return frozen


def prepare():
    if DATASET.exists() or REPLAY.exists() or (HERE / "freeze.json").exists():
        raise FileExistsError("Never overwrite a prepared random-frame experiment")
    ready = json.loads((DATA_SOURCE / "READY.json").read_text(encoding="utf-8"))
    ready_sources = {"frames_sha256": OLD / "corpus_v1/frames.csv",
        "segments_sha256": OLD / "corpus_v1/segments.csv",
        "reviews_sha256": OLD / "corpus_v1/manual_review_status.csv",
        "label_manifest_sha256": DATA_SOURCE / "label_hashes.csv"}
    for key, path in ready_sources.items():
        if sha(path) != ready[key]:
            raise ValueError(f"Original reviewed membership changed: {key}")
    if ready["frames"] != 40542 or ready["scope"] != "lindian":
        raise ValueError("Unexpected original prepared dataset")
    if sha(ROOT / "yolo11n-pose.pt") != "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0":
        raise ValueError("Pretrained weight changed")
    print("Verified original reviewed membership and label manifest hashes", flush=True)
    original = json.loads((BASELINE / "freeze.json").read_text(encoding="utf-8"))
    for entry in original["files"] + original["generated"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Baseline source changed: {entry['path']}")
    frames = pd.read_csv(OLD / "corpus_v1/frames.csv", dtype=str).fillna("")
    keys = frames.segment_id + ":" + frames.source_frame.map(lambda x: f"{int(x):06d}")
    frames["frame_key"] = keys
    label_hashes = pd.read_csv(DATA_SOURCE / "label_hashes.csv").set_index("frame_key")
    frames = frames[frames.frame_key.isin(label_hashes.index)].copy()
    if len(frames) != 40542 or frames.cohort.eq("R3_jiufu").any() or frames.frame_key.duplicated().any():
        raise ValueError("Prepared membership differs from the approved 40542 frames")
    assignment = assignments(frames.frame_key)
    manifest = []
    for number, row in enumerate(frames.itertuples(index=False), 1):
        basename = f"{row.segment_id}__{int(row.source_frame):06d}"
        image = DATA_SOURCE / "images" / row.split / f"{basename}{Path(row.image_path).suffix}"
        label = DATA_SOURCE / "labels" / row.split / f"{basename}.txt"
        if sha(label) != label_hashes.loc[row.frame_key, "label_sha256"]:
            raise ValueError(f"Label changed: {row.frame_key}")
        new_split = assignment[row.frame_key]
        target_image = DATASET / "images" / new_split / image.name
        target_label = DATASET / "labels" / new_split / label.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        shutil.copy2(label, target_label)
        image_sha, label_sha = sha(image), sha(label)
        if sha(target_image) != image_sha or sha(target_label) != label_sha:
            raise ValueError(f"Materialized bytes differ: {row.frame_key}")
        manifest.append({"frame_key": row.frame_key, "segment_id": row.segment_id, "source_frame": int(row.source_frame),
            "cohort": row.cohort, "cow_group": row.cow_group, "grouped_split": row.split,
            "random_split": new_split, "source_image": str(image), "source_label": str(label),
            "image": str(target_image), "label": str(target_label), "image_sha256": image_sha, "label_sha256": label_sha})
        if number % 5000 == 0:
            print(f"Copied and verified {number}/40542 frames", flush=True)
    table = pd.DataFrame(manifest)
    save(HERE / "frame_split_manifest.csv", table)
    source_group = table.groupby("segment_id").grouped_split.first()
    exposures = table.groupby(["segment_id", "random_split"]).size().unstack(fill_value=0).reindex(columns=["train", "val"], fill_value=0)
    exposures = exposures.rename(columns={"train": "train_frames", "val": "val_frames"})
    exposures["grouped_split"] = source_group
    exposures["split"] = np.where(exposures.train_frames.gt(0) & exposures.val_frames.gt(0), "mixed_frame",
                                    np.where(exposures.train_frames.gt(0), "train_only", "val_only"))
    save(HERE / "segment_exposure.csv", exposures.reset_index())
    shared_groups = table.groupby("cow_group").random_split.nunique().eq(2)
    image_hash_overlap = set(table[table.random_split.eq("train")].image_sha256) & set(table[table.random_split.eq("val")].image_sha256)
    write_json(HERE / "split_audit.json", {"frames": len(table), "train": int(table.random_split.eq("train").sum()),
        "val": int(table.random_split.eq("val").sum()), "segments": len(exposures),
        "mixed_frame_segments": int(exposures.split.eq("mixed_frame").sum()), "cow_groups": len(shared_groups),
        "cross_split_cow_groups": int(shared_groups.sum()), "cross_split_exact_image_hashes": len(image_hash_overlap),
        "cross_split_hashes": sorted(image_hash_overlap), "no_independent_test": True,
        "interpretation": "Random frame validation with temporal and identity exposure; not unseen-cow validation"})
    yaml = (f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [nose]\n"
            "kpt_shape: [2, 3]\nflip_idx: [1, 0]\n")
    (DATASET / "dataset.yaml").write_text(yaml, encoding="utf-8")
    REPLAY.mkdir()
    registry = pd.read_csv(BASELINE / "reference_registry.csv")
    registry = registry.rename(columns={"split": "grouped_split"}).merge(exposures.drop(columns="grouped_split"), left_on="segment_id", right_index=True, validate="one_to_one")
    assert len(registry) == 199 and registry.segment_id.is_unique
    save(REPLAY / "reference_registry.csv", registry)
    old_frames = pd.read_csv(BASELINE / "frames.csv", dtype=str).fillna("")
    old_frames = old_frames.rename(columns={"split": "grouped_split"})
    replay_keys = old_frames.segment_id + ":" + old_frames.source_frame.map(lambda x: f"{int(x):06d}")
    old_frames["random_split"] = replay_keys.map(assignment)
    assert old_frames.random_split.notna().all() and len(old_frames) == 40542
    image_mapping = table.set_index("frame_key")
    old_frames["source_image_path"] = old_frames.image_path
    old_frames["image_path"] = replay_keys.map(image_mapping.image)
    old_frames["label_path"] = replay_keys.map(image_mapping.label)
    old_frames["split"] = old_frames.random_split
    save(REPLAY / "frames.csv", old_frames)
    for name in ("frame_times.csv", "reference_events.csv"):
        shutil.copy2(BASELINE / name, REPLAY / name)
    prepared_files = [HERE / "frame_split_manifest.csv", HERE / "segment_exposure.csv", HERE / "split_audit.json", DATASET / "dataset.yaml"]
    prepared_files += list(REPLAY.glob("*.csv"))
    source_paths = {Path(entry["path"]) for entry in original["files"] + original["generated"]}
    source_paths.update([DATA_SOURCE / "READY.json", DATA_SOURCE / "label_hashes.csv", BASELINE / "predictions.csv",
        OLD / "train_grouped_pose.py", OLD / "replay_reviewed_corpus.py", ROOT / "models/best.pt", ROOT / "yolo11n-pose.pt",
        Path(__file__), HERE / "PROTOCOL.md", HERE / "test_run.py"])
    frozen = {"sources": [{"path": str(path), "sha256": sha(path)} for path in sorted(source_paths)],
        "generated": [{"path": str(path.relative_to(HERE)), "sha256": sha(path)} for path in sorted(prepared_files)],
        "frame_count": 40542, "seed": SEED, "sampling": "Sorted frame_key then default_rng permutation, first ceil(.2*N) val",
        "scope": "Exposed-frame development experiment; no independent test; no default adoption"}
    write_json(HERE / "freeze.json", frozen)
    write_json(REPLAY / "freeze.json", {"files": frozen["sources"],
        "generated": [{"path": str(HERE / item["path"]), "sha256": item["sha256"]} for item in frozen["generated"]],
        "arms": original["arms"], "roi_radius": original["roi_radius"], "min_temperature": original["min_temperature"],
        "keypoint_confidence": original["keypoint_confidence"], "scope": frozen["scope"], "duration_reference": "unchanged from baseline"})
    preflight_new()
    print(json.dumps({"prepared": True, "train": 32433, "val": 8109, "segments": 199}), flush=True)


def preflight_new():
    verify_sources()
    table = pd.read_csv(HERE / "frame_split_manifest.csv")
    assert len(table) == 40542 and not table.frame_key.duplicated().any()
    expected = assignments(table.frame_key)
    assert table.frame_key.map(expected).eq(table.random_split).all()
    assert table.random_split.value_counts().to_dict() == {"train": 32433, "val": 8109}
    for row in table.itertuples(index=False):
        if sha(row.image) != row.image_sha256 or sha(row.label) != row.label_sha256:
            raise ValueError(f"Dataset bytes changed: {row.frame_key}")
    print("PREFLIGHT: 40542 image/label pairs and frame split verified", flush=True)


def load_replay():
    spec = importlib.util.spec_from_file_location("historical_fixed_rr_replay", OLD / "replay_reviewed_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train():
    preflight_new()
    from ultralytics import YOLO
    os.environ["YOLO_CONFIG_DIR"] = str(HERE / "runtime/ultralytics")
    model = YOLO(str(ROOT / "yolo11n-pose.pt"))
    model.train(data=str(DATASET / "dataset.yaml"), epochs=200, batch=8, imgsz=640, device="0", patience=50,
        lr0=.001, weight_decay=.0005, seed=SEED, workers=4, project=str(HERE / "training_runs"),
        name=NAME, exist_ok=False, save=True, plots=True)


def summarize():
    baseline, current = (pd.read_csv(path / "predictions.csv") for path in (BASELINE, REPLAY))
    keys = ["segment_id", "arm"]
    assert not current.duplicated(keys).any() and len(current) == len(baseline) == 796
    paired = current.merge(baseline, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(paired) == 796
    for name in ("eligible_rr", "eligible_events", "truth_count", "duration_seconds", "cow_group", "cohort"):
        a, b = paired[f"{name}_new"], paired[f"{name}_old"]
        assert (a.eq(b) | (a.isna() & b.isna())).all(), name
    records = []
    scopes = {"primary47": lambda f: f.cohort_new.eq("anchored49"), "all82_approx30": lambda f: f.cohort_new.ne("short73"),
              "old73": lambda f: f.cohort_new.eq("short73"), "all155_mixed_duration": lambda f: pd.Series(True, index=f.index)}
    scopes.update({name: lambda f, c=name: f.cohort_new.eq(c) for name in ("P1", "P2", "P3")})
    for scope, selector in scopes.items():
        pool = paired[selector(paired) & paired.eligible_rr_new]
        for arm, group in pool.groupby("arm", sort=False):
            row = {"scope": scope, "arm": arm, "windows": len(group)}
            for source in ("old", "new"):
                y, p = group[f"truth_rr_bpm_{source}"].to_numpy(), group[f"predicted_rr_bpm_{source}"].to_numpy()
                e, sst = p-y, np.sum((y-y.mean())**2)
                ev = group[group[f"eligible_events_{source}"]]
                tp, fp, fn = (ev[f"{key}_{source}"].sum() for key in ("tp", "fp", "fn"))
                row.update({f"{source}_mae_bpm": float(abs(e).mean()), f"{source}_rmse_bpm": float(np.sqrt(np.mean(e**2))),
                    f"{source}_r2": float(1-np.sum(e**2)/sst) if sst else np.nan,
                    f"{source}_mean_count_accuracy": float(group[f"mean_count_accuracy_{source}"].mean()),
                    f"{source}_exact_count": int(group[f"exact_count_{source}"].sum()),
                    f"{source}_event_f1": float(2*tp/(2*tp+fp+fn)) if len(ev) and 2*tp+fp+fn else np.nan})
            row["delta_mae_bpm"] = row["new_mae_bpm"] - row["old_mae_bpm"]
            records.append(row)
    result = pd.DataFrame(records)
    save(HERE / "comparison_metrics.csv", result)
    fields = ["scope", "arm", "windows", "old_mae_bpm", "new_mae_bpm", "delta_mae_bpm", "old_r2", "new_r2"]
    lines = ["# 帧随机与牛号分组训练的同窗开发性RR比较", "", "全窗包含训练帧，不是独立测试；验证mAP来自不同验证成员，不能作同测试集准确度提升。",
             "旧73和155混合时长只追溯；质量选择历史补齐混杂限制不变。默认方法不自动采用。", "", "| " + " | ".join(fields) + " |", "|" + "---|"*len(fields)]
    for _, row in result.iterrows():
        lines.append("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key]) for key in fields) + " |")
    (HERE / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    verify_sources()


def pipeline():
    verify_sources()
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    if (logs / "status.json").exists():
        raise FileExistsError("A supervisor has already started; inspect before any recovery")
    status = {"pid": os.getpid(), "name": NAME, "default_changed": False, "no_independent_test": True}

    def update(stage, **extra):
        status.update(stage=stage, heartbeat_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        temp = logs / "status.tmp"
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, logs / "status.json")

    def execute(stage):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), stage]
        update(stage, child_pid=None, command=command)
        with (logs / f"{stage}.log").open("x", encoding="utf-8") as output:
            child = subprocess.Popen(command, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT)
            update(stage, child_pid=child.pid)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update(stage, child_pid=child.pid)
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; preserved log")

    try:
        execute("train")
        execute("replay")
        execute("summarize")
        assert (REPLAY / "COMPLETED.json").is_file() and (HERE / "comparison_metrics.csv").is_file()
        verify_sources()
        weights = HERE / "training_runs" / NAME / "weights/best.pt"
        write_json(HERE / "COMPLETED.json", {"status": "complete", "weight_sha256": sha(weights), "frames": 40542,
            "segments": 199, "rr_predictions": 796, "default_changed": False, "independent_test": False,
            "claim": "Exposed-frame development comparison; no automatic method or superiority adoption"})
        update("complete", child_pid=None)
    except BaseException as error:
        update("failed", child_pid=None, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "train", "replay", "summarize", "pipeline"))
    stage = parser.parse_args().stage
    if stage == "replay":
        verify_sources()
        load_replay().score(REPLAY, HERE / "training_runs" / NAME / "weights/best.pt")
    else:
        {"prepare": prepare, "verify": preflight_new, "train": train, "summarize": summarize, "pipeline": pipeline}[stage]()
