"""Resumable local training -> fixed RR replay, with logs and liveness heartbeats."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from review_gate import HERE
from inspect_pipeline import process_alive


def wait_with_heartbeat(child, heartbeat, interval=30):
    while True:
        try:
            return child.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            heartbeat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    replay = HERE / args.replay
    if not (replay / "freeze.json").is_file():
        raise FileNotFoundError("Prepare frozen references before launching")
    root_logs = HERE / "pipeline_logs" / args.name
    status_path = root_logs / "status.json"
    if args.resume_checkpoint:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if any(process_alive(previous.get(key)) for key in ("pid", "child_pid")):
            raise RuntimeError("Previous pipeline or trainer still exists; do not start a duplicate")
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        logs = root_logs / "resumes" / timestamp
        logs.mkdir(parents=True, exist_ok=False)
        shutil.copy2(status_path, logs / "previous_status.json")
    else:
        logs = root_logs
        logs.mkdir(parents=True, exist_ok=False)
    status = {"pid": os.getpid(), "name": args.name, "dataset": args.dataset,
              "replay": str(replay), "epochs_max": args.epochs, "batch": args.batch,
              "logs": str(logs), "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
              "state_semantics": "Stage is not liveness evidence; verify PID and heartbeat via inspect_pipeline.py"}

    def update(stage, **extra):
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        status.update(stage=stage, updated_at=now, heartbeat_at=now, **extra)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, status_path)

    def execute(stage, script, arguments):
        command = [sys.executable, "-u", str(HERE / script), *arguments]
        update(stage, command=command, child_pid=None)
        with (logs / f"{stage}.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=str(HERE.parents[1]), stdout=log, stderr=subprocess.STDOUT)
            update(stage, child_pid=child.pid)
            code = wait_with_heartbeat(child, lambda: update(stage, child_pid=child.pid))
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {logs / (stage + '.log')}")

    try:
        common = ["--dataset", args.dataset, "--name", args.name]
        if args.resume_checkpoint:
            execute("resume_training", "resume_grouped_pose.py", common + [
                "--checkpoint", str(args.resume_checkpoint.resolve()), "--backup", str(logs / "checkpoint_before_resume")])
        else:
            execute("preflight", "train_grouped_pose.py", common + ["--preflight-only"])
            execute("training", "train_grouped_pose.py", common + ["--epochs", str(args.epochs), "--batch", str(args.batch)])
        weights = HERE / "training_runs" / args.name / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Training exited without best.pt: {weights}")
        execute("rr_replay", "replay_reviewed_corpus.py", ["score", "--output", str(replay), "--weights", str(weights)])
        if not (replay / "COMPLETED.json").is_file():
            raise RuntimeError("Replay exited without completion marker")
        update("complete", child_pid=None, weights=str(weights))
    except BaseException as error:
        update("failed", child_pid=None, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
