import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock

from inspect_pipeline import observed_state
from resume_grouped_pose import check_checkpoint
from run_lindian_pipeline import wait_with_heartbeat


class PipelineRecoveryTests(unittest.TestCase):
    def test_saved_training_with_dead_parent_is_interrupted(self):
        self.assertEqual(observed_state({"stage": "training", "child_pid": 7}, False, False, 1), "interrupted")

    def test_live_process_with_stale_heartbeat_is_not_claimed_running(self):
        self.assertEqual(observed_state({"stage": "training", "child_pid": 7}, True, True, 150), "stale_heartbeat")
        self.assertEqual(observed_state({"stage": "training", "child_pid": 7}, True, True, 3), "running")

    def test_terminal_status_is_preserved(self):
        self.assertEqual(observed_state({"stage": "complete"}, False, False, None), "complete")

    def test_wait_updates_heartbeat_without_restarting_child(self):
        child = Mock()
        child.wait.side_effect = [subprocess.TimeoutExpired("train", 30), subprocess.TimeoutExpired("train", 30), 0]
        heartbeat = Mock()
        self.assertEqual(wait_with_heartbeat(child, heartbeat), 0)
        self.assertEqual(heartbeat.call_count, 2)
        self.assertEqual(child.wait.call_count, 3)

    def fixture(self):
        dataset = Path("dataset").resolve()
        run = Path("training_runs/test").resolve()
        ckpt = {"epoch": 27, "optimizer": {"state": {}}, "ema": object(), "train_args": {
            "data": str(dataset / "dataset.yaml"), "save_dir": str(run), "name": run.name,
            "epochs": 200, "batch": 8, "imgsz": 640, "seed": 20260929, "patience": 50}}
        return ckpt, run, dataset

    def test_resume_from_28_completed_epochs(self):
        ckpt, run, dataset = self.fixture()
        self.assertEqual(check_checkpoint(ckpt, run, dataset, list(range(1, 29))), 28)

    def test_missing_optimizer_or_inconsistent_history_is_rejected(self):
        ckpt, run, dataset = self.fixture()
        with self.assertRaisesRegex(ValueError, "epochs differ"):
            check_checkpoint(ckpt, run, dataset, list(range(1, 30)))
        ckpt["optimizer"] = None
        with self.assertRaisesRegex(ValueError, "optimizer"):
            check_checkpoint(ckpt, run, dataset, list(range(1, 29)))


if __name__ == "__main__":
    unittest.main()
