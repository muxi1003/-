import tempfile
import unittest
from pathlib import Path

import pandas as pd

from replay_reviewed_corpus import summarize


class ReplayMetricsTests(unittest.TestCase):
    def test_four_arms_keep_paired_windows_and_report_both_splits(self):
        rows = []
        for arm in ("C0P0", "C1P0", "C0P1", "C1P1"):
            for index, split in enumerate(("train", "val")):
                error = 2 if arm == "C0P0" else 1
                rows.append({"arm": arm, "segment_id": str(index), "cow_group": f"cow:{index}",
                    "cohort": "test", "split": split, "eligible_rr": True, "eligible_events": True,
                    "truth_rr_bpm": 30 + index * 10, "predicted_rr_bpm": 30 + index * 10 + error,
                    "absolute_rr_error_bpm": error, "exact_count": 0, "mean_count_accuracy": .9,
                    "tp": 5, "fp": 1, "fn": 1})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summarize(output, pd.DataFrame(rows))
            metrics = pd.read_csv(output / "metrics.csv")
            intervals = pd.read_csv(output / "paired_intervals.csv")
            self.assertEqual(len(metrics), 12)
            self.assertEqual(len(intervals), 9)
            self.assertTrue(intervals.delta_mae_bpm.eq(-1).all())
            self.assertTrue(intervals.ci95_low.eq(-1).all())
            self.assertTrue(intervals.ci95_high.eq(-1).all())
            self.assertTrue(metrics.event_f1.sub(10/12).abs().lt(1e-12).all())
            self.assertTrue(metrics[metrics.split.eq("all")].scored_windows.eq(2).all())


if __name__ == "__main__":
    unittest.main()
