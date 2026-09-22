import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "02_冻结独立测试"))
from score_external_counts import join_reference, cluster_intervals
from frozen_rr_evaluation import validate_predictions


class ExternalTransferTests(unittest.TestCase):
    def setUp(self):
        self.windows = pd.DataFrame([dict(window_id="a", source_path="E:/a.mp4", start_seconds="0", duration_seconds="30", verified_cow_id="1")])
        self.pred = pd.DataFrame([dict(window_id="a", prediction_status="ok", predicted_count="10", duration_seconds="30", method_manifest_sha256="h")])
        self.labels = pd.DataFrame([dict(window_id="a", source_path="E:/a.mp4", source_start_seconds="0", duration_seconds="30", annotation_round="R1", annotation_status="complete", manual_breath_count="10", predictions_hidden="true", annotator="1")])

    def test_complete_join(self):
        self.assertEqual(len(join_reference(self.pred, self.windows, self.labels)), 1)

    def test_reject_reference_origin_change(self):
        self.labels.loc[0, "source_start_seconds"] = "1"
        with self.assertRaises(ValueError):
            join_reference(self.pred, self.windows, self.labels)

    def test_reject_reference_duration_change(self):
        self.labels.loc[0, "duration_seconds"] = "20"
        with self.assertRaises(ValueError):
            join_reference(self.pred, self.windows, self.labels)

    def test_reject_unobservable_zero(self):
        self.labels.loc[0, ["annotation_status", "manual_breath_count"]] = ["unobservable", "0"]
        with self.assertRaises(ValueError):
            join_reference(self.pred, self.windows, self.labels)

    def test_allow_unobservable_blank(self):
        self.labels.loc[0, ["annotation_status", "manual_breath_count"]] = ["unobservable", ""]
        self.assertEqual(len(join_reference(self.pred, self.windows, self.labels)), 1)

    def test_reject_nonblind_complete(self):
        self.labels.loc[0, "predictions_hidden"] = "false"
        with self.assertRaises(ValueError):
            join_reference(self.pred, self.windows, self.labels)

    def test_reject_missing_prediction(self):
        with self.assertRaises(ValueError):
            validate_predictions(self.pred.iloc[:0], self.windows, "h")

    def test_reject_abstain_zero(self):
        self.pred.loc[0, ["prediction_status", "predicted_count"]] = ["abstain", "0"]
        with self.assertRaises(ValueError):
            validate_predictions(self.pred, self.windows, "h")

    def test_bootstrap_keeps_cow_groups(self):
        table = pd.DataFrame(dict(verified_cow_id=["a", "a", "b"], truth_count=[10, 20, 30], predicted_count=[10, 20, 30], truth_rr_bpm=[20, 40, 60], predicted_rr_bpm=[20, 40, 60]))
        result = cluster_intervals(table, draws=20)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["intervals"]["rr_mae_bpm"]["high"], 0)


if __name__ == "__main__":
    unittest.main()
