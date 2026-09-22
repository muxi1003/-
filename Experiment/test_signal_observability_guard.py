"""Synthetic provenance tests, not animal-level validation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "02_冻结独立测试"))
from signal_observability_guard import protect_window


def sample(n=20):
    return pd.DataFrame({f"{side}_{key}": [value]*n for side in ("left", "right")
                         for key, value in [("temp",35.),("source","detected"),("conf",.9),("x",10.),("y",10.)]})


class SignalGuardTests(unittest.TestCase):
    def test_complete_direct_signal(self):
        self.assertEqual(protect_window(sample(), "left")[2]["prediction_status"], "ready")

    def test_internal_three_frame_gap_permitted(self):
        table = sample()
        table.loc[5:7, "left_source"] = "tracked"
        mask, _, status = protect_window(table, "left")
        self.assertEqual(status["prediction_status"], "ready")
        self.assertEqual(int(mask.short_gap_permitted.sum()), 3)

    def test_long_gap_abstains_even_with_finite_inferred_temperature(self):
        table = sample()
        table.loc[5:8, "left_source"] = "inferred"
        self.assertEqual(protect_window(table, "left")[2]["prediction_status"], "abstain")

    def test_boundary_not_filled(self):
        table = sample()
        table.loc[0, "left_temp"] = np.nan
        self.assertEqual(protect_window(table, "left")[2]["prediction_status"], "abstain")

    def test_unused_side_missing_does_not_reject_single_side(self):
        table = sample()
        table["right_source"] = "missing"
        self.assertEqual(protect_window(table, "left")[2]["prediction_status"], "ready")
        self.assertEqual(protect_window(table, "mean")[2]["prediction_status"], "abstain")

    def test_low_confidence_not_direct(self):
        table = sample()
        table["left_conf"] = .1
        self.assertEqual(protect_window(table, "left")[2]["prediction_status"], "abstain")

    def test_context_padding_does_not_wrap(self):
        table = sample()
        table.loc[0, "left_source"] = "missing"
        mask, _, _ = protect_window(table, "left")
        self.assertTrue(mask.unsafe_event_context.iloc[:4].all())
        self.assertFalse(mask.unsafe_event_context.iloc[-1])

    def test_missing_schema_fails_closed(self):
        with self.assertRaises(ValueError):
            protect_window(sample().drop(columns="left_source"), "left")

    def test_unknown_fusion_rejected(self):
        with self.assertRaises(ValueError):
            protect_window(sample(), "adaptive")


if __name__ == "__main__":
    unittest.main()
