"""Synthetic timestamp cases, not empirical animal accuracy evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "02_冻结独立测试"))
from timestamp_adapter import sample_map


class TimestampAdapterTests(unittest.TestCase):
    def test_normal_terminal_frame_at_lower_source_fps(self):
        fps = 8.651
        mapping, status = sample_map(np.arange(260)/fps, 260/fps)
        self.assertIsNotNone(mapping)
        self.assertEqual(status["prediction_status"], "ready_for_signal_candidate")

    def test_regular_grid_identity(self):
        t = np.arange(261) / 8.7
        mapping, status = sample_map(t, 30)
        self.assertEqual(status["prediction_status"], "ready_for_signal_candidate")
        np.testing.assert_array_equal(mapping.source_frame_index, np.arange(261))

    def test_long_gap_abstains_not_zero(self):
        t = np.r_[np.arange(0, 10, .1), np.arange(12, 30.1, .1)]
        mapping, status = sample_map(t, 30)
        self.assertIsNone(mapping)
        self.assertEqual(status["reason"], "long_timestamp_gap")
        self.assertIsNone(status["predicted_count"])

    def test_nonmonotonic_abstains(self):
        self.assertEqual(sample_map([0, .1, .1, .2], 30)[1]["reason"], "invalid_timestamp_sequence")

    def test_missing_tail_abstains(self):
        self.assertEqual(sample_map(np.arange(0, 29, .1), 30)[1]["reason"], "unsupported_window_end")

    def test_boundary_frame_never_sampled(self):
        mapping, _ = sample_map(np.arange(0, 30.1, .1), 30)
        self.assertTrue((mapping.source_time_seconds < 30).all())
        self.assertTrue((mapping.target_time_seconds < 30).all())

    def test_origin_not_silently_shifted(self):
        self.assertEqual(sample_map([2, 2.1, 2.2], 30)[1]["reason"], "window_origin_not_verified_zero")


if __name__ == "__main__":
    unittest.main()
