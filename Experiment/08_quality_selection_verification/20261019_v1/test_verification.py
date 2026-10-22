import unittest

import numpy as np
import pandas as pd

import run_verification as run


def temperature_table(left, right):
    return pd.DataFrame({"left_temp": left, "right_temp": right,
                         "left_source": ["detected" if np.isfinite(x) else "missing" for x in left],
                         "right_source": ["detected" if np.isfinite(x) else "missing" for x in right]})


class VerificationTests(unittest.TestCase):
    def test_filling_policies_include_endpoints(self):
        values = pd.Series([np.nan, 0., np.nan, 1., np.nan])
        np.testing.assert_allclose(run.fill_curve(values, "HALF"), [.5, 0, .5, 1, .5])
        np.testing.assert_allclose(run.fill_curve(values, "LINEAR"), [0, 0, .5, 1, 1])
        np.testing.assert_allclose(run.fill_curve(pd.Series([np.nan] * 5), "LINEAR"), [.5] * 5)
        with self.assertRaises(ValueError):
            run.fill_curve(values, "unknown")

    def test_single_side_and_bilateral_missing(self):
        table = temperature_table([30., 32., np.nan, 34., 30., 31., 32., 30.], [np.nan] * 8)
        result, quality = run.predict_signals(table)
        self.assertEqual(len(result), 16)
        self.assertEqual(len(quality), 10)
        for mode in ("MEAN", "MAX", "MIN", "LEFT"):
            np.testing.assert_allclose(result[f"{mode}_LINEAR"]["values"], result["LEFT_LINEAR"]["values"])
        np.testing.assert_allclose(result["RIGHT_LINEAR"]["values"], [.5] * 8)
        empty, _ = run.predict_signals(temperature_table([np.nan] * 8, [np.nan] * 8))
        for item in empty.values():
            self.assertEqual(len(run.peaks_from_curve(item["values"])[1]), 0)

    def test_constant_curve_and_tie_order(self):
        table = temperature_table([31.] * 20, [32.] * 20)
        result, _ = run.predict_signals(table)
        self.assertEqual(result["SELECT_LINEAR"]["mode"], "mean")
        self.assertEqual(result["SELECT_HALF"]["mode"], "mean")
        for arm in run.ARMS:
            self.assertEqual(len(run.peaks_from_curve(result[arm]["values"])[1]), 0)

    def test_ablation_removes_exactly_one_term(self):
        quality = {"candidate_median_prominence": .1, "candidate_amplitude": .8,
                   "candidate_interval_cv": .2, "candidate_missing_rate": .25}
        self.assertAlmostEqual(run.score_value(quality), 1.1)
        self.assertAlmostEqual(run.score_value(quality, "prominence"), 1.)
        self.assertAlmostEqual(run.score_value(quality, "amplitude"), -.5)
        self.assertAlmostEqual(run.score_value(quality, "interval_cv"), 1.2)
        self.assertAlmostEqual(run.score_value(quality, "missing"), 1.6)

    def test_p0_center_frame_mapping_and_duration_formula(self):
        values = (np.sin(np.arange(90) * 2 * np.pi / 12) + 1) / 2
        smoothed, indices = run.peaks_from_curve(values)
        self.assertEqual(len(smoothed), 88)
        self.assertTrue(np.all((indices >= 1) & (indices < len(values) - 1)))
        count = len(indices)
        self.assertAlmostEqual(60 * count / 30., 2 * count)
        self.assertAlmostEqual(60 * count / 29.9, count * (60 / 29.9))

    def test_rr_metrics_recompute_sse_not_average_r2(self):
        frame = pd.DataFrame({"truth_rr_bpm": [10., 20., 30.], "predicted_rr_bpm": [12., 18., 30.],
                              "cow_group": ["a", "a", "b"], "exact_count": [0, 0, 1],
                              "mean_count_accuracy": [.8, .9, 1.], "tp": [2, 3, 4], "fp": [1, 0, 1], "fn": [0, 1, 0]})
        result = run.measurements(frame)
        self.assertAlmostEqual(result["rr_mae_bpm"], 4 / 3)
        self.assertAlmostEqual(result["rr_r2"], .96)
        self.assertAlmostEqual(result["event_f1"], 18 / 21)
        self.assertAlmostEqual(result["mean_count_accuracy"], .9)
        self.assertTrue(np.isnan(run.measurements(frame.iloc[:1])["rr_r2"]))

    def test_cluster_pairing_keeps_whole_cow_and_checks_membership(self):
        base = pd.DataFrame({"segment_id": ["a1", "a2", "b1"], "cow_group": ["a", "a", "b"],
                             "absolute_rr_error_bpm": [3., 3., 3.]})
        selected = base.copy()
        selected["absolute_rr_error_bpm"] = [2., 2., 1.]
        result = run.paired_ci(selected, base, draws=1000, seed=3)
        self.assertEqual(result["cow_groups"], 2)
        self.assertAlmostEqual(result["delta_mae_bpm"], -4 / 3)
        self.assertAlmostEqual(result["ci95_low"], -2.)
        self.assertAlmostEqual(result["ci95_high"], -1.)
        with self.assertRaises(ValueError):
            run.paired_ci(selected.iloc[:2], base)

    def test_global_fixed_mode_ignores_validation_errors_and_uses_declared_ties(self):
        rows = []
        for mode in run.MODES:
            for split in ("train", "val"):
                rows.append({"arm": f"{mode.upper()}_LINEAR", "split": split,
                             "absolute_rr_error_bpm": 2. if split == "train" else (0. if mode == "right" else 100.)})
        chosen, _ = run.global_fixed(pd.DataFrame(rows))
        self.assertEqual(chosen["mode"], "mean")


if __name__ == "__main__":
    unittest.main()
