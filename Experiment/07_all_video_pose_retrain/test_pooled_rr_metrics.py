import unittest

import numpy as np
import pandas as pd

from pooled_rr_metrics import ARMS, aggregate, paired


def example_predictions():
    rows = []
    for index, error in enumerate((0., 2., 2., 2.)):
        for arm in ARMS:
            truth = 10. + index * 2
            reduction = 1. if index else 0.
            final_error = error - reduction if arm == "C1P0" else error
            rows.append({"segment_id": f"window_{index}", "cohort": "small" if index == 0 else "large",
                         "split": "train" if index < 3 else "val", "cow_group": "cow_a" if index < 2 else "cow_b",
                         "arm": arm, "eligible_rr": True, "eligible_events": index == 0,
                         "truth_rr_bpm": truth, "predicted_rr_bpm": truth + final_error,
                         "absolute_rr_error_bpm": final_error, "exact_count": float(final_error == 0),
                         "mean_count_accuracy": 1 - final_error / truth,
                         "tp": 1. if index == 0 else np.nan,
                         "fp": 1. if index == 0 else np.nan,
                         "fn": 0. if index == 0 else np.nan})
    return pd.DataFrame(rows)


class PooledMetricsTests(unittest.TestCase):
    def test_pool_is_window_weighted_and_r2_is_recomputed(self):
        source = example_predictions()
        result = aggregate(source).query("split == 'all' and arm == 'C0P0'").iloc[0]
        self.assertAlmostEqual(result.rr_mae_bpm, 1.5)
        macro = source[source.arm.eq("C0P0")].groupby("cohort").absolute_rr_error_bpm.mean().mean()
        self.assertAlmostEqual(macro, 1.)
        self.assertNotEqual(result.rr_mae_bpm, macro)
        self.assertAlmostEqual(result.rr_rmse_bpm, np.sqrt(3))
        self.assertAlmostEqual(result.rr_r2, .4)
        self.assertEqual(result.exact_count, 1)
        self.assertAlmostEqual(result.exact_count_rate, .25)

    def test_event_denominator_is_not_all_rr_windows(self):
        result = aggregate(example_predictions()).query("split == 'all' and arm == 'C0P0'").iloc[0]
        self.assertEqual(result.scored_windows, 4)
        self.assertEqual(result.event_scored_windows, 1)
        self.assertAlmostEqual(result.event_f1, 2 / 3)

    def test_cluster_pairing_merges_cow_across_cohorts(self):
        source = example_predictions()
        output = paired(source, draws=100, seed=17)
        row = output[(output.split.eq("all")) & output.candidate.eq("C1P0") & output.control.eq("C0P0")].iloc[0]
        self.assertEqual(row.cow_groups, 2)
        self.assertEqual(row.windows, 4)
        self.assertAlmostEqual(row.delta_mae_bpm, -.75)
        self.assertAlmostEqual(row.relative_mae_reduction, .5)
        draws = np.random.default_rng(17).integers(0, 2, size=(100, 2))
        boot = np.array([-1., -2.])[draws].sum(axis=1) / np.array([2, 2])[draws].sum(axis=1)
        np.testing.assert_allclose([row.ci95_low, row.ci95_high], np.quantile(boot, [.025, .975]))

    def test_empty_scoring_group_and_constant_reference_do_not_get_fake_r2(self):
        source = example_predictions()
        source.loc[source.split.eq("val"), "eligible_rr"] = False
        source["truth_rr_bpm"] = 10.
        output = aggregate(source)
        self.assertFalse(output.split.eq("val").any())
        self.assertTrue(output.rr_r2.isna().all())


if __name__ == "__main__":
    unittest.main()
