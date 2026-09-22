from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import paper_repro_rr as rr


def candidate(
    mode: str,
    count: int,
    *,
    median_prominence: float,
    amplitude: float,
    interval_cv: float,
    missing_rate: float,
    config: rr.ReproConfig,
) -> dict[str, object]:
    quality: dict[str, float | int | str] = {
        "mode": mode,
        "candidate_peaks": count,
        "candidate_median_prominence": median_prominence,
        "candidate_amplitude": amplitude,
        "candidate_interval_cv": interval_cv,
        "candidate_missing_rate": missing_rate,
    }
    quality["score"] = rr.fusion_score_from_quality(
        quality,
        amplitude_weight=config.fusion_amplitude_weight,
        interval_cv_weight=config.fusion_interval_cv_weight,
        missing_weight=config.fusion_missing_weight,
    )
    return {
        "mode": mode,
        "candidate": pd.Series([0.0, 1.0, 0.0]),
        "quality": quality,
        "final_peak_count": count,
    }


class RobustPeakPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            rr.fast_fusion_quality_config(True),
            fusion_rescue_interval_cv_weight=1.1,
            fusion_rescue_outlier_threshold=3,
            source_peak_gate=True,
            source_peak_support_radius=1,
        )

    def test_source_gate_rejects_peak_without_direct_detection(self) -> None:
        temp_df = pd.DataFrame(
            {
                "left_source": ["detected"] * 6 + ["tracked"] * 4,
                "right_source": ["detected"] * 6 + ["tracked"] * 4,
            }
        )
        peaks, rejected = rr.filter_peaks_by_source_support(
            np.asarray([3, 8]),
            temp_df,
            smooth_offset=0,
            enabled=True,
            radius=1,
        )
        self.assertEqual(peaks.tolist(), [3])
        self.assertEqual(rejected, [8])

    def test_guarded_rescue_switches_large_consensus_outlier(self) -> None:
        values = [
            candidate("mean", 28, median_prominence=0.135, amplitude=0.381, interval_cv=0.210, missing_rate=0.035, config=self.config),
            candidate("max", 27, median_prominence=0.099, amplitude=0.329, interval_cv=0.266, missing_rate=0.035, config=self.config),
            candidate("min", 27, median_prominence=0.185, amplitude=0.703, interval_cv=0.230, missing_rate=0.035, config=self.config),
            candidate("left", 23, median_prominence=0.084, amplitude=0.830, interval_cv=0.385, missing_rate=0.069, config=self.config),
            candidate("right", 28, median_prominence=0.190, amplitude=0.525, interval_cv=0.209, missing_rate=0.000, config=self.config),
        ]
        selected = rr.choose_adaptive_fusion_candidate(values, self.config)
        self.assertEqual(selected["mode"], "min")
        self.assertEqual(selected["quality"]["legacy_selected_fusion_mode"], "left")
        self.assertEqual(selected["quality"]["fusion_rescue_applied"], "True")

    def test_guarded_rescue_keeps_small_consensus_deviation(self) -> None:
        values = [
            candidate("mean", 19, median_prominence=0.308, amplitude=0.477, interval_cv=0.515, missing_rate=0.000, config=self.config),
            candidate("max", 20, median_prominence=0.142, amplitude=0.335, interval_cv=0.555, missing_rate=0.000, config=self.config),
            candidate("min", 20, median_prominence=0.424, amplitude=0.641, interval_cv=0.535, missing_rate=0.000, config=self.config),
            candidate("left", 24, median_prominence=0.142, amplitude=0.406, interval_cv=0.379, missing_rate=0.000, config=self.config),
            candidate("right", 18, median_prominence=0.410, amplitude=0.675, interval_cv=0.620, missing_rate=0.000, config=self.config),
        ]
        selected = rr.choose_adaptive_fusion_candidate(values, self.config)
        self.assertEqual(selected["mode"], "right")
        self.assertEqual(selected["quality"]["fusion_rescue_applied"], "False")
        self.assertEqual(selected["quality"]["fusion_consensus_count"], 20.0)


class ShortGapRoiRepairTests(unittest.TestCase):
    class ConstantTemperatureModel:
        def predict(self, pixels: np.ndarray) -> np.ndarray:
            return np.full(len(pixels), 30.0)

    def setUp(self) -> None:
        self.config = replace(
            rr.fast_fusion_quality_config(True),
            radius=10,
            min_temp=None,
            max_track_gap=3,
            max_track_anchor_shift=1.0,
        )
        self.images = [Path(f"frame_{index:03d}.jpg") for index in range(7)]

    @staticmethod
    def table(missing: list[int], xs: list[float] | None = None) -> pd.DataFrame:
        xs = xs or [10.0] * 7
        values = [30.0] * 7
        sources = ["detected"] * 7
        for index in missing:
            values[index] = np.nan
            sources[index] = "missing"
        return pd.DataFrame(
            {
                "left_temp": values,
                "left_x": xs,
                "left_y": [10.0] * 7,
                "left_source": sources,
                "roi_radius": [10] * 7,
            }
        )

    @patch.object(rr.cv2, "imread", return_value=np.zeros((24, 24, 3), dtype=np.uint8))
    def test_repairs_only_short_interior_gap_with_direct_anchors(self, _imread) -> None:
        table = self.table([2, 3])
        repaired = rr.repair_short_missing_runs(
            table,
            self.images,
            self.ConstantTemperatureModel(),
            "left",
            self.config,
        )
        self.assertEqual(repaired, 2)
        self.assertEqual(table.loc[2:3, "left_source"].tolist(), ["tracked", "tracked"])

    @patch.object(rr.cv2, "imread", return_value=np.zeros((24, 24, 3), dtype=np.uint8))
    def test_rejects_edge_long_and_large_motion_gaps(self, _imread) -> None:
        edge = self.table([0, 1])
        long_gap = self.table([1, 2, 3, 4])
        large_motion = self.table([2], xs=[10.0, 10.0, 10.0, 30.1, 30.1, 30.1, 30.1])
        model = self.ConstantTemperatureModel()
        self.assertEqual(rr.repair_short_missing_runs(edge, self.images, model, "left", self.config), 0)
        self.assertEqual(rr.repair_short_missing_runs(long_gap, self.images, model, "left", self.config), 0)
        self.assertEqual(rr.repair_short_missing_runs(large_motion, self.images, model, "left", self.config), 0)


if __name__ == "__main__":
    unittest.main()
