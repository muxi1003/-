from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import paper_repro_rr as rr
from evaluate_lindian_adaptive_roi import radius_policy_table


class MotionRobustPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            rr.fast_fusion_quality_config(True),
            motion_center_step_threshold=0.20,
            motion_scale_step_threshold=0.12,
            motion_angle_step_threshold=0.18,
            motion_mad_multiplier=6.0,
            motion_mask_radius=0,
        )

    def test_abrupt_nostril_geometry_change_is_flagged(self) -> None:
        temp_df = pd.DataFrame(
            {
                "left_x": [10.0, 10.0, 10.0, 30.0, 30.0],
                "left_y": [10.0] * 5,
                "right_x": [0.0, 0.0, 0.0, 10.0, 10.0],
                "right_y": [10.0] * 5,
            }
        )
        motion = rr.nostril_motion_features(temp_df, self.config)
        self.assertFalse(bool(motion.loc[2, "motion_artifact_raw"]))
        self.assertTrue(bool(motion.loc[3, "motion_artifact_raw"]))
        self.assertTrue(bool(motion.loc[3, "motion_center_flag"]))
        self.assertTrue(bool(motion.loc[3, "motion_scale_flag"]))

    def test_motion_signal_coupling_detects_aligned_temperature_jump(self) -> None:
        signal = np.asarray([0.0, 0.1, 0.2, 1.2, 1.3, 1.4])
        motion = np.asarray([False, False, False, True, False, False])
        coupling = rr.motion_signal_coupling(signal, motion)
        self.assertGreater(coupling, 0.0)
        self.assertEqual(rr.motion_signal_coupling(signal, np.zeros(6, dtype=bool)), 0.0)

    def test_adaptive_radius_tracks_spacing_and_damping_reduces_range(self) -> None:
        table = pd.DataFrame(
            {
                "frame_name": [f"frame_{index:06d}.jpg" for index in range(4)],
                "left_x": [10.0, 10.0, 20.0, 20.0],
                "left_y": [0.0] * 4,
                "right_x": [0.0] * 4,
                "right_y": [0.0] * 4,
                "left_conf": [1.0] * 4,
                "right_conf": [1.0] * 4,
                "left_source": ["detected"] * 4,
                "right_source": ["detected"] * 4,
                "status": ["ok"] * 4,
            }
        )
        for side in ["left", "right"]:
            for radius in range(14, 29):
                table[f"{side}_temp_r{radius}"] = float(radius)
        _, linear = radius_policy_table(
            table,
            base_radius=20,
            exponent=1.0,
            clip_low=14,
            clip_high=28,
        )
        _, damped = radius_policy_table(
            table,
            base_radius=20,
            exponent=0.5,
            clip_low=14,
            clip_high=28,
        )
        self.assertGreater(int(linear.max() - linear.min()), int(damped.max() - damped.min()))


if __name__ == "__main__":
    unittest.main()
