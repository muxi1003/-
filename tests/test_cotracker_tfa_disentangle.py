from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_innovation73_cotracker_tfa_disentangle as experiment


class CoTrackerTfaDisentangleTests(unittest.TestCase):
    def test_chunk_ranges_cover_sequence_with_bounded_windows(self) -> None:
        ranges = experiment.chunk_ranges(137, window_length=60, overlap=12)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 137)
        self.assertTrue(all(end - start <= 60 for start, end in ranges))
        covered = set()
        for start, end in ranges:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(137)))

    def test_constellation_centers_match_bilateral_points(self) -> None:
        left = np.asarray([100.0, 80.0])
        right = np.asarray([200.0, 80.0])
        points = experiment.face_constellation(left, right)
        np.testing.assert_allclose(np.mean(points[:5], axis=0), left, atol=1e-6)
        np.testing.assert_allclose(np.mean(points[5:], axis=0), right, atol=1e-6)

    def test_low_confidence_second_point_can_only_seed_a_weak_anchor(self) -> None:
        table = pd.DataFrame(
            {
                "left_x": [100.0, 101.0],
                "left_y": [80.0, 80.0],
                "right_x": [200.0, 201.0],
                "right_y": [80.0, 80.0],
                "left_conf": [0.99, 0.99],
                "right_conf": [0.01, 0.02],
            }
        )
        self.assertFalse(experiment.valid_pair(table).any())
        self.assertTrue(experiment.finite_plausible_pair(table).all())
        self.assertEqual(experiment.choose_chunk_anchor(table, 0, 2, 2), 0)
        reference_index, _, _ = experiment.choose_reference_pair(table)
        self.assertIn(reference_index, (0, 1))

    def test_short_repair_rejects_boundary_and_long_runs(self) -> None:
        direct = np.asarray([False, True, False, True, False, False, False, False, True])
        eligible = np.ones(len(direct), dtype=bool)
        selected = experiment.short_internal_repair_mask(direct, eligible, max_gap=3)
        np.testing.assert_array_equal(
            selected,
            np.asarray([False, False, True, False, False, False, False, False, False]),
        )

    def test_selective_candidates_only_modify_gated_frames(self) -> None:
        frames = 7
        source = pd.DataFrame(
            {
                "frame_name": [f"frame_{index:06d}.jpg" for index in range(frames)],
                "left_x": np.linspace(100.0, 106.0, frames),
                "left_y": np.full(frames, 80.0),
                "right_x": np.linspace(180.0, 186.0, frames),
                "right_y": np.full(frames, 80.0),
                "left_temp": np.full(frames, 35.0),
                "right_temp": np.full(frames, 35.0),
                "left_conf": np.ones(frames),
                "right_conf": np.ones(frames),
                "left_source": ["detected"] * frames,
                "right_source": ["detected"] * 3 + ["low_conf_inferred"] + ["detected"] * 3,
            }
        )
        tracks = pd.DataFrame(
            {
                "left_visible_fraction": np.ones(frames),
                "right_visible_fraction": np.ones(frames),
                "left_tracker_rejected": np.zeros(frames, dtype=bool),
                "right_tracker_rejected": np.zeros(frames, dtype=bool),
            }
        )
        joint = source.copy()
        joint[["left_temp", "right_temp"]] += 1.0
        tfa = source.copy()
        tfa[["left_temp", "right_temp"]] += 2.0
        disentangled = source.copy()
        disentangled[["left_temp", "right_temp"]] += 3.0
        diagnostics = pd.DataFrame(
            {
                "registration_applied": np.ones(frames, dtype=bool),
                "registration_source": ["joint_points"] * frames,
                "registration_residual_px": np.ones(frames),
            }
        )
        baseline_row = pd.Series(
            {
                "selected_fusion_mode": "left",
                "smooth_window": 3,
                "peak_distance": 5,
                "selected_peak_prominence": 0.05,
            }
        )
        motion = pd.DataFrame(
            {"motion_artifact": [False, False, True, True, True, False, False]}
        )
        with mock.patch.object(experiment.rr, "nostril_motion_features", return_value=motion):
            repair, motion_tfa, final, stats = experiment.conservative_selective_candidates(
                source, tracks, joint, tfa, disentangled, diagnostics, baseline_row
            )
        self.assertEqual(float(repair.loc[3, "right_temp"]), 36.0)
        self.assertEqual(stats["right_repair_frames"], 1)
        self.assertEqual(float(motion_tfa.loc[2, "left_temp"]), 35.5)
        self.assertEqual(float(motion_tfa.loc[3, "right_temp"]), 36.0)
        self.assertEqual(float(final.loc[2, "left_temp"]), 36.0)
        self.assertEqual(stats["left_tfa_frames"], 3)
        self.assertEqual(stats["right_tfa_frames"], 2)

    def test_pair_similarity_maps_current_pair_to_reference(self) -> None:
        source_left = np.asarray([10.0, 30.0])
        source_right = np.asarray([10.0, 50.0])
        target_left = np.asarray([100.0, 100.0])
        target_right = np.asarray([140.0, 100.0])
        matrix = experiment.pair_similarity_matrix(
            source_left, source_right, target_left, target_right
        )
        self.assertIsNotNone(matrix)
        mapped = cv2.transform(
            np.asarray([[source_left, source_right]], dtype=np.float32), matrix
        )[0]
        np.testing.assert_allclose(mapped[0], target_left, atol=1e-5)
        np.testing.assert_allclose(mapped[1], target_right, atol=1e-5)

    def test_motion_disentanglement_reduces_known_geometry_nuisance(self) -> None:
        fps = 8.7
        frames = 240
        time = np.arange(frames) / fps
        motion = np.sin(2 * np.pi * 0.82 * time)
        respiration = np.sin(2 * np.pi * 1.27 * time + 0.4)
        left_x = 100.0 + 12.0 * motion
        right_x = left_x + 80.0
        true_temperature = 35.0 + 0.10 * respiration
        contaminated = true_temperature + 0.32 * motion
        table = pd.DataFrame(
            {
                "left_x": left_x,
                "left_y": np.full(frames, 80.0),
                "right_x": right_x,
                "right_y": np.full(frames, 80.0),
                "left_temp": contaminated,
                "right_temp": contaminated,
            }
        )
        corrected, diagnostics = experiment.disentangle_motion_component(table, fps=fps)
        raw_error = float(np.mean((contaminated - true_temperature) ** 2))
        corrected_error = float(
            np.mean((corrected["left_temp"].to_numpy() - true_temperature) ** 2)
        )
        self.assertGreater(diagnostics["left_motion_explained_cv"], 0.15)
        self.assertGreater(diagnostics["left_disentangle_gain"], 0.0)
        self.assertLess(corrected_error, raw_error)


if __name__ == "__main__":
    unittest.main()
