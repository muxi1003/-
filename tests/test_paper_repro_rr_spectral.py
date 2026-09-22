from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import paper_repro_rr as rr
from evaluate_paper73_butterworth_spectral import spectral_features


class ButterworthSpectralTests(unittest.TestCase):
    def test_butterworth_preserves_length_and_suppresses_high_frequency(self) -> None:
        fps = 10.0
        time = np.arange(500, dtype=float) / fps
        low = np.sin(2 * np.pi * 0.7 * time)
        high = 0.8 * np.sin(2 * np.pi * 3.0 * time)
        filtered, offset = rr.smooth_curve(
            low + high,
            3,
            method="butterworth",
            fps=fps,
            butterworth_order=4,
            butterworth_cutoff_hz=1.5,
        )
        self.assertEqual(len(filtered), len(time))
        self.assertEqual(offset, 0)
        low_error = float(np.sqrt(np.mean((filtered - low) ** 2)))
        raw_error = float(np.sqrt(np.mean((low + high - low) ** 2)))
        self.assertLess(low_error, raw_error * 0.2)

    def test_spectral_estimate_recovers_known_frequency(self) -> None:
        fps = 8.7
        time = np.arange(260, dtype=float) / fps
        signal = np.sin(2 * np.pi * 0.8 * time)
        features = spectral_features(signal, fps, 0.2, 1.8)
        self.assertAlmostEqual(features["dominant_frequency_hz"], 0.8, delta=0.01)
        self.assertAlmostEqual(features["spectral_rr_bpm"], 48.0, delta=0.6)

    def test_butterworth_rejects_cutoff_at_nyquist(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            rr.smooth_curve(
                np.arange(20, dtype=float),
                3,
                method="butterworth",
                fps=8.0,
                butterworth_cutoff_hz=4.0,
            )


if __name__ == "__main__":
    unittest.main()
