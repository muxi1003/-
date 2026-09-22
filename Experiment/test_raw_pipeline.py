"""Synthetic decoding contract tests; not empirical RR accuracy evidence."""
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "02_冻结独立测试"))
from validate_raw_pipeline_internal49 import frames_from_anchor
from apply_frozen_roi_to_raw_validation import longest_missing, read_bgr_strict
import cv2


class FakeCapture:
    def __init__(self, times, opened=True):
        self.times, self.index, self.opened = times, -1, opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        self.index += 1
        return (True, np.zeros((2, 2, 3), dtype=np.uint8)) if self.index < len(self.times) else (False, None)

    def get(self, _):
        return self.times[self.index] * 1000

    def release(self):
        self.released = True


class RawPipelineTests(unittest.TestCase):
    def test_unicode_path_bgr_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "鼻孔输入.png"
            image = np.full((3, 5, 3), [10, 80, 220], dtype=np.uint8)
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            path.write_bytes(data.tobytes())
            np.testing.assert_array_equal(read_bgr_strict(path), image)

    def test_corrupt_image_fails_not_missing_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.jpg"
            path.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                read_bgr_strict(path)

    def test_failed_open_releases_capture(self):
        cap = FakeCapture([], opened=False)
        with patch("validate_raw_pipeline_internal49.cv2.VideoCapture", return_value=cap):
            with self.assertRaises(ValueError):
                list(frames_from_anchor("dummy", 0))
        self.assertTrue(cap.released)

    def test_missing_run_not_total_missing_count(self):
        self.assertEqual(longest_missing([True, True, False, True]), 2)

    def test_missing_run_empty_or_fully_observed(self):
        self.assertEqual(longest_missing([]), 0)
        self.assertEqual(longest_missing([False, False]), 0)

    def test_missing_run_all_missing(self):
        self.assertEqual(longest_missing([True] * 20), 20)

    def test_anchor_and_half_open_boundary_evidence(self):
        cap = FakeCapture([4, 5, 5.1, 5.3, 5.5, 6])
        with patch("validate_raw_pipeline_internal49.cv2.VideoCapture", return_value=cap):
            rows = list(frames_from_anchor("dummy", 1, .4))
        self.assertEqual([r[0] for r in rows], [0, 1, 2, 3])
        np.testing.assert_allclose([r[1] for r in rows], [0, .1, .3, .5])
        self.assertEqual(rows[0][2], 5)
        self.assertTrue(cap.released)

    def test_early_eof_is_not_padded(self):
        cap = FakeCapture([0, .1])
        with patch("validate_raw_pipeline_internal49.cv2.VideoCapture", return_value=cap):
            rows = list(frames_from_anchor("dummy", 0))
        self.assertEqual(len(rows), 2)
        self.assertTrue(cap.released)

    def test_anchor_beyond_eof_yields_no_frames(self):
        cap = FakeCapture([0, .1])
        with patch("validate_raw_pipeline_internal49.cv2.VideoCapture", return_value=cap):
            self.assertEqual(list(frames_from_anchor("dummy", 3)), [])
        self.assertTrue(cap.released)

    def test_early_consumer_close_releases_capture(self):
        cap = FakeCapture([0, .1])
        with patch("validate_raw_pipeline_internal49.cv2.VideoCapture", return_value=cap):
            iterator = frames_from_anchor("dummy", 0)
            next(iterator)
            iterator.close()
        self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
