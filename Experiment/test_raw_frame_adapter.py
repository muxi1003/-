"""Image preprocessing contract tests; synthetic images are not research results."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "02_冻结独立测试"))
from raw_frame_adapter import canonical_bgr


class RawFrameAdapterTests(unittest.TestCase):
    def test_matches_legacy_disk_jpeg95(self):
        image = np.random.default_rng(21).integers(0, 256, (80, 60, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            path = str(Path(root)/"legacy.jpg")
            self.assertTrue(cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95]))
            np.testing.assert_array_equal(canonical_bgr(image), cv2.imread(path))

    def test_preserves_native_geometry_and_channel_order(self):
        image = np.empty((80, 60, 3), dtype=np.uint8)
        image[:] = [15, 90, 220]
        result = canonical_bgr(image)
        self.assertEqual(result.shape, image.shape)
        self.assertLess(abs(float(result[...,0].mean())-15), 3)
        self.assertLess(abs(float(result[...,2].mean())-220), 3)

    def test_rejects_grayscale(self):
        with self.assertRaises(ValueError):
            canonical_bgr(np.zeros((10,10), dtype=np.uint8))

    def test_rejects_float_temperature_matrix(self):
        with self.assertRaises(ValueError):
            canonical_bgr(np.zeros((10,10,3), dtype=float))


if __name__ == "__main__":
    unittest.main()
