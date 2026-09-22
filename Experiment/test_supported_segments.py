import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "02_冻结独立测试"))
from supported_segments import segments


class SegmentTests(unittest.TestCase):
    def test_full_window_not_trimmed(self):
        self.assertEqual(segments(np.ones(300),30)[0]["core_seconds"],30)

    def test_gap_not_concatenated(self):
        spans = segments(np.r_[np.ones(100),np.zeros(100),np.ones(100)],30)
        self.assertEqual(len(spans),2)
        self.assertEqual(spans[0]["core_stop_frame_exclusive"],97)
        self.assertEqual(spans[1]["core_start_frame"],203)
        self.assertAlmostEqual(sum(s["core_seconds"] for s in spans),19.4)

    def test_short_fragments_not_eligible(self):
        self.assertFalse(segments(np.r_[np.zeros(10),np.ones(50),np.zeros(10)],7)[0]["eligible_duration"])

    def test_no_support_no_segments(self):
        self.assertEqual(segments(np.zeros(100),10),[])

    def test_single_frame_margin_cannot_invert_interval(self):
        s = segments([False,True,False],.3)[0]
        self.assertEqual(s["core_seconds"],0)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            segments([True],float("nan"))


if __name__ == "__main__":
    unittest.main()
