import copy
import unittest

from import_reviewed_xanylabeling import bounded_boxes, convert_pose, negative_keys


class ReviewedImportTests(unittest.TestCase):
    def sample(self):
        return {"imageWidth": 1080, "imageHeight": 1440, "shapes": [
            {"label": "nose", "shape_type": "rectangle", "points": [[-0.724, 20], [120, 450]], "group_id": 0},
            {"label": "left_nostril", "shape_type": "point", "points": [[60, 200]], "group_id": 0}]}

    def test_subpixel_box_clip_preserves_original_and_point(self):
        data = self.sample()
        original = copy.deepcopy(data)
        fixed, changes = bounded_boxes(data)
        self.assertEqual(data, original)
        self.assertEqual(fixed["shapes"][1], data["shapes"][1])
        self.assertEqual(fixed["shapes"][0]["points"][0][0], 0)
        self.assertAlmostEqual(changes[0]["old"], -0.724)
        self.assertEqual(convert_pose(fixed)[0][7], 2)

    def test_larger_box_overshoot_rejected(self):
        data = self.sample()
        data["shapes"][0]["points"][0][0] = -1.01
        with self.assertRaisesRegex(ValueError, "exceeds"):
            bounded_boxes(data)

    def test_outside_keypoint_not_clipped(self):
        data = self.sample()
        data["shapes"][1]["points"] = [[-0.1, 100]]
        fixed, _ = bounded_boxes(data)
        with self.assertRaisesRegex(ValueError, "coordinates outside"):
            convert_pose(fixed)

    def test_explicit_negative_range_is_exact(self):
        keys = negative_keys({"confirmed_negative_ranges": [{"segment_id": "A49_200713",
                              "first_source_frame": 0, "stop_source_frame_exclusive": 54}]})
        self.assertEqual(len(keys), 54)
        self.assertIn("A49_200713:000053", keys)
        self.assertNotIn("A49_200713:000054", keys)


if __name__ == "__main__":
    unittest.main()
