import unittest
import numpy as np
from score_confirmed_source_pose import high_points, iou


class ScoringTests(unittest.TestCase):
    def test_no_box_is_not_a_point(self):
        self.assertEqual(high_points({'boxes':[]}).shape,(0,2))

    def test_highest_box_and_visibility(self):
        p=dict(boxes=[[0,0,5,5],[0,0,10,10]],box_conf=[.3,.8],width=100,height=100,
               keypoints=[[[1,1,.9],[2,2,.9]],[[10,20,.5],[40,50,.49]]])
        np.testing.assert_equal(high_points(p),[[10,20]])

    def test_outside_points_rejected(self):
        p=dict(boxes=[[0,0,5,5]],box_conf=[.9],width=100,height=100,
               keypoints=[[[-1,3,.9],[100,5,.9]]])
        self.assertEqual(high_points(p).shape,(0,2))

    def test_box_iou(self):
        self.assertAlmostEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[20,20,30,30]),0)
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,15,10]),1/3)


if __name__=='__main__':unittest.main()
