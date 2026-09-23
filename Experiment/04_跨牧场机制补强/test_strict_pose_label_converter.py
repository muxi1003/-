import unittest
from strict_pose_label_converter import convert_pose, AnnotationConflict


def sample(box, point):
    return dict(imageWidth=100,imageHeight=100,shapes=[
        dict(label='nose',shape_type='rectangle',points=box,group_id=0),
        dict(label='left_nostril',shape_type='point',points=[point],group_id=0)])


class StrictPoseConversionTests(unittest.TestCase):
    def test_four_and_two_point_boxes_match(self):
        a=convert_pose(sample([[10,10],[30,10],[30,30],[10,30]],[20,20]))
        b=convert_pose(sample([[30,30],[10,10]],[20,20]))
        self.assertEqual(a,b); self.assertEqual(a[0][4],.2)

    def test_border_annotation_is_not_silently_removed(self):
        row=convert_pose(sample([[10,10],[30,30]],[10,20]))[0]
        self.assertEqual(row[5:8],[.1,.2,2])

    def test_fractional_coordinates_preserved(self):
        row=convert_pose(sample([[10.1,10.1],[30.9,30.9]],[10.2,20.3]))[0]
        self.assertAlmostEqual(row[5],.102); self.assertEqual(row[7],2)

    def test_outside_annotation_rejected_not_zeroed(self):
        with self.assertRaisesRegex(AnnotationConflict,'outside associated'):
            convert_pose(sample([[10,10],[30,30]],[31,20]))

    def test_group_mismatch_rejected(self):
        data=sample([[10,10],[30,30]],[20,20]); data['shapes'][1]['group_id']=1
        with self.assertRaisesRegex(AnnotationConflict,'group'):
            convert_pose(data)

    def test_missing_annotation_not_invented(self):
        self.assertEqual(convert_pose(sample([[10,10],[30,30]],[20,20]))[0][8:],[0,0,0])

    def test_unassigned_group_with_unique_containing_box_preserved(self):
        data=sample([[10,10],[30,30]],[20,20]); data['shapes'][1]['group_id']=None
        self.assertEqual(convert_pose(data)[0][5:8],[.2,.2,2])

    def test_unassigned_group_with_overlapping_boxes_rejected(self):
        data=sample([[10,10],[30,30]],[20,20]); data['shapes'][1]['group_id']=None
        data['shapes'].append(dict(label='nose',shape_type='rectangle',points=[[15,15],[35,35]],group_id=1))
        with self.assertRaisesRegex(AnnotationConflict,'ambiguous'):
            convert_pose(data)

    def test_no_box_with_point_is_not_background(self):
        data=sample([[10,10],[30,30]],[20,20]); data['shapes']=data['shapes'][1:]
        with self.assertRaisesRegex(AnnotationConflict,'without nose box'):
            convert_pose(data)


if __name__=='__main__': unittest.main()
