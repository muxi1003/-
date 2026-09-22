"""Synthetic contract tests; synthetic labels are never research references."""
import unittest
import numpy as np
from probe_pose_mechanism import check_transforms
from analyze_extremum_phase import extrema, finite_runs
from evaluate_nostril_reference import validate_submission, normalized_distance, match_regions, outside_fraction


class PosePhaseTests(unittest.TestCase):
    def test_affine_roundtrip_native_coordinates(self):
        check_transforms()

    def test_finite_runs_do_not_join_gaps(self):
        self.assertEqual(finite_runs(np.array([np.nan,1,2,np.nan,3,4,5,np.nan])),[(1,3),(4,7)])

    def test_extrema_no_gap_crossing(self):
        t=np.arange(261)/8.7;y=np.sin(2*np.pi*t)
        y[70:130]=np.nan
        points=extrema(t,y)
        self.assertTrue(points)
        self.assertTrue(all(not 69/8.7<=p['event_time_seconds']<=131/8.7 for p in points))
        self.assertTrue(all(p['observed_run_end']<70/8.7 or p['observed_run_start']>=130/8.7 for p in points))

    def test_maf_phase_center_not_shifted(self):
        t=np.arange(261)/8.7;y=np.cos(2*np.pi*t)
        maxima=[p['event_time_seconds'] for p in extrema(t,y) if p['kind']=='maximum']
        self.assertTrue(maxima)
        self.assertLess(max(abs(x-round(x)) for x in maxima),1/8.7)

    def test_minimum_not_maximum(self):
        t=np.arange(261)/8.7;y=np.cos(2*np.pi*t)
        minima=[p['event_time_seconds'] for p in extrema(t,y) if p['kind']=='minimum']
        self.assertLess(max(abs((x-.5)-round(x-.5)) for x in minima),1/8.7)

    def test_region_matching_unordered(self):
        regions=[dict(cx=20,cy=30,rx=5,ry=5),dict(cx=80,cy=30,rx=5,ry=5)]
        matches=match_regions(np.array([[80,30],[20,30]]),regions)
        self.assertEqual([(i,j) for i,j,d in matches],[(0,1),(1,0)])
        self.assertTrue(all(d==0 for i,j,d in matches))

    def test_off_target_is_not_success(self):
        r=dict(cx=20,cy=30,rx=5,ry=5)
        self.assertGreater(normalized_distance((40,30),r),1)
        self.assertEqual(sum(d<=1 for i,j,d in match_regions([[40,30]],[r])),0)

    def test_roi_background_fraction(self):
        r=dict(cx=50,cy=50,rx=10,ry=10)
        self.assertEqual(outside_fraction((50,50),10,r,100,100),0)
        self.assertGreater(outside_fraction((50,50),20,r,100,100),.7)

    def fixture(self):
        p=dict(package_id='synthetic_test_only',frames=[dict(frame_id='test',width=100,height=100)])
        d=dict(schema_version=1,package_id=p['package_id'],prediction_overlay_shown=False,records={
            'test':dict(status='complete',visibility='one_visible',regions={'A':dict(cx=50,cy=50,rx=10,ry=10),'B':None},annotator='synthetic_test_only',notes='')})
        return p,d

    def test_valid_single_region(self):
        p,d=self.fixture();self.assertIn('test',validate_submission(d,p))

    def test_wrong_package_rejected(self):
        p,d=self.fixture();d['package_id']='wrong'
        with self.assertRaises(ValueError):validate_submission(d,p)

    def test_synthetic_export_rejected(self):
        p,d=self.fixture();d['synthetic_test_only']=True
        with self.assertRaises(ValueError):validate_submission(d,p)

    def test_visibility_region_conflict_rejected(self):
        p,d=self.fixture();d['records']['test']['visibility']='two_visible'
        with self.assertRaises(ValueError):validate_submission(d,p)

    def test_outside_ellipse_rejected(self):
        p,d=self.fixture();d['records']['test']['regions']['A']['cx']=0
        with self.assertRaises(ValueError):validate_submission(d,p)

    def test_pending_not_interpreted_as_zero(self):
        p,d=self.fixture();d['records']['test']={'status':'pending'}
        self.assertIn('test',validate_submission(d,p))

    def test_uncertain_needs_reason(self):
        p,d=self.fixture();d['records']['test']['visibility']='uncertain'
        with self.assertRaises(ValueError):validate_submission(d,p)


if __name__=='__main__':unittest.main()
