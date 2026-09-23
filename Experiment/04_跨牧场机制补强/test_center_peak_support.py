import unittest
import numpy as np
import pandas as pd
from audit_center_peak_support import has_center_support


def tables(left_sources, right_sources=None, mode='left'):
    if right_sources is None:right_sources=['missing']*len(left_sources)
    curve=pd.DataFrame(dict(selected_fusion_mode=[mode]*len(left_sources),
                            left_norm=[.4]*len(left_sources),right_norm=[.6]*len(left_sources)))
    temp=pd.DataFrame(dict(left_source=left_sources,right_source=right_sources,
                           left_temp=[30. if s=='detected' else np.nan for s in left_sources],
                           right_temp=[30. if s=='detected' else np.nan for s in right_sources]))
    return curve,temp


class CenterSupportTests(unittest.TestCase):
    def test_direct_center(self):
        curve,temp=tables(['missing','detected','missing'])
        self.assertTrue(has_center_support(curve,temp,1)[0])

    def test_one_sample_interior_bridge(self):
        curve,temp=tables(['detected','missing','detected'])
        keep,center,bridge=has_center_support(curve,temp,1)
        self.assertTrue(keep);self.assertEqual(center,[]);self.assertEqual(bridge,['left'])

    def test_no_boundary_extrapolation(self):
        curve,temp=tables(['missing','detected','detected'])
        self.assertFalse(has_center_support(curve,temp,0)[0])

    def test_other_side_does_not_support_selected_left(self):
        curve,temp=tables(['missing','missing','missing'],['detected']*3)
        self.assertFalse(has_center_support(curve,temp,1)[0])

    def test_mean_can_use_direct_contributing_side(self):
        curve,temp=tables(['missing']*3,['missing','detected','missing'],mode='mean')
        self.assertTrue(has_center_support(curve,temp,1)[0])


if __name__=='__main__':unittest.main()
