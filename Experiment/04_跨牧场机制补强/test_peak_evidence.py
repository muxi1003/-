import unittest
import numpy as np
from audit_frozen_peak_evidence import contributing_sides,temperature_origin


class PeakEvidenceTests(unittest.TestCase):
    def test_left_mode_does_not_borrow_right_support(self):
        self.assertEqual(contributing_sides(dict(left_norm=.2,right_norm=.8),'left'),['left'])

    def test_max_tracks_normalized_contributor(self):
        self.assertEqual(contributing_sides(dict(left_norm=.8,right_norm=.2),'max'),['left'])

    def test_ties_retain_both(self):
        self.assertEqual(contributing_sides(dict(left_norm=.4,right_norm=.4),'min'),['left','right'])

    def test_missing_fused_contributors_are_unknown(self):
        self.assertEqual(contributing_sides(dict(left_norm=np.nan,right_norm=np.nan),'max'),[])

    def test_imputation_not_direct(self):
        self.assertEqual(temperature_origin(np.nan,32.,'detected'),'temperature_interpolated')

    def test_inferred_coords_not_direct(self):
        self.assertEqual(temperature_origin(32.,32.,'opposite_inferred'),'raw_inferred')

    def test_changed_raw_value_not_direct(self):
        self.assertEqual(temperature_origin(32.,31.,'detected'),'modified_temperature')


if __name__=='__main__':unittest.main()
