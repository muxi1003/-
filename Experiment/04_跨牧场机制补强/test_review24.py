import unittest
from evaluate_review24 import canonical, matched, rr_stats


class ReviewTests(unittest.TestCase):
    def test_json_csv_boolean_normalization(self):
        self.assertEqual(canonical([{'hidden':True,'count':'2'}],['hidden','count']),
                         canonical([{'hidden':'true','count':'2'}],['hidden','count']))

    def test_missing_values_remain_blank(self):
        self.assertEqual(canonical([{'n':None}],['n']),[('',)])
        self.assertNotEqual(canonical([{'n':None}],['n']),canonical([{'n':0}],['n']))

    def test_no_phase_shift_fitting(self):
        _,m=matched([1.,2.],[1.4,2.4],.3)
        self.assertEqual(m['tp'],0)
        self.assertEqual(m['fn'],2)

    def test_viewing_duration_is_used(self):
        m=rr_stats([10,20],[11,19],[30,60])
        self.assertEqual(m['rr_mae_bpm'],1.5)
        self.assertIsNone(m['rr_r2'])


if __name__=='__main__':
    unittest.main(verbosity=2)
