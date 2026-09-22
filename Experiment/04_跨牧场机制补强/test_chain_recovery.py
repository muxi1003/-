import sys
from pathlib import Path
import unittest
import numpy as np
from recover_timestamp_segments import mapping_and_support, CachedRF, segments

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'02_冻结独立测试'))
from timestamp_adapter import sample_map


class FakeRF:
    def predict(self,x):
        return x[:,0].astype(float)+2*x[:,1].astype(float)+3*x[:,2].astype(float)


class ChainRecoveryTests(unittest.TestCase):
    def test_no_gap_grid_matches_frozen_adapter(self):
        t=np.arange(301)/10
        old,_=sample_map(t,30)
        new,gaps=mapping_and_support(t)
        np.testing.assert_array_equal(old.source_frame_index,new.source_frame_index)
        np.testing.assert_allclose(old.target_time_seconds,new.target_time_seconds,atol=1e-12)
        self.assertTrue(new.time_supported.all())
        self.assertEqual(gaps,[])

    def test_gap_not_interpolated_or_time_compressed(self):
        t=np.r_[np.arange(101)/10,12.5+np.arange(176)/10]
        m,gaps=mapping_and_support(t)
        self.assertEqual(len(gaps),1)
        self.assertFalse(m.loc[(m.target_time_seconds>10)&(m.target_time_seconds<12.5),'time_supported'].any())
        self.assertAlmostEqual(m.target_time_seconds.iloc[-1],260/8.7)

    def test_no_segment_core_spans_gap(self):
        m,gaps=mapping_and_support(np.r_[np.arange(101)/10,12.5+np.arange(176)/10])
        for s in segments(m.time_supported.to_numpy(),30):
            self.assertTrue(s['core_stop_seconds']<=10.1 or s['core_start_seconds']>=12.5)

    def test_unsupported_end_not_filled(self):
        m,_=mapping_and_support(np.arange(201)/10)
        self.assertFalse(m.loc[m.target_time_seconds>20.15,'time_supported'].any())

    def test_short_segment_not_eligible(self):
        mask=np.r_[np.ones(30,dtype=bool),np.zeros(231,dtype=bool)]
        self.assertFalse(segments(mask,30)[0]['eligible_duration'])

    def test_exact_color_cache_preserves_BGR_order(self):
        p=np.array([[1,2,3],[255,0,1],[1,2,3],[3,2,1]],dtype=np.uint8)
        c=CachedRF(FakeRF())
        np.testing.assert_array_equal(c.predict(p),FakeRF().predict(p))
        np.testing.assert_array_equal(c.predict(p[::-1]),FakeRF().predict(p[::-1]))


if __name__=='__main__':unittest.main(verbosity=2)
