import unittest
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'03_可靠事件参考'))
from analyze_transfer import metrics
from reference_tools import match_events
spec=importlib.util.spec_from_file_location('guard',HERE.parent/'02_冻结独立测试/signal_observability_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


class EvidenceTests(unittest.TestCase):
    def table(self,n=20):
        d={}
        for side in ['left','right']:
            for col,val in [('temp',33.),('x',100.),('y',100.),('conf',.9),('source','detected')]:d[f'{side}_{col}']=[val]*n
        return pd.DataFrame(d)

    def test_abstain_all_fn(self):
        matched,fp,fn=match_events([], [1.,2.],.3)
        self.assertEqual((len(matched),len(fp),len(fn)),(0,0,2))

    def test_zero_peak_output_is_valid(self):
        indices=np.array([],dtype=int)
        self.assertEqual(len(np.ones(10,dtype=bool)[indices]),0)

    def test_no_double_match(self):
        matched,fp,fn=match_events([1.,1.1],[1.05],.3)
        self.assertEqual((len(matched),len(fp),len(fn)),(1,1,0))

    def test_correct_count_can_be_wrong_events(self):
        matches,fp,fn=match_events([1.,3.],[2.,4.],.3)
        self.assertEqual(len(fp)-len(fn),0)
        self.assertEqual(metrics(len(matches),len(fp),len(fn))['f1'],0.)

    def test_short_interior_gap_permitted(self):
        t=self.table();t.loc[8:10,'left_source']='tracked'
        mask,_,result=guard.protect_window(t,'left')
        self.assertTrue(result['prediction_status']=='ready')
        self.assertTrue(mask.support_after_short_gap_policy.all())

    def test_boundary_gap_not_full_rr(self):
        t=self.table();t.loc[0,'left_source']='missing'
        mask,_,result=guard.protect_window(t,'left')
        self.assertEqual(result['prediction_status'],'abstain')
        self.assertTrue(mask.unsafe_event_context.iloc[0])
        self.assertFalse(mask.unsafe_event_context.iloc[10])

    def test_mixed_support_is_conservative_not_visibility_truth(self):
        t=self.table();t.loc[:,'right_source']='low_conf_inferred'
        _,_,mixed=guard.protect_window(t,'max');_,_,single=guard.protect_window(t,'left')
        self.assertEqual(mixed['prediction_status'],'abstain')
        self.assertEqual(single['prediction_status'],'ready')

    def test_affine_normalization_invariance_only(self):
        x=np.array([31.,32.,33.,31.5]);y=2*x+3
        self.assertTrue(np.allclose((x-x.min())/np.ptp(x),(y-y.min())/np.ptp(y)))


if __name__=='__main__':unittest.main(verbosity=2)
