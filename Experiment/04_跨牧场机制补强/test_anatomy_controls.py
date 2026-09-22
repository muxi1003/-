import unittest
import numpy as np
import pandas as pd
from run_direct_roi_control import mask_inferred
from analyze_anatomy_reference import ellipse_pixels,proxy_mean


class AnatomyControlTests(unittest.TestCase):
    def test_inferred_mask_preserves_direct_and_input(self):
        frame=pd.DataFrame(dict(left_temp=[31.,32.,np.nan],right_temp=[30.,29.,28.],
            left_source=['detected','opposite_inferred','missing'],right_source=['low_conf_inferred','detected','temporal_interpolated']))
        original=frame.copy(deep=True)
        result,counts=mask_inferred(frame)
        pd.testing.assert_frame_equal(frame,original)
        self.assertEqual(result.left_temp.iloc[0],31.)
        self.assertTrue(np.isnan(result.left_temp.iloc[1]))
        self.assertEqual(result.right_temp.iloc[1],29.)
        self.assertTrue(result.right_temp.iloc[[0,2]].isna().all())
        self.assertEqual(counts,{'left':1,'right':2})

    def test_ellipse_native_pixels(self):
        im=np.zeros((11,11,3),np.uint8);im[5,5]=[1,2,3]
        pixels=ellipse_pixels(im,dict(cx=5,cy=5,rx=1,ry=1))
        self.assertEqual(len(pixels),5)
        np.testing.assert_array_equal(pixels.sum(axis=0),[1,2,3])

    def test_strict_temperature_threshold(self):
        class FakeRF:
            def predict(self,pixels):return np.array([20.,21.,19.])
        self.assertEqual(proxy_mean(np.zeros((3,3),np.uint8),FakeRF()),21.)


if __name__=='__main__':unittest.main()
