import unittest
import numpy as np
from local_contrast_roi_probe import refine


class LocalContrastTests(unittest.TestCase):
    def test_flat_field_does_not_move(self):
        x,y,k,gain,_=refine(np.ones((100,100))*30,50.,50.,20)
        self.assertEqual((x,y,k),(50.,50.,0));self.assertEqual(gain,0)

    def test_blob_recenters_in_both_polarities(self):
        yy,xx=np.indices((120,120));blob=np.exp(-((xx-50)**2+(yy-50)**2)/50)
        for sign in [-1,1]:
            x,y,k,gain,_=refine(30+sign*blob,60.,50.,20)
            self.assertEqual((x,y),(50.,50.));self.assertGreater(gain,0)

    def test_all_invalid_preserves_seed(self):
        x,y,k,_,_=refine(np.full((100,100),np.nan),50.,50.,20)
        self.assertEqual((x,y,k),(50.,50.,0))

    def test_displacement_bound(self):
        field=np.random.default_rng(1).normal(30,1,(100,100))
        x,y,_,_,_=refine(field,50.5,50.25,20)
        self.assertLessEqual(np.hypot(x-50.5,y-50.25),10+1e-9)


if __name__=='__main__':unittest.main()
