"""Native BGR-to-temperature spot checks independent of pooled CSV arithmetic."""
import sys
import joblib
import numpy as np
from analyze_transfer import read,write_json,sha256
from run_oof_nostril_signal import OUT,FROZEN,TRANSFER
from probe_pose_mechanism import load_bgr


def main():
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    model=joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl')
    if hasattr(model,'n_jobs'):model.n_jobs=1
    checked=[]
    for p in read(OUT/'prediction_windows.csv').itertuples():
        path=OUT/p.method/p.window_id/'detections.csv'
        if not path.exists():continue
        d=read(path)
        if d.empty:continue
        row=d.iloc[0];frame=TRANSFER/'windows'/p.window_id/'frames'/f'frame_{int(row.frame_index):06d}.jpg'
        value=rr.circle_temperature(load_bgr(frame),model,float(row.x),float(row.y),20,20.)
        np.testing.assert_allclose(value,row.temperature,atol=1e-9,rtol=0,equal_nan=True)
        checked.append(dict(method=p.method,window_id=p.window_id,frame_index=int(row.frame_index),frame_sha256=sha256(frame)))
    assert len(checked)==18
    write_json(OUT/'rf_sample_verification.json',dict(status='PASS',sampled_ROIs=18,
        selection='first saved detection per method/window',samples=checked,
        scope='Native pixels, saved center, fixed radius20, BGR RF and strict>20C cutoff; not all ROI pixels re-inferred'))
    print('PASS:18 original-frame BGR RF temperature spot checks')


if __name__=='__main__':main()
