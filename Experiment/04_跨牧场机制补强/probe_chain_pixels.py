"""Bounded localization probes and exact stored-ROI RF replay; no RR tuning."""
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import joblib
import pandas as pd
from analyze_transfer import HERE, HOLDOUT, read, write_csv, write_json, sha256

ROOT=HERE/'20260921_chain_v3'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2/windows'


def frame(path):
    out=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    assert out is not None
    return out


def main():
    out=ROOT/'pixel_probe';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    model=YOLO(str(FROZEN/'weights/YOLO11n-best.pt'))
    rf=joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl')
    audit=read(ROOT/'review12_stage_audit.csv')
    results=[];replay=[]
    for row in audit.itertuples():
        root=TRANSFER/row.window_id
        if not (root/'temperatures.csv').exists():continue
        temp=read(root/'temperatures.csv')
        for i in [0,87,174]:
            t=temp.iloc[i];im=frame(root/'frames'/t.frame_name)
            for side in ['left','right']:
                stored=t[side+'_temp'];x,y=t[side+'_x'],t[side+'_y']
                if not np.isfinite([stored,x,y]).all():continue
                actual=rr.circle_temperature(im,rf,x,y,int(t.adaptive_roi_radius),20.)
                replay.append(dict(review_id=row.review_id,window_id=row.window_id,frame_index=i,side=side,
                    stored=stored,recomputed=actual,absolute_error=abs(stored-actual),
                    frame_sha256=sha256(root/'frames'/t.frame_name)))
                assert abs(stored-actual)<1e-9
            if row.review_id not in ['R3-01','R3-02','R3-10']:continue
            for name,size,conf in [('baseline640',640,.25),('resolution1280',1280,.25),('low_box_conf',640,.05)]:
                result=model(im,imgsz=size,conf=conf,verbose=False)[0]
                k=rr.select_detection(result)
                record=dict(review_id=row.review_id,window_id=row.window_id,frame_index=i,variant=name,imgsz=size,box_threshold=conf,
                    detections=len(result.boxes),best_box_confidence=None,high_conf_nostril_count=0)
                if k is not None:
                    points=result.keypoints.data[k].cpu().numpy()
                    record.update(best_box_confidence=float(result.boxes.conf[k]),
                        high_conf_nostril_count=int((points[:2,2]>=.5).sum()))
                    for j,side in enumerate(['left','right']):
                        record.update({side+'_x':float(points[j,0]),side+'_y':float(points[j,1]),side+'_conf':float(points[j,2])})
                results.append(record)
    write_csv(out/'localization_probes.csv',results);write_csv(out/'RF_pixel_replay.csv',replay)
    write_json(out/'verification.json',dict(status='PASS_PIXEL_ROI_REPLAY',replayed_roi_count=len(replay),
        max_absolute_error=max(x['absolute_error'] for x in replay),
        caveat='exact pseudocolor mapping replay, not absolute radiometric or anatomical accuracy',
        predictor_scope='27 single-frame diagnostic probes, not all271 improvement',default_changed=False,
        weights_sha256=sha256(FROZEN/'weights/YOLO11n-best.pt')))
    print(pd.DataFrame(results).groupby(['review_id','variant']).agg(frames=('frame_index','size'),
        detections=('detections','sum'),high_conf_points=('high_conf_nostril_count','sum')).to_string())
    print('RF replay',len(replay),'ROIs PASS')


if __name__=='__main__':main()
