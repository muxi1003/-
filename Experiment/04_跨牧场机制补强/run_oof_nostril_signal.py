"""Continuous held-out-cow detector temperatures; references used only downstream."""
import sys
import json
from pathlib import Path
import cv2
import joblib
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from analyze_transfer import read,write_csv,write_json,sha256,HOLDOUT
from train_nostril_grouped_probe import ROOT,OUT as SPLIT_ROOT

OUT=ROOT/'oof_nostril_signal_v1'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2'


def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    config=json.loads((FROZEN/'method_config.json').read_text(encoding='utf-8'))['signal_config']
    config.update(fusion_mode='left',track_missing=False,infer_missing_nostril=False,infer_low_confidence_nostril=False)
    cfg=rr.ReproConfig(**config)
    assert cfg.truth_csv is None and not cfg.optimize_peaks and not cfg.adaptive_peak_retuning
    split=read(SPLIT_ROOT/'group_split.csv')
    members=split[['window_id','cow_id','heldout_fold']].drop_duplicates()
    assert len(members)==12
    original=read(TRANSFER/'prediction_windows.csv').set_index('window_id')
    rf=joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl')
    if hasattr(rf,'n_jobs'):rf.n_jobs=1
    torch.set_num_threads(2);cv2.setNumThreads(1)
    write_json(OUT/'protocol_before_predictions.json',dict(
        role='retrospective continuous held-out-cow diagnostic; not new independent external test',
        cohort_windows=12,split_sha256=sha256(SPLIT_ROOT/'group_split.csv'),
        models=['direct_nostril_longer_v1','direct_nostril_unfrozen_v1'],
        temperature='BGR RF, each detected center radius20, keep predicted pixel temperatures strictly>20C; arithmetic mean of available ROI means',
        side_identity='unordered pooled temperature stored in left_temp only; not anatomical left',
        missing='no detected pixels -> missing, never zero or inferred ROI; frozen signal repair/MAF3/peak/source-gate retained',
        detector=dict(conf=.25,iou=.7,max_det=2,imgsz=640),signal_config=config,
        time_policy='reuse exact original grid; retain timestamp-based abstentions; no-signal abstention may recover',
        original_pipeline_is_context_not_single_factor_control=True,no_reference_events_read=True,
        default_changed=False,rf_sha256=sha256(FROZEN/'weights/clf_model_RGB_20240906.pkl'),
        script_sha256=sha256(Path(__file__))))
    rows=[];events=[];seal_inputs=[]
    for method,modelroot in [('frozen_direct',ROOT/'direct_nostril_longer_v1'),('unfrozen_direct',ROOT/'direct_nostril_unfrozen_v1')]:
        for fold in range(3):
            weights=modelroot/'runs'/f'fold{fold}'/'weights/last.pt'
            completed=json.loads((modelroot/f'fold{fold}_completed.json').read_text())
            assert sha256(weights)==completed['sha256']
            training_cows=set(split[split.heldout_fold.ne(fold)].cow_id.astype(str))
            detector=YOLO(str(weights))
            for m in members[members.heldout_fold.eq(fold)].itertuples():
                assert str(m.cow_id) not in training_cows
                folder=TRANSFER/'windows'/m.window_id
                if not (folder/'map.csv').exists():
                    assert original.loc[m.window_id,'prediction_status']=='abstain'
                    rows.append(dict(method=method,window_id=m.window_id,cow_id=m.cow_id,fold=fold,prediction_status='abstain',
                                     predicted_count=None,reason=original.loc[m.window_id,'reason'],valid_frames=0))
                    continue
                mapping=read(folder/'map.csv');assert len(mapping)==261
                np.testing.assert_allclose(mapping.target_time_seconds,np.arange(261)/8.7,atol=1e-8)
                temperatures=[];boxes_out=[]
                for r in mapping.itertuples():
                    path=folder/'frames'/f'frame_{r.target_index:06d}.jpg'
                    im=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR);assert im is not None
                    p=detector(im,conf=.25,iou=.7,max_det=2,imgsz=640,device=0,verbose=False)[0]
                    values=[]
                    for k,(box,conf) in enumerate(zip(p.boxes.xyxy.cpu().numpy(),p.boxes.conf.cpu().numpy())):
                        x0,y0,x1,y1=map(float,box);x,y=(x0+x1)/2,(y0+y1)/2
                        value=rr.circle_temperature(im,rf,x,y,20,20.)
                        if np.isfinite(value):values.append(value)
                        boxes_out.append(dict(frame_index=r.target_index,time_seconds=r.target_time_seconds,det_index=k,
                            x=x,y=y,x0=x0,y0=y0,x1=x1,y1=y1,confidence=float(conf),temperature=value))
                    value=float(np.mean(values)) if values else np.nan
                    temperatures.append(dict(frame_name=path.name,left_temp=value,right_temp=np.nan,
                        left_source='detected' if values else 'missing',right_source='missing',roi_radius=20,
                        status='ok' if values else 'no_valid_nostril'))
                table=pd.DataFrame(temperatures)
                target=OUT/method/m.window_id
                write_csv(target/'temperatures.csv',table)
                write_csv(target/'detections.csv',boxes_out,columns=['frame_index','time_seconds','det_index','x','y','x0','y0','x1','y1','confidence','temperature'])
                valid=int(table.left_temp.notna().sum())
                if valid:
                    curve,_=rr.fuse_temperature_curve(table,cfg,truth_row=None)
                    indices=np.flatnonzero(curve.is_peak)
                    write_csv(target/'curve.csv',curve)
                    events.extend(dict(method=method,window_id=m.window_id,event_id=f'p{int(i)}',event_time_seconds=float(i/8.7)) for i in indices)
                    status='ok';count=len(indices);reason=''
                else:status='abstain';count=None;reason='no_valid_temperature_signal'
                rows.append(dict(method=method,window_id=m.window_id,cow_id=m.cow_id,fold=fold,prediction_status=status,
                                 predicted_count=count,reason=reason,valid_frames=valid))
                seal_inputs.append(dict(method=method,window_id=m.window_id,checkpoint_sha256=sha256(weights),
                    mapping_sha256=sha256(folder/'map.csv'),temperatures_sha256=sha256(target/'temperatures.csv'),
                    detections_sha256=sha256(target/'detections.csv')))
                print(method,m.window_id,'valid',valid,'peaks',count,flush=True)
            del detector;torch.cuda.empty_cache()
    write_csv(OUT/'prediction_windows.csv',rows)
    write_csv(OUT/'prediction_events.csv',events,columns=['method','window_id','event_id','event_time_seconds'])
    write_csv(OUT/'input_output_hashes.csv',seal_inputs)
    write_json(OUT/'prediction_seal.json',dict(rows=len(rows),windows=12,
        files={n:sha256(OUT/n) for n in ['prediction_windows.csv','prediction_events.csv','input_output_hashes.csv','protocol_before_predictions.json']},
        default_changed=False))


if __name__=='__main__':main()
