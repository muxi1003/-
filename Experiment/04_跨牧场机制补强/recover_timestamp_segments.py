"""Recover partial events around timestamp gaps; no reference input or whole-window RR."""
from pathlib import Path
import sys
import json
import math
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import cv2
import joblib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment_common import HOLDOUT, sha256, write_csv, write_json
sys.path.insert(0,str(HOLDOUT))
from supported_segments import segments, POLICY as SEGMENT_POLICY

TRANSFER=HOLDOUT/'external_transfer/20260914_v2'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'
OUT=HERE/'20260921_chain_v3/partial_timestamp'


def mapping_and_support(times,duration=30.,fps=8.7):
    t=np.asarray(times,float)
    assert len(t)>1 and np.all(np.diff(t)>0) and abs(t[0])<1e-6
    source=t[t<duration]
    grid=np.arange(int(np.ceil(duration*fps)))/fps
    right=np.clip(np.searchsorted(source,grid),0,len(source)-1)
    left=np.maximum(0,right-1)
    idx=np.where(abs(source[left]-grid)<=abs(source[right]-grid),left,right)
    support=abs(source[idx]-grid)<=.25+1e-9
    gaps=[]
    for k in np.flatnonzero(np.diff(t)>.5):
        a,b=float(t[k]),float(t[k+1])
        if a>=duration:continue
        support[(grid>a)&(grid<b)]=False
        gaps.append(dict(start_seconds=a,end_seconds=min(b,duration),duration_seconds=min(b,duration)-a))
    terminal=min(.25,1.5*float(np.median(np.diff(t))))
    support[grid>source[-1]+terminal]=False
    return pd.DataFrame(dict(target_index=np.arange(len(grid)),target_time_seconds=grid,
        source_frame_index=idx,source_time_seconds=source[idx],time_supported=support)),gaps


class BGRProxy:
    def __getattr__(self,name):return getattr(cv2,name)
    def imread(self,path):
        im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError(path)
        return im


class CachedRF:
    """Cache exact uint8 BGR predictions, without quantization or model changes."""
    def __init__(self,model):
        self.model=model
        self.cache=np.full(256**3,np.nan,dtype=np.float64)
    def predict(self,pixels):
        x=np.asarray(pixels)
        assert x.dtype==np.uint8 and x.ndim==2 and x.shape[1]==3
        v=x.astype(np.int32)
        key=v[:,0]+256*v[:,1]+65536*v[:,2]
        missing=np.unique(key[np.isnan(self.cache[key])])
        if len(missing):
            colors=np.column_stack([missing%256,(missing//256)%256,missing//65536]).astype(np.uint8)
            self.cache[missing]=self.model.predict(colors)
        return self.cache[key]


def main():
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((FROZEN/'method_config.json').read_text(encoding='utf-8'))
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    import evaluate_lindian_adaptive_roi as roi
    from ultralytics import YOLO
    config=rr.ReproConfig(**cfg['signal_config']);policy=cfg['roi_policy']
    assert config.truth_csv is None and not config.optimize_peaks and not config.adaptive_peak_retuning
    predictions=pd.read_csv(TRANSFER/'prediction_windows.csv')
    members=pd.read_csv(TRANSFER/'test_windows.csv').set_index('window_id')
    targets=predictions[predictions.reason.isin(['long_timestamp_gap','unsupported_window_end'])]
    assert len(targets)==53
    inputs=[FROZEN/'method_config.json',FROZEN/'weights/YOLO11n-best.pt',FROZEN/'weights/clf_model_RGB_20240906.pkl',
            FROZEN/'scripts/paper_repro_rr.py',FROZEN/'scripts/evaluate_lindian_adaptive_roi.py',
            HOLDOUT/'supported_segments.py',Path(__file__),TRANSFER/'prediction_windows.csv',TRANSFER/'test_windows.csv']
    hashes={str(p):sha256(p) for p in inputs}
    write_json(OUT/'protocol_before_predictions.json',dict(
        role='retrospective_candidate_not_new_holdout',members=targets.window_id.tolist(),inputs=hashes,
        selected_by='frozen timestamp rejection only, no human counts/events',
        change='retain continuous timestamp-supported spans instead of whole-window event abstention',
        segment_policy=SEGMENT_POLICY,frame_gap_threshold_seconds=.5,
        signal_policy='unchanged frozen YOLO/RF/ROI and adaptive signal config applied separately per span',
        full_window_count=None,full_window_rr=None,partial_events_only=True,
        no_time_compression=True,no_interpolation_across_timestamp_gaps=True,
        reference_tables_read=False,existing216_outputs_unchanged=True,
        RF_cache='exact uint8 BGR, no quantization, checked against original regressor'))
    cv2.setNumThreads(1)
    rr.cv2=BGRProxy();roi.cv2=BGRProxy()
    detector=YOLO(str(FROZEN/'weights/YOLO11n-best.pt'))
    raw_model=joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl')
    model=CachedRF(raw_model)
    sample=np.random.default_rng(20260921).integers(0,256,(1000,3),dtype=np.uint8)
    np.testing.assert_allclose(model.predict(sample),raw_model.predict(sample),rtol=0,atol=1e-12)
    output_windows=[];all_events=[];all_spans=[];all_gaps=[]
    start=time.monotonic()
    for ordinal,w in enumerate(targets.itertuples(),1):
        wid=w.window_id;dest=OUT/'windows'/wid;dest.mkdir(parents=True)
        source=Path(members.loc[wid,'source_path'])
        assert sha256(source)==members.loc[wid,'source_sha256']
        time_path=TRANSFER/'windows'/wid/'timestamps.csv'
        original_times=pd.read_csv(time_path).relative_seconds.to_numpy(float)
        mapping,gaps=mapping_and_support(original_times)
        spans=segments(mapping.time_supported.to_numpy(bool),30.)
        write_csv(dest/'map.csv',mapping)
        chosen={}
        for k,span in enumerate(spans):
            span['segment_id']=k
            if not span['eligible_duration']:continue
            frame_dir=dest/f'segment_{k:02d}'/'frames';frame_dir.mkdir(parents=True)
            for i in range(span['start_frame'],span['stop_frame_exclusive']):
                si=int(mapping.iloc[i].source_frame_index)
                chosen.setdefault(si,[]).append(frame_dir/f'frame_{i:06d}.jpg')
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG)
        assert cap.isOpened()
        for i in range(max(chosen,default=-1)+1):
            ok,im=cap.read()
            if not ok:raise ValueError(f'Decode failed {wid} {i}')
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            assert abs(pts-original_times[i])<1e-6
            if i in chosen:
                ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                for p in chosen[i]:b.tofile(str(p))
        cap.release()
        event_count=0;eligible_seconds=0.;signal_spans=0
        for span in spans:
            record=dict(window_id=wid,**span,event_count=0,signal_status='short_span')
            if span['eligible_duration']:
                d=dest/f"segment_{span['segment_id']:02d}"
                base=rr.extract_temperatures(d/'frames',detector,model,config)
                write_csv(d/'base_temperatures.csv',base)
                table=roi.extract_radius_table(d/'frames',base,model,
                    radii=list(range(policy['min_radius'],policy['max_radius']+1)),min_temp=config.min_temp,batch_frames=24)
                spacing_cv=roi.nostril_spacing_cv(table)
                exponent=policy['damped_exponent'] if math.isfinite(spacing_cv) and spacing_cv>policy['spacing_cv_threshold'] else policy['linear_exponent']
                selected,radii=roi.radius_policy_table(table,base_radius=policy['base_radius'],exponent=exponent,
                    clip_low=policy['min_radius'],clip_high=policy['max_radius'])
                write_csv(d/'temperatures.csv',selected)
                if np.isfinite(selected[['left_temp','right_temp']].to_numpy(float)).any():
                    curve,summary=rr.fuse_temperature_curve(selected,config,truth_row=None)
                    curve['target_time_seconds']=(curve.frame_index+span['start_frame'])/config.fps
                    write_csv(d/'curve.csv',curve)
                    indices=np.flatnonzero(curve.is_peak.to_numpy(bool))+span['start_frame']
                    indices=indices[(indices>=span['core_start_frame'])&(indices<span['core_stop_frame_exclusive'])]
                    for i in indices:
                        all_events.append(dict(window_id=wid,event_id=f"partial_{span['segment_id']}_{i}",
                            event_time_seconds=i/config.fps,segment_id=span['segment_id'],grid_index=int(i)))
                    record.update(event_count=len(indices),signal_status='partial_events_only')
                    event_count+=len(indices);eligible_seconds+=span['core_seconds'];signal_spans+=1
                else:record['signal_status']='no_temperature_signal'
            all_spans.append(record)
        all_gaps += [dict(window_id=wid,**g) for g in gaps]
        result=dict(window_id=wid,original_status='abstain',original_reason=w.reason,
            status='partial_events_only' if signal_spans else 'abstain',full_window_count=None,full_window_rr_bpm=None,
            partial_event_count=event_count,eligible_seconds=eligible_seconds,signal_segments=signal_spans)
        write_json(dest/'result.json',result)
        write_json(dest/'checkpoint.json',{'files':{p.relative_to(dest).as_posix():sha256(p) for p in dest.rglob('*') if p.is_file()}})
        output_windows.append(result)
        print(f'{ordinal}/53 {wid}: {signal_spans} spans, {event_count} partial events; elapsed {time.monotonic()-start:.1f}s',flush=True)
    write_csv(OUT/'prediction_windows.csv',output_windows)
    write_csv(OUT/'prediction_events.csv',all_events,columns=['window_id','event_id','event_time_seconds','segment_id','grid_index'])
    write_csv(OUT/'segments.csv',all_spans);write_csv(OUT/'timestamp_gaps.csv',all_gaps)
    for p,h in hashes.items():assert sha256(Path(p))==h
    write_json(OUT/'prediction_seal.json',dict(reference_input_read=False,inputs_unchanged=True,
        windows=53,events=len(all_events),elapsed_seconds=time.monotonic()-start,
        files={p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':main()
