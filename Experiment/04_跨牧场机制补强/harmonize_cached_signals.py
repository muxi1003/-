"""Fixed-r20, same-config controls on the original annotated viewing windows.

This does not re-extract YOLO, undo historical CFR conversion, or establish an
end-to-end domain effect. Existing external temporal refusals remain refusals.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment_common import ABLATION,HOLDOUT,REFERENCE,sha256,write_json,write_csv,measurements
sys.path.insert(0,str(HOLDOUT));sys.path.insert(0,str(REFERENCE))
from timestamp_adapter import sample_map,POLICY
from reference_tools import match_events
from analyze_transfer import metrics


def main():
    out=HERE/'20260921_v1/common_cached_r20'
    out.mkdir(parents=True,exist_ok=False)
    frozen=HOLDOUT/'method_snapshots/20260909_v1'
    sys.path.insert(0,str(frozen/'scripts'))
    import paper_repro_rr as rr
    assert Path(rr.__file__).resolve()==(frozen/'scripts/paper_repro_rr.py').resolve()
    for item in json.loads((frozen/'freeze_manifest.json').read_text(encoding='utf-8'))['files']:
        assert sha256(frozen/item['snapshot_relative_path'])==item['sha256']
    cfg=json.loads((frozen/'method_config.json').read_text(encoding='utf-8'))['signal_config']
    config=rr.ReproConfig(**cfg)
    assert config.truth_csv is None and not config.optimize_peaks
    run=HOLDOUT/'external_transfer/20260914_v2'
    seal=json.loads((run/'prediction_seal.json').read_text(encoding='utf-8'))
    assert sha256(run/'prediction_windows.csv')==seal['predictions_sha256']
    external=pd.read_csv(run/'prediction_windows.csv',usecols=['window_id','prediction_status']).set_index('window_id')
    refdir=REFERENCE/'submissions/20260918_r2_status_v1/raw'
    # Only immutable membership/time fields, never human outcomes, enter generation.
    ws=pd.read_csv(refdir/'annotation_windows.csv',usecols=['window_id','video_id','cohort','duration_seconds'])
    write_json(out/'protocol_before_results.json',dict(role='retrospective_cached_signal_control',
        fixed_radius=20,signal_config=cfg,sampling_policy=POLICY,
        arms=['r20_same_config_native_viewing_times','r20_same_config_uniform_viewing_grid'],
        original_viewing_windows_unchanged=True,old_jiufu_refusals_retained=True,
        new_detection_or_pixel_extraction=False,no_new_short_gap_remeasurement=True,
        limitation='Cached upstream detection/provenance and CFR history remain cohort-specific; NOT full raw-video harmonization',
        truth_access='only membership metadata parsed before prediction seal',
        parent_config_sha256=sha256(frozen/'method_config.json')))
    pred=[];events=[];hashes=[]
    input_manifest=pd.read_csv(ABLATION/'input_snapshots/20260909_v1/matched_windows.csv',
        usecols=['cohort','video_id','input_sha256'])
    expected=input_manifest[input_manifest.cohort.eq('anchored49')].set_index('video_id').input_sha256.to_dict()
    for w in ws.itertuples():
        if w.cohort=='jiufu271' and external.loc[w.window_id,'prediction_status']!='ok':
            for arm in ['native','uniform']:
                pred.append(dict(cohort=w.cohort,window_id=w.window_id,arm=arm,prediction_status='abstain',
                    predicted_count=None,duration_seconds=w.duration_seconds,reason='original_frozen_refusal_retained'))
            continue
        if w.cohort=='lindian49':
            tp=ABLATION/f'input_snapshots/20260909_v1/inputs/anchored49/{w.video_id}.csv'
            timepath=HOLDOUT/f'adapter_validation/20260912_v2/decoded_timestamps/{w.video_id}.csv'
            assert sha256(tp)==expected[str(w.video_id)]
            times=pd.read_csv(timepath,float_precision='round_trip').annotation_video_time_seconds.to_numpy()
        else:
            root=run/'windows'/w.window_id
            assert sha256(root/'checkpoint.json')==seal['checkpoints'][w.window_id]
            checkpoint=json.loads((root/'checkpoint.json').read_text(encoding='utf-8'))['files']
            tp=root/'radius_table.csv';timepath=root/'map.csv'
            for x in [tp,timepath]:assert sha256(x)==checkpoint[x.name]
            times=pd.read_csv(timepath,float_precision='round_trip').target_time_seconds.to_numpy()
        hashes.extend(dict(path=str(x),sha256=sha256(x)) for x in [tp,timepath])
        table=pd.read_csv(tp,float_precision='round_trip')
        assert len(table)==len(times)
        for side in ['left','right']:table[f'{side}_temp']=table[f'{side}_temp_r20']
        table['roi_radius']=20
        for arm in ['native','uniform']:
            record=dict(cohort=w.cohort,window_id=w.window_id,arm=arm,duration_seconds=w.duration_seconds)
            if arm=='uniform':
                mapping,status=sample_map(times,float(w.duration_seconds))
                if mapping is None:
                    pred.append(dict(**record,prediction_status='abstain',predicted_count=None,reason=status['reason']));continue
                t=table.iloc[mapping.source_frame_index].copy().reset_index(drop=True)
                clock=mapping.target_time_seconds.to_numpy()
                t['frame_name']=[f'frame_{i:06d}.jpg' for i in range(len(t))]
            else:t=table.copy();clock=times
            if not np.isfinite(t[['left_temp','right_temp']].to_numpy(float)).any():
                pred.append(dict(**record,prediction_status='abstain',predicted_count=None,reason='no_r20_signal'));continue
            curve,summary=rr.fuse_temperature_curve(t,config,truth_row=None)
            peaks=np.flatnonzero(curve.is_peak.to_numpy(bool))
            pred.append(dict(**record,prediction_status='ok',predicted_count=len(peaks),reason='',
                selected_mode=summary['selected_fusion_mode'],frames=len(t)))
            events.extend(dict(cohort=w.cohort,window_id=w.window_id,arm=arm,event_id=f'p{k}',event_time_seconds=float(clock[k])) for k in peaks)
    pw=pd.DataFrame(pred);pe=pd.DataFrame(events)
    write_csv(out/'prediction_windows.csv',pw);write_csv(out/'prediction_events.csv',pe)
    write_json(out/'prediction_seal_before_reference.json',dict(
        windows_sha256=sha256(out/'prediction_windows.csv'),events_sha256=sha256(out/'prediction_events.csv')))
    refs=pd.read_csv(refdir/'reference_events.csv');refs=refs[refs.confidence.eq('confirmed')]
    meta=pd.read_csv(refdir/'annotation_windows.csv');meta=meta[meta.annotation_status.eq('complete')]
    refgroups={w:g.event_time_seconds.to_numpy(float) for w,g in refs.groupby('window_id')}
    results=[];bywindow=[];pairs=[];delta=[]
    for (cohort,arm),g in pw.groupby(['cohort','arm']):
        total=np.zeros(3,dtype=int);paired=[]
        for w in meta[meta.cohort.eq(cohort)].itertuples():
            p=g[g.window_id.eq(w.window_id)].iloc[0]
            pt=pe[(pe.arm.eq(arm))&pe.window_id.eq(w.window_id)].event_time_seconds
            r=refgroups.get(w.window_id,[])
            matched,fp,fn=match_events(pt,r,.3);nums=[len(matched),len(fp),len(fn)];total+=nums
            bywindow.append(dict(cohort=cohort,arm=arm,window_id=w.window_id,**metrics(*nums)))
            if p.prediction_status=='ok':
                yc=len(r);pc=int(p.predicted_count);d=float(w.duration_seconds)
                row=dict(cohort=cohort,arm=arm,window_id=w.window_id,predicted_count=pc,truth_count=yc,
                    predicted_rr_bpm=60*pc/d,truth_rr_bpm=60*yc/d)
                paired.append(row);pairs.append(row)
        results.append(dict(cohort=cohort,arm=arm,total_windows=len(g),all_output_windows=int(g.prediction_status.eq('ok').sum()),
            complete_reference_windows=len(meta[meta.cohort.eq(cohort)]),**metrics(*total),**measurements(pd.DataFrame(paired))))
    for cohort,baseline in [('lindian49',ABLATION/'reference_updates/20260917_r2_v2/F1G1P1'),
                            ('jiufu271',HOLDOUT/'event_evaluation/20260918_r2_status_v1')]:
        b=pd.read_csv(baseline/'paired_count_results.csv').set_index('window_id')
        q=pd.DataFrame(pairs)
        for arm in ['native','uniform']:
            z=q[q.cohort.eq(cohort)&q.arm.eq(arm)].set_index('window_id')
            assert set(z.index)==set(b.index), 'Analysis-set change requires explicit matched-only report'
            d=(abs(z.predicted_rr_bpm-z.truth_rr_bpm)-abs(b.predicted_rr_bpm-b.truth_rr_bpm)).rename('delta')
            ms=meta.set_index('window_id')
            clusters={}
            for wid,val in d.items():
                src=str(ms.loc[wid,'source_path'])
                key=src if cohort=='lindian49' else src.replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0]
                clusters.setdefault(key,[]).append(val)
            arr=list(clusters.values());rng=np.random.default_rng(20260921);draw=[]
            for _ in range(2000):draw.append(np.concatenate([arr[i] for i in rng.integers(len(arr),size=len(arr))]).mean())
            delta.append(dict(cohort=cohort,arm=arm,n=len(d),clusters=len(arr),delta_mae_vs_original=float(d.mean()),
                ci95_low=float(np.quantile(draw,.025)),ci95_high=float(np.quantile(draw,.975))))
    write_csv(out/'metrics.csv',results);write_csv(out/'events_by_window.csv',bywindow)
    write_csv(out/'paired_counts.csv',pairs);write_csv(out/'paired_mae_changes.csv',delta);write_csv(out/'input_hashes.csv',hashes)
    assert all(sha256(Path(x['path']))==x['sha256'] for x in hashes)
    write_json(out/'verification.json',dict(status='PASS',members=len(ws),arms=2,
        predictions_generated_before_reference_loading=True,source_inputs_unchanged=True,
        baseline_count_analysis_sets_identical=True,default_unchanged=True,code_sha256=sha256(Path(__file__))))
    print(pd.DataFrame(results)[['cohort','arm','n','f1','rr_r2','rr_mae_bpm']].to_string(index=False))
    print(pd.DataFrame(delta).to_string(index=False))


if __name__=='__main__':main()
