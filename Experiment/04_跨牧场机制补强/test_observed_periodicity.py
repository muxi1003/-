"""Single retrospective selector candidate: observed-only periodic power times coverage."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from analyze_transfer import HERE, HOLDOUT, read, write_csv, write_json, sha256

ROOT=HERE/'20260921_chain_v3'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2'
MODES=['mean','max','min','left','right']


def observed_score(y,fps=8.7,min_rr=20.,max_rr=100.):
    y=np.asarray(y,float);valid=np.isfinite(y)
    if valid.sum()<6:return 0.
    t=np.arange(len(y))/fps;tv=t[valid];v=y[valid]
    residual=v-np.polyval(np.polyfit(tv,v,1),tv)
    if np.sum(residual**2)<1e-12:return 0.
    frequencies=np.arange(min_rr/60,max_rr/60+1e-9,1/(4*(len(y)/fps)))
    power=lombscargle(tv,residual,2*np.pi*frequencies,normalize=True)
    return float(max(0,np.nanmax(power))*valid.mean())


def main():
    out=ROOT/'observed_periodicity';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    config=rr.ReproConfig(**json.loads((FROZEN/'method_config.json').read_text(encoding='utf-8'))['signal_config'])
    assert config.truth_csv is None and not config.optimize_peaks and not config.adaptive_peak_retuning
    predictions=read(TRANSFER/'prediction_windows.csv')
    hashes={str(p):sha256(p) for p in [FROZEN/'scripts/paper_repro_rr.py',FROZEN/'method_config.json',Path(__file__),TRANSFER/'prediction_windows.csv']}
    write_json(out/'protocol_before_predictions.json',dict(role='single_retrospective_fusion_selector_test',
        input_hashes=hashes,score='max normalized Lomb-Scargle power after linear detrending, multiplied by finite sample coverage',
        fit_samples='raw finite mapped temperatures only; never interpolated samples',
        frequency_range_bpm=[config.min_rr_bpm,config.max_rr_bpm],frequency_step='1/(4*window_duration)',
        changed_component='candidate selection only; existing peak rules/ROI/model/time unchanged',
        reference_read_for_prediction=False,no_phase_shift=True,all_original_abstentions_retained=True,
        candidate_count=1,default_changed=False))
    original_choose=rr.choose_adaptive_fusion_candidate
    windows=[];events=[];scores_rows=[];inputrows=[]
    for ordinal,p in enumerate(predictions.itertuples(),1):
        if p.prediction_status!='ok':
            windows.append(dict(window_id=p.window_id,prediction_status='abstain',predicted_count=None,selected_mode=None))
            continue
        folder=TRANSFER/'windows'/p.window_id
        table=read(folder/'temperatures.csv');saved=read(folder/'curve.csv')
        inputrows.append(dict(window_id=p.window_id,temperature_sha256=sha256(folder/'temperatures.csv'),curve_sha256=sha256(folder/'curve.csv')))
        rr.choose_adaptive_fusion_candidate=original_choose
        control,_=rr.fuse_temperature_curve(table,config,truth_row=None)
        np.testing.assert_array_equal(np.flatnonzero(control.is_peak),np.flatnonzero(saved.is_peak))
        l=rr.normalize_series(table.left_temp);r=rr.normalize_series(table.right_temp)
        scores={m:observed_score(rr.fuse_columns(l,r,m).to_numpy(float),config.fps,config.min_rr_bpm,config.max_rr_bpm) for m in MODES}
        def choose(candidates,cfg):
            baseline=original_choose(candidates,cfg)
            if max(scores.values())<=0:return baseline
            item=max(candidates,key=lambda x:(scores[x['mode']],x['mode']==baseline['mode']))
            return {**item,'quality':{**item['quality'],'observed_periodicity_score':scores[item['mode']]}}
        rr.choose_adaptive_fusion_candidate=choose
        curve,summary=rr.fuse_temperature_curve(table,config,truth_row=None)
        rr.choose_adaptive_fusion_candidate=original_choose
        indices=np.flatnonzero(curve.is_peak)
        windows.append(dict(window_id=p.window_id,prediction_status='ok',predicted_count=len(indices),
            selected_mode=curve.selected_fusion_mode.iloc[0],baseline_mode=saved.selected_fusion_mode.iloc[0]))
        scores_rows.extend(dict(window_id=p.window_id,mode=m,score=s) for m,s in scores.items())
        events.extend(dict(window_id=p.window_id,event_id=f'p{int(i)}',event_time_seconds=float(i/config.fps)) for i in indices)
        write_csv(out/'curves'/f'{p.window_id}.csv',curve)
        if ordinal%40==0:print(f'{ordinal}/271 completed selector replay',flush=True)
    write_csv(out/'prediction_windows.csv',windows);write_csv(out/'prediction_events.csv',events)
    write_csv(out/'candidate_scores.csv',scores_rows);write_csv(out/'input_hashes.csv',inputrows)
    for p,h in hashes.items():assert sha256(Path(p))==h
    write_json(out/'prediction_seal.json',dict(original_output_peak_replay_verified=216,reference_read=False,
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':main()
