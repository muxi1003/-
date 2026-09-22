"""One anatomy-informed retrospective control, no per-window truth selection."""
import sys
import json
from pathlib import Path
import numpy as np
from analyze_transfer import HERE,HOLDOUT,read,write_csv,write_json,sha256

FROZEN=HOLDOUT/'method_snapshots/20260909_v1'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2'
OUT=HERE/'20260922_pose_phase_v1/direct_roi_control_v1'


def mask_inferred(table):
    result=table.copy()
    counts={}
    for side in ['left','right']:
        mask=result[side+'_temp'].notna()&result[side+'_source'].ne('detected')
        counts[side]=int(mask.sum())
        result.loc[mask,side+'_temp']=np.nan
        result.loc[mask,side+'_source']='missing'
    return result,counts


def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    cfg=rr.ReproConfig(**json.loads((FROZEN/'method_config.json').read_text(encoding='utf-8'))['signal_config'])
    assert cfg.truth_csv is None and not cfg.optimize_peaks and not cfg.adaptive_peak_retuning
    pred=read(TRANSFER/'prediction_windows.csv')
    write_json(OUT/'protocol_before_predictions.json',dict(
        role='single_anatomy_informed_retrospective_counterfactual',
        rule='mask finite temperature when source is not detected; retain frozen fusion/repair/peak rules',
        hypothesis='inferred ROI temperatures contaminate waveform',
        fallback='if no direct temperatures anywhere, retain frozen result and flag fallback',
        annotation_used_to_design_rule=True,reference_read_by_predictor=False,
        no_human_coordinate_substitution=True,original_abstentions_retained=True,
        no_default_change=True,no_new_model=True,no_claim_of_independent_external_validation=True,
        input_sha256={str(p):sha256(p) for p in [FROZEN/'method_config.json',FROZEN/'scripts/paper_repro_rr.py',Path(__file__)]}))
    windows=[];events=[];masked=[];hashes=[]
    for ordinal,p in enumerate(pred.itertuples(),1):
        if p.prediction_status!='ok':
            windows.append(dict(window_id=p.window_id,prediction_status='abstain',predicted_count=None))
            continue
        folder=TRANSFER/'windows'/p.window_id
        temp=read(folder/'temperatures.csv');old=read(folder/'curve.csv')
        baseline,_=rr.fuse_temperature_curve(temp,cfg,truth_row=None)
        np.testing.assert_array_equal(np.flatnonzero(baseline.is_peak),np.flatnonzero(old.is_peak))
        clean,removed=mask_inferred(temp)
        fallback=not clean[['left_temp','right_temp']].notna().any().any()
        if fallback:
            curve=baseline
        else:
            curve,_=rr.fuse_temperature_curve(clean,cfg,truth_row=None)
        idx=np.flatnonzero(curve.is_peak)
        windows.append(dict(window_id=p.window_id,prediction_status='ok',predicted_count=len(idx),
            fallback_no_direct_signal=fallback,selected_mode=curve.selected_fusion_mode.iloc[0]))
        events.extend(dict(window_id=p.window_id,event_id=f'p{int(i)}',event_time_seconds=float(i/cfg.fps)) for i in idx)
        masked.append(dict(window_id=p.window_id,removed_left=removed['left'],removed_right=removed['right'],
            fallback=fallback,baseline_count=int(old.is_peak.sum()),new_count=len(idx)))
        hashes.append(dict(window_id=p.window_id,temp_sha256=sha256(folder/'temperatures.csv'),curve_sha256=sha256(folder/'curve.csv')))
        write_csv(OUT/'curves'/f'{p.window_id}.csv',curve)
        if ordinal%40==0:print(ordinal,'/271 done',flush=True)
    write_csv(OUT/'prediction_windows.csv',windows);write_csv(OUT/'prediction_events.csv',events)
    write_csv(OUT/'temperature_mask_audit.csv',masked);write_csv(OUT/'input_hashes.csv',hashes)
    write_json(OUT/'prediction_seal.json',dict(original_output_peak_replay_verified=216,
        files={p.name:sha256(p) for p in OUT.iterdir() if p.is_file()},no_reference_read_by_predictor=True))
    print('271 members completed; score separately')


if __name__=='__main__':main()
