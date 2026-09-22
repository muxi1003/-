"""Remove only the whole-curve selection block from sealed external processing."""
import json
import sys
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment_common import HOLDOUT, ABLATION, REFERENCE, sha256, write_csv, write_json, measurements
sys.path.insert(0,str(REFERENCE))
from reference_tools import match_events
from analyze_transfer import metrics, bootstrap_delta


def main():
    out=HERE/'20260921_v1/fusion_transfer'
    out.mkdir(parents=True,exist_ok=False)
    frozen=HOLDOUT/'method_snapshots/20260909_v1'
    manifest=json.loads((frozen/'freeze_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha256(frozen/item['snapshot_relative_path'])==item['sha256']
    sys.path.insert(0,str(frozen/'scripts'))
    import paper_repro_rr as rr
    assert Path(rr.__file__).resolve()==(frozen/'scripts/paper_repro_rr.py').resolve()
    config=rr.ReproConfig(**json.loads((frozen/'method_config.json').read_text(encoding='utf-8'))['signal_config'])
    assert config.truth_csv is None and not config.optimize_peaks and not config.adaptive_peak_retuning
    off=replace(config,fusion_mode='max',fusion_rescue_outlier_threshold=0)
    write_json(out/'protocol_before_results.json',dict(
        role='retrospective_same_input_fusion_block_ablation_not_new_independent_test',
        contrast='F_on whole-curve selection+its rescue vs F_off same-frame max; all other signal rules identical',
        model_manifest_sha256=sha256(frozen/'freeze_manifest.json'),
        F_on=asdict(config),F_off=asdict(off),
        human_reference_generation_input=False,adoption='no automatic change; explanatory ablation only'))
    run=HOLDOUT/'external_transfer/20260914_v2'
    seal=json.loads((run/'prediction_seal.json').read_text(encoding='utf-8'))
    assert sha256(run/'prediction_windows.csv')==seal['predictions_sha256']
    windows=pd.read_csv(run/'prediction_windows.csv',usecols=['window_id','prediction_status','predicted_count','duration_seconds'])
    events=[];predictions=[];hashes=[];replayed=0
    for w in windows.itertuples():
        root=run/'windows'/w.window_id
        assert sha256(root/'checkpoint.json')==seal['checkpoints'][w.window_id]
        checkpoint=json.loads((root/'checkpoint.json').read_text(encoding='utf-8'))['files']
        if w.prediction_status!='ok':
            predictions.append(dict(window_id=w.window_id,prediction_status='abstain',predicted_count=None,duration_seconds=w.duration_seconds))
            continue
        for name in ['temperatures.csv','curve.csv','map.csv']:
            assert sha256(root/name)==checkpoint[name]
            hashes.append(dict(path=str(root/name),sha256=checkpoint[name]))
        table=pd.read_csv(root/'temperatures.csv',float_precision='round_trip')
        old=pd.read_csv(root/'curve.csv',float_precision='round_trip')
        mapping=pd.read_csv(root/'map.csv',float_precision='round_trip')
        replay,summary=rr.fuse_temperature_curve(table.copy(),config,truth_row=None)
        assert np.array_equal(replay.is_peak.to_numpy(bool),old.is_peak.to_numpy(bool))
        assert int(summary['peaks'])==int(w.predicted_count)
        replayed+=1
        curve,s=rr.fuse_temperature_curve(table.copy(),off,truth_row=None)
        peaks=np.flatnonzero(curve.is_peak.to_numpy(bool))
        predictions.append(dict(window_id=w.window_id,prediction_status='ok',predicted_count=len(peaks),duration_seconds=w.duration_seconds))
        events.extend(dict(window_id=w.window_id,event_id=f'p{k}',event_time_seconds=float(mapping.iloc[k].target_time_seconds)) for k in peaks)
    # Prediction generation is complete before any reference table is opened.
    pw=pd.DataFrame(predictions);pe=pd.DataFrame(events)
    write_csv(out/'F_off_prediction_windows.csv',pw);write_csv(out/'F_off_prediction_events.csv',pe)
    write_csv(out/'input_hashes.csv',hashes)
    reference=REFERENCE/'submissions/20260918_r2_status_v1/raw'
    refs=pd.read_csv(reference/'reference_events.csv');refs=refs[refs.confidence.eq('confirmed')]
    ws=pd.read_csv(reference/'annotation_windows.csv',dtype=str,keep_default_na=False)
    refgroups={w:g.event_time_seconds.to_numpy(float) for w,g in refs.groupby('window_id')}
    rows=[];sums=[];count_rows=[];deltas=[]
    for cohort in ['lindian49','jiufu271']:
        selected=ws[ws.cohort.eq(cohort)&ws.annotation_status.eq('complete')]
        if cohort=='lindian49':
            onroot=ABLATION/'reference_updates/20260917_r2_v2/F1G1P1'
            offroot=ABLATION/'reference_updates/20260917_r2_v2/F0G1P1'
            offpw=pd.read_csv(offroot/'prediction_windows.csv').set_index('window_id')
            offpe=pd.read_csv(offroot/'prediction_events.csv')
        else:
            onroot=HOLDOUT/'event_evaluation/20260918_r2_status_v1'
            offpw=pw.set_index('window_id');offpe=pe
        on=pd.read_csv(onroot/'event_metrics_by_window.csv').set_index('window_id')
        oncounts=pd.read_csv(onroot/'paired_count_results.csv').set_index('window_id')
        local=[];paired=[]
        for w in selected.itertuples():
            r=refgroups.get(w.window_id,np.array([]));pp=offpe[offpe.window_id.eq(w.window_id)]
            matches,fp,fn=match_events(pp.event_time_seconds,r,.30)
            source=str(w.source_path)
            cluster=source if cohort=='lindian49' else source.replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0]
            b=on.loc[w.window_id]
            record=dict(cohort=cohort,window_id=w.window_id,cluster=cluster,
                        base_tp=int(b.tp),base_fp=int(b.fp),base_fn=int(b.fn),**metrics(len(matches),len(fp),len(fn)))
            rows.append(record);local.append(record)
            estimate=offpw.loc[w.window_id]
            if estimate.prediction_status=='ok':
                duration=float(w.duration_seconds);yc=len(r);pc=int(estimate.predicted_count)
                rr_on=float(oncounts.loc[w.window_id,'predicted_rr_bpm'])
                paired.append(dict(window_id=w.window_id,truth_count=yc,predicted_count=pc,
                     truth_rr_bpm=60*yc/duration,predicted_rr_bpm=60*pc/duration))
                deltas.append(dict(cohort=cohort,window_id=w.window_id,cluster=cluster,
                    on_rr_bpm=rr_on,off_rr_bpm=60*pc/duration,
                    error_on_minus_off=abs(rr_on-60*yc/duration)-abs(60*pc/duration-60*yc/duration)))
        g=pd.DataFrame(local)
        stats=metrics(*g[['tp','fp','fn']].sum());ci=bootstrap_delta(g)
        sums.append(dict(cohort=cohort,arm='F_off',**stats,**ci,**measurements(pd.DataFrame(paired))))
        onm=json.loads((onroot/'count_metrics.json').read_text(encoding='utf-8'))['metrics']
        sums.append(dict(cohort=cohort,arm='F_on',**metrics(*g[['base_tp','base_fp','base_fn']].sum()),**onm))
    # Paired MAE contrast F_on-F_off resamples full source/cow clusters.
    contrasts=[];d=pd.DataFrame(deltas)
    for cohort,g in d.groupby('cohort'):
        groups=[x.error_on_minus_off.to_numpy(float) for _,x in g.groupby('cluster')]
        rng=np.random.default_rng(20260921);samples=[]
        for _ in range(2000):samples.append(np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean())
        contrasts.append(dict(cohort=cohort,n=len(g),clusters=len(groups),delta_mae_on_minus_off=g.error_on_minus_off.mean(),
            ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975))))
    write_csv(out/'metrics.csv',sums);write_csv(out/'event_contrasts_by_window.csv',rows)
    write_csv(out/'rr_contrasts_by_window.csv',deltas);write_csv(out/'rr_contrasts.csv',contrasts)
    assert all(sha256(Path(h['path']))==h['sha256'] for h in hashes)
    write_json(out/'verification.json',dict(status='PASS',external_numeric_replays=replayed,external_members=len(windows),
         input_hashes_unchanged=True,frozen_predictor_replay_exact=True,all_refusal_members_retained=True,
         no_new_YOLO_inference=True,no_new_RF_training=True,default_unchanged=True,
         reference_hashes={n:sha256(reference/n) for n in ['annotation_windows.csv','reference_events.csv']},
         code_sha256=sha256(Path(__file__))))
    print(pd.DataFrame(sums)[['cohort','arm','f1','rr_r2','rr_mae_bpm']].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__=='__main__':main()
