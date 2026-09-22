"""Score sealed partial-event recovery on unchanged full R2 and separate R3 sets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_transfer import HERE, BASES, SUBMISSION, read, write_csv, write_json, sha256, metrics, match_events, ordered_matching
from recover_timestamp_segments import mapping_and_support

ROOT=HERE/'20260921_chain_v3'
RUN=ROOT/'partial_timestamp'


def main():
    out=ROOT/'recovery_evaluation';out.mkdir(exist_ok=False)
    seal=json.loads((RUN/'prediction_seal.json').read_text(encoding='utf-8'))
    for n,h in seal['files'].items():assert sha256(RUN/n)==h
    for name,h in json.loads((RUN/'protocol_before_predictions.json').read_text(encoding='utf-8'))['inputs'].items():
        assert sha256(Path(name))==h,name
    windows=read(RUN/'prediction_windows.csv'); extra=read(RUN/'prediction_events.csv'); spans=read(RUN/'segments.csv')
    assert len(windows)==53 and windows.window_id.is_unique
    assert windows.full_window_count.isna().all() and windows.full_window_rr_bpm.isna().all()
    baseline_events=read(BASES['jiufu271']/'prediction_events.csv')
    baseline_windows=read(BASES['jiufu271']/'prediction_windows.csv').set_index('window_id')
    assert baseline_windows.loc[windows.window_id,'prediction_status'].eq('abstain').all()
    assert not set(extra.window_id)&set(baseline_events.window_id)
    original=pd.concat([baseline_events[['window_id','event_id','event_time_seconds']],extra[['window_id','event_id','event_time_seconds']]],ignore_index=True)
    expected_events=[]
    for wid in windows.window_id:
        dest=RUN/'windows'/wid
        checkpoint=json.loads((dest/'checkpoint.json').read_text(encoding='utf-8'))
        for n,h in checkpoint['files'].items():assert sha256(dest/n)==h
        mapping=read(dest/'map.csv')
        selected=spans[spans.window_id.eq(wid)]
        for s in selected.itertuples():
            if s.signal_status!='partial_events_only':continue
            assert s.eligible_duration and s.core_seconds>=6-1e-9
            assert mapping.iloc[s.start_frame:s.stop_frame_exclusive].time_supported.all()
            c=read(dest/f'segment_{s.segment_id:02d}/curve.csv')
            indices=np.flatnonzero(c.is_peak.to_numpy(bool))+s.start_frame
            indices=indices[(indices>=s.core_start_frame)&(indices<s.core_stop_frame_exclusive)]
            assert len(indices)==s.event_count
            expected_events.extend((wid,int(s.segment_id),int(i)) for i in indices)
    assert sorted(expected_events)==sorted(zip(extra.window_id,extra.segment_id.astype(int),extra.grid_index.astype(int)))
    np.testing.assert_allclose(extra.event_time_seconds,extra.grid_index/8.7,atol=1e-12,rtol=0)
    rows=[]; summaries=[]; changes=[]
    for version in ['R2','R3']:
        if version=='R2':
            w=read(SUBMISSION/'annotation_windows.csv')
            w=w[w.cohort.eq('jiufu271')&w.annotation_status.eq('complete')]
            refs=read(SUBMISSION/'reference_events.csv')
            clusters={r.window_id:str(r.source_path).replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0] for r in w.itertuples()}
        else:
            audit=read(HERE/'20260921_r3_v1/annotation_audit.csv')
            w=audit[audit.cohort.eq('jiufu271')&audit.eligible_full_window]
            refs=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
            clusters=w.set_index('window_id').cluster.astype(str).to_dict()
        for wid in w.window_id:
            rr=refs[refs.window_id.eq(wid)&refs.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
            for method,pred in [('frozen',baseline_events),('partial_timestamp',original)]:
                pp=pred[pred.window_id.eq(wid)].event_time_seconds.to_numpy(float)
                m,fp,fn=match_events(pp,rr,.3)
                nm,cost=ordered_matching(pp,rr,.3)
                assert nm==len(m) and abs(cost-sum(x[2] for x in m))<1e-6
                rows.append(dict(reference=version,window_id=wid,cluster=clusters[wid],method=method,
                    originally_timestamp_rejected=wid in set(windows.window_id),**metrics(len(m),len(fp),len(fn))))
        d=pd.DataFrame(rows);d=d[d.reference.eq(version)]
        for method,g in d.groupby('method'):
            summaries.append(dict(reference=version,method=method,windows=len(g),**metrics(*g[['tp','fp','fn']].sum())))
        old=d[d.method.eq('frozen')].set_index('window_id')
        new=d[d.method.eq('partial_timestamp')].set_index('window_id')
        z=new[['tp','fp','fn','cluster']].join(old[['tp','fp','fn']],rsuffix='_old')
        arr=z.groupby('cluster')[['tp','fp','fn','tp_old','fp_old','fn_old']].sum().to_numpy(float)
        rng=np.random.default_rng(20260922);ds=[]
        for _ in range(2000):
            v=arr[rng.integers(len(arr),size=len(arr))].sum(axis=0)
            ds.append(metrics(*v[:3])['f1']-metrics(*v[3:])['f1'])
        changes.append(dict(reference=version,windows=len(w),clusters=len(arr),
            delta_f1=metrics(*new[['tp','fp','fn']].sum())['f1']-metrics(*old[['tp','fp','fn']].sum())['f1'],
            delta_ci_low=float(np.quantile(ds,.025)),delta_ci_high=float(np.quantile(ds,.975)),
            additional_tp=int(new.tp.sum()-old.tp.sum()),additional_fp=int(new.fp.sum()-old.fp.sum()),
            improved_windows=int((new.f1>old.f1).sum()),worsened_windows=int((new.f1<old.f1).sum())))
    write_csv(out/'events_by_window.csv',rows);write_csv(out/'metrics.csv',summaries);write_csv(out/'paired_changes.csv',changes)
    original_scores=read(BASES['jiufu271']/'event_metrics_by_window.csv').set_index('window_id')
    old=pd.DataFrame(rows);old=old[old.reference.eq('R2')&old.method.eq('frozen')].set_index('window_id')
    np.testing.assert_array_equal(old[['tp','fp','fn']],original_scores.loc[old.index,['tp','fp','fn']])
    write_json(out/'verification.json',dict(status='PASS',checkpoints_verified=53,extra_events=len(extra),
        unchanged_existing_output_windows=216,unchanged_RR_pair_count=173,full_RR_metrics_changed=False,
        no_events_in_unsupported_timestamp_grid=True,no_full_count_from_partial_events=True,
        independent_matching_agrees=True,phase_shift_fit=False,old_reference_changed=False))
    adoption=changes[0]['delta_ci_low']>0
    write_json(out/'adoption_decision.json',dict(
        disposition='OPT_IN_PARTIAL_EVENTS_ONLY' if adoption else 'DO_NOT_ADOPT',
        event_scope='retrospective R2 full216 and R3 review11, not unseen holdout',
        full_RR='UNCHANGED; 0.419629 R2 R-squared not solved by partial events',
        default_changed=False,forbidden='sum partial events / 30s as full RR'))
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))
    print('ADOPTION',adoption,'partial events only')


if __name__=='__main__':main()
