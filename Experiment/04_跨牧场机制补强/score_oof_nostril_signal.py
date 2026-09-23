"""Same-reference event matching and common-support count comparison."""
import json
import numpy as np
import pandas as pd
from analyze_transfer import HERE,BASES,read,write_csv,write_json,sha256,metrics,match_events
from run_oof_nostril_signal import OUT


def main():
    seal=json.loads((OUT/'prediction_seal.json').read_text())
    for n,h in seal['files'].items():assert sha256(OUT/n)==h
    w=read(OUT/'prediction_windows.csv');e=read(OUT/'prediction_events.csv')
    old=read(BASES['jiufu271']/'prediction_windows.csv')
    old=old[old.window_id.isin(w.window_id)].copy();old['method']='original_pipeline'
    oe=read(BASES['jiufu271']/'prediction_events.csv');oe=oe[oe.window_id.isin(w.window_id)].copy();oe['method']='original_pipeline'
    windows=pd.concat([w,old],ignore_index=True);events=pd.concat([e,oe],ignore_index=True)
    audit=read(HERE/'20260921_r3_v1/annotation_audit.csv')
    audit=audit[audit.cohort.eq('jiufu271')&audit.eligible_full_window]
    truth_windows=read(HERE/'20260921_r3_v1/raw/annotation_windows.csv').set_index('window_id')
    truth_events=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    methods=['original_pipeline','frozen_direct','unfrozen_direct']
    detail=[];pairs=[];event_detail=[];summaries=[]
    for cohort,subset in [('R3_full11',audit),('R3_strict9',audit[audit.strict_visible_phase])]:
        assert len(subset)==(11 if cohort=='R3_full11' else 9)
        common=set(subset.window_id)
        for method in methods:common &= set(windows[windows.method.eq(method)&windows.prediction_status.eq('ok')].window_id)
        for r in subset.itertuples():
            t=truth_events[truth_events.window_id.eq(r.window_id)&truth_events.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
            assert len(t)==float(truth_windows.loc[r.window_id,'manual_breath_count'])
            for method in methods:
                p=events[events.method.eq(method)&events.window_id.eq(r.window_id)].sort_values('event_time_seconds')
                times=p.event_time_seconds.to_numpy(float)
                matched,fp,fn=match_events(times,t,.3)
                pw=windows[windows.method.eq(method)&windows.window_id.eq(r.window_id)].iloc[0]
                if pw.prediction_status=='ok':assert float(pw.predicted_count)==len(times)
                else:assert len(times)==0
                detail.append(dict(cohort=cohort,method=method,window_id=r.window_id,review_id=r.review_id,
                    prediction_status=pw.prediction_status,truth_count=len(t),predicted_events=len(times),
                    **metrics(len(matched),len(fp),len(fn))))
                # Explicit per-event times retain unmatched peaks and missed human events.
                for i,j,_ in matched:event_detail.append(dict(cohort=cohort,method=method,window_id=r.window_id,
                    status='TP',predicted_time=times[i],reference_time=t[j],absolute_error=abs(times[i]-t[j])))
                for i in fp:event_detail.append(dict(cohort=cohort,method=method,window_id=r.window_id,status='FP',predicted_time=times[i],reference_time=None))
                for j in fn:event_detail.append(dict(cohort=cohort,method=method,window_id=r.window_id,status='FN',predicted_time=None,reference_time=t[j]))
                if r.window_id in common:
                    duration=float(truth_windows.loc[r.window_id,'duration_seconds'])
                    pairs.append(dict(cohort=cohort,method=method,window_id=r.window_id,
                                      truth_rr=len(t)*60/duration,predicted_rr=len(times)*60/duration))
        for method in methods:
            d=pd.DataFrame(detail);d=d[d.cohort.eq(cohort)&d.method.eq(method)]
            pp=pd.DataFrame(pairs);pp=pp[pp.cohort.eq(cohort)&pp.method.eq(method)]
            error=pp.predicted_rr-pp.truth_rr
            denominator=((pp.truth_rr-pp.truth_rr.mean())**2).sum()
            summaries.append(dict(cohort=cohort,method=method,event_windows=len(d),output_windows=int(d.prediction_status.eq('ok').sum()),
                common_count_pairs=len(pp),rr_mae_common=float(error.abs().mean()),
                rr_r2_common=float(1-(error**2).sum()/denominator) if denominator>0 else None,
                **metrics(*d[['tp','fp','fn']].sum())))
    baseline=next(s for s in summaries if s['cohort']=='R3_full11' and s['method']=='original_pipeline')
    assert [baseline[k] for k in ['tp','fp','fn']]==[41,42,102]
    write_csv(OUT/'event_metrics_by_window.csv',detail);write_csv(OUT/'event_correspondence.csv',event_detail)
    write_csv(OUT/'common_count_pairs.csv',pairs);write_csv(OUT/'metrics.csv',summaries)
    write_json(OUT/'scoring_protocol.json',dict(reference='R3 existing eligible_full_window and strict_visible_phase',
        tolerance_seconds=.3,original_full11_counts_reproduced=[41,42,102],
        count_cohort='intersection of output windows across all3 methods within each reference set',
        abstention='No output treated as no predicted events in event recall; never imputed zero count for RR pairs',
        restrictions='Not full271 evaluation, not new independent external test; time-policy rejected windows remain abstentions',
        original_pipeline_not_same_pooling_control=True,default_changed=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':main()
