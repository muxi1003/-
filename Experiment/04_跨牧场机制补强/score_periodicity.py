"""Separate evaluation of the already sealed single selector candidate."""
import json
import numpy as np
import pandas as pd
from analyze_transfer import HERE, BASES, SUBMISSION, read, write_csv, write_json, sha256, metrics, match_events

ROOT=HERE/'20260921_chain_v3/observed_periodicity'


def main():
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    seal=json.loads((ROOT/'prediction_seal.json').read_text(encoding='utf-8'))
    for n,h in seal['files'].items():assert sha256(ROOT/n)==h
    cp=read(ROOT/'prediction_windows.csv').set_index('window_id')
    ce=read(ROOT/'prediction_events.csv')
    bp=read(BASES['jiufu271']/'prediction_windows.csv').set_index('window_id')
    be=read(BASES['jiufu271']/'prediction_events.csv')
    assert set(cp.index)==set(bp.index) and cp.prediction_status.eq(bp.loc[cp.index,'prediction_status']).all()
    summary=[];paired=[];events=[];deltas=[]
    for ref in ['R2','R3']:
        if ref=='R2':
            w=read(SUBMISSION/'annotation_windows.csv')
            w=w[w.cohort.eq('jiufu271')&w.annotation_status.eq('complete')].set_index('window_id')
            truth=read(SUBMISSION/'reference_events.csv')
            cluster={wid:str(r.source_path).replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0] for wid,r in w.iterrows()}
        else:
            w=read(HERE/'20260921_r3_v1/raw/annotation_windows.csv').set_index('window_id')
            a=read(HERE/'20260921_r3_v1/annotation_audit.csv')
            a=a[a.cohort.eq('jiufu271')&a.eligible_full_window]
            w=w.loc[a.window_id];cluster=a.set_index('window_id').cluster.astype(str).to_dict()
            truth=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
        for wid,r in w.iterrows():
            tt=truth[truth.window_id.eq(wid)&truth.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
            for method,p,e in [('frozen',bp,be),('observed_periodicity',cp,ce)]:
                pp=e[e.window_id.eq(wid)].event_time_seconds.to_numpy(float)
                m,fp,fn=match_events(pp,tt,.3)
                events.append(dict(reference=ref,window_id=wid,cluster=cluster[wid],method=method,**metrics(len(m),len(fp),len(fn))))
                if p.loc[wid,'prediction_status']=='ok':
                    count=p.loc[wid,'predicted_count'];assert count==len(pp)
                    true_count=float(r.manual_breath_count)
                    assert true_count==len(tt)
                    paired.append(dict(reference=ref,window_id=wid,cluster=cluster[wid],method=method,
                        truth_count=true_count,predicted_count=count,truth_rr=true_count*60/float(r.duration_seconds),
                        predicted_rr=count*60/float(r.duration_seconds)))
        ed=pd.DataFrame(events);ed=ed[ed.reference.eq(ref)]
        pd_=pd.DataFrame(paired);pd_=pd_[pd_.reference.eq(ref)]
        for method in ['frozen','observed_periodicity']:
            eg=ed[ed.method.eq(method)];g=pd_[pd_.method.eq(method)]
            error=g.predicted_rr-g.truth_rr
            summary.append(dict(reference=ref,method=method,event_windows=len(eg),count_pairs=len(g),
                rr_r2=float(1-(error**2).sum()/((g.truth_rr-g.truth_rr.mean())**2).sum()),rr_mae=float(error.abs().mean()),
                **metrics(*eg[['tp','fp','fn']].sum())))
        eg=ed[ed.method.eq('frozen')].set_index('window_id')[['tp','fp','fn','cluster']]
        eg=eg.join(ed[ed.method.eq('observed_periodicity')].set_index('window_id')[['tp','fp','fn']],rsuffix='_new')
        ea=eg.groupby('cluster')[['tp','fp','fn','tp_new','fp_new','fn_new']].sum().to_numpy(float)
        pg=pd_[pd_.method.eq('frozen')].set_index('window_id').copy()
        q=pd_[pd_.method.eq('observed_periodicity')].set_index('window_id')
        pg['delta_abs_rr']=(q.predicted_rr-pg.truth_rr).abs()-(pg.predicted_rr-pg.truth_rr).abs()
        pa=[g.delta_abs_rr.to_numpy() for _,g in pg.groupby('cluster')]
        rng=np.random.default_rng(20260922);ds=[];ms=[]
        for _ in range(2000):
            x=ea[rng.integers(len(ea),size=len(ea))].sum(axis=0)
            ds.append(metrics(*x[3:])['f1']-metrics(*x[:3])['f1'])
            ms.append(np.concatenate([pa[i] for i in rng.integers(len(pa),size=len(pa))]).mean())
        deltas.append(dict(reference=ref,rr_mae_delta=float(pg.delta_abs_rr.mean()),
            rr_mae_ci_low=float(np.quantile(ms,.025)),rr_mae_ci_high=float(np.quantile(ms,.975)),
            event_f1_ci_low=float(np.quantile(ds,.025)),event_f1_ci_high=float(np.quantile(ds,.975))))
    write_csv(out/'metrics.csv',summary);write_csv(out/'paired_count_results.csv',paired)
    write_csv(out/'event_metrics_by_window.csv',events);write_csv(out/'paired_deltas.csv',deltas)
    passed=deltas[0]['rr_mae_ci_high']<0 and deltas[0]['event_f1_ci_low']>=0
    write_json(out/'adoption_decision.json',dict(disposition='REQUIRES_INTERNAL_REGRESSION' if passed else 'DO_NOT_ADOPT',
        default_changed=False,scope='single retrospective candidate; no model or threshold search',unchanged_R2_pair_count=173))
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(deltas).to_string(index=False));print('passed',passed)


if __name__=='__main__':main()
