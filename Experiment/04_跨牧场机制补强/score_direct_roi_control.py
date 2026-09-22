"""Evaluate a sealed direct-ROI control using existing exact R2/R3 analysis sets."""
import json
import numpy as np
import pandas as pd
from analyze_transfer import HERE,BASES,SUBMISSION,read,write_csv,write_json,sha256,metrics,match_events

ROOT=HERE/'20260922_pose_phase_v1/direct_roi_control_v1'


def main():
    seal=json.loads((ROOT/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,h in seal['files'].items():assert sha256(ROOT/name)==h
    new=read(ROOT/'prediction_windows.csv').set_index('window_id');ne=read(ROOT/'prediction_events.csv')
    old=read(BASES['jiufu271']/'prediction_windows.csv').set_index('window_id');oe=read(BASES['jiufu271']/'prediction_events.csv')
    assert set(new.index)==set(old.index) and new.prediction_status.eq(old.loc[new.index,'prediction_status']).all()
    results=[];event_rows=[];pairs=[];deltas=[]
    for reference in ['R2','R3']:
        if reference=='R2':
            w=read(SUBMISSION/'annotation_windows.csv');w=w[w.cohort.eq('jiufu271')&w.annotation_status.eq('complete')].set_index('window_id')
            truth=read(SUBMISSION/'reference_events.csv')
            clusters={wid:str(r.source_path).replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0] for wid,r in w.iterrows()}
        else:
            w=read(HERE/'20260921_r3_v1/raw/annotation_windows.csv').set_index('window_id')
            a=read(HERE/'20260921_r3_v1/annotation_audit.csv');a=a[a.cohort.eq('jiufu271')&a.eligible_full_window]
            w=w.loc[a.window_id];clusters=a.set_index('window_id').cluster.astype(str).to_dict()
            truth=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
        for wid,r in w.iterrows():
            tt=truth[truth.window_id.eq(wid)&truth.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
            assert len(tt)==float(r.manual_breath_count)
            for method,p,e in [('frozen',old,oe),('direct_roi_only',new,ne)]:
                pp=e[e.window_id.eq(wid)].event_time_seconds.to_numpy(float)
                m,fp,fn=match_events(pp,tt,.3)
                event_rows.append(dict(reference=reference,method=method,window_id=wid,cluster=clusters[wid],**metrics(len(m),len(fp),len(fn))))
                if p.loc[wid,'prediction_status']=='ok':
                    count=p.loc[wid,'predicted_count'];assert count==len(pp)
                    pairs.append(dict(reference=reference,method=method,window_id=wid,cluster=clusters[wid],truth_count=len(tt),predicted_count=count,
                        truth_rr=len(tt)*60/float(r.duration_seconds),predicted_rr=count*60/float(r.duration_seconds)))
        e=pd.DataFrame(event_rows);e=e[e.reference.eq(reference)]
        p=pd.DataFrame(pairs);p=p[p.reference.eq(reference)]
        for method in ['frozen','direct_roi_only']:
            g=p[p.method.eq(method)];h=e[e.method.eq(method)];error=g.predicted_rr-g.truth_rr
            results.append(dict(reference=reference,method=method,event_windows=len(h),count_pairs=len(g),
                rr_r2=float(1-(error**2).sum()/((g.truth_rr-g.truth_rr.mean())**2).sum()),rr_mae=float(error.abs().mean()),
                **metrics(*h[['tp','fp','fn']].sum())))
        a=e[e.method.eq('frozen')].set_index('window_id')[['cluster','tp','fp','fn']]
        a=a.join(e[e.method.eq('direct_roi_only')].set_index('window_id')[['tp','fp','fn']],rsuffix='_new')
        ea=a.groupby('cluster')[['tp','fp','fn','tp_new','fp_new','fn_new']].sum().to_numpy(float)
        b=p[p.method.eq('frozen')].set_index('window_id');c=p[p.method.eq('direct_roi_only')].set_index('window_id')
        b['mae_delta']=(c.predicted_rr-b.truth_rr).abs()-(b.predicted_rr-b.truth_rr).abs()
        pa=[g.mae_delta.to_numpy() for _,g in b.groupby('cluster')]
        rng=np.random.default_rng(20260922);fd=[];md=[]
        for _ in range(2000):
            x=ea[rng.integers(len(ea),size=len(ea))].sum(axis=0)
            fd.append(metrics(*x[3:])['f1']-metrics(*x[:3])['f1'])
            md.append(np.concatenate([pa[i] for i in rng.integers(len(pa),size=len(pa))]).mean())
        deltas.append(dict(reference=reference,rr_mae_delta=float(b.mae_delta.mean()),rr_mae_ci_low=float(np.quantile(md,.025)),
            rr_mae_ci_high=float(np.quantile(md,.975)),event_f1_ci_low=float(np.quantile(fd,.025)),event_f1_ci_high=float(np.quantile(fd,.975))))
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    write_csv(out/'metrics.csv',results);write_csv(out/'paired_count_results.csv',pairs)
    write_csv(out/'event_metrics_by_window.csv',event_rows);write_csv(out/'paired_deltas.csv',deltas)
    provisional=deltas[0]['rr_mae_ci_high']<0 and deltas[0]['event_f1_ci_low']>=0
    write_json(out/'adoption_decision.json',dict(disposition='REQUIRES_INTERNAL_REGRESSION' if provisional else 'DO_NOT_ADOPT',
        default_changed=False,retrospective_only=True,no_anatomical_truth_used_per_prediction=True,
        kept_R2_count_pairs=173,kept_R2_event_windows=216))
    print(pd.DataFrame(results).to_string(index=False));print(pd.DataFrame(deltas).to_string(index=False))


if __name__=='__main__':main()
