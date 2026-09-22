"""Audit R3 exports and compare unchanged predictions on identical R2/R3 windows."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_transfer import (HERE, BASES, SUBMISSION, read, metrics, match_events,
                              ordered_matching, sha256, write_csv, write_json)

REVIEW = HERE / '20260921_v1/review24'
FILES = ['annotation_windows.csv', 'reference_events.csv',
         'unobservable_intervals.csv', 'event_reference_R3_backup.json']
TOLS = [.1, .2, .3, .5, .75, 1.0]


def strings(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def canonical(records, columns):
    def val(x):
        if x is None:
            return ''
        if isinstance(x, bool):
            return str(x).lower()
        return str(x)
    return sorted(tuple(val(r.get(k, '')) for k in columns) for r in records)


def matched(p, r, tolerance):
    m, fp, fn = match_events(p, r, tolerance)
    n, cost = ordered_matching(p, r, tolerance)
    assert n == len(m) and abs(cost - sum(x[2] for x in m)) < 1e-6
    return m, metrics(len(m), len(fp), len(fn))


def rr_stats(counts, predictions, durations):
    y = np.asarray(counts, float) * 60 / np.asarray(durations, float)
    p = np.asarray(predictions, float) * 60 / np.asarray(durations, float)
    return dict(paired_windows=len(y), rr_mae_bpm=float(abs(y-p).mean()),
                rr_r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))
                if len(y)>1 and np.ptp(y)>0 else None)


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    raw = out / 'raw'; raw.mkdir()
    paths = [source/n for n in FILES]
    paths += [REVIEW/n for n in ['annotation_windows.csv', 'decoded_frame_pts.json',
                                'selection_key_DO_NOT_VIEW_DURING_REVIEW.csv']]
    paths += [SUBMISSION/n for n in FILES[:3]]
    for folder in BASES.values():
        paths += [folder/n for n in ['prediction_events.csv', 'prediction_windows.csv']]
    paths += [HERE/'20260921_v1/analysis_v3/candidate_event_decisions.csv', Path(__file__)]
    hashes = {str(p):sha256(p) for p in paths}
    for name in FILES:
        shutil.copy2(source/name, raw/name)
        assert sha256(source/name) == sha256(raw/name)
    w, e, iv = [strings(raw/n) for n in FILES[:3]]
    backup = json.loads((raw/FILES[3]).read_text(encoding='utf-8-sig'))
    for key, table in zip(['windows','events','intervals'],[w,e,iv]):
        assert canonical(table.to_dict('records'), table.columns) == canonical(backup[key], table.columns), key
    initial = strings(REVIEW/'annotation_windows.csv').set_index('window_id')
    mapping = strings(REVIEW/'selection_key_DO_NOT_VIEW_DURING_REVIEW.csv').set_index('window_id')
    pts = json.loads((REVIEW/'decoded_frame_pts.json').read_text(encoding='utf-8'))
    assert len(w)==24 and w.window_id.is_unique and set(w.window_id)==set(initial.index)
    assert e.event_id.is_unique and set(e.window_id)<=set(w.window_id)
    assert set(w.annotation_round)=={'R3'} and set(e.annotation_round)=={'R3'}
    r2w = strings(SUBMISSION/'annotation_windows.csv').set_index('window_id')
    r2e = read(SUBMISSION/'reference_events.csv')
    r2e = r2e[r2e.confidence.eq('confirmed')]
    audits, issues = [], []
    for row in w.itertuples():
        wid = row.window_id
        for col in ['source_path','browser_video_path','source_start_seconds','view_start_seconds','duration_seconds']:
            assert getattr(row,col)==initial.loc[wid,col], (wid,col)
        ev = e[e.window_id.eq(wid)]
        tt = ev.event_time_seconds.to_numpy(float)
        assert np.all(np.isfinite(tt)) and np.all((tt>=0)&(tt<float(row.duration_seconds)))
        assert len(set(tt))==len(tt), ('duplicate_times',wid)
        frame_error = max((min(abs(np.asarray(pts[wid])-t)) for t in tt), default=0.)
        assert frame_error < .00101, ('not_decoded_PTS',wid,frame_error)
        confirmed = ev.confidence.eq('confirmed').sum()
        uncertain = ev.confidence.ne('confirmed').sum()
        if row.annotation_status=='complete':
            assert float(row.manual_breath_count)==confirmed and uncertain==0
        discrepancy = len(ev)>0 and not ev.phase_basis.eq(row.phase_basis).all()
        if discrepancy:
            issues.append(dict(review_id=row.video_id,window_id=wid,issue='window_event_phase_disagreement',
                               detail=f'window={row.phase_basis}; events={ev.phase_basis.value_counts().to_dict()}'))
        # A declared off-screen suffix contradicts whole-window completeness.
        # Flag only; never manufacture interval boundaries or rewrite annotation status.
        note_conflict = row.annotation_status=='complete' and '结尾' in row.reference_notes and '范围外' in row.reference_notes
        if note_conflict:
            issues.append(dict(review_id=row.video_id,window_id=wid,issue='complete_vs_unobservable_note',detail=row.reference_notes))
        if row.annotation_status!='complete':
            issues.append(dict(review_id=row.video_id,window_id=wid,issue='not_complete',detail=row.annotation_status))
        valid = row.annotation_status=='complete' and not note_conflict
        visible = valid and row.phase_basis=='visible_expiration' and ev.phase_basis.eq('visible_expiration').all()
        audits.append(dict(review_id=row.video_id,window_id=wid,cohort=mapping.loc[wid,'cohort'],
            sampling_stratum=mapping.loc[wid,'sampling_stratum'],cluster=mapping.loc[wid,'sampling_group'],
            exported_status=row.annotation_status,confirmed_events=int(confirmed),uncertain_events=int(uncertain),
            phase_basis=row.phase_basis,phase_field_disagreement=discrepancy,note_conflict=note_conflict,
            eligible_full_window=valid,strict_visible_phase=visible,frame_rounding_error_seconds=frame_error,
            reference_notes=row.reference_notes))
    audit=pd.DataFrame(audits)
    write_csv(out/'annotation_audit.csv',audit)
    write_csv(out/'clarifications.csv',issues)
    write_json(out/'analysis_protocol.json',dict(
        reference='R3 paired repeat-observer diagnostic; R2 original retained',tolerances=TOLS,
        main_tolerance=.3, fitted_phase_shift=False,changed_predictions=False,
        full_window_rule='complete, count/event consistent, no explicit contradictory unobservable suffix note',
        strict_phase_rule='both window and every event visible_expiration',
        phase_disagreements='retain original values; broad descriptive set allowed, strict phase set excludes',
        no_intervals='empty export is absence of markings, not verified full observability',
        sampling='12 internal/12 external planned, external abstentions enriched; not representative holdout',
        date='2026-09-21'))
    prediction_events={c:read(p/'prediction_events.csv') for c,p in BASES.items()}
    prediction_windows={c:read(p/'prediction_windows.csv').set_index('window_id') for c,p in BASES.items()}
    guard=read(HERE/'20260921_v1/analysis_v3/candidate_event_decisions.csv')
    bywindow, reference_comparison, shifts, countchanges, plotdata = [], [], [], [], {}
    for a in audit[audit.eligible_full_window].itertuples():
        wid=a.window_id; row=w[w.window_id.eq(wid)].iloc[0]
        old=r2e[r2e.window_id.eq(wid)].event_time_seconds.to_numpy(float)
        new=e[e.window_id.eq(wid)&e.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
        duration=float(row.duration_seconds)
        assert abs(duration-float(r2w.loc[wid,'duration_seconds']))<1e-9
        pred=prediction_events[a.cohort]
        pp=pred[pred.window_id.eq(wid)].event_time_seconds.to_numpy(float)
        status=prediction_windows[a.cohort].loc[wid,'prediction_status']
        if status!='ok':
            assert len(pp)==0
        countchanges.append(dict(cohort=a.cohort,review_id=a.review_id,window_id=wid,cluster=a.cluster,
            strict_visible_phase=a.strict_visible_phase,r2_count=len(old),r3_count=len(new),
            r3_minus_r2_count=len(new)-len(old),duration_seconds=duration,prediction_status=status,
            predicted_count=len(pp) if status=='ok' else None))
        plotdata[wid]=(a,old,new,pp,duration)
        for tol in TOLS:
            mm,m=matched(old,new,tol)
            reference_comparison.append(dict(cohort=a.cohort,review_id=a.review_id,window_id=wid,
                strict_visible_phase=a.strict_visible_phase,tolerance_seconds=tol,**m))
            for i,j,d in mm:
                shifts.append(dict(cohort=a.cohort,review_id=a.review_id,window_id=wid,
                    tolerance_seconds=tol,r2_time=old[i],r3_time=new[j],r2_minus_r3_seconds=old[i]-new[j],
                    caveat='truncated_matched_subset_not_reaction_latency'))
            for version,refs in [('R2',old),('R3',new)]:
                _,m=matched(pp,refs,tol)
                bywindow.append(dict(cohort=a.cohort,review_id=a.review_id,window_id=wid,cluster=a.cluster,
                    strict_visible_phase=a.strict_visible_phase,prediction_status=status,version=version,
                    method='frozen_baseline',tolerance_seconds=tol,**m))
        gg=guard[guard.window_id.eq(wid)&guard.kept.astype(str).str.lower().eq('true')]
        kept=gg.event_time_seconds.to_numpy(float)
        assert all(np.min(abs(pp-t))<1e-9 for t in kept)
        _,m=matched(kept,new,.3)
        bywindow.append(dict(cohort=a.cohort,review_id=a.review_id,window_id=wid,cluster=a.cluster,
            strict_visible_phase=a.strict_visible_phase,prediction_status=status,version='R3',
            method='previously_rejected_local_guard',tolerance_seconds=.3,**m))
    bw=pd.DataFrame(bywindow); rc=pd.DataFrame(reference_comparison); ct=pd.DataFrame(countchanges)
    sd=pd.DataFrame(shifts)
    summary=[]; refsum=[]; delta=[]
    for scope in ['eligible_full_window','strict_visible_phase']:
        ids=set(audit.loc[audit[scope],'window_id'])
        for (cohort,version,method,tol),g in bw[bw.window_id.isin(ids)].groupby(['cohort','version','method','tolerance_seconds']):
            summary.append(dict(scope=scope,cohort=cohort,version=version,method=method,tolerance_seconds=tol,
                windows=len(g),output_windows=int(g.prediction_status.eq('ok').sum()),**metrics(*g[['tp','fp','fn']].sum())))
        for cohort,g in ct[ct.window_id.isin(ids)].groupby('cohort'):
            paired=g[g.prediction_status.eq('ok')]
            for version in ['R2','R3']:
                refsum.append(dict(scope=scope,cohort=cohort,version=version,windows=len(g),
                    mean_count=float(g[version.lower()+'_count'].mean()),
                    **rr_stats(paired[version.lower()+'_count'],paired.predicted_count,paired.duration_seconds)))
            s=bw[bw.window_id.isin(g.window_id)&bw.method.eq('frozen_baseline')&bw.tolerance_seconds.eq(.3)]
            a=s[s.version.eq('R2')].set_index('window_id').loc[g.window_id]
            b=s[s.version.eq('R3')].set_index('window_id').loc[g.window_id]
            # One distinct source/cow per selected window; resample matched windows.
            assert g.cluster.nunique()==len(g)
            rng=np.random.default_rng(20260921); ds=[]
            for _ in range(2000):
                ix=rng.integers(len(g),size=len(g))
                ds.append(metrics(*b.iloc[ix][['tp','fp','fn']].sum())['f1']-metrics(*a.iloc[ix][['tp','fp','fn']].sum())['f1'])
            delta.append(dict(scope=scope,cohort=cohort,windows=len(g),
                r3_minus_r2_f1=metrics(*b[['tp','fp','fn']].sum())['f1']-metrics(*a[['tp','fp','fn']].sum())['f1'],
                ci_low=float(np.quantile(ds,.025)),ci_high=float(np.quantile(ds,.975)),
                count_changed_windows=int(g.r3_minus_r2_count.ne(0).sum()),
                mean_absolute_reference_count_change=float(g.r3_minus_r2_count.abs().mean()),
                note='reference sensitivity, NOT model improvement; small stratified sample'))
    for name,table in [('events_by_window',bw),('r2_r3_reference_matching',rc),('r2_r3_matched_times',sd),
                       ('count_changes',ct),('event_metrics',pd.DataFrame(summary)),
                       ('rr_metrics',pd.DataFrame(refsum)),('reference_sensitivity',pd.DataFrame(delta))]:
        write_csv(out/(name+'.csv'),table)
    figdir=out/'figures';figdir.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    ss=pd.DataFrame(summary)
    for axis,cohort in zip(ax,['lindian49','jiufu271']):
        for version,color in [('R2','#087f8c'),('R3','#c75c24')]:
            z=ss[(ss.scope=='eligible_full_window')&(ss.cohort==cohort)&(ss.method=='frozen_baseline')&(ss.version==version)]
            axis.plot(z.tolerance_seconds,z.f1,'o-',label=version,color=color)
        axis.set(title=cohort+' (same reviewed windows)',xlabel='Tolerance (s)',ylabel='Frozen prediction event F1',ylim=(0,1));axis.legend()
        axis.axvline(.3,color='gray',ls='--')
    fig.savefig(figdir/'01_reference_sensitivity.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,5.5),layout='constrained')
    for axis,cohort in zip(axs,['lindian49','jiufu271']):
        # Deterministic illustration chosen by largest count disagreement, disclosed in caption.
        g=ct[ct.cohort.eq(cohort)].copy();g['diff']=g.r3_minus_r2_count.abs()
        wid=g.sort_values(['diff','window_id'],ascending=[False,True]).iloc[0].window_id
        a,old,new,pp,duration=plotdata[wid]
        axis.eventplot([old,new,pp],lineoffsets=[2,1,0],linelengths=.6,colors=['#087f8c','#c75c24','#333333'])
        axis.set(yticks=[0,1,2],yticklabels=['Frozen prediction','R3','R2'],xlim=(0,duration),xlabel='Viewing time (s)',
                 title=f'{a.review_id}: {cohort}; largest reference count difference (illustration only)')
    fig.savefig(figdir/'02_event_rasters.png',dpi=200);plt.close(fig)
    for path,h in hashes.items():
        assert sha256(Path(path))==h, ('input_changed',path)
    write_csv(out/'input_hashes.csv',[dict(path=p,sha256=h) for p,h in hashes.items()])
    write_json(out/'verification.json',dict(status='PASS_WITH_ANNOTATION_CLARIFICATIONS',
        submitted_windows=len(w),submitted_events=len(e),submitted_intervals=len(iv),
        csv_json_equal=True,window_members_and_time_binding_unchanged=True,
        eligible_windows=int(audit.eligible_full_window.sum()),strict_visible_windows=int(audit.strict_visible_phase.sum()),
        matching_independently_verified=True,source_hashes_unchanged=True,
        phase_disagreement_windows=int(audit.phase_field_disagreement.sum()),
        raw_labels_changed=False,default_predictions_changed=False))
    print(pd.DataFrame(summary).query("scope=='eligible_full_window' and tolerance_seconds==0.3").to_string(index=False))
    print(pd.DataFrame(refsum).to_string(index=False))
    print(pd.DataFrame(delta).to_string(index=False))
    print('AUDIT',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.out)
