"""Isolated retrospective analysis of sealed predictions, not a new holdout test."""
from pathlib import Path
import sys
import json
import argparse
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment_common import REPO, ABLATION, HOLDOUT, REFERENCE, sha256, write_csv, write_json
from report_event_submission import ordered_matching
sys.path.insert(0, str(REFERENCE))
from reference_tools import match_events

BASES = {
    'lindian49': ABLATION / 'reference_updates/20260917_r2_v2/F1G1P1',
    'jiufu271': HOLDOUT / 'event_evaluation/20260918_r2_status_v1',
}
SUBMISSION = REFERENCE / 'submissions/20260918_r2_status_v1/raw'
TRANSFER = HOLDOUT / 'external_transfer/20260914_v2/windows'
RUN = ABLATION / 'runs/20260909_v1'
SEED = 20260921
TOLERANCES = [.1, .2, .3, .5, .75, 1.0]


def read(path):
    return pd.read_csv(path, encoding='utf-8-sig')


def metrics(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/(tp+fn)) if tp+fn else 0.,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)


def bootstrap_delta(table):
    groups = [g[['tp', 'fp', 'fn', 'base_tp', 'base_fp', 'base_fn']].sum().to_numpy()
              for _, g in table.groupby('cluster')]
    arr = np.asarray(groups)
    rng = np.random.default_rng(SEED)
    ds = []
    for _ in range(2000):
        v = arr[rng.integers(len(arr), size=len(arr))].sum(axis=0)
        ds.append(metrics(*v[:3])['f1'] - metrics(*v[3:])['f1'])
    return dict(clusters=len(arr), draws=2000, seed=SEED,
                delta_f1_ci_low=float(np.quantile(ds, .025)),
                delta_f1_ci_high=float(np.quantile(ds, .975)))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    source_paths = [SUBMISSION/n for n in ['annotation_windows.csv', 'reference_events.csv', 'unobservable_intervals.csv']]
    for folder in BASES.values():
        source_paths += [folder/n for n in ['event_metrics_by_window.csv', 'prediction_events.csv',
                                           'paired_count_results.csv', 'prediction_windows.csv', 'cluster_uncertainty.json']]
    source_paths += [RUN/'paired_predictions.csv', Path(__file__),
                     HOLDOUT/'signal_observability_guard.py']
    before = {str(p):sha256(p) for p in source_paths}
    spec = importlib.util.spec_from_file_location('guard', HOLDOUT/'signal_observability_guard.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    write_json(out/'analysis_lock.json', dict(
        role='retrospective_diagnostic_not_independent_new_test', seed=SEED,
        main_tolerance_seconds=.30, sensitivity_tolerances=TOLERANCES,
        phase_shift_fitting=False, truth_used_to_generate_candidate=False,
        candidate='same_existing_guard_mask_evaluated_at_each_frozen_peak_not_entire_window',
        policy=guard.POLICY, no_recovered_events_in_abstentions=True,
        adoption='no_default_replacement; require_nonnegative_source_F1_CI_and_R3_phase_visibility_review',
        comparison='within_each_cohort_same_frozen_peaks_and_inputs; between_cohort_pipeline_confound_remains',
        references='R2 20260918; playing-Q response latency unknown; R3 not yet available'))
    windows = read(SUBMISSION/'annotation_windows.csv')
    refs = read(SUBMISSION/'reference_events.csv')
    refs = refs[refs.confidence.eq('confirmed')]
    legacy = read(RUN/'paired_predictions.csv')
    legacy = legacy[legacy.cohort.eq('anchored49') & legacy.variant.eq('F1G1P1')].set_index('video_id')
    reference_groups = {w:g.event_time_seconds.to_numpy(float) for w,g in refs.groupby('window_id')}
    tolerance_rows, losses, distributions, cancellation, frequency, candidate_rows, audits = [], [], [], [], [], [], []
    kept_events, feature_rows, source_details, signed_errors = [], [], [], []
    for cohort, folder in BASES.items():
        ev = read(folder/'event_metrics_by_window.csv')
        pred = read(folder/'prediction_events.csv')
        paired = read(folder/'paired_count_results.csv')
        ws = windows[windows.cohort.eq(cohort)].set_index('window_id')
        truth_complete = ws[ws.annotation_status.eq('complete')]
        assert set(ev.window_id) == set(truth_complete.index)
        y, p = paired.truth_rr_bpm.to_numpy(), paired.predicted_rr_bpm.to_numpy()
        distributions.append(dict(cohort=cohort, n=len(y), scope='complete_and_output_only',
            mean=float(y.mean()), sd=float(y.std(ddof=1)), min=float(y.min()), max=float(y.max()),
            variance_population=float(y.var()), mse=float(np.mean((p-y)**2)),
            rr_r2=float(1-np.mean((p-y)**2)/y.var()), below20=int((y<20).sum()),
            mae=float(np.mean(abs(p-y)))))
        for status, g in ev.groupby('prediction_status'):
            losses.append(dict(cohort=cohort, status=status, windows=len(g),
                               truth_events=int(g.truth_count.sum()), **metrics(*g[['tp','fp','fn']].sum())))
        for row in ev.itertuples():
            # With maximum one-to-one matching, count error is FP-FN exactly.
            if row.prediction_status=='ok':
                assert row.predicted_count-row.truth_count == row.fp-row.fn
            else:
                assert row.tp==0 and row.fp==0 and row.fn==row.truth_count
            cancellation.append(dict(cohort=cohort, window_id=row.window_id, status=row.prediction_status,
                count_error=int(row.fp-row.fn), unmatched_events=int(row.fp+row.fn),
                cancelling_pairs=int(min(row.fp,row.fn)), exact_count=row.fp==row.fn,
                f1=metrics(row.tp,row.fp,row.fn)['f1']))
        for low, high in [(0,20),(20,40),(40,60),(60,np.inf)]:
            g=paired[(paired.truth_rr_bpm>=low)&(paired.truth_rr_bpm<high)]
            em=ev[ev.window_id.isin(g.window_id)]
            frequency.append(dict(cohort=cohort, band=f'[{low},{high})', n=len(g),
                rr_mae_bpm=float(abs(g.predicted_rr_bpm-g.truth_rr_bpm).mean()) if len(g) else None,
                bias_bpm=float((g.predicted_rr_bpm-g.truth_rr_bpm).mean()) if len(g) else None,
                **metrics(*em[['tp','fp','fn']].sum())))
        pg = {w:g for w,g in pred.groupby('window_id')}
        for tol in TOLERANCES:
            totals = np.zeros(3, dtype=int)
            for row in ev.itertuples():
                pp=pg.get(row.window_id, pred.iloc[:0])
                rt=reference_groups.get(row.window_id, np.array([]))
                matches, fp, fn=match_events(pp.event_time_seconds, rt, tol)
                totals += [len(matches),len(fp),len(fn)]
                independent, cost=ordered_matching(pp.event_time_seconds,rt,tol)
                assert independent==len(matches)
                assert abs(cost-sum(m[2] for m in matches))<1e-6
                if tol==.3:
                    assert (len(matches),len(fp),len(fn))==(row.tp,row.fp,row.fn)
                    for i,j,error in matches:
                        signed_errors.append(dict(cohort=cohort,window_id=row.window_id,
                            predicted_minus_reference_seconds=float(pp.iloc[i].event_time_seconds-rt[j]),
                            scope='matched_at_0.30_only_truncated_not_latency_estimate'))
            tolerance_rows.append(dict(cohort=cohort,tolerance_seconds=tol,**metrics(*totals)))
        paired_cows = paired.set_index('window_id')
        for row in ev.itertuples():
            w=ws.loc[row.window_id]
            pp=pg.get(row.window_id,pred.iloc[:0])
            rt=reference_groups.get(row.window_id,np.array([]))
            if cohort=='jiufu271':
                cluster = str(w.source_path).replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0]
            else:
                cluster = str(w.source_path)
            if row.prediction_status!='ok':
                for variant in ['local_event_guard','whole_window_guard']:
                    candidate_rows.append(dict(cohort=cohort,window_id=row.window_id,cluster=cluster,
                        variant=variant,base_tp=row.tp,base_fp=row.fp,base_fn=row.fn,
                        count_status='abstain', **metrics(0,0,len(rt))))
                continue
            if cohort=='jiufu271':
                root=TRANSFER/row.window_id
                tpath,cpath=root/'temperatures.csv',root/'curve.csv'
                curve=read(cpath); mode=curve.selected_fusion_mode.iloc[0]
            else:
                vid=str(w.video_id)
                tpath=ABLATION/f'input_snapshots/20260909_v1/inputs/anchored49/{vid}.csv'
                cpath=RUN/f'curves/anchored49/F1G1P1/{vid}.csv'
                curve=read(cpath); mode=legacy.loc[vid,'selected_fusion_mode']
            source_details.extend([dict(path=str(z),sha256=sha256(z)) for z in [tpath,cpath]])
            table=read(tpath)
            assert len(curve)==len(table)
            mask,gaps,info=guard.protect_window(table,mode)
            indices=np.array([int(str(x)[1:]) for x in pp.event_id],dtype=int)
            assert set(indices)==set(curve.loc[curve.is_peak.astype(str).str.lower().eq('true'),'frame_index'].astype(int))
            safe=~mask.unsafe_event_context.to_numpy()
            direct=mask.selected_inputs_direct.to_numpy()
            spacing=np.hypot(table.left_x-table.right_x,table.left_y-table.right_y)
            feature_rows.append(dict(cohort=cohort,window_id=row.window_id,selected_mode=mode,
                direct_support_fraction=float(direct.mean()), safe_context_fraction=float(safe.mean()),
                longest_gap_frames=info['longest_selected_gap_frames'],
                nostril_spacing_median_px=float(np.nanmedian(spacing)),
                frame_count=len(table), rr_error=float(paired_cows.loc[row.window_id,'predicted_rr_bpm']-paired_cows.loc[row.window_id,'truth_rr_bpm'])))
            for variant in ['local_event_guard','whole_window_guard']:
                keep = safe[indices] if variant=='local_event_guard' else np.full(len(indices),info['prediction_status']=='ready')
                kp=pp[keep]
                matches,fp,fn=match_events(kp.event_time_seconds,rt,.3)
                status='complete_output' if info['prediction_status']=='ready' else 'partial_events_no_full_RR'
                candidate_rows.append(dict(cohort=cohort,window_id=row.window_id,cluster=cluster,variant=variant,
                    base_tp=row.tp,base_fp=row.fp,base_fn=row.fn,count_status=status,
                    **metrics(len(matches),len(fp),len(fn))))
                if variant=='local_event_guard':
                    for event, idx, retained in zip(pp.itertuples(),indices,keep):
                        kept_events.append(dict(cohort=cohort,window_id=row.window_id,event_id=event.event_id,
                            event_time_seconds=event.event_time_seconds,frame_index=int(idx),kept=bool(retained),
                            reason='supported_context' if retained else 'near_long_or_boundary_provenance_gap'))
                    audits.append(dict(cohort=cohort,window_id=row.window_id,
                        full_support=info['prediction_status']=='ready',
                        removed_events=int((~keep).sum()),total_events=len(keep)))
    cand=pd.DataFrame(candidate_rows)
    candidate_summary=[]
    for (cohort,variant),g in cand.groupby(['cohort','variant']):
        m=metrics(*g[['tp','fp','fn']].sum())
        base=metrics(*g[['base_tp','base_fp','base_fn']].sum())
        candidate_summary.append(dict(cohort=cohort,variant=variant,windows=len(g),**m,
            delta_f1=m['f1']-base['f1'],**bootstrap_delta(g),
            full_count_output_windows=int(g.count_status.eq('complete_output').sum()),
            disposition='DIAGNOSTIC_ONLY_PENDING_R3_NO_DEFAULT_CHANGE'))
    tables = dict(tolerance_sensitivity=tolerance_rows,loss_decomposition=losses,
        rr_distribution=distributions,count_event_cancellation=cancellation,rr_strata=frequency,
        candidate_by_window=candidate_rows,candidate_summary=candidate_summary,
        candidate_event_decisions=kept_events,observability_features=feature_rows,
        candidate_coverage=audits,matched_signed_errors=signed_errors,signal_input_hashes=source_details)
    for name,rows in tables.items(): write_csv(out/f'{name}.csv',rows)
    # These plots contain recorded data only; no synthetic breathing waveform is used.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#087f8c','#cd6b26']
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for color,(cohort,folder) in zip(colors,BASES.items()):
        d=read(folder/'paired_count_results.csv')
        axs[0].hist(d.truth_rr_bpm,bins=np.arange(0,81,5),histtype='step',lw=2,color=color,label=f'{cohort} n={len(d)}')
        axs[1].scatter(d.truth_rr_bpm,d.predicted_rr_bpm,s=14,alpha=.65,color=color)
    axs[0].set(xlabel='Reference RR (breaths/min)',ylabel='Paired windows');axs[0].legend(fontsize=8)
    axs[1].plot([0,80],[0,80],color='gray',ls='--');axs[1].set(xlabel='Reference RR',ylabel='Predicted RR',xlim=(0,80),ylim=(0,80))
    fig.savefig(out/'figures/01_distribution_scatter.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    td=pd.DataFrame(tolerance_rows)
    for color,(cohort,g) in zip(colors,td.groupby('cohort',sort=False)):
        axs[0].plot(g.tolerance_seconds,g.f1,'o-',color=color,label=cohort)
    axs[0].axvline(.3,color='gray',ls='--');axs[0].set(xlabel='Matching tolerance (s)',ylabel='Event F1',ylim=(0,1));axs[0].legend()
    ld=pd.DataFrame(losses); ext=ld[ld.cohort.eq('jiufu271')]
    axs[1].bar(['TP','FP','FN output','FN abstain'],[ext.tp.sum(),ext.fp.sum(),int(ext.loc[ext.status.eq('ok'),'fn'].sum()),int(ext.loc[ext.status.ne('ok'),'fn'].sum())],color=['#087f8c','#c74343','#cd6b26','#74858d'])
    axs[1].set(ylabel='Events',title='Jiufu complete references (n=216)')
    fig.savefig(out/'figures/02_events_and_tolerance.png',dpi=200);plt.close(fig)
    ab=read(ABLATION/'reference_updates/20260917_r2_v2/metrics_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    axs[0].plot(ab.variant,ab.rr_r2,'o-',label='RR R2',color=colors[0]);axs[0].plot(ab.variant,ab.f1,'s-',label='Event F1',color=colors[1]);axs[0].tick_params(axis='x',rotation=45);axs[0].legend();axs[0].set(ylim=(0,1),title='Same-input internal ablation (47 windows)')
    cd=pd.DataFrame(cancellation)
    for color,(cohort,g) in zip(colors,cd.groupby('cohort',sort=False)):
        g=g[g.status.eq('ok')];axs[1].scatter(abs(g.count_error),g.unmatched_events,s=15,alpha=.6,color=color,label=cohort)
    axs[1].set(xlabel='Absolute count error',ylabel='FP + FN at 0.30 s');axs[1].legend()
    fig.savefig(out/'figures/03_ablation_cancellation.png',dpi=200);plt.close(fig)
    # Deterministic illustrative cases are separate from the blinded review selection.
    for cohort,folder in BASES.items():
        cd0=cd[(cd.cohort.eq(cohort))&(cd.status.eq('ok'))&(cd.exact_count)]
        chosen=cd0.sort_values(['unmatched_events','window_id'],ascending=[False,True]).iloc[0].window_id
        w=windows.set_index('window_id').loc[chosen]
        if cohort=='jiufu271':
            cp=TRANSFER/chosen/'curve.csv';c=read(cp);t=c.target_time_seconds
        else:
            cp=RUN/f'curves/anchored49/F1G1P1/{w.video_id}.csv';c=read(cp)
            t=read(HOLDOUT/f'adapter_validation/20260912_v2/decoded_timestamps/{w.video_id}.csv').annotation_video_time_seconds
        pp=read(folder/'prediction_events.csv');pp=pp[pp.window_id.eq(chosen)]
        rr=reference_groups[chosen]
        fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,2,1]})
        axs[0].plot(t,c.left_temp,label='Left',color=colors[0]);axs[0].plot(t,c.right_temp,label='Right',color=colors[1]);axs[0].set_ylabel('RF proxy');axs[0].legend(ncol=2)
        axs[1].plot(t,c.fused_norm,color='#91bbc8',label='Fused');axs[1].plot(t,c.smoothed_norm,color='#175b7d',label='Smoothed');axs[1].set_ylabel('Normalized');axs[1].legend(ncol=2)
        axs[2].eventplot([pp.event_time_seconds,rr],lineoffsets=[1,0],linelengths=.6,colors=['#c74343','#22855e']);axs[2].set(yticks=[0,1],yticklabels=['R2 human','Algorithm'],xlabel='Annotation viewing time (s)',xlim=(0,float(w.duration_seconds)))
        fig.suptitle(f'{cohort} / {chosen}\nEqual total counts do not imply event agreement (illustrative, not random)')
        fig.savefig(out/f'figures/04_case_{cohort}.png',dpi=200);plt.close(fig)
    assert all(sha256(Path(p))==h for p,h in before.items())
    write_json(out/'verification.json',dict(status='PASS',source_inputs_unchanged=True,
        source_hashes=before,matcher='Hungarian independently checked by ordered DP at all six tolerances',
        baseline_per_window_counts_verified=True,default_algorithm_changed=False,
        candidate_peak_indices_verified=True,reference_modified=False,
        limitation='Neither causal farm attribution nor human physiological phase validation'))
    print(json.dumps(dict(distributions=distributions,candidates=candidate_summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'20260921_v1/analysis_v3')
    run(parser.parse_args().out)
