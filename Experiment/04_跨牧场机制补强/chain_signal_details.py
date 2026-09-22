"""Align saved ROIs, missingness, curves, and human/frozen events without retuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_transfer import HERE, HOLDOUT, BASES, SUBMISSION, read, write_csv, write_json

ROOT=HERE/'20260921_chain_v3'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2/windows'


def main():
    a=read(ROOT/'all271_stage_audit.csv')
    old=read(SUBMISSION/'reference_events.csv')
    new=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    pred=read(BASES['jiufu271']/'prediction_events.csv')
    rows=[];gap_rows=[]
    for row in a.itertuples():
        ts=read(TRANSFER/row.window_id/'timestamps.csv').relative_seconds.to_numpy(float)
        gaps=[(ts[k],min(ts[k+1],30.)) for k in np.flatnonzero(np.diff(ts)>.5) if ts[k]<30]
        refs=old[old.window_id.eq(row.window_id)&old.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
        gap_rows.append(dict(window_id=row.window_id,reason=row.reason,large_gap_count=len(gaps),
            large_gap_seconds=sum(b-a for a,b in gaps),
            reference_events_in_gap_open_interior=sum(any(a<t<b for a,b in gaps) for t in refs)))
        if row.first_blocking_stage!='signal_available':continue
        c=read(TRANSFER/row.window_id/'curve.csv')
        t=read(TRANSFER/row.window_id/'temperatures.csv')
        raw_missing=~np.isfinite(t[['left_temp','right_temp']].to_numpy(float)).any(axis=1)
        runs=[];start=None
        for i,value in enumerate(np.r_[raw_missing,False]):
            if value and start is None:start=i
            if not value and start is not None:runs.append((start,i));start=None
        rows.append(dict(window_id=row.window_id,review_id=row.review_id,finite_signal_fraction=row.finite_signal_fraction,
            longest_bilateral_missing_seconds=max((b-a for a,b in runs),default=0)/8.7,
            selected_mode=c.selected_fusion_mode.iloc[0],raw_absent_frames=int(raw_missing.sum()),
            missing_but_numeric_fused_frames=int((raw_missing & np.isfinite(c.fused_norm)).sum()),
            reference_events_during_missing=sum(raw_missing[min(len(raw_missing)-1,int(round(ti*8.7)))] for ti in refs),
            tp=row.tp,fp=row.fp,fn=row.fn,rr_abs_error=row.rr_abs_error))
        if not isinstance(row.review_id,str) or not row.review_id:continue
        time=c.target_time_seconds.to_numpy(float)
        fig,axs=plt.subplots(4,1,figsize=(11,8),sharex=True,layout='constrained',
                             gridspec_kw={'height_ratios':[2,2,.7,1]})
        for side,color in [('left','#087f8c'),('right','#c75c24')]:
            axs[0].plot(time,t[side+'_temp'],label=side+' ROI',color=color,lw=1.2)
            axs[0].plot(time,c[side+'_temp_used'],color=color,ls=':',lw=.7,alpha=.7)
        axs[0].set(ylabel='RF proxy temperature',title=f'{row.review_id} | raw ROI samples (solid), used after repair (dotted)')
        axs[0].legend(loc='upper right',fontsize=8)
        axs[1].plot(time,c.fused_norm,color='#a1b0b6',label='fused')
        axs[1].plot(time,c.smoothed_norm,color='#174e75',label='smoothed')
        ix=np.flatnonzero(c.is_peak.to_numpy(bool))
        axs[1].scatter(time[ix],c.smoothed_norm.iloc[ix],s=22,color='black',label='frozen peaks')
        axs[1].set(ylabel='Normalized signal',title=f'Selected fusion: {c.selected_fusion_mode.iloc[0]}')
        axs[1].legend(fontsize=8,loc='upper right')
        axs[2].fill_between(time,0,(~raw_missing).astype(int),step='mid',color='#24865a')
        axs[2].set(ylim=(0,1),yticks=[0,1],ylabel='Any ROI\navailable')
        r2=refs
        r3=new[new.window_id.eq(row.window_id)&new.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
        pp=pred[pred.window_id.eq(row.window_id)].event_time_seconds.to_numpy(float)
        axs[3].eventplot([r2,r3,pp],lineoffsets=[2,1,0],linelengths=.6,colors=['#087f8c','#c75c24','#333333'])
        axs[3].set(yticks=[0,1,2],yticklabels=['Frozen','R3','R2'],xlabel='Original viewing time (s)',xlim=(0,30))
        for ax in axs[:3]:
            for start,stop in runs:
                ax.axvspan(start/8.7,min(stop/8.7,30),color='#c74343',alpha=.12)
        fig.savefig(ROOT/'figures'/f'{row.review_id}_signal_chain.png',dpi=180);plt.close(fig)
    table=pd.DataFrame(rows)
    write_csv(ROOT/'signal_stage_details.csv',table)
    write_csv(ROOT/'timestamp_gap_details.csv',gap_rows)
    complete=table[table.rr_abs_error.notna()].copy()
    complete['signal_coverage_band']=pd.cut(complete.finite_signal_fraction,[0,.25,.5,.75,1.000001],
        labels=['(0,.25]','(.25,.50]','(.50,.75]','(.75,1]'],include_lowest=True)
    group=complete.groupby('signal_coverage_band',observed=True).agg(windows=('window_id','size'),
        rr_mae=('rr_abs_error','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'))
    group['f1']=2*group.tp/(2*group.tp+group.fp+group.fn)
    write_csv(ROOT/'signal_coverage_strata.csv',group.reset_index())
    print(group.to_string())
    print(table[table.review_id.notna()][['review_id','selected_mode','longest_bilateral_missing_seconds','missing_but_numeric_fused_frames']].to_string(index=False))


if __name__=='__main__':main()
