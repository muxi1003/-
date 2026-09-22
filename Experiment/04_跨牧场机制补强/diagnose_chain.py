"""Stage-wise audit of archived inputs, never rewriting frozen predictions."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_transfer import HERE, HOLDOUT, BASES, SUBMISSION, read, sha256, write_csv, write_json

OUT=HERE/'20260921_chain_v3'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'


def load_bgr(path):
    frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(path)
    return frame


def main():
    OUT.mkdir(exist_ok=False)
    pred=read(TRANSFER/'prediction_windows.csv').set_index('window_id')
    events=read(BASES['jiufu271']/'event_metrics_by_window.csv').set_index('window_id')
    counts=read(BASES['jiufu271']/'paired_count_results.csv').set_index('window_id')
    selected=read(HERE/'20260921_v1/review24/selection_key_DO_NOT_VIEW_DURING_REVIEW.csv')
    selected=selected[selected.cohort.eq('jiufu271')].set_index('window_id')
    members=read(TRANSFER/'test_windows.csv').set_index('window_id')
    hashes=[]; rows=[]
    for wid,p in pred.iterrows():
        root=TRANSFER/'windows'/wid
        base=read(root/'base_temperatures.csv') if (root/'base_temperatures.csv').exists() else None
        temp=read(root/'temperatures.csv') if (root/'temperatures.csv').exists() else None
        r=dict(window_id=wid,prediction_status=p.prediction_status,reason=p.reason,
               frame_shape=p.frame_shape if 'frame_shape' in p else '',review_id=selected.loc[wid,'review_id'] if wid in selected.index else '')
        if base is not None:
            hashes += [dict(path=str(root/n),sha256=sha256(root/n)) for n in ['base_temperatures.csv','temperatures.csv','map.csv','radius_table.csv']]
            finite=np.isfinite(temp[['left_temp','right_temp']].to_numpy(float))
            high=(base[['left_conf','right_conf']].to_numpy(float)>=.5).any(axis=1)
            r.update(frames=len(base),detected_frame_fraction=float(base.status.ne('no_detection').mean()),
                finite_signal_fraction=float(finite.any(axis=1).mean()),both_finite_fraction=float(finite.all(axis=1).mean()),
                high_keypoint_fraction=float(high.mean()),
                left_source_direct_fraction=float(temp.left_source.eq('detected').mean()),
                right_source_direct_fraction=float(temp.right_source.eq('detected').mean()),
                left_span=float(temp.left_temp.max()-temp.left_temp.min()),
                right_span=float(temp.right_temp.max()-temp.right_temp.min()),
                median_spacing=float(base.nostril_spacing.median()),
                radius_median=float(temp.adaptive_roi_radius.median()))
            if base.status.eq('no_detection').all():stage='no_pose_detection_all_frames'
            elif not finite.any():stage='detections_but_no_valid_temperature'
            else:stage='signal_available'
            r['first_blocking_stage']=stage
        else:
            r['first_blocking_stage']='timestamp_sampling'
        if wid in events.index:
            r.update({k:events.loc[wid,k] for k in ['tp','fp','fn','truth_count']})
        if wid in counts.index:
            r['rr_abs_error']=abs(counts.loc[wid,'predicted_rr_bpm']-counts.loc[wid,'truth_rr_bpm'])
        rows.append(r)
    table=pd.DataFrame(rows)
    write_csv(OUT/'all271_stage_audit.csv',table)
    write_csv(OUT/'input_hashes.csv',hashes)
    groups=table.groupby(['first_blocking_stage','prediction_status'],dropna=False).agg(
        windows=('window_id','size'),complete_references=('truth_count','count'),
        tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),rr_mae=('rr_abs_error','mean'))
    write_csv(OUT/'stage_totals.csv',groups.reset_index())
    review=table[table.window_id.isin(selected.index)].copy()
    write_csv(OUT/'review12_stage_audit.csv',review)
    figdir=OUT/'figures';figdir.mkdir()
    # Source pixels are not modified; markers are drawn in the same native coordinate axes.
    for wid in selected.index:
        folder=TRANSFER/'windows'/wid
        t=read(folder/'temperatures.csv') if (folder/'temperatures.csv').exists() else None
        mapping=read(folder/'map.csv') if t is not None else None
        raw_frames={}
        if t is None:
            times=read(folder/'timestamps.csv').relative_seconds.to_numpy(float)
            wanted={int(np.argmin(abs(times-i/8.7))) for i in [0,43,87,130,174,217]}
            cap=cv2.VideoCapture(str(members.loc[wid,'source_path']))
            for j in range(max(wanted)+1):
                ok,frame=cap.read()
                if not ok:raise ValueError('Decode failed')
                if j in wanted:raw_frames[j]=frame
            cap.release()
        fig,axs=plt.subplots(2,3,figsize=(10,9),layout='constrained')
        for ax,idx in zip(axs.flat,[0,43,87,130,174,217]):
            if t is None:
                j=int(np.argmin(abs(times-idx/8.7)));frame=raw_frames[j]
                ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                ax.set_title(f'{times[j]:.2f}s | timestamp-rejected\nYOLO/ROI not executed',fontsize=8)
                ax.axis('off');continue
            row=t.iloc[idx];frame=load_bgr(folder/'frames'/row.frame_name)
            ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            for side,color in [('left','#00ffff'),('right','#00ff00')]:
                x,y=row[side+'_x'],row[side+'_y']
                if np.isfinite([x,y]).all():
                    ax.add_patch(plt.Circle((x,y),row.adaptive_roi_radius,fill=False,color=color,lw=1.3))
                    ax.plot(x,y,'+',color=color,ms=6)
            ax.set_title(f"{mapping.iloc[idx].target_time_seconds:.2f}s | {row.status}\nL={row.left_conf:.2f} R={row.right_conf:.2f}",fontsize=8)
            ax.axis('off')
        fig.suptitle(f"{selected.loc[wid,'review_id']} / {wid}\nArchived native frames and recorded ROI; cyan=L, green=R",fontsize=11)
        fig.savefig(figdir/f"{selected.loc[wid,'review_id']}_frames.png",dpi=150);plt.close(fig)
    write_json(OUT/'protocol.json',dict(role='retrospective_failure_diagnosis',
        original_predictions_unchanged=True,annotation_version='R2 full and R3 review kept separate',
        initial_hypotheses=['no_pose_or_bad_coordinates','ROI_temperature_loss','interpolation_waveform_artifact','fusion_peak_rules'],
        visual_sampling='six fixed times per Jiufu review window, not outcome optimized',
        new_model_selection_requires_same_input_comparison=True))
    print(groups.to_string())
    print(review[['review_id','first_blocking_stage','finite_signal_fraction','high_keypoint_fraction','tp','fp','fn','rr_abs_error']].to_string(index=False))


if __name__=='__main__':main()
