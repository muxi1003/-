"""Anatomical reference audit and human-ROI diagnostic control, not RR prediction."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from analyze_transfer import HERE,HOLDOUT,read,write_csv,write_json,sha256
from evaluate_nostril_reference import validate_submission,match_regions,outside_fraction
from probe_pose_mechanism import load_bgr
from recover_timestamp_segments import CachedRF

ROOT=HERE/'20260922_pose_phase_v1'
OUT=ROOT/'anatomy_reference_142648_v1'
FROZEN=HOLDOUT/'method_snapshots/20260909_v1'


def ellipse_pixels(im,e):
    h,w=im.shape[:2]
    xx,yy=np.meshgrid(np.arange(max(0,int(np.floor(e['cx']-e['rx']))),min(w,int(np.ceil(e['cx']+e['rx']))+1)),
        np.arange(max(0,int(np.floor(e['cy']-e['ry']))),min(h,int(np.ceil(e['cy']+e['ry']))+1)))
    mask=((xx-e['cx'])/e['rx'])**2+((yy-e['cy'])/e['ry'])**2<=1
    return im[yy[mask],xx[mask]]


def proxy_mean(pixels,rf):
    if not len(pixels):return np.nan
    pred=rf.predict(pixels);pred=pred[pred>20]
    return float(pred.mean()) if len(pred) else np.nan


def main():
    source=OUT/'reference_original.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    frame_map=validate_submission(data,package)
    manifest=read(ROOT/'frame_manifest.csv').set_index('frame_id')
    archive_meta=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    assert sha256(source)==archive_meta['reference_sha256']
    summary=read(OUT/'summary.csv');detail=read(OUT/'frame_results.csv')
    pp=read(ROOT/'pose_keypoints.csv');pp=pp[pp.confidence.ge(.5)&pp.inside_native].copy();pp['radius']=20
    frozen=read(ROOT/'frozen_roi_keypoints.csv')
    pts=pd.concat([pp,frozen],ignore_index=True)
    probe=read(ROOT/'pose_probes.csv');baseline=probe[probe.variant.eq('baseline')].set_index('frame_id')
    stages=read(HERE/'20260921_chain_v3/review12_stage_audit.csv').set_index('window_id')
    annotations=[];regions=[]
    for fid,r in data['records'].items():
        if r['status']!='complete':continue
        fm=manifest.loc[fid];b=baseline.loc[fid]
        visible=r['visibility'] in ['one_visible','two_visible']
        if not visible:stage='human_not_visible' if r['visibility']=='not_visible' else 'uncertain_reference'
        elif b.detections==0:stage='visible_but_no_box'
        elif b.high_keypoints==0:stage='box_but_no_high_keypoint'
        else:stage='high_keypoint_available'
        annotations.append(dict(frame_id=fid,review_id=fm.review_id,window_id=fm.window_id,
            visibility=r['visibility'],notes=r.get('notes',''),expected_nostrils=sum(e is not None for e in r['regions'].values()),
            stage=stage,baseline_boxes=int(b.detections),baseline_high_points=int(b.high_keypoints),
            upstream_timestamp_rejected=stages.loc[fm.window_id,'first_blocking_stage']=='timestamp_sampling'))
        for key,e in r['regions'].items():
            if e is None:continue
            regions.append(dict(frame_id=fid,region=key,**e,
                minor_diameter_native=2*min(e['rx'],e['ry']),
                minor_diameter_at_640=2*min(e['rx'],e['ry'])*640/max(fm.width,fm.height)))
    ad=pd.DataFrame(annotations);rd=pd.DataFrame(regions)
    write_csv(OUT/'annotation_audit.csv',ad);write_csv(OUT/'human_region_geometry.csv',rd)
    stage_summary=ad.groupby('stage').agg(frames=('frame_id','size'),nostrils=('expected_nostrils','sum')).reset_index()
    write_csv(OUT/'anatomical_stage_summary.csv',stage_summary)
    # Compare actual frozen output only on frames where timestamp gate allowed execution.
    executed=set(ad[~ad.upstream_timestamp_rejected].frame_id)
    conditional=[]
    for variant in ['baseline','frozen_roi']:
        g=detail[detail.variant.eq(variant)&detail.frame_id.isin(executed)]
        conditional.append(dict(variant=variant,frames=len(g),visible_frames=int(g.expected_nostrils.gt(0).sum()),
            expected_nostrils=int(g.expected_nostrils.sum()),predicted_points=int(g.predicted_points.sum()),
            correct_points=int(g.correct_points.sum()),missed_nostrils=int(g.missed_nostrils.sum()),
            off_target_or_extra_points=int(g.extra_or_off_target_points.sum())))
    write_csv(OUT/'timestamp_executed_same_frames.csv',conditional)
    b=detail[detail.variant.eq('baseline')].set_index('frame_id')
    lower=detail[detail.variant.eq('lower75')].set_index('frame_id')
    changed=[];fallback=b.copy()
    for fid in b.index[b.predicted_points.eq(0)]:
        fallback.loc[fid]=lower.loc[fid]
        if lower.loc[fid,'predicted_points']:
            changed.append(dict(frame_id=fid,original_points=0,new_points=int(lower.loc[fid,'predicted_points']),
                new_correct_points=int(lower.loc[fid,'correct_points'])))
    write_json(OUT/'crop_fallback_decision.json',dict(
        rule='only when baseline has no >=.5 point, use lower75 crop points',
        role='post_reference_diagnostic_not_adopted',correct_before=int(b.correct_points.sum()),
        correct_after=int(fallback.correct_points.sum()),points_before=int(b.predicted_points.sum()),
        points_after=int(fallback.predicted_points.sum()),changed_frames=changed,adopted=False))
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    rf=CachedRF(joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl'))
    controls=[];geometry=[]
    for fid,r in data['records'].items():
        if r['status']!='complete' or r['visibility'] not in ['one_visible','two_visible']:continue
        fm=manifest.loc[fid];im=load_bgr(ROOT/'frames'/f'{fid}.png')
        assert sha256(ROOT/'frames'/f'{fid}.png')==fm.image_sha256
        els=[e for e in r['regions'].values() if e is not None]
        for variant in ['baseline','frozen_roi']:
            sel=pts[pts.frame_id.eq(fid)&pts.variant.eq(variant)]
            for i,j,dist in match_regions(sel[['x','y']].to_numpy(float),els):
                p=sel.iloc[i];e=els[j];radius=int(p.radius)
                actual=rr.circle_temperature(im,rf,p.x,p.y,radius,20.)
                centered=rr.circle_temperature(im,rf,e['cx'],e['cy'],radius,20.)
                anatomical=proxy_mean(ellipse_pixels(im,e),rf)
                controls.append(dict(frame_id=fid,variant=variant,keypoint=int(p.keypoint),
                    normalized_center_distance=dist,center_inside=dist<=1,radius=radius,
                    current_RF_temperature=actual,human_center_same_radius_RF_temperature=centered,
                    human_ellipse_RF_temperature=anatomical,
                    current_minus_human_ellipse=actual-anatomical,
                    recentered_minus_human_ellipse=centered-anatomical,
                    role='human_reference_assisted_single_frame_control_not_RR_or_absolute_temperature_truth'))
                for scale in [.8,1.,1.2]:
                    enlarged={**e,'rx':e['rx']*scale,'ry':e['ry']*scale}
                    geometry.append(dict(frame_id=fid,variant=variant,keypoint=int(p.keypoint),ellipse_scale=scale,
                        center_inside=dist<=1,
                        outside_current=outside_fraction((p.x,p.y),radius,enlarged,int(fm.width),int(fm.height)),
                        outside_after_human_recentering=outside_fraction((e['cx'],e['cy']),radius,enlarged,int(fm.width),int(fm.height))))
    write_csv(OUT/'human_ROI_temperature_control.csv',controls)
    write_csv(OUT/'ROI_geometry_sensitivity.csv',geometry)
    plots=OUT/'figures';plots.mkdir(exist_ok=False)
    for fid in ['R3-01_t00','R3-02_t10','R3-09_t00','R3-11_t20','R3-14_t10']:
        r=data['records'][fid];im=load_bgr(ROOT/'frames'/f'{fid}.png')[:,:,::-1]
        fig,axes=plt.subplots(1,2,figsize=(10,7),layout='constrained')
        for ax,variant in zip(axes,['baseline','frozen_roi']):
            ax.imshow(im)
            for name,e in r['regions'].items():
                if e is not None:
                    ax.add_patch(Ellipse((e['cx'],e['cy']),2*e['rx'],2*e['ry'],fill=False,color='#21ef36',lw=2))
                    ax.text(e['cx'],e['cy']-e['ry']-8,'Human '+name,color='#21ef36',fontsize=9,
                        bbox=dict(facecolor='black',alpha=.6,edgecolor='none'))
            selected=pts[pts.frame_id.eq(fid)&pts.variant.eq(variant)]
            for p in selected.itertuples():
                ax.add_patch(Circle((p.x,p.y),p.radius,fill=False,color='#00dcff',lw=1.7))
                ax.plot(p.x,p.y,'+',color='#00dcff',ms=8)
            ax.set_title(f'{variant}: {len(selected)} ROI points\nReference: {r["visibility"]}',fontsize=10);ax.axis('off')
        fig.suptitle(f'{fid}: submitted human regions (green), original predictions (cyan)',fontsize=11)
        fig.savefig(plots/f'{fid}_anatomical_overlay.png',dpi=160);plt.close(fig)
    geo=pd.DataFrame(geometry);ctrl=pd.DataFrame(controls)
    stats=dict(reference_sha256=sha256(source),completed_frames=len(ad),visible_frames=int(ad.expected_nostrils.gt(0).sum()),
        expected_nostrils=int(ad.expected_nostrils.sum()),original_reference_unchanged=True,
        human_minor_diameter_640_median=float(rd.minor_diameter_at_640.median()),
        original_frozen_models_unchanged=True,full_RR_rerun=False)
    for variant in ['baseline','frozen_roi']:
        g=geo[geo.variant.eq(variant)&geo.ellipse_scale.eq(1)]
        c=ctrl[ctrl.variant.eq(variant)]
        stats[variant]=dict(geometric_matched_pairs=len(g),outside_current_median=float(g.outside_current.median()),
            outside_centered_median=float(g.outside_after_human_recentering.median()),
            outside_current_mean=float(g.outside_current.mean()),
            single_frame_RF_difference_abs_median=float(c.current_minus_human_ellipse.abs().median()))
    write_json(OUT/'anatomy_findings.json',stats)
    print(stage_summary.to_string(index=False));print(pd.DataFrame(conditional).to_string(index=False))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
