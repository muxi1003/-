"""Per-emitted-peak evidence, distinct from physiological ground-truth claims."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_transfer import HERE, BASES, HOLDOUT, read, write_csv, write_json, sha256, match_events
from evaluate_nostril_reference import match_regions

ROOT=HERE/'20260922_pose_phase_v1'
OUT=ROOT/'frozen_peak_evidence_v1'
TRANSFER=HOLDOUT/'external_transfer/20260914_v2/windows'


def contributing_sides(row,mode):
    values={s:float(row[s+'_norm']) for s in ['left','right'] if np.isfinite(row[s+'_norm'])}
    if mode in ['left','right']: return [mode] if mode in values else []
    if mode=='mean': return list(values)
    if mode in ['min','max']:
        if not values:return []
        optimum=(min if mode=='min' else max)(values.values())
        return [s for s,v in values.items() if abs(v-optimum)<1e-12]
    raise ValueError('Unrecognized saved fusion mode: '+mode)


def temperature_origin(raw,used,source):
    if not np.isfinite(raw):return 'temperature_interpolated'
    if not np.isfinite(used) or abs(float(raw)-float(used))>1e-8:return 'modified_temperature'
    return 'raw_direct' if source=='detected' else 'raw_inferred'


def spatial_support(temperature,record):
    regions=[e for e in record['regions'].values() if e is not None]
    sides=[s for s in ['left','right'] if np.isfinite(temperature[s+'_temp'])
           and np.isfinite(temperature[s+'_x']) and np.isfinite(temperature[s+'_y'])]
    points=[[temperature[s+'_x'],temperature[s+'_y']] for s in sides]
    matched={s:False for s in sides}
    for i,j,d in match_regions(points,regions):matched[sides[i]]=d<=1
    return matched


def main():
    OUT.mkdir(exist_ok=False)
    audit=read(HERE/'20260921_r3_v1/annotation_audit.csv')
    audit=audit[audit.cohort.eq('jiufu271')&audit.eligible_full_window]
    refs=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    predictions=read(BASES['jiufu271']/'prediction_events.csv')
    raw=json.loads((ROOT/'anatomy_reference_142648_v1/reference_original.json').read_text(encoding='utf-8'))
    frames=read(ROOT/'frame_manifest.csv')
    reviewed={str(Path(r.source).resolve()).lower():r.frame_id for r in frames.itertuples()
              if r.frame_id in raw['records'] and raw['records'][r.frame_id]['status']=='complete'}
    sources=[HERE/'20260921_r3_v1/annotation_audit.csv',HERE/'20260921_r3_v1/raw/reference_events.csv',
             BASES['jiufu271']/'prediction_events.csv',ROOT/'anatomy_reference_142648_v1/reference_original.json',
             ROOT/'frame_manifest.csv']
    write_json(OUT/'protocol.json',dict(role='posthoc_evidence_audit_not_new_detector',
        reference='R3 paused visible-expiration review; strict9 separately from eligible11',
        tolerance_seconds=.3,matching='one-to-one maximum cardinality then minimum time error',
        reference_time_shift_seconds=0,MAF_window=3,anatomical_match='exact archived image path only; no propagation across frames',
        max_min_ties='all equal-valued participating sides retained conservatively',
        physiological_truth_claim=False,no_prediction_or_reference_changes=True,
        input_hashes={str(p):sha256(p) for p in sources}))
    peaks=[];support=[];references=[];input_hashes=[]
    for a in audit.itertuples():
        wid=a.window_id; folder=TRANSFER/wid
        r=refs[refs.window_id.eq(wid)&refs.confidence.eq('confirmed')].reset_index(drop=True)
        p=predictions[predictions.window_id.eq(wid)].reset_index(drop=True)
        matches,fp,fn=match_events(p.event_time_seconds.to_numpy(float),r.event_time_seconds.to_numpy(float),.3)
        match_by_pred={i:(j,delta) for i,j,delta in matches}
        matched_ref={j:i for i,j,_ in matches}
        for j,event in r.iterrows():
            references.append(dict(window_id=wid,review_id=a.review_id,event_id=event.event_id,
                reference_seconds=float(event.event_time_seconds),phase_basis=event.phase_basis,
                strict_visible_phase=bool(a.strict_visible_phase),
                prediction_event_id=p.iloc[matched_ref[j]].event_id if j in matched_ref else '',
                matching_status='TP' if j in matched_ref else 'FN'))
        if p.empty:continue
        curve=read(folder/'curve.csv');temperature=read(folder/'temperatures.csv')
        assert len(curve)==len(temperature) and set(curve.selected_fusion_mode)=={curve.selected_fusion_mode.iloc[0]}
        assert set(curve.smoothing_method)=={'moving_average'}
        smooth=np.convolve(curve.fused_norm.to_numpy(float),np.ones(3)/3,mode='valid')
        np.testing.assert_allclose(smooth,curve.smoothed_norm.to_numpy(float)[1:-1],rtol=1e-8,atol=1e-8,equal_nan=True)
        np.testing.assert_allclose(curve.loc[curve.is_peak,'target_time_seconds'],p.event_time_seconds,atol=1e-8)
        for name in ['curve.csv','temperatures.csv']:
            input_hashes.append(dict(path=str(folder/name),sha256=sha256(folder/name)))
        mode=curve.selected_fusion_mode.iloc[0]
        for i,event in p.iterrows():
            index=int(round(event.event_time_seconds*8.7))
            assert curve.iloc[index].is_peak and 0<index<len(curve)-1
            contributors=[]
            for q in range(index-1,index+2):
                row=curve.iloc[q];t=temperature.iloc[q]
                sides=contributing_sides(row,mode)
                path=str((folder/'frames'/t.frame_name).resolve()).lower()
                fid=reviewed.get(path)
                anatomical=spatial_support(t,raw['records'][fid]) if fid else {}
                for side in sides or ['fused']:
                    origin=(temperature_origin(t[side+'_temp'],row[side+'_temp_used'],t[side+'_source'])
                            if side!='fused' else 'fused_interpolated')
                    contributors.append(dict(window_id=wid,prediction_event_id=event.event_id,
                        contributor_frame=q,center_frame=index,side=side,origin=origin,
                        reference_frame_id=fid or '',spatially_reviewed=bool(fid),
                        center_inside_human_region=bool(anatomical.get(side,False)) if fid else None))
            support.extend(contributors)
            if i in match_by_pred:
                j,delta=match_by_pred[i];reference=r.iloc[j]
                refid=reference.event_id;rt=float(reference.event_time_seconds);phase=reference.phase_basis
            else:refid='';rt=None;phase='';delta=None
            counts={o:sum(c['origin']==o for c in contributors) for o in
                    ['raw_direct','raw_inferred','temperature_interpolated','modified_temperature','fused_interpolated']}
            peaks.append(dict(window_id=wid,review_id=a.review_id,prediction_event_id=event.event_id,
                peak_seconds=float(event.event_time_seconds),selected_mode=mode,
                temporal_match='TP' if i in match_by_pred else 'FP',reference_event_id=refid,
                reference_seconds=rt,phase_basis=phase,time_error_seconds=delta,
                strict_visible_phase=bool(a.strict_visible_phase),contributor_count=len(contributors),
                all_contributors_direct=all(c['origin']=='raw_direct' for c in contributors),
                any_temperature_interpolated=any('interpolated' in c['origin'] for c in contributors),
                any_same_image_anatomical_review=any(c['spatially_reviewed'] for c in contributors),
                all_contributors_have_correct_anatomical_reference=all(c['spatially_reviewed'] and c['center_inside_human_region'] for c in contributors),
                **{k+'_contributors':v for k,v in counts.items()}))
    d=pd.DataFrame(peaks);r=pd.DataFrame(references)
    assert len(d)==83 and len(r)==143 and d.temporal_match.eq('TP').sum()==41
    assert r.matching_status.eq('FN').sum()==102
    summary=[]
    for scope in ['eligible11','strict9']:
        x=d if scope=='eligible11' else d[d.strict_visible_phase]
        y=r if scope=='eligible11' else r[r.strict_visible_phase]
        for temporal,g in x.groupby('temporal_match'):
            summary.append(dict(scope=scope,temporal_match=temporal,peaks=len(g),
                all_contributors_direct=int(g.all_contributors_direct.sum()),
                at_least_one_inferred_ROI=int(g.raw_inferred_contributors.gt(0).sum()),
                any_temperature_interpolated=int(g.any_temperature_interpolated.sum()),
                any_same_image_anatomical_review=int(g.any_same_image_anatomical_review.sum()),
                all_contributors_have_correct_anatomical_reference=int(g.all_contributors_have_correct_anatomical_reference.sum())))
    write_csv(OUT/'emitted_peak_evidence.csv',peaks)
    write_csv(OUT/'smoothing_contributors.csv',support)
    write_csv(OUT/'reference_matching.csv',references)
    write_csv(OUT/'summary.csv',summary);write_csv(OUT/'input_hashes.csv',input_hashes)
    write_json(OUT/'verification.json',dict(status='PASS',eligible_windows=11,strict_windows=9,
        actual_predictions_replayed=83,reference_events=143,tp=41,fp=42,fn=102,
        MAF3_saved_values_verified=True,no_default_or_reference_changes=True,
        exact_frame_anatomy_required_no_nearest_time_label_substitution=True))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
