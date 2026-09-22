"""Anatomical scoring after candidate sealing; never feed labels into predictions."""
import json
import pandas as pd
import numpy as np
from analyze_transfer import read,write_csv,write_json,sha256
from evaluate_nostril_reference import validate_submission,match_regions,outside_fraction
from local_contrast_roi_probe import ROOT,OUT


def main():
    seal=json.loads((OUT/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,h in seal['files'].items():assert sha256(OUT/name)==h
    ref_path=ROOT/'anatomy_reference_142648_v1/reference_original.json'
    data=json.loads(ref_path.read_text(encoding='utf-8'));pack=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    frames=validate_submission(data,pack)
    new=read(OUT/'prediction_points.csv')
    old=new.copy();old['variant']=old.seed_variant;old['x']=old.seed_x;old['y']=old.seed_y
    points=pd.concat([old,new],ignore_index=True)
    records=[];pairs=[];summary=[]
    variants=['baseline','baseline_local_contrast','frozen_roi','frozen_roi_local_contrast']
    for variant in variants:
        for fid,r in data['records'].items():
            if r['status']!='complete' or r['visibility']=='uncertain':continue
            p=points[points.frame_id.eq(fid)&points.variant.eq(variant)]
            els=[e for e in r['regions'].values() if e is not None]
            matched=match_regions(p[['x','y']].to_numpy(float),els)
            correct=sum(d<=1 for i,j,d in matched)
            records.append(dict(variant=variant,frame_id=fid,cluster=fid.split('_')[0],expected=len(els),
                predicted=len(p),correct=correct,false_points=len(p)-correct,missed=len(els)-correct))
            for i,j,d in matched:
                point=p.iloc[i]
                pairs.append(dict(variant=variant,frame_id=fid,keypoint=int(point.keypoint),human_region=j,
                    center_inside=d<=1,normalized_distance=d,outside_fraction=outside_fraction(
                        (point.x,point.y),point.radius,els[j],frames[fid]['width'],frames[fid]['height'])))
    results=pd.DataFrame(records);pr=pd.DataFrame(pairs)
    for v,g in results.groupby('variant'):
        d=pr[pr.variant.eq(v)]
        summary.append(dict(variant=v,frames=len(g),expected=int(g.expected.sum()),predicted=int(g.predicted.sum()),correct=int(g.correct.sum()),
            point_recall=float(g.correct.sum()/g.expected.sum()),point_precision=float(g.correct.sum()/g.predicted.sum()),
            median_normalized_distance=float(d.normalized_distance.median()),median_ROI_outside=float(d.outside_fraction.median())))
    changes=[]
    for v in ['baseline','frozen_roi']:
        a=results[results.variant.eq(v)].set_index('frame_id');b=results[results.variant.eq(v+'_local_contrast')].set_index('frame_id')
        a['delta_correct']=b.correct-a.correct
        clusters=a.groupby('cluster')[['delta_correct','expected']].sum().to_numpy(float)
        rng=np.random.default_rng(20260922);ds=[]
        for _ in range(2000):
            c=clusters[rng.integers(len(clusters),size=len(clusters))].sum(axis=0)
            if c[1]:ds.append(c[0]/c[1])
        changes.append(dict(seed_variant=v,delta_correct=int(a.delta_correct.sum()),
            better_frames=int((a.delta_correct>0).sum()),worse_frames=int((a.delta_correct<0).sum()),
            delta_recall_ci_low=float(np.quantile(ds,.025)),delta_recall_ci_high=float(np.quantile(ds,.975))))
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    write_csv(dest/'frame_metrics.csv',records);write_csv(dest/'matched_regions.csv',pairs)
    write_csv(dest/'summary.csv',summary);write_csv(dest/'paired_changes.csv',changes)
    passed=all(c['delta_correct']>0 and c['delta_recall_ci_low']>0 for c in changes)
    write_json(dest/'adoption_decision.json',dict(disposition='NEEDS_TEMPORAL_AND_FULL_COHORT_VALIDATION' if passed else 'DO_NOT_ADOPT',
        reference_sha256=sha256(ref_path),default_changed=False,full_RR_run=False,
        limitation='36 selected frames from12 existing review windows; not new external test or physiological proof'))
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))


if __name__=='__main__':main()
