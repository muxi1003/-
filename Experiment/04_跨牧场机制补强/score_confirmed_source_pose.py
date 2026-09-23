"""Score sealed pose predictions without checkpoint/threshold selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from analyze_transfer import read, write_csv, write_json, sha256
from train_confirmed_source_pose import OUT
from build_confirmed_pose_dataset import DEST
from probe_pose_mechanism import ROOT
from evaluate_nostril_reference import validate_submission, match_regions, outside_fraction


def high_points(p):
    if not p['boxes']:return np.zeros((0,2))
    best=int(np.argmax(p['box_conf']))
    k=np.asarray(p['keypoints'][best],float)
    mask=(k[:,2]>=.5)&(k[:,0]>=0)&(k[:,0]<p['width'])&(k[:,1]>=0)&(k[:,1]<p['height'])
    return k[mask,:2]


def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return inter/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter+1e-12)


def main():
    reference=ROOT/'anatomy_reference_142648_v1/reference_original.json'
    data=json.loads(reference.read_text(encoding='utf-8-sig'))
    package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    frames=validate_submission(data,package)
    records={k:v for k,v in data['records'].items() if v['status']=='complete' and v['visibility']!='uncertain'}
    seal=json.loads((OUT/'prediction_seal.json').read_text())
    for name,digest in seal['files'].items():assert sha256(OUT/name)==digest
    summary=[];detail=[];pairs=[];source_rows=[]
    source_manifest=read(DEST/'manifest.csv').set_index('frame_id')
    for variant in ['baseline','candidate']:
        pred=json.loads((OUT/f'{variant}_predictions.json').read_text(encoding='utf-8'))
        for p in pred:
            if p['cohort']=='source_validation':
                lab=np.array([list(map(float,l.split())) for l in
                    Path(source_manifest.loc[p['frame_id'],'label_path']).read_text().splitlines()]).reshape(-1,11)
                gt=[]
                for row in lab:
                    cx,cy,w,h=row[1:5]*[p['width'],p['height'],p['width'],p['height']]
                    gt.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
                npoints=int((lab[:,[7,10]]==2).sum())
                ncorr=0;nboxes=0;distances=[]
                if len(gt) and p['boxes']:
                    cost=np.array([[1-iou(a,b) for b in p['boxes']] for a in gt])
                    aa,bb=linear_sum_assignment(cost)
                    for a,b in zip(aa,bb):
                        if cost[a,b]>.5:continue
                        nboxes+=1
                        diag=np.hypot(gt[a][2]-gt[a][0],gt[a][3]-gt[a][1])
                        for k,start in enumerate([5,8]):
                            if lab[a,start+2]!=2:continue
                            q=p['keypoints'][b][k]
                            if q[2]<.5:continue
                            d=np.linalg.norm(np.array(q[:2])-lab[a,start:start+2]*[p['width'],p['height']])/diag
                            distances.append(d);ncorr+=int(d<=.1)
                source_rows.append(dict(variant=variant,frame_id=p['frame_id'],expected_boxes=len(gt),matched_boxes_iou50=nboxes,
                    expected_points=npoints,correct_points_pck10=ncorr,
                    confident_keypoints=sum(int(k[2]>=.5) for inst in p['keypoints'] for k in inst),
                    mean_matched_normalized_error=float(np.mean(distances)) if distances else None))
                continue
            if p['frame_id'] not in records:continue
            fid=p['frame_id'];r=records[fid];xy=high_points(p)
            regions=[v for v in r['regions'].values() if v is not None]
            matches=match_regions(xy,regions);correct=sum(d<=1 for _,_,d in matches)
            detail.append(dict(variant=variant,frame_id=fid,visible=bool(regions),expected=len(regions),
                               predicted=len(xy),correct=correct,box_absent=not bool(p['boxes']),
                               no_high_point=not len(xy),no_correct=correct==0))
            for i,j,d in matches:
                pairs.append(dict(variant=variant,frame_id=fid,human_region=j,normalized_distance=d,
                    center_inside=d<=1,outside_fraction_r20=outside_fraction(xy[i],20,regions[j],p['width'],p['height'])))
    df=pd.DataFrame(detail)
    for variant,g in df.groupby('variant'):
        n=int(g.expected.sum());pred=int(g.predicted.sum());tp=int(g.correct.sum())
        visible=g[g.visible]
        pp=pd.DataFrame(pairs);pp=pp[pp.variant.eq(variant)]
        summary.append(dict(variant=variant,frames=len(g),visible_frames=len(visible),human_regions=n,
            predicted_points=pred,correct_points=tp,point_recall=tp/n,point_precision=tp/pred if pred else 0,
            point_F1=2*tp/(n+pred),visible_no_box=int(visible.box_absent.sum()),
            visible_no_high_point=int(visible.no_high_point.sum()),visible_no_correct=int(visible.no_correct.sum()),
            matched_ROI_pairs=len(pp),median_ROI_outside_all_assigned=float(pp.outside_fraction_r20.median()),
            median_ROI_outside_correct_centers=float(pp[pp.center_inside].outside_fraction_r20.median())))
    write_csv(OUT/'anatomy_frame_results.csv',detail)
    write_csv(OUT/'anatomy_pair_results.csv',pairs)
    write_csv(OUT/'anatomy_summary.csv',summary)
    write_csv(OUT/'source_validation_frames.csv',source_rows)
    ss=[]
    for v,g in pd.DataFrame(source_rows).groupby('variant'):
        ss.append(dict(variant=v,frames=len(g),expected_boxes=int(g.expected_boxes.sum()),
            matched_boxes_iou50=int(g.matched_boxes_iou50.sum()),box_recall_iou50=g.matched_boxes_iou50.sum()/g.expected_boxes.sum(),
            expected_points=int(g.expected_points.sum()),correct_points_pck10=int(g.correct_points_pck10.sum()),
            pck10_all_annotated=g.correct_points_pck10.sum()/g.expected_points.sum()))
    write_csv(OUT/'source_validation_summary.csv',ss)
    baseline=next(s for s in summary if s['variant']=='baseline')
    assert baseline['correct_points']==18 and baseline['predicted_points']==27
    matched=pd.DataFrame(pairs).pivot(index=['frame_id','human_region'],columns='variant',values='outside_fraction_r20').dropna()
    write_csv(OUT/'paired_ROI_comparison.csv',matched.reset_index())
    write_json(OUT/'paired_ROI_summary.json',dict(paired_regions=len(matched),
        baseline_median=float(matched.baseline.median()) if len(matched) else None,
        candidate_median=float(matched.candidate.median()) if len(matched) else None,
        median_paired_change=float((matched.candidate-matched.baseline).median()) if len(matched) else None,
        warning='Conditional on both models having an assigned point; cannot replace full-region recall'))
    write_json(OUT/'scoring_protocol.json',dict(reference_sha256=sha256(reference),
        completed_frames=len(records),match='unordered one-to-one inside human ellipse',radius=20,
        source_PCK='correct labeled side, confidence >=.5, box IoU >=.5, distance <= .1 reference box diagonal; denominator all annotated points',
        ROI_limit='Approximate ellipse containment, not pixelwise background truth; assigned pairs differ across models',
        not_RR_evaluation=True,not_new_external_validation=True,default_changed=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(ss).to_string(index=False))


if __name__=='__main__':main()
