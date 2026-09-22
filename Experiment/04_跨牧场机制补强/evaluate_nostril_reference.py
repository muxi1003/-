"""Score actual imported anatomical references; never infer missing human labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from analyze_transfer import HERE, read, write_csv, write_json, sha256

ROOT=HERE/'20260922_pose_phase_v1'


def validate_submission(data,package):
    if data.get('synthetic_test_only'):
        raise ValueError('Synthetic QA data is not a research reference')
    if data.get('schema_version')!=1 or data.get('package_id')!=package['package_id']:
        raise ValueError('Annotation schema/package mismatch')
    if data.get('prediction_overlay_shown') is not False:
        raise ValueError('Annotation blinding declaration missing or not false; use separate analysis')
    frames={f['frame_id']:f for f in package['frames']}
    records=data.get('records')
    if not isinstance(records,dict):raise ValueError('Invalid records')
    for fid,r in records.items():
        if fid not in frames:raise ValueError('Unknown frame')
        if r.get('status') not in ['pending','complete']:raise ValueError('Invalid status')
        if r['status']!='complete':continue
        if r.get('visibility') not in ['two_visible','one_visible','not_visible','uncertain']:
            raise ValueError('Completed frame has invalid visibility')
        regions=r.get('regions',{})
        if set(regions)!=set(['A','B']):raise ValueError('Expected A/B regions')
        valid=[s for s in regions.values() if s is not None]
        for s in valid:
            if not isinstance(s,dict) or not all(isinstance(s.get(k),(float,int)) and not isinstance(s[k],bool) and np.isfinite(s[k]) for k in ['cx','cy','rx','ry']):
                raise ValueError('Invalid coordinates')
            w,h=frames[fid]['width'],frames[fid]['height']
            if min(s['rx'],s['ry'])<2 or s['cx']-s['rx']<-.01 or s['cy']-s['ry']<-.01 or s['cx']+s['rx']>w+.01 or s['cy']+s['ry']>h+.01:
                raise ValueError('Ellipse outside frame or too small')
        expected={'two_visible':2,'one_visible':1,'not_visible':0}.get(r['visibility'])
        if expected is not None and len(valid)!=expected:raise ValueError('Region count contradicts visibility')
        if not str(r.get('annotator','')).strip():raise ValueError('Missing annotator')
        if r['visibility'] in ['uncertain','not_visible'] and not str(r.get('notes','')).strip():raise ValueError('Missing reason')
    return frames


def normalized_distance(point,ellipse):
    return float(np.hypot((point[0]-ellipse['cx'])/ellipse['rx'],(point[1]-ellipse['cy'])/ellipse['ry']))


def match_regions(points,regions):
    if not len(points) or not len(regions):return []
    costs=np.array([[normalized_distance(p,r) for r in regions] for p in points])
    # Maximize number inside an ellipse before minimizing remaining distance.
    bounded=np.minimum(costs,100.)
    penalty=101*(min(costs.shape)+1)
    a,b=linear_sum_assignment(bounded+(costs>1)*penalty)
    return [(int(i),int(j),float(costs[i,j])) for i,j in zip(a,b)]


def outside_fraction(point,radius,ellipse,width,height):
    x,y=point;radius=int(radius)
    xx,yy=np.meshgrid(np.arange(max(0,int(round(x))-radius),min(width,int(round(x))+radius+1)),
        np.arange(max(0,int(round(y))-radius),min(height,int(round(y))+radius+1)))
    disk=(xx-round(x))**2+(yy-round(y))**2<=radius**2
    region=((xx-ellipse['cx'])/ellipse['rx'])**2+((yy-ellipse['cy'])/ellipse['ry'])**2<=1
    return float((disk&~region).sum()/disk.sum()) if disk.any() else np.nan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    data=json.loads(args.input.read_text(encoding='utf-8-sig'))
    frames=validate_submission(data,package)
    records=data['records'];completed={k:v for k,v in records.items() if v['status']=='complete'}
    if not completed:raise ValueError('No actual completed annotations; blank references cannot be scored')
    probes=read(ROOT/'pose_keypoints.csv');probes=probes[probes.confidence.ge(.5)&probes.inside_native].copy();probes['radius']=20
    frozen=read(ROOT/'frozen_roi_keypoints.csv')
    points=pd.concat([probes,frozen],ignore_index=True)
    variants=read(ROOT/'pose_probe_summary.csv').variant.tolist()+['frozen_roi']
    rows=[];pair_rows=[];summary=[]
    for variant in variants:
        for fid,r in completed.items():
            if r['visibility']=='uncertain':continue
            pp=points[points.frame_id.eq(fid)&points.variant.eq(variant)]
            xy=pp[['x','y']].to_numpy(float)
            regions=[v for v in r['regions'].values() if v is not None]
            matches=match_regions(xy,regions)
            correct=sum(d<=1 for i,j,d in matches)
            rows.append(dict(variant=variant,frame_id=fid,visibility=r['visibility'],
                expected_nostrils=len(regions),predicted_points=len(xy),correct_points=correct,
                missed_nostrils=len(regions)-correct,extra_or_off_target_points=len(xy)-correct,
                no_prediction_on_visible_frame=bool(regions) and not len(xy),
                no_correct_point_on_visible_frame=bool(regions) and correct==0))
            for i,j,d in matches:
                pair_rows.append(dict(variant=variant,frame_id=fid,predicted_keypoint=int(pp.iloc[i].keypoint),
                    human_region=j,normalized_center_distance=d,center_inside=d<=1,
                    radius=float(pp.iloc[i].radius),ROI_outside_human_ellipse_fraction=outside_fraction(
                        xy[i],pp.iloc[i].radius,regions[j],frames[fid]['width'],frames[fid]['height'])))
    detail=pd.DataFrame(rows)
    for variant,g in detail.groupby('variant'):
        expected=int(g.expected_nostrils.sum());predicted=int(g.predicted_points.sum());correct=int(g.correct_points.sum())
        summary.append(dict(variant=variant,scored_frames=len(g),visible_frames=int(g.expected_nostrils.gt(0).sum()),
            expected_nostrils=expected,predicted_points=predicted,correct_points=correct,
            point_recall=correct/expected if expected else None,point_precision=correct/predicted if predicted else None,
            no_prediction_visible_frames=int(g.no_prediction_on_visible_frame.sum()),
            no_correct_point_visible_frames=int(g.no_correct_point_on_visible_frame.sum())))
    args.out.mkdir(parents=True,exist_ok=False)
    write_csv(args.out/'frame_results.csv',rows);write_csv(args.out/'matched_region_results.csv',pair_rows)
    write_csv(args.out/'summary.csv',summary)
    write_json(args.out/'provenance.json',dict(reference_file=str(args.input),reference_sha256=sha256(args.input),
        package_id=package['package_id'],completed=len(completed),pending=72-len(completed),
        uncertain=sum(r['visibility']=='uncertain' for r in completed.values()),
        scope='submitted frames only; not all271 detection AP; one-observer approximate anatomical ellipses',
        point_match='unordered one-to-one; normalized ellipse distance <=1',
        diagnostic_variant_radius=20,frozen_roi_radius='actual stored adaptive radius',
        incomplete_annotations_not_zeroed=True,no_training_or_default_change=True))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
