"""Single bounded center-refinement probe; human labels are not predictor inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from analyze_transfer import HERE,HOLDOUT,read,write_csv,write_json,sha256
from probe_pose_mechanism import load_bgr
from recover_timestamp_segments import CachedRF

ROOT=HERE/'20260922_pose_phase_v1'
OUT=ROOT/'local_contrast_roi_v1'
POLICY=dict(candidate_offsets_in_radii=[[0,0],[-.5,0],[.5,0],[0,-.5],[0,.5]],
    inner_radius_fraction=.5,ring_inner_radius_fraction=1.,ring_outer_radius_fraction=1.5,
    min_finite_kernel_fraction=.8,min_RF_temperature_strict=20.,
    score='absolute(mean(inner)-mean(ring)); ties retain original then smaller displacement',
    anatomy_used_for_policy_motivation=True,anatomy_not_read_by_predictor=True,
    no_default_change=True,temperature_dependent_center_may_distort_RR_must_validate_temporally=True)


def region_score(field,cx,cy,radius):
    yy,xx=np.indices(field.shape)
    dd=(xx-cx)**2+(yy-cy)**2
    inner=dd<=(radius*.5)**2
    ring=(dd>radius**2)&(dd<=(radius*1.5)**2)
    finite=np.isfinite(field)
    # Reject clipped kernels rather than treating image borders as local contrast.
    expected_inner=np.pi*(radius*.5)**2
    expected_ring=np.pi*((radius*1.5)**2-radius**2)
    if (inner&finite).sum()<.8*expected_inner or (ring&finite).sum()<.8*expected_ring:
        return -np.inf
    return float(abs(field[inner&finite].mean()-field[ring&finite].mean()))


def refine(field,cx,cy,radius):
    candidates=[]
    for dx,dy in POLICY['candidate_offsets_in_radii']:
        candidates.append((cx+dx*radius,cy+dy*radius))
    scores=[region_score(field,x,y,radius) for x,y in candidates]
    if not np.isfinite(scores).any():return cx,cy,0,0.,scores
    best=max(range(len(scores)),key=lambda i:(scores[i],-(candidates[i][0]-cx)**2-(candidates[i][1]-cy)**2))
    x,y=candidates[best]
    return x,y,best,float(scores[best]-scores[0]) if np.isfinite(scores[0]) else np.nan,scores


def main():
    OUT.mkdir(exist_ok=False)
    inputs=[ROOT/'frame_manifest.csv',ROOT/'pose_keypoints.csv',ROOT/'frozen_roi_keypoints.csv',Path(__file__)]
    hashes={str(p):sha256(p) for p in inputs}
    write_json(OUT/'protocol_before_predictions.json',dict(role='retrospective_local_ROI_candidate',policy=POLICY,input_hashes=hashes))
    frames=read(ROOT/'frame_manifest.csv').set_index('frame_id')
    baseline=read(ROOT/'pose_keypoints.csv');baseline=baseline[baseline.variant.eq('baseline')&baseline.confidence.ge(.5)&baseline.inside_native].copy()
    baseline['radius']=20
    frozen=read(ROOT/'frozen_roi_keypoints.csv')
    seeds=pd.concat([baseline,frozen],ignore_index=True)
    rf=CachedRF(joblib.load(HOLDOUT/'method_snapshots/20260909_v1/weights/clf_model_RGB_20240906.pkl'))
    records=[];scores=[]
    for fid,group in seeds.groupby('frame_id'):
        path=ROOT/'frames'/f'{fid}.png';assert sha256(path)==frames.loc[fid,'image_sha256']
        im=load_bgr(path)
        for p in group.itertuples():
            r=float(p.radius);pad=int(np.ceil(2*r+2))
            x0=max(0,int(np.floor(p.x))-pad);x1=min(im.shape[1],int(np.ceil(p.x))+pad+1)
            y0=max(0,int(np.floor(p.y))-pad);y1=min(im.shape[0],int(np.ceil(p.y))+pad+1)
            patch=im[y0:y1,x0:x1];assert patch.size
            field=rf.predict(patch.reshape(-1,3)).reshape(patch.shape[:2])
            field=np.where(field>20,field,np.nan)
            nx,ny,index,gain,values=refine(field,p.x-x0,p.y-y0,r)
            newx,newy=nx+x0,ny+y0
            assert np.hypot(newx-p.x,newy-p.y)<=r*.5+1e-6
            records.append(dict(frame_id=fid,variant=p.variant+'_local_contrast',seed_variant=p.variant,keypoint=p.keypoint,
                x=newx,y=newy,radius=r,seed_x=p.x,seed_y=p.y,seed_confidence=p.confidence,
                choice=index,shift_pixels=float(np.hypot(newx-p.x,newy-p.y)),score_gain=gain))
            for choice,score in enumerate(values):scores.append(dict(frame_id=fid,seed_variant=p.variant,keypoint=p.keypoint,choice=choice,
                score=score if np.isfinite(score) else None))
    write_csv(OUT/'prediction_points.csv',records);write_csv(OUT/'candidate_scores.csv',scores)
    for p,h in hashes.items():assert sha256(Path(p))==h
    write_json(OUT/'prediction_seal.json',dict(reference_read=False,points=len(records),frames=seeds.frame_id.nunique(),
        files={n:sha256(OUT/n) for n in ['prediction_points.csv','candidate_scores.csv','protocol_before_predictions.json']}))
    print(pd.DataFrame(records).groupby('seed_variant').agg(points=('choice','size'),shifted=('choice',lambda x:int((x!=0).sum()))))


if __name__=='__main__':main()
