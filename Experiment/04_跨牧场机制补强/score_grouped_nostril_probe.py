"""Score cow-held-out direct detector and fixed no-keypoint fallback."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_transfer import read,write_csv,write_json,sha256
from train_nostril_grouped_probe import ROOT,OUT,REF
from evaluate_nostril_reference import validate_submission,match_regions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction-root',type=Path,default=OUT)
    output=parser.parse_args().prediction_root
    seal=json.loads((output/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,h in seal['files'].items():assert sha256(output/name)==h
    raw=json.loads(REF.read_text(encoding='utf-8'));package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    frames=validate_submission(raw,package);split=read(OUT/'group_split.csv').set_index('frame_id')
    try:new=read(output/'out_of_fold_predictions.csv')
    except pd.errors.EmptyDataError:new=pd.DataFrame(columns=['frame_id','fold','cow_id','prediction_index','x','y','confidence'])
    for p in new.itertuples():assert int(split.loc[p.frame_id,'heldout_fold'])==p.fold
    for fold in range(3):
        train=set(split[split.heldout_fold.ne(fold)].cow_id.astype(str));val=set(split[split.heldout_fold.eq(fold)].cow_id.astype(str))
        assert not train&val and len(train)==8 and len(val)==4
    old=read(ROOT/'pose_keypoints.csv');old=old[old.variant.eq('baseline')&old.confidence.ge(.5)&old.inside_native]
    records=[];matched=[]
    for fid,r in raw['records'].items():
        if r['status']!='complete' or r['visibility']=='uncertain':continue
        regions=[e for e in r['regions'].values() if e is not None]
        baseline=old[old.frame_id.eq(fid)][['x','y']].to_numpy(float)
        candidate=new[new.frame_id.eq(fid)][['x','y']].to_numpy(float)
        for method,points in [('original_pose',baseline),('direct_nostril_oof',candidate),
                              ('no_keypoint_fallback_oof',baseline if len(baseline) else candidate)]:
            pairs=match_regions(points,regions);correct=sum(d<=1 for _,_,d in pairs)
            records.append(dict(method=method,frame_id=fid,cow_id=str(split.loc[fid,'cow_id']),
                fold=int(split.loc[fid,'heldout_fold']),visibility=r['visibility'],expected=len(regions),
                predicted=len(points),correct=correct,missed=len(regions)-correct,false_or_off_target=len(points)-correct,
                no_correct_when_visible=bool(regions) and correct==0))
            matched.extend(dict(method=method,frame_id=fid,predicted_index=i,human_region=j,normalized_distance=d,inside=d<=1) for i,j,d in pairs)
    d=pd.DataFrame(records);summary=[];differences=[]
    for method,g in d.groupby('method'):
        correct=int(g.correct.sum());pred=int(g.predicted.sum());truth=int(g.expected.sum())
        summary.append(dict(method=method,frames=len(g),cows=g.cow_id.nunique(),expected_nostrils=truth,
            predicted_points=pred,correct_points=correct,precision=correct/pred if pred else 0.,
            recall=correct/truth,point_f1=2*correct/(pred+truth),
            no_correct_visible_frames=int(g.no_correct_when_visible.sum())))
    baseline=d[d.method.eq('original_pose')].set_index('frame_id')
    for method in ['direct_nostril_oof','no_keypoint_fallback_oof']:
        g=d[d.method.eq(method)].set_index('frame_id');delta=g.correct-baseline.correct
        differences.append(dict(method=method,delta_correct=int(delta.sum()),better_frames=int((delta>0).sum()),
            worse_frames=int((delta<0).sum()),extra_false_or_off_target=int(g.false_or_off_target.sum()-baseline.false_or_off_target.sum())))
    dest=output/'evaluation';dest.mkdir(exist_ok=False)
    write_csv(dest/'summary.csv',summary);write_csv(dest/'per_frame.csv',records);write_csv(dest/'matched_regions.csv',matched)
    write_csv(dest/'paired_changes.csv',differences)
    write_json(dest/'adoption_decision.json',dict(
        disposition='DO_NOT_ADOPT' if differences[1]['delta_correct']<=0 or differences[1]['extra_false_or_off_target']>0 else 'REQUIRES_FULL_SIGNAL_VALIDATION',
        default_changed=False,goal_complete=False,reference_sha256=sha256(REF),
        restriction='small retrospective cow-grouped adaptation feasibility, no full-video RR or independent-farm proof',
        all36_frames_evaluated_even_zero_predictions=True,training_used_only_other_cows_in_each_fold=True))
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(differences).to_string(index=False))


if __name__=='__main__':main()
