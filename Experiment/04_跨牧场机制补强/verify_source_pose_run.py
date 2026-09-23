"""Verify fixed training config and replay native-coordinate predictions."""
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from analyze_transfer import read, write_json, sha256
from train_confirmed_source_pose import OUT, INIT, TRAIN, infer
from build_confirmed_pose_dataset import DEST


def main():
    protocol=json.loads((OUT/'protocol_before_training.json').read_text(encoding='utf-8'))
    done=json.loads((OUT/'training_complete.json').read_text())
    args=yaml.safe_load((OUT/'runs/fold0/args.yaml').read_text(encoding='utf-8'))
    for k,v in TRAIN.items():
        if k=='device':assert str(args[k])==str(v)
        else:assert args[k]==v,(k,args[k],v)
    assert args['resume'] is False and args['fraction']==1.
    assert len(read(OUT/'runs/fold0/results.csv'))==40
    assert sha256(INIT)==protocol['init_sha256']
    assert sha256(DEST/'manifest.csv')==protocol['source_manifest_sha256']
    assert sha256(Path(done['checkpoint']))==done['sha256']
    model=YOLO(done['checkpoint']);torch.set_num_threads(2)
    assert model.model.yaml['kpt_shape']==[2,3]
    predictions=json.loads((OUT/'candidate_predictions.json').read_text(encoding='utf-8'))
    assert len(predictions)==645 and len({p['frame_id'] for p in predictions})==645
    selected=[p for p in predictions if p['cohort']=='jiufu_review72']
    selected += sorted([p for p in predictions if p['cohort']=='source_validation'],key=lambda p:p['frame_id'])[::29]
    for p in selected:
        assert sha256(Path(p['image_path']))==p['image_sha256']
        q=infer(model,p['image_path'],p['frame_id'],p['cohort'])
        for key in ['boxes','box_conf','keypoints']:
            np.testing.assert_allclose(q[key],p[key],atol=1e-4,rtol=1e-6,err_msg=p['frame_id'])
    d=read(OUT/'anatomy_frame_results.csv')
    s=read(OUT/'anatomy_summary.csv').set_index('variant')
    for variant,g in d.groupby('variant'):
        assert len(g)==36 and g.expected.sum()==50
        assert g.correct.sum()==s.loc[variant,'correct_points']
        assert (g.correct<=g.expected).all() and (g.correct<=g.predicted).all()
    write_json(OUT/'verification.json',dict(status='PASS',epochs=40,training_config_matches=True,
        keypoint_shape=[2,3],unique_candidate_predictions=645,replayed_predictions=len(selected),
        replay_scope='all72 Jiufu diagnostic frames plus every29th of573 sorted source validation frames',
        summary_arithmetic_pass=True,init_and_source_manifest_unchanged=True,
        scope_limit='No full RR evaluation or independent anatomical re-annotation'))
    print('PASS: configuration, checkpoint, 645 IDs,',len(selected),'prediction replays and anatomy totals')


if __name__=='__main__':main()
