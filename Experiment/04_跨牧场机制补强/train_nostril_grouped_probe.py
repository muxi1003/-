"""Cow-grouped feasibility probe: directly detect submitted nostril ellipse boxes."""
import os
os.environ['WANDB_MODE']='disabled'
os.environ['YOLO_AUTOINSTALL']='false'
import sys
import json
import time
import shutil
import hashlib
import contextlib
from pathlib import Path
import numpy as np
import torch
import yaml
from analyze_transfer import HERE,HOLDOUT,read,write_csv,write_json,sha256
from evaluate_nostril_reference import validate_submission
from probe_pose_mechanism import load_bgr

ROOT=HERE/'20260922_pose_phase_v1'
OUT=ROOT/'direct_nostril_grouped_v1'
REF=ROOT/'anatomy_reference_142648_v1/reference_original.json'
WEIGHTS=HOLDOUT/'method_snapshots/20260909_v1/weights/YOLO11n-best.pt'
TRAIN=dict(epochs=30,imgsz=640,batch=4,device=0,workers=0,seed=20260922,deterministic=True,
    optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,warmup_epochs=1,patience=100,
    freeze=10,amp=False,cache=False,plots=False,val=False,save=True,save_period=-1,
    hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=.05,scale=.1,shear=0.,perspective=0.,
    flipud=0.,fliplr=.5,mosaic=0.,mixup=0.,copy_paste=0.,close_mosaic=0,verbose=False)


def prepare():
    OUT.mkdir(exist_ok=False)
    data=json.loads(REF.read_text(encoding='utf-8'))
    package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    frames=validate_submission(data,package)
    audit=read(HERE/'20260921_r3_v1/annotation_audit.csv').set_index('window_id')
    records={fid:r for fid,r in data['records'].items() if r['status']=='complete' and r['visibility']!='uncertain'}
    groups={fid:str(audit.loc[frames[fid]['window_id'],'cluster']) for fid in records}
    cows=sorted(set(groups.values()),key=lambda s:hashlib.sha256(('20260922:'+s).encode()).hexdigest())
    fold_of={cow:i%3 for i,cow in enumerate(cows)}
    rows=[]
    for fid,r in records.items():
        f=frames[fid]
        rows.append(dict(frame_id=fid,window_id=f['window_id'],cow_id=groups[fid],heldout_fold=fold_of[groups[fid]],
            visibility=r['visibility'],region_count=sum(e is not None for e in r['regions'].values()),image_sha256=f['image_sha256']))
    assert len(rows)==36 and len(cows)==12
    write_csv(OUT/'group_split.csv',rows)
    protocol=dict(role='retrospective_3fold_cow_grouped_feasibility_not_new_external_test',
        target='one nostril per bounding rectangle enclosing submitted human ellipse',
        original_nose_pose_weights='backbone initialization, not modified',model='installed yolo11n detect architecture',
        training=TRAIN,fold_count=3,groups=len(cows),frames=len(rows),
        checkpoint='last.pt after fixed30 epochs, never choose best.pt using heldout metrics',
        primary_prediction=dict(conf=.25,iou=.7,max_det=2,imgsz=640),
        fixed_comparisons=['direct_nostril_centers','fallback_only_when_original_has_no_high_confidence_points'],
        no_epoch_or_hyperparameter_search=True,no_online_integrations=True,
        scope='repeated views of12 selected cows; not representative full271 AP or validated RR',
        no_RR_truth_read_by_training=True,reference_sha256=sha256(REF),init_sha256=sha256(WEIGHTS),
        script_sha256=sha256(Path(__file__)),default_changed=False,
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),reuse='existing plant_gpu; no install or rebuild',
            seeded_kernel_witness='32x32 finite CUDA matmul verified before launch'))
    write_json(OUT/'protocol_before_training.json',protocol)
    for fold in range(3):
        base=OUT/f'data_fold{fold}'
        for split in ['train','val']:
            (base/'images'/split).mkdir(parents=True)
            (base/'labels'/split).mkdir(parents=True)
        train_cows=set();val_cows=set()
        for row in rows:
            fid=row['frame_id'];f=frames[fid];r=records[fid]
            split='val' if row['heldout_fold']==fold else 'train'
            (val_cows if split=='val' else train_cows).add(row['cow_id'])
            source=ROOT/'frames'/f'{fid}.png';assert sha256(source)==row['image_sha256']
            shutil.copy2(source,base/'images'/split/f'{fid}.png')
            labels=[]
            for e in r['regions'].values():
                if e is None:continue
                labels.append(f"0 {e['cx']/f['width']:.8f} {e['cy']/f['height']:.8f} {2*e['rx']/f['width']:.8f} {2*e['ry']/f['height']:.8f}")
            (base/'labels'/split/f'{fid}.txt').write_text('\n'.join(labels)+('\n' if labels else ''),encoding='ascii')
        assert not train_cows&val_cows and len(train_cows)==8 and len(val_cows)==4
        config=dict(path=str(base.resolve()),train='images/train',val='images/val',names={0:'nostril'})
        (base/'dataset.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    return rows


def main():
    from ultralytics import YOLO
    from ultralytics.utils import callbacks,LOGGER
    # Local-only run: suppress optional network logging integration callbacks in this process.
    callbacks.add_integration_callbacks=lambda instance:None
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    rows=prepare()
    results=[];fold_records=[];started=time.monotonic()
    for fold in range(3):
        print(f'Fold {fold+1}/3 starting:8 train cows,4 heldout cows',flush=True)
        model=YOLO('yolo11n.yaml').load(str(WEIGHTS))
        with (OUT/f'fold{fold}_training.log').open('w',encoding='utf-8') as log:
            streams=[]
            for handler in LOGGER.handlers:
                if hasattr(handler,'setStream'):
                    streams.append((handler,handler.stream));handler.setStream(log)
            try:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    model.train(data=str(OUT/f'data_fold{fold}/dataset.yaml'),project=str(OUT/'runs'),
                        name=f'fold{fold}',exist_ok=False,**TRAIN)
            finally:
                for handler,stream in streams:handler.setStream(stream)
        last=OUT/'runs'/f'fold{fold}'/'weights/last.pt';assert last.exists()
        detector=YOLO(str(last))
        for row in rows:
            if row['heldout_fold']!=fold:continue
            im=load_bgr(ROOT/'frames'/f"{row['frame_id']}.png")
            pred=detector(im,conf=.25,iou=.7,max_det=2,imgsz=640,verbose=False)[0]
            for k,(box,conf) in enumerate(zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy())):
                x0,y0,x1,y1=map(float,box)
                results.append(dict(frame_id=row['frame_id'],fold=fold,cow_id=row['cow_id'],prediction_index=k,
                    x=(x0+x1)/2,y=(y0+y1)/2,x0=x0,y0=y0,x1=x1,y1=y1,confidence=float(conf)))
        fold_records.append(dict(fold=fold,train_cows=8,heldout_cows=4,train_frames=24,heldout_frames=12,
            checkpoint=str(last),sha256=sha256(last),epochs=30))
        write_json(OUT/f'fold{fold}_completed.json',fold_records[-1])
        print(f'Fold {fold+1}/3 complete, elapsed {time.monotonic()-started:.1f}s',flush=True)
        del detector,model;torch.cuda.empty_cache()
    write_csv(OUT/'out_of_fold_predictions.csv',results)
    write_csv(OUT/'fold_checkpoints.csv',fold_records)
    assert sha256(REF)==json.loads((OUT/'protocol_before_training.json').read_text(encoding='utf-8'))['reference_sha256']
    assert sha256(WEIGHTS)==json.loads((OUT/'protocol_before_training.json').read_text(encoding='utf-8'))['init_sha256']
    write_json(OUT/'prediction_seal.json',dict(folds_completed=3,heldout_frames=36,heldout_cows=12,
        no_cow_crosses_train_val_within_fold=True,all_predictions_out_of_fold=True,
        original_model_unchanged=True,elapsed_seconds=time.monotonic()-started,
        files={name:sha256(OUT/name) for name in ['out_of_fold_predictions.csv','group_split.csv','fold_checkpoints.csv','protocol_before_training.json']}))
    print('Grouped training and OOF predictions sealed; evaluation separate',flush=True)


if __name__=='__main__':main()
