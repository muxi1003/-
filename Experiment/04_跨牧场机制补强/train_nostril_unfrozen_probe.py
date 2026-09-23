"""Fixed200 three-fold nostril probe: change only the training freeze setting."""
import os
os.environ['WANDB_MODE']='disabled'
os.environ['YOLO_AUTOINSTALL']='false'
import contextlib
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import callbacks, LOGGER
from analyze_transfer import read,write_csv,write_json,sha256
from train_nostril_grouped_probe import ROOT,OUT as BASE,REF,WEIGHTS
from probe_pose_mechanism import load_bgr

CONTROL=ROOT/'direct_nostril_longer_v1'
OUT=ROOT/'direct_nostril_unfrozen_v1'


def main():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((CONTROL/'protocol_before_training.json').read_text())
    cfg=dict(prior['training'],freeze=0)
    assert [k for k in cfg if cfg[k]!=prior['training'][k]]==['freeze']
    assert sha256(REF)==prior['reference_sha256'] and sha256(WEIGHTS)==prior['init_sha256']
    assert sha256(BASE/'group_split.csv')==prior['shared_split_sha256']
    callbacks.add_integration_callbacks=lambda instance:None
    torch.set_num_threads(2)
    torch.manual_seed(20260922)
    x=torch.randn(32,32,device='cuda');assert torch.isfinite(x@x).all()
    # Model construction follows the existing runner; trainer applies its fixed seed.
    write_json(OUT/'protocol_before_training.json',dict(
        hypothesis='Frozen backbone may prevent adaptation to local nostril appearance',
        role='retrospective cow-grouped adaptation, not new independent external validation',
        training=cfg,changed_from_fixed200=['freeze:10->0'],
        control_protocol_sha256=sha256(CONTROL/'protocol_before_training.json'),
        shared_split_sha256=prior['shared_split_sha256'],primary_prediction=prior['primary_prediction'],
        checkpoint='last.pt at fixed epoch200, never best.pt',
        reference_sha256=sha256(REF),init_sha256=sha256(WEIGHTS),script_sha256=sha256(Path(__file__)),
        no_default_change=True,no_RR_truth_read=True,
        initialization_limit='Same initialization checkpoint and trainer seed; pre-trainer random construction is not archived in the historical control',
        environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),reuse='plant_gpu; no rebuild',
                         seeded_cuda_matmul='32x32 finite passed')))
    split=read(BASE/'group_split.csv');frames=[];points=[];checkpoints=[];start=time.monotonic()
    for fold in range(3):
        print(f'Fold {fold+1}/3: unfrozen fixed200 starting',flush=True)
        model=YOLO('yolo11n.yaml').load(str(WEIGHTS))
        with (OUT/f'fold{fold}_training.log').open('w',encoding='utf-8') as log:
            streams=[]
            for h in LOGGER.handlers:
                if hasattr(h,'setStream'):streams.append((h,h.stream));h.setStream(log)
            try:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    model.train(data=str(BASE/f'data_fold{fold}/dataset.yaml'),
                                project=str(OUT/'runs'),name=f'fold{fold}',exist_ok=False,**cfg)
            finally:
                for h,stream in streams:h.setStream(stream)
        last=OUT/'runs'/f'fold{fold}'/'weights/last.pt'
        assert len(read(OUT/'runs'/f'fold{fold}'/'results.csv'))==200
        detector=YOLO(str(last))
        for r in split[split.heldout_fold.eq(fold)].itertuples():
            result=detector(load_bgr(ROOT/'frames'/f'{r.frame_id}.png'),device=0,verbose=False,**prior['primary_prediction'])[0]
            frames.append(dict(frame_id=r.frame_id,fold=fold,cow_id=r.cow_id,prediction_count=len(result.boxes)))
            for k,(box,conf) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                x0,y0,x1,y1=map(float,box)
                points.append(dict(frame_id=r.frame_id,fold=fold,cow_id=r.cow_id,prediction_index=k,
                                   x=(x0+x1)/2,y=(y0+y1)/2,x0=x0,y0=y0,x1=x1,y1=y1,confidence=float(conf)))
        checkpoints.append(dict(fold=fold,checkpoint=str(last),sha256=sha256(last),epochs=200))
        write_json(OUT/f'fold{fold}_completed.json',checkpoints[-1])
        del detector,model;torch.cuda.empty_cache()
        print(f'Fold {fold+1}/3 complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
    write_csv(OUT/'out_of_fold_predictions.csv',points,columns=['frame_id','fold','cow_id','prediction_index','x','y','x0','y0','x1','y1','confidence'])
    write_csv(OUT/'frame_predictions.csv',frames);write_csv(OUT/'fold_checkpoints.csv',checkpoints)
    assert len(frames)==36 and sha256(REF)==prior['reference_sha256'] and sha256(WEIGHTS)==prior['init_sha256']
    write_json(OUT/'prediction_seal.json',dict(folds_completed=3,heldout_frames=36,
        elapsed_seconds=time.monotonic()-start,original_model_unchanged=True,
        files={n:sha256(OUT/n) for n in ['out_of_fold_predictions.csv','frame_predictions.csv','fold_checkpoints.csv','protocol_before_training.json']}))
    print('Three-fold predictions sealed.',flush=True)


if __name__=='__main__':main()
