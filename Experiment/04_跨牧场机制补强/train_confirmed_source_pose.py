"""Fixed source-group pose training; Jiufu images are inference-only diagnostics."""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['YOLO_AUTOINSTALL'] = 'false'
import contextlib
import json
import time
from pathlib import Path
import torch
import cv2
from ultralytics import YOLO
from ultralytics.utils import callbacks, LOGGER
from analyze_transfer import read, write_json, write_csv, sha256
from build_confirmed_pose_dataset import DEST
from probe_pose_mechanism import ROOT, FROZEN, load_bgr

OUT = ROOT/'confirmed_source_pose_fold0_v1'
INIT = Path(__file__).resolve().parents[2]/'yolo11n-pose.pt'
TRAIN = dict(epochs=40,imgsz=640,batch=8,device=0,workers=2,seed=20260923,
    deterministic=True,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,
    warmup_epochs=2,patience=0,freeze=0,amp=False,cache=False,plots=False,
    val=False,save=True,save_period=-1,hsv_h=0.,hsv_s=0.,hsv_v=0.,
    degrees=10.,translate=.1,scale=.2,shear=0.,perspective=0.,flipud=0.,
    fliplr=.5,mosaic=0.,mixup=0.,copy_paste=0.,close_mosaic=0,verbose=False)


def infer(model, image, frame_id, cohort):
    im = load_bgr(image)
    result = model(im, imgsz=640,conf=.25,iou=.7,max_det=10,device=0,verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    return dict(frame_id=frame_id,cohort=cohort,image_path=str(image),image_sha256=sha256(Path(image)),
                width=im.shape[1],height=im.shape[0],boxes=boxes,
                box_conf=result.boxes.conf.cpu().numpy().tolist(),
                keypoints=result.keypoints.data.cpu().numpy().tolist())


def main():
    OUT.mkdir(exist_ok=False)
    assert json.loads((DEST/'verification.json').read_text())['status']=='PASS'
    manifest = read(DEST/'manifest.csv')
    included = manifest[manifest.status.eq('included')]
    train = included[included.fold.ne(0)]
    val = included[included.fold.eq(0)]
    assert not set(train.group)&set(val.group)
    callbacks.add_integration_callbacks = lambda instance: None
    torch.set_num_threads(2); cv2.setNumThreads(1)
    protocol = dict(role='fixed single-fold source-group training plus retrospective Jiufu diagnostic',
        training=TRAIN,fold=0,train_frames=len(train),validation_frames=len(val),
        target='nose box and left/right nostril keypoints; not ellipse segmentation',
        initialization=str(INIT),init_sha256=sha256(INIT),
        init_metadata='local person17-keypoint checkpoint, not old cattle Pose checkpoint',
        checkpoint_rule='last.pt at fixed epoch40; never choose checkpoint from Jiufu or best.pt',
        primary_inference=dict(box_conf=.25,kpt_conf=.5,imgsz=640,iou=.7,highest_conf_box=True),
        source_manifest_sha256=sha256(DEST/'manifest.csv'),
        source_config_sha256=sha256(DEST/'fold0.yaml'),
        script_sha256=sha256(Path(__file__)),
        jiufu_reference_not_used_in_training=True,default_changed=False,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                         reuse='plant_gpu; no install/rebuild',seeded_cuda_matmul='32x32 finite witness passed'),
        limitation='One predefined source fold, not completed five-fold CV; viewed Jiufu is not a new independent external test')
    write_json(OUT/'protocol_before_training.json',protocol)
    start = time.monotonic()
    def progress(trainer):
        print(f'Epoch {trainer.epoch+1}/{TRAIN["epochs"]} complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
    model = YOLO(str(INIT))
    model.add_callback('on_fit_epoch_end',progress)
    with (OUT/'training.log').open('w',encoding='utf-8') as log:
        streams=[]
        for h in LOGGER.handlers:
            if hasattr(h,'setStream'):
                streams.append((h,h.stream));h.setStream(log)
        try:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                model.train(data=str(DEST/'fold0.yaml'),project=str(OUT/'runs'),name='fold0',exist_ok=False,**TRAIN)
        finally:
            for h,stream in streams:h.setStream(stream)
    checkpoint=OUT/'runs/fold0/weights/last.pt'
    assert len(read(OUT/'runs/fold0/results.csv'))==40
    assert sha256(INIT)==protocol['init_sha256']
    assert sha256(DEST/'manifest.csv')==protocol['source_manifest_sha256']
    write_json(OUT/'training_complete.json',dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint),
                                               elapsed_seconds=time.monotonic()-start,epochs=40))
    print('Training complete. Fixed last checkpoint sealed; inference starting.',flush=True)
    frames=read(ROOT/'frame_manifest.csv')
    # Infer every predefined diagnostic frame without loading anatomical or RR reference values.
    for name,weights in [('candidate',checkpoint),('baseline',FROZEN/'weights/YOLO11n-best.pt')]:
        predictor=YOLO(str(weights));predictions=[]
        for r in frames.itertuples():
            predictions.append(infer(predictor,r.image_path,r.frame_id,'jiufu_review72'))
        for r in val.itertuples():
            predictions.append(infer(predictor,r.image_path,r.frame_id,'source_validation'))
        write_json(OUT/f'{name}_predictions.json',predictions)
        print(name, len(predictions),'predictions sealed',flush=True)
        del predictor
        torch.cuda.empty_cache()
    write_json(OUT/'prediction_seal.json',dict(default_changed=False,
        files={n:sha256(OUT/n) for n in ['candidate_predictions.json','baseline_predictions.json','training_complete.json']},
        elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
