"""Fixed longer schedule after evidence of poor training-set fit; no threshold search."""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['YOLO_AUTOINSTALL'] = 'false'
import contextlib
import json
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import callbacks, LOGGER
from analyze_transfer import read, write_csv, write_json, sha256
from train_nostril_grouped_probe import ROOT, OUT as BASE, REF, WEIGHTS, TRAIN
from probe_pose_mechanism import load_bgr

OUT = ROOT / 'direct_nostril_longer_v1'


def main():
    OUT.mkdir(exist_ok=False)
    callbacks.add_integration_callbacks = lambda instance: None
    torch.set_num_threads(2)
    split = read(BASE / 'group_split.csv')
    cfg = dict(TRAIN, epochs=200, patience=0)
    write_json(OUT / 'protocol_before_training.json', dict(
        hypothesis='30 epochs failed to fit training targets; test fixed longer schedule',
        role='retrospective cow-grouped adaptation, not new independent external validation',
        training=cfg, changed_from_30epoch=['epochs:30->200', 'patience:100->0 (disable early stopping)'],
        shared_split_sha256=sha256(BASE / 'group_split.csv'),
        primary_prediction=dict(conf=.25, iou=.7, max_det=2, imgsz=640),
        checkpoint='last.pt at fixed epoch200, never best.pt',
        reference_sha256=sha256(REF), init_sha256=sha256(WEIGHTS),
        script_sha256=sha256(Path(__file__)), no_hyperparameter_search=True,
        no_default_change=True, no_RR_truth_read=True))
    frames = []; points = []; checkpoints = []
    started = time.monotonic()
    for fold in range(3):
        print(f'Fold {fold+1}/3: fixed200 epochs starting', flush=True)
        model = YOLO('yolo11n.yaml').load(str(WEIGHTS))
        with (OUT / f'fold{fold}_training.log').open('w', encoding='utf-8') as log:
            streams = []
            for handler in LOGGER.handlers:
                if hasattr(handler, 'setStream'):
                    streams.append((handler, handler.stream)); handler.setStream(log)
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    model.train(data=str(BASE / f'data_fold{fold}/dataset.yaml'),
                                project=str(OUT / 'runs'), name=f'fold{fold}', exist_ok=False, **cfg)
            finally:
                for handler, stream in streams:
                    handler.setStream(stream)
        last = OUT / 'runs' / f'fold{fold}' / 'weights/last.pt'
        assert len(read(OUT / 'runs' / f'fold{fold}' / 'results.csv')) == 200
        detector = YOLO(str(last))
        for row in split[split.heldout_fold.eq(fold)].itertuples():
            result = detector(load_bgr(ROOT / 'frames' / f'{row.frame_id}.png'),
                              conf=.25, iou=.7, max_det=2, imgsz=640, device=0, verbose=False)[0]
            frames.append(dict(frame_id=row.frame_id, fold=fold, cow_id=row.cow_id, prediction_count=len(result.boxes)))
            for k, (box, conf) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())):
                x0, y0, x1, y1 = map(float, box)
                points.append(dict(frame_id=row.frame_id, fold=fold, cow_id=row.cow_id, prediction_index=k,
                    x=(x0+x1)/2, y=(y0+y1)/2, x0=x0, y0=y0, x1=x1, y1=y1, confidence=float(conf)))
        checkpoints.append(dict(fold=fold, checkpoint=str(last), sha256=sha256(last), epochs=200))
        write_json(OUT / f'fold{fold}_completed.json', checkpoints[-1])
        del detector, model
        torch.cuda.empty_cache()
        print(f'Fold {fold+1}/3 complete; elapsed {time.monotonic()-started:.1f}s', flush=True)
    assert len(frames) == 36
    write_csv(OUT / 'out_of_fold_predictions.csv', points,
              columns=['frame_id','fold','cow_id','prediction_index','x','y','x0','y0','x1','y1','confidence'])
    write_csv(OUT / 'frame_predictions.csv', frames)
    write_csv(OUT / 'fold_checkpoints.csv', checkpoints)
    before = json.loads((OUT / 'protocol_before_training.json').read_text(encoding='utf-8'))
    assert sha256(REF) == before['reference_sha256'] and sha256(WEIGHTS) == before['init_sha256']
    write_json(OUT / 'prediction_seal.json', dict(folds_completed=3, heldout_frames=36,
        elapsed_seconds=time.monotonic()-started, original_model_unchanged=True,
        files={n:sha256(OUT/n) for n in ['out_of_fold_predictions.csv','frame_predictions.csv',
             'fold_checkpoints.csv','protocol_before_training.json']}))
    print('Predictions sealed; no evaluation-driven checkpoint selection.', flush=True)


if __name__ == '__main__':
    main()
