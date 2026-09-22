"""Replay each held-out frame and retain explicit zero-detection records."""
import json
import argparse
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO
from analyze_transfer import read, write_csv, write_json, sha256
from train_nostril_grouped_probe import ROOT, OUT, REF, WEIGHTS
from probe_pose_mechanism import load_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, default=OUT)
    output = parser.parse_args().model_root
    torch.set_num_threads(2)
    protocol = json.loads((output / 'protocol_before_training.json').read_text(encoding='utf-8'))
    seal = json.loads((output / 'prediction_seal.json').read_text(encoding='utf-8'))
    assert sha256(REF) == protocol['reference_sha256']
    assert sha256(WEIGHTS) == protocol['init_sha256']
    for name, digest in seal['files'].items():
        assert sha256(output / name) == digest
    split = read(OUT / 'group_split.csv')
    if 'shared_split_sha256' in protocol:
        assert sha256(OUT / 'group_split.csv') == protocol['shared_split_sha256']
    try:
        sealed_predictions = read(output / 'out_of_fold_predictions.csv')
    except pd.errors.EmptyDataError:
        sealed_predictions = pd.DataFrame(columns=['frame_id', 'x', 'y', 'confidence'])
    rows = []
    for fold in range(3):
        heldout = split[split.heldout_fold == fold]
        training = split[split.heldout_fold != fold]
        assert len(heldout) == 12 and len(training) == 24
        assert not set(heldout.cow_id) & set(training.cow_id)
        completed = json.loads((output / f'fold{fold}_completed.json').read_text(encoding='utf-8'))
        checkpoint = output / 'runs' / f'fold{fold}' / 'weights/last.pt'
        assert sha256(checkpoint) == completed['sha256']
        epochs = read(output / 'runs' / f'fold{fold}' / 'results.csv')
        assert len(epochs) == protocol['training']['epochs']
        model = YOLO(str(checkpoint))
        for row in heldout.itertuples():
            source = ROOT / 'frames' / f'{row.frame_id}.png'
            assert sha256(source) == row.image_sha256
            result = model(load_bgr(source), device=0, verbose=False,
                           **protocol['primary_prediction'])[0]
            expected = sealed_predictions[sealed_predictions.frame_id.eq(row.frame_id)]
            assert len(result.boxes) == len(expected)
            if len(expected):
                import numpy as np
                box = result.boxes.xyxy.cpu().numpy()
                np.testing.assert_allclose((box[:, :2]+box[:, 2:])/2,
                                           expected[['x','y']].to_numpy(float), atol=1e-4)
                np.testing.assert_allclose(result.boxes.conf.cpu().numpy(),
                                           expected.confidence.to_numpy(float), atol=1e-5)
            rows.append(dict(frame_id=row.frame_id, fold=fold, cow_id=row.cow_id,
                             prediction_count=len(result.boxes),
                             checkpoint_sha256=completed['sha256']))
    assert len(rows) == 36 and len({r['frame_id'] for r in rows}) == 36
    count = sum(r['prediction_count'] for r in rows)
    assert count == len(sealed_predictions)
    write_csv(output / 'verification_frame_predictions.csv', rows)
    write_json(output / 'verification.json', dict(status='PASS', replayed_frames=36,
        predicted_boxes_at_conf025=count, folds=3, epochs_each=protocol['training']['epochs'],
        cow_isolation_verified=True, original_model_and_reference_unchanged=True,
        prediction_seal_verified=True, training_rerun=False))
    print(f'PASS: 36 held-out frames replayed; {count} boxes at fixed conf=.25; coordinates, confidence, hashes and cow isolation verified.')


if __name__ == '__main__':
    main()
