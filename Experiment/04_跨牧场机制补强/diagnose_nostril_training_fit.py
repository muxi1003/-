"""Training-set fit versus held-out fit; low confidence is diagnostic only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from analyze_transfer import read, write_csv, write_json, sha256
from train_nostril_grouped_probe import ROOT, OUT, REF
from probe_pose_mechanism import load_bgr
from evaluate_nostril_reference import match_regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, default=OUT)
    args = parser.parse_args()
    dest = args.model_root / 'training_fit_diagnostic'
    dest.mkdir(exist_ok=False)
    raw = json.loads(REF.read_text(encoding='utf-8'))
    split = read(OUT / 'group_split.csv')
    package = json.loads((ROOT / 'annotation_package.json').read_text(encoding='utf-8'))
    frames = {f['frame_id']: f for f in package['frames']}
    write_json(dest / 'protocol.json', dict(
        purpose='distinguish failure to fit training targets from held-out generalization',
        thresholds=[.25, .001], max_det=2, iou=.7, imgsz=640,
        low_threshold_diagnostic_only=True, default_changed=False,
        reference_sha256=sha256(REF)))
    torch.set_num_threads(2)
    details = []
    labels_checked = 0
    for fold in range(3):
        model = YOLO(str(args.model_root / 'runs' / f'fold{fold}' / 'weights/last.pt'))
        for row in split.itertuples():
            fid = row.frame_id
            subset = 'heldout' if row.heldout_fold == fold else 'training'
            region = [e for e in raw['records'][fid]['regions'].values() if e is not None]
            label_path = OUT / f'data_fold{fold}/labels' / ('val' if subset == 'heldout' else 'train') / f'{fid}.txt'
            labels = [list(map(float, line.split())) for line in label_path.read_text().splitlines() if line.strip()]
            assert len(labels) == len(region)
            for lab, e in zip(labels, region):
                f = frames[fid]
                np.testing.assert_allclose(np.array(lab[1:]) * [f['width'], f['height'], f['width'], f['height']],
                                           [e['cx'], e['cy'], 2*e['rx'], 2*e['ry']], atol=.0001)
            labels_checked += 1
            image = load_bgr(ROOT / 'frames' / f'{fid}.png')
            assert image.shape[:2] == (frames[fid]['height'], frames[fid]['width'])
            for threshold in [.25, .001]:
                pred = model(image, conf=threshold, imgsz=640, max_det=2, iou=.7, device=0, verbose=False)[0]
                boxes = pred.boxes.xyxy.cpu().numpy()
                points = (boxes[:, :2] + boxes[:, 2:]) / 2
                matches = match_regions(points, region)
                correct = sum(d <= 1 for _, _, d in matches)
                conf = pred.boxes.conf.cpu().numpy()
                details.append(dict(fold=fold, subset=subset, frame_id=fid, cow_id=row.cow_id,
                    threshold=threshold, expected=len(region), predicted=len(boxes), correct=correct,
                    max_confidence=float(conf.max()) if len(conf) else 0.,
                    correct_max_confidence=max([float(conf[i]) for i, _, d in matches if d <= 1], default=0.)))
    d = pd.DataFrame(details)
    summary = []
    for (subset, threshold), g in d.groupby(['subset', 'threshold']):
        correct, predicted, expected = [int(g[col].sum()) for col in ['correct', 'predicted', 'expected']]
        summary.append(dict(subset=subset, threshold=threshold, frame_model_pairs=len(g),
            expected=expected, predicted=predicted, correct=correct,
            precision=correct/predicted if predicted else 0., recall=correct/expected,
            max_confidence=float(g.max_confidence.max()),
            median_frame_max_confidence=float(g.max_confidence.median())))
    write_csv(dest / 'per_frame.csv', details)
    write_csv(dest / 'summary.csv', summary)
    write_json(dest / 'verification.json', dict(label_files_roundtrip_checked=labels_checked,
        image_dimensions_checked=True, train_frame_model_pairs=72, heldout_frame_model_pairs=36,
        same_training_frame_appears_in_two_models=True, no_new_independent_validation=True))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
