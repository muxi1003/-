"""Fixed-input pose probes; transformed pixels never enter temperature mapping."""
from pathlib import Path
import json
import time
import cv2
import numpy as np
import pandas as pd
from analyze_transfer import HERE, HOLDOUT, read, write_csv, write_json, sha256

ROOT = HERE / '20260922_pose_phase_v1'
TRANSFER = HOLDOUT / 'external_transfer/20260914_v2'
FROZEN = HOLDOUT / 'method_snapshots/20260909_v1'
TIMES = [0, 5, 10, 15, 20, 25]
VARIANTS = ['baseline', 'rotate_minus30', 'rotate_plus30', 'gray', 'clahe_l', 'lower75', 'center75']


def load_bgr(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(path)
    return im


def transform(im, variant):
    h, w = im.shape[:2]
    matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    if variant.startswith('rotate'):
        angle = -30 if variant == 'rotate_minus30' else 30
        matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.)
        output = cv2.warpAffine(im, matrix, (w, h), borderValue=(114, 114, 114))
    elif variant == 'gray':
        output = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif variant == 'clahe_l':
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif variant in ['lower75', 'center75']:
        x = 0 if variant == 'lower75' else w // 8
        y = h // 4 if variant == 'lower75' else h // 8
        x1 = w if variant == 'lower75' else w - w // 8
        y1 = h if variant == 'lower75' else h - h // 8
        matrix[:, 2] = [-x, -y]
        output = im[y:y1, x:x1].copy()
    elif variant == 'baseline':
        output = im.copy()
    else:
        raise ValueError(variant)
    return output, matrix


def map_points(points, matrix):
    p = np.asarray(points, dtype=float)
    return np.c_[p, np.ones(len(p))] @ matrix.T


def check_transforms():
    im = np.zeros((1440, 1080, 3), np.uint8)
    p = np.array([[450., 750.], [600., 750.], [530., 900.]])
    for variant in VARIANTS:
        _, mat = transform(im, variant)
        np.testing.assert_allclose(map_points(map_points(p, mat), cv2.invertAffineTransform(mat)), p, atol=1e-9)


def main():
    from ultralytics import YOLO
    check_transforms()
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'frames').mkdir()
    review = read(HERE / '20260921_chain_v3/review12_stage_audit.csv').sort_values('review_id')
    members = read(TRANSFER / 'test_windows.csv').set_index('window_id')
    write_json(ROOT / 'protocol_before_probes.json', dict(
        role='retrospective_mechanistic_diagnosis_not_new_holdout',
        hypothesis=['pose_orientation_or_scale', 'pseudocolor_contrast', 'anatomical_ROI_error', 'extremum_phase_mismatch'],
        sample_times=TIMES, variants=VARIANTS, detector_imgsz=640, box_conf=.25, kpt_conf=.5,
        crop_bounds='lower75: full width, bottom75% height; center75: central75% width/height',
        rotation='fixed canvas; inverse native coordinates; no RR from transformed pixels',
        truth_times_used_for_detection=False, original_models_modified=False,
        selection='same12 existing Jiufu review windows, fixed6 timepoints',
        interpretation='detections are predictions, not verified anatomical labels',
        weights_sha256=sha256(FROZEN / 'weights/YOLO11n-best.pt'),
        script_sha256=sha256(Path(__file__))))
    cv2.setNumThreads(1)
    detector = YOLO(str(FROZEN / 'weights/YOLO11n-best.pt'))
    manifest, probes, points = [], [], []
    started = time.monotonic()
    for row in review.itertuples():
        folder = TRANSFER / 'windows' / row.window_id
        mapping_path = folder / 'map.csv'
        frames = {}
        if mapping_path.exists():
            mapping = read(mapping_path)
            for target in TIMES:
                ix = int(np.argmin(abs(mapping.target_time_seconds.to_numpy()-target)))
                p = folder / 'frames' / f'frame_{ix:06d}.jpg'
                frames[target] = (load_bgr(p), float(mapping.iloc[ix].source_time_seconds), str(p), sha256(p))
        else:
            ts = read(folder / 'timestamps.csv').relative_seconds.to_numpy(float)
            desired = {target: int(np.argmin(abs(ts-target))) for target in TIMES}
            cap = cv2.VideoCapture(str(members.loc[row.window_id, 'source_path']))
            assert cap.isOpened()
            for ix in range(max(desired.values())+1):
                ok, im = cap.read()
                if not ok:
                    raise ValueError('Source decode failed')
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                assert abs(pts-ts[ix]) < 1e-6
                for target, wanted in desired.items():
                    if wanted == ix:
                        frames[target] = (im.copy(), float(ts[ix]), str(members.loc[row.window_id, 'source_path']), str(members.loc[row.window_id, 'source_sha256']))
            cap.release()
        for target, (im, actual, source, source_hash) in frames.items():
            frame_id = f'{row.review_id}_t{target:02d}'
            image_path = ROOT / 'frames' / f'{frame_id}.png'
            ok, buf = cv2.imencode('.png', im); assert ok
            buf.tofile(str(image_path))
            manifest.append(dict(frame_id=frame_id, review_id=row.review_id, window_id=row.window_id,
                target_seconds=target, source_seconds=actual, width=im.shape[1], height=im.shape[0],
                image_path=str(image_path), image_sha256=sha256(image_path), source=source, source_sha256=source_hash))
            for variant in VARIANTS:
                transformed, matrix = transform(im, variant)
                tic = time.monotonic()
                res = detector(transformed, imgsz=640, conf=.25, verbose=False)[0]
                elapsed = time.monotonic()-tic
                best = int(res.boxes.conf.argmax()) if len(res.boxes) else None
                record = dict(frame_id=frame_id, review_id=row.review_id, window_id=row.window_id,
                    variant=variant, detections=len(res.boxes), high_keypoints=0, elapsed_seconds=elapsed)
                if best is not None:
                    kp = res.keypoints.data[best].cpu().numpy()
                    xy = map_points(kp[:, :2], cv2.invertAffineTransform(matrix))
                    inside = (xy[:, 0]>=0)&(xy[:, 0]<im.shape[1])&(xy[:, 1]>=0)&(xy[:, 1]<im.shape[0])
                    record.update(box_conf=float(res.boxes.conf[best]), high_keypoints=int(((kp[:, 2]>=.5)&inside).sum()))
                    for k, (pos, confidence) in enumerate(zip(xy, kp[:, 2])):
                        points.append(dict(frame_id=frame_id, variant=variant, keypoint=k,
                            x=float(pos[0]), y=float(pos[1]), confidence=float(confidence), inside_native=bool(inside[k])))
                probes.append(record)
        print(row.review_id, 'done', round(time.monotonic()-started, 1), 's', flush=True)
    write_csv(ROOT / 'frame_manifest.csv', manifest)
    write_csv(ROOT / 'pose_probes.csv', probes)
    write_csv(ROOT / 'pose_keypoints.csv', points)
    summary = pd.DataFrame(probes).groupby('variant').agg(frames=('frame_id','size'),
        detected_frames=('detections',lambda s: int((s>0).sum())), high_keypoints=('high_keypoints','sum')).reset_index()
    write_csv(ROOT / 'pose_probe_summary.csv', summary)
    write_json(ROOT / 'probe_verification.json', dict(status='COMPLETED_NOT_ANATOMICAL_VALIDATION',
        frames=len(manifest), inferences=len(probes), affine_roundtrip_pass=True,
        elapsed_seconds=time.monotonic()-started,
        files={n:sha256(ROOT/n) for n in ['frame_manifest.csv','pose_probes.csv','pose_keypoints.csv']}))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
