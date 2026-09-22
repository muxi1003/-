"""Read-only consistency audit of legacy labels and their embedded images."""
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from analyze_transfer import HERE, read, write_csv, write_json, sha256
from probe_pose_mechanism import load_bgr

PROJECT = HERE.parents[1]
SOURCE = PROJECT / 'Dataset_new/label_labelme_手工标注'
YOLO = PROJECT / 'Dataset_new/labels1_手工标注'
ROOT = HERE / '20260922_pose_phase_v1'
OUT = ROOT / 'legacy_annotation_audit_v1'


def read_shapes(data):
    groups = {}
    issues = []
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in ['nose', 'left_nostril', 'right_nostril']:
            issues.append('unknown_label:' + str(label)); continue
        group = str(shape.get('group_id'))
        slot = groups.setdefault(group, {})
        if label in slot:
            issues.append('duplicate_group_label:' + label); continue
        points = np.asarray(shape.get('points', []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            issues.append('invalid_points:' + label); continue
        if label == 'nose':
            if len(points) < 2:
                issues.append('invalid_box'); continue
            slot[label] = [*points.min(axis=0), *points.max(axis=0)]
        elif len(points) == 1:
            slot[label] = points[0].tolist()
        else:
            issues.append('nonpoint_nostril:' + label)
    return groups, issues


def parse_yolo(text, width, height):
    result = []
    for line in text.splitlines():
        if not line.strip(): continue
        v = list(map(float, line.split()))
        if len(v) != 11 or v[0] != 0 or not np.isfinite(v).all():
            raise ValueError('expected class0 and 11 finite values')
        if any(x < 0 or x > 1 for x in [*v[1:5], v[5], v[6], v[8], v[9]]):
            raise ValueError('normalized coordinates outside [0,1]')
        if any(x not in [0, 1, 2] for x in [v[7], v[10]]):
            raise ValueError('invalid keypoint visibility')
        cx, cy, w, h = np.array(v[1:5])*[width,height,width,height]
        result.append(dict(nose=[cx-w/2,cy-h/2,cx+w/2,cy+h/2],
            left_nostril=[v[5]*width,v[6]*height], right_nostril=[v[8]*width,v[9]*height],
            left_visibility=v[7],right_visibility=v[10]))
    return result


def pixel_sha(image):
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    paths = sorted(SOURCE.glob('*.json'))
    package = json.loads((ROOT / 'annotation_package.json').read_text(encoding='utf-8'))
    review_hashes = {pixel_sha(load_bgr(ROOT / 'frames' / (f['frame_id']+'.png'))): f['frame_id']
                     for f in package['frames']}
    rows = []; points = []; labels = Counter(); shape_types = Counter(); errors = []
    for number, path in enumerate(paths, 1):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        groups, issues = read_shapes(data)
        w, h = data.get('imageWidth'), data.get('imageHeight')
        row = dict(frame_id=path.stem, video_group=path.stem.rsplit('_frame_',1)[0],
            json_path=str(path), json_sha256=sha256(path), declared_width=w, declared_height=h,
            image_path=data.get('imagePath'), has_embedded_image=bool(data.get('imageData')),
            group_count=len(groups), shape_count=len(data.get('shapes', [])),
            scored_shapes=sum(s.get('score') is not None for s in data.get('shapes', [])),
            image_decoded=False, image_size_matches=False, pixel_sha256='',
            exact_review_pixel_overlap='', label_issues='|'.join(issues))
        for shape in data.get('shapes', []):
            labels[str(shape.get('label'))] += 1
            shape_types[str(shape.get('shape_type'))] += 1
        if row['has_embedded_image']:
            payload = base64.b64decode(data['imageData'], validate=True)
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                row['image_decoded'] = True
                row['image_size_matches'] = image.shape[:2] == (h,w)
                row['pixel_sha256'] = pixel_sha(image)
                row['exact_review_pixel_overlap'] = review_hashes.get(row['pixel_sha256'], '')
        for gid, group in groups.items():
            for label in ['left_nostril','right_nostril']:
                if label not in group: continue
                x,y = group[label]
                points.append(dict(frame_id=path.stem, video_group=row['video_group'], group_id=gid,
                    label=label, x=x,y=y, in_image=0<=x<w and 0<=y<h,
                    has_nose_box='nose' in group,
                    inside_nose_box=('nose' in group and group['nose'][0]<=x<=group['nose'][2]
                                     and group['nose'][1]<=y<=group['nose'][3])))
        txt = YOLO / (path.stem+'.txt')
        row['yolo_exists'] = txt.exists(); row['yolo_parse_ok'] = False
        row['yolo_instances'] = None; row['yolo_sha256'] = ''
        row['max_point_delta_px'] = None; row['max_box_delta_px'] = None
        row['same_instance_count'] = False
        if txt.exists():
            row['yolo_sha256'] = sha256(txt)
            try:
                yolo = parse_yolo(txt.read_text(encoding='utf-8-sig'), w, h)
                row['yolo_parse_ok'] = True; row['yolo_instances'] = len(yolo)
                row['same_instance_count'] = len(yolo) == len(groups)
                # Only compare unambiguous single-instance files; do not guess associations.
                if len(groups) == len(yolo) == 1:
                    group = next(iter(groups.values())); yy = yolo[0]
                    dd = [float(np.linalg.norm(np.array(group[label])-yy[label]))
                          for label in ['left_nostril','right_nostril']
                          if label in group and yy[label.split('_')[0]+'_visibility'] > 0]
                    row['max_point_delta_px'] = max(dd) if dd else None
                    row['max_box_delta_px'] = (float(np.max(np.abs(np.array(group['nose'])-yy['nose'])))
                                               if 'nose' in group else None)
            except (ValueError, TypeError) as exc:
                errors.append(dict(frame_id=path.stem, reason=str(exc)))
        rows.append(row)
        if number % 500 == 0: print(f'{number}/{len(paths)} read', flush=True)
    table = pd.DataFrame(rows); pp = pd.DataFrame(points)
    valid = table[table.pixel_sha256.ne('')]
    duplicates = valid[valid.pixel_sha256.duplicated(keep=False)].sort_values('pixel_sha256')
    per_video = table.groupby('video_group').agg(frames=('frame_id','size'),
        decoded=('image_decoded','sum'), max_coordinate_difference=('max_point_delta_px','max')).reset_index()
    summary = dict(json_files=len(table), video_groups=table.video_group.nunique(),
        embedded_images=int(table.has_embedded_image.sum()), decoded_images=int(table.image_decoded.sum()),
        size_matches=int(table.image_size_matches.sum()), unique_pixel_images=int(valid.pixel_sha256.nunique()),
        exact_duplicate_image_rows=len(duplicates), review_exact_pixel_overlap=int(table.exact_review_pixel_overlap.ne('').sum()),
        shapes=dict(labels), shape_types=dict(shape_types), files_with_scores=int(table.scored_shapes.gt(0).sum()),
        files_with_parse_issues=int(table.label_issues.ne('').sum()),
        points=len(pp), points_outside_image=int((~pp.in_image).sum()), points_outside_nose_box=int((~pp.inside_nose_box).sum()),
        yolo_files_present=int(table.yolo_exists.sum()), yolo_parse_ok=int(table.yolo_parse_ok.sum()),
        single_instance_point_comparisons=int(table.max_point_delta_px.notna().sum()),
        point_delta_gt_1px=int(table.max_point_delta_px.gt(1).sum()),
        point_delta_gt_5px=int(table.max_point_delta_px.gt(5).sum()),
        median_max_point_delta_px=float(table.max_point_delta_px.median()),
        max_point_delta_px=float(table.max_point_delta_px.max()),
        files_unmatched_by_basename=[p.name for p in YOLO.glob('*.txt') if p.stem not in set(table.frame_id)],
        training_eligibility='UNRESOLVED_LABEL_VERSION_AND_PROVENANCE',
        no_training_run=True, no_original_files_changed=True,
        limitation='exact decoded pixel duplicates only; not near-duplicate or full271 nonoverlap proof; score field does not disprove human review')
    write_csv(OUT/'file_audit.csv', rows); write_csv(OUT/'point_audit.csv', points)
    write_csv(OUT/'per_video.csv', per_video); write_csv(OUT/'exact_duplicate_images.csv', duplicates)
    write_csv(OUT/'parse_errors.csv', errors, columns=['frame_id','reason'])
    write_json(OUT/'summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
