"""Trace current local training labels without claiming historical checkpoint lineage."""
import base64
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from analyze_transfer import read, write_csv, write_json, sha256
from audit_legacy_localization_data import PROJECT, SOURCE, YOLO, OUT as AUDIT, read_shapes, parse_yolo
from probe_pose_mechanism import load_bgr

DATA = PROJECT / 'Dataset_new/72video'
OUT = AUDIT / 'current_layout'


def replay_batch(data):
    """Mirror existing converter's integer rounding and strict in-box selection."""
    lines = []; dropped = []
    width, height = data['imageWidth'], data['imageHeight']
    for shape in data['shapes']:
        if shape['shape_type'] != 'rectangle': continue
        p = np.array(shape['points'], float)
        x0,y0 = p.min(axis=0).astype(int); x1,y1 = p.max(axis=0).astype(int)
        v = [0, int((x0+x1)/2)/width, int((y0+y1)/2)/height, (x1-x0)/width, (y1-y0)/height]
        candidates = {}
        for point in data['shapes']:
            if point['shape_type'] != 'point': continue
            x,y = [int(a) for a in point['points'][0]]
            if x0 < x < x1 and y0 < y < y1:
                candidates[point['label']] = (x,y)
            else:
                dropped.append(dict(label=point['label'], x=x,y=y,
                    on_int_box_border=x in [x0,x1] or y in [y0,y1],
                    outside_int_box=not(x0<=x<=x1 and y0<=y<=y1)))
        for label in ['left_nostril','right_nostril']:
            if label in candidates:
                x,y = candidates[label]; v += [x/width,y/height,2]
            else: v += [0,0,0]
        lines.append(np.array([float(f'{a:.5f}') for a in v]))
    return np.array(lines).reshape(-1,11), dropped


def numeric_labels(path):
    return np.array([[float(a) for a in l.split()] for l in path.read_text(encoding='utf-8-sig').splitlines()
                     if l.strip()]).reshape(-1,11)


def main():
    OUT.mkdir(exist_ok=False)
    inventory = []
    lookup = {}
    for split in ['train','val']:
        for image in sorted((DATA/'images'/split).iterdir()):
            if image.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp']: continue
            fid = image.stem; group, ordinal = fid.rsplit('_frame_',1)
            label = DATA/'labels'/split/(fid+'.txt')
            inventory.append(dict(frame_id=fid, video_group=group, frame_index=int(ordinal),
                split=split,image_path=str(image),label_path=str(label),label_exists=label.exists(),
                manual_json_exists=(SOURCE/(fid+'.json')).exists()))
            lookup.setdefault(fid,[]).append((image,label,split))
    inv = pd.DataFrame(inventory)
    train_groups = set(inv[inv.split.eq('train')].video_group)
    val_groups = set(inv[inv.split.eq('val')].video_group)
    nearest = []
    for group,g in inv.groupby('video_group'):
        tr = g[g.split.eq('train')].frame_index.to_numpy(int)
        for row in g[g.split.eq('val')].itertuples():
            nearest.append(dict(frame_id=row.frame_id,video_group=group,
                nearest_train_frame_distance=int(np.min(abs(tr-row.frame_index))) if len(tr) else None))
    rows = []; losses = []
    files = sorted(SOURCE.glob('*.json'))
    for index,path in enumerate(files,1):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        converted,dropped = replay_batch(data)
        row = dict(frame_id=path.stem, locations=len(lookup.get(path.stem,[])),
            json_sha256=sha256(path), replay_missing_points=len(dropped))
        if row['locations'] == 1:
            image,label,split = lookup[path.stem][0]
            current = numeric_labels(label)
            legacy = numeric_labels(YOLO/(path.stem+'.txt'))
            row.update(split=split,image_path=str(image),label_path=str(label),
                label_sha256=sha256(label),image_sha256=sha256(image),
                current_matches_json_replay=current.shape==converted.shape and np.allclose(current,converted,atol=1e-7,rtol=0),
                current_matches_legacy_txt=current.shape==legacy.shape and np.allclose(current,legacy,atol=1e-7,rtol=0))
            if data.get('imageData'):
                embedded=cv2.imdecode(np.frombuffer(base64.b64decode(data['imageData']),np.uint8),cv2.IMREAD_COLOR)
                external=load_bgr(image)
                row['image_dimensions_match']=embedded.shape==external.shape
                if embedded.shape==external.shape:
                    row['embedded_vs_current_pixel_MAE']=float(np.abs(embedded.astype(np.float32)-external).mean())
                    a=cv2.resize(embedded,(108,144)).astype(float).reshape(-1)
                    b=cv2.resize(external,(108,144)).astype(float).reshape(-1)
                    row['embedded_vs_current_thumbnail_correlation']=float(np.corrcoef(a,b)[0,1])
            for d in dropped:
                losses.append(dict(frame_id=path.stem,split=split,current_matches_json_replay=row['current_matches_json_replay'],**d))
        rows.append(row)
        if index%500==0: print(f'{index}/{len(files)} current labels and images compared',flush=True)
    d = pd.DataFrame(rows); nn = pd.DataFrame(nearest)
    summary = dict(current_train_frames=int(inv.split.eq('train').sum()), current_val_frames=int(inv.split.eq('val').sum()),
        train_video_groups=len(train_groups),val_video_groups=len(val_groups),shared_video_groups=len(train_groups&val_groups),
        validation_frames_with_training_video=int(inv[inv.split.eq('val')].video_group.isin(train_groups).sum()),
        validation_frames_with_train_neighbor_le1=int(nn.nearest_train_frame_distance.le(1).sum()),
        validation_frames_with_train_neighbor_le3=int(nn.nearest_train_frame_distance.le(3).sum()),
        manual_json_single_current_match=int(d.locations.eq(1).sum()),
        labels_match_json_converter=int(d.current_matches_json_replay.fillna(False).sum()),
        labels_match_legacy_txt=int(d.current_matches_legacy_txt.fillna(False).sum()),
        replay_point_drops=len(losses), drop_files=len(set(r['frame_id'] for r in losses)),
        embedded_current_image_compared=int(d.embedded_vs_current_pixel_MAE.notna().sum()),
        median_image_pixel_MAE=float(d.embedded_vs_current_pixel_MAE.median()),
        max_image_pixel_MAE=float(d.embedded_vs_current_pixel_MAE.max()),
        thumbnail_corr_min=float(d.embedded_vs_current_thumbnail_correlation.min()),
        interpretation='current local layout only; historical Ubuntu training image manifest is absent; no checkpoint lineage claim',
        current_layout_not_independent_video_validation=True, no_original_files_changed=True,
        no_training_or_RR_changes=True)
    write_csv(OUT/'inventory.csv', inv); write_csv(OUT/'manual_to_current.csv', rows)
    write_csv(OUT/'conversion_point_drops.csv', losses,
              columns=['frame_id','split','current_matches_json_replay','label','x','y','on_int_box_border','outside_int_box'])
    write_csv(OUT/'validation_training_neighbors.csv', nearest)
    write_json(OUT/'summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
