"""Materialize an isolated pose dataset from user-confirmed final JSON labels."""
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_transfer import read, write_csv, write_json, sha256
from audit_legacy_localization_data import OUT
from strict_pose_label_converter import convert_pose, serialize_pose


DEST = OUT / 'confirmed_json_pose_v1'


def main():
    plan = read(OUT / 'grouped_rebuild_draft/partition_manifest.csv')
    audited = read(OUT / 'file_audit.csv').set_index('frame_id')
    DEST.mkdir(exist_ok=False)
    (DEST / 'images').mkdir()
    (DEST / 'labels').mkdir()
    records = []
    for r in plan.itertuples():
        source = Path(r.json_path)
        source_hash = sha256(source)
        assert source_hash == audited.loc[r.frame_id, 'json_sha256'], source
        if r.status == 'quarantine':
            records.append(dict(frame_id=r.frame_id, status='quarantine', reason=r.reason,
                                json_path=str(source), json_sha256=source_hash,
                                group=r.partition_group, fold=int(r.proposed_fold)))
            continue
        data = json.loads(source.read_text(encoding='utf-8-sig'))
        rows = convert_pose(data)
        if data.get('imageData'):
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data['imageData']), dtype=np.uint8), cv2.IMREAD_COLOR)
            image_source = 'json_embedded'
            image_source_path = str(source)
        else:
            image_source_path = r.current_image_path
            pixels = cv2.imdecode(np.fromfile(image_source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            image_source = 'external_basename_size_association'
        assert pixels is not None and pixels.shape[:2] == (data['imageHeight'], data['imageWidth'])
        image = DEST / 'images' / (r.frame_id + '.png')
        label = DEST / 'labels' / (r.frame_id + '.txt')
        ok, encoded = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        assert ok
        image.write_bytes(encoded.tobytes())
        label.write_text(serialize_pose(rows), encoding='ascii')
        replay = np.array([list(map(float, line.split())) for line in label.read_text().splitlines()])
        assert replay.shape == np.array(rows).shape and np.allclose(replay, rows, atol=5.1e-9, rtol=0)
        visible = sum(int(row[7] == 2) + int(row[10] == 2) for row in rows)
        records.append(dict(frame_id=r.frame_id, status='included', reason='', json_path=str(source),
                            json_sha256=source_hash, group=r.partition_group, fold=int(r.proposed_fold),
                            image_source=image_source, image_source_path=image_source_path,
                            image_source_sha256=sha256(Path(image_source_path)),
                            image_path=str(image), image_sha256=sha256(image),
                            label_path=str(label), label_sha256=sha256(label),
                            instances=len(rows), visible_points=visible))
    write_csv(DEST / 'manifest.csv', records)
    included = [r for r in records if r['status'] == 'included']
    for fold in range(5):
        train = [r for r in included if r['fold'] != fold]
        val = [r for r in included if r['fold'] == fold]
        assert not ({r['group'] for r in train} & {r['group'] for r in val})
        for split, members in [('train', train), ('val', val)]:
            (DEST / f'fold{fold}_{split}.txt').write_text(''.join(Path(r['image_path']).as_posix()+'\n' for r in members), encoding='utf-8')
        # JSON syntax is valid YAML and preserves Windows/Unicode paths safely.
        (DEST / f'fold{fold}.yaml').write_text(json.dumps(dict(
            path=DEST.as_posix(), train=f'fold{fold}_train.txt', val=f'fold{fold}_val.txt',
            names={0:'nose'}, kpt_shape=[2,3], flip_idx=[1,0]), ensure_ascii=False, indent=2), encoding='utf-8')
    write_json(DEST / 'protocol.json', dict(
        status='MATERIALIZED_NOT_TRAINED', confirmation_date='2026-09-23',
        label_source='X-AnyLabeling assisted JSON; user manually reviewed and confirmed final version',
        legacy_txt='User manually labeled separate version; preserved, not used or declared incorrect',
        included_frames=len(included), quarantined_frames=len(records)-len(included),
        visible_points=sum(r['visible_points'] for r in included),
        image_source_counts={k:sum(r['image_source']==k for r in included) for k in ['json_embedded','external_basename_size_association']},
        association_limit='External-only images match basename and size; correspondence is not independently established',
        grouping='Video plus equal numeric identifier after stripping letter prefix; not verified biological cow identity',
        evaluation_limit='Grouped internal development only; frozen pose initialization may have seen these groups. Viewed Jiufu remains retrospective.',
        training_started=False, default_model_or_rr_changed=False,
        draft_manifest_sha256=sha256(OUT/'grouped_rebuild_draft/partition_manifest.csv'),
        converter_sha256=sha256(Path(__file__).with_name('strict_pose_label_converter.py')),
        manifest_sha256=sha256(DEST/'manifest.csv')))
    print(json.dumps(dict(included=len(included), quarantined=len(records)-len(included),
                         visible_points=sum(r['visible_points'] for r in included)), indent=2))


if __name__ == '__main__':
    main()
