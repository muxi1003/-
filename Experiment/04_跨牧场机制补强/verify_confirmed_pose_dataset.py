"""Read back generated images, labels and grouped split membership."""
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from analyze_transfer import read, write_json, sha256
from build_confirmed_pose_dataset import DEST
from strict_pose_label_converter import convert_pose


def main():
    manifest = read(DEST/'manifest.csv')
    included = manifest[manifest.status.eq('included')]
    points = 0
    for r in included.itertuples():
        assert sha256(Path(r.json_path)) == r.json_sha256
        data = json.loads(Path(r.json_path).read_text(encoding='utf-8-sig'))
        source_bytes = base64.b64decode(data['imageData']) if data.get('imageData') else Path(r.image_source_path).read_bytes()
        source = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        generated = cv2.imdecode(np.fromfile(r.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(source, generated), r.frame_id
        actual = np.array([list(map(float,line.split())) for line in Path(r.label_path).read_text().splitlines()]).reshape(-1,11)
        expected = np.array(convert_pose(data)).reshape(-1,11)
        assert actual.shape == expected.shape and np.allclose(actual, expected, atol=5.1e-9, rtol=0)
        points += int((actual[:,[7,10]] == 2).sum())
        assert sha256(Path(r.image_path)) == r.image_sha256
        assert sha256(Path(r.label_path)) == r.label_sha256
    expected_ids = set(included.frame_id)
    for fold in range(5):
        train = {Path(p).stem for p in (DEST/f'fold{fold}_train.txt').read_text(encoding='utf-8').splitlines()}
        val = {Path(p).stem for p in (DEST/f'fold{fold}_val.txt').read_text(encoding='utf-8').splitlines()}
        assert not train & val and train | val == expected_ids
        assert not (set(included[included.frame_id.isin(train)].group) & set(included[included.frame_id.isin(val)].group))
        assert val == set(included[included.fold.eq(fold)].frame_id)
    for r in manifest[manifest.status.eq('quarantine')].itertuples():
        assert not (DEST/'images'/f'{r.frame_id}.png').exists()
        assert not (DEST/'labels'/f'{r.frame_id}.txt').exists()
    result = dict(status='PASS', verified_images=len(included), pixel_exact_to_selected_source=True,
                  verified_labels=len(included), visible_points=points, grouped_folds=5,
                  quarantine_excluded=int(manifest.status.eq('quarantine').sum()),
                  scope='File conversion and partition integrity, not anatomical accuracy or detector performance',
                  manifest_sha256=sha256(DEST/'manifest.csv'))
    write_json(DEST/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
