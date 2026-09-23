"""Dry-run strict conversion; retain conflicts without generating training labels."""
import json
import argparse
from collections import Counter
from audit_legacy_localization_data import SOURCE, OUT
from strict_pose_label_converter import convert_pose, AnnotationConflict
from analyze_transfer import write_csv, write_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default='v2')
    tag=parser.parse_args().tag
    rows=[]
    for path in sorted(SOURCE.glob('*.json')):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        row=dict(frame_id=path.stem, status='valid_geometry_only', reason='', preserved_points=0)
        try:
            converted=convert_pose(data)
            row['preserved_points']=sum(v[7]>0 for v in converted)+sum(v[10]>0 for v in converted)
            row['instances']=len(converted)
        except AnnotationConflict as exc:
            row.update(status='quarantined_geometry_conflict',reason=str(exc))
        rows.append(row)
    write_csv(OUT/f'strict_conversion_dry_run_{tag}.csv',rows)
    summary=dict(files=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        conflict_reasons=dict(Counter(r['reason'] for r in rows if r['reason'])),
        training_labels_generated=False, original_annotations_changed=False,
        provenance_still_unconfirmed=True, geometry_validation_is_not_anatomical_truth_validation=True)
    write_json(OUT/f'strict_conversion_summary_{tag}.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
