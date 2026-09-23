"""Deterministic consistency checks, not independent scientific review."""
import json
from pathlib import Path
from analyze_transfer import read, write_json, sha256
from audit_legacy_localization_data import OUT


def main():
    files=read(OUT/'file_audit.csv')
    for r in files.itertuples():
        assert sha256(Path(r.json_path))==r.json_sha256
    current=read(OUT/'current_layout/manual_to_current.csv')
    for r in current.itertuples():
        assert sha256(Path(r.label_path))==r.label_sha256
    geometry=read(OUT/'strict_conversion_dry_run_v2.csv')
    assert len(geometry)==len(files)==2518
    assert geometry.preserved_points.sum()==4668
    rejected=geometry[geometry.status.eq('quarantined_geometry_conflict')]
    assert rejected.frame_id.tolist()==['bs177611_frame_000012']
    assert current.current_matches_json_replay.all()
    assert not current.current_matches_legacy_txt.any()
    draft=read(OUT/'grouped_rebuild_draft/partition_manifest.csv')
    assert set(draft.frame_id)==set(files.frame_id)
    assert draft.groupby('partition_group').proposed_fold.nunique().max()==1
    assert draft.groupby('video_group').proposed_fold.nunique().max()==1
    assert draft.status.eq('quarantine').sum()==3
    protocol=json.loads((OUT/'grouped_rebuild_draft/protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['source_hashes'].items(): assert sha256(OUT/name)==digest
    assert protocol['training_allowed'] is False
    write_json(OUT/'verification.json',dict(status='PASS',
        scope='file consistency and split constraints only; not human anatomical adjudication',
        source_json_hashes_rechecked=2518,current_training_label_hashes_rechecked=2518,
        strict_converter_preserves_points_on_valid_records=4668,
        two_orphan_points_quarantined_not_zeroed=True,
        same_video_and_numeric_group_cross_fold_overlap=0,
        draft_still_not_training_enabled=True,independent_reviewer_used=False))
    print('PASS: 2518 JSON/label hashes, 4668 preserved points, orphan quarantine and grouped draft verified.')


if __name__=='__main__':main()
