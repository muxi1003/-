"""Draft group partition only; no training labels or images are rewritten."""
import hashlib
import re
import pandas as pd
from analyze_transfer import read, write_csv, write_json, sha256
from audit_legacy_localization_data import OUT


def main():
    destination=OUT/'grouped_rebuild_draft'
    destination.mkdir(exist_ok=False)
    files=read(OUT/'file_audit.csv')
    layout=read(OUT/'current_layout/manual_to_current.csv').set_index('frame_id')
    geometry=read(OUT/'strict_conversion_dry_run_v2.csv').set_index('frame_id')
    duplicated=set(read(OUT/'exact_duplicate_images.csv').frame_id)
    def group_key(video):
        match=re.fullmatch(r'[A-Za-z]*(\d+)',video)
        return match[1] if match else video
    files['partition_group']=files.video_group.map(group_key)
    groups=sorted(set(files.partition_group),key=lambda k:hashlib.sha256(('legacy-draft-20260923:'+k).encode()).hexdigest())
    fold={g:i%5 for i,g in enumerate(groups)}
    rows=[]
    for r in files.itertuples():
        reasons=[]
        if geometry.loc[r.frame_id,'status']!='valid_geometry_only':reasons.append('point_without_nose_box')
        if r.frame_id in duplicated:reasons.append('duplicate_embedded_image_needs_pairing_review')
        rows.append(dict(frame_id=r.frame_id,video_group=r.video_group,partition_group=r.partition_group,
            proposed_fold=fold[r.partition_group],status='quarantine' if reasons else 'draft_provenance_pending',
            reason=';'.join(reasons),json_path=r.json_path,current_image_path=layout.loc[r.frame_id,'image_path'],
            current_label_path=layout.loc[r.frame_id,'label_path'],legacy_txt_version_unresolved=True))
    d=pd.DataFrame(rows)
    # Prefix aliases and all adjacent frames of a video stay together.
    assert d.groupby('video_group').proposed_fold.nunique().max()==1
    assert d.groupby('partition_group').proposed_fold.nunique().max()==1
    write_csv(destination/'partition_manifest.csv',rows)
    write_csv(destination/'fold_summary.csv',d.groupby(['proposed_fold','status']).size().reset_index(name='frames'))
    write_json(destination/'protocol.json',dict(
        status='DRAFT_NOT_A_TRAINING_DATASET',training_allowed=False,
        frames=len(d),partition_groups=len(groups),video_groups=d.video_group.nunique(),
        quarantined_frames=int(d.status.eq('quarantine').sum()),
        label_source_choice_pending_user_confirmation=True,
        grouping='same video plus equal numeric identifiers after stripping letter prefix; conservative aliases, not verified biological cow IDs',
        same_video_train_validation_overlap=0,split_count=5,
        frozen_weights_may_have_seen_all_source_groups=True,
        independent_validation_requires_training_provenance_and_untouched_test=True,
        no_original_images_labels_weights_or_RR_changed=True,
        source_hashes={n:sha256(OUT/n) for n in ['file_audit.csv','current_layout/manual_to_current.csv','strict_conversion_dry_run_v2.csv']}))
    print(d.groupby(['proposed_fold','status']).size().to_string())


if __name__=='__main__':main()
