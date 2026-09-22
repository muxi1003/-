"""Requirement-level delivery verification without inventing unavailable human evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common import ROOT, ABLATION, HOLDOUT, REFERENCE, read_csv, sha256, write_json


def independent_metrics(table):
    n=table.truth_count.to_numpy(float);p=table.predicted_count.to_numpy(float)
    d=table.duration_seconds.to_numpy(float)
    if not np.isfinite(np.column_stack([n,p,d])).all() or np.any(d<=0):
        raise ValueError('Invalid count/duration')
    y=60*n/d;z=60*p/d;e=z-y;den=((y-y.mean())**2).sum();ce=abs(n-p)
    return dict(rr_r2=float(1-(e**2).sum()/den) if den>0 else None,
                rr_mae_bpm=float(abs(e).mean()),rr_rmse_bpm=float(np.sqrt((e**2).mean())),
                count_mae=float(ce.mean()),exact_count=int((ce==0).sum()),within_one_count=int((ce<=1).sum()),
                mean_count_accuracy_percent=float(100*np.maximum(0,1-ce[n>0]/n[n>0]).mean()) if np.any(n>0) else None)


def assert_metrics(actual, expected):
    for key,value in expected.items():
        if value is not None and abs(float(actual[key])-value)>1e-8:
            raise ValueError(f'Metric mismatch: {key}')


def main():
    out=ROOT/'delivery_20260914_v1'
    checks=[]
    index=read_csv(out/'文件用途与哈希.csv')
    for row in index.itertuples():
        p=Path(row.path)
        if not p.is_file() or sha256(p)!=row.sha256:
            raise ValueError(f'Delivery artifact changed: {p}')
    checks.append(f'all_{len(index)}_indexed_artifacts_exist_and_hash_match')
    original_hashes={
      '20260909_v1':'64b6522745dd7d212a501d4f5ef5f02c17fec943b96947e08957656ba735c214',
      'jiufu271_frozen_v1':'004f8ec22fafd28770782dceae1e26c71598a0efe1803f8aa17c99024b09b41a'}
    for folder,digest in original_hashes.items():
        if sha256(REFERENCE/'annotations'/folder/'annotation_windows.csv')!=digest:
            raise ValueError('Original human count file changed')
    checks.append('both_original_R1_count_files_byte_unchanged')
    ab=ABLATION/'runs/20260909_v1'
    pred=pd.read_csv(ab/'paired_predictions.csv',dtype={'video_id':str})
    if len(pred)!=976 or pred.duplicated(['cohort','video_id','variant']).any() or pred.variant.nunique()!=8:
        raise ValueError('Ablation matrix incomplete')
    for (cohort,variant),frame in pred.groupby(['cohort','variant']):
        if len(frame)!={'internal73':73,'anchored49':49}[cohort]:
            raise ValueError('Wrong per-arm cohort')
    metrics=pd.read_csv(ab/'metrics.csv')
    if len(metrics)!=24:
        raise ValueError('Ablation metric groups incomplete')
    for row in metrics.to_dict('records'):
        data=pred[pred.cohort.eq(row['cohort'])&pred.variant.eq(row['variant'])]
        if row['analysis_set']=='primary_completed':data=data[data.include_primary.astype(str).str.lower().eq('true')]
        if len(data)!=row['n']:raise ValueError('Metric n mismatch')
        assert_metrics(row,independent_metrics(data))
    checks.append('all976_ablation_rows_8_arms_24_metric_groups_independently_recomputed')
    for row in pred.itertuples():
        curve=pd.read_csv(ab/'curves'/row.cohort/row.variant/f'{row.video_id}.csv')
        if curve.is_peak.astype(str).str.lower().eq('true').sum()!=row.predicted_count:
            raise ValueError('Ablation curve peak count mismatch')
    checks.append('all976_ablation_curve_peak_counts_match_summary')
    run=HOLDOUT/'external_transfer/20260914_v2'
    plan=json.loads((run/'evaluation_plan_lock.json').read_text(encoding='utf-8'))
    for name,digest in plan['files'].items():
        if sha256(HOLDOUT/name)!=digest:raise ValueError('Pre-outcome evaluation script changed')
    result=json.loads((run/'count_evaluation/metrics.json').read_text(encoding='utf-8'))
    all_rows=read_csv(run/'count_evaluation/all_271_windows.csv')
    paired=pd.read_csv(run/'count_evaluation/paired_count_results.csv')
    seal=json.loads((run/'prediction_seal.json').read_text(encoding='utf-8'))
    if sha256(run/'prediction_windows.csv')!=seal['predictions_sha256'] or sha256(run/'prediction_seal.json')!=result['prediction_seal_sha256']:
        raise ValueError('Scored result does not bind to sealed predictions')
    gate=json.loads((run/'engineering_verification.json').read_text(encoding='utf-8'))
    if gate['code_sha256']!=plan['files']['verify_external_transfer.py'] or gate['prediction_seal_sha256']!=result['prediction_seal_sha256']:
        raise ValueError('Engineering gate code/seal binding mismatch')
    sealed_rows=read_csv(run/'prediction_windows.csv').set_index('window_id')
    reference_path=REFERENCE/'submissions/20260911_v2/jiufu271/count_reference_only.csv'
    if sha256(reference_path)!=result['reference_sha256']:
        raise ValueError('Scored reference changed')
    refs=read_csv(reference_path).set_index('window_id')
    if set(gate['window_ids'])!=set(sealed_rows.index):
        raise ValueError('Engineering gate missing released windows')
    for row in all_rows.itertuples():
        saved=sealed_rows.loc[row.window_id]; ref=refs.loc[row.window_id]
        if row.prediction_status!=saved.prediction_status or row.annotation_status!=ref.annotation_status:
            raise ValueError('Joined statuses differ from authoritative inputs')
        if row.predicted_count!=saved.predicted_count or row.manual_breath_count!=ref.manual_breath_count:
            raise ValueError('Joined counts differ from authoritative inputs')
        if abs(float(row.duration_seconds)-float(saved.duration_seconds))>1e-8 or abs(float(row.duration_seconds)-float(ref.duration_seconds))>1e-8:
            raise ValueError('Joined durations changed')
    for row in paired.itertuples():
        if float(row.predicted_count)!=float(sealed_rows.loc[row.window_id,'predicted_count']) or float(row.truth_count)!=float(refs.loc[row.window_id,'manual_breath_count']):
            raise ValueError('Paired analysis altered sealed or reference count')
    if len(all_rows)!=271 or all_rows.window_id.nunique()!=271 or len(paired)!=result['n_primary']:
        raise ValueError('External analysis membership incomplete')
    selected=all_rows.annotation_status.eq('complete')&all_rows.prediction_status.eq('ok')
    if set(all_rows.loc[selected,'window_id'])!=set(paired.window_id):raise ValueError('External analysis mask mismatch')
    if len(paired):assert_metrics(result['metrics'],independent_metrics(paired))
    if result['event_metrics'] is not None:raise ValueError('Unsupported event metric')
    checks.append('external271_membership_complete_intersection_and_metrics_independently_recomputed_no_event_F1')
    ui=REFERENCE/'event_workspace/20260914_v3'
    if len(read_csv(ui/'annotation_windows.csv'))!=320 or len(read_csv(ui/'reference_events.csv')) or len(read_csv(ui/'unobservable_intervals.csv')):
        raise ValueError('Expected count-free pending human event work')
    window_table=read_csv(ui/'annotation_windows.csv')
    if not window_table.manual_breath_count.eq('').all() or not window_table.annotation_status.eq('pending').all():
        raise ValueError('Unexpected generated human count/status')
    if not window_table.view_start_seconds.astype(float).eq(0).all():raise ValueError('Viewer origin wrong')
    check=json.loads((ui/'ui_validation.json').read_text(encoding='utf-8'))
    if check['status']!='PASS' or check['videoPixelRange']<10 or not check['eventJumpCorrectOrigin']:
        raise ValueError('Incomplete annotation UI test')
    media=read_csv(REFERENCE/'event_workspace/browser_media_v1/viewing_copy_manifest.csv')
    if len(media)!=49 or media.max_timestamp_delta_seconds.astype(float).max()>.001 or media.max_decoded_pixel_delta.astype(int).max()>1:
        raise ValueError('Viewing copy equivalence missing')
    checks.append('320_pending_event_windows_zero_fake_events_49_time_pixel_verified_viewing_copies_UI_passed')
    write_json(out/'completion_audit.json',dict(status='PASS_AUTOMATABLE_DELIVERY_WITH_EXPLICIT_HUMAN_DEPENDENCY',checks=checks,
      completed=['same_policy_signal_ablation','locked_external_count_test','event_entry_and_evaluation_tools','artifact_purpose_index'],
      human_required=['actual_reference_event_times','unobservable_intervals_and_honest_R2_blinding_declaration'],
      not_claimed=['completed_human_event_reference','event_F1','absolute_temperature_validation','same_split_network_training_ablation'],
      all_three_scientific_evidence_blocks_complete=False,validator_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(status='PASS',checks=checks),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
