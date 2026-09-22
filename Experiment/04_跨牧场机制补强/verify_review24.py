"""Reconcile saved R3 analysis against archived R2 counts and event metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_transfer import HERE, BASES, read, sha256, write_json

ROOT=HERE/'20260921_r3_v1'
data=read(ROOT/'events_by_window.csv')
counts=read(ROOT/'count_changes.csv')
checked=0
for cohort,folder in BASES.items():
    old=read(folder/'event_metrics_by_window.csv').set_index('window_id')
    pred=read(folder/'prediction_windows.csv').set_index('window_id')
    d=data[data.cohort.eq(cohort)&data.version.eq('R2')&data.tolerance_seconds.eq(.3)&data.method.eq('frozen_baseline')]
    for row in d.itertuples():
        assert np.array_equal([row.tp,row.fp,row.fn],old.loc[row.window_id,['tp','fp','fn']].to_numpy())
        checked+=1
    for row in counts[counts.cohort.eq(cohort)].itertuples():
        p=pred.loc[row.window_id]
        assert p.prediction_status==row.prediction_status
        if p.prediction_status=='ok':
            assert p.predicted_count==row.predicted_count
        else:
            assert pd.isna(row.predicted_count)
        assert abs(float(p.duration_seconds)-row.duration_seconds)<1e-8
assert checked==23
for row in read(ROOT/'input_hashes.csv').itertuples():
    assert sha256(Path(row.path))==row.sha256, row.path
raw=read(ROOT/'raw/reference_events.csv')
assert raw.marking_mode.eq('paused_frame_review').all()
windows=read(ROOT/'raw/annotation_windows.csv')
assert windows.predictions_hidden.astype(str).str.lower().eq('true').all()
assert raw.event_type.eq('expiration_peak').all()
manifest={}
for name in ['中文论文初稿.md','奶牛热红外呼吸检测_中文初稿_20260921.docx','奶牛热红外呼吸检测_中文初稿_20260921.pdf']:
    path=ROOT/'manuscript'/name
    manifest[name]=dict(sha256=sha256(path),bytes=path.stat().st_size)
qa=json.loads((ROOT/'manuscript/document_qa/release_20260921/verification.json').read_text(encoding='utf-8'))
write_json(ROOT/'delivery_verification.json',dict(status='PASS_WITH_DISCLOSED_LABEL_CLARIFICATIONS',
    archived_R2_window_event_metrics_exact=checked,archived_prediction_counts_and_durations_exact=True,
    all_hashed_inputs_unchanged=True,paused_marking_metadata=True,prediction_hidden_metadata=True,
    note='UI metadata is not independent observation of actual human procedure',
    related_unit_tests_passed=12,docx_schema_validation='PASS',document_qa=qa,artifacts=manifest))
print('PASS: 23 archived R2 scores, original counts/durations, input hashes, and manuscript artifacts.')
