"""Archive R2 human exports and evaluate pre-existing predictions without tuning."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiment_common import *

sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(HOLDOUT))
import reference_tools as ref
from run_frozen_external import verify_lock, verify_checkpoint
from score_external_counts import clean, cluster_intervals


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_export(csv, records):
    other = pd.DataFrame(records, columns=csv.columns).fillna('').astype(str)
    require(csv.shape == other.shape, 'Backup/CSV dimensions differ')
    for column in csv:
        if column.endswith('_seconds'):
            a = pd.to_numeric(csv[column], errors='coerce').to_numpy(float)
            b = pd.to_numeric(other[column], errors='coerce').to_numpy(float)
            require(np.allclose(a, b, atol=1e-10, rtol=0, equal_nan=True), f'Backup differs: {column}')
            require(csv[column].eq('').equals(other[column].eq('')), 'Blank time differs')
        else:
            require(csv[column].equals(other[column]), f'Backup differs: {column}')


def evaluate(annotations, windows, events, out, groups):
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'prediction_windows.csv', windows)
    write_csv(out / 'prediction_events.csv', events,
              columns=['window_id', 'event_id', 'event_time_seconds'])
    ref.score(annotations, out / 'prediction_events.csv', out / 'prediction_windows.csv', out, 'R2')
    reference = read_csv(annotations / 'annotation_windows.csv')
    paired = reference[reference.annotation_status.eq('complete')].merge(windows, on='window_id', suffixes=('', '_prediction'), validate='one_to_one')
    paired = paired[paired.prediction_status.eq('ok')].copy()
    paired['truth_count'] = paired.manual_breath_count.astype(float)
    paired['predicted_count'] = paired.predicted_count.astype(float)
    paired['truth_rr_bpm'] = 60 * paired.truth_count / paired.duration_seconds.astype(float)
    paired['predicted_rr_bpm'] = 60 * paired.predicted_count / paired.duration_seconds.astype(float)
    paired['verified_cow_id'] = paired.window_id.map(groups)
    require(paired.verified_cow_id.notna().all(), 'Missing bootstrap group')
    metrics = clean(measurements(paired)) if len(paired) else None
    write_csv(out / 'paired_count_results.csv', paired)
    write_json(out / 'count_metrics.json', {'reference_round': 'R2', 'metrics': metrics,
               'policy': 'complete_reference_AND_ok_only; abstentions_not_zero; no_phase_or_tolerance_fitting'})
    ev = read_csv(out / 'event_metrics_by_window.csv')
    ev['cluster'] = ev.window_id.map(groups)
    arrays = [g[['tp', 'fp', 'fn']].astype(int).sum().to_numpy() for _, g in ev.groupby('cluster')]
    rng = np.random.default_rng(20260917)
    values = []
    for _ in range(2000):
        tp, fp, fn = np.asarray(arrays)[rng.integers(0, len(arrays), len(arrays))].sum(axis=0)
        if 2 * tp + fp + fn:
            values.append(2 * tp / (2 * tp + fp + fn))
    write_json(out / 'cluster_uncertainty.json', clean({
        'group_basis': 'verified_cow_for_jiufu; raw_source_video_for_lindian_not_verified_cow',
        'event_f1': {'clusters': len(arrays), 'draws': 2000, 'seed': 20260917,
                     'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975))} if values else None,
        'counts': cluster_intervals(paired, draws=2000, seed=20260917)}))
    event_metrics = json.loads((out / 'event_metrics.json').read_text(encoding='utf-8'))
    return {**{k: event_metrics[k] for k in ['complete_reference_windows', 'algorithm_output_coverage_on_complete', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']}, **(metrics or {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', default='20260917_r2_v1')
    parser.add_argument('--downloads', type=Path, default=Path('C:/Users/muxi/Downloads'))
    args = parser.parse_args()
    submission = REFERENCE / 'submissions' / args.id
    internal = ABLATION / 'reference_updates' / args.id
    external = HOLDOUT / 'event_evaluation' / args.id
    delivery = ROOT / ('delivery_' + args.id)
    require(not any(p.exists() for p in [submission, internal, external, delivery]), 'Output already exists')
    raw = submission / 'raw'
    raw.mkdir(parents=True)
    sources = {'annotation_windows.csv': 'annotation_windows (1).csv', 'reference_events.csv': 'reference_events.csv',
               'unobservable_intervals.csv': 'unobservable_intervals.csv', 'backup.json': 'event_reference_R2_backup.json'}
    provenance = []
    for target, name in sources.items():
        src = args.downloads / name
        shutil.copyfile(src, raw / target)
        provenance.append({'source': str(src), 'snapshot': str(raw / target), 'sha256': sha256(src)})
    shutil.copyfile(Path(__file__), submission / 'evaluator_snapshot.py')
    write_json(submission / 'analysis_policy_before_scoring.json', {
        'created_at': stamp(), 'round': 'R2', 'tolerance_seconds': .30,
        'primary': 'complete_only; all_abstained_reference_events_count_as_FN',
        'pending_user_clarification': '2026-09-17 user: 152 pending windows have notes explaining why not annotated; retain original status and notes',
        'no_tuning': True, 'no_phase_offset_fitting': True,
        'scope': 'new_reference_evaluation_of_existing_internal_ablation_and_presealed_external_predictions; not_new_independent_test',
        'source_exports': provenance})
    w, e, intervals = ref.load_reference(raw)
    backup = json.loads((raw / 'backup.json').read_text(encoding='utf-8-sig'))
    for table, key in [(w, 'windows'), (e, 'events'), (intervals, 'intervals')]:
        compare_export(table, backup[key])
    original = read_csv(REFERENCE / 'event_workspace/20260914_v3/annotation_windows.csv').set_index('window_id')
    require(set(w.window_id) == set(original.index) and len(w) == 320, 'Reference inventory changed')
    immutable = ['video_id', 'video_path', 'source_path', 'source_start_seconds', 'duration_seconds', 'cohort',
                 'annotation_round', 'event_definition', 'browser_video_path', 'view_start_seconds', 'prior_algorithm_exposure']
    for row in w.itertuples():
        for column in immutable:
            require(str(getattr(row, column)) == original.loc[row.window_id, column], f'Window metadata changed: {row.window_id}/{column}')
    errors = ref.validate_tables(w, e, intervals)
    write_json(submission / 'reference_validation.json', {'errors': errors, 'windows': len(w), 'events': len(e),
               'intervals': len(intervals), 'backup_csv_semantically_equal': True, 'window_metadata_unchanged': True})
    require(not errors, str(errors[:10]))
    review = w.copy()
    review['review_disposition'] = np.select([
        w.annotation_status.eq('pending') & w.reference_notes.str.strip().ne(''), w.annotation_status.eq('pending'),
        w.annotation_status.eq('complete')], ['excluded_with_user_reason', 'pending_without_reason', 'complete_event_reference'], default='excluded_' + w.annotation_status)
    write_csv(submission / 'window_dispositions.csv', review)
    write_csv(submission / 'coverage.csv', review.groupby(['cohort', 'annotation_status', 'review_disposition']).size().reset_index(name='n'))
    write_csv(submission / 'excluded_windows_with_notes.csv', review[review.annotation_status.ne('complete')])
    for cohort in ['lindian49', 'jiufu271']:
        subset = w[w.cohort.eq(cohort)]
        for name, table in [('annotation_windows.csv', subset), ('reference_events.csv', e[e.window_id.isin(subset.window_id)]),
                            ('unobservable_intervals.csv', intervals[intervals.window_id.isin(subset.window_id)])]:
            write_csv(submission / cohort / name, table)
    print('Archived and validated human exports', flush=True)

    # Align saved image-index peaks with decoded viewing-video timestamps, never nominal 8.7 fps.
    ablation = ABLATION / 'runs/20260909_v1'
    pred = read_csv(ablation / 'paired_predictions.csv')
    pred = pred[pred.cohort.eq('anchored49')]
    matched = read_csv(ABLATION / 'input_snapshots/20260909_v1/matched_windows.csv')
    matched = matched[matched.cohort.eq('anchored49')].set_index('video_id')
    view = read_csv(REFERENCE / 'event_workspace/browser_media_v1/viewing_copy_manifest.csv').set_index('video_id')
    adapter = HOLDOUT / 'adapter_validation/20260912_v2'
    lw = w[w.cohort.eq('lindian49')].set_index('video_id')
    times, bindings = {}, []
    cv2.setNumThreads(1)
    for vid, row in lw.iterrows():
        vm = view.loc[vid]
        require(sha256(Path(vm.source_path)) == vm.source_sha256 and sha256(Path(vm.browser_video_path)) == vm.browser_sha256, 'Viewing video changed')
        cap = cv2.VideoCapture(vm.browser_video_path, cv2.CAP_FFMPEG)
        ts = []
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        finally:
            cap.release()
        saved = read_csv(adapter / 'decoded_timestamps' / f'{vid}.csv').annotation_video_time_seconds.astype(float).to_numpy()
        table = read_csv(Path(matched.loc[vid, 'input_temperature_csv']))
        require(sha256(Path(matched.loc[vid, 'input_temperature_csv'])) == matched.loc[vid, 'input_sha256'], 'Temperature input changed')
        require(len(ts) == len(saved) == len(table) == int(vm.frames), 'Frame count differs')
        require(np.allclose(ts, saved, atol=.001, rtol=0) and np.all(np.diff(ts) > 0), 'PTS alignment differs')
        require(table.frame_name.tolist() == [f'frame_{i:06d}.jpg' for i in range(len(ts))], 'Nonsequential image binding')
        require(abs(float(matched.loc[vid, 'duration_seconds']) - float(row.duration_seconds)) < 1e-6, 'Duration differs')
        times[vid] = np.asarray(ts)
        bindings.extend({'video_id': vid, 'frame_index': i, 'annotation_video_time_seconds': t} for i, t in enumerate(ts))
    write_csv(submission / 'lindian_frame_time_bindings.csv', bindings)
    print('49 viewing-video time axes verified', flush=True)
    summaries = []
    for variant, group in pred.groupby('variant', sort=True):
        windows, events = [], []
        require(set(group.video_id) == set(lw.index), 'Ablation inventory differs')
        for row in group.itertuples():
            r = lw.loc[row.video_id]
            curve_path = ablation / 'curves/anchored49' / variant / f'{row.video_id}.csv'
            curve = read_csv(curve_path)
            indices = np.flatnonzero(curve.is_peak.map(is_true).to_numpy())
            declared = [int(x) for x in row.peak_frames_zero_based.split(';') if x]
            require(indices.tolist() == declared and len(indices) == int(row.predicted_count), 'Peak replay differs')
            require(len(curve) == len(times[row.video_id]), 'Curve/time length differs')
            require(curve.frame_name.tolist() == [f'frame_{i:06d}.jpg' for i in range(len(curve))], 'Curve frame binding differs')
            windows.append(dict(window_id=r.window_id, duration_seconds=r.duration_seconds, predicted_count=row.predicted_count,
                           prediction_status='ok', timebase_verified='true', timebase='annotation_video_seconds'))
            events.extend(dict(window_id=r.window_id, event_id=f'p{k}', event_time_seconds=times[row.video_id][k]) for k in indices)
        groups = {r.window_id: matched.loc[v, 'cluster_id'] for v, r in lw.iterrows()}
        result = evaluate(submission / 'lindian49', pd.DataFrame(windows), pd.DataFrame(events), internal / variant, groups)
        summaries.append(dict(cohort='lindian49', variant=variant, **result))
    print('Eight fixed internal ablations scored', flush=True)

    run = HOLDOUT / 'external_transfer/20260914_v2'
    seal = json.loads((run / 'prediction_seal.json').read_text(encoding='utf-8'))
    gate = json.loads((run / 'engineering_verification.json').read_text(encoding='utf-8'))
    require(gate['status'] == 'PASS_ENGINEERING_GATE_NOT_ACCURACY' and gate['prediction_seal_sha256'] == sha256(run / 'prediction_seal.json'), 'Engineering gate changed')
    lock, released = verify_lock(run)
    require(sha256(run / 'prediction_windows.csv') == seal['predictions_sha256'] and sha256(run / 'protocol_lock.json') == seal['protocol_sha256'], 'External seal changed')
    jp = read_csv(run / 'prediction_windows.csv').set_index('window_id')
    jw = w[w.cohort.eq('jiufu271')].set_index('window_id')
    require(set(jw.index) == set(jp.index) == set(released.window_id), 'External inventory differs')
    windows, events = [], []
    for n, row in enumerate(released.itertuples(), 1):
        wid = row.window_id
        r, prediction = jw.loc[wid], jp.loc[wid]
        require(Path(row.source_path) == Path(r.source_path) == Path(r.video_path), 'Source binding differs')
        require(float(row.start_seconds) == float(r.source_start_seconds) == float(r.view_start_seconds) == 0, 'Source start differs')
        require(abs(float(row.duration_seconds) - float(r.duration_seconds)) < 1e-6, 'External duration differs')
        directory = run / 'windows' / wid
        require(sha256(directory / 'checkpoint.json') == seal['checkpoints'][wid], 'Checkpoint seal changed')
        checkpoint = verify_checkpoint(directory)
        require(checkpoint['prediction_status'] == prediction.prediction_status, 'Status replay differs')
        require(checkpoint['predicted_count'] is None if prediction.prediction_status == 'abstain' else float(checkpoint['predicted_count']) == float(prediction.predicted_count), 'Count replay differs')
        windows.append(dict(window_id=wid, duration_seconds=r.duration_seconds, predicted_count=prediction.predicted_count,
                       prediction_status=prediction.prediction_status, timebase_verified='true', timebase='annotation_video_seconds'))
        if prediction.prediction_status == 'ok':
            peaks = read_csv(directory / 'algorithm_events.csv')
            mapping = read_csv(directory / 'map.csv')
            curve = read_csv(directory / 'curve.csv')
            indices = np.flatnonzero(curve.is_peak.map(is_true).to_numpy())
            require(indices.tolist() == peaks.grid_index.astype(int).tolist() and len(peaks) == int(float(prediction.predicted_count)), 'External peak replay differs')
            require(np.allclose(peaks.time_seconds.astype(float), mapping.iloc[indices].target_time_seconds.astype(float), atol=1e-9), 'External time map differs')
            events.extend(dict(window_id=wid, event_id=f'p{p.grid_index}', event_time_seconds=float(p.time_seconds)) for p in peaks.itertuples())
        if n % 50 == 0:
            print(f'External sealed artifacts verified {n}/271', flush=True)
    result = evaluate(submission / 'jiufu271', pd.DataFrame(windows), pd.DataFrame(events), external,
                      dict(zip(released.window_id, released.verified_cow_id)))
    summaries.append(dict(cohort='jiufu271', variant='frozen_transfer_20260914_v2', **result))
    write_json(external / 'prediction_provenance.json', {'prediction_seal_sha256': sha256(run / 'prediction_seal.json'),
               'all_271_checkpoints_verified': True, 'no_predictions_recomputed': True, 'no_parameters_updated': True,
               'timebase': 'presealed_target_grid_in_source_video_seconds_not_posthoc_fitted_offset'})
    write_csv(internal / 'metrics_summary.csv', pd.DataFrame(summaries[:-1]))
    write_csv(external / 'metrics_summary.csv', pd.DataFrame(summaries[-1:]))
    delivery.mkdir()
    write_csv(delivery / 'metrics_summary.csv', summaries)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    write_json(delivery / 'status.json', {'status': 'SCORED_SUBMITTED_COMPLETE_REFERENCES', 'submission': str(submission),
               'internal': str(internal), 'external': str(external), 'not_double_blind': True,
               'not_new_independent_cohort': True, 'absolute_temperature_unverified': True,
               'network_training_ablation_not_run': True, 'human_labels_changed': False,
               'pending_with_notes': int((review.review_disposition == 'excluded_with_user_reason').sum()),
               'pending_without_notes': int((review.review_disposition == 'pending_without_reason').sum())})
    artifact_rows = []
    for directory in [submission, internal, external, delivery]:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                artifact_rows.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
    write_csv(delivery / 'artifact_manifest.csv', artifact_rows)


if __name__ == '__main__':
    main()
