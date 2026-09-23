"""Frozen-peak counterfactual: retain peaks with observed center or one-frame bridge."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_transfer import BASES, HERE, HOLDOUT, SUBMISSION, match_events, metrics, read, sha256, write_csv, write_json
from audit_frozen_peak_evidence import contributing_sides


TRANSFER = HOLDOUT / 'external_transfer/20260914_v2'
OUT = HERE / '20260922_pose_phase_v1/center_peak_support_v1'


def directly_observed(temp, side):
    return temp[f'{side}_source'] == 'detected' and np.isfinite(temp[f'{side}_temp'])


def has_center_support(curve, temperature, index):
    """Use only nostrils contributing to the saved fused signal at this peak."""
    mode = str(curve.iloc[index].selected_fusion_mode)
    sides = contributing_sides(curve.iloc[index], mode)
    at_peak = [side for side in sides if directly_observed(temperature.iloc[index], side)]
    bridged = [side for side in sides if 0 < index < len(temperature) - 1
               and directly_observed(temperature.iloc[index - 1], side)
               and directly_observed(temperature.iloc[index + 1], side)]
    return bool(at_peak or bridged), at_peak, bridged


def metric_rows(windows, original, candidate, reference):
    truth_windows, truth_events = reference
    all_events = []
    rr_pairs = []
    summaries = []
    for method, events in [('original', original), ('center_support', candidate)]:
        for w in windows.itertuples():
            ref = truth_events[(truth_events.window_id == w.window_id)
                               & (truth_events.confidence == 'confirmed')].event_time_seconds.to_numpy(float)
            assert len(ref) == int(w.manual_breath_count)
            pred = events[events.window_id == w.window_id].event_time_seconds.to_numpy(float)
            matched, fp, fn = match_events(pred, ref, .3)
            all_events.append(dict(method=method, window_id=w.window_id,
                                   **metrics(len(matched), len(fp), len(fn))))
            status = truth_windows.loc[w.window_id, 'prediction_status']
            if status == 'ok':
                duration = float(w.duration_seconds)
                rr_pairs.append(dict(method=method, window_id=w.window_id, truth_rr=len(ref)*60/duration,
                                     predicted_rr=len(pred)*60/duration, truth_count=len(ref), predicted_count=len(pred)))
        detail = pd.DataFrame(all_events)
        g = detail[detail.method == method]
        pairs = pd.DataFrame(rr_pairs)
        p = pairs[pairs.method == method]
        err = p.predicted_rr - p.truth_rr
        den = ((p.truth_rr - p.truth_rr.mean())**2).sum()
        summaries.append(dict(method=method, event_windows=len(g), count_pairs=len(p),
                              rr_mae=float(err.abs().mean()), rr_r2=float(1 - (err**2).sum()/den),
                              **metrics(*g[['tp', 'fp', 'fn']].sum())))
    return all_events, rr_pairs, summaries


def main():
    OUT.mkdir(exist_ok=False)
    frozen = BASES['jiufu271']
    original = read(frozen / 'prediction_events.csv')
    prediction_windows = read(frozen / 'prediction_windows.csv').set_index('window_id')
    external = read(TRANSFER / 'prediction_windows.csv').set_index('window_id')
    assert set(prediction_windows.index) == set(external.index)
    source_files = [frozen / 'prediction_events.csv', frozen / 'prediction_windows.csv',
                    TRANSFER / 'prediction_windows.csv']
    write_json(OUT / 'protocol_before_prediction.json', dict(
        role='retrospective, anatomy-informed peak-support filter; no new peak locations',
        rule='selected contributing side directly measured at peak OR same side directly measured at both adjacent frames',
        side_selection='saved normalized left/right/mean/min/max fusion mode; same-side bridge spans one missing sample',
        detector_and_fusion_unchanged=True, original_abstentions_unchanged=True,
        event_reference_not_read_by_predictor=True, reference_time_shift=0,
        selection='fixed from physical observability, not selected using reference score',
        baseline_input_sha256={str(p): sha256(p) for p in source_files},
        new_independent_test=False, default_changed=False, script_sha256=sha256(Path(__file__))))
    rows = []
    peak_audit = []
    file_hashes = []
    for ordinal, (wid, w) in enumerate(external.iterrows(), 1):
        baseline = original[original.window_id == wid]
        if w.prediction_status != 'ok':
            assert prediction_windows.loc[wid, 'prediction_status'] != 'ok' and baseline.empty
            continue
        assert prediction_windows.loc[wid, 'prediction_status'] == 'ok'
        folder = TRANSFER / 'windows' / wid
        curve = read(folder / 'curve.csv')
        temperature = read(folder / 'temperatures.csv')
        assert len(curve) == len(temperature)
        peak_indices = np.flatnonzero(curve.is_peak.to_numpy(bool))
        assert len(peak_indices) == len(baseline) == int(w.predicted_count)
        np.testing.assert_allclose(peak_indices / 8.7, baseline.event_time_seconds.to_numpy(float), atol=1e-8)
        for idx, p in zip(peak_indices, baseline.itertuples()):
            supported, center, bridge = has_center_support(curve, temperature, int(idx))
            peak_audit.append(dict(window_id=wid, event_id=p.event_id, frame_index=int(idx),
                                   event_time_seconds=p.event_time_seconds, retained=supported,
                                   direct_center_sides=','.join(center), bracket_sides=','.join(bridge)))
            if supported:
                rows.append(dict(window_id=wid, event_id=p.event_id, event_time_seconds=p.event_time_seconds))
        file_hashes.append(dict(window_id=wid, curve_sha256=sha256(folder/'curve.csv'),
                                temperatures_sha256=sha256(folder/'temperatures.csv')))
        if ordinal % 50 == 0:
            print(ordinal, '/271 inspected', flush=True)
    write_csv(OUT/'candidate_events.csv', rows, columns=['window_id','event_id','event_time_seconds'])
    write_csv(OUT/'peak_support.csv', peak_audit)
    write_csv(OUT/'input_hashes.csv', file_hashes)
    write_json(OUT/'prediction_seal.json', dict(checked_windows=len(external), output_windows=len(file_hashes),
        original_peaks=len(peak_audit), kept_peaks=len(rows),
        files={name:sha256(OUT/name) for name in ['candidate_events.csv','peak_support.csv','input_hashes.csv',
                                                   'protocol_before_prediction.json']}))

    # References enter only after the complete candidate event list has been sealed.
    print('271 windows screened and sealed; score with score_center_peak_support.py')


if __name__=='__main__':main()
