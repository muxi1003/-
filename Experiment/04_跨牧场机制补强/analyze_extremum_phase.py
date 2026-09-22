"""Reference-relative extrema diagnosis, without time shifts or gap filling."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from analyze_transfer import HERE, HOLDOUT, read, write_csv, write_json, sha256, match_events, metrics

ROOT = HERE / '20260922_pose_phase_v1'
TRANSFER = HOLDOUT / 'external_transfer/20260914_v2/windows'
PARTIAL = HERE / '20260921_chain_v3/partial_timestamp/windows'


def finite_runs(x):
    mask = np.r_[False, np.isfinite(x), False]
    starts = np.flatnonzero(np.diff(mask.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(mask.astype(int)) == -1)
    return list(zip(starts, stops))


def extrema(time, values):
    """Frozen MAF3/distance5/prominence.035, no smoothing across missing frames."""
    output = []
    x = np.asarray(values, float)
    time = np.asarray(time, float)
    finite = x[np.isfinite(x)]
    if len(finite) < 7 or np.ptp(finite) <= 1e-12:
        return output
    norm = (x-finite.min()) / np.ptp(finite)
    for start, stop in finite_runs(norm):
        if stop-start < 7:
            continue
        # Convolution valid samples align with central input timestamps.
        smooth = np.convolve(norm[start:stop], np.ones(3)/3, mode='valid')
        ids = np.arange(start+1, stop-1)
        for kind, sign in [('maximum', 1), ('minimum', -1)]:
            peaks, properties = find_peaks(sign*smooth, distance=5, prominence=.035)
            for k, prom in zip(peaks, properties['prominences']):
                ix = int(ids[k])
                output.append(dict(kind=kind, frame_index=ix, event_time_seconds=float(time[ix]),
                    normalized_value=float(smooth[k]), prominence=float(prom),
                    observed_run_start=float(time[start]), observed_run_end=float(time[stop-1])))
    return output


def main():
    out = ROOT / 'phase_audit'; out.mkdir(exist_ok=False)
    audit = read(HERE / '20260921_r3_v1/annotation_audit.csv')
    audit = audit[audit.cohort.eq('jiufu271')]
    refs = read(HERE / '20260921_r3_v1/raw/reference_events.csv')
    candidates, details, scores, input_hashes = [], [], [], []
    for row in audit.itertuples():
        wid = row.window_id
        roots = [TRANSFER/wid] if (TRANSFER/wid/'temperatures.csv').exists() else sorted((PARTIAL/wid).glob('segment_*'))
        tables = []
        for folder in roots:
            if not (folder/'temperatures.csv').exists():
                continue
            temp = read(folder/'temperatures.csv')
            curve = read(folder/'curve.csv') if (folder/'curve.csv').exists() else None
            mapping = read(TRANSFER/wid/'map.csv') if (TRANSFER/wid/'map.csv').exists() else None
            time = curve.target_time_seconds.to_numpy(float) if curve is not None else mapping.target_time_seconds.to_numpy(float)
            assert len(time) == len(temp) and np.all(np.diff(time)>0)
            tables.append((time, temp))
            input_hashes.append(dict(path=str(folder/'temperatures.csv'), sha256=sha256(folder/'temperatures.csv')))
            for side in ['left', 'right']:
                for peak in extrema(time, temp[side+'_temp']):
                    candidates.append(dict(review_id=row.review_id, window_id=wid, side=side,
                        segment=str(folder.name), eligible_full_window=row.eligible_full_window,
                        strict_visible_phase=row.strict_visible_phase, **peak))
        re = refs[refs.window_id.eq(wid)&refs.confidence.eq('confirmed')]
        window_candidates = [c for c in candidates if c['window_id']==wid]
        for event in re.itertuples():
            t = float(event.event_time_seconds)
            record = dict(review_id=row.review_id, window_id=wid, event_id=event.event_id,
                reference_seconds=t, phase_basis=event.phase_basis, eligible_full_window=row.eligible_full_window,
                strict_visible_phase=row.strict_visible_phase)
            available = {}
            for side in ['left', 'right']:
                available[side] = any(np.isfinite(df[side+'_temp'].iloc[ix]) and abs(times[ix]-t)<=.25
                    for times, df in tables for ix in [int(np.argmin(abs(times-t)))])
                record[side+'_sample_available'] = available[side]
                for kind in ['maximum','minimum']:
                    cc = [c for c in window_candidates if c['side']==side and c['kind']==kind]
                    nearest = min(cc, key=lambda c:abs(c['event_time_seconds']-t), default=None)
                    record[side+'_'+kind+'_delta_s'] = nearest['event_time_seconds']-t if nearest else np.nan
            maximum = any(abs(record[s+'_maximum_delta_s'])<=.3 for s in ['left','right'])
            minimum = any(abs(record[s+'_minimum_delta_s'])<=.3 for s in ['left','right'])
            record['classification'] = ('no_observed_ROI_sample' if not any(available.values()) else
                'both_extrema_nearby' if maximum and minimum else 'maximum_only' if maximum else
                'minimum_only' if minimum else 'no_extremum_within_tolerance')
            details.append(record)
        if row.eligible_full_window:
            truth = re.event_time_seconds.to_numpy(float)
            for side in ['left','right']:
                for kind in ['maximum','minimum']:
                    pp = [c['event_time_seconds'] for c in window_candidates if c['side']==side and c['kind']==kind]
                    matched, fp, fn = match_events(pp,truth,.3)
                    scores.append(dict(review_id=row.review_id,window_id=wid,side=side,kind=kind,
                        strict_visible_phase=row.strict_visible_phase,**metrics(len(matched),len(fp),len(fn))))
    write_csv(out/'extremum_candidates.csv',candidates)
    write_csv(out/'reference_event_audit.csv',details)
    write_csv(out/'per_window_matching.csv',scores)
    write_csv(out/'input_hashes.csv',input_hashes)
    d=pd.DataFrame(details);s=pd.DataFrame(scores); summaries=[]
    for scope,group in [('all_eligible',d[d.eligible_full_window]),('strict_visible',d[d.strict_visible_phase])]:
        for classification,g in group.groupby('classification'):
            summaries.append(dict(scope=scope,classification=classification,events=len(g),windows=g.window_id.nunique()))
    write_csv(out/'event_classification_summary.csv',summaries)
    write_json(out/'protocol_and_limits.json',dict(
        role='diagnosis_only_not_new_RR_predictor',reference='R3 20260921 19:21',
        smoothing='MAF3 only within finite same-side run, valid central timestamps',
        distance_frames=5,prominence=.035,tolerance_seconds=.3,reference_time_shift=0,
        signal='archived native-pixel RF ROI samples, no missing-value interpolation',
        matching='per-side per-polarity one-to-one; per-reference proximity is descriptive only',
        no_observed_ROI_sample='nearest sampled timestamp within .25 s; not proof of human invisibility',
        nearby_extremum='temporal agreement only, not independent physiological validation',
        no_default_or_reference_changes=True))
    print(pd.DataFrame(summaries).to_string(index=False))
    for key,g in s.groupby(['side','kind']):
        print(key,metrics(*g[['tp','fp','fn']].sum()))


if __name__=='__main__':main()
