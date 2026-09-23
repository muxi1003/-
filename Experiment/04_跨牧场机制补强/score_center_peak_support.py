"""Score sealed center-support peaks against unchanged R2 and R3 references."""
import json
import pandas as pd
from analyze_transfer import BASES, HERE, SUBMISSION, read, sha256, write_csv, write_json
from audit_center_peak_support import OUT, metric_rows


def main():
    seal=json.loads((OUT/'prediction_seal.json').read_text(encoding='utf-8'))
    for filename,digest in seal['files'].items():assert sha256(OUT/filename)==digest
    frozen=BASES['jiufu271']
    prediction_windows=read(frozen/'prediction_windows.csv').set_index('window_id')
    original=read(frozen/'prediction_events.csv')
    candidate=read(OUT/'candidate_events.csv')
    support=read(OUT/'peak_support.csv')
    assert len(support)==seal['original_peaks'] and len(candidate)==seal['kept_peaks']
    r2=read(SUBMISSION/'annotation_windows.csv')
    r2=r2[(r2.cohort=='jiufu271') & (r2.annotation_status=='complete')]
    r2_events=read(SUBMISSION/'reference_events.csv')
    audit=read(HERE/'20260921_r3_v1/annotation_audit.csv')
    audit=audit[(audit.cohort=='jiufu271') & audit.eligible_full_window]
    r3=read(HERE/'20260921_r3_v1/raw/annotation_windows.csv')
    r3_events=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    results=[];detail=[];counts=[]
    for name,cohort,events in [
        ('R2_complete216',r2,r2_events),
        ('R3_full11',r3[r3.window_id.isin(audit.window_id)],r3_events),
        ('R3_strict9',r3[r3.window_id.isin(audit[audit.strict_visible_phase].window_id)],r3_events)]:
        event_rows,count_rows,summary=metric_rows(cohort,original,candidate,(prediction_windows,events))
        detail.extend(dict(cohort=name,**row) for row in event_rows)
        counts.extend(dict(cohort=name,**row) for row in count_rows)
        results.extend(dict(cohort=name,**row) for row in summary)
    expected=next(row for row in results if row['cohort']=='R2_complete216' and row['method']=='original')
    assert expected['event_windows']==216 and expected['count_pairs']==173
    assert (expected['tp'],expected['fp'],expected['fn'])==(1324,1028,1706)
    assert abs(expected['rr_r2']-0.419629)<1e-5
    write_csv(OUT/'metrics.csv',results)
    write_csv(OUT/'event_window_results.csv',detail)
    write_csv(OUT/'rr_pairs.csv',counts)
    write_json(OUT/'scoring_provenance.json',dict(
        r2_reference_sha256={str(p):sha256(p) for p in [SUBMISSION/'annotation_windows.csv',SUBMISSION/'reference_events.csv']},
        r3_reference_sha256={str(p):sha256(p) for p in [HERE/'20260921_r3_v1/raw/annotation_windows.csv',
            HERE/'20260921_r3_v1/raw/reference_events.csv',HERE/'20260921_r3_v1/annotation_audit.csv']},
        baseline_R2_replayed=True,matching_tolerance_seconds=.3,count_pairs_unchanged=173,
        peak_filter_read_only_presealed=True,default_changed=False))
    print(pd.DataFrame(results).to_string(index=False))


if __name__=='__main__':main()
