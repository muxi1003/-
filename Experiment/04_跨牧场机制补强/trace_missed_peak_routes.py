"""Separate abstention, missing signal and later candidate losses without retuning."""
import numpy as np
import pandas as pd
from analyze_transfer import read,write_csv,write_json,HOLDOUT
from audit_frozen_peak_evidence import ROOT,OUT,TRANSFER


def main():
    refs=read(OUT/'reference_loss_stage.csv')
    refs=refs[refs.strict_visible_phase&refs.matching_status.eq('FN')]
    candidates=read(ROOT/'phase_audit/extremum_candidates.csv')
    status=read(HOLDOUT/'external_transfer/20260914_v2/prediction_windows.csv').set_index('window_id')
    rows=[]
    for r in refs.itertuples():
        s=status.loc[r.window_id]
        c=candidates[candidates.window_id.eq(r.window_id)&candidates.event_time_seconds.sub(r.reference_seconds).abs().le(.3)]
        positive=c[c.kind.eq('maximum')]
        record=dict(window_id=r.window_id,review_id=r.review_id,reference_event_id=r.event_id,
            reference_seconds=r.reference_seconds,previous_classification=r.classification,
            frozen_prediction_status=s.prediction_status,abstention_reason=s.reason,
            nearby_maximum_count=len(positive),nearby_minimum_count=int(c.kind.eq('minimum').sum()),
            selected_mode='',selected_candidate_flags='')
        if s.prediction_status!='ok':
            route='whole_window_abstention'
        elif r.classification=='no_observed_ROI_sample':
            route='no_ROI_temperature_near_reference'
        elif c.empty:
            route='no_raw_extremum_near_reference'
        elif positive.empty:
            route='minimum_only_not_positive_peak'
        else:
            curve=read(TRANSFER/r.window_id/'curve.csv')
            mode=curve.selected_fusion_mode.iloc[0];record['selected_mode']=mode
            output=curve[curve.is_peak].target_time_seconds
            if output.sub(r.reference_seconds).abs().le(.3).any():
                route='nearby_output_used_by_other_reference'
            elif mode in ['left','right'] and not positive.side.eq(mode).any():
                route='maximum_only_in_nonselected_side'
            else:
                selected=positive[positive.side.eq(mode)] if mode in ['left','right'] else positive
                flags=[]
                for candidate in selected.itertuples():
                    ix=int(candidate.frame_index)
                    assert abs(curve.target_time_seconds.iloc[ix]-candidate.event_time_seconds)<1e-8
                    ff=[name for name in ['source_peak_rejected','quality_peak_rejected','motion_peak_rejected',
                        'motion_interval_peak_rejected','mad_ibi_peak_rejected'] if bool(curve.iloc[ix][name])]
                    flags.append(','.join(ff) or 'no_explicit_rejection_flag')
                record['selected_candidate_flags']=';'.join(flags)
                route=('selected_maxima_rejected_by_source_gate'
                       if flags and all(f=='source_peak_rejected' for f in flags)
                       else 'selected_signal_or_peak_rule_difference')
        record['observed_route']=route;rows.append(record)
    d=pd.DataFrame(rows)
    assert len(d)==84
    summary=d.groupby('observed_route').agg(events=('reference_event_id','size'),windows=('window_id','nunique')).reset_index()
    write_csv(OUT/'strict_FN_route_evidence.csv',rows)
    write_csv(OUT/'strict_FN_route_summary.csv',summary)
    write_json(OUT/'route_limits.json',dict(
        scope='strict9 R3 windows; 84 frozen false negatives only',
        routes='observed pipeline conditions, not exclusive physiological causes',
        raw_extrema_on_abstained_windows='existing partial-time diagnostic outputs are not frozen whole-window predictions',
        spatial_ground_truth_not_propagated=True,no_counterfactual_count_or_threshold_change=True,
        maximum_near_reference_does_not_prove_recoverable_true_breath=True))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
