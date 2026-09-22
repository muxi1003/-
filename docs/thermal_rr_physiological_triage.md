# RR-Only Physiological Triage Layer

Generated: 2026-07-09T11:03:16+00:00

Risk threshold: `0.30`
Conformal interval: `0.90` target coverage, `risk_adaptive_fixed_scale`

This asset converts the strongest current non-truth RR estimate into a review and alert workflow: automatic reporting for low algorithmic-risk videos, high-RR candidate flags from predicted RR only, and explicit context-review boundaries. It does not use reference truth for triage decisions.

## Internal Audit

| label | videos | coverage | rr_r2 | rr_mae | rr_rmse | exact_count | exact_rate | abs_count_error_ge2 | selection_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_safe_gate_predictions | 73 | 1.0000 | 0.9520 | 1.2884 | 2.7915 | 59 | 0.8082 | 0 | all videos |
| auto_report_subset | 22 | 0.3014 | 0.9872 | 0.5058 | 1.3783 | 20 | 0.9091 | 0 | algorithmic_review_risk_score <= frozen deployment threshold |
| manual_review_subset | 51 | 0.6986 | 0.9380 | 1.6260 | 3.2147 | 39 | 0.7647 | 0 | algorithmic_review_risk_score > frozen deployment threshold |
| predicted_upper_quartile_rr_candidate | 22 | 0.3014 | 0.8491 | 1.6504 | 3.3613 | 17 | 0.7727 | 0 | predicted safe-gate RR >= internal q75 (60.000 bpm) |
| auto_high_rr_candidate | 6 | 0.0822 | 0.9987 | 0.1111 | 0.1925 | 6 | 1.0000 | 0 | auto-report candidate and predicted safe-gate RR >= internal q75 |
| predicted_top_decile_rr_candidate | 8 | 0.1096 | 0.5291 | 3.0387 | 4.8377 | 5 | 0.6250 | 0 | predicted safe-gate RR >= internal q90 (67.333 bpm) |

## Triage Actions

| triage_recommended_action | videos |
| --- | --- |
| manual_review_low_confidence | 35 |
| auto_report_routine_rr | 17 |
| manual_review_high_rr_candidate | 16 |
| auto_report_high_rr_candidate_context_review | 5 |

## Metadata Boundary

| field | nonempty | total_videos | coverage_percent | triage_interpretation |
| --- | --- | --- | --- | --- |
| collection_start_date | 73 | 73 | 100.0000 | available_context_or_split_field |
| collection_end_date | 73 | 73 | 100.0000 | available_context_or_split_field |
| collection_location_country | 73 | 73 | 100.0000 | available_context_or_split_field |
| collection_location_province | 73 | 73 | 100.0000 | available_context_or_split_field |
| collection_location_county | 73 | 73 | 100.0000 | available_context_or_split_field |
| collection_site | 73 | 73 | 100.0000 | available_context_or_split_field |
| ambient_temperature_c | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| relative_humidity_percent | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| thi | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| athi | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| head_motion_score_0_3 | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| occlusion_score_0_3 | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| nostril_visibility_score_0_3 | 0 | 73 | 0.0000 | missing_do_not_claim_stratified_physiology_or_quality |
| external_test_split | 73 | 73 | 100.0000 | available_context_or_split_field |

## Source Register

| source_id | source_type | url | paper_use |
| --- | --- | --- | --- |
| lin_2025_irt_head_movement_rr | thermal_rr_baseline | https://doi.org/10.1016/j.jtherbio.2025.104154 | Positions thermal nostril RR as current baseline; triage adds reliability and review value. |
| yan_2024_environmental_ml_rr | environmental_rr_context | https://doi.org/10.1016/j.biosystemseng.2024.01.010 | Supports the need for environment variables before physiological interpretation. |
| cartwright_2022_heat_challenge | heat_stress_physiology_context | https://arxiv.org/abs/2201.02675 | Shows RR and THI are linked in heat-challenge studies; current dataset lacks synchronized THI. |
| sadeghi_2024_calves_thermal_vitals | thermal_vital_sign_context | https://arxiv.org/abs/2405.11532 | Supports thermal vital-sign monitoring as a PLF direction while preserving species/age boundaries. |

## Manuscript Boundary

Use this as an internal RR-only deployment and interpretation supplement. The current data support high-precision automatic RR reporting for a selected low-risk subset, not a heat-stress diagnosis. Heat-stress, welfare, motion, occlusion, or cow-level physiological claims still require synchronized environment metadata, manual or validated automated quality labels, and independent external validation.
