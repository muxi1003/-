# Thermal RR Algorithmic Engineering Manuscript Insert

This file is generated from the current paper assets. It is designed for the
algorithmic engineering route in which environment metadata and manual visual
quality scores cannot be recovered. Use it to update the manuscript without
turning location/date context into heat-stress or quality-stratified evidence.

## Proposed Framing

The feasible manuscript route is a method-frozen agricultural sensing study of
thermal-video respiratory-rate estimation with algorithmic quality control,
safe gating, selective reporting, and uncertainty reporting. The recommended
target-route record is `algorithmic_engineering_route_no_heat_quality_claims`, with current status
`hold_for_external_validation` and missing must-have gates
`external_validation_ready;fieldwork_algorithmic_q2_ready`. The contribution should be
presented as an internally validated algorithmic improvement plus a frozen
external-validation protocol, not as heat-stress monitoring, THI modeling, or
manual visual-quality robustness.

## Methods Insert

The current dataset contains 73 thermal videos collected at a ranch in Lindian
County, Heilongjiang, China, during 2023-08-05 to 2023-08-10. The numeric video
identifier was copied into `cow_id` as a dataset-level label, and
`external_test_split` was set to `internal` for all current videos. In this
workflow, `external_test_split` is a provenance label that separates internal
development/evaluation videos from future method-frozen external or holdout
videos; it is not a quality score and it does not make the present 73 videos an
external test set. Per-video barn temperature, humidity, THI/ATHI, camera ID,
exact acquisition time, and manual head-motion, occlusion, or nostril-visibility
scores were unavailable. Therefore, location and collection-window information
were retained only as provenance context.

The algorithmic route evaluates a default thermal RR pipeline and non-truth
residual-correction extensions. The conservative signal-aware safe gate applies
only prediction-time signal and model-consistency criteria and does not use the
reference RR, reference breath count, or truth-calibrated review parameters when
making a prediction. The frozen method manifest currently reports
`freeze_status=locked` and
`method_freeze_id=5de9befb2ee92ab1ebfab565cfd628637687f5fdc18f9bebcf8f313accedbe12`. Under this
claim boundary, truth-calibrated RR is retained only as an oracle diagnostic
for residual signal availability and is excluded from the main method row,
abstract performance claim, and external-validation claim.

## Results Insert

On the current 73-video internal set, the default thermal RR pipeline achieved
RR R2=0.9290, MAE=1.971 breaths/min, RMSE=3.395 breaths/min, exact=49/73, within-one=73/73. The conservative signal-aware safe gate achieved the
strongest internal non-truth result, with RR R2=0.9520, MAE=1.288 breaths/min, RMSE=2.792 breaths/min, exact=59/73, within-one=73/73. Under the
prefix-group internal stress test, the same safe-gate candidate reached
RR R2=0.9499, MAE=1.357 breaths/min, RMSE=2.852 breaths/min, exact=58/73, within-one=73/73. These values support an internal precision and
robustness claim, but they do not yet establish external generalization because
the external split has no scored external rows.

The reliability layer can be described as an internal engineering control.
At the internal risk <= 0.30 operating point, 22/73 videos were assigned to automatic reporting and 51/73 to review; the auto-report subset reached RR R2=0.9872, MAE=0.506 breaths/min, and exact=20/22. The internal risk-adaptive conformal interval targeted 0.90 coverage, achieved coverage=0.9041, and had mean width=11.791 breaths/min; low-risk coverage was 0.9545. RR-only auto-report triage selected 22/73 low-risk videos (coverage=0.301) with RR R2=0.9872, MAE=0.506 breaths/min, and exact rate=0.909; the auto-report high-RR candidate subset contained 6 videos with RR R2=0.9987, MAE=0.111 breaths/min, and exact rate=1.000; all predicted upper-quartile RR candidates contained 22 videos and should be treated as context-review candidates rather than heat-stress diagnoses. bilateral nostril consistency screening retained 19/73 auto-report videos (coverage=0.260) with RR R2=0.9910, MAE=0.349 breaths/min, RMSE=1.159 breaths/min, and exact rate=0.947. These
analyses should be reported as internal selective-reporting, RR-only
review/alert, bilateral signal-consistency, and uncertainty experiments whose
thresholds must remain frozen before external evaluation. The
truth-calibrated upper bound
reached RR R2=0.9997, MAE=0.139 breaths/min, RMSE=0.237 breaths/min, exact=73/73, within-one=73/73, but this number uses reference information and
should appear only as a supplementary oracle diagnostic, not as the headline RR
R2.

A narrow consensus rollback guard was also evaluated as an exploratory precision supplement after the safe gate. In fixed out-of-fold validation it reached RR R2=0.9574, MAE=1.186 breaths/min, RMSE=2.630 breaths/min, exact=60/73, within-one=73/73, and under prefix-group stress it reached RR R2=0.9553, MAE=1.254 breaths/min, RMSE=2.695 breaths/min, exact=59/73, within-one=73/73. The paired fixed comparison gave paired delta RR R2=0.0054 (95% bootstrap CI 0.0000 to 0.0164), delta exact=1, McNemar one-sided p=0.500; the prefix-group comparison gave paired delta RR R2=0.0054 (95% bootstrap CI 0.0000 to 0.0161), delta exact=1, McNemar one-sided p=0.500. The rollback-specific bootstrap confidence interval touches zero, so this is a promising internal error-analysis signal rather than statistically confirmed superiority. Because this rule was derived from internal error analysis and rescues a rare harmful correction pattern, it should be treated as a supplementary candidate until frozen and externally validated.

## External Validation Insert

The next manuscript gate for the no-environment, no-manual-quality route is the
`q2_algorithmic_external_validation` tier. The current collection plan specifies
104 external videos with required fields
`cow_id;collection_date;camera_id;scene_id;external_test_split;manual_breath_count;manual_duration_seconds;manual_rr_bpm;reference_rr_annotator`.
Its design rule is: freeze the conservative signal-aware safe gate before scoring; sample prospectively across cows, sessions, and camera/scene groups; do not require environment or manual quality scores unless making those claims.
The execution dashboard currently marks
`q2_algorithmic_external_validation_fieldwork` as
`FAIL`, and the external split gate
`algorithmic engineering external claim allowed` is
`FAIL` with threshold
`external_n >= 104 + method frozen + all algorithmic absolute gates PASS`.
Before this gate passes, the paper may state that the external protocol is
specified and the method is frozen, but it must not report an external
performance claim.

The predefined external acceptance rules for the algorithmic engineering route
are: external RR R2 >= 0.90 and preferably lower 95% CI >= 0.90; external MAE <= 2.5 bpm and preferably upper 95% CI <= 2.5 bpm; external RMSE <= 4.0 bpm and preferably upper 95% CI <= 4.0 bpm; and external exact-count agreement should not be worse than the default external pipeline; within-one agreement >= 95%. Passing this tier
would support an external algorithmic RR accuracy claim for the frozen
safe-gate workflow, including algorithmic quality control and uncertainty
reporting. It would still not support heat-stress, THI-stratified,
head-motion-stratified, occlusion-stratified, or nostril-visibility-stratified
claims unless those real metadata fields are later collected.

## Discussion Insert

The main distinction for the paper is between an RR algorithmic claim and a
biological heat-stress claim. The former can be pursued with independent
external videos, manual RR truth, cow/session/camera provenance, and a frozen
method. The latter requires real synchronized temperature and humidity or THI
measurements and, for visual-quality robustness, manual head-motion, occlusion,
and nostril-visibility scores. Because those fields are unavailable in the
current dataset, the manuscript should explicitly state that the study evaluates
thermal-video RR estimation and reliability triage rather than validated
heat-stress monitoring. The Heilongjiang/Lindian County collection context is
safe to report as provenance, but it should not be converted into per-video
environment exposure or quality labels.

## Claim Boundary Table

| claim_family | status | paper_location | recommended_wording | do_not_write |
| --- | --- | --- | --- | --- |
| current_internal_algorithmic_rr_accuracy | allowed_now_internal_only | Results and limitations | Report the current 73-video set as internal validation. Default workflow: RR R2=0.9290, MAE=1.971 breaths/min, RMSE=3.395 breaths/min, exact=49/73, within-one=73/73. Conservative signal-aware safe gate: RR R2=0.9520, MAE=1.288 breaths/min, RMSE=2.792 breaths/min, exact=59/73, within-one=73/73. | Do not call this external, prospective, cross-farm, or deployment-ready validation. |
| prefix_group_stress_evidence | allowed_now_as_internal_stress_test | Secondary robustness results | Use prefix-group validation as an internal domain-risk stress test only: RR R2=0.9499, MAE=1.357 breaths/min, RMSE=2.852 breaths/min, exact=58/73, within-one=73/73. | Do not call prefix groups true cow-level, camera-level, date-level, or farm-level validation. |
| consensus_rollback_guard | exploratory_internal_only | Supplementary error analysis | A narrow consensus rollback guard can be reported as an exploratory internal probe for rare harmful residual corrections: RR R2=0.9574, MAE=1.186 breaths/min, RMSE=2.630 breaths/min, exact=60/73, within-one=73/73; paired delta RR R2=0.0054 (95% bootstrap CI 0.0000 to 0.0164), delta exact=1, McNemar one-sided p=0.500. | Do not present the rollback guard as the frozen primary method, as statistically confirmed superiority, or as externally validated precision improvement. |
| truth_calibrated_rr_r2 | upper_bound_only | Supplementary diagnostic, not abstract | Truth-calibrated RR can be described as an oracle upper bound showing residual respiratory information in the curves: RR R2=0.9997, MAE=0.139 breaths/min, RMSE=0.237 breaths/min, exact=73/73, within-one=73/73. | Do not use truth-calibrated R2 as the main method R2 or compare it as a deployable model. |
| algorithmic_engineering_external_validation | FAIL | Future external validation or main table only after PASS | After independent external scoring, claim a frozen algorithmic RR validation only if the 104-video algorithmic tier and all predefined RR gates pass. | Do not use the current 73 internal videos as external validation. Current target route status: hold_for_external_validation. |
| heat_stress_thi_or_environment | not_allowed_with_current_metadata | Limitations only | Report Heilongjiang, Lindian County ranch and the 2023-08-05 to 2023-08-10 collection window as provenance context only. | Do not infer per-video temperature, humidity, THI, ATHI, or heat-stress strata from location/date context. |
| manual_quality_stratification | not_allowed_with_current_metadata | Limitations only | State that manual head-motion, occlusion, and nostril-visibility scores were unavailable. | Do not replace manual scores with algorithmic risk scores. |
| algorithmic_quality_and_uncertainty | allowed_now_internal_engineering_evidence | Methods, Results, Discussion | Algorithmic quality tiers, selective reporting, safe gating, and conformal RR intervals may be reported as internal engineering controls that must be frozen before external use. | Do not describe these controls as clinically or prospectively validated reliability. |
| rr_only_physiological_triage | allowed_now_internal_review_alert_evidence | Methods, Results, Discussion, Supplement | RR-only auto-report triage selected 22/73 low-risk videos (coverage=0.301) with RR R2=0.9872, MAE=0.506 breaths/min, and exact rate=0.909; the auto-report high-RR candidate subset contained 6 videos with RR R2=0.9987, MAE=0.111 breaths/min, and exact rate=1.000; all predicted upper-quartile RR candidates contained 22 videos and should be treated as context-review candidates rather than heat-stress diagnoses. | Do not describe high-RR candidate flags as heat-stress, disease, welfare, or treatment labels without synchronized environment and animal-context metadata. |
| bilateral_nostril_consistency_gate | allowed_now_internal_selective_reporting_evidence | Methods, Results, Discussion, Supplement | bilateral nostril consistency screening retained 19/73 auto-report videos (coverage=0.260) with RR R2=0.9910, MAE=0.349 breaths/min, RMSE=1.159 breaths/min, and exact rate=0.947. | Do not describe bilateral consistency as external validation, heat-stress diagnosis, welfare classification, disease detection, or true animal-identity evidence. |
