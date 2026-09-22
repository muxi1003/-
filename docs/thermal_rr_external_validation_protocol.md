# Thermal RR External Validation And Metadata Protocol

Updated: 2026-07-09

This protocol defines how the current thermal-video respiratory-rate package should move from an internally ready algorithmic manuscript to a Q2-or-higher submission package. It is aligned with the current evidence state: the 73 existing videos are internal development data, `cow_id` is the numeric video label, the collection window and site are known, and synchronized barn environment records plus manual visual-quality scores are unavailable.

## 1. Current Evidence Boundary

The current 73-video internal package is manuscript-ready for a bounded algorithmic claim. The default thermal RR pipeline reports `RR R2 = 0.928992`, `MAE = 1.9713 bpm`, `RMSE = 3.3950 bpm`, and exact count `49/73`. The fixed-threshold quality-aware residual correction reports `RR R2 = 0.942885`, `MAE = 1.6187 bpm`, `RMSE = 3.0448 bpm`, and exact count `53/73`. The conservative signal-aware safe gate is the highest-precision internal candidate, with fixed out-of-fold `RR R2 = 0.951993`, `MAE = 1.2884 bpm`, and exact count `59/73`; its leave-one-prefix-group-out stress test reports `RR R2 = 0.949883`, `MAE = 1.3569 bpm`, exact count `58/73`, and no count errors of two or more.

These results support internal method development and a frozen external-validation plan. They do not yet support a Q2+ final submission claim because there are zero independent external rows. The truth-calibrated result remains an oracle upper bound only and must not be reported as main model performance.

## 2. Metadata Boundary

The following current metadata fields are defensibly filled for all 73 internal videos:

- `cow_id`: numeric part of `video_id`, per current dataset convention.
- `collection_start_date`: `2023-08-05`.
- `collection_end_date`: `2023-08-10`.
- `collection_location_country`: `China`.
- `collection_location_province`: `Heilongjiang`.
- `collection_location_county`: `Lindian County`.
- `collection_site`: `Lindian County ranch`.
- `scene_id`: Lindian County ranch collection context.
- `external_test_split`: `internal`.

The following fields must remain blank unless real records or manual labels become available:

- Exact per-video `collection_date` and `collection_time`.
- `camera_id`.
- `ambient_temperature_c`, `relative_humidity_percent`, `thi`, and `athi`.
- `head_motion_score_0_3`, `occlusion_score_0_3`, and `nostril_visibility_score_0_3`.

These missing fields are not blockers for the algorithmic external-validation route. They are blockers only for heat-stress, welfare, camera-domain, or manual-quality-stratified claims. Do not fill them from model errors, regional weather proxies, or subjective inference unless the source is explicitly described and appropriate for the claim.

## 3. Primary Q2+ Route

The current primary route is an algorithmic agricultural-sensing paper, with Computers and Electronics in Agriculture as the first target after method-frozen external validation. The required next evidence is not more internal tuning. The required next evidence is an independent external set evaluated with the frozen method package.

Minimum algorithmic external-validation target:

- At least 104 independent external videos for the current Q2-oriented route.
- External rows must be prospectively defined or imported as independent data.
- Every external row needs `external_video_id`, `cow_id`, `collection_date` or session label, `camera_id` or `scene_id`, manual reference breath count, manual duration, manual RR, reference annotator/source, and `external_test_split=external`.
- Environment and manual visual-quality scores are optional unless the manuscript claims heat-stress, THI, motion, occlusion, or ROI-quality stratified performance.

If only 50 external videos can be obtained, report it as a minimum feasibility holdout, not as a strong Q2 external-performance package. If 209 or more external videos can be obtained, the paired external CI for relative improvement over the default method becomes more realistic.

## 4. Frozen External Workflow

The current method freeze is locked and includes the external postprocessor applier. The current freeze ID is:

`5de9befb2ee92ab1ebfab565cfd628637687f5fdc18f9bebcf8f313accedbe12`

The external workflow is generated in:

`Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_external_validation_run_commands.csv`

If that file is open in a spreadsheet application, the generator writes the updated command table to `paper_external_validation_run_commands_pending.csv`.

The current annotation-to-submission gate dashboard is:

`Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_external_validation_annotation_submission_dashboard.md`

The current workflow is:

1. Build blinded, randomized A and B breath-annotation HTML packets.
2. Have two annotators count breaths without using model output, error reports, cow ID, collection date/time, or source-session labels.
3. Run `scripts/build_rr_external_annotation_progress_monitor.py` after each export to check A/B completion, duplicate IDs, and missing required fields.
4. Run `scripts/audit_rr_external_breath_annotation_agreement.py` to create the agreement audit, consensus CSV, and adjudication template.
5. Resolve all missing or disagreeing rows in the adjudication template and rerun the agreement audit until `needs_adjudication_rows=0`.
6. Merge the consensus annotation CSV into a consensus fieldwork worksheet with `scripts/merge_rr_external_breath_annotations.py`.
7. Regenerate the external truth and metadata templates from the consensus fieldwork worksheet.
8. Validate the filled external fieldwork worksheet.
9. Extract external MP4 clips into frame folders under `Dataset_new/72video/external_al_images`.
10. Run the default frozen RR pipeline on the external frame folders.
11. Apply the frozen residual, signal-consensus, signal-aware, and safe-gate postprocessors using `scripts/apply_rr_frozen_external_postprocessors.py`.
12. Score the external split with `scripts/rr_external_split_validation.py`, then refresh readiness, submission-gap, route, and paper-asset outputs.

After the two browser annotation exports exist, the downstream steps can also
be run through one guarded command:

`E:\real\anaconda\envs\plant_gpu\python.exe scripts\run_rr_external_validation_after_annotation.py`

The runner writes `paper_external_validation_after_annotation_run_log.csv` and
`paper_external_validation_after_annotation_run_report.md`. It stops before
frame extraction or scoring if the A/B exports are missing, incomplete, or still
need adjudication.

During annotation, the progress monitor writes
`paper_external_validation_annotation_progress.md`,
`paper_external_validation_annotation_progress_summary.csv`,
`paper_external_validation_annotation_progress_by_video.csv`, and
`paper_external_validation_annotation_progress_missing_or_review.csv`.

The postprocessor applier may use external default predictions and external curve-derived features. It must not use external `truth_count`, `truth_rr`, external residual errors, or external metric results for training, feature selection, threshold selection, or method tuning. External truth can be used only after predictions exist, for metric computation.

## 5. External Data Layout

Place external frame folders under:

`Dataset_new/72video/external_al_images/<external_video_id>/`

Each external folder should contain the inputs required by `scripts/paper_repro_rr.py` to generate per-video temperature, curve, and peak-review outputs. The external truth table is generated from the fieldwork worksheet as:

`paper_external_validation_truth_template.csv`

The external metadata table is generated as:

`paper_external_validation_metadata_template.csv`

The field `external_test_split` means split provenance. Use `internal` for the current 73 videos. Use `external`, `holdout`, or `test` only for rows that were not used for method development, threshold selection, model fitting, or manuscript tuning.

## 6. Reference RR Rules

Manual reference RR should be recorded independently of model output. The annotator should not use predicted peaks, residual-corrector decisions, truth-calibrated curves, external error reports, cow ID, collection date/time, or source-session labels when deciding the reference breath count. The preferred protocol is blinded dual independent annotation followed by consensus or adjudication. At minimum, record:

- `manual_breath_count`
- `manual_duration_seconds`
- `manual_rr_bpm`, or leave it blank so it can be computed as `manual_breath_count * 60 / manual_duration_seconds`
- `reference_rr_annotator`
- `reference_protocol_notes`

For the `split_all_use` external batch, the current protocol uses two full blinded randomized annotation exports, not only a subset. The A packet uses seed `20260710`; the B packet uses seed `20260711`. Exact agreement is accepted automatically by default. Missing values or disagreements are routed to the adjudication template, and only consensus-ready rows should enter external scoring.

## 7. Reporting Plan

Primary external metrics:

- RR regression `R2`
- Pearson `R2`
- MAE
- RMSE
- exact breath-count agreement
- within-one-count agreement
- count errors of two or more

Primary external methods:

- Default thermal RR pipeline.
- Frozen quality-aware residual correction.
- Conservative signal-aware safe gate as the secondary engineering candidate.

The manuscript can claim external algorithmic performance only after the external split is nonempty, method-freeze status is locked, and external acceptance gates pass. If the safe gate passes the external sample-size and performance gates, it becomes the preferred algorithmic engineering result. If it does not, keep it as an internal precision candidate or secondary analysis.

## 8. Claim Boundaries

Allowed now:

- Internal algorithmic method development.
- Internal fixed out-of-fold, nested-CV, and prefix-group stress-test results.
- Current collection provenance: Lindian County ranch, Heilongjiang, China, 2023-08-05 to 2023-08-10.
- A frozen external-validation protocol and command pack.

Not allowed yet:

- Final Q2+ external-validation claim.
- Animal-level generalization unless numeric `cow_id` values are verified as real cow identities.
- Date, camera, heat-stress, THI, head-motion, occlusion, or nostril-visibility stratified claims.
- Truth-calibrated performance as a model result.

Allowed after external validation passes:

- Frozen external RR performance.
- Algorithmic external-validation table.
- Computers and Electronics in Agriculture or Biosystems Engineering route, depending on the final framing and external results.

Allowed only after new biological/environmental metadata:

- Heat-stress interpretation.
- THI/ATHI analysis.
- Dairy-health, welfare, or thermal-physiology claims.
- Stronger Journal of Dairy Science or Journal of Thermal Biology route.

## 9. Acceptance Gates

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Current metadata provenance | 73/73 current videos have numeric `cow_id`, collection window, collection site, scene label, and `external_test_split=internal` | Met |
| Method freeze | `paper_method_freeze_summary.csv` has `freeze_status=locked` and includes the frozen external postprocessor applier | Met |
| External input pack | Run commands include dual annotation QC, consensus merge, default RR, frozen postprocessors, external split scoring, and asset refresh | Met |
| Independent external rows | Nonzero external/holdout/test rows with manual RR truth | Not met |
| Q2 algorithmic external sample | Target 104 independent external videos | Not met |
| External acceptance gates | External RR R2/MAE/RMSE/within-one and safe-gate checks pass without retuning | Not met |
| Heat-stress or quality-stratified claims | Real temperature/humidity/THI or manual visual-quality scores | Not met and not required for algorithmic route |
| Submission readiness | `Q2-or-higher submission readiness` changes from `not_ready` to `ready` | Not met |

## 10. Immediate Next Action

The immediate next action is to complete the blinded A/B breath-count annotation for the 119 included `split_all_use` external clips, export both annotation CSVs, and run `scripts\run_rr_external_validation_after_annotation.py`. Do not relabel the current 73 internal videos as external. Do not fabricate environment or quality fields. If using the manual route instead of the runner, run the commands in `paper_external_validation_run_commands_pending.csv` or the unlocked `paper_external_validation_run_commands.csv` exactly in order after the consensus fieldwork worksheet exists, then review `paper_repro_external_split_acceptance.csv`, `paper_repro_external_split_metrics.csv`, and `paper_repro_quality_residual_submission_readiness_summary.md`.
