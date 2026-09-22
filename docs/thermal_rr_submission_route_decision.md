# Thermal RR Submission Route Decision

Updated: 2026-07-08

This document summarizes the current Q2-or-higher submission route after the metadata boundary update. The authoritative generated files are:

- `Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_submission_gap_route_decision.md`
- `Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_submission_gap_route_journal_fit.csv`
- `Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_submission_gap_route_minimum_data.csv`
- `Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_submission_gap_route_next_actions.csv`

## Current Decision

The strongest current route is an algorithmic agricultural-sensing manuscript targeting `Computers and Electronics in Agriculture` after independent external validation. `Biosystems Engineering` is the second route if the paper is framed as an operational biosystems engineering workflow with a clear Science4Impact statement. `Animals` is a fallback after external validation and animal-welfare framing. `Journal of Dairy Science` and `Journal of Thermal Biology` are not the current primary routes because they require biological, dairy-management, or synchronized thermal-environment metadata that are not available for the current 73 videos.

## Evidence Boundary

The current best manuscript-usable internal candidate is the conservative signal-aware safe gate. In the refreshed assets, it reaches internal RR R2 of about 0.952, MAE of about 1.288 bpm, and exact breath-count agreement of 59/73. This can be written as internal method evidence only. It is not external validation.

The current 73 videos must remain `internal` in `external_test_split`. They were used for development and cannot be relabeled as external validation. The required next dataset for the primary Q2+ algorithmic route is the 104-video method-frozen external validation tier with independent manual RR references.

## Metadata Boundary

The current safe metadata prefill is:

- `cow_id`: numeric video label.
- collection window: `2023-08-05` to `2023-08-10`.
- collection site: Lindian County ranch, Heilongjiang, China.
- `scene_id`: `CN_HLJ_Lindian_ranch_2023-08-05_to_2023-08-10`.
- `external_test_split`: `internal`.

Synchronized barn temperature, relative humidity, THI/ATHI, head-motion score, occlusion score, and nostril-visibility score remain unavailable. These fields are claim-specific limits, not fields to fabricate. Their absence blocks heat-stress, welfare-outcome, and manual-quality stratified claims, but it does not block the algorithmic external-validation route.

## Minimum Next Step

Collect or import at least 104 independent videos for `q2_algorithmic_external_validation`. Each row should include `external_test_split`, `cow_id`, `collection_date`, `camera_id`, `scene_id`, `manual_breath_count`, `manual_duration_seconds`, `manual_rr_bpm`, and `reference_rr_annotator`. Do not tune thresholds on these videos. The frozen method must be applied first, then external metrics can be reported.

Truth-calibrated RR remains an oracle upper-bound diagnostic only and must not be used in the abstract, highlights, or primary performance claim.
