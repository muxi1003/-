# Bounded Peak Ablation

## Scope

This internal ablation tested four mechanisms requested for the 73-video thermal RR
workflow: a fixed-duration quality-selected window, periodic boundary-peak completion,
shallow double-peak suppression, and bilateral nostril quality fusion. All candidate
signals were generated from saved temperature curves and frame duration without reading
manual RR during prediction.

The fixed duration was `12 s`, defined as `ceil(median(raw_duration_seconds))` on the
internal inventory. Videos shorter than that duration retained their complete available
curve. The selected fixed window is offline because it chooses the highest-quality window
within a clip.

## Results

The direct replacement candidates did not beat the current frozen internal summary:

| Method | RR R2 | MAE (bpm) | RMSE (bpm) | Exact counts |
| --- | ---: | ---: | ---: | ---: |
| Current frozen summary | 0.9466 | 1.6917 | 3.0064 | 51/73 |
| Adaptive fusion plus double-peak suppression | 0.7911 | 4.1801 | 5.9461 | 36/73 |
| Adaptive fusion plus periodic edge completion | 0.5948 | 6.3410 | 8.2815 | 20/73 |
| Adaptive fusion plus all fixed-window mechanisms | 0.6746 | 5.3985 | 7.4217 | 25/73 |
| Structured diagnostic residual corrector, stratified OOF | 0.9437 | 1.7739 | 3.0874 | 50/73 |
| Structured diagnostic residual corrector, prefix-group holdout | 0.9461 | 1.7046 | 3.0217 | 51/73 |

The explicit periodic-edge rule over-counted because a high boundary maximum is not
enough evidence that a complete respiratory cycle occurred outside the window. Direct
continuous bilateral weighting also underperformed the existing adaptive fusion selector.
The current pipeline already contains shallow-peak merging, adaptive dense/sparse peak
retuning, and adaptive nostril fusion; replacing them with the new fixed rules removes
useful established behavior.

## Decision

Do not replace the current internal main method or safe gate with these candidates.
Keep the generated tables as negative ablation evidence. Future data collection should
use a fixed acquisition duration of at least 20-30 seconds, which reduces the RR impact
of one boundary-count error. For the current short clips, preserve adaptive fusion and
use endpoint, double-peak, and bilateral diagnostics only for manual-review triage or
future external evaluation rather than automatic peak-count overrides.

## Artifacts

- `paper_bounded_bilateral_peak_fusion_config.csv`
- `paper_bounded_bilateral_peak_fusion_predictions.csv`
- `paper_bounded_bilateral_peak_fusion_metrics.csv`
- `paper_bounded_bilateral_peak_fusion_bootstrap_ci.csv`
- `paper_bounded_bilateral_peak_fusion_report.md`
- `paper_structured_peak_diagnostic_residual_metrics.csv`
- `paper_structured_peak_diagnostic_residual_prefix_group_predictions.csv`
- `paper_structured_peak_diagnostic_residual_report.md`
