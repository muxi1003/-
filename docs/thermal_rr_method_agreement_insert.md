# RR Method Agreement Manuscript Insert

This generated insert converts the current RR Bland-Altman agreement outputs
into manuscript-ready wording. The values come from the internal 73-video set
and must not be described as independent external validation.

## Methods Wording

Bland-Altman agreement was used as a complementary evaluation to identity-line
RR R2, MAE, RMSE, and exact breath-count agreement. Agreement error was defined
as predicted respiratory rate minus manual reference respiratory rate. For each
method, the mean bias, error standard deviation, and 95% limits of agreement
(LoA; bias +/- 1.96 SD) were computed over videos with both manual and predicted
RR. Proportional bias was screened by regressing the agreement error against
the mean of manual and predicted RR. Respiratory-rate strata were summarized as
low RR (<50 breaths/min), mid RR (50 to <70 breaths/min), and high RR (>=70
breaths/min). These analyses were applied only to the current internal
development set and should be rerun unchanged after the `split_all_use`
external videos receive blinded A/B consensus reference labels.

## Results Wording

To complement the R2 and count-agreement results, Bland-Altman analysis showed
that the proposed corrections tightened the error distribution around the manual
reference. The default thermal RR pipeline had a mean bias of -0.011 breaths/min, 95% limits of agreement from -6.711 to 6.689 breaths/min, a LoA width of 13.400 breaths/min, MAE of 1.971 breaths/min, and exact count agreement in 49/73 videos. The quality-aware residual correction
reduced the LoA width to 12.018 breaths/min, and
the signal-consensus supplement further reduced it to
11.639 breaths/min. The conservative signal-aware
safe gate produced the tightest current non-truth agreement, with a bias of
-0.117 breaths/min and 95% LoA from
-5.621 to 5.388 breaths/min.
Relative to the default pipeline, the safe gate narrowed the LoA width by
2.392 breaths/min, decreased
MAE by 0.683 breaths/min, decreased
RMSE by 0.603 breaths/min, and
increased exact count agreement by 10
videos, without introducing any >=2-breath count errors. In the prefix-group
internal stress test, the safe-gate LoA width remained
11.256 breaths/min, supporting the interpretation
that the safe gate improves agreement while preserving grouped internal
robustness.

The stratum analysis indicated that the agreement gain was clearest in the
lower and middle RR ranges. In the safe-gate result, the low-RR stratum
contained 37 videos with MAE
1.206 breaths/min, and the mid-RR stratum contained
30 videos with MAE
1.077 breaths/min. The high-RR stratum remained weakly supported because it contained only 6 videos; in that stratum the safe gate had bias -2.857 breaths/min and MAE 2.857 breaths/min. This result should be reported as a limitation rather than as evidence for high-RR or heat-stress deployment.

## Discussion Wording

The Bland-Altman results strengthen the manuscript claim because they show that
the improvement is not limited to a higher RR R2. The safe gate also reduced
the absolute error distribution and narrowed the limits of agreement, which is
more directly relevant to whether an automatically reported respiratory rate
can be interpreted against manual counting. However, this remains an internal
agreement analysis. The method should therefore be described as an internally
validated precision candidate with agreement tightening, while external
generalization, animal/session/camera robustness, and high-RR biological claims
remain conditional on the frozen `split_all_use` external scoring workflow.

## Figure Captions

**paper_rr_method_agreement_bland_altman_default.png.** Bland-Altman plot for Default thermal RR pipeline on the internal 73-video set. The vertical axis shows predicted minus manual respiratory rate, and the horizontal axis shows the mean of predicted and manual respiratory rate. The solid line denotes the mean bias (-0.011 breaths/min), and dashed lines denote the 95% limits of agreement (-6.711 to 6.689 breaths/min).

**paper_rr_method_agreement_bland_altman_quality_residual_fixed_oof.png.** Bland-Altman plot for Quality-aware residual correction on the internal 73-video set. The vertical axis shows predicted minus manual respiratory rate, and the horizontal axis shows the mean of predicted and manual respiratory rate. The solid line denotes the mean bias (-0.005 breaths/min), and dashed lines denote the 95% limits of agreement (-6.015 to 6.004 breaths/min).

**paper_rr_method_agreement_bland_altman_signal_aware_safe_fixed_oof.png.** Bland-Altman plot for Conservative signal-aware safe gate on the internal 73-video set. The vertical axis shows predicted minus manual respiratory rate, and the horizontal axis shows the mean of predicted and manual respiratory rate. The solid line denotes the mean bias (-0.117 breaths/min), and dashed lines denote the 95% limits of agreement (-5.621 to 5.388 breaths/min).

## Claim Boundary

These Bland-Altman results can be used in the paper as internal method-agreement
evidence. They should not be used as an external validation result, and they do
not replace the default RR R2/MAE/RMSE table. After A/B annotation is complete
for the 119 included `split_all_use` external clips, the same agreement analysis
should be rerun on external paired predictions before making an external
agreement claim.
