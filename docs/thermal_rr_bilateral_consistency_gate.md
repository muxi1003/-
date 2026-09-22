# Bilateral Nostril Consistency Gate

Generated: 2026-07-09T11:03:18+00:00

## Purpose

This internal reliability layer asks whether the left and right nostril temperature curves carry compatible respiratory periodicity before a low-risk RR estimate is auto-reported. It is designed to improve the physiological interpretability of selective RR reporting, not to infer heat stress or animal health status.

## Frozen Candidate Rule

- Decision rule: existing RR-only auto-report candidate AND `bilateral_best_corr >= 0.20` AND `bilateral_fft_count_delta <= 1.0` AND `bilateral_amplitude_ratio >= 0.10`.
- The rule uses only prediction-time curve features and existing non-truth triage output.
- Manual reference RR is used only to evaluate internal performance after the decision is made.

## Internal Metrics

| subset | videos | coverage | rr_r2 | rr_mae_bpm | rr_rmse_bpm | exact_count | exact_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_safe_gate_predictions | 73 | 1.0 | 0.9519928084458789 | 1.2883965352770061 | 2.791503777074494 | 59/73 | 0.8082191780821918 |
| rr_only_auto_report_subset | 22 | 0.3013698630136986 | 0.9872464468201202 | 0.505827505957376 | 1.3782932155944636 | 20/22 | 0.9090909090909091 |
| bilateral_consistent_all_predictions | 55 | 0.7534246575342466 | 0.9596233910422156 | 1.164054127973602 | 2.585133277393999 | 45/55 | 0.8181818181818182 |
| bilateral_consistent_auto_report_subset | 19 | 0.2602739726027397 | 0.9909586546695432 | 0.34885290148447984 | 1.1594742034165855 | 18/19 | 0.9473684210526315 |
| auto_report_rejected_by_bilateral_gate | 3 | 0.0410958904109589 | 0.3466219918334471 | 1.5000000009523855 | 2.327373340628157 | 2/3 | 0.6666666666666666 |

## Manuscript-Safe Result

The bilateral-consistent auto-report subset reached n=19, coverage=0.260, RR R2=0.9910, MAE=0.349 breaths/min, RMSE=1.159 breaths/min, and exact=18/19 on the current internal set.

## Claim Boundary

Report this as an internal physiological-consistency reliability screen. Do not describe it as external validation, heat-stress diagnosis, welfare classification, disease detection, or true animal-identity evidence. The rule should be frozen before any independent external scoring.
