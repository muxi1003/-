# P3 frozen-output failure attribution protocol

Date: 2026-09-29. This is a bounded, post-hoc diagnostic of the immutable
`20260929_p3_four_arm_v1` outputs, not another method-selection experiment.
No weights, temperature mapping, ROI, fusion, peak thresholds, references,
or scored predictions may change. The strict main set remains 24/53 windows.

Cases are locked before inspecting per-event transitions or original frames:

- Four largest main-set C1P1 vs C0P0 RR absolute-error regressions (ties kept):
  `20230810T145443n170131_000_030`,
  `20230810T164658n221284_000_030`,
  `20230808T170359n197250_030_060`,
  `20230808T093014n211075_000_030`.
- One largest improvement, plus an improvement with a different peak-rule
  response among the -4 bpm ties: `20230810T164018n221381_000_030`,
  `20230810T071310n211109_000_030`. The second is purposive, not the
  uniquely second-largest improvement.
- The two user-claimed 36-count windows without timed reference peaks:
  `20230809T084030n170020_030_060`,
  `20230809T073104n200792_060_090`.

Read-only checks: decompose four-arm count differences; compare C0P0 and
C1P1 matched-reference event IDs; inspect availability of the selected side
within 0.30 s of changed events; inspect selected original frames and fixed
20px ROI geometry. A keypoint jump is only a coordinate proxy, not proof of
head motion. The two count-only windows have no event F1 or timed-event
attribution. No selection or hypothesis from this diagnostic can be scored
again on P3 as a fresh confirmation.

Stop after one pass over these eight cases. Output to
`20260929_p3_failure_diagnosis_v1/` with input hashes and a decision note.
