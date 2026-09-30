# P3 frozen-output failure attribution (bounded post-hoc audit)

Date: 2026-09-29. This report uses exactly the eight cases locked in
`../20260929_p3_failure_diagnosis_protocol.md`. It reads frozen P3 predictions,
temperature tables, matched events and four original MP4s; it does not retrain,
change references, retune thresholds, or recompute scored predictions. The
input file hashes are in `input_hashes.json`; the four original MP4 hashes
were also rechecked against the frozen segment manifest. The script and
machine-readable observations are `../diagnose_p3_frozen_failures.py`,
`case_metrics.csv`, and `changed_reference_events.csv`.

## Overall result and concentration of failure

The strict reference set remains **24/53 fixed 30-second P3 windows**.
Against C0P0, C1P1 RR MAE is 4.00 vs 3.75 breaths/min; paired difference
is +0.25, 95% interval [-1.167, 1.833]. The two largest deteriorations,
`170131` and `221284`, each add +10 breaths/min absolute RR error. Together
they contribute +20 to the 24-window sum of paired absolute-error differences;
the other 22 windows sum to -14, yielding the observed net +6 and mean +0.25.
This is a concentration description, **not** a leave-two-out replacement
result or a reason to remove them from the analysis.

| Case | Reference | C0P0 -> C1P1 counts | RR absolute-error change | Matched timed events C0P0 -> C1P1 | Attribution from frozen arms and evidence |
|---|---:|---:|---:|---:|---|
| `170131_000_030` | 28 | 30 -> 21 | +10 | 27 -> 15 | Selection of left side drives loss: C1P0 is 20, whereas C0P1 stays 30. Left has a 26-frame gap at 0.00-2.92 s; 3 of 12 lost reference events are in/near it. The other 9 lost events have nearby left temperature but weak/non-matching selected-left peaks. |
| `221284_000_030` | 25 | 25 -> 20 | +10 | 24 -> 20 | Selection of right side drives loss: C1P0 is 20, whereas C0P1 is 26. Right has a 34-frame gap at 4.90-8.67 s; 3 of 4 lost events have zero right temperatures within +/-0.30 s and the fourth only 2/6 nearby frames. Left is available near all four. |
| `197250_030_060` | 26 | 32 -> 34 | +4 | 25 -> 26 | Peak rule worsens existing overcount: C0P1 rises 32 -> 35, adding one TP but two FPs. C1 reduces count by one; it does not offset P1's excess. |
| `211075_000_030` | 22 | 22 -> 24 | +4 | 19 -> 20 | Both modules independently make 23, combined 24. Baseline exact total already hides 3 FPs/3 FNs; C1P1 has 4 FPs/2 FNs. Count deterioration is not the same as loss of event recall. |
| `221381_000_030` | 16 | 19 -> 16 | -6 | 9 -> 7 | Exact count after selection is misleading: TP falls 9 -> 7 and FN rises 7 -> 9. FP/FN cancellation produces the RR improvement, not better event detection. |
| `211109_000_030` | 32 | 28 -> 30 | -4 | 17 -> 22 | Genuine timed-event gain in this case: TP +5, FP -3, FN -5. C0P1 alone reaches 24 TP but predicts 34; C1 selection partly counteracts it. This case shows P1 can help events without proving cohort RR MAE improvement. |

For `170131`, 13 timed reference events fall before 13 s. C0P0 outputs
13 peaks there, while C1P1 outputs four; the source image at 1.75 s still
shows the muzzle/right-side ROI and absent left temperature. By 7.00 and
12.61 s both temperatures exist, but the selected-left curve remains much
weaker than the framewise-max curve. For `221284`, original frames at
5.48/6.39/8.67 s show changing head orientation, with a left ROI visible in
the scored data while the selected right temperature is absent. The adaptive
fusion implementation interpolates its selected single-side series across
missing runs before peak finding. These observations support a **specific
signal-path failure**. They do not prove the anatomical correctness of any
ROI or that actual head motion, rather than keypoint instability, alone caused
the missing series. The coordinate-jump columns in `case_metrics.csv` are
proxies, not a head-motion ground truth.

## Two 36-count windows: limited evidence

User-stated whole-window counts are 36 each, but neither has reliable timed
reference events. They remain outside the 24-window main set and event F1.
`170020_030_060` is predicted 24 by C0P0 and 20 by C1P1; selected-left
temperature is present on 189/263 frames, with a 53-frame missing run at
0.34-6.27 s. The sampled original frame at 5.02 s does not show a usable
muzzle ROI, while 15.06 and 24.98 s do. `200792_060_090` is predicted
31/30; left temperature exists on only 37/263 frames but right on 263/263.
The three sampled frames show a muzzle in view, so a simple “whole nose out
of frame” explanation does not fit that case. Without independently timed
events, it is impossible to allocate the 5-6 missing counts to particular
phases, localization, mapping, or peak filtering. A 36/30 s claim implies
about 0.83 s per cycle, only about seven frames at this video's frame rate;
that is a sampling-resolution concern, not a demonstrated cause here.

## Decision and near-term paper work

1. Keep C0P0 as the P3 default comparison. Neither C1P1 nor either module
   has met the prespecified primary RR MAE gain criterion. Do not exclude
   these failure windows, tune on P3 and call it independent confirmation, or
   use the 36-count claims to synthesize event labels.
2. Draft the paper's methods, cohort/reference flow, four-arm results and
   limitations now, including P1/P2/P3 mixed outcomes and the 24/53 strict
   reference coverage. A claim of a validated RR improvement is unavailable.
3. A single later development hypothesis may be tested separately:
   prevent whole-window side selection from filling multi-second missing
   runs with artificial signal; use observed-side fallback or explicit
   abstention. Freeze the implementation **before** a new, independently
   annotated 30-second confirmation set. If the paired MAE gain criterion
   again fails, report it and stop algorithmic expansion for this manuscript.

This is a diagnostic stopping point, not an additional result table for
publication as if independently validated. The writing target clarified by
the user is to finish a paper **this year**; it is not a claim that formal
volume/issue pagination will be available this year.
