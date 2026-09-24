# 190588 selected source-time windows: algorithm scoring

This is an exploratory within-source development result, not an independent cow or cross-farm test. The user selected windows 1, 2, and 6 after reviewing the 190588 recording. All three are from the same original MP4 and annotation round `L190588-P1` by one annotator.

## Inputs and methods

- Source-time bins: `000_030`, `030_060`, `180_210`; 264, 263, and 262 frames respectively.
- Reference: desktop export at `C:/Users/muxi/Desktop/实验/190588标注`, captured by SHA-256 in `manifest.json`. All three windows are complete with 18, 17, and 22 confirmed expiration-peak events and no unobservable intervals.
- Current project YOLO11n-Pose `models/best.pt` and BGR random forest `clf_model_RGB_20240906.pkl` were run on decoded frames of the original MP4, not the lossy viewing copies. Fixed nostril radius 20 px, keypoint confidence 0.5, temperature cutoff >20 C; no missing-ROI remeasurement.
- Four signal-rule arms are imported unchanged from `Experiment/05_方法核证/run.py`: C0 = framewise bilateral maximum; C1 = whole-curve quality choice; P0 = paper MAF3/peak-distance-6/prominence-0.05 rule; P1 = the frozen constrained peak package. All arms use the same freshly extracted temperatures.
- RR = `60 * predicted_count / 30` bpm. Predicted peak frames were mapped to playback time through `frame_time_map.csv`. The source-MP4 and browser-WebM maps matched for all 789 selected frames, with zero playback-time difference. Events were matched one-to-one within 0.30 s.
- This is **not** an end-to-end reproduction of the original authors' trained models.

## Results

| Arm | Counts, windows 1 / 2 / 6 | RR MAE, bpm | Exact counts | Event TP / FP / FN | Event F1 | R-squared* |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| C0P0 | 15 / 18 / 24 | 4.000 | 0/3 | 51 / 6 / 6 | 0.895 | 0.000 |
| C1P0 | 10 / 16 / 23 | 6.667 | 0/3 | 44 / 5 / 13 | 0.830 | -3.714 |
| C0P1 | 16 / 19 / 23 | 3.333 | 0/3 | 52 / 6 / 5 | 0.904 | 0.357 |
| C1P1 | 11 / 17 / 22 | 4.667 | 2/3 | 45 / 5 / 12 | 0.841 | -2.500 |

*R-squared is calculated across only three correlated windows of one cow. It is descriptive and is not evidence of generalization or a stable method ranking. The selected videos were not held out before method development.

C1P1 exactly counts windows 2 and 6, but undercounts window 1 by seven; its event F1 and MAE are worse than C0P0 on the three-window aggregate. C0P1 has the lowest exploratory MAE and highest event F1 here but zero exact-count windows. These data do not justify replacing the pre-specified four-arm decision based on the 47-window analysis.

`predictions.csv` contains all per-window scores, `predicted_events.csv` the predicted event times, `event_matches.csv` the TP/FP/FN assignments, `temperature_*.csv` the direct per-frame measurements, `curve_*_*.csv` the four-arm curves, and `manifest.json` the input and code hashes. The earlier sibling directory `190588_selected_126_20260924_v1` is an incomplete run stopped by a bookkeeping error before scoring; use only this v2 directory.
