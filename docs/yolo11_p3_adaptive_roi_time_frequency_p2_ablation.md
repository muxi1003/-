# YOLO11 P3/P2、Adaptive ROI and Time-Frequency Ablation

Date: 2026-07-21

## Decision

Keep the existing `paper_repro` pipeline as the authoritative default. Do not promote the P2 head, adaptive ROI, or automatic time-frequency count correction. The experimental code and outputs are retained only so the negative ablation can be reproduced.

The default behavior remains:

- existing production pose checkpoint
- fixed 20-pixel bilateral circular ROI
- validated adaptive left/right fusion
- time-domain peak count

## Model ablation

All model rows use the same complete training set, three epochs, seed 0, 640-pixel input, and the same 1497-image validation set.

| Variant | Pose mAP50-95 | Box mAP50-95 | GFLOPs | Validation inference ms/image | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| YOLO11n-Pose | 0.994805 | 0.869209 | 6.5598 | 1.958 | Keep baseline |
| P3-CBAM | 0.994652 | 0.874871 | 6.5660 | 1.864 | Detector-only candidate, not promoted |
| P2/4 Pose Head | 0.994495 | 0.838822 | 10.0635 | 3.162 | Roll back |

The P2 head reduced Pose mAP50-95 by 0.00031 and Box mAP50-95 by 0.03039 while adding 3.5037 GFLOPs. Its batch-1 latency increased from 8.11 ms to 13.95 ms in the unified benchmark.

## RR ablation on 73 videos

| Method | RR R2 | MAE | RMSE | Exact count | Mean count accuracy | Within one | Error >= 2 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default `paper_repro` | **0.946595** | **1.6917** | **3.0064** | **51/73** | **96.67%** | **73/73** | **0** | Keep |
| Time-frequency consensus | 0.911478 | 2.2841 | 3.8707 | 47/73 | 95.56% | 70/73 | 3 | Roll back |
| P3-CBAM + adaptive ROI + peaks | 0.908784 | 2.5749 | 3.9291 | 40/73 | 94.95% | 71/73 | 2 | Roll back |
| P3-CBAM + adaptive ROI + time-frequency | 0.888951 | 2.9451 | 4.3353 | 37/73 | 94.22% | 70/73 | 3 | Roll back |
| Keypoint tracking + motion registration | 0.913996 | 2.3738 | 3.8152 | 43/73 | 95.31% | 71/73 | 2 | Roll back |

Time-frequency consensus changed 15 videos: 4 improved and 11 worsened. The P3-CBAM/adaptive-ROI peak version changed 24 videos: 6 improved and 18 worsened. The full combination changed 30 videos: 7 improved, 22 worsened, and one retained the same absolute error.

The adaptive radius used `round(0.14 * inter_nostril_distance)` clipped to 12--32 pixels. Across 7489 frames, the median radius was 20 pixels and the mean was 20.56; 86 frames reached the lower bound and 141 reached the upper bound. Although scaling behaved as designed, changing the sampled thermal area destabilized more curves than it repaired.

The short clips do not provide endpoint phase information. A spectral rate and autocorrelation period can agree while the number of complete visible cycles is still one lower or higher. Therefore frequency evidence is useful as a diagnostic/confidence feature, but it is not reliable enough here to overwrite the integer time-domain count.

## Keypoint tracking and motion registration

The optional tracking stage applies a confidence-aware One Euro filter independently to the left and right nostril coordinates. Missing or low-confidence measurements carry the previous filtered state. When both nostrils are available, a bilateral similarity transform (translation, rotation, and scale) maps the raw nostril axis onto the tracked axis before the two circular ROIs are sampled. Nearest-neighbor warping preserves the discrete thermal palette used by the random-forest temperature mapper.

On the 73-video set, registration was applied to 7018 frames. Mean tracking displacement was 7.44 pixels and the mean per-video P95 displacement was 25.73 pixels. Counts changed in 12 videos: one improved (`ns211030`) and 11 worsened. RR R2 fell from 0.946595 to 0.913996, exact-count accuracy fell from 69.86% to 58.90%, and mean count accuracy fell from 96.67% to 95.31%. Tracking and registration therefore remain opt-in diagnostic features and are disabled by default.

Mean count accuracy is calculated per video and then averaged:

`mean(max(0, 1 - abs(predicted_count - truth_count) / truth_count)) * 100%`

This metric should be reported together with RR R2, MAE, RMSE, and exact-count accuracy; by itself it can look high when most errors are only one breath.

## Reproducibility artifacts

- `runs/pose_attention/full3_seed0_p2_comparison.csv`
- `runs/pose_attention/p2_full3_seed0/weights/best.pt`
- `Dataset_new/72video/al_images/paper_repro_20260721_method_ablation.csv`
- `Dataset_new/72video/al_images/paper_repro_timefreq_metrics.csv`
- `Dataset_new/72video/al_images/paper_repro_p3_adaptive_metrics.csv`
- `Dataset_new/72video/al_images/paper_repro_p3_adaptive_tf_metrics.csv`
- `Dataset_new/72video/al_images/paper_repro_tracking_registration_metrics.csv`
- `Dataset_new/72video/al_images/paper_repro_tracking_registration_summary.csv`

Every experimental prefix has 73 temperature CSVs, 73 curve CSVs, 73 curve PNGs, and 73 peak-review PNGs. The authoritative `paper_repro_summary.csv` predictions were not overwritten; its metrics file was refreshed only to add the two count-accuracy fields.
