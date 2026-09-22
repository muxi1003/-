# External Thermal RR Tracking and Sequential-Fusion Literature Digest

Generated: 2026-07-11

Purpose: identify a defensible innovation route after the frozen internal method
failed on the provisional Jiufu Ranch external cohort. This digest records the
sources consulted before implementing the duration-normalized windowed
sequential-fusion probe.

## Primary Cattle RR Sources

1. Chen et al. (2025), *Respiratory rate detection of dairy cows based on
   infrared thermography in head movement scenarios*, Journal of Thermal
   Biology 130, 104154. DOI: https://doi.org/10.1016/j.jtherbio.2025.104154
   - Uses YOLOv8-Pose nostril keypoints, RGB-to-temperature random forest
     mapping, and bilateral nostril curve fusion for head movement.
   - Reports 246 thermal videos, RR R2 0.92, RMSE 3.53 bpm, and 96.3% average
     accuracy.
   - Directly supports bilateral thermal-signal fusion, but does not establish
     cross-farm performance for the current Jiufu acquisition domain.

2. Zhao et al. (2023), *Detection of Respiratory Rate of Dairy Cows Based on
   Infrared Thermography and Deep Learning*, Agriculture 13, 1939.
   DOI: https://doi.org/10.3390/agriculture13101939
   - Detects/tracks the nose, segments nostrils, maps false color to
     temperature, filters the curve, and uses sliding-window peak detection.
   - Supports retaining explicit window-level signal processing rather than
     treating an entire long clip as stationary.

3. Mantovani et al. (2024), *Predicting respiration rate in unrestrained dairy
   cows using image analysis and fast Fourier transform*, JDS Communications
   5, 310-316. DOI: https://doi.org/10.3168/jdsc.2023-0442
   - Uses frequency filtering and inverse FFT on flank image intensity.
   - Reports development R2 around 0.77 and external calf validation R2 0.73,
     while explicitly excluding videos affected by movement, lighting, or ROI
     occlusion.
   - Demonstrates the value of external validation and also the fragility of a
     single whole-window frequency estimate under uncontrolled conditions.

4. Wang et al. (2024), *Learning end-to-end respiratory rate prediction of
   dairy cows from RGB videos*, Journal of Dairy Science 107(11).
   DOI: https://doi.org/10.3168/jds.2023-24601
   - Uses VideoMAE to avoid error propagation across separately trained ROI and
     signal modules.
   - Reports MAE 2.58 bpm and RMSE 3.52 bpm.
   - Motivates an eventual learned temporal representation, but the current
     dataset is too small for an externally confirmatory end-to-end model.

5. Shu et al. (2024), *Non-contact respiration rate measurement of multiple
   cows in a free-stall barn using computer vision methods*, Computers and
   Electronics in Agriculture 218, 108678.
   DOI: https://doi.org/10.1016/j.compag.2024.108678
   - Combines cow/abdomen localization, optical flow, and multi-object
     processing in a commercial barn.
   - Supports motion-derived secondary signals and explicit tracking as future
     extensions when nostril temperature alone is unstable.

6. Sadeghi et al. (2024), *Non-Invasive Monitoring of Vital Signs in Calves
   Using Thermal Imaging Technology*. arXiv:2405.11532.
   URL: https://arxiv.org/abs/2405.11532
   - Uses Kernelised Correlation Filters for thermal ROI tracking and reports
     respiration MAPE 3.08%.
   - Supports a label-independent tracking-continuity layer before temperature
     signal extraction.

7. *A Non-Invasive Continuous Respiration Rate Monitoring Device for Dairy
   Cattle Under Commercial Farm Conditions* (2026).
   PubMed: https://pubmed.ncbi.nlm.nih.gov/41897960/
   - Uses multiscale periodicity detection, robust consensus, and rolling median
     reporting over longer intervals.
   - Reports 10-min rolling-median MAE 1.47 bpm, RMSE 1.92 bpm, and R2 0.96.
   - Provides direct cattle-specific support for multiscale candidate consensus
     and temporal aggregation rather than one estimate from one long segment.

## Transferable Physiological-Signal Methods

8. *Respiratory rate estimation from photoplethysmogram baseline wandering by
   harmonic analysis and sequential fusion* (2025), Biomedical Signal
   Processing and Control 100C, 107006.
   DOI: https://doi.org/10.1016/j.bspc.2024.107006
   - Scores respiratory fundamentals using harmonic power and then fuses
     estimates sequentially across windows to reduce frequency ambiguity.
   - Supports explicit harmonic/half-rate diagnostics plus window-to-window
     continuity, but the direct FFT implementation must be validated because it
     was unstable on the current thermal curves.

9. Li et al. (2024), *Bi-TTA: Bidirectional Test-Time Adapter for Remote
   Physiological Measurement*. arXiv:2409.17316.
   URL: https://arxiv.org/abs/2409.17316
   - Uses expert physiological priors for unlabeled test-time adaptation while
     stabilizing parameters against catastrophic forgetting.
   - Supports future label-free target-domain adaptation, but it is not yet
     justified for the current non-neural signal pipeline.

10. *Fully Test-Time rPPG Estimation via Synthetic Signal-Guided Feature
    Learning* (2024). arXiv:2407.13322.
    URL: https://arxiv.org/abs/2407.13322
    - Addresses cross-domain degradation in camera-derived physiological
      signals using unlabeled target data.
    - Supports treating the observed cross-farm failure as a domain-shift
      problem rather than retuning against external reference RR.

## Current Workspace Evidence

- Internal reference videos have median duration about 11 s and median RR about
  49 bpm.
- The provisional Jiufu external primary set has 94 clips, mostly about 30 s,
  with median RR about 30 bpm.
- The frozen whole-clip method produced RR R2 -0.8628 externally; one-count
  reference sensitivity and session aggregation did not explain the failure.
- External curves have higher missingness and interval irregularity and lower
  peak prominence/selection score than internal curves.
- Direct FFT, harmonic-sum FFT, harmonic-product spectrum, and simple
  autocorrelation estimators were all worse than the peak baseline on both
  cohorts. Frequency analysis therefore remains a diagnostic signal, not a
  replacement estimator in the next fixed probe.

## Innovation Selected for Implementation

**Duration-normalized quality-weighted sequential fusion**:

1. Derive a fixed subwindow length from the internal acquisition duration,
   without external RR labels.
2. Keep the existing frozen estimator for short clips.
3. For long clips, estimate RR in overlapping fixed-duration windows so head
   movement and missingness cannot contaminate the full 30-s interval at once.
4. Weight window estimates using label-independent curve amplitude,
   prominence, interval regularity, and missingness.
5. Stabilize adjacent clips from the same source long video using a fixed
   rolling-median shrinkage rule.
6. Preserve abstention/uncertainty and source-session clustered reporting.

This is a development innovation probe. Because the Jiufu labels have already
been inspected at cohort level, improvement on this cohort is not confirmatory
external evidence. A newly collected or untouched source-session cohort is
required for the final Q2-or-higher performance claim.

