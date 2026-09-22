# Thermal RR Innovation Source Digest

Generated: 2026-07-08

Purpose: document the literature inputs used to guide the thermal respiratory-rate
innovation route in this workspace. This digest supports the current
algorithmic-engineering route and the new frame-motion-aware innovation probe.

## Online Sources Checked

1. Zhao et al. 2023. Detection of Respiratory Rate of Dairy Cows Based on
   Infrared Thermography and Deep Learning. Agriculture, 13(10), 1939.
   URL: https://www.mdpi.com/2077-0472/13/10/1939
   DOI: https://doi.org/10.3390/agriculture13101939
   Relevance: YOLOv8 thermal nose detection, nostril segmentation, color-to-
   temperature mapping, filtering, peak detection, and RR correlation reporting.

2. Sadeghi et al. 2024. Non-Invasive Monitoring of Vital Signs in Calves Using
   Thermal Imaging Technology. arXiv:2405.11532.
   URL: https://arxiv.org/abs/2405.11532
   DOI: https://doi.org/10.48550/arXiv.2405.11532
   Relevance: thermal imaging vital-sign monitoring with motion tracking and
   signal processing; supports adding an objective frame-motion/track-stability
   layer for head-movement scenarios.

3. Bhujel et al. 2024. Public Computer Vision Datasets for Precision Livestock
   Farming: A Systematic Survey. arXiv:2406.10628.
   URL: https://arxiv.org/abs/2406.10628
   DOI: https://doi.org/10.48550/arXiv.2406.10628
   Relevance: highlights limited high-quality annotated livestock CV datasets
   and missing contextual metadata as bottlenecks; supports external validation,
   metadata gates, and careful claim boundaries.

4. Melki et al. 2025. Uncertainty Guarantees on Automated Precision Weeding
   using Conformal Prediction. arXiv:2501.07185.
   URL: https://arxiv.org/abs/2501.07185
   DOI: https://doi.org/10.48550/arXiv.2501.07185
   Relevance: precision-agriculture trust and conformal prediction framing;
   supports reporting RR uncertainty intervals and selective auto-report/review
   decisions instead of only point accuracy.

## Local PDF Sources Inspected

The local folder `E:/real/learning/respiration literature` (displayed in the
desktop as `E:/real/学习/呼吸文献`) contains papers on end-to-end RGB-video RR,
machine-learning RR prediction with environmental/animal variables, FFT/image
analysis RR estimation, infrared thermography/deep learning, IoT vital-sign
devices, and Chinese RR monitoring reviews.

The user-provided PDF `E:/real/possible-combined-literature/1-s2.0-
S0306456525001111-main.pdf` (displayed in the desktop as
`E:/real/可能结合文献/...`) was inspected by metadata/text extraction. Its title is
`Respiratory rate detection of dairy cows based on infrared thermography in head
movement scenarios`. Relevance: directly motivates the new frame-motion-aware
head-movement proxy and the requirement that motion robustness claims be
validated rather than assumed.

## Workspace Actions Derived From These Sources

1. Keep the main claim as algorithmic RR estimation and reliability triage, not
   heat-stress/THI physiology, because synchronized per-video environment fields
   are unavailable.
2. Preserve truth-calibrated RR as an oracle upper bound only.
3. Add a frame-motion-aware probe from raw video frames to test whether objective
   motion features explain residual errors or improve residual correction.
4. Use the frame-motion result as an external-validation sampling stratum unless
   it later beats the safe gate under nested, prefix-group, and external tests.
5. Keep conformal RR intervals and selective auto-report/review as a trust layer
   for an algorithmic-engineering journal route.
