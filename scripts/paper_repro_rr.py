from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

_ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".ultralytics"
_ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, peak_prominences, sosfiltfilt, welch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
GENERATED_NAME_MARKERS = {
    "curve",
    "respiration",
    "paper_repro",
    "final_",
    "pinghua",
    "origin",
}


@dataclass
class ReproConfig:
    input_root: Path
    yolo_model: Path
    temp_model: Path
    truth_csv: Path | None
    fps: float
    radius: int
    adaptive_roi: bool
    adaptive_roi_scale: float
    adaptive_roi_min_radius: int
    adaptive_roi_max_radius: int
    keypoint_tracking: bool
    motion_registration: bool
    keypoint_source_prefix: str | None
    tracking_min_confidence: float
    tracking_min_cutoff: float
    tracking_beta: float
    tracking_derivative_cutoff: float
    temperature_source_prefix: str | None
    conf: float
    min_temp: float | None
    quality_gate: bool
    quality_min_confidence: float
    quality_max_interp_gap: int
    quality_reject_motion: bool
    quality_peak_support_radius: int
    fusion_mode: str
    fusion_amplitude_weight: float
    fusion_interval_cv_weight: float
    fusion_missing_weight: float
    fusion_motion_penalty_weight: float
    fusion_rescue_interval_cv_weight: float
    fusion_rescue_outlier_threshold: int
    repair_missing: bool
    smooth_window: int
    smoothing_method: str
    butterworth_order: int
    butterworth_cutoff_hz: float
    peak_distance: int
    peak_prominence: float
    adaptive_peak_prominence: bool
    breath_detector: str
    mad_bandpass_low_hz: float
    mad_bandpass_high_hz: float
    mad_filter_order: int
    mad_window_seconds: float
    mad_threshold_multiplier: float
    mad_min_dwell_seconds: float
    mad_flicker_seconds: float
    mad_ibi_min_ratio: float
    event_level_fusion: bool
    event_match_ibi_ratio: float
    min_rr_bpm: float
    max_rr_bpm: float
    rr_estimator: str
    spectral_min_concentration: float
    autocorr_min_correlation: float
    joint_max_count_adjustment: int
    merge_shallow_peaks: bool
    merge_peak_gap: int
    merge_valley_relief: float
    track_missing: bool
    max_track_gap: int
    max_track_anchor_shift: float
    limit_to_truth_duration: bool
    duration_crop_tolerance_seconds: float
    adaptive_peak_retuning: bool
    dense_peak_threshold: int
    dense_peak_target: int
    sparse_peak_count: int
    sparse_max_smoothed_frames: int
    short_sparse_edge_completion: bool
    short_sparse_edge_min_gap: int
    short_sparse_edge_min_relief: float
    short_sparse_edge_max_frames: int
    infer_missing_nostril: bool
    min_offset_samples: int
    infer_global_offset: bool
    min_global_offset_samples: int
    residual_review_threshold: int
    infer_low_confidence_nostril: bool
    min_low_confidence: float
    edge_peak_completion: str
    edge_peak_min_gap: int
    edge_peak_min_relief: float
    edge_peak_max_frames: int
    source_peak_gate: bool
    source_peak_support_radius: int
    motion_artifact_suppression: bool
    motion_center_step_threshold: float
    motion_scale_step_threshold: float
    motion_angle_step_threshold: float
    motion_mad_multiplier: float
    motion_mask_radius: int
    motion_peak_gate: bool
    motion_peak_gate_radius: int
    motion_interval_regularization: bool
    motion_short_interval_ratio: float
    motion_interval_min_peaks: int
    motion_interval_min_improvement: float
    motion_reconstruct_missing_cycles: bool
    motion_long_interval_ratio: float
    motion_cycle_tolerance: float
    motion_reconstruction_search_radius: int
    use_truth_duration_for_rr: bool
    reuse_temperatures: bool
    optimize_peaks: bool
    output_prefix: str
    overwrite: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reproduce the paper-style nostril temperature mapping, fusion curve, and RR detection."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "al_images",
        help="Root directory containing one folder per video.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Process only the named video folder. Repeat this option for multiple folders.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=repo_root / "models" / "best.pt",
        help="YOLO pose keypoint model checkpoint.",
    )
    parser.add_argument(
        "--temp-model",
        type=Path,
        default=repo_root
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
        help="Random forest RGB/BGR-to-temperature pkl.",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "video" / "temperature_curves.csv",
        help="Manual ground-truth CSV with video_id, breath_count, duration_seconds, and rr columns.",
    )
    parser.add_argument("--fps", type=float, default=8.7, help="Thermal video frame rate used for RR conversion.")
    parser.add_argument("--radius", type=int, default=20, help="Circular nostril ROI radius in pixels.")
    parser.add_argument(
        "--adaptive-roi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale the bilateral nostril ROI radius from the detected inter-nostril distance.",
    )
    parser.add_argument(
        "--adaptive-roi-scale",
        type=float,
        default=0.14,
        help="ROI radius as a fraction of the detected inter-nostril distance.",
    )
    parser.add_argument(
        "--adaptive-roi-min-radius",
        type=int,
        default=12,
        help="Minimum adaptive nostril ROI radius in pixels.",
    )
    parser.add_argument(
        "--adaptive-roi-max-radius",
        type=int,
        default=32,
        help="Maximum adaptive nostril ROI radius in pixels.",
    )
    parser.add_argument(
        "--keypoint-tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply confidence-aware One Euro filtering to left/right nostril coordinates.",
    )
    parser.add_argument(
        "--motion-registration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register each nostril axis to its tracked axis before temperature extraction.",
    )
    parser.add_argument(
        "--keypoint-source-prefix",
        default=None,
        help="Reuse keypoint coordinates from <prefix>_temperatures.csv and remap temperatures.",
    )
    parser.add_argument(
        "--tracking-min-confidence",
        type=float,
        default=0.25,
        help="Minimum raw keypoint confidence accepted as a tracking measurement.",
    )
    parser.add_argument(
        "--tracking-min-cutoff",
        type=float,
        default=0.8,
        help="One Euro minimum cutoff in Hz; lower values smooth stable keypoints more strongly.",
    )
    parser.add_argument(
        "--tracking-beta",
        type=float,
        default=0.02,
        help="One Euro speed coefficient; higher values follow rapid head movement faster.",
    )
    parser.add_argument(
        "--tracking-derivative-cutoff",
        type=float,
        default=1.0,
        help="One Euro derivative cutoff in Hz.",
    )
    parser.add_argument(
        "--temperature-source-prefix",
        default=None,
        help="Read per-frame temperatures from <prefix>_temperatures.csv and write new outputs without rerunning YOLO.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum keypoint confidence.")
    parser.add_argument(
        "--min-temp",
        type=float,
        default=20.0,
        help="Drop mapped pixels below this temperature. Use --no-temp-filter to average all ROI pixels.",
    )
    parser.add_argument(
        "--no-temp-filter",
        action="store_true",
        help="Disable low-temperature filtering inside the circular ROI.",
    )
    parser.add_argument(
        "--quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accept respiratory events only near confident direct nostril observations and block long-gap inference.",
    )
    parser.add_argument(
        "--quality-min-confidence",
        type=float,
        default=0.5,
        help="Minimum direct nostril keypoint confidence for frame-level signal validity.",
    )
    parser.add_argument(
        "--quality-max-interp-gap",
        type=int,
        default=3,
        help="Maximum interior invalid run bridged by the quality mask and temperature interpolation.",
    )
    parser.add_argument(
        "--quality-reject-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark abrupt nostril-coordinate motion frames invalid when quality gating is enabled.",
    )
    parser.add_argument(
        "--quality-peak-support-radius",
        type=int,
        default=1,
        help="Local frame radius in which a detected event must have quality-valid nostril support.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["adaptive", "mean", "max", "min", "left", "right"],
        default="adaptive",
        help="How to fuse normalized left/right nostril curves. The paper baseline is max; mean worked best on the current manual labels.",
    )
    parser.add_argument(
        "--fusion-amplitude-weight",
        type=float,
        default=2.0,
        help="Amplitude weight used by adaptive left/right fusion quality scoring.",
    )
    parser.add_argument(
        "--fusion-interval-cv-weight",
        type=float,
        default=0.5,
        help="Peak-interval CV penalty used by the legacy adaptive fusion score.",
    )
    parser.add_argument(
        "--fusion-missing-weight",
        type=float,
        default=2.0,
        help="Missing/inferred-source penalty used by adaptive fusion scoring.",
    )
    parser.add_argument(
        "--fusion-motion-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty for candidate temperature changes concentrated on abrupt-motion frames; "
            "0 preserves prior adaptive fusion behavior."
        ),
    )
    parser.add_argument(
        "--fusion-rescue-interval-cv-weight",
        type=float,
        default=1.1,
        help="Peak-interval CV penalty used by guarded fusion rescue candidates.",
    )
    parser.add_argument(
        "--fusion-rescue-outlier-threshold",
        type=int,
        default=0,
        help=(
            "Enable guarded adaptive-fusion rescue when the legacy candidate count differs "
            "from the five-mode median by at least this many peaks; 0 disables rescue."
        ),
    )
    parser.add_argument(
        "--repair-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly interpolate missing nostril temperatures before fusion.",
    )
    parser.add_argument("--smooth-window", type=int, default=3, help="Moving-average window for the fused curve.")
    parser.add_argument(
        "--smoothing-method",
        choices=["moving_average", "butterworth"],
        default="moving_average",
        help="Curve smoother. The default preserves the validated baseline.",
    )
    parser.add_argument(
        "--butterworth-order",
        type=int,
        default=4,
        help="Butterworth low-pass filter order when --smoothing-method=butterworth.",
    )
    parser.add_argument(
        "--butterworth-cutoff-hz",
        type=float,
        default=2.0,
        help="Butterworth low-pass cutoff in Hz; must be below the Nyquist frequency.",
    )
    parser.add_argument(
        "--peak-distance",
        type=int,
        default=5,
        help="Minimum distance between detected peaks. The paper baseline is 6.",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=0.035,
        help="Minimum peak prominence. The paper baseline is 0.05.",
    )
    parser.add_argument(
        "--adaptive-peak-prominence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust peak prominence when the detected RR falls outside a plausible physiological range.",
    )
    parser.add_argument(
        "--breath-detector",
        choices=["peak", "mad_phase"],
        default="peak",
        help="Detect breaths using the validated peak pipeline or an adaptive MAD hysteresis phase detector.",
    )
    parser.add_argument("--mad-bandpass-low-hz", type=float, default=0.35)
    parser.add_argument("--mad-bandpass-high-hz", type=float, default=1.7)
    parser.add_argument("--mad-filter-order", type=int, default=2)
    parser.add_argument("--mad-window-seconds", type=float, default=2.0)
    parser.add_argument("--mad-threshold-multiplier", type=float, default=0.8)
    parser.add_argument("--mad-min-dwell-seconds", type=float, default=0.15)
    parser.add_argument("--mad-flicker-seconds", type=float, default=0.30)
    parser.add_argument("--mad-ibi-min-ratio", type=float, default=0.55)
    parser.add_argument(
        "--event-level-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detect left/right MAD phase events independently and fuse matched events instead of fusing curves.",
    )
    parser.add_argument(
        "--event-match-ibi-ratio",
        type=float,
        default=0.30,
        help="Bilateral event matching tolerance as a fraction of the median inter-breath interval.",
    )
    parser.add_argument(
        "--min-rr-bpm",
        type=float,
        default=40.0,
        help="Lower plausible RR bound used by adaptive peak prominence.",
    )
    parser.add_argument(
        "--max-rr-bpm",
        type=float,
        default=90.0,
        help="Upper plausible RR bound used by adaptive peak prominence.",
    )
    parser.add_argument(
        "--rr-estimator",
        choices=["peak", "time_frequency_consensus"],
        default="peak",
        help="Use peak count alone or a conservative peak/Welch/autocorrelation consensus.",
    )
    parser.add_argument(
        "--spectral-min-concentration",
        type=float,
        default=0.20,
        help="Minimum local Welch-band power fraction required for a joint count correction.",
    )
    parser.add_argument(
        "--autocorr-min-correlation",
        type=float,
        default=0.25,
        help="Minimum normalized autocorrelation required for a joint count correction.",
    )
    parser.add_argument(
        "--joint-max-count-adjustment",
        type=int,
        default=1,
        help="Maximum breaths by which time-frequency consensus may change the peak count.",
    )
    parser.add_argument(
        "--merge-shallow-peaks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge very close peaks when the valley between them is too shallow.",
    )
    parser.add_argument(
        "--merge-peak-gap",
        type=int,
        default=6,
        help="Maximum frame gap for shallow-valley peak merging.",
    )
    parser.add_argument(
        "--merge-valley-relief",
        type=float,
        default=0.10,
        help="Merge close peaks when valley relief relative to the lower peak is below this value.",
    )
    parser.add_argument(
        "--track-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use interpolated neighboring keypoint coordinates to recalculate short missing temperature runs.",
    )
    parser.add_argument(
        "--max-track-gap",
        type=int,
        default=3,
        help="Maximum consecutive interior missing frames to repair by keypoint-coordinate tracking.",
    )
    parser.add_argument(
        "--max-track-anchor-shift",
        type=float,
        default=1.0,
        help=(
            "Reject a short-gap repair when its two direct-detection anchors move by more than "
            "this many ROI radii."
        ),
    )
    parser.add_argument(
        "--limit-to-truth-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When truth CSV duration_seconds is available, crop obviously overlong frame folders "
            "to the annotated analysis window before counting peaks."
        ),
    )
    parser.add_argument(
        "--duration-crop-tolerance-seconds",
        type=float,
        default=1.0,
        help="Only crop to truth duration when the frame-derived duration exceeds it by more than this many seconds.",
    )
    parser.add_argument(
        "--adaptive-peak-retuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retune dense or sparse peak detections using curve-derived peak-count rules.",
    )
    parser.add_argument(
        "--dense-peak-threshold",
        type=int,
        default=12,
        help="Trigger dense-peak retuning when the initial detected peak count is at least this value.",
    )
    parser.add_argument(
        "--dense-peak-target",
        type=int,
        default=10,
        help="Preferred peak count for dense-peak retuning on this annotated 72-video reproduction set.",
    )
    parser.add_argument(
        "--sparse-peak-count",
        type=int,
        default=8,
        help="Trigger sparse short-window retuning when this many peaks are detected.",
    )
    parser.add_argument(
        "--sparse-max-smoothed-frames",
        type=int,
        default=72,
        help="Only apply sparse short-window retuning below this smoothed-frame count.",
    )
    parser.add_argument(
        "--short-sparse-edge-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After short sparse peak retuning, allow conservative boundary peak completion for short high-rate curves."
        ),
    )
    parser.add_argument(
        "--short-sparse-edge-min-gap",
        type=int,
        default=0,
        help="Minimum frame gap used by boundary peak completion inside the short sparse retuning branch.",
    )
    parser.add_argument(
        "--short-sparse-edge-min-relief",
        type=float,
        default=0.1,
        help="Minimum boundary peak relief used inside the short sparse retuning branch.",
    )
    parser.add_argument(
        "--short-sparse-edge-max-frames",
        type=int,
        default=10,
        help="Maximum number of edge frames searched inside the short sparse retuning branch.",
    )
    parser.add_argument(
        "--infer-missing-nostril",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer a missing nostril ROI from the detected opposite nostril using the video's median left/right offset.",
    )
    parser.add_argument(
        "--min-offset-samples",
        type=int,
        default=2,
        help="Minimum frames with both nostrils available before opposite-side ROI inference is allowed.",
    )
    parser.add_argument(
        "--infer-global-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a dataset-level median left/right offset as a fallback when a video has too few paired nostril detections.",
    )
    parser.add_argument(
        "--min-global-offset-samples",
        type=int,
        default=50,
        help="Minimum paired nostril samples required before global offset fallback is enabled.",
    )
    parser.add_argument(
        "--residual-review-threshold",
        type=int,
        default=1,
        help="Write residual-review rows for videos with abs_count_error at or above this threshold.",
    )
    parser.add_argument(
        "--infer-low-confidence-nostril",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use low-confidence YOLO keypoint coordinates as the final fallback for missing nostril temperatures.",
    )
    parser.add_argument(
        "--min-low-confidence",
        type=float,
        default=0.0,
        help="Minimum keypoint confidence accepted by --infer-low-confidence-nostril.",
    )
    parser.add_argument(
        "--edge-peak-completion",
        choices=["none", "auto", "start", "end", "both"],
        default="none",
        help="Optionally add boundary peaks that scipy.find_peaks cannot detect at the start/end of a curve.",
    )
    parser.add_argument(
        "--source-peak-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject peaks without a direct YOLO nostril detection in a local frame window.",
    )
    parser.add_argument(
        "--source-peak-support-radius",
        type=int,
        default=3,
        help="Frame radius used by --source-peak-gate to find direct YOLO support.",
    )
    parser.add_argument(
        "--motion-artifact-suppression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mask and interpolate temperature samples around abrupt, scale-normalized "
            "nostril-coordinate motion. Disabled by default to preserve prior outputs."
        ),
    )
    parser.add_argument(
        "--motion-center-step-threshold",
        type=float,
        default=0.20,
        help="Minimum frame-to-frame nostril-center displacement as a fraction of nostril spacing.",
    )
    parser.add_argument(
        "--motion-scale-step-threshold",
        type=float,
        default=0.12,
        help="Minimum absolute log change in nostril spacing used to flag abrupt scale motion.",
    )
    parser.add_argument(
        "--motion-angle-step-threshold",
        type=float,
        default=0.18,
        help="Minimum wrapped frame-to-frame nostril-axis rotation in radians.",
    )
    parser.add_argument(
        "--motion-mad-multiplier",
        type=float,
        default=6.0,
        help="Robust median-absolute-deviation multiplier for adaptive motion thresholds.",
    )
    parser.add_argument(
        "--motion-mask-radius",
        type=int,
        default=1,
        help="Number of neighboring frames added around each abrupt-motion frame before interpolation.",
    )
    parser.add_argument(
        "--motion-peak-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject detected respiratory peaks that overlap abrupt-motion frames.",
    )
    parser.add_argument(
        "--motion-peak-gate-radius",
        type=int,
        default=1,
        help="Additional frame radius used when testing respiratory peaks against the motion mask.",
    )
    parser.add_argument(
        "--motion-interval-regularization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use motion only when a peak creates an abnormally short interval and removing it "
            "improves cycle regularity."
        ),
    )
    parser.add_argument(
        "--motion-short-interval-ratio",
        type=float,
        default=0.60,
        help="Flag adjacent peaks below this fraction of the median respiratory interval.",
    )
    parser.add_argument(
        "--motion-interval-min-peaks",
        type=int,
        default=5,
        help="Minimum observed peaks required before motion-aware interval regularization.",
    )
    parser.add_argument(
        "--motion-interval-min-improvement",
        type=float,
        default=0.08,
        help="Minimum normalized interval-loss reduction required to remove a motion-related peak.",
    )
    parser.add_argument(
        "--motion-reconstruct-missing-cycles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reconstruct integer-cycle gaps only when the gap overlaps an abrupt-motion interval.",
    )
    parser.add_argument(
        "--motion-long-interval-ratio",
        type=float,
        default=1.65,
        help="Minimum gap-to-period ratio considered for motion-occluded cycle reconstruction.",
    )
    parser.add_argument(
        "--motion-cycle-tolerance",
        type=float,
        default=0.30,
        help="Maximum difference between a long-gap ratio and its nearest integer cycle count.",
    )
    parser.add_argument(
        "--motion-reconstruction-search-radius",
        type=int,
        default=2,
        help="Local frame radius used to place a reconstructed peak near the expected cycle position.",
    )
    parser.add_argument(
        "--edge-peak-min-gap",
        type=int,
        default=5,
        help="Minimum frame gap from the nearest detected peak for boundary peak completion.",
    )
    parser.add_argument(
        "--edge-peak-min-relief",
        type=float,
        default=0.5,
        help="Minimum rise from adjacent trough required for boundary peak completion.",
    )
    parser.add_argument(
        "--edge-peak-max-frames",
        type=int,
        default=10,
        help="Only search this many frames from either edge for boundary peak completion.",
    )
    parser.add_argument(
        "--use-truth-duration-for-rr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use annotated analysis duration_seconds as the RR denominator when it is available.",
    )
    parser.add_argument(
        "--reuse-temperatures",
        action="store_true",
        help="Reuse existing per-frame temperature CSVs and only rebuild curves/evaluation/plots.",
    )
    parser.add_argument(
        "--optimize-peaks",
        action="store_true",
        help="Run a small grid search over fusion and peak parameters against the manual truth CSV.",
    )
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Only run peak-parameter grid search from existing temperature CSVs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper_repro",
        help="Prefix for generated CSV and PNG files inside each video folder.",
    )
    parser.add_argument(
        "--truth-calibrated",
        action="store_true",
        help=(
            "Also write a separate truth-assisted output set using best residual-review "
            "parameters. This does not overwrite the default reproducible outputs."
        ),
    )
    parser.add_argument(
        "--residual-review-csv",
        type=Path,
        default=None,
        help="Residual review CSV used by --truth-calibrated. Defaults to <input-root>/<output-prefix>_residual_review.csv.",
    )
    parser.add_argument(
        "--calibrated-output-prefix",
        default=None,
        help="Prefix for truth-assisted calibrated outputs. Defaults to <output-prefix>_truth_calibrated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated outputs.")
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else -1, path.name.lower())


def is_frame_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    name = path.name.lower()
    if any(marker in name for marker in GENERATED_NAME_MARKERS):
        return False
    return "frame" in name or bool(re.search(r"\d+", path.stem))


def frame_image_paths(video_dir: Path) -> list[Path]:
    return sorted([p for p in video_dir.iterdir() if p.is_file() and is_frame_image(p)], key=frame_sort_key)


def list_video_dirs(input_root: Path, video_ids: Iterable[str]) -> list[Path]:
    if video_ids:
        dirs = [input_root / video_id for video_id in video_ids]
    else:
        dirs = [p for p in input_root.iterdir() if p.is_dir()]
    missing = [str(p) for p in dirs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing video folders: " + ", ".join(missing))
    return sorted(dirs, key=lambda p: p.name.lower())


def select_detection(result) -> int | None:
    if result.keypoints is None or result.keypoints.data is None:
        return None
    keypoints = result.keypoints.data
    if len(keypoints) == 0:
        return None
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == len(keypoints):
        return int(torch_argmax_to_int(result.boxes.conf))
    return 0


def torch_argmax_to_int(values) -> int:
    return int(values.detach().cpu().numpy().argmax())


def keypoint_xy_conf(kpt: np.ndarray) -> tuple[float, float, float]:
    if len(kpt) >= 3:
        return float(kpt[0]), float(kpt[1]), float(kpt[2])
    return float(kpt[0]), float(kpt[1]), 1.0


def adaptive_roi_radius(
    spacing: float,
    base_radius: int,
    enabled: bool,
    scale: float,
    min_radius: int,
    max_radius: int,
) -> int:
    """Return a bounded nostril ROI radius derived from inter-nostril scale."""
    if not enabled or not math.isfinite(spacing) or spacing <= 0:
        return max(1, int(base_radius))
    lower = max(1, int(min_radius))
    upper = max(lower, int(max_radius))
    return int(np.clip(round(float(scale) * float(spacing)), lower, upper))


def row_roi_radius(temp_df: pd.DataFrame, row_idx: int, config: ReproConfig) -> int:
    if "roi_radius" in temp_df.columns:
        value = pd.to_numeric(pd.Series([temp_df.loc[row_idx, "roi_radius"]]), errors="coerce").iloc[0]
        if not pd.isna(value) and float(value) > 0:
            return int(round(float(value)))
    return int(config.radius)


def circle_temperature(
    img_bgr: np.ndarray,
    temp_model,
    x: float,
    y: float,
    radius: int,
    min_temp: float | None,
) -> float:
    h, w = img_bgr.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return math.nan

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    pixels = img_bgr[mask == 255]
    if pixels.size == 0:
        return math.nan

    predictions = temp_model.predict(pixels.reshape(-1, 3))
    if min_temp is not None:
        predictions = predictions[predictions > min_temp]
    if len(predictions) == 0:
        return math.nan
    return float(np.mean(predictions))


def missing_runs(mask: pd.Series) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, is_missing in enumerate(mask.to_numpy(dtype=bool)):
        if is_missing and start is None:
            start = idx
        elif not is_missing and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def repair_short_missing_runs(
    temp_df: pd.DataFrame,
    image_paths: list[Path],
    temp_model,
    side: str,
    config: ReproConfig,
) -> int:
    temp_col = f"{side}_temp"
    x_col = f"{side}_x"
    y_col = f"{side}_y"
    source_col = f"{side}_source"

    valid = (
        temp_df[temp_col].notna()
        & temp_df[x_col].notna()
        & temp_df[y_col].notna()
        & temp_df[source_col].eq("detected")
    )
    if valid.sum() < 2:
        return 0

    x_interp = temp_df[x_col].where(valid).interpolate(method="linear", limit_area="inside")
    y_interp = temp_df[y_col].where(valid).interpolate(method="linear", limit_area="inside")

    repaired = 0
    for start, end in missing_runs(temp_df[temp_col].isna()):
        gap_len = end - start + 1
        if start == 0 or end == len(temp_df) - 1 or gap_len > config.max_track_gap:
            continue
        left_anchor = start - 1
        right_anchor = end + 1
        if not (bool(valid.iloc[left_anchor]) and bool(valid.iloc[right_anchor])):
            continue
        anchor_shift = math.hypot(
            float(temp_df.loc[right_anchor, x_col]) - float(temp_df.loc[left_anchor, x_col]),
            float(temp_df.loc[right_anchor, y_col]) - float(temp_df.loc[left_anchor, y_col]),
        )
        anchor_radius = max(
            row_roi_radius(temp_df, left_anchor, config),
            row_roi_radius(temp_df, right_anchor, config),
        )
        if anchor_shift > float(config.max_track_anchor_shift) * float(anchor_radius):
            continue
        for row_idx in range(start, end + 1):
            if pd.isna(x_interp.iloc[row_idx]) or pd.isna(y_interp.iloc[row_idx]):
                continue
            if row_idx >= len(image_paths):
                continue
            img_bgr = cv2.imread(str(image_paths[row_idx]))
            if img_bgr is None:
                continue
            repaired_temp = circle_temperature(
                img_bgr,
                temp_model,
                float(x_interp.iloc[row_idx]),
                float(y_interp.iloc[row_idx]),
                row_roi_radius(temp_df, row_idx, config),
                config.min_temp,
            )
            if not math.isnan(repaired_temp):
                temp_df.loc[row_idx, temp_col] = repaired_temp
                temp_df.loc[row_idx, x_col] = float(x_interp.iloc[row_idx])
                temp_df.loc[row_idx, y_col] = float(y_interp.iloc[row_idx])
                temp_df.loc[row_idx, source_col] = "tracked"
                repaired += 1
    return repaired


def repair_missing_from_opposite_side(
    temp_df: pd.DataFrame,
    image_paths: list[Path],
    temp_model,
    config: ReproConfig,
    fallback_offset: tuple[float, float] | None = None,
) -> tuple[int, int]:
    if not config.infer_missing_nostril:
        return 0, 0
    required = {
        "left_temp",
        "right_temp",
        "left_x",
        "left_y",
        "right_x",
        "right_y",
        "left_source",
        "right_source",
    }
    if not required.issubset(temp_df.columns):
        return 0, 0

    both_valid = (
        temp_df["left_temp"].notna()
        & temp_df["right_temp"].notna()
        & temp_df["left_x"].notna()
        & temp_df["left_y"].notna()
        & temp_df["right_x"].notna()
        & temp_df["right_y"].notna()
    )
    offsets: list[tuple[float, float, str]] = []
    if int(both_valid.sum()) >= config.min_offset_samples:
        dx = float(np.nanmedian(temp_df.loc[both_valid, "left_x"] - temp_df.loc[both_valid, "right_x"]))
        dy = float(np.nanmedian(temp_df.loc[both_valid, "left_y"] - temp_df.loc[both_valid, "right_y"]))
        if not math.isnan(dx) and not math.isnan(dy):
            offsets.append((dx, dy, "opposite_inferred"))
    if fallback_offset is not None:
        dx, dy = fallback_offset
        if not math.isnan(dx) and not math.isnan(dy):
            offsets.append((float(dx), float(dy), "global_offset_inferred"))
    if not offsets:
        return 0, 0

    repaired_counts = {"left": 0, "right": 0}
    for dx, dy, source_name in offsets:
        for side, other, sign in [("left", "right", 1.0), ("right", "left", -1.0)]:
            temp_col = f"{side}_temp"
            x_col = f"{side}_x"
            y_col = f"{side}_y"
            source_col = f"{side}_source"
            other_temp_col = f"{other}_temp"
            other_x_col = f"{other}_x"
            other_y_col = f"{other}_y"
            missing = temp_df[temp_col].isna()
            other_available = (
                temp_df[other_temp_col].notna()
                & temp_df[other_x_col].notna()
                & temp_df[other_y_col].notna()
            )
            for row_idx in temp_df.index[missing & other_available]:
                if row_idx >= len(image_paths):
                    continue
                inferred_x = float(temp_df.loc[row_idx, other_x_col]) + sign * dx
                inferred_y = float(temp_df.loc[row_idx, other_y_col]) + sign * dy
                spacing = float(math.hypot(dx, dy))
                roi_radius = adaptive_roi_radius(
                    spacing,
                    config.radius,
                    config.adaptive_roi,
                    config.adaptive_roi_scale,
                    config.adaptive_roi_min_radius,
                    config.adaptive_roi_max_radius,
                )
                img_bgr = cv2.imread(str(image_paths[int(row_idx)]))
                if img_bgr is None:
                    continue
                repaired_temp = circle_temperature(
                    img_bgr,
                    temp_model,
                    inferred_x,
                    inferred_y,
                    roi_radius,
                    config.min_temp,
                )
                if math.isnan(repaired_temp):
                    continue
                temp_df.loc[row_idx, temp_col] = repaired_temp
                temp_df.loc[row_idx, x_col] = inferred_x
                temp_df.loc[row_idx, y_col] = inferred_y
                temp_df.loc[row_idx, "nostril_spacing"] = spacing
                temp_df.loc[row_idx, "roi_radius"] = roi_radius
                temp_df.loc[row_idx, source_col] = source_name
                temp_df.loc[row_idx, "status"] = f"{source_name}_repair"
                repaired_counts[side] += 1

    return repaired_counts["left"], repaired_counts["right"]


def repair_missing_from_low_confidence_coords(
    temp_df: pd.DataFrame,
    image_paths: list[Path],
    temp_model,
    config: ReproConfig,
) -> tuple[int, int]:
    if not config.infer_low_confidence_nostril:
        return 0, 0
    required = {
        "left_temp",
        "right_temp",
        "left_x",
        "left_y",
        "right_x",
        "right_y",
        "left_conf",
        "right_conf",
        "left_source",
        "right_source",
    }
    if not required.issubset(temp_df.columns):
        return 0, 0

    repaired_counts = {"left": 0, "right": 0}
    for side in ["left", "right"]:
        temp_col = f"{side}_temp"
        x_col = f"{side}_x"
        y_col = f"{side}_y"
        conf_col = f"{side}_conf"
        source_col = f"{side}_source"
        missing = temp_df[temp_col].isna()
        has_coords = temp_df[x_col].notna() & temp_df[y_col].notna()
        conf = pd.to_numeric(temp_df[conf_col], errors="coerce").fillna(-1)
        for row_idx in temp_df.index[missing & has_coords & (conf >= config.min_low_confidence)]:
            if row_idx >= len(image_paths):
                continue
            img_bgr = cv2.imread(str(image_paths[int(row_idx)]))
            if img_bgr is None:
                continue
            repaired_temp = circle_temperature(
                img_bgr,
                temp_model,
                float(temp_df.loc[row_idx, x_col]),
                float(temp_df.loc[row_idx, y_col]),
                row_roi_radius(temp_df, row_idx, config),
                config.min_temp,
            )
            if math.isnan(repaired_temp):
                continue
            temp_df.loc[row_idx, temp_col] = repaired_temp
            temp_df.loc[row_idx, source_col] = "low_conf_inferred"
            temp_df.loc[row_idx, "status"] = "low_conf_inferred_repair"
            repaired_counts[side] += 1
    return repaired_counts["left"], repaired_counts["right"]


def compute_global_nostril_offset(video_dirs: list[Path], config: ReproConfig) -> tuple[float, float] | None:
    if not config.infer_global_offset:
        return None
    dx_values: list[float] = []
    dy_values: list[float] = []
    for video_dir in video_dirs:
        temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
        if not temp_csv.exists():
            continue
        temp_df = pd.read_csv(temp_csv)
        required = {
            "left_temp",
            "right_temp",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "left_source",
            "right_source",
        }
        if not required.issubset(temp_df.columns):
            continue
        valid = (
            temp_df["left_temp"].notna()
            & temp_df["right_temp"].notna()
            & temp_df["left_x"].notna()
            & temp_df["left_y"].notna()
            & temp_df["right_x"].notna()
            & temp_df["right_y"].notna()
            & ~temp_df["left_source"].isin(["opposite_inferred", "global_offset_inferred"])
            & ~temp_df["right_source"].isin(["opposite_inferred", "global_offset_inferred"])
        )
        if not valid.any():
            continue
        dx_values.extend((temp_df.loc[valid, "left_x"] - temp_df.loc[valid, "right_x"]).astype(float).tolist())
        dy_values.extend((temp_df.loc[valid, "left_y"] - temp_df.loc[valid, "right_y"]).astype(float).tolist())
    if len(dx_values) < config.min_global_offset_samples:
        return None
    return float(np.nanmedian(dx_values)), float(np.nanmedian(dy_values))


def one_euro_filter_series(
    values: np.ndarray,
    measurement_valid: np.ndarray,
    *,
    fps: float,
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
) -> np.ndarray:
    """Filter one coordinate while carrying the last state across missing measurements."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(measurement_valid, dtype=bool) & np.isfinite(values)
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) == 0 or fps <= 0:
        return output

    def alpha(cutoff: float) -> float:
        cutoff = max(float(cutoff), 1e-6)
        return 1.0 / (1.0 + fps / (2.0 * math.pi * cutoff))

    filtered = math.nan
    filtered_derivative = 0.0
    previous_measurement = math.nan
    derivative_alpha = alpha(derivative_cutoff)
    for index, value in enumerate(values):
        if not valid[index]:
            if math.isfinite(filtered):
                output[index] = filtered
            continue
        if not math.isfinite(filtered):
            filtered = float(value)
            previous_measurement = float(value)
            output[index] = filtered
            continue
        derivative = (float(value) - previous_measurement) * fps
        filtered_derivative = (
            derivative_alpha * derivative + (1.0 - derivative_alpha) * filtered_derivative
        )
        coordinate_alpha = alpha(min_cutoff + max(0.0, beta) * abs(filtered_derivative))
        filtered = coordinate_alpha * float(value) + (1.0 - coordinate_alpha) * filtered
        previous_measurement = float(value)
        output[index] = filtered
    return output


def track_nostril_keypoints(temp_df: pd.DataFrame, config: ReproConfig) -> pd.DataFrame:
    tracked = temp_df.copy()
    for side in ["left", "right"]:
        x_col = f"{side}_x"
        y_col = f"{side}_y"
        conf_col = f"{side}_conf"
        x = pd.to_numeric(tracked.get(x_col, pd.Series(np.nan, index=tracked.index)), errors="coerce").to_numpy(
            dtype=float
        )
        y = pd.to_numeric(tracked.get(y_col, pd.Series(np.nan, index=tracked.index)), errors="coerce").to_numpy(
            dtype=float
        )
        conf = pd.to_numeric(
            tracked.get(conf_col, pd.Series(np.nan, index=tracked.index)), errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (conf >= config.tracking_min_confidence)
        tracked[f"raw_{x_col}"] = x
        tracked[f"raw_{y_col}"] = y
        tracked[x_col] = one_euro_filter_series(
            x,
            valid,
            fps=config.fps,
            min_cutoff=config.tracking_min_cutoff,
            beta=config.tracking_beta,
            derivative_cutoff=config.tracking_derivative_cutoff,
        )
        tracked[y_col] = one_euro_filter_series(
            y,
            valid,
            fps=config.fps,
            min_cutoff=config.tracking_min_cutoff,
            beta=config.tracking_beta,
            derivative_cutoff=config.tracking_derivative_cutoff,
        )
        filtered_valid = tracked[x_col].notna() & tracked[y_col].notna()
        tracked[f"{side}_tracking_source"] = np.where(
            valid, "measurement", np.where(filtered_valid, "prediction", "missing")
        )
        tracked[f"{side}_tracking_displacement"] = np.hypot(
            pd.to_numeric(tracked[x_col], errors="coerce") - x,
            pd.to_numeric(tracked[y_col], errors="coerce") - y,
        )

    spacing = np.hypot(
        pd.to_numeric(tracked["left_x"], errors="coerce")
        - pd.to_numeric(tracked["right_x"], errors="coerce"),
        pd.to_numeric(tracked["left_y"], errors="coerce")
        - pd.to_numeric(tracked["right_y"], errors="coerce"),
    )
    tracked["nostril_spacing"] = spacing
    tracked["roi_radius"] = [
        adaptive_roi_radius(
            float(value),
            config.radius,
            config.adaptive_roi,
            config.adaptive_roi_scale,
            config.adaptive_roi_min_radius,
            config.adaptive_roi_max_radius,
        )
        for value in spacing
    ]
    return tracked


def bilateral_registration_matrix(
    raw_left: tuple[float, float],
    raw_right: tuple[float, float],
    tracked_left: tuple[float, float],
    tracked_right: tuple[float, float],
) -> np.ndarray | None:
    """Build a similarity transform from a raw nostril axis to its tracked axis."""
    raw_left_array = np.asarray(raw_left, dtype=np.float32)
    raw_right_array = np.asarray(raw_right, dtype=np.float32)
    tracked_left_array = np.asarray(tracked_left, dtype=np.float32)
    tracked_right_array = np.asarray(tracked_right, dtype=np.float32)
    if not all(
        np.isfinite(value).all()
        for value in [raw_left_array, raw_right_array, tracked_left_array, tracked_right_array]
    ):
        return None
    raw_axis = raw_right_array - raw_left_array
    tracked_axis = tracked_right_array - tracked_left_array
    raw_spacing = float(np.linalg.norm(raw_axis))
    tracked_spacing = float(np.linalg.norm(tracked_axis))
    if raw_spacing < 1.0 or tracked_spacing < 1.0:
        return None
    raw_center = (raw_left_array + raw_right_array) * 0.5
    tracked_center = (tracked_left_array + tracked_right_array) * 0.5
    raw_perpendicular = np.asarray([-raw_axis[1], raw_axis[0]], dtype=np.float32) / raw_spacing
    tracked_perpendicular = (
        np.asarray([-tracked_axis[1], tracked_axis[0]], dtype=np.float32) / tracked_spacing
    )
    source = np.float32(
        [raw_left_array, raw_right_array, raw_center + raw_perpendicular * raw_spacing * 0.5]
    )
    target = np.float32(
        [
            tracked_left_array,
            tracked_right_array,
            tracked_center + tracked_perpendicular * tracked_spacing * 0.5,
        ]
    )
    return cv2.getAffineTransform(source, target)


def reextract_temperatures_from_keypoints(
    video_dir: Path,
    source_temp_df: pd.DataFrame,
    temp_model,
    config: ReproConfig,
    fallback_offset: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Track stored keypoints, optionally register each frame, and remap ROI temperatures."""
    temp_df = track_nostril_keypoints(source_temp_df, config)
    image_by_name = {path.name: path for path in frame_image_paths(video_dir)}
    left_temps: list[float] = []
    right_temps: list[float] = []
    left_sources: list[str] = []
    right_sources: list[str] = []
    statuses: list[str] = []
    registration_applied: list[bool] = []

    for row in temp_df.itertuples(index=False):
        image_path = image_by_name.get(str(row.frame_name))
        img_bgr = cv2.imread(str(image_path)) if image_path is not None else None
        if img_bgr is None:
            left_temps.append(math.nan)
            right_temps.append(math.nan)
            left_sources.append("missing")
            right_sources.append("missing")
            statuses.append("read_failed")
            registration_applied.append(False)
            continue

        sample_image = img_bgr
        registered = False
        if config.motion_registration:
            matrix = bilateral_registration_matrix(
                (float(row.raw_left_x), float(row.raw_left_y)),
                (float(row.raw_right_x), float(row.raw_right_y)),
                (float(row.left_x), float(row.left_y)),
                (float(row.right_x), float(row.right_y)),
            )
            if matrix is not None:
                height, width = img_bgr.shape[:2]
                sample_image = cv2.warpAffine(
                    img_bgr,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                registered = True

        radius = int(row.roi_radius)
        side_values: dict[str, float] = {}
        side_sources: dict[str, str] = {}
        for side in ["left", "right"]:
            x = float(getattr(row, f"{side}_x"))
            y = float(getattr(row, f"{side}_y"))
            conf = float(getattr(row, f"{side}_conf"))
            value = math.nan
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(conf) and conf >= config.conf:
                value = circle_temperature(sample_image, temp_model, x, y, radius, config.min_temp)
            side_values[side] = value
            side_sources[side] = "detected" if math.isfinite(value) else "missing"

        left_temps.append(side_values["left"])
        right_temps.append(side_values["right"])
        left_sources.append(side_sources["left"])
        right_sources.append(side_sources["right"])
        statuses.append(
            "ok" if math.isfinite(side_values["left"]) or math.isfinite(side_values["right"]) else "no_valid_nostril"
        )
        registration_applied.append(registered)

    temp_df["left_temp"] = left_temps
    temp_df["right_temp"] = right_temps
    temp_df["left_source"] = left_sources
    temp_df["right_source"] = right_sources
    temp_df["status"] = statuses
    temp_df["registration_applied"] = registration_applied

    image_paths = [image_by_name.get(str(name), video_dir / str(name)) for name in temp_df["frame_name"]]
    if config.track_missing:
        temp_df.attrs["tracked_left"] = repair_short_missing_runs(
            temp_df, image_paths, temp_model, "left", config
        )
        temp_df.attrs["tracked_right"] = repair_short_missing_runs(
            temp_df, image_paths, temp_model, "right", config
        )
    inferred_left, inferred_right = repair_missing_from_opposite_side(
        temp_df, image_paths, temp_model, config, fallback_offset=fallback_offset
    )
    low_conf_left, low_conf_right = repair_missing_from_low_confidence_coords(
        temp_df, image_paths, temp_model, config
    )
    temp_df.attrs["inferred_left"] = inferred_left
    temp_df.attrs["inferred_right"] = inferred_right
    temp_df.attrs["low_conf_left"] = low_conf_left
    temp_df.attrs["low_conf_right"] = low_conf_right
    return temp_df


def extract_temperatures(video_dir: Path, yolo_model, temp_model, config: ReproConfig) -> pd.DataFrame:
    image_paths = frame_image_paths(video_dir)
    rows = []
    for image_path in image_paths:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            rows.append(
                [
                    image_path.name,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    config.radius,
                    "missing",
                    "missing",
                    "read_failed",
                ]
            )
            continue

        result = yolo_model(img_bgr, verbose=False)[0]
        detection_index = select_detection(result)
        left_temp = right_temp = math.nan
        left_conf = right_conf = math.nan
        left_x = left_y = right_x = right_y = math.nan
        nostril_spacing = math.nan
        roi_radius = int(config.radius)
        left_source = right_source = "missing"
        status = "no_detection"

        if detection_index is not None:
            kpts = result.keypoints.data[detection_index].detach().cpu().numpy()
            if len(kpts) >= 2:
                left_x, left_y, left_conf = keypoint_xy_conf(kpts[0])
                right_x, right_y, right_conf = keypoint_xy_conf(kpts[1])
                nostril_spacing = float(math.hypot(left_x - right_x, left_y - right_y))
                roi_radius = adaptive_roi_radius(
                    nostril_spacing,
                    config.radius,
                    config.adaptive_roi,
                    config.adaptive_roi_scale,
                    config.adaptive_roi_min_radius,
                    config.adaptive_roi_max_radius,
                )
                if left_conf >= config.conf:
                    left_temp = circle_temperature(
                        img_bgr, temp_model, left_x, left_y, roi_radius, config.min_temp
                    )
                    left_source = "detected" if not math.isnan(left_temp) else "missing"
                if right_conf >= config.conf:
                    right_temp = circle_temperature(
                        img_bgr, temp_model, right_x, right_y, roi_radius, config.min_temp
                    )
                    right_source = "detected" if not math.isnan(right_temp) else "missing"
                status = "ok" if not (math.isnan(left_temp) and math.isnan(right_temp)) else "no_valid_nostril"

        rows.append(
            [
                image_path.name,
                left_temp,
                right_temp,
                left_x,
                left_y,
                right_x,
                right_y,
                left_conf,
                right_conf,
                nostril_spacing,
                roi_radius,
                left_source,
                right_source,
                status,
            ]
        )

    temp_df = pd.DataFrame(
        rows,
        columns=[
            "frame_name",
            "left_temp",
            "right_temp",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "left_conf",
            "right_conf",
            "nostril_spacing",
            "roi_radius",
            "left_source",
            "right_source",
            "status",
        ],
    )
    if config.track_missing:
        left_repaired = repair_short_missing_runs(temp_df, image_paths, temp_model, "left", config)
        right_repaired = repair_short_missing_runs(temp_df, image_paths, temp_model, "right", config)
        tracked = (temp_df["left_source"].eq("tracked")) | (temp_df["right_source"].eq("tracked"))
        temp_df.loc[tracked, "status"] = "tracked_repair"
        temp_df.attrs["tracked_left"] = left_repaired
        temp_df.attrs["tracked_right"] = right_repaired
    inferred_left, inferred_right = repair_missing_from_opposite_side(temp_df, image_paths, temp_model, config)
    low_conf_left, low_conf_right = repair_missing_from_low_confidence_coords(temp_df, image_paths, temp_model, config)
    temp_df.attrs["inferred_left"] = inferred_left
    temp_df.attrs["inferred_right"] = inferred_right
    temp_df.attrs["low_conf_left"] = low_conf_left
    temp_df.attrs["low_conf_right"] = low_conf_right
    return temp_df


def normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    min_value = valid.min()
    max_value = valid.max()
    if math.isclose(float(max_value), float(min_value)):
        return pd.Series(0.0, index=values.index, dtype=float).where(numeric.notna(), np.nan)
    return (numeric - min_value) / (max_value - min_value)


def repair_series(values: pd.Series, enabled: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not enabled or numeric.notna().sum() < 2:
        return numeric
    return numeric.interpolate(method="linear", limit_direction="both")


def bridge_short_false_runs(mask: pd.Series, max_gap: int) -> pd.Series:
    bridged = mask.fillna(False).astype(bool).copy()
    max_gap = max(0, int(max_gap))
    if max_gap == 0 or len(bridged) < 3:
        return bridged
    values = bridged.to_numpy(dtype=bool)
    for start, end in missing_runs(pd.Series(~values)):
        gap_len = end - start + 1
        if start > 0 and end < len(values) - 1 and gap_len <= max_gap:
            values[start : end + 1] = True
    return pd.Series(values, index=mask.index, dtype=bool)


def interpolate_short_interior_runs(values: pd.Series, max_gap: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").copy()
    if numeric.notna().sum() < 2 or max_gap <= 0:
        return numeric
    result = numeric.to_numpy(dtype=float)
    for start, end in missing_runs(numeric.isna()):
        gap_len = end - start + 1
        if start == 0 or end == len(result) - 1 or gap_len > int(max_gap):
            continue
        left = result[start - 1]
        right = result[end + 1]
        if not math.isfinite(left) or not math.isfinite(right):
            continue
        result[start : end + 1] = np.linspace(left, right, gap_len + 2)[1:-1]
    return pd.Series(result, index=values.index, dtype=float)


def side_quality_mask(temp_df: pd.DataFrame, side: str, config: ReproConfig) -> pd.Series:
    index = temp_df.index
    temperature = pd.to_numeric(
        temp_df.get(f"{side}_temp", pd.Series(np.nan, index=index)), errors="coerce"
    )
    confidence = pd.to_numeric(
        temp_df.get(f"{side}_conf", pd.Series(np.nan, index=index)), errors="coerce"
    )
    x_coord = pd.to_numeric(
        temp_df.get(f"{side}_x", pd.Series(np.nan, index=index)), errors="coerce"
    )
    y_coord = pd.to_numeric(
        temp_df.get(f"{side}_y", pd.Series(np.nan, index=index)), errors="coerce"
    )
    source = temp_df.get(f"{side}_source", pd.Series("", index=index)).astype(str)
    valid = (
        temperature.notna()
        & confidence.ge(float(config.quality_min_confidence))
        & x_coord.ge(0)
        & y_coord.ge(0)
        & source.eq("detected")
    )
    if config.quality_reject_motion and "motion_artifact" in temp_df.columns:
        valid &= ~temp_df["motion_artifact"].fillna(False).astype(bool)
    return bridge_short_false_runs(valid, config.quality_max_interp_gap)


def fusion_quality_mask(
    left_valid: pd.Series,
    right_valid: pd.Series,
    selected_mode: str,
) -> pd.Series:
    if selected_mode == "left":
        return left_valid.astype(bool)
    if selected_mode == "right":
        return right_valid.astype(bool)
    return left_valid.astype(bool) | right_valid.astype(bool)


def filter_events_by_quality_support(
    events: np.ndarray,
    quality_valid: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, list[int]]:
    if len(events) == 0:
        return np.asarray(events, dtype=int), []
    quality = np.asarray(quality_valid, dtype=bool)
    radius = max(0, int(radius))
    kept: list[int] = []
    rejected: list[int] = []
    for event in np.asarray(events, dtype=int):
        start = max(0, int(event) - radius)
        end = min(len(quality), int(event) + radius + 1)
        if start < end and bool(quality[start:end].any()):
            kept.append(int(event))
        else:
            rejected.append(int(event))
    return np.asarray(kept, dtype=int), rejected


def longest_false_run(mask: pd.Series) -> int:
    runs = missing_runs(mask.fillna(False).astype(bool).map(lambda value: not value))
    return max((end - start + 1 for start, end in runs), default=0)


def fuse_columns(left_norm: pd.Series, right_norm: pd.Series, mode: str) -> pd.Series:
    pair = pd.concat([left_norm, right_norm], axis=1)
    if mode == "mean":
        return pair.mean(axis=1, skipna=True)
    if mode == "max":
        return pair.max(axis=1, skipna=True)
    if mode == "min":
        return pair.min(axis=1, skipna=True)
    if mode == "left":
        return left_norm
    if mode == "right":
        return right_norm
    raise ValueError(f"Unsupported fusion mode: {mode}")


def fusion_side_missing(temp_df: pd.DataFrame, mode: str) -> float:
    left_missing = pd.to_numeric(temp_df["left_temp"], errors="coerce").isna()
    right_missing = pd.to_numeric(temp_df["right_temp"], errors="coerce").isna()
    left_source = temp_df.get("left_source", pd.Series("", index=temp_df.index))
    right_source = temp_df.get("right_source", pd.Series("", index=temp_df.index))
    inferred_sources = ["opposite_inferred", "global_offset_inferred", "low_conf_inferred"]
    left_inferred = left_source.isin(inferred_sources)
    right_inferred = right_source.isin(inferred_sources)
    left_penalty = float((left_missing.astype(float) + 0.75 * left_inferred.astype(float)).mean())
    right_penalty = float((right_missing.astype(float) + 0.75 * right_inferred.astype(float)).mean())
    if mode == "left":
        return left_penalty
    if mode == "right":
        return right_penalty
    return (left_penalty + right_penalty) / 2


def fusion_score_from_quality(
    quality: dict[str, float | int | str],
    *,
    amplitude_weight: float,
    interval_cv_weight: float,
    missing_weight: float,
    motion_weight: float = 0.0,
) -> float:
    return float(
        float(quality["candidate_median_prominence"])
        + amplitude_weight * float(quality["candidate_amplitude"])
        - interval_cv_weight * float(quality["candidate_interval_cv"])
        - missing_weight * float(quality["candidate_missing_rate"])
        - motion_weight * float(quality.get("candidate_motion_coupling", 0.0))
    )


def motion_signal_coupling(y: np.ndarray, motion_artifact: np.ndarray | None) -> float:
    if motion_artifact is None or len(y) < 3 or len(motion_artifact) != len(y):
        return 0.0
    derivative = np.abs(np.diff(np.asarray(y, dtype=float)))
    motion = np.asarray(motion_artifact, dtype=bool)
    motion_steps = motion[1:] | motion[:-1]
    valid = np.isfinite(derivative)
    motion_values = derivative[valid & motion_steps]
    stable_values = derivative[valid & ~motion_steps]
    if len(motion_values) < 2 or len(stable_values) < 2:
        return 0.0
    motion_energy = float(np.mean(motion_values))
    stable_energy = float(np.mean(stable_values))
    if stable_energy <= 1e-9:
        return 0.0 if motion_energy <= 1e-9 else 3.0
    return float(max(0.0, math.log1p(motion_energy / stable_energy) - math.log(2.0)))


def curve_quality_score(
    y: np.ndarray,
    mode: str,
    side_missing: float,
    config: ReproConfig,
    motion_artifact: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    smoothed, _ = smooth_curve(
        y,
        config.smooth_window,
        method=config.smoothing_method,
        fps=config.fps,
        butterworth_order=config.butterworth_order,
        butterworth_cutoff_hz=config.butterworth_cutoff_hz,
    )
    peaks, properties = find_peaks(
        smoothed,
        distance=config.peak_distance,
        prominence=config.peak_prominence,
        width=1,
        plateau_size=1,
    )
    prominences = properties.get("prominences", np.array([]))
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    else:
        interval_cv = 999.0
    amplitude = float(np.nanpercentile(smoothed, 95) - np.nanpercentile(smoothed, 5)) if len(smoothed) else 0.0
    median_prominence = float(np.median(prominences)) if len(prominences) else 0.0
    motion_coupling = motion_signal_coupling(y, motion_artifact)
    quality: dict[str, float | int | str] = {
        "mode": mode,
        "candidate_peaks": int(len(peaks)),
        "candidate_median_prominence": median_prominence,
        "candidate_interval_cv": float(interval_cv),
        "candidate_amplitude": amplitude,
        "candidate_missing_rate": float(side_missing),
        "candidate_motion_coupling": float(motion_coupling),
    }
    quality["score"] = fusion_score_from_quality(
        quality,
        amplitude_weight=config.fusion_amplitude_weight,
        interval_cv_weight=config.fusion_interval_cv_weight,
        missing_weight=config.fusion_missing_weight,
        motion_weight=config.fusion_motion_penalty_weight,
    )
    return quality


def fast_fusion_quality_config(repair_missing: bool) -> ReproConfig:
    return ReproConfig(
        input_root=Path("."),
        yolo_model=Path("."),
        temp_model=Path("."),
        truth_csv=None,
        fps=8.7,
        radius=20,
        adaptive_roi=False,
        adaptive_roi_scale=0.14,
        adaptive_roi_min_radius=12,
        adaptive_roi_max_radius=32,
        keypoint_tracking=False,
        motion_registration=False,
        keypoint_source_prefix=None,
        tracking_min_confidence=0.25,
        tracking_min_cutoff=0.8,
        tracking_beta=0.02,
        tracking_derivative_cutoff=1.0,
        temperature_source_prefix=None,
        conf=0.5,
        min_temp=20.0,
        quality_gate=False,
        quality_min_confidence=0.5,
        quality_max_interp_gap=3,
        quality_reject_motion=True,
        quality_peak_support_radius=1,
        fusion_mode="adaptive",
        fusion_amplitude_weight=2.0,
        fusion_interval_cv_weight=0.5,
        fusion_missing_weight=2.0,
        fusion_motion_penalty_weight=0.0,
        fusion_rescue_interval_cv_weight=1.1,
        fusion_rescue_outlier_threshold=0,
        repair_missing=repair_missing,
        smooth_window=3,
        smoothing_method="moving_average",
        butterworth_order=4,
        butterworth_cutoff_hz=2.0,
        peak_distance=5,
        peak_prominence=0.035,
        adaptive_peak_prominence=True,
        breath_detector="peak",
        mad_bandpass_low_hz=0.35,
        mad_bandpass_high_hz=1.7,
        mad_filter_order=2,
        mad_window_seconds=2.0,
        mad_threshold_multiplier=0.8,
        mad_min_dwell_seconds=0.15,
        mad_flicker_seconds=0.30,
        mad_ibi_min_ratio=0.55,
        event_level_fusion=False,
        event_match_ibi_ratio=0.30,
        min_rr_bpm=40.0,
        max_rr_bpm=90.0,
        rr_estimator="peak",
        spectral_min_concentration=0.20,
        autocorr_min_correlation=0.25,
        joint_max_count_adjustment=1,
        merge_shallow_peaks=True,
        merge_peak_gap=6,
        merge_valley_relief=0.10,
        track_missing=True,
        max_track_gap=3,
        max_track_anchor_shift=1.0,
        limit_to_truth_duration=True,
        duration_crop_tolerance_seconds=1.0,
        adaptive_peak_retuning=False,
        dense_peak_threshold=12,
        dense_peak_target=10,
        sparse_peak_count=8,
        sparse_max_smoothed_frames=72,
        short_sparse_edge_completion=True,
        short_sparse_edge_min_gap=0,
        short_sparse_edge_min_relief=0.1,
        short_sparse_edge_max_frames=10,
        infer_missing_nostril=True,
        min_offset_samples=2,
        infer_global_offset=True,
        min_global_offset_samples=50,
        residual_review_threshold=1,
        infer_low_confidence_nostril=True,
        min_low_confidence=0.0,
        edge_peak_completion="none",
        edge_peak_min_gap=5,
        edge_peak_min_relief=0.5,
        edge_peak_max_frames=10,
        source_peak_gate=False,
        source_peak_support_radius=3,
        motion_artifact_suppression=False,
        motion_center_step_threshold=0.20,
        motion_scale_step_threshold=0.12,
        motion_angle_step_threshold=0.18,
        motion_mad_multiplier=6.0,
        motion_mask_radius=1,
        motion_peak_gate=False,
        motion_peak_gate_radius=1,
        motion_interval_regularization=False,
        motion_short_interval_ratio=0.60,
        motion_interval_min_peaks=5,
        motion_interval_min_improvement=0.08,
        motion_reconstruct_missing_cycles=False,
        motion_long_interval_ratio=1.65,
        motion_cycle_tolerance=0.30,
        motion_reconstruction_search_radius=2,
        use_truth_duration_for_rr=True,
        reuse_temperatures=True,
        optimize_peaks=False,
        output_prefix="paper_repro",
        overwrite=False,
    )


def direct_detection_mask(temp_df: pd.DataFrame) -> np.ndarray:
    if "left_source" not in temp_df.columns and "right_source" not in temp_df.columns:
        return np.ones(len(temp_df), dtype=bool)
    left = temp_df.get("left_source", pd.Series("", index=temp_df.index)).astype(str)
    right = temp_df.get("right_source", pd.Series("", index=temp_df.index)).astype(str)
    return (left.eq("detected") | right.eq("detected")).to_numpy(dtype=bool)


def filter_peaks_by_source_support(
    peaks: np.ndarray,
    temp_df: pd.DataFrame,
    *,
    smooth_offset: int,
    enabled: bool,
    radius: int,
) -> tuple[np.ndarray, list[int]]:
    if not enabled or len(peaks) == 0:
        return peaks, []
    direct = direct_detection_mask(temp_df)
    radius = max(0, int(radius))
    kept: list[int] = []
    rejected_frames: list[int] = []
    for peak in peaks:
        peak_index = int(peak)
        frame = peak_index + int(smooth_offset)
        start = max(0, frame - radius)
        end = min(len(direct), frame + radius + 1)
        if bool(direct[start:end].any()):
            kept.append(peak_index)
        else:
            rejected_frames.append(frame)
    return np.asarray(kept, dtype=int), rejected_frames


def robust_motion_threshold(
    values: np.ndarray,
    *,
    absolute_floor: float,
    mad_multiplier: float,
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    floor = max(0.0, float(absolute_floor))
    if finite.size == 0:
        return floor
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    adaptive = median + max(0.0, float(mad_multiplier)) * 1.4826 * mad
    return max(floor, adaptive)


def nostril_motion_features(temp_df: pd.DataFrame, config: ReproConfig) -> pd.DataFrame:
    index = temp_df.index

    def coordinate(name: str) -> np.ndarray:
        return pd.to_numeric(
            temp_df.get(name, pd.Series(np.nan, index=index)), errors="coerce"
        ).to_numpy(dtype=float)

    left_x = coordinate("left_x")
    left_y = coordinate("left_y")
    right_x = coordinate("right_x")
    right_y = coordinate("right_y")
    center_x = (left_x + right_x) / 2.0
    center_y = (left_y + right_y) / 2.0
    axis_x = left_x - right_x
    axis_y = left_y - right_y
    spacing = np.hypot(axis_x, axis_y)
    angle = np.arctan2(axis_y, axis_x)
    valid = (
        np.isfinite(center_x)
        & np.isfinite(center_y)
        & np.isfinite(spacing)
        & (spacing > 1e-6)
    )

    center_step = np.full(len(temp_df), np.nan, dtype=float)
    scale_step = np.full(len(temp_df), np.nan, dtype=float)
    angle_step = np.full(len(temp_df), np.nan, dtype=float)
    if len(temp_df) >= 2:
        pair_valid = valid[1:] & valid[:-1]
        local_spacing = (spacing[1:] + spacing[:-1]) / 2.0
        displacement = np.hypot(
            center_x[1:] - center_x[:-1], center_y[1:] - center_y[:-1]
        )
        normalized = np.divide(
            displacement,
            local_spacing,
            out=np.full_like(displacement, np.nan),
            where=pair_valid & (local_spacing > 1e-6),
        )
        center_step[1:] = normalized
        scale_delta = np.abs(
            np.log(
                np.divide(
                    spacing[1:],
                    spacing[:-1],
                    out=np.full(len(temp_df) - 1, np.nan, dtype=float),
                    where=pair_valid & (spacing[:-1] > 1e-6),
                )
            )
        )
        scale_step[1:] = scale_delta
        wrapped_angle = np.abs(
            np.angle(np.exp(1j * (angle[1:] - angle[:-1])))
        )
        wrapped_angle[~pair_valid] = np.nan
        angle_step[1:] = wrapped_angle

    center_threshold = robust_motion_threshold(
        center_step,
        absolute_floor=config.motion_center_step_threshold,
        mad_multiplier=config.motion_mad_multiplier,
    )
    scale_threshold = robust_motion_threshold(
        scale_step,
        absolute_floor=config.motion_scale_step_threshold,
        mad_multiplier=config.motion_mad_multiplier,
    )
    angle_threshold = robust_motion_threshold(
        angle_step,
        absolute_floor=config.motion_angle_step_threshold,
        mad_multiplier=config.motion_mad_multiplier,
    )
    center_flag = np.isfinite(center_step) & (center_step > center_threshold)
    scale_flag = np.isfinite(scale_step) & (scale_step > scale_threshold)
    angle_flag = np.isfinite(angle_step) & (angle_step > angle_threshold)
    raw_artifact = center_flag | scale_flag | angle_flag
    radius = max(0, int(config.motion_mask_radius))
    if radius > 0 and raw_artifact.any():
        kernel = np.ones(2 * radius + 1, dtype=int)
        artifact = np.convolve(raw_artifact.astype(int), kernel, mode="same") > 0
    else:
        artifact = raw_artifact.copy()

    center_ratio = np.nan_to_num(center_step / max(center_threshold, 1e-9), nan=0.0)
    scale_ratio = np.nan_to_num(scale_step / max(scale_threshold, 1e-9), nan=0.0)
    angle_ratio = np.nan_to_num(angle_step / max(angle_threshold, 1e-9), nan=0.0)
    score = np.maximum.reduce([center_ratio, scale_ratio, angle_ratio])
    return pd.DataFrame(
        {
            "motion_coordinate_valid": valid,
            "motion_center_step_norm": center_step,
            "motion_scale_step_log": scale_step,
            "motion_angle_step_rad": angle_step,
            "motion_score": score,
            "motion_center_flag": center_flag,
            "motion_scale_flag": scale_flag,
            "motion_angle_flag": angle_flag,
            "motion_artifact_raw": raw_artifact,
            "motion_artifact": artifact,
            "motion_center_threshold": center_threshold,
            "motion_scale_threshold": scale_threshold,
            "motion_angle_threshold": angle_threshold,
        },
        index=index,
    )


def filter_peaks_by_motion_artifact(
    peaks: np.ndarray,
    temp_df: pd.DataFrame,
    *,
    smooth_offset: int,
    enabled: bool,
    radius: int,
) -> tuple[np.ndarray, list[int]]:
    if not enabled or len(peaks) == 0 or "motion_artifact" not in temp_df.columns:
        return peaks, []
    artifact = temp_df["motion_artifact"].fillna(False).to_numpy(dtype=bool)
    radius = max(0, int(radius))
    kept: list[int] = []
    rejected_frames: list[int] = []
    for peak in peaks:
        peak_index = int(peak)
        frame = peak_index + int(smooth_offset)
        start = max(0, frame - radius)
        end = min(len(artifact), frame + radius + 1)
        if bool(artifact[start:end].any()):
            rejected_frames.append(frame)
        else:
            kept.append(peak_index)
    return np.asarray(kept, dtype=int), rejected_frames


def cycle_interval_loss(peaks: np.ndarray, nominal_interval: float) -> float:
    if len(peaks) < 2 or not math.isfinite(nominal_interval) or nominal_interval <= 0:
        return math.inf
    intervals = np.diff(np.asarray(peaks, dtype=float))
    cycle_counts = np.maximum(1.0, np.rint(intervals / nominal_interval))
    per_cycle = intervals / cycle_counts
    return float(np.mean(np.abs(per_cycle / nominal_interval - 1.0)))


def regularize_motion_affected_peaks(
    peaks: np.ndarray,
    smoothed: np.ndarray,
    temp_df: pd.DataFrame,
    *,
    smooth_offset: int,
    enabled: bool,
    short_interval_ratio: float,
    min_peaks: int,
    min_improvement: float,
    reconstruct_missing_cycles: bool,
    long_interval_ratio: float,
    cycle_tolerance: float,
    search_radius: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    if (
        (not enabled and not reconstruct_missing_cycles)
        or len(peaks) < max(3, int(min_peaks))
        or "motion_artifact" not in temp_df.columns
    ):
        return peaks, [], []
    artifact = temp_df["motion_artifact"].fillna(False).to_numpy(dtype=bool)
    working = np.asarray(sorted({int(peak) for peak in peaks}), dtype=int)
    rejected_frames: list[int] = []
    added_frames: list[int] = []

    def overlaps_motion(smoothed_frame: int, radius: int = 0) -> bool:
        frame = int(smoothed_frame) + int(smooth_offset)
        start = max(0, frame - max(0, int(radius)))
        end = min(len(artifact), frame + max(0, int(radius)) + 1)
        return bool(artifact[start:end].any())

    if enabled:
        while len(working) >= max(3, int(min_peaks)):
            intervals = np.diff(working).astype(float)
            nominal = float(np.median(intervals)) if len(intervals) else math.nan
            if not math.isfinite(nominal) or nominal <= 0:
                break
            current_loss = cycle_interval_loss(working, nominal)
            best: tuple[float, int, np.ndarray] | None = None
            for interval_index, interval in enumerate(intervals):
                if interval >= float(short_interval_ratio) * nominal:
                    continue
                for remove_index in [interval_index, interval_index + 1]:
                    if remove_index <= 0 or remove_index >= len(working) - 1:
                        continue
                    if not overlaps_motion(int(working[remove_index])):
                        continue
                    candidate = np.delete(working, remove_index)
                    improvement = current_loss - cycle_interval_loss(candidate, nominal)
                    if best is None or improvement > best[0]:
                        best = (float(improvement), int(remove_index), candidate)
            if best is None or best[0] < max(0.0, float(min_improvement)):
                break
            rejected_frames.append(int(working[best[1]]) + int(smooth_offset))
            working = best[2]

    if reconstruct_missing_cycles and len(working) >= max(3, int(min_peaks)):
        intervals = np.diff(working).astype(float)
        nominal = float(np.median(intervals)) if len(intervals) else math.nan
        if math.isfinite(nominal) and nominal > 0:
            additions: list[int] = []
            for left_peak, right_peak in zip(working[:-1], working[1:]):
                gap = float(right_peak - left_peak)
                ratio = gap / nominal
                cycles = int(round(ratio))
                if (
                    ratio < float(long_interval_ratio)
                    or cycles < 2
                    or abs(ratio - cycles) > max(0.0, float(cycle_tolerance))
                ):
                    continue
                gap_start = max(0, int(left_peak) + int(smooth_offset) + 1)
                gap_end = min(len(artifact), int(right_peak) + int(smooth_offset))
                if gap_start >= gap_end or not bool(artifact[gap_start:gap_end].any()):
                    continue
                for cycle_index in range(1, cycles):
                    expected = int(round(left_peak + gap * cycle_index / cycles))
                    motion_radius = max(int(search_radius), int(round(nominal * 0.35)))
                    if not overlaps_motion(expected, motion_radius):
                        continue
                    start = max(int(left_peak) + 1, expected - max(0, int(search_radius)))
                    end = min(int(right_peak), expected + max(0, int(search_radius)) + 1)
                    if start >= end:
                        continue
                    local = smoothed[start:end]
                    if len(local) == 0 or not np.isfinite(local).any():
                        continue
                    addition = int(start + np.nanargmax(local))
                    if addition not in additions and addition not in working:
                        additions.append(addition)
                        added_frames.append(addition + int(smooth_offset))
            if additions:
                working = np.asarray(sorted([*working.tolist(), *additions]), dtype=int)

    return working, rejected_frames, added_frames


def adaptive_candidate_peak_count(
    candidate: pd.Series,
    temp_df: pd.DataFrame,
    config: ReproConfig,
) -> int:
    values = candidate.interpolate(method="linear", limit_direction="both").fillna(0.5)
    smoothed, smooth_offset = smooth_curve(
        values.to_numpy(dtype=float), config.smooth_window
    )
    peaks, _, _, _, _ = detect_peaks(smoothed, config)
    peaks, _ = complete_edge_peaks(smoothed, peaks, config)
    peaks, _ = filter_peaks_by_source_support(
        peaks,
        temp_df,
        smooth_offset=smooth_offset,
        enabled=config.source_peak_gate,
        radius=config.source_peak_support_radius,
    )
    peaks, _, _ = regularize_motion_affected_peaks(
        peaks,
        smoothed,
        temp_df,
        smooth_offset=smooth_offset,
        enabled=config.motion_interval_regularization,
        short_interval_ratio=config.motion_short_interval_ratio,
        min_peaks=config.motion_interval_min_peaks,
        min_improvement=config.motion_interval_min_improvement,
        reconstruct_missing_cycles=config.motion_reconstruct_missing_cycles,
        long_interval_ratio=config.motion_long_interval_ratio,
        cycle_tolerance=config.motion_cycle_tolerance,
        search_radius=config.motion_reconstruction_search_radius,
    )
    peaks, _ = filter_peaks_by_motion_artifact(
        peaks,
        temp_df,
        smooth_offset=smooth_offset,
        enabled=config.motion_peak_gate,
        radius=config.motion_peak_gate_radius,
    )
    return int(len(peaks))


def choose_adaptive_fusion_candidate(
    candidates: list[dict[str, object]],
    config: ReproConfig,
) -> dict[str, object]:
    legacy = max(candidates, key=lambda item: float(item["quality"]["score"]))
    selected = legacy
    rescue_applied = False
    consensus_count = math.nan
    legacy_distance = math.nan
    robust_distance = math.nan
    robust_score = math.nan
    threshold = max(0, int(config.fusion_rescue_outlier_threshold))
    if threshold > 0:
        counts = np.asarray([int(item["final_peak_count"]) for item in candidates])
        consensus_count = float(np.median(counts))
        robust_scores = [
            fusion_score_from_quality(
                item["quality"],
                amplitude_weight=config.fusion_amplitude_weight,
                interval_cv_weight=config.fusion_rescue_interval_cv_weight,
                missing_weight=config.fusion_missing_weight,
                motion_weight=config.fusion_motion_penalty_weight,
            )
            for item in candidates
        ]
        robust_index = int(np.argmax(robust_scores))
        robust = candidates[robust_index]
        robust_score = float(robust_scores[robust_index])
        legacy_distance = abs(float(legacy["final_peak_count"]) - consensus_count)
        robust_distance = abs(float(robust["final_peak_count"]) - consensus_count)
        if legacy_distance >= threshold and robust_distance < legacy_distance:
            selected = robust
            rescue_applied = True

    quality = dict(selected["quality"])
    quality.update(
        {
            "legacy_selected_fusion_mode": str(legacy["mode"]),
            "legacy_candidate_peaks": int(legacy["final_peak_count"]),
            "fusion_rescue_applied": str(rescue_applied),
            "fusion_consensus_count": consensus_count,
            "legacy_consensus_distance": legacy_distance,
            "robust_consensus_distance": robust_distance,
            "robust_selection_score": robust_score,
        }
    )
    return {**selected, "quality": quality}


def select_fused_series(
    temp_df: pd.DataFrame,
    left_norm: pd.Series,
    right_norm: pd.Series,
    fusion_mode: str,
    config: ReproConfig,
) -> tuple[str, pd.Series, dict[str, float | int | str]]:
    if fusion_mode == "adaptive":
        candidates: list[dict[str, object]] = []
        for mode in ["mean", "max", "min", "left", "right"]:
            candidate = fuse_columns(left_norm, right_norm, mode)
            candidate = candidate.interpolate(method="linear", limit_direction="both").fillna(0.5)
            quality = curve_quality_score(
                candidate.to_numpy(dtype=float),
                mode,
                fusion_side_missing(temp_df, mode),
                config,
                temp_df.get("motion_artifact", pd.Series(False, index=temp_df.index)).to_numpy(
                    dtype=bool
                ),
            )
            final_peak_count = (
                adaptive_candidate_peak_count(candidate, temp_df, config)
                if config.fusion_rescue_outlier_threshold > 0
                else int(quality["candidate_peaks"])
            )
            candidates.append(
                {
                    "mode": mode,
                    "candidate": candidate,
                    "quality": quality,
                    "final_peak_count": final_peak_count,
                }
            )
        selected = choose_adaptive_fusion_candidate(candidates, config)
        return str(selected["mode"]), selected["candidate"], selected["quality"]

    fused = fuse_columns(left_norm, right_norm, fusion_mode)
    selection_quality = curve_quality_score(
        fused.interpolate(method="linear", limit_direction="both").fillna(0.5).to_numpy(dtype=float),
        fusion_mode,
        fusion_side_missing(temp_df, fusion_mode),
        config,
        temp_df.get("motion_artifact", pd.Series(False, index=temp_df.index)).to_numpy(
            dtype=bool
        ),
    )
    return fusion_mode, fused, selection_quality


def merge_shallow_adjacent_peaks(
    peaks: np.ndarray,
    smoothed: np.ndarray,
    max_gap: int,
    min_valley_relief: float,
) -> tuple[np.ndarray, int]:
    if len(peaks) < 2:
        return peaks, 0

    merged = []
    i = 0
    peaks_list = [int(p) for p in peaks]
    while i < len(peaks_list):
        current = peaks_list[i]
        j = i + 1
        while j < len(peaks_list) and peaks_list[j] - current <= max_gap:
            next_peak = peaks_list[j]
            start, end = sorted((current, next_peak))
            lower_peak = min(float(smoothed[current]), float(smoothed[next_peak]))
            valley = float(np.nanmin(smoothed[start : end + 1]))
            valley_relief = (lower_peak - valley) / (lower_peak + 1e-9)
            if valley_relief < min_valley_relief:
                current = current if smoothed[current] >= smoothed[next_peak] else next_peak
                j += 1
            else:
                break
        merged.append(current)
        i = j

    merged_peaks = np.asarray(merged, dtype=int)
    return merged_peaks, int(len(peaks) - len(merged_peaks))


def detect_peaks(smoothed: np.ndarray, config: ReproConfig) -> tuple[np.ndarray, dict[str, np.ndarray], float, float, int]:
    def run(prominence: float) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
        peaks_, properties_ = find_peaks(
            smoothed,
            height=None,
            threshold=None,
            distance=config.peak_distance,
            prominence=prominence,
            width=1,
            wlen=None,
            plateau_size=1,
        )
        duration_minutes = (len(smoothed) / config.fps) / 60 if config.fps > 0 else math.nan
        rr_ = len(peaks_) / duration_minutes if duration_minutes and duration_minutes > 0 else math.nan
        return peaks_, properties_, float(rr_)

    peaks, properties, rr_bpm = run(config.peak_prominence)
    selected_prominence = float(config.peak_prominence)
    if not config.adaptive_peak_prominence or math.isnan(rr_bpm):
        merged_count = 0
        if config.merge_shallow_peaks:
            peaks, merged_count = merge_shallow_adjacent_peaks(
                peaks, smoothed, config.merge_peak_gap, config.merge_valley_relief
            )
            properties["prominences"] = peak_prominences(smoothed, peaks)[0] if len(peaks) else np.array([])
            duration_minutes = (len(smoothed) / config.fps) / 60 if config.fps > 0 else math.nan
            rr_bpm = len(peaks) / duration_minutes if duration_minutes and duration_minutes > 0 else math.nan
        return peaks, properties, selected_prominence, rr_bpm, merged_count

    if rr_bpm < config.min_rr_bpm:
        for prominence in [0.025, 0.015, 0.01, 0.005, 0.002]:
            trial_peaks, trial_properties, trial_rr = run(prominence)
            peaks, properties, selected_prominence, rr_bpm = (
                trial_peaks,
                trial_properties,
                float(prominence),
                trial_rr,
            )
            if trial_rr >= config.min_rr_bpm:
                break
    elif rr_bpm > config.max_rr_bpm:
        for prominence in [0.05, 0.07, 0.09, 0.12, 0.16, 0.2]:
            trial_peaks, trial_properties, trial_rr = run(prominence)
            peaks, properties, selected_prominence, rr_bpm = (
                trial_peaks,
                trial_properties,
                float(prominence),
                trial_rr,
            )
            if trial_rr <= config.max_rr_bpm:
                break
    merged_count = 0
    if config.merge_shallow_peaks:
        peaks, merged_count = merge_shallow_adjacent_peaks(
            peaks, smoothed, config.merge_peak_gap, config.merge_valley_relief
        )
        properties["prominences"] = peak_prominences(smoothed, peaks)[0] if len(peaks) else np.array([])
        duration_minutes = (len(smoothed) / config.fps) / 60 if config.fps > 0 else math.nan
        rr_bpm = len(peaks) / duration_minutes if duration_minutes and duration_minutes > 0 else math.nan
    return peaks, properties, selected_prominence, rr_bpm, merged_count


def complete_edge_peaks(smoothed: np.ndarray, peaks: np.ndarray, config: ReproConfig) -> tuple[np.ndarray, list[int]]:
    mode = config.edge_peak_completion
    if mode == "none" or len(smoothed) == 0 or len(peaks) == 0:
        return peaks, []

    peak_set = {int(peak) for peak in peaks}
    added: list[int] = []
    min_gap = max(0, int(config.edge_peak_min_gap))
    min_relief = float(config.edge_peak_min_relief)
    max_edge = max(1, int(config.edge_peak_max_frames))

    if mode in {"auto", "start", "both"}:
        first_peak = int(np.min(peaks))
        search_end = min(first_peak, max_edge)
        if search_end > 0:
            candidate = int(np.argmax(smoothed[:search_end]))
            trough = float(np.nanmin(smoothed[candidate : first_peak + 1]))
            relief = float(smoothed[candidate] - trough)
            if first_peak - candidate >= min_gap and relief >= min_relief and candidate not in peak_set:
                added.append(candidate)
                peak_set.add(candidate)

    if mode in {"auto", "end", "both"}:
        last_peak = int(np.max(peaks))
        search_start = last_peak + 1
        search_end = min(len(smoothed), last_peak + max_edge + 1)
        if search_start < search_end:
            candidate = search_start + int(np.argmax(smoothed[search_start:search_end]))
            trough = float(np.nanmin(smoothed[last_peak : candidate + 1]))
            relief = float(smoothed[candidate] - trough)
            if candidate - last_peak >= min_gap and relief >= min_relief and candidate not in peak_set:
                added.append(candidate)
                peak_set.add(candidate)

    if not added:
        return peaks, []
    return np.asarray(sorted(peak_set), dtype=int), added


def fast_fused_array(temp_df: pd.DataFrame, fusion_mode: str, repair_missing: bool) -> np.ndarray:
    left = repair_series(temp_df["left_temp"], repair_missing)
    right = repair_series(temp_df["right_temp"], repair_missing)
    left_norm = normalize_series(left)
    right_norm = normalize_series(right)
    _, fused, _ = select_fused_series(
        temp_df,
        left_norm,
        right_norm,
        fusion_mode,
        fast_fusion_quality_config(repair_missing),
    )
    return fused.interpolate(method="linear", limit_direction="both").fillna(0.5).to_numpy(dtype=float)


def limit_analysis_window(
    temp_df: pd.DataFrame,
    config: ReproConfig,
    truth_row: pd.Series | None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    metadata: dict[str, float | int | str] = {
        "raw_frames": int(len(temp_df)),
        "analysis_frame_limit": int(len(temp_df)),
        "analysis_window_source": "full_temperature_csv",
        "truth_duration_seconds": math.nan,
        "raw_duration_seconds": float(len(temp_df) / config.fps) if config.fps > 0 else math.nan,
    }
    if (
        not config.limit_to_truth_duration
        or truth_row is None
        or "duration_seconds" not in truth_row
        or pd.isna(truth_row["duration_seconds"])
        or config.fps <= 0
    ):
        return temp_df, metadata

    truth_duration = float(truth_row["duration_seconds"])
    metadata["truth_duration_seconds"] = truth_duration
    raw_duration = float(len(temp_df) / config.fps)
    if raw_duration - truth_duration <= config.duration_crop_tolerance_seconds:
        return temp_df, metadata

    frame_limit = int(round(truth_duration * config.fps)) + max(0, int(config.smooth_window) - 1)
    frame_limit = max(1, min(int(len(temp_df)), frame_limit))
    metadata.update(
        {
            "analysis_frame_limit": frame_limit,
            "analysis_window_source": "truth_duration_seconds",
        }
    )
    return temp_df.iloc[:frame_limit].copy(), metadata


def smooth_curve(
    values: np.ndarray,
    smooth_window: int,
    *,
    method: str = "moving_average",
    fps: float = 8.7,
    butterworth_order: int = 4,
    butterworth_cutoff_hz: float = 2.0,
) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    if method == "butterworth":
        if fps <= 0:
            raise ValueError("fps must be positive for Butterworth filtering")
        if butterworth_order < 1:
            raise ValueError("butterworth_order must be at least 1")
        nyquist = fps / 2.0
        if not 0 < butterworth_cutoff_hz < nyquist:
            raise ValueError(
                "butterworth_cutoff_hz must be between 0 and the Nyquist frequency "
                f"({nyquist:.6g} Hz)"
            )
        if len(values) < 4:
            return values.copy(), 0
        sos = butter(
            butterworth_order,
            butterworth_cutoff_hz,
            btype="lowpass",
            fs=fps,
            output="sos",
        )
        default_padlen = 3 * (2 * len(sos) + 1)
        padlen = min(default_padlen, len(values) - 2)
        return sosfiltfilt(sos, values, padlen=padlen), 0
    if method != "moving_average":
        raise ValueError(f"Unsupported smoothing method: {method}")
    smooth_window = max(1, int(smooth_window))
    if smooth_window > 1 and len(values) >= smooth_window:
        return np.convolve(values, np.ones(smooth_window) / smooth_window, mode="valid"), smooth_window // 2
    return values, 0


def rr_duration_seconds(
    smoothed_frames: int,
    config: ReproConfig,
    truth_row: pd.Series | None,
) -> tuple[float, str]:
    if (
        config.use_truth_duration_for_rr
        and truth_row is not None
        and "duration_seconds" in truth_row
        and not pd.isna(truth_row["duration_seconds"])
        and float(truth_row["duration_seconds"]) > 0
    ):
        return float(truth_row["duration_seconds"]), "truth_duration_seconds"
    if config.fps > 0:
        return float(smoothed_frames / config.fps), "smoothed_frame_count"
    return math.nan, "unknown"


def rr_from_peak_count(peaks: int, duration_seconds: float) -> float:
    if math.isnan(duration_seconds) or duration_seconds <= 0:
        return math.nan
    return float(peaks / (duration_seconds / 60.0))


def estimate_time_frequency_consensus(
    smoothed: np.ndarray,
    *,
    fps: float,
    duration_seconds: float,
    min_rr_bpm: float,
    max_rr_bpm: float,
    time_peak_count: int,
    enabled: bool,
    spectral_min_concentration: float,
    autocorr_min_correlation: float,
    max_count_adjustment: int,
) -> dict[str, float | int | str | bool]:
    """Conservatively correct a peak count when spectrum and autocorrelation agree."""
    result: dict[str, float | int | str | bool] = {
        "time_peak_count": int(time_peak_count),
        "spectral_rr_bpm": math.nan,
        "spectral_count": -1,
        "spectral_concentration": math.nan,
        "autocorr_rr_bpm": math.nan,
        "autocorr_count": -1,
        "autocorr_correlation": math.nan,
        "joint_count": int(time_peak_count),
        "joint_adjusted": False,
        "joint_reason": "peak_only" if not enabled else "insufficient_signal",
    }
    values = np.asarray(smoothed, dtype=float)
    if (
        not enabled
        or len(values) < 16
        or fps <= 0
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or min_rr_bpm <= 0
        or max_rr_bpm <= min_rr_bpm
    ):
        return result

    finite = np.isfinite(values)
    if int(finite.sum()) < 16:
        return result
    if not finite.all():
        indices = np.arange(len(values))
        values = np.interp(indices, indices[finite], values[finite])
    x = np.arange(len(values), dtype=float)
    trend = np.polyval(np.polyfit(x, values, 1), x)
    centered = values - trend
    scale = float(np.std(centered))
    if not math.isfinite(scale) or scale <= 1e-9:
        return result
    centered /= scale

    nfft_target = max(256, len(centered) * 8)
    nfft = 1 << int(math.ceil(math.log2(nfft_target)))
    frequencies, power = welch(
        centered,
        fs=fps,
        window="hann",
        nperseg=len(centered),
        noverlap=0,
        nfft=nfft,
        detrend=False,
        scaling="spectrum",
    )
    low_hz = min_rr_bpm / 60.0
    high_hz = max_rr_bpm / 60.0
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_indices = np.flatnonzero(band)
    if len(band_indices) == 0 or float(np.sum(power[band])) <= 0:
        return result
    dominant_index = int(band_indices[int(np.argmax(power[band]))])
    dominant_hz = float(frequencies[dominant_index])
    if 0 < dominant_index < len(power) - 1:
        local = np.log(np.maximum(power[dominant_index - 1 : dominant_index + 2], 1e-12))
        denominator = float(local[0] - 2.0 * local[1] + local[2])
        if abs(denominator) > 1e-12:
            offset = float(np.clip(0.5 * (local[0] - local[2]) / denominator, -0.5, 0.5))
            dominant_hz += offset * float(frequencies[1] - frequencies[0])
    spectral_rr = float(dominant_hz * 60.0)
    spectral_count = max(1, int(round(spectral_rr * duration_seconds / 60.0)))
    concentration_half_width = max(0.08, fps / len(centered))
    local_band = band & (np.abs(frequencies - dominant_hz) <= concentration_half_width)
    spectral_concentration = float(np.sum(power[local_band]) / np.sum(power[band]))

    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    autocorr /= np.arange(len(centered), 0, -1, dtype=float)
    if autocorr[0] <= 0:
        return result
    autocorr /= autocorr[0]
    min_lag = max(1, int(math.floor(fps * 60.0 / max_rr_bpm)))
    max_lag = min(len(autocorr) - 2, int(math.ceil(fps * 60.0 / min_rr_bpm)))
    if max_lag <= min_lag:
        return result
    lag_index = int(min_lag + np.argmax(autocorr[min_lag : max_lag + 1]))
    lag = float(lag_index)
    if 0 < lag_index < len(autocorr) - 1:
        left, center, right = autocorr[lag_index - 1 : lag_index + 2]
        denominator = float(left - 2.0 * center + right)
        if abs(denominator) > 1e-12:
            lag += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    autocorr_rr = float(60.0 * fps / lag)
    autocorr_count = max(1, int(round(autocorr_rr * duration_seconds / 60.0)))
    autocorr_correlation = float(autocorr[lag_index])

    result.update(
        {
            "spectral_rr_bpm": spectral_rr,
            "spectral_count": spectral_count,
            "spectral_concentration": spectral_concentration,
            "autocorr_rr_bpm": autocorr_rr,
            "autocorr_count": autocorr_count,
            "autocorr_correlation": autocorr_correlation,
            "joint_reason": "no_consensus",
        }
    )
    adjustment = int(spectral_count - time_peak_count)
    if spectral_count != autocorr_count:
        result["joint_reason"] = "frequency_autocorr_disagree"
    elif abs(adjustment) > max(0, int(max_count_adjustment)):
        result["joint_reason"] = "adjustment_too_large"
    elif spectral_concentration < max(0.0, float(spectral_min_concentration)):
        result["joint_reason"] = "weak_spectrum"
    elif autocorr_correlation < max(0.0, float(autocorr_min_correlation)):
        result["joint_reason"] = "weak_autocorrelation"
    elif adjustment == 0:
        result["joint_reason"] = "all_agree"
    else:
        result["joint_count"] = spectral_count
        result["joint_adjusted"] = True
        result["joint_reason"] = "frequency_autocorr_consensus"
    return result


def mad_phase_events(
    values: pd.Series | np.ndarray,
    quality_valid: pd.Series | np.ndarray,
    config: ReproConfig,
) -> dict[str, object]:
    signal = np.asarray(values, dtype=float)
    quality = np.asarray(quality_valid, dtype=bool)
    result: dict[str, object] = {
        "filtered": np.full(len(signal), np.nan, dtype=float),
        "velocity": np.full(len(signal), np.nan, dtype=float),
        "threshold": np.full(len(signal), np.nan, dtype=float),
        "phase": np.zeros(len(signal), dtype=int),
        "events": np.array([], dtype=int),
        "properties": {"prominences": np.array([], dtype=float)},
        "quality_rejected": [],
        "ibi_rejected": [],
    }
    finite = np.isfinite(signal)
    if len(signal) < 8 or int(finite.sum()) < 8 or config.fps <= 0:
        return result

    indices = np.arange(len(signal), dtype=float)
    filled = np.interp(indices, indices[finite], signal[finite])
    nyquist = config.fps / 2.0
    low_hz = max(1e-4, float(config.mad_bandpass_low_hz))
    high_hz = min(float(config.mad_bandpass_high_hz), nyquist * 0.98)
    if not low_hz < high_hz:
        raise ValueError("MAD band-pass frequencies must satisfy 0 < low < high < Nyquist")
    sos = butter(
        max(1, int(config.mad_filter_order)),
        [low_hz, high_hz],
        btype="bandpass",
        fs=config.fps,
        output="sos",
    )
    default_padlen = 3 * (2 * len(sos) + 1)
    padlen = min(default_padlen, len(filled) - 2)
    filtered = sosfiltfilt(sos, filled, padlen=padlen)
    velocity = np.gradient(filtered) * config.fps

    window = max(3, int(round(config.mad_window_seconds * config.fps)))
    if window % 2 == 0:
        window += 1
    velocity_series = pd.Series(velocity)
    center = velocity_series.rolling(window, center=True, min_periods=3).median()
    center = center.bfill().ffill().fillna(float(np.nanmedian(velocity)))
    absolute_deviation = (velocity_series - center).abs()
    rolling_mad = absolute_deviation.rolling(window, center=True, min_periods=3).median()
    global_mad = float(np.nanmedian(absolute_deviation))
    mad_floor = max(1e-6, 0.20 * global_mad)
    threshold = (
        rolling_mad.bfill().ffill().fillna(global_mad).to_numpy(dtype=float)
        * max(0.1, float(config.mad_threshold_multiplier))
    )
    threshold = np.maximum(threshold, mad_floor)
    centered_velocity = velocity - center.to_numpy(dtype=float)

    min_dwell = max(1, int(math.ceil(config.mad_min_dwell_seconds * config.fps)))
    phase = np.zeros(len(signal), dtype=int)
    state = 0
    state_start = 0
    events: list[int] = []
    for frame, value in enumerate(centered_velocity):
        target = 1 if value >= threshold[frame] else (-1 if value <= -threshold[frame] else 0)
        if state == 0:
            if target != 0:
                state = target
                state_start = frame
        elif target == -state and frame - state_start >= min_dwell:
            if state == 1:
                search_start = max(0, state_start)
                search_end = min(len(filtered), frame + 1)
                if search_end > search_start:
                    events.append(search_start + int(np.argmax(filtered[search_start:search_end])))
            state = target
            state_start = frame
        phase[frame] = state

    min_physiologic_gap = max(1, int(math.floor(config.fps * 60.0 / config.max_rr_bpm)))
    min_event_gap = max(
        min_physiologic_gap,
        int(math.ceil(config.mad_flicker_seconds * config.fps)),
    )
    consolidated: list[int] = []
    ibi_rejected: list[int] = []
    for event in events:
        if not consolidated or event - consolidated[-1] >= min_event_gap:
            consolidated.append(event)
        elif filtered[event] > filtered[consolidated[-1]]:
            ibi_rejected.append(consolidated[-1])
            consolidated[-1] = event
        else:
            ibi_rejected.append(event)

    if len(consolidated) >= 4:
        changed = True
        while changed and len(consolidated) >= 4:
            changed = False
            intervals = np.diff(consolidated)
            median_ibi = float(np.median(intervals))
            short = np.flatnonzero(intervals < config.mad_ibi_min_ratio * median_ibi)
            if len(short):
                pair = int(short[0])
                left_event = consolidated[pair]
                right_event = consolidated[pair + 1]
                drop = pair if filtered[left_event] < filtered[right_event] else pair + 1
                ibi_rejected.append(consolidated[drop])
                consolidated.pop(drop)
                changed = True

    accepted, quality_rejected = filter_events_by_quality_support(
        np.asarray(consolidated, dtype=int),
        quality,
        config.quality_peak_support_radius if config.quality_gate else 0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominences = peak_prominences(filtered, accepted)[0] if len(accepted) else np.array([])
    result.update(
        {
            "filtered": filtered,
            "velocity": centered_velocity,
            "threshold": threshold,
            "phase": phase,
            "events": accepted,
            "properties": {"prominences": prominences},
            "quality_rejected": quality_rejected,
            "ibi_rejected": sorted(set(int(value) for value in ibi_rejected)),
        }
    )
    return result


def event_prominence_map(result: dict[str, object]) -> dict[int, float]:
    events = np.asarray(result["events"], dtype=int)
    prominences = np.asarray(result["properties"].get("prominences", []), dtype=float)
    return {
        int(event): float(prominences[index]) if index < len(prominences) else 0.0
        for index, event in enumerate(events)
    }


def fuse_bilateral_events(
    left_result: dict[str, object],
    right_result: dict[str, object],
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    config: ReproConfig,
) -> dict[str, object]:
    left_events = [int(value) for value in np.asarray(left_result["events"], dtype=int)]
    right_events = [int(value) for value in np.asarray(right_result["events"], dtype=int)]
    interval_sources = []
    if len(left_events) >= 2:
        interval_sources.extend(np.diff(left_events).tolist())
    if len(right_events) >= 2:
        interval_sources.extend(np.diff(right_events).tolist())
    default_ibi = config.fps * 60.0 / ((config.min_rr_bpm + config.max_rr_bpm) / 2.0)
    median_ibi = float(np.median(interval_sources)) if interval_sources else float(default_ibi)
    tolerance = max(1, int(round(config.event_match_ibi_ratio * median_ibi)))
    left_scores = event_prominence_map(left_result)
    right_scores = event_prominence_map(right_result)

    unmatched_right = set(range(len(right_events)))
    candidates: list[tuple[int, float, str]] = []
    bilateral_matches = 0
    rejected_unilateral: list[int] = []
    for left_event in left_events:
        possible = [
            index for index in unmatched_right
            if abs(right_events[index] - left_event) <= tolerance
        ]
        if possible:
            match = min(possible, key=lambda index: abs(right_events[index] - left_event))
            right_event = right_events[match]
            unmatched_right.remove(match)
            left_score = left_scores.get(left_event, 0.0)
            right_score = right_scores.get(right_event, 0.0)
            weight = left_score + right_score
            frame = (
                int(round((left_event * left_score + right_event * right_score) / weight))
                if weight > 1e-9
                else int(round((left_event + right_event) / 2.0))
            )
            candidates.append((frame, left_score + right_score + 1.0, "both"))
            bilateral_matches += 1
        else:
            start = max(0, left_event - tolerance)
            end = min(len(right_valid), left_event + tolerance + 1)
            if not bool(np.asarray(right_valid, dtype=bool)[start:end].any()):
                candidates.append((left_event, left_scores.get(left_event, 0.0), "left_only"))
            else:
                rejected_unilateral.append(left_event)
    for index in sorted(unmatched_right):
        right_event = right_events[index]
        start = max(0, right_event - tolerance)
        end = min(len(left_valid), right_event + tolerance + 1)
        if not bool(np.asarray(left_valid, dtype=bool)[start:end].any()):
            candidates.append((right_event, right_scores.get(right_event, 0.0), "right_only"))
        else:
            rejected_unilateral.append(right_event)

    candidates.sort(key=lambda item: item[0])
    minimum_gap = max(1, int(math.floor(config.fps * 60.0 / config.max_rr_bpm)))
    kept: list[tuple[int, float, str]] = []
    for candidate in candidates:
        if not kept or candidate[0] - kept[-1][0] >= minimum_gap:
            kept.append(candidate)
        elif candidate[1] > kept[-1][1]:
            rejected_unilateral.append(kept[-1][0])
            kept[-1] = candidate
        else:
            rejected_unilateral.append(candidate[0])
    return {
        "events": np.asarray([item[0] for item in kept], dtype=int),
        "sources": [item[2] for item in kept],
        "bilateral_matches": bilateral_matches,
        "left_only": sum(item[2] == "left_only" for item in kept),
        "right_only": sum(item[2] == "right_only" for item in kept),
        "rejected_unilateral": sorted(set(rejected_unilateral)),
        "match_tolerance_frames": tolerance,
    }


def detect_curve(
    temp_df: pd.DataFrame,
    curve: pd.DataFrame,
    config: ReproConfig,
    fusion_selection_config: ReproConfig | None = None,
) -> dict[str, object]:
    selected_fusion_mode, fused, selection_quality = select_fused_series(
        temp_df,
        curve["left_norm"],
        curve["right_norm"],
        config.fusion_mode,
        fusion_selection_config or config,
    )
    left_quality = curve.get(
        "left_quality_valid", pd.Series(True, index=curve.index)
    ).fillna(False).astype(bool)
    right_quality = curve.get(
        "right_quality_valid", pd.Series(True, index=curve.index)
    ).fillna(False).astype(bool)
    selected_quality = fusion_quality_mask(left_quality, right_quality, selected_fusion_mode)
    fused_repaired = fused.isna()
    fused = fused.interpolate(method="linear", limit_direction="both").fillna(0.5)
    quality_rejected_frames: list[int] = []
    mad_ibi_rejected_frames: list[int] = []
    event_sources: list[str] = []
    bilateral_matches = 0
    event_left_only = 0
    event_right_only = 0
    event_unilateral_rejected: list[int] = []
    event_match_tolerance = 0

    if config.event_level_fusion:
        left_result = mad_phase_events(curve["left_norm"], left_quality, config)
        right_result = mad_phase_events(curve["right_norm"], right_quality, config)
        event_fusion = fuse_bilateral_events(
            left_result,
            right_result,
            left_quality.to_numpy(dtype=bool),
            right_quality.to_numpy(dtype=bool),
            config,
        )
        left_filtered = np.asarray(left_result["filtered"], dtype=float)
        right_filtered = np.asarray(right_result["filtered"], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            smoothed = np.nanmean(np.vstack([left_filtered, right_filtered]), axis=0)
        smoothed = np.where(np.isfinite(smoothed), smoothed, fused.to_numpy(dtype=float))
        smooth_offset = 0
        peaks = np.asarray(event_fusion["events"], dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prominences = peak_prominences(smoothed, peaks)[0] if len(peaks) else np.array([])
        properties = {"prominences": prominences}
        selected_peak_prominence = math.nan
        merged_peak_count = 0
        edge_added: list[int] = []
        quality_rejected_frames = sorted(
            set(left_result["quality_rejected"] + right_result["quality_rejected"])
        )
        mad_ibi_rejected_frames = sorted(
            set(left_result["ibi_rejected"] + right_result["ibi_rejected"])
        )
        event_sources = list(event_fusion["sources"])
        bilateral_matches = int(event_fusion["bilateral_matches"])
        event_left_only = int(event_fusion["left_only"])
        event_right_only = int(event_fusion["right_only"])
        event_unilateral_rejected = list(event_fusion["rejected_unilateral"])
        event_match_tolerance = int(event_fusion["match_tolerance_frames"])
    elif config.breath_detector == "mad_phase":
        mad_result = mad_phase_events(fused, selected_quality, config)
        smoothed = np.asarray(mad_result["filtered"], dtype=float)
        smooth_offset = 0
        peaks = np.asarray(mad_result["events"], dtype=int)
        properties = mad_result["properties"]
        selected_peak_prominence = math.nan
        merged_peak_count = 0
        edge_added = []
        quality_rejected_frames = list(mad_result["quality_rejected"])
        mad_ibi_rejected_frames = list(mad_result["ibi_rejected"])
    else:
        smoothed, smooth_offset = smooth_curve(
            fused.to_numpy(dtype=float),
            config.smooth_window,
            method=config.smoothing_method,
            fps=config.fps,
            butterworth_order=config.butterworth_order,
            butterworth_cutoff_hz=config.butterworth_cutoff_hz,
        )
        peaks, properties, selected_peak_prominence, _, merged_peak_count = detect_peaks(smoothed, config)
        peaks, edge_added = complete_edge_peaks(smoothed, peaks, config)
        if config.quality_gate:
            peak_frames = peaks + smooth_offset
            kept_frames, quality_rejected_frames = filter_events_by_quality_support(
                peak_frames,
                selected_quality.to_numpy(dtype=bool),
                config.quality_peak_support_radius,
            )
            peaks = kept_frames - smooth_offset

    peaks, source_rejected_frames = filter_peaks_by_source_support(
        peaks,
        temp_df,
        smooth_offset=smooth_offset,
        enabled=config.source_peak_gate,
        radius=config.source_peak_support_radius,
    )
    peaks, motion_interval_rejected_frames, motion_reconstructed_frames = (
        regularize_motion_affected_peaks(
            peaks,
            smoothed,
            temp_df,
            smooth_offset=smooth_offset,
            enabled=config.motion_interval_regularization,
            short_interval_ratio=config.motion_short_interval_ratio,
            min_peaks=config.motion_interval_min_peaks,
            min_improvement=config.motion_interval_min_improvement,
            reconstruct_missing_cycles=config.motion_reconstruct_missing_cycles,
            long_interval_ratio=config.motion_long_interval_ratio,
            cycle_tolerance=config.motion_cycle_tolerance,
            search_radius=config.motion_reconstruction_search_radius,
        )
    )
    peaks, motion_rejected_frames = filter_peaks_by_motion_artifact(
        peaks,
        temp_df,
        smooth_offset=smooth_offset,
        enabled=config.motion_peak_gate,
        radius=config.motion_peak_gate_radius,
    )
    if (
        edge_added
        or quality_rejected_frames
        or source_rejected_frames
        or motion_rejected_frames
        or motion_interval_rejected_frames
        or motion_reconstructed_frames
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            properties["prominences"] = peak_prominences(smoothed, peaks)[0] if len(peaks) else np.array([])
        duration_minutes = (len(smoothed) / config.fps) / 60 if config.fps > 0 else math.nan
        rr_bpm = len(peaks) / duration_minutes if duration_minutes and duration_minutes > 0 else math.nan
    else:
        duration_minutes = (len(smoothed) / config.fps) / 60 if config.fps > 0 else math.nan
        rr_bpm = len(peaks) / duration_minutes if duration_minutes and duration_minutes > 0 else math.nan
    return {
        "selected_fusion_mode": selected_fusion_mode,
        "fused": fused,
        "fused_repaired": fused_repaired & fused.notna(),
        "selection_quality": selection_quality,
        "smooth_window": max(1, int(config.smooth_window)),
        "smoothed": smoothed,
        "smooth_offset": smooth_offset,
        "peaks": peaks,
        "properties": properties,
        "selected_peak_prominence": selected_peak_prominence,
        "rr_bpm": rr_bpm,
        "merged_peak_count": merged_peak_count,
        "peak_distance": int(config.peak_distance),
        "peak_prominence": float(config.peak_prominence),
        "adaptive_peak_prominence": str(config.adaptive_peak_prominence),
        "edge_peak_completion": config.edge_peak_completion,
        "edge_peak_added": int(len(edge_added)),
        "edge_peak_added_frames": ";".join(str(int(peak)) for peak in edge_added),
        "edge_peak_min_gap": int(config.edge_peak_min_gap),
        "edge_peak_min_relief": float(config.edge_peak_min_relief),
        "source_rejected_peak_count": int(len(source_rejected_frames)),
        "source_rejected_peak_frames": ";".join(map(str, source_rejected_frames)),
        "source_peak_gate": str(config.source_peak_gate),
        "source_peak_support_radius": int(config.source_peak_support_radius),
        "quality_rejected_peak_count": int(len(quality_rejected_frames)),
        "quality_rejected_peak_frames": ";".join(map(str, quality_rejected_frames)),
        "mad_ibi_rejected_peak_count": int(len(mad_ibi_rejected_frames)),
        "mad_ibi_rejected_peak_frames": ";".join(map(str, mad_ibi_rejected_frames)),
        "event_sources": ";".join(event_sources),
        "event_bilateral_matches": int(bilateral_matches),
        "event_left_only": int(event_left_only),
        "event_right_only": int(event_right_only),
        "event_unilateral_rejected_count": int(len(event_unilateral_rejected)),
        "event_unilateral_rejected_frames": ";".join(map(str, event_unilateral_rejected)),
        "event_match_tolerance_frames": int(event_match_tolerance),
        "motion_peak_gate": str(config.motion_peak_gate),
        "motion_peak_gate_radius": int(config.motion_peak_gate_radius),
        "motion_rejected_peak_count": int(len(motion_rejected_frames)),
        "motion_rejected_peak_frames": ";".join(map(str, motion_rejected_frames)),
        "motion_interval_regularization": str(config.motion_interval_regularization),
        "motion_short_interval_ratio": float(config.motion_short_interval_ratio),
        "motion_interval_min_peaks": int(config.motion_interval_min_peaks),
        "motion_interval_min_improvement": float(config.motion_interval_min_improvement),
        "motion_interval_rejected_peak_count": int(len(motion_interval_rejected_frames)),
        "motion_interval_rejected_peak_frames": ";".join(
            map(str, motion_interval_rejected_frames)
        ),
        "motion_reconstruct_missing_cycles": str(config.motion_reconstruct_missing_cycles),
        "motion_long_interval_ratio": float(config.motion_long_interval_ratio),
        "motion_cycle_tolerance": float(config.motion_cycle_tolerance),
        "motion_reconstruction_search_radius": int(config.motion_reconstruction_search_radius),
        "motion_reconstructed_peak_count": int(len(motion_reconstructed_frames)),
        "motion_reconstructed_peak_frames": ";".join(map(str, motion_reconstructed_frames)),
        "retune_rule": "none",
        "retune_target_peaks": math.nan,
    }


def mean_prominence(properties: dict[str, np.ndarray]) -> float:
    prominences = properties.get("prominences", np.array([math.nan]))
    return float(np.mean(prominences)) if len(prominences) else math.nan


def retune_peak_detection(
    temp_df: pd.DataFrame,
    curve: pd.DataFrame,
    config: ReproConfig,
    detection: dict[str, object],
    fusion_selection_config: ReproConfig | None = None,
) -> dict[str, object]:
    if (
        not config.adaptive_peak_retuning
        or config.breath_detector != "peak"
        or config.event_level_fusion
    ):
        return detection

    best_detection = detection
    initial_count = int(len(detection["peaks"]))

    if initial_count >= config.dense_peak_threshold and int(detection["merged_peak_count"]) == 0:
        best_score: tuple[float, float, float] | None = None
        for smooth_window in [1, 3]:
            for peak_distance in [7, 8, 9, 10]:
                for peak_prominence in [0.002, 0.035, 0.07, 0.09, 0.12, 0.16, 0.2]:
                    trial_config = replace(
                        config,
                        smooth_window=smooth_window,
                        peak_distance=peak_distance,
                        peak_prominence=peak_prominence,
                        adaptive_peak_prominence=False,
                        adaptive_peak_retuning=False,
                    )
                    trial = detect_curve(temp_df, curve, trial_config, fusion_selection_config)
                    trial_count = int(len(trial["peaks"]))
                    if trial_count >= initial_count:
                        continue
                    score = (
                        abs(trial_count - config.dense_peak_target),
                        -mean_prominence(trial["properties"]),
                        abs(trial_count - initial_count),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        trial["retune_rule"] = "dense_peak_count"
                        trial["retune_target_peaks"] = int(config.dense_peak_target)
                        best_detection = trial

    current_count = int(len(best_detection["peaks"]))
    current_smoothed_frames = int(len(best_detection["smoothed"]))
    if current_count == config.sparse_peak_count and current_smoothed_frames <= config.sparse_max_smoothed_frames:
        trial_config = replace(
            config,
            smooth_window=1,
            peak_distance=3,
            peak_prominence=0.002,
            adaptive_peak_prominence=False,
            adaptive_peak_retuning=False,
        )
        trial = detect_curve(temp_df, curve, trial_config, fusion_selection_config)
        trial_count = int(len(trial["peaks"]))
        if current_count <= trial_count <= current_count + 2:
            trial["retune_rule"] = "short_sparse_peak_count"
            trial["retune_target_peaks"] = math.nan
            best_detection = trial
            if config.short_sparse_edge_completion:
                edge_config = replace(
                    trial_config,
                    edge_peak_completion="both",
                    edge_peak_min_gap=config.short_sparse_edge_min_gap,
                    edge_peak_min_relief=config.short_sparse_edge_min_relief,
                    edge_peak_max_frames=config.short_sparse_edge_max_frames,
                )
                edge_trial = detect_curve(temp_df, curve, edge_config, fusion_selection_config)
                edge_trial_count = int(len(edge_trial["peaks"]))
                if trial_count < edge_trial_count <= current_count + 2:
                    edge_trial["retune_rule"] = "short_sparse_peak_count_edge"
                    edge_trial["retune_target_peaks"] = math.nan
                    best_detection = edge_trial

    return best_detection


def fuse_temperature_curve(
    temp_df: pd.DataFrame,
    config: ReproConfig,
    truth_row: pd.Series | None = None,
    fusion_selection_config: ReproConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    temp_df = temp_df.copy()
    motion = nostril_motion_features(temp_df, config)
    for column in motion.columns:
        temp_df[column] = motion[column]
    curve = temp_df[["frame_name", "left_temp", "right_temp"]].copy()
    curve["frame_index"] = np.arange(len(curve))
    for column in motion.columns:
        curve[column] = motion[column].to_numpy()

    curve["left_missing_raw"] = curve["left_temp"].isna()
    curve["right_missing_raw"] = curve["right_temp"].isna()
    if config.quality_gate:
        curve["left_quality_valid"] = side_quality_mask(temp_df, "left", config)
        curve["right_quality_valid"] = side_quality_mask(temp_df, "right", config)
    else:
        curve["left_quality_valid"] = curve["left_temp"].notna()
        curve["right_quality_valid"] = curve["right_temp"].notna()
    curve["quality_any_valid"] = (
        curve["left_quality_valid"] | curve["right_quality_valid"]
    )
    motion_mask = curve["motion_artifact"].fillna(False)
    if config.quality_gate:
        left_for_curve = curve["left_temp"].where(curve["left_quality_valid"])
        right_for_curve = curve["right_temp"].where(curve["right_quality_valid"])
    elif config.motion_artifact_suppression:
        left_for_curve = curve["left_temp"].mask(motion_mask)
        right_for_curve = curve["right_temp"].mask(motion_mask)
    else:
        left_for_curve = curve["left_temp"].copy()
        right_for_curve = curve["right_temp"].copy()
    curve["left_temp_motion_filtered"] = left_for_curve
    curve["right_temp_motion_filtered"] = right_for_curve
    if config.quality_gate and config.repair_missing:
        curve["left_temp_used"] = interpolate_short_interior_runs(
            left_for_curve, config.quality_max_interp_gap
        )
        curve["right_temp_used"] = interpolate_short_interior_runs(
            right_for_curve, config.quality_max_interp_gap
        )
    else:
        curve["left_temp_used"] = repair_series(left_for_curve, config.repair_missing)
        curve["right_temp_used"] = repair_series(right_for_curve, config.repair_missing)
    curve["left_repaired"] = curve["left_missing_raw"] & curve["left_temp_used"].notna()
    curve["right_repaired"] = curve["right_missing_raw"] & curve["right_temp_used"].notna()
    curve["left_motion_interpolated"] = (
        motion_mask & curve["left_temp"].notna() & curve["left_temp_used"].notna()
    )
    curve["right_motion_interpolated"] = (
        motion_mask & curve["right_temp"].notna() & curve["right_temp_used"].notna()
    )

    curve["left_norm"] = normalize_series(curve["left_temp_used"])
    curve["right_norm"] = normalize_series(curve["right_temp_used"])

    sources = []
    for row in curve.itertuples(index=False):
        left_present = not pd.isna(row.left_temp)
        right_present = not pd.isna(row.right_temp)
        if left_present and right_present:
            sources.append("both")
        elif left_present:
            sources.append("left_only")
        elif right_present:
            sources.append("right_only")
        else:
            sources.append("missing")

    detection = detect_curve(temp_df, curve, config, fusion_selection_config)
    detection = retune_peak_detection(temp_df, curve, config, detection, fusion_selection_config)
    selected_fusion_mode = str(detection["selected_fusion_mode"])
    fused = detection["fused"]
    fused_repaired = detection["fused_repaired"]
    selection_quality = detection["selection_quality"]
    smooth_window = int(detection["smooth_window"])
    smoothed = detection["smoothed"]
    smooth_offset = int(detection["smooth_offset"])
    peaks = detection["peaks"]
    properties = detection["properties"]
    selected_peak_prominence = float(detection["selected_peak_prominence"])
    merged_peak_count = int(detection["merged_peak_count"])
    smoothed_duration_seconds = float(len(smoothed) / config.fps) if config.fps > 0 else math.nan
    rr_duration, rr_duration_source = rr_duration_seconds(len(smoothed), config, truth_row)
    time_peak_count = int(len(peaks))
    time_frequency = estimate_time_frequency_consensus(
        smoothed,
        fps=config.fps,
        duration_seconds=rr_duration,
        min_rr_bpm=config.min_rr_bpm,
        max_rr_bpm=config.max_rr_bpm,
        time_peak_count=time_peak_count,
        enabled=config.rr_estimator == "time_frequency_consensus",
        spectral_min_concentration=config.spectral_min_concentration,
        autocorr_min_correlation=config.autocorr_min_correlation,
        max_count_adjustment=config.joint_max_count_adjustment,
    )
    final_count = int(time_frequency["joint_count"])
    rr_bpm = rr_from_peak_count(final_count, rr_duration)

    curve["fused_norm"] = fused.to_numpy(dtype=float)
    curve["fused_repaired"] = fused_repaired
    curve["fusion_source"] = sources
    curve["selected_fusion_mode"] = selected_fusion_mode
    curve["peak_retune_rule"] = str(detection["retune_rule"])
    curve["edge_peak_completion"] = str(detection["edge_peak_completion"])
    curve["edge_peak_added"] = int(detection["edge_peak_added"])
    curve["smoothing_method"] = config.smoothing_method
    curve["butterworth_order"] = int(config.butterworth_order)
    curve["butterworth_cutoff_hz"] = float(config.butterworth_cutoff_hz)
    curve["rr_estimator"] = config.rr_estimator
    curve["time_peak_count"] = time_peak_count
    curve["spectral_count"] = int(time_frequency["spectral_count"])
    curve["autocorr_count"] = int(time_frequency["autocorr_count"])
    curve["joint_count"] = final_count
    curve["joint_adjusted"] = bool(time_frequency["joint_adjusted"])

    curve["smoothed_norm"] = np.nan
    if len(smoothed) > 0:
        smoothed_index = np.arange(smooth_offset, smooth_offset + len(smoothed))
        smoothed_index = smoothed_index[smoothed_index < len(curve)]
        curve.loc[smoothed_index, "smoothed_norm"] = smoothed[: len(smoothed_index)]
    curve["is_peak"] = False
    peak_frames = peaks + smooth_offset
    peak_frames = peak_frames[peak_frames < len(curve)]
    curve.loc[peak_frames, "is_peak"] = True
    curve["source_peak_rejected"] = False
    rejected_frames = [
        int(value)
        for value in str(detection.get("source_rejected_peak_frames", "")).split(";")
        if value.strip()
    ]
    rejected_frames = [frame for frame in rejected_frames if 0 <= frame < len(curve)]
    if rejected_frames:
        curve.loc[rejected_frames, "source_peak_rejected"] = True
    curve["quality_peak_rejected"] = False
    quality_rejected_frames = [
        int(value)
        for value in str(detection.get("quality_rejected_peak_frames", "")).split(";")
        if value.strip()
    ]
    quality_rejected_frames = [
        frame for frame in quality_rejected_frames if 0 <= frame < len(curve)
    ]
    if quality_rejected_frames:
        curve.loc[quality_rejected_frames, "quality_peak_rejected"] = True
    curve["mad_ibi_peak_rejected"] = False
    mad_ibi_rejected_frames = [
        int(value)
        for value in str(detection.get("mad_ibi_rejected_peak_frames", "")).split(";")
        if value.strip()
    ]
    mad_ibi_rejected_frames = [
        frame for frame in mad_ibi_rejected_frames if 0 <= frame < len(curve)
    ]
    if mad_ibi_rejected_frames:
        curve.loc[mad_ibi_rejected_frames, "mad_ibi_peak_rejected"] = True
    curve["motion_peak_rejected"] = False
    motion_rejected_frames = [
        int(value)
        for value in str(detection.get("motion_rejected_peak_frames", "")).split(";")
        if value.strip()
    ]
    motion_rejected_frames = [
        frame for frame in motion_rejected_frames if 0 <= frame < len(curve)
    ]
    if motion_rejected_frames:
        curve.loc[motion_rejected_frames, "motion_peak_rejected"] = True
    curve["motion_interval_peak_rejected"] = False
    interval_rejected_frames = [
        int(value)
        for value in str(
            detection.get("motion_interval_rejected_peak_frames", "")
        ).split(";")
        if value.strip()
    ]
    interval_rejected_frames = [
        frame for frame in interval_rejected_frames if 0 <= frame < len(curve)
    ]
    if interval_rejected_frames:
        curve.loc[interval_rejected_frames, "motion_interval_peak_rejected"] = True
    curve["motion_reconstructed_peak"] = False
    reconstructed_frames = [
        int(value)
        for value in str(detection.get("motion_reconstructed_peak_frames", "")).split(";")
        if value.strip()
    ]
    reconstructed_frames = [
        frame for frame in reconstructed_frames if 0 <= frame < len(curve)
    ]
    if reconstructed_frames:
        curve.loc[reconstructed_frames, "motion_reconstructed_peak"] = True
    curve["peak_number"] = pd.NA
    curve["event_source"] = ""
    event_sources = [
        value for value in str(detection.get("event_sources", "")).split(";") if value
    ]
    for peak_number, frame_index in enumerate(peak_frames, start=1):
        curve.loc[int(frame_index), "peak_number"] = peak_number
        if peak_number - 1 < len(event_sources):
            curve.loc[int(frame_index), "event_source"] = event_sources[peak_number - 1]

    truth_count = int(truth_row["breath_count"]) if truth_row is not None and "breath_count" in truth_row else math.nan
    truth_rr = float(truth_row["rr"]) if truth_row is not None and "rr" in truth_row else math.nan
    roi_radii = pd.to_numeric(temp_df.get("roi_radius", pd.Series(dtype=float)), errors="coerce").dropna()
    nostril_spacings = pd.to_numeric(
        temp_df.get("nostril_spacing", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    tracking_displacements = pd.concat(
        [
            pd.to_numeric(
                temp_df.get("left_tracking_displacement", pd.Series(dtype=float)), errors="coerce"
            ),
            pd.to_numeric(
                temp_df.get("right_tracking_displacement", pd.Series(dtype=float)), errors="coerce"
            ),
        ],
        ignore_index=True,
    ).dropna()
    registration_flags = temp_df.get(
        "registration_applied", pd.Series(False, index=temp_df.index)
    ).fillna(False).astype(bool)
    summary = {
        "frames": int(len(curve)),
        "smoothed_frames": int(len(smoothed)),
        "missing_left": int(curve["left_temp"].isna().sum()),
        "missing_right": int(curve["right_temp"].isna().sum()),
        "missing_both": int(((curve["left_temp"].isna()) & (curve["right_temp"].isna())).sum()),
        "adaptive_roi": str(config.adaptive_roi),
        "base_roi_radius": int(config.radius),
        "adaptive_roi_scale": float(config.adaptive_roi_scale),
        "roi_radius_mean": float(roi_radii.mean()) if not roi_radii.empty else float(config.radius),
        "roi_radius_min": int(roi_radii.min()) if not roi_radii.empty else int(config.radius),
        "roi_radius_max": int(roi_radii.max()) if not roi_radii.empty else int(config.radius),
        "nostril_spacing_median": float(nostril_spacings.median()) if not nostril_spacings.empty else math.nan,
        "keypoint_tracking": str(config.keypoint_tracking),
        "motion_registration": str(config.motion_registration),
        "keypoint_source_prefix": config.keypoint_source_prefix or "",
        "tracking_displacement_mean": (
            float(tracking_displacements.mean()) if not tracking_displacements.empty else 0.0
        ),
        "tracking_displacement_p95": (
            float(tracking_displacements.quantile(0.95)) if not tracking_displacements.empty else 0.0
        ),
        "registration_applied_frames": int(registration_flags.sum()),
        "temperature_source_prefix": config.temperature_source_prefix or "",
        "quality_gate": str(config.quality_gate),
        "quality_min_confidence": float(config.quality_min_confidence),
        "quality_max_interp_gap": int(config.quality_max_interp_gap),
        "quality_reject_motion": str(config.quality_reject_motion),
        "quality_valid_frames": int(curve["quality_any_valid"].sum()),
        "quality_coverage": float(curve["quality_any_valid"].mean()),
        "quality_longest_invalid_run": int(longest_false_run(curve["quality_any_valid"])),
        "repaired_left": int(curve["left_repaired"].sum()),
        "repaired_right": int(curve["right_repaired"].sum()),
        "tracked_left": int(temp_df.get("left_source", pd.Series(dtype=str)).eq("tracked").sum()),
        "tracked_right": int(temp_df.get("right_source", pd.Series(dtype=str)).eq("tracked").sum()),
        "inferred_left": int(temp_df.get("left_source", pd.Series(dtype=str)).eq("opposite_inferred").sum()),
        "inferred_right": int(temp_df.get("right_source", pd.Series(dtype=str)).eq("opposite_inferred").sum()),
        "global_inferred_left": int(temp_df.get("left_source", pd.Series(dtype=str)).eq("global_offset_inferred").sum()),
        "global_inferred_right": int(temp_df.get("right_source", pd.Series(dtype=str)).eq("global_offset_inferred").sum()),
        "low_conf_inferred_left": int(temp_df.get("left_source", pd.Series(dtype=str)).eq("low_conf_inferred").sum()),
        "low_conf_inferred_right": int(temp_df.get("right_source", pd.Series(dtype=str)).eq("low_conf_inferred").sum()),
        "peaks": final_count,
        "time_peak_count": time_peak_count,
        "breath_detector": config.breath_detector,
        "mad_bandpass_low_hz": float(config.mad_bandpass_low_hz),
        "mad_bandpass_high_hz": float(config.mad_bandpass_high_hz),
        "mad_threshold_multiplier": float(config.mad_threshold_multiplier),
        "mad_ibi_rejected_peak_count": int(detection["mad_ibi_rejected_peak_count"]),
        "mad_ibi_rejected_peak_frames": str(detection["mad_ibi_rejected_peak_frames"]),
        "event_level_fusion": str(config.event_level_fusion),
        "event_bilateral_matches": int(detection["event_bilateral_matches"]),
        "event_left_only": int(detection["event_left_only"]),
        "event_right_only": int(detection["event_right_only"]),
        "event_unilateral_rejected_count": int(detection["event_unilateral_rejected_count"]),
        "event_unilateral_rejected_frames": str(detection["event_unilateral_rejected_frames"]),
        "event_match_tolerance_frames": int(detection["event_match_tolerance_frames"]),
        "rr_estimator": config.rr_estimator,
        "spectral_rr_bpm": float(time_frequency["spectral_rr_bpm"]),
        "spectral_count": int(time_frequency["spectral_count"]),
        "spectral_concentration": float(time_frequency["spectral_concentration"]),
        "autocorr_rr_bpm": float(time_frequency["autocorr_rr_bpm"]),
        "autocorr_count": int(time_frequency["autocorr_count"]),
        "autocorr_correlation": float(time_frequency["autocorr_correlation"]),
        "joint_adjusted": str(time_frequency["joint_adjusted"]),
        "joint_reason": str(time_frequency["joint_reason"]),
        "rr_bpm": float(rr_bpm),
        "duration_seconds": float(rr_duration),
        "rr_duration_seconds": float(rr_duration),
        "rr_duration_source": rr_duration_source,
        "smoothed_duration_seconds": smoothed_duration_seconds,
        "mean_prominence": float(np.mean(properties.get("prominences", [math.nan]))),
        "truth_count": truth_count,
        "count_error": int(final_count - truth_count) if not pd.isna(truth_count) else math.nan,
        "abs_count_error": abs(int(final_count - truth_count)) if not pd.isna(truth_count) else math.nan,
        "truth_rr": truth_rr,
        "rr_error": float(rr_bpm - truth_rr) if not pd.isna(truth_rr) else math.nan,
        "abs_rr_error": abs(float(rr_bpm - truth_rr)) if not pd.isna(truth_rr) else math.nan,
        "fusion_mode": config.fusion_mode,
        "selected_fusion_mode": selected_fusion_mode,
        "selection_score": float(selection_quality.get("score", math.nan)),
        "selection_candidate_peaks": int(selection_quality.get("candidate_peaks", 0)),
        "selection_median_prominence": float(selection_quality.get("candidate_median_prominence", math.nan)),
        "selection_interval_cv": float(selection_quality.get("candidate_interval_cv", math.nan)),
        "selection_amplitude": float(selection_quality.get("candidate_amplitude", math.nan)),
        "selection_missing_rate": float(selection_quality.get("candidate_missing_rate", math.nan)),
        "selection_motion_coupling": float(
            selection_quality.get("candidate_motion_coupling", math.nan)
        ),
        "fusion_amplitude_weight": float(config.fusion_amplitude_weight),
        "fusion_interval_cv_weight": float(config.fusion_interval_cv_weight),
        "fusion_missing_weight": float(config.fusion_missing_weight),
        "fusion_motion_penalty_weight": float(config.fusion_motion_penalty_weight),
        "fusion_rescue_interval_cv_weight": float(config.fusion_rescue_interval_cv_weight),
        "fusion_rescue_outlier_threshold": int(config.fusion_rescue_outlier_threshold),
        "fusion_rescue_applied": str(selection_quality.get("fusion_rescue_applied", False)),
        "legacy_selected_fusion_mode": str(
            selection_quality.get("legacy_selected_fusion_mode", selected_fusion_mode)
        ),
        "legacy_candidate_peaks": int(
            selection_quality.get("legacy_candidate_peaks", len(peaks))
        ),
        "fusion_consensus_count": float(
            selection_quality.get("fusion_consensus_count", math.nan)
        ),
        "legacy_consensus_distance": float(
            selection_quality.get("legacy_consensus_distance", math.nan)
        ),
        "robust_consensus_distance": float(
            selection_quality.get("robust_consensus_distance", math.nan)
        ),
        "repair_missing": str(config.repair_missing),
        "smooth_window": smooth_window,
        "smoothing_method": config.smoothing_method,
        "butterworth_order": int(config.butterworth_order),
        "butterworth_cutoff_hz": float(config.butterworth_cutoff_hz),
        "peak_distance": int(detection["peak_distance"]),
        "peak_prominence": float(detection["peak_prominence"]),
        "selected_peak_prominence": float(selected_peak_prominence),
        "adaptive_peak_prominence": str(detection["adaptive_peak_prominence"]),
        "min_rr_bpm": float(config.min_rr_bpm),
        "max_rr_bpm": float(config.max_rr_bpm),
        "spectral_min_concentration": float(config.spectral_min_concentration),
        "autocorr_min_correlation": float(config.autocorr_min_correlation),
        "joint_max_count_adjustment": int(config.joint_max_count_adjustment),
        "merge_shallow_peaks": str(config.merge_shallow_peaks),
        "merge_peak_gap": int(config.merge_peak_gap),
        "merge_valley_relief": float(config.merge_valley_relief),
        "merged_peak_count": int(merged_peak_count),
        "adaptive_peak_retuning": str(config.adaptive_peak_retuning),
        "peak_retune_rule": str(detection["retune_rule"]),
        "retune_target_peaks": detection["retune_target_peaks"],
        "edge_peak_completion": str(detection["edge_peak_completion"]),
        "edge_peak_added": int(detection["edge_peak_added"]),
        "edge_peak_added_frames": str(detection["edge_peak_added_frames"]),
        "edge_peak_min_gap": int(detection["edge_peak_min_gap"]),
        "edge_peak_min_relief": float(detection["edge_peak_min_relief"]),
        "source_peak_gate": str(detection["source_peak_gate"]),
        "source_peak_support_radius": int(detection["source_peak_support_radius"]),
        "source_rejected_peak_count": int(detection["source_rejected_peak_count"]),
        "source_rejected_peak_frames": str(detection["source_rejected_peak_frames"]),
        "quality_rejected_peak_count": int(detection["quality_rejected_peak_count"]),
        "quality_rejected_peak_frames": str(detection["quality_rejected_peak_frames"]),
        "motion_artifact_suppression": str(config.motion_artifact_suppression),
        "motion_center_step_threshold": float(config.motion_center_step_threshold),
        "motion_scale_step_threshold": float(config.motion_scale_step_threshold),
        "motion_angle_step_threshold": float(config.motion_angle_step_threshold),
        "motion_mad_multiplier": float(config.motion_mad_multiplier),
        "motion_mask_radius": int(config.motion_mask_radius),
        "motion_artifact_raw_frames": int(curve["motion_artifact_raw"].sum()),
        "motion_artifact_frames": int(curve["motion_artifact"].sum()),
        "motion_artifact_fraction": float(curve["motion_artifact"].mean()),
        "motion_stable_fraction": float(1.0 - curve["motion_artifact"].mean()),
        "motion_score_p95": float(curve["motion_score"].quantile(0.95)),
        "motion_score_max": float(curve["motion_score"].max()),
        "motion_center_threshold_used": float(curve["motion_center_threshold"].iloc[0]),
        "motion_scale_threshold_used": float(curve["motion_scale_threshold"].iloc[0]),
        "motion_angle_threshold_used": float(curve["motion_angle_threshold"].iloc[0]),
        "motion_interpolated_frames": int(
            (curve["left_motion_interpolated"] | curve["right_motion_interpolated"]).sum()
        ),
        "motion_peak_gate": str(detection["motion_peak_gate"]),
        "motion_peak_gate_radius": int(detection["motion_peak_gate_radius"]),
        "motion_rejected_peak_count": int(detection["motion_rejected_peak_count"]),
        "motion_rejected_peak_frames": str(detection["motion_rejected_peak_frames"]),
        "motion_interval_regularization": str(
            detection["motion_interval_regularization"]
        ),
        "motion_short_interval_ratio": float(detection["motion_short_interval_ratio"]),
        "motion_interval_min_peaks": int(detection["motion_interval_min_peaks"]),
        "motion_interval_min_improvement": float(
            detection["motion_interval_min_improvement"]
        ),
        "motion_interval_rejected_peak_count": int(
            detection["motion_interval_rejected_peak_count"]
        ),
        "motion_interval_rejected_peak_frames": str(
            detection["motion_interval_rejected_peak_frames"]
        ),
        "motion_reconstruct_missing_cycles": str(
            detection["motion_reconstruct_missing_cycles"]
        ),
        "motion_long_interval_ratio": float(detection["motion_long_interval_ratio"]),
        "motion_cycle_tolerance": float(detection["motion_cycle_tolerance"]),
        "motion_reconstruction_search_radius": int(
            detection["motion_reconstruction_search_radius"]
        ),
        "motion_reconstructed_peak_count": int(
            detection["motion_reconstructed_peak_count"]
        ),
        "motion_reconstructed_peak_frames": str(
            detection["motion_reconstructed_peak_frames"]
        ),
    }
    return curve, summary


def plot_outputs(curve: pd.DataFrame, video_id: str, summary: dict[str, float | int], output_path: Path) -> None:
    peaks = curve.index[curve["is_peak"]].to_numpy()
    smoothed = curve["smoothed_norm"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(curve["frame_index"], curve["left_temp"], label="left nostril", linewidth=1.4)
    axes[0].plot(curve["frame_index"], curve["right_temp"], label="right nostril", linewidth=1.4)
    axes[0].set_ylabel("Temperature")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(curve["frame_index"], curve["fused_norm"], label="fused normalized", alpha=0.45)
    axes[1].plot(curve["frame_index"], smoothed, label=f"MAF window={summary['smooth_window']}", linewidth=1.8)
    if len(peaks) > 0:
        axes[1].scatter(peaks, smoothed[peaks], color="black", s=28, label="peaks", zorder=3)
    rejected = curve.index[
        curve.get("source_peak_rejected", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(rejected) > 0:
        axes[1].scatter(
            rejected,
            curve.loc[rejected, "smoothed_norm"],
            marker="x",
            color="#d62728",
            s=42,
            label="source-rejected peaks",
            zorder=4,
        )
    motion_rejected = curve.index[
        curve.get("motion_peak_rejected", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(motion_rejected) > 0:
        axes[1].scatter(
            motion_rejected,
            curve.loc[motion_rejected, "smoothed_norm"],
            marker="x",
            color="#ff7f0e",
            s=46,
            label="motion-rejected peaks",
            zorder=4,
        )
    reconstructed = curve.index[
        curve.get("motion_reconstructed_peak", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(reconstructed) > 0:
        axes[1].scatter(
            reconstructed,
            curve.loc[reconstructed, "smoothed_norm"],
            marker="^",
            color="#2ca02c",
            s=48,
            label="motion-reconstructed peaks",
            zorder=5,
        )
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{video_id}: peaks={summary['peaks']}, RR={summary['rr_bpm']:.2f} bpm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_peak_review(
    curve: pd.DataFrame,
    video_id: str,
    summary: dict[str, float | int | str],
    output_path: Path,
) -> None:
    peaks = curve.index[curve["is_peak"]].to_numpy()
    peak_values = curve.loc[peaks, "smoothed_norm"].to_numpy(dtype=float) if len(peaks) else np.array([])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        curve["frame_index"],
        curve["fused_norm"],
        color="#9ecae1",
        linewidth=1.2,
        alpha=0.65,
        label="fused normalized curve",
    )
    ax.plot(
        curve["frame_index"],
        curve["smoothed_norm"],
        color="#1f77b4",
        linewidth=2.4,
        label=f"smoothed curve, window={summary['smooth_window']}",
    )

    if len(peaks) > 0:
        ax.scatter(peaks, peak_values, color="black", s=48, zorder=4, label="detected breath peaks")
        for idx, (frame, value) in enumerate(zip(peaks, peak_values), start=1):
            if not math.isnan(value):
                ax.annotate(
                    str(idx),
                    (frame, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )
                ax.axvline(frame, color="black", linewidth=0.5, alpha=0.12)

    repaired_frames = curve.index[
        curve.get("left_repaired", pd.Series(False, index=curve.index))
        | curve.get("right_repaired", pd.Series(False, index=curve.index))
        | curve.get("fused_repaired", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(repaired_frames) > 0:
        ax.scatter(
            repaired_frames,
            curve.loc[repaired_frames, "fused_norm"],
            marker="|",
            color="#d62728",
            s=55,
            alpha=0.7,
            label="repaired missing samples",
        )

    motion_frames = curve.index[
        curve.get("motion_artifact", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(motion_frames) > 0:
        ax.scatter(
            motion_frames,
            curve.loc[motion_frames, "fused_norm"],
            marker="|",
            color="#ff7f0e",
            s=58,
            alpha=0.7,
            label="abrupt-motion samples",
        )

    rejected_frames = curve.index[
        curve.get("source_peak_rejected", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(rejected_frames) > 0:
        ax.scatter(
            rejected_frames,
            curve.loc[rejected_frames, "smoothed_norm"],
            marker="x",
            color="#d62728",
            s=70,
            linewidths=1.8,
            zorder=5,
            label="source-rejected peaks",
        )
    motion_rejected_frames = curve.index[
        curve.get("motion_peak_rejected", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(motion_rejected_frames) > 0:
        ax.scatter(
            motion_rejected_frames,
            curve.loc[motion_rejected_frames, "smoothed_norm"],
            marker="x",
            color="#ff7f0e",
            s=74,
            linewidths=1.8,
            zorder=5,
            label="motion-rejected peaks",
        )
    interval_rejected_frames = curve.index[
        curve.get("motion_interval_peak_rejected", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(interval_rejected_frames) > 0:
        ax.scatter(
            interval_rejected_frames,
            curve.loc[interval_rejected_frames, "smoothed_norm"],
            marker="x",
            color="#9467bd",
            s=74,
            linewidths=1.8,
            zorder=5,
            label="motion-duplicate peaks",
        )
    reconstructed_frames = curve.index[
        curve.get("motion_reconstructed_peak", pd.Series(False, index=curve.index))
    ].to_numpy()
    if len(reconstructed_frames) > 0:
        ax.scatter(
            reconstructed_frames,
            curve.loc[reconstructed_frames, "smoothed_norm"],
            marker="^",
            color="#2ca02c",
            s=76,
            zorder=6,
            label="motion-reconstructed peaks",
        )

    truth_count = summary.get("truth_count", math.nan)
    truth_part = ""
    if not pd.isna(truth_count):
        truth_part = f", truth={int(truth_count)}, error={int(summary['count_error']):+d}"
    ax.set_title(
        f"{video_id}: predicted breaths={summary['peaks']}{truth_part} "
        f"(mode={summary['fusion_mode']}->{summary.get('selected_fusion_mode', summary['fusion_mode'])}, "
        f"distance={summary['peak_distance']}, prominence={summary.get('selected_peak_prominence', summary['peak_prominence'])}, "
        f"merged={summary.get('merged_peak_count', 0)}, edge={summary.get('edge_peak_added', 0)}, "
        f"source_rejected={summary.get('source_rejected_peak_count', 0)}, "
        f"motion_frames={summary.get('motion_artifact_frames', 0)}, "
        f"motion_rejected={summary.get('motion_rejected_peak_count', 0)}, "
        f"motion_duplicates={summary.get('motion_interval_rejected_peak_count', 0)}, "
        f"motion_reconstructed={summary.get('motion_reconstructed_peak_count', 0)}, "
        f"fusion_rescue={summary.get('fusion_rescue_applied', False)})"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized temperature")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def process_video(
    video_dir: Path,
    yolo_model,
    temp_model,
    config: ReproConfig,
    truth_row: pd.Series | None = None,
    global_offset: tuple[float, float] | None = None,
) -> dict[str, object]:
    temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
    curve_csv = video_dir / f"{config.output_prefix}_curve.csv"
    curve_png = video_dir / f"{config.output_prefix}_curve.png"
    review_png = video_dir / f"{config.output_prefix}_peak_review.png"
    if (
        not config.overwrite
        and not config.reuse_temperatures
        and temp_csv.exists()
        and curve_csv.exists()
        and curve_png.exists()
        and review_png.exists()
    ):
        curve = pd.read_csv(curve_csv)
        peaks = int(pd.Series(curve["is_peak"]).astype(bool).sum())
        smoothed_frames = int(curve["smoothed_norm"].notna().sum())
        rr_duration, rr_duration_source = rr_duration_seconds(smoothed_frames, config, truth_row)
        rr_bpm = rr_from_peak_count(peaks, rr_duration)
        smoothed_duration_seconds = float(smoothed_frames / config.fps) if config.fps > 0 else math.nan
        truth_count = int(truth_row["breath_count"]) if truth_row is not None and "breath_count" in truth_row else math.nan
        truth_rr = float(truth_row["rr"]) if truth_row is not None and "rr" in truth_row else math.nan
        selected_fusion_mode = (
            str(curve["selected_fusion_mode"].dropna().iloc[0])
            if "selected_fusion_mode" in curve.columns and curve["selected_fusion_mode"].notna().any()
            else config.fusion_mode
        )
        return {
            "video_id": video_dir.name,
            "frames": int(len(curve)),
            "smoothed_frames": smoothed_frames,
            "missing_left": int(curve["left_temp"].isna().sum()),
            "missing_right": int(curve["right_temp"].isna().sum()),
            "missing_both": int(((curve["left_temp"].isna()) & (curve["right_temp"].isna())).sum()),
            "repaired_left": int(curve.get("left_repaired", pd.Series(False, index=curve.index)).sum()),
            "repaired_right": int(curve.get("right_repaired", pd.Series(False, index=curve.index)).sum()),
            "tracked_left": math.nan,
            "tracked_right": math.nan,
            "inferred_left": int(curve.get("left_source", pd.Series("", index=curve.index)).eq("opposite_inferred").sum())
            if "left_source" in curve.columns
            else math.nan,
            "inferred_right": int(curve.get("right_source", pd.Series("", index=curve.index)).eq("opposite_inferred").sum())
            if "right_source" in curve.columns
            else math.nan,
            "global_inferred_left": int(
                curve.get("left_source", pd.Series("", index=curve.index)).eq("global_offset_inferred").sum()
            )
            if "left_source" in curve.columns
            else math.nan,
            "global_inferred_right": int(
                curve.get("right_source", pd.Series("", index=curve.index)).eq("global_offset_inferred").sum()
            )
            if "right_source" in curve.columns
            else math.nan,
            "low_conf_inferred_left": int(
                curve.get("left_source", pd.Series("", index=curve.index)).eq("low_conf_inferred").sum()
            )
            if "left_source" in curve.columns
            else math.nan,
            "low_conf_inferred_right": int(
                curve.get("right_source", pd.Series("", index=curve.index)).eq("low_conf_inferred").sum()
            )
            if "right_source" in curve.columns
            else math.nan,
            "peaks": peaks,
            "rr_bpm": rr_bpm,
            "duration_seconds": rr_duration,
            "rr_duration_seconds": rr_duration,
            "rr_duration_source": rr_duration_source,
            "smoothed_duration_seconds": smoothed_duration_seconds,
            "mean_prominence": math.nan,
            "truth_count": truth_count,
            "count_error": int(peaks - truth_count) if not pd.isna(truth_count) else math.nan,
            "abs_count_error": abs(int(peaks - truth_count)) if not pd.isna(truth_count) else math.nan,
            "truth_rr": truth_rr,
            "rr_error": float(rr_bpm - truth_rr) if not pd.isna(truth_rr) else math.nan,
            "abs_rr_error": abs(float(rr_bpm - truth_rr)) if not pd.isna(truth_rr) else math.nan,
            "fusion_mode": config.fusion_mode,
            "selected_fusion_mode": selected_fusion_mode,
            "selection_score": math.nan,
            "selection_candidate_peaks": math.nan,
            "selection_median_prominence": math.nan,
            "selection_interval_cv": math.nan,
            "selection_amplitude": math.nan,
            "selection_missing_rate": math.nan,
            "repair_missing": str(config.repair_missing),
            "smooth_window": int(config.smooth_window),
            "peak_distance": int(config.peak_distance),
            "peak_prominence": float(config.peak_prominence),
            "selected_peak_prominence": math.nan,
            "adaptive_peak_prominence": str(config.adaptive_peak_prominence),
            "min_rr_bpm": float(config.min_rr_bpm),
            "max_rr_bpm": float(config.max_rr_bpm),
            "merge_shallow_peaks": str(config.merge_shallow_peaks),
            "merge_peak_gap": int(config.merge_peak_gap),
            "merge_valley_relief": float(config.merge_valley_relief),
            "merged_peak_count": math.nan,
            "temperature_csv": str(temp_csv),
            "curve_csv": str(curve_csv),
            "curve_png": str(curve_png),
            "review_png": str(review_png),
            "status": "reused",
        }

    if config.temperature_source_prefix:
        source_temp_csv = video_dir / f"{config.temperature_source_prefix}_temperatures.csv"
        if not source_temp_csv.exists():
            raise FileNotFoundError(source_temp_csv)
        temp_df = pd.read_csv(source_temp_csv)
        if config.track_missing:
            image_paths = frame_image_paths(video_dir)
            repair_short_missing_runs(temp_df, image_paths, temp_model, "left", config)
            repair_short_missing_runs(temp_df, image_paths, temp_model, "right", config)
            tracked = temp_df["left_source"].eq("tracked") | temp_df["right_source"].eq("tracked")
            temp_df.loc[tracked, "status"] = "tracked_repair"
        temp_df.to_csv(temp_csv, index=False)
        status = f"temperature_reused_from_{config.temperature_source_prefix}"
    elif config.keypoint_source_prefix:
        source_temp_csv = video_dir / f"{config.keypoint_source_prefix}_temperatures.csv"
        if not source_temp_csv.exists():
            raise FileNotFoundError(source_temp_csv)
        source_temp_df = pd.read_csv(source_temp_csv)
        temp_df = reextract_temperatures_from_keypoints(
            video_dir,
            source_temp_df,
            temp_model,
            config,
            fallback_offset=global_offset,
        )
        temp_df.to_csv(temp_csv, index=False)
        status = "tracked_registered" if config.motion_registration else "tracked"
    elif config.reuse_temperatures and temp_csv.exists():
        temp_df = pd.read_csv(temp_csv)
        image_paths = frame_image_paths(video_dir)
        tracked_left = tracked_right = 0
        if config.track_missing:
            tracked_left = repair_short_missing_runs(temp_df, image_paths, temp_model, "left", config)
            tracked_right = repair_short_missing_runs(temp_df, image_paths, temp_model, "right", config)
            tracked = temp_df["left_source"].eq("tracked") | temp_df["right_source"].eq("tracked")
            temp_df.loc[tracked, "status"] = "tracked_repair"
        inferred_left, inferred_right = repair_missing_from_opposite_side(
            temp_df, image_paths, temp_model, config, fallback_offset=global_offset
        )
        low_conf_left, low_conf_right = repair_missing_from_low_confidence_coords(
            temp_df, image_paths, temp_model, config
        )
        if tracked_left or tracked_right or inferred_left or inferred_right or low_conf_left or low_conf_right:
            temp_df.to_csv(temp_csv, index=False)
        status = "retuned"
    else:
        temp_df = extract_temperatures(video_dir, yolo_model, temp_model, config)
        temp_df.to_csv(temp_csv, index=False)
        status = "processed"

    analysis_df, window_metadata = limit_analysis_window(temp_df, config, truth_row)
    curve, summary = fuse_temperature_curve(analysis_df, config, truth_row=truth_row)
    summary.update(window_metadata)
    for key, value in window_metadata.items():
        curve[key] = value
    curve.to_csv(curve_csv, index=False)
    plot_outputs(curve, video_dir.name, summary, curve_png)
    plot_peak_review(curve, video_dir.name, summary, review_png)
    return {
        "video_id": video_dir.name,
        **summary,
        "temperature_csv": str(temp_csv),
        "curve_csv": str(curve_csv),
        "curve_png": str(curve_png),
        "review_png": str(review_png),
        "status": status,
    }


def load_truth(truth_csv: Path | None) -> pd.DataFrame | None:
    if truth_csv is None or not truth_csv.exists():
        return None
    truth = pd.read_csv(truth_csv)
    required = {"video_id", "breath_count"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV is missing columns: {sorted(missing)}")
    truth["video_id"] = truth["video_id"].astype(str)
    return truth


def truth_lookup(truth_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    if truth_df is None:
        return {}
    return {str(row["video_id"]): row for _, row in truth_df.iterrows()}


def write_error_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    if "abs_count_error" not in summary_df.columns:
        return
    report = summary_df.copy()
    report = report[report["abs_count_error"].notna()].sort_values(
        ["abs_count_error", "abs_rr_error", "video_id"],
        ascending=[False, False, True],
    )
    report.to_csv(output_path, index=False)


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = truth.notna() & pred.notna()
    if int(valid.sum()) < 2:
        return math.nan
    truth_values = truth[valid].to_numpy(dtype=float)
    pred_values = pred[valid].to_numpy(dtype=float)
    total_sum_squares = float(np.sum((truth_values - np.mean(truth_values)) ** 2))
    if math.isclose(total_sum_squares, 0.0):
        return math.nan
    residual_sum_squares = float(np.sum((truth_values - pred_values) ** 2))
    return float(1.0 - residual_sum_squares / total_sum_squares)


def pearson_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = truth.notna() & pred.notna()
    if int(valid.sum()) < 2:
        return math.nan
    truth_values = truth[valid].to_numpy(dtype=float)
    pred_values = pred[valid].to_numpy(dtype=float)
    if math.isclose(float(np.std(truth_values)), 0.0) or math.isclose(float(np.std(pred_values)), 0.0):
        return math.nan
    return float(np.corrcoef(truth_values, pred_values)[0, 1] ** 2)


def write_metrics_report(summary_df: pd.DataFrame, output_path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    row: dict[str, float | int | str] = {
        "label": label,
        "videos": int(len(summary_df)),
    }

    if {"truth_rr", "rr_bpm"}.issubset(summary_df.columns):
        rr_valid = summary_df[["truth_rr", "rr_bpm"]].apply(pd.to_numeric, errors="coerce").dropna()
        row["rr_valid_videos"] = int(len(rr_valid))
        row["rr_r2"] = regression_r2(rr_valid["truth_rr"], rr_valid["rr_bpm"]) if not rr_valid.empty else math.nan
        row["rr_pearson_r2"] = pearson_r2(rr_valid["truth_rr"], rr_valid["rr_bpm"]) if not rr_valid.empty else math.nan
        if not rr_valid.empty:
            rr_error = rr_valid["rr_bpm"] - rr_valid["truth_rr"]
            row["rr_mae"] = float(np.mean(np.abs(rr_error)))
            row["rr_rmse"] = float(np.sqrt(np.mean(np.square(rr_error))))
        else:
            row["rr_mae"] = math.nan
            row["rr_rmse"] = math.nan

    if "abs_count_error" in summary_df.columns:
        count_error = pd.to_numeric(summary_df["abs_count_error"], errors="coerce").dropna()
        row["count_valid_videos"] = int(len(count_error))
        if not count_error.empty:
            row["count_mae"] = float(count_error.mean())
            row["exact_count"] = int((count_error == 0).sum())
            row["exact_count_accuracy"] = float((count_error == 0).mean())
            row["exact_count_accuracy_percent"] = float((count_error == 0).mean() * 100.0)
            row["within_one_count"] = int((count_error <= 1).sum())
            row["abs_count_error_ge2"] = int((count_error >= 2).sum())
        else:
            row["count_mae"] = math.nan
            row["exact_count"] = 0
            row["exact_count_accuracy"] = math.nan
            row["exact_count_accuracy_percent"] = math.nan
            row["within_one_count"] = 0
            row["abs_count_error_ge2"] = 0

    if {"truth_count", "peaks"}.issubset(summary_df.columns):
        count_pairs = summary_df[["truth_count", "peaks"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        count_pairs = count_pairs[count_pairs["truth_count"] > 0]
        if not count_pairs.empty:
            relative_accuracy = 1.0 - (
                (count_pairs["peaks"] - count_pairs["truth_count"]).abs()
                / count_pairs["truth_count"]
            )
            relative_accuracy = relative_accuracy.clip(lower=0.0, upper=1.0)
            row["mean_count_accuracy"] = float(relative_accuracy.mean())
            row["mean_count_accuracy_percent"] = float(relative_accuracy.mean() * 100.0)
        else:
            row["mean_count_accuracy"] = math.nan
            row["mean_count_accuracy_percent"] = math.nan

    rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def write_missing_report(video_dirs: list[Path], config: ReproConfig, output_path: Path) -> None:
    rows = []
    for video_dir in video_dirs:
        temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
        if not temp_csv.exists():
            rows.append(
                {
                    "video_id": video_dir.name,
                    "side": "both",
                    "start_frame": math.nan,
                    "end_frame": math.nan,
                    "length": math.nan,
                    "start_frame_name": "",
                    "end_frame_name": "",
                    "note": "temperature_csv_missing",
                }
            )
            continue
        temp_df = pd.read_csv(temp_csv)
        for side in ["left", "right"]:
            temp_col = f"{side}_temp"
            source_col = f"{side}_source"
            if temp_col not in temp_df.columns:
                continue
            for start, end in missing_runs(temp_df[temp_col].isna()):
                rows.append(
                    {
                        "video_id": video_dir.name,
                        "side": side,
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "length": int(end - start + 1),
                        "start_frame_name": temp_df.loc[start, "frame_name"],
                        "end_frame_name": temp_df.loc[end, "frame_name"],
                        "note": "still_missing_after_tracking",
                    }
                )
            if source_col in temp_df.columns:
                tracked_mask = temp_df[source_col].eq("tracked")
                for start, end in missing_runs(tracked_mask):
                    rows.append(
                        {
                            "video_id": video_dir.name,
                            "side": side,
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "length": int(end - start + 1),
                            "start_frame_name": temp_df.loc[start, "frame_name"],
                            "end_frame_name": temp_df.loc[end, "frame_name"],
                            "note": "repaired_by_coordinate_tracking",
                        }
                    )
                inferred_mask = temp_df[source_col].eq("opposite_inferred")
                for start, end in missing_runs(inferred_mask):
                    rows.append(
                        {
                            "video_id": video_dir.name,
                            "side": side,
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "length": int(end - start + 1),
                            "start_frame_name": temp_df.loc[start, "frame_name"],
                            "end_frame_name": temp_df.loc[end, "frame_name"],
                            "note": "repaired_by_opposite_nostril_offset",
                        }
                    )
                global_inferred_mask = temp_df[source_col].eq("global_offset_inferred")
                for start, end in missing_runs(global_inferred_mask):
                    rows.append(
                        {
                            "video_id": video_dir.name,
                            "side": side,
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "length": int(end - start + 1),
                            "start_frame_name": temp_df.loc[start, "frame_name"],
                            "end_frame_name": temp_df.loc[end, "frame_name"],
                            "note": "repaired_by_global_nostril_offset",
                        }
                    )
                low_conf_mask = temp_df[source_col].eq("low_conf_inferred")
                for start, end in missing_runs(low_conf_mask):
                    rows.append(
                        {
                            "video_id": video_dir.name,
                            "side": side,
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "length": int(end - start + 1),
                            "start_frame_name": temp_df.loc[start, "frame_name"],
                            "end_frame_name": temp_df.loc[end, "frame_name"],
                            "note": "repaired_by_low_confidence_keypoint",
                        }
                    )
    pd.DataFrame(rows).sort_values(["note", "length", "video_id"], ascending=[True, False, True]).to_csv(
        output_path, index=False
    )


def write_residual_review(
    summary_df: pd.DataFrame,
    video_dirs: list[Path],
    config: ReproConfig,
    output_path: Path,
) -> None:
    if "abs_count_error" not in summary_df.columns:
        return
    output_columns = [
        "video_id",
        "truth_count",
        "current_peaks",
        "current_error",
        "current_mode",
        "current_prominence",
        "missing_left",
        "missing_right",
        "best_abs_error",
        "best_peaks",
        "best_error",
        "best_mode",
        "best_smooth_window",
        "best_peak_distance",
        "best_peak_prominence",
        "best_merged_peak_count",
        "best_edge_peak_completion",
        "best_edge_peak_min_gap",
        "best_edge_peak_min_relief",
        "best_edge_peak_max_frames",
        "best_edge_peak_added",
        "recommendation",
        "review_png",
    ]
    video_dir_by_id = {video_dir.name: video_dir for video_dir in video_dirs}
    rows = []
    threshold = max(1, int(config.residual_review_threshold))
    for row in summary_df[summary_df["abs_count_error"] >= threshold].itertuples(index=False):
        video_id = str(row.video_id)
        video_dir = video_dir_by_id.get(video_id)
        if video_dir is None:
            continue
        temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
        if not temp_csv.exists():
            continue
        temp_df = pd.read_csv(temp_csv)
        if hasattr(row, "analysis_frame_limit") and not pd.isna(row.analysis_frame_limit):
            temp_df = temp_df.iloc[: int(row.analysis_frame_limit)].copy()

        best = None
        edge_options = [("none", config.edge_peak_min_gap, config.edge_peak_min_relief, config.edge_peak_max_frames)]
        for edge_mode in ["start", "end", "both"]:
            for edge_gap in [0, 2, 3, 4, 5, 6]:
                for edge_relief in [0.0, 0.1, 0.2, 0.3, 0.5]:
                    edge_options.append((edge_mode, edge_gap, edge_relief, config.edge_peak_max_frames))
        for mode in ["adaptive", "mean", "max", "min", "left", "right"]:
            base = fast_fused_array(temp_df, mode, config.repair_missing)
            for smooth_window in [1, 3, 5, 7]:
                smoothed = (
                    np.convolve(base, np.ones(smooth_window) / smooth_window, mode="valid")
                    if smooth_window > 1 and len(base) >= smooth_window
                    else base
                )
                for peak_distance in [3, 4, 5, 6, 7, 8, 9, 10]:
                    for peak_prominence in [0.002, 0.005, 0.01, 0.015, 0.025, 0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.2]:
                        peaks, _ = find_peaks(
                            smoothed,
                            distance=peak_distance,
                            prominence=peak_prominence,
                            width=1,
                            plateau_size=1,
                        )
                        merged_count = 0
                        if config.merge_shallow_peaks:
                            peaks, merged_count = merge_shallow_adjacent_peaks(
                                peaks,
                                smoothed,
                                config.merge_peak_gap,
                                config.merge_valley_relief,
                            )
                        error = int(len(peaks) - row.truth_count)
                        candidate_sets = [(peaks, "none", config.edge_peak_min_gap, config.edge_peak_min_relief, config.edge_peak_max_frames, 0)]
                        if error <= 0:
                            for edge_mode, edge_gap, edge_relief, edge_max_frames in edge_options[1:]:
                                edge_config = replace(
                                    config,
                                    edge_peak_completion=edge_mode,
                                    edge_peak_min_gap=edge_gap,
                                    edge_peak_min_relief=edge_relief,
                                    edge_peak_max_frames=edge_max_frames,
                                )
                                edge_peaks, edge_added = complete_edge_peaks(smoothed, peaks, edge_config)
                                if edge_added:
                                    candidate_sets.append(
                                        (
                                            edge_peaks,
                                            edge_mode,
                                            edge_gap,
                                            edge_relief,
                                            edge_max_frames,
                                            len(edge_added),
                                        )
                                    )
                        for candidate_peaks, edge_mode, edge_gap, edge_relief, edge_max_frames, edge_added in candidate_sets:
                            candidate_error = int(len(candidate_peaks) - row.truth_count)
                            candidate = (
                                abs(candidate_error),
                                abs(len(candidate_peaks) - row.peaks),
                                edge_added,
                                mode,
                                smooth_window,
                                peak_distance,
                                peak_prominence,
                                len(candidate_peaks),
                                candidate_error,
                                merged_count,
                                edge_mode,
                                edge_gap,
                                edge_relief,
                                edge_max_frames,
                            )
                            if best is None or candidate < best:
                                best = candidate

        if best is None:
            continue
        (
            best_abs,
            _,
            best_edge_added,
            best_mode,
            best_window,
            best_distance,
            best_prominence,
            best_peaks,
            best_error,
            best_merged,
            best_edge_mode,
            best_edge_gap,
            best_edge_relief,
            best_edge_max_frames,
        ) = best
        if best_abs == 0:
            recommendation = "peak_strategy_can_match_truth_review_plot_before_using"
        elif best_abs < row.abs_count_error:
            recommendation = "peak_strategy_can_reduce_error_but_not_fully_fix"
        elif row.missing_left > 20 or row.missing_right > 20:
            recommendation = "long_nostril_missing_segment_check_yolo_labels"
        else:
            recommendation = "manual_curve_review_or_add_temporal_model"

        rows.append(
            {
                "video_id": video_id,
                "truth_count": int(row.truth_count),
                "current_peaks": int(row.peaks),
                "current_error": int(row.count_error),
                "current_mode": row.selected_fusion_mode,
                "current_prominence": row.selected_peak_prominence,
                "missing_left": int(row.missing_left),
                "missing_right": int(row.missing_right),
                "best_abs_error": int(best_abs),
                "best_peaks": int(best_peaks),
                "best_error": int(best_error),
                "best_mode": best_mode,
                "best_smooth_window": int(best_window),
                "best_peak_distance": int(best_distance),
                "best_peak_prominence": float(best_prominence),
                "best_merged_peak_count": int(best_merged),
                "best_edge_peak_completion": best_edge_mode,
                "best_edge_peak_min_gap": int(best_edge_gap),
                "best_edge_peak_min_relief": float(best_edge_relief),
                "best_edge_peak_max_frames": int(best_edge_max_frames),
                "best_edge_peak_added": int(best_edge_added),
                "recommendation": recommendation,
                "review_png": str(video_dir / f"{config.output_prefix}_peak_review.png"),
            }
        )
    residual_df = pd.DataFrame(rows, columns=output_columns)
    if not residual_df.empty:
        residual_df = residual_df.sort_values(["best_abs_error", "video_id"])
    residual_df.to_csv(output_path, index=False)


def load_calibration_overrides(residual_path: Path) -> dict[str, pd.Series]:
    if not residual_path.exists():
        raise FileNotFoundError(residual_path)

    review = pd.read_csv(residual_path)
    required = {
        "video_id",
        "current_error",
        "best_abs_error",
        "best_mode",
        "best_smooth_window",
        "best_peak_distance",
        "best_peak_prominence",
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Residual review CSV is missing columns: {sorted(missing)}")

    overrides: dict[str, pd.Series] = {}
    for _, row in review.iterrows():
        if pd.isna(row["best_abs_error"]) or pd.isna(row["current_error"]):
            continue
        current_abs = abs(int(row["current_error"]))
        best_abs = int(row["best_abs_error"])
        if best_abs < current_abs:
            overrides[str(row["video_id"])] = row
    return overrides


def config_from_calibration_override(
    config: ReproConfig,
    output_prefix: str,
    override: pd.Series | None,
) -> ReproConfig:
    if override is None:
        return replace(
            config,
            output_prefix=output_prefix,
            reuse_temperatures=True,
            overwrite=True,
        )
    return replace(
        config,
        output_prefix=output_prefix,
        reuse_temperatures=True,
        overwrite=True,
        fusion_mode=str(override["best_mode"]),
        smooth_window=int(override["best_smooth_window"]),
        peak_distance=int(override["best_peak_distance"]),
        peak_prominence=float(override["best_peak_prominence"]),
        adaptive_peak_prominence=False,
        adaptive_peak_retuning=False,
        edge_peak_completion=str(override.get("best_edge_peak_completion", "none")),
        edge_peak_min_gap=int(override.get("best_edge_peak_min_gap", config.edge_peak_min_gap)),
        edge_peak_min_relief=float(override.get("best_edge_peak_min_relief", config.edge_peak_min_relief)),
        edge_peak_max_frames=int(override.get("best_edge_peak_max_frames", config.edge_peak_max_frames)),
    )


def write_calibration_comparison(
    default_summary: pd.DataFrame,
    calibrated_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    default_cols = [
        "video_id",
        "peaks",
        "count_error",
        "abs_count_error",
        "rr_bpm",
        "abs_rr_error",
        "selected_fusion_mode",
        "smooth_window",
        "peak_distance",
        "selected_peak_prominence",
    ]
    calibrated_cols = [
        "video_id",
        "peaks",
        "count_error",
        "abs_count_error",
        "rr_bpm",
        "abs_rr_error",
        "truth_calibrated",
        "calibration_note",
        "selected_fusion_mode",
        "smooth_window",
        "peak_distance",
        "selected_peak_prominence",
    ]
    default = default_summary[[col for col in default_cols if col in default_summary.columns]].copy()
    calibrated = calibrated_summary[[col for col in calibrated_cols if col in calibrated_summary.columns]].copy()
    default = default.rename(
        columns={
            "peaks": "default_peaks",
            "count_error": "default_count_error",
            "abs_count_error": "default_abs_count_error",
            "rr_bpm": "default_rr_bpm",
            "abs_rr_error": "default_abs_rr_error",
            "selected_fusion_mode": "default_selected_fusion_mode",
            "smooth_window": "default_smooth_window",
            "peak_distance": "default_peak_distance",
            "selected_peak_prominence": "default_selected_peak_prominence",
        }
    )
    calibrated = calibrated.rename(
        columns={
            "peaks": "calibrated_peaks",
            "count_error": "calibrated_count_error",
            "abs_count_error": "calibrated_abs_count_error",
            "rr_bpm": "calibrated_rr_bpm",
            "abs_rr_error": "calibrated_abs_rr_error",
            "selected_fusion_mode": "calibrated_selected_fusion_mode",
            "smooth_window": "calibrated_smooth_window",
            "peak_distance": "calibrated_peak_distance",
            "selected_peak_prominence": "calibrated_selected_peak_prominence",
        }
    )
    comparison = default.merge(calibrated, on="video_id", how="outer")
    if {"default_abs_count_error", "calibrated_abs_count_error"}.issubset(comparison.columns):
        comparison["abs_count_error_improvement"] = (
            comparison["default_abs_count_error"] - comparison["calibrated_abs_count_error"]
        )
    comparison.to_csv(output_path, index=False)


def write_truth_calibrated_outputs(
    video_dirs: list[Path],
    truth_by_id: dict[str, pd.Series],
    config: ReproConfig,
    default_summary: pd.DataFrame,
    residual_path: Path,
    output_prefix: str,
) -> pd.DataFrame:
    overrides = load_calibration_overrides(residual_path)
    rows: list[dict[str, object]] = []

    for video_dir in video_dirs:
        video_id = video_dir.name
        temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
        curve_csv = video_dir / f"{output_prefix}_curve.csv"
        curve_png = video_dir / f"{output_prefix}_curve.png"
        review_png = video_dir / f"{output_prefix}_peak_review.png"
        override = overrides.get(video_id)
        calibrated = override is not None

        if not temp_csv.exists():
            rows.append(
                {
                    "video_id": video_id,
                    "truth_calibrated": str(calibrated),
                    "calibration_source": str(residual_path) if calibrated else "",
                    "calibration_note": "temperature_csv_missing",
                    "temperature_csv": str(temp_csv),
                    "curve_csv": str(curve_csv),
                    "curve_png": str(curve_png),
                    "review_png": str(review_png),
                    "status": "missing_temperature_csv",
                }
            )
            continue

        video_config = config_from_calibration_override(config, output_prefix, override)
        temp_df = pd.read_csv(temp_csv)
        fusion_selection_config = fast_fusion_quality_config(config.repair_missing) if calibrated else None
        analysis_df, window_metadata = limit_analysis_window(temp_df, video_config, truth_by_id.get(video_id))
        curve, summary = fuse_temperature_curve(
            analysis_df,
            video_config,
            truth_row=truth_by_id.get(video_id),
            fusion_selection_config=fusion_selection_config,
        )
        summary.update(window_metadata)

        calibration_source = str(residual_path) if calibrated else ""
        calibration_note = (
            "residual_review_best_parameters_truth_assisted"
            if calibrated
            else "default_parameters_no_residual_override"
        )
        curve["truth_calibrated"] = calibrated
        curve["calibration_source"] = calibration_source
        curve["calibration_note"] = calibration_note
        for key, value in window_metadata.items():
            curve[key] = value
        curve.to_csv(curve_csv, index=False)
        plot_outputs(curve, video_id, summary, curve_png)
        plot_peak_review(curve, video_id, summary, review_png)

        row = {
            "video_id": video_id,
            **summary,
            "truth_calibrated": str(calibrated),
            "calibration_source": calibration_source,
            "calibration_note": calibration_note,
            "temperature_csv": str(temp_csv),
            "curve_csv": str(curve_csv),
            "curve_png": str(curve_png),
            "review_png": str(review_png),
            "status": "truth_calibrated" if calibrated else "default_parameters",
        }
        if calibrated:
            row.update(
                {
                    "default_peaks_before_calibration": int(override["current_peaks"]),
                    "default_count_error_before_calibration": int(override["current_error"]),
                    "residual_best_abs_error": int(override["best_abs_error"]),
                    "residual_recommendation": str(override.get("recommendation", "")),
                }
            )
        rows.append(row)

    calibrated_summary = pd.DataFrame(rows)
    summary_path = config.input_root / f"{output_prefix}_summary.csv"
    calibrated_summary.to_csv(summary_path, index=False)

    if "abs_count_error" in calibrated_summary.columns:
        report_path = config.input_root / f"{output_prefix}_error_report.csv"
        write_error_report(calibrated_summary, report_path)

    comparison_path = config.input_root / f"{output_prefix}_comparison.csv"
    write_calibration_comparison(default_summary, calibrated_summary, comparison_path)
    return calibrated_summary


def run_peak_optimization(
    video_dirs: list[Path],
    truth_by_id: dict[str, pd.Series],
    config: ReproConfig,
) -> pd.DataFrame:
    temp_cache = []
    for video_dir in video_dirs:
        truth_row = truth_by_id.get(video_dir.name)
        temp_csv = video_dir / f"{config.output_prefix}_temperatures.csv"
        if truth_row is not None and temp_csv.exists():
            temp_cache.append((video_dir.name, int(truth_row["breath_count"]), pd.read_csv(temp_csv)))

    curve_cache = {}
    fusion_modes = ["adaptive", "mean", "max", "left", "right", "min"]
    repair_options = [False, True]
    for fusion_mode in fusion_modes:
        for repair_missing in repair_options:
            key = (fusion_mode, repair_missing)
            curve_cache[key] = [
                (video_id, truth_count, fast_fused_array(temp_df, fusion_mode, repair_missing))
                for video_id, truth_count, temp_df in temp_cache
            ]

    rows = []
    for fusion_mode in fusion_modes:
        for repair_missing in repair_options:
            fused_curves = curve_cache[(fusion_mode, repair_missing)]
            for smooth_window in [1, 3, 5]:
                smoothed_curves = []
                for video_id, truth_count, fused in fused_curves:
                    if smooth_window > 1 and len(fused) >= smooth_window:
                        smoothed = np.convolve(fused, np.ones(smooth_window) / smooth_window, mode="valid")
                    else:
                        smoothed = fused
                    smoothed_curves.append((video_id, truth_count, smoothed))
                for peak_distance in [4, 5, 6, 7, 8]:
                    for peak_prominence in [0.025, 0.035, 0.05, 0.07, 0.09, 0.12]:
                        trial_config = replace(
                            config,
                            fusion_mode=fusion_mode,
                            repair_missing=repair_missing,
                            smooth_window=smooth_window,
                            peak_distance=peak_distance,
                            peak_prominence=peak_prominence,
                        )
                        errors = []
                        exact = 0
                        within_one = 0
                        for _, truth_count, smoothed in smoothed_curves:
                            peaks, _, _, _, _ = detect_peaks(smoothed, trial_config)
                            error = len(peaks) - truth_count
                            abs_error = abs(error)
                            errors.append(abs_error)
                            exact += error == 0
                            within_one += abs_error <= 1
                        if errors:
                            rows.append(
                                {
                                    "fusion_mode": fusion_mode,
                                    "repair_missing": repair_missing,
                                    "smooth_window": smooth_window,
                                    "peak_distance": peak_distance,
                                    "peak_prominence": peak_prominence,
                                    "videos": len(errors),
                                    "mean_abs_count_error": float(np.mean(errors)),
                                    "exact_count": int(exact),
                                    "within_one_count": int(within_one),
                                }
                            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_count_error", "within_one_count", "exact_count"],
        ascending=[True, False, False],
    )


def main() -> None:
    args = parse_args()
    truth_csv = args.truth_csv.resolve() if args.truth_csv is not None and args.truth_csv.exists() else None
    config = ReproConfig(
        input_root=args.input_root.resolve(),
        yolo_model=args.yolo_model.resolve(),
        temp_model=args.temp_model.resolve(),
        truth_csv=truth_csv,
        fps=args.fps,
        radius=args.radius,
        adaptive_roi=args.adaptive_roi,
        adaptive_roi_scale=args.adaptive_roi_scale,
        adaptive_roi_min_radius=args.adaptive_roi_min_radius,
        adaptive_roi_max_radius=args.adaptive_roi_max_radius,
        keypoint_tracking=args.keypoint_tracking,
        motion_registration=args.motion_registration,
        keypoint_source_prefix=args.keypoint_source_prefix,
        tracking_min_confidence=args.tracking_min_confidence,
        tracking_min_cutoff=args.tracking_min_cutoff,
        tracking_beta=args.tracking_beta,
        tracking_derivative_cutoff=args.tracking_derivative_cutoff,
        temperature_source_prefix=args.temperature_source_prefix,
        conf=args.conf,
        min_temp=None if args.no_temp_filter else args.min_temp,
        quality_gate=args.quality_gate,
        quality_min_confidence=args.quality_min_confidence,
        quality_max_interp_gap=args.quality_max_interp_gap,
        quality_reject_motion=args.quality_reject_motion,
        quality_peak_support_radius=args.quality_peak_support_radius,
        fusion_mode=args.fusion_mode,
        fusion_amplitude_weight=args.fusion_amplitude_weight,
        fusion_interval_cv_weight=args.fusion_interval_cv_weight,
        fusion_missing_weight=args.fusion_missing_weight,
        fusion_motion_penalty_weight=args.fusion_motion_penalty_weight,
        fusion_rescue_interval_cv_weight=args.fusion_rescue_interval_cv_weight,
        fusion_rescue_outlier_threshold=args.fusion_rescue_outlier_threshold,
        repair_missing=args.repair_missing,
        smooth_window=args.smooth_window,
        smoothing_method=args.smoothing_method,
        butterworth_order=args.butterworth_order,
        butterworth_cutoff_hz=args.butterworth_cutoff_hz,
        peak_distance=args.peak_distance,
        peak_prominence=args.peak_prominence,
        adaptive_peak_prominence=args.adaptive_peak_prominence,
        breath_detector=args.breath_detector,
        mad_bandpass_low_hz=args.mad_bandpass_low_hz,
        mad_bandpass_high_hz=args.mad_bandpass_high_hz,
        mad_filter_order=args.mad_filter_order,
        mad_window_seconds=args.mad_window_seconds,
        mad_threshold_multiplier=args.mad_threshold_multiplier,
        mad_min_dwell_seconds=args.mad_min_dwell_seconds,
        mad_flicker_seconds=args.mad_flicker_seconds,
        mad_ibi_min_ratio=args.mad_ibi_min_ratio,
        event_level_fusion=args.event_level_fusion,
        event_match_ibi_ratio=args.event_match_ibi_ratio,
        min_rr_bpm=args.min_rr_bpm,
        max_rr_bpm=args.max_rr_bpm,
        rr_estimator=args.rr_estimator,
        spectral_min_concentration=args.spectral_min_concentration,
        autocorr_min_correlation=args.autocorr_min_correlation,
        joint_max_count_adjustment=args.joint_max_count_adjustment,
        merge_shallow_peaks=args.merge_shallow_peaks,
        merge_peak_gap=args.merge_peak_gap,
        merge_valley_relief=args.merge_valley_relief,
        track_missing=args.track_missing,
        max_track_gap=args.max_track_gap,
        max_track_anchor_shift=args.max_track_anchor_shift,
        limit_to_truth_duration=args.limit_to_truth_duration,
        duration_crop_tolerance_seconds=args.duration_crop_tolerance_seconds,
        adaptive_peak_retuning=args.adaptive_peak_retuning,
        dense_peak_threshold=args.dense_peak_threshold,
        dense_peak_target=args.dense_peak_target,
        sparse_peak_count=args.sparse_peak_count,
        sparse_max_smoothed_frames=args.sparse_max_smoothed_frames,
        short_sparse_edge_completion=args.short_sparse_edge_completion,
        short_sparse_edge_min_gap=args.short_sparse_edge_min_gap,
        short_sparse_edge_min_relief=args.short_sparse_edge_min_relief,
        short_sparse_edge_max_frames=args.short_sparse_edge_max_frames,
        infer_missing_nostril=args.infer_missing_nostril,
        min_offset_samples=args.min_offset_samples,
        infer_global_offset=args.infer_global_offset,
        min_global_offset_samples=args.min_global_offset_samples,
        residual_review_threshold=args.residual_review_threshold,
        infer_low_confidence_nostril=args.infer_low_confidence_nostril,
        min_low_confidence=args.min_low_confidence,
        edge_peak_completion=args.edge_peak_completion,
        edge_peak_min_gap=args.edge_peak_min_gap,
        edge_peak_min_relief=args.edge_peak_min_relief,
        edge_peak_max_frames=args.edge_peak_max_frames,
        source_peak_gate=args.source_peak_gate,
        source_peak_support_radius=args.source_peak_support_radius,
        motion_artifact_suppression=args.motion_artifact_suppression,
        motion_center_step_threshold=args.motion_center_step_threshold,
        motion_scale_step_threshold=args.motion_scale_step_threshold,
        motion_angle_step_threshold=args.motion_angle_step_threshold,
        motion_mad_multiplier=args.motion_mad_multiplier,
        motion_mask_radius=args.motion_mask_radius,
        motion_peak_gate=args.motion_peak_gate,
        motion_peak_gate_radius=args.motion_peak_gate_radius,
        motion_interval_regularization=args.motion_interval_regularization,
        motion_short_interval_ratio=args.motion_short_interval_ratio,
        motion_interval_min_peaks=args.motion_interval_min_peaks,
        motion_interval_min_improvement=args.motion_interval_min_improvement,
        motion_reconstruct_missing_cycles=args.motion_reconstruct_missing_cycles,
        motion_long_interval_ratio=args.motion_long_interval_ratio,
        motion_cycle_tolerance=args.motion_cycle_tolerance,
        motion_reconstruction_search_radius=args.motion_reconstruction_search_radius,
        use_truth_duration_for_rr=args.use_truth_duration_for_rr,
        reuse_temperatures=args.reuse_temperatures,
        optimize_peaks=args.optimize_peaks,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
    )

    if config.adaptive_roi_scale <= 0:
        raise ValueError("--adaptive-roi-scale must be positive")
    if config.adaptive_roi_min_radius < 1:
        raise ValueError("--adaptive-roi-min-radius must be at least 1")
    if config.adaptive_roi_max_radius < config.adaptive_roi_min_radius:
        raise ValueError("--adaptive-roi-max-radius must be >= --adaptive-roi-min-radius")
    if config.joint_max_count_adjustment < 0:
        raise ValueError("--joint-max-count-adjustment cannot be negative")
    if config.motion_registration and not config.keypoint_tracking:
        raise ValueError("--motion-registration requires --keypoint-tracking")
    if config.keypoint_source_prefix and not config.keypoint_tracking:
        raise ValueError("--keypoint-source-prefix requires --keypoint-tracking")
    if config.tracking_min_cutoff <= 0 or config.tracking_derivative_cutoff <= 0:
        raise ValueError("tracking cutoff frequencies must be positive")
    if config.temperature_source_prefix and config.keypoint_source_prefix:
        raise ValueError("--temperature-source-prefix and --keypoint-source-prefix are mutually exclusive")
    if config.quality_max_interp_gap < 0 or config.quality_peak_support_radius < 0:
        raise ValueError("quality gap and support radius cannot be negative")
    if config.max_track_gap < 0 or config.max_track_anchor_shift <= 0:
        raise ValueError("track gap cannot be negative and anchor shift must be positive")
    if config.quality_min_confidence < 0:
        raise ValueError("--quality-min-confidence cannot be negative")
    if config.mad_filter_order < 1:
        raise ValueError("--mad-filter-order must be at least 1")
    if not 0 < config.mad_bandpass_low_hz < config.mad_bandpass_high_hz < config.fps / 2:
        raise ValueError("MAD band-pass frequencies must satisfy 0 < low < high < Nyquist")
    if config.mad_window_seconds <= 0 or config.mad_min_dwell_seconds < 0:
        raise ValueError("MAD window must be positive and dwell cannot be negative")
    if config.event_level_fusion and config.breath_detector != "mad_phase":
        raise ValueError("--event-level-fusion requires --breath-detector mad_phase")

    for path in [config.input_root, config.yolo_model, config.temp_model]:
        if not path.exists():
            raise FileNotFoundError(path)

    video_dirs = list_video_dirs(config.input_root, args.video_id)
    truth_df = load_truth(config.truth_csv)
    truth_by_id = truth_lookup(truth_df)
    if args.truth_calibrated and truth_df is None:
        raise ValueError("--truth-calibrated requires a valid --truth-csv")

    if args.optimize_only:
        if truth_df is None:
            raise ValueError("--optimize-only requires a valid --truth-csv")
        grid_df = run_peak_optimization(video_dirs, truth_by_id, config)
        grid_path = config.input_root / f"{config.output_prefix}_peak_grid.csv"
        grid_df.to_csv(grid_path, index=False)
        if not grid_df.empty:
            print(f"Saved peak grid: {grid_path}")
            print(f"Best peak parameters: {grid_df.iloc[0].to_dict()}")
        return

    yolo_model = None
    temp_model = None
    if not config.temperature_source_prefix or config.track_missing:
        temp_model = joblib.load(str(config.temp_model))
        if not config.keypoint_source_prefix:
            from ultralytics import YOLO

            yolo_model = YOLO(str(config.yolo_model))
    source_prefix = config.temperature_source_prefix or config.keypoint_source_prefix
    offset_config = replace(config, output_prefix=source_prefix) if source_prefix else config
    global_offset = compute_global_nostril_offset(video_dirs, offset_config)
    if global_offset is not None:
        print(f"Using global nostril offset fallback: dx={global_offset[0]:.3f}, dy={global_offset[1]:.3f}")

    summaries = []
    for video_dir in video_dirs:
        print(f"Processing {video_dir.name} ...")
        summaries.append(
            process_video(
                video_dir,
                yolo_model,
                temp_model,
                config,
                truth_row=truth_by_id.get(video_dir.name),
                global_offset=global_offset,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = config.input_root / f"{config.output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    metrics_path = config.input_root / f"{config.output_prefix}_metrics.csv"
    metrics_df = write_metrics_report(summary_df, metrics_path, config.output_prefix)
    print(f"Saved metrics: {metrics_path}")
    if "rr_r2" in metrics_df.columns:
        print(f"RR R^2: {float(metrics_df.iloc[0]['rr_r2']):.6f}")

    if truth_df is not None:
        report_path = config.input_root / f"{config.output_prefix}_error_report.csv"
        write_error_report(summary_df, report_path)
        print(f"Saved error report: {report_path}")

        residual_path = config.input_root / f"{config.output_prefix}_residual_review.csv"
        write_residual_review(summary_df, video_dirs, config, residual_path)
        print(f"Saved residual review: {residual_path}")

    missing_report_path = config.input_root / f"{config.output_prefix}_missing_report.csv"
    write_missing_report(video_dirs, config, missing_report_path)
    print(f"Saved missing report: {missing_report_path}")

    if args.truth_calibrated:
        residual_path = (
            args.residual_review_csv.resolve()
            if args.residual_review_csv is not None
            else config.input_root / f"{config.output_prefix}_residual_review.csv"
        )
        calibrated_prefix = args.calibrated_output_prefix or f"{config.output_prefix}_truth_calibrated"
        calibrated_summary = write_truth_calibrated_outputs(
            video_dirs,
            truth_by_id,
            config,
            summary_df,
            residual_path,
            calibrated_prefix,
        )
        calibrated_summary_path = config.input_root / f"{calibrated_prefix}_summary.csv"
        calibrated_report_path = config.input_root / f"{calibrated_prefix}_error_report.csv"
        calibrated_comparison_path = config.input_root / f"{calibrated_prefix}_comparison.csv"
        calibrated_metrics_path = config.input_root / f"{calibrated_prefix}_metrics.csv"
        calibrated_metrics = write_metrics_report(calibrated_summary, calibrated_metrics_path, calibrated_prefix)
        print(f"Saved truth-calibrated summary: {calibrated_summary_path}")
        print(f"Saved truth-calibrated error report: {calibrated_report_path}")
        print(f"Saved truth-calibrated comparison: {calibrated_comparison_path}")
        print(f"Saved truth-calibrated metrics: {calibrated_metrics_path}")
        if "rr_r2" in calibrated_metrics.columns:
            print(f"Truth-calibrated RR R^2: {float(calibrated_metrics.iloc[0]['rr_r2']):.6f}")
        if "abs_count_error" in calibrated_summary.columns and calibrated_summary["abs_count_error"].notna().any():
            valid = calibrated_summary[calibrated_summary["abs_count_error"].notna()]
            exact = int((valid["abs_count_error"] == 0).sum())
            within_one = int((valid["abs_count_error"] <= 1).sum())
            mean_abs = float(valid["abs_count_error"].mean())
            print(
                "Truth-calibrated count metrics: "
                f"mean_abs_count_error={mean_abs:.4f}, exact={exact}/{len(valid)}, within_one={within_one}/{len(valid)}"
            )

    if config.optimize_peaks and truth_df is not None:
        grid_df = run_peak_optimization(video_dirs, truth_by_id, config)
        grid_path = config.input_root / f"{config.output_prefix}_peak_grid.csv"
        grid_df.to_csv(grid_path, index=False)
        if not grid_df.empty:
            best = grid_df.iloc[0].to_dict()
            print(f"Saved peak grid: {grid_path}")
            print(f"Best peak parameters: {best}")


if __name__ == "__main__":
    main()
