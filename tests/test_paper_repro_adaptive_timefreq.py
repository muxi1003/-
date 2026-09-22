import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.paper_repro_rr import (
    adaptive_roi_radius,
    bilateral_registration_matrix,
    bridge_short_false_runs,
    estimate_time_frequency_consensus,
    fast_fusion_quality_config,
    fuse_bilateral_events,
    interpolate_short_interior_runs,
    mad_phase_events,
    one_euro_filter_series,
    parse_args,
    write_metrics_report,
)


def test_adaptive_roi_radius_scales_and_clamps():
    assert adaptive_roi_radius(100.0, 20, True, 0.14, 12, 32) == 14
    assert adaptive_roi_radius(200.0, 20, True, 0.14, 12, 32) == 28
    assert adaptive_roi_radius(20.0, 20, True, 0.14, 12, 32) == 12
    assert adaptive_roi_radius(400.0, 20, True, 0.14, 12, 32) == 32
    assert adaptive_roi_radius(float("nan"), 20, True, 0.14, 12, 32) == 20
    assert adaptive_roi_radius(200.0, 20, False, 0.14, 12, 32) == 20


def test_time_frequency_consensus_repairs_one_count_error():
    fps = 8.7
    duration = 12.0
    time = np.arange(round(fps * duration)) / fps
    signal = np.sin(2.0 * np.pi * time) + 0.05 * np.sin(2.0 * np.pi * 2.7 * time)

    result = estimate_time_frequency_consensus(
        signal,
        fps=fps,
        duration_seconds=duration,
        min_rr_bpm=40.0,
        max_rr_bpm=90.0,
        time_peak_count=11,
        enabled=True,
        spectral_min_concentration=0.20,
        autocorr_min_correlation=0.25,
        max_count_adjustment=1,
    )

    assert result["spectral_count"] == 12
    assert result["autocorr_count"] == 12
    assert result["joint_count"] == 12
    assert result["joint_adjusted"] is True


def test_time_frequency_consensus_is_disabled_by_default():
    result = estimate_time_frequency_consensus(
        np.ones(32),
        fps=8.7,
        duration_seconds=4.0,
        min_rr_bpm=40.0,
        max_rr_bpm=90.0,
        time_peak_count=4,
        enabled=False,
        spectral_min_concentration=0.20,
        autocorr_min_correlation=0.25,
        max_count_adjustment=1,
    )
    assert result["joint_count"] == 4
    assert result["joint_adjusted"] is False


def test_experimental_rr_features_are_opt_in():
    original_argv = sys.argv
    try:
        sys.argv = ["paper_repro_rr.py"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    assert args.radius == 20
    assert args.adaptive_roi is False
    assert args.keypoint_tracking is False
    assert args.motion_registration is False
    assert args.quality_gate is False
    assert args.breath_detector == "peak"
    assert args.event_level_fusion is False
    assert args.rr_estimator == "peak"


def test_one_euro_filter_smooths_jitter_and_carries_missing_state():
    values = np.asarray([0.0, 0.5, 10.0, np.nan, 10.0])
    valid = np.isfinite(values)
    filtered = one_euro_filter_series(
        values,
        valid,
        fps=8.7,
        min_cutoff=0.8,
        beta=0.02,
        derivative_cutoff=1.0,
    )
    assert 0.0 < filtered[2] < 10.0
    assert filtered[3] == filtered[2]
    assert filtered[4] > filtered[3]


def test_bilateral_registration_maps_raw_axis_to_tracked_axis():
    matrix = bilateral_registration_matrix(
        (10.0, 10.0),
        (30.0, 10.0),
        (20.0, 20.0),
        (20.0, 40.0),
    )
    assert matrix is not None
    source = np.asarray([[10.0, 10.0, 1.0], [30.0, 10.0, 1.0]])
    mapped = source @ matrix.T
    assert np.allclose(mapped, np.asarray([[20.0, 20.0], [20.0, 40.0]]), atol=1e-5)


def test_metrics_include_mean_breath_count_accuracy():
    summary = pd.DataFrame(
        {
            "truth_count": [10, 10],
            "peaks": [9, 10],
            "abs_count_error": [1, 0],
            "truth_rr": [60.0, 60.0],
            "rr_bpm": [54.0, 60.0],
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        metrics = write_metrics_report(summary, Path(directory) / "metrics.csv", "test")
    assert np.isclose(metrics.loc[0, "mean_count_accuracy"], 0.95)
    assert np.isclose(metrics.loc[0, "mean_count_accuracy_percent"], 95.0)
    assert np.isclose(metrics.loc[0, "exact_count_accuracy"], 0.5)


def test_quality_gate_bridges_only_short_interior_gaps():
    mask = pd.Series([False, True, False, True, False, False, True, False])
    bridged = bridge_short_false_runs(mask, max_gap=1)
    assert bridged.tolist() == [False, True, True, True, False, False, True, False]

    values = pd.Series([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, 6.0, np.nan])
    repaired = interpolate_short_interior_runs(values, max_gap=1)
    assert np.isnan(repaired.iloc[0])
    assert repaired.iloc[2] == 2.0
    assert repaired.iloc[4:6].isna().all()
    assert np.isnan(repaired.iloc[7])


def test_mad_phase_detector_counts_synthetic_breaths():
    config = replace(
        fast_fusion_quality_config(True),
        breath_detector="mad_phase",
        min_rr_bpm=30.0,
        max_rr_bpm=90.0,
    )
    duration = 12.0
    time = np.arange(round(duration * config.fps)) / config.fps
    signal = np.sin(2.0 * np.pi * (50.0 / 60.0) * time)
    result = mad_phase_events(signal, np.ones(len(signal), dtype=bool), config)
    assert len(result["events"]) == 10


def test_event_level_fusion_matches_and_rejects_unsupported_unilateral_events():
    config = replace(
        fast_fusion_quality_config(True),
        breath_detector="mad_phase",
        event_level_fusion=True,
        min_rr_bpm=30.0,
        max_rr_bpm=90.0,
    )
    left = {
        "events": np.asarray([10, 20, 30]),
        "properties": {"prominences": np.asarray([1.0, 1.0, 1.0])},
    }
    right = {
        "events": np.asarray([11, 21]),
        "properties": {"prominences": np.asarray([1.0, 1.0])},
    }
    left_valid = np.ones(40, dtype=bool)
    right_valid = np.ones(40, dtype=bool)
    fused = fuse_bilateral_events(left, right, left_valid, right_valid, config)
    assert fused["events"].tolist() == [10, 20]
    assert fused["bilateral_matches"] == 2
    assert fused["rejected_unilateral"] == [30]

    right_valid[27:34] = False
    fused_with_missing_side = fuse_bilateral_events(left, right, left_valid, right_valid, config)
    assert fused_with_missing_side["events"].tolist() == [10, 20, 30]
    assert fused_with_missing_side["left_only"] == 1
