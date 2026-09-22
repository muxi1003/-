"""Opt-in timestamp sampling candidate; no interpolation across long frame gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd


POLICY = {
    "version": "timestamp_candidate_20260912_v2",
    "target_fps": 8.7,
    "long_gap_seconds": 0.5,
    "sampling": "nearest_existing_frame_on_half_open_uniform_grid; ties_choose_earlier",
    "long_gap_action": "abstain_entire_window; never_compress_time_or_fill_breaths",
    "short_interval_action": "existing_frame_may_be_reused; no_new_temperature_interpolation",
    "boundary_support_rule": "min(long_gap/2,1.5*median_observed_frame_interval)",
    "uses_reference_counts": False,
    "adoption": "candidate_requires_internal_validation",
}


def sample_map(timestamps, duration, policy=None):
    policy = dict(POLICY if policy is None else policy)
    t = np.asarray(timestamps, dtype=float)
    d, fps = float(duration), float(policy["target_fps"])
    gap = float(policy["long_gap_seconds"])
    if t.ndim != 1 or not np.isfinite(d) or d <= 0 or fps <= 0 or gap <= 0:
        raise ValueError("Invalid sampling inputs")
    base = {"prediction_status": "abstain", "duration_seconds": d, "predicted_count": None}
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return None, {**base, "reason": "invalid_timestamp_sequence"}
    if abs(t[0]) > 1e-6:
        return None, {**base, "reason": "window_origin_not_verified_zero"}
    relevant = (t[:-1] < d) & (t[1:] > 0)
    gaps = np.flatnonzero(relevant & (np.diff(t) > gap))
    if len(gaps):
        return None, {**base, "reason": "long_timestamp_gap", "long_gap_count": len(gaps)}
    inside = np.flatnonzero((t >= 0) & (t < d))
    terminal_support = min(gap / 2, 1.5 * float(np.median(np.diff(t))))
    if len(inside) < 2 or d - t[inside[-1]] > terminal_support + 1e-6:
        return None, {**base, "reason": "unsupported_window_end"}
    source = t[inside]
    target = np.arange(int(np.ceil(d * fps)), dtype=float) / fps
    target = target[target < d]
    right = np.clip(np.searchsorted(source, target, side="left"), 0, len(source)-1)
    left = np.maximum(0, right-1)
    nearest = np.where(abs(source[left]-target) <= abs(source[right]-target), left, right)
    error = abs(source[nearest]-target)
    if np.any(error > gap / 2 + 1e-6):
        return None, {**base, "reason": "unsupported_sampling_grid"}
    result = pd.DataFrame({"target_index": np.arange(len(target)), "target_time_seconds": target,
                           "source_frame_index": inside[nearest], "source_time_seconds": source[nearest],
                           "nearest_time_error_seconds": error})
    result["reused_source_frame"] = result.source_frame_index.duplicated()
    return result, {"prediction_status": "ready_for_signal_candidate", "reason": "",
                    "duration_seconds": d, "source_frames": len(inside), "target_frames": len(result),
                    "reused_source_frames": int(result.reused_source_frame.sum()),
                    "max_nearest_time_error_seconds": float(error.max())}
