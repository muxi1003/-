"""Half-open supported spans; no concatenation across missing intervals."""
import numpy as np

POLICY = {
    "version": "supported_segments_20260914_v1",
    "minimum_core_seconds": 6.0,
    "minimum_duration_origin": "two_cycles_at_existing_min_RR_20_bpm; not_fitted_to_counts",
    "gap_boundary_margin_frames": 3,
    "minimum_peaks_for_segment_RR": 2,
    "side_selection": "retain_upstream_selected_fusion; no_new_side_search",
    "partial_RR": "60_times_sum_eligible_segment_peaks_over_sum_eligible_core_seconds",
    "full_window_count": "null_unless_the_entire_original_window_is_eligible",
    "support_scope": "inherited_detection_provenance_with_permitted_short_gaps_not_visibility_truth",
}


def segments(mask, duration, policy=None):
    policy = POLICY if policy is None else policy
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid segment timebase")
    if not len(mask):
        return []
    dt = float(duration)/len(mask)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        margin = policy["gap_boundary_margin_frames"]
        a = min(stop, start + (margin if start else 0))
        b = max(a, stop - (margin if stop < len(mask) else 0))
        seconds = (b-a)*dt
        result.append({"start_frame": int(start), "stop_frame_exclusive": int(stop),
                       "core_start_frame": int(a), "core_stop_frame_exclusive": int(b),
                       "core_start_seconds": a*dt, "core_stop_seconds": b*dt,
                       "core_seconds": seconds, "eligible_duration": seconds+1e-9 >= policy["minimum_core_seconds"]})
    return result
