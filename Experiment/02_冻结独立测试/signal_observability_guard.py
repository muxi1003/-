"""Opt-in complete-window protection; partial event counts are never full-window RR."""
import numpy as np
import pandas as pd

POLICY = {
    "version": "observability_guard_20260914_v1",
    "max_short_gap_frames": 3,
    "confidence_minimum": 0.5,
    "boundary_missing": "abstain",
    "mixed_fusion_support": "both_inputs_direct; conservative_not_anatomical_visibility_truth",
    "long_gap_action": "abstain_complete_window; keep_diagnostic_events_only",
    "event_context_frames": 3,
    "parameter_origin": "existing_frozen_max_track_gap_conf_and_source_peak_support_radius",
    "truth_used": False,
}


def false_runs(mask):
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, ~values, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def protect_window(table, selected_mode, policy=None):
    policy = POLICY if policy is None else policy
    n = len(table)
    direct = {}
    for side in ("left", "right"):
        required = [f"{side}_temp", f"{side}_source", f"{side}_conf", f"{side}_x", f"{side}_y"]
        if any(column not in table for column in required):
            raise ValueError("Missing observation-provenance columns")
        finite = np.isfinite(table[[f"{side}_temp", f"{side}_conf", f"{side}_x", f"{side}_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
        direct[side] = finite & table[f"{side}_source"].eq("detected").to_numpy() & table[f"{side}_conf"].ge(policy["confidence_minimum"]).to_numpy()
    if selected_mode in direct:
        observed = direct[selected_mode].copy()
    elif selected_mode in ("mean", "min", "max"):
        observed = direct["left"] & direct["right"]
    else:
        raise ValueError(f"Unsupported fusion mode: {selected_mode}")
    supported = observed.copy()
    gaps = []
    for start, stop in false_runs(observed):
        boundary = start == 0 or stop == n
        allowed = not boundary and stop-start <= policy["max_short_gap_frames"]
        if allowed:
            supported[start:stop] = True
        gaps.append({"start_frame": int(start), "stop_frame_exclusive": int(stop),
                     "length_frames": int(stop-start), "boundary": boundary,
                     "permitted_short_gap": allowed})
    unsafe_context = ~supported
    radius = policy["event_context_frames"]
    for index in np.flatnonzero(~supported):
        unsafe_context[max(0,index-radius):min(n,index+radius+1)] = True
    ready = n > 0 and bool(supported.all())
    return pd.DataFrame({"frame_index": np.arange(n), "left_direct": direct["left"],
                         "right_direct": direct["right"], "selected_inputs_direct": observed,
                         "short_gap_permitted": supported & ~observed,
                         "support_after_short_gap_policy": supported,
                         "unsafe_event_context": unsafe_context}), gaps, {
        "prediction_status": "ready" if ready else "abstain",
        "reason": "" if ready else "incomplete_selected_signal_support",
        "selected_mode": selected_mode, "direct_selected_frames": int(observed.sum()),
        "unsupported_frames": int((~supported).sum()),
        "longest_selected_gap_frames": max([g["length_frames"] for g in gaps], default=0)}
