"""Frozen CPU-only attribution of whole-curve selection versus gap filling."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "Experiment/07_all_video_pose_retrain/replay_lindian_20261016_v1"
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "runtime/ultralytics"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "Experiment/03_可靠事件参考"))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import paper_repro_rr as rr
from reference_tools import match_events

MODES = ("mean", "max", "min", "left", "right")
POLICIES = ("HALF", "LINEAR")
TERMS = ("prominence", "amplitude", "interval_cv", "missing")
ARMS = tuple(f"{mode.upper()}_{policy}" for policy in POLICIES for mode in MODES) + (
    "SELECT_HALF", "SELECT_LINEAR", *[f"SELECT_LINEAR_NO_{term.upper()}" for term in TERMS])
CONFIG = replace(rr.fast_fusion_quality_config(False), peak_distance=6, peak_prominence=.05,
                 adaptive_peak_prominence=False, merge_shallow_peaks=False, fusion_rescue_outlier_threshold=0)
RULES_PATH = REPO / "Experiment/05_方法核证/run.py"
SPLITS = ("all", "train", "val")
SCOPES = ("primary47", "all82", "anchored49", "P1", "P2", "P3")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        pd.DataFrame(frame).to_csv(stream, index=False)


def read(name):
    return pd.read_csv(HERE / "inputs" / name)


def freeze():
    if (HERE / "freeze.json").exists():
        verify_frozen()
        return
    registry = pd.read_csv(SOURCE / "reference_registry.csv")
    selected = registry[registry.cohort.ne("short73") & registry.eligible_rr].copy()
    assert len(selected) == 82 and selected.cohort.value_counts().to_dict() == {"anchored49": 47, "P3": 24, "P2": 6, "P1": 5}
    assert selected.split.value_counts().to_dict() == {"train": 69, "val": 13}
    assert selected.eligible_events.all() and selected.segment_id.is_unique
    ids = set(selected.segment_id)
    source_files = [RULES_PATH, Path(rr.__file__), REPO / "Experiment/03_可靠事件参考/reference_tools.py",
                    REPO / "models/best.pt",
                    REPO / "Experiment/07_all_video_pose_retrain/training_runs/yolo11n_pose_lindian_20261016_v1/weights/best.pt",
                    SOURCE / "freeze.json", SOURCE / "COMPLETED.json"]
    save_csv(HERE / "inputs/reference_registry.csv", selected)
    candidates = registry[registry.cohort.ne("short73")].copy()
    candidates["reason"] = np.where(candidates.eligible_rr, "included_complete_frozen_reference", "excluded_original_incomplete_reference")
    save_csv(HERE / "inputs/all_candidate_reference_audit.csv", candidates)
    source_files.append(SOURCE / "reference_registry.csv")
    for name in ("frame_times.csv", "reference_events.csv", "predictions.csv", "predicted_events.csv", "event_matches.csv"):
        path = SOURCE / name
        source_files.append(path)
        frame = pd.read_csv(path)
        save_csv(HERE / "inputs" / name, frame[frame.segment_id.isin(ids)])
    old_freeze = json.loads((SOURCE / "freeze.json").read_text(encoding="utf-8"))
    source_files += [Path(entry["path"]) for entry in old_freeze["files"] if Path(entry["path"]).suffix.lower() == ".pkl"]
    for segment in selected.segment_id:
        source = SOURCE / "temperatures" / f"{segment}.csv"
        destination = HERE / "inputs/temperatures" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        source_files.append(source)
    fingerprints = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(set(source_files))]
    snapshots = [{"path": str(path.relative_to(HERE)), "sha256": sha256(path)}
                 for path in sorted((HERE / "inputs").rglob("*")) if path.is_file()]
    code = [HERE / "PROTOCOL.md", Path(__file__), HERE / "test_verification.py"]
    write_json(HERE / "freeze.json", {"client_date": "2026-10-19", "classification": "exploratory_development_attribution",
        "windows": 82, "primary_windows": 47, "train": 69, "val": 13, "arms": list(ARMS),
        "primary_contrast": "SELECT_LINEAR minus MAX_LINEAR on primary47", "rr": "60*N/frozen_duration",
        "no_yolo_training_or_inference": True, "sources": fingerprints, "snapshots": snapshots,
        "implementation": [{"path": str(path.relative_to(HERE)), "sha256": sha256(path)} for path in code],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}})
    print("Frozen 82 complete windows, 47 primary; 69 train / 13 val.", flush=True)


def verify_frozen():
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["sources"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Original source changed: {item['path']}")
    for key in ("snapshots", "implementation"):
        for item in frozen[key]:
            if sha256(HERE / item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen {key} changed: {item['path']}")
    return frozen


def fill_curve(values, policy):
    numeric = pd.to_numeric(values, errors="coerce")
    if policy == "HALF":
        return numeric.fillna(.5)
    if policy == "LINEAR":
        return numeric.interpolate(method="linear", limit_direction="both").fillna(.5)
    raise ValueError(policy)


def score_value(quality, disabled=None):
    weights = {"prominence": 1., "amplitude": 2., "interval_cv": -.5, "missing": -2.}
    columns = {"prominence": "candidate_median_prominence", "amplitude": "candidate_amplitude",
               "interval_cv": "candidate_interval_cv", "missing": "candidate_missing_rate"}
    return float(sum((0. if term == disabled else weight) * quality[columns[term]] for term, weight in weights.items()))


def choose(candidates, disabled=None):
    return max(candidates, key=lambda item: score_value(item["quality"], disabled))


def peaks_from_curve(curve):
    values = np.asarray(curve, float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Invalid completed curve")
    smoothed = np.convolve(values, np.ones(3) / 3, mode="valid")
    peaks, _ = find_peaks(smoothed, distance=6, prominence=.05)
    return smoothed, peaks + 1


def predict_signals(table):
    left, right = rr.normalize_series(table.left_temp), rr.normalize_series(table.right_temp)
    candidates, outputs, qualities = {}, {}, []
    for policy in POLICIES:
        candidates[policy] = []
        for mode in MODES:
            values = fill_curve(rr.fuse_columns(left, right, mode), policy)
            quality = rr.curve_quality_score(values.to_numpy(float), mode, rr.fusion_side_missing(table, mode), CONFIG)
            item = {"mode": mode, "values": values, "quality": quality}
            candidates[policy].append(item)
            qualities.append({"policy": policy, "mode": mode, **quality})
            outputs[f"{mode.upper()}_{policy}"] = item
        outputs[f"SELECT_{policy}"] = choose(candidates[policy])
    for term in TERMS:
        outputs[f"SELECT_LINEAR_NO_{term.upper()}"] = choose(candidates["LINEAR"], term)
    assert set(outputs) == set(ARMS)
    return outputs, qualities


def legacy_rules():
    spec = importlib.util.spec_from_file_location("frozen_quality_attribution_rules", RULES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_acceptance():
    if (HERE / "replay_acceptance.json").exists():
        return
    verify_frozen()
    rules = legacy_rules()
    registry, prior, events = read("reference_registry.csv"), read("predictions.csv"), read("predicted_events.csv")
    times = read("frame_times.csv")
    checked = []
    for win in registry.itertuples(index=False):
        table = read(f"temperatures/{win.segment_id}.csv")
        binding = times[times.segment_id.eq(win.segment_id)].sort_values("frame_index").time_seconds.to_numpy(float)
        outputs, _ = predict_signals(table)
        for old_arm, new_arm in (("C0P0", "MAX_HALF"), ("C1P0", "SELECT_LINEAR")):
            curve, frames, selected, prominence = rules.run_one(table, old_arm)
            stored = prior[prior.segment_id.eq(win.segment_id) & prior.arm.eq(old_arm)].iloc[0]
            stored_events = events[events.segment_id.eq(win.segment_id) & events.arm.eq(old_arm)].sort_values("frame_index")
            _, new_frames = peaks_from_curve(outputs[new_arm]["values"])
            if not (len(frames) == int(stored.predicted_count) and selected == stored.fusion
                    and np.array_equal(frames, stored_events.frame_index.to_numpy(int))
                    and np.array_equal(frames, new_frames)
                    and np.allclose(binding[frames], stored_events.time_seconds, rtol=0, atol=1e-12)
                    and outputs[new_arm]["mode"] == ("max" if old_arm == "C0P0" else selected)
                    and np.allclose(curve.fused_norm, outputs[new_arm]["values"], rtol=0, atol=1e-12)):
                raise AssertionError(f"Replay mismatch: {win.segment_id} {old_arm}")
            checked.append({"segment_id": win.segment_id, "arm": old_arm, "new_arm": new_arm,
                            "count": len(frames), "selected": selected, "peak_frames_match": True,
                            "time_match": True, "curve_match": True})
    save_csv(HERE / "replay_acceptance.csv", checked)
    write_json(HERE / "replay_acceptance.json", {"status": "passed", "window_arms": len(checked), "windows": len(registry),
               "counts_peaks_modes_curves_and_times_match": True, "new_implementation_matches_original": True})
    print("Replay acceptance passed: 164 window-arms, exact peaks/counts/modes.", flush=True)


def run_predictions():
    if (HERE / "predictions.csv").exists():
        raise FileExistsError("Never overwrite a completed prediction snapshot")
    verify_frozen()
    acceptance = json.loads((HERE / "replay_acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "passed"
    registry, refs, times = read("reference_registry.csv"), read("reference_events.csv"), read("frame_times.csv")
    predictions, candidate_rows, peak_rows, match_rows = [], [], [], []
    for number, win in enumerate(registry.itertuples(index=False), 1):
        table = read(f"temperatures/{win.segment_id}.csv")
        binding = times[times.segment_id.eq(win.segment_id)].sort_values("frame_index").time_seconds.to_numpy(float)
        reference = refs[refs.segment_id.eq(win.segment_id)].sort_values("time_seconds").time_seconds.to_numpy(float)
        assert len(reference) == win.truth_count and len(binding) == len(table)
        assert np.isfinite(binding).all() and np.all(np.diff(binding) > 0) and win.duration_seconds > 0
        assert set(table.left_source) <= {"detected", "missing"} and set(table.right_source) <= {"detected", "missing"}
        outputs, qualities = predict_signals(table)
        candidate_rows.extend({"segment_id": win.segment_id, **quality} for quality in qualities)
        left_valid, right_valid = table.left_temp.notna().mean(), table.right_temp.notna().mean()
        missing = (table.left_temp.isna() & table.right_temp.isna()).mean()
        curves = pd.DataFrame({"frame_index": np.arange(len(table)), "time_seconds": binding,
                               "left_norm": rr.normalize_series(table.left_temp), "right_norm": rr.normalize_series(table.right_temp)})
        for arm in ARMS:
            item = outputs[arm]
            smooth, peaks = peaks_from_curve(item["values"])
            count = len(peaks)
            truth_rr, predicted_rr = 60 * win.truth_count / win.duration_seconds, 60 * count / win.duration_seconds
            hits, fps, fns = match_events(binding[peaks], reference, .30)
            row = {**win._asdict(), "arm": arm, "selected_mode": item["mode"],
                   "predicted_count": count, "truth_rr_bpm": truth_rr, "predicted_rr_bpm": predicted_rr,
                   "absolute_rr_error_bpm": abs(predicted_rr - truth_rr), "exact_count": int(count == win.truth_count),
                   "mean_count_accuracy": max(0., 1 - abs(count - win.truth_count) / win.truth_count) if win.truth_count > 0 else float(count == 0),
                   "tp": len(hits), "fp": len(fps), "fn": len(fns),
                   "left_valid_ratio": left_valid, "right_valid_ratio": right_valid,
                   "valid_ratio_difference": abs(left_valid - right_valid), "bilateral_missing_ratio": missing,
                   "signal_observed": bool(table[["left_temp", "right_temp"]].notna().any().any())}
            predictions.append(row)
            peak_rows.extend({"segment_id": win.segment_id, "arm": arm, "frame_index": int(frame), "time_seconds": float(binding[frame])} for frame in peaks)
            match_rows.extend({"segment_id": win.segment_id, "arm": arm, "type": "TP", "predicted_index": i, "reference_index": j, "error_seconds": float(error)} for i, j, error in hits)
            match_rows.extend({"segment_id": win.segment_id, "arm": arm, "type": "FP", "predicted_index": i, "reference_index": None} for i in fps)
            match_rows.extend({"segment_id": win.segment_id, "arm": arm, "type": "FN", "predicted_index": None, "reference_index": j} for j in fns)
            curves[f"{arm}_fused"] = item["values"].to_numpy(float)
            curves[f"{arm}_smooth"] = np.r_[np.nan, smooth, np.nan]
            curves[f"{arm}_peak"] = np.isin(np.arange(len(table)), peaks).astype(int)
        save_csv(HERE / "curves" / f"{win.segment_id}.csv", curves)
        print(f"Attribution {number}/82: {win.segment_id}", flush=True)
    save_csv(HERE / "predictions.csv", predictions)
    save_csv(HERE / "candidate_quality.csv", candidate_rows)
    save_csv(HERE / "predicted_events.csv", peak_rows)
    save_csv(HERE / "event_matches.csv", match_rows)


def select_scope(frame, scope, split):
    selected = frame if scope == "all82" else frame[frame.cohort.eq("anchored49" if scope == "primary47" else scope)]
    return selected if split == "all" else selected[selected.split.eq(split)]


def measurements(frame):
    truth, prediction = frame.truth_rr_bpm.to_numpy(float), frame.predicted_rr_bpm.to_numpy(float)
    assert np.isfinite(truth).all() and np.isfinite(prediction).all() and len(frame)
    error = prediction - truth
    sst = np.sum((truth - truth.mean()) ** 2)
    tp, fp, fn = (int(frame[key].sum()) for key in ("tp", "fp", "fn"))
    return {"scored_windows": len(frame), "cow_groups": frame.cow_group.nunique(),
            "rr_mae_bpm": float(np.abs(error).mean()), "rr_rmse_bpm": float(np.sqrt(np.mean(error ** 2))),
            "rr_r2": float(1 - np.sum(error ** 2) / sst) if sst > 0 else np.nan,
            "exact_count": int(frame.exact_count.sum()), "exact_count_rate": float(frame.exact_count.mean()),
            "mean_count_accuracy": float(frame.mean_count_accuracy.mean()),
            "tp": tp, "fp": fp, "fn": fn, "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}


def paired_ci(candidate, control, draws=10000, seed=20260929):
    left, right = candidate.set_index("segment_id"), control.set_index("segment_id")
    if set(left.index) != set(right.index) or not left.index.is_unique or not right.index.is_unique:
        raise ValueError("Paired window mismatch")
    right = right.reindex(left.index)
    delta = left.absolute_rr_error_bpm - right.absolute_rr_error_bpm
    clusters = pd.DataFrame({"cow_group": left.cow_group, "delta": delta}).groupby("cow_group").delta.agg(["sum", "size"])
    ix = np.random.default_rng(seed).integers(0, len(clusters), size=(draws, len(clusters)))
    boot = clusters["sum"].to_numpy()[ix].sum(axis=1) / clusters["size"].to_numpy()[ix].sum(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    baseline = float(right.absolute_rr_error_bpm.mean())
    return {"windows": len(left), "cow_groups": len(clusters), "delta_mae_bpm": float(delta.mean()),
            "ci95_low": float(low), "ci95_high": float(high), "draws": draws, "seed": seed,
            "relative_mae_reduction": float(-delta.mean() / baseline) if baseline > 0 else np.nan}


def global_fixed(predictions):
    records = []
    for rank, mode in enumerate(MODES):
        train = predictions[predictions.arm.eq(f"{mode.upper()}_LINEAR") & predictions.split.eq("train")]
        records.append({"mode": mode, "tie_rank": rank, "train_mae_bpm": float(train.absolute_rr_error_bpm.mean())})
    chosen = min(records, key=lambda item: (item["train_mae_bpm"], item["tie_rank"]))
    return chosen, records


def evaluate():
    predictions = pd.read_csv(HERE / "predictions.csv")
    assert len(predictions) == 82 * 16 and not predictions.duplicated(["segment_id", "arm"]).any()
    rows, comparisons = [], []
    contrasts = [("SELECT_LINEAR", "MAX_LINEAR", "selection_given_linear_primary"),
                 ("SELECT_HALF", "MAX_HALF", "selection_given_half"),
                 ("MAX_LINEAR", "MAX_HALF", "filling_given_max"),
                 ("SELECT_LINEAR", "SELECT_HALF", "filling_given_selection"),
                 ("SELECT_LINEAR", "MAX_HALF", "original_package")]
    contrasts += [(f"SELECT_LINEAR_NO_{term.upper()}", "SELECT_LINEAR", f"remove_{term}") for term in TERMS]
    for scope in SCOPES:
        for split in SPLITS:
            pool = select_scope(predictions, scope, split)
            if pool.empty:
                continue
            for arm in ARMS:
                rows.append({"scope": scope, "split": split, "arm": arm, **measurements(pool[pool.arm.eq(arm)])})
            for candidate, control, name in contrasts:
                comparisons.append({"scope": scope, "split": split, "candidate": candidate, "control": control,
                                    "contrast": name, **paired_ci(pool[pool.arm.eq(candidate)], pool[pool.arm.eq(control)])})
    save_csv(HERE / "metrics.csv", rows)
    save_csv(HERE / "paired_intervals.csv", comparisons)
    chosen, fixed = global_fixed(predictions)
    val = predictions[predictions.split.eq("val")]
    write_json(HERE / "global_fixed_mode.json", {"selection_uses": "69 train-group complete references only",
        "classification": "development comparator, not unseen RR validation", "candidates": fixed,
        "chosen_mode": chosen["mode"], "chosen_arm": f"{chosen['mode'].upper()}_LINEAR",
        "validation_windows": 13, "validation_fixed": measurements(val[val.arm.eq(f"{chosen['mode'].upper()}_LINEAR")]),
        "validation_selector": measurements(val[val.arm.eq("SELECT_LINEAR")])})
    control = predictions[predictions.arm.eq("MAX_LINEAR")].set_index("segment_id")
    selected = predictions[predictions.arm.eq("SELECT_LINEAR")].set_index("segment_id")
    attribution = selected[["cohort", "split", "cow_group", "selected_mode", "left_valid_ratio", "right_valid_ratio",
                            "valid_ratio_difference", "bilateral_missing_ratio", "absolute_rr_error_bpm"]].copy()
    attribution["control_absolute_rr_error_bpm"] = control.absolute_rr_error_bpm
    attribution["delta_mae_bpm"] = selected.absolute_rr_error_bpm - control.absolute_rr_error_bpm
    for term in TERMS:
        arm = f"SELECT_LINEAR_NO_{term.upper()}"
        without = predictions[predictions.arm.eq(arm)].set_index("segment_id")
        attribution[f"without_{term}_mode"] = without.selected_mode
        attribution[f"without_{term}_changed"] = without.selected_mode.ne(selected.selected_mode)
        attribution[f"without_{term}_delta_abs_error"] = without.absolute_rr_error_bpm - selected.absolute_rr_error_bpm
    save_csv(HERE / "attribution_by_window.csv", attribution.reset_index())
    ranked = attribution.reset_index()
    better = ranked[ranked.delta_mae_bpm.lt(0)].sort_values(["delta_mae_bpm", "segment_id"]).head(3).assign(case_type="largest_improvement")
    worse = ranked[ranked.delta_mae_bpm.gt(0)].sort_values(["delta_mae_bpm", "segment_id"], ascending=[False, True]).head(3).assign(case_type="largest_regression")
    save_csv(HERE / "failure_cases.csv", pd.concat([better, worse], ignore_index=True))
    verify_frozen()
    write_json(HERE / "NUMERIC_COMPLETED.json", {"windows": 82, "arms": 16, "predictions": len(predictions),
        "metric_rows": len(rows), "paired_intervals": len(comparisons), "original_sources_unchanged": True,
        "novelty_and_final_claim_review": "pending", "no_automatic_adoption": True})
    print(json.dumps({"numeric_complete": True, "predictions": len(predictions), "fixed_mode": chosen["mode"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acceptance", "predict", "evaluate", "verify", "all"))
    args = parser.parse_args()
    stages = {"freeze": freeze, "acceptance": replay_acceptance, "predict": run_predictions,
              "evaluate": evaluate, "verify": verify_frozen}
    if args.stage == "all":
        for name in ("freeze", "acceptance", "predict", "evaluate"):
            stages[name]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
