"""Separate R2-only pixel remeasurement sensitivity, never part of the four arms."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run import ROOT, HERE, SNAP, REF, direct_temperatures, read, run_one, save_csv, save_json
from run import rr, match_events, sha256, metrics_for, bootstrap


OUT = HERE / "runs" / "gap_sensitivity_v1"
MODEL = ROOT / "Experiment" / "02_冻结独立测试" / "method_snapshots" / "20260909_v1" / "weights" / "clf_model_RGB_20240906.pkl"
MAIN = HERE / "runs" / "full_v2"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    annotations = read(REF / "lindian49" / "annotation_windows.csv")
    annotations = annotations[annotations.annotation_status.eq("complete")].set_index("video_id")
    assert len(annotations) == 47
    windows = read(SNAP / "matched_windows.csv")
    windows = windows[windows.cohort.eq("anchored49") & windows.video_id.isin(annotations.index)]
    bindings = read(REF / "lindian_frame_time_bindings.csv")
    reference_events = read(REF / "lindian49" / "reference_events.csv")
    baseline = read(MAIN / "predictions.csv")
    baseline = baseline[baseline.cohort.eq("anchored49") & baseline.arm.eq("C1P1") & baseline.reference_status.eq("complete")]
    assert len(baseline) == 47
    model = joblib.load(MODEL)
    config = replace(rr.fast_fusion_quality_config(False), max_track_gap=3,
                     max_track_anchor_shift=1.0, min_temp=20.0, radius=20)
    OUT.mkdir(parents=True)
    records, altered, event_rows, image_hashes = [], [], [], []
    for win in windows.itertuples(index=False):
        table = direct_temperatures(Path(win.input_temperature_csv), win.input_sha256, int(win.frames))
        before = table[["left_temp", "right_temp"]].copy()
        paths = [Path(win.image_dir) / name for name in table.frame_name]
        counts = {side: rr.repair_short_missing_runs(table, paths, model, side, config) for side in ("left", "right")}
        changed = []
        for side in ("left", "right"):
            indices = np.flatnonzero(before[f"{side}_temp"].isna() & table[f"{side}_temp"].notna())
            assert len(indices) == counts[side]
            for index in indices:
                image = paths[int(index)]
                digest = sha256(image)
                image_hashes.append({"video_id": win.video_id, "frame_index": int(index), "image": str(image), "sha256": digest})
                changed.append({"video_id": win.video_id, "frame_index": int(index), "side": side,
                                "new_temperature": float(table.loc[index, f"{side}_temp"]), "image_sha256": digest})
        altered.extend(changed)
        curve, frames, selected, prominence = run_one(table, "C1P1")
        save_csv(OUT / "curves" / f"{win.video_id}.csv", curve)
        ar = annotations.loc[win.video_id]
        duration, truth = float(ar.duration_seconds), int(ar.manual_breath_count)
        times = bindings[bindings.video_id.eq(win.video_id)].sort_values("frame_index")
        if len(times) != len(table) or not np.array_equal(times.frame_index.to_numpy(int), np.arange(len(table))):
            raise ValueError(f"Frame timing mismatch: {win.video_id}")
        p_times = times.annotation_video_time_seconds.to_numpy(float)[frames]
        r_times = reference_events[reference_events.window_id.eq(ar.window_id)].event_time_seconds.to_numpy(float)
        hit, fp, fn = match_events(p_times, r_times, .30)
        event_rows.append({"video_id": win.video_id, "tp": len(hit), "fp": len(fp), "fn": len(fn)})
        records.append({"video_id": win.video_id, "cluster_id": win.cluster_id, "arm": "C1P1_gap_remeasure",
                        "truth_count": truth, "truth_rr_bpm": 60 * truth / duration,
                        "predicted_count": len(frames), "predicted_rr_bpm": 60 * len(frames) / duration,
                        "duration_seconds": duration, "selected_mode": selected,
                        "peak_frames": ";".join(map(str, frames)), "remeasured_left": counts["left"], "remeasured_right": counts["right"]})
    result = pd.DataFrame(records)
    original = baseline[["video_id", "predicted_count", "predicted_rr_bpm"]].rename(columns={"predicted_count": "baseline_count", "predicted_rr_bpm": "baseline_rr_bpm"})
    result = result.merge(original, on="video_id", validate="one_to_one")
    save_csv(OUT / "predictions.csv", result)
    save_csv(OUT / "remeasured_samples.csv", altered)
    save_csv(OUT / "image_hashes.csv", image_hashes)
    save_csv(OUT / "event_metrics_by_window.csv", event_rows)
    score = metrics_for(result)
    interval = bootstrap(result, baseline)
    events = pd.DataFrame(event_rows)[["tp", "fp", "fn"]].sum()
    f1 = 2 * events.tp / (2 * events.tp + events.fp + events.fn)
    original_events = pd.read_csv(MAIN / "metrics.csv")
    original_f1 = float(original_events[(original_events.cohort == "anchored49") & (original_events.arm == "C1P1")].event_f1.iloc[0])
    report = {"status": "separate_sensitivity_only_not_adopted", "n": len(result),
              "source_run": str(MAIN), "model_path": str(MODEL), "model_sha256": sha256(MODEL),
              "main_manifest_sha256": sha256(MAIN / "freeze_manifest.json"),
              "code_sha256": sha256(Path(__file__)), "remeasured_side_frames": len(altered),
              "changed_prediction_windows": int(result.predicted_count.ne(result.baseline_count).sum()),
              "rr_metrics": score, "paired_vs_C1P1_no_gap": interval,
              "event_f1": float(f1), "event_f1_no_gap": original_f1,
              "decision": "No effect on the main four-arm adjudication; this is a post-main sensitivity."}
    save_json(OUT / "sensitivity_summary.json", report)
    print(json.dumps({k: report[k] for k in ("n", "remeasured_side_frames", "changed_prediction_windows", "event_f1", "event_f1_no_gap")}, ensure_ascii=False))
    print(score["rr_mae_bpm"], interval)
    print(OUT)


if __name__ == "__main__":
    main()
