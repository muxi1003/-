"""Replay fixed BGR/ROI and four frozen RR arms after grouped YOLO retraining."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from review_gate import CORPUS, HERE, sha256

ROOT = HERE.parents[1]
EXP6 = ROOT / "Experiment" / "06_林甸扩展试验"
os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics_config")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(EXP6))
sys.path.insert(0, str(ROOT / "Experiment" / "05_方法核证"))

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import score_lindian_expansion as p1
import score_lindian_expansion_p2 as p2
import score_lindian_expansion_p3 as p3
import run as rules


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def prepare(output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frames = pd.read_csv(CORPUS / "frames.csv", dtype=str).fillna("")
    frames = frames[frames.cohort.ne("R3_jiufu")]
    registry, mappings, events, files = [], [], [], []
    matched = rules.read(rules.SNAP / "matched_windows.csv")
    a49, refs49, bind49 = rules.prepare_references(matched)
    b73 = rules.read(ROOT / "Dataset_new/72video/al_images/innovation_repro_summary.csv").set_index("video_id")
    a49 = a49.set_index("video_id")
    refs_root = ROOT / "Experiment/03_可靠事件参考/submissions/20260917_r2_v2"
    files.extend([rules.SNAP / "matched_windows.csv", ROOT / "Dataset_new/72video/al_images/innovation_repro_summary.csv",
                  refs_root / "lindian49/annotation_windows.csv", refs_root / "lindian49/reference_events.csv",
                  refs_root / "lindian_frame_time_bindings.csv"])
    windows1, _, records1, paths1 = p1.validate(Path("C:/Users/muxi/Desktop/实验/呼吸事件标注 LX-20260924-P1"))
    windows2, records2, paths2 = p2.validate(Path("C:/Users/muxi/Desktop/实验/呼吸事件标注 LX-20260928-P2"))
    windows3, records3, _, paths3 = p3.validate(EXP6 / "20260929_p3_reference_adjudicated_v1")
    files.extend(paths1 + paths2 + paths3)
    expansion = {"P1": (windows1.set_index("video_id"), records1),
                 "P2": (windows2.set_index("video_id"), records2),
                 "P3": (windows3.set_index("video_id"), records3)}
    for segment, group in frames.groupby("segment_id", sort=False):
        row = group.iloc[0]
        cohort = row.cohort
        refs = pd.DataFrame()
        if cohort == "short73":
            video_id = segment.removeprefix("S73_")
            reference = b73.loc[video_id]
            eligible, count, duration, status = True, float(reference.truth_count), float(reference.duration_seconds), "B73-I_count_only"
            times = np.arange(len(group)) / 8.7
        elif cohort == "anchored49":
            video_id = segment.removeprefix("A49_")
            reference = a49.loc[video_id]
            eligible = reference.annotation_status == "complete"
            count = float(reference.manual_breath_count) if eligible else np.nan
            duration, status = float(reference.duration_seconds), reference.annotation_status
            times = bind49[bind49.video_id.eq(video_id)].sort_values("frame_index").annotation_video_time_seconds.to_numpy(float)
            if eligible:
                refs = refs49[refs49.window_id.eq(reference.window_id)]
        else:
            video_id = segment.removeprefix(cohort + "_")
            windows, records = expansion[cohort]
            reference = windows.loc[video_id]
            _, binding, selected = records[reference.window_id]
            if cohort == "P2":
                status = reference.analysis_status
                eligible = status == "complete"
            elif cohort == "P3":
                status = reference.analysis_disposition
                eligible = status == "complete_primary"
            else:
                status = reference.annotation_status
                eligible = status == "complete"
            count = float(reference.manual_breath_count) if eligible else np.nan
            duration = float(reference.duration_seconds)
            binding = binding.sort_values("source_frame")
            if not np.array_equal(binding.source_frame.to_numpy(int), group.source_frame.to_numpy(int)):
                raise ValueError(f"Source-frame binding changed: {segment}")
            times = binding.playback_seconds.to_numpy(float)
            refs = selected if eligible else pd.DataFrame()
        if len(times) != len(group) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Frame/time binding invalid: {segment}")
        registry.append({"segment_id": segment, "video_id": video_id, "cohort": cohort,
                         "split": row.split, "cow_group": row.cow_group, "eligible_rr": eligible,
                         "eligible_events": eligible and cohort != "short73", "reference_status": status,
                         "truth_count": count, "duration_seconds": duration})
        mappings.extend({"segment_id": segment, "frame_index": index, "time_seconds": float(time)}
                        for index, time in enumerate(times))
        if not refs.empty:
            if len(refs) != int(count):
                raise ValueError(f"Complete reference count/events differ: {segment}")
            events.extend({"segment_id": segment, "event_id": ref.event_id, "time_seconds": ref.event_time_seconds}
                          for ref in refs.sort_values("event_time_seconds").itertuples())
    save(output / "reference_registry.csv", registry)
    save(output / "frame_times.csv", mappings)
    save(output / "reference_events.csv", events)
    save(output / "frames.csv", frames)
    files.extend([CORPUS / "frames.csv", CORPUS / "segments.csv", p1.RF,
                  Path(__file__), Path(rules.__file__), Path(p1.rr.__file__)])
    unique = sorted(set(Path(path) for path in files))
    frozen = [{"path": str(path), "sha256": sha256(path)} for path in unique]
    generated = [{"path": str(output / name), "sha256": sha256(output / name)} for name in
                 ("reference_registry.csv", "frame_times.csv", "reference_events.csv", "frames.csv")]
    (output / "freeze.json").write_text(json.dumps({"files": frozen, "generated": generated,
        "arms": list(rules.ARMS), "roi_radius": 20, "min_temperature": 20., "keypoint_confidence": .5,
        "event_tolerance_seconds": .30, "scope": "same-farm development replay, not unseen RR confirmation",
        "time": "Frozen cohort-specific duration and frame-time binding; RR=60*N/T",
        "input": "Frozen corpus image pixels; legacy JPG retained, new PNG sequentially decoded; no short-gap repair",
        "R1": "Aliases P2; unresolved reference disputes do not silently replace the frozen P2 six-window analysis",
        "R3_jiufu": "Deferred; not trained or replayed"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(registry).groupby("cohort").eligible_rr.agg(["size", "sum"]).to_string(), flush=True)


def temperatures(group, detector, rf):
    rows = []
    ordered = group.sort_values("source_frame", key=lambda x: x.astype(int))
    for frame in ordered.itertuples(index=False):
        bgr = cv2.imdecode(np.fromfile(frame.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Cannot read BGR image: {frame.image_path}")
        result = detector(bgr, verbose=False, device=0)[0]
        detection = p1.rr.select_detection(result)
        values = {f"{side}_{key}": np.nan for side in ("left", "right") for key in ("temp", "x", "y", "conf")}
        values.update(left_source="missing", right_source="missing")
        if detection is not None:
            points = result.keypoints.data[detection].detach().cpu().numpy()
            for index, side in enumerate(("left", "right")):
                x, y, conf = p1.rr.keypoint_xy_conf(points[index])
                values.update({f"{side}_x": x, f"{side}_y": y, f"{side}_conf": conf})
                if conf >= .5:
                    temp = p1.rr.circle_temperature(bgr, rf, x, y, 20, 20.)
                    values[f"{side}_temp"] = temp
                    values[f"{side}_source"] = "detected" if np.isfinite(temp) else "missing"
        rows.append({"frame_name": Path(frame.image_path).stem, "source_frame": int(frame.source_frame), **values})
    return pd.DataFrame(rows)


def score(output, weights):
    if (output / "COMPLETED.json").exists() or (output / "inference_started.json").exists():
        raise FileExistsError("Never overwrite a started RR run")
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for entry in freeze["files"] + freeze["generated"]:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Frozen replay input changed: {entry['path']}")
    (output / "inference_started.json").write_text(json.dumps({"weights": str(weights), "sha256": sha256(weights)}, indent=2), encoding="utf-8")
    detector, rf = YOLO(str(weights)), joblib.load(p1.RF)
    frames = pd.read_csv(output / "frames.csv", dtype=str).fillna("")
    registry = pd.read_csv(output / "reference_registry.csv")
    times_all, refs = pd.read_csv(output / "frame_times.csv"), pd.read_csv(output / "reference_events.csv")
    predictions, predicted_events, matches = [], [], []
    for number, win in enumerate(registry.itertuples(index=False), 1):
        group = frames[frames.segment_id.eq(win.segment_id)]
        table = temperatures(group, detector, rf)
        save(output / "temperatures" / f"{win.segment_id}.csv", table)
        times = times_all[times_all.segment_id.eq(win.segment_id)].sort_values("frame_index").time_seconds.to_numpy(float)
        selected_refs = refs[refs.segment_id.eq(win.segment_id)].sort_values("time_seconds")
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        axes[0, 0].plot(times, table.left_temp, label="Left", color="#176eaf")
        axes[0, 0].plot(times, table.right_temp, label="Right", color="#db7426")
        axes[0, 0].set(title=win.segment_id, ylabel="Mapped temperature (C)")
        axes[0, 0].legend()
        axes[0, 1].axis("off")
        axes[0, 1].text(0, .7, f"YOLO split: {win.split}\nReference: {win.reference_status}\nDuration: {win.duration_seconds:g} s\nDevelopment replay, not independent confirmation", fontsize=10)
        for index, arm in enumerate(rules.ARMS):
            curve, peak_frames, selected, prominence = rules.run_one(table, arm)
            curve["time_seconds"] = times
            save(output / "curves" / arm / f"{win.segment_id}.csv", curve)
            count = len(peak_frames)
            record = {**win._asdict(), "arm": arm, "predicted_count": count,
                      "predicted_rr_bpm": 60 * count / win.duration_seconds, "fusion": selected, "prominence": prominence,
                      "left_valid_frames": int(table.left_temp.notna().sum()), "right_valid_frames": int(table.right_temp.notna().sum()),
                      "output_status": "fixed_rule_output" if table[["left_temp", "right_temp"]].notna().any().any() else "zero_count_from_all_missing_signal",
                      "tp": np.nan, "fp": np.nan, "fn": np.nan}
            if win.eligible_rr:
                record.update(truth_rr_bpm=60 * win.truth_count / win.duration_seconds,
                              absolute_rr_error_bpm=60 * abs(count - win.truth_count) / win.duration_seconds,
                              exact_count=int(count == win.truth_count),
                              mean_count_accuracy=max(0., 1 - abs(count - win.truth_count) / win.truth_count) if win.truth_count > 0 else float(count == 0))
            if win.eligible_events:
                hits, fp, fn = p1.match_events(times[peak_frames], selected_refs.time_seconds.to_numpy(float), .30)
                record.update(tp=len(hits), fp=len(fp), fn=len(fn))
                matches.extend({"segment_id": win.segment_id, "arm": arm, "type": "TP", "predicted_index": int(i), "reference_index": int(j), "error_seconds": float(error)} for i, j, error in hits)
                matches.extend({"segment_id": win.segment_id, "arm": arm, "type": "FP", "predicted_index": int(i), "reference_index": None} for i in fp)
                matches.extend({"segment_id": win.segment_id, "arm": arm, "type": "FN", "predicted_index": None, "reference_index": int(j)} for j in fn)
            predictions.append(record)
            predicted_events.extend({"segment_id": win.segment_id, "arm": arm, "frame_index": int(frame), "time_seconds": float(times[frame])} for frame in peak_frames)
            axis = axes[1 + index // 2, index % 2]
            axis.plot(times, curve.smoothed_norm, color="#176eaf", linewidth=1)
            axis.scatter(times[peak_frames], curve.smoothed_norm.iloc[peak_frames], color="#c9423b", s=14)
            axis.set(title=f"{arm} | {selected} | N={count}", xlabel="Time (s)", ylabel="Normalized signal")
        plots = output / "plots"
        plots.mkdir(exist_ok=True)
        fig.savefig(plots / f"{win.segment_id}.png", dpi=130)
        plt.close(fig)
        save(output / "predictions.csv", predictions)
        print(f"RR replay {number}/{len(registry)}: {win.segment_id}", flush=True)
    save(output / "predicted_events.csv", predicted_events)
    save(output / "event_matches.csv", matches)
    summarize(output, pd.DataFrame(predictions))
    (output / "COMPLETED.json").write_text(json.dumps({"segments": len(registry), "arms": list(rules.ARMS),
        "predictions": len(predictions), "weights_sha256": sha256(weights),
        "claim": "Development replay only; no automatic default adoption or paper-level innovation claim"}, indent=2), encoding="utf-8")


def summarize(output, predictions):
    metrics, intervals = [], []
    for cohort, whole in predictions.groupby("cohort", sort=False):
        for split in ("all", "train", "val"):
            pool = whole if split == "all" else whole[whole.split.eq(split)]
            scored = pool[pool.eligible_rr]
            for arm, group in scored.groupby("arm", sort=False):
                truth, pred = group.truth_rr_bpm.to_numpy(float), group.predicted_rr_bpm.to_numpy(float)
                total = np.sum((truth - truth.mean()) ** 2)
                ev = group[group.eligible_events]
                tp, fp, fn = (float(ev[key].sum()) for key in ("tp", "fp", "fn"))
                metrics.append({"cohort": cohort, "split": split, "arm": arm, "scored_windows": len(group),
                    "selected_windows": int(pool.segment_id.nunique()), "rr_mae_bpm": float(np.mean(abs(pred-truth))),
                    "rr_rmse_bpm": float(np.sqrt(np.mean((pred-truth)**2))),
                    "rr_r2": 1 - np.sum((pred-truth)**2) / total if total > 0 else np.nan,
                    "exact_count": int(group.exact_count.sum()), "mean_count_accuracy": float(group.mean_count_accuracy.mean()),
                    "tp": tp if len(ev) else np.nan, "fp": fp if len(ev) else np.nan, "fn": fn if len(ev) else np.nan,
                    "event_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan})
            base = scored[scored.arm.eq("C0P0")].set_index("segment_id")
            for candidate in rules.ARMS[1:]:
                other = scored[scored.arm.eq(candidate)].set_index("segment_id").reindex(base.index)
                if not len(base):
                    continue
                deltas = pd.DataFrame({"cow_group": base.cow_group, "delta": other.absolute_rr_error_bpm - base.absolute_rr_error_bpm})
                clusters = deltas.groupby("cow_group").delta.agg(["sum", "size"])
                rng = np.random.default_rng(20260929)
                samples = rng.integers(0, len(clusters), size=(10000, len(clusters)))
                boot = clusters["sum"].to_numpy()[samples].sum(axis=1) / clusters["size"].to_numpy()[samples].sum(axis=1)
                low, high = np.quantile(boot, [.025, .975])
                intervals.append({"cohort": cohort, "split": split, "candidate": candidate, "control": "C0P0",
                    "windows": len(base), "cow_groups": len(clusters), "delta_mae_bpm": float(deltas.delta.mean()),
                    "ci95_low": float(low), "ci95_high": float(high), "draws": 10000, "seed": 20260929,
                    "interpretation": "Exploratory development comparison; previously used RR windows"})
    save(output / "metrics.csv", metrics)
    save(output / "paired_intervals.csv", intervals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve())
    elif args.weights is None:
        parser.error("score requires --weights")
    else:
        score(args.output.resolve(), args.weights.resolve())
