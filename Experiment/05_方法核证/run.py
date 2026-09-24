"""Frozen, same-input Chen-rule replay versus curve selection and constrained peaks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "Experiment"))
sys.path.insert(0, str(ROOT / "Experiment" / "03_可靠事件参考"))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import paper_repro_rr as rr
from experiment_common import measurements, paired_cluster_bootstrap, sha256
from reference_tools import match_events

HERE = Path(__file__).resolve().parent
SNAP = ROOT / "Experiment" / "01_同口径消融" / "input_snapshots" / "20260909_v1"
REF = ROOT / "Experiment" / "03_可靠事件参考" / "submissions" / "20260917_r2_v2"
PAPER = Path("E:/real/学习/呼吸文献/1-s2.0-S0306456525001111-main.pdf")
SOURCE = Path("E:/real/use_code/代码/站立呼吸/yoloV8/temperature_extraction/single_nose/double_noses0117_20.py")
PILOT_IDS = ("16170075", "170333", "zs197000")
ARMS = ("C0P0", "C1P0", "C0P1", "C1P1")


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"video_id": str}, encoding="utf-8-sig")


def freeze_files(windows: pd.DataFrame) -> list[dict]:
    base = [HERE / "PROTOCOL.md", HERE / "run.py", PAPER, SOURCE,
            ROOT / "scripts" / "paper_repro_rr.py",
            ROOT / "Experiment" / "experiment_common.py",
            ROOT / "Experiment" / "03_可靠事件参考" / "reference_tools.py",
            SNAP / "matched_windows.csv", SNAP / "input_hashes.csv",
            ROOT / "Dataset_new" / "72video" / "al_images" / "innovation_repro_summary.csv",
            REF / "lindian49" / "annotation_windows.csv",
            REF / "lindian49" / "reference_events.csv",
            REF / "lindian_frame_time_bindings.csv"]
    base += [Path(p) for p in windows.input_temperature_csv]
    if any(not p.is_file() for p in base):
        raise FileNotFoundError([str(p) for p in base if not p.is_file()])
    return [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in base]


def prepare_references(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(windows) != 122 or set(windows.cohort.value_counts().to_dict().items()) != {("anchored49", 49), ("internal73", 73)}:
        raise ValueError("Matched input set is not 49+73")
    if windows.duplicated(["cohort", "video_id"]).any():
        raise ValueError("Duplicate input IDs")
    annotations = read(REF / "lindian49" / "annotation_windows.csv")
    if len(annotations) != 49 or annotations.video_id.duplicated().any():
        raise ValueError("R2 annotation windows not 49 unique IDs")
    if annotations.annotation_status.eq("complete").sum() != 47:
        raise ValueError("R2 complete analysis set changed")
    events = read(REF / "lindian49" / "reference_events.csv")
    bindings = read(REF / "lindian_frame_time_bindings.csv")
    if events.duplicated(["window_id", "event_id"]).any() or bindings.duplicated(["video_id", "frame_index"]).any():
        raise ValueError("Duplicate event or time binding")
    complete = annotations[annotations.annotation_status.eq("complete")]
    counts = events.groupby("window_id").size()
    for row in complete.itertuples():
        if counts.get(row.window_id, 0) != int(row.manual_breath_count):
            raise ValueError(f"R2 event/count conflict: {row.window_id}")
    original = read(ROOT / "Dataset_new" / "72video" / "al_images" / "innovation_repro_summary.csv")
    if len(original) != 73 or original.video_id.duplicated().any():
        raise ValueError("B73-I reference not 73 unique IDs")
    b73 = windows[windows.cohort.eq("internal73")].merge(
        original[["video_id", "truth_count", "duration_seconds"]], on="video_id", validate="one_to_one", suffixes=("", "_original"))
    if not np.allclose(b73.duration_seconds.astype(float), b73.duration_seconds_original.astype(float)):
        raise ValueError("B73-I durations differ")
    return annotations, events, bindings


def direct_temperatures(path: Path, expected_hash: str, frames: int) -> pd.DataFrame:
    if sha256(path) != expected_hash:
        raise ValueError(f"Temperature input hash changed: {path}")
    table = pd.read_csv(path)
    if len(table) != frames or not {"frame_name", "left_temp", "right_temp", "left_source", "right_source"}.issubset(table):
        raise ValueError(f"Invalid temperature table: {path}")
    for side in ("left", "right"):
        table[f"{side}_temp"] = pd.to_numeric(table[f"{side}_temp"], errors="coerce").where(table[f"{side}_source"].eq("detected"))
    return table


def normalize(values: pd.Series) -> pd.Series:
    return rr.normalize_series(values)


def peak_rule(fused: np.ndarray, constrained: bool) -> tuple[np.ndarray, np.ndarray, float]:
    smoothed = np.convolve(fused, np.ones(3) / 3, mode="valid")
    if not constrained:
        peaks, _ = find_peaks(smoothed, distance=6, prominence=0.05)
        return smoothed, peaks + 1, 0.05
    config = replace(rr.fast_fusion_quality_config(False), peak_distance=5,
                     peak_prominence=0.035, adaptive_peak_prominence=True,
                     merge_shallow_peaks=True, min_rr_bpm=20., max_rr_bpm=100.)
    peaks, _, prominence, _, _ = rr.detect_peaks(smoothed, config)
    return smoothed, peaks + 1, float(prominence)


def run_one(table: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, np.ndarray, str, float]:
    left, right = normalize(table.left_temp), normalize(table.right_temp)
    if mode[1] == "0":
        fused = pd.concat([left, right], axis=1).max(axis=1, skipna=True).fillna(0.5)
        selected = "framewise_max"
    else:
        quality_config = replace(rr.fast_fusion_quality_config(False), peak_distance=6,
                                 peak_prominence=0.05, adaptive_peak_prominence=False,
                                 merge_shallow_peaks=False, fusion_rescue_outlier_threshold=0)
        selected, fused, _ = rr.select_fused_series(table, left, right, "adaptive", quality_config)
    smoothed, peak_frames, prominence = peak_rule(fused.to_numpy(float), mode[3] == "1")
    if (peak_frames < 0).any() or (peak_frames >= len(table)).any():
        raise ValueError("Peak frame outside input")
    curve = pd.DataFrame({"frame_index": np.arange(len(table)), "frame_name": table.frame_name,
                          "left_norm": left, "right_norm": right, "fused_norm": fused,
                          "smoothed_norm": np.r_[np.nan, smoothed, np.nan],
                          "is_peak": np.isin(np.arange(len(table)), peak_frames),
                          "selected_mode": selected})
    return curve, peak_frames, selected, prominence


def metrics_for(group: pd.DataFrame) -> dict:
    if len(group) == 0:
        raise ValueError("Empty analysis set")
    return measurements(group)


def bootstrap(group: pd.DataFrame, control: pd.DataFrame) -> dict:
    both = group.merge(control[["video_id", "predicted_rr_bpm"]], on="video_id", suffixes=("", "_control"), validate="one_to_one")
    both["paired_abs_error_delta"] = abs(both.predicted_rr_bpm - both.truth_rr_bpm) - abs(both.predicted_rr_bpm_control - both.truth_rr_bpm)
    return paired_cluster_bootstrap(both, draws=10000, seed=20260924)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    out = HERE / "runs" / ("pilot_v2" if args.stage == "pilot" else "full_v2")
    if out.exists():
        raise FileExistsError(f"Immutable run already exists: {out}")
    windows = read(SNAP / "matched_windows.csv")
    annotations, refs, bindings = prepare_references(windows)
    manifest = {"stage": args.stage, "protocol": "v2", "files": freeze_files(windows),
                "arms": list(ARMS), "pilot_ids": list(PILOT_IDS),
                "note": "Frozen sources hashed in place; no complete original-model replay"}
    out.mkdir(parents=True)
    save_json(out / "freeze_manifest.json", manifest)
    save_csv(out / "reference_disposition.csv", annotations[["video_id", "window_id", "annotation_status", "manual_breath_count", "duration_seconds", "reference_notes"]])
    old73 = windows[windows.cohort.eq("internal73")][["video_id", "truth_count"]].merge(
        read(ROOT / "Dataset_new" / "72video" / "al_images" / "innovation_repro_summary.csv")[["video_id", "truth_count"]],
        on="video_id", validate="one_to_one", suffixes=("_matched_legacy", "_B73I"))
    save_csv(out / "B73_truth_discrepancies.csv", old73[old73.truth_count_matched_legacy.astype(int).ne(old73.truth_count_B73I.astype(int))])
    if args.stage == "pilot":
        windows = windows[windows.video_id.isin(PILOT_IDS)]
    source_data = read(ROOT / "Dataset_new" / "72video" / "al_images" / "innovation_repro_summary.csv").set_index("video_id")
    annotations = annotations.set_index("video_id")
    predictions, predicted_events, matching, case_rows = [], [], [], []
    for win in windows.itertuples(index=False):
        input_path = Path(win.input_temperature_csv)
        table = direct_temperatures(input_path, win.input_sha256, int(win.frames))
        is_primary = win.cohort == "anchored49" and annotations.loc[win.video_id, "annotation_status"] == "complete"
        if win.cohort == "anchored49":
            ar = annotations.loc[win.video_id]
            truth_count, duration = (int(ar.manual_breath_count), float(ar.duration_seconds)) if is_primary else (None, float(ar.duration_seconds))
            event_ref = refs[refs.window_id.eq(ar.window_id)].sort_values("event_time_seconds") if is_primary else pd.DataFrame()
            time_map = bindings[bindings.video_id.eq(win.video_id)].sort_values("frame_index")
            if len(time_map) != len(table) or not np.array_equal(time_map.frame_index.to_numpy(int), np.arange(len(table))):
                raise ValueError(f"Missing or unordered video frame timestamps: {win.video_id}")
            times = time_map.annotation_video_time_seconds.to_numpy(float)
            if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
                raise ValueError(f"Invalid frame time binding: {win.video_id}")
        else:
            orig = source_data.loc[win.video_id]
            truth_count, duration = int(orig.truth_count), float(orig.duration_seconds)
            event_ref, times = pd.DataFrame(), np.arange(len(table)) / 8.7
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError("Invalid frozen duration")
        cluster = win.cluster_id
        for arm in ARMS:
            curve, frames, selected, prominence = run_one(table, arm)
            save_csv(out / "curves" / win.cohort / arm / f"{win.video_id}.csv", curve)
            count = len(frames)
            record = {"cohort": win.cohort, "video_id": win.video_id, "arm": arm,
                      "cluster_id": cluster, "reference_status": "complete" if is_primary else (str(annotations.loc[win.video_id, "annotation_status"]) if win.cohort == "anchored49" else "B73-I count only"),
                      "duration_seconds": duration, "truth_count": truth_count,
                      "truth_rr_bpm": 60 * truth_count / duration if truth_count is not None else np.nan,
                      "predicted_count": count, "predicted_rr_bpm": 60 * count / duration,
                      "selected_mode": selected, "selected_prominence": prominence,
                      "output_status": "ok", "peak_frames": ";".join(map(str, frames))}
            predictions.append(record)
            for k, frame in enumerate(frames):
                predicted_events.append({"cohort": win.cohort, "video_id": win.video_id, "arm": arm,
                                         "event_id": f"p{k+1}", "frame_index": int(frame),
                                         "time_seconds": float(times[frame]),
                                         "timebase": "R2_bound_frame_PTS" if win.cohort == "anchored49" else "nominal_8.7_no_event_truth"})
            if is_primary:
                refs_time = event_ref.event_time_seconds.to_numpy(float)
                p_time = times[frames]
                hits, fp, fn = match_events(p_time, refs_time, .30)
                for i, j, error in hits:
                    matching.append({"video_id": win.video_id, "arm": arm, "type": "TP", "predicted_event_id": f"p{i+1}", "reference_event_id": event_ref.iloc[j].event_id, "time_error_seconds": error})
                for i in fp:
                    matching.append({"video_id": win.video_id, "arm": arm, "type": "FP", "predicted_event_id": f"p{i+1}", "reference_event_id": "", "time_error_seconds": np.nan})
                for j in fn:
                    matching.append({"video_id": win.video_id, "arm": arm, "type": "FN", "predicted_event_id": "", "reference_event_id": event_ref.iloc[j].event_id, "time_error_seconds": np.nan})
                case_rows.append({"video_id": win.video_id, "arm": arm, "truth_count": truth_count,
                                  "predicted_count": count, "count_error": count - truth_count,
                                  "tp": len(hits), "fp": len(fp), "fn": len(fn),
                                  "missing_direct_left": int(table.left_temp.isna().sum()),
                                  "missing_direct_right": int(table.right_temp.isna().sum())})
        if win.cohort == "anchored49":
            baseline = run_one(table, "C0P0")[0].fused_norm.to_numpy(float)
            source_smoothed = np.convolve(baseline, np.ones(3) / 3, mode="valid")
            source_peaks, _ = find_peaks(source_smoothed, distance=6, prominence=.035, width=1, plateau_size=1)
            predictions.append({"cohort": "source_sensitivity", "video_id": win.video_id, "arm": "C0P0_source035",
                                "cluster_id": cluster, "reference_status": "sensitivity_only", "duration_seconds": duration,
                                "truth_count": truth_count, "truth_rr_bpm": 60 * truth_count / duration if truth_count is not None else np.nan,
                                "predicted_count": len(source_peaks), "predicted_rr_bpm": 60 * len(source_peaks) / duration,
                                "selected_mode": "framewise_max", "selected_prominence": .035, "output_status": "ok",
                                "peak_frames": ";".join(map(str, source_peaks + 1)),
                                "source_8p6_rr_bpm_historical": 60 * len(source_peaks) * 8.6 / len(source_smoothed)})
    pred = pd.DataFrame(predictions)
    save_csv(out / "predictions.csv", pred)
    save_csv(out / "predicted_events.csv", predicted_events)
    save_csv(out / "event_matches.csv", matching)
    save_csv(out / "cases.csv", case_rows)
    if args.stage == "pilot":
        print(pred[["cohort", "video_id", "arm", "truth_count", "predicted_count", "selected_mode"]].to_string(index=False))
        print(f"Pilot saved: {out}")
        return
    metric_rows, paired_rows = [], []
    for cohort in ("anchored49", "internal73"):
        frame = pred[pred.cohort.eq(cohort) & pred.arm.isin(ARMS)].copy()
        if cohort == "anchored49":
            frame = frame[frame.reference_status.eq("complete")]
        if len(frame) != (47 if cohort == "anchored49" else 73) * 4:
            raise ValueError("Analysis denominator changed")
        for arm in ARMS:
            group = frame[frame.arm.eq(arm)]
            score = metrics_for(group)
            events = pd.DataFrame(matching)
            if cohort == "anchored49":
                matched = events[events.arm.eq(arm)]
                tp, fp, fn = [(matched.type == t).sum() for t in ("TP", "FP", "FN")]
                score.update({"tp": int(tp), "fp": int(fp), "fn": int(fn),
                              "event_precision": tp / (tp + fp) if tp + fp else 0.,
                              "event_recall": tp / (tp + fn) if tp + fn else 0.,
                              "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
            metric_rows.append({"cohort": cohort, "arm": arm, "coverage": f"{len(group)}/{len(group)}", **score})
        for candidate, control in (("C1P0", "C0P0"), ("C0P1", "C0P0"),
                                   ("C1P1", "C0P0"), ("C1P1", "C0P1"), ("C1P1", "C1P0")):
            paired_rows.append({"cohort": cohort, "candidate": candidate, "control": control,
                                **bootstrap(frame[frame.arm.eq(candidate)], frame[frame.arm.eq(control)])})
    metrics = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    save_csv(out / "metrics.csv", metrics)
    save_csv(out / "paired_intervals.csv", paired)
    source = pred[pred.cohort.eq("source_sensitivity") & pred.reference_status.eq("sensitivity_only") & pred.truth_count.notna()].copy()
    if len(source) != 47:
        raise ValueError("Source sensitivity denominator changed")
    source_score = metrics_for(source)
    source_control = pred[pred.cohort.eq("anchored49") & pred.arm.eq("C0P0") & pred.reference_status.eq("complete")]
    source_pair = source.merge(source_control[["video_id", "predicted_count"]], on="video_id", suffixes=("", "_paper"), validate="one_to_one")
    source_summary = {"analysis_set": "R2_complete47", "source_peak_rule": "prominence=.035,width=1,plateau_size=1,distance=6",
                      "paper_peak_rule": "prominence=.05,distance=6,other_defaults",
                      "changed_count_windows": int(source_pair.predicted_count.ne(source_pair.predicted_count_paper).sum()),
                      "mean_abs_rr_formula_difference_bpm": float(abs(source.source_8p6_rr_bpm_historical - source.predicted_rr_bpm).mean()),
                      "max_abs_rr_formula_difference_bpm": float(abs(source.source_8p6_rr_bpm_historical - source.predicted_rr_bpm).max()),
                      **source_score}
    save_json(out / "source_sensitivity.json", source_summary)
    base, proposed = [metrics[(metrics.cohort == "anchored49") & (metrics.arm == arm)].iloc[0] for arm in ("C0P0", "C1P1")]
    ci = paired[(paired.cohort == "anchored49") & (paired.candidate == "C1P1") & (paired.control == "C0P0")].iloc[0]
    supported = bool(ci.delta_mae_ci95_high < 0 and proposed.exact_count >= base.exact_count and proposed.event_f1 >= base.event_f1)
    decision = {"status": "supported_on_internal_development_only" if supported else "method_gain_not_demonstrated",
                "primary_comparison": "C1P1-C0P0", "delta_mae_bpm": float(ci.delta_mae_bpm),
                "ci95": [float(ci.delta_mae_ci95_low), float(ci.delta_mae_ci95_high)],
                "exact_count_control_candidate": [int(base.exact_count), int(proposed.exact_count)],
                "event_f1_control_candidate": [float(base.event_f1), float(proposed.event_f1)],
                "limitations": ["Internal development set", "Cached BGR RF temperatures, not Chen end-to-end",
                                "C1 candidate missing-value interpolation differs from C0 fill-0.5",
                                "Single-annotator R2 reference", "No untouched external confirmation"]}
    save_json(out / "decision.json", decision)
    cases = pd.DataFrame(case_rows)
    wide = cases.pivot(index="video_id", columns="arm", values="count_error")
    failures = cases[cases.arm.eq("C1P1")].copy()
    failures["baseline_abs_error"] = wide.C0P0.abs().reindex(failures.video_id).to_numpy()
    failures["candidate_abs_error"] = failures.count_error.abs()
    failures["delta_abs_count_error"] = failures.candidate_abs_error - failures.baseline_abs_error
    failures = failures.sort_values(["delta_abs_count_error", "candidate_abs_error"], ascending=False)
    save_csv(out / "failure_cases.csv", failures)
    lines = ["# 同输入四组核证结果", "", "内部开发集；不是原作者完整端到端复现或新外测。详细规则见 PROTOCOL.md，文件哈希见 freeze_manifest.json。", "",
             "| Cohort | Arm | n | RR R² | RR MAE | Exact | Event F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        lines.append(f"| {row.cohort} | {row.arm} | {row.n} | {row.rr_r2:.4f} | {row.rr_mae_bpm:.3f} | {row.exact_count}/{row.n} | {getattr(row, 'event_f1', float('nan')):.4f} |")
    lines += ["", f"主比较 C1P1-C0P0 配对 MAE 差值 {ci.delta_mae_bpm:.3f} bpm，原视频簇 bootstrap 95% CI [{ci.delta_mae_ci95_low:.3f}, {ci.delta_mae_ci95_high:.3f}]。",
              f"预定判据裁决：{decision['status']}。", "", "R2 其余2窗未完成参考，不计主结果。B73-I仅计数，无同口径人工事件时间。", "",
              f"源码差异敏感性 (n=47)：0.035+width/plateau使{source_summary['changed_count_windows']}窗峰数变化，MAE {source_summary['rr_mae_bpm']:.3f}；8.6 fps历史公式与冻结时长公式平均绝对差 {source_summary['mean_abs_rr_formula_difference_bpm']:.3f} bpm。该组不是主对照。", "未进行真实像素短缺口重取敏感性。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(metrics[["cohort", "arm", "rr_r2", "rr_mae_bpm", "exact_count"]].to_string(index=False))
    print(decision)
    print(f"Full run saved: {out}")


if __name__ == "__main__":
    main()
