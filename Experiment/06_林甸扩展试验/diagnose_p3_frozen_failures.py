"""Bounded, read-only attribution of eight preselected P3 cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE / "20260929_p3_four_arm_v1"
OUT = HERE / "20260929_p3_failure_diagnosis_v1"
ARMS = ("C0P0", "C1P0", "C0P1", "C1P1")
CASES = (
    ("regression", "20230810T145443n170131_000_030"),
    ("regression", "20230810T164658n221284_000_030"),
    ("regression", "20230808T170359n197250_030_060"),
    ("regression", "20230808T093014n211075_000_030"),
    ("improvement", "20230810T164018n221381_000_030"),
    ("improvement_purposive", "20230810T071310n211109_000_030"),
    ("count_only", "20230809T084030n170020_030_060"),
    ("count_only", "20230809T073104n200792_060_090"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def jump_stats(table: pd.DataFrame, side: str) -> tuple[float, int]:
    x = table[f"{side}_x"].to_numpy(float)
    y = table[f"{side}_y"].to_numpy(float)
    valid = table[f"{side}_temp"].notna().to_numpy() & np.isfinite(x) & np.isfinite(y)
    pairs = valid[1:] & valid[:-1]
    jumps = np.hypot(np.diff(x), np.diff(y))[pairs]
    return (float(np.percentile(jumps, 95)), int(np.sum(jumps > 20))) if len(jumps) else (float("nan"), 0)


def local_coverage(table: pd.DataFrame, second: float, side: str) -> tuple[int, int]:
    near = table[(table.playback_seconds - second).abs() <= 0.30]
    return int(near[f"{side}_temp"].notna().sum()), len(near)


def plot_curves(video_id: str, table: pd.DataFrame, matched: pd.DataFrame, plots: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    t = table.playback_seconds.to_numpy(float)
    refs = matched.loc[matched.kind.isin(["TP", "FN"]), "reference_time"].dropna().unique()
    for ax, arm in zip(axes, ("C0P0", "C1P1")):
        curve = pd.read_csv(RUN / "curves" / arm / f"{video_id}.csv")
        if len(curve) != len(table):
            raise ValueError(f"Curve and temperature lengths differ: {video_id} {arm}")
        ax.plot(t, curve.smoothed_norm, lw=1, color="#155e75")
        ax.scatter(t[curve.is_peak.to_numpy(bool)], curve.smoothed_norm[curve.is_peak],
                   s=14, color="#dc2626", zorder=3)
        for ref in refs:
            ax.axvline(ref, color="#64748b", alpha=0.28, lw=0.7)
        ax.set_ylabel(f"{arm} normalized")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Window time (s); gray=timed reference, red=predicted peak")
    fig.suptitle(video_id, fontsize=10)
    fig.savefig(plots / f"{video_id}_curves.png", dpi=145)
    plt.close(fig)


def frame_montage(video_id: str, table: pd.DataFrame, source: Path,
                  seconds: list[float], plots: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    fig, axes = plt.subplots(1, len(seconds), figsize=(5 * len(seconds), 4.8), constrained_layout=True)
    if len(seconds) == 1:
        axes = [axes]
    try:
        for ax, second in zip(axes, seconds):
            row = table.iloc[(table.playback_seconds - second).abs().argmin()]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Frame read failed: {video_id} source_frame={row.source_frame}")
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for side, color in (("left", "#22c55e"), ("right", "#ef4444")):
                x, y = row[f"{side}_x"], row[f"{side}_y"]
                if np.isfinite(x) and np.isfinite(y):
                    ax.add_patch(plt.Circle((x, y), 20, color=color, fill=False, lw=2))
                    ax.text(x + 21, y, side[0].upper(), color=color, fontsize=12, weight="bold")
            ax.set_title(f"t={row.playback_seconds:.2f}s  source frame {int(row.source_frame)}\n"
                         f"L={'yes' if pd.notna(row.left_temp) else 'no'} / "
                         f"R={'yes' if pd.notna(row.right_temp) else 'no'}", fontsize=9)
            ax.axis("off")
        fig.savefig(plots / f"{video_id}_source_roi.png", dpi=120)
    finally:
        cap.release()
        plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUT}")
    OUT.mkdir()
    plots = OUT / "plots"
    plots.mkdir()
    preds = pd.read_csv(RUN / "predictions.csv")
    matches = pd.read_csv(RUN / "event_matches.csv")
    segs = pd.concat([
        pd.read_csv(HERE / "20260928_expansion_p3_v2" / f"annotation_batch_{batch}" / "segments.csv")
        for batch in ("A", "B")
    ], ignore_index=True).set_index("video_id")
    records: list[dict] = []
    changes: list[dict] = []
    hashes: dict[str, str] = {}
    for path in (RUN / "predictions.csv", RUN / "event_matches.csv", RUN / "manifest.json",
                 HERE / "20260929_p3_failure_diagnosis_protocol.md", Path(__file__)):
        hashes[str(path)] = sha256(path)

    for selection, video_id in CASES:
        per = preds[preds.video_id.eq(video_id)].set_index("arm")
        if set(per.index) != set(ARMS):
            raise ValueError(f"Missing arm: {video_id}")
        primary = per.loc["C0P0", "analysis_disposition"] == "complete_primary"
        if primary != (selection != "count_only"):
            raise ValueError(f"Unexpected reference disposition: {video_id}")
        temp_path = RUN / "temperatures" / f"{video_id}.csv"
        table = pd.read_csv(temp_path)
        hashes[str(temp_path)] = sha256(temp_path)
        for arm in ARMS:
            path = RUN / "curves" / arm / f"{video_id}.csv"
            hashes[str(path)] = sha256(path)
        left = table.left_temp.notna().to_numpy()
        right = table.right_temp.notna().to_numpy()
        mode = str(per.loc["C1P1", "fusion"])
        selected_valid = left if mode == "left" else right if mode == "right" else (left | right)
        ljump, llarge = jump_stats(table, "left")
        rjump, rlarge = jump_stats(table, "right")
        row: dict = {
            "selection": selection, "video_id": video_id, "primary": primary,
            "reference_count": per.loc["C0P0", "truth_count"],
            "frames": len(table), "left_valid": int(left.sum()), "right_valid": int(right.sum()),
            "both_valid": int((left & right).sum()), "any_valid": int((left | right).sum()),
            "selected_mode": mode, "selected_valid": int(selected_valid.sum()),
            "selected_longest_missing_frames": longest_run(~selected_valid),
            "left_coordinate_jump_p95_px": ljump, "right_coordinate_jump_p95_px": rjump,
            "left_consecutive_jumps_gt20px": llarge, "right_consecutive_jumps_gt20px": rlarge,
        }
        for arm in ARMS:
            p = per.loc[arm]
            row.update({f"{arm}_count": int(p.predicted_count),
                        f"{arm}_rr_abs_error": float(p.absolute_rr_error_bpm) if pd.notna(p.absolute_rr_error_bpm) else np.nan,
                        f"{arm}_tp": int(p.tp) if pd.notna(p.tp) else np.nan,
                        f"{arm}_fp": int(p.fp) if pd.notna(p.fp) else np.nan,
                        f"{arm}_fn": int(p.fn) if pd.notna(p.fn) else np.nan})
        row["joint_minus_baseline_abs_error"] = row["C1P1_rr_abs_error"] - row["C0P0_rr_abs_error"]

        m = matches[matches.video_id.eq(video_id)]
        if primary:
            control = m[m.arm.eq("C0P0") & m.kind.eq("TP")].set_index("reference_event_id")
            joint = m[m.arm.eq("C1P1") & m.kind.eq("TP")].set_index("reference_event_id")
            lost = sorted(set(control.index) - set(joint.index))
            gained = sorted(set(joint.index) - set(control.index))
            row["lost_matched_reference_events"] = len(lost)
            row["gained_matched_reference_events"] = len(gained)
            for kind, ids, source in (("lost_by_joint", lost, control), ("gained_by_joint", gained, joint)):
                for ref_id in ids:
                    ref = source.loc[ref_id]
                    sec = float(ref.reference_time)
                    lvalid, n = local_coverage(table, sec, "left")
                    rvalid, _ = local_coverage(table, sec, "right")
                    changes.append({"video_id": video_id, "change": kind,
                                    "reference_event_id": ref_id, "reference_time_seconds": sec,
                                    "selected_mode": mode, "nearby_frame_count": n,
                                    "left_valid_within_0p3s": lvalid, "right_valid_within_0p3s": rvalid})
            plot_curves(video_id, table, m, plots)
        else:
            row["lost_matched_reference_events"] = np.nan
            row["gained_matched_reference_events"] = np.nan
            plot_curves(video_id, table, m, plots)
        records.append(row)

        if video_id.endswith("n170131_000_030"):
            seconds = [1.77, 7.04, 12.57]
        elif video_id.endswith("n221284_000_030"):
            seconds = [5.45, 6.39, 8.63]
        elif selection == "count_only":
            seconds = [5.0, 15.0, 25.0]
        else:
            continue
        source = Path(segs.loc[video_id, "source_path"])
        frame_montage(video_id, table, source, seconds, plots)
        hashes[f"source_sha256_from_frozen_segments:{source}"] = str(segs.loc[video_id, "source_sha256"])

    cases = pd.DataFrame(records)
    transitions = pd.DataFrame(changes)
    cases.to_csv(OUT / "case_metrics.csv", index=False)
    transitions.to_csv(OUT / "changed_reference_events.csv", index=False)
    primary = preds[preds.analysis_disposition.eq("complete_primary")]
    errors = primary.pivot(index="video_id", columns="arm", values="absolute_rr_error_bpm")
    delta = errors.C1P1 - errors.C0P0
    top_two = cases.iloc[:2].video_id.tolist()
    summary = {
        "scope": "eight preselected cases, frozen P3 outputs; no re-scoring",
        "main_windows": len(errors),
        "main_paired_absolute_error_sum_delta_bpm": float(delta.sum()),
        "two_largest_regressions_sum_delta_bpm": float(delta.loc[top_two].sum()),
        "other_main_windows_sum_delta_bpm": float(delta.drop(top_two).sum()),
        "case_count": len(cases), "changed_timed_events": len(transitions),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "input_hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(cases[["video_id", "selected_mode", "C0P0_count", "C1P1_count",
                 "joint_minus_baseline_abs_error", "lost_matched_reference_events",
                 "gained_matched_reference_events"]].to_string(index=False))


if __name__ == "__main__":
    main()
