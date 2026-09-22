"""Run a frozen 2x2x2 internal ablation on paired temperature inputs."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def config_for(rr, cohort: str, fusion: int, gap: int, peak: int):
    config = replace(rr.fast_fusion_quality_config(True),
                     fusion_mode="adaptive" if fusion else "max", peak_distance=5 if peak else 6,
                     peak_prominence=.035 if peak else .05, adaptive_peak_prominence=bool(peak),
                     merge_shallow_peaks=bool(peak), adaptive_peak_retuning=bool(peak and cohort == "internal73"),
                     short_sparse_edge_completion=bool(peak and cohort == "internal73"),
                     source_peak_gate=bool(peak and cohort == "anchored49"),
                     fusion_rescue_outlier_threshold=3 if fusion and cohort == "anchored49" else 0,
                     min_rr_bpm=40. if cohort == "internal73" else 20.,
                     max_rr_bpm=90. if cohort == "internal73" else 100.,
                     truth_csv=None, use_truth_duration_for_rr=False, limit_to_truth_duration=False,
                     infer_missing_nostril=False, infer_global_offset=False, infer_low_confidence_nostril=False,
                     track_missing=bool(gap), output_prefix="experiment_only", overwrite=False)
    return config


def historical_replay_config(rr, row: pd.Series, cohort: str):
    config = config_for(rr, cohort, 1, 0, 1)
    return replace(config, fusion_mode=str(row.selected_fusion_mode),
                   peak_distance=int(row.peak_distance), peak_prominence=float(row.selected_peak_prominence),
                   adaptive_peak_prominence=False, adaptive_peak_retuning=False,
                   short_sparse_edge_completion=False, smooth_window=int(row.smooth_window),
                   merge_shallow_peaks=is_true(row.merge_shallow_peaks),
                   source_peak_gate=is_true(row.get("source_peak_gate", False)),
                   edge_peak_completion=str(row.edge_peak_completion) if int(row.edge_peak_added) else "none",
                   edge_peak_min_gap=int(row.edge_peak_min_gap), edge_peak_min_relief=float(row.edge_peak_min_relief))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run-id", default=stamp())
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unsafe run id")
    if args.bootstrap_draws < 100:
        raise ValueError("At least 100 bootstrap draws required")
    package = json.loads(args.package.read_text(encoding="utf-8"))
    frozen = Path(package["method_snapshot"])
    manifest = json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(frozen / item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Method snapshot integrity failure")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    input_root = Path(package["matched_inputs"])
    windows = read_csv(input_root / "matched_windows.csv")
    for cohort, n in (("internal73", 73), ("anchored49", 49)):
        selected = windows[windows.cohort == cohort]
        if len(selected) != n or selected.video_id.duplicated().any():
            raise ValueError(f"Invalid {cohort} matched cohort")
    out = ABLATION / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    model = joblib.load(frozen / "weights" / "clf_model_RGB_20240906.pkl")
    variants = list(itertools.product((0, 1), repeat=3))
    configurations = {}
    for cohort in windows.cohort.unique():
        for f, g, p in variants:
            name = f"F{f}G{g}P{p}"
            configurations[f"{cohort}/{name}"] = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config_for(rr, cohort, f, g, p)).items()}
    write_json(out / "run_config_before_results.json", {
        "started_at": datetime.now().astimezone().isoformat(), "scope": "internal_development_ablation_only",
        "input_manifest_sha256": sha256(input_root / "matched_windows.csv"),
        "method_freeze_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "factors": {"F": "whole_curve_quality_selection", "G": "max3frame_anchor_constrained_ROI_remeasurement", "P": "existing_cohort_peak_rule_package"},
        "shared_conditions": {"radius": 20, "fps": 8.7, "normalization": "per_side_minmax", "smoother": "MAF3",
                              "legacy_curve_interpolation": "same_existing_interpolation_all_arms",
                              "roi_inferred_values": "retained_as_shared_cached_input_except_tracked_sources"},
        "rr_reference": "60*manual_count/frozen_duration; legacy rounded RR kept separately",
        "configurations": configurations})
    predictions, repairs, replay = [], [], []
    s73 = pd.read_csv(input_root / "historical_sources" / "paper_repro_summary.csv", dtype={"video_id": str}).set_index("video_id")
    s49 = pd.read_csv(input_root / "historical_sources" / "lindian_anchored30_motion_robust_rr_predictions.csv", dtype={"video_id": str}).set_index("video_id")
    for i, row in enumerate(windows.itertuples()):
        source_path = Path(row.input_temperature_csv)
        if sha256(source_path) != row.input_sha256:
            raise ValueError(f"Input changed: {source_path}")
        table = pd.read_csv(source_path)
        original_row = (s73 if row.cohort == "internal73" else s49).loc[row.video_id]
        historical_table = pd.read_csv(Path(original_row.temperature_csv)).iloc[:int(row.frames)].copy()
        replay_config = historical_replay_config(rr, original_row, row.cohort)
        _, hist_summary = rr.fuse_temperature_curve(historical_table, replay_config, truth_row=None)
        expected = int(row.historical_predicted_count)
        replay.append({"cohort": row.cohort, "video_id": row.video_id, "saved_count": expected,
                       "replay_count": int(hist_summary["peaks"]), "matches": int(hist_summary["peaks"]) == expected,
                       "scope": "historical_per_clip_config_replay_not_transferable_test_method"})
        # Remove prior tracked samples so the G contrast actually toggles pixel remeasurement.
        for side in ("left", "right"):
            tracked = table[f"{side}_source"].eq("tracked")
            table.loc[tracked, f"{side}_temp"] = np.nan
        repaired = table.copy()
        images = [Path(row.image_dir) / n for n in table.frame_name]
        for side in ("left", "right"):
            missing_before = repaired[f"{side}_temp"].isna()
            count = rr.repair_short_missing_runs(repaired, images, model, side, config_for(rr, row.cohort, 0, 1, 0))
            for frame in np.flatnonzero((missing_before & repaired[f"{side}_temp"].notna()).to_numpy()):
                repairs.append({"cohort": row.cohort, "video_id": row.video_id, "side": side, "frame_index": int(frame),
                                "temperature": float(repaired.loc[frame, f"{side}_temp"]), "image_path": str(images[frame]),
                                "rule": "internal_gap<=3;direct_anchors;shift<=20px"})
        for f, g, p in variants:
            name = f"F{f}G{g}P{p}"
            config = config_for(rr, row.cohort, f, g, p)
            curve, summary = rr.fuse_temperature_curve(repaired if g else table, config, truth_row=None)
            predicted = int(summary["peaks"])
            peak_frames = np.flatnonzero(curve.is_peak.to_numpy(bool))
            duration = float(row.duration_seconds)
            record = {"cohort": row.cohort, "video_id": row.video_id, "variant": name,
                      "fusion_factor": f, "gap_factor": g, "peak_factor": p,
                      "predicted_count": predicted, "predicted_rr_bpm": predicted * 60 / duration,
                      "truth_count": int(row.truth_count), "truth_rr_bpm": float(row.truth_rr_bpm),
                      "legacy_truth_rr_bpm": float(row.legacy_truth_rr_bpm), "duration_seconds": duration,
                      "include_primary": is_true(row.include_primary), "reference_reliability": row.reference_reliability,
                      "cluster_id": row.cluster_id, "selected_fusion_mode": summary["selected_fusion_mode"],
                      "peak_frames_zero_based": ";".join(map(str, peak_frames)),
                      "peak_times_seconds_nominal": ";".join(f"{v/8.7:.6f}" for v in peak_frames),
                      "event_time_status": "nominal_8.7fps_requires_frame_PTS_alignment_for_reference",
                      "source_rejected_count": summary.get("source_rejected_peak_count", 0)}
            predictions.append(record)
            columns = [c for c in ["frame_name", "frame_index", "left_temp", "right_temp", "left_norm", "right_norm", "fused_norm", "smoothed_norm", "is_peak", "source_peak_rejected", "fused_repaired"] if c in curve]
            write_csv(out / "curves" / row.cohort / name / f"{row.video_id}.csv", curve[columns])
        if (i + 1) % 10 == 0:
            print(f"Ablation: {i + 1}/{len(windows)} windows, {len(predictions)} predictions", flush=True)
    pred = pd.DataFrame(predictions)
    write_csv(out / "paired_predictions.csv", pred)
    write_csv(out / "roi_remeasurement_events.csv", repairs, columns=["cohort", "video_id", "side", "frame_index", "temperature", "image_path", "rule"])
    write_csv(out / "historical_replay_audit.csv", replay)
    metric_rows, paired_rows, contrasts = [], [], []
    for cohort, data in pred.groupby("cohort", sort=True):
        analyses = {"all_numeric": data}
        if cohort == "anchored49":
            analyses["primary_completed"] = data[data.include_primary]
        for analysis, subset in analyses.items():
            base = subset[subset.variant == "F0G0P0"].set_index("video_id")
            for variant, group in subset.groupby("variant", sort=True):
                metric_rows.append({"cohort": cohort, "analysis_set": analysis, "variant": variant, **measurements(group)})
                paired = group.merge(base[["predicted_rr_bpm", "predicted_count"]], left_on="video_id", right_index=True, suffixes=("", "_control"), validate="one_to_one")
                paired["paired_abs_error_delta"] = abs(paired.predicted_rr_bpm - paired.truth_rr_bpm) - abs(paired.predicted_rr_bpm_control - paired.truth_rr_bpm)
                stats = paired_cluster_bootstrap(paired, args.bootstrap_draws)
                paired_rows.append({"cohort": cohort, "analysis_set": analysis, "variant": variant, "control": "F0G0P0", **stats,
                                    "evidence_status": "exploratory_source_cluster_CI_not_independent_test"})
            # Paired factorial contrasts report each module in every fixed background.
            for factor_index, factor in enumerate("FGP"):
                for background in itertools.product((0, 1), repeat=2):
                    off = list(background)
                    off.insert(factor_index, 0)
                    on = off.copy()
                    on[factor_index] = 1
                    a, b = [f"F{x[0]}G{x[1]}P{x[2]}" for x in (off, on)]
                    ga, gb = subset[subset.variant == a].set_index("video_id"), subset[subset.variant == b].copy()
                    pair = gb.merge(ga[["predicted_rr_bpm"]], left_on="video_id", right_index=True, suffixes=("", "_off"), validate="one_to_one")
                    pair["paired_abs_error_delta"] = abs(pair.predicted_rr_bpm - pair.truth_rr_bpm) - abs(pair.predicted_rr_bpm_off - pair.truth_rr_bpm)
                    contrasts.append({"cohort": cohort, "analysis_set": analysis, "factor": factor, "off": a, "on": b, **paired_cluster_bootstrap(pair, args.bootstrap_draws)})
    metrics = pd.DataFrame(metric_rows)
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "paired_vs_control.csv", paired_rows)
    write_csv(out / "factorial_contrasts.csv", contrasts)
    for (cohort, analysis), group in metrics.groupby(["cohort", "analysis_set"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        colors = ["#087e8b" if n != "F1G1P1" else "#ea7b26" for n in group.variant]
        axes[0].bar(group.variant, group.rr_mae_bpm, color=colors)
        axes[0].set_ylabel("RR MAE (breaths/min)")
        axes[1].bar(group.variant, group.rr_r2, color=colors)
        axes[1].set_ylabel("R-squared")
        for ax in axes:
            ax.tick_params(axis="x", rotation=40)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(f"{cohort} / {analysis} / matched radius-20 inputs")
        fig.savefig(out / f"ablation_{cohort}_{analysis}.png", dpi=180)
        fig.savefig(out / f"ablation_{cohort}_{analysis}.pdf")
        plt.close(fig)
    replay_df = pd.DataFrame(replay)
    report = ["# 同口径消融运行结果", "", "状态：已运行的内部开发消融；不是独立测试结果。", "",
              "F=整条曲线质量选择；G=短内部缺口ROI像素重取；P=现有峰值规则包。所有组使用相同输入、MAF3、20像素ROI与窗口时长。",
              "", "73短片与49固定窗分别分析；49另报告39条completed。参考RR统一用人工次数和冻结时长换算，因此不直接复用旧73表中四舍五入的RR。",
              "", "此矩阵复用了已保存YOLO定位/温度输入，不能宣称已完成YOLOv8与YOLO11端到端网络对照。49输入取半径表r20，不能把与旧自适应ROI基线的差值归因于单一后处理模块。",
              "", "旧插值步骤在各组保持一致；G只代表真实图像上的短缺口重取温度，不代表已禁止整条曲线中的长缺口插值。73的P含开发集专用dense-peak目标10规则，禁止直接迁移为独立测试方法。",
              "", f"历史逐视频配置回放匹配：{int(replay_df.matches.sum())}/{len(replay_df)}。这是历史可复现性检查，非可迁移参数的证据。",
              "", f"ROI重取帧侧数：{len(repairs)}；涉及视频：{len({(r['cohort'], r['video_id']) for r in repairs})}。", "",
              "| 分析集 | 方法 | n | R² | MAE | RMSE | 完全计数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        report.append(f"| {row.cohort}/{row.analysis_set} | {row.variant} | {row.n} | {row.rr_r2:.6f} | {row.rr_mae_bpm:.6f} | {row.rr_rmse_bpm:.6f} | {row.exact_count}/{row.n} |")
    report += ["", "配对置信区间按原长视频聚类，尚未核实为真实牛身份；同一开发集的多重比较只作探索性证据。delta<0表示MAE下降；不能仅按最大R²自动更新默认流程。", "",
               "下一步：补YOLOv8/YOLO11同训练划分的网络对照、参考事件时间与独立测试。当前源温度中的推断来源和长插值行为也需在正式方法中如实说明。"]
    write_text(out / "RESULTS.md", "\n".join(report) + "\n")
    print(metrics[["cohort", "analysis_set", "variant", "rr_r2", "rr_mae_bpm"]].to_string(index=False), flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
