"""Score sealed transfer predictions against archived human TOTAL counts, not events."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import ROOT, REFERENCE, measurements, read_csv, sha256, write_csv, write_json, write_text
from run_frozen_external import verify_lock, verify_checkpoint
from frozen_rr_evaluation import validate_predictions


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def join_reference(pred, windows, labels):
    if labels.window_id.duplicated().any() or set(labels.window_id) != set(windows.window_id):
        raise ValueError("Reference cohort mismatch")
    merged = windows.merge(pred, on="window_id", validate="one_to_one", suffixes=("", "_prediction"))
    merged = merged.merge(labels, on="window_id", validate="one_to_one", suffixes=("", "_reference"))
    for row in merged.itertuples():
        if (Path(row.source_path) != Path(row.source_path_reference)
                or abs(float(row.start_seconds)-float(row.source_start_seconds)) > 1e-6
                or abs(float(row.duration_seconds)-float(row.duration_seconds_reference)) > 1e-6
                or row.annotation_round != "R1"):
            raise ValueError("Reference/source time binding mismatch")
        if row.annotation_status not in {"complete", "unobservable"}:
            raise ValueError("Unexpected submitted status; requires explicit new reference version")
        if row.annotation_status == "complete":
            count = float(row.manual_breath_count)
            if not np.isfinite(count) or count < 0 or not count.is_integer():
                raise ValueError("Invalid reference count")
            if str(row.predictions_hidden).lower() != "true" or not row.annotator.strip():
                raise ValueError("Missing blinded-observer declaration")
        elif str(row.manual_breath_count).strip():
            raise ValueError("Unobservable reference must not be converted to zero")
    return merged


def cluster_intervals(primary, draws=2000, seed=20260914):
    if primary.empty:
        return {"clusters": 0, "intervals": {}}
    groups = [g for _, g in primary.groupby("verified_cow_id", sort=True)]
    rng = np.random.default_rng(seed)
    keys = ["rr_r2", "rr_mae_bpm", "rr_rmse_bpm", "count_mae", "mean_count_accuracy_percent", "bias_bpm"]
    values = {k: [] for k in keys}
    for _ in range(draws):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        metrics = measurements(sample)
        for k in keys:
            if np.isfinite(metrics[k]):
                values[k].append(metrics[k])
    return {"clusters": len(groups), "draws": draws, "seed": seed,
            "method": "percentile_bootstrap_cow_clusters_keep_all_windows_per_sampled_cow",
            "scope": "paired_complete_and_ok_set_conditional_on_coverage_not_missing_outcome_imputation",
            "intervals": {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "finite_draws": len(v)} if v else None for k, v in values.items()}}


def score(run, out):
    seal_path = run / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate_path = run / "engineering_verification.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_ENGINEERING_GATE_NOT_ACCURACY" or gate.get("prediction_seal_sha256") != sha256(seal_path):
        raise ValueError("Missing/changed additional engineering gate")
    lock, windows = verify_lock(run)
    if sha256(run / "protocol_lock.json") != seal["protocol_sha256"] or sha256(run / "prediction_windows.csv") != seal["predictions_sha256"]:
        raise ValueError("Predictions or protocol changed after sealing")
    pred = read_csv(run / "prediction_windows.csv")
    validate_predictions(pred, windows, lock["method_manifest_sha256"])
    for row in windows.itertuples():
        directory = run / "windows" / row.window_id
        if sha256(directory / "checkpoint.json") != seal["checkpoints"][row.window_id]:
            raise ValueError("Checkpoint seal changed")
        checkpoint = verify_checkpoint(directory)
        saved = pred.set_index("window_id").loc[row.window_id]
        count = "" if checkpoint["predicted_count"] is None else float(checkpoint["predicted_count"])
        if checkpoint["prediction_status"] != saved.prediction_status or (count == "" and saved.predicted_count != "") or (count != "" and count != float(saved.predicted_count)):
            raise ValueError("Prediction does not match per-window checkpoint")
    # References are loaded only after verifying the sealed prediction evidence.
    submission = REFERENCE / "submissions/20260911_v2"
    reference_path = submission / "jiufu271/count_reference_only.csv"
    manifest = read_csv(submission / "delivery_artifact_manifest.csv")
    expected = manifest.loc[manifest.path.map(lambda p: ROOT / p).eq(reference_path), "sha256"]
    if len(expected) != 1 or sha256(reference_path) != expected.iloc[0]:
        raise ValueError("Archived reference hash mismatch")
    merged = join_reference(pred, windows, read_csv(reference_path))
    merged["primary_included"] = merged.annotation_status.eq("complete") & merged.prediction_status.eq("ok")
    merged["exclusion_reason"] = np.select([merged.annotation_status.ne("complete"), merged.prediction_status.ne("ok")], ["human_unobservable", "algorithm_abstention"], default="included")
    primary = merged.loc[merged.primary_included].copy()
    primary["truth_count"] = primary.manual_breath_count.astype(float)
    primary["predicted_count"] = primary.predicted_count.astype(float)
    primary["truth_rr_bpm"] = 60*primary.truth_count/primary.duration_seconds.astype(float)
    primary["predicted_rr_bpm"] = 60*primary.predicted_count/primary.duration_seconds.astype(float)
    metrics = measurements(primary) if len(primary) else None
    intervals = cluster_intervals(primary)
    coverage = merged.groupby(["annotation_status", "prediction_status", "reason"], dropna=False).size().reset_index(name="n")
    report = clean({"status": "SCORED_COUNT_REFERENCE_ONLY" if metrics else "NO_PAIRED_COUNT_OUTPUTS",
                   "n_released": len(merged), "n_cows": merged.verified_cow_id.nunique(),
                   "n_complete_reference": int(merged.annotation_status.eq("complete").sum()),
                   "n_unobservable_reference": int(merged.annotation_status.eq("unobservable").sum()),
                   "n_algorithm_ok": int(merged.prediction_status.eq("ok").sum()), "n_primary": len(primary),
                   "output_coverage_all271": float(merged.prediction_status.eq("ok").mean()),
                   "paired_coverage_complete_reference": len(primary)/int(merged.annotation_status.eq("complete").sum()),
                   "metrics": metrics, "cluster_ci95": intervals,
                   "prediction_seal_sha256": sha256(seal_path), "reference_sha256": sha256(reference_path),
                   "event_metrics": None, "event_reference_status": "NOT_AVAILABLE_NO_EVENT_TIMES",
                   "blinding": "Jiufu_single_observer_counts_hidden_from_algorithm_predictions_per_user_declaration; not_double_observer",
                   "limitations": [lock["calibration_scope"], lock["spatial_limit"], lock["training_scope"],
                                   "same_cow_repeated_windows_not_independent", "no_parameter_updates_from_this_holdout",
                                   "algorithm_rejection_and_human_unobservability_not_zero_counts", "no_claim_all271_accuracy"]})
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "all_271_windows.csv", merged)
    write_csv(out / "paired_count_results.csv", primary)
    write_csv(out / "coverage.csv", coverage)
    write_json(out / "metrics.json", report)
    write_json(out / "scoring_provenance.json", {"script_sha256": sha256(Path(__file__)), "predictions_presealed": True,
               "reference_path": str(reference_path), "reference_hash": sha256(reference_path), "test_used_for_tuning": False})
    lines = ["# 久福冻结外测：人工总次数参考", "", "仅评估一次锁定的迁移方法，不证明绝对温度标定有效，不是事件F1。", "",
             f"全部 {len(merged)} 窗；人工可计数 {report['n_complete_reference']} 窗；算法输出 {report['n_algorithm_ok']} 窗；两者交集 {len(primary)} 窗。", ""]
    if metrics:
        lines += ["| 指标 | 结果 | 按牛聚类95%区间 |", "|---|---:|---| "]
        for k in ["rr_r2", "rr_mae_bpm", "rr_rmse_bpm", "count_mae", "mean_count_accuracy_percent"]:
            ci = intervals["intervals"][k]
            lines.append(f"| {k} | {metrics[k]:.6f} | {ci['low']:.6f} 至 {ci['high']:.6f} |")
    lines += ["", "不能将交集性能写成全部271窗的性能。114个人工不可观察窗口不填0；拒绝输出不偷偷删除。所有排除见all_271_windows.csv。", "", "结果无论好坏均保留，不根据久福结果调参后继续称独立测试。单人总次数不是传感器金标准；缺少事件时刻时不计算Precision/Recall/F1。"]
    write_text(out / "RESULTS.md", "\n".join(lines)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    score(args.run, args.out)
