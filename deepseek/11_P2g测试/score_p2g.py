# -*- coding: utf-8 -*-
"""
D012  P2g 评分：把冻结的 7 通道相对伪彩集合接到当前 271 窗 R2 参考上。

做法：
  - 复用项目模块 build_rr_calibration_free_thermal_index 的 fused_curve / estimate_curve
    / selected_member_predictions，**通道与门限一律用内部冻结值，不重排、不调参**；
  - 只把 extract_video_signals 换成读取 deepseek/11_P2g测试/signals/ 下我提取的通道表
    （选项 A：ROI 与当前外测一致，仅替换信号定义）；
  - 参考用 R2 20260918 状态修订版（216 完整参考 / 173 计数配对窗）；
  - 配对 bootstrap 按牛号聚类。

只读项目文件；产物写 deepseek/11_P2g测试/。
"""
from __future__ import annotations
import io, os, sys, csv, math, random
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(r"E:\real\use_code\yoloV8")
sys.path.insert(0, str(REPO / "scripts"))
import build_rr_calibration_free_thermal_index as p2g   # noqa: E402

SIGDIR = REPO / "deepseek" / "11_P2g测试" / "signals"
OUT = REPO / "deepseek" / "11_P2g测试"
RELEASE = REPO / "Experiment" / "02_冻结独立测试" / "release_20260909_v1" / "test_windows.csv"
WINROOT = REPO / "Experiment" / "02_冻结独立测试" / "external_transfer" / "20260914_v2" / "windows"
R2DISP = (REPO / "Experiment" / "03_可靠事件参考" / "submissions"
          / "20260918_r2_status_v1" / "window_dispositions.csv")
RF_METRICS = (REPO / "Experiment" / "02_冻结独立测试" / "event_evaluation"
              / "20260918_r2_status_v1")
INTERNAL_SUMMARY = (REPO / "Dataset_new" / "72video" / "al_images" / "paper_repro_summary.csv")

# 内部冻结的 7 个成员与显著度（来自 paper_p2g_..._provisional_metrics.csv）
MEMBERS_TEXT = ("gray:direct;lab_b:direct;red_minus_blue:inverted;blue:direct;"
                "lab_b:inverted;lab_a:inverted;red_minus_blue:direct")
PROMINENCE = 0.05
FPS = 8.7


def load_csv_any(path):
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            with io.open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    raise IOError(path)


def r2_score(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[m], ref[m]
    return 1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()


def mae(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    return float(np.abs(pred[m] - ref[m]).mean())


def main():
    # ---- 参考：R2 20260918 ----
    disp = load_csv_any(R2DISP)
    truth = {}
    for r in disp:
        if r.get("cohort") != "jiufu271":
            continue
        if r.get("annotation_status") != "complete":
            continue
        try:
            c, d = float(r["manual_breath_count"]), float(r["duration_seconds"])
        except Exception:
            continue
        truth[r["window_id"]] = dict(count=c, dur=d, rr=60.0 * c / d,
                                     cow=str(r.get("verified_cow_id") or r.get("video_id")))
    print("R2 久福完整参考窗: %d" % len(truth))

    # ---- 待评窗口：有已提取信号的 ----
    sig = sorted(p.stem.replace("_calibration_free_signals", "") for p in SIGDIR.glob("*_calibration_free_signals.csv"))
    ids = [w for w in sig if w in truth]
    print("有信号且进入完整参考的窗: %d" % len(ids))
    if not ids:
        print("尚无可用信号，退出。")
        return

    # ---- 配置：沿用内部派生的冻结配置 ----
    internal_summary = pd.read_csv(INTERNAL_SUMMARY)
    config = p2g.derive_config(internal_summary, FPS)
    config = p2g.replace(config, base_prominence=PROMINENCE)
    members = [tuple(m.split(":")) for m in MEMBERS_TEXT.split(";")]
    print("成员 %d 个: %s" % (len(members), MEMBERS_TEXT))

    # ---- 把信号源换成我提取的通道表 ----
    def patched_extract(video_dir, prefix, radius, overwrite):
        wid = Path(video_dir).name
        return pd.read_csv(SIGDIR / (wid + "_calibration_free_signals.csv"))

    p2g.extract_video_signals = patched_extract

    summary = pd.DataFrame({"video_id": ids})
    preds = p2g.selected_member_predictions(SIGDIR, "paper_repro", summary, config,
                                            members, 20, "external_p2g")
    preds = preds[["video_id", "rr_bpm", "selected_members"]].copy()
    preds["window_id"] = preds["video_id"]
    preds["truth_rr"] = [truth[w]["rr"] for w in preds["video_id"]]
    preds["truth_count"] = [truth[w]["count"] for w in preds["video_id"]]
    preds["cow"] = [truth[w]["cow"] for w in preds["video_id"]]
    preds.to_csv(OUT / "p2g_predictions_on_271.csv", index=False)

    # ---- 与 RF 代理（E055）在同一配对集上比较 ----
    rf_path = WINROOT / ".." / ".." / "event_evaluation" / "20260918_r2_status_v1"
    rf_pred = None
    for cand in [rf_path / "paired_counts.csv", rf_path / "rr_metrics.csv"]:
        if cand.exists():
            rf_pred = pd.read_csv(cand)
            print("RF 预测表:", cand.name, list(rf_pred.columns)[:10])
            break

    print()
    print("=== P2g 在 %d 个完整参考窗上的结果（自评，未配对）===" % len(preds))
    print("  RR R2 = %.6f   MAE = %.4f" % (r2_score(preds["rr_bpm"], preds["truth_rr"]),
                                          mae(preds["rr_bpm"], preds["truth_rr"])))
    print("  非有限预测数: %d" % (~np.isfinite(preds["rr_bpm"].astype(float))).sum())


if __name__ == "__main__":
    main()
