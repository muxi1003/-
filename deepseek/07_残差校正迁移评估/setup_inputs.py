# -*- coding: utf-8 -*-
"""
D009 步骤1：为"质量感知残差校正"在 49/47 窗集上搭建可运行输入目录。

背景：r20 固定（原文忠实基线）**没有保存曲线 CSV**，校正器无法为其计算曲线特征；
      消融的 8 个变体全部基于同一 r20 输入且**保存了 49 条曲线**，故取
      F1G1P1（初稿主线）与 F1G0P1（最佳配置）作为可运行基底。

产物全部写入 deepseek/ 下，不修改任何项目文件。
"""
import csv, io, os, shutil

R = r"E:\real\use_code\yoloV8"
ABL = os.path.join(R, "Experiment", "01_同口径消融", "runs", "20260909_v1")
CURVES = os.path.join(ABL, "curves", "anchored49")
PAIRED = os.path.join(ABL, "paired_predictions.csv")
R2DISP = os.path.join(R, "Experiment", "03_可靠事件参考", "submissions",
                      "20260917_r2_v2", "window_dispositions.csv")
OUTROOT = os.path.join(R, "deepseek", "07_残差校正迁移评估")

VARIANTS = ["F1G1P1", "F1G0P1"]


def load(path, encodings=("utf-8-sig", "gb18030", "utf-8")):
    last = None
    for enc in encodings:
        try:
            with io.open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f)), enc
        except Exception as e:
            last = e
    raise last


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


# ---- 统一参考：R2 47 窗 ----
disp, e1 = load(R2DISP)
truth = {}
for r in disp:
    if r.get("cohort") != "lindian49" or r.get("annotation_status") != "complete":
        continue
    c, d = fnum(r.get("manual_breath_count")), fnum(r.get("duration_seconds"))
    if c is None or d is None:
        continue
    truth[r["video_id"]] = dict(count=c, dur=d, rr=60.0 * c / d)
print("R2 lindian49 complete 窗: %d (编码 %s)" % (len(truth), e1))

# ---- 消融预测 ----
paired, e2 = load(PAIRED)
print("paired_predictions 编码: %s" % e2)

for variant in VARIANTS:
    rows = [r for r in paired
            if r.get("cohort") == "anchored49" and r.get("variant") == variant
            and r.get("video_id") in truth]
    run_dir = os.path.join(OUTROOT, "run_" + variant)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    out_rows = []
    copied = 0
    for r in sorted(rows, key=lambda x: x["video_id"]):
        vid = r["video_id"]
        t = truth[vid]
        peaks = fnum(r.get("predicted_count"))
        if peaks is None:
            continue
        # 曲线：curves/anchored49/{variant}/{vid}.csv -> {run}/{vid}/paper_repro_curve.csv
        src = os.path.join(CURVES, variant, vid + ".csv")
        if os.path.isfile(src):
            d = os.path.join(run_dir, vid)
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(src, os.path.join(d, "paper_repro_curve.csv"))
            copied += 1
        row = {
            "video_id": vid,
            "peaks": peaks,
            "rr_bpm": 60.0 * peaks / t["dur"],
            "duration_seconds": t["dur"],
            "truth_count": t["count"],
            "truth_rr": t["rr"],
            # 可用的 summary 侧特征（49 窗集本就稀少）
            "source_rejected_count": fnum(r.get("source_rejected_count")) or 0,
            "selected_fusion_mode": r.get("selected_fusion_mode") or "missing",
            "reference_reliability": r.get("reference_reliability") or "missing",
        }
        out_rows.append(row)

    summ = os.path.join(run_dir, "summary.csv")
    cols = ["video_id", "peaks", "rr_bpm", "duration_seconds", "truth_count", "truth_rr",
            "source_rejected_count", "selected_fusion_mode", "reference_reliability"]
    with io.open(summ, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print("  %-8s 样本 %d，曲线复制 %d，summary -> %s" % (variant, len(out_rows), copied, summ))
print("完成。")
