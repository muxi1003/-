# -*- coding: utf-8 -*-
"""
D006：两件事的核对脚本
  任务1：查明 49 窗表的 short_clip_reference_rr_bpm_not_window_truth 列
         与 73 表 temperature_curves.csv 的 rr 列 为何不一致
  任务2：把 r20 固定（原文方法候选基线）与 F1G1P1 / F1G0P1（本文方法）
         放在**统一的 R2 47 窗参考**上重算，并做配对 bootstrap

只读：不写入任何项目数据文件。
"""
import csv, io, os, random, math, statistics as st

ROOT = r"E:\real\use_code\yoloV8"
AL = os.path.join(ROOT, "Dataset_new", "72video", "al_images")
ASSETS = os.path.join(AL, "paper_repro_quality_residual_paper_assets")
ABL = os.path.join(ROOT, "Experiment", "01_同口径消融", "runs", "20260909_v1")
R2DIR = os.path.join(ROOT, "Experiment", "03_可靠事件参考", "submissions", "20260917_r2_v2")

F_WIN49 = os.path.join(ASSETS, "lindian_anchored30_quality_gated_annotation_template.csv")
F_73    = os.path.join(ROOT, "Dataset_new", "72video", "video", "temperature_curves.csv")
F_R20   = os.path.join(ASSETS, "lindian_adaptive_roi_predictions.csv")
F_ABL   = os.path.join(ABL, "paired_predictions.csv")
F_R2    = os.path.join(R2DIR, "window_dispositions.csv")


def load(path, encodings=("utf-8-sig", "gb18030", "utf-8")):
    last = None
    for enc in encodings:
        try:
            with io.open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f)), enc
        except Exception as e:
            last = e
    raise last


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def r2_score(pred, ref):
    m = st.mean(ref)
    ss_res = sum((p - r) ** 2 for p, r in zip(pred, ref))
    ss_tot = sum((r - m) ** 2 for r in ref)
    return 1.0 - ss_res / ss_tot


def mae(pred, ref):
    return st.mean(abs(p - r) for p, r in zip(pred, ref))


def rmse(pred, ref):
    return math.sqrt(st.mean((p - r) ** 2 for p, r in zip(pred, ref)))


print("=" * 78)
print("TASK 1  短片参考两列的差异")
print("=" * 78)
win49, e1 = load(F_WIN49)
t73, e2 = load(F_73)
print("49窗表编码=%s 行=%d | 73表编码=%s 行=%d" % (e1, len(win49), e2, len(t73)))

rr73 = {r["video_id"]: f(r["rr"]) for r in t73}
cnt73 = {r["video_id"]: f(r["breath_count"]) for r in t73}
dur73 = {r["video_id"]: f(r["duration_seconds"]) for r in t73}

print("\n73 表内部一致性: rr == round(60*breath_count/duration_seconds) ?")
bad = 0
for r in t73:
    calc = 60.0 * f(r["breath_count"]) / f(r["duration_seconds"])
    if abs(round(calc) - f(r["rr"])) > 1e-9:
        bad += 1
        if bad <= 5:
            print("   不一致 %s: rr=%s 计算=%.4f" % (r["video_id"], r["rr"], calc))
print("   不一致行数: %d / %d" % (bad, len(t73)))

print("\n逐行对比（completed 窗）: B=49窗表短片列, C=73表rr列")
print("%-11s %6s %6s %6s   %8s %8s %8s" % ("video_id", "B", "C", "B-C", "c73", "dur73", "60c/dur"))
diff_rows = []
for r in win49:
    if r["annotation_status"] != "completed":
        continue
    vid = r["video_id"]
    B = f(r["short_clip_reference_rr_bpm_not_window_truth"])
    C = rr73.get(vid)
    if B is None or C is None:
        continue
    c73, d73 = cnt73[vid], dur73[vid]
    calc = 60.0 * c73 / d73
    if abs(B - C) > 1e-9:
        diff_rows.append((vid, B, C, B - C, c73, d73, calc))
print("   有差异的行数: %d" % len(diff_rows))
for vid, B, C, d, c73, d73, calc in sorted(diff_rows, key=lambda x: -abs(x[3])):
    print("%-11s %6.1f %6.1f %6.1f   %8.0f %8.1f %8.3f" % (vid, B, C, d, c73, d73, calc))

print()
print("=" * 78)
print("TASK 2  统一 R2 47 窗参考上的配对对照")
print("=" * 78)
r2rows, e3 = load(F_R2)
print("window_dispositions 编码=%s 行=%d" % (e3, len(r2rows)))
base = {}
for r in r2rows:
    if r.get("cohort") != "lindian49":
        continue
    if r.get("annotation_status") != "complete":
        continue
    mc, dur = f(r["manual_breath_count"]), f(r["duration_seconds"])
    if mc is None or dur is None:
        continue
    base[r["video_id"]] = dict(window_id=r["window_id"], count=mc, dur=dur,
                               rr=60.0 * mc / dur)
print("lindian49 complete 窗数: %d" % len(base))

# --- r20 固定 ---
r20rows, e4 = load(F_R20)
r20 = {}
for r in r20rows:
    if r["policy_id"] != "fixed_radius20_reextracted":
        continue
    vid = r["video_id"]
    if vid not in base:
        continue
    pc = f(r["predicted_breath_count"])
    r20[vid] = dict(count=pc, rr=60.0 * pc / base[vid]["dur"])
print("r20 固定 与 R2 47窗 交集: %d" % len(r20))

# --- 消融各配置 ---
abl, e5 = load(F_ABL)
variants = {}
for r in abl:
    if r.get("cohort") != "anchored49":
        continue
    v = r["variant"]
    vid = r["video_id"]
    if vid not in base:
        continue
    pc = f(r["predicted_count"])
    variants.setdefault(v, {})[vid] = dict(count=pc, rr=60.0 * pc / base[vid]["dur"])
print("消融配置可用: %s" % sorted(variants))
for v in sorted(variants):
    print("   %s 与 R2 交集: %d" % (v, len(variants[v])))


def evaluate(pred_map, ids):
    pred = [pred_map[i]["rr"] for i in ids]
    ref = [base[i]["rr"] for i in ids]
    ex = sum(1 for i in ids if abs(pred_map[i]["count"] - base[i]["count"]) < 1e-9)
    return dict(n=len(ids), r2=r2_score(pred, ref), mae=mae(pred, ref),
                rmse=rmse(pred, ref), exact=ex)


def paired_bootstrap(predA, predB, ids, draws=2000, seed=20260922):
    """返回 ΔMAE(A-B) 的 95% 区间；每个窗=一头不同牛，故按窗重采样"""
    rng = random.Random(seed)
    d = []
    n = len(ids)
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        a = st.mean(abs(predA[ids[i]]["rr"] - base[ids[i]]["rr"]) for i in idx)
        b = st.mean(abs(predB[ids[i]]["rr"] - base[ids[i]]["rr"]) for i in idx)
        d.append(a - b)
    d.sort()
    return st.mean(d), d[int(0.025 * draws)], d[int(0.975 * draws) - 1]


ids = sorted(r20)
rows = [("r20 固定（原文候选基线）", evaluate(r20, ids))]
for v in ["F1G1P1", "F1G0P1", "F1G1P0", "F1G0P0", "F0G0P0", "F0G0P1"]:
    if v in variants:
        sub = [i for i in ids if i in variants[v]]
        if len(sub) == len(ids):
            rows.append((v, evaluate(variants[v], ids)))
        else:
            rows.append((v + " (交集%d)" % len(sub), evaluate(variants[v], sub)))

print("\n统一参考 = R2 20260917_v2 的 lindian49 complete 窗 (%d 窗)" % len(ids))
print("%-26s %4s %9s %9s %9s %8s" % ("方法", "n", "R2", "MAE", "RMSE", "完全计数"))
for name, m in rows:
    print("%-26s %4d %9.6f %9.4f %9.4f %8s" % (name, m["n"], m["r2"], m["mae"], m["rmse"], "%d/%d" % (m["exact"], m["n"])))

print("\n配对 bootstrap（2000 次，按窗重采样，seed=20260922）")
print("ΔMAE = MAE(本文方法) - MAE(r20固定)；负值=本文方法更好")
for v in ["F1G1P1", "F1G0P1", "F1G1P0", "F1G0P0"]:
    if v not in variants:
        continue
    sub = [i for i in ids if i in variants[v]]
    if len(sub) != len(ids):
        continue
    m, lo, hi = paired_bootstrap(variants[v], r20, ids)
    verdict = "区间跨0 -> 无显著差异" if (lo < 0 < hi) else ("本文更好" if hi < 0 else "基线更好")
    print("   %-8s ΔMAE=%+8.4f  95%%CI=[%+8.4f, %+8.4f]  %s" % (v, m, lo, hi, verdict))
