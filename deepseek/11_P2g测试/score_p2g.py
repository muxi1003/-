# -*- coding: utf-8 -*-
"""
D013  P2g 评分（终版）：把内部冻结的 7 通道相对伪彩集合，放到当前 271 窗 R2 参考上，
与 RF 绝对温度代理（E055）在同一批 173 配对窗上做配对对照。

规则全部冻结，不重排通道、不调 prominence、不按外部结果选参。
只读项目文件；产物写 deepseek/11_P2g测试/。
"""
from __future__ import annotations
import sys, math, random
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(r"E:\real\use_code\yoloV8")
sys.path.insert(0, str(REPO / "scripts"))
import build_rr_calibration_free_thermal_index as p2g   # noqa: E402

SIGDIR = REPO / "deepseek" / "11_P2g测试" / "signals"
OUT = REPO / "deepseek" / "11_P2g测试"
PAIRED = (REPO / "Experiment" / "02_冻结独立测试" / "event_evaluation"
          / "20260918_r2_status_v1" / "paired_count_results.csv")
INTERNAL_SUMMARY = REPO / "Dataset_new" / "72video" / "al_images" / "paper_repro_summary.csv"

MEMBERS_TEXT = ("gray:direct;lab_b:direct;red_minus_blue:inverted;blue:direct;"
                "lab_b:inverted;lab_a:inverted;red_minus_blue:direct")
PROMINENCE = 0.05
FPS = 8.7
SEED = 20260922
DRAWS = 5000


def r2_score(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[m], ref[m]
    return 1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()


def mae(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    return float(np.abs(pred[m] - ref[m]).mean())


def rmse(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    return float(math.sqrt(((pred[m] - ref[m]) ** 2).mean()))


def main():
    paired = pd.read_csv(PAIRED)
    print("E055 配对窗: %d" % len(paired))

    internal = pd.read_csv(INTERNAL_SUMMARY)
    config = p2g.derive_config(internal, FPS)
    config = replace(config, base_prominence=PROMINENCE)
    members = [tuple(m.split(":")) for m in MEMBERS_TEXT.split(";")]
    print("冻结成员 %d 个；prominence=%.3f" % (len(members), PROMINENCE))

    rows = []
    miss = 0
    for r in paired.itertuples(index=False):
        wid = str(r.window_id)
        f = SIGDIR / (wid + "_calibration_free_signals.csv")
        if not f.exists():
            miss += 1
            continue
        try:
            table = pd.read_csv(f)
            est = [p2g.estimate_curve(p2g.fused_curve(table, s, p == "inverted", config), config)[0]
                   for s, p in members]
            rr = float(np.mean(est))
        except Exception as e:
            print("  FAILED %s: %s" % (wid, e))
            rr = math.nan
        rows.append(dict(window_id=wid, cow=str(r.verified_cow_id),
                         p2g_rr_bpm=rr,
                         rf_rr_bpm=float(r.predicted_rr_bpm),
                         truth_rr_bpm=float(r.truth_rr_bpm),
                         truth_count=float(r.truth_count),
                         predicted_count_rf=float(r.predicted_count)))
    pred = pd.DataFrame(rows)
    print("成功评分 %d 窗（缺信号 %d）" % (len(pred), miss))
    ok = pred[np.isfinite(pred["p2g_rr_bpm"])].copy()
    print("P2g 给出有限预测: %d / %d" % (len(ok), len(pred)))
    pred.to_csv(OUT / "p2g_vs_rf_paired_predictions.csv", index=False, encoding="utf-8")

    ref = ok["truth_rr_bpm"].to_numpy(float)
    p = ok["p2g_rr_bpm"].to_numpy(float)
    r = ok["rf_rr_bpm"].to_numpy(float)

    print()
    print("=" * 74)
    print("同一批 %d 个配对窗（R2 20260918 参考）" % len(ok))
    print("%-26s %9s %9s %9s" % ("method", "R2", "MAE", "RMSE"))
    print("%-26s %9.6f %9.4f %9.4f" % ("RF 绝对温度代理 (E055)", r2_score(r, ref), mae(r, ref), rmse(r, ref)))
    print("%-26s %9.6f %9.4f %9.4f" % ("P2g 相对伪彩集合", r2_score(p, ref), mae(p, ref), rmse(p, ref)))
    print("%-26s %+9.6f %+9.4f %+9.4f" % ("Δ (P2g − RF)", r2_score(p, ref) - r2_score(r, ref),
                                          mae(p, ref) - mae(r, ref), rmse(p, ref) - rmse(r, ref)))
    print()
    print("完全计数(P2g 取整): %d / %d" % ((np.round(p) == ok["truth_count"].to_numpy(float)).sum(), len(ok)))
    print("完全计数(RF):       %d / %d" % ((ok["predicted_count_rf"].to_numpy(float) == ok["truth_count"].to_numpy(float)).sum(), len(ok)))

    # ---- 配对 bootstrap：按牛号聚类 ----
    cows = ok["cow"].astype(str).to_numpy()
    uniq = sorted(set(cows))
    idx_by_cow = {c: np.where(cows == c)[0] for c in uniq}
    print()
    print("按牛号聚类配对 bootstrap（%d 次，%d 个牛号，seed=%d）" % (DRAWS, len(uniq), SEED))
    rng = random.Random(SEED)
    dR2, dMAE, dRMSE = [], [], []
    for _ in range(DRAWS):
        pick = [uniq[rng.randrange(len(uniq))] for _ in range(len(uniq))]
        ii = np.concatenate([idx_by_cow[c] for c in pick])
        rr_, pp_, ff_ = ref[ii], p[ii], r[ii]
        dR2.append(r2_score(pp_, rr_) - r2_score(ff_, rr_))
        dMAE.append(mae(pp_, rr_) - mae(ff_, rr_))
        dRMSE.append(rmse(pp_, rr_) - rmse(ff_, rr_))

    def ci(v):
        v = sorted(v)
        return float(np.mean(v)), v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]

    res = []
    for name, v in [("delta_rr_r2", dR2), ("delta_rr_mae", dMAE), ("delta_rr_rmse", dRMSE)]:
        m, lo, hi = ci(v)
        verdict = "跨0 -> 无显著差异" if lo < 0 < hi else ("P2g 更好" if hi < 0 else "P2g 更差")
        if name == "delta_rr_r2":
            verdict = "跨0 -> 无显著差异" if lo < 0 < hi else ("P2g 更好" if lo > 0 else "P2g 更差")
        print("  %-16s %+9.4f  95%%CI [%+9.4f, %+9.4f]  %s" % (name, m, lo, hi, verdict))
        res.append(dict(metric=name, estimate=m, ci_low=lo, ci_high=hi, verdict=verdict))
    pd.DataFrame(res).to_csv(OUT / "p2g_vs_rf_bootstrap.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
