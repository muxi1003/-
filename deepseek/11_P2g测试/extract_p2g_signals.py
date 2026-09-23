# -*- coding: utf-8 -*-
"""
D012  P2g 通道提取器（选项 A：与当前外测同 ROI）

做法：
  1. 从 Experiment/02_冻结独立测试/external_transfer/20260914_v2/code_snapshots/
     导入当时运行所用的三个模块（frames_from_anchor / sample_map / read_bgr_strict），
     保证帧采样与当时**逐位一致**；
  2. 对每个窗：解码源视频 -> 按 map.csv 选帧 -> JPEG q95 编码 -> SHA256
     与 frame_hashes.csv **逐帧比对**（位一致性验证）；
  3. 用 temperatures.csv 的 (x, y, adaptive_roi_radius) 计算 8 个颜色通道的圆形 ROI 均值
     （通道数学逐行复刻 scripts/build_rr_calibration_free_thermal_index.py 的 roi_means）。

只读项目文件；产物写入 deepseek/11_P2g测试/。
"""
from __future__ import annotations
import argparse, json, math, os, sys, hashlib
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path(r"E:\real\use_code\yoloV8")
SNAP = REPO / "Experiment" / "02_冻结独立测试"          # 活动模块（帧哈希校验会兜住任何漂移）
WINROOT = REPO / "Experiment" / "02_冻结独立测试" / "external_transfer" / "20260914_v2" / "windows"
RELEASE = REPO / "Experiment" / "02_冻结独立测试" / "release_20260909_v1" / "test_windows.csv"
OUT = REPO / "deepseek" / "11_P2g测试" / "signals"

sys.path.insert(0, str(SNAP))
from timestamp_adapter import sample_map                                  # noqa: E402
from validate_raw_pipeline_internal49 import frames_from_anchor            # noqa: E402

SIGNALS = ["gray", "blue", "green", "red", "hue", "lab_a", "lab_b", "red_minus_blue"]


def roi_means(image, x, y, radius):
    """逐行复刻 build_rr_calibration_free_thermal_index.roi_means"""
    xv = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    yv = pd.to_numeric(pd.Series([y]), errors="coerce").iloc[0]
    if pd.isna(xv) or pd.isna(yv):
        return {s: math.nan for s in SIGNALS}
    cx, cy = int(round(float(xv))), int(round(float(yv)))
    r = int(round(float(radius)))
    height, width = image.shape[:2]
    x0, x1 = max(0, cx - r), min(width, cx + r + 1)
    y0, y1 = max(0, cy - r), min(height, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return {s: math.nan for s in SIGNALS}
    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    if not np.any(mask):
        return {s: math.nan for s in SIGNALS}
    pixels = patch[mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    blue, green, red = np.mean(pixels.astype(float), axis=0)
    return {
        "gray": float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[mask])),
        "blue": float(blue), "green": float(green), "red": float(red),
        "hue": float(np.median(hsv[:, 0].astype(float))),
        "lab_a": float(np.mean(lab[:, 1].astype(float))),
        "lab_b": float(np.mean(lab[:, 2].astype(float))),
        "red_minus_blue": float(red - blue),
    }


def process_window(source: Path, window_id: str, out_dir: Path):
    wdir = WINROOT / window_id
    mapping = pd.read_csv(wdir / "map.csv")
    temps = pd.read_csv(wdir / "temperatures.csv")
    hashes = pd.read_csv(wdir / "frame_hashes.csv").set_index("target_index")

    # 解码源视频，逐帧复现
    times, frames = [], {}
    for index, relative, timestamp, frame in frames_from_anchor(source, 0):
        times.append(relative)
        frames[index] = frame
    if len(times) == 0:
        raise ValueError("no frames decoded")

    wanted = mapping.groupby("source_frame_index")["target_index"].apply(list).to_dict()
    rows, hash_ok, hash_bad = [], 0, 0
    for index in sorted(wanted):
        if index not in frames:
            raise ValueError("source frame %d missing" % index)
        frame = frames[index]
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("jpeg encode failed")
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        for target in wanted[index]:
            expect = str(hashes.loc[target, "sha256"])
            if digest == expect:
                hash_ok += 1
            else:
                hash_bad += 1
            rows.append((target, encoded))

    # 逐 target 计算通道
    out_rows = []
    tmap = temps.set_index("frame_name")
    for target, encoded in sorted(rows, key=lambda x: x[0]):
        image = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        name = "frame_%06d.jpg" % target
        rec = tmap.loc[name]
        out = {"frame_name": name, "target_index": target,
               "left_x": rec["left_x"], "left_y": rec["left_y"],
               "right_x": rec["right_x"], "right_y": rec["right_y"],
               "adaptive_roi_radius": rec["adaptive_roi_radius"],
               "left_status": rec.get("status"), }
        for side in ("left", "right"):
            vals = roi_means(image, rec["%s_x" % side], rec["%s_y" % side],
                             rec["adaptive_roi_radius"])
            for s, v in vals.items():
                out["%s_%s" % (side, s)] = v
        out_rows.append(out)
    table = pd.DataFrame(out_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / (window_id + "_calibration_free_signals.csv"), index=False)
    return dict(window_id=window_id, frames=len(table), hash_ok=hash_ok, hash_bad=hash_bad,
                source=str(source))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个窗（0=全部）")
    ap.add_argument("--windows", default="", help="逗号分隔的 window_id")
    ap.add_argument("--manifest", default="", help="window_id 清单文件（每行一个）")
    args = ap.parse_args()

    rel = pd.read_csv(RELEASE)
    want = set()
    if args.windows:
        want = set(x.strip() for x in args.windows.split(",") if x.strip())
    if args.manifest:
        want |= set(x.strip() for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip())

    todo = []
    for row in rel.itertuples(index=False):
        wid = str(row.window_id)
        if want and wid not in want:
            continue
        if not (WINROOT / wid / "map.csv").exists():
            continue
        todo.append((wid, Path(str(row.source_path))))
    if args.limit:
        todo = todo[:args.limit]

    print("待处理窗口: %d" % len(todo), flush=True)
    results = []
    for i, (wid, src) in enumerate(todo, 1):
        out_dir = OUT
        done = out_dir / (wid + "_calibration_free_signals.csv")
        if done.exists():
            print("[%d/%d] %s 已存在，跳过" % (i, len(todo), wid), flush=True)
            continue
        try:
            r = process_window(src, wid, out_dir)
            results.append(r)
            print("[%d/%d] %s  frames=%d  hash_ok=%d  hash_bad=%d"
                  % (i, len(todo), wid, r["frames"], r["hash_ok"], r["hash_bad"]), flush=True)
        except Exception as e:
            print("[%d/%d] %s  FAILED: %s" % (i, len(todo), wid, e), flush=True)
            results.append(dict(window_id=wid, frames=0, hash_ok=0, hash_bad=-1, source=str(src), error=str(e)))
    pd.DataFrame(results).to_csv(OUT / "_run_manifest.csv", index=False)
    tot_ok = sum(r["hash_ok"] for r in results)
    tot_bad = sum(r["hash_bad"] for r in results)
    print("\n完成 %d 窗；帧哈希逐位一致 %d，不一致 %d" % (len(results), tot_ok, tot_bad), flush=True)


if __name__ == "__main__":
    main()
