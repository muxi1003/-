# -*- coding: utf-8 -*-
"""
D011 任务A：排查哪些文件继承了"被调整过的 73 集真值"。

判定依据：4 个被改视频 210994 / 211049 / ns196527 / ns211117
  原始(innovation_repro)计数: 7 / 4 / 10 / 10
  调整后(paper_repro)计数:    6 / 5 / 11 / 11

对每个 CSV：找出计数类列，看这 4 个视频取的是哪一套。
只读，不写任何项目文件。
"""
import os, sys, io, csv as csvmod

R = r"E:\real\use_code\yoloV8"
ROOTS = [
    os.path.join(R, "Dataset_new", "72video", "al_images"),
    os.path.join(R, "Dataset_new", "72video"),
    os.path.join(R, "Dataset_new", "external_al_images_single_reference"),
    os.path.join(R, "Experiment"),
]
TARGETS = {"210994": (7, 6), "211049": (4, 5), "ns196527": (10, 11), "ns211117": (10, 11)}
# 计数类列名（小写匹配）
COUNT_HINT = ("truth_count", "manual_breath_count", "breath_count", "reference_count",
              "truth_breaths", "ref_count")
ID_HINT = ("video_id", "clip_id", "window_id", "id")


def sniff(path):
    for enc in ("utf-8-sig", "gb18030", "utf-8", "latin-1"):
        try:
            with io.open(path, encoding=enc, newline="") as f:
                rd = csvmod.reader(f)
                head = next(rd)
                rows = list(rd)
            return enc, head, rows
        except StopIteration:
            return enc, [], []
        except Exception:
            continue
    return None, None, None


def classify(head, rows):
    if not head:
        return None
    low = [h.strip().lower() for h in head]
    id_idx = None
    for i, h in enumerate(low):
        if h in ("video_id", "clip_id"):
            id_idx = i
            break
    if id_idx is None:
        return None
    cnt_idx = None
    for i, h in enumerate(low):
        if h in COUNT_HINT:
            cnt_idx = i
            break
    if cnt_idx is None:
        return None
    orig_hits = adj_hits = 0
    total = 0
    for r in rows:
        if len(r) <= max(id_idx, cnt_idx):
            continue
        vid = r[id_idx].strip()
        if vid not in TARGETS:
            continue
        try:
            v = float(r[cnt_idx])
        except Exception:
            continue
        o, a = TARGETS[vid]
        total += 1
        if abs(v - o) < 1e-9:
            orig_hits += 1
        elif abs(v - a) < 1e-9:
            adj_hits += 1
    if total == 0:
        return None
    return dict(col=head[cnt_idx].strip(), total=total, orig=orig_hits, adj=adj_hits)


adj_files, clean_files, mixed = [], [], []
seen = set()
for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "figures", "curve_png")]
        for f in fn:
            if not f.lower().endswith(".csv"):
                continue
            p = os.path.join(dp, f)
            if p in seen:
                continue
            seen.add(p)
            try:
                if os.path.getsize(p) > 8 * 1024 * 1024:
                    continue
            except OSError:
                continue
            enc, head, rows = sniff(p)
            if head is None:
                continue
            res = classify(head, rows)
            if not res:
                continue
            rel = os.path.relpath(p, R)
            tag = None
            if res["adj"] > 0 and res["orig"] == 0:
                tag = "ADJUSTED"
                adj_files.append((rel, res))
            elif res["orig"] > 0 and res["adj"] == 0:
                tag = "clean"
                clean_files.append((rel, res))
            elif res["orig"] > 0 and res["adj"] > 0:
                tag = "MIXED"
                mixed.append((rel, res))
            if tag:
                print("%-9s %-46s 列=%-22s 命中%d/%d (原始%d 调整%d)" %
                      (tag, rel, res["col"], res["total"], 4, res["orig"], res["adj"]))

print()
print("=" * 78)
print("汇总：命中 4 个目标视频且含计数列的 CSV")
print("  ADJUSTED（全部用调整值）: %d" % len(adj_files))
print("  clean   （全部用原始值）: %d" % len(clean_files))
print("  MIXED   （两者混杂）    : %d" % len(mixed))
print()
if adj_files:
    print("!! 继承调整真值的文件：")
    for rel, res in adj_files:
        print("   ", rel)
if mixed:
    print("!! 混杂文件（需人工判读）：")
    for rel, res in mixed:
        print("   ", rel, res)
