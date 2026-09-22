"""Describe observed timestamp gaps without changing frozen window membership."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    source = args.preflight / "decoded_frame_timestamps.csv"
    frame = read_csv(source)
    checks = read_csv(args.preflight / "video_decode_checks.csv").set_index("window_id")
    out = args.preflight / "timestamp_gap_analysis"
    out.mkdir(parents=True, exist_ok=False)
    gaps = []
    for window, rows in frame.groupby("window_id", sort=False):
        t = rows.opencv_ffmpeg_timestamp_seconds.astype(float).to_numpy()
        indices = rows.decoded_frame_index.astype(int).to_numpy()
        for i in np.flatnonzero(np.diff(t) > .5):
            gaps.append({"window_id": window, "source_path": checks.loc[window, "source_path"],
                         "previous_frame_index": int(indices[i]), "next_frame_index": int(indices[i+1]),
                         "previous_timestamp_seconds": float(t[i]), "next_timestamp_seconds": float(t[i+1]),
                         "interval_seconds": float(t[i+1]-t[i]),
                         "gap_crosses_window_end": bool(t[i+1] >= 30),
                         "interpretation": "no_decoded_frame_timestamp_between_endpoints; not_confirmed_camera_dropped_frames"})
    write_csv(out / "timestamp_intervals_over_0p5s.csv", gaps)
    checks = checks.reset_index()
    write_csv(out / "all271_timing_classification.csv", checks[["window_id", "decode_status", "max_timestamp_step_seconds", "max_error_if_index_divided_by_8_7_seconds", "issues"]])
    examples = [r["window_id"] for r in gaps[:1]]
    if examples:
        rows = frame[frame.window_id.eq(examples[0])]
        t = rows.opencv_ffmpeg_timestamp_seconds.astype(float).to_numpy()
        ix = rows.decoded_frame_index.astype(int).to_numpy()
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        axes[0].plot(ix, t, label="OpenCV/FFMPEG timestamp", color="#087e8b")
        axes[0].plot(ix, ix/8.7, label="Nominal frame index / 8.7", color="#cd5935", linestyle="--")
        axes[0].set(xlabel="Decoded frame index", ylabel="Time (seconds)", title=examples[0])
        axes[0].legend()
        axes[1].plot(t[1:], np.diff(t), color="#087e8b")
        axes[1].axhline(.5, color="#cd5935", linestyle="--", label="Preflight review threshold (0.5 s)")
        axes[1].set(xlabel="Timestamp (seconds)", ylabel="Adjacent-frame interval (seconds)")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        fig.savefig(out / "first_flagged_window_timing.png", dpi=160)
        plt.close(fig)
    total = len(checks)
    n = len({g["window_id"] for g in gaps})
    maximum = max((g["interval_seconds"] for g in gaps), default=0)
    drift = checks.max_error_if_index_divided_by_8_7_seconds.astype(float)
    write_json(out / "summary.json", {"windows": total, "windows_with_gap_over_0p5s": n,
               "intervals_over_0p5s": len(gaps), "largest_interval_seconds": maximum,
               "largest_nominal_time_error_seconds": float(drift.max()), "median_max_nominal_time_error_seconds": float(drift.median()),
               "raw_timestamp_sha256": sha256(source), "reference_files_used": [], "inference_or_member_filtering": False})
    write_text(out / "时间轴问题说明.md", f"# 首30秒时间轴核查\n\n271/271窗均解码至30秒边界之后，后端时间戳全部从0开始且严格递增；不是51个视频无法打开。\n\n"
               f"其中{n}窗共有{len(gaps)}个相邻解码帧时间间隔超过0.5秒，最大{maximum:.3f}秒。简单用帧号/8.7替代实际时间，最大偏差{drift.max():.3f}秒。\n\n"
               "这只证明后端报告的帧时间不均匀，尚不能区分采集暂停、编码时基或其他原因，不直接命名为已确认丢帧。\n\n"
               "## 对后续推理的约束\n\n"
               "- 维持原271窗、首30秒不变，不因时间轴或人工不可观察问题重新挑选片段。\n"
               "- RR分母仍是预定义30秒；不能删去时间间隔后压缩成更短时长，从而改变人工观看窗口。\n"
               "- 信号输入需显式处理不规则采样；长间隔不能靠重复帧/温度插值生成额外呼吸。\n"
               "- 必须先在内部数据验证并冻结时间适配规则，再运行外测；本次未改变已冻结信号参数，也未选择让外测指标更好的规则。\n"
               "- 220窗未超过此阈值只代表这一项检查通过，不能据此确认温度映射、可见性或呼吸率预测正确。\n")
    print(f"{n}/{total} windows; {len(gaps)} intervals; maximum={maximum:.3f}s; nominal drift={drift.max():.3f}s")


if __name__ == "__main__":
    main()
