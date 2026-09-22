from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


REPRO_BLUE = "#0072B2"
SKY_BLUE = "#56B4E9"
INNOVATION_ORANGE = "#D55E00"
AMBER = "#E69F00"
TEAL = "#009E73"
PURPLE = "#CC79A7"
DARK = "#263442"
MUTED = "#617181"
LIGHT_GRID = "#D8E0E7"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description="Build a publication-style multi-panel technical route for the 73-video RR study."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=default_assets)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> str:
    candidates = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "axes.linewidth": 0.8,
        }
    )
    return selected


def panel_box(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    title: str,
    *,
    kind: str,
    label: str,
) -> None:
    x, y, w, h = rect
    if kind == "reproduction":
        edge = REPRO_BLUE
        fill = "#F1F7FB"
        tag = "原文复现"
    elif kind == "innovation":
        edge = INNOVATION_ORANGE
        fill = "#FFF6ED"
        tag = "本文改进"
    else:
        edge = TEAL
        fill = "#F1F8F6"
        tag = "内部验证"

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=fill,
        transform=fig.transFigure,
        zorder=-20,
    )
    fig.add_artist(patch)
    fig.text(x + 0.011, y + h - 0.026, label, fontsize=15, fontweight="bold", color=edge)
    fig.text(x + 0.034, y + h - 0.026, title, fontsize=12.3, fontweight="bold", color=DARK)
    tag_width = 0.058 if kind != "validation" else 0.066
    tag_patch = FancyBboxPatch(
        (x + w - tag_width - 0.012, y + h - 0.035),
        tag_width,
        0.022,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=0.8,
        edgecolor=edge,
        facecolor="white",
        transform=fig.transFigure,
        zorder=5,
    )
    fig.add_artist(tag_patch)
    fig.text(
        x + w - tag_width / 2 - 0.012,
        y + h - 0.024,
        tag,
        ha="center",
        va="center",
        fontsize=7.5,
        color=edge,
        fontweight="bold",
        zorder=6,
    )


def figure_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#405365",
    connectionstyle: str = "arc3,rad=0",
    linewidth: float = 1.8,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        transform=fig.transFigure,
        clip_on=False,
        zorder=30,
    )
    fig.add_artist(arrow)


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#82909C")
    ax.spines["bottom"].set_color("#82909C")
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT_GRID, linewidth=0.6, alpha=0.75)


def draw_network(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [
        (0.02, 0.32, "Backbone", ["Conv", "C2f", "SPPF"], "#D9EAD3"),
        (0.36, 0.32, "Neck", ["C2f", "Concat", "Upsample"], "#D9E2F3"),
        (0.70, 0.32, "Pose head", ["Box", "L-keypoint", "R-keypoint"], "#E4D7F5"),
    ]
    for x, width, title, blocks, color in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.08),
                width - 0.03,
                0.82,
                boxstyle="round,pad=0.012",
                facecolor="white",
                edgecolor="#8DA0B2",
                linewidth=0.9,
            )
        )
        ax.text(x + (width - 0.03) / 2, 0.94, title, ha="center", va="center", fontsize=7.3, fontweight="bold")
        for idx, block in enumerate(blocks):
            by = 0.66 - idx * 0.22
            ax.add_patch(Rectangle((x + 0.04, by), width - 0.11, 0.12, facecolor=color, edgecolor="#5E6C78", lw=0.7))
            ax.text(x + (width - 0.03) / 2, by + 0.06, block, ha="center", va="center", fontsize=6.3)
        if x < 0.7:
            ax.add_patch(FancyArrowPatch((x + width - 0.02, 0.48), (x + width + 0.025, 0.48), arrowstyle="-|>", mutation_scale=10, color=DARK, lw=0.9))


def draw_random_forest(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rgb_colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]
    for idx, color in enumerate(rgb_colors):
        ax.add_patch(Rectangle((0.025 + (idx % 2) * 0.07, 0.63 - (idx // 2) * 0.13), 0.055, 0.09, facecolor=color, edgecolor="white", lw=0.7))
    ax.text(0.087, 0.43, "像素特征\n(R, G, B)", ha="center", va="top", fontsize=7.5, color=DARK)
    ax.add_patch(FancyArrowPatch((0.16, 0.59), (0.25, 0.59), arrowstyle="-|>", mutation_scale=11, color=DARK, lw=1.0))

    sample_x = [0.27, 0.39, 0.51]
    for idx, x in enumerate(sample_x, start=1):
        ax.add_patch(FancyBboxPatch((x, 0.70), 0.085, 0.13, boxstyle="round,pad=0.01", facecolor="#FFF0D5", edgecolor=AMBER, lw=0.8))
        ax.text(x + 0.043, 0.765, f"D{idx}", ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.add_patch(FancyBboxPatch((x, 0.37), 0.085, 0.15, boxstyle="round,pad=0.01", facecolor="#E8F3F7", edgecolor=REPRO_BLUE, lw=0.8))
        ax.text(x + 0.043, 0.445, f"Tree {idx}", ha="center", va="center", fontsize=6.8)
        ax.add_patch(FancyArrowPatch((x + 0.043, 0.70), (x + 0.043, 0.53), arrowstyle="-|>", mutation_scale=9, color=DARK, lw=0.8))
        ax.add_patch(FancyArrowPatch((x + 0.043, 0.37), (0.68, 0.27), arrowstyle="-|>", mutation_scale=8, color="#667785", lw=0.7))
    ax.text(0.475, 0.765, "…", fontsize=13, ha="center", va="center", color=MUTED)
    ax.add_patch(FancyBboxPatch((0.63, 0.16), 0.18, 0.17, boxstyle="round,pad=0.012", facecolor="#DDF0EA", edgecolor=TEAL, lw=1.0))
    ax.text(0.72, 0.245, "Ensemble\naverage", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.81, 0.245), (0.90, 0.245), arrowstyle="-|>", mutation_scale=11, color=DARK, lw=1.0))

    ax.add_patch(Rectangle((0.91, 0.24), 0.025, 0.34, facecolor="#F47B5B", edgecolor="#7C4D3F", lw=0.8))
    ax.add_patch(Circle((0.9225, 0.20), 0.055, facecolor="#F47B5B", edgecolor="#7C4D3F", lw=0.8))
    ax.text(0.92, 0.68, "温度 T", ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK)
    ax.text(0.67, 0.00, r"$\hat{T}=\frac{1}{K}\sum_{k=1}^{K} f_k(R,G,B)$", ha="center", va="bottom", fontsize=8.2, color=DARK)


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def draw_repair_flow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = ["短缺失\n关键点跟踪", "对侧鼻孔\n几何偏移推断", "低置信度\n关键点回退"]
    colors = ["#DCEAF5", "#E4F2ED", "#FDEBD7"]
    xs = [0.02, 0.36, 0.70]
    for idx, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.add_patch(FancyBboxPatch((x, 0.10), 0.25, 0.77, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#7E8E9B", lw=0.9))
        ax.text(x + 0.125, 0.49, label, ha="center", va="center", fontsize=7.4, color=DARK, fontweight="bold")
        if idx < 2:
            ax.add_patch(FancyArrowPatch((x + 0.26, 0.49), (xs[idx + 1] - 0.015, 0.49), arrowstyle="-|>", mutation_scale=10, color=DARK, lw=0.9))


def draw_fusion_flow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    modes = ["Left", "Right", "Mean", "Max", "Min"]
    for idx, mode in enumerate(modes):
        y = 0.82 - idx * 0.19
        selected = mode == "Right"
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.07),
                0.28,
                0.12,
                boxstyle="round,pad=0.012",
                facecolor="#DDF0EA" if selected else "white",
                edgecolor=TEAL if selected else "#96A5B1",
                linewidth=1.2 if selected else 0.8,
            )
        )
        ax.text(0.16, y - 0.01, mode + ("  已选" if selected else ""), ha="center", va="center", fontsize=7.4, fontweight="bold" if selected else "normal", color=TEAL if selected else DARK)
        ax.add_patch(FancyArrowPatch((0.30, y - 0.01), (0.49, 0.46), arrowstyle="-|>", mutation_scale=8, color="#7A8793", lw=0.7))

    diamond = Polygon([[0.49, 0.46], [0.66, 0.67], [0.83, 0.46], [0.66, 0.25]], closed=True, facecolor="#FFF0D5", edgecolor=INNOVATION_ORANGE, lw=1.1)
    ax.add_patch(diamond)
    ax.text(0.66, 0.50, "质量评分 Q", ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK)
    ax.text(0.66, 0.40, "显著性 · 振幅\n周期稳定性 · 缺失率", ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.add_patch(FancyArrowPatch((0.83, 0.46), (0.98, 0.46), arrowstyle="-|>", mutation_scale=11, color=TEAL, lw=1.2))


def draw_peak_panels(fig: plt.Figure, curve_df: pd.DataFrame) -> None:
    x = pd.to_numeric(curve_df["frame_index"], errors="coerce").to_numpy(dtype=float)
    raw = pd.to_numeric(curve_df["fused_norm"], errors="coerce").to_numpy(dtype=float)
    smoothed = pd.to_numeric(curve_df["smoothed_norm"], errors="coerce").to_numpy(dtype=float)
    peaks = curve_df.loc[curve_df["is_peak"].fillna(False), ["frame_index", "smoothed_norm"]].apply(pd.to_numeric, errors="coerce").dropna()

    ax1 = fig.add_axes([0.041, 0.145, 0.180, 0.115])
    ax1.plot(x, raw, color=SKY_BLUE, alpha=0.45, lw=1.0, label="融合曲线")
    ax1.plot(x, smoothed, color=REPRO_BLUE, lw=1.7, label="3帧移动平均")
    ax1.scatter(peaks["frame_index"], peaks["smoothed_norm"], s=14, color=DARK, zorder=4)
    ax1.set_xlabel("帧", fontsize=7)
    ax1.set_ylabel("归一化温度", fontsize=7)
    ax1.set_title("自适应显著度计峰", fontsize=8.5, fontweight="bold", color=DARK, pad=4)
    clean_axis(ax1)

    ax2 = fig.add_axes([0.252, 0.145, 0.180, 0.115])
    xx = np.linspace(0, 1, 200)
    yy = 0.22 + 0.72 * np.exp(-((xx - 0.40) / 0.11) ** 2) + 0.60 * np.exp(-((xx - 0.55) / 0.10) ** 2)
    yy -= 0.13 * np.exp(-((xx - 0.48) / 0.055) ** 2)
    ax2.plot(xx, yy, color=REPRO_BLUE, lw=1.8)
    p1 = int(np.argmax(yy[:100]))
    p2 = 100 + int(np.argmax(yy[100:]))
    ax2.scatter([xx[p1], xx[p2]], [yy[p1], yy[p2]], color=[TEAL, INNOVATION_ORANGE], s=30, zorder=4)
    ax2.text(xx[p2], yy[p2] + 0.08, "×", color=INNOVATION_ORANGE, fontsize=15, ha="center", va="center", fontweight="bold")
    valley = float(np.min(yy[p1:p2 + 1]))
    ax2.annotate("浅谷", xy=(0.48, valley), xytext=(0.70, 0.33), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8), fontsize=7, color=MUTED)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("噪声双峰抑制", fontsize=8.5, fontweight="bold", color=DARK, pad=4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    clean_axis(ax2, grid=False)

    ax3 = fig.add_axes([0.463, 0.145, 0.180, 0.115])
    xx = np.linspace(0, 1, 180)
    yy = 0.50 + 0.35 * np.cos(6 * np.pi * xx + 0.25)
    ax3.plot(xx, yy, color=REPRO_BLUE, lw=1.7)
    normal_peaks = [0.32, 0.65, 0.98]
    for peak in normal_peaks:
        yv = 0.50 + 0.35 * math.cos(6 * math.pi * peak + 0.25)
        ax3.scatter([peak], [yv], s=22, color=DARK, zorder=4)
    edge_x = 0.02
    edge_y = 0.50 + 0.35 * math.cos(6 * math.pi * edge_x + 0.25)
    ax3.scatter([edge_x], [edge_y], s=38, facecolor="white", edgecolor=AMBER, lw=1.5, zorder=5)
    ax3.axvline(0.08, color=AMBER, ls="--", lw=1.0)
    ax3.text(0.10, 0.08, "边界候选", fontsize=7, color=AMBER)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("保守端点周期补全", fontsize=8.5, fontweight="bold", color=DARK, pad=4)
    ax3.set_xticks([])
    ax3.set_yticks([])
    clean_axis(ax3, grid=False)

    fig.text(0.131, 0.115, "平滑 + prominence + 最小峰距", ha="center", va="center", fontsize=7.1, color=MUTED)
    fig.text(0.342, 0.115, "近邻峰且谷深不足时保留主峰", ha="center", va="center", fontsize=7.1, color=MUTED)
    fig.text(0.553, 0.115, "仅在短时稀疏曲线满足周期条件时启用", ha="center", va="center", fontsize=7.1, color=MUTED)


def build_figure(repo_root: Path, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    al_images = repo_root / "Dataset_new" / "72video" / "al_images"
    primary_dir = al_images / "zs197000"
    missing_dir = al_images / "zs170065"

    frame_paths = [primary_dir / f"zs197000_frame_{index:06d}.jpg" for index in (0, 40, 80)]
    missing_frames = [missing_dir / f"zs170065_frame_{index:06d}.jpg" for index in (0, 60)]
    required = [
        *frame_paths,
        *missing_frames,
        primary_dir / "paper_repro_temperatures.csv",
        primary_dir / "paper_repro_curve.csv",
        missing_dir / "paper_repro_temperatures.csv",
        al_images / "paper_repro_summary.csv",
        al_images / "paper_repro_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing route-figure inputs: " + ", ".join(missing))

    primary_temp = pd.read_csv(primary_dir / "paper_repro_temperatures.csv")
    primary_curve = pd.read_csv(primary_dir / "paper_repro_curve.csv")
    missing_temp = pd.read_csv(missing_dir / "paper_repro_temperatures.csv")
    summary = pd.read_csv(al_images / "paper_repro_summary.csv")
    metrics = pd.read_csv(al_images / "paper_repro_metrics.csv").iloc[0]

    fig = plt.figure(figsize=(18, 11.25), facecolor="white")
    fig.text(0.5, 0.967, "基于热红外图像的奶牛呼吸频率检测与质量感知增强技术路线", ha="center", va="center", fontsize=22, fontweight="bold", color="#17232E")
    fig.text(0.5, 0.938, "73段内部视频：原文关键点定位与温度映射复现 + 双鼻孔信号重建与约束计峰改进", ha="center", va="center", fontsize=10.5, color=MUTED)

    legend_y = 0.915
    fig.add_artist(Rectangle((0.718, legend_y - 0.006), 0.018, 0.012, transform=fig.transFigure, facecolor="#F1F7FB", edgecolor=REPRO_BLUE, lw=1.2))
    fig.text(0.741, legend_y, "原文复现模块", va="center", fontsize=8.2, color=REPRO_BLUE)
    fig.add_artist(Rectangle((0.820, legend_y - 0.006), 0.018, 0.012, transform=fig.transFigure, facecolor="#FFF6ED", edgecolor=INNOVATION_ORANGE, lw=1.2))
    fig.text(0.843, legend_y, "本文改进模块", va="center", fontsize=8.2, color=INNOVATION_ORANGE)

    panel_box(fig, (0.02, 0.67, 0.56, 0.235), "热红外序列与鼻孔关键点定位", kind="reproduction", label="A")
    panel_box(fig, (0.61, 0.67, 0.37, 0.235), "RGB–温度随机森林映射", kind="reproduction", label="B")
    panel_box(fig, (0.02, 0.375, 0.50, 0.255), "单侧缺失与温度序列重建", kind="innovation", label="C")
    panel_box(fig, (0.55, 0.375, 0.43, 0.255), "质量感知双鼻孔融合", kind="innovation", label="D")
    panel_box(fig, (0.02, 0.08, 0.65, 0.25), "约束计峰与呼吸周期判定", kind="innovation", label="E")
    panel_box(fig, (0.70, 0.08, 0.28, 0.25), "RR计算与73视频内部验证", kind="validation", label="F")

    # A: real thermal sequence.
    sequence_positions = [0.038, 0.075, 0.112]
    for zorder, (path, x) in enumerate(zip(frame_paths, sequence_positions), start=1):
        ax = fig.add_axes([x, 0.715, 0.105, 0.145], zorder=zorder)
        image = mpimg.imread(path)
        ax.imshow(image[80:1230, 110:970])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.0)
    fig.text(0.119, 0.687, "热红外视频序列", ha="center", fontsize=8.3, fontweight="bold", color=DARK)
    figure_arrow(fig, (0.215, 0.785), (0.232, 0.785), linewidth=1.4)

    ax_network = fig.add_axes([0.232, 0.708, 0.174, 0.160])
    draw_network(ax_network)
    fig.text(0.319, 0.687, "YOLOv8n-Pose", ha="center", fontsize=8.3, fontweight="bold", color=DARK)
    figure_arrow(fig, (0.408, 0.785), (0.423, 0.785), linewidth=1.4)

    crop_image = mpimg.imread(frame_paths[0])
    crop_x0, crop_y0, crop_x1, crop_y1 = 400, 730, 800, 1090
    ax_crop = fig.add_axes([0.425, 0.715, 0.125, 0.145])
    ax_crop.imshow(crop_image[crop_y0:crop_y1, crop_x0:crop_x1])
    row = primary_temp.iloc[0]
    for side, color in [("left", AMBER), ("right", TEAL)]:
        cx = float(row[f"{side}_x"]) - crop_x0
        cy = float(row[f"{side}_y"]) - crop_y0
        ax_crop.add_patch(Circle((cx, cy), 20, fill=False, edgecolor=color, linewidth=2.2))
        ax_crop.scatter([cx], [cy], s=13, color=color, zorder=5)
        ax_crop.text(cx, cy - 35, "L" if side == "left" else "R", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax_crop.set_xticks([])
    ax_crop.set_yticks([])
    for spine in ax_crop.spines.values():
        spine.set_color(REPRO_BLUE)
        spine.set_linewidth(1.2)
    fig.text(0.4875, 0.687, "圆形鼻孔ROI（r = 20 px）", ha="center", fontsize=8.3, fontweight="bold", color=DARK)

    # B: random forest temperature mapping.
    ax_rf = fig.add_axes([0.625, 0.700, 0.335, 0.170])
    draw_random_forest(ax_rf)

    # C: actual missing-side temperature reconstruction case.
    ax_temp = fig.add_axes([0.040, 0.445, 0.292, 0.125])
    frame_idx = np.arange(len(missing_temp))
    left_temp = pd.to_numeric(missing_temp["left_temp"], errors="coerce")
    right_temp = pd.to_numeric(missing_temp["right_temp"], errors="coerce")
    ax_temp.plot(frame_idx, left_temp, color=REPRO_BLUE, lw=1.25, label="左鼻孔")
    ax_temp.plot(frame_idx, right_temp, color=AMBER, lw=1.25, label="右鼻孔")
    inferred_mask = missing_temp["left_source"].astype(str).ne("detected").to_numpy()
    for start, end in contiguous_spans(inferred_mask):
        ax_temp.axvspan(start - 0.5, end + 0.5, facecolor=INNOVATION_ORANGE, alpha=0.10, edgecolor="none")
    inferred_indices = np.flatnonzero(inferred_mask)
    ax_temp.scatter(inferred_indices[::5], left_temp.iloc[inferred_indices[::5]], s=12, marker="x", color=INNOVATION_ORANGE, linewidths=0.8, label="推断/回退")
    ax_temp.set_xlabel("帧", fontsize=7)
    ax_temp.set_ylabel("温度 (°C)", fontsize=7)
    ax_temp.set_title("zs170065：左鼻孔长时不可见", fontsize=8.2, fontweight="bold", pad=3)
    ax_temp.legend(loc="lower right", frameon=False, fontsize=6.3, ncol=3, handlelength=1.4, columnspacing=0.8)
    clean_axis(ax_temp)

    for index, (path, x, label) in enumerate(zip(missing_frames, [0.352, 0.430], ["偏转前", "偏转后"])):
        ax = fig.add_axes([x, 0.438, 0.070, 0.137])
        image = mpimg.imread(path)
        ax.imshow(image[40:1160, 40:1040])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(INNOVATION_ORANGE if index else "#8AA0AF")
            spine.set_linewidth(1.0)
        fig.text(x + 0.035, 0.425, label, ha="center", fontsize=7.0, color=MUTED)

    ax_repair = fig.add_axes([0.055, 0.392, 0.430, 0.042])
    draw_repair_flow(ax_repair)
    fig.text(0.270, 0.382, "无需人工呼吸次数参与修复", ha="center", va="center", fontsize=6.8, color=INNOVATION_ORANGE, fontweight="bold")

    # D: quality-aware candidate fusion.
    ax_fusion = fig.add_axes([0.568, 0.410, 0.205, 0.175])
    draw_fusion_flow(ax_fusion)
    fig.text(0.663, 0.596, r"$Q=P_{med}+2A-0.5CV_I-2M$", ha="center", va="center", fontsize=9.4, color=DARK)

    left_norm = (primary_temp["left_temp"] - primary_temp["left_temp"].min()) / (primary_temp["left_temp"].max() - primary_temp["left_temp"].min())
    right_norm = (primary_temp["right_temp"] - primary_temp["right_temp"].min()) / (primary_temp["right_temp"].max() - primary_temp["right_temp"].min())
    ax_selected = fig.add_axes([0.795, 0.445, 0.160, 0.115])
    ax_selected.plot(left_norm, color=SKY_BLUE, lw=0.9, alpha=0.55, label="Left")
    ax_selected.plot(right_norm, color=TEAL, lw=1.8, label="Selected: Right")
    ax_selected.set_xlabel("帧", fontsize=7)
    ax_selected.set_ylabel("归一化温度", fontsize=7)
    ax_selected.set_title("候选曲线择优", fontsize=8.2, fontweight="bold", pad=3)
    ax_selected.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.3)
    clean_axis(ax_selected)
    fig.text(0.875, 0.420, "73段中仅3段选择固定Max；70段选择其他质量更优候选", ha="center", va="center", fontsize=6.8, color=INNOVATION_ORANGE, fontweight="bold")
    fig.text(0.875, 0.400, "输出：单条连续呼吸温度曲线", ha="center", va="center", fontsize=7.2, color=DARK)

    # E: constrained peak counting.
    draw_peak_panels(fig, primary_curve)

    # F: RR equation, agreement scatter and metrics.
    fig.text(0.840, 0.274, r"$RR=\dfrac{60N}{T}$", ha="center", va="center", fontsize=20, color="#A33434", fontweight="bold")
    fig.text(0.840, 0.247, "N：有效呼吸峰数    T：有效视频时长", ha="center", va="center", fontsize=7.0, color=MUTED)

    truth_rr = pd.to_numeric(summary["truth_rr"], errors="coerce")
    pred_rr = pd.to_numeric(summary["rr_bpm"], errors="coerce")
    valid = truth_rr.notna() & pred_rr.notna()
    ax_scatter = fig.add_axes([0.718, 0.115, 0.118, 0.110])
    low = float(min(truth_rr[valid].min(), pred_rr[valid].min())) - 2
    high = float(max(truth_rr[valid].max(), pred_rr[valid].max())) + 2
    ax_scatter.plot([low, high], [low, high], color="#7A8793", lw=1.0, ls="--")
    ax_scatter.scatter(truth_rr[valid], pred_rr[valid], s=18, color=TEAL, edgecolor="white", linewidth=0.45, alpha=0.85)
    ax_scatter.set_xlim(low, high)
    ax_scatter.set_ylim(low, high)
    ax_scatter.set_xlabel("人工RR (次/min)", fontsize=6.5)
    ax_scatter.set_ylabel("预测RR (次/min)", fontsize=6.5)
    clean_axis(ax_scatter)

    metric_x = 0.852
    fig.text(metric_x, 0.215, f"RR R² = {float(metrics['rr_r2']):.4f}", fontsize=11.3, fontweight="bold", color="#15384A")
    fig.text(metric_x, 0.184, f"MAE = {float(metrics['rr_mae']):.4f} 次/min", fontsize=9.0, fontweight="bold", color=DARK)
    fig.text(metric_x, 0.155, f"RMSE = {float(metrics['rr_rmse']):.4f} 次/min", fontsize=9.0, fontweight="bold", color=DARK)
    fig.text(metric_x, 0.126, f"完全正确 = {int(metrics['exact_count'])}/73", fontsize=9.0, fontweight="bold", color=DARK)
    fig.text(metric_x, 0.097, f"误差 ≤ 1次 = {int(metrics['within_one_count'])}/73", fontsize=9.0, fontweight="bold", color=DARK)

    # Main processing arrows.
    figure_arrow(fig, (0.585, 0.785), (0.606, 0.785), linewidth=2.0)
    figure_arrow(fig, (0.650, 0.669), (0.515, 0.632), connectionstyle="arc3,rad=0", linewidth=1.8)
    figure_arrow(fig, (0.523, 0.505), (0.546, 0.505), linewidth=2.0)
    figure_arrow(fig, (0.715, 0.372), (0.647, 0.334), connectionstyle="arc3,rad=0.10", linewidth=1.8)
    figure_arrow(fig, (0.675, 0.205), (0.696, 0.205), linewidth=2.0)

    fig.text(0.020, 0.047, "示例帧：zs197000、zs170065；结果来源：paper_repro_summary.csv 与 paper_repro_metrics.csv。", fontsize=6.8, color=MUTED)
    fig.text(0.980, 0.047, "频谱/Butterworth仅作为辅助质量诊断，不替代主时域流程。", ha="right", fontsize=6.8, color=MUTED)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "paper73_technical_route_composite"
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=dpi, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=dpi, facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = build_figure(args.repo_root.resolve(), args.output_dir.resolve(), args.dpi)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
