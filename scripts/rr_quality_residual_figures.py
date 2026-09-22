from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


METHODS = {
    "baseline": {
        "label": "Default pipeline",
        "rr_column": "rr_bpm",
        "count_column": "peaks",
        "color": OKABE_ITO["black"],
        "marker": "o",
    },
    "fixed": {
        "label": "Quality residual, fixed threshold",
        "rr_column": "corrected_rr_bpm",
        "count_column": "corrected_peaks",
        "color": OKABE_ITO["blue"],
        "marker": "s",
    },
    "nested": {
        "label": "Quality residual, nested CV",
        "rr_column": "nested_corrected_rr_bpm",
        "count_column": "nested_corrected_peaks",
        "color": OKABE_ITO["orange"],
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Create publication-ready figures for quality residual RR validation."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def regression_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 2:
        return math.nan
    truth = y_true[valid]
    pred = y_pred[valid]
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= 0:
        return math.nan
    return float(1.0 - np.sum((pred - truth) ** 2) / denominator)


def method_metrics(data: pd.DataFrame, rr_column: str, count_column: str) -> dict[str, float]:
    truth_rr = numeric(data["truth_rr"])
    pred_rr = numeric(data[rr_column])
    truth_count = numeric(data["truth_count"])
    pred_count = numeric(data[count_column])
    rr_error = pred_rr - truth_rr
    count_error = pred_count - truth_count
    return {
        "r2": regression_r2(truth_rr, pred_rr),
        "mae": float(np.nanmean(np.abs(rr_error))),
        "rmse": float(np.sqrt(np.nanmean(rr_error**2))),
        "exact": int(np.nansum(np.abs(count_error) == 0)),
    }


def load_inputs(input_root: Path, prefix: str) -> dict[str, pd.DataFrame]:
    paths = {
        "predictions": input_root / f"{prefix}_predictions.csv",
        "nested_predictions": input_root / f"{prefix}_nested_predictions.csv",
        "validation_summary": input_root / f"{prefix}_validation_summary.csv",
        "bootstrap_ci": input_root / f"{prefix}_bootstrap_ci.csv",
        "threshold_grid": input_root / f"{prefix}_threshold_grid.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def scatter_figure(
    predictions: pd.DataFrame, nested_predictions: pd.DataFrame, output_dir: Path, dpi: int
) -> list[Path]:
    datasets = {
        "baseline": predictions,
        "fixed": predictions,
        "nested": nested_predictions,
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=True, sharey=True)
    truth_all = numeric(predictions["truth_rr"])
    axis_min = float(np.nanmin(truth_all)) - 4
    axis_max = float(np.nanmax(truth_all)) + 4
    for panel, (key, ax) in zip("ABC", zip(["baseline", "fixed", "nested"], axes)):
        spec = METHODS[key]
        data = datasets[key]
        truth = numeric(data["truth_rr"])
        pred = numeric(data[spec["rr_column"]])
        metrics = method_metrics(data, spec["rr_column"], spec["count_column"])
        ax.scatter(
            truth,
            pred,
            s=18,
            marker=spec["marker"],
            facecolor=spec["color"],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
        )
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="0.35", lw=0.8, ls="--")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_title(spec["label"])
        ax.set_xlabel("Reference RR (breaths/min)")
        ax.text(
            0.03,
            0.97,
            panel,
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.text(
            0.05,
            0.08,
            f"R2={metrics['r2']:.3f}\nMAE={metrics['mae']:.2f}\nExact={metrics['exact']}/73",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.85", "linewidth": 0.4, "alpha": 0.9},
        )
        despine(ax)
    axes[0].set_ylabel("Predicted RR (breaths/min)")
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, output_dir, "rr_prediction_scatter", dpi)


def bland_altman_figure(
    predictions: pd.DataFrame, nested_predictions: pd.DataFrame, output_dir: Path, dpi: int
) -> list[Path]:
    datasets = {
        "baseline": predictions,
        "fixed": predictions,
        "nested": nested_predictions,
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharey=True)
    all_diffs: list[np.ndarray] = []
    for key, data in datasets.items():
        spec = METHODS[key]
        all_diffs.append(numeric(data[spec["rr_column"]]) - numeric(data["truth_rr"]))
    y_min = min(float(np.nanmin(diff)) for diff in all_diffs) - 1.0
    y_max = max(float(np.nanmax(diff)) for diff in all_diffs) + 1.0

    for panel, (key, ax) in zip("ABC", zip(["baseline", "fixed", "nested"], axes)):
        spec = METHODS[key]
        data = datasets[key]
        truth = numeric(data["truth_rr"])
        pred = numeric(data[spec["rr_column"]])
        mean_rr = (truth + pred) / 2.0
        diff = pred - truth
        bias = float(np.nanmean(diff))
        sd = float(np.nanstd(diff, ddof=1))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
        ax.scatter(
            mean_rr,
            diff,
            s=18,
            marker=spec["marker"],
            facecolor=spec["color"],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
        )
        ax.axhline(0, color="0.2", lw=0.7)
        ax.axhline(bias, color=spec["color"], lw=1.0)
        ax.axhline(loa_low, color=spec["color"], lw=0.8, ls="--")
        ax.axhline(loa_high, color=spec["color"], lw=0.8, ls="--")
        ax.set_ylim(y_min, y_max)
        ax.set_title(spec["label"])
        ax.set_xlabel("Mean RR (breaths/min)")
        ax.text(0.03, 0.97, panel, transform=ax.transAxes, fontweight="bold", va="top")
        ax.text(
            0.05,
            0.08,
            f"Bias={bias:.2f}\nLoA={loa_low:.2f} to {loa_high:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.85", "linewidth": 0.4, "alpha": 0.9},
        )
        despine(ax)
    axes[0].set_ylabel("Prediction error (breaths/min)")
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, output_dir, "rr_bland_altman", dpi)


def bootstrap_ci_figure(ci: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    rows = ci[ci["metric"].isin(["delta_rr_r2", "delta_rr_mae", "delta_rr_rmse", "delta_exact_count"])].copy()
    label_map = {
        "delta_rr_r2": "Delta R2",
        "delta_rr_mae": "Delta MAE (breaths/min)",
        "delta_rr_rmse": "Delta RMSE (breaths/min)",
        "delta_exact_count": "Delta exact count",
    }
    rows["label"] = rows["metric"].map(label_map)
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.9))
    for panel, (_, row), ax in zip("ABCD", rows.iterrows(), axes.ravel()):
        estimate = float(row["estimate"])
        low = float(row["ci_low_2_5"])
        high = float(row["ci_high_97_5"])
        ax.errorbar(
            estimate,
            0,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=OKABE_ITO["blue"],
            ecolor=OKABE_ITO["blue"],
            elinewidth=1.2,
            capsize=3,
            markersize=4,
        )
        ax.axvline(0, color="0.25", lw=0.8, ls="--")
        ax.set_yticks([])
        ax.set_xlabel(str(row["label"]))
        ax.text(0.03, 0.92, panel, transform=ax.transAxes, fontweight="bold", va="top")
        ax.text(
            0.50,
            0.12,
            f"{estimate:.3f}\n95% CI {low:.3f} to {high:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
        despine(ax)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    return save_figure(fig, output_dir, "rr_bootstrap_ci", dpi)


def threshold_heatmap_figure(grid: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    metrics = [("rr_r2", "R2"), ("exact_count", "Exact count")]
    for panel, (metric, title), ax in zip("AB", metrics, axes):
        pivot = grid.pivot_table(
            index="margin_threshold",
            columns="confidence_threshold",
            values=metric,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
        )
        best = grid.sort_values(
            ["rr_r2", "rr_mae", "rr_rmse", "exact_count"],
            ascending=[False, True, True, False],
        ).iloc[0]
        ax.scatter(
            [float(best["confidence_threshold"])],
            [float(best["margin_threshold"])],
            marker="*",
            s=70,
            color=OKABE_ITO["vermillion"],
            edgecolor="white",
            linewidth=0.5,
            label="Best grid",
        )
        ax.scatter(
            [0.54],
            [0.26],
            marker="o",
            s=32,
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
            label="Fixed threshold",
        )
        ax.set_title(title)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Margin threshold")
        ax.text(0.03, 0.97, panel, transform=ax.transAxes, fontweight="bold", va="top", color="white")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.1)
    return save_figure(fig, output_dir, "rr_threshold_sensitivity", dpi)


def combined_figure(
    predictions: pd.DataFrame,
    nested_predictions: pd.DataFrame,
    ci: pd.DataFrame,
    grid: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(7.1, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bland = fig.add_subplot(gs[0, 1])
    ax_ci = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    truth = numeric(predictions["truth_rr"])
    base = numeric(predictions["rr_bpm"])
    fixed = numeric(predictions["corrected_rr_bpm"])
    axis_min = float(np.nanmin(truth)) - 4
    axis_max = float(np.nanmax(truth)) + 4
    ax_scatter.scatter(truth, base, s=17, color=OKABE_ITO["black"], alpha=0.45, label="Default")
    ax_scatter.scatter(truth, fixed, s=17, color=OKABE_ITO["blue"], alpha=0.75, marker="s", label="Fixed threshold")
    ax_scatter.plot([axis_min, axis_max], [axis_min, axis_max], color="0.35", lw=0.8, ls="--")
    ax_scatter.set_xlim(axis_min, axis_max)
    ax_scatter.set_ylim(axis_min, axis_max)
    ax_scatter.set_xlabel("Reference RR (breaths/min)")
    ax_scatter.set_ylabel("Predicted RR (breaths/min)")
    ax_scatter.legend(frameon=False, loc="lower right")
    despine(ax_scatter)

    mean_rr = (truth + fixed) / 2.0
    diff = fixed - truth
    bias = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1))
    ax_bland.scatter(mean_rr, diff, s=17, color=OKABE_ITO["blue"], alpha=0.75, marker="s")
    ax_bland.axhline(0, color="0.25", lw=0.8)
    ax_bland.axhline(bias, color=OKABE_ITO["blue"], lw=1.0)
    ax_bland.axhline(bias - 1.96 * sd, color=OKABE_ITO["blue"], lw=0.8, ls="--")
    ax_bland.axhline(bias + 1.96 * sd, color=OKABE_ITO["blue"], lw=0.8, ls="--")
    ax_bland.set_xlabel("Mean RR (breaths/min)")
    ax_bland.set_ylabel("Prediction error (breaths/min)")
    despine(ax_bland)

    ci_rows = ci[ci["metric"].isin(["delta_rr_r2", "delta_rr_mae", "delta_exact_count"])].copy()
    labels = ["Delta R2", "Delta MAE", "Delta exact"]
    y = np.arange(len(ci_rows))
    estimates = ci_rows["estimate"].to_numpy(dtype=float)
    lows = ci_rows["ci_low_2_5"].to_numpy(dtype=float)
    highs = ci_rows["ci_high_97_5"].to_numpy(dtype=float)
    ax_ci.errorbar(
        estimates,
        y,
        xerr=[estimates - lows, highs - estimates],
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        capsize=3,
    )
    ax_ci.axvline(0, color="0.25", lw=0.8, ls="--")
    ax_ci.set_yticks(y, labels)
    ax_ci.set_xlabel("Fixed-threshold improvement")
    despine(ax_ci)

    pivot = grid.pivot_table(
        index="margin_threshold",
        columns="confidence_threshold",
        values="rr_r2",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    image = ax_heat.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
    )
    ax_heat.scatter([0.54], [0.26], marker="o", s=36, facecolor="none", edgecolor="white", linewidth=1.0)
    ax_heat.set_xlabel("Confidence threshold")
    ax_heat.set_ylabel("Margin threshold")
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("R2", labelpad=2)

    for label, ax in zip("ABCD", [ax_scatter, ax_bland, ax_ci, ax_heat]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontweight="bold", va="top")

    return save_figure(fig, output_dir, "rr_quality_residual_combined", dpi)


def write_manifest(paths: list[Path], output_dir: Path) -> Path:
    rows = [
        {
            "figure_file": str(path),
            "format": path.suffix.lstrip("."),
            "bytes": int(path.stat().st_size),
        }
        for path in paths
    ]
    manifest = output_dir / "figure_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    args = parse_args()
    configure_style()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_figures"
    )
    data = load_inputs(input_root, args.corrected_prefix)

    generated: list[Path] = []
    generated.extend(
        scatter_figure(
            data["predictions"],
            data["nested_predictions"],
            output_dir,
            int(args.dpi),
        )
    )
    generated.extend(
        bland_altman_figure(
            data["predictions"],
            data["nested_predictions"],
            output_dir,
            int(args.dpi),
        )
    )
    generated.extend(bootstrap_ci_figure(data["bootstrap_ci"], output_dir, int(args.dpi)))
    generated.extend(threshold_heatmap_figure(data["threshold_grid"], output_dir, int(args.dpi)))
    generated.extend(
        combined_figure(
            data["predictions"],
            data["nested_predictions"],
            data["bootstrap_ci"],
            data["threshold_grid"],
            output_dir,
            int(args.dpi),
        )
    )
    manifest = write_manifest(generated, output_dir)

    print(f"Saved figures to: {output_dir}")
    print(f"Saved manifest: {manifest}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
