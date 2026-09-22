from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rr_quality_residual_figures import (
    OKABE_ITO,
    configure_style,
    despine,
    method_metrics,
    numeric,
    save_figure,
)


METHODS = {
    "default": {
        "label": "Default pipeline",
        "rr_column": "rr_bpm",
        "count_column": "peaks",
        "color": OKABE_ITO["black"],
        "marker": "o",
    },
    "signal_consensus": {
        "label": "Signal consensus",
        "rr_column": "signal_consensus_rr_bpm",
        "count_column": "signal_consensus_peaks",
        "color": OKABE_ITO["green"],
        "marker": "s",
    },
    "signal_aware": {
        "label": "Signal-aware residual",
        "rr_column": "signal_aware_final_rr_bpm",
        "count_column": "signal_aware_final_peaks",
        "color": OKABE_ITO["purple"],
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Create publication figures for the signal-aware residual candidate."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs(input_root: Path, output_prefix: str) -> dict[str, pd.DataFrame]:
    files = {
        "predictions": input_root / f"{output_prefix}_signal_aware_residual_predictions.csv",
        "metrics": input_root / f"{output_prefix}_signal_aware_residual_metrics.csv",
        "bootstrap_ci": input_root / f"{output_prefix}_signal_aware_residual_bootstrap_ci.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required signal-aware files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in files.items()}


def scatter_comparison_figure(predictions: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=True, sharey=True)
    truth_all = numeric(predictions["truth_rr"])
    axis_min = float(np.nanmin(truth_all)) - 4
    axis_max = float(np.nanmax(truth_all)) + 4
    for panel, (key, ax) in zip("ABC", zip(METHODS, axes)):
        spec = METHODS[key]
        truth = numeric(predictions["truth_rr"])
        pred = numeric(predictions[spec["rr_column"]])
        metrics = method_metrics(predictions, spec["rr_column"], spec["count_column"])
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
        ax.text(0.03, 0.97, panel, transform=ax.transAxes, fontweight="bold", va="top")
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
    return save_figure(fig, output_dir, "signal_aware_rr_prediction_scatter", dpi)


def bootstrap_ci_figure(ci: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    rows = ci[
        ci["metric"].isin(["delta_rr_r2", "delta_rr_mae", "delta_exact_count"])
        & ci["comparison"].isin(["signal_aware_vs_signal_consensus", "signal_aware_vs_default"])
    ].copy()
    rows["display"] = rows["comparison"].map(
        {
            "signal_aware_vs_signal_consensus": "vs signal consensus",
            "signal_aware_vs_default": "vs default",
        }
    )
    metric_labels = {
        "delta_rr_r2": "Delta R2",
        "delta_rr_mae": "Delta MAE (breaths/min)",
        "delta_exact_count": "Delta exact count",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for panel, (metric, ax) in zip("ABC", zip(metric_labels, axes)):
        subset = rows[rows["metric"] == metric].copy()
        subset = subset.sort_values("comparison")
        y = np.arange(len(subset))
        estimate = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci_low_2_5"].to_numpy(dtype=float)
        high = subset["ci_high_97_5"].to_numpy(dtype=float)
        colors = [OKABE_ITO["green"], OKABE_ITO["black"]]
        for idx in range(len(subset)):
            ax.errorbar(
                estimate[idx],
                y[idx],
                xerr=[[estimate[idx] - low[idx]], [high[idx] - estimate[idx]]],
                fmt="o",
                color=colors[idx],
                ecolor=colors[idx],
                capsize=3,
                markersize=4,
                elinewidth=1.1,
            )
        ax.axvline(0, color="0.25", lw=0.8, ls="--")
        ax.set_yticks(y, subset["display"].tolist())
        ax.set_xlabel(metric_labels[metric])
        ax.text(0.03, 0.95, panel, transform=ax.transAxes, fontweight="bold", va="top")
        despine(ax)
    fig.tight_layout(w_pad=1.1)
    return save_figure(fig, output_dir, "signal_aware_bootstrap_ci", dpi)


def peak_mask(curve: pd.DataFrame) -> pd.Series:
    if "is_peak" not in curve.columns:
        return pd.Series(False, index=curve.index)
    if pd.api.types.is_bool_dtype(curve["is_peak"]):
        return curve["is_peak"].fillna(False)
    return curve["is_peak"].astype(str).str.lower().isin(["true", "1", "yes"])


def time_axis(curve: pd.DataFrame, duration_seconds: float) -> np.ndarray:
    if len(curve) <= 1 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return np.arange(len(curve), dtype=float)
    return np.linspace(0.0, duration_seconds, len(curve))


def likely_removed_peak(curve: pd.DataFrame, peak_indices: np.ndarray) -> int | None:
    if len(peak_indices) == 0 or "smoothed_norm" not in curve.columns:
        return None
    smoothed = numeric_series(curve["smoothed_norm"]).to_numpy(dtype=float)
    peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(smoothed))]
    if len(peak_indices) == 0:
        return None
    values = smoothed[peak_indices]
    if np.isfinite(values).any():
        return int(peak_indices[int(np.nanargmin(values))])
    return int(peak_indices[-1])


def select_signal_aware_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    for column in [
        "truth_count",
        "signal_consensus_peaks",
        "signal_aware_final_peaks",
        "signal_aware_applied_adjust",
        "signal_aware_confidence",
        "signal_aware_margin",
        "signal_aware_count_error",
    ]:
        data[column] = numeric_series(data[column])
    changed = data[data["signal_aware_applied_adjust"] != 0].copy()
    changed["fixed_error"] = (
        (changed["signal_consensus_peaks"] != changed["truth_count"])
        & (changed["signal_aware_final_peaks"] == changed["truth_count"])
    )
    changed["introduced_error"] = (
        (changed["signal_consensus_peaks"] == changed["truth_count"])
        & (changed["signal_aware_final_peaks"] != changed["truth_count"])
    )
    fixed = changed[changed["fixed_error"]].sort_values(
        ["signal_aware_confidence", "signal_aware_margin"], ascending=False
    )
    caution = changed[changed["introduced_error"]].sort_values(
        ["signal_aware_confidence", "signal_aware_margin"], ascending=False
    )
    rows = []
    for _, row in fixed.head(3).iterrows():
        copied = row.copy()
        copied["case_role"] = "successful correction"
        rows.append(copied)
    if not caution.empty:
        copied = caution.iloc[0].copy()
        copied["case_role"] = "caution: false correction"
        rows.append(copied)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_case(ax: plt.Axes, input_root: Path, row: pd.Series, panel: str) -> dict[str, object]:
    video_id = str(row["video_id"])
    curve_path = input_root / video_id / "paper_repro_curve.csv"
    curve = pd.read_csv(curve_path)
    duration = finite_float(row.get("rr_duration_seconds"))
    if not math.isfinite(duration):
        duration = finite_float(row.get("duration_seconds"))
    time_s = time_axis(curve, duration)
    fused = numeric_series(curve.get("fused_norm", pd.Series(index=curve.index, dtype=float)))
    smoothed = numeric_series(curve.get("smoothed_norm", pd.Series(index=curve.index, dtype=float)))
    peaks = peak_mask(curve).to_numpy(dtype=bool)
    peak_indices = np.where(peaks)[0]
    removed = likely_removed_peak(curve, peak_indices)

    ax.plot(time_s, fused, color="0.78", lw=0.8, alpha=0.75, label="Fused curve")
    ax.plot(time_s, smoothed, color=OKABE_ITO["purple"], lw=1.2, label="Smoothed curve")
    ax.scatter(
        time_s[peaks],
        smoothed[peaks].to_numpy(dtype=float),
        s=18,
        color=OKABE_ITO["black"],
        zorder=3,
        label="Detected peaks",
    )
    if removed is not None:
        ax.scatter(
            [time_s[removed]],
            [smoothed.iloc[removed]],
            marker="x",
            s=58,
            color=OKABE_ITO["vermillion"],
            lw=1.6,
            zorder=4,
            label="Removed-count cue",
        )
    role = str(row["case_role"])
    truth = int(finite_float(row["truth_count"]))
    consensus = int(finite_float(row["signal_consensus_peaks"]))
    final = int(finite_float(row["signal_aware_final_peaks"]))
    conf = finite_float(row["signal_aware_confidence"])
    margin = finite_float(row["signal_aware_margin"])
    ax.set_title(f"{panel}. {video_id} | {role}", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized signal")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(True, axis="y", color="0.9", lw=0.5)
    ax.text(
        0.01,
        0.98,
        f"truth={truth}, consensus={consensus}, signal-aware={final}\nconfidence={conf:.2f}, margin={margin:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "lw": 0.5},
    )
    despine(ax)
    return {
        "video_id": video_id,
        "case_role": role,
        "truth_count": truth,
        "signal_consensus_peaks": consensus,
        "signal_aware_final_peaks": final,
        "signal_aware_confidence": conf,
        "signal_aware_margin": margin,
        "curve_csv": str(curve_path),
        "marked_frame": removed if removed is not None else np.nan,
    }


def case_figure(
    input_root: Path,
    predictions: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    cases = select_signal_aware_cases(predictions)
    if cases.empty:
        raise ValueError("No signal-aware changed cases are available for plotting.")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    marker_rows = []
    for ax, panel, (_, row) in zip(axes.ravel(), "ABCD", cases.iterrows()):
        marker_rows.append(plot_case(ax, input_root, row, panel))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    paths = save_figure(fig, output_dir, "signal_aware_case_examples", dpi)
    case_table = pd.DataFrame(marker_rows)
    case_table.to_csv(output_dir / "signal_aware_case_examples.csv", index=False)
    return paths, case_table


def write_manifest(paths: list[Path], output_dir: Path) -> Path:
    rows = [
        {
            "figure": path.stem,
            "figure_file": str(path),
            "format": path.suffix.lstrip("."),
            "bytes": int(path.stat().st_size),
        }
        for path in paths
    ]
    manifest = output_dir / "signal_aware_figure_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    args = parse_args()
    configure_style()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.output_prefix}_signal_aware_residual_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(input_root, args.output_prefix)
    generated: list[Path] = []
    generated.extend(scatter_comparison_figure(data["predictions"], output_dir, int(args.dpi)))
    generated.extend(bootstrap_ci_figure(data["bootstrap_ci"], output_dir, int(args.dpi)))
    case_paths, case_table = case_figure(
        input_root,
        data["predictions"],
        output_dir,
        int(args.dpi),
    )
    generated.extend(case_paths)
    manifest = write_manifest(generated, output_dir)
    print(f"Saved signal-aware figures to: {output_dir}")
    print(f"Saved signal-aware figure manifest: {manifest}")
    print(case_table.to_string(index=False))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
