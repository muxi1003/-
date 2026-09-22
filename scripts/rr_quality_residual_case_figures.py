from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rr_quality_residual_figures import OKABE_ITO, configure_style, save_figure


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Create representative respiratory-curve case figures."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_inputs(input_root: Path, output_prefix: str, corrected_prefix: str) -> dict[str, pd.DataFrame]:
    files = {
        "predictions": input_root / f"{corrected_prefix}_predictions.csv",
        "taxonomy": input_root / f"{output_prefix}_error_taxonomy.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    return {name: pd.read_csv(path) for name, path in files.items()}


def numeric_value(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_number_list(value: object) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return []
    numbers = []
    for part in value.split(";"):
        number = numeric_value(part)
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def merged_cases(predictions: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    taxonomy_columns = [
        "video_id",
        "error_type",
        "error_reason",
        "first_gap",
        "last_gap",
        "median_interval",
        "peak_indices",
        "peak_intervals",
        "peak_prominences",
    ]
    available = [column for column in taxonomy_columns if column in taxonomy.columns]
    data = data.merge(taxonomy[available], on="video_id", how="left")
    data["error_type"] = data["error_type"].fillna("no_default_count_error")
    data["error_reason"] = data["error_reason"].fillna("")
    for column in [
        "truth_count",
        "peaks",
        "count_error",
        "target_adjust",
        "predicted_adjust",
        "applied_adjust",
        "corrected_peaks",
        "corrected_count_error",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "residual_confidence",
        "residual_margin",
    ]:
        if column in data.columns:
            data[column] = numeric_series(data[column])
    return data


def pick_case(
    data: pd.DataFrame,
    *,
    case_label: str,
    description: str,
    predicate: pd.Series,
    preferred: list[str],
) -> pd.Series:
    candidates = data[predicate].copy()
    if candidates.empty:
        raise ValueError(f"No candidate found for case: {case_label}")
    for video_id in preferred:
        match = candidates[candidates["video_id"].astype(str) == video_id]
        if not match.empty:
            row = match.iloc[0].copy()
            row["case_label"] = case_label
            row["case_description"] = description
            return row
    candidates = candidates.sort_values(
        ["residual_confidence", "residual_margin"], ascending=[False, False]
    )
    row = candidates.iloc[0].copy()
    row["case_label"] = case_label
    row["case_description"] = description
    return row


def select_cases(data: pd.DataFrame) -> pd.DataFrame:
    corrected = data["corrected_count_error"].fillna(np.inf) == 0
    applied_plus = data["applied_adjust"].fillna(0) > 0
    applied_minus = data["applied_adjust"].fillna(0) < 0
    rows = [
        pick_case(
            data,
            case_label="endpoint_missing_corrected",
            description="Default pipeline missed one breath; residual corrector added one count.",
            predicate=corrected
            & applied_plus
            & data["error_type"].eq("endpoint_or_boundary_missing_peak"),
            preferred=["170333", "210944", "210994"],
        ),
        pick_case(
            data,
            case_label="boundary_extra_corrected",
            description="Default pipeline counted one extra boundary peak; residual corrector removed one count.",
            predicate=corrected
            & applied_minus
            & data["error_type"].eq("boundary_extra_peak_or_window_mismatch"),
            preferred=["211040"],
        ),
        pick_case(
            data,
            case_label="noise_overcount_corrected",
            description="Default pipeline overcounted a short/noisy local peak; residual corrector removed one count.",
            predicate=corrected
            & applied_minus
            & data["error_type"].eq("noise_or_double_peak_overcount"),
            preferred=["ns196527"],
        ),
        pick_case(
            data,
            case_label="false_positive_limitation",
            description="Residual corrector changed a video that was already exactly counted.",
            predicate=(data["count_error"].fillna(np.inf) == 0)
            & (data["applied_adjust"].fillna(0) != 0),
            preferred=["200939"],
        ),
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


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


def frame_to_time(frame_index: float, curve: pd.DataFrame, time_s: np.ndarray) -> float:
    if len(curve) == 0:
        return math.nan
    index = int(round(frame_index))
    index = max(0, min(index, len(curve) - 1))
    return float(time_s[index])


def nearest_y(curve: pd.DataFrame, frame_index: float) -> float:
    if "smoothed_norm" not in curve.columns or len(curve) == 0:
        return math.nan
    index = int(round(frame_index))
    index = max(0, min(index, len(curve) - 1))
    value = numeric_value(curve.iloc[index]["smoothed_norm"])
    if math.isfinite(value):
        return value
    return numeric_series(curve["smoothed_norm"]).dropna().median()


def missed_region_frame(row: pd.Series, curve: pd.DataFrame) -> float | None:
    peaks = parse_number_list(row.get("peak_indices", ""))
    intervals = parse_number_list(row.get("peak_intervals", ""))
    if peaks and intervals:
        median_interval = numeric_value(row.get("median_interval"))
        max_index = int(np.argmax(intervals))
        if (
            0 <= max_index < len(peaks) - 1
            and math.isfinite(median_interval)
            and intervals[max_index] >= 1.25 * median_interval
        ):
            return float((peaks[max_index] + peaks[max_index + 1]) / 2.0)
    first_gap = numeric_value(row.get("first_gap"))
    last_gap = numeric_value(row.get("last_gap"))
    median_interval = numeric_value(row.get("median_interval"))
    if math.isfinite(first_gap) and math.isfinite(median_interval) and first_gap >= 0.75 * median_interval:
        return max(0.0, first_gap / 2.0)
    if math.isfinite(last_gap) and math.isfinite(median_interval) and last_gap >= 0.75 * median_interval:
        return float(max(0, len(curve) - 1 - last_gap / 2.0))
    return None


def suspected_extra_peak_frame(row: pd.Series) -> float | None:
    peaks = parse_number_list(row.get("peak_indices", ""))
    intervals = parse_number_list(row.get("peak_intervals", ""))
    prominences = parse_number_list(row.get("peak_prominences", ""))
    if not peaks:
        return None
    error_type = str(row.get("error_type", ""))
    if "boundary_extra" in error_type:
        first_gap = numeric_value(row.get("first_gap"))
        last_gap = numeric_value(row.get("last_gap"))
        if math.isfinite(first_gap) and math.isfinite(last_gap):
            return float(peaks[0] if first_gap <= last_gap else peaks[-1])
        return float(peaks[-1])
    if "noise" in error_type and intervals:
        min_index = int(np.argmin(intervals))
        candidates = [min_index, min_index + 1]
        if len(prominences) == len(peaks):
            candidates = [idx for idx in candidates if idx < len(prominences)]
            if candidates:
                chosen = min(candidates, key=lambda idx: prominences[idx])
                return float(peaks[chosen])
        if min_index + 1 < len(peaks):
            return float(peaks[min_index + 1])
    if len(prominences) == len(peaks):
        return float(peaks[int(np.argmin(prominences))])
    return None


def plot_case(ax: plt.Axes, input_root: Path, row: pd.Series, panel: str) -> dict[str, object]:
    video_id = str(row["video_id"])
    curve_path = input_root / video_id / "paper_repro_curve.csv"
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing curve CSV for {video_id}: {curve_path}")
    curve = pd.read_csv(curve_path)
    duration = numeric_value(row.get("duration_seconds"))
    time_s = time_axis(curve, duration)
    smoothed = numeric_series(curve.get("smoothed_norm", pd.Series(index=curve.index, dtype=float)))
    fused = numeric_series(curve.get("fused_norm", pd.Series(index=curve.index, dtype=float)))

    ax.plot(time_s, fused, color="0.78", lw=0.8, alpha=0.75, label="Fused curve")
    ax.plot(time_s, smoothed, color=OKABE_ITO["blue"], lw=1.2, label="Smoothed curve")
    peaks = peak_mask(curve)
    ax.scatter(
        time_s[peaks.to_numpy()],
        smoothed[peaks].to_numpy(dtype=float),
        s=18,
        color=OKABE_ITO["black"],
        zorder=3,
        label="Default peaks",
    )

    marker_frame = None
    marker_note = ""
    if numeric_value(row.get("applied_adjust")) > 0:
        marker_frame = missed_region_frame(row, curve)
        marker_note = "Candidate missed cycle"
        if marker_frame is not None:
            x = frame_to_time(marker_frame, curve, time_s)
            ax.axvspan(
                max(time_s[0], x - 0.25),
                min(time_s[-1], x + 0.25),
                color=OKABE_ITO["orange"],
                alpha=0.22,
                lw=0,
                label=marker_note,
            )
            ax.text(
                x,
                0.05,
                "+1",
                color=OKABE_ITO["vermillion"],
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    elif numeric_value(row.get("applied_adjust")) < 0:
        marker_frame = suspected_extra_peak_frame(row)
        marker_note = "Suspected extra peak"
        if marker_frame is not None:
            x = frame_to_time(marker_frame, curve, time_s)
            y = nearest_y(curve, marker_frame)
            ax.scatter(
                [x],
                [y],
                marker="x",
                s=60,
                color=OKABE_ITO["vermillion"],
                lw=1.6,
                zorder=4,
                label=marker_note,
            )

    default_count = int(numeric_value(row.get("peaks")))
    truth_count = int(numeric_value(row.get("truth_count")))
    corrected_count = int(numeric_value(row.get("corrected_peaks")))
    applied = int(numeric_value(row.get("applied_adjust")))
    confidence = numeric_value(row.get("residual_confidence"))
    margin = numeric_value(row.get("residual_margin"))
    title = f"{panel}. {video_id} | {row['case_label'].replace('_', ' ')}"
    ax.set_title(title, loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized nostril signal")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(True, axis="y", color="0.9", lw=0.5)
    ax.text(
        0.01,
        0.98,
        (
            f"truth={truth_count}, default={default_count}, corrected={corrected_count}\n"
            f"adjust={applied:+d}, confidence={confidence:.2f}, margin={margin:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "lw": 0.5},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return {
        "video_id": video_id,
        "curve_csv": str(curve_path),
        "marker_frame": marker_frame if marker_frame is not None else np.nan,
        "marker_note": marker_note,
    }


def case_figure(input_root: Path, cases: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
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
    paths = save_figure(fig, output_dir, "rr_representative_cases", dpi)
    marker_table = pd.DataFrame(marker_rows)
    marker_table.to_csv(output_dir / "rr_representative_case_markers.csv", index=False)
    return paths


def write_case_table(cases: pd.DataFrame, input_root: Path, output_dir: Path) -> tuple[Path, Path]:
    columns = [
        "case_label",
        "case_description",
        "video_id",
        "error_type",
        "error_reason",
        "truth_count",
        "peaks",
        "count_error",
        "applied_adjust",
        "corrected_peaks",
        "corrected_count_error",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "residual_confidence",
        "residual_margin",
    ]
    table = cases[columns].copy()
    root_table = input_root / "paper_repro_quality_residual_case_examples.csv"
    figure_table = output_dir / "rr_representative_case_examples.csv"
    table.to_csv(root_table, index=False)
    table.to_csv(figure_table, index=False)
    return root_table, figure_table


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
    manifest = output_dir / "representative_case_figure_manifest.csv"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(input_root, args.output_prefix, args.corrected_prefix)
    cases = select_cases(merged_cases(data["predictions"], data["taxonomy"]))
    root_table, figure_table = write_case_table(cases, input_root, output_dir)
    paths = case_figure(input_root, cases, output_dir, int(args.dpi))
    manifest = write_manifest(paths, output_dir)
    print(f"Saved case examples: {root_table}")
    print(f"Saved figure case table: {figure_table}")
    print(f"Saved representative case figures to: {output_dir}")
    print(f"Saved manifest: {manifest}")
    print(cases[["case_label", "video_id", "truth_count", "peaks", "applied_adjust", "corrected_peaks"]].to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
