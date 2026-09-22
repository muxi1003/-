from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import peak_prominences


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Classify remaining RR count errors into curve-level failure modes."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-output-csv", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def peak_features(curve: pd.DataFrame) -> dict[str, object]:
    if "smoothed_norm" not in curve.columns or "is_peak" not in curve.columns:
        return {
            "first_gap": math.nan,
            "last_gap": math.nan,
            "median_interval": math.nan,
            "min_interval_ratio": math.nan,
            "max_interval_ratio": math.nan,
            "min_prominence_ratio": math.nan,
            "peak_indices": "",
            "peak_intervals": "",
            "peak_prominences": "",
        }

    smoothed_mask = curve["smoothed_norm"].notna().to_numpy()
    if not smoothed_mask.any():
        return {
            "first_gap": math.nan,
            "last_gap": math.nan,
            "median_interval": math.nan,
            "min_interval_ratio": math.nan,
            "max_interval_ratio": math.nan,
            "min_prominence_ratio": math.nan,
            "peak_indices": "",
            "peak_intervals": "",
            "peak_prominences": "",
        }

    smoothed = curve.loc[smoothed_mask, "smoothed_norm"].to_numpy(dtype=float)
    first_smoothed_index = int(np.where(smoothed_mask)[0][0])
    peak_frames = curve.index[curve["is_peak"].astype(bool)].to_numpy(dtype=int)
    peaks = peak_frames - first_smoothed_index
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed))]
    intervals = np.diff(peaks) if len(peaks) > 1 else np.array([], dtype=float)

    if len(peaks) > 0:
        with np.errstate(invalid="ignore"):
            prominences = peak_prominences(smoothed, peaks)[0]
    else:
        prominences = np.array([], dtype=float)

    median_interval = float(np.median(intervals)) if len(intervals) else math.nan
    min_interval_ratio = (
        float(np.min(intervals) / (median_interval + 1e-9))
        if len(intervals) and not math.isnan(median_interval)
        else math.nan
    )
    max_interval_ratio = (
        float(np.max(intervals) / (median_interval + 1e-9))
        if len(intervals) and not math.isnan(median_interval)
        else math.nan
    )
    median_prominence = float(np.median(prominences)) if len(prominences) else math.nan
    min_prominence_ratio = (
        float(np.min(prominences) / (median_prominence + 1e-9))
        if len(prominences) and not math.isnan(median_prominence)
        else math.nan
    )

    return {
        "first_gap": int(peaks[0]) if len(peaks) else math.nan,
        "last_gap": int((len(smoothed) - 1) - peaks[-1]) if len(peaks) else math.nan,
        "median_interval": median_interval,
        "min_interval_ratio": min_interval_ratio,
        "max_interval_ratio": max_interval_ratio,
        "min_prominence_ratio": min_prominence_ratio,
        "peak_indices": ";".join(str(int(peak)) for peak in peaks),
        "peak_intervals": ";".join(str(int(interval)) for interval in intervals),
        "peak_prominences": ";".join(f"{float(prominence):.4f}" for prominence in prominences),
    }


def classify_error(row: pd.Series, features: dict[str, object]) -> tuple[str, str]:
    count_error = int(row["count_error"])
    first_gap = finite_float(features.get("first_gap"))
    last_gap = finite_float(features.get("last_gap"))
    median_interval = finite_float(features.get("median_interval"))
    min_interval_ratio = finite_float(features.get("min_interval_ratio"))
    max_interval_ratio = finite_float(features.get("max_interval_ratio"))
    min_prominence_ratio = finite_float(features.get("min_prominence_ratio"))

    endpoint_gap = False
    if not math.isnan(median_interval) and median_interval > 0:
        endpoint_gap = (
            (not math.isnan(first_gap) and first_gap >= 0.8 * median_interval)
            or (not math.isnan(last_gap) and last_gap >= 0.8 * median_interval)
        )

    if count_error < 0:
        if endpoint_gap:
            return "endpoint_or_boundary_missing_peak", "long edge gap relative to median breath interval"
        if not math.isnan(max_interval_ratio) and max_interval_ratio >= 1.45:
            return "weak_or_merged_missing_peak", "long internal interval suggests an undetected weak peak"
        if str(row.get("selected_fusion_mode", "")) in {"left", "right", "min", "max"}:
            return "single_channel_or_fusion_missing_peak", "selected curve depends on one dominant nostril channel"
        return "undercount_periodicity_ambiguous", "undercount remains but no strong endpoint or interval signature"

    if count_error > 0:
        if not math.isnan(min_interval_ratio) and min_interval_ratio <= 0.75:
            return "noise_or_double_peak_overcount", "short internal interval suggests an extra local peak"
        if not math.isnan(min_prominence_ratio) and min_prominence_ratio <= 0.25:
            return "weak_noise_peak_overcount", "one detected peak has much lower prominence than the median peak"
        if endpoint_gap:
            return "boundary_extra_peak_or_window_mismatch", "edge gap suggests analysis-window or boundary ambiguity"
        return "overcount_periodicity_ambiguous", "overcount remains but no strong short-interval signature"

    return "no_count_error", "count matches reference"


def build_taxonomy(summary: pd.DataFrame, input_root: Path, output_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    error_rows = summary[pd.to_numeric(summary["abs_count_error"], errors="coerce") > 0].copy()

    for row in error_rows.itertuples(index=False):
        row_dict = row._asdict()
        video_id = str(row_dict["video_id"])
        curve_csv = input_root / video_id / f"{output_prefix}_curve.csv"
        if curve_csv.exists():
            curve = pd.read_csv(curve_csv)
            features = peak_features(curve)
        else:
            features = peak_features(pd.DataFrame())

        label, reason = classify_error(pd.Series(row_dict), features)
        rows.append(
            {
                "video_id": video_id,
                "truth_rr": row_dict.get("truth_rr", math.nan),
                "rr_bpm": row_dict.get("rr_bpm", math.nan),
                "rr_error": row_dict.get("rr_error", math.nan),
                "truth_count": row_dict.get("truth_count", math.nan),
                "peaks": row_dict.get("peaks", math.nan),
                "count_error": row_dict.get("count_error", math.nan),
                "abs_count_error": row_dict.get("abs_count_error", math.nan),
                "error_type": label,
                "error_reason": reason,
                "selected_fusion_mode": row_dict.get("selected_fusion_mode", ""),
                "peak_retune_rule": row_dict.get("peak_retune_rule", ""),
                "edge_peak_added": row_dict.get("edge_peak_added", math.nan),
                "duration_seconds": row_dict.get("duration_seconds", math.nan),
                "rr_duration_source": row_dict.get("rr_duration_source", ""),
                **features,
                "review_png": row_dict.get("review_png", ""),
                "curve_csv": str(curve_csv),
            }
        )

    taxonomy = pd.DataFrame(rows)
    if not taxonomy.empty:
        taxonomy = taxonomy.sort_values(["error_type", "video_id"]).reset_index(drop=True)
    return taxonomy


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    summary_csv = args.summary_csv or input_root / f"{args.output_prefix}_summary.csv"
    output_csv = args.output_csv or input_root / f"{args.output_prefix}_error_taxonomy.csv"
    summary_output_csv = (
        args.summary_output_csv or input_root / f"{args.output_prefix}_error_taxonomy_summary.csv"
    )

    summary = pd.read_csv(summary_csv)
    taxonomy = build_taxonomy(summary, input_root, args.output_prefix)
    taxonomy.to_csv(output_csv, index=False)

    if taxonomy.empty:
        taxonomy_summary = pd.DataFrame(columns=["error_type", "videos"])
    else:
        taxonomy_summary = (
            taxonomy.groupby("error_type", dropna=False)
            .size()
            .reset_index(name="videos")
            .sort_values(["videos", "error_type"], ascending=[False, True])
        )
    taxonomy_summary.to_csv(summary_output_csv, index=False)

    print(f"Saved taxonomy: {output_csv}")
    print(f"Saved taxonomy summary: {summary_output_csv}")
    if not taxonomy_summary.empty:
        print(taxonomy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
