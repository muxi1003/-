from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rr_quality_residual_validation import metric_dict


STRATEGIES = [
    "core20_mean",
    "core20_p90",
    "core12_p90_bg",
    "core20_p90_bg",
    "ring20_35_p95_bg",
    "wedge20_35_p95_bg",
    "multi_max_bg",
    "multi_quality_weighted",
]
DISTANCES = [4, 5, 6, 7, 8]
PROMINENCES = [0.020, 0.035, 0.050, 0.065, 0.090]
SWITCH_MARGIN = 0.10


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a conservative multi-ROI candidate selector against the current "
            "safe-gate RR baseline using current truth labels and held-video/group CV."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path(r"E:\real\code\yolo\repro_outputs\multi_roi"),
    )
    parser.add_argument("--random-state", type=int, default=20260710)
    parser.add_argument("--switch-margin", type=float, default=SWITCH_MARGIN)
    return parser.parse_args()


def numeric(data: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def video_prefix(value: object) -> str:
    match = re.match(r"^[A-Za-z]+", str(value))
    return match.group(0).lower() if match else "numeric"


def fill(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(len(arr), np.nan, dtype=float)
    if not valid.any():
        return out
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    out[valid] = 0.0 if hi <= lo else (arr[valid] - lo) / (hi - lo)
    return out


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    arr = fill(values)
    if window <= 1 or len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def fuse_sides(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = minmax(left)
    right_norm = minmax(right)
    stack = np.vstack([left_norm, right_norm])
    fused = np.full(stack.shape[1], np.nan, dtype=float)
    valid = ~np.all(np.isnan(stack), axis=0)
    fused[valid] = np.nanmax(stack[:, valid], axis=0)
    return smooth(fused)


def bg_corrected(table: pd.DataFrame, side: str, roi: str, statistic: str) -> np.ndarray:
    return (
        numeric(table, f"{side}_{roi}_{statistic}").to_numpy(dtype=float)
        - numeric(table, f"{side}_bg35_50_median").to_numpy(dtype=float)
    )


def side_curve(table: pd.DataFrame, strategy: str, side: str) -> np.ndarray:
    if strategy == "core20_mean":
        return numeric(table, f"{side}_core20_mean").to_numpy(dtype=float)
    if strategy == "core20_p90":
        return numeric(table, f"{side}_core20_p90").to_numpy(dtype=float)
    if strategy == "core12_p90_bg":
        return bg_corrected(table, side, "core12", "p90")
    if strategy == "core20_p90_bg":
        return bg_corrected(table, side, "core20", "p90")
    if strategy == "ring20_35_p95_bg":
        return bg_corrected(table, side, "ring20_35", "p95")
    if strategy == "wedge20_35_p95_bg":
        return bg_corrected(table, side, "wedge20_35", "p95")
    raise ValueError(strategy)


def multi_candidates(table: pd.DataFrame) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for side in ["left", "right"]:
        candidates.extend(
            [
                bg_corrected(table, side, "core12", "p90"),
                bg_corrected(table, side, "core20", "p90"),
                bg_corrected(table, side, "ring20_35", "p95"),
                bg_corrected(table, side, "wedge20_35", "p95"),
            ]
        )
    return candidates


def strategy_curve(table: pd.DataFrame, strategy: str) -> np.ndarray:
    if strategy not in {"multi_max_bg", "multi_quality_weighted"}:
        return fuse_sides(side_curve(table, strategy, "left"), side_curve(table, strategy, "right"))
    arrays = multi_candidates(table)
    normalized: list[np.ndarray] = []
    weights: list[float] = []
    for values in arrays:
        valid = values[np.isfinite(values)]
        if len(valid) < 5:
            continue
        normalized.append(minmax(values))
        amplitude = float(np.nanpercentile(valid, 95) - np.nanpercentile(valid, 5))
        differences = np.diff(valid)
        noise = float(np.median(np.abs(differences - np.median(differences)))) if len(differences) else 0.0
        availability = float(np.mean(np.isfinite(values)))
        weights.append(max(amplitude, 0.0) * availability / (noise + 1e-3))
    if not normalized:
        return np.array([], dtype=float)
    stack = np.vstack(normalized)
    if strategy == "multi_max_bg":
        fused = np.full(stack.shape[1], np.nan, dtype=float)
        valid = ~np.all(np.isnan(stack), axis=0)
        fused[valid] = np.nanmax(stack[:, valid], axis=0)
    else:
        weight = np.asarray(weights, dtype=float)
        valid = np.isfinite(stack)
        weighted = np.where(valid, stack * weight[:, None], 0.0)
        denominator = np.sum(np.where(valid, weight[:, None], 0.0), axis=0)
        fused = np.divide(
            np.sum(weighted, axis=0),
            denominator,
            out=np.full(stack.shape[1], np.nan, dtype=float),
            where=denominator > 0,
        )
    return smooth(fused)


def curve_features(curve: np.ndarray, peaks: np.ndarray, prominence_values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(curve, dtype=float)
    valid = arr[np.isfinite(arr)]
    amplitude = float(np.nanpercentile(valid, 95) - np.nanpercentile(valid, 5)) if len(valid) else 0.0
    differences = np.diff(valid)
    noise = float(np.median(np.abs(differences - np.median(differences)))) if len(differences) else 0.0
    intervals = np.diff(peaks.astype(float))
    interval_cv = (
        float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if len(intervals) >= 2
        else 9.0
    )
    median_prominence = float(np.median(prominence_values)) if len(prominence_values) else 0.0
    first_gap_ratio = float(peaks[0] / (np.median(intervals) + 1e-9)) if len(peaks) and len(intervals) else 9.0
    last_gap_ratio = (
        float((len(arr) - 1 - peaks[-1]) / (np.median(intervals) + 1e-9))
        if len(peaks) and len(intervals)
        else 9.0
    )
    return {
        "curve_amplitude": amplitude,
        "curve_noise_mad": noise,
        "curve_snr_proxy": amplitude / (noise + 1e-3),
        "peak_median_prominence": median_prominence,
        "peak_interval_cv": interval_cv,
        "first_gap_ratio": first_gap_ratio,
        "last_gap_ratio": last_gap_ratio,
        "curve_points": float(len(arr)),
    }


def load_baseline(input_root: Path, output_prefix: str) -> pd.DataFrame:
    path = input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    data["video_id"] = data["video_id"].astype(str)
    data["video_prefix_group"] = data["video_id"].map(video_prefix)
    quality_path = input_root / f"{output_prefix}_algorithmic_quality_predictions.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        quality["video_id"] = quality["video_id"].astype(str)
        keep = [column for column in quality.columns if column == "video_id" or column.startswith("algorithmic_")]
        data = data.merge(quality[keep], on="video_id", how="left", validate="one_to_one")
    return data


def candidate_context(source: pd.Series, candidate_count: int) -> dict[str, float]:
    safe_count = float(source["signal_aware_safe_final_peaks"])
    result: dict[str, float] = {
        "candidate_count": float(candidate_count),
        "safe_count": safe_count,
        "count_delta_vs_safe": float(candidate_count - safe_count),
        "selection_interval_cv": float(source.get("selection_interval_cv", math.nan)),
        "selection_amplitude": float(source.get("selection_amplitude", math.nan)),
        "algorithmic_review_risk_score": float(source.get("algorithmic_review_risk_score", math.nan)),
        "algorithmic_signal_agreement": float(source.get("algorithmic_signal_agreement", math.nan)),
        "algorithmic_interval_instability_score": float(source.get("algorithmic_interval_instability_score", math.nan)),
        "algorithmic_model_ambiguity_score": float(source.get("algorithmic_model_ambiguity_score", math.nan)),
    }
    for column in ["spectral_count_estimate", "fft_count_estimate", "autocorr_count_estimate"]:
        value = pd.to_numeric(pd.Series([source.get(column, math.nan)]), errors="coerce").iloc[0]
        result[f"abs_delta_{column}"] = abs(float(value) - candidate_count) if pd.notna(value) else math.nan
    return result


def build_candidates(
    baseline: pd.DataFrame,
    input_root: Path,
    output_prefix: str,
    features_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for _, source in baseline.iterrows():
        video_id = str(source["video_id"])
        feature_path = features_root / video_id / "multi_roi_features.csv"
        if not feature_path.exists():
            coverage_rows.append({"video_id": video_id, "status": "missing_multi_roi_features", "path": str(feature_path)})
            continue
        table = pd.read_csv(feature_path)
        frame_limit = pd.to_numeric(pd.Series([source.get("analysis_frame_limit", len(table))]), errors="coerce").iloc[0]
        if pd.notna(frame_limit):
            table = table.iloc[: max(1, int(frame_limit))].copy()
        safe_count = int(round(float(source["signal_aware_safe_final_peaks"])))
        truth_count = float(source["truth_count"])
        duration = float(source.get("rr_duration_seconds", source.get("duration_seconds", math.nan)))
        safe_curve_path = input_root / video_id / f"{output_prefix}_curve.csv"
        safe_curve = pd.read_csv(safe_curve_path) if safe_curve_path.exists() else pd.DataFrame()
        safe_values = numeric(safe_curve, "smoothed_norm").dropna().to_numpy(dtype=float)
        safe_peaks = (
            pd.to_numeric(
                safe_curve.loc[
                    safe_curve.get("is_peak", pd.Series(False, index=safe_curve.index)).astype(str).str.lower().isin(["true", "1"]),
                    "frame_index",
                ],
                errors="coerce",
            ).dropna().to_numpy(dtype=int)
            if not safe_curve.empty and "frame_index" in safe_curve.columns
            else np.array([], dtype=int)
        )
        safe_features = curve_features(safe_values, safe_peaks, np.array([], dtype=float))
        candidate_rows.append(
            {
                "video_id": video_id,
                "video_prefix_group": source["video_prefix_group"],
                "strategy": "safe_gate",
                "distance": float(source.get("peak_distance", 5)),
                "prominence": float(source.get("selected_peak_prominence", 0.035)),
                **candidate_context(source, safe_count),
                **safe_features,
                "duration_seconds": duration,
                "candidate_rr_bpm": safe_count * 60.0 / duration,
                "truth_count": truth_count,
                "truth_rr": float(source["truth_rr"]),
                "target_abs_count_error": abs(safe_count - truth_count),
            }
        )
        generated = 0
        for strategy in STRATEGIES:
            curve = strategy_curve(table, strategy)
            if len(curve) < 3:
                continue
            for distance in DISTANCES:
                for prominence in PROMINENCES:
                    peaks, properties = find_peaks(
                        curve,
                        distance=distance,
                        prominence=prominence,
                        width=1,
                    )
                    count = int(len(peaks))
                    if abs(count - safe_count) > 1:
                        continue
                    features = curve_features(
                        curve,
                        peaks,
                        properties.get("prominences", np.array([], dtype=float)),
                    )
                    candidate_rows.append(
                        {
                            "video_id": video_id,
                            "video_prefix_group": source["video_prefix_group"],
                            "strategy": strategy,
                            "distance": distance,
                            "prominence": prominence,
                            **candidate_context(source, count),
                            **features,
                            "duration_seconds": duration,
                            "candidate_rr_bpm": count * 60.0 / duration,
                            "truth_count": truth_count,
                            "truth_rr": float(source["truth_rr"]),
                            "target_abs_count_error": abs(count - truth_count),
                        }
                    )
                    generated += 1
        coverage_rows.append(
            {
                "video_id": video_id,
                "status": "available",
                "path": str(feature_path),
                "feature_rows": len(table),
                "candidate_rows": generated + 1,
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return candidates, pd.DataFrame(coverage_rows)
    count_support = (
        candidates.groupby(["video_id", "candidate_count"]).size().rename("candidate_count_support").reset_index()
    )
    candidates = candidates.merge(
        count_support,
        on=["video_id", "candidate_count"],
        how="left",
        validate="many_to_one",
    )
    total = candidates.groupby("video_id").size().rename("video_candidate_total")
    candidates = candidates.merge(total, on="video_id", how="left", validate="many_to_one")
    candidates["candidate_count_support_rate"] = (
        candidates["candidate_count_support"] / candidates["video_candidate_total"]
    )
    return candidates, pd.DataFrame(coverage_rows)


def feature_columns(candidates: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["strategy"]
    excluded = {
        "video_id",
        "video_prefix_group",
        "truth_count",
        "truth_rr",
        "target_abs_count_error",
        "candidate_rr_bpm",
        "duration_seconds",
    }
    numerical = [
        column
        for column in candidates.columns
        if column not in excluded | set(categorical)
        and pd.to_numeric(candidates[column], errors="coerce").notna().sum() >= 5
    ]
    return categorical, numerical


def build_model(categorical: list[str], numerical: list[str], random_state: int) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("number", SimpleImputer(strategy="median"), numerical),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=8,
        max_features=0.7,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("features", transformer), ("model", model)])


def select_video_candidate(table: pd.DataFrame, margin: float) -> pd.Series:
    safe = table[table["strategy"].astype(str).eq("safe_gate")].iloc[0]
    best = table.sort_values(
        ["predicted_abs_count_error", "candidate_count_support_rate", "strategy"],
        ascending=[True, False, True],
    ).iloc[0]
    if (
        str(best["strategy"]) != "safe_gate"
        and float(best["predicted_abs_count_error"]) + margin
        < float(safe["predicted_abs_count_error"])
    ):
        selected = best.copy()
        selected["selector_reason"] = "predicted_error_margin_switch"
    else:
        selected = safe.copy()
        selected["selector_reason"] = "safe_gate_retained"
    selected["predicted_improvement_vs_safe"] = float(safe["predicted_abs_count_error"]) - float(best["predicted_abs_count_error"])
    return selected


def evaluate_folds(
    candidates: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray, str]],
    categorical: list[str],
    numerical: list[str],
    random_state: int,
    margin: float,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_ids = np.asarray(sorted(candidates["video_id"].unique()), dtype=str)
    selected_rows: list[pd.Series] = []
    fold_rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index, held_out) in enumerate(folds, start=1):
        train_videos = set(video_ids[train_index])
        test_videos = set(video_ids[test_index])
        train = candidates[candidates["video_id"].isin(train_videos)].copy()
        test = candidates[candidates["video_id"].isin(test_videos)].copy()
        model = build_model(categorical, numerical, random_state + fold_index)
        model.fit(train[categorical + numerical], train["target_abs_count_error"].astype(float))
        test["predicted_abs_count_error"] = model.predict(test[categorical + numerical])
        for video_id, group in test.groupby("video_id", sort=True):
            selected = select_video_candidate(group, margin)
            selected["validation_mode"] = mode
            selected["fold"] = fold_index
            selected["held_out_group"] = held_out
            selected_rows.append(selected)
        fold_rows.append(
            {
                "validation_mode": mode,
                "fold": fold_index,
                "held_out_group": held_out,
                "train_videos": len(train_videos),
                "test_videos": len(test_videos),
                "train_candidate_rows": len(train),
                "test_candidate_rows": len(test),
            }
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True), pd.DataFrame(fold_rows)


def stratified_folds(candidates: pd.DataFrame, random_state: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    safe = candidates[candidates["strategy"].astype(str).eq("safe_gate")].sort_values("video_id")
    video_ids = safe["video_id"].astype(str).to_numpy()
    target = np.clip(
        np.rint(safe["truth_count"].to_numpy(dtype=float) - safe["safe_count"].to_numpy(dtype=float)),
        -1,
        1,
    ).astype(int)
    min_class = int(pd.Series(target).value_counts().min())
    n_splits = max(2, min(5, min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ordered = np.asarray(sorted(video_ids), dtype=str)
    index_by_video = {video_id: index for index, video_id in enumerate(ordered)}
    rows: list[tuple[np.ndarray, np.ndarray, str]] = []
    for train, test in splitter.split(video_ids, target):
        rows.append(
            (
                np.asarray([index_by_video[video_ids[index]] for index in train], dtype=int),
                np.asarray([index_by_video[video_ids[index]] for index in test], dtype=int),
                "",
            )
        )
    return rows


def prefix_folds(candidates: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    videos = (
        candidates[["video_id", "video_prefix_group"]]
        .drop_duplicates("video_id")
        .sort_values("video_id")
        .reset_index(drop=True)
    )
    groups = videos["video_prefix_group"].astype(str).to_numpy()
    rows: list[tuple[np.ndarray, np.ndarray, str]] = []
    for group in sorted(np.unique(groups)):
        rows.append((np.flatnonzero(groups != group), np.flatnonzero(groups == group), str(group)))
    return rows


def metric_row(label: str, predictions: pd.DataFrame, note: str) -> dict[str, object]:
    return metric_dict(
        label,
        predictions["truth_rr"].to_numpy(dtype=float),
        predictions["candidate_rr_bpm"].to_numpy(dtype=float),
        predictions["truth_count"].to_numpy(dtype=float),
        predictions["candidate_count"].to_numpy(dtype=float),
        evaluation_note=note,
    )


def baseline_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates[candidates["strategy"].astype(str).eq("safe_gate")].copy()


def oracle_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.sort_values(
            ["video_id", "target_abs_count_error", "strategy"],
            ascending=[True, True, True],
        )
        .groupby("video_id", as_index=False)
        .first()
    )


def margin_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for margin in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        selected = []
        for video_id, group in predictions.groupby("video_id", sort=True):
            selected.append(select_video_candidate(group, margin))
        selected_table = pd.DataFrame(selected)
        row = metric_row(
            f"multi_roi_selector_margin_{margin:.2f}",
            selected_table,
            "held_video_oof_margin_sensitivity",
        )
        row["switch_margin"] = margin
        row["switched_videos"] = int((selected_table["strategy"] != "safe_gate").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    selector: pd.DataFrame,
    features_root: Path,
) -> None:
    def describe(label: str) -> str:
        row = metrics[metrics["label"].astype(str).eq(label)].iloc[0]
        return (
            f"RR R2={float(row['rr_r2']):.4f}, MAE={float(row['rr_mae']):.3f}, "
            f"RMSE={float(row['rr_rmse']):.3f}, exact={int(row['exact_count'])}/"
            f"{int(row['count_valid_videos'])}, >=2 errors={int(row['abs_count_error_ge2'])}"
        )

    safe = metrics[metrics["label"].astype(str).eq("safe_gate_baseline")].iloc[0]
    oof = metrics[metrics["label"].astype(str).eq("multi_roi_selector_stratified_oof")].iloc[0]
    grouped = metrics[metrics["label"].astype(str).eq("multi_roi_selector_prefix_group")].iloc[0]
    promoted = bool(
        int(oof["abs_count_error_ge2"]) == 0
        and int(grouped["abs_count_error_ge2"]) == 0
        and float(oof["rr_mae"]) < float(safe["rr_mae"])
        and float(grouped["rr_mae"]) <= float(safe["rr_mae"])
    )
    lines = [
        "# Multi-ROI Conservative Selector Probe",
        "",
        f"Precomputed feature source: `{features_root}`.",
        "",
        f"Feature coverage: `{int((coverage['status'] == 'available').sum())}/{len(coverage)}` videos.",
        "",
        "All respiratory-rate metrics were recomputed against the current yoloV8 truth "
        "table; no RR result from the source workspace was reused. Candidate selection "
        "was evaluated by held-video stratified folds and leave-one-prefix-group-out folds.",
        "",
        "## Results",
        "",
        f"Safe-gate baseline: {describe('safe_gate_baseline')}.",
        f"Multi-ROI held-video selector: {describe('multi_roi_selector_stratified_oof')}.",
        f"Multi-ROI prefix-group selector: {describe('multi_roi_selector_prefix_group')}.",
        f"Multi-ROI oracle feature-space bound: {describe('multi_roi_oracle_upper_bound')}.",
        "",
        f"Promotion gate: `{'PASS' if promoted else 'FAIL'}`.",
        "",
        f"Held-video selector switched away from the safe gate in "
        f"`{int((selector['strategy'] != 'safe_gate').sum())}` videos.",
        "",
        "## Interpretation",
        "",
        "A PASS result would justify porting the multi-ROI extractor into this repository "
        "and freezing the selector for external validation. A FAIL result means the "
        "precomputed feature space may remain useful for error analysis, but it does not "
        "replace the current safe gate. The source features are derived from the same 73 "
        "videos, so they are not external evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = input_root / f"{args.corrected_prefix}_paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(input_root, args.output_prefix)
    candidates, coverage = build_candidates(
        baseline,
        input_root,
        args.output_prefix,
        args.features_root.resolve(),
    )
    if candidates.empty:
        raise RuntimeError("No multi-ROI candidate rows were generated.")
    categorical, numerical = feature_columns(candidates)
    stratified_selected, stratified_audit = evaluate_folds(
        candidates,
        stratified_folds(candidates, args.random_state),
        categorical,
        numerical,
        args.random_state,
        args.switch_margin,
        "held_video_stratified_oof",
    )
    prefix_selected, prefix_audit = evaluate_folds(
        candidates,
        prefix_folds(candidates),
        categorical,
        numerical,
        args.random_state + 100,
        args.switch_margin,
        "leave_one_prefix_group_out",
    )
    safe = baseline_predictions(candidates)
    oracle = oracle_predictions(candidates)
    metrics = pd.DataFrame(
        [
            metric_row("safe_gate_baseline", safe, "existing_safe_gate_predictions"),
            metric_row(
                "multi_roi_selector_stratified_oof",
                stratified_selected,
                "held_video_random_forest_candidate_error_selector",
            ),
            metric_row(
                "multi_roi_selector_prefix_group",
                prefix_selected,
                "leave_one_prefix_group_out_candidate_error_selector",
            ),
            metric_row(
                "multi_roi_oracle_upper_bound",
                oracle,
                "truth_selected_multi_roi_candidate_upper_bound_only",
            ),
        ]
    )
    # Margin sensitivity must reuse fold-level candidate predictions, so reconstruct it
    # from all test candidate predictions in a second pass below.
    all_test_candidates: list[pd.DataFrame] = []
    video_ids = np.asarray(sorted(candidates["video_id"].unique()), dtype=str)
    for fold_index, (train_index, test_index, _) in enumerate(
        stratified_folds(candidates, args.random_state), start=1
    ):
        train_videos = set(video_ids[train_index])
        test_videos = set(video_ids[test_index])
        train = candidates[candidates["video_id"].isin(train_videos)].copy()
        test = candidates[candidates["video_id"].isin(test_videos)].copy()
        model = build_model(categorical, numerical, args.random_state + fold_index)
        model.fit(train[categorical + numerical], train["target_abs_count_error"].astype(float))
        test["predicted_abs_count_error"] = model.predict(test[categorical + numerical])
        all_test_candidates.append(test)
    margins = margin_grid(pd.concat(all_test_candidates, ignore_index=True))
    fold_audit = pd.concat([stratified_audit, prefix_audit], ignore_index=True)

    candidates_path = output_dir / "paper_multi_roi_selector_candidates.csv"
    coverage_path = output_dir / "paper_multi_roi_selector_feature_coverage.csv"
    predictions_path = output_dir / "paper_multi_roi_selector_predictions.csv"
    prefix_predictions_path = output_dir / "paper_multi_roi_selector_prefix_predictions.csv"
    metrics_path = output_dir / "paper_multi_roi_selector_metrics.csv"
    margin_path = output_dir / "paper_multi_roi_selector_margin_sensitivity.csv"
    folds_path = output_dir / "paper_multi_roi_selector_fold_audit.csv"
    report_path = output_dir / "paper_multi_roi_selector_report.md"
    candidates.to_csv(candidates_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    stratified_selected.to_csv(predictions_path, index=False)
    prefix_selected.to_csv(prefix_predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    margins.to_csv(margin_path, index=False)
    fold_audit.to_csv(folds_path, index=False)
    write_report(
        report_path,
        metrics,
        coverage,
        stratified_selected,
        args.features_root.resolve(),
    )
    print(f"Saved multi-ROI candidate table: {candidates_path}")
    print(f"Saved multi-ROI metrics: {metrics_path}")
    print(f"Saved multi-ROI report: {report_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
