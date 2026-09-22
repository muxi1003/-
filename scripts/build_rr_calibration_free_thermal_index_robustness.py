from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Audit reference uncertainty and clustered robustness for the internally "
            "selected calibration-free thermal-color RR estimator."
        )
    )
    parser.add_argument("--assets-dir", type=Path, default=assets)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def metric_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    y = truth[valid]
    p = prediction[valid]
    error = p - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "videos": int(valid.sum()),
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
    }


def load_data(assets: Path) -> pd.DataFrame:
    gated = pd.read_csv(assets / "paper_calibration_free_thermal_index_gated_predictions.csv")
    reference = pd.read_csv(assets / "paper_external_validation_single_reference_rows.csv")
    data = gated[gated["cohort"].astype(str).eq("external_provisional")].copy()
    reference = reference.rename(columns={"external_video_id": "video_id"}).copy()
    if "raw_video_path" not in reference.columns:
        reference["raw_video_path"] = ""
    reference = reference[
        ["video_id", "raw_video_path", "cow_id", "source_session_id", "collection_date"]
    ]
    fieldwork = assets / "paper_external_validation_split_all_use_fieldwork_template.csv"
    if fieldwork.exists():
        paths = pd.read_csv(fieldwork)
        if {"external_video_id", "raw_video_path"}.issubset(paths.columns):
            paths = paths[["external_video_id", "raw_video_path"]].rename(
                columns={"external_video_id": "video_id", "raw_video_path": "template_raw_video_path"}
            )
            reference = reference.merge(paths, on="video_id", how="left", validate="one_to_one")
            missing_path = reference["raw_video_path"].isna() | reference["raw_video_path"].eq("")
            reference.loc[missing_path, "raw_video_path"] = reference.loc[
                missing_path, "template_raw_video_path"
            ]
            reference = reference.drop(columns=["template_raw_video_path"])
    data = data.merge(reference, on="video_id", how="left", validate="one_to_one")
    for column in [
        "truth_rr",
        "truth_count",
        "truth_duration_seconds",
        "duration_gated_rr_bpm",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def one_count_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    prediction = data["duration_gated_rr_bpm"].to_numpy(float)
    count = data["truth_count"].to_numpy(float)
    duration = data["truth_duration_seconds"].to_numpy(float)
    candidates = np.column_stack(
        [(count + offset) * 60.0 / duration for offset in [-1.0, 0.0, 1.0]]
    )
    candidates = np.where(candidates > 0, candidates, np.nan)
    squared_error = (candidates - prediction[:, None]) ** 2
    optimistic = candidates[np.arange(len(data)), np.nanargmin(squared_error, axis=1)]
    conservative = candidates[np.arange(len(data)), np.nanargmax(squared_error, axis=1)]
    rows = []
    for label, truth, note in [
        ("point_single_annotator", data["truth_rr"].to_numpy(float), "recorded count"),
        (
            "one_count_optimistic_rowwise",
            optimistic,
            "each count may move by one breath toward prediction; sensitivity only",
        ),
        (
            "one_count_conservative_rowwise",
            conservative,
            "each count may move by one breath away from prediction; sensitivity only",
        ),
    ]:
        rows.append({"analysis": label, **metric_dict(truth, prediction), "note": note})
    long = data[data["truth_duration_seconds"].ge(29.0)]
    rows.append(
        {
            "analysis": "point_duration_at_least_29_seconds",
            **metric_dict(
                long["truth_rr"].to_numpy(float), long["duration_gated_rr_bpm"].to_numpy(float)
            ),
            "note": "reduces count quantization from short clips",
        }
    )
    return pd.DataFrame(rows)


def aggregate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in data.groupby("source_session_id", sort=True):
        duration = group["truth_duration_seconds"].to_numpy(float)
        rows.append(
            {
                "source_session_id": session,
                "clips": len(group),
                "total_duration_seconds": float(np.sum(duration)),
                "truth_count": float(np.sum(group["truth_count"])),
                "truth_rr": float(np.sum(group["truth_count"]) * 60.0 / np.sum(duration)),
                "rr_bpm": float(
                    np.average(group["duration_gated_rr_bpm"].to_numpy(float), weights=duration)
                ),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap(data: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    groups = sorted(data["source_session_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat(
            [data[data["source_session_id"].astype(str).eq(group)] for group in sampled],
            ignore_index=True,
        )
        metrics = metric_dict(
            replicate["truth_rr"].to_numpy(float),
            replicate["duration_gated_rr_bpm"].to_numpy(float),
        )
        for metric in values:
            values[metric].append(metrics[metric])
    point = metric_dict(
        data["truth_rr"].to_numpy(float), data["duration_gated_rr_bpm"].to_numpy(float)
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "resamples": int(resamples),
                "cluster_unit": "source_session_id",
                "clusters": len(groups),
            }
            for metric, samples in values.items()
        ]
    )


def date_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in data.groupby("collection_date", sort=True):
        rows.append(
            {
                "collection_date": date,
                "sessions": int(group["source_session_id"].nunique()),
                **metric_dict(
                    group["truth_rr"].to_numpy(float),
                    group["duration_gated_rr_bpm"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def reannotation_priority(data: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    session_error = sessions.assign(
        session_abs_rr_error=lambda frame: np.abs(frame["rr_bpm"] - frame["truth_rr"])
    )[["source_session_id", "session_abs_rr_error", "clips"]]
    review = data.merge(session_error, on="source_session_id", how="left", validate="many_to_one")
    review["abs_rr_error_bpm"] = np.abs(
        review["duration_gated_rr_bpm"] - review["truth_rr"]
    )
    predicted_count = np.rint(
        review["duration_gated_rr_bpm"] * review["truth_duration_seconds"] / 60.0
    )
    review["predicted_breath_count"] = predicted_count
    review["abs_count_error"] = np.abs(predicted_count - review["truth_count"])

    def priority(row: pd.Series) -> str:
        if row["session_abs_rr_error"] >= 7.0 or row["abs_rr_error_bpm"] >= 8.0:
            return "P0_dual_blinded_recount_and_adjudication"
        if row["session_abs_rr_error"] >= 4.0 or row["abs_rr_error_bpm"] >= 5.0:
            return "P1_second_independent_recount"
        return "P2_retain_for_random_quality_control"

    review["review_priority"] = review.apply(priority, axis=1)
    review["review_reason"] = (
        "P2g absolute RR/count discrepancy; distinguish model failure from single-annotator "
        "reference uncertainty before using this clip in confirmatory external validation"
    )
    columns = [
        "review_priority",
        "video_id",
        "raw_video_path",
        "cow_id",
        "collection_date",
        "source_session_id",
        "truth_count",
        "truth_duration_seconds",
        "truth_rr",
        "duration_gated_rr_bpm",
        "predicted_breath_count",
        "abs_count_error",
        "abs_rr_error_bpm",
        "session_abs_rr_error",
        "clips",
        "review_reason",
    ]
    return review[columns].sort_values(
        ["review_priority", "session_abs_rr_error", "abs_rr_error_bpm"],
        ascending=[True, False, False],
    )


def write_report(
    path: Path,
    sensitivity: pd.DataFrame,
    session: pd.DataFrame,
    dates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    reannotation: pd.DataFrame,
) -> None:
    point = sensitivity[sensitivity["analysis"].eq("point_single_annotator")].iloc[0]
    optimistic = sensitivity[
        sensitivity["analysis"].eq("one_count_optimistic_rowwise")
    ].iloc[0]
    conservative = sensitivity[
        sensitivity["analysis"].eq("one_count_conservative_rowwise")
    ].iloc[0]
    session_metric = metric_dict(session["truth_rr"].to_numpy(float), session["rr_bpm"].to_numpy(float))
    r2 = bootstrap[bootstrap["metric"].eq("rr_r2")].iloc[0]
    lines = [
        "# Calibration-Free Thermal-Color Robustness Audit",
        "",
        "Status: `provisional_single_annotator_robustness_audit_not_confirmatory_validation`",
        "",
        "The 94 clips use one manual annotator. The +/-1 analyses are reference-uncertainty "
        "sensitivity bounds, not alternative ground truth. Source sessions are the bootstrap "
        "unit because adjacent clips originate from one long recording.",
        "",
        "## Clip-Level Point and Count-Uncertainty Sensitivity",
        "",
        f"Point reference: R2 `{point['rr_r2']:.6f}`, MAE `{point['rr_mae']:.6f}`, RMSE `{point['rr_rmse']:.6f}` bpm.",
        f"Optimistic +/-1 count sensitivity: R2 `{optimistic['rr_r2']:.6f}`, MAE `{optimistic['rr_mae']:.6f}`, RMSE `{optimistic['rr_rmse']:.6f}` bpm.",
        f"Conservative +/-1 count sensitivity: R2 `{conservative['rr_r2']:.6f}`, MAE `{conservative['rr_mae']:.6f}`, RMSE `{conservative['rr_rmse']:.6f}` bpm.",
        "",
        "## Clustered and Session-Level Views",
        "",
        f"Source-session cluster bootstrap for clip-level R2: `{r2['estimate']:.6f}` (95% CI `{r2['ci_low']:.6f}` to `{r2['ci_high']:.6f}`; `{int(r2['clusters'])}` sessions).",
        f"After duration-weighted session aggregation: R2 `{session_metric['rr_r2']:.6f}`, MAE `{session_metric['rr_mae']:.6f}`, RMSE `{session_metric['rr_rmse']:.6f}` bpm across `{len(session)}` sessions.",
        "",
        "## Date-Stratified Descriptive Results",
        "",
        "| date | clips | sessions | R2 | MAE | RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dates.itertuples(index=False):
        lines.append(
            f"| {row.collection_date} | {row.videos} | {row.sessions} | {row.rr_r2:.6f} | {row.rr_mae:.6f} | {row.rr_rmse:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reannotation Priority",
            "",
            f"The review queue contains `{int((reannotation['review_priority'] == 'P0_dual_blinded_recount_and_adjudication').sum())}` P0 clips and "
            f"`{int((reannotation['review_priority'] == 'P1_second_independent_recount').sum())}` P1 clips. Priority reflects disagreement, not a conclusion that the model or the manual reference is wrong.",
            "",
            "Date rows are descriptive because clips are not independent and dates were "
            "not an external holdout selected before development. The analysis does not "
            "convert the provisional point reference into confirmatory external validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    assets = args.assets_dir.resolve()
    data = load_data(assets)
    sensitivity = one_count_sensitivity(data)
    sessions = aggregate_sessions(data)
    session_metrics = pd.DataFrame(
        [
            {
                "analysis": "duration_weighted_source_session_aggregation",
                **metric_dict(sessions["truth_rr"].to_numpy(float), sessions["rr_bpm"].to_numpy(float)),
                "note": "sum counts and duration within source long video",
            }
        ]
    )
    dates = date_metrics(data)
    bootstrap = cluster_bootstrap(data, args.bootstrap_resamples, args.seed)
    reannotation = reannotation_priority(data, sessions)
    sensitivity = pd.concat([sensitivity, session_metrics], ignore_index=True)
    sensitivity_path = assets / "paper_calibration_free_thermal_index_robustness_metrics.csv"
    session_path = assets / "paper_calibration_free_thermal_index_session_metrics.csv"
    date_path = assets / "paper_calibration_free_thermal_index_date_metrics.csv"
    bootstrap_path = assets / "paper_calibration_free_thermal_index_robustness_bootstrap_ci.csv"
    report_path = assets / "paper_calibration_free_thermal_index_robustness_report.md"
    reannotation_path = assets / "paper_calibration_free_thermal_index_reannotation_priority.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    sessions.to_csv(session_path, index=False)
    dates.to_csv(date_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    reannotation.to_csv(reannotation_path, index=False)
    write_report(report_path, sensitivity, sessions, dates, bootstrap, reannotation)
    print(f"Saved robustness metrics: {sensitivity_path}")
    print(f"Saved session metrics: {session_path}")
    print(f"Saved date metrics: {date_path}")
    print(f"Saved robustness bootstrap CI: {bootstrap_path}")
    print(f"Saved robustness report: {report_path}")
    print(f"Saved reannotation priority queue: {reannotation_path}")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
