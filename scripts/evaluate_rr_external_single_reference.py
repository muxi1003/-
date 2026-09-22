from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    method: str
    filename: str
    rr_column: str
    count_column: str


METHODS = [
    MethodSpec(
        "external_default_pipeline",
        "{prefix}_summary.csv",
        "rr_bpm",
        "peaks",
    ),
    MethodSpec(
        "external_quality_residual_frozen",
        "paper_repro_quality_residual_predictions.csv",
        "corrected_rr_bpm",
        "corrected_peaks",
    ),
    MethodSpec(
        "external_signal_consensus_frozen",
        "{prefix}_signal_consensus_predictions.csv",
        "signal_consensus_rr_bpm",
        "signal_consensus_peaks",
    ),
    MethodSpec(
        "external_signal_aware_residual_frozen",
        "{prefix}_signal_aware_residual_predictions.csv",
        "signal_aware_final_rr_bpm",
        "signal_aware_final_peaks",
    ),
    MethodSpec(
        "external_signal_aware_safe_gate_frozen",
        "{prefix}_signal_aware_safe_policy_predictions.csv",
        "signal_aware_safe_final_rr_bpm",
        "signal_aware_safe_final_peaks",
    ),
]


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
            "Evaluate provisional external RR predictions against a single-annotator "
            "reference with one-count sensitivity and source-session cluster bootstrap."
        )
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument(
        "--reference-audit-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
    )
    parser.add_argument("--paper-assets-dir", type=Path, default=assets)
    parser.add_argument("--output-prefix", default="external_repro_single_reference")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def regression_metrics(
    truth_rr: np.ndarray,
    predicted_rr: np.ndarray,
    truth_count: np.ndarray,
    predicted_count: np.ndarray,
) -> dict[str, float | int]:
    rr_valid = np.isfinite(truth_rr) & np.isfinite(predicted_rr)
    count_valid = np.isfinite(truth_count) & np.isfinite(predicted_count)
    result: dict[str, float | int] = {
        "videos": int(len(truth_rr)),
        "rr_valid_videos": int(rr_valid.sum()),
        "count_valid_videos": int(count_valid.sum()),
    }
    if rr_valid.sum() >= 2:
        truth = truth_rr[rr_valid]
        predicted = predicted_rr[rr_valid]
        error = predicted - truth
        denominator = float(np.sum((truth - float(np.mean(truth))) ** 2))
        result["rr_r2"] = (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        )
        if float(np.std(truth)) > 0 and float(np.std(predicted)) > 0:
            result["rr_pearson_r2"] = float(np.corrcoef(truth, predicted)[0, 1] ** 2)
        else:
            result["rr_pearson_r2"] = float("nan")
        result["rr_mae"] = float(np.mean(np.abs(error)))
        result["rr_rmse"] = float(np.sqrt(np.mean(error**2)))
    else:
        for key in ["rr_r2", "rr_pearson_r2", "rr_mae", "rr_rmse"]:
            result[key] = float("nan")
    if count_valid.any():
        count_error = np.abs(predicted_count[count_valid] - truth_count[count_valid])
        result["count_mae"] = float(np.mean(count_error))
        result["exact_count"] = int(np.isclose(count_error, 0).sum())
        result["within_one_count"] = int((count_error <= 1 + 1e-9).sum())
        result["abs_count_error_ge2"] = int((count_error >= 2 - 1e-9).sum())
    else:
        result["count_mae"] = float("nan")
        result["exact_count"] = 0
        result["within_one_count"] = 0
        result["abs_count_error_ge2"] = 0
    return result


def load_methods(external_root: Path, prefix: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for spec in METHODS:
        path = external_root / spec.filename.format(prefix=prefix)
        if not path.exists():
            raise FileNotFoundError(f"Missing method predictions: {path}")
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"video_id", spec.rr_column, spec.count_column}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        table = data[["video_id", spec.rr_column, spec.count_column]].copy()
        table = table.rename(
            columns={spec.rr_column: "predicted_rr", spec.count_column: "predicted_count"}
        )
        table["method"] = spec.method
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def reference_scenario(data: pd.DataFrame, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    count_point = finite(data["manual_breath_count"])
    count_low = finite(data["breath_count_lower"])
    count_high = finite(data["breath_count_upper"])
    rr_point = finite(data["manual_rr_bpm_point"])
    rr_low = finite(data["manual_rr_lower_bpm"])
    rr_high = finite(data["manual_rr_upper_bpm"])
    predicted_rr = finite(data["predicted_rr"])
    candidates_rr = np.vstack([rr_low, rr_point, rr_high])
    candidates_count = np.vstack([count_low, count_point, count_high])
    if scenario == "point_reference":
        return rr_point, count_point
    if scenario == "all_counts_minus_one":
        return rr_low, count_low
    if scenario == "all_counts_plus_one":
        return rr_high, count_high
    distances = np.abs(candidates_rr - predicted_rr[np.newaxis, :])
    if scenario == "rowwise_nearest_within_one_count":
        selected = np.nanargmin(distances, axis=0)
    elif scenario == "rowwise_farthest_within_one_count":
        selected = np.nanargmax(distances, axis=0)
    else:
        raise ValueError(f"Unknown reference scenario: {scenario}")
    columns = np.arange(len(data))
    return candidates_rr[selected, columns], candidates_count[selected, columns]


def build_sensitivity(merged: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "point_reference",
        "all_counts_minus_one",
        "all_counts_plus_one",
        "rowwise_nearest_within_one_count",
        "rowwise_farthest_within_one_count",
    ]
    subsets = {
        "primary_ge20s": merged["primary_analysis_include"].astype(str).str.lower().eq("true"),
        "strict_ge29s": (
            merged["primary_analysis_include"].astype(str).str.lower().eq("true")
            & (pd.to_numeric(merged["manual_duration_seconds"], errors="coerce") >= 29.0)
        ),
    }
    rows: list[dict[str, object]] = []
    for method, method_data in merged.groupby("method", sort=False):
        for subset_name, subset_mask in subsets.items():
            data = method_data.loc[subset_mask.loc[method_data.index]].reset_index(drop=True)
            for scenario in scenarios:
                truth_rr, truth_count = reference_scenario(data, scenario)
                metrics = regression_metrics(
                    truth_rr,
                    finite(data["predicted_rr"]),
                    truth_count,
                    finite(data["predicted_count"]),
                )
                rows.append(
                    {
                        "method": method,
                        "analysis_subset": subset_name,
                        "reference_scenario": scenario,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    data: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    clusters = data["source_session_id"].astype(str).unique()
    cluster_indices = {
        cluster: np.flatnonzero(data["source_session_id"].astype(str).to_numpy() == cluster)
        for cluster in clusters
    }
    sampled: dict[str, list[float]] = {
        key: [] for key in ["rr_r2", "rr_pearson_r2", "rr_mae", "rr_rmse", "count_mae"]
    }
    truth_rr = finite(data["manual_rr_bpm_point"])
    predicted_rr = finite(data["predicted_rr"])
    truth_count = finite(data["manual_breath_count"])
    predicted_count = finite(data["predicted_count"])
    for _ in range(reps):
        selected_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in selected_clusters])
        metrics = regression_metrics(
            truth_rr[indices],
            predicted_rr[indices],
            truth_count[indices],
            predicted_count[indices],
        )
        for key in sampled:
            value = float(metrics[key])
            if np.isfinite(value):
                sampled[key].append(value)
    return {
        key: (
            float(np.percentile(values, 2.5)) if values else float("nan"),
            float(np.percentile(values, 97.5)) if values else float("nan"),
        )
        for key, values in sampled.items()
    }


def build_bootstrap(
    merged: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_index, (method, method_data) in enumerate(merged.groupby("method", sort=False)):
        data = method_data[
            method_data["primary_analysis_include"].astype(str).str.lower().eq("true")
        ].reset_index(drop=True)
        point = regression_metrics(
            finite(data["manual_rr_bpm_point"]),
            finite(data["predicted_rr"]),
            finite(data["manual_breath_count"]),
            finite(data["predicted_count"]),
        )
        intervals = cluster_bootstrap(
            data,
            reps=reps,
            rng=np.random.default_rng(seed + method_index),
        )
        for metric, (low, high) in intervals.items():
            rows.append(
                {
                    "method": method,
                    "analysis_subset": "primary_ge20s",
                    "reference_scenario": "point_reference",
                    "cluster_unit": "source_session_id",
                    "clusters": int(data["source_session_id"].nunique()),
                    "videos": len(data),
                    "bootstrap_reps": reps,
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_2_5": low,
                    "ci_97_5": high,
                }
            )
    return pd.DataFrame(rows)


def build_session_aggregation(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    session_tables: list[pd.DataFrame] = []
    for method, method_data in merged.groupby("method", sort=False):
        data = method_data[
            method_data["primary_analysis_include"].astype(str).str.lower().eq("true")
        ].copy()
        for column in [
            "manual_breath_count",
            "manual_duration_seconds",
            "predicted_count",
        ]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        grouped = (
            data.groupby("source_session_id", as_index=False)
            .agg(
                cow_id=("cow_id", "first"),
                clips=("video_id", "size"),
                duration_seconds=("manual_duration_seconds", "sum"),
                truth_count=("manual_breath_count", "sum"),
                predicted_count=("predicted_count", "sum"),
                manual_count_min=("manual_breath_count", "min"),
                manual_count_max=("manual_breath_count", "max"),
            )
        )
        grouped["truth_rr"] = grouped["truth_count"] * 60.0 / grouped["duration_seconds"]
        grouped["predicted_rr"] = (
            grouped["predicted_count"] * 60.0 / grouped["duration_seconds"]
        )
        grouped["manual_count_range"] = (
            grouped["manual_count_max"] - grouped["manual_count_min"]
        )
        grouped["method"] = method
        session_tables.append(grouped)
        metrics = regression_metrics(
            grouped["truth_rr"].to_numpy(dtype=float),
            grouped["predicted_rr"].to_numpy(dtype=float),
            grouped["truth_count"].to_numpy(dtype=float),
            grouped["predicted_count"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "method": method,
                "analysis_unit": "source_session_id",
                "reference_scenario": "point_reference",
                **metrics,
            }
        )
    return pd.DataFrame(metric_rows), pd.concat(session_tables, ignore_index=True)


def metric_value(
    metrics: pd.DataFrame,
    method: str,
    subset: str,
    scenario: str,
    column: str,
) -> float:
    row = metrics[
        metrics["method"].eq(method)
        & metrics["analysis_subset"].eq(subset)
        & metrics["reference_scenario"].eq(scenario)
    ]
    return float(row.iloc[0][column])


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    merged: pd.DataFrame,
    session_metrics: pd.DataFrame,
) -> None:
    method = "external_default_pipeline"
    point_r2 = metric_value(metrics, method, "primary_ge20s", "point_reference", "rr_r2")
    point_mae = metric_value(metrics, method, "primary_ge20s", "point_reference", "rr_mae")
    optimistic_r2 = metric_value(
        metrics, method, "primary_ge20s", "rowwise_nearest_within_one_count", "rr_r2"
    )
    conservative_r2 = metric_value(
        metrics, method, "primary_ge20s", "rowwise_farthest_within_one_count", "rr_r2"
    )
    strict_r2 = metric_value(metrics, method, "strict_ge29s", "point_reference", "rr_r2")
    session_row = session_metrics[session_metrics["method"].eq(method)].iloc[0]
    ci = bootstrap[
        bootstrap["method"].eq(method) & bootstrap["metric"].eq("rr_r2")
    ].iloc[0]
    primary = merged[
        merged["method"].eq(method)
        & merged["primary_analysis_include"].astype(str).str.lower().eq("true")
    ]
    lines = [
        "# Provisional Single-Annotator External RR Evaluation",
        "",
        "Status: `external_domain_failure_under_provisional_reference`",
        "",
        "## Main Result",
        "",
        f"The frozen default pipeline was evaluated on `{len(primary)}` clips from "
        f"`{primary['source_session_id'].nunique()}` source sessions. Against the point "
        f"single-annotator reference, RR R2 was `{point_r2:.6f}` and MAE was "
        f"`{point_mae:.6f}` bpm. The source-session cluster-bootstrap 95% interval for "
        f"R2 was `[{float(ci['ci_2_5']):.6f}, {float(ci['ci_97_5']):.6f}]`.",
        "",
        "## Reference-Uncertainty Sensitivity",
        "",
        f"Allowing every manual count to vary by one breath produced a rowwise optimistic "
        f"R2 of `{optimistic_r2:.6f}` and a rowwise conservative R2 of "
        f"`{conservative_r2:.6f}`. Restricting the point-reference analysis to clips of "
        f"at least 29 s gave R2 `{strict_r2:.6f}`. These scenarios do not restore acceptable "
        "external performance.",
        "",
        "## Session Aggregation",
        "",
        f"After summing counts and durations within each source long video, the analysis "
        f"contained `{int(session_row['videos'])}` independent sessions. Session-level R2 "
        f"was `{float(session_row['rr_r2']):.6f}` and MAE was "
        f"`{float(session_row['rr_mae']):.6f}` bpm. Aggregation reduces clip-level count "
        "quantization but does not recover acceptable external performance.",
        "",
        "## Interpretation Boundary",
        "",
        "This is a provisional single-annotator analysis, not confirmatory consensus truth. "
        "However, the failure is too large to be explained by one-count uncertainty alone. "
        "The result should be used as evidence of external-domain mismatch and to motivate "
        "domain-shift diagnostics, not for tuning the frozen method on these labels.",
        "",
        "A second independent blinded count and adjudication remain required before final "
        "paper reporting. Environment and visual-quality stratification remain unavailable.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be at least 100")
    external_root = args.external_root.resolve()
    reference = pd.read_csv(args.reference_audit_csv.resolve(), dtype=str, keep_default_na=False)
    required_reference = {
        "external_video_id",
        "source_session_id",
        "manual_breath_count",
        "manual_duration_seconds",
        "manual_rr_bpm_point",
        "breath_count_lower",
        "breath_count_upper",
        "manual_rr_lower_bpm",
        "manual_rr_upper_bpm",
        "primary_analysis_include",
    }
    missing = required_reference - set(reference.columns)
    if missing:
        raise ValueError(f"Reference audit missing columns: {sorted(missing)}")
    predictions = load_methods(external_root, args.output_prefix)
    merged = predictions.merge(
        reference,
        left_on="video_id",
        right_on="external_video_id",
        how="left",
        validate="many_to_one",
    )
    if merged["external_video_id"].eq("").any() or merged["external_video_id"].isna().any():
        missing_ids = merged.loc[
            merged["external_video_id"].eq("") | merged["external_video_id"].isna(),
            "video_id",
        ].unique()
        raise ValueError(f"Predictions without reference rows: {missing_ids[:10].tolist()}")

    metrics = build_sensitivity(merged)
    bootstrap = build_bootstrap(merged, reps=args.bootstrap_reps, seed=args.seed)
    session_metrics, session_predictions = build_session_aggregation(merged)
    rows_path = external_root / f"{args.output_prefix}_single_reference_predictions.csv"
    metrics_path = external_root / f"{args.output_prefix}_reference_sensitivity_metrics.csv"
    bootstrap_path = external_root / f"{args.output_prefix}_cluster_bootstrap_ci.csv"
    session_metrics_path = external_root / f"{args.output_prefix}_session_aggregated_metrics.csv"
    session_predictions_path = external_root / f"{args.output_prefix}_session_aggregated_predictions.csv"
    report_path = external_root / f"{args.output_prefix}_single_reference_evaluation.md"
    paper_assets_dir = args.paper_assets_dir.resolve()
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(rows_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    session_metrics.to_csv(session_metrics_path, index=False)
    session_predictions.to_csv(session_predictions_path, index=False)
    write_report(report_path, metrics, bootstrap, merged, session_metrics)
    metrics.to_csv(
        paper_assets_dir / "paper_external_validation_single_reference_metrics.csv",
        index=False,
    )
    bootstrap.to_csv(
        paper_assets_dir
        / "paper_external_validation_single_reference_cluster_bootstrap_ci.csv",
        index=False,
    )
    session_metrics.to_csv(
        paper_assets_dir / "paper_external_validation_single_reference_session_metrics.csv",
        index=False,
    )
    write_report(
        paper_assets_dir / "paper_external_validation_single_reference_evaluation.md",
        metrics,
        bootstrap,
        merged,
        session_metrics,
    )

    default_point = metrics[
        metrics["method"].eq("external_default_pipeline")
        & metrics["analysis_subset"].eq("primary_ge20s")
        & metrics["reference_scenario"].eq("point_reference")
    ].iloc[0]
    print(f"Saved reference sensitivity metrics: {metrics_path}")
    print(f"Saved cluster bootstrap intervals: {bootstrap_path}")
    print(f"Saved session-aggregated metrics: {session_metrics_path}")
    print(f"Saved provisional evaluation report: {report_path}")
    print(f"default_rr_r2={float(default_point['rr_r2']):.6f}")
    print(f"default_rr_mae={float(default_point['rr_mae']):.6f}")


if __name__ == "__main__":
    main()
