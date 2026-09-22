from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_rr_duration_normalized_windowed_innovation import external_truth, metric_dict


METHODS = [
    "clipwise_no_context",
    "causal_running_median",
    "causal_running_quality_mean",
    "causal_current_prior_blend",
    "offline_session_median",
    "offline_session_quality_mean",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed unsupervised temporal source-session context for the frozen "
            "calibration-free thermal-color RR branch."
        )
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
    )
    parser.add_argument(
        "--p2g-gated-predictions-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_gated_predictions.csv",
    )
    parser.add_argument(
        "--consensus-predictions-csv",
        type=Path,
        default=assets / "paper_calibration_free_consensus_ensemble_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    return parser.parse_args()


def clip_index(video_id: str) -> int:
    match = re.search(r"_clip(\d+)$", str(video_id))
    return int(match.group(1)) if match else 0


def quality_by_video(predictions: pd.DataFrame) -> pd.DataFrame:
    members = predictions[
        predictions.get("cohort", pd.Series("", index=predictions.index))
        .astype(str)
        .eq("external_provisional")
        & predictions.get("member", pd.Series("", index=predictions.index)).notna()
    ].copy()
    if members.empty:
        return pd.DataFrame(columns=["video_id", "member_quality"])
    members["quality_mean"] = pd.to_numeric(members["quality_mean"], errors="coerce")
    quality = members.groupby("video_id", as_index=False)["quality_mean"].median()
    return quality.rename(columns={"quality_mean": "member_quality"})


def quality_weights(values: pd.Series) -> np.ndarray:
    quality = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weights = np.exp(np.clip(quality, -2.0, 2.0))
    weights[~np.isfinite(weights)] = 1.0
    return weights


def session_context(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_id, group in data.groupby("source_session_id", sort=False):
        ordered = group.sort_values("clip_index").copy()
        predictions = ordered["p2g_rr_bpm"].to_numpy(float)
        weights = quality_weights(ordered["member_quality"])
        full_median = float(np.median(predictions))
        full_weighted_mean = float(np.average(predictions, weights=weights))
        for position, row in enumerate(ordered.itertuples(index=False)):
            history = predictions[: position + 1]
            history_weights = weights[: position + 1]
            prior_mean = (
                float(np.average(predictions[:position], weights=weights[:position]))
                if position
                else float(predictions[position])
            )
            values = {
                "clipwise_no_context": float(predictions[position]),
                "causal_running_median": float(np.median(history)),
                "causal_running_quality_mean": float(
                    np.average(history, weights=history_weights)
                ),
                "causal_current_prior_blend": float(
                    0.5 * predictions[position] + 0.5 * prior_mean
                ),
                "offline_session_median": full_median,
                "offline_session_quality_mean": full_weighted_mean,
            }
            for method, rr_bpm in values.items():
                rows.append(
                    {
                        "video_id": str(row.video_id),
                        "source_session_id": str(session_id),
                        "clip_index": int(row.clip_index),
                        "method": method,
                        "session_context_rr_bpm": rr_bpm,
                        "p2g_rr_bpm": float(predictions[position]),
                        "member_quality": float(row.member_quality),
                    }
                )
    return pd.DataFrame(rows)


def metric_values(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    truth = frame["truth_rr"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
    }


def cluster_bootstrap(
    data: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = sorted(data["source_session_id"].astype(str).unique())
    rows = []
    for method, group in data.groupby("method", sort=False):
        if method == "clipwise_no_context":
            continue
        samples = {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
        for _ in range(int(resamples)):
            selected = rng.choice(groups, size=len(groups), replace=True)
            replicate = pd.concat(
                [group[group["source_session_id"].astype(str).eq(session)] for session in selected],
                ignore_index=True,
            )
            candidate = metric_values(replicate, "session_context_rr_bpm")
            baseline = metric_values(replicate, "p2g_rr_bpm")
            for metric in samples:
                samples[metric].append(candidate[metric] - baseline[metric])
        candidate_point = metric_values(group, "session_context_rr_bpm")
        baseline_point = metric_values(group, "p2g_rr_bpm")
        for metric, values in samples.items():
            rows.append(
                {
                    "comparison": f"{method}_minus_clipwise_p2g",
                    "metric": metric,
                    "estimate": candidate_point[metric] - baseline_point[metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "resamples": int(resamples),
                    "cluster_unit": "source_session_id",
                }
            )
    return pd.DataFrame(rows)


def score_methods(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in data.groupby("method", sort=False):
        row = metric_dict(
            "external_provisional",
            f"calibration_free_session_context_{method}",
            group["truth_rr"].to_numpy(float),
            group["session_context_rr_bpm"].to_numpy(float),
            group["truth_count"].to_numpy(float),
            group["truth_duration_seconds"].to_numpy(float),
            "fixed_unsupervised_source_session_context_no_external_rr_used_for_method_definition",
        )
        row.update(
            {
                "temporal_mode": "causal" if method.startswith("causal_") else "offline" if method.startswith("offline_") else "none",
                "method": method,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, metrics: pd.DataFrame, bootstrap: pd.DataFrame, data: pd.DataFrame) -> None:
    lines = [
        "# Calibration-Free Source-Session Context Probe",
        "",
        "Status: `fixed_unsupervised_external_development_sensitivity_probe`",
        "",
        "Each context rule is predefined and uses only the ordering, source-session identifier, "
        "frozen clipwise P2g prediction, and predicted curve quality. No external RR is used to "
        "define a rule. Offline rows use future clips and are therefore not real-time methods.",
        "",
        "## Metrics",
        "",
        "| method | temporal mode | R2 | MAE | RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.temporal_mode} | {row.rr_r2:.6f} | {row.rr_mae:.6f} | {row.rr_rmse:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Source-Session Cluster Bootstrap Against Clipwise P2g",
            "",
            "| comparison | metric | estimate | 95% CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.estimate:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Scored clips: `{len(data) // len(METHODS)}`; source sessions: `{data['source_session_id'].nunique()}`.",
            "This is not independent external validation because the current external manual "
            "reference is single-annotator and was inspected during development.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_csv.resolve())
    active = reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    reference = reference.loc[active].copy()
    reference = reference.rename(columns={"external_video_id": "video_id"})
    p2g = pd.read_csv(args.p2g_gated_predictions_csv.resolve())
    p2g = p2g[p2g["cohort"].astype(str).eq("external_provisional")][
        ["video_id", "duration_gated_rr_bpm"]
    ].rename(columns={"duration_gated_rr_bpm": "p2g_rr_bpm"})
    quality = quality_by_video(pd.read_csv(args.consensus_predictions_csv.resolve()))
    truth = external_truth(reference.rename(columns={"video_id": "external_video_id"}))
    base = p2g.merge(quality, on="video_id", how="left", validate="one_to_one")
    base = base.merge(
        reference[["video_id", "source_session_id"]],
        on="video_id",
        how="inner",
        validate="one_to_one",
    )
    base = base.merge(truth, on="video_id", how="inner", validate="one_to_one")
    base["clip_index"] = base["video_id"].map(clip_index)
    if len(base) != len(reference):
        raise ValueError(f"Expected {len(reference)} primary clips, found {len(base)} prediction/truth rows")
    context = session_context(base)
    scored = context.merge(
        base[["video_id", "truth_rr", "truth_count", "truth_duration_seconds"]],
        on="video_id",
        how="inner",
        validate="many_to_one",
    )
    metrics = score_methods(scored)
    bootstrap = cluster_bootstrap(scored, args.bootstrap_resamples, args.bootstrap_seed)
    session_metrics = (
        scored.groupby(["source_session_id", "method"], as_index=False)
        .agg(
            clips=("video_id", "size"),
            truth_rr_mean=("truth_rr", "mean"),
            rr_bpm_mean=("session_context_rr_bpm", "mean"),
            mean_abs_rr_error=("session_context_rr_bpm", lambda series: math.nan),
        )
    )
    session_metrics["mean_abs_rr_error"] = session_metrics.apply(
        lambda row: abs(float(row["rr_bpm_mean"]) - float(row["truth_rr_mean"])), axis=1
    )
    metrics.to_csv(output_dir / "paper_calibration_free_session_context_metrics.csv", index=False)
    scored.to_csv(output_dir / "paper_calibration_free_session_context_predictions.csv", index=False)
    session_metrics.to_csv(output_dir / "paper_calibration_free_session_context_session_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paper_calibration_free_session_context_bootstrap_ci.csv", index=False)
    write_report(output_dir / "paper_calibration_free_session_context_report.md", metrics, bootstrap, scored)
    print("Metrics:")
    print(metrics.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
