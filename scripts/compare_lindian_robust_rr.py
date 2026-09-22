"""Paired comparison of frozen baseline and robust Lindian RR predictions."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=assets / "lindian_anchored30_rr_predictions.csv"
    )
    parser.add_argument(
        "--robust", type=Path, default=assets / "lindian_anchored30_robust_rr_predictions.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument(
        "--comparison-prefix",
        default="lindian_anchored30_robust_comparison",
        help="Filename prefix for paired comparison outputs.",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return np.nan if denominator <= np.finfo(float).eps else 1.0 - float(
        np.sum((prediction - truth) ** 2)
    ) / denominator


def metrics(truth: np.ndarray, prediction: np.ndarray, count_error: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "rr_r2": r2(truth, prediction),
        "rr_mae_bpm": float(np.mean(np.abs(error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(error**2))),
        "exact_count_rate": float(np.mean(count_error == 0)),
        "within_one_count_rate": float(np.mean(np.abs(count_error) <= 1)),
        "count_error_ge_two_rate": float(np.mean(np.abs(count_error) >= 2)),
    }


def bootstrap_delta(
    table: pd.DataFrame,
    *,
    analysis_set: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = table["manual_rr_bpm"].to_numpy(dtype=float)
    baseline = table["baseline_rr_bpm"].to_numpy(dtype=float)
    robust = table["robust_rr_bpm"].to_numpy(dtype=float)
    baseline_count_error = table["baseline_count_error"].to_numpy(dtype=float)
    robust_count_error = table["robust_count_error"].to_numpy(dtype=float)
    baseline_point = metrics(truth, baseline, baseline_count_error)
    robust_point = metrics(truth, robust, robust_count_error)
    names = tuple(baseline_point)
    samples = {name: [] for name in names}
    for _ in range(iterations):
        indices = rng.integers(0, len(table), size=len(table))
        baseline_sample = metrics(
            truth[indices], baseline[indices], baseline_count_error[indices]
        )
        robust_sample = metrics(
            truth[indices], robust[indices], robust_count_error[indices]
        )
        for name in names:
            delta = robust_sample[name] - baseline_sample[name]
            if np.isfinite(delta):
                samples[name].append(delta)

    rows = []
    for name in names:
        values = np.asarray(samples[name], dtype=float)
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "analysis_set": analysis_set,
                "metric": name,
                "baseline": baseline_point[name],
                "robust": robust_point[name],
                "delta_robust_minus_baseline": robust_point[name] - baseline_point[name],
                "delta_ci_lower": lower,
                "delta_ci_upper": upper,
                "bootstrap_iterations_valid": len(values),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def build_report(comparison: pd.DataFrame, cases: pd.DataFrame) -> str:
    lines = [
        "# Lindian Robust RR Paired Comparison",
        "",
        "The robust policy is truth-independent at inference time, but its development used this dataset. These results are internal development evidence, not confirmatory external validation.",
        "",
    ]
    for analysis_set in ("primary_completed", "sensitivity_all_numeric"):
        rows = comparison.loc[comparison["analysis_set"].eq(analysis_set)]
        lines.extend([f"## {analysis_set}", ""])
        for metric in ("rr_r2", "rr_mae_bpm", "rr_rmse_bpm"):
            row = rows.loc[rows["metric"].eq(metric)].iloc[0]
            lines.append(
                f"- {metric}: {row['baseline']:.4f} -> {row['robust']:.4f}; "
                f"delta {row['delta_robust_minus_baseline']:+.4f} "
                f"(paired bootstrap 95% CI {row['delta_ci_lower']:+.4f} to {row['delta_ci_upper']:+.4f})"
            )
        lines.append("")
    improved = int((cases["abs_count_error_change"] < 0).sum())
    worsened = int((cases["abs_count_error_change"] > 0).sum())
    unchanged = int((cases["abs_count_error_change"] == 0).sum())
    lines.extend(
        [
            "## Case changes",
            "",
            f"- Improved videos: {improved}",
            f"- Worsened videos: {worsened}",
            f"- Unchanged videos: {unchanged}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline, dtype={"video_id": str})
    robust = pd.read_csv(args.robust, dtype={"video_id": str})
    keep = [
        "video_id",
        "manual_rr_bpm",
        "manual_breath_count",
        "include_primary_analysis",
        "truth_reliability",
        "predicted_rr_bpm",
        "predicted_breath_count",
        "count_error",
        "selected_fusion_mode",
    ]
    baseline = baseline[[column for column in keep if column in baseline.columns]].rename(
        columns={
            "predicted_rr_bpm": "baseline_rr_bpm",
            "predicted_breath_count": "baseline_breath_count",
            "count_error": "baseline_count_error",
            "selected_fusion_mode": "baseline_fusion_mode",
        }
    )
    robust = robust[
        [
            "video_id",
            "predicted_rr_bpm",
            "predicted_breath_count",
            "count_error",
            "selected_fusion_mode",
            "fusion_rescue_applied",
            "source_rejected_peak_count",
            "source_rejected_peak_frames",
        ]
    ].rename(
        columns={
            "predicted_rr_bpm": "robust_rr_bpm",
            "predicted_breath_count": "robust_breath_count",
            "count_error": "robust_count_error",
            "selected_fusion_mode": "robust_fusion_mode",
        }
    )
    paired = baseline.merge(robust, on="video_id", how="inner", validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(robust):
        raise ValueError("Baseline and robust video IDs do not match")
    paired["include_primary_analysis"] = (
        paired["include_primary_analysis"].astype(str).str.lower().eq("true")
    )
    paired["baseline_abs_count_error"] = paired["baseline_count_error"].abs()
    paired["robust_abs_count_error"] = paired["robust_count_error"].abs()
    paired["abs_count_error_change"] = (
        paired["robust_abs_count_error"] - paired["baseline_abs_count_error"]
    )
    paired["prediction_changed"] = (
        paired["baseline_breath_count"] != paired["robust_breath_count"]
    )

    comparisons = []
    for index, (analysis_set, table) in enumerate(
        (
            ("primary_completed", paired.loc[paired["include_primary_analysis"]]),
            ("sensitivity_all_numeric", paired),
        )
    ):
        comparisons.append(
            bootstrap_delta(
                table,
                analysis_set=analysis_set,
                iterations=args.iterations,
                seed=args.seed + index,
            )
        )
    comparison = pd.concat(comparisons, ignore_index=True)
    cases = paired.sort_values(
        ["abs_count_error_change", "video_id"], ascending=[True, True]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / f"{args.comparison_prefix}_metrics.csv"
    cases_path = args.output_dir / f"{args.comparison_prefix}_cases.csv"
    report_path = args.output_dir / f"{args.comparison_prefix}.md"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(comparison, cases), encoding="utf-8")
    print(f"Saved comparison metrics: {comparison_path.resolve()}")
    print(f"Saved comparison cases: {cases_path.resolve()}")
    print(f"Saved comparison report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
