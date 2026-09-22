"""Paired bootstrap comparison for the frozen 73-video adaptive-ROI stress test."""

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
        "--predictions", type=Path, default=assets / "paper73_adaptive_roi_predictions.csv"
    )
    parser.add_argument("--policy-id", default="adaptive_roi_damped_cv0.06")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return 1.0 - float(np.sum((prediction - truth) ** 2)) / denominator


def metrics(
    truth_rr: np.ndarray,
    predicted_rr: np.ndarray,
    count_error: np.ndarray,
) -> dict[str, float]:
    rr_error = predicted_rr - truth_rr
    return {
        "rr_r2": r2(truth_rr, predicted_rr),
        "rr_mae_bpm": float(np.mean(np.abs(rr_error))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(rr_error**2))),
        "exact_count_rate": float(np.mean(count_error == 0)),
        "within_one_count_rate": float(np.mean(np.abs(count_error) <= 1)),
        "count_error_ge_two_rate": float(np.mean(np.abs(count_error) >= 2)),
    }


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.predictions, dtype={"video_id": str})
    data = data.loc[data["policy_id"].eq(args.policy_id)].copy()
    if len(data) != 73 or data["video_id"].nunique() != 73:
        raise ValueError("Selected policy must contain exactly 73 unique videos")
    truth = data["truth_rr"].to_numpy(dtype=float)
    baseline = data["baseline_rr_bpm"].to_numpy(dtype=float)
    candidate = data["predicted_rr_bpm"].to_numpy(dtype=float)
    baseline_count_error = data["baseline_count_error"].to_numpy(dtype=float)
    candidate_count_error = data["count_error"].to_numpy(dtype=float)
    baseline_point = metrics(truth, baseline, baseline_count_error)
    candidate_point = metrics(truth, candidate, candidate_count_error)
    rng = np.random.default_rng(args.seed)
    samples = {name: [] for name in baseline_point}
    for _ in range(args.iterations):
        indices = rng.integers(0, len(data), size=len(data))
        baseline_sample = metrics(
            truth[indices], baseline[indices], baseline_count_error[indices]
        )
        candidate_sample = metrics(
            truth[indices], candidate[indices], candidate_count_error[indices]
        )
        for name in samples:
            delta = candidate_sample[name] - baseline_sample[name]
            if np.isfinite(delta):
                samples[name].append(delta)

    rows = []
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        lower, upper = np.percentile(array, [2.5, 97.5])
        rows.append(
            {
                "policy_id": args.policy_id,
                "metric": name,
                "baseline": baseline_point[name],
                "candidate": candidate_point[name],
                "delta_candidate_minus_baseline": candidate_point[name]
                - baseline_point[name],
                "delta_ci_lower": lower,
                "delta_ci_upper": upper,
                "bootstrap_iterations_valid": len(array),
                "seed": args.seed,
            }
        )
    comparison = pd.DataFrame(rows)
    cases = data.sort_values(
        ["abs_count_error_change", "video_id"], ascending=[False, True]
    ).reset_index(drop=True)
    changed = cases.loc[cases["prediction_changed"]].copy()
    report_lines = [
        "# Paper73 Adaptive-ROI Transfer Stress Test",
        "",
        f"Frozen policy: `{args.policy_id}`.",
        "",
        "This applies the policy developed on the fixed 30-second set to the authoritative 73-video short-clip baseline. It is an internal transfer stress test, not independent external validation.",
        "",
        "## Paired results",
        "",
    ]
    for row in comparison.itertuples(index=False):
        report_lines.append(
            f"- {row.metric}: {row.baseline:.6f} -> {row.candidate:.6f}; "
            f"delta {row.delta_candidate_minus_baseline:+.6f} "
            f"(paired bootstrap 95% CI {row.delta_ci_lower:+.6f} to {row.delta_ci_upper:+.6f})"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Changed videos: {len(changed)}; improved: {int((changed['abs_count_error_change'] < 0).sum())}; worsened: {int((changed['abs_count_error_change'] > 0).sum())}.",
            "",
            "The frozen adaptive-ROI policy does not transfer to the 73-video baseline and must not replace the fixed-radius main result.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "paper73_adaptive_roi_comparison_metrics.csv"
    cases_path = args.output_dir / "paper73_adaptive_roi_comparison_cases.csv"
    report_path = args.output_dir / "paper73_adaptive_roi_comparison.md"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
