"""Validate global fixed nostril ROI radii on the authoritative 73-video set."""

from __future__ import annotations

import argparse
import re
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
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--reference-radius", type=int, default=20)
    return parser.parse_args()


def radius_from_policy(policy_id: str) -> int | None:
    if policy_id == "fixed_radius20_reextracted":
        return 20
    match = re.fullmatch(r"fixed_radius(\d+)_global", policy_id)
    return int(match.group(1)) if match else None


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return np.nan if denominator <= np.finfo(float).eps else 1.0 - float(
        np.sum((prediction - truth) ** 2)
    ) / denominator


def metric_row(table: pd.DataFrame, radius: int) -> dict[str, object]:
    truth_rr = table["truth_rr"].to_numpy(dtype=float)
    predicted_rr = table["predicted_rr_bpm"].to_numpy(dtype=float)
    count_error = table["count_error"].to_numpy(dtype=float)
    return {
        "radius": radius,
        "videos": len(table),
        "rr_r2": r2(truth_rr, predicted_rr),
        "rr_mae_bpm": float(np.mean(np.abs(predicted_rr - truth_rr))),
        "rr_rmse_bpm": float(np.sqrt(np.mean((predicted_rr - truth_rr) ** 2))),
        "exact_count": int(np.sum(count_error == 0)),
        "within_one_count": int(np.sum(np.abs(count_error) <= 1)),
        "count_error_ge_two": int(np.sum(np.abs(count_error) >= 2)),
    }


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.predictions, dtype={"video_id": str})
    data["radius"] = data["policy_id"].map(radius_from_policy)
    fixed = data.loc[data["radius"].notna()].copy()
    fixed["radius"] = fixed["radius"].astype(int)
    expected_radii = set(range(14, 29))
    if set(fixed["radius"]) != expected_radii:
        raise ValueError("Fixed-radius predictions must cover every radius from 14 to 28")
    if fixed.groupby("radius")["video_id"].nunique().ne(73).any():
        raise ValueError("Every fixed radius must contain 73 unique videos")

    grid = pd.DataFrame(
        [metric_row(group, int(radius)) for radius, group in fixed.groupby("radius")]
    ).sort_values("radius")
    reference = fixed.loc[fixed["radius"].eq(args.reference_radius)].set_index("video_id")
    changed_rows = []
    for radius, group in fixed.groupby("radius"):
        current = group.set_index("video_id")
        changed_rows.append(
            {
                "radius": int(radius),
                "changed_videos_vs_radius20": int(
                    (current["predicted_count"] != reference["predicted_count"]).sum()
                ),
                "improved_videos_vs_radius20": int(
                    (current["abs_count_error_change"] < 0).sum()
                ),
                "worsened_videos_vs_radius20": int(
                    (current["abs_count_error_change"] > 0).sum()
                ),
            }
        )
    grid = grid.merge(pd.DataFrame(changed_rows), on="radius", how="left")

    loo_rows = []
    video_ids = sorted(fixed["video_id"].unique())
    for held_out in video_ids:
        training = fixed.loc[fixed["video_id"].ne(held_out)]
        candidates = []
        for radius, group in training.groupby("radius"):
            truth = group["truth_rr"].to_numpy(dtype=float)
            prediction = group["predicted_rr_bpm"].to_numpy(dtype=float)
            candidates.append(
                {
                    "radius": int(radius),
                    "training_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                    "training_mae": float(np.mean(np.abs(prediction - truth))),
                }
            )
        candidate_table = pd.DataFrame(candidates)
        best_rmse = float(candidate_table["training_rmse"].min())
        tied = candidate_table.loc[
            np.isclose(candidate_table["training_rmse"], best_rmse, atol=1e-12)
        ].copy()
        tied["distance_from_reference"] = (
            tied["radius"] - int(args.reference_radius)
        ).abs()
        selected = tied.sort_values(
            ["distance_from_reference", "training_mae", "radius"]
        ).iloc[0]
        held = fixed.loc[
            fixed["video_id"].eq(held_out)
            & fixed["radius"].eq(int(selected["radius"]))
        ].iloc[0]
        loo_rows.append(
            {
                "video_id": held_out,
                "selected_radius": int(selected["radius"]),
                "training_rmse": float(selected["training_rmse"]),
                "training_mae": float(selected["training_mae"]),
                "truth_rr": float(held["truth_rr"]),
                "predicted_rr_bpm": float(held["predicted_rr_bpm"]),
                "truth_count": int(held["truth_count"]),
                "predicted_count": int(held["predicted_count"]),
                "count_error": int(held["count_error"]),
            }
        )
    loo = pd.DataFrame(loo_rows)
    loo_metric = metric_row(loo, -1)
    radius_counts = loo["selected_radius"].value_counts().sort_index()
    report = [
        "# Paper73 Fixed ROI Radius Validation",
        "",
        "All radii from 14 through 28 were evaluated with frozen per-video fusion and peak parameters.",
        "",
        "## Grid result",
        "",
        f"Radius 20 R2: {float(grid.loc[grid['radius'].eq(20), 'rr_r2'].iloc[0]):.6f}.",
        f"Radius 19 R2: {float(grid.loc[grid['radius'].eq(19), 'rr_r2'].iloc[0]):.6f}; changed videos versus radius 20: {int(grid.loc[grid['radius'].eq(19), 'changed_videos_vs_radius20'].iloc[0])}.",
        "",
        "## Leave-one-video-out selection",
        "",
        f"LOO R2: {float(loo_metric['rr_r2']):.6f}; MAE: {float(loo_metric['rr_mae_bpm']):.6f}; RMSE: {float(loo_metric['rr_rmse_bpm']):.6f}.",
        f"Selected-radius counts: {', '.join(f'{int(radius)}={int(count)}' for radius, count in radius_counts.items())}.",
        "",
        "Radius 19 is empirically equivalent to radius 20 on this dataset, but radius 20 remains the conservative default because LOO tie-breaking favors the pre-specified reference and no accuracy gain is demonstrated.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "paper73_fixed_radius_grid_metrics.csv", index=False)
    loo.to_csv(args.output_dir / "paper73_fixed_radius_loo_predictions.csv", index=False)
    pd.DataFrame([loo_metric]).to_csv(
        args.output_dir / "paper73_fixed_radius_loo_metrics.csv", index=False
    )
    (args.output_dir / "paper73_fixed_radius_validation.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(grid.to_string(index=False))
    print(pd.DataFrame([loo_metric]).to_string(index=False))
    print(radius_counts.to_string())


if __name__ == "__main__":
    main()
