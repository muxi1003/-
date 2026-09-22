from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def assets_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )


def parse_args() -> argparse.Namespace:
    assets = assets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen P2g predictions against a desktop single-annotator import. "
            "The outputs are explicitly provisional and non-confirmatory."
        )
    )
    parser.add_argument(
        "--single-annotation-csv",
        type=Path,
        default=assets / "paper_external_validation_desktop_single_annotation_import.csv",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=assets / "paper_p2g_primary_plus_extension_frozen_predictions.csv",
    )
    parser.add_argument(
        "--primary-fieldwork-csv",
        type=Path,
        default=assets / "paper_p2g_primary_all_fieldwork_subset.csv",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
        help="Provides source-session IDs only; its counts are never used for this evaluation.",
    )
    parser.add_argument(
        "--method-id",
        default="duration_gated_calibration_free_internal_ranked_ensemble",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper_p2g_desktop_single_annotator_provisional",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=assets,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(data: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(data.get(column, pd.Series(np.nan, index=data.index)), errors="coerce")


def regression_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    truth = data["truth_rr_bpm"].to_numpy(dtype=float)
    prediction = data["p2g_rr_bpm"].to_numpy(dtype=float)
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    predicted_count = np.rint(prediction * data["manual_duration_seconds"].to_numpy(dtype=float) / 60.0)
    count_error = np.abs(predicted_count - data["manual_breath_count"].to_numpy(dtype=float))
    return {
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_pearson_r2": (
            float(np.corrcoef(truth, prediction)[0, 1] ** 2)
            if len(truth) >= 2 and np.std(truth) > 0 and np.std(prediction) > 0
            else math.nan
        ),
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
        "count_mae": float(np.mean(count_error)),
        "exact_count": int(np.isclose(count_error, 0).sum()),
        "within_one_count": int((count_error <= 1.0 + 1e-9).sum()),
        "abs_count_error_ge2": int((count_error >= 2.0 - 1e-9).sum()),
    }


def session_level_outputs(data: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    grouped = data.copy()
    grouped["predicted_breaths"] = (
        grouped["p2g_rr_bpm"] * grouped["manual_duration_seconds"] / 60.0
    )
    grouped = (
        grouped.groupby("source_session_id", as_index=False)
        .agg(
            clips=("external_video_id", "size"),
            manual_breath_count=("manual_breath_count", "sum"),
            manual_duration_seconds=("manual_duration_seconds", "sum"),
            predicted_breaths=("predicted_breaths", "sum"),
        )
        .sort_values("source_session_id")
        .reset_index(drop=True)
    )
    grouped["truth_rr_bpm"] = (
        grouped["manual_breath_count"] * 60.0 / grouped["manual_duration_seconds"]
    )
    grouped["p2g_rr_bpm"] = (
        grouped["predicted_breaths"] * 60.0 / grouped["manual_duration_seconds"]
    )
    return regression_metrics(grouped), grouped


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.single_annotation_csv, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(args.predictions_csv, dtype=str, keep_default_na=False)
    primary = pd.read_csv(args.primary_fieldwork_csv, dtype=str, keep_default_na=False)
    reference = pd.read_csv(args.reference_csv, dtype=str, keep_default_na=False)
    for name, data, required in [
        ("single annotation", annotations, {"external_video_id", "manual_breath_count", "manual_duration_seconds", "manual_rr_from_count_bpm", "reference_evidence_tier"}),
        ("frozen predictions", predictions, {"video_id", "duration_gated_rr_bpm", "selected_members", "selected_prominence"}),
        ("primary fieldwork", primary, {"external_video_id"}),
        ("reference metadata", reference, {"external_video_id", "source_session_id"}),
    ]:
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{name} lacks columns: {sorted(missing)}")

    primary_ids = set(primary["external_video_id"].astype(str))
    annotations = annotations[annotations["external_video_id"].astype(str).isin(primary_ids)].copy()
    annotations["manual_breath_count"] = numeric(annotations, "manual_breath_count")
    annotations["manual_duration_seconds"] = numeric(annotations, "manual_duration_seconds")
    annotations["truth_rr_bpm"] = numeric(annotations, "manual_rr_from_count_bpm")
    predictions["p2g_rr_bpm"] = numeric(predictions, "duration_gated_rr_bpm")
    merged = annotations.merge(
        predictions,
        left_on="external_video_id",
        right_on="video_id",
        how="left",
        validate="one_to_one",
    )
    valid = (
        merged["reference_evidence_tier"].astype(str).eq("single_annotator_provisional")
        & np.isfinite(merged["manual_breath_count"])
        & np.isfinite(merged["manual_duration_seconds"])
        & (merged["manual_duration_seconds"] > 0)
        & np.isfinite(merged["truth_rr_bpm"])
        & np.isfinite(merged["p2g_rr_bpm"])
    )
    scored = merged[valid].copy()
    missing_primary = sorted(primary_ids - set(scored["external_video_id"].astype(str)))
    if missing_primary:
        raise ValueError(
            "The frozen primary scope must be fully labeled before preliminary scoring; "
            f"missing IDs={missing_primary[:10]}"
        )
    if len(scored) < 2:
        raise ValueError("At least two valid primary rows are required for RR R2")
    source_sessions = reference[["external_video_id", "source_session_id"]].copy()
    if source_sessions["external_video_id"].duplicated().any():
        raise ValueError("reference metadata has duplicate external video IDs")
    scored = scored.merge(
        source_sessions,
        on="external_video_id",
        how="left",
        validate="one_to_one",
    )
    if scored["source_session_id"].astype(str).str.strip().eq("").any() or scored[
        "source_session_id"
    ].isna().any():
        raise ValueError("source-session IDs are required for session-level evaluation")

    keep = [
        "external_video_id",
        "source_session_id",
        "cow_id",
        "manual_breath_count",
        "manual_duration_seconds",
        "truth_rr_bpm",
        "p2g_rr_bpm",
        "baseline_rr_bpm",
        "duration_gate_activated",
        "prediction_origin",
        "reference_rr_annotator",
        "reference_quality_status",
        "reference_evidence_tier",
        "source_csv_sha256",
    ]
    scored = scored[[column for column in keep if column in scored.columns]].copy()
    scored["rr_error_bpm"] = scored["p2g_rr_bpm"] - scored["truth_rr_bpm"]
    scored["abs_rr_error_bpm"] = scored["rr_error_bpm"].abs()
    metrics = regression_metrics(scored)
    session_metrics, session_predictions = session_level_outputs(scored)
    metric_row = {
        "scope": "frozen_primary_94",
        "evaluation_status": "provisional_single_annotator_complete",
        "confirmatory_status": "not_confirmatory_single_annotator",
        "method": args.method_id,
        "pre_registered_videos": len(primary_ids),
        "valid_labelled_videos": len(scored),
        "missing_labelled_videos": len(primary_ids) - len(scored),
        "selected_members": str(predictions.iloc[0]["selected_members"]),
        "selected_prominence": float(predictions.iloc[0]["selected_prominence"]),
        "evaluation_note": (
            "Frozen P2g predictions compared with a non-blinded desktop single-annotator "
            "count. This is a progress estimate only and cannot update manuscript, target-"
            "journal, or confirmatory external-validation claims."
        ),
        **metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.output_prefix}_predictions.csv"
    metrics_path = args.output_dir / f"{args.output_prefix}_metrics.csv"
    session_metrics_path = args.output_dir / f"{args.output_prefix}_session_metrics.csv"
    session_predictions_path = args.output_dir / f"{args.output_prefix}_session_predictions.csv"
    manifest_path = args.output_dir / f"{args.output_prefix}_freeze_manifest.json"
    report_path = args.output_dir / f"{args.output_prefix}_report.md"
    scored.to_csv(predictions_path, index=False)
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "scope": "frozen_primary_94",
                "evaluation_status": "provisional_single_annotator_complete",
                "confirmatory_status": "not_confirmatory_single_annotator",
                "method": args.method_id,
                "analysis_unit": "source_session_id",
                "source_sessions": int(len(session_predictions)),
                "clips_aggregated": int(len(scored)),
                **session_metrics,
            }
        ]
    ).to_csv(session_metrics_path, index=False)
    session_predictions.to_csv(session_predictions_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "evaluation_status": metric_row["evaluation_status"],
                "confirmatory_status": metric_row["confirmatory_status"],
                "single_annotation_csv": str(args.single_annotation_csv.resolve()),
                "single_annotation_csv_sha256": sha256(args.single_annotation_csv),
                "frozen_predictions_csv": str(args.predictions_csv.resolve()),
                "frozen_predictions_csv_sha256": sha256(args.predictions_csv),
                "primary_fieldwork_csv": str(args.primary_fieldwork_csv.resolve()),
                "primary_fieldwork_csv_sha256": sha256(args.primary_fieldwork_csv),
                "reference_metadata_csv": str(args.reference_csv.resolve()),
                "reference_metadata_csv_sha256": sha256(args.reference_csv),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "# Frozen RR Desktop Single-Annotation Preliminary Evaluation\n\n"
        "Status: `provisional_single_annotator_complete`\n\n"
        f"- Frozen primary videos: `{metric_row['pre_registered_videos']}`\n"
        f"- Valid single-annotator labels: `{metric_row['valid_labelled_videos']}`\n"
        f"- RR R2: `{metric_row['rr_r2']:.6f}`\n"
        f"- RR MAE: `{metric_row['rr_mae']:.6f}` BPM\n"
        f"- RR RMSE: `{metric_row['rr_rmse']:.6f}` BPM\n\n"
        f"- Time-weighted source-session RR R2: `{session_metrics['rr_r2']:.6f}` across `{len(session_predictions)}` sessions\n"
        f"- Time-weighted source-session MAE: `{session_metrics['rr_mae']:.6f}` BPM\n\n"
        "This result is not confirmatory: the same annotation source is not a blinded A/B "
        "consensus and has no adjudication record. Do not use it as the paper's confirmed "
        "external result or to satisfy a target-journal gate.\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")
    print(f"RR_R2={metric_row['rr_r2']:.6f}")
    print(f"RR_MAE={metric_row['rr_mae']:.6f}")
    print(f"RR_RMSE={metric_row['rr_rmse']:.6f}")


if __name__ == "__main__":
    main()
