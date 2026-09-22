from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_rr_duration_normalized_windowed_innovation import metric_dict


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Score the frozen P2g relative thermal-color prediction against blinded dual-"
            "annotation consensus without refitting any P2g component."
        )
    )
    parser.add_argument("--consensus-csv", type=Path, required=True)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=assets / "paper_external_validation_single_reference_rows.csv",
    )
    parser.add_argument(
        "--p2g-predictions-csv",
        "--predictions-csv",
        dest="predictions_csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_gated_predictions.csv",
    )
    parser.add_argument(
        "--p2g-selection-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_selection.csv",
    )
    parser.add_argument(
        "--method-selection-csv",
        type=Path,
        default=None,
        help=(
            "Optional method-specific selection artifact. The P2g channel-selection CSV is "
            "always recorded separately for traceability."
        ),
    )
    parser.add_argument(
        "--method-id",
        default="duration_gated_calibration_free_internal_ranked_ensemble",
        help="Frozen method identifier written to metrics, reports, and manifests.",
    )
    parser.add_argument(
        "--method-selection-boundary",
        default=(
            "Members, polarity, prominence, and duration gate were selected on the 73-video "
            "internal development cohort before blinded-consensus scoring."
        ),
        help="Human-readable statement of the frozen method-selection boundary.",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper_p2g_blinded_consensus",
        help="Prefix for scorer outputs; retain the default for the original frozen P2g method.",
    )
    parser.add_argument(
        "--priority-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_reannotation_priority.csv",
    )
    parser.add_argument(
        "--extension-fieldwork-csv",
        type=Path,
        default=assets / "paper_p2g_extension_all_fieldwork_subset.csv",
    )
    parser.add_argument(
        "--extension-inventory-csv",
        type=Path,
        default=assets / "paper_external_validation_split_all_use_inventory.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument(
        "--scope",
        choices=["primary_all", "priority_p0_p1", "extension_all", "primary_plus_extension"],
        default="primary_all",
        help=(
            "primary_all is the complete confirmatory scope. priority_p0_p1 is a label-"
            "reliability subset and must not be presented as full external validation. "
            "extension_all scores pre-registered duration-eligible extension clips; "
            "primary_plus_extension requires all 94 primary clips plus at least 10 registered "
            "extension clips."
        ),
    )
    parser.add_argument("--minimum-extension-clips", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table.get(column, pd.Series(np.nan, index=table.index)), errors="coerce")


def scope_ids(
    reference: pd.DataFrame,
    priority: pd.DataFrame,
    extension_fieldwork: pd.DataFrame,
    scope: str,
) -> tuple[set[str], set[str]]:
    primary = reference[
        reference["primary_analysis_include"].astype(str).str.lower().eq("true")
    ]["external_video_id"].astype(str)
    primary_ids = set(primary)
    if scope == "primary_all":
        return primary_ids, set()
    priority_ids = priority[
        priority.get("review_priority", pd.Series("", index=priority.index))
        .astype(str)
        .isin({"P0_dual_blinded_recount_and_adjudication", "P1_second_independent_recount"})
    ]["video_id"].astype(str)
    if scope == "priority_p0_p1":
        return primary_ids.intersection(priority_ids), set()
    extension_ids = set(extension_fieldwork["external_video_id"].astype(str))
    if scope == "extension_all":
        return set(), extension_ids
    return primary_ids, extension_ids


def validate_scope(
    consensus_ids: set[str],
    primary_ids: set[str],
    extension_ids: set[str],
    scope: str,
    minimum_extension_clips: int,
) -> set[str]:
    if scope in {"primary_all", "priority_p0_p1"}:
        expected = primary_ids
        unexpected = sorted(consensus_ids - expected)
        missing = sorted(expected - consensus_ids)
        if unexpected or missing:
            message = []
            if unexpected:
                message.append(f"unexpected consensus IDs={unexpected[:10]}")
            if missing:
                message.append(f"missing scope IDs={missing[:10]}")
            raise ValueError("; ".join(message))
        return expected

    allowed = extension_ids if scope == "extension_all" else primary_ids.union(extension_ids)
    unexpected = sorted(consensus_ids - allowed)
    missing_primary = sorted(primary_ids - consensus_ids) if scope == "primary_plus_extension" else []
    extension_count = len(consensus_ids.intersection(extension_ids))
    if unexpected or missing_primary or extension_count < int(minimum_extension_clips):
        message = []
        if unexpected:
            message.append(f"unexpected consensus IDs={unexpected[:10]}")
        if missing_primary:
            message.append(f"missing primary IDs={missing_primary[:10]}")
        if extension_count < int(minimum_extension_clips):
            message.append(
                f"registered extension consensus clips={extension_count}, minimum={minimum_extension_clips}"
            )
        raise ValueError("; ".join(message))
    return consensus_ids


def load_consensus(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"external_video_id", "manual_breath_count", "reference_rr_annotator"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Consensus CSV missing columns: {sorted(missing)}")
    keep = sorted(
        (
            required
            | {
                "manual_rr_bpm",
                "camera_id",
                "annotation_notes",
                "reference_quality_status",
                "reference_count_uncertainty_breaths",
            }
        ).intersection(
            data.columns
        )
    )
    data = data[keep].copy()
    data["external_video_id"] = data["external_video_id"].astype(str)
    if data["external_video_id"].duplicated().any():
        duplicates = data.loc[data["external_video_id"].duplicated(), "external_video_id"].tolist()
        raise ValueError(f"Consensus CSV has duplicate video IDs: {duplicates[:10]}")
    data["manual_breath_count"] = numeric(data, "manual_breath_count")
    if data["manual_breath_count"].isna().any():
        missing_ids = data.loc[data["manual_breath_count"].isna(), "external_video_id"].tolist()
        raise ValueError(f"Consensus CSV has non-numeric counts: {missing_ids[:10]}")
    annotators = data["reference_rr_annotator"].astype(str).str.strip()
    if annotators.eq("").any():
        missing_ids = data.loc[annotators.eq(""), "external_video_id"].tolist()
        raise ValueError(f"Consensus CSV lacks consensus annotator provenance: {missing_ids[:10]}")
    if "reference_quality_status" not in data.columns:
        data["reference_quality_status"] = "not_recorded"
    else:
        data["reference_quality_status"] = data["reference_quality_status"].astype(str).replace("", "not_recorded")
    if "reference_count_uncertainty_breaths" not in data.columns:
        data["reference_count_uncertainty_breaths"] = ""
    return data


def regression_values(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    truth = frame["truth_rr"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
    }


def session_level_outputs(
    scored: pd.DataFrame,
    method_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate contiguous clips to the source-session deployment unit."""
    data = scored.copy()
    data["predicted_breaths"] = numeric(data, "p2g_rr_bpm") * numeric(
        data, "truth_duration_seconds"
    ) / 60.0
    grouped = (
        data.groupby("source_session_id", as_index=False)
        .agg(
            clips=("video_id", "size"),
            truth_count=("truth_count", "sum"),
            duration_seconds=("truth_duration_seconds", "sum"),
            predicted_breaths=("predicted_breaths", "sum"),
        )
        .sort_values("source_session_id")
        .reset_index(drop=True)
    )
    grouped["truth_rr"] = grouped["truth_count"] * 60.0 / grouped["duration_seconds"]
    grouped["p2g_rr_bpm"] = grouped["predicted_breaths"] * 60.0 / grouped[
        "duration_seconds"
    ]
    metrics = metric_dict(
        "blinded_consensus_external",
        method_id,
        grouped["truth_rr"].to_numpy(float),
        grouped["p2g_rr_bpm"].to_numpy(float),
        grouped["truth_count"].to_numpy(float),
        grouped["duration_seconds"].to_numpy(float),
        "time_weighted_source_session_aggregation_of_frozen_clip_predictions",
    )
    metrics.update(
        {
            "analysis_unit": "source_session_id",
            "source_sessions": int(len(grouped)),
            "clips_aggregated": int(len(scored)),
        }
    )
    return pd.DataFrame([metrics]), grouped


def cluster_bootstrap(data: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    groups = sorted(data["source_session_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat(
            [data[data["source_session_id"].astype(str).eq(group)] for group in sampled],
            ignore_index=True,
        )
        result = regression_values(replicate, "p2g_rr_bpm")
        for metric in samples:
            samples[metric].append(result[metric])
    point = regression_values(data, "p2g_rr_bpm")
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "resamples": int(resamples),
                "cluster_unit": "source_session_id",
            }
            for metric, values in samples.items()
        ]
    )


def method_freeze_manifest(
    args: argparse.Namespace,
    selection: pd.DataFrame,
    predictions: pd.DataFrame,
    scope: str,
) -> dict[str, object]:
    row = selection.iloc[0] if not selection.empty else pd.Series(dtype=object)
    prediction_row = predictions.iloc[0] if not predictions.empty else pd.Series(dtype=object)
    method_selection_file = args.method_selection_csv or args.p2g_selection_csv
    return {
        "method": args.method_id,
        "scope": scope,
        "selected_members": str(prediction_row.get("selected_members", row.get("selected_members", ""))),
        "selected_prominence": float(
            pd.to_numeric(
                pd.Series([prediction_row.get("selected_prominence", row.get("selected_prominence"))]),
                errors="coerce",
            ).iloc[0]
        ),
        "track_strategy": str(prediction_row.get("track_strategy", "")),
        "duration_gate_seconds": 21.0,
        "selection_boundary": args.method_selection_boundary,
        "prediction_file": str(args.predictions_csv.resolve()),
        "prediction_file_sha256": sha256(args.predictions_csv.resolve()),
        "selection_file": str(args.p2g_selection_csv.resolve()),
        "selection_file_sha256": sha256(args.p2g_selection_csv.resolve()),
        "method_selection_file": str(method_selection_file.resolve()),
        "method_selection_file_sha256": sha256(method_selection_file.resolve()),
        "reference_file": str(args.reference_csv.resolve()),
        "reference_file_sha256": sha256(args.reference_csv.resolve()),
        "extension_fieldwork_file": str(args.extension_fieldwork_csv.resolve()),
        "extension_fieldwork_file_sha256": sha256(args.extension_fieldwork_csv.resolve()),
        "extension_inventory_file": str(args.extension_inventory_csv.resolve()),
        "extension_inventory_file_sha256": sha256(args.extension_inventory_csv.resolve()),
        "consensus_file": str(args.consensus_csv.resolve()),
        "consensus_file_sha256": sha256(args.consensus_csv.resolve()),
    }


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    session_metrics: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    row = metrics.iloc[0]
    session_row = session_metrics.iloc[0]
    scope = str(manifest["scope"])
    scope_notes = {
        "primary_all": (
            "This is the complete primary scope and can support confirmatory external reporting "
            "only after all consensus and metadata gates are satisfied."
        ),
        "priority_p0_p1": (
            "This is a P0/P1 label-reliability subset and must not be reported as complete external validation."
        ),
        "extension_all": (
            "This is the pre-registered duration-eligible extension scope. Report every excluded "
            "unreadable clip in the flow table and do not use it as a replacement for the primary scope."
        ),
        "primary_plus_extension": (
            "This combines all 94 primary clips with at least 10 pre-registered extension clips. "
            "Every unavailable extension clip must remain documented in the flow table."
        ),
    }
    scope_note = scope_notes[scope]
    lines = [
        "# Frozen RR Method Blinded-Consensus Score",
        "",
        f"Scope: `{scope}`",
        "",
        "The frozen prediction file is reused as-is. This scorer does not rerun ROI extraction, "
        "channel ranking, peak-prominence selection, duration gating, or model fitting.",
        "",
        f"RR R2 `{row['rr_r2']:.6f}`, MAE `{row['rr_mae']:.6f}`, RMSE `{row['rr_rmse']:.6f}` bpm.",
        (
            "Annotation quality: "
            f"clear={int(row.get('clear_consensus_clips', 0))}, "
            f"uncertain={int(row.get('uncertain_consensus_clips', 0))}, "
            f"not_recorded={int(row.get('quality_not_recorded_clips', 0))}."
        ),
        (
            "Time-weighted source-session endpoint: "
            f"sessions={int(session_row['source_sessions'])}, RR R2={session_row['rr_r2']:.6f}, "
            f"MAE={session_row['rr_mae']:.6f}, RMSE={session_row['rr_rmse']:.6f} bpm."
        ),
        "",
        "## Source-Session Cluster Bootstrap",
        "",
        "| metric | estimate | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for item in bootstrap.itertuples(index=False):
        lines.append(
            f"| {item.metric} | {item.estimate:.6f} | [{item.ci_low:.6f}, {item.ci_high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            scope_note,
            "",
            "## Freeze Manifest",
            "",
            "```json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_csv.resolve(), dtype=str, keep_default_na=False)
    priority = pd.read_csv(args.priority_csv.resolve(), dtype=str, keep_default_na=False)
    extension_fieldwork = pd.read_csv(
        args.extension_fieldwork_csv.resolve(), dtype=str, keep_default_na=False
    )
    extension_inventory = pd.read_csv(
        args.extension_inventory_csv.resolve(), dtype=str, keep_default_na=False
    )
    primary_ids, extension_ids = scope_ids(
        reference, priority, extension_fieldwork, args.scope
    )
    consensus = load_consensus(args.consensus_csv.resolve())
    consensus_ids = set(consensus["external_video_id"])
    active_ids = validate_scope(
        consensus_ids,
        primary_ids,
        extension_ids,
        args.scope,
        args.minimum_extension_clips,
    )
    predictions = pd.read_csv(args.predictions_csv.resolve())
    predictions = predictions[
        predictions["video_id"].astype(str).isin(active_ids)
    ][["video_id", "duration_gated_rr_bpm"]].copy()
    predictions["video_id"] = predictions["video_id"].astype(str)
    if predictions["video_id"].duplicated().any():
        duplicates = predictions.loc[predictions["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"Frozen prediction CSV has duplicate scoped video IDs: {duplicates[:10]}")
    predictions = predictions.rename(columns={"duration_gated_rr_bpm": "p2g_rr_bpm"})
    reference_metadata = reference.rename(columns={"external_video_id": "video_id"})[
        ["video_id", "source_session_id", "manual_duration_seconds"]
    ].copy()
    # The reference worksheet retains non-primary historical rows. Extension rows must
    # come only from the pre-registered extension worksheet below, never both sources.
    reference_metadata = reference_metadata[
        reference_metadata["video_id"].astype(str).isin(primary_ids)
    ].copy()
    extension_metadata = extension_fieldwork[["external_video_id", "manual_duration_seconds"]].rename(
        columns={"external_video_id": "video_id"}
    )
    extension_sessions = extension_inventory[["external_video_id", "source_session_id"]].rename(
        columns={"external_video_id": "video_id"}
    )
    extension_metadata = extension_metadata.merge(
        extension_sessions, on="video_id", how="left", validate="one_to_one"
    )
    metadata = pd.concat([reference_metadata, extension_metadata], ignore_index=True)
    metadata = metadata[metadata["video_id"].astype(str).isin(active_ids)].copy()
    if metadata["video_id"].duplicated().any():
        duplicates = metadata.loc[metadata["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"Scoped metadata has duplicate video IDs: {duplicates[:10]}")
    metadata["manual_duration_seconds"] = numeric(metadata, "manual_duration_seconds")
    scored = consensus.rename(columns={"external_video_id": "video_id"}).merge(
        metadata, on="video_id", how="inner", validate="one_to_one"
    )
    scored = scored.merge(predictions, on="video_id", how="inner", validate="one_to_one")
    if len(scored) != len(active_ids):
        raise ValueError(f"Expected {len(active_ids)} frozen prediction rows, found {len(scored)}")
    scored["truth_count"] = numeric(scored, "manual_breath_count")
    scored["truth_duration_seconds"] = numeric(scored, "manual_duration_seconds")
    scored["truth_rr"] = scored["truth_count"] * 60.0 / scored["truth_duration_seconds"]
    scored["predicted_count"] = np.rint(
        numeric(scored, "p2g_rr_bpm") * scored["truth_duration_seconds"] / 60.0
    )
    metrics = metric_dict(
        "blinded_consensus_external",
        args.method_id,
        scored["truth_rr"].to_numpy(float),
        numeric(scored, "p2g_rr_bpm").to_numpy(float),
        scored["truth_count"].to_numpy(float),
        scored["truth_duration_seconds"].to_numpy(float),
        "frozen_method_predictions_scored_after_blinded_consensus_without_refit",
    )
    quality = scored["reference_quality_status"].astype(str)
    metrics.update(
        {
            "scope": args.scope,
            "consensus_clips": int(len(scored)),
            "clear_consensus_clips": int(quality.eq("clear_dual_annotation").sum()),
            "uncertain_consensus_clips": int(quality.eq("uncertain_dual_annotation").sum()),
            "quality_not_recorded_clips": int(quality.eq("not_recorded").sum()),
        }
    )
    metrics_table = pd.DataFrame([metrics])
    bootstrap = cluster_bootstrap(scored, args.bootstrap_resamples, args.bootstrap_seed)
    session_metrics, session_predictions = session_level_outputs(scored, args.method_id)
    selection = pd.read_csv(args.p2g_selection_csv.resolve())
    manifest = method_freeze_manifest(args, selection, predictions, args.scope)
    prefix = f"{args.output_prefix}_{args.scope}"
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    bootstrap_path = output_dir / f"{prefix}_bootstrap_ci.csv"
    session_metrics_path = output_dir / f"{prefix}_session_metrics.csv"
    session_predictions_path = output_dir / f"{prefix}_session_predictions.csv"
    manifest_path = output_dir / f"{prefix}_freeze_manifest.json"
    report_path = output_dir / f"{prefix}_report.md"
    metrics_table.to_csv(metrics_path, index=False)
    scored.to_csv(predictions_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    session_metrics.to_csv(session_metrics_path, index=False)
    session_predictions.to_csv(session_predictions_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, metrics_table, bootstrap, session_metrics, manifest)
    print(f"Saved frozen-method blinded-consensus metrics: {metrics_path}")
    print(f"Saved frozen-method blinded-consensus predictions: {predictions_path}")
    print(f"Saved frozen-method bootstrap: {bootstrap_path}")
    print(f"Saved frozen-method source-session metrics: {session_metrics_path}")
    print(f"Saved frozen-method source-session predictions: {session_predictions_path}")
    print(f"Saved frozen-method manifest: {manifest_path}")
    print(f"Saved frozen-method blinded-consensus report: {report_path}")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
