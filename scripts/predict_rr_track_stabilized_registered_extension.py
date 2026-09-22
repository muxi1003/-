from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from build_rr_calibration_free_consensus_ensemble import aggregate_members, duration_gated, selected_members
from build_rr_calibration_free_track_stabilized_roi import STRATEGIES, member_predictions
from build_rr_duration_normalized_windowed_innovation import derive_config


def assets_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = assets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Freeze bilateral trajectory-stabilized RR predictions for the registered primary "
            "and unlabelled extension clips without reading manual counts or RR labels."
        )
    )
    parser.add_argument("--internal-root", type=Path, default=repo / "Dataset_new" / "72video" / "al_images")
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_p2g_extension",
    )
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--primary-prefix", default="external_repro_single_reference")
    parser.add_argument("--extension-prefix", default="external_repro_p2g_extension")
    parser.add_argument(
        "--primary-fieldwork-csv",
        type=Path,
        default=assets / "paper_p2g_primary_all_fieldwork_subset.csv",
    )
    parser.add_argument(
        "--extension-fieldwork-csv",
        type=Path,
        default=assets / "paper_p2g_extension_all_fieldwork_subset.csv",
    )
    parser.add_argument(
        "--p2g-selection-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_selection.csv",
    )
    parser.add_argument(
        "--track-selection-csv",
        type=Path,
        default=assets / "paper_calibration_free_track_stabilized_roi_selected_strategy.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--minimum-radius", type=int, default=8)
    parser.add_argument("--maximum-radius", type=int, default=48)
    parser.add_argument(
        "--freeze-version",
        default="v2_duration_fallback",
        help=(
            "Versioned artifact suffix. The default preserves the original v1 frozen "
            "prediction files while recording the deterministic duration-source fix."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registered_ids(path: Path, label: str) -> set[str]:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "external_video_id" not in data.columns:
        raise ValueError(f"{label} lacks external_video_id")
    ids = set(data["external_video_id"].astype(str))
    if not ids or len(ids) != len(data):
        raise ValueError(f"{label} must contain unique non-empty external video IDs")
    return ids


def artifact_prefix(version: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", version):
        raise ValueError("freeze-version may contain only letters, numbers, and underscores")
    return f"paper_track_stabilized_{version}"


def registered_summary(root: Path, prefix: str, ids: set[str], label: str) -> pd.DataFrame:
    summary_path = root / f"{prefix}_summary.csv"
    summary = pd.read_csv(summary_path)
    selected = summary[summary["video_id"].astype(str).isin(ids)].copy()
    observed = set(selected["video_id"].astype(str))
    if observed != ids:
        raise ValueError(f"{label} summary does not match registered IDs: {sorted(ids - observed)[:8]}")
    return selected


def freeze_predictions(
    root: Path,
    prefix: str,
    summary: pd.DataFrame,
    config: object,
    members: list[tuple[str, str]],
    strategy: str,
    args: argparse.Namespace,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    member_table, diagnostics = member_predictions(
        root,
        prefix,
        summary,
        config,
        members,
        [strategy],
        args.minimum_radius,
        args.maximum_radius,
        cohort,
    )
    aggregate = aggregate_members(member_table, members, "mean")
    gated = duration_gated(aggregate, summary, config.long_clip_threshold_seconds)
    output = gated.rename(columns={"video_id": "video_id"}).copy()
    output["cohort"] = cohort
    output["selected_members"] = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    output["selected_prominence"] = config.base_prominence
    output["track_strategy"] = strategy
    output["prediction_origin"] = "frozen_track_stabilized_prediction_without_manual_rr_read"
    diagnostics["cohort"] = cohort
    return output, diagnostics


def main() -> None:
    args = parse_args()
    primary_ids = read_registered_ids(args.primary_fieldwork_csv, "primary fieldwork")
    extension_ids = read_registered_ids(args.extension_fieldwork_csv, "extension fieldwork")
    if primary_ids.intersection(extension_ids):
        raise ValueError("primary and extension registered IDs overlap")
    track_selection = pd.read_csv(args.track_selection_csv)
    if len(track_selection) != 1 or "strategy" not in track_selection.columns:
        raise ValueError("track selection must contain exactly one selected strategy")
    strategy = str(track_selection.iloc[0]["strategy"])
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown track strategy: {strategy}")
    p2g_selection = pd.read_csv(args.p2g_selection_csv)
    members, prominence = selected_members(p2g_selection)
    internal_summary = pd.read_csv(args.internal_root / f"{args.internal_prefix}_summary.csv")
    config = derive_config(internal_summary, args.fps)
    config = config.__class__(**{**config.__dict__, "base_prominence": prominence})
    primary_summary = registered_summary(args.primary_root, args.primary_prefix, primary_ids, "primary")
    extension_summary = registered_summary(args.extension_root, args.extension_prefix, extension_ids, "extension")

    primary, primary_diagnostics = freeze_predictions(
        args.primary_root,
        args.primary_prefix,
        primary_summary,
        config,
        members,
        strategy,
        args,
        "external_primary_track_frozen",
    )
    extension, extension_diagnostics = freeze_predictions(
        args.extension_root,
        args.extension_prefix,
        extension_summary,
        config,
        members,
        strategy,
        args,
        "external_extension_track_frozen",
    )
    combined = pd.concat([primary, extension], ignore_index=True, sort=False)
    if len(combined) != len(primary_ids) + len(extension_ids):
        raise ValueError("frozen prediction row count does not match registered scope")
    if combined["video_id"].astype(str).duplicated().any():
        raise ValueError("frozen prediction table contains duplicate video IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = artifact_prefix(args.freeze_version)
    primary_path = args.output_dir / f"{prefix}_primary_frozen_predictions.csv"
    extension_path = args.output_dir / f"{prefix}_extension_frozen_predictions.csv"
    combined_path = args.output_dir / f"{prefix}_primary_plus_extension_frozen_predictions.csv"
    diagnostics_path = args.output_dir / f"{prefix}_primary_plus_extension_frozen_diagnostics.csv"
    manifest_path = args.output_dir / f"{prefix}_primary_plus_extension_freeze_manifest.json"
    report_path = args.output_dir / f"{prefix}_primary_plus_extension_freeze_report.md"
    primary.to_csv(primary_path, index=False)
    extension.to_csv(extension_path, index=False)
    combined.to_csv(combined_path, index=False)
    pd.concat([primary_diagnostics, extension_diagnostics], ignore_index=True).to_csv(
        diagnostics_path, index=False
    )
    manifest_path.write_text(
        json.dumps(
            {
                "method": "bilateral_trajectory_stabilized_calibration_free_thermal_color_ensemble",
                "method_status": "post_hoc_development_candidate_not_confirmatory",
                "freeze_version": args.freeze_version,
                "selection_boundary": "73_video_internal_development_only",
                "external_manual_rr_read_by_this_command": False,
                "duration_gate_policy": (
                    "raw_duration_seconds, then duration_seconds, then rr_duration_seconds; "
                    "v2 fixes missing raw-duration fallbacks without reading labels"
                ),
                "track_strategy": strategy,
                "selected_members": ";".join(f"{signal}:{polarity}" for signal, polarity in members),
                "selected_prominence": float(prominence),
                "duration_gate_seconds": float(config.long_clip_threshold_seconds),
                "primary_registered_clips": len(primary_ids),
                "extension_registered_clips": len(extension_ids),
                "track_selection_csv": str(args.track_selection_csv.resolve()),
                "track_selection_csv_sha256": sha256(args.track_selection_csv),
                "p2g_selection_csv": str(args.p2g_selection_csv.resolve()),
                "p2g_selection_csv_sha256": sha256(args.p2g_selection_csv),
                "primary_fieldwork_csv": str(args.primary_fieldwork_csv.resolve()),
                "primary_fieldwork_csv_sha256": sha256(args.primary_fieldwork_csv),
                "extension_fieldwork_csv": str(args.extension_fieldwork_csv.resolve()),
                "extension_fieldwork_csv_sha256": sha256(args.extension_fieldwork_csv),
                "primary_predictions_csv": str(primary_path.resolve()),
                "extension_predictions_csv": str(extension_path.resolve()),
                "combined_predictions_csv": str(combined_path.resolve()),
                "combined_predictions_csv_sha256": sha256(combined_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                "# Versioned Track-Stabilized Frozen Predictions",
                "",
                f"Freeze version: `{args.freeze_version}`.",
                "",
                "The original v1 files are retained unchanged. This version repairs only the "
                "duration-gate source order: `raw_duration_seconds`, then frame-derived "
                "`duration_seconds`, then `rr_duration_seconds`. No manual breath count, "
                "reference RR, or error column is read by this command.",
                "",
                f"Primary clips: `{len(primary)}`; duration gate active: `{int(primary['duration_gate_activated'].sum())}`.",
                f"Extension clips: `{len(extension)}`; duration gate active: `{int(extension['duration_gate_activated'].sum())}`.",
                f"Combined clips: `{len(combined)}`.",
                "",
                "This is a post-hoc development candidate. It must be scored only after blinded "
                "A/B consensus or on an untouched external cohort; it cannot replace the frozen "
                "P2g primary result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"strategy={strategy}")
    print(f"primary_frozen_rows={len(primary)}")
    print(f"extension_frozen_rows={len(extension)}")
    print(f"combined_predictions={combined_path}")
    print(f"freeze_report={report_path}")


if __name__ == "__main__":
    main()
