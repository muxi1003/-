from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from build_rr_calibration_free_thermal_index import (
    estimate_curve,
    extract_video_signals,
    fused_curve,
)
from build_rr_duration_normalized_windowed_innovation import derive_config


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Generate label-free frozen P2g predictions for the pre-registered external "
            "extension clips. This command never reads manual breath counts or RR values."
        )
    )
    parser.add_argument("--assets-dir", type=Path, default=assets)
    parser.add_argument(
        "--internal-root", type=Path, default=repo / "Dataset_new" / "72video" / "al_images"
    )
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_p2g_extension",
    )
    parser.add_argument(
        "--extension-output-prefix", default="external_repro_p2g_extension"
    )
    parser.add_argument(
        "--extension-fieldwork-csv",
        type=Path,
        default=assets / "paper_p2g_extension_all_fieldwork_subset.csv",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_selection.csv",
    )
    parser.add_argument(
        "--primary-gated-predictions-csv",
        type=Path,
        default=assets / "paper_calibration_free_thermal_index_gated_predictions.csv",
    )
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--fps", type=float, default=8.7)
    parser.add_argument("--overwrite-signal-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_members(selection: pd.DataFrame) -> tuple[list[tuple[str, str]], float]:
    if len(selection) != 1:
        raise ValueError("Expected exactly one frozen P2g selection row")
    row = selection.iloc[0]
    members: list[tuple[str, str]] = []
    for item in str(row.get("selected_members", "")).split(";"):
        signal, polarity = item.split(":", maxsplit=1)
        members.append((signal, polarity))
    prominence = float(pd.to_numeric(pd.Series([row.get("selected_prominence")]), errors="coerce").iloc[0])
    if not members or not np.isfinite(prominence):
        raise ValueError("Frozen P2g selection lacks members or peak prominence")
    return members, prominence


def require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(table.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def primary_predictions(table: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        table,
        {"cohort", "video_id", "selected_members", "selected_prominence", "raw_duration_seconds", "duration_gated_rr_bpm", "duration_gate_activated"},
        "Primary P2g predictions",
    )
    primary = table[table["cohort"].astype(str).eq("external_provisional")].copy()
    baseline_column = "rr_bpm_y" if "rr_bpm_y" in primary.columns else "rr_bpm"
    if baseline_column not in primary.columns:
        raise ValueError("Primary P2g predictions lack the frozen baseline RR column")
    primary = primary[
        [
            "video_id",
            "selected_members",
            "selected_prominence",
            baseline_column,
            "raw_duration_seconds",
            "duration_gated_rr_bpm",
            "duration_gate_activated",
        ]
    ].rename(columns={baseline_column: "baseline_rr_bpm"})
    if primary["video_id"].duplicated().any():
        raise ValueError("Primary P2g prediction file has duplicate external video IDs")
    primary.insert(0, "cohort", "external_primary_frozen")
    primary["prediction_origin"] = "frozen_primary_prediction_reused_without_refit"
    return primary


def extension_predictions(
    extension_root: Path,
    output_prefix: str,
    summary: pd.DataFrame,
    members: list[tuple[str, str]],
    config: object,
    radius: int,
    overwrite_signal_cache: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    member_text = ";".join(f"{signal}:{polarity}" for signal, polarity in members)
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        print(f"[extension {index}/{len(summary)}] {video_id}")
        signals = extract_video_signals(
            extension_root / video_id,
            output_prefix,
            radius,
            overwrite_signal_cache,
        )
        estimates = []
        for signal, polarity in members:
            curve = fused_curve(signals, signal, polarity == "inverted", config)
            estimates.append(estimate_curve(curve, config)[0])
        raw_duration = pd.to_numeric(pd.Series([getattr(row, "raw_duration_seconds", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(raw_duration):
            raw_duration = pd.to_numeric(pd.Series([getattr(row, "duration_seconds", np.nan)]), errors="coerce").iloc[0]
        baseline = pd.to_numeric(pd.Series([getattr(row, "rr_bpm", np.nan)]), errors="coerce").iloc[0]
        color = float(np.mean(estimates))
        gate = bool(np.isfinite(raw_duration) and raw_duration >= config.long_clip_threshold_seconds)
        rows.append(
            {
                "cohort": "external_extension_frozen",
                "video_id": video_id,
                "selected_members": member_text,
                "selected_prominence": float(config.base_prominence),
                "baseline_rr_bpm": baseline,
                "color_rr_bpm": color,
                "raw_duration_seconds": raw_duration,
                "duration_gated_rr_bpm": color if gate else baseline,
                "duration_gate_activated": gate,
                "prediction_origin": "pre_registered_extension_frozen_members_without_truth",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    assets = args.assets_dir.resolve()
    extension_root = args.extension_root.resolve()
    internal_root = args.internal_root.resolve()
    fieldwork = pd.read_csv(args.extension_fieldwork_csv.resolve(), dtype=str, keep_default_na=False)
    require_columns(fieldwork, {"external_video_id", "include_in_external_validation"}, "Extension fieldwork")
    expected_ids = set(
        fieldwork.loc[
            fieldwork["include_in_external_validation"].astype(str).str.lower().eq("yes"),
            "external_video_id",
        ].astype(str)
    )
    if len(expected_ids) < 10:
        raise ValueError(f"Expected at least 10 registered extension clips, found {len(expected_ids)}")
    if len(expected_ids) != len(fieldwork):
        raise ValueError("Extension fieldwork must retain every pre-registered clip as included")

    summary_path = extension_root / f"{args.extension_output_prefix}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing extension frozen-RR summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    require_columns(summary, {"video_id", "rr_bpm"}, "Extension frozen-RR summary")
    summary["video_id"] = summary["video_id"].astype(str)
    found_ids = set(summary["video_id"])
    missing = sorted(expected_ids - found_ids)
    unexpected = sorted(found_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"Extension summary scope mismatch; missing={missing[:10]}; unexpected={unexpected[:10]}"
        )
    if summary["video_id"].duplicated().any():
        raise ValueError("Extension frozen-RR summary has duplicate video IDs")

    selection_path = args.selection_csv.resolve()
    selection = pd.read_csv(selection_path)
    members, prominence = selected_members(selection)
    internal_summary = pd.read_csv(internal_root / f"{args.internal_prefix}_summary.csv")
    config = derive_config(internal_summary, args.fps)
    config = replace(config, base_prominence=prominence)

    extension = extension_predictions(
        extension_root,
        args.extension_output_prefix,
        summary.sort_values("video_id"),
        members,
        config,
        args.radius,
        args.overwrite_signal_cache,
    )
    primary_path = args.primary_gated_predictions_csv.resolve()
    primary = primary_predictions(pd.read_csv(primary_path))
    combined = pd.concat([primary, extension], ignore_index=True, sort=False)
    if combined["video_id"].duplicated().any():
        duplicates = combined.loc[combined["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"Combined P2g prediction IDs are duplicated: {duplicates[:10]}")
    if len(extension) != len(expected_ids):
        raise ValueError("Extension prediction count does not match the pre-registered scope")

    extension_path = assets / "paper_p2g_extension_all_frozen_predictions.csv"
    combined_path = assets / "paper_p2g_primary_plus_extension_frozen_predictions.csv"
    manifest_path = assets / "paper_p2g_extension_all_freeze_manifest.json"
    report_path = assets / "paper_p2g_extension_all_frozen_prediction_report.md"
    extension.to_csv(extension_path, index=False)
    combined.to_csv(combined_path, index=False)
    manifest = {
        "method": "duration_gated_calibration_free_internal_ranked_ensemble",
        "scope": "primary_plus_pre_registered_extension",
        "primary_prediction_rows": int(len(primary)),
        "extension_prediction_rows": int(len(extension)),
        "combined_prediction_rows": int(len(combined)),
        "selected_members": ";".join(f"{signal}:{polarity}" for signal, polarity in members),
        "selected_prominence": float(config.base_prominence),
        "duration_gate_seconds": float(config.long_clip_threshold_seconds),
        "selection_file": str(selection_path),
        "selection_file_sha256": sha256(selection_path),
        "primary_prediction_file": str(primary_path),
        "primary_prediction_file_sha256": sha256(primary_path),
        "extension_fieldwork_file": str(args.extension_fieldwork_csv.resolve()),
        "extension_fieldwork_file_sha256": sha256(args.extension_fieldwork_csv.resolve()),
        "extension_summary_file": str(summary_path),
        "extension_summary_file_sha256": sha256(summary_path),
        "label_policy": "No manual breath count, reference RR, or prediction-error field is read by this generator.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# P2G Pre-Registered Extension Frozen Predictions",
                "",
                f"Extension clips: `{len(extension)}`.",
                f"Combined primary-plus-extension clips: `{len(combined)}`.",
                f"Selected members: `{manifest['selected_members']}`.",
                f"Duration gate: `{manifest['duration_gate_seconds']:.3f}` seconds.",
                "",
                "No RR accuracy metric is reported here because no extension manual reference is read. "
                "After blinded A/B annotation and adjudication, score these fixed predictions with "
                "`score_rr_p2g_blinded_consensus.py --scope primary_plus_extension` and the "
                "combined prediction CSV.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved extension frozen predictions: {extension_path}")
    print(f"Saved combined frozen predictions: {combined_path}")
    print(f"Saved extension freeze manifest: {manifest_path}")
    print(f"Saved extension prediction report: {report_path}")


if __name__ == "__main__":
    main()
