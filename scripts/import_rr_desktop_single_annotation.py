from __future__ import annotations

import argparse
import hashlib
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
            "Import a manually completed desktop RR sheet as a traceable, single-annotator "
            "provisional reference. This command cannot create confirmatory external metrics."
        )
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--fieldwork-csv",
        type=Path,
        default=assets / "paper_external_validation_split_all_use_fieldwork_template.csv",
    )
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
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--annotator-id", default="desktop_single_annotator")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"external_video_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")
    data["external_video_id"] = data["external_video_id"].astype(str).str.strip()
    if data["external_video_id"].eq("").any() or data["external_video_id"].duplicated().any():
        raise ValueError(f"{label} must contain one non-empty row per external_video_id")
    return data


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def count_ids(data: pd.DataFrame, ids: set[str]) -> int:
    return int(data["external_video_id"].isin(ids).sum())


def main() -> None:
    args = parse_args()
    source = load_unique(args.source_csv, "source CSV")
    fieldwork = load_unique(args.fieldwork_csv, "fieldwork CSV")
    required = {"manual_breath_count", "manual_duration_seconds"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"source CSV lacks columns: {sorted(missing)}")

    source_ids = set(source["external_video_id"])
    fieldwork_ids = set(fieldwork["external_video_id"])
    if source_ids != fieldwork_ids:
        raise ValueError(
            "source and registered fieldwork IDs differ: "
            f"source_only={sorted(source_ids - fieldwork_ids)[:5]}, "
            f"fieldwork_only={sorted(fieldwork_ids - source_ids)[:5]}"
        )

    primary = load_unique(args.primary_fieldwork_csv, "primary fieldwork CSV")
    extension = load_unique(args.extension_fieldwork_csv, "extension fieldwork CSV")
    primary_ids = set(primary["external_video_id"])
    extension_ids = set(extension["external_video_id"])
    if primary_ids.intersection(extension_ids):
        raise ValueError("primary and extension fieldwork sets overlap")
    if not primary_ids.union(extension_ids).issubset(fieldwork_ids):
        raise ValueError("registered P2g scope IDs are not present in fieldwork")

    imported = source.copy()
    count = numeric(imported["manual_breath_count"])
    duration = numeric(imported["manual_duration_seconds"])
    valid_count = count.notna() & np.isfinite(count) & (count >= 0) & np.isclose(count, np.rint(count))
    valid_duration = duration.notna() & np.isfinite(duration) & (duration > 0)
    valid_label = valid_count & valid_duration
    imported["manual_breath_count"] = count.where(valid_count).round().astype("Int64")
    imported["manual_duration_seconds"] = duration.where(valid_duration)
    imported["submitted_manual_rr_bpm"] = imported.get("manual_rr_bpm", "")
    imported["manual_rr_from_count_bpm"] = np.where(
        valid_label,
        60.0 * count / duration,
        np.nan,
    )
    imported["reference_rr_annotator"] = np.where(
        valid_label,
        args.annotator_id,
        "",
    )
    imported["reference_quality_status"] = np.where(
        valid_label,
        "single_annotator_unverified",
        "single_annotator_missing_or_invalid",
    )
    imported["reference_count_uncertainty_breaths"] = ""
    imported["reference_evidence_tier"] = "single_annotator_provisional"
    imported["confirmatory_eligibility"] = False
    imported["import_status"] = np.where(valid_label, "imported", "missing_or_invalid_label")
    imported["source_csv_path"] = str(args.source_csv.resolve())
    imported["source_csv_sha256"] = sha256(args.source_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imported_path = args.output_dir / "paper_external_validation_desktop_single_annotation_import.csv"
    report_path = args.output_dir / "paper_external_validation_desktop_single_annotation_import_report.csv"
    readme_path = args.output_dir / "paper_external_validation_desktop_single_annotation_import.md"
    imported.to_csv(imported_path, index=False)

    valid = imported[imported["import_status"].eq("imported")]
    report_rows = [
        ("source_rows", len(source)),
        ("registered_fieldwork_rows", len(fieldwork)),
        ("source_id_set_matches_registered_fieldwork", True),
        ("valid_single_annotator_labels", int(valid_label.sum())),
        ("missing_or_invalid_single_annotator_labels", int((~valid_label).sum())),
        ("primary_registered_rows", len(primary_ids)),
        ("primary_valid_single_annotator_labels", count_ids(valid, primary_ids)),
        ("extension_registered_rows", len(extension_ids)),
        ("extension_valid_single_annotator_labels", count_ids(valid, extension_ids)),
        ("confirmatory_eligible_rows", 0),
        ("source_csv_sha256", sha256(args.source_csv)),
        ("fieldwork_csv_sha256", sha256(args.fieldwork_csv)),
    ]
    pd.DataFrame(report_rows, columns=["metric", "value"]).to_csv(report_path, index=False)
    readme_path.write_text(
        "# Desktop Single-Annotation Import\n\n"
        "This artifact preserves a completed desktop sheet as a single-annotator, "
        "provisional reference. RR is recomputed row by row as ``60 * count / "
        "manual_duration_seconds``. It is not a blinded A/B consensus, cannot change "
        "confirmatory external-validation claims, and must not be used to update target-"
        "journal gates or manuscript confirmed metrics.\n",
        encoding="utf-8",
    )
    print(f"Wrote {imported_path}")
    print(f"Wrote {report_path}")
    print(f"valid_single_annotator_labels={int(valid_label.sum())}")


if __name__ == "__main__":
    main()
