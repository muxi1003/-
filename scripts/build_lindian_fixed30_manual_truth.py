"""Normalize manually counted Lindian anchored-window truth without altering the annotation template."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


YES_VALUES = {"yes", "y", "true", "1", "include", "included"}
NO_VALUES = {"no", "n", "false", "0", "exclude", "excluded"}


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
        "--annotation-template",
        type=Path,
        default=assets / "lindian_anchored30_quality_gated_annotation_template.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=assets / "lindian_anchored30_manual_truth.csv",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=assets / "lindian_anchored30_manual_truth.md",
    )
    return parser.parse_args()


def read_annotation(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "cp936"]:
        try:
            return pd.read_csv(path, dtype={"video_id": str}, keep_default_na=False, encoding=encoding), encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise UnicodeError("Could not decode annotation template: " + " | ".join(errors))


def normalize(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def main() -> None:
    args = parse_args()
    annotation, source_encoding = read_annotation(args.annotation_template)
    required = {
        "video_id",
        "window_id",
        "window_seconds",
        "manual_breath_count",
        "annotation_status",
        "include_in_fixed_window_evaluation",
    }
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"Annotation template is missing columns: {sorted(missing)}")
    if annotation["video_id"].duplicated().any():
        duplicates = annotation.loc[annotation["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"Duplicate video IDs: {duplicates}")

    count = pd.to_numeric(annotation["manual_breath_count"], errors="coerce")
    duration = pd.to_numeric(annotation["window_seconds"], errors="coerce")
    invalid = count.isna() | duration.isna() | count.le(0) | duration.le(0) | ~np.isclose(count, np.round(count))
    if invalid.any():
        bad = annotation.loc[invalid, ["video_id", "manual_breath_count", "window_seconds"]]
        raise ValueError("Invalid manual truth rows:\n" + bad.to_string(index=False))

    status = annotation["annotation_status"].map(normalize).str.lower()
    explicit_include = annotation["include_in_fixed_window_evaluation"].map(normalize).str.lower()
    unknown_include = ~explicit_include.isin(YES_VALUES | NO_VALUES | {""})
    if unknown_include.any():
        values = sorted(set(explicit_include[unknown_include]))
        raise ValueError(f"Unknown inclusion values: {values}")
    include_primary = explicit_include.isin(YES_VALUES) | (
        explicit_include.eq("") & status.eq("completed")
    )
    include_primary &= ~explicit_include.isin(NO_VALUES)

    truth = annotation.copy()
    truth["breath_count"] = np.round(count).astype(int)
    truth["duration_seconds"] = duration.astype(float)
    truth["manual_rr_bpm"] = truth["breath_count"] * 60.0 / truth["duration_seconds"]
    truth["rr"] = truth["manual_rr_bpm"]
    truth["include_primary_analysis"] = include_primary
    truth["include_sensitivity_analysis"] = True
    truth["truth_reliability"] = np.where(status.eq("completed"), "completed", "uncertain")
    truth["annotation_source_encoding"] = source_encoding

    ordered = [
        "window_id",
        "video_id",
        "breath_count",
        "duration_seconds",
        "rr",
        "manual_rr_bpm",
        "annotation_status",
        "truth_reliability",
        "include_primary_analysis",
        "include_sensitivity_analysis",
        "bilateral_valid_fraction_1hz",
        "labeler_id",
        "review_notes",
        "raw_source_path",
        "window_start_seconds",
        "annotation_source_encoding",
    ]
    ordered = [column for column in ordered if column in truth.columns]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    truth[ordered].to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    primary = truth.loc[truth["include_primary_analysis"]]
    report = [
        "# Lindian Anchored 30-Second Manual Truth",
        "",
        f"- Source encoding detected: {source_encoding}",
        f"- Numeric manual counts: {len(truth)}/{len(truth)}",
        f"- Primary completed annotations: {len(primary)}",
        f"- Sensitivity annotations including uncertain rows: {len(truth)}",
        f"- Count range: {truth['breath_count'].min()} to {truth['breath_count'].max()}",
        f"- Primary RR range: {primary['manual_rr_bpm'].min():.3f} to {primary['manual_rr_bpm'].max():.3f} bpm",
        "",
        "Primary analysis uses completed rows unless the inclusion column explicitly says otherwise. Uncertain rows remain available only for sensitivity analysis.",
    ]
    args.output_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved normalized manual truth: {args.output_csv.resolve()}")
    print(f"Saved truth audit report: {args.output_report.resolve()}")
    print(f"Primary rows: {len(primary)}; sensitivity rows: {len(truth)}")


if __name__ == "__main__":
    main()
