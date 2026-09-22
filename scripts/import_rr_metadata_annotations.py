from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "video_id",
    "cow_id",
    "collection_date",
    "collection_start_date",
    "collection_end_date",
    "collection_time",
    "collection_location_country",
    "collection_location_province",
    "collection_location_county",
    "collection_site",
    "camera_id",
    "scene_id",
    "ambient_temperature_c",
    "relative_humidity_percent",
    "thi",
    "athi",
    "posture",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
    "operator_or_annotator",
    "external_test_split",
    "notes",
]

SCORE_COLUMNS = [
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
]

NUMERIC_COLUMNS = [
    "ambient_temperature_c",
    "relative_humidity_percent",
    "thi",
    "athi",
]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
ALLOWED_SPLITS = {"", "internal", "external", "holdout", "train", "validation", "test"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Import filled metadata annotations into paper_metadata_template.csv "
            "with validation. Defaults to dry-run."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--metadata-template", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write validated metadata columns to paper_metadata_template.csv.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def load_annotation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV does not exist: {path}")
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "video_id" not in data.columns:
        raise ValueError(f"Annotation CSV is missing required column: video_id")
    data["video_id"] = data["video_id"].map(normalize_text)
    if data["video_id"].eq("").any():
        raise ValueError("Annotation CSV contains empty video_id values")
    duplicates = data["video_id"][data["video_id"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Annotation CSV contains duplicate video_id values: {duplicates[:10]}")
    for column in METADATA_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data[METADATA_COLUMNS].fillna("").map(normalize_text)


def load_summary_video_ids(input_root: Path, output_prefix: str) -> list[str]:
    summary_csv = input_root / f"{output_prefix}_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")
    summary = pd.read_csv(summary_csv, usecols=["video_id"], dtype=str)
    return summary["video_id"].astype(str).tolist()


def validate_metadata(metadata: pd.DataFrame, summary_video_ids: list[str]) -> pd.DataFrame:
    summary_set = set(summary_video_ids)
    rows: list[dict[str, object]] = []

    annotation_set = set(metadata["video_id"].astype(str))
    missing = sorted(summary_set - annotation_set)
    extra = sorted(annotation_set - summary_set)
    rows.append(
        {
            "check": "all summary videos present",
            "status": "PASS" if not missing else "FAIL",
            "affected_rows": len(missing),
            "detail": ";".join(missing[:20]),
            "recommended_action": "Add missing video_id rows to the annotation sheet.",
        }
    )
    rows.append(
        {
            "check": "no extra video ids",
            "status": "PASS" if not extra else "WARN",
            "affected_rows": len(extra),
            "detail": ";".join(extra[:20]),
            "recommended_action": "Remove or verify extra annotation rows.",
        }
    )

    for column in METADATA_COLUMNS:
        if column == "video_id":
            continue
        values = metadata[column].map(normalize_text)
        rows.append(
            {
                "check": f"{column} coverage",
                "status": "PASS" if values.ne("").all() else "WARN",
                "affected_rows": int(values.eq("").sum()),
                "detail": f"nonempty={int(values.ne('').sum())}/{len(values)}",
                "recommended_action": f"Fill {column} for all videos if required for the target analysis.",
            }
        )

    for column in SCORE_COLUMNS:
        invalid = []
        for video_id, value in zip(metadata["video_id"], metadata[column]):
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                invalid.append(str(video_id))
                continue
            if number not in {0.0, 1.0, 2.0, 3.0}:
                invalid.append(str(video_id))
        rows.append(
            {
                "check": f"{column} values in 0-3",
                "status": "PASS" if not invalid else "FAIL",
                "affected_rows": len(invalid),
                "detail": ";".join(invalid[:20]),
                "recommended_action": f"Use integer scores 0, 1, 2, or 3 for {column}.",
            }
        )

    for column in NUMERIC_COLUMNS:
        invalid = []
        for video_id, value in zip(metadata["video_id"], metadata[column]):
            if value == "":
                continue
            try:
                float(value)
            except ValueError:
                invalid.append(str(video_id))
        rows.append(
            {
                "check": f"{column} numeric",
                "status": "PASS" if not invalid else "FAIL",
                "affected_rows": len(invalid),
                "detail": ";".join(invalid[:20]),
                "recommended_action": f"Use numeric values for {column}.",
            }
        )

    for column, regex, example in [
        ("collection_date", DATE_RE, "YYYY-MM-DD"),
        ("collection_start_date", DATE_RE, "YYYY-MM-DD"),
        ("collection_end_date", DATE_RE, "YYYY-MM-DD"),
        ("collection_time", TIME_RE, "HH:MM or HH:MM:SS"),
    ]:
        invalid = [
            str(video_id)
            for video_id, value in zip(metadata["video_id"], metadata[column])
            if value != "" and not regex.match(value)
        ]
        rows.append(
            {
                "check": f"{column} format",
                "status": "PASS" if not invalid else "WARN",
                "affected_rows": len(invalid),
                "detail": ";".join(invalid[:20]),
                "recommended_action": f"Prefer {example} for {column}.",
            }
        )

    invalid_splits = [
        str(video_id)
        for video_id, value in zip(metadata["video_id"], metadata["external_test_split"])
        if value.lower() not in ALLOWED_SPLITS
    ]
    rows.append(
        {
            "check": "external_test_split controlled values",
            "status": "PASS" if not invalid_splits else "WARN",
            "affected_rows": len(invalid_splits),
            "detail": ";".join(invalid_splits[:20]),
            "recommended_action": (
                "Use internal, external, holdout, train, validation, or test for "
                "external_test_split."
            ),
        }
    )

    cow_values = metadata["cow_id"].map(normalize_text)
    rows.append(
        {
            "check": "cow_id numeric-video-label grouping ready",
            "status": "PASS" if cow_values.ne("").all() and cow_values.nunique() >= 2 else "FAIL",
            "affected_rows": int(cow_values.eq("").sum()),
            "detail": f"nonempty={int(cow_values.ne('').sum())}/{len(cow_values)}, unique={cow_values[cow_values.ne('')].nunique()}",
            "recommended_action": "Fill cow_id from the numeric video label for all videos; use a separate real-cow identity map before animal-level grouping claims.",
        }
    )

    temp = metadata["ambient_temperature_c"].map(normalize_text)
    rh = metadata["relative_humidity_percent"].map(normalize_text)
    thi = metadata["thi"].map(normalize_text)
    heat_ready = (temp.ne("") & rh.ne("")) | thi.ne("")
    rows.append(
        {
            "check": "THI inputs ready",
            "status": "PASS" if heat_ready.all() else "FAIL",
            "affected_rows": int((~heat_ready).sum()),
            "detail": f"ready={int(heat_ready.sum())}/{len(heat_ready)}",
            "recommended_action": "Fill temperature+humidity or provide THI for every video.",
        }
    )
    return pd.DataFrame(rows)


def progress_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in METADATA_COLUMNS:
        if column == "video_id":
            continue
        values = metadata[column].map(normalize_text)
        rows.append(
            {
                "field": column,
                "nonempty": int(values.ne("").sum()),
                "total_videos": int(len(metadata)),
                "coverage_percent": values.ne("").sum() / max(len(metadata), 1) * 100.0,
                "unique_values": int(values[values.ne("")].nunique()),
            }
        )
    return pd.DataFrame(rows)


def write_report(validation: pd.DataFrame, progress: pd.DataFrame, output_path: Path, wrote: bool) -> None:
    failing = validation[validation["status"].isin(["FAIL", "WARN"])].copy()
    text = [
        "# RR Metadata Import Validation",
        "",
        f"Template write performed: `{wrote}`",
        "",
        "## Validation Issues",
        "",
    ]
    if failing.empty:
        text.append("No validation issues detected.")
    else:
        text.append(to_markdown(failing, ["check", "status", "affected_rows", "detail", "recommended_action"]))
    text.extend(["", "## Field Coverage", "", to_markdown(progress, ["field", "nonempty", "total_videos", "coverage_percent", "unique_values"])])
    output_path.write_text("\n".join(text), encoding="utf-8")


def to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    paper_assets = input_root / f"{args.corrected_prefix}_paper_assets"
    annotation_csv = args.annotation_csv or input_root / f"{args.output_prefix}_metadata_annotation_sheet.csv"
    metadata_template = args.metadata_template or paper_assets / "paper_metadata_template.csv"

    metadata = load_annotation(annotation_csv)
    summary_video_ids = load_summary_video_ids(input_root, args.output_prefix)
    metadata = pd.DataFrame({"video_id": summary_video_ids}).merge(
        metadata, on="video_id", how="left"
    )
    for column in METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata[METADATA_COLUMNS].fillna("").map(normalize_text)

    validation = validate_metadata(metadata, summary_video_ids)
    progress = progress_table(metadata)

    validation_csv = input_root / f"{args.output_prefix}_metadata_import_validation.csv"
    progress_csv = input_root / f"{args.output_prefix}_metadata_import_progress.csv"
    report_md = input_root / f"{args.output_prefix}_metadata_import_validation.md"
    validation.to_csv(validation_csv, index=False)
    progress.to_csv(progress_csv, index=False)
    write_report(validation, progress, report_md, bool(args.write_template))

    if args.write_template:
        metadata_template.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_template, index=False)

    print(f"Saved validation: {validation_csv}")
    print(f"Saved progress: {progress_csv}")
    print(f"Saved report: {report_md}")
    print(f"Template write performed: {bool(args.write_template)}")
    if args.write_template:
        print(f"Wrote metadata template: {metadata_template}")
    print(validation[validation["status"].isin(["FAIL", "WARN"])].to_string(index=False))


if __name__ == "__main__":
    main()
