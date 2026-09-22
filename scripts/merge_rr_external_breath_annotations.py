from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UPDATE_COLUMNS = [
    "manual_breath_count",
    "manual_rr_bpm",
    "reference_rr_annotator",
    "camera_id",
    "reference_quality_status",
    "reference_count_uncertainty_breaths",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Merge exported external breath-count annotations back into an "
            "external validation fieldwork CSV."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--annotations-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the source fieldwork CSV after writing the merge report.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_fieldwork_csv(output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_split_all_use_fieldwork_template.csv"


def default_annotations_csv(output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_breath_annotation_export.csv"


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def numeric_text(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.6f}".rstrip("0").rstrip(".")


def compute_rr(count_text: str, duration_text: str) -> str:
    count = pd.to_numeric(pd.Series([count_text]), errors="coerce").iloc[0]
    duration = pd.to_numeric(pd.Series([duration_text]), errors="coerce").iloc[0]
    if pd.isna(count) or pd.isna(duration) or float(duration) <= 0:
        return ""
    return f"{float(count) * 60.0 / float(duration):.6f}".rstrip("0").rstrip(".")


def merge_annotations(
    fieldwork: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = fieldwork.copy()
    for column in UPDATE_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    if "reference_protocol_notes" not in merged.columns:
        merged["reference_protocol_notes"] = ""

    annotations_by_id = {
        normalize_text(row.get("external_video_id", "")): row
        for _, row in annotations.iterrows()
        if normalize_text(row.get("external_video_id", ""))
    }
    report_rows: list[dict[str, object]] = []
    matched = 0
    updated = 0
    for idx, row in merged.iterrows():
        video_id = normalize_text(row.get("external_video_id", ""))
        annotation = annotations_by_id.get(video_id)
        if annotation is None:
            continue
        matched += 1
        row_updates = 0
        for column in [
            "manual_breath_count",
            "reference_rr_annotator",
            "camera_id",
            "reference_quality_status",
            "reference_count_uncertainty_breaths",
        ]:
            value = normalize_text(annotation.get(column, ""))
            if value:
                if column == "manual_breath_count":
                    value = numeric_text(value)
                elif column == "reference_count_uncertainty_breaths":
                    value = numeric_text(value)
                merged.at[idx, column] = value
                row_updates += 1
        rr_value = normalize_text(annotation.get("manual_rr_bpm", ""))
        if rr_value:
            merged.at[idx, "manual_rr_bpm"] = numeric_text(rr_value)
            row_updates += 1
        elif normalize_text(merged.at[idx, "manual_breath_count"]):
            computed = compute_rr(
                merged.at[idx, "manual_breath_count"],
                merged.at[idx, "manual_duration_seconds"],
            )
            if computed:
                merged.at[idx, "manual_rr_bpm"] = computed
                row_updates += 1

        notes = normalize_text(annotation.get("annotation_notes", ""))
        if notes:
            existing = normalize_text(merged.at[idx, "reference_protocol_notes"])
            merged.at[idx, "reference_protocol_notes"] = (
                f"{existing} Annotation note: {notes}".strip()
                if existing
                else f"Annotation note: {notes}"
            )
            row_updates += 1
        if row_updates:
            updated += 1
        report_rows.append(
            {
                "external_video_id": video_id,
                "matched_annotation": True,
                "updated_fields": row_updates,
                "manual_breath_count": normalize_text(merged.at[idx, "manual_breath_count"]),
                "manual_rr_bpm": normalize_text(merged.at[idx, "manual_rr_bpm"]),
                "reference_rr_annotator": normalize_text(
                    merged.at[idx, "reference_rr_annotator"]
                ),
                "camera_id": normalize_text(merged.at[idx, "camera_id"]),
            }
        )

    missing_in_fieldwork = sorted(
        video_id
        for video_id in annotations_by_id
        if video_id not in set(merged["external_video_id"].fillna("").astype(str))
    )
    for video_id in missing_in_fieldwork:
        report_rows.append(
            {
                "external_video_id": video_id,
                "matched_annotation": False,
                "updated_fields": 0,
                "manual_breath_count": "",
                "manual_rr_bpm": "",
                "reference_rr_annotator": "",
                "camera_id": "",
            }
        )
    report = pd.DataFrame(report_rows)
    summary = pd.DataFrame(
        [
            {
                "external_video_id": "__summary__",
                "matched_annotation": True,
                "updated_fields": updated,
                "manual_breath_count": f"matched_rows={matched}",
                "manual_rr_bpm": f"annotation_rows={len(annotations_by_id)}",
                "reference_rr_annotator": f"updated_rows={updated}",
                "camera_id": f"missing_in_fieldwork={len(missing_in_fieldwork)}",
            }
        ]
    )
    report = pd.concat([summary, report], ignore_index=True)
    return merged, report


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldwork_csv = (
        args.fieldwork_csv.resolve()
        if args.fieldwork_csv is not None
        else default_fieldwork_csv(output_dir).resolve()
    )
    annotations_csv = (
        args.annotations_csv.resolve()
        if args.annotations_csv is not None
        else default_annotations_csv(output_dir).resolve()
    )
    output_csv = (
        fieldwork_csv
        if args.in_place
        else (
            args.output_csv.resolve()
            if args.output_csv is not None
            else output_dir
            / "paper_external_validation_split_all_use_fieldwork_annotated.csv"
        )
    )
    if not fieldwork_csv.exists():
        raise FileNotFoundError(f"Missing fieldwork CSV: {fieldwork_csv}")
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Missing annotation CSV: {annotations_csv}")
    fieldwork = pd.read_csv(fieldwork_csv, dtype=str, keep_default_na=False)
    annotations = pd.read_csv(annotations_csv, dtype=str, keep_default_na=False)
    if "external_video_id" not in annotations.columns:
        raise ValueError("Annotation CSV must include external_video_id")
    merged, report = merge_annotations(fieldwork, annotations)

    report_path = output_dir / "paper_external_validation_breath_annotation_merge_report.csv"
    merged.to_csv(output_csv, index=False)
    report.to_csv(report_path, index=False)

    summary = report.iloc[0]
    print(f"Saved merged fieldwork CSV: {output_csv}")
    print(f"Saved annotation merge report: {report_path}")
    print(
        f"{summary['manual_breath_count']}; {summary['manual_rr_bpm']}; "
        f"{summary['reference_rr_annotator']}; {summary['camera_id']}"
    )


if __name__ == "__main__":
    main()
