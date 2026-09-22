from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRUTH_COLUMNS = [
    "video_id",
    "cow_id",
    "breath_count",
    "duration_seconds",
    "rr",
]

METADATA_COLUMNS = [
    "video_id",
    "cow_id",
    "source_session_id",
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

COMMAND_COLUMNS = [
    "step_order",
    "step",
    "purpose",
    "command",
    "expected_output",
]

EXTERNAL_SPLITS = {"external", "holdout", "test"}
EXTERNAL_VIDEO_ID_RE = re.compile(
    r"^NM_Hulunbuir_Jiufu_(?P<datetime>\d{8}T\d{6})_cow(?P<cow>.+?)_clip\d+$"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Convert a filled external-validation fieldwork worksheet into the "
            "truth CSV, metadata CSV, and frozen-evaluation command pack expected "
            "by the thermal RR pipeline."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--external-input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", default=r"E:\real\anaconda\envs\plant_gpu\python.exe")
    parser.add_argument("--external-output-prefix", default="external_repro")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_fieldwork_csv(args: argparse.Namespace) -> Path:
    return output_dir_for(args) / "paper_external_validation_fieldwork_template.csv"


def default_external_input_root(args: argparse.Namespace) -> Path:
    return args.input_root.parent / "external_al_images"


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1", "include", "included"]
    )


def active_mask(data: pd.DataFrame) -> pd.Series:
    columns = [
        "external_video_id",
        "raw_video_path",
        "cow_id",
        "manual_breath_count",
        "manual_duration_seconds",
        "manual_rr_bpm",
    ]
    present = [column for column in columns if column in data.columns]
    if not present:
        return pd.Series(False, index=data.index)
    active = pd.Series(False, index=data.index)
    for column in present:
        active = active | data[column].map(normalize_text).ne("")
    return active


def included_rows(fieldwork: pd.DataFrame) -> pd.DataFrame:
    if "include_in_external_validation" not in fieldwork.columns:
        return fieldwork.iloc[0:0].copy()
    included = fieldwork[
        yes_mask(fieldwork["include_in_external_validation"]) & active_mask(fieldwork)
    ].copy()
    return included.reset_index(drop=True)


def standard_video_id(row: pd.Series) -> str:
    external_id = normalize_text(row.get("external_video_id", ""))
    if external_id:
        return external_id
    raw_path = normalize_text(row.get("raw_video_path", ""))
    if raw_path:
        return Path(raw_path).stem
    return ""


def source_session_id(row: pd.Series) -> str:
    explicit = normalize_text(row.get("source_session_id", ""))
    if explicit:
        return explicit
    match = EXTERNAL_VIDEO_ID_RE.match(standard_video_id(row))
    if match:
        return f"{match.group('datetime')}-{match.group('cow')}"
    return normalize_text(row.get("scene_id", ""))


def numeric_text(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(number):
        return text
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.6f}".rstrip("0").rstrip(".")


def computed_rr(row: pd.Series) -> str:
    rr = pd.to_numeric(pd.Series([row.get("manual_rr_bpm", "")]), errors="coerce").iloc[0]
    if pd.notna(rr):
        return numeric_text(rr)
    count = pd.to_numeric(pd.Series([row.get("manual_breath_count", "")]), errors="coerce").iloc[0]
    duration = pd.to_numeric(pd.Series([row.get("manual_duration_seconds", "")]), errors="coerce").iloc[0]
    if pd.notna(count) and pd.notna(duration) and float(duration) > 0:
        return f"{float(count) * 60.0 / float(duration):.6f}".rstrip("0").rstrip(".")
    return ""


def build_truth(included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in included.iterrows():
        rows.append(
            {
                "video_id": standard_video_id(row),
                "cow_id": normalize_text(row.get("cow_id", "")),
                "breath_count": numeric_text(row.get("manual_breath_count", "")),
                "duration_seconds": numeric_text(row.get("manual_duration_seconds", "")),
                "rr": computed_rr(row),
            }
        )
    return pd.DataFrame(rows, columns=TRUTH_COLUMNS)


def build_metadata(included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in included.iterrows():
        split = normalize_text(row.get("external_test_split", "")).lower()
        if not split:
            split = "external"
        notes = [
            "Prepared from external validation fieldwork worksheet.",
            normalize_text(row.get("external_collection_notes", "")),
            normalize_text(row.get("reference_protocol_notes", "")),
        ]
        metadata = {
            "video_id": standard_video_id(row),
            "cow_id": normalize_text(row.get("cow_id", "")),
            "source_session_id": source_session_id(row),
            "collection_date": normalize_text(row.get("collection_date", "")),
            "collection_start_date": normalize_text(row.get("collection_date", "")),
            "collection_end_date": normalize_text(row.get("collection_date", "")),
            "collection_time": normalize_text(row.get("collection_time", "")),
            "collection_location_country": normalize_text(
                row.get("collection_location_country", "")
            ),
            "collection_location_province": normalize_text(
                row.get("collection_location_province", "")
            ),
            "collection_location_county": normalize_text(
                row.get("collection_location_county", "")
            ),
            "collection_site": normalize_text(row.get("collection_site", "")),
            "camera_id": normalize_text(row.get("camera_id", "")),
            "scene_id": normalize_text(row.get("scene_id", "")),
            "ambient_temperature_c": numeric_text(row.get("ambient_temperature_c", "")),
            "relative_humidity_percent": numeric_text(
                row.get("relative_humidity_percent", "")
            ),
            "thi": numeric_text(row.get("thi", "")),
            "athi": numeric_text(row.get("athi", "")),
            "posture": normalize_text(row.get("posture", "")),
            "head_motion_score_0_3": numeric_text(row.get("head_motion_score_0_3", "")),
            "occlusion_score_0_3": numeric_text(row.get("occlusion_score_0_3", "")),
            "nostril_visibility_score_0_3": numeric_text(
                row.get("nostril_visibility_score_0_3", "")
            ),
            "operator_or_annotator": normalize_text(
                row.get("reference_rr_annotator", "")
            ),
            "external_test_split": split,
            "notes": " ".join(note for note in notes if note),
        }
        rows.append(metadata)
    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def build_issues(fieldwork: pd.DataFrame, included: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if fieldwork.empty:
        rows.append(
            {
                "severity": "FAIL",
                "issue": "fieldwork worksheet is empty",
                "evidence": "rows=0",
                "recommended_action": "Fill paper_external_validation_fieldwork_template.csv.",
            }
        )
    if included.empty:
        rows.append(
            {
                "severity": "FAIL",
                "issue": "no included external rows",
                "evidence": "include_in_external_validation has no active yes rows",
                "recommended_action": "Fill independent external rows and set include_in_external_validation=yes.",
            }
        )
    for column in TRUTH_COLUMNS:
        if truth.empty:
            continue
        missing = int(truth[column].map(normalize_text).eq("").sum())
        if missing:
            rows.append(
                {
                    "severity": "FAIL",
                    "issue": f"truth column missing values: {column}",
                    "evidence": f"missing={missing}/{len(truth)}",
                    "recommended_action": f"Fill {column} for every included external row.",
                }
            )
    if not truth.empty:
        duplicate_ids = truth["video_id"][truth["video_id"].duplicated()].dropna().astype(str).tolist()
        if duplicate_ids:
            rows.append(
                {
                    "severity": "FAIL",
                    "issue": "duplicate external video IDs",
                    "evidence": ";".join(duplicate_ids[:20]),
                    "recommended_action": "Use one unique external_video_id per external video folder.",
                }
            )
    if "external_test_split" in included.columns:
        splits = included["external_test_split"].map(lambda value: normalize_text(value).lower())
        bad = int((~splits.isin(EXTERNAL_SPLITS)).sum()) if len(splits) else 0
        if bad:
            rows.append(
                {
                    "severity": "FAIL",
                    "issue": "included rows are not all labeled external/holdout/test",
                    "evidence": f"bad={bad}/{len(included)}",
                    "recommended_action": "Only independent rows should be included and labeled external, holdout, or test.",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["severity", "issue", "evidence", "recommended_action"],
    )


def quote_path(path: Path) -> str:
    return f'"{path}"'


def build_commands(
    args: argparse.Namespace,
    external_input_root: Path,
    fieldwork_csv: Path,
    truth_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    py = quote_path(Path(args.python))
    output_prefix = args.external_output_prefix
    method_freeze_csv = output_dir / "paper_method_freeze_summary.csv"
    annotator_a_export = output_dir / "paper_external_validation_breath_annotation_annotator_a_export.csv"
    annotator_b_export = output_dir / "paper_external_validation_breath_annotation_annotator_b_export.csv"
    consensus_csv = output_dir / "paper_external_validation_breath_annotation_consensus.csv"
    adjudication_csv = output_dir / "paper_external_validation_breath_annotation_adjudication_template.csv"
    annotated_fieldwork_csv = output_dir / "paper_external_validation_split_all_use_fieldwork_consensus.csv"
    rows = [
        {
            "step_order": 1,
            "step": "build_annotator_a_packet",
            "purpose": "Build an independent HTML packet for annotator A.",
            "command": (
                f"& {py} scripts\\build_rr_external_breath_annotation_packet.py "
                "--annotator-id annotator_a "
                "--export-name paper_external_validation_breath_annotation_annotator_a_export.csv "
                "--blind --blind-seed 20260710"
            ),
            "expected_output": "paper_external_validation_breath_annotation_packet_annotator_a.html",
        },
        {
            "step_order": 2,
            "step": "build_annotator_b_packet",
            "purpose": "Build an independent HTML packet for annotator B.",
            "command": (
                f"& {py} scripts\\build_rr_external_breath_annotation_packet.py "
                "--annotator-id annotator_b "
                "--export-name paper_external_validation_breath_annotation_annotator_b_export.csv "
                "--blind --blind-seed 20260711"
            ),
            "expected_output": "paper_external_validation_breath_annotation_packet_annotator_b.html",
        },
        {
            "step_order": 3,
            "step": "audit_dual_annotation_agreement",
            "purpose": (
                "Compare two exported breath-count annotation CSVs and create "
                "a consensus CSV plus adjudication template."
            ),
            "command": (
                f"& {py} scripts\\audit_rr_external_breath_annotation_agreement.py "
                f"--annotation-a-csv {quote_path(annotator_a_export)} "
                f"--annotation-b-csv {quote_path(annotator_b_export)}"
            ),
            "expected_output": "paper_external_validation_breath_annotation_consensus.csv",
        },
        {
            "step_order": 4,
            "step": "fill_adjudication_template_if_needed",
            "purpose": (
                "Fill adjudicated_breath_count for rows listed in the "
                "adjudication template when A/B labels are missing or disagree."
            ),
            "command": str(adjudication_csv),
            "expected_output": "adjudication template has no unresolved included rows",
        },
        {
            "step_order": 5,
            "step": "rerun_agreement_with_adjudication",
            "purpose": (
                "Regenerate the consensus CSV after adjudication so only "
                "consensus-ready rows enter external scoring."
            ),
            "command": (
                f"& {py} scripts\\audit_rr_external_breath_annotation_agreement.py "
                f"--annotation-a-csv {quote_path(annotator_a_export)} "
                f"--annotation-b-csv {quote_path(annotator_b_export)} "
                f"--adjudication-csv {quote_path(adjudication_csv)}"
            ),
            "expected_output": "paper_external_validation_breath_annotation_consensus.csv",
        },
        {
            "step_order": 6,
            "step": "merge_consensus_annotations",
            "purpose": "Merge consensus breath counts into an annotated fieldwork worksheet.",
            "command": (
                f"& {py} scripts\\merge_rr_external_breath_annotations.py "
                f"--annotations-csv {quote_path(consensus_csv)} "
                f"--output-csv {quote_path(annotated_fieldwork_csv)}"
            ),
            "expected_output": str(annotated_fieldwork_csv),
        },
        {
            "step_order": 7,
            "step": "regenerate_input_pack_from_consensus_fieldwork",
            "purpose": "Regenerate truth, metadata, and run commands from consensus fieldwork.",
            "command": (
                f"& {py} scripts\\prepare_rr_external_validation_inputs.py "
                f"--fieldwork-csv {quote_path(annotated_fieldwork_csv)}"
            ),
            "expected_output": "paper_external_validation_truth_template.csv",
        },
        {
            "step_order": 8,
            "step": "preflight_fieldwork",
            "purpose": "Validate the filled fieldwork worksheet before scoring.",
            "command": (
                f"& {py} scripts\\preflight_rr_external_fieldwork.py "
                f"--fieldwork-csv {quote_path(annotated_fieldwork_csv)}"
            ),
            "expected_output": "paper_external_validation_fieldwork_preflight.csv",
        },
        {
            "step_order": 9,
            "step": "extract_external_clip_frames",
            "purpose": "Extract raw mp4 clips into one frame folder per external_video_id.",
            "command": (
                f"& {py} scripts\\extract_rr_external_clip_frames.py "
                f"--fieldwork-csv {quote_path(annotated_fieldwork_csv)} "
                f"--external-input-root {quote_path(external_input_root)} "
                "--include-only --overwrite"
            ),
            "expected_output": f"{external_input_root}\\external_clip_frame_extraction_report.csv",
        },
        {
            "step_order": 10,
            "step": "run_frozen_external_rr",
            "purpose": "Run the frozen paper-style RR pipeline on external frame folders.",
            "command": (
                f"& {py} scripts\\paper_repro_rr.py "
                f"--input-root {quote_path(external_input_root)} "
                f"--truth-csv {quote_path(truth_csv)} "
                f"--output-prefix {output_prefix} "
                "--overwrite"
            ),
            "expected_output": f"{external_input_root}\\{output_prefix}_summary.csv",
        },
        {
            "step_order": 11,
            "step": "apply_frozen_postprocessors",
            "purpose": (
                "Apply the frozen internal residual, signal-consensus, and safe-gate "
                "postprocessors to external default RR outputs without using external truth."
            ),
            "command": (
                f"& {py} scripts\\apply_rr_frozen_external_postprocessors.py "
                f"--input-root {quote_path(args.input_root)} "
                f"--external-input-root {quote_path(external_input_root)} "
                f"--external-output-prefix {output_prefix} "
                f"--corrected-prefix {args.corrected_prefix}"
            ),
            "expected_output": (
                f"{external_input_root}\\{output_prefix}_signal_aware_safe_policy_predictions.csv"
            ),
        },
        {
            "step_order": 12,
            "step": "score_external_split",
            "purpose": "Evaluate external metrics after predictions exist.",
            "command": (
                f"& {py} scripts\\rr_external_split_validation.py "
                f"--input-root {quote_path(external_input_root)} "
                f"--metadata-csv {quote_path(metadata_csv)} "
                f"--output-prefix {output_prefix} "
                f"--corrected-prefix {args.corrected_prefix} "
                f"--method-freeze-csv {quote_path(method_freeze_csv)}"
            ),
            "expected_output": f"{external_input_root}\\{output_prefix}_external_split_metrics.csv",
        },
        {
            "step_order": 13,
            "step": "refresh_submission_assets",
            "purpose": "Refresh manuscript and asset summaries after external scoring.",
            "command": (
                f"& {py} scripts\\refresh_rr_q2_package.py "
                "--steps external_split,readiness,submission_gap,submission_route_pack,assets_final"
            ),
            "expected_output": "paper_repro_quality_residual_paper_assets refreshed",
        },
    ]
    return pd.DataFrame(rows, columns=COMMAND_COLUMNS)


def markdown_table(data: pd.DataFrame, max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows available._"
    table = data.head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.to_numpy():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    truth: pd.DataFrame,
    metadata: pd.DataFrame,
    issues: pd.DataFrame,
    commands: pd.DataFrame,
    external_input_root: Path,
) -> None:
    status = "ready_to_run_external_rr" if issues.empty and not truth.empty else "not_ready"
    lines = [
        "# External Validation Input Pack",
        "",
        f"Status: `{status}`",
        f"Included external rows: `{len(truth)}`",
        f"External input root: `{external_input_root}`",
        "",
        "## Issues",
        "",
        markdown_table(issues),
        "",
        "## Generated Truth Preview",
        "",
        markdown_table(truth),
        "",
        "## Generated Metadata Preview",
        "",
        markdown_table(metadata[["video_id", "cow_id", "collection_date", "camera_id", "scene_id", "external_test_split"]]),
        "",
        "## Run Commands",
        "",
        markdown_table(commands, max_rows=20),
        "",
        "## Boundary",
        "",
        "Use this pack only for independent external videos. The current 73 development videos remain internal. Do not rerun threshold selection on the external rows.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_with_fallback(data: pd.DataFrame, path: Path) -> Path:
    try:
        data.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_pending{path.suffix}")
        data.to_csv(fallback, index=False)
        return fallback


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldwork_csv = (
        args.fieldwork_csv.resolve()
        if args.fieldwork_csv is not None
        else default_fieldwork_csv(args).resolve()
    )
    external_input_root = (
        args.external_input_root.resolve()
        if args.external_input_root is not None
        else default_external_input_root(args).resolve()
    )
    if not fieldwork_csv.exists():
        raise FileNotFoundError(f"Missing fieldwork CSV: {fieldwork_csv}")

    fieldwork = pd.read_csv(fieldwork_csv, dtype=str, keep_default_na=False)
    included = included_rows(fieldwork)
    truth = build_truth(included)
    metadata = build_metadata(included)
    issues = build_issues(fieldwork, included, truth)

    truth_path = output_dir / "paper_external_validation_truth_template.csv"
    metadata_path = output_dir / "paper_external_validation_metadata_template.csv"
    issues_path = output_dir / "paper_external_validation_input_pack_issues.csv"
    commands_path = output_dir / "paper_external_validation_run_commands.csv"
    report_path = output_dir / "paper_external_validation_input_pack.md"

    commands = build_commands(
        args,
        external_input_root,
        fieldwork_csv,
        truth_path,
        metadata_path,
        output_dir,
    )
    truth_path_written = write_csv_with_fallback(truth, truth_path)
    metadata_path_written = write_csv_with_fallback(metadata, metadata_path)
    issues_path_written = write_csv_with_fallback(issues, issues_path)
    commands_path_written = write_csv_with_fallback(commands, commands_path)
    write_report(report_path, truth, metadata, issues, commands, external_input_root)

    print(f"Saved external truth template: {truth_path_written}")
    print(f"Saved external metadata template: {metadata_path_written}")
    print(f"Saved external input-pack issues: {issues_path_written}")
    print(f"Saved external run commands: {commands_path_written}")
    print(f"Saved external input-pack report: {report_path}")
    print(f"Included external rows: {len(truth)}")
    print(f"Issues: {len(issues)}")


if __name__ == "__main__":
    main()
