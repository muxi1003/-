from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
GENERATED_MARKERS = {
    "curve",
    "respiration",
    "paper_repro",
    "quality_residual",
    "review",
    "final_",
    "pinghua",
    "origin",
}

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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build an annotation sheet and codebook for filling cow-level, "
            "scene-level, heat-stress, and video-quality metadata."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def non_generated_images(video_dir: Path) -> list[Path]:
    images = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lower = path.name.lower()
        if any(marker in lower for marker in GENERATED_MARKERS):
            continue
        images.append(path)
    return images


def sample_frame_paths(video_dir: Path) -> tuple[str, str, str]:
    images = non_generated_images(video_dir)
    if not images:
        return "", "", ""
    indices = [0, len(images) // 2, len(images) - 1]
    return tuple(str(images[index]) for index in indices)  # type: ignore[return-value]


def ensure_metadata_columns(metadata: pd.DataFrame, video_ids: pd.Series) -> pd.DataFrame:
    if metadata.empty:
        metadata = pd.DataFrame({"video_id": video_ids.astype(str)})
    if "video_id" not in metadata.columns:
        raise ValueError("Existing metadata file is missing required column: video_id")
    metadata["video_id"] = metadata["video_id"].astype(str)
    base = pd.DataFrame({"video_id": video_ids.astype(str)})
    merged = base.merge(metadata, on="video_id", how="left")
    for column in METADATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    return merged[METADATA_COLUMNS].fillna("")


def build_annotation_sheet(
    input_root: Path,
    output_prefix: str,
    corrected_prefix: str,
    output_dir: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(input_root / f"{output_prefix}_summary.csv", dtype=str, keep_default_na=False)
    predictions = read_optional_csv(input_root / f"{corrected_prefix}_predictions.csv")
    signal_predictions = read_optional_csv(input_root / f"{output_prefix}_signal_consensus_predictions.csv")
    signal_aware_predictions = read_optional_csv(
        input_root / f"{output_prefix}_signal_aware_residual_predictions.csv"
    )
    signal_aware_group_predictions = read_optional_csv(
        input_root / f"{output_prefix}_signal_aware_residual_group_predictions.csv"
    )
    selective_predictions = read_optional_csv(input_root / f"{output_prefix}_selective_rr_predictions.csv")
    taxonomy = read_optional_csv(input_root / f"{output_prefix}_error_taxonomy.csv")
    case_examples = read_optional_csv(input_root / f"{corrected_prefix}_case_examples.csv")
    metadata_path = output_dir / "paper_metadata_template.csv"
    metadata = ensure_metadata_columns(read_optional_csv(metadata_path), summary["video_id"])

    context_columns = [
        "video_id",
        "truth_rr",
        "rr_bpm",
        "truth_count",
        "peaks",
        "count_error",
        "abs_count_error",
        "duration_seconds",
        "selected_fusion_mode",
        "selection_missing_rate",
        "curve_png",
        "review_png",
    ]
    available_summary = [column for column in context_columns if column in summary.columns]
    context = summary[available_summary].copy()
    if not predictions.empty:
        prediction_columns = [
            "video_id",
            "applied_adjust",
            "corrected_peaks",
            "corrected_count_error",
            "corrected_rr_bpm",
            "residual_confidence",
            "residual_margin",
        ]
        available = [column for column in prediction_columns if column in predictions.columns]
        context = context.merge(predictions[available], on="video_id", how="left")
    if not signal_predictions.empty:
        signal_columns = [
            "video_id",
            "signal_consensus_peaks",
            "signal_consensus_rr_bpm",
            "signal_consensus_adjust",
            "signal_consensus_adjust_reason",
            "signal_consensus_adjust_signal_votes",
            "spectral_count_estimate",
            "fft_count_estimate",
            "autocorr_count_estimate",
        ]
        available = [column for column in signal_columns if column in signal_predictions.columns]
        context = context.merge(signal_predictions[available], on="video_id", how="left")
    if not signal_aware_predictions.empty:
        signal_aware_columns = [
            "video_id",
            "signal_aware_applied_adjust",
            "signal_aware_confidence",
            "signal_aware_margin",
            "signal_aware_final_peaks",
            "signal_aware_final_rr_bpm",
            "signal_aware_count_error",
            "signal_aware_abs_count_error",
            "signal_aware_abs_rr_error",
            "signal_aware_evaluation_note",
        ]
        available = [column for column in signal_aware_columns if column in signal_aware_predictions.columns]
        context = context.merge(signal_aware_predictions[available], on="video_id", how="left")
    if not signal_aware_group_predictions.empty:
        signal_aware_group_columns = [
            "video_id",
            "signal_aware_group_applied_adjust",
            "signal_aware_group_confidence",
            "signal_aware_group_margin",
            "signal_aware_group_final_peaks",
            "signal_aware_group_final_rr_bpm",
            "signal_aware_group_count_error",
            "signal_aware_group_abs_count_error",
            "signal_aware_group_abs_rr_error",
            "signal_aware_group_evaluation_note",
        ]
        available = [
            column
            for column in signal_aware_group_columns
            if column in signal_aware_group_predictions.columns
        ]
        context = context.merge(signal_aware_group_predictions[available], on="video_id", how="left")
    if not selective_predictions.empty:
        selective_columns = [
            "video_id",
            "signal_count_vote_agreement",
            "selective_review_score",
            "strict_auto_accept",
            "score_auto_accept",
            "selective_action",
            "selective_triage_reason",
            "selective_abs_count_error",
            "selective_abs_rr_error",
        ]
        available = [column for column in selective_columns if column in selective_predictions.columns]
        context = context.merge(selective_predictions[available], on="video_id", how="left")
    if not taxonomy.empty:
        taxonomy_columns = ["video_id", "error_type", "error_reason"]
        available = [column for column in taxonomy_columns if column in taxonomy.columns]
        context = context.merge(taxonomy[available], on="video_id", how="left")
    if not case_examples.empty:
        cases = case_examples[["video_id", "case_label"]].copy()
        context = context.merge(cases, on="video_id", how="left")

    sample_rows = []
    for video_id in summary["video_id"].astype(str):
        first, middle, last = sample_frame_paths(input_root / video_id)
        sample_rows.append(
            {
                "video_id": video_id,
                "sample_frame_first": first,
                "sample_frame_middle": middle,
                "sample_frame_last": last,
            }
        )
    samples = pd.DataFrame(sample_rows)

    sheet = metadata.merge(context, on="video_id", how="left").merge(samples, on="video_id", how="left")
    sheet["q2_missing_required_fields"] = sheet.apply(missing_required_fields, axis=1)
    sheet["q2_annotation_score"] = sheet.apply(q2_annotation_score, axis=1)
    sheet["q2_annotation_batch"] = sheet.apply(q2_annotation_batch, axis=1)
    sheet["annotation_priority"] = sheet.apply(annotation_priority, axis=1)
    sheet["annotation_reason"] = sheet.apply(annotation_reason, axis=1)
    ordered = [
        *METADATA_COLUMNS,
        "annotation_priority",
        "q2_annotation_score",
        "q2_annotation_batch",
        "q2_missing_required_fields",
        "annotation_reason",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "signal_consensus_rr_bpm",
        "signal_aware_final_rr_bpm",
        "signal_aware_group_final_rr_bpm",
        "truth_count",
        "peaks",
        "corrected_peaks",
        "signal_consensus_peaks",
        "signal_aware_final_peaks",
        "signal_aware_group_final_peaks",
        "count_error",
        "corrected_count_error",
        "signal_aware_count_error",
        "signal_aware_group_count_error",
        "selective_abs_count_error",
        "abs_count_error",
        "signal_aware_abs_count_error",
        "signal_aware_group_abs_count_error",
        "applied_adjust",
        "signal_consensus_adjust",
        "signal_aware_applied_adjust",
        "signal_aware_group_applied_adjust",
        "signal_consensus_adjust_reason",
        "signal_consensus_adjust_signal_votes",
        "residual_confidence",
        "residual_margin",
        "signal_aware_confidence",
        "signal_aware_margin",
        "signal_aware_group_confidence",
        "signal_aware_group_margin",
        "signal_count_vote_agreement",
        "selective_review_score",
        "strict_auto_accept",
        "score_auto_accept",
        "selective_action",
        "selective_triage_reason",
        "selective_abs_rr_error",
        "signal_aware_abs_rr_error",
        "signal_aware_group_abs_rr_error",
        "signal_aware_evaluation_note",
        "signal_aware_group_evaluation_note",
        "error_type",
        "error_reason",
        "case_label",
        "duration_seconds",
        "selected_fusion_mode",
        "selection_missing_rate",
        "spectral_count_estimate",
        "fft_count_estimate",
        "autocorr_count_estimate",
        "sample_frame_first",
        "sample_frame_middle",
        "sample_frame_last",
        "curve_png",
        "review_png",
    ]
    for column in ordered:
        if column not in sheet.columns:
            sheet[column] = ""
    return sheet[ordered].fillna("")


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clean_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def as_bool(value: object) -> bool:
    text = clean_text(value).lower()
    return text in {"true", "1", "yes", "y"}


def missing_required_fields(row: pd.Series) -> str:
    required = [
        "cow_id",
        "collection_start_date",
        "collection_end_date",
        "collection_location_country",
        "collection_location_province",
        "collection_location_county",
        "collection_site",
        "scene_id",
        "external_test_split",
    ]
    missing = [column for column in required if not clean_text(row.get(column, ""))]
    return ";".join(missing)


def q2_annotation_score(row: pd.Series) -> int:
    score = 0
    if clean_text(row.get("case_label", "")):
        score += 30
    if as_float(row.get("abs_count_error")) >= 1:
        score += 25
    if as_float(row.get("selective_abs_count_error")) >= 1:
        score += 25
    if as_float(row.get("applied_adjust")) != 0:
        score += 15
    if as_float(row.get("signal_consensus_adjust")) != 0:
        score += 15
    if as_float(row.get("signal_aware_applied_adjust")) != 0:
        score += 20
    if as_float(row.get("signal_aware_group_applied_adjust")) != 0:
        score += 15
    if as_float(row.get("signal_aware_abs_count_error")) >= 1:
        score += 25
    if as_float(row.get("signal_aware_group_abs_count_error")) >= 1:
        score += 15
    if clean_text(row.get("selective_action", "")) == "manual_review_required":
        score += 10
    if as_float(row.get("selective_review_score")) >= 0.60:
        score += 10
    if clean_text(row.get("error_type", "")):
        score += 10
    if as_bool(row.get("strict_auto_accept", "")):
        score += 3
    if missing_required_fields(row):
        score += 5
    return int(score)


def q2_annotation_batch(row: pd.Series) -> str:
    if clean_text(row.get("case_label", "")) or as_float(row.get("abs_count_error")) >= 1:
        return "batch_1_residual_errors_and_representative_cases"
    if (
        as_float(row.get("selective_abs_count_error")) >= 1
        or as_float(row.get("applied_adjust")) != 0
        or as_float(row.get("signal_aware_applied_adjust")) != 0
        or as_float(row.get("signal_aware_group_applied_adjust")) != 0
        or as_float(row.get("signal_aware_abs_count_error")) >= 1
        or as_float(row.get("signal_aware_group_abs_count_error")) >= 1
    ):
        return "batch_2_corrected_or_selective_error_cases"
    if clean_text(row.get("selective_action", "")) == "manual_review_required" or as_float(
        row.get("selective_review_score")
    ) >= 0.60:
        return "batch_3_high_uncertainty_manual_review"
    if as_bool(row.get("strict_auto_accept", "")):
        return "batch_4_high_confidence_auto_report_controls"
    return "batch_5_remaining_metadata_completion"


def annotation_priority(row: pd.Series) -> str:
    score = q2_annotation_score(row)
    if score >= 55:
        return "critical"
    if clean_text(row.get("case_label", "")):
        return "high"
    if as_float(row.get("applied_adjust")) != 0:
        return "high"
    if as_float(row.get("signal_consensus_adjust")) != 0:
        return "high"
    if as_float(row.get("signal_aware_applied_adjust")) != 0:
        return "high"
    if as_float(row.get("signal_aware_abs_count_error")) >= 1:
        return "high"
    if as_float(row.get("signal_aware_group_abs_count_error")) >= 1:
        return "high"
    if as_float(row.get("abs_count_error")) >= 1:
        return "high"
    if as_float(row.get("selective_abs_count_error")) >= 1:
        return "high"
    if clean_text(row.get("error_type", "")):
        return "medium"
    if clean_text(row.get("selective_action", "")) == "manual_review_required":
        return "medium"
    return "normal"


def annotation_reason(row: pd.Series) -> str:
    reasons = []
    case_label = clean_text(row.get("case_label", ""))
    error_type = clean_text(row.get("error_type", ""))
    if case_label:
        reasons.append(f"representative_case={case_label}")
    if as_float(row.get("applied_adjust")) != 0:
        reasons.append(f"residual_adjust={row.get('applied_adjust')}")
    if as_float(row.get("signal_consensus_adjust")) != 0:
        reasons.append(f"signal_consensus_adjust={row.get('signal_consensus_adjust')}")
    if as_float(row.get("signal_aware_applied_adjust")) != 0:
        reasons.append(f"signal_aware_adjust={row.get('signal_aware_applied_adjust')}")
    if as_float(row.get("signal_aware_group_applied_adjust")) != 0:
        reasons.append(
            f"signal_aware_group_adjust={row.get('signal_aware_group_applied_adjust')}"
        )
    if as_float(row.get("abs_count_error")) >= 1:
        reasons.append("default_count_error")
    if as_float(row.get("selective_abs_count_error")) >= 1:
        reasons.append("signal_consensus_count_error")
    if as_float(row.get("signal_aware_abs_count_error")) >= 1:
        reasons.append("signal_aware_count_error")
    if as_float(row.get("signal_aware_group_abs_count_error")) >= 1:
        reasons.append("signal_aware_group_count_error")
    if error_type:
        reasons.append(f"error_type={error_type}")
    selective_action = clean_text(row.get("selective_action", ""))
    if selective_action:
        reasons.append(f"selective_action={selective_action}")
    review_score = clean_text(row.get("selective_review_score", ""))
    if review_score:
        reasons.append(f"selective_review_score={review_score}")
    missing = clean_text(row.get("q2_missing_required_fields", ""))
    if missing:
        reasons.append(f"missing_q2_fields={missing}")
    return "; ".join(reasons)


def progress_table(sheet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in METADATA_COLUMNS:
        if column == "video_id":
            continue
        values = sheet[column].astype(str).str.strip()
        nonempty = int((values != "").sum())
        rows.append(
            {
                "field": column,
                "nonempty": nonempty,
                "total_videos": int(len(sheet)),
                "coverage_percent": nonempty / max(len(sheet), 1) * 100.0,
                "unique_values": int(values[values != ""].nunique()),
            }
        )
    return pd.DataFrame(rows)


def priority_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch, batch_df in sheet.groupby("q2_annotation_batch", sort=True):
        priority_counts = batch_df["annotation_priority"].value_counts().to_dict()
        rows.append(
            {
                "q2_annotation_batch": batch,
                "videos": int(len(batch_df)),
                "critical": int(priority_counts.get("critical", 0)),
                "high": int(priority_counts.get("high", 0)),
                "medium": int(priority_counts.get("medium", 0)),
                "normal": int(priority_counts.get("normal", 0)),
                "max_q2_annotation_score": int(
                    pd.to_numeric(batch_df["q2_annotation_score"], errors="coerce").max()
                ),
                "mean_q2_annotation_score": float(
                    pd.to_numeric(batch_df["q2_annotation_score"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_fill_template(sheet: pd.DataFrame) -> pd.DataFrame:
    table = sheet.copy()
    table["q2_annotation_score_numeric"] = pd.to_numeric(
        table["q2_annotation_score"], errors="coerce"
    ).fillna(0)
    table = table.sort_values(
        ["q2_annotation_score_numeric", "q2_annotation_batch", "video_id"],
        ascending=[False, True, True],
    )
    return table[METADATA_COLUMNS].copy()


def q2_unblock_role(row: pd.Series) -> str:
    batch = clean_text(row.get("q2_annotation_batch", ""))
    if batch == "batch_1_residual_errors_and_representative_cases":
        return "residual_error_or_representative_case"
    if (
        as_float(row.get("signal_aware_applied_adjust")) != 0
        or as_float(row.get("signal_aware_abs_count_error")) >= 1
        or as_float(row.get("signal_aware_group_abs_count_error")) >= 1
    ):
        return "signal_aware_transition_or_failure_case"
    if clean_text(row.get("selective_action", "")) == "manual_review_required":
        return "uncertainty_review_case"
    if as_bool(row.get("strict_auto_accept", "")):
        return "high_confidence_control_case"
    return "metadata_completion_case"


def build_q2_unblock_queue(sheet: pd.DataFrame) -> pd.DataFrame:
    table = sheet.copy()
    table["q2_annotation_score_numeric"] = pd.to_numeric(
        table["q2_annotation_score"], errors="coerce"
    ).fillna(0)
    table["q2_unblock_role"] = table.apply(q2_unblock_role, axis=1)
    table["external_split_instruction"] = table["external_test_split"].apply(
        lambda value: (
            "already_filled_verify_no_leakage"
            if clean_text(value)
            else "fill_from_acquisition_design_only_do_not_infer_from_model_output"
        )
    )
    table["q2_claims_unlocked_by_fields"] = (
        "cow_id: numeric video label grouping per current dataset convention; "
        "external_test_split: split provenance, with frozen external validation only after independent external rows exist; "
        "ambient_temperature_c+relative_humidity_percent: optional synchronized barn measurements required only for THI claims; regional weather proxies are context only; "
        "head_motion+occlusion+nostril_visibility: optional manual scores required only for robustness stratification"
    )
    table["required_q2_metadata_fields"] = table["q2_missing_required_fields"]
    table = table.sort_values(
        [
            "q2_annotation_score_numeric",
            "q2_annotation_batch",
            "annotation_priority",
            "video_id",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    table.insert(0, "q2_unblock_rank", range(1, len(table) + 1))
    columns = [
        "q2_unblock_rank",
        "video_id",
        "q2_unblock_role",
        "annotation_priority",
        "q2_annotation_score",
        "q2_annotation_batch",
        "required_q2_metadata_fields",
        "external_split_instruction",
        "q2_claims_unlocked_by_fields",
        "annotation_reason",
        "truth_count",
        "peaks",
        "corrected_peaks",
        "signal_consensus_peaks",
        "signal_aware_final_peaks",
        "signal_aware_group_final_peaks",
        "abs_count_error",
        "signal_aware_abs_count_error",
        "signal_aware_group_abs_count_error",
        "selective_action",
        "selective_review_score",
        "sample_frame_first",
        "sample_frame_middle",
        "sample_frame_last",
        "curve_png",
        "review_png",
    ]
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    return table[columns].fillna("")


def write_q2_unblock_report(
    output_path: Path,
    queue: pd.DataFrame,
    progress: pd.DataFrame,
    priority: pd.DataFrame,
) -> Path:
    top_queue = queue.head(20).fillna("").astype(str)
    progress_table = progress.fillna("").astype(str)
    priority_table = priority.fillna("").astype(str)
    lines = [
        "# Q2 Unblock Annotation Queue",
        "",
        "This queue turns the current Q2-or-higher blockers into per-video annotation work. Use the numeric video label for `cow_id` when that is the available animal/video identifier, fill `external_test_split` from acquisition design, and do not infer temperature, humidity, or visual-quality scores from prediction errors.",
        "",
        "## Current Metadata Coverage",
        "",
        to_markdown(
            progress_table,
            ["field", "nonempty", "total_videos", "coverage_percent", "unique_values"],
        ),
        "",
        "## Batch Counts",
        "",
        to_markdown(
            priority_table,
            [
                "q2_annotation_batch",
                "videos",
                "critical",
                "high",
                "medium",
                "normal",
                "max_q2_annotation_score",
            ],
        ),
        "",
        "## Top 20 Videos To Annotate First",
        "",
        to_markdown(
            top_queue,
            [
                "q2_unblock_rank",
                "video_id",
                "q2_unblock_role",
                "annotation_priority",
                "q2_annotation_score",
                "required_q2_metadata_fields",
                "annotation_reason",
            ],
        ),
        "",
        "## External Validation Note",
        "",
        "For the current 73 videos, `external_test_split` must come from the acquisition design. If all current videos were used during method development, mark them as `internal` and collect a genuinely independent external/holdout set before making Q2+ generalization claims.",
        "",
        "Recommended next proof commands after filling metadata:",
        "",
        "```powershell",
        "& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\import_rr_metadata_annotations.py --annotation-csv Dataset_new\\72video\\al_images\\paper_repro_quality_residual_paper_assets\\paper_metadata_annotation_fill_template.csv --write-template",
        "& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\audit_rr_metadata_quality.py",
        "& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_quality_residual_metadata_group_validation.py --group-columns cow_id",
        "& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_external_split_validation.py",
        "& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\audit_rr_submission_readiness.py",
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_minimum_q2_checklist(
    output_path: Path,
    progress: pd.DataFrame,
    priority: pd.DataFrame,
) -> Path:
    progress_by_field = {
        str(row["field"]): row for _, row in progress.iterrows()
    }

    def coverage(field: str) -> str:
        row = progress_by_field.get(field)
        if row is None:
            return "not generated"
        return (
            f"{int(row['nonempty'])}/{int(row['total_videos'])} "
            f"({float(row['coverage_percent']):.1f}%)"
        )

    rows = [
        {
            "priority": "P0",
            "field": "external_test_split",
            "current_coverage": coverage("external_test_split"),
            "fill_from": "real acquisition design or genuinely independent holdout set",
            "proof_command": "python scripts/rr_external_split_validation.py",
            "claim_unlocked": "split provenance now; frozen external or holdout validation only after independent external rows exist",
        },
        {
            "priority": "P0",
            "field": "cow_id",
            "current_coverage": coverage("cow_id"),
            "fill_from": "numeric video label per current dataset convention; farm ID/collar/visual identity if later available",
            "proof_command": "python scripts/rr_quality_residual_metadata_group_validation.py --group-columns cow_id",
            "claim_unlocked": "video-derived numeric-label grouped validation only; true animal-level wording requires a separate real-cow identity map",
        },
        {
            "priority": "P0",
            "field": "collection_start_date",
            "current_coverage": coverage("collection_start_date"),
            "fill_from": "known collection window: 2023-08-05",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "collection-window provenance",
        },
        {
            "priority": "P0",
            "field": "collection_end_date",
            "current_coverage": coverage("collection_end_date"),
            "fill_from": "known collection window: 2023-08-10",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "collection-window provenance",
        },
        {
            "priority": "P0",
            "field": "collection_site",
            "current_coverage": coverage("collection_site"),
            "fill_from": "Lindian County ranch, Heilongjiang, China",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "collection-location provenance",
        },
        {
            "priority": "P2_claim_specific",
            "field": "ambient_temperature_c",
            "current_coverage": coverage("ambient_temperature_c"),
            "fill_from": "synchronized barn sensor/logger/acquisition record only; leave blank if unavailable",
            "proof_command": "python scripts/build_rr_heat_stress_context.py",
            "claim_unlocked": "THI and heat-stress context only when real measurements exist",
        },
        {
            "priority": "P2_claim_specific",
            "field": "relative_humidity_percent",
            "current_coverage": coverage("relative_humidity_percent"),
            "fill_from": "synchronized humidity logger or barn record only; leave blank if unavailable",
            "proof_command": "python scripts/build_rr_heat_stress_context.py",
            "claim_unlocked": "THI and heat-stress context only when real measurements exist",
        },
        {
            "priority": "P2_claim_specific",
            "field": "collection_date",
            "current_coverage": coverage("collection_date"),
            "fill_from": "video acquisition log or folder/session record",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "date/session leakage and stratification checks",
        },
        {
            "priority": "P2_claim_specific",
            "field": "camera_id",
            "current_coverage": coverage("camera_id"),
            "fill_from": "camera log, filename convention, or acquisition record",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "camera-domain robustness wording",
        },
        {
            "priority": "P0",
            "field": "scene_id",
            "current_coverage": coverage("scene_id"),
            "fill_from": "barn/pen/session label or visual review",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "scene-level validation and leakage checks",
        },
        {
            "priority": "P2_claim_specific",
            "field": "head_motion_score_0_3",
            "current_coverage": coverage("head_motion_score_0_3"),
            "fill_from": "manual visual review only; leave blank if no review scores can be produced",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "motion-robustness stratified analysis",
        },
        {
            "priority": "P2_claim_specific",
            "field": "occlusion_score_0_3",
            "current_coverage": coverage("occlusion_score_0_3"),
            "fill_from": "manual visual review only; leave blank if no review scores can be produced",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "occlusion robustness/error-source analysis",
        },
        {
            "priority": "P2_claim_specific",
            "field": "nostril_visibility_score_0_3",
            "current_coverage": coverage("nostril_visibility_score_0_3"),
            "fill_from": "manual visual review only; leave blank if no review scores can be produced",
            "proof_command": "python scripts/audit_rr_metadata_quality.py",
            "claim_unlocked": "ROI-quality stratified analysis",
        },
    ]
    checklist = pd.DataFrame(rows)
    priority_table = priority.fillna("").astype(str)
    lines = [
        "# Minimum Q2 Metadata Checklist",
        "",
        "This is the shortest field list that moves the current internal manuscript package toward an algorithmic Q2-or-higher route. Fill real values only; do not infer cow identity, external split, environmental data, or visual-quality scores from model errors.",
        "",
        "## Blocking And Claim-Specific Fields",
        "",
        to_markdown(checklist, ["priority", "field", "current_coverage", "fill_from", "proof_command", "claim_unlocked"]),
        "",
        "## Annotation Batch Order",
        "",
        to_markdown(
            priority_table,
            [
                "q2_annotation_batch",
                "videos",
                "critical",
                "high",
                "medium",
                "normal",
                "max_q2_annotation_score",
            ],
        ),
        "",
        "## Safe Import Workflow",
        "",
        "1. Fill `paper_metadata_annotation_fill_template.csv` with real metadata values.",
        "2. Validate without writing:",
        "",
        "```powershell",
        "python scripts\\import_rr_metadata_annotations.py --annotation-csv Dataset_new\\72video\\al_images\\paper_repro_quality_residual_paper_assets\\paper_metadata_annotation_fill_template.csv",
        "```",
        "",
        "3. Only after validation, write the template:",
        "",
        "```powershell",
        "python scripts\\import_rr_metadata_annotations.py --annotation-csv Dataset_new\\72video\\al_images\\paper_repro_quality_residual_paper_assets\\paper_metadata_annotation_fill_template.csv --write-template",
        "```",
        "",
        "4. Rerun metadata quality, external split validation, readiness audit, and paper asset build.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("\n", " ") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_codebook(output_dir: Path, corrected_prefix: str) -> Path:
    codebook = output_dir / "paper_metadata_annotation_codebook.md"
    text = f"""# Thermal RR Metadata Annotation Codebook

Use `paper_metadata_annotation_sheet.csv` to fill the same metadata fields used by `paper_metadata_template.csv`. After annotation, copy the filled metadata columns back to `paper_metadata_template.csv` or replace that template with the filled sheet after removing context-only columns.

## Required Fields For Algorithmic Q2+ Readiness

- `cow_id`: current dataset convention uses the numeric part of `video_id`, per user clarification. This supports numeric video-label grouping only; it is not evidence of real animal identity. True animal-level wording requires a separate real-cow identity map.
- `collection_start_date`, `collection_end_date`, and collection-location fields: known provenance context for the current dataset. These fields document the Lindian County, Heilongjiang, China collection window.
- `collection_date`: exact per-video acquisition date, preferably `YYYY-MM-DD`, only if an acquisition log exists. The known date window is not a substitute for per-video date-level validation.
- `scene_id`: current shared barn/farm scene identifier. `camera_id` is optional unless camera-domain robustness is claimed.
- `ambient_temperature_c` and `relative_humidity_percent`: claim-specific fields required for THI only when synchronized barn sensor/logger values are available. Regional NASA POWER values are stored as context-only proxy tables and should not be copied into these fields for heat-stress claims.
- `head_motion_score_0_3`, `occlusion_score_0_3`, `nostril_visibility_score_0_3`: claim-specific manual video-quality context for robustness analysis. Leave blank if these scores cannot be produced; do not replace them with algorithmic risk scores.
- `external_test_split`: use values such as `internal`, `external`, or `holdout`. The current 73 videos should remain `internal` if they were used during development; only method-frozen, independently collected videos should be marked `external` or `holdout`.

## Score Definitions

`head_motion_score_0_3`

- 0: head and nostril region are stable for the analysis window.
- 1: mild motion; nostril region remains mostly visible.
- 2: frequent motion or short tracking interruptions.
- 3: severe motion, large displacement, or repeated loss of the nostril region.

`occlusion_score_0_3`

- 0: no occlusion.
- 1: minor partial occlusion that does not affect most breaths.
- 2: repeated or moderate occlusion.
- 3: severe occlusion or nostril region often blocked.

`nostril_visibility_score_0_3`

- 0: nostrils not reliably visible.
- 1: visible only intermittently or weakly.
- 2: mostly visible with some uncertainty.
- 3: clearly visible throughout most of the analysis window.

## Recommended Workflow

1. Open `paper_metadata_annotation_dashboard.html` to inspect frames and curves.
2. Start from `paper_metadata_annotation_q2_unblock_queue.csv`; sort by `q2_unblock_rank`.
3. Fill `paper_metadata_annotation_fill_template.csv`; it contains only import-safe metadata columns.
4. Use `paper_metadata_annotation_sheet.csv` only as context; it includes many non-metadata columns.
5. Sort by `q2_annotation_score` descending, then `q2_annotation_batch` when you need a secondary review order.
6. Start with `batch_1_residual_errors_and_representative_cases` and `batch_2_corrected_or_selective_error_cases`.
7. Fill all metadata fields for every video, not only high-priority videos.
8. Validate with `scripts/import_rr_metadata_annotations.py` before writing to `paper_metadata_template.csv`.
9. Use `strict_auto_accept`, `selective_action`, and signal-aware correction fields only as review-priority context, not as substitutes for metadata.
10. Rerun:

```powershell
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_quality_residual_metadata_group_validation.py --group-columns cow_id
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_heat_stress_context.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_paper_assets.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_metadata_annotation_dashboard.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_external_split_validation.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\audit_rr_submission_readiness.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_paper_assets.py
```

Current readiness table is expected at `{corrected_prefix}_submission_readiness.csv`.
"""
    codebook.write_text(text, encoding="utf-8")
    return codebook


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sheet = build_annotation_sheet(
        input_root,
        args.output_prefix,
        args.corrected_prefix,
        output_dir,
    )
    progress = progress_table(sheet)
    priority = priority_summary(sheet)
    q2_unblock_queue = build_q2_unblock_queue(sheet)
    fill_template = build_fill_template(sheet)

    sheet_root = input_root / f"{args.output_prefix}_metadata_annotation_sheet.csv"
    fill_template_root = input_root / f"{args.output_prefix}_metadata_annotation_fill_template.csv"
    checklist_root = input_root / f"{args.output_prefix}_metadata_annotation_minimum_q2_checklist.md"
    progress_root = input_root / f"{args.output_prefix}_metadata_annotation_progress.csv"
    priority_root = input_root / f"{args.output_prefix}_metadata_annotation_priority_summary.csv"
    q2_unblock_queue_root = input_root / f"{args.output_prefix}_metadata_annotation_q2_unblock_queue.csv"
    q2_unblock_report_root = input_root / f"{args.output_prefix}_metadata_annotation_q2_unblock_queue.md"
    sheet_assets = output_dir / "paper_metadata_annotation_sheet.csv"
    fill_template_assets = output_dir / "paper_metadata_annotation_fill_template.csv"
    checklist_assets = output_dir / "paper_metadata_annotation_minimum_q2_checklist.md"
    progress_assets = output_dir / "paper_metadata_annotation_progress.csv"
    priority_assets = output_dir / "paper_metadata_annotation_priority_summary.csv"
    q2_unblock_queue_assets = output_dir / "paper_metadata_annotation_q2_unblock_queue.csv"
    q2_unblock_report_assets = output_dir / "paper_metadata_annotation_q2_unblock_queue.md"
    sheet.to_csv(sheet_root, index=False)
    sheet.to_csv(sheet_assets, index=False)
    fill_template.to_csv(fill_template_root, index=False)
    fill_template.to_csv(fill_template_assets, index=False)
    progress.to_csv(progress_root, index=False)
    progress.to_csv(progress_assets, index=False)
    priority.to_csv(priority_root, index=False)
    priority.to_csv(priority_assets, index=False)
    q2_unblock_queue.to_csv(q2_unblock_queue_root, index=False)
    q2_unblock_queue.to_csv(q2_unblock_queue_assets, index=False)
    write_minimum_q2_checklist(checklist_root, progress, priority)
    write_minimum_q2_checklist(checklist_assets, progress, priority)
    write_q2_unblock_report(q2_unblock_report_root, q2_unblock_queue, progress, priority)
    write_q2_unblock_report(q2_unblock_report_assets, q2_unblock_queue, progress, priority)
    codebook = write_codebook(output_dir, args.corrected_prefix)

    print(f"Saved annotation sheet: {sheet_root}")
    print(f"Saved paper-assets annotation sheet: {sheet_assets}")
    print(f"Saved annotation fill template: {fill_template_root}")
    print(f"Saved paper-assets fill template: {fill_template_assets}")
    print(f"Saved minimum Q2 checklist: {checklist_assets}")
    print(f"Saved Q2 unblock queue: {q2_unblock_queue_assets}")
    print(f"Saved annotation progress: {progress_root}")
    print(f"Saved annotation priority summary: {priority_root}")
    print(f"Saved codebook: {codebook}")
    print(progress.to_string(index=False))
    print("\nPriority summary:")
    print(priority.to_string(index=False))


if __name__ == "__main__":
    main()
