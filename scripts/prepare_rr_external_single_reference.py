from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


YES_VALUES = {"yes", "y", "true", "1", "include", "included"}
TRUTH_COLUMNS = ["video_id", "cow_id", "breath_count", "duration_seconds", "rr"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a provisional single-annotator external RR reference set "
            "without representing it as dual-annotator consensus."
        )
    )
    parser.add_argument(
        "--fieldwork-csv",
        type=Path,
        default=assets / "paper_external_validation_split_all_use_fieldwork_template.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--annotator-id", default="single_manual_annotator")
    parser.add_argument("--count-uncertainty", type=int, default=1)
    parser.add_argument("--primary-min-duration-seconds", type=float, default=20.0)
    parser.add_argument("--strict-min-duration-seconds", type=float, default=29.0)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def yes(value: object) -> bool:
    return normalize_text(value).lower() in YES_VALUES


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def numeric_text(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def append_note(existing: object, note: str) -> str:
    current = normalize_text(existing)
    if note in current:
        return current
    return f"{current} {note}".strip()


def source_session_id(row: pd.Series) -> str:
    scene = normalize_text(row.get("scene_id", ""))
    if scene:
        return scene
    video_id = normalize_text(row.get("external_video_id", ""))
    marker = video_id.rsplit("_clip", 1)
    return marker[0] if len(marker) == 2 else video_id


def build_row_audit(
    fieldwork: pd.DataFrame,
    count_uncertainty: int,
    primary_min_duration: float,
    strict_min_duration: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in fieldwork.iterrows():
        count = pd.to_numeric(
            pd.Series([row.get("manual_breath_count", "")]), errors="coerce"
        ).iloc[0]
        duration = pd.to_numeric(
            pd.Series([row.get("manual_duration_seconds", "")]), errors="coerce"
        ).iloc[0]
        original_include = yes(row.get("include_in_external_validation", ""))
        valid_count = bool(pd.notna(count) and float(count) >= 0)
        valid_duration = bool(pd.notna(duration) and float(duration) > 0)
        valid_reference = valid_count and valid_duration
        point_rr = float(count) * 60.0 / float(duration) if valid_reference else np.nan
        count_low = max(0.0, float(count) - count_uncertainty) if valid_reference else np.nan
        count_high = float(count) + count_uncertainty if valid_reference else np.nan
        rr_low = count_low * 60.0 / float(duration) if valid_reference else np.nan
        rr_high = count_high * 60.0 / float(duration) if valid_reference else np.nan
        quantization = count_uncertainty * 60.0 / float(duration) if valid_reference else np.nan

        if not original_include:
            status = "excluded_by_fieldwork"
            reason = "fieldwork include flag is not yes"
        elif not valid_count:
            status = "excluded_unreadable_or_missing_count"
            reason = "manual breath count is missing or invalid"
        elif not valid_duration:
            status = "excluded_invalid_duration"
            reason = "manual duration is missing or non-positive"
        elif float(duration) < primary_min_duration:
            status = "short_visible_sensitivity_only"
            reason = (
                f"duration below primary threshold ({primary_min_duration:g} s); "
                "one-count error causes excessive RR uncertainty"
            )
        elif float(duration) >= strict_min_duration:
            status = "primary_strict_duration"
            reason = "eligible for primary and strict-duration analyses"
        else:
            status = "primary_eligible"
            reason = "eligible for primary analysis"

        rows.append(
            {
                "external_video_id": normalize_text(row.get("external_video_id", "")),
                "cow_id": normalize_text(row.get("cow_id", "")),
                "source_session_id": source_session_id(row),
                "collection_date": normalize_text(row.get("collection_date", "")),
                "original_include": original_include,
                "manual_breath_count": numeric_text(float(count)) if valid_count else "",
                "manual_duration_seconds": numeric_text(float(duration)) if valid_duration else "",
                "manual_rr_bpm_point": numeric_text(point_rr),
                "count_uncertainty_breaths": count_uncertainty if valid_reference else "",
                "breath_count_lower": numeric_text(count_low),
                "breath_count_upper": numeric_text(count_high),
                "manual_rr_lower_bpm": numeric_text(rr_low),
                "manual_rr_upper_bpm": numeric_text(rr_high),
                "rr_uncertainty_plus_minus_bpm": numeric_text(quantization),
                "reference_status": status,
                "exclusion_or_tier_reason": reason,
                "primary_analysis_include": status.startswith("primary_"),
                "all_visible_sensitivity_include": bool(original_include and valid_reference),
            }
        )
    return pd.DataFrame(rows)


def prepare_fieldwork(
    fieldwork: pd.DataFrame,
    audit: pd.DataFrame,
    annotator_id: str,
    include_column: str,
) -> pd.DataFrame:
    output = fieldwork.copy()
    audit_by_id = audit.set_index("external_video_id", drop=False)
    for column in [
        "manual_rr_bpm",
        "reference_rr_annotator",
        "reference_protocol_notes",
        "reference_quality_status",
        "reference_count_uncertainty_breaths",
        "reference_rr_lower_bpm",
        "reference_rr_upper_bpm",
    ]:
        if column not in output.columns:
            output[column] = ""

    for idx, row in output.iterrows():
        video_id = normalize_text(row.get("external_video_id", ""))
        if video_id not in audit_by_id.index:
            continue
        info = audit_by_id.loc[video_id]
        include = bool(info[include_column])
        output.at[idx, "include_in_external_validation"] = "yes" if include else "no"
        output.at[idx, "reference_quality_status"] = info["reference_status"]
        output.at[idx, "reference_count_uncertainty_breaths"] = info[
            "count_uncertainty_breaths"
        ]
        output.at[idx, "reference_rr_lower_bpm"] = info["manual_rr_lower_bpm"]
        output.at[idx, "reference_rr_upper_bpm"] = info["manual_rr_upper_bpm"]
        if info["manual_rr_bpm_point"]:
            output.at[idx, "manual_rr_bpm"] = info["manual_rr_bpm_point"]
            if not normalize_text(output.at[idx, "reference_rr_annotator"]):
                output.at[idx, "reference_rr_annotator"] = annotator_id
        note = (
            "Single-annotator provisional reference; not dual-annotator consensus. "
            f"Count uncertainty is +/-{info['count_uncertainty_breaths']} breath(s)."
            if info["manual_rr_bpm_point"]
            else "Manual count unavailable or unreadable; excluded from provisional scoring."
        )
        output.at[idx, "reference_protocol_notes"] = append_note(
            output.at[idx, "reference_protocol_notes"], note
        )
    return output


def build_truth(fieldwork: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in fieldwork.iterrows():
        if not yes(row.get("include_in_external_validation", "")):
            continue
        rows.append(
            {
                "video_id": normalize_text(row.get("external_video_id", "")),
                "cow_id": normalize_text(row.get("cow_id", "")),
                "breath_count": normalize_text(row.get("manual_breath_count", "")),
                "duration_seconds": normalize_text(row.get("manual_duration_seconds", "")),
                "rr": normalize_text(row.get("manual_rr_bpm", "")),
            }
        )
    return pd.DataFrame(rows, columns=TRUTH_COLUMNS)


def summary_rows(fieldwork: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    point_rr = numeric(audit["manual_rr_bpm_point"])
    duration = numeric(audit["manual_duration_seconds"])
    valid = audit["all_visible_sensitivity_include"].astype(bool)
    primary = audit["primary_analysis_include"].astype(bool)
    strict = audit["reference_status"].eq("primary_strict_duration")
    missing_count = numeric(fieldwork["manual_breath_count"]).isna()
    rows = [
        ("source_rows", len(fieldwork), "All rows in the user-edited fieldwork CSV."),
        (
            "original_include_yes_rows",
            int(fieldwork["include_in_external_validation"].map(yes).sum()),
            "Rows still marked yes before reference normalization.",
        ),
        ("manual_count_available_rows", int(numeric(fieldwork["manual_breath_count"]).notna().sum()), "Rows with a numeric manual count."),
        ("manual_count_missing_rows", int(missing_count.sum()), "Missing counts are not imputed."),
        ("all_visible_sensitivity_rows", int(valid.sum()), "Original include=yes with a valid count and duration."),
        ("primary_rows", int(primary.sum()), "Valid references meeting the pre-prediction minimum duration."),
        ("strict_approximately_30s_rows", int(strict.sum()), "Primary rows with duration at least the strict threshold."),
        ("short_visible_sensitivity_only_rows", int(audit["reference_status"].eq("short_visible_sensitivity_only").sum()), "Valid but short clips excluded from the primary result."),
        ("primary_source_sessions", int(audit.loc[primary, "source_session_id"].nunique()), "Use source-session grouped intervals; clips are not independent sessions."),
        ("primary_cows", int(audit.loc[primary, "cow_id"].nunique()), "Unique cow labels in the primary provisional set."),
        ("primary_collection_dates", int(audit.loc[primary, "collection_date"].nunique()), "Unique collection dates in the primary provisional set."),
        ("visible_rr_min_bpm", float(point_rr[valid].min()), "Point reference only; annotation uncertainty remains."),
        ("visible_rr_mean_bpm", float(point_rr[valid].mean()), "Point reference only; annotation uncertainty remains."),
        ("visible_rr_max_bpm", float(point_rr[valid].max()), "Point reference only; annotation uncertainty remains."),
        ("visible_duration_min_seconds", float(duration[valid].min()), "Short duration increases RR quantization error."),
        ("visible_duration_median_seconds", float(duration[valid].median()), "Median counted duration."),
        ("visible_duration_max_seconds", float(duration[valid].max()), "Maximum counted duration."),
        ("duplicate_external_video_ids", int(fieldwork["external_video_id"].duplicated().sum()), "Must remain zero."),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    table = data.loc[:, columns].head(max_rows).fillna("").astype(str)
    if table.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_eda_report(
    path: Path,
    source_path: Path,
    fieldwork: pd.DataFrame,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    count_uncertainty: int,
    primary_min_duration: float,
    strict_min_duration: float,
) -> None:
    summary_map = dict(zip(summary["metric"], summary["value"]))
    missing = pd.DataFrame(
        {
            "column": fieldwork.columns,
            "missing_rows": [
                int(fieldwork[column].map(normalize_text).eq("").sum())
                for column in fieldwork.columns
            ],
        }
    ).sort_values(["missing_rows", "column"], ascending=[False, True])
    status_counts = (
        audit["reference_status"]
        .value_counts(dropna=False)
        .rename_axis("reference_status")
        .reset_index(name="rows")
    )
    short = audit[audit["reference_status"].eq("short_visible_sensitivity_only")]
    generated = datetime.now().isoformat(timespec="seconds")
    file_info = source_path.stat()
    lines = [
        "# External Manual Breath-Count EDA and Reference Audit",
        "",
        f"Generated: `{generated}`",
        "",
        "## Basic Information",
        "",
        f"- Source: `{source_path}`",
        f"- File size: `{file_info.st_size}` bytes",
        f"- Last modified: `{datetime.fromtimestamp(file_info.st_mtime).isoformat(timespec='seconds')}`",
        f"- Format: CSV tabular fieldwork data, `{len(fieldwork)}` rows x `{len(fieldwork.columns)}` columns",
        "",
        "## Outcome",
        "",
        f"The file contains `{int(summary_map['manual_count_available_rows'])}` numeric manual counts. "
        f"The provisional primary set contains `{int(summary_map['primary_rows'])}` clips from "
        f"`{int(summary_map['primary_source_sessions'])}` source sessions and "
        f"`{int(summary_map['primary_cows'])}` cow labels. These are single-annotator references, "
        "not dual-annotator consensus truth.",
        "",
        "## Reference Tiers",
        "",
        markdown_table(status_counts, ["reference_status", "rows"]),
        "",
        "The primary duration threshold is defined before algorithm scoring. "
        f"Primary clips require at least `{primary_min_duration:g}` s; the strict sensitivity subset "
        f"requires at least `{strict_min_duration:g}` s. Every counted reference receives a "
        f"`+/-{count_uncertainty}` breath interval. At 30 s this equals approximately "
        f"`+/-{count_uncertainty * 2:g}` bpm, while short clips have much wider RR uncertainty.",
        "",
        "## Missingness",
        "",
        markdown_table(missing, ["column", "missing_rows"], max_rows=len(missing)),
        "",
        "Environment, camera, and visual quality fields remain missing. They must not be inferred "
        "from location/date or used for stratified environmental/quality claims.",
        "",
        "## Short Counted Clips",
        "",
        markdown_table(
            short,
            [
                "external_video_id",
                "manual_breath_count",
                "manual_duration_seconds",
                "manual_rr_bpm_point",
                "rr_uncertainty_plus_minus_bpm",
            ],
        ),
        "",
        "## Statistical Use",
        "",
        "Report the point-reference RR R2, MAE, and RMSE only as provisional single-annotator "
        "external results. Also report source-session cluster bootstrap intervals and repeat the "
        "metrics for the lower/upper one-count reference scenarios. Do not tune thresholds or "
        "select methods using these external labels.",
        "",
        "## Required Upgrade for Confirmatory Claims",
        "",
        "A second independent blinded count is still required for confirmatory external validation. "
        "Resolve disagreements by adjudication, then replace this provisional reference with the "
        "consensus/adjudicated count. Rows that cannot be judged should remain excluded rather than "
        "being imputed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.count_uncertainty < 0:
        raise ValueError("--count-uncertainty must be non-negative")
    if args.primary_min_duration_seconds <= 0:
        raise ValueError("--primary-min-duration-seconds must be positive")
    if args.strict_min_duration_seconds < args.primary_min_duration_seconds:
        raise ValueError("--strict-min-duration-seconds must be >= primary threshold")

    source_path = args.fieldwork_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldwork = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    required = {
        "external_video_id",
        "cow_id",
        "manual_breath_count",
        "manual_duration_seconds",
        "include_in_external_validation",
    }
    missing = required - set(fieldwork.columns)
    if missing:
        raise ValueError(f"Fieldwork CSV missing columns: {sorted(missing)}")

    audit = build_row_audit(
        fieldwork,
        count_uncertainty=args.count_uncertainty,
        primary_min_duration=args.primary_min_duration_seconds,
        strict_min_duration=args.strict_min_duration_seconds,
    )
    primary = prepare_fieldwork(
        fieldwork,
        audit,
        annotator_id=args.annotator_id,
        include_column="primary_analysis_include",
    )
    all_visible = prepare_fieldwork(
        fieldwork,
        audit,
        annotator_id=args.annotator_id,
        include_column="all_visible_sensitivity_include",
    )
    summary = summary_rows(fieldwork, audit)

    outputs = {
        "audit": output_dir / "paper_external_validation_single_reference_rows.csv",
        "summary": output_dir / "paper_external_validation_single_reference_summary.csv",
        "primary_fieldwork": output_dir / "paper_external_validation_split_all_use_fieldwork_single_annotator_primary.csv",
        "all_visible_fieldwork": output_dir / "paper_external_validation_split_all_use_fieldwork_single_annotator_all_visible.csv",
        "primary_truth": output_dir / "paper_external_validation_single_reference_truth_primary.csv",
        "all_visible_truth": output_dir / "paper_external_validation_single_reference_truth_all_visible.csv",
        "report": output_dir / "paper_external_validation_single_reference_eda.md",
    }
    audit.to_csv(outputs["audit"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    primary.to_csv(outputs["primary_fieldwork"], index=False)
    all_visible.to_csv(outputs["all_visible_fieldwork"], index=False)
    build_truth(primary).to_csv(outputs["primary_truth"], index=False)
    build_truth(all_visible).to_csv(outputs["all_visible_truth"], index=False)
    write_eda_report(
        outputs["report"],
        source_path,
        fieldwork,
        audit,
        summary,
        count_uncertainty=args.count_uncertainty,
        primary_min_duration=args.primary_min_duration_seconds,
        strict_min_duration=args.strict_min_duration_seconds,
    )

    print(f"Saved single-reference audit: {outputs['audit']}")
    print(f"Saved primary provisional fieldwork: {outputs['primary_fieldwork']}")
    print(f"Saved EDA report: {outputs['report']}")
    for metric in [
        "source_rows",
        "manual_count_available_rows",
        "primary_rows",
        "strict_approximately_30s_rows",
        "short_visible_sensitivity_only_rows",
    ]:
        value = summary.loc[summary["metric"].eq(metric), "value"].iloc[0]
        print(f"{metric}={value}")


if __name__ == "__main__":
    main()
