from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AGREEMENT_COLUMNS = [
    "external_video_id",
    "cow_id",
    "source_session_id",
    "manual_duration_seconds",
    "count_a",
    "annotator_a",
    "visibility_a",
    "count_uncertainty_a_breaths",
    "count_b",
    "annotator_b",
    "visibility_b",
    "count_uncertainty_b_breaths",
    "abs_count_diff",
    "agreement_status",
    "consensus_breath_count",
    "consensus_rr_bpm",
    "consensus_annotator",
    "reference_quality_status",
    "reference_count_uncertainty_breaths",
    "camera_id",
    "needs_adjudication",
    "notes",
]

CONSENSUS_COLUMNS = [
    "external_video_id",
    "manual_breath_count",
    "manual_rr_bpm",
    "reference_rr_annotator",
    "camera_id",
    "reference_quality_status",
    "reference_count_uncertainty_breaths",
    "annotation_notes",
]

ADJUDICATION_COLUMNS = [
    "external_video_id",
    "cow_id",
    "source_session_id",
    "manual_duration_seconds",
    "count_a",
    "annotator_a",
    "visibility_a",
    "count_uncertainty_a_breaths",
    "count_b",
    "annotator_b",
    "visibility_b",
    "count_uncertainty_b_breaths",
    "abs_count_diff",
    "adjudicated_breath_count",
    "adjudicator",
    "camera_id",
    "adjudication_notes",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Audit dual manual breath-count annotation agreement for external "
            "RR validation and generate consensus/adjudication files."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--worklist-csv", type=Path, default=None)
    parser.add_argument("--annotation-a-csv", type=Path, required=True)
    parser.add_argument("--annotation-b-csv", type=Path, required=True)
    parser.add_argument("--adjudication-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--auto-consensus-max-diff",
        type=float,
        default=0.0,
        help=(
            "Maximum absolute breath-count difference allowed for automatic "
            "consensus. Default 0 requires exact agreement."
        ),
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_fieldwork_csv(output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_split_all_use_fieldwork_template.csv"


def default_worklist_csv(output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_breath_annotation_worklist.csv"


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1", "include", "included"]
    )


def numeric(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def count_text(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:.6f}".rstrip("0").rstrip(".")


def rr_text(count: float | None, duration: float | None) -> str:
    if count is None or duration is None or duration <= 0:
        return ""
    rr = float(count) * 60.0 / float(duration)
    return f"{rr:.6f}".rstrip("0").rstrip(".")


def visibility_text(value: object) -> str:
    text = normalize_text(value).lower()
    return text if text in {"clear", "uncertain", "unreadable"} else "not_recorded"


def uncertainty_value(value: object) -> float | None:
    result = numeric(value)
    return result if result is not None and result >= 0 else None


def read_annotations(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "external_video_id" not in data.columns:
        raise ValueError(f"{label} annotation CSV must include external_video_id")
    rename = {
        "manual_breath_count": f"count_{label}",
        "reference_rr_annotator": f"annotator_{label}",
        "camera_id": f"camera_id_{label}",
        "annotation_visibility": f"visibility_{label}",
        "count_uncertainty_breaths": f"count_uncertainty_{label}_breaths",
        "annotation_notes": f"notes_{label}",
    }
    keep = ["external_video_id"] + [column for column in rename if column in data.columns]
    data = data[keep].copy().rename(columns=rename)
    for column in rename.values():
        if column not in data.columns:
            data[column] = ""
    return data


def base_rows(fieldwork: pd.DataFrame, worklist: pd.DataFrame | None) -> pd.DataFrame:
    included = fieldwork[
        yes_mask(fieldwork.get("include_in_external_validation", pd.Series("", index=fieldwork.index)))
    ].copy()
    columns = [
        "external_video_id",
        "cow_id",
        "manual_duration_seconds",
        "raw_video_path",
        "camera_id",
    ]
    base = included[[column for column in columns if column in included.columns]].copy()
    if worklist is not None and not worklist.empty:
        keep = [
            column
            for column in [
                "external_video_id",
                "source_session_id",
                "clip_index",
                "collection_date",
                "collection_time",
            ]
            if column in worklist.columns
        ]
        base = base.merge(worklist[keep].drop_duplicates("external_video_id"), on="external_video_id", how="left")
    for column in ["source_session_id", "clip_index", "collection_date", "collection_time"]:
        if column not in base.columns:
            base[column] = ""
    return base


def read_adjudication(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["external_video_id", "adjudicated_breath_count", "adjudicator", "camera_id", "adjudication_notes"])
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "external_video_id" not in data.columns:
        raise ValueError("Adjudication CSV must include external_video_id")
    for column in ["adjudicated_breath_count", "adjudicator", "camera_id", "adjudication_notes"]:
        if column not in data.columns:
            data[column] = ""
    return data[["external_video_id", "adjudicated_breath_count", "adjudicator", "camera_id", "adjudication_notes"]].copy()


def choose_camera(row: pd.Series) -> str:
    for column in ["camera_id", "camera_id_a", "camera_id_b"]:
        value = normalize_text(row.get(column, ""))
        if value:
            return value
    return ""


def build_agreement(
    fieldwork: pd.DataFrame,
    worklist: pd.DataFrame | None,
    annotation_a: pd.DataFrame,
    annotation_b: pd.DataFrame,
    adjudication: pd.DataFrame,
    max_diff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = base_rows(fieldwork, worklist)
    data = data.merge(annotation_a, on="external_video_id", how="left")
    data = data.merge(annotation_b, on="external_video_id", how="left")
    if not adjudication.empty:
        data = data.merge(adjudication, on="external_video_id", how="left")
    else:
        for column in ["adjudicated_breath_count", "adjudicator", "adjudication_notes"]:
            data[column] = ""

    rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    adjudication_rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        video_id = normalize_text(row.get("external_video_id", ""))
        duration = numeric(row.get("manual_duration_seconds", ""))
        count_a = numeric(row.get("count_a", ""))
        count_b = numeric(row.get("count_b", ""))
        visibility_a = visibility_text(row.get("visibility_a", ""))
        visibility_b = visibility_text(row.get("visibility_b", ""))
        uncertainty_a = uncertainty_value(row.get("count_uncertainty_a_breaths", ""))
        uncertainty_b = uncertainty_value(row.get("count_uncertainty_b_breaths", ""))
        adjudicated = numeric(row.get("adjudicated_breath_count", ""))
        diff = None
        if count_a is not None and count_b is not None:
            diff = abs(count_a - count_b)
        consensus = None
        status = "missing_both"
        needs_adjudication = True
        unreadable = "unreadable" in {visibility_a, visibility_b}
        if adjudicated is not None:
            consensus = adjudicated
            status = "adjudicated"
            needs_adjudication = False
        elif unreadable:
            status = "unreadable_by_annotator"
            needs_adjudication = False
        elif count_a is None and count_b is None:
            status = "missing_both"
        elif count_a is None:
            status = "missing_a"
        elif count_b is None:
            status = "missing_b"
        elif diff is not None and diff <= max_diff:
            consensus = round((count_a + count_b) / 2.0, 6)
            status = "exact_agreement" if diff == 0 else "auto_consensus_within_tolerance"
            needs_adjudication = False
        else:
            status = "disagreement"

        if status == "adjudicated":
            quality_status = "adjudicated_after_unreadable_mark" if unreadable else "adjudicated"
        elif unreadable:
            quality_status = "unreadable"
        elif consensus is not None and visibility_a == visibility_b == "clear" and (uncertainty_a or 0.0) == 0.0 and (uncertainty_b or 0.0) == 0.0:
            quality_status = "clear_dual_annotation"
        elif consensus is not None and "not_recorded" not in {visibility_a, visibility_b}:
            quality_status = "uncertain_dual_annotation"
        else:
            quality_status = "visibility_not_recorded"
        consensus_uncertainty = max(
            value for value in [uncertainty_a, uncertainty_b] if value is not None
        ) if any(value is not None for value in [uncertainty_a, uncertainty_b]) else None

        camera_id = normalize_text(row.get("camera_id", "")) or choose_camera(row)
        consensus_annotator = ""
        if consensus is not None:
            if status == "adjudicated":
                consensus_annotator = normalize_text(row.get("adjudicator", "")) or "adjudicated"
            else:
                names = [
                    normalize_text(row.get("annotator_a", "")),
                    normalize_text(row.get("annotator_b", "")),
                ]
                consensus_annotator = "+".join(name for name in names if name) or "dual_annotator_agreement"
        notes = "; ".join(
            note
            for note in [
                normalize_text(row.get("notes_a", "")),
                normalize_text(row.get("notes_b", "")),
                normalize_text(row.get("adjudication_notes", "")),
            ]
            if note
        )
        agreement_row = {
            "external_video_id": video_id,
            "cow_id": normalize_text(row.get("cow_id", "")),
            "source_session_id": normalize_text(row.get("source_session_id", "")),
            "manual_duration_seconds": normalize_text(row.get("manual_duration_seconds", "")),
            "count_a": count_text(count_a),
            "annotator_a": normalize_text(row.get("annotator_a", "")),
            "visibility_a": visibility_a,
            "count_uncertainty_a_breaths": count_text(uncertainty_a),
            "count_b": count_text(count_b),
            "annotator_b": normalize_text(row.get("annotator_b", "")),
            "visibility_b": visibility_b,
            "count_uncertainty_b_breaths": count_text(uncertainty_b),
            "abs_count_diff": "" if diff is None else f"{diff:.6f}".rstrip("0").rstrip("."),
            "agreement_status": status,
            "consensus_breath_count": count_text(consensus),
            "consensus_rr_bpm": rr_text(consensus, duration),
            "consensus_annotator": consensus_annotator,
            "reference_quality_status": quality_status,
            "reference_count_uncertainty_breaths": count_text(consensus_uncertainty),
            "camera_id": camera_id,
            "needs_adjudication": bool(needs_adjudication),
            "notes": notes,
        }
        rows.append(agreement_row)
        if consensus is not None:
            consensus_rows.append(
                {
                    "external_video_id": video_id,
                    "manual_breath_count": count_text(consensus),
                    "manual_rr_bpm": rr_text(consensus, duration),
                    "reference_rr_annotator": consensus_annotator,
                    "camera_id": camera_id,
                    "reference_quality_status": quality_status,
                    "reference_count_uncertainty_breaths": count_text(consensus_uncertainty),
                    "annotation_notes": f"Consensus source: {status}; quality: {quality_status}. {notes}".strip(),
                }
            )
        if needs_adjudication:
            adjudication_rows.append(
                {
                    "external_video_id": video_id,
                    "cow_id": normalize_text(row.get("cow_id", "")),
                    "source_session_id": normalize_text(row.get("source_session_id", "")),
                    "manual_duration_seconds": normalize_text(row.get("manual_duration_seconds", "")),
                    "count_a": count_text(count_a),
                    "annotator_a": normalize_text(row.get("annotator_a", "")),
                    "visibility_a": visibility_a,
                    "count_uncertainty_a_breaths": count_text(uncertainty_a),
                    "count_b": count_text(count_b),
                    "annotator_b": normalize_text(row.get("annotator_b", "")),
                    "visibility_b": visibility_b,
                    "count_uncertainty_b_breaths": count_text(uncertainty_b),
                    "abs_count_diff": "" if diff is None else f"{diff:.6f}".rstrip("0").rstrip("."),
                    "adjudicated_breath_count": "",
                    "adjudicator": "",
                    "camera_id": camera_id,
                    "adjudication_notes": "",
                }
            )
    agreement = pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
    consensus_df = pd.DataFrame(consensus_rows, columns=CONSENSUS_COLUMNS)
    adjudication_df = pd.DataFrame(adjudication_rows, columns=ADJUDICATION_COLUMNS)
    summary = summarize(agreement)
    return agreement, consensus_df, adjudication_df, summary


def summarize(agreement: pd.DataFrame) -> pd.DataFrame:
    total = len(agreement)
    complete_pair = agreement["count_a"].astype(str).ne("") & agreement["count_b"].astype(str).ne("")
    exact = agreement["agreement_status"].astype(str).eq("exact_agreement")
    auto = agreement["agreement_status"].astype(str).eq("auto_consensus_within_tolerance")
    adjudicated = agreement["agreement_status"].astype(str).eq("adjudicated")
    needs = agreement["needs_adjudication"].astype(bool) if total else pd.Series(dtype=bool)
    quality = agreement.get("reference_quality_status", pd.Series("", index=agreement.index)).astype(str)
    consensus_ready = agreement.get("consensus_breath_count", pd.Series("", index=agreement.index)).astype(str).ne("")
    diff = pd.to_numeric(agreement.get("abs_count_diff", pd.Series(dtype=str)), errors="coerce")
    rows = [
        {
            "metric": "included_external_rows",
            "value": total,
            "interpretation": "Rows selected for external RR reference annotation.",
        },
        {
            "metric": "complete_dual_annotation_rows",
            "value": int(complete_pair.sum()) if total else 0,
            "interpretation": "Rows with non-missing counts from both annotators.",
        },
        {
            "metric": "exact_agreement_rows",
            "value": int(exact.sum()) if total else 0,
            "interpretation": "Rows accepted automatically under exact agreement.",
        },
        {
            "metric": "auto_consensus_within_tolerance_rows",
            "value": int(auto.sum()) if total else 0,
            "interpretation": "Rows accepted automatically by the configured tolerance.",
        },
        {
            "metric": "adjudicated_rows",
            "value": int(adjudicated.sum()) if total else 0,
            "interpretation": "Rows resolved by adjudication CSV.",
        },
        {
            "metric": "needs_adjudication_rows",
            "value": int(needs.sum()) if total else 0,
            "interpretation": "Rows that should not enter external scoring yet.",
        },
        {
            "metric": "consensus_ready_rows",
            "value": int(consensus_ready.sum()) if total else 0,
            "interpretation": "Rows with a consensus breath count and RR.",
        },
        {
            "metric": "clear_dual_annotation_rows",
            "value": int(quality.eq("clear_dual_annotation").sum()) if total else 0,
            "interpretation": "Consensus rows marked clear by both annotators with zero stated count uncertainty.",
        },
        {
            "metric": "uncertain_dual_annotation_rows",
            "value": int(quality.eq("uncertain_dual_annotation").sum()) if total else 0,
            "interpretation": "Consensus rows retained with an explicit uncertain visibility or count-uncertainty mark.",
        },
        {
            "metric": "unreadable_rows",
            "value": int(quality.eq("unreadable").sum()) if total else 0,
            "interpretation": "Rows excluded from scoring because at least one annotator marked the clip unreadable and no adjudicated count exists.",
        },
        {
            "metric": "mean_abs_count_difference_complete_pairs",
            "value": "" if diff.dropna().empty else round(float(diff.dropna().mean()), 6),
            "interpretation": "Mean absolute breath-count difference across complete dual annotations.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(data: pd.DataFrame, max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows._"
    view = data.head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    adjudication: pd.DataFrame,
    consensus: pd.DataFrame,
    max_diff: float,
) -> None:
    status = "ready_for_consensus_merge" if not consensus.empty and adjudication.empty else "not_ready"
    if not consensus.empty and not adjudication.empty:
        status = "partial_consensus_needs_adjudication"
    text = f"""# External Breath Annotation Agreement Audit

Status: `{status}`

Automatic consensus tolerance: `{max_diff}` breaths.

Use this report to keep the external RR reference defensible for a Q2+ paper.
Rows requiring adjudication should not enter external RR scoring until the
adjudication template is filled and this audit is rerun.

## Summary

{markdown_table(summary)}

## Adjudication Preview

{markdown_table(adjudication)}

## Consensus Preview

{markdown_table(consensus)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldwork_csv = (
        args.fieldwork_csv.resolve()
        if args.fieldwork_csv is not None
        else default_fieldwork_csv(output_dir).resolve()
    )
    worklist_csv = (
        args.worklist_csv.resolve()
        if args.worklist_csv is not None
        else default_worklist_csv(output_dir).resolve()
    )
    if not fieldwork_csv.exists():
        raise FileNotFoundError(f"Missing fieldwork CSV: {fieldwork_csv}")
    fieldwork = pd.read_csv(fieldwork_csv, dtype=str, keep_default_na=False)
    worklist = (
        pd.read_csv(worklist_csv, dtype=str, keep_default_na=False)
        if worklist_csv.exists()
        else None
    )
    annotation_a = read_annotations(args.annotation_a_csv.resolve(), "a")
    annotation_b = read_annotations(args.annotation_b_csv.resolve(), "b")
    adjudication = read_adjudication(args.adjudication_csv.resolve() if args.adjudication_csv else None)

    agreement, consensus, adjudication_template, summary = build_agreement(
        fieldwork,
        worklist,
        annotation_a,
        annotation_b,
        adjudication,
        args.auto_consensus_max_diff,
    )

    agreement_path = output_dir / "paper_external_validation_breath_annotation_agreement.csv"
    consensus_path = output_dir / "paper_external_validation_breath_annotation_consensus.csv"
    adjudication_path = output_dir / "paper_external_validation_breath_annotation_adjudication_template.csv"
    summary_path = output_dir / "paper_external_validation_breath_annotation_agreement_summary.csv"
    report_path = output_dir / "paper_external_validation_breath_annotation_agreement.md"

    agreement.to_csv(agreement_path, index=False)
    consensus.to_csv(consensus_path, index=False)
    adjudication_template.to_csv(adjudication_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, summary, adjudication_template, consensus, args.auto_consensus_max_diff)

    print(f"Saved agreement audit: {agreement_path}")
    print(f"Saved consensus annotation CSV: {consensus_path}")
    print(f"Saved adjudication template: {adjudication_path}")
    print(f"Saved agreement summary: {summary_path}")
    print(f"Saved agreement report: {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
