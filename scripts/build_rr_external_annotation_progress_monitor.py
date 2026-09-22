from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "annotator",
    "export_csv",
    "export_exists",
    "expected_rows",
    "export_rows",
    "matched_expected_rows",
    "extra_rows",
    "duplicate_video_ids",
    "breath_count_filled",
    "annotator_filled",
    "camera_id_filled",
    "complete_required_rows",
    "missing_required_rows",
    "completion_rate",
    "status",
    "next_action",
]

PER_VIDEO_COLUMNS = [
    "external_video_id",
    "blinded_id_a",
    "display_order_a",
    "blinded_id_b",
    "display_order_b",
    "a_breath_count",
    "a_reference_rr_annotator",
    "a_camera_id",
    "a_complete",
    "b_breath_count",
    "b_reference_rr_annotator",
    "b_camera_id",
    "b_complete",
    "dual_complete",
    "count_disagreement",
    "needs_annotation_or_review",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Monitor blinded A/B external breath-count annotation export progress "
            "before running frozen external RR validation."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--worklist-a-csv",
        type=Path,
        action="append",
        default=None,
        help="Annotator A worklist. Repeat for separate frozen packet scopes.",
    )
    parser.add_argument(
        "--worklist-b-csv",
        type=Path,
        action="append",
        default=None,
        help="Annotator B worklist. Repeat for separate frozen packet scopes.",
    )
    parser.add_argument(
        "--annotation-a-csv",
        type=Path,
        action="append",
        default=None,
        help="Annotator A export. Repeat when multiple packets were exported separately.",
    )
    parser.add_argument(
        "--annotation-b-csv",
        type=Path,
        action="append",
        default=None,
        help="Annotator B export. Repeat when multiple packets were exported separately.",
    )
    parser.add_argument(
        "--report-prefix",
        default="paper_external_validation",
        help="Prefix for progress-monitor output files.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_path(output_dir: Path, name: str) -> Path:
    return output_dir / name


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def complete_mask(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=bool)
    # Camera identity is useful provenance but is not required to adjudicate RR counts.
    required = ["manual_breath_count", "reference_rr_annotator"]
    mask = pd.Series(True, index=data.index)
    for column in required:
        if column not in data.columns:
            mask = mask & False
        else:
            mask = mask & data[column].map(normalize_text).ne("")
    return mask


def expected_from_worklists(worklist_a: pd.DataFrame, worklist_b: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, data in [("a", worklist_a), ("b", worklist_b)]:
        if data.empty or "external_video_id" not in data.columns:
            continue
        keep = ["external_video_id"]
        for column in ["blinded_id", "display_order"]:
            if column in data.columns:
                keep.append(column)
        frame = data[keep].drop_duplicates("external_video_id").copy()
        rename = {
            "blinded_id": f"blinded_id_{label}",
            "display_order": f"display_order_{label}",
        }
        frame = frame.rename(columns=rename)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["external_video_id"])
    expected = frames[0]
    for frame in frames[1:]:
        expected = expected.merge(frame, on="external_video_id", how="outer")
    expected["external_video_id"] = expected["external_video_id"].map(normalize_text)
    expected = expected[expected["external_video_id"].ne("")]
    return expected.sort_values("external_video_id", kind="stable").reset_index(drop=True)


def summarize_annotator(
    label: str,
    paths: list[Path],
    expected_ids: set[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    frames = [read_optional_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    missing_paths = [path for path in paths if not path.exists()]
    export_exists = bool(paths) and not missing_paths
    if data.empty or "external_video_id" not in data.columns:
        export_ids: list[str] = []
        complete = pd.Series(dtype=bool)
    else:
        data = data.copy()
        data["external_video_id"] = data["external_video_id"].map(normalize_text)
        data = data[data["external_video_id"].ne("")]
        export_ids = data["external_video_id"].tolist()
        complete = complete_mask(data)
    export_id_set = set(export_ids)
    duplicate_ids = len(export_ids) - len(export_id_set)
    matched = len(export_id_set & expected_ids)
    extra = len(export_id_set - expected_ids)

    def filled(column: str) -> int:
        if data.empty or column not in data.columns:
            return 0
        return int(data[column].map(normalize_text).ne("").sum())

    complete_required = int(complete.sum()) if len(complete) else 0
    expected_count = len(expected_ids)
    missing_required = max(expected_count - complete_required, 0)
    completion_rate = complete_required / expected_count if expected_count else 0.0
    if not export_exists:
        status = "MISSING_EXPORT"
        missing_names = ", ".join(path.name for path in missing_paths)
        next_action = (
            f"Export annotator {label.upper()} CSV from the blinded HTML packet(s): "
            f"{missing_names}."
        )
    elif duplicate_ids:
        status = "DUPLICATE_VIDEO_IDS"
        next_action = f"Remove duplicate external_video_id rows from annotator {label.upper()} export."
    elif extra:
        status = "EXTRA_ROWS"
        next_action = f"Remove rows not present in the expected worklist from annotator {label.upper()} export."
    elif complete_required < expected_count:
        status = "INCOMPLETE"
        next_action = (
            f"Fill breath_count and reference_rr_annotator for every annotator "
            f"{label.upper()} row."
        )
    else:
        status = "COMPLETE"
        next_action = "Ready for agreement audit."
    summary = {
        "annotator": label,
        "export_csv": "; ".join(str(path) for path in paths),
        "export_exists": bool(export_exists),
        "expected_rows": expected_count,
        "export_rows": len(data),
        "matched_expected_rows": matched,
        "extra_rows": extra,
        "duplicate_video_ids": duplicate_ids,
        "breath_count_filled": filled("manual_breath_count"),
        "annotator_filled": filled("reference_rr_annotator"),
        "camera_id_filled": filled("camera_id"),
        "complete_required_rows": complete_required,
        "missing_required_rows": missing_required,
        "completion_rate": f"{completion_rate:.6f}",
        "status": status,
        "next_action": next_action,
    }
    return summary, data


def resolve_paths(paths: list[Path] | None, default: Path) -> list[Path]:
    return [path.resolve() for path in paths] if paths else [default]


def count_value(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def build_per_video(expected: pd.DataFrame, export_a: pd.DataFrame, export_b: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_label = {}
    for label, export in [("a", export_a), ("b", export_b)]:
        if export.empty or "external_video_id" not in export.columns:
            by_label[label] = {}
            continue
        export = export.copy()
        export["external_video_id"] = export["external_video_id"].map(normalize_text)
        by_label[label] = {
            normalize_text(row.get("external_video_id", "")): row
            for _, row in export.iterrows()
            if normalize_text(row.get("external_video_id", ""))
        }
    for _, row in expected.iterrows():
        video_id = normalize_text(row.get("external_video_id", ""))
        a = by_label["a"].get(video_id, pd.Series(dtype=object))
        b = by_label["b"].get(video_id, pd.Series(dtype=object))
        a_complete = all(
            normalize_text(a.get(column, ""))
            for column in ["manual_breath_count", "reference_rr_annotator", "camera_id"]
        )
        b_complete = all(
            normalize_text(b.get(column, ""))
            for column in ["manual_breath_count", "reference_rr_annotator", "camera_id"]
        )
        a_count = count_value(a.get("manual_breath_count", ""))
        b_count = count_value(b.get("manual_breath_count", ""))
        count_disagreement = (
            ""
            if a_count is None or b_count is None
            else bool(abs(float(a_count) - float(b_count)) > 0)
        )
        needs_review = bool((not a_complete) or (not b_complete) or count_disagreement is True)
        rows.append(
            {
                "external_video_id": video_id,
                "blinded_id_a": normalize_text(row.get("blinded_id_a", "")),
                "display_order_a": normalize_text(row.get("display_order_a", "")),
                "blinded_id_b": normalize_text(row.get("blinded_id_b", "")),
                "display_order_b": normalize_text(row.get("display_order_b", "")),
                "a_breath_count": normalize_text(a.get("manual_breath_count", "")),
                "a_reference_rr_annotator": normalize_text(a.get("reference_rr_annotator", "")),
                "a_camera_id": normalize_text(a.get("camera_id", "")),
                "a_complete": bool(a_complete),
                "b_breath_count": normalize_text(b.get("manual_breath_count", "")),
                "b_reference_rr_annotator": normalize_text(b.get("reference_rr_annotator", "")),
                "b_camera_id": normalize_text(b.get("camera_id", "")),
                "b_complete": bool(b_complete),
                "dual_complete": bool(a_complete and b_complete),
                "count_disagreement": count_disagreement,
                "needs_annotation_or_review": needs_review,
            }
        )
    return pd.DataFrame(rows, columns=PER_VIDEO_COLUMNS)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if data.empty:
        return "_No rows available._"
    table = data[columns].head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for values in table.to_numpy():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    per_video: pd.DataFrame,
    ready_for_runner: bool,
    output_dir: Path,
) -> None:
    if ready_for_runner:
        status = "ready_for_after_annotation_runner"
        next_action = "Run scripts/run_rr_external_validation_after_annotation.py."
    else:
        status = "annotation_exports_incomplete"
        incomplete = summary[summary["status"].astype(str) != "COMPLETE"]
        next_action = "; ".join(incomplete["next_action"].astype(str).tolist())
    missing = per_video[per_video["needs_annotation_or_review"].astype(bool)].copy()
    text = f"""# External A/B Annotation Progress Monitor

Status: `{status}`

Next action: {next_action}

Ready for after-annotation runner: `{ready_for_runner}`

Output directory: `{output_dir}`

## Annotator Summary

{markdown_table(summary, SUMMARY_COLUMNS)}

## Rows Needing Annotation Or Review

{markdown_table(missing, ['external_video_id', 'blinded_id_a', 'display_order_a', 'blinded_id_b', 'display_order_b', 'a_complete', 'b_complete', 'count_disagreement'], max_rows=40)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worklist_a_paths = resolve_paths(
        args.worklist_a_csv,
        default_path(output_dir, "paper_external_validation_breath_annotation_worklist_annotator_a.csv"),
    )
    worklist_b_paths = resolve_paths(
        args.worklist_b_csv,
        default_path(output_dir, "paper_external_validation_breath_annotation_worklist_annotator_b.csv"),
    )
    annotation_a_paths = resolve_paths(
        args.annotation_a_csv,
        default_path(output_dir, "paper_external_validation_breath_annotation_annotator_a_export.csv"),
    )
    annotation_b_paths = resolve_paths(
        args.annotation_b_csv,
        default_path(output_dir, "paper_external_validation_breath_annotation_annotator_b_export.csv"),
    )
    worklist_a = pd.concat([read_optional_csv(path) for path in worklist_a_paths], ignore_index=True)
    worklist_b = pd.concat([read_optional_csv(path) for path in worklist_b_paths], ignore_index=True)
    expected = expected_from_worklists(worklist_a, worklist_b)
    expected_ids = set(expected["external_video_id"].astype(str).tolist())
    summary_a, export_a = summarize_annotator("a", annotation_a_paths, expected_ids)
    summary_b, export_b = summarize_annotator("b", annotation_b_paths, expected_ids)
    summary = pd.DataFrame([summary_a, summary_b], columns=SUMMARY_COLUMNS)
    per_video = build_per_video(expected, export_a, export_b)
    missing = per_video[per_video["needs_annotation_or_review"].astype(bool)].copy()
    ready_for_runner = bool(
        not summary.empty
        and summary["status"].astype(str).eq("COMPLETE").all()
        and not per_video.empty
        and per_video["dual_complete"].astype(bool).all()
    )

    prefix = args.report_prefix.strip() or "paper_external_validation"
    summary_path = output_dir / f"{prefix}_annotation_progress_summary.csv"
    per_video_path = output_dir / f"{prefix}_annotation_progress_by_video.csv"
    missing_path = output_dir / f"{prefix}_annotation_progress_missing_or_review.csv"
    report_path = output_dir / f"{prefix}_annotation_progress.md"
    summary.to_csv(summary_path, index=False)
    per_video.to_csv(per_video_path, index=False)
    missing.to_csv(missing_path, index=False)
    write_report(report_path, summary, per_video, ready_for_runner, output_dir)
    print(f"Saved annotation progress summary: {summary_path}")
    print(f"Saved annotation progress by-video table: {per_video_path}")
    print(f"Saved annotation missing/review table: {missing_path}")
    print(f"Saved annotation progress report: {report_path}")
    print(summary[["annotator", "status", "complete_required_rows", "expected_rows", "next_action"]].to_string(index=False))


if __name__ == "__main__":
    main()
