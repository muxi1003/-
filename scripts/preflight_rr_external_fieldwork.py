from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_REQUIRED_COLUMNS = [
    "external_video_id",
    "raw_video_path",
    "cow_id",
    "collection_date",
    "camera_id",
    "scene_id",
    "external_test_split",
    "manual_breath_count",
    "manual_duration_seconds",
    "reference_rr_annotator",
    "include_in_external_validation",
]

OPTIONAL_REFERENCE_COLUMNS = [
    "manual_rr_bpm",
]

Q2_REQUIRED_COLUMNS = [
    "ambient_temperature_c",
    "relative_humidity_percent",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
]

SCORE_COLUMNS = [
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
]
ISSUE_COLUMNS = [
    "row_index",
    "external_video_id",
    "issue_type",
    "field",
    "severity",
    "evidence",
]

TIER_RULES = [
    {
        "tier": "minimum_holdout_smoke_test",
        "min_videos": 50,
        "min_cows": 5,
        "min_dates": 2,
        "min_camera_or_scene": 2,
        "requires_q2_metadata": False,
    },
    {
        "tier": "q2_target_absolute_validation",
        "min_videos": 104,
        "min_cows": 8,
        "min_dates": 4,
        "min_camera_or_scene": 2,
        "requires_q2_metadata": True,
    },
    {
        "tier": "q2_algorithmic_external_validation",
        "min_videos": 104,
        "min_cows": 8,
        "min_dates": 4,
        "min_camera_or_scene": 2,
        "requires_q2_metadata": False,
    },
    {
        "tier": "q2_strong_relative_improvement",
        "min_videos": 209,
        "min_cows": 12,
        "min_dates": 6,
        "min_camera_or_scene": 3,
        "requires_q2_metadata": True,
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Preflight-check the external validation fieldwork worksheet before "
            "frozen external RR scoring. The script reports readiness tiers and "
            "row-level issues without treating the current internal videos as "
            "external evidence."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--rr-tolerance-bpm",
        type=float,
        default=0.25,
        help="Allowed difference between manual_rr_bpm and breath_count/duration.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def default_fieldwork_csv(args: argparse.Namespace) -> Path:
    return output_dir_for(args) / "paper_external_validation_fieldwork_template.csv"


def nonempty(series: pd.Series) -> pd.Series:
    return ~series.fillna("").astype(str).str.strip().isin(["", "nan", "None", "<NA>"])


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1", "include", "included"]
    )


def active_fieldwork_mask(data: pd.DataFrame) -> pd.Series:
    identity_columns = [
        "external_video_id",
        "raw_video_path",
        "cow_id",
        "manual_breath_count",
        "manual_duration_seconds",
        "manual_rr_bpm",
    ]
    present = [column for column in identity_columns if column in data.columns]
    if not present:
        return pd.Series(False, index=data.index)
    active = pd.Series(False, index=data.index)
    for column in present:
        active = active | nonempty(data[column])
    return active


def numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def status_row(
    category: str,
    check: str,
    status: str,
    evidence: str,
    recommended_action: str,
    q2_blocking: bool,
    tier: str = "",
) -> dict[str, object]:
    return {
        "tier": tier,
        "category": category,
        "check": check,
        "status": status,
        "q2_blocking": bool(q2_blocking),
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def missing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in data.columns]


def unique_nonempty(data: pd.DataFrame, columns: list[str]) -> int:
    present = [column for column in columns if column in data.columns]
    if not present:
        return 0
    values = set()
    for column in present:
        values.update(
            value
            for value in data[column].fillna("").astype(str).str.strip().tolist()
            if value
        )
    return len(values)


def build_overall_status(data: pd.DataFrame, included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_required = BASE_REQUIRED_COLUMNS + Q2_REQUIRED_COLUMNS
    missing = missing_columns(data, all_required)
    rows.append(
        status_row(
            "structure",
            "required fieldwork columns present",
            "PASS" if not missing else "FAIL",
            "missing=" + ("none" if not missing else ";".join(missing)),
            "Regenerate paper_external_validation_fieldwork_template.csv if required columns are missing.",
            bool(missing),
        )
    )
    active_rows = int(active_fieldwork_mask(data).sum())
    rows.append(
        status_row(
            "structure",
            "active external fieldwork rows present",
            "PASS" if active_rows > 0 else "FAIL",
            f"active_rows={active_rows}; worksheet_rows={len(data)}",
            "Fill at least one independent external video row; blank template rows are not counted.",
            True,
        )
    )
    rows.append(
        status_row(
            "structure",
            "included external rows present",
            "PASS" if len(included) > 0 else "FAIL",
            f"included_rows={len(included)}",
            "Fill external fieldwork rows and set include_in_external_validation=yes.",
            True,
        )
    )
    if "external_test_split" in included.columns and not included.empty:
        external_mask = included["external_test_split"].fillna("").astype(str).str.strip().str.lower().isin(
            ["external", "holdout", "test"]
        )
        bad = int((~external_mask).sum())
    else:
        bad = len(included)
    rows.append(
        status_row(
            "split",
            "included rows labeled external",
            "PASS" if len(included) > 0 and bad == 0 else "FAIL",
            f"bad_external_split_labels={bad}; included_rows={len(included)}",
            "Label only independent external/holdout/test rows as external; keep current development rows internal.",
            True,
        )
    )
    return pd.DataFrame(rows)


def column_coverage(included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = BASE_REQUIRED_COLUMNS + OPTIONAL_REFERENCE_COLUMNS + Q2_REQUIRED_COLUMNS + [
        "collection_time",
        "posture",
        "athi",
        "reference_protocol_notes",
        "error_driven_planning_stratum",
        "matched_internal_prototype_video_id",
    ]
    total = len(included)
    for column in columns:
        if column not in included.columns:
            rows.append(
                {
                    "field": column,
                    "present": False,
                    "nonempty": 0,
                    "included_rows": total,
                    "coverage_percent": 0.0,
                    "q2_required": column in Q2_REQUIRED_COLUMNS or column in BASE_REQUIRED_COLUMNS,
                }
            )
            continue
        count = int(nonempty(included[column]).sum())
        rows.append(
            {
                "field": column,
                "present": True,
                "nonempty": count,
                "included_rows": total,
                "coverage_percent": (100.0 * count / total) if total else 0.0,
                "q2_required": column in Q2_REQUIRED_COLUMNS or column in BASE_REQUIRED_COLUMNS,
            }
        )
    return pd.DataFrame(rows)


def build_value_checks(included: pd.DataFrame, rr_tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(included)

    def add_numeric_range(column: str, low: float, high: float, q2_blocking: bool) -> None:
        values = numeric(included, column)
        present = values.notna()
        invalid = int((present & ~values.between(low, high)).sum())
        missing = int((~present).sum())
        rows.append(
            status_row(
                "value_quality",
                f"{column} numeric range",
                "PASS" if total > 0 and invalid == 0 else "FAIL",
                f"missing={missing}/{total}; invalid={invalid}; valid_range=[{low},{high}]",
                f"Fill {column} with numeric values in the valid range.",
                q2_blocking,
            )
        )

    add_numeric_range("manual_breath_count", 1, 200, True)
    add_numeric_range("manual_duration_seconds", 5, 600, True)
    add_numeric_range("manual_rr_bpm", 1, 120, False)
    add_numeric_range("ambient_temperature_c", -40, 60, True)
    add_numeric_range("relative_humidity_percent", 0, 100, True)
    for column in SCORE_COLUMNS:
        add_numeric_range(column, 0, 3, True)

    if {"manual_breath_count", "manual_duration_seconds", "manual_rr_bpm"}.issubset(
        included.columns
    ):
        count = numeric(included, "manual_breath_count")
        duration = numeric(included, "manual_duration_seconds")
        rr = numeric(included, "manual_rr_bpm")
        computed = count * 60.0 / duration.replace(0.0, np.nan)
        comparable = count.notna() & duration.notna() & rr.notna() & computed.notna()
        diff = (rr - computed).abs()
        inconsistent = int((comparable & (diff > rr_tolerance)).sum())
        rows.append(
            status_row(
                "reference_rr",
                "manual_rr_bpm consistent with count and duration",
                "PASS" if total > 0 and inconsistent == 0 else "FAIL",
                (
                    f"comparable={int(comparable.sum())}/{total}; "
                    f"inconsistent={inconsistent}; tolerance_bpm={rr_tolerance}"
                ),
                "Compute manual_rr_bpm as manual_breath_count * 60 / manual_duration_seconds or leave it blank until computed consistently.",
                bool(inconsistent),
            )
        )
    return pd.DataFrame(rows)


def tier_status(included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(included)
    cow_count = unique_nonempty(included, ["cow_id"])
    date_count = unique_nonempty(included, ["collection_date"])
    camera_scene_count = unique_nonempty(included, ["camera_id", "scene_id"])
    base_missing = missing_columns(included, BASE_REQUIRED_COLUMNS)
    q2_missing = missing_columns(included, Q2_REQUIRED_COLUMNS)

    base_complete = False
    if not base_missing and total:
        base_complete = all(nonempty(included[column]).all() for column in BASE_REQUIRED_COLUMNS)
    q2_complete = False
    if not q2_missing and total:
        q2_complete = all(nonempty(included[column]).all() for column in Q2_REQUIRED_COLUMNS)

    for rule in TIER_RULES:
        tier = str(rule["tier"])
        checks = {
            "videos": total >= int(rule["min_videos"]),
            "cow_id": cow_count >= int(rule["min_cows"]),
            "collection_date": date_count >= int(rule["min_dates"]),
            "camera_or_scene": camera_scene_count >= int(rule["min_camera_or_scene"]),
            "base_required_fields": base_complete,
            "q2_required_fields": q2_complete if rule["requires_q2_metadata"] else True,
        }
        passed = all(checks.values())
        rows.append(
            status_row(
                "tier_readiness",
                f"{tier} ready",
                "PASS" if passed else "FAIL",
                (
                    f"included={total}/{rule['min_videos']}; cows={cow_count}/{rule['min_cows']}; "
                    f"dates={date_count}/{rule['min_dates']}; camera_or_scene={camera_scene_count}/{rule['min_camera_or_scene']}; "
                    f"base_fields_complete={base_complete}; q2_fields_complete={q2_complete}"
                ),
                (
                    "Fill the external fieldwork template to satisfy this tier before "
                    "claiming it. The q2_algorithmic_external_validation tier supports "
                    "algorithmic/external-RR claims only; heat-stress and manual-quality "
                    "stratified claims still require the Q2 metadata fields."
                ),
                bool(rule["requires_q2_metadata"] or tier != "minimum_holdout_smoke_test"),
                tier=tier,
            )
        )
    return pd.DataFrame(rows)


def issue_rows(data: pd.DataFrame, included: pd.DataFrame, rr_tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = BASE_REQUIRED_COLUMNS + Q2_REQUIRED_COLUMNS
    for idx, item in included.iterrows():
        identifier = item.get("external_video_id", "")
        row_label = identifier if str(identifier).strip() else f"row_{idx + 2}"
        for column in required:
            claim_dependent = column in Q2_REQUIRED_COLUMNS
            if column not in included.columns:
                rows.append(
                    {
                        "row_index": int(idx),
                        "external_video_id": row_label,
                        "issue_type": (
                            "missing_claim_dependent_column"
                            if claim_dependent
                            else "missing_column"
                        ),
                        "field": column,
                        "severity": "WARN" if claim_dependent else "FAIL",
                        "evidence": (
                            "column not present; required for heat-stress/manual-quality claims"
                            if claim_dependent
                            else "column not present"
                        ),
                    }
                )
            elif not nonempty(pd.Series([item.get(column, "")])).iloc[0]:
                rows.append(
                    {
                        "row_index": int(idx),
                        "external_video_id": row_label,
                        "issue_type": (
                            "missing_claim_dependent_value"
                            if claim_dependent
                            else "missing_value"
                        ),
                        "field": column,
                        "severity": "WARN" if claim_dependent else "FAIL",
                        "evidence": (
                            "claim-dependent field blank; do not make heat-stress/manual-quality claims"
                            if claim_dependent
                            else "required field blank"
                        ),
                    }
                )
        for column in SCORE_COLUMNS:
            if column in included.columns:
                value = pd.to_numeric(pd.Series([item.get(column)]), errors="coerce").iloc[0]
                if pd.notna(value) and not (0 <= value <= 3):
                    rows.append(
                        {
                            "row_index": int(idx),
                            "external_video_id": row_label,
                            "issue_type": "invalid_score",
                            "field": column,
                            "severity": "FAIL",
                            "evidence": f"value={value}",
                        }
                    )
        if {"manual_breath_count", "manual_duration_seconds", "manual_rr_bpm"}.issubset(
            included.columns
        ):
            count = pd.to_numeric(pd.Series([item.get("manual_breath_count")]), errors="coerce").iloc[0]
            duration = pd.to_numeric(pd.Series([item.get("manual_duration_seconds")]), errors="coerce").iloc[0]
            rr = pd.to_numeric(pd.Series([item.get("manual_rr_bpm")]), errors="coerce").iloc[0]
            if pd.notna(count) and pd.notna(duration) and pd.notna(rr) and duration > 0:
                computed = count * 60.0 / duration
                diff = abs(float(rr) - float(computed))
                if diff > rr_tolerance:
                    rows.append(
                        {
                            "row_index": int(idx),
                            "external_video_id": row_label,
                            "issue_type": "rr_consistency",
                            "field": "manual_rr_bpm",
                            "severity": "FAIL",
                            "evidence": f"manual_rr={rr:.4f}; computed_rr={computed:.4f}; diff={diff:.4f}",
                        }
                    )
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows available._"
    view = data[[column for column in columns if column in data.columns]].head(max_rows).copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    preflight: pd.DataFrame,
    tiers: pd.DataFrame,
    coverage: pd.DataFrame,
    issues: pd.DataFrame,
    fieldwork_csv: Path,
) -> None:
    blockers = preflight[
        preflight["status"].astype(str).isin(["FAIL", "WARN"])
        | preflight["q2_blocking"].astype(bool)
    ].copy()
    text = f"""# External Fieldwork Preflight

Fieldwork CSV: `{fieldwork_csv}`

This report checks whether the external validation worksheet is ready for
method-frozen external RR scoring. It does not score RR predictions and it does
not convert internal development videos into external evidence.

## Tier Readiness

{markdown_table(tiers, ['tier', 'check', 'status', 'evidence', 'recommended_action'])}

## Blocking Or Caution Checks

{markdown_table(blockers, ['category', 'check', 'status', 'q2_blocking', 'evidence', 'recommended_action'])}

## Required Field Coverage

{markdown_table(coverage, ['field', 'present', 'nonempty', 'included_rows', 'coverage_percent', 'q2_required'])}

## Row-Level Issues

{markdown_table(issues, ['row_index', 'external_video_id', 'issue_type', 'field', 'severity', 'evidence'])}

## Use Boundary

Only rows from an independent collection should be marked external. The current
73 development videos remain internal and should not be relabeled to pass this
preflight.
"""
    path.write_text(text, encoding="utf-8")


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
    if not fieldwork_csv.exists():
        raise FileNotFoundError(f"Missing external fieldwork CSV: {fieldwork_csv}")

    data = pd.read_csv(fieldwork_csv, dtype=str, keep_default_na=False)
    if "include_in_external_validation" in data.columns:
        included = data[
            yes_mask(data["include_in_external_validation"])
            & active_fieldwork_mask(data)
        ].copy()
    else:
        included = data.iloc[0:0].copy()

    overall = build_overall_status(data, included)
    values = build_value_checks(included, args.rr_tolerance_bpm)
    tiers = tier_status(included)
    coverage = column_coverage(included)
    issues = issue_rows(data, included, args.rr_tolerance_bpm)
    preflight = pd.concat([overall, values, tiers], ignore_index=True)

    preflight_path = output_dir / "paper_external_validation_fieldwork_preflight.csv"
    tiers_path = output_dir / "paper_external_validation_fieldwork_tier_status.csv"
    coverage_path = output_dir / "paper_external_validation_fieldwork_field_coverage.csv"
    issues_path = output_dir / "paper_external_validation_fieldwork_row_issues.csv"
    report_path = output_dir / "paper_external_validation_fieldwork_preflight.md"
    preflight.to_csv(preflight_path, index=False)
    tiers.to_csv(tiers_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    issues.to_csv(issues_path, index=False)
    write_report(report_path, preflight, tiers, coverage, issues, fieldwork_csv)

    print(f"Saved external fieldwork preflight: {preflight_path}")
    print(f"Saved external fieldwork tier status: {tiers_path}")
    print(f"Saved external fieldwork field coverage: {coverage_path}")
    print(f"Saved external fieldwork row issues: {issues_path}")
    print(f"Saved external fieldwork preflight report: {report_path}")
    print("\nTier readiness:")
    print(tiers[["tier", "status", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
