from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_METADATA_FIELDS = [
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


FIELD_GUIDANCE = {
    "cow_id": {
        "valid_values": "numeric video label in the current dataset; stable animal identity only if later verified",
        "source": "numeric part of video_id per current dataset convention",
        "unlocks": "numeric-video-label grouped stress testing only; true animal-level wording requires separate real cow identities",
    },
    "collection_date": {
        "valid_values": "YYYY-MM-DD",
        "source": "video acquisition log or file/session record",
        "unlocks": "optional date/session stratification and leakage checks",
    },
    "collection_start_date": {
        "valid_values": "YYYY-MM-DD",
        "source": "known collection window",
        "unlocks": "collection-window provenance",
    },
    "collection_end_date": {
        "valid_values": "YYYY-MM-DD",
        "source": "known collection window",
        "unlocks": "collection-window provenance",
    },
    "collection_location_country": {
        "valid_values": "country name",
        "source": "known collection provenance",
        "unlocks": "collection-location provenance",
    },
    "collection_location_province": {
        "valid_values": "province/state name",
        "source": "known collection provenance",
        "unlocks": "collection-location provenance",
    },
    "collection_location_county": {
        "valid_values": "county name",
        "source": "known collection provenance",
        "unlocks": "collection-location provenance",
    },
    "collection_site": {
        "valid_values": "farm/ranch/site label",
        "source": "known collection provenance",
        "unlocks": "collection-site provenance",
    },
    "camera_id": {
        "valid_values": "stable camera name or ID",
        "source": "camera log or acquisition folder metadata",
        "unlocks": "optional camera-domain validation and deployment discussion",
    },
    "scene_id": {
        "valid_values": "stable barn/pen/location/session scene label",
        "source": "acquisition notes or visual review",
        "unlocks": "scene-level validation and robustness analysis",
    },
    "ambient_temperature_c": {
        "valid_values": "numeric Celsius value",
        "source": "barn sensor or synchronized acquisition record only; regional weather proxy can support context only and should remain separate",
        "unlocks": "THI calculation and heat-stress context only when real synchronized measurements exist",
    },
    "relative_humidity_percent": {
        "valid_values": "numeric percent from 0 to 100",
        "source": "barn sensor or synchronized acquisition record only; regional weather proxy can support context only and should remain separate",
        "unlocks": "THI calculation and heat-stress context only when real synchronized measurements exist",
    },
    "thi": {
        "valid_values": "numeric; optional if temperature and humidity are present",
        "source": "computed by scripts/build_rr_heat_stress_context.py when input fields exist",
        "unlocks": "THI-stratified RR and error analysis",
    },
    "athi": {
        "valid_values": "numeric if your study defines adjusted THI",
        "source": "manual calculation from the chosen ATHI formula; do not auto-impute",
        "unlocks": "ATHI-based heat-stress comparison",
    },
    "head_motion_score_0_3": {
        "valid_values": "0=no/minimal, 1=mild, 2=moderate, 3=severe",
        "source": "manual visual review only; leave blank if review scores cannot be produced",
        "unlocks": "motion-robustness stratified performance claim",
    },
    "occlusion_score_0_3": {
        "valid_values": "0=none, 1=mild, 2=moderate, 3=severe",
        "source": "manual visual review only; leave blank if review scores cannot be produced",
        "unlocks": "occlusion robustness/error-source analysis",
    },
    "nostril_visibility_score_0_3": {
        "valid_values": "0=not visible, 1=poor, 2=partial, 3=clear",
        "source": "manual visual review only; leave blank if review scores cannot be produced",
        "unlocks": "ROI-quality stratified performance claim",
    },
    "external_test_split": {
        "valid_values": "internal or external; use only from real study design, not from error outcome",
        "source": "collection design or genuinely independent holdout dataset",
        "unlocks": "frozen external/holdout validation metrics",
    },
}


ACTION_MAP = {
    "cow_id numeric-video-label grouping readiness": {
        "required_fields": "cow_id",
        "deliverable": "fill cow_id from the numeric video label for all videos or all videos in the validation subset",
        "proof_command": "python scripts/rr_quality_residual_metadata_group_validation.py --group-columns cow_id",
        "proof_output": "paper_repro_quality_residual_metadata_group_readiness.csv ready_for_group_validation=True",
        "claim_unlocked": "replace heuristic prefix groups with numeric-video-label grouped stress testing; real cow-level validation still requires true animal IDs",
        "priority": "P0",
    },
    "external split labels complete": {
        "required_fields": "external_test_split",
        "deliverable": "keep current rows internal unless a method-frozen independent external set exists",
        "proof_command": "python scripts/rr_external_split_validation.py",
        "proof_output": "paper_repro_external_split_readiness.csv external split validation ready=PASS",
        "claim_unlocked": "report frozen holdout/external performance instead of internal-only validation",
        "priority": "P0",
    },
    "frozen external split validation ready": {
        "required_fields": "external_test_split",
        "deliverable": "rerun frozen split validation after split labels are present",
        "proof_command": "python scripts/rr_external_split_validation.py",
        "proof_output": "paper_repro_external_split_metrics.csv with nonzero external rows",
        "claim_unlocked": "external validation paragraph and table",
        "priority": "P0",
    },
    "ambient temperature context complete": {
        "required_fields": "ambient_temperature_c",
        "deliverable": "fill only if synchronized barn/session temperature records exist; otherwise keep as unavailable limitation",
        "proof_command": "python scripts/build_rr_heat_stress_context.py",
        "proof_output": "paper_heat_stress_readiness.csv ambient_temperature_c ready=True",
        "claim_unlocked": "optional environmental context and THI calculation",
        "priority": "P2",
    },
    "relative humidity context complete": {
        "required_fields": "relative_humidity_percent",
        "deliverable": "fill only if synchronized barn/session humidity records exist; otherwise keep as unavailable limitation",
        "proof_command": "python scripts/build_rr_heat_stress_context.py",
        "proof_output": "paper_heat_stress_readiness.csv relative_humidity_percent ready=True",
        "claim_unlocked": "optional THI calculation and heat-stress stratification",
        "priority": "P2",
    },
    "THI analysis readiness": {
        "required_fields": "ambient_temperature_c;relative_humidity_percent;thi",
        "deliverable": "provide real temperature and humidity, then compute THI; do not use regional proxy as per-video THI evidence",
        "proof_command": "python scripts/build_rr_heat_stress_context.py",
        "proof_output": "paper_heat_stress_association_table.csv with THI rows when enough values exist",
        "claim_unlocked": "optional heat-stress/welfare meaning beyond RR detection accuracy",
        "priority": "P2",
    },
    "head-motion robustness metadata complete": {
        "required_fields": "head_motion_score_0_3",
        "deliverable": "score head motion from 0 to 3 only if manual visual review is feasible; otherwise report unavailable",
        "proof_command": "python scripts/build_rr_metadata_annotation_pack.py",
        "proof_output": "paper_metadata_annotation_progress.csv head_motion_score_0_3 coverage>0",
        "claim_unlocked": "optional motion-robustness stratified validation",
        "priority": "P2",
    },
    "occlusion robustness metadata complete": {
        "required_fields": "occlusion_score_0_3",
        "deliverable": "score nostril/face occlusion from 0 to 3 only if manual visual review is feasible; otherwise report unavailable",
        "proof_command": "python scripts/build_rr_metadata_annotation_pack.py",
        "proof_output": "paper_metadata_annotation_progress.csv occlusion_score_0_3 coverage>0",
        "claim_unlocked": "optional occlusion-specific performance/error analysis",
        "priority": "P2",
    },
    "nostril visibility metadata complete": {
        "required_fields": "nostril_visibility_score_0_3",
        "deliverable": "score nostril visibility from 0 to 3 only if manual visual review is feasible; otherwise report unavailable",
        "proof_command": "python scripts/build_rr_metadata_annotation_pack.py",
        "proof_output": "paper_metadata_annotation_progress.csv nostril_visibility_score_0_3 coverage>0",
        "claim_unlocked": "optional ROI-quality stratified performance analysis",
        "priority": "P2",
    },
    "Delta RR R2 bootstrap CI excludes zero": {
        "required_fields": "external_test_split;cow_id;more_videos_if_available",
        "deliverable": "strengthen evidence with external/metadata-stratified validation or a larger dataset",
        "proof_command": "python scripts/audit_rr_submission_readiness.py",
        "proof_output": "paper_repro_quality_residual_submission_readiness.csv statistics check PASS or evidence explained cautiously",
        "claim_unlocked": "stronger statistical support for the precision-improvement claim",
        "priority": "P2",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build a Q2+ submission gap action pack from current readiness outputs."
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--top-videos", type=int, default=30)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_action_plan(readiness: pd.DataFrame) -> pd.DataFrame:
    blockers = readiness[
        readiness["q2_blocking"].map(truthy) | readiness["status"].isin(["FAIL", "WARN"])
    ].copy()
    rows = []
    for _, row in blockers.iterrows():
        check = str(row["check"])
        mapped = ACTION_MAP.get(check, {})
        rows.append(
            {
                "category": row.get("category", ""),
                "check": check,
                "status": row.get("status", ""),
                "priority": mapped.get("priority", "P2"),
                "required_fields": mapped.get("required_fields", ""),
                "deliverable": mapped.get("deliverable", row.get("recommended_action", "")),
                "proof_command": mapped.get("proof_command", "python scripts/audit_rr_submission_readiness.py"),
                "proof_output": mapped.get("proof_output", ""),
                "claim_unlocked": mapped.get("claim_unlocked", ""),
                "current_evidence": row.get("evidence", ""),
                "recommended_action": row.get("recommended_action", ""),
            }
        )
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    result = pd.DataFrame(rows)
    result["_priority_order"] = result["priority"].map(priority_order).fillna(99)
    result = result.sort_values(["_priority_order", "category", "check"]).drop(
        columns=["_priority_order"]
    )
    return result


def build_field_checklist(progress: pd.DataFrame | None, action_plan: pd.DataFrame) -> pd.DataFrame:
    needed = set(REQUIRED_METADATA_FIELDS + ["thi", "athi"])
    for fields in action_plan["required_fields"].dropna():
        for field in str(fields).split(";"):
            field = field.strip()
            if field and field != "more_videos_if_available":
                needed.add(field)
    progress_by_field = {}
    if progress is not None and not progress.empty and "field" in progress.columns:
        progress_by_field = {
            str(row["field"]): row for _, row in progress.iterrows()
        }
    rows = []
    for field in sorted(needed):
        guidance = FIELD_GUIDANCE.get(
            field,
            {
                "valid_values": "",
                "source": "",
                "unlocks": "",
            },
        )
        p = progress_by_field.get(field)
        rows.append(
            {
                "field": field,
                "current_nonempty": int(float(p["nonempty"])) if p is not None and "nonempty" in p else "",
                "total_videos": int(float(p["total_videos"])) if p is not None and "total_videos" in p else "",
                "coverage_percent": float(p["coverage_percent"]) if p is not None and "coverage_percent" in p else "",
                "valid_values": guidance["valid_values"],
                "recommended_source": guidance["source"],
                "downstream_claim_unlocked": guidance["unlocks"],
                "fill_priority": (
                    "P0"
                    if field in set(REQUIRED_METADATA_FIELDS)
                    else "P2_claim_specific"
                ),
            }
        )
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P2_claim_specific": 2}
    result = pd.DataFrame(rows)
    result["_priority_order"] = result["fill_priority"].map(priority_order)
    result = result.sort_values(["_priority_order", "field"]).drop(columns=["_priority_order"])
    return result


def metadata_boundary(progress: pd.DataFrame | None) -> str:
    if progress is None or progress.empty or "field" not in progress.columns:
        return "metadata progress table not generated"
    by_field = {str(row["field"]): row for _, row in progress.iterrows()}

    def coverage(field: str) -> tuple[int, int, int]:
        row = by_field.get(field)
        if row is None:
            return 0, 0, 0
        return (
            int(float(row.get("nonempty", 0))),
            int(float(row.get("total_videos", 0))),
            int(float(row.get("unique_values", 0))),
        )

    filled = []
    for field in [
        "cow_id",
        "collection_start_date",
        "collection_end_date",
        "collection_site",
        "scene_id",
        "ambient_temperature_c",
        "relative_humidity_percent",
        "thi",
        "external_test_split",
    ]:
        nonempty, total, unique = coverage(field)
        filled.append(f"{field}={nonempty}/{total} ({unique} unique)")
    still_missing = [
        field
        for field in [
            "collection_date",
            "camera_id",
            "head_motion_score_0_3",
            "occlusion_score_0_3",
            "nostril_visibility_score_0_3",
        ]
        if coverage(field)[0] == 0
    ]
    return (
        "filled/current fields: "
        + ", ".join(filled)
        + "; remaining gaps: "
        + (", ".join(still_missing) if still_missing else "none")
    )


def build_video_queue(annotation_sheet: pd.DataFrame, top_videos: int) -> pd.DataFrame:
    sheet = annotation_sheet.copy()
    for column in [
        "q2_annotation_score",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "signal_consensus_rr_bpm",
        "selective_review_score",
        "abs_count_error",
        "selective_abs_rr_error",
    ]:
        if column in sheet.columns:
            sheet[column] = pd.to_numeric(sheet[column], errors="coerce")
    priority_rank = {"critical": 0, "high": 1, "medium": 2, "normal": 3}
    sheet["_priority_rank"] = sheet.get("annotation_priority", "").map(priority_rank).fillna(9)
    sheet = sheet.sort_values(
        ["_priority_rank", "q2_annotation_score", "selective_review_score"],
        ascending=[True, False, False],
    )
    keep_columns = [
        "video_id",
        "annotation_priority",
        "q2_annotation_score",
        "q2_annotation_batch",
        "q2_missing_required_fields",
        "annotation_reason",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "signal_consensus_rr_bpm",
        "truth_count",
        "peaks",
        "corrected_peaks",
        "signal_consensus_peaks",
        "count_error",
        "corrected_count_error",
        "error_type",
        "selective_action",
        "selective_review_score",
        "strict_auto_accept",
        "sample_frame_middle",
        "curve_png",
        "review_png",
    ]
    for column in keep_columns:
        if column not in sheet.columns:
            sheet[column] = ""
    queue = sheet[keep_columns].head(top_videos).copy()
    queue["first_pass_fields_to_fill"] = ";".join(REQUIRED_METADATA_FIELDS)
    queue["annotation_instruction"] = (
        "Use the dashboard frames, curve plot, and peak-review plot; fill real metadata only, "
        "then rerun metadata/import/readiness scripts."
    )
    return queue


def write_markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    table = table.fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    action_plan: pd.DataFrame,
    field_checklist: pd.DataFrame,
    video_queue: pd.DataFrame,
    main_results: pd.DataFrame | None,
    progress: pd.DataFrame | None,
) -> Path:
    report = output_dir / "paper_submission_gap_action_pack.md"
    dashboard = output_dir / "paper_metadata_annotation_dashboard.html"
    safe_gate_summary = "Conservative signal-aware safe gate metrics not found."
    if main_results is not None and not main_results.empty:
        mask = main_results["method"].astype(str).str.contains(
            "Conservative signal-aware safe gate", case=False, regex=False
        )
        if mask.any():
            rows = []
            for _, row in main_results[mask].iterrows():
                rows.append(
                    "{method}: RR R2={r2:.4f}, MAE={mae:.3f}, exact={exact}, paper_use={paper_use}".format(
                        method=row["method"],
                        r2=float(row["rr_r2"]),
                        mae=float(row["rr_mae_bpm"]),
                        exact=row["exact_count"],
                        paper_use=row["paper_use"],
                    )
                )
            safe_gate_summary = "; ".join(rows)
    text = f"""# Q2+ Submission Gap Action Pack

Generated from the current readiness and metadata-annotation outputs. This file is an execution checklist for moving from an internally ready manuscript package to a Q2-or-higher submission-ready package.

## Current Boundary

- Internal manuscript package: ready.
- Q2-or-higher submission readiness: not_ready.
- Metadata status: {metadata_boundary(progress)}.
- Main reason Q2+ is still not ready: a nonzero independent external/holdout split is still missing. Exact per-video date, camera ID, synchronized environment records, and manual quality scores are claim-specific limitations, not fields to fabricate.
- Current highest-precision internal candidate: {safe_gate_summary}
- Freeze/use boundary: quality-aware residual correction remains the primary frozen method; the conservative signal-aware safe gate is a secondary candidate until it passes real cow/session and external split validation.
- Do not use truth-calibrated RR as main performance; keep it as upper-bound diagnostics only.

## Priority Action Plan

{write_markdown_table(action_plan, ["priority", "category", "check", "status", "required_fields", "deliverable", "claim_unlocked"])}

## Field Checklist

{write_markdown_table(field_checklist, ["fill_priority", "field", "current_nonempty", "total_videos", "coverage_percent", "valid_values", "downstream_claim_unlocked"])}

## First Videos To Annotate

{write_markdown_table(video_queue, ["video_id", "annotation_priority", "q2_annotation_score", "q2_annotation_batch", "q2_missing_required_fields", "error_type", "selective_action"], max_rows=20)}

## Dashboard

`{dashboard}`

Use the dashboard to review sample frames, respiratory curves, and peak-review images. Fill only verified metadata; do not infer external split or manual quality scores from prediction errors. Regional weather proxies may be used only as collection-context covariates unless synchronized barn measurements become available.

## Rerun Sequence After Filling Metadata

```powershell
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\import_rr_metadata_annotations.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\import_rr_metadata_annotations.py --write-template
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_quality_residual_metadata_group_validation.py --group-columns cow_id
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_quality_residual_metadata_group_validation.py --group-columns cow_id collection_date
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_heat_stress_context.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\rr_external_split_validation.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_external_validation_sample_plan.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\audit_rr_submission_readiness.py
& 'E:\\real\\anaconda\\envs\\plant_gpu\\python.exe' scripts\\build_rr_paper_assets.py
```
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness = pd.read_csv(
        require_file(input_root / f"{args.corrected_prefix}_submission_readiness.csv")
    )
    progress_path = output_dir / "paper_metadata_annotation_progress.csv"
    progress = pd.read_csv(progress_path) if progress_path.exists() else None
    annotation_sheet = pd.read_csv(
        require_file(output_dir / "paper_metadata_annotation_sheet.csv"), dtype=str
    )
    main_results_path = output_dir / "paper_main_results_table.csv"
    main_results = pd.read_csv(main_results_path) if main_results_path.exists() else None

    action_plan = build_action_plan(readiness)
    field_checklist = build_field_checklist(progress, action_plan)
    video_queue = build_video_queue(annotation_sheet, args.top_videos)

    action_path = output_dir / "paper_submission_gap_action_plan.csv"
    field_path = output_dir / "paper_submission_gap_field_checklist.csv"
    queue_path = output_dir / "paper_submission_gap_video_queue.csv"
    action_plan.to_csv(action_path, index=False)
    field_checklist.to_csv(field_path, index=False)
    video_queue.to_csv(queue_path, index=False)
    report = write_report(
        output_dir,
        action_plan,
        field_checklist,
        video_queue,
        main_results,
        progress,
    )

    print(f"Saved submission gap action plan: {action_path}")
    print(f"Saved submission gap field checklist: {field_path}")
    print(f"Saved submission gap video queue: {queue_path}")
    print(f"Saved submission gap report: {report}")
    print("\nPriority actions:")
    print(action_plan[["priority", "category", "check", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
