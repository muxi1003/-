from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DASHBOARD_COLUMNS = [
    "gate_order",
    "gate",
    "status",
    "evidence",
    "next_action",
    "q2_blocking",
    "paper_use",
]

ACTION_COLUMNS = [
    "action_order",
    "action",
    "why_it_matters",
    "input_artifact",
    "expected_output",
    "blocks_q2_external_claim",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build an annotation-to-submission dashboard for the external RR "
            "validation batch."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def metric_value(table: pd.DataFrame | None, metric: str, default: str = "0") -> str:
    if table is None or table.empty or "metric" not in table.columns:
        return default
    row = table[table["metric"].astype(str).eq(metric)]
    if row.empty:
        return default
    return str(row.iloc[0].get("value", default))


def numeric_text_to_int(value: object, default: int = 0) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return int(number)


def readiness_evidence(readiness: pd.DataFrame | None, check: str, default: str = "missing") -> str:
    if readiness is None or readiness.empty or "check" not in readiness.columns:
        return default
    row = readiness[readiness["check"].astype(str).eq(check)]
    if row.empty:
        return default
    return str(row.iloc[0].get("evidence", default))


def tier_row(tiers: pd.DataFrame | None, tier: str) -> tuple[str, str]:
    if tiers is None or tiers.empty or "tier" not in tiers.columns:
        return "MISSING", "tier_status_missing"
    row = tiers[tiers["tier"].astype(str).eq(tier)]
    if row.empty:
        return "MISSING", f"{tier}_row_missing"
    item = row.iloc[0]
    return str(item.get("status", "MISSING")), str(item.get("evidence", ""))


def dashboard_row(
    order: int,
    gate: str,
    status: str,
    evidence: str,
    next_action: str,
    q2_blocking: bool,
    paper_use: str,
) -> dict[str, object]:
    return {
        "gate_order": order,
        "gate": gate,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
        "q2_blocking": bool(q2_blocking),
        "paper_use": paper_use,
    }


def file_status(path: Path) -> str:
    return "PASS" if path.exists() else "FAIL"


def build_dashboard(output_dir: Path) -> pd.DataFrame:
    readiness = read_csv(output_dir / "paper_external_validation_split_all_use_readiness.csv")
    agreement = read_csv(output_dir / "paper_external_validation_breath_annotation_agreement_summary.csv")
    tiers = read_csv(output_dir / "paper_external_validation_fieldwork_tier_status.csv")
    execution = read_csv(output_dir / "paper_external_validation_execution_dashboard.csv")
    target_strategy = read_csv(output_dir / "paper_target_journal_strategy.csv")
    input_issues = read_csv(output_dir / "paper_external_validation_input_pack_issues.csv")
    annotation_progress = read_csv(
        output_dir / "paper_external_validation_annotation_progress_summary.csv"
    )

    included_evidence = readiness_evidence(
        readiness,
        "default included clips after duration screen",
        "included_clips=0; short_excluded=NA",
    )
    q2_size_evidence = readiness_evidence(
        readiness,
        "104-video Q2 algorithmic external tier by clips",
        "included_clips=0; target=104",
    )
    q2_size_status = "FAIL"
    if readiness is not None and not readiness.empty and "check" in readiness.columns:
        match = readiness[
            readiness["check"].astype(str).eq("104-video Q2 algorithmic external tier by clips")
        ]
        if not match.empty:
            q2_size_status = str(match.iloc[0].get("status", "FAIL"))

    included_rows = numeric_text_to_int(metric_value(agreement, "included_external_rows", "0"))
    complete_dual = numeric_text_to_int(metric_value(agreement, "complete_dual_annotation_rows", "0"))
    consensus_ready = numeric_text_to_int(metric_value(agreement, "consensus_ready_rows", "0"))
    needs_adjudication = numeric_text_to_int(metric_value(agreement, "needs_adjudication_rows", "0"))

    algorithmic_tier_status, algorithmic_tier_evidence = tier_row(
        tiers, "q2_algorithmic_external_validation"
    )
    smoke_status, smoke_evidence = tier_row(tiers, "minimum_holdout_smoke_test")
    target_status, target_evidence = tier_row(tiers, "q2_target_absolute_validation")

    freeze_status = "MISSING"
    freeze_evidence = "execution_dashboard_missing"
    external_split_status = "MISSING"
    external_split_evidence = "execution_dashboard_missing"
    if execution is not None and not execution.empty and "gate" in execution.columns:
        freeze = execution[execution["gate"].astype(str).eq("method_freeze_locked")]
        if not freeze.empty:
            freeze_status = str(freeze.iloc[0].get("status", "MISSING"))
            freeze_evidence = str(freeze.iloc[0].get("evidence", ""))
        split = execution[execution["gate"].astype(str).eq("frozen_external_split_validation")]
        if not split.empty:
            external_split_status = str(split.iloc[0].get("status", "MISSING"))
            external_split_evidence = str(split.iloc[0].get("evidence", ""))

    issue_count = 0 if input_issues is None else len(input_issues)
    issue_text = "none" if issue_count == 0 else f"issues={issue_count}"
    if input_issues is not None and not input_issues.empty and "issue" in input_issues.columns:
        issue_text = "; ".join(input_issues["issue"].astype(str).head(4).tolist())

    progress_text = "annotation_progress_not_generated"
    if (
        annotation_progress is not None
        and not annotation_progress.empty
        and {"annotator", "status", "complete_required_rows", "expected_rows"}.issubset(annotation_progress.columns)
    ):
        progress_text = "; ".join(
            f"{row['annotator']}={row['status']} "
            f"({row['complete_required_rows']}/{row['expected_rows']})"
            for _, row in annotation_progress.iterrows()
        )

    route_evidence = "target_strategy_missing"
    route_status = "MISSING"
    if target_strategy is not None and not target_strategy.empty:
        row = target_strategy.iloc[0]
        route_status = str(row.get("journal_strategy_status", "MISSING"))
        route_evidence = (
            f"{row.get('target_journal', '')}: "
            f"{row.get('missing_must_have_gates', '')}"
        )

    rows = [
        dashboard_row(
            1,
            "external_batch_quantity",
            q2_size_status,
            f"{included_evidence}; {q2_size_evidence}",
            "Keep the six short final fragments excluded unless manually justified.",
            False,
            "external sample size gate",
        ),
        dashboard_row(
            2,
            "dual_annotation_packets_exist",
            "PASS"
            if (
                file_status(output_dir / "paper_external_validation_breath_annotation_packet_annotator_a.html") == "PASS"
                and file_status(output_dir / "paper_external_validation_breath_annotation_packet_annotator_b.html") == "PASS"
            )
            else "FAIL",
            "annotator_a_html="
            + file_status(output_dir / "paper_external_validation_breath_annotation_packet_annotator_a.html")
            + "; annotator_b_html="
            + file_status(output_dir / "paper_external_validation_breath_annotation_packet_annotator_b.html"),
            "Use the A/B HTML packets for independent manual breath-count annotation.",
            True,
            "reference RR acquisition",
        ),
        dashboard_row(
            3,
            "dual_annotation_completion",
            "PASS" if included_rows > 0 and complete_dual >= included_rows else "FAIL",
            f"complete_dual={complete_dual}/{included_rows}; {progress_text}",
            "Use the annotation progress monitor, then export both annotator CSV files after all included clips are counted.",
            True,
            "reference RR quality control",
        ),
        dashboard_row(
            4,
            "consensus_ready",
            "PASS" if included_rows > 0 and consensus_ready >= included_rows else "FAIL",
            f"consensus_ready={consensus_ready}/{included_rows}; needs_adjudication={needs_adjudication}",
            "Fill the adjudication template and rerun the agreement audit until no rows need adjudication.",
            True,
            "reference RR quality control",
        ),
        dashboard_row(
            5,
            "input_pack_truth_ready",
            "PASS" if issue_count == 0 else "FAIL",
            issue_text,
            "Regenerate the input pack from the consensus fieldwork worksheet after agreement is ready.",
            True,
            "external truth and metadata templates",
        ),
        dashboard_row(
            6,
            "fieldwork_smoke_tier",
            smoke_status,
            smoke_evidence,
            "Complete required fieldwork fields and rerun preflight.",
            False,
            "minimum external feasibility",
        ),
        dashboard_row(
            7,
            "fieldwork_algorithmic_q2_tier",
            algorithmic_tier_status,
            algorithmic_tier_evidence,
            "Complete manual breath counts, reference annotator, and camera_id for the included external clips.",
            True,
            "Q2 algorithmic external validation",
        ),
        dashboard_row(
            8,
            "fieldwork_full_q2_metadata_tier",
            target_status,
            target_evidence,
            "Only pursue this route after real temperature/humidity and manual quality labels exist.",
            False,
            "heat-stress or manual-quality claims",
        ),
        dashboard_row(
            9,
            "method_freeze_locked",
            freeze_status,
            freeze_evidence,
            "Keep method files and thresholds frozen until external scoring is complete.",
            True,
            "external validation validity",
        ),
        dashboard_row(
            10,
            "external_split_scored",
            external_split_status,
            external_split_evidence,
            "Run frame extraction, frozen RR prediction, frozen postprocessors, and external split scoring after consensus truth exists.",
            True,
            "external RR metrics",
        ),
        dashboard_row(
            11,
            "target_journal_route",
            "HOLD" if route_status == "hold_for_external_validation" else route_status,
            route_evidence,
            "Do not submit to the selected Q2+ target until the external validation and fieldwork gates pass.",
            True,
            "submission strategy",
        ),
    ]
    return pd.DataFrame(rows, columns=DASHBOARD_COLUMNS)


def build_action_queue(dashboard: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "action_order": 1,
            "action": "complete_annotator_a_and_b_exports",
            "why_it_matters": "No external RR R2 can be computed without independent manual reference counts.",
            "input_artifact": "paper_external_validation_breath_annotation_packet_annotator_a.html; paper_external_validation_breath_annotation_packet_annotator_b.html",
            "expected_output": "paper_external_validation_breath_annotation_annotator_a_export.csv; paper_external_validation_breath_annotation_annotator_b_export.csv",
            "blocks_q2_external_claim": True,
        },
        {
            "action_order": 2,
            "action": "run_agreement_audit_and_adjudicate",
            "why_it_matters": "Dual annotation and adjudication make the external truth defensible for review.",
            "input_artifact": "two annotator export CSVs",
            "expected_output": "paper_external_validation_breath_annotation_consensus.csv with consensus_ready_rows equal to included rows",
            "blocks_q2_external_claim": True,
        },
        {
            "action_order": 3,
            "action": "merge_consensus_and_regenerate_external_input_pack",
            "why_it_matters": "The RR pipeline reads truth and metadata templates, not the HTML annotation state.",
            "input_artifact": "paper_external_validation_breath_annotation_consensus.csv",
            "expected_output": "paper_external_validation_truth_template.csv with no missing breath_count or rr",
            "blocks_q2_external_claim": True,
        },
        {
            "action_order": 4,
            "action": "extract_frames_and_score_frozen_external_split",
            "why_it_matters": "The manuscript needs method-frozen external RR metrics, not internal-only precision.",
            "input_artifact": "consensus fieldwork; external mp4 clips",
            "expected_output": "external_repro_external_split_metrics.csv and acceptance outputs",
            "blocks_q2_external_claim": True,
        },
        {
            "action_order": 5,
            "action": "refresh_submission_assets_and_claim_scope",
            "why_it_matters": "The manuscript wording must match the external evidence actually achieved.",
            "input_artifact": "external metrics and acceptance outputs",
            "expected_output": "submission readiness and target journal strategy PASS or explicit hold",
            "blocks_q2_external_claim": True,
        },
    ]
    if dashboard[dashboard["status"].astype(str).eq("FAIL")].empty:
        rows.insert(
            0,
            {
                "action_order": 0,
                "action": "all_dashboard_gates_passed",
                "why_it_matters": "No blocking action was detected in the dashboard.",
                "input_artifact": "dashboard",
                "expected_output": "review final submission readiness",
                "blocks_q2_external_claim": False,
            },
        )
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def markdown_table(data: pd.DataFrame, max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows._"
    view = data.head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, dashboard: pd.DataFrame, actions: pd.DataFrame) -> None:
    blocking = dashboard[
        dashboard["q2_blocking"].astype(bool)
        & ~dashboard["status"].astype(str).isin(["PASS"])
    ].copy()
    if blocking.empty:
        status = "ready_for_q2_external_claim_review"
        next_action = "Review external metrics and target-journal fit."
    else:
        status = "not_ready_for_q2_external_claim"
        first = blocking.sort_values("gate_order").iloc[0]
        next_action = str(first["next_action"])
    text = f"""# External Annotation To Submission Dashboard

Status: `{status}`

Next action: {next_action}

This dashboard combines the `split_all_use` external-batch inventory, dual
manual breath-count annotation QC, fieldwork preflight, method freeze, external
split status, and target-journal route. It is designed to prevent a Q2+
manuscript from claiming external validation before the external reference RR
and frozen scoring gates are actually complete.

## Gate Dashboard

{markdown_table(dashboard)}

## Action Queue

{markdown_table(actions)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard = build_dashboard(output_dir)
    actions = build_action_queue(dashboard)

    dashboard_path = output_dir / "paper_external_validation_annotation_submission_dashboard.csv"
    actions_path = output_dir / "paper_external_validation_annotation_submission_actions.csv"
    report_path = output_dir / "paper_external_validation_annotation_submission_dashboard.md"

    dashboard.to_csv(dashboard_path, index=False)
    actions.to_csv(actions_path, index=False)
    write_report(report_path, dashboard, actions)

    print(f"Saved annotation submission dashboard: {dashboard_path}")
    print(f"Saved annotation submission actions: {actions_path}")
    print(f"Saved annotation submission report: {report_path}")
    print(dashboard[["gate_order", "gate", "status", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
