from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build the execution plan for method-frozen external validation: input "
            "contract, command sequence, readiness dashboard, and claim gates."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ready"}


def get_status(table: pd.DataFrame, check: str) -> str:
    if table.empty or "check" not in table.columns:
        return "MISSING"
    match = table[table["check"].astype(str) == check]
    if match.empty:
        return "MISSING"
    return str(match.iloc[0].get("status", "MISSING")).upper()


def freeze_status(method_freeze: pd.DataFrame) -> tuple[bool, str, str]:
    if method_freeze.empty:
        return False, "", "method freeze summary missing"
    item = method_freeze.iloc[0]
    freeze_id = str(item.get("method_freeze_id", ""))
    status = str(item.get("freeze_status", ""))
    try:
        missing = int(float(item.get("required_files_missing", 1)))
    except (TypeError, ValueError):
        missing = 1
    locked = bool(freeze_id and status == "locked" and missing == 0)
    return locked, freeze_id, f"freeze_status={status}; required_files_missing={missing}"


def tier_status(tiers: pd.DataFrame, tier: str) -> str:
    if tiers.empty:
        return "MISSING"
    match = tiers[tiers["tier"].astype(str) == tier]
    if match.empty:
        return "MISSING"
    return str(match.iloc[0].get("status", "MISSING")).upper()


def build_dashboard(
    output_dir: Path,
    input_root: Path,
    output_prefix: str,
) -> pd.DataFrame:
    fieldwork = read_optional(output_dir / "paper_external_validation_fieldwork_preflight.csv")
    tiers = read_optional(output_dir / "paper_external_validation_fieldwork_tier_status.csv")
    external_split = read_optional(input_root / f"{output_prefix}_external_split_readiness.csv")
    method_freeze = read_optional(output_dir / "paper_method_freeze_summary.csv")
    metadata_audit = read_optional(output_dir / "paper_metadata_quality_audit.csv")
    target_gate = read_optional(output_dir / "paper_target_journal_gate_matrix.csv")
    locked, freeze_id, freeze_evidence = freeze_status(method_freeze)

    metadata_blockers = 0
    if not metadata_audit.empty and "q2_blocking" in metadata_audit.columns:
        metadata_blockers = int(metadata_audit["q2_blocking"].map(truthy).sum())

    active_rows_status = get_status(fieldwork, "active external fieldwork rows present")
    smoke_status = tier_status(tiers, "minimum_holdout_smoke_test")
    q2_status = tier_status(tiers, "q2_target_absolute_validation")
    algorithmic_q2_status = tier_status(tiers, "q2_algorithmic_external_validation")
    external_status = get_status(external_split, "external split validation ready")

    rows = [
        {
            "gate_order": 1,
            "gate": "method_freeze_locked",
            "status": "PASS" if locked else "FAIL",
            "evidence": freeze_evidence,
            "next_action": "Use this freeze ID for all external scoring." if locked else "Run freeze_rr_external_method.py before external scoring.",
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 2,
            "gate": "active_external_fieldwork_rows",
            "status": active_rows_status,
            "evidence": "paper_external_validation_fieldwork_preflight.csv",
            "next_action": "Fill independent external rows in the fieldwork template; blank template rows are not counted.",
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 3,
            "gate": "minimum_holdout_smoke_test",
            "status": smoke_status,
            "evidence": "50 videos, >=5 cows, >=2 dates/sessions, >=2 camera/scene groups, manual RR protocol",
            "next_action": "Reach this tier before claiming external feasibility.",
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 4,
            "gate": "q2_algorithmic_external_validation_fieldwork",
            "status": algorithmic_q2_status,
            "evidence": (
                "104 videos, >=8 cows, >=4 dates/sessions, >=2 camera/scene "
                "groups, and manual RR protocol; excludes heat-stress and "
                "manual-quality stratified claims"
            ),
            "next_action": (
                "Reach this tier for a Q2 algorithmic/engineering submission "
                "route when environment and manual quality metadata are unavailable."
            ),
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 5,
            "gate": "q2_absolute_validation_fieldwork",
            "status": q2_status,
            "evidence": "104 videos plus Q2 metadata and manual quality scores",
            "next_action": "Reach this tier only for heat-stress or manual-quality stratified claims.",
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 6,
            "gate": "metadata_quality_q2_claims",
            "status": "PASS" if metadata_blockers == 0 and not metadata_audit.empty else "FAIL",
            "evidence": f"metadata_q2_blockers={metadata_blockers}",
            "next_action": "Fill real environment and manual quality fields; do not use regional proxy as per-video environment evidence.",
            "method_freeze_id": freeze_id,
        },
        {
            "gate_order": 7,
            "gate": "frozen_external_split_validation",
            "status": external_status,
            "evidence": "paper_repro_external_split_readiness.csv",
            "next_action": "Run rr_external_split_validation.py after predictions and external metadata are present.",
            "method_freeze_id": freeze_id,
        },
    ]
    if not target_gate.empty:
        for _, item in target_gate.iterrows():
            rows.append(
                {
                    "gate_order": 20 + len(rows),
                    "gate": f"target_gate_{item.get('gate')}",
                    "status": item.get("status", "MISSING"),
                    "evidence": item.get("evidence", ""),
                    "next_action": "Use paper_target_journal_strategy.csv for journal-specific gating.",
                    "method_freeze_id": freeze_id,
                }
            )
    return pd.DataFrame(rows)


def build_input_contract() -> pd.DataFrame:
    rows = [
        {
            "input_artifact": "external thermal frame folders",
            "required": True,
            "expected_location_or_format": "<external_input_root>/<external_video_id>/*.jpg|png|bmp",
            "owner_action": "Create one folder per independent external video; avoid generated curve/plot files as input frames.",
            "checked_by": "scripts/paper_repro_rr.py --input-root <external_input_root>",
        },
        {
            "input_artifact": "manual external RR reference CSV",
            "required": True,
            "expected_location_or_format": "CSV with video_id, breath_count, duration_seconds, and rr columns",
            "owner_action": "Record manual breath count for every included external video; rr can be computed from breath_count and duration.",
            "checked_by": "scripts/prepare_rr_external_validation_inputs.py and scripts/paper_repro_rr.py --truth-csv <external_truth_csv>",
        },
        {
            "input_artifact": "manual breath annotation packet",
            "required": "helper",
            "expected_location_or_format": "paper_external_validation_breath_annotation_packet_annotator_a.html and _annotator_b.html plus exported annotation CSVs",
            "owner_action": "Use blinded randomized A/B HTML packets to watch external clips and count breaths before consensus or adjudication.",
            "checked_by": "scripts/audit_rr_external_breath_annotation_agreement.py",
        },
        {
            "input_artifact": "manual annotation progress monitor",
            "required": "helper",
            "expected_location_or_format": "paper_external_validation_annotation_progress_summary.csv and paper_external_validation_annotation_progress.md",
            "owner_action": "Run the progress monitor after each browser export to find missing A/B rows, duplicate video IDs, and incomplete required fields before the agreement audit.",
            "checked_by": "scripts/build_rr_external_annotation_progress_monitor.py",
        },
        {
            "input_artifact": "dual-annotator agreement audit",
            "required": True,
            "expected_location_or_format": "paper_external_validation_breath_annotation_agreement.csv, consensus.csv, and adjudication_template.csv",
            "owner_action": "Resolve all missing or disagreeing breath counts before creating the external truth CSV.",
            "checked_by": "scripts/audit_rr_external_breath_annotation_agreement.py",
        },
        {
            "input_artifact": "after-annotation external validation runner",
            "required": "helper",
            "expected_location_or_format": "scripts/run_rr_external_validation_after_annotation.py plus paper_external_validation_after_annotation_run_log.csv",
            "owner_action": "After exporting A/B annotation CSVs, run the orchestrator to audit, merge, preflight, extract frames, apply the frozen method, score the external split, and refresh assets. It stops safely if consensus is incomplete.",
            "checked_by": "paper_external_validation_after_annotation_run_report.md",
        },
        {
            "input_artifact": "fieldwork metadata worksheet",
            "required": True,
            "expected_location_or_format": "paper_external_validation_fieldwork_template.csv filled with independent external rows",
            "owner_action": "Fill cow_id, date/time, camera/scene, manual RR reference, reference annotator, and include flag. Fill environment and quality scores only for heat-stress or quality-stratified claims.",
            "checked_by": "scripts/preflight_rr_external_fieldwork.py",
        },
        {
            "input_artifact": "synchronized temperature and humidity",
            "required": "claim_dependent",
            "expected_location_or_format": "ambient_temperature_c and relative_humidity_percent per external video or session",
            "owner_action": "Use barn/session measurements; do not fill from regional weather proxy for per-video claims.",
            "checked_by": "scripts/audit_rr_metadata_quality.py and scripts/build_rr_heat_stress_context.py",
        },
        {
            "input_artifact": "manual quality scores",
            "required": "claim_dependent",
            "expected_location_or_format": "0-3 scores for head motion, occlusion, nostril visibility",
            "owner_action": "Score from visual review; leave blank only if avoiding quality-stratified claims.",
            "checked_by": "scripts/preflight_rr_external_fieldwork.py",
        },
        {
            "input_artifact": "method freeze manifest",
            "required": True,
            "expected_location_or_format": "paper_method_freeze_summary.csv with freeze_status=locked",
            "owner_action": "Do not change thresholds or model files after this point without a new freeze ID.",
            "checked_by": "scripts/freeze_rr_external_method.py",
        },
    ]
    return pd.DataFrame(rows)


def build_execution_steps(input_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "step_order": 1,
            "phase": "external_inventory",
            "step": "build_split_all_use_manifest",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "scripts\\build_rr_split_all_use_external_manifest.py "
                "--source-root E:\\real\\use_code\\split_all_use"
            ),
            "can_run_now": True,
            "success_output": "paper_external_validation_split_all_use_fieldwork_template.csv",
            "stop_if_fails": True,
            "paper_use": "external data inventory",
        },
        {
            "step_order": 2,
            "phase": "external_annotation",
            "step": "build_dual_breath_annotation_packets",
            "command_template": (
                "Run scripts\\build_rr_external_breath_annotation_packet.py twice: "
                "--annotator-id annotator_a --export-name paper_external_validation_breath_annotation_annotator_a_export.csv --blind --blind-seed 20260710; "
                "--annotator-id annotator_b --export-name paper_external_validation_breath_annotation_annotator_b_export.csv --blind --blind-seed 20260711"
            ),
            "can_run_now": True,
            "success_output": "paper_external_validation_breath_annotation_packet_annotator_a.html and _annotator_b.html",
            "stop_if_fails": True,
            "paper_use": "independent manual RR annotation interface",
        },
        {
            "step_order": 3,
            "phase": "external_annotation",
            "step": "fill_dual_breath_annotation_packets",
            "command_template": (
                str(output_dir / "paper_external_validation_breath_annotation_packet_annotator_a.html")
                + " and "
                + str(output_dir / "paper_external_validation_breath_annotation_packet_annotator_b.html")
            ),
            "can_run_now": False,
            "success_output": "two exported annotation CSV files from browser export",
            "stop_if_fails": True,
            "paper_use": "independent manual RR reference",
        },
        {
            "step_order": 4,
            "phase": "external_annotation",
            "step": "audit_dual_annotation_agreement",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "scripts\\audit_rr_external_breath_annotation_agreement.py "
                "--annotation-a-csv <annotator_a_export.csv> "
                "--annotation-b-csv <annotator_b_export.csv>; "
                "or run scripts\\run_rr_external_validation_after_annotation.py to orchestrate this and all downstream frozen external-scoring steps"
            ),
            "can_run_now": False,
            "success_output": "paper_external_validation_breath_annotation_consensus.csv and adjudication_template.csv",
            "stop_if_fails": True,
            "paper_use": "manual RR truth quality control",
        },
        {
            "step_order": 5,
            "phase": "external_annotation",
            "step": "adjudicate_disagreements_if_needed",
            "command_template": (
                "Fill paper_external_validation_breath_annotation_adjudication_template.csv, "
                "then rerun audit_rr_external_breath_annotation_agreement.py with --adjudication-csv."
            ),
            "can_run_now": False,
            "success_output": "needs_adjudication_rows=0 and consensus_ready_rows equals included rows",
            "stop_if_fails": True,
            "paper_use": "manual RR truth quality control",
        },
        {
            "step_order": 6,
            "phase": "external_annotation",
            "step": "merge_consensus_annotations_into_fieldwork",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "scripts\\merge_rr_external_breath_annotations.py "
                "--annotations-csv <paper_external_validation_breath_annotation_consensus.csv> "
                "--output-csv <annotated_fieldwork_csv>"
            ),
            "can_run_now": False,
            "success_output": "annotated fieldwork CSV with manual_breath_count, rr, annotator, and camera_id",
            "stop_if_fails": True,
            "paper_use": "external data provenance",
        },
        {
            "step_order": 7,
            "phase": "current_workspace_preflight",
            "step": "preflight_annotated_fieldwork_and_prepare_inputs",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "scripts\\prepare_rr_external_validation_inputs.py "
                "--fieldwork-csv <annotated_fieldwork_csv>"
            ),
            "can_run_now": False,
            "success_output": "paper_external_validation_truth_template.csv and metadata template",
            "stop_if_fails": True,
            "paper_use": "external validation input pack",
        },
        {
            "step_order": 8,
            "phase": "external_rr_prediction",
            "step": "extract_external_clip_frames",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "scripts\\extract_rr_external_clip_frames.py "
                "--fieldwork-csv <annotated_fieldwork_csv> "
                "--external-input-root <external_input_root> --include-only --overwrite"
            ),
            "can_run_now": False,
            "success_output": "<external_input_root>/external_clip_frame_extraction_report.csv",
            "stop_if_fails": True,
            "paper_use": "external frame extraction",
        },
        {
            "step_order": 9,
            "phase": "external_rr_prediction",
            "step": "run_frozen_temperature_curve_pipeline_on_external_frames",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe scripts\\paper_repro_rr.py "
                "--input-root <external_input_root> --truth-csv <external_truth_csv> "
                "--output-prefix paper_repro --overwrite"
            ),
            "can_run_now": False,
            "success_output": "<external_input_root>/paper_repro_summary.csv and paper_repro_metrics.csv",
            "stop_if_fails": True,
            "paper_use": "external prediction generation",
        },
        {
            "step_order": 10,
            "phase": "external_method_application",
            "step": "apply_frozen_primary_and_candidate_methods",
            "command_template": (
                "Run the frozen residual, signal-consensus, safe-gate, algorithmic-quality, "
                "deployment-decision, and conformal scripts on the external prediction root "
                "using the frozen parameters in paper_method_freeze_parameters.csv."
            ),
            "can_run_now": False,
            "success_output": "external-root method prediction CSVs with the frozen method_freeze_id recorded",
            "stop_if_fails": True,
            "paper_use": "method-frozen external scoring",
        },
        {
            "step_order": 11,
            "phase": "external_validation",
            "step": "evaluate_frozen_external_split",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe scripts\\rr_external_split_validation.py "
                "--input-root <combined_or_external_scoring_root> "
                "--metadata-csv <external_or_combined_metadata_csv> "
                "--method-freeze-csv "
                + str(output_dir / "paper_method_freeze_summary.csv")
            ),
            "can_run_now": False,
            "success_output": "paper_repro_external_split_metrics.csv and paper_repro_external_split_acceptance.csv",
            "stop_if_fails": True,
            "paper_use": "external validation result",
        },
        {
            "step_order": 12,
            "phase": "submission_refresh",
            "step": "refresh_readiness_and_target_journal_strategy",
            "command_template": (
                "E:\\real\\anaconda\\envs\\plant_gpu\\python.exe scripts\\refresh_rr_q2_package.py "
                "--python E:\\real\\anaconda\\envs\\plant_gpu\\python.exe "
                "--steps metadata_quality,external_split,readiness,q2_roadmap,target_journal_strategy,assets_final"
            ),
            "can_run_now": True,
            "success_output": "paper_repro_quality_residual_submission_readiness.csv and paper_target_journal_strategy.csv",
            "stop_if_fails": True,
            "paper_use": "final submission gate",
        },
    ]
    return pd.DataFrame(rows)


def build_acceptance_dashboard(output_dir: Path) -> pd.DataFrame:
    acceptance = read_optional(output_dir / "paper_external_validation_acceptance_criteria.csv")
    target_strategy = read_optional(output_dir / "paper_target_journal_strategy.csv")
    rows = []
    if not acceptance.empty:
        for _, item in acceptance.iterrows():
            rows.append(
                {
                    "acceptance_family": item.get("claim_type", ""),
                    "metric_or_gate": item.get("metric", ""),
                    "acceptance_rule": item.get("acceptance_rule", ""),
                    "current_status": "not_evaluable_until_external_metrics_exist",
                    "claim_if_passed": item.get("manuscript_use_if_passed", ""),
                }
            )
    if not target_strategy.empty:
        for _, item in target_strategy.iterrows():
            rows.append(
                {
                    "acceptance_family": "target_journal_route",
                    "metric_or_gate": item.get("target_journal", ""),
                    "acceptance_rule": "all must-have gates pass and current journal quartile/scope manually verified",
                    "current_status": item.get("journal_strategy_status", ""),
                    "claim_if_passed": item.get("lead_claim_if_ready", ""),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows available._"
    view = data[[column for column in columns if column in data.columns]].head(max_rows)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    dashboard: pd.DataFrame,
    contract: pd.DataFrame,
    steps: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> None:
    failed = dashboard[dashboard["status"].astype(str) != "PASS"]
    if failed.empty:
        position = "All execution gates currently pass; proceed to external scoring and final submission refresh."
    else:
        first = failed.sort_values("gate_order").iloc[0]
        position = (
            f"Current execution phase is blocked at `{first['gate']}`. "
            f"Next action: {first['next_action']}"
        )
    text = f"""# External Validation Execution Plan

{position}

This plan is an execution checklist. It is not an external performance result.
It keeps the current 73 videos as internal/development data and requires
independent external rows before external claims.

## Execution Dashboard

{markdown_table(dashboard, ['gate_order', 'gate', 'status', 'evidence', 'next_action', 'method_freeze_id'])}

## Input Contract

{markdown_table(contract, ['input_artifact', 'required', 'expected_location_or_format', 'owner_action', 'checked_by'])}

## Command Sequence

{markdown_table(steps, ['step_order', 'phase', 'step', 'can_run_now', 'command_template', 'success_output', 'paper_use'])}

## Acceptance Dashboard

{markdown_table(acceptance, ['acceptance_family', 'metric_or_gate', 'acceptance_rule', 'current_status', 'claim_if_passed'])}
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard = build_dashboard(output_dir, args.input_root, args.output_prefix)
    contract = build_input_contract()
    steps = build_execution_steps(args.input_root, output_dir)
    acceptance = build_acceptance_dashboard(output_dir)

    dashboard_path = output_dir / "paper_external_validation_execution_dashboard.csv"
    contract_path = output_dir / "paper_external_validation_input_contract.csv"
    steps_path = output_dir / "paper_external_validation_execution_steps.csv"
    acceptance_path = output_dir / "paper_external_validation_acceptance_dashboard.csv"
    report_path = output_dir / "paper_external_validation_execution_plan.md"

    dashboard.to_csv(dashboard_path, index=False)
    contract.to_csv(contract_path, index=False)
    steps.to_csv(steps_path, index=False)
    acceptance.to_csv(acceptance_path, index=False)
    write_report(report_path, dashboard, contract, steps, acceptance)

    print(f"Saved external validation execution dashboard: {dashboard_path}")
    print(f"Saved external validation input contract: {contract_path}")
    print(f"Saved external validation execution steps: {steps_path}")
    print(f"Saved external validation acceptance dashboard: {acceptance_path}")
    print(f"Saved external validation execution report: {report_path}")
    print("\nExecution dashboard:")
    print(dashboard[["gate_order", "gate", "status", "next_action"]].to_string(index=False))


if __name__ == "__main__":
    main()
