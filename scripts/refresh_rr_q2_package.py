from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_STEPS = [
    "safe_policy",
    "algorithmic_quality",
    "post_safe_gate_diagnostics",
    "physiology_sequence_decoder",
    "deployment_decision",
    "conformal_uncertainty",
    "physiological_triage",
    "bilateral_consistency",
    "method_stats",
    "method_agreement",
    "method_agreement_manuscript_insert",
    "assets_pre_freeze",
    "external_validation_sample_plan",
    "error_driven_external_queue",
    "external_fieldwork_preflight",
    "external_input_pack",
    "method_freeze",
    "metadata_context_prefill",
    "external_split",
    "metadata_preflight",
    "metadata_pack",
    "metadata_group_validation",
    "heat_stress",
    "metadata_quality",
    "assets_pre_readiness",
    "readiness",
    "assets_pre_gap",
    "submission_gap",
    "assets_pre_claims",
    "claim_update",
    "consensus_rollback_probe",
    "frame_motion_innovation",
    "q2_roadmap",
    "target_journal_strategy",
    "external_validation_execution_plan",
    "external_annotation_progress_monitor",
    "external_annotation_submission_dashboard",
    "external_validation_reporting_insert",
    "claim_scope_strategy",
    "literature_recent_refresh",
    "q2_reviewer_risk_response",
    "submission_route_pack",
    "algorithmic_manuscript_insert",
    "manuscript_draft_sync",
    "assets_final",
]


SCRIPT_BY_STEP = {
    "safe_policy": "rr_signal_aware_safe_policy.py",
    "algorithmic_quality": "build_rr_algorithmic_quality_context.py",
    "post_safe_gate_diagnostics": "build_rr_post_safe_gate_error_diagnostics.py",
    "physiology_sequence_decoder": "build_rr_physiology_sequence_decoder.py",
    "multi_roi_selector_probe": "build_rr_multi_roi_selector_probe.py",
    "duration_windowed_innovation": "build_rr_duration_normalized_windowed_innovation.py",
    "external_farm_calibration": "build_rr_external_farm_calibration_probe.py",
    "calibration_free_thermal_index": "build_rr_calibration_free_thermal_index.py",
    "deployment_decision": "build_rr_deployment_decision_curve.py",
    "conformal_uncertainty": "build_rr_conformal_uncertainty.py",
    "physiological_triage": "build_rr_physiological_triage.py",
    "bilateral_consistency": "build_rr_bilateral_consistency_gate.py",
    "method_stats": "rr_method_statistical_tests.py",
    "method_agreement": "build_rr_method_agreement_analysis.py",
    "method_agreement_manuscript_insert": "build_rr_method_agreement_manuscript_insert.py",
    "assets_pre_freeze": "build_rr_paper_assets.py",
    "external_validation_sample_plan": "build_rr_external_validation_sample_plan.py",
    "error_driven_external_queue": "build_rr_error_driven_external_validation_queue.py",
    "external_fieldwork_preflight": "preflight_rr_external_fieldwork.py",
    "external_validation_execution_plan": "build_rr_external_validation_execution_plan.py",
    "external_annotation_progress_monitor": "build_rr_external_annotation_progress_monitor.py",
    "external_annotation_submission_dashboard": "build_rr_external_annotation_submission_dashboard.py",
    "external_validation_reporting_insert": "build_rr_external_validation_reporting_insert.py",
    "method_freeze": "freeze_rr_external_method.py",
    "external_input_pack": "prepare_rr_external_validation_inputs.py",
    "metadata_context_prefill": "prefill_rr_metadata_known_context.py",
    "external_split": "rr_external_split_validation.py",
    "metadata_preflight": "preflight_rr_metadata_annotations.py",
    "metadata_pack": "build_rr_metadata_annotation_pack.py",
    "metadata_group_validation": "rr_quality_residual_metadata_group_validation.py",
    "heat_stress": "build_rr_heat_stress_context.py",
    "metadata_quality": "audit_rr_metadata_quality.py",
    "assets_pre_readiness": "build_rr_paper_assets.py",
    "readiness": "audit_rr_submission_readiness.py",
    "assets_pre_gap": "build_rr_paper_assets.py",
    "submission_gap": "build_rr_submission_gap_action_pack.py",
    "assets_pre_claims": "build_rr_paper_assets.py",
    "claim_update": "build_rr_manuscript_claim_update.py",
    "consensus_rollback_probe": "build_rr_consensus_rollback_probe.py",
    "q2_roadmap": "build_rr_q2_plus_innovation_roadmap.py",
    "target_journal_strategy": "build_rr_target_journal_strategy.py",
    "claim_scope_strategy": "build_rr_claim_scope_strategy.py",
    "literature_recent_refresh": "build_rr_literature_recent_refresh.py",
    "q2_reviewer_risk_response": "build_rr_q2_reviewer_risk_response.py",
    "submission_route_pack": "build_rr_submission_route_pack.py",
    "frame_motion_innovation": "build_rr_frame_motion_innovation.py",
    "algorithmic_manuscript_insert": "build_rr_algorithmic_engineering_manuscript_insert.py",
    "manuscript_draft_sync": "sync_rr_manuscript_draft.py",
    "assets_final": "build_rr_paper_assets.py",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the thermal RR Q2+ package in dependency order. This is an "
            "orchestrator only; each step delegates to the existing source script."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--steps",
        default=",".join(DEFAULT_STEPS),
        help=(
            "Comma-separated step list. Available: "
            + ", ".join(DEFAULT_STEPS)
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def output_dir(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets"


def command_for_step(args: argparse.Namespace, step: str) -> list[str]:
    script = Path(__file__).resolve().parent / SCRIPT_BY_STEP[step]
    if step in {
        "duration_windowed_innovation",
        "external_farm_calibration",
        "calibration_free_thermal_index",
    }:
        return [str(args.python), str(script)]
    command = [
        str(args.python),
        str(script),
        "--input-root",
        str(args.input_root),
    ]
    if step not in {"safe_policy", "heat_stress", "metadata_context_prefill"}:
        command.extend(["--output-prefix", args.output_prefix])
    if step not in {
        "safe_policy",
        "algorithmic_quality",
        "conformal_uncertainty",
        "method_stats",
        "q2_roadmap",
    }:
        command.extend(["--corrected-prefix", args.corrected_prefix])
    if step == "metadata_context_prefill":
        command.extend(
            [
                "--write-fill-template",
                "--write-metadata-template",
                "--no-weather-fetch",
            ]
        )
    return command


def write_log(log_path: Path, rows: list[dict[str, object]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "step_order",
        "step",
        "script",
        "status",
        "return_code",
        "duration_seconds",
        "started_at",
        "finished_at",
        "command",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report_path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Q2 Package Refresh Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Input root: `{args.input_root}`",
        "",
        f"Output dir: `{output_dir(args.input_root, args.corrected_prefix)}`",
        "",
        f"Overall status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| order | step | status | seconds | script |",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {order} | {step} | {status} | {seconds:.2f} | `{script}` |".format(
                order=row["step_order"],
                step=row["step"],
                status=row["status"],
                seconds=float(row["duration_seconds"]),
                script=row["script"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS means the package was refreshed in dependency order. It does not mean "
            "Q2+ submission readiness is achieved; use "
            "`paper_repro_quality_residual_submission_readiness.csv` for that gate.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.python = args.python.resolve()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in SCRIPT_BY_STEP]
    if unknown:
        raise ValueError(f"Unknown refresh steps: {unknown}")

    out_dir = output_dir(args.input_root, args.corrected_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "paper_q2_package_refresh_log.csv"
    report_path = out_dir / "paper_q2_package_refresh_report.md"

    rows: list[dict[str, object]] = []
    for index, step in enumerate(steps, start=1):
        command = command_for_step(args, step)
        started_at = datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        status = "DRY_RUN"
        return_code = 0
        print(f"[{index}/{len(steps)}] {step}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
            return_code = int(completed.returncode)
            status = "PASS" if return_code == 0 else "FAIL"
        finished_at = datetime.now().isoformat(timespec="seconds")
        duration = time.perf_counter() - start_time
        rows.append(
            {
                "step_order": index,
                "step": step,
                "script": SCRIPT_BY_STEP[step],
                "status": status,
                "return_code": return_code,
                "duration_seconds": f"{duration:.3f}",
                "started_at": started_at,
                "finished_at": finished_at,
                "command": " ".join(command),
            }
        )
        write_log(log_path, rows)
        write_report(report_path, rows, args)
        if return_code != 0 and not args.continue_on_error:
            raise SystemExit(return_code)

    print(f"Saved refresh log: {log_path}")
    print(f"Saved refresh report: {report_path}")


if __name__ == "__main__":
    main()
