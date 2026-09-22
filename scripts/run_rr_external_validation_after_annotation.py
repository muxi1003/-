from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_COLUMNS = [
    "step_order",
    "step",
    "status",
    "return_code",
    "duration_seconds",
    "started_at",
    "finished_at",
    "command",
    "output_artifact",
    "message",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_external_root = repo_root / "Dataset_new" / "72video" / "external_al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Run the method-frozen external RR validation steps after blinded A/B "
            "manual breath-count annotation exports are available."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--external-input-root", type=Path, default=default_external_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--external-output-prefix", default="external_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--annotation-a-csv", type=Path, default=None)
    parser.add_argument("--annotation-b-csv", type=Path, default=None)
    parser.add_argument("--adjudication-csv", type=Path, default=None)
    parser.add_argument("--annotated-fieldwork-csv", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--skip-frame-extraction", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-postprocessors", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when annotation exports or consensus are incomplete.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_path(output_dir: Path, name: str) -> Path:
    return output_dir / name


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def read_metric(summary: pd.DataFrame, metric: str) -> str:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return ""
    match = summary[summary["metric"].astype(str) == metric]
    if match.empty:
        return ""
    return normalize_text(match.iloc[0].get("value", ""))


def numeric_metric(summary: pd.DataFrame, metric: str) -> float:
    value = pd.to_numeric(pd.Series([read_metric(summary, metric)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def run_command(
    rows: list[dict[str, object]],
    step_order: int,
    step: str,
    command: list[str],
    output_artifact: Path | str,
    *,
    cwd: Path,
    dry_run: bool,
) -> bool:
    started = now_text()
    start_time = time.monotonic()
    if dry_run:
        rows.append(
            {
                "step_order": step_order,
                "step": step,
                "status": "DRY_RUN",
                "return_code": "",
                "duration_seconds": "0.000",
                "started_at": started,
                "finished_at": started,
                "command": command_text(command),
                "output_artifact": str(output_artifact),
                "message": "Command not executed.",
            }
        )
        return True
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    finished = now_text()
    duration = time.monotonic() - start_time
    stdout_tail = "\n".join(completed.stdout.splitlines()[-8:])
    stderr_tail = "\n".join(completed.stderr.splitlines()[-8:])
    message = " ".join(part for part in [stdout_tail, stderr_tail] if part).strip()
    rows.append(
        {
            "step_order": step_order,
            "step": step,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "return_code": completed.returncode,
            "duration_seconds": f"{duration:.3f}",
            "started_at": started,
            "finished_at": finished,
            "command": command_text(command),
            "output_artifact": str(output_artifact),
            "message": message,
        }
    )
    return completed.returncode == 0


def append_blocked(
    rows: list[dict[str, object]],
    step_order: int,
    step: str,
    output_artifact: Path | str,
    message: str,
) -> None:
    now = now_text()
    rows.append(
        {
            "step_order": step_order,
            "step": step,
            "status": "BLOCKED",
            "return_code": "",
            "duration_seconds": "0.000",
            "started_at": now,
            "finished_at": now,
            "command": "",
            "output_artifact": str(output_artifact),
            "message": message,
        }
    )


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    overall_status: str,
    next_action: str,
    agreement_summary: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(rows, columns=LOG_COLUMNS)
    log_path = output_dir / "paper_external_validation_after_annotation_run_log.csv"
    report_path = output_dir / "paper_external_validation_after_annotation_run_report.md"
    log.to_csv(log_path, index=False)
    agreement_lines: list[str] = []
    if agreement_summary is not None and not agreement_summary.empty:
        for metric in [
            "included_external_rows",
            "complete_dual_annotation_rows",
            "exact_agreement_rows",
            "consensus_ready_rows",
            "needs_adjudication_rows",
        ]:
            agreement_lines.append(f"- {metric}: `{read_metric(agreement_summary, metric)}`")
    else:
        agreement_lines.append("- agreement summary: `not generated in this run`")
    table = log[["step_order", "step", "status", "output_artifact", "message"]].fillna("")
    report_lines = [
        "# External Validation After-Annotation Runner",
        "",
        f"Status: `{overall_status}`",
        "",
        f"Next action: {next_action}",
        "",
        "## Agreement Snapshot",
        "",
        *agreement_lines,
        "",
        "## Step Log",
        "",
        "| step_order | step | status | output_artifact | message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        values = [
            str(row.get("step_order", "")),
            str(row.get("step", "")),
            str(row.get("status", "")),
            str(row.get("output_artifact", "")),
            str(row.get("message", "")).replace("\n", " ")[:500],
        ]
        report_lines.append("| " + " | ".join(values) + " |")
    report_lines.extend(["", f"CSV log: `{log_path}`", ""])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved after-annotation run log: {log_path}")
    print(f"Saved after-annotation run report: {report_path}")
    print(f"Status: {overall_status}")
    print(f"Next action: {next_action}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.input_root = args.input_root.resolve()
    args.external_input_root = args.external_input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldwork_csv = (
        args.fieldwork_csv.resolve()
        if args.fieldwork_csv is not None
        else default_path(output_dir, "paper_external_validation_split_all_use_fieldwork_template.csv")
    )
    annotation_a = (
        args.annotation_a_csv.resolve()
        if args.annotation_a_csv is not None
        else default_path(output_dir, "paper_external_validation_breath_annotation_annotator_a_export.csv")
    )
    annotation_b = (
        args.annotation_b_csv.resolve()
        if args.annotation_b_csv is not None
        else default_path(output_dir, "paper_external_validation_breath_annotation_annotator_b_export.csv")
    )
    adjudication = (
        args.adjudication_csv.resolve()
        if args.adjudication_csv is not None
        else default_path(output_dir, "paper_external_validation_breath_annotation_adjudication_template.csv")
    )
    annotated_fieldwork = (
        args.annotated_fieldwork_csv.resolve()
        if args.annotated_fieldwork_csv is not None
        else default_path(output_dir, "paper_external_validation_split_all_use_fieldwork_consensus.csv")
    )
    consensus_csv = default_path(output_dir, "paper_external_validation_breath_annotation_consensus.csv")
    truth_csv = default_path(output_dir, "paper_external_validation_truth_template.csv")
    metadata_csv = default_path(output_dir, "paper_external_validation_metadata_template.csv")
    method_freeze_csv = default_path(output_dir, "paper_method_freeze_summary.csv")

    rows: list[dict[str, object]] = []
    missing = [path for path in [fieldwork_csv, annotation_a, annotation_b] if not path.exists()]
    if missing:
        append_blocked(
            rows,
            1,
            "check_required_annotation_exports",
            output_dir,
            "Missing required file(s): " + "; ".join(str(path) for path in missing),
        )
        write_outputs(
            output_dir,
            rows,
            "blocked_missing_annotation_exports",
            "Open the blinded A/B HTML packets, export both annotation CSV files, then rerun this runner.",
        )
        return 2 if args.fail_on_blocked else 0

    audit_command = [
        str(args.python),
        str(repo_root / "scripts" / "audit_rr_external_breath_annotation_agreement.py"),
        "--input-root",
        str(args.input_root),
        "--corrected-prefix",
        args.corrected_prefix,
        "--fieldwork-csv",
        str(fieldwork_csv),
        "--annotation-a-csv",
        str(annotation_a),
        "--annotation-b-csv",
        str(annotation_b),
    ]
    if adjudication.exists():
        audit_command.extend(["--adjudication-csv", str(adjudication)])
    if not run_command(
        rows,
        1,
        "audit_dual_annotation_agreement",
        audit_command,
        consensus_csv,
        cwd=repo_root,
        dry_run=args.dry_run,
    ):
        write_outputs(
            output_dir,
            rows,
            "failed_agreement_audit",
            "Fix the A/B annotation CSV format and rerun the agreement audit.",
        )
        return 1
    if args.dry_run:
        write_outputs(output_dir, rows, "dry_run", "Review the dry-run command list before executing.")
        return 0

    agreement_summary_path = output_dir / "paper_external_validation_breath_annotation_agreement_summary.csv"
    agreement = pd.read_csv(agreement_summary_path, dtype=str, keep_default_na=False)
    included = numeric_metric(agreement, "included_external_rows")
    consensus_ready = numeric_metric(agreement, "consensus_ready_rows")
    needs_adjudication = numeric_metric(agreement, "needs_adjudication_rows")
    if not (included > 0 and consensus_ready >= included and needs_adjudication == 0):
        append_blocked(
            rows,
            2,
            "check_consensus_ready",
            agreement_summary_path,
            (
                f"Consensus is incomplete: included={included:g}, "
                f"consensus_ready={consensus_ready:g}, needs_adjudication={needs_adjudication:g}."
            ),
        )
        write_outputs(
            output_dir,
            rows,
            "blocked_needs_annotation_consensus",
            "Complete both A/B exports and fill adjudication rows until consensus_ready_rows equals included_external_rows.",
            agreement,
        )
        return 2 if args.fail_on_blocked else 0

    steps: list[tuple[str, list[str], Path | str]] = [
        (
            "merge_consensus_annotations",
            [
                str(args.python),
                str(repo_root / "scripts" / "merge_rr_external_breath_annotations.py"),
                "--input-root",
                str(args.input_root),
                "--corrected-prefix",
                args.corrected_prefix,
                "--fieldwork-csv",
                str(fieldwork_csv),
                "--annotations-csv",
                str(consensus_csv),
                "--output-csv",
                str(annotated_fieldwork),
            ],
            annotated_fieldwork,
        ),
        (
            "regenerate_external_input_pack",
            [
                str(args.python),
                str(repo_root / "scripts" / "prepare_rr_external_validation_inputs.py"),
                "--input-root",
                str(args.input_root),
                "--corrected-prefix",
                args.corrected_prefix,
                "--fieldwork-csv",
                str(annotated_fieldwork),
                "--external-input-root",
                str(args.external_input_root),
                "--external-output-prefix",
                args.external_output_prefix,
            ],
            truth_csv,
        ),
        (
            "preflight_consensus_fieldwork",
            [
                str(args.python),
                str(repo_root / "scripts" / "preflight_rr_external_fieldwork.py"),
                "--input-root",
                str(args.input_root),
                "--output-prefix",
                args.output_prefix,
                "--corrected-prefix",
                args.corrected_prefix,
                "--fieldwork-csv",
                str(annotated_fieldwork),
            ],
            default_path(output_dir, "paper_external_validation_fieldwork_preflight.csv"),
        ),
    ]
    if not args.skip_frame_extraction:
        extract_command = [
            str(args.python),
            str(repo_root / "scripts" / "extract_rr_external_clip_frames.py"),
            "--fieldwork-csv",
            str(annotated_fieldwork),
            "--external-input-root",
            str(args.external_input_root),
            "--include-only",
        ]
        if args.overwrite_frames:
            extract_command.append("--overwrite")
        steps.append(
            (
                "extract_external_clip_frames",
                extract_command,
                args.external_input_root / "external_clip_frame_extraction_report.csv",
            )
        )
    if not args.skip_prediction:
        steps.append(
            (
                "run_frozen_external_rr",
                [
                    str(args.python),
                    str(repo_root / "scripts" / "paper_repro_rr.py"),
                    "--input-root",
                    str(args.external_input_root),
                    "--truth-csv",
                    str(truth_csv),
                    "--output-prefix",
                    args.external_output_prefix,
                    "--overwrite",
                ],
                args.external_input_root / f"{args.external_output_prefix}_summary.csv",
            )
        )
    if not args.skip_postprocessors:
        steps.append(
            (
                "apply_frozen_external_postprocessors",
                [
                    str(args.python),
                    str(repo_root / "scripts" / "apply_rr_frozen_external_postprocessors.py"),
                    "--input-root",
                    str(args.input_root),
                    "--external-input-root",
                    str(args.external_input_root),
                    "--output-prefix",
                    args.output_prefix,
                    "--external-output-prefix",
                    args.external_output_prefix,
                    "--corrected-prefix",
                    args.corrected_prefix,
                ],
                args.external_input_root
                / f"{args.external_output_prefix}_signal_aware_safe_policy_predictions.csv",
            )
        )
    if not args.skip_scoring:
        steps.append(
            (
                "score_external_split",
                [
                    str(args.python),
                    str(repo_root / "scripts" / "rr_external_split_validation.py"),
                    "--input-root",
                    str(args.external_input_root),
                    "--metadata-csv",
                    str(metadata_csv),
                    "--output-prefix",
                    args.external_output_prefix,
                    "--corrected-prefix",
                    args.corrected_prefix,
                    "--method-freeze-csv",
                    str(method_freeze_csv),
                ],
                args.external_input_root / f"{args.external_output_prefix}_external_split_metrics.csv",
            )
        )
    if not args.skip_refresh:
        steps.append(
            (
                "refresh_submission_assets",
                [
                    str(args.python),
                    str(repo_root / "scripts" / "refresh_rr_q2_package.py"),
                    "--input-root",
                    str(args.input_root),
                    "--output-prefix",
                    args.output_prefix,
                    "--corrected-prefix",
                    args.corrected_prefix,
                    "--steps",
                    "external_split,readiness,submission_gap,submission_route_pack,"
                    "external_annotation_submission_dashboard,q2_roadmap,"
                    "target_journal_strategy,external_validation_reporting_insert,"
                    "manuscript_draft_sync,assets_final",
                    "--python",
                    str(args.python),
                ],
                default_path(output_dir, "paper_assets_summary.md"),
            )
        )

    for offset, (step, command, artifact) in enumerate(steps, start=2):
        ok = run_command(
            rows,
            offset,
            step,
            command,
            artifact,
            cwd=repo_root,
            dry_run=False,
        )
        if not ok:
            write_outputs(
                output_dir,
                rows,
                f"failed_{step}",
                f"Fix the failure in step `{step}`, then rerun the after-annotation runner.",
                agreement,
            )
            return 1

    write_outputs(
        output_dir,
        rows,
        "complete_external_scoring_refresh",
        "Review external split metrics, acceptance outputs, and submission readiness before writing paper claims.",
        agreement,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
