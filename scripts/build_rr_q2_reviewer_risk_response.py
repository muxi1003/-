from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a Q2+ reviewer-risk response matrix from the current thermal "
            "RR evidence package, literature refresh, and readiness blockers."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame()


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def contains_row(table: pd.DataFrame, column: str, text: str) -> pd.Series:
    if table.empty or column not in table.columns:
        return pd.Series(dtype=object)
    mask = table[column].astype(str).str.contains(text, case=False, regex=False)
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def method_text(row: pd.Series) -> str:
    if row.empty:
        return "not generated"
    return (
        f"RR R2={fmt(row.get('rr_r2'))}, "
        f"MAE={fmt(row.get('rr_mae_bpm', row.get('rr_mae')), 3)} bpm, "
        f"RMSE={fmt(row.get('rr_rmse_bpm', row.get('rr_rmse')), 3)} bpm, "
        f"exact={row.get('exact_count', 'NA')}"
    )


def readiness_blockers(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_submission_readiness_table.csv")
    if table.empty:
        return "readiness table not generated"
    status = str(table.get("q2_submission_status", pd.Series(["unknown"])).iloc[0])
    mask = (
        table.get("q2_blocking", pd.Series("", index=table.index)).astype(str).str.lower().eq("true")
        & table.get("status", pd.Series("", index=table.index)).astype(str).isin(["FAIL", "WARN"])
    )
    blockers = "; ".join(table.loc[mask, "check"].astype(str).head(8).tolist())
    if int(mask.sum()) > 8:
        blockers += f"; +{int(mask.sum()) - 8} more"
    return f"Q2 status={status}; blockers={blockers or 'none'}"


def metadata_snapshot(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_metadata_annotation_progress.csv")
    if table.empty:
        return "metadata progress not generated"
    fields = [
        "cow_id",
        "collection_start_date",
        "collection_end_date",
        "collection_location_country",
        "collection_location_province",
        "collection_location_county",
        "collection_site",
        "collection_date",
        "camera_id",
        "external_test_split",
        "ambient_temperature_c",
        "relative_humidity_percent",
        "thi",
        "head_motion_score_0_3",
        "occlusion_score_0_3",
        "nostril_visibility_score_0_3",
    ]
    rows = []
    for field in fields:
        match = table[table["field"].astype(str).eq(field)]
        if match.empty:
            rows.append(f"{field}=not generated")
        else:
            item = match.iloc[0]
            rows.append(
                f"{field}={int(float(item.get('nonempty', 0)))}/"
                f"{int(float(item.get('total_videos', 0)))}"
            )
    return "; ".join(rows)


def unique_nonempty(table: pd.DataFrame, column: str, limit: int = 3) -> str:
    if table.empty or column not in table.columns:
        return "not generated"
    values = [
        str(value).strip()
        for value in table[column].dropna().astype(str).tolist()
        if str(value).strip()
    ]
    if not values:
        return "blank"
    unique = list(dict.fromkeys(values))
    if len(unique) > limit:
        return ", ".join(unique[:limit]) + f", +{len(unique) - limit} more"
    return ", ".join(unique)


def metadata_context(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_metadata_template.csv")
    if table.empty:
        return "metadata template not generated"
    location = "/".join(
        [
            unique_nonempty(table, "collection_location_country"),
            unique_nonempty(table, "collection_location_province"),
            unique_nonempty(table, "collection_location_county"),
            unique_nonempty(table, "collection_site"),
        ]
    )
    return (
        "cow_id convention=numeric video label; "
        f"external_test_split={unique_nonempty(table, 'external_test_split')}; "
        f"collection_window={unique_nonempty(table, 'collection_start_date')} to "
        f"{unique_nonempty(table, 'collection_end_date')}; "
        f"location={location}; "
        f"scene_id={unique_nonempty(table, 'scene_id')}"
    )


def external_workflow_evidence(output_dir: Path) -> str:
    freeze = read_csv(output_dir / "paper_method_freeze_summary.csv")
    commands = read_csv(output_dir / "paper_external_validation_run_commands.csv")
    if freeze.empty:
        freeze_text = "method freeze not generated"
    else:
        item = freeze.iloc[0]
        freeze_id = str(item.get("method_freeze_id", ""))
        freeze_id = freeze_id[:12] + "..." if len(freeze_id) > 12 else freeze_id
        freeze_text = (
            f"freeze_status={item.get('freeze_status', 'unknown')}; "
            f"method_freeze_id={freeze_id}; "
            f"required_files={item.get('required_files_present', 'NA')}/"
            f"{item.get('required_files', 'NA')}"
        )
    if commands.empty:
        commands_text = "external run commands not generated"
    else:
        steps = commands["step"].astype(str).tolist() if "step" in commands.columns else []
        commands_text = (
            f"external workflow steps={len(commands)}; "
            f"apply_frozen_postprocessors={'yes' if 'apply_frozen_postprocessors' in steps else 'no'}; "
            f"score_external_split={'yes' if 'score_external_split' in steps else 'no'}"
        )
    return f"{freeze_text}; {commands_text}"


def source_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_literature_recent_source_verification.csv")
    if table.empty:
        return "recent literature source verification not generated"
    verified = int(table["crossref_status"].astype(str).eq("verified").sum())
    return f"recent source verification rows={len(table)}, Crossref verified={verified}"


def bilateral_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_rr_bilateral_consistency_summary.csv")
    if table.empty:
        return "bilateral consistency gate not generated"
    row = contains_row(table, "subset", "bilateral_consistent_auto_report_subset")
    if row.empty:
        return "bilateral consistency gate generated but auto-report row missing"
    return (
        f"bilateral auto-report n={row.get('videos', 'NA')}, "
        f"coverage={fmt(row.get('coverage'), 3)}, "
        f"RR R2={fmt(row.get('rr_r2'))}, "
        f"MAE={fmt(row.get('rr_mae_bpm'), 3)} bpm, "
        f"exact_rate={fmt(row.get('exact_rate'), 3)}"
    )


def build_rows(output_dir: Path) -> pd.DataFrame:
    main = read_csv(output_dir / "paper_main_results_table.csv")
    tests = read_csv(output_dir / "paper_rr_method_statistical_tests_table.csv")
    default = contains_row(main, "method", "Default thermal RR pipeline")
    safe = contains_row(main, "method", "Conservative signal-aware safe gate (fixed")
    prefix_safe = contains_row(main, "method", "Conservative signal-aware safe gate (leave-one-prefix")
    truth = contains_row(main, "method", "Truth-calibrated upper bound")
    safe_test = contains_row(tests, "method_id", "signal_aware_safe_fixed_oof")
    delta = "not generated"
    if not safe_test.empty:
        delta = (
            f"delta RR R2={fmt(safe_test.get('delta_rr_r2'))}, "
            f"CI={fmt(safe_test.get('delta_rr_r2_ci_low'))} to "
            f"{fmt(safe_test.get('delta_rr_r2_ci_high'))}, "
            f"delta exact={safe_test.get('delta_exact_count', 'NA')}"
        )

    rows = [
        {
            "risk_id": "R1",
            "reviewer_risk": "The innovation may look incremental because thermal nostril RR and YOLO-style detection already exist.",
            "risk_level": "high",
            "current_evidence": f"{source_evidence(output_dir)}; default={method_text(default)}; safe_gate={method_text(safe)}",
            "response_strategy": "Frame novelty after signal extraction: residual breath-count error taxonomy, quality-aware residual correction, signal consensus, conservative safe gate, and reliability triage.",
            "manuscript_location": "Introduction; Discussion; Table 1; Figure 1",
            "required_action_before_q2": "Keep the literature-refresh table in supplementary material and use the P0 gap wording instead of claiming novelty in localization alone.",
            "claim_boundary": "Do not claim that basic thermal mapping or YOLO localization is novel.",
        },
        {
            "risk_id": "R2",
            "reviewer_risk": "The sample size is small and internal validation may overstate the gain.",
            "risk_level": "high",
            "current_evidence": f"n=73; safe gate={method_text(safe)}; prefix stress={method_text(prefix_safe)}; {delta}",
            "response_strategy": "Report default, safe gate, bootstrap CI, McNemar/exact agreement, nested/group stress tests, and classify the evidence as internal precision improvement.",
            "manuscript_location": "Abstract; Results 3.4; Discussion; Limitations",
            "required_action_before_q2": "Add method-frozen external videos if making strong external or relative-improvement claims.",
            "claim_boundary": "Internal improvement is reportable; broad generalization is not.",
        },
        {
            "risk_id": "R3",
            "reviewer_risk": "There is no true external or holdout set.",
            "risk_level": "critical",
            "current_evidence": f"{readiness_blockers(output_dir)}; {external_workflow_evidence(output_dir)}",
            "response_strategy": "State that all current videos are internal and present external validation as a frozen protocol with frozen postprocessing, acceptance gates, and sample targets.",
            "manuscript_location": "Manuscript Status; Methods 2.9; Results 3.9; Limitations",
            "required_action_before_q2": "Collect or import independent external/holdout videos after method freeze, run the generated external workflow, and label them external/holdout/test.",
            "claim_boundary": "No external performance claim from the current 73 videos.",
        },
        {
            "risk_id": "R4",
            "reviewer_risk": "Heat-stress or welfare interpretation is under-supported without synchronized environment and animal-context metadata.",
            "risk_level": "high",
            "current_evidence": f"{metadata_context(output_dir)}; {metadata_snapshot(output_dir)}",
            "response_strategy": "Use Lindian County and collection-window provenance only as dataset context; keep THI/ATHI and heat-stress associations as future or conditional analyses.",
            "manuscript_location": "Introduction; Methods 2.8; Results 3.8; Discussion; Limitations",
            "required_action_before_q2": "Add per-video or session-level barn temperature, humidity, THI/ATHI, posture, and collection date/time if targeting dairy-science heat-stress claims.",
            "claim_boundary": "Do not report THI-stratified performance or heat-stress monitoring from provenance alone.",
        },
        {
            "risk_id": "R5",
            "reviewer_risk": "Head movement, occlusion, and nostril visibility are not manually scored.",
            "risk_level": "high",
            "current_evidence": metadata_snapshot(output_dir),
            "response_strategy": "Distinguish algorithmic risk scores from manual quality labels; use them for triage, not for manual-quality robustness claims.",
            "manuscript_location": "Methods 2.6; Results 3.5; Limitations",
            "required_action_before_q2": "Either annotate 0-3 head-motion, occlusion, and nostril-visibility scores or build and validate automated quality labels against manual review.",
            "claim_boundary": "Algorithmic quality risk is not a manual motion/occlusion label.",
        },
        {
            "risk_id": "R6",
            "reviewer_risk": "Truth-calibrated results could be mistaken for deployable performance or leakage.",
            "risk_level": "critical",
            "current_evidence": f"truth-calibrated upper bound={method_text(truth)}",
            "response_strategy": "Keep truth-calibrated RR R2 in an oracle/upper-bound supplement only; exclude it from the abstract's main performance sentence.",
            "manuscript_location": "Methods 2.2; Limitations; Supplementary table",
            "required_action_before_q2": "Maintain separate output files and table labels for default, method-frozen, and truth-calibrated branches.",
            "claim_boundary": "Truth-assisted calibration is diagnostic only.",
        },
        {
            "risk_id": "R7",
            "reviewer_risk": "Selective reporting may inflate accuracy by excluding hard videos.",
            "risk_level": "medium",
            "current_evidence": (
                "algorithmic triage reports coverage, automatic subset size, "
                f"manual-review load, conformal interval coverage, and {bilateral_evidence(output_dir)}"
            ),
            "response_strategy": "Always pair selective accuracy with coverage and review-load metrics; describe it as triage rather than replacement for whole-dataset performance.",
            "manuscript_location": "Results 3.5; Supplementary table; Discussion",
            "required_action_before_q2": "Freeze triage and bilateral-consistency thresholds and test them on external videos before prospective reliability claims.",
            "claim_boundary": "Selective accuracy is conditional on coverage.",
        },
        {
            "risk_id": "R8",
            "reviewer_risk": "Cow identity and grouped validation may be weak because cow_id is derived from video labels.",
            "risk_level": "medium",
            "current_evidence": f"{metadata_context(output_dir)}; {metadata_snapshot(output_dir)}",
            "response_strategy": "State the current cow_id convention explicitly and avoid animal-level generalization unless labels are confirmed as real cows.",
            "manuscript_location": "Methods 2.1; Methods 2.7; Limitations",
            "required_action_before_q2": "Confirm cow IDs, camera IDs, and date/session labels, then rerun GroupKFold and external validation.",
            "claim_boundary": "Numeric video-label cow_id is a grouping aid, not proven identity metadata.",
        },
        {
            "risk_id": "R9",
            "reviewer_risk": "Bilateral nostril consistency could be overinterpreted as physiological diagnosis or external reliability.",
            "risk_level": "medium",
            "current_evidence": bilateral_evidence(output_dir),
            "response_strategy": (
                "Present bilateral consistency as an internal selective-reporting "
                "screen that checks left/right periodicity agreement before "
                "auto-reporting; keep whole-dataset safe-gate metrics as the main "
                "performance row."
            ),
            "manuscript_location": "Methods; Results selective-reporting subsection; Limitations",
            "required_action_before_q2": (
                "Keep the bilateral rule in the method-freeze manifest and validate "
                "it on independent external videos with coverage and manual-review "
                "load reported."
            ),
            "claim_boundary": (
                "Bilateral consistency is not heat-stress diagnosis, welfare "
                "classification, disease detection, or true animal identity evidence."
            ),
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("\n", " ") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_report(path: Path, rows: pd.DataFrame) -> None:
    high = rows[rows["risk_level"].isin(["high", "critical"])]
    text = "\n".join(
        [
            "# Q2+ Reviewer-Risk Response Matrix",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "This matrix converts the current evidence package into likely reviewer objections, defensible responses, and actions required before strong Q2+ claims.",
            "",
            f"High or critical risks: {len(high)}/{len(rows)}",
            "",
            markdown_table(
                rows,
                [
                    "risk_id",
                    "reviewer_risk",
                    "risk_level",
                    "current_evidence",
                    "response_strategy",
                    "required_action_before_q2",
                    "claim_boundary",
                ],
            ),
            "",
            "## Use In Manuscript",
            "",
            "Use this file as a reviewer-preemption checklist. It should not be copied verbatim into the manuscript, but every high or critical risk should be addressed in the Introduction, Methods, Results, Discussion, or Limitations before submission.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(output_dir)
    csv_path = output_dir / "paper_literature_q2_reviewer_risk_response.csv"
    md_path = output_dir / "paper_literature_q2_reviewer_risk_response.md"
    rows.to_csv(csv_path, index=False)
    write_report(md_path, rows)
    print(f"Saved reviewer-risk matrix: {csv_path}")
    print(f"Saved reviewer-risk report: {md_path}")
    print(rows[["risk_id", "risk_level", "reviewer_risk"]].to_string(index=False))


if __name__ == "__main__":
    main()
