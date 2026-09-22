from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


JOURNAL_TARGETS = [
    {
        "target_journal": "Computers and Electronics in Agriculture",
        "route": "calibration_free_cross_farm_engineering_route",
        "scope_fit": "computer vision, sensors, algorithmic quality control, uncertainty, and deployment triage in agriculture",
        "lead_claim_if_ready": (
            "internally selected duration-gated calibration-free relative thermal-color "
            "RR estimation with blinded cross-farm validation and explicit negative "
            "robustness ablations"
        ),
        "must_have": [
            "p2g_method_freeze_ready",
            "p2g_blinded_consensus_scored",
            "p2g_q2_external_sample_target",
            "algorithmic_quality_ready",
        ],
        "nice_to_have": [
            "conformal_uncertainty_ready",
            "error_driven_external_design_ready",
            "p2g_complete_annotation_packets_ready",
        ],
        "claim_exclusions": (
            "No heat-stress, THI-stratified, manual head-motion, occlusion, or "
            "nostril-visibility stratified claims unless those real metadata "
            "fields are later collected."
        ),
        "current_quartile_note": "High-selectivity engineering target; verify current JCR/SJR quartile before submission.",
        "source_basis": (
            "The official scope emphasizes investigator-developed innovation and rigorous "
            "data collection; the P2g method freeze, negative ablations, and fully blinded "
            "cross-farm scoring are therefore required before submission."
        ),
        "source_urls": "https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture",
    },
    {
        "target_journal": "Smart Agricultural Technology",
        "route": "applied_cross_farm_rr_deployment_route",
        "scope_fit": "practical smart-agriculture application, imaging, sensing, and livestock welfare",
        "lead_claim_if_ready": (
            "frozen calibration-free thermal-color RR monitoring with blinded 94-clip "
            "cross-farm consensus validation and review routing"
        ),
        "must_have": [
            "p2g_method_freeze_ready",
            "p2g_blinded_consensus_scored",
            "algorithmic_quality_ready",
        ],
        "nice_to_have": [
            "error_driven_external_design_ready",
            "p2g_q2_external_sample_target",
            "conformal_uncertainty_ready",
        ],
        "claim_exclusions": (
            "Avoid heat-stress and manual-quality stratified claims unless real "
            "environment and reviewer-score metadata become available."
        ),
        "current_quartile_note": "Applied fallback; verify current JCR/SJR quartile and APC before submission.",
        "source_basis": (
            "The official scope explicitly accepts practical smart-technology applications "
            "in livestock farming, making it a realistic route after blinded P2g validation."
        ),
        "source_urls": "https://www.sciencedirect.com/journal/smart-agricultural-technology",
    },
    {
        "target_journal": "Biosystems Engineering",
        "route": "engineering_validation_route",
        "scope_fit": "biosystems sensors, farm engineering, validation under deployment conditions",
        "lead_claim_if_ready": (
            "method-frozen thermal RR pipeline with external validation, robustness "
            "checks, and fieldwork-ready metadata"
        ),
        "must_have": [
            "p2g_method_freeze_ready",
            "p2g_blinded_consensus_scored",
            "p2g_q2_external_sample_target",
            "metadata_quality_ready",
        ],
        "nice_to_have": [
            "conformal_uncertainty_ready",
            "error_driven_external_design_ready",
        ],
        "claim_exclusions": (
            "Avoid heat-stress and manual-quality stratified claims unless real "
            "environment and reviewer-score metadata become available."
        ),
        "current_quartile_note": "Q2-or-higher engineering target; verify current JCR/SJR quartile before submission.",
        "source_basis": (
            "Best fit if the study is written as a validated biosystems sensing and "
            "decision-support workflow rather than only a curve-processing experiment."
        ),
        "source_urls": "https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture",
    },
    {
        "target_journal": "Journal of Dairy Science",
        "route": "dairy_health_and_management_route",
        "scope_fit": "dairy cattle physiology, welfare, health monitoring, herd-management relevance",
        "lead_claim_if_ready": (
            "non-contact dairy-cow RR monitoring with external dairy-farm validation "
            "and animal/session-level robustness"
        ),
        "must_have": [
            "p2g_blinded_consensus_scored",
            "p2g_q2_external_sample_target",
            "metadata_quality_ready",
            "biological_context_ready",
        ],
        "nice_to_have": [
            "cow_group_validation_ready",
            "manual_quality_scores_ready",
        ],
        "claim_exclusions": "Do not use this route without biological context and real metadata.",
        "current_quartile_note": "Strong dairy target; verify current scope, article type, and quartile before submission.",
        "source_basis": (
            "Use only if biological interpretation is strengthened with dairy-farm "
            "metadata, animal/session grouping, and welfare/health relevance."
        ),
        "source_urls": "https://www.journalofdairyscience.org/",
    },
    {
        "target_journal": "Journal of Thermal Biology",
        "route": "thermal_physiology_route",
        "scope_fit": "temperature biology, animal thermal responses, heat-stress physiology",
        "lead_claim_if_ready": (
            "thermal physiology interpretation of nostril-temperature RR signals "
            "under measured barn temperature/humidity or THI strata"
        ),
        "must_have": [
            "p2g_blinded_consensus_scored",
            "p2g_q2_external_sample_target",
            "biological_context_ready",
            "metadata_quality_ready",
        ],
        "nice_to_have": [
            "conformal_uncertainty_ready",
            "manual_quality_scores_ready",
        ],
        "claim_exclusions": "Do not use this route without real temperature/humidity or THI metadata.",
        "current_quartile_note": "Thermal-biology route; verify current JCR/SJR quartile and scope before submission.",
        "source_basis": (
            "Thermal Biology is attractive only after real temperature/humidity or "
            "THI context is available; current internal dataset cannot support "
            "heat-stress claims."
        ),
        "source_urls": (
            "https://www.sciencedirect.com/journal/journal-of-thermal-biology"
        ),
    },
    {
        "target_journal": "Animals",
        "route": "animal_welfare_fallback_route",
        "scope_fit": "animal welfare, livestock monitoring, applied validation",
        "lead_claim_if_ready": (
            "non-invasive respiratory monitoring and review-triage workflow for "
            "cattle welfare applications"
        ),
        "must_have": [
            "p2g_method_freeze_ready",
            "p2g_blinded_consensus_scored",
        ],
        "nice_to_have": [
            "metadata_quality_ready",
            "biological_context_ready",
        ],
        "claim_exclusions": (
            "Avoid heat-stress and manual-quality stratified claims unless real "
            "environment and reviewer-score metadata become available."
        ),
        "current_quartile_note": "Possible fallback; verify current quartile, APC, and fit before submission.",
        "source_basis": (
            "Use if external validation is feasible but engineering or thermal-"
            "physiology novelty is not strong enough for the more selective targets."
        ),
        "source_urls": "https://www.mdpi.com/journal/animals",
    },
]


GATE_DESCRIPTIONS = {
    "external_validation_ready": "Frozen external split readiness is PASS and external metrics exist.",
    "fieldwork_smoke_ready": "At least the 50-video external smoke-test fieldwork tier is PASS.",
    "fieldwork_q2_ready": "At least the 104-video Q2 absolute-validation fieldwork tier is PASS.",
    "fieldwork_algorithmic_q2_ready": (
        "At least the 104-video algorithmic external-validation tier is PASS "
        "without requiring heat-stress or manual-quality metadata."
    ),
    "method_freeze_locked": "Frozen method manifest is locked before external scoring.",
    "metadata_quality_ready": "Metadata quality audit has no Q2-blocking failures.",
    "biological_context_ready": "Real temperature/humidity or THI context is available; no regional proxy-only claim.",
    "manual_quality_scores_ready": "Head motion, occlusion, and nostril visibility scores are complete enough for stratification.",
    "cow_group_validation_ready": (
        "Metadata grouping is ready for stress testing; current cow_id is a numeric "
        "video label, so cow-level wording still requires true animal identities."
    ),
    "algorithmic_quality_ready": "Algorithmic quality tiers, deployment decision curve, and conformal intervals are generated.",
    "conformal_uncertainty_ready": "Conformal RR uncertainty recommendation is generated.",
    "error_driven_external_design_ready": "Error-driven external validation queue and strata are generated.",
    "p2g_method_freeze_ready": (
        "P2g selected members, prominence, duration-gated predictions, and frozen consensus scorer are present."
    ),
    "p2g_complete_annotation_packets_ready": (
        "Two 94-clip P2g primary-all blinded worklists are generated with no prefilled counts."
    ),
    "p2g_blinded_consensus_scored": (
        "A 94-clip primary-all dual-annotator consensus score exists for frozen P2g predictions."
    ),
    "p2g_q2_external_sample_target": (
        "At least 104 independently annotated scorable external clips are available for the project Q2 target."
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a target-journal readiness strategy for the thermal RR Q2+ "
            "submission package from current generated evidence."
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


def first_status(table: pd.DataFrame, check: str) -> str:
    if table.empty or "check" not in table.columns:
        return ""
    match = table[table["check"].astype(str) == check]
    if match.empty:
        return ""
    return str(match.iloc[0].get("status", ""))


def tier_pass(tier_status: pd.DataFrame, tier: str) -> bool:
    if tier_status.empty or "tier" not in tier_status.columns:
        return False
    match = tier_status[tier_status["tier"].astype(str) == tier]
    if match.empty:
        return False
    return str(match.iloc[0].get("status", "")).upper() == "PASS"


def gate_state(output_dir: Path, input_root: Path, output_prefix: str) -> dict[str, dict[str, object]]:
    readiness = read_optional(input_root / "paper_repro_quality_residual_submission_readiness.csv")
    if readiness.empty:
        readiness = read_optional(input_root / f"{output_prefix}_quality_residual_submission_readiness.csv")
    external_split = read_optional(input_root / f"{output_prefix}_external_split_readiness.csv")
    metadata_audit = read_optional(output_dir / "paper_metadata_quality_audit.csv")
    metadata_field = read_optional(output_dir / "paper_metadata_quality_field_status.csv")
    tier_status_table = read_optional(output_dir / "paper_external_validation_fieldwork_tier_status.csv")
    method_freeze = read_optional(output_dir / "paper_method_freeze_summary.csv")
    algorithmic_quality = read_optional(output_dir / "paper_algorithmic_quality_tier_metrics_table.csv")
    deployment = read_optional(output_dir / "paper_deployment_decision_recommendation_table.csv")
    conformal = read_optional(output_dir / "paper_conformal_rr_recommendation_table.csv")
    error_queue = read_optional(output_dir / "paper_external_validation_error_driven_queue.csv")
    error_strata = read_optional(output_dir / "paper_external_validation_error_driven_strata.csv")
    p2g_selection = read_optional(output_dir / "paper_calibration_free_thermal_index_selection.csv")
    p2g_predictions = read_optional(output_dir / "paper_calibration_free_thermal_index_gated_predictions.csv")
    p2g_packets = read_optional(output_dir / "paper_p2g_primary_all_packet_summary.csv")
    p2g_extension_packets = read_optional(
        output_dir / "paper_p2g_extension_all_packet_summary.csv"
    )
    p2g_extension_predictions = read_optional(
        output_dir / "paper_p2g_extension_all_frozen_predictions.csv"
    )
    p2g_consensus = read_optional(
        output_dir / "paper_p2g_blinded_consensus_primary_all_metrics.csv"
    )

    q2_status = "unknown"
    if not readiness.empty and "q2_submission_status" in readiness.columns:
        q2_status = str(readiness["q2_submission_status"].iloc[0])

    external_ready = first_status(external_split, "external split validation ready") == "PASS"
    metadata_blockers = 0
    if not metadata_audit.empty and "q2_blocking" in metadata_audit.columns:
        metadata_blockers = int(metadata_audit["q2_blocking"].map(truthy).sum())

    def field_claim(field: str) -> bool:
        if metadata_field.empty or "field" not in metadata_field.columns:
            return False
        match = metadata_field[metadata_field["field"].astype(str) == field]
        if match.empty:
            return False
        if "claim_unlocked" in match.columns:
            return truthy(match.iloc[0].get("claim_unlocked"))
        if "status" in match.columns:
            return str(match.iloc[0].get("status", "")).upper() == "PASS"
        return False

    biological_ready = field_claim("thi") or (
        field_claim("ambient_temperature_c") and field_claim("relative_humidity_percent")
    )
    manual_quality_ready = all(
        field_claim(field)
        for field in [
            "head_motion_score_0_3",
            "occlusion_score_0_3",
            "nostril_visibility_score_0_3",
        ]
    )
    cow_group_ready = field_claim("cow_id")

    freeze_locked = False
    freeze_evidence = "method freeze summary missing"
    if not method_freeze.empty:
        item = method_freeze.iloc[0]
        freeze_locked = bool(
            str(item.get("freeze_status", "")) == "locked"
            and int(float(item.get("required_files_missing", 1))) == 0
        )
        freeze_evidence = (
            f"freeze_status={item.get('freeze_status')}; "
            f"method_freeze_id={item.get('method_freeze_id')}"
        )

    p2g_scorer = Path(__file__).resolve().parents[1] / "scripts" / "score_rr_p2g_blinded_consensus.py"
    p2g_freeze_ready = (
        not p2g_selection.empty
        and not p2g_predictions.empty
        and p2g_scorer.exists()
    )
    p2g_packet_clips = 0
    p2g_packets_ready = False
    if not p2g_packets.empty and {"annotator_id", "clips"}.issubset(p2g_packets.columns):
        p2g_packet_clips = int(pd.to_numeric(p2g_packets["clips"], errors="coerce").max())
        p2g_packets_ready = len(p2g_packets) == 2 and p2g_packet_clips == 94
    p2g_extension_packet_clips = 0
    p2g_extension_packets_ready = False
    if not p2g_extension_packets.empty and {"annotator_id", "clips"}.issubset(
        p2g_extension_packets.columns
    ):
        p2g_extension_packet_clips = int(
            pd.to_numeric(p2g_extension_packets["clips"], errors="coerce").max()
        )
        p2g_extension_packets_ready = (
            len(p2g_extension_packets) == 2 and p2g_extension_packet_clips >= 10
        )
    p2g_extension_prediction_clips = int(len(p2g_extension_predictions))
    p2g_extension_predictions_ready = (
        p2g_extension_prediction_clips == p2g_extension_packet_clips
        and p2g_extension_prediction_clips >= 10
        and "duration_gated_rr_bpm" in p2g_extension_predictions.columns
    )
    p2g_consensus_clips = 0
    p2g_consensus_ready = False
    if not p2g_consensus.empty and {"scope", "consensus_clips"}.issubset(p2g_consensus.columns):
        primary = p2g_consensus[p2g_consensus["scope"].astype(str).eq("primary_all")]
        if not primary.empty:
            p2g_consensus_clips = int(
                pd.to_numeric(primary.iloc[0].get("consensus_clips"), errors="coerce")
            )
            p2g_consensus_ready = p2g_consensus_clips >= 94

    states = {
        "external_validation_ready": {
            "passed": external_ready,
            "evidence": (
                "external split validation ready PASS"
                if external_ready
                else "external split validation not ready; external rows=0"
            ),
        },
        "fieldwork_smoke_ready": {
            "passed": tier_pass(tier_status_table, "minimum_holdout_smoke_test"),
            "evidence": "minimum_holdout_smoke_test tier status from fieldwork preflight",
        },
        "fieldwork_q2_ready": {
            "passed": tier_pass(tier_status_table, "q2_target_absolute_validation"),
            "evidence": "q2_target_absolute_validation tier status from fieldwork preflight",
        },
        "fieldwork_algorithmic_q2_ready": {
            "passed": tier_pass(tier_status_table, "q2_algorithmic_external_validation"),
            "evidence": (
                "q2_algorithmic_external_validation tier status from fieldwork "
                "preflight; no heat-stress/manual-quality claims unlocked"
            ),
        },
        "method_freeze_locked": {
            "passed": freeze_locked,
            "evidence": freeze_evidence,
        },
        "metadata_quality_ready": {
            "passed": metadata_blockers == 0 and not metadata_audit.empty,
            "evidence": f"metadata_q2_blockers={metadata_blockers}",
        },
        "biological_context_ready": {
            "passed": biological_ready,
            "evidence": "real temperature/humidity/THI field readiness",
        },
        "manual_quality_scores_ready": {
            "passed": manual_quality_ready,
            "evidence": "head motion, occlusion, and nostril visibility claim readiness",
        },
        "cow_group_validation_ready": {
            "passed": cow_group_ready,
            "evidence": (
                "cow_id numeric-video-label readiness; verify true animal identity "
                "before cow-level wording"
            ),
        },
        "algorithmic_quality_ready": {
            "passed": not algorithmic_quality.empty and not deployment.empty,
            "evidence": (
                f"algorithmic_quality_rows={len(algorithmic_quality)}; "
                f"deployment_rows={len(deployment)}"
            ),
        },
        "conformal_uncertainty_ready": {
            "passed": not conformal.empty,
            "evidence": f"conformal_recommendation_rows={len(conformal)}",
        },
        "error_driven_external_design_ready": {
            "passed": not error_queue.empty and not error_strata.empty,
            "evidence": f"queue_rows={len(error_queue)}; strata_rows={len(error_strata)}",
        },
        "q2_submission_ready": {
            "passed": q2_status == "ready",
            "evidence": f"q2_submission_status={q2_status}",
        },
        "p2g_method_freeze_ready": {
            "passed": p2g_freeze_ready,
            "evidence": (
                "P2g selection, gated prediction, and frozen consensus scorer present"
                if p2g_freeze_ready
                else "P2g selection/prediction/scorer artifact missing"
            ),
        },
        "p2g_complete_annotation_packets_ready": {
            "passed": p2g_packets_ready,
            "evidence": f"primary-all blinded packet clips={p2g_packet_clips}; annotator packets={len(p2g_packets)}",
        },
        "p2g_blinded_consensus_scored": {
            "passed": p2g_consensus_ready,
            "evidence": f"primary-all blinded consensus clips={p2g_consensus_clips}/94",
        },
        "p2g_q2_external_sample_target": {
            "passed": p2g_consensus_clips >= 104,
            "evidence": (
                f"blinded-consensus clips={p2g_consensus_clips}/104; "
                f"prepared primary packet clips={p2g_packet_clips}/104; "
                f"pre-registered extension packet clips={p2g_extension_packet_clips}; "
                f"extension A/B ready={p2g_extension_packets_ready}; "
                f"extension frozen predictions={p2g_extension_prediction_clips}; "
                f"extension prediction freeze ready={p2g_extension_predictions_ready}; "
                f"additional consensus-scored clips needed after primary scope={max(0, 104 - p2g_packet_clips)}"
            ),
        },
    }
    return states


def build_gate_matrix(states: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for gate, description in GATE_DESCRIPTIONS.items():
        state = states.get(gate, {"passed": False, "evidence": "missing"})
        rows.append(
            {
                "gate": gate,
                "description": description,
                "status": "PASS" if state["passed"] else "FAIL",
                "evidence": state["evidence"],
            }
        )
    return pd.DataFrame(rows)


def build_strategy(states: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for target in JOURNAL_TARGETS:
        must = target["must_have"]
        nice = target["nice_to_have"]
        missing_must = [gate for gate in must if not bool(states.get(gate, {}).get("passed", False))]
        missing_nice = [gate for gate in nice if not bool(states.get(gate, {}).get("passed", False))]
        if not missing_must:
            status = "ready_after_quartile_and_author_guideline_verification"
            recommendation = "Prepare journal-specific cover letter and final formatting."
        elif (
            "external_validation_ready" in missing_must
            or "fieldwork_q2_ready" in missing_must
            or "fieldwork_algorithmic_q2_ready" in missing_must
            or "fieldwork_smoke_ready" in missing_must
        ):
            status = "hold_for_external_validation"
            recommendation = "Do not submit to this target until independent external validation is complete."
        elif (
            "p2g_blinded_consensus_scored" in missing_must
            or "p2g_q2_external_sample_target" in missing_must
        ):
            status = "hold_for_p2g_blinded_external_validation"
            actions = [
                "Complete the 94-clip P2G_FULL_A/B annotations and adjudication, then run frozen P2g consensus scoring."
            ]
            if "p2g_q2_external_sample_target" in missing_must:
                actions.append(
                    "Complete the pre-registered 15-clip P2g extension A/B recount, retain every "
                    "unreadable exclusion in the flow table, and score at least 10 additional valid clips "
                    "against the already frozen predictions for the 104-clip Q2 target."
                )
            if "biological_context_ready" in missing_must:
                actions.append(
                    "Add synchronized temperature/humidity or THI only if retaining a thermal-physiology claim."
                )
            recommendation = " ".join(actions)
        else:
            status = "hold_for_metadata_or_biological_context"
            recommendation = "Strengthen metadata/biological interpretation before submitting to this target."
        rows.append(
            {
                "target_journal": target["target_journal"],
                "route": target["route"],
                "scope_fit": target["scope_fit"],
                "journal_strategy_status": status,
                "lead_claim_if_ready": target["lead_claim_if_ready"],
                "missing_must_have_gates": ";".join(missing_must) if missing_must else "none",
                "missing_nice_to_have_gates": ";".join(missing_nice) if missing_nice else "none",
                "claim_exclusions": target.get("claim_exclusions", ""),
                "recommended_next_action": recommendation,
                "current_quartile_note": target["current_quartile_note"],
                "source_basis": target["source_basis"],
                "source_urls": target["source_urls"],
                "paper_use": "submission_strategy_not_current_acceptance_claim",
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


def write_report(path: Path, strategy: pd.DataFrame, gates: pd.DataFrame) -> None:
    recommended = strategy[strategy["journal_strategy_status"].astype(str).str.startswith("ready")]
    if recommended.empty:
        current_position = (
            "No Q2+ target is submission-ready yet. The package is internally "
            "manuscript-ready, but all journal routes that can support a strong "
            "claim still require independent external validation and/or real "
            "metadata."
        )
    else:
        current_position = (
            "At least one target route is ready after journal-specific quartile, "
            "scope, and author-guideline verification."
        )
    text = f"""# Target Journal Strategy

{current_position}

Quartile and impact-factor labels are intentionally not hard-coded as final
claims because they change over time. Verify the current JCR/SJR classification
before submission.

## Journal Route Matrix

{markdown_table(strategy, [
    'target_journal',
    'route',
    'journal_strategy_status',
    'lead_claim_if_ready',
    'missing_must_have_gates',
    'claim_exclusions',
    'recommended_next_action',
    'current_quartile_note',
])}

## Evidence Gates

{markdown_table(gates, ['gate', 'description', 'status', 'evidence'])}

## Manuscript Position

The current best paper framing is an internally validated thermal-RR algorithm
and external-validation design package. If environment and manual-quality
metadata cannot be recovered, use the algorithmic engineering route and avoid
heat-stress, THI-stratified, head-motion, occlusion, and nostril-visibility
stratified claims. The stronger Q2+ framing still requires the external
fieldwork and frozen external split gates to pass.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    states = gate_state(output_dir, args.input_root, args.output_prefix)
    gates = build_gate_matrix(states)
    strategy = build_strategy(states)

    strategy_path = output_dir / "paper_target_journal_strategy.csv"
    gate_path = output_dir / "paper_target_journal_gate_matrix.csv"
    report_path = output_dir / "paper_target_journal_strategy.md"
    strategy.to_csv(strategy_path, index=False)
    gates.to_csv(gate_path, index=False)
    write_report(report_path, strategy, gates)

    print(f"Saved target journal strategy: {strategy_path}")
    print(f"Saved target journal gate matrix: {gate_path}")
    print(f"Saved target journal strategy report: {report_path}")
    print("\nTarget strategy:")
    print(
        strategy[
            [
                "target_journal",
                "journal_strategy_status",
                "missing_must_have_gates",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
