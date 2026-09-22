from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE_VERIFICATION_DATE = "2026-07-13"


JOURNAL_ROUTES = [
    {
        "rank": 1,
        "target_journal": "Computers and Electronics in Agriculture",
        "route": "primary_algorithmic_agricultural_sensing_route",
        "official_source_url": "https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture",
        "cite_score": 15.1,
        "impact_factor": 8.9,
        "apc_usd": 3730,
        "official_scope_basis": (
            "Agricultural computer hardware, software, electronic instrumentation, "
            "control systems, AI, sensors, and machine vision for animal/livestock farming. "
            "The journal emphasizes novelty and views prior-dataset prototyping as preliminary "
            "until a rigorous controlled/reported dataset is generated."
        ),
        "fit_to_current_innovation": (
            "Best match after validation because the current contribution is an internally "
            "selected duration-gated calibration-free relative thermal-color RR method with "
            "cross-farm robustness ablations and a frozen scoring path."
        ),
        "current_state": (
            "The P2g method and 94-clip dual-blind packets are frozen; consensus counts are "
            "not yet available."
        ),
        "minimum_to_submit": (
            "Score the frozen P2g method against 94 blinded-consensus clips, then add 10 "
            "independently annotated scorable clips for the project 104-clip engineering tier."
        ),
        "claim_allowed_after_minimum": (
            "External algorithmic RR accuracy, safe-gate behavior, quality-control triage, "
            "and uncertainty reporting."
        ),
        "claim_excluded_without_extra_data": (
            "No heat-stress, THI, manual head-motion, occlusion, or nostril-visibility "
            "stratified claims."
        ),
    },
    {
        "rank": 2,
        "target_journal": "Smart Agricultural Technology",
        "route": "smart_agricultural_technology_applied_route",
        "official_source_url": "https://www.sciencedirect.com/journal/smart-agricultural-technology",
        "cite_score": 7.7,
        "impact_factor": 5.7,
        "apc_usd": 2010,
        "official_scope_basis": (
            "Practical smart-system applications that integrate advanced computing, imaging, "
            "sensing, or controls for production agriculture, including livestock health and welfare."
        ),
        "fit_to_current_innovation": (
            "A realistic applied route after frozen P2g cross-farm consensus validation, "
            "provided the current JCR/SJR quartile is verified before submission."
        ),
        "current_state": (
            "The P2g method is frozen and its 94-clip blinded packet is ready; consensus "
            "counts are still pending."
        ),
        "minimum_to_submit": (
            "Complete dual-blinded consensus scoring for the 94 P2g clips and retain the "
            "current no-heat-stress claim boundary."
        ),
        "claim_allowed_after_minimum": (
            "Practical cross-farm non-contact RR monitoring accuracy and review-routing evidence."
        ),
        "claim_excluded_without_extra_data": (
            "No heat-stress, disease, welfare-outcome, or THI claim without real matching metadata."
        ),
    },
    {
        "rank": 3,
        "target_journal": "Biosystems Engineering",
        "route": "engineering_impact_route",
        "official_source_url": "https://www.sciencedirect.com/journal/biosystems-engineering",
        "cite_score": 10.1,
        "impact_factor": 5.3,
        "apc_usd": 3350,
        "official_scope_basis": (
            "Engineering and physical-science research for biological systems, including "
            "data-driven findings, instrumentation, and equipment. The journal expects "
            "new engineering insight, representative conditions, operational implementation, "
            "application impact, and a Science4Impact statement."
        ),
        "fit_to_current_innovation": (
            "Viable if written as an operational biosystems sensing workflow with external "
            "field validation and clear decision-support impact."
        ),
        "current_state": (
            "Internal algorithmic evidence is usable, but representative external operation "
            "and application-impact evidence are not yet present."
        ),
        "minimum_to_submit": (
            "Add the 104-video external tier plus an operational-use description, failure "
            "handling, and Science4Impact statement."
        ),
        "claim_allowed_after_minimum": (
            "Operational thermal RR sensing workflow and decision-support readiness under "
            "reported farm conditions."
        ),
        "claim_excluded_without_extra_data": (
            "No broad deployment or biological-impact claim from internal-only validation."
        ),
    },
    {
        "rank": 4,
        "target_journal": "Animals",
        "route": "animal_welfare_applied_route",
        "official_source_url": "https://www.mdpi.com/journal/animals/about",
        "cite_score": 5.5,
        "impact_factor": 3.2,
        "apc_usd": None,
        "official_scope_basis": (
            "International interdisciplinary animal journal covering animal science, ethics, "
            "welfare, management, and interactions between animals and the outside world."
        ),
        "fit_to_current_innovation": (
            "Fallback route if the paper emphasizes non-invasive monitoring and welfare "
            "decision support after external validation."
        ),
        "current_state": (
            "Current internal algorithm evidence can support a draft, but external validation "
            "and animal-welfare framing are still needed."
        ),
        "minimum_to_submit": (
            "At least the 50-video external smoke test, preferably 104 external videos, plus "
            "ethics/provenance and welfare use-case wording."
        ),
        "claim_allowed_after_minimum": (
            "Non-invasive cattle RR monitoring feasibility for welfare-oriented observation."
        ),
        "claim_excluded_without_extra_data": (
            "No welfare outcome, heat-stress, or animal-response conclusion without matching "
            "biological metadata."
        ),
    },
    {
        "rank": 5,
        "target_journal": "Journal of Dairy Science",
        "route": "dairy_health_management_route",
        "official_source_url": "https://www.sciencedirect.com/journal/journal-of-dairy-science",
        "cite_score": 7.8,
        "impact_factor": 4.4,
        "apc_usd": 3500,
        "official_scope_basis": (
            "Official ADSA journal for dairy production and dairy foods. The page reports "
            "Q1 in Agriculture, Dairy, and Animal Science and Q2 in Food Science and Technology."
        ),
        "fit_to_current_innovation": (
            "High-value route only if the algorithm is tied to dairy production, health, "
            "welfare, or management evidence rather than precision alone."
        ),
        "current_state": (
            "Current provenance identifies a dairy-farm collection window, but environment, "
            "health, management, and biological outcome metadata are unavailable."
        ),
        "minimum_to_submit": (
            "External dairy-farm validation plus cow/session metadata and a dairy-management "
            "or health-monitoring interpretation."
        ),
        "claim_allowed_after_minimum": (
            "Dairy-cow non-contact RR monitoring with management relevance."
        ),
        "claim_excluded_without_extra_data": (
            "No dairy health, heat-stress, or productivity implication from the current "
            "internal-only dataset."
        ),
    },
    {
        "rank": 6,
        "target_journal": "Journal of Thermal Biology",
        "route": "thermal_physiology_route",
        "official_source_url": "https://www.sciencedirect.com/journal/journal-of-thermal-biology",
        "cite_score": 5.7,
        "impact_factor": 2.9,
        "apc_usd": 3580,
        "official_scope_basis": (
            "Thermal biology of humans and animals, including thermal responses, heat/cold "
            "effects, thermoregulation, stress, and techniques for measuring thermal responses."
        ),
        "fit_to_current_innovation": (
            "Only appropriate if the paper becomes a thermal-response or heat-stress study, "
            "not merely an RR algorithm paper."
        ),
        "current_state": (
            "No synchronized barn temperature, humidity, THI, ATHI, or thermal-response "
            "strata exist for the current dataset."
        ),
        "minimum_to_submit": (
            "Collect synchronized environment and animal-response metadata with external RR "
            "validation."
        ),
        "claim_allowed_after_minimum": (
            "Thermal-response measurement technique or heat-stress related RR interpretation."
        ),
        "claim_excluded_without_extra_data": (
            "Do not submit on this route using regional provenance alone."
        ),
    },
]


MINIMUM_DATA_ROWS = [
    {
        "route": "current_internal_algorithmic_manuscript",
        "minimum_new_videos": 0,
        "required_now": (
            "Use current internal data only for supervisor review, methods development, "
            "and a carefully bounded draft."
        ),
        "currently_available": "yes",
        "unlocks": "Internal algorithmic manuscript draft, not Q2+ final submission.",
        "cannot_unlock": "External validation, strong generalization, or final Q2+ submission.",
    },
    {
        "route": "minimum_holdout_smoke_test",
        "minimum_new_videos": 50,
        "required_now": (
            "Independent external_test_split rows, cow_id, collection_date, camera_id, "
            "scene_id, manual_breath_count, manual_duration_seconds, manual_rr_bpm."
        ),
        "currently_available": "no",
        "unlocks": "External feasibility evidence only.",
        "cannot_unlock": "Strong relative improvement or broad deployment claim.",
    },
    {
        "route": "q2_algorithmic_external_validation",
        "minimum_new_videos": 104,
        "required_now": (
            "Method-frozen independent videos with split provenance, cow/session/camera/scene "
            "labels, manual reference RR, and annotator/source fields. Environment and manual "
            "quality scores are optional unless those claims are made."
        ),
        "currently_available": "no",
        "unlocks": (
            "Best current Q2+ route for Computers and Electronics in Agriculture or "
            "Biosystems Engineering."
        ),
        "cannot_unlock": "Heat-stress or manual-quality stratified claims.",
    },
    {
        "route": "q2_biological_or_thermal_route",
        "minimum_new_videos": 104,
        "required_now": (
            "All algorithmic external fields plus synchronized barn temperature, humidity, "
            "THI/ATHI if used, and relevant animal/session context."
        ),
        "currently_available": "no",
        "unlocks": "Dairy-health, welfare, heat-stress, or thermal-biology route.",
        "cannot_unlock": "Not feasible from current 73 videos alone.",
    },
    {
        "route": "q2_strong_relative_improvement",
        "minimum_new_videos": 209,
        "required_now": (
            "Paired default-vs-safe-gate external predictions with all thresholds frozen "
            "before scoring."
        ),
        "currently_available": "no",
        "unlocks": "Statistically stronger external delta RR R2 claim if CI excludes zero.",
        "cannot_unlock": "Guaranteed positive CI. The external data must prove it.",
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build a journal route and minimum-data decision pack for the thermal RR Q2+ submission plan."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def check_status(readiness: pd.DataFrame, check: str) -> tuple[str, str]:
    if readiness.empty or "check" not in readiness.columns:
        return "MISSING", "readiness table unavailable"
    match = readiness[readiness["check"].astype(str) == check]
    if match.empty:
        return "MISSING", f"check not found: {check}"
    row = match.iloc[0]
    return str(row.get("status", "")), str(row.get("evidence", ""))


def best_internal_method(main_results: pd.DataFrame) -> dict[str, object]:
    if main_results.empty:
        return {
            "method": "",
            "rr_r2": "",
            "rr_mae_bpm": "",
            "exact_count": "",
            "paper_use": "",
        }
    usable = main_results[
        ~main_results["paper_use"].astype(str).isin(["upper_bound_only", "sensitivity_only"])
    ].copy()
    if usable.empty:
        usable = main_results.copy()
    usable["rr_r2_numeric"] = pd.to_numeric(usable["rr_r2"], errors="coerce")
    row = usable.sort_values("rr_r2_numeric", ascending=False).iloc[0]
    return {
        "method": row.get("method", ""),
        "rr_r2": row.get("rr_r2", ""),
        "rr_mae_bpm": row.get("rr_mae_bpm", ""),
        "exact_count": row.get("exact_count", ""),
        "paper_use": row.get("paper_use", ""),
    }


def summarize_current_state(
    readiness: pd.DataFrame,
    main_results: pd.DataFrame,
    collection_tiers: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    external_status, external_evidence = check_status(
        readiness, "frozen external split validation ready"
    )
    freeze_status, freeze_evidence = check_status(
        readiness, "external method freeze manifest locked"
    )
    delta_status, delta_evidence = check_status(
        readiness, "Delta RR R2 bootstrap CI excludes zero"
    )
    metadata_status, metadata_evidence = check_status(
        readiness, "metadata quality Q2 claim gates"
    )
    best = best_internal_method(main_results)
    q2_algorithmic_n = ""
    if not collection_tiers.empty and "collection_tier" in collection_tiers.columns:
        match = collection_tiers[
            collection_tiers["collection_tier"].astype(str)
            == "q2_algorithmic_external_validation"
        ]
        if not match.empty:
            q2_algorithmic_n = match.iloc[0].get("recommended_videos", "")
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
    p2g_packet_clips = 0
    if not p2g_packets.empty and "clips" in p2g_packets.columns:
        p2g_packet_clips = int(pd.to_numeric(p2g_packets["clips"], errors="coerce").max())
    p2g_extension_packet_clips = 0
    if not p2g_extension_packets.empty and "clips" in p2g_extension_packets.columns:
        p2g_extension_packet_clips = int(
            pd.to_numeric(p2g_extension_packets["clips"], errors="coerce").max()
        )
    p2g_extension_prediction_clips = int(len(p2g_extension_predictions))
    p2g_consensus_clips = 0
    if not p2g_consensus.empty and "consensus_clips" in p2g_consensus.columns:
        p2g_consensus_clips = int(
            pd.to_numeric(p2g_consensus.iloc[0].get("consensus_clips"), errors="coerce")
        )
    return {
        "best_internal_method": best["method"],
        "best_internal_rr_r2": best["rr_r2"],
        "best_internal_rr_mae_bpm": best["rr_mae_bpm"],
        "best_internal_exact_count": best["exact_count"],
        "best_internal_paper_use": best["paper_use"],
        "external_validation_status": external_status,
        "external_validation_evidence": external_evidence,
        "method_freeze_status": freeze_status,
        "method_freeze_evidence": freeze_evidence,
        "delta_rr_r2_ci_status": delta_status,
        "delta_rr_r2_ci_evidence": delta_evidence,
        "metadata_claim_gate_status": metadata_status,
        "metadata_claim_gate_evidence": metadata_evidence,
        "q2_algorithmic_external_target_n": q2_algorithmic_n,
        "p2g_packet_clips": p2g_packet_clips,
        "p2g_extension_packet_clips": p2g_extension_packet_clips,
        "p2g_extension_prediction_clips": p2g_extension_prediction_clips,
        "p2g_consensus_clips": p2g_consensus_clips,
    }


def route_decision(row: dict[str, object], state: dict[str, object]) -> dict[str, object]:
    external_ready = str(state["external_validation_status"]).upper() == "PASS"
    method_locked = str(state["method_freeze_status"]).upper() == "PASS"
    route = str(row["route"])
    p2g_consensus_ready = int(state["p2g_consensus_clips"]) >= 94
    p2g_q2_ready = int(state["p2g_consensus_clips"]) >= 104
    if not method_locked:
        decision = "not_ready_method_not_frozen"
        next_gate = "Lock method freeze before any external scoring."
    elif route == "primary_algorithmic_agricultural_sensing_route" and not p2g_q2_ready:
        decision = "best_target_after_external_validation"
        next_gate = (
            "Complete blinded P2g consensus scoring for 94 clips, then complete the "
            "pre-registered 15-clip extension recount and score at least 10 valid extension clips "
            "against the already frozen prediction file for the 104-clip engineering target."
        )
    elif route == "engineering_impact_route" and not p2g_q2_ready:
        decision = "second_target_after_external_and_impact_statement"
        next_gate = "Complete 104-clip P2g consensus validation plus operational-use and Science4Impact evidence."
    elif route == "smart_agricultural_technology_applied_route" and not p2g_consensus_ready:
        decision = "applied_target_after_p2g_consensus_validation"
        next_gate = "Complete the 94-clip P2g blinded consensus validation and verify current journal quartile/APC."
    elif route == "animal_welfare_applied_route" and not p2g_consensus_ready:
        decision = "fallback_after_external_validation"
        next_gate = "Complete the 94-clip P2g blinded consensus validation and animal-welfare framing."
    elif route in {"dairy_health_management_route", "thermal_physiology_route"}:
        decision = "not_current_primary_route"
        next_gate = "Requires new biological/environmental metadata beyond current data."
    else:
        decision = "submission_candidate"
        next_gate = "Run final manuscript and journal-specific compliance checks."
    output = dict(row)
    output.update(
        {
            "source_verification_date": SOURCE_VERIFICATION_DATE,
            "current_best_internal_method": state["best_internal_method"],
            "current_best_internal_rr_r2": state["best_internal_rr_r2"],
            "current_best_internal_rr_mae_bpm": state["best_internal_rr_mae_bpm"],
            "current_best_internal_exact_count": state["best_internal_exact_count"],
            "method_freeze_status": state["method_freeze_status"],
            "external_validation_status": state["external_validation_status"],
            "metadata_claim_gate_status": state["metadata_claim_gate_status"],
            "delta_rr_r2_ci_status": state["delta_rr_r2_ci_status"],
            "route_decision": decision,
            "next_gate": next_gate,
            "write_in_current_paper": (
                "Use as target-route strategy only; do not claim journal-ready external performance."
            ),
        }
    )
    return output


def build_next_actions(state: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "priority": "P0",
            "action": "Keep all current 73 videos internal",
            "why": "They were used for development and cannot serve as external validation.",
            "proof_output": "paper_metadata_template.csv external_test_split=internal for current rows.",
        },
        {
            "priority": "P0",
            "action": "Complete P2G_FULL_A/B blinded breath counts and adjudication",
            "why": "The 94-clip P2g primary validation packet is ready, but no dual-annotator consensus score exists.",
            "proof_output": (
                f"primary packet clips={state['p2g_packet_clips']}; "
                f"current blinded consensus clips={state['p2g_consensus_clips']}"
            ),
        },
        {
            "priority": "P0",
            "action": "Run frozen P2g consensus scoring, then complete the pre-registered 15-clip extension recount",
            "why": "The complete packet covers 94 clips; the pre-registered extension pool can supply the 10 additional valid clips required for the 104-clip engineering-Q2 target, and its P2g predictions are already frozen.",
            "proof_output": "paper_p2g_blinded_consensus_primary_all_metrics.csv, paper_p2g_extension_all_packet_summary.csv, paper_p2g_primary_plus_extension_frozen_predictions.csv, and the 104-clip collection tier.",
        },
        {
            "priority": "P1",
            "action": "Record manual reference RR for every external video",
            "why": "RR R2, MAE, RMSE, and exact count require independent manual reference values.",
            "proof_output": "manual_breath_count, manual_duration_seconds, manual_rr_bpm, reference_rr_annotator.",
        },
        {
            "priority": "P1",
            "action": "Keep environment and manual visual-quality scores optional",
            "why": "They are unavailable for current data and are needed only for THI or robustness-stratified claims.",
            "proof_output": "submission readiness marks these missing fields as nonblocking claim-specific limits.",
        },
        {
            "priority": "P2",
            "action": "Only pursue Journal of Dairy Science or Journal of Thermal Biology after new biological metadata",
            "why": "Those routes require dairy/thermal meaning beyond algorithmic precision.",
            "proof_output": "paper_submission_gap_route_journal_fit.csv ranks them below the algorithmic route.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
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
    journal_fit: pd.DataFrame,
    minimum_data: pd.DataFrame,
    next_actions: pd.DataFrame,
    state: dict[str, object],
) -> Path:
    report = output_dir / "paper_submission_gap_route_decision.md"
    lines = [
        "# Submission Route Decision Pack",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source verification date: `{SOURCE_VERIFICATION_DATE}`",
        "",
        "## Bottom Line",
        "",
        (
            "The current best route is a calibration-free cross-farm agricultural-sensing paper "
            "for Computers and Electronics in Agriculture after frozen P2g external validation. "
            "The 94-clip blinded packet is ready, but the package is not Q2+ submission-ready "
            "until consensus scoring is complete and the 104-clip engineering target is met."
        ),
        "",
        "## Current Evidence",
        "",
        f"- Best internal method: `{state['best_internal_method']}`.",
        f"- Best internal RR R2: `{state['best_internal_rr_r2']}`.",
        f"- Best internal MAE: `{state['best_internal_rr_mae_bpm']}` bpm.",
        f"- Best internal exact count: `{state['best_internal_exact_count']}`.",
        f"- Method freeze: `{state['method_freeze_status']}`.",
        f"- External validation: `{state['external_validation_status']}`.",
        f"- Metadata claim gate: `{state['metadata_claim_gate_status']}`.",
        f"- Delta RR R2 CI gate: `{state['delta_rr_r2_ci_status']}`.",
        f"- P2g primary packet clips: `{state['p2g_packet_clips']}`.",
        f"- P2g blinded consensus clips: `{state['p2g_consensus_clips']}`.",
        f"- Pre-registered P2g extension packet clips: `{state['p2g_extension_packet_clips']}`.",
        f"- Frozen P2g extension prediction clips: `{state['p2g_extension_prediction_clips']}`.",
        "",
        "## Journal Route Ranking",
        "",
        markdown_table(
            journal_fit,
            [
                "rank",
                "target_journal",
                "route",
                "impact_factor",
                "cite_score",
                "route_decision",
                "next_gate",
            ],
        ),
        "",
        "## Minimum Data Choices",
        "",
        markdown_table(
            minimum_data,
            [
                "route",
                "minimum_new_videos",
                "currently_available",
                "required_now",
                "unlocks",
            ],
        ),
        "",
        "## Next Actions",
        "",
        markdown_table(next_actions, ["priority", "action", "why", "proof_output"]),
        "",
        "## Claim Boundary",
        "",
        (
            "Truth-calibrated RR remains an oracle upper bound only. Environment, THI, "
            "head-motion, occlusion, and nostril-visibility fields should stay blank "
            "unless real measurements or manual labels exist. They should not block the "
            "algorithmic route, but they must block heat-stress, welfare-outcome, or "
            "manual-quality stratified conclusions."
        ),
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness = read_optional(args.input_root / f"{args.corrected_prefix}_submission_readiness.csv")
    main_results = read_optional(output_dir / "paper_main_results_table.csv")
    collection_tiers = read_optional(output_dir / "paper_external_validation_collection_tiers.csv")

    state = summarize_current_state(readiness, main_results, collection_tiers, output_dir)
    journal_fit = pd.DataFrame(
        [route_decision(route, state) for route in JOURNAL_ROUTES]
    )
    minimum_data = pd.DataFrame(MINIMUM_DATA_ROWS)
    next_actions = build_next_actions(state)

    fit_path = output_dir / "paper_submission_gap_route_journal_fit.csv"
    min_path = output_dir / "paper_submission_gap_route_minimum_data.csv"
    actions_path = output_dir / "paper_submission_gap_route_next_actions.csv"
    journal_fit.to_csv(fit_path, index=False)
    minimum_data.to_csv(min_path, index=False)
    next_actions.to_csv(actions_path, index=False)
    report = write_report(output_dir, journal_fit, minimum_data, next_actions, state)

    print(f"Saved submission route journal fit: {fit_path}")
    print(f"Saved submission route minimum data: {min_path}")
    print(f"Saved submission route next actions: {actions_path}")
    print(f"Saved submission route decision report: {report}")
    print(journal_fit[["rank", "target_journal", "route_decision", "next_gate"]].to_string(index=False))


if __name__ == "__main__":
    main()
