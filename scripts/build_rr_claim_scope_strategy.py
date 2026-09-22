from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a claim-scope strategy for the thermal RR paper package when "
            "environment metadata and manual quality scores are unavailable."
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


def status_for_check(table: pd.DataFrame, check: str) -> str:
    if table.empty or "check" not in table.columns:
        return "MISSING"
    match = table[table["check"].astype(str) == check]
    if match.empty:
        return "MISSING"
    return str(match.iloc[0].get("status", "MISSING")).upper()


def tier_status(table: pd.DataFrame, tier: str) -> str:
    if table.empty or "tier" not in table.columns:
        return "MISSING"
    match = table[table["tier"].astype(str) == tier]
    if match.empty:
        return "MISSING"
    return str(match.iloc[0].get("status", "MISSING")).upper()


def gate_status(gates: pd.DataFrame, gate: str) -> str:
    if gates.empty or "gate" not in gates.columns:
        return "MISSING"
    match = gates[gates["gate"].astype(str) == gate]
    if match.empty:
        return "MISSING"
    return str(match.iloc[0].get("status", "MISSING")).upper()


def metric_value(table: pd.DataFrame, method_contains: str, column: str) -> str:
    if table.empty or "method" not in table.columns or column not in table.columns:
        return ""
    match = table[table["method"].astype(str).str.contains(method_contains, case=False, regex=False)]
    if match.empty:
        return ""
    value = match.iloc[0].get(column, "")
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def build_routes(
    main_results: pd.DataFrame,
    gates: pd.DataFrame,
    tiers: pd.DataFrame,
    external_split: pd.DataFrame,
) -> pd.DataFrame:
    default_r2 = metric_value(main_results, "Default thermal RR pipeline", "rr_r2")
    safe_r2 = metric_value(main_results, "Conservative signal-aware safe gate (fixed threshold", "rr_r2")
    safe_exact = ""
    if not main_results.empty and "method" in main_results.columns:
        match = main_results[
            main_results["method"]
            .astype(str)
            .str.contains("Conservative signal-aware safe gate (fixed threshold", case=False, regex=False)
        ]
        if not match.empty:
            safe_exact = str(match.iloc[0].get("exact_count", ""))

    external_ready = status_for_check(external_split, "external split validation ready")
    algorithmic_q2 = tier_status(tiers, "q2_algorithmic_external_validation")
    full_q2 = tier_status(tiers, "q2_target_absolute_validation")

    routes = [
        {
            "route": "internal_algorithm_manuscript_package",
            "current_status": "ready_internal_only",
            "required_next_gate": "external_validation_ready",
            "main_claim_allowed_now": (
                "Internal 73-video algorithm development and validation only."
            ),
            "main_claim_after_next_gate": (
                "Method-frozen external RR validation if an independent external "
                "set is collected and scored."
            ),
            "excluded_claims": (
                "No external generalization, heat-stress, THI-stratified, or "
                "manual quality-stratified claims."
            ),
            "evidence": (
                f"default_rr_r2={default_r2}; safe_gate_rr_r2={safe_r2}; "
                f"safe_gate_exact={safe_exact}; external_ready={external_ready}"
            ),
            "paper_use": "current_methods_and_results_with_limitations",
        },
        {
            "route": "q2_algorithmic_engineering_route",
            "current_status": (
                "ready_after_external_validation"
                if external_ready == "PASS" and algorithmic_q2 == "PASS"
                else "hold_for_algorithmic_external_validation"
            ),
            "required_next_gate": "q2_algorithmic_external_validation and frozen external split validation",
            "main_claim_allowed_now": (
                "Not yet; current evidence is internal and development-set based."
            ),
            "main_claim_after_next_gate": (
                "External RR accuracy of a frozen thermal-video algorithm with "
                "algorithmic quality control, safe gating, and uncertainty reporting."
            ),
            "excluded_claims": (
                "No heat-stress, THI-stratified, head-motion, occlusion, or "
                "nostril-visibility stratified claims."
            ),
            "evidence": (
                f"q2_algorithmic_external_validation={algorithmic_q2}; "
                f"external_validation_ready={external_ready}; "
                f"algorithmic_quality_ready={gate_status(gates, 'algorithmic_quality_ready')}; "
                f"conformal_uncertainty_ready={gate_status(gates, 'conformal_uncertainty_ready')}"
            ),
            "paper_use": "preferred_route_when_environment_and_quality_scores_are_unavailable",
        },
        {
            "route": "thermal_physiology_or_heat_stress_route",
            "current_status": "not_supported_by_current_metadata",
            "required_next_gate": "biological_context_ready and q2_target_absolute_validation",
            "main_claim_allowed_now": "No.",
            "main_claim_after_next_gate": (
                "Thermal physiology or THI-stratified RR interpretation only if "
                "real temperature/humidity or THI metadata are collected."
            ),
            "excluded_claims": "All heat-stress and THI claims until real environment metadata exist.",
            "evidence": (
                f"biological_context_ready={gate_status(gates, 'biological_context_ready')}; "
                f"q2_target_absolute_validation={full_q2}"
            ),
            "paper_use": "do_not_use_for_current_submission",
        },
        {
            "route": "manual_quality_robustness_route",
            "current_status": "not_supported_by_current_metadata",
            "required_next_gate": "manual_quality_scores_ready",
            "main_claim_allowed_now": "No.",
            "main_claim_after_next_gate": (
                "Robustness stratified by manually reviewed head motion, occlusion, "
                "and nostril visibility only if those scores are produced."
            ),
            "excluded_claims": "All manual quality-stratified robustness claims.",
            "evidence": f"manual_quality_scores_ready={gate_status(gates, 'manual_quality_scores_ready')}",
            "paper_use": "do_not_use_for_current_submission",
        },
    ]
    return pd.DataFrame(routes)


def build_claims(routes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_family": "internal_rr_accuracy",
                "status": "allowed_now_with_internal_only_caveat",
                "recommended_wording": (
                    "On the current 73-video internal set, the frozen signal-aware "
                    "safe-gate pipeline improved RR agreement over the default "
                    "thermal RR workflow."
                ),
                "must_not_say": "Do not call this an external or cross-farm result.",
            },
            {
                "claim_family": "truth_calibrated_rr_r2",
                "status": "upper_bound_only",
                "recommended_wording": (
                    "Truth-calibrated RR was used only as an oracle upper bound "
                    "showing that the temperature curves retain respiratory signal."
                ),
                "must_not_say": "Do not report truth-calibrated R2 as the primary model performance.",
            },
            {
                "claim_family": "algorithmic_quality_control",
                "status": "allowed_now_as_internal_engineering_evidence",
                "recommended_wording": (
                    "Algorithmic quality tiers, deployment decision curves, and "
                    "conformal intervals define when the pipeline should auto-report "
                    "or defer to review."
                ),
                "must_not_say": "Do not present algorithmic quality tiers as manual visual quality scores.",
            },
            {
                "claim_family": "external_validation",
                "status": "not_allowed_until_external_rows_are_scored",
                "recommended_wording": (
                    "The method is frozen and the external validation protocol is "
                    "specified; independent external data remain required."
                ),
                "must_not_say": "Do not call the current 73 videos external validation.",
            },
            {
                "claim_family": "heat_stress_or_thi",
                "status": "not_allowed",
                "recommended_wording": (
                    "The collection location and date window can be reported as "
                    "provenance, but heat-stress interpretation was not evaluated."
                ),
                "must_not_say": "Do not infer per-video THI from regional weather or notes.",
            },
            {
                "claim_family": "manual_quality_stratification",
                "status": "not_allowed",
                "recommended_wording": (
                    "Manual head-motion, occlusion, and nostril-visibility scores "
                    "were unavailable; quality-stratified robustness was not claimed."
                ),
                "must_not_say": "Do not replace manual scores with model-derived risk scores.",
            },
        ]
    )


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    if data.empty:
        return "_No rows available._"
    view = data[[column for column in columns if column in data.columns]].fillna("").astype(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, routes: pd.DataFrame, claims: pd.DataFrame) -> None:
    preferred = routes[routes["route"].eq("q2_algorithmic_engineering_route")]
    status = str(preferred.iloc[0]["current_status"]) if not preferred.empty else "unknown"
    text = f"""# RR Claim-Scope Strategy

Current preferred route: `q2_algorithmic_engineering_route`

Current preferred-route status: `{status}`

This file exists because synchronized environment metadata and manual quality
scores are unavailable. It narrows the manuscript to claims that can be
supported without fabricating those fields.

## Submission Routes

{markdown_table(routes, [
    'route',
    'current_status',
    'required_next_gate',
    'main_claim_allowed_now',
    'main_claim_after_next_gate',
    'excluded_claims',
    'evidence',
    'paper_use',
])}

## Claim Wording

{markdown_table(claims, ['claim_family', 'status', 'recommended_wording', 'must_not_say'])}

## Practical Decision

Use the algorithmic engineering route for a Q2-or-higher target if independent
external RR videos can be collected. Do not pursue the thermal-physiology route
unless real temperature/humidity or THI metadata become available.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results = read_optional(output_dir / "paper_main_results_table.csv")
    gates = read_optional(output_dir / "paper_target_journal_gate_matrix.csv")
    tiers = read_optional(output_dir / "paper_external_validation_fieldwork_tier_status.csv")
    external_split = read_optional(args.input_root / f"{args.output_prefix}_external_split_readiness.csv")

    routes = build_routes(main_results, gates, tiers, external_split)
    claims = build_claims(routes)

    routes_path = output_dir / "paper_claim_scope_routes.csv"
    claims_path = output_dir / "paper_claim_scope_allowed_claims.csv"
    report_path = output_dir / "paper_claim_scope_strategy.md"
    routes.to_csv(routes_path, index=False)
    claims.to_csv(claims_path, index=False)
    write_report(report_path, routes, claims)

    print(f"Saved claim-scope routes: {routes_path}")
    print(f"Saved claim-scope allowed claims: {claims_path}")
    print(f"Saved claim-scope strategy: {report_path}")
    print(routes[["route", "current_status", "required_next_gate"]].to_string(index=False))


if __name__ == "__main__":
    main()
