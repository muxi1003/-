from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript-ready wording for the thermal RR algorithmic "
            "engineering route when environment metadata and manual visual "
            "quality scores are unavailable."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=repo_root / "docs")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row(table: pd.DataFrame, column: str, contains: str) -> pd.Series:
    if table.empty or column not in table.columns:
        return pd.Series(dtype=object)
    mask = table[column].astype(str).str.contains(contains, case=False, regex=False)
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def first_exact(table: pd.DataFrame, column: str, value: str) -> pd.Series:
    if table.empty or column not in table.columns:
        return pd.Series(dtype=object)
    mask = table[column].astype(str) == value
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def int_text(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def total_videos(row: pd.Series) -> str:
    if row.empty:
        return ""
    if "videos" in row and str(row.get("videos", "")).strip():
        return int_text(row.get("videos"))
    return ""


def exact_text(row: pd.Series) -> str:
    if row.empty:
        return ""
    raw_exact = str(row.get("exact_count", "")).strip()
    if "/" in raw_exact:
        return raw_exact
    exact = int_text(raw_exact)
    total = total_videos(row)
    return f"{exact}/{total}" if exact and total else exact


def within_one_text(row: pd.Series) -> str:
    if row.empty:
        return ""
    raw_within = str(row.get("within_one_count", "")).strip()
    if "/" in raw_within:
        return raw_within
    within = int_text(raw_within)
    total = total_videos(row)
    return f"{within}/{total}" if within and total else within


def metric_text(row: pd.Series) -> str:
    if row.empty:
        return "metrics unavailable"
    mae = row.get("rr_mae_bpm", row.get("rr_mae", ""))
    rmse = row.get("rr_rmse_bpm", row.get("rr_rmse", ""))
    parts = [
        f"RR R2={fmt(row.get('rr_r2'), 4)}",
        f"MAE={fmt(mae, 3)} breaths/min",
        f"RMSE={fmt(rmse, 3)} breaths/min",
        f"exact={exact_text(row)}",
        f"within-one={within_one_text(row)}",
    ]
    return ", ".join(part for part in parts if part.split("=")[-1])


def rollback_stats_text(row: pd.Series) -> str:
    if row.empty:
        return "paired statistics unavailable"
    return (
        f"paired delta RR R2={fmt(row.get('delta_rr_r2'), 4)} "
        f"(95% bootstrap CI {fmt(row.get('delta_rr_r2_ci_low'), 4)} to "
        f"{fmt(row.get('delta_rr_r2_ci_high'), 4)}), "
        f"delta exact={int_text(row.get('delta_exact_count'))}, "
        f"McNemar one-sided p={fmt(row.get('mcnemar_exact_p_method_better'), 3)}"
    )


def triage_metrics_text(table: pd.DataFrame) -> str:
    if table.empty:
        return "RR-only physiological triage not generated"

    def row(label: str) -> pd.Series:
        return first_exact(table, "label", label)

    auto = row("auto_report_subset")
    high_auto = row("auto_high_rr_candidate")
    upper = row("predicted_upper_quartile_rr_candidate")
    if auto.empty:
        return "RR-only physiological triage generated but auto-report row missing"
    text = (
        f"RR-only auto-report triage selected {int_text(auto.get('videos'))}/73 "
        f"low-risk videos (coverage={fmt(auto.get('coverage'), 3)}) with "
        f"RR R2={fmt(auto.get('rr_r2'), 4)}, MAE={fmt(auto.get('rr_mae'), 3)} "
        f"breaths/min, and exact rate={fmt(auto.get('exact_rate'), 3)}"
    )
    if not high_auto.empty:
        text += (
            f"; the auto-report high-RR candidate subset contained "
            f"{int_text(high_auto.get('videos'))} videos with "
            f"RR R2={fmt(high_auto.get('rr_r2'), 4)}, "
            f"MAE={fmt(high_auto.get('rr_mae'), 3)} breaths/min, and "
            f"exact rate={fmt(high_auto.get('exact_rate'), 3)}"
        )
    if not upper.empty:
        text += (
            f"; all predicted upper-quartile RR candidates contained "
            f"{int_text(upper.get('videos'))} videos and should be treated as "
            "context-review candidates rather than heat-stress diagnoses"
        )
    return text + "."


def bilateral_metrics_text(table: pd.DataFrame) -> str:
    if table.empty:
        return "bilateral nostril consistency gate not generated"
    row = first_exact(table, "subset", "bilateral_consistent_auto_report_subset")
    if row.empty:
        return "bilateral nostril consistency gate generated but auto-report row missing"
    return (
        f"bilateral nostril consistency screening retained "
        f"{int_text(row.get('videos'))}/73 auto-report videos "
        f"(coverage={fmt(row.get('coverage'), 3)}) with "
        f"RR R2={fmt(row.get('rr_r2'), 4)}, "
        f"MAE={fmt(row.get('rr_mae_bpm'), 3)} breaths/min, "
        f"RMSE={fmt(row.get('rr_rmse_bpm'), 3)} breaths/min, and "
        f"exact rate={fmt(row.get('exact_rate'), 3)}."
    )


def status_for_gate(table: pd.DataFrame, gate: str) -> str:
    row = first_exact(table, "gate", gate)
    if row.empty:
        return "MISSING"
    return str(row.get("status", "MISSING")).upper()


def status_for_tier(table: pd.DataFrame, tier: str) -> str:
    row = first_exact(table, "collection_tier", tier)
    if row.empty:
        row = first_exact(table, "tier", tier)
    if row.empty:
        return "MISSING"
    return str(row.get("status", row.get("recommended_videos", "MISSING")))


def acceptance_rule(acceptance: pd.DataFrame, claim_type: str, metric: str) -> str:
    if acceptance.empty:
        return ""
    mask = (
        acceptance.get("claim_type", pd.Series("", index=acceptance.index))
        .astype(str)
        .eq(claim_type)
        & acceptance.get("metric", pd.Series("", index=acceptance.index))
        .astype(str)
        .eq(metric)
    )
    if not mask.any():
        return ""
    return str(acceptance.loc[mask].iloc[0].get("acceptance_rule", ""))


def first_strategy(strategy: pd.DataFrame) -> pd.Series:
    row = first_exact(
        strategy,
        "route",
        "algorithmic_engineering_route_no_heat_quality_claims",
    )
    if not row.empty:
        return row
    if strategy.empty:
        return pd.Series(dtype=object)
    return strategy.iloc[0]


def build_claim_table(
    main_results: pd.DataFrame,
    external_acceptance: pd.DataFrame,
    target_strategy: pd.DataFrame,
    split_acceptance: pd.DataFrame,
    rollback_metrics: pd.DataFrame,
    rollback_stats: pd.DataFrame,
    triage_summary: pd.DataFrame,
    bilateral_summary: pd.DataFrame,
) -> pd.DataFrame:
    default = first_row(main_results, "method", "Default thermal RR pipeline")
    safe_gate = first_row(
        main_results,
        "method",
        "Conservative signal-aware safe gate (fixed threshold",
    )
    safe_group = first_row(
        main_results,
        "method",
        "Conservative signal-aware safe gate (leave-one-prefix",
    )
    truth = first_row(main_results, "method", "Truth-calibrated upper bound")
    algorithmic_gate = first_exact(
        split_acceptance,
        "gate",
        "algorithmic engineering external claim allowed",
    )
    route = first_strategy(target_strategy)
    rollback_fixed = first_exact(
        rollback_metrics,
        "label",
        "consensus_rollback_probe_fixed",
    )
    rollback_fixed_stats = first_exact(
        rollback_stats,
        "validation_mode",
        "fixed",
    )

    return pd.DataFrame(
        [
            {
                "claim_family": "current_internal_algorithmic_rr_accuracy",
                "status": "allowed_now_internal_only",
                "paper_location": "Results and limitations",
                "recommended_wording": (
                    "Report the current 73-video set as internal validation. "
                    f"Default workflow: {metric_text(default)}. Conservative "
                    f"signal-aware safe gate: {metric_text(safe_gate)}."
                ),
                "do_not_write": "Do not call this external, prospective, cross-farm, or deployment-ready validation.",
            },
            {
                "claim_family": "prefix_group_stress_evidence",
                "status": "allowed_now_as_internal_stress_test",
                "paper_location": "Secondary robustness results",
                "recommended_wording": (
                    "Use prefix-group validation as an internal domain-risk stress "
                    f"test only: {metric_text(safe_group)}."
                ),
                "do_not_write": "Do not call prefix groups true cow-level, camera-level, date-level, or farm-level validation.",
            },
            {
                "claim_family": "consensus_rollback_guard",
                "status": "exploratory_internal_only",
                "paper_location": "Supplementary error analysis",
                "recommended_wording": (
                    "A narrow consensus rollback guard can be reported as an "
                    "exploratory internal probe for rare harmful residual "
                    f"corrections: {metric_text(rollback_fixed)}; "
                    f"{rollback_stats_text(rollback_fixed_stats)}."
                ),
                "do_not_write": (
                    "Do not present the rollback guard as the frozen primary "
                    "method, as statistically confirmed superiority, or as "
                    "externally validated precision improvement."
                ),
            },
            {
                "claim_family": "truth_calibrated_rr_r2",
                "status": "upper_bound_only",
                "paper_location": "Supplementary diagnostic, not abstract",
                "recommended_wording": (
                    "Truth-calibrated RR can be described as an oracle upper "
                    f"bound showing residual respiratory information in the "
                    f"curves: {metric_text(truth)}."
                ),
                "do_not_write": "Do not use truth-calibrated R2 as the main method R2 or compare it as a deployable model.",
            },
            {
                "claim_family": "algorithmic_engineering_external_validation",
                "status": str(algorithmic_gate.get("status", "FAIL")).upper()
                if not algorithmic_gate.empty
                else "FAIL",
                "paper_location": "Future external validation or main table only after PASS",
                "recommended_wording": (
                    "After independent external scoring, claim a frozen "
                    "algorithmic RR validation only if the 104-video algorithmic "
                    "tier and all predefined RR gates pass."
                ),
                "do_not_write": (
                    "Do not use the current 73 internal videos as external "
                    "validation. Current target route status: "
                    f"{route.get('journal_strategy_status', 'unknown')}."
                ),
            },
            {
                "claim_family": "heat_stress_thi_or_environment",
                "status": "not_allowed_with_current_metadata",
                "paper_location": "Limitations only",
                "recommended_wording": (
                    "Report Heilongjiang, Lindian County ranch and the "
                    "2023-08-05 to 2023-08-10 collection window as provenance "
                    "context only."
                ),
                "do_not_write": "Do not infer per-video temperature, humidity, THI, ATHI, or heat-stress strata from location/date context.",
            },
            {
                "claim_family": "manual_quality_stratification",
                "status": "not_allowed_with_current_metadata",
                "paper_location": "Limitations only",
                "recommended_wording": (
                    "State that manual head-motion, occlusion, and nostril-"
                    "visibility scores were unavailable."
                ),
                "do_not_write": "Do not replace manual scores with algorithmic risk scores.",
            },
            {
                "claim_family": "algorithmic_quality_and_uncertainty",
                "status": "allowed_now_internal_engineering_evidence",
                "paper_location": "Methods, Results, Discussion",
                "recommended_wording": (
                    "Algorithmic quality tiers, selective reporting, safe gating, "
                    "and conformal RR intervals may be reported as internal "
                    "engineering controls that must be frozen before external use."
                ),
                "do_not_write": "Do not describe these controls as clinically or prospectively validated reliability.",
            },
            {
                "claim_family": "rr_only_physiological_triage",
                "status": "allowed_now_internal_review_alert_evidence",
                "paper_location": "Methods, Results, Discussion, Supplement",
                "recommended_wording": triage_metrics_text(triage_summary),
                "do_not_write": (
                    "Do not describe high-RR candidate flags as heat-stress, "
                    "disease, welfare, or treatment labels without synchronized "
                    "environment and animal-context metadata."
                ),
            },
            {
                "claim_family": "bilateral_nostril_consistency_gate",
                "status": "allowed_now_internal_selective_reporting_evidence",
                "paper_location": "Methods, Results, Discussion, Supplement",
                "recommended_wording": bilateral_metrics_text(bilateral_summary),
                "do_not_write": (
                    "Do not describe bilateral consistency as external validation, "
                    "heat-stress diagnosis, welfare classification, disease "
                    "detection, or true animal-identity evidence."
                ),
            },
        ]
    )


def build_manuscript_text(
    main_results: pd.DataFrame,
    acceptance: pd.DataFrame,
    collection: pd.DataFrame,
    execution: pd.DataFrame,
    target_strategy: pd.DataFrame,
    split_acceptance: pd.DataFrame,
    freeze: pd.DataFrame,
    deployment: pd.DataFrame,
    conformal: pd.DataFrame,
    rollback_metrics: pd.DataFrame,
    rollback_stats: pd.DataFrame,
    triage_summary: pd.DataFrame,
    bilateral_summary: pd.DataFrame,
    claim_table: pd.DataFrame,
) -> str:
    default = first_row(main_results, "method", "Default thermal RR pipeline")
    safe_gate = first_row(
        main_results,
        "method",
        "Conservative signal-aware safe gate (fixed threshold",
    )
    safe_group = first_row(
        main_results,
        "method",
        "Conservative signal-aware safe gate (leave-one-prefix",
    )
    truth = first_row(main_results, "method", "Truth-calibrated upper bound")
    route = first_strategy(target_strategy)
    algorithmic_split_gate = first_exact(
        split_acceptance,
        "gate",
        "algorithmic engineering external claim allowed",
    )
    q2_tier = first_exact(collection, "collection_tier", "q2_algorithmic_external_validation")
    q2_execution = first_exact(execution, "gate", "q2_algorithmic_external_validation_fieldwork")
    freeze_row = freeze.iloc[0] if not freeze.empty else pd.Series(dtype=object)
    deploy_row = deployment.iloc[0] if not deployment.empty else pd.Series(dtype=object)
    conformal_row = conformal.iloc[0] if not conformal.empty else pd.Series(dtype=object)
    rollback_fixed = first_exact(
        rollback_metrics,
        "label",
        "consensus_rollback_probe_fixed",
    )
    rollback_group = first_exact(
        rollback_metrics,
        "label",
        "consensus_rollback_probe_prefix_group",
    )
    rollback_fixed_stats = first_exact(
        rollback_stats,
        "validation_mode",
        "fixed",
    )
    rollback_group_stats = first_exact(
        rollback_stats,
        "validation_mode",
        "prefix_group",
    )

    r2_rule = acceptance_rule(
        acceptance,
        "algorithmic_engineering_external_performance",
        "RR R2",
    )
    mae_rule = acceptance_rule(
        acceptance,
        "algorithmic_engineering_external_performance",
        "MAE",
    )
    rmse_rule = acceptance_rule(
        acceptance,
        "algorithmic_engineering_external_performance",
        "RMSE",
    )
    exact_rule = acceptance_rule(
        acceptance,
        "algorithmic_engineering_external_performance",
        "exact count agreement",
    )

    auto_n = int_text(deploy_row.get("auto_videos", ""))
    review_n = int_text(deploy_row.get("review_videos", ""))
    deploy_metrics = ""
    if not deploy_row.empty:
        deploy_metrics = (
            f"At the internal risk <= {fmt(deploy_row.get('risk_threshold'), 2)} "
            f"operating point, {auto_n}/73 videos were assigned to automatic "
            f"reporting and {review_n}/73 to review; the auto-report subset "
            f"reached RR R2={fmt(deploy_row.get('rr_r2'), 4)}, "
            f"MAE={fmt(deploy_row.get('rr_mae'), 3)} breaths/min, and "
            f"exact={int_text(deploy_row.get('exact_count'))}/"
            f"{int_text(deploy_row.get('count_valid_videos'))}."
        )
    conformal_metrics = ""
    if not conformal_row.empty:
        conformal_metrics = (
            f"The internal risk-adaptive conformal interval targeted "
            f"{fmt(conformal_row.get('target_coverage'), 2)} coverage, achieved "
            f"coverage={fmt(conformal_row.get('coverage'), 4)}, and had mean "
            f"width={fmt(conformal_row.get('mean_width_bpm'), 3)} breaths/min; "
            f"low-risk coverage was {fmt(conformal_row.get('low_risk_coverage'), 4)}."
        )
    triage_metrics = triage_metrics_text(triage_summary)
    bilateral_metrics = bilateral_metrics_text(bilateral_summary)
    rollback_metrics_text = ""
    if not rollback_fixed.empty and not rollback_group.empty:
        rollback_metrics_text = (
            "A narrow consensus rollback guard was also evaluated as an "
            "exploratory precision supplement after the safe gate. In fixed "
            f"out-of-fold validation it reached {metric_text(rollback_fixed)}, "
            f"and under prefix-group stress it reached {metric_text(rollback_group)}. "
            f"The paired fixed comparison gave {rollback_stats_text(rollback_fixed_stats)}; "
            f"the prefix-group comparison gave {rollback_stats_text(rollback_group_stats)}. "
            "The rollback-specific bootstrap confidence interval touches zero, "
            "so this is a promising internal error-analysis signal rather than "
            "statistically confirmed superiority. "
            "Because this rule was derived from internal error analysis and rescues "
            "a rare harmful correction pattern, it should be treated as a "
            "supplementary candidate until frozen and externally validated."
        )

    claim_lines = [
        "| claim_family | status | paper_location | recommended_wording | do_not_write |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in claim_table.iterrows():
        claim_lines.append(
            "| "
            + " | ".join(
                str(row[column]).replace("\n", " ")
                for column in [
                    "claim_family",
                    "status",
                    "paper_location",
                    "recommended_wording",
                    "do_not_write",
                ]
            )
            + " |"
        )

    text = f"""# Thermal RR Algorithmic Engineering Manuscript Insert

This file is generated from the current paper assets. It is designed for the
algorithmic engineering route in which environment metadata and manual visual
quality scores cannot be recovered. Use it to update the manuscript without
turning location/date context into heat-stress or quality-stratified evidence.

## Proposed Framing

The feasible manuscript route is a method-frozen agricultural sensing study of
thermal-video respiratory-rate estimation with algorithmic quality control,
safe gating, selective reporting, and uncertainty reporting. The recommended
target-route record is `{route.get('route', 'unknown')}`, with current status
`{route.get('journal_strategy_status', 'unknown')}` and missing must-have gates
`{route.get('missing_must_have_gates', 'unknown')}`. The contribution should be
presented as an internally validated algorithmic improvement plus a frozen
external-validation protocol, not as heat-stress monitoring, THI modeling, or
manual visual-quality robustness.

## Methods Insert

The current dataset contains 73 thermal videos collected at a ranch in Lindian
County, Heilongjiang, China, during 2023-08-05 to 2023-08-10. The numeric video
identifier was copied into `cow_id` as a dataset-level label, and
`external_test_split` was set to `internal` for all current videos. In this
workflow, `external_test_split` is a provenance label that separates internal
development/evaluation videos from future method-frozen external or holdout
videos; it is not a quality score and it does not make the present 73 videos an
external test set. Per-video barn temperature, humidity, THI/ATHI, camera ID,
exact acquisition time, and manual head-motion, occlusion, or nostril-visibility
scores were unavailable. Therefore, location and collection-window information
were retained only as provenance context.

The algorithmic route evaluates a default thermal RR pipeline and non-truth
residual-correction extensions. The conservative signal-aware safe gate applies
only prediction-time signal and model-consistency criteria and does not use the
reference RR, reference breath count, or truth-calibrated review parameters when
making a prediction. The frozen method manifest currently reports
`freeze_status={freeze_row.get('freeze_status', 'unknown')}` and
`method_freeze_id={freeze_row.get('method_freeze_id', 'unknown')}`. Under this
claim boundary, truth-calibrated RR is retained only as an oracle diagnostic
for residual signal availability and is excluded from the main method row,
abstract performance claim, and external-validation claim.

## Results Insert

On the current 73-video internal set, the default thermal RR pipeline achieved
{metric_text(default)}. The conservative signal-aware safe gate achieved the
strongest internal non-truth result, with {metric_text(safe_gate)}. Under the
prefix-group internal stress test, the same safe-gate candidate reached
{metric_text(safe_group)}. These values support an internal precision and
robustness claim, but they do not yet establish external generalization because
the external split has no scored external rows.

The reliability layer can be described as an internal engineering control.
{deploy_metrics} {conformal_metrics} {triage_metrics} {bilateral_metrics} These
analyses should be reported as internal selective-reporting, RR-only
review/alert, bilateral signal-consistency, and uncertainty experiments whose
thresholds must remain frozen before external evaluation. The
truth-calibrated upper bound
reached {metric_text(truth)}, but this number uses reference information and
should appear only as a supplementary oracle diagnostic, not as the headline RR
R2.

{rollback_metrics_text}

## External Validation Insert

The next manuscript gate for the no-environment, no-manual-quality route is the
`q2_algorithmic_external_validation` tier. The current collection plan specifies
{q2_tier.get('recommended_videos', '104')} external videos with required fields
`{q2_tier.get('required_metadata', 'cow_id, collection_date, camera_id, scene_id, external_test_split, manual RR truth')}`.
Its design rule is: {q2_tier.get('design_rule', 'freeze the method before external scoring')}.
The execution dashboard currently marks
`q2_algorithmic_external_validation_fieldwork` as
`{q2_execution.get('status', 'MISSING')}`, and the external split gate
`algorithmic engineering external claim allowed` is
`{algorithmic_split_gate.get('status', 'MISSING')}` with threshold
`{algorithmic_split_gate.get('threshold', 'external_n >= 104 + method frozen + predefined gates PASS')}`.
Before this gate passes, the paper may state that the external protocol is
specified and the method is frozen, but it must not report an external
performance claim.

The predefined external acceptance rules for the algorithmic engineering route
are: {r2_rule}; {mae_rule}; {rmse_rule}; and {exact_rule}. Passing this tier
would support an external algorithmic RR accuracy claim for the frozen
safe-gate workflow, including algorithmic quality control and uncertainty
reporting. It would still not support heat-stress, THI-stratified,
head-motion-stratified, occlusion-stratified, or nostril-visibility-stratified
claims unless those real metadata fields are later collected.

## Discussion Insert

The main distinction for the paper is between an RR algorithmic claim and a
biological heat-stress claim. The former can be pursued with independent
external videos, manual RR truth, cow/session/camera provenance, and a frozen
method. The latter requires real synchronized temperature and humidity or THI
measurements and, for visual-quality robustness, manual head-motion, occlusion,
and nostril-visibility scores. Because those fields are unavailable in the
current dataset, the manuscript should explicitly state that the study evaluates
thermal-video RR estimation and reliability triage rather than validated
heat-stress monitoring. The Heilongjiang/Lindian County collection context is
safe to report as provenance, but it should not be converted into per-video
environment exposure or quality labels.

## Claim Boundary Table

{chr(10).join(claim_lines)}
"""
    return text


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = args.docs_dir.resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    main_results = read_optional(output_dir / "paper_main_results_table.csv")
    acceptance = read_optional(output_dir / "paper_external_validation_acceptance_criteria.csv")
    collection = read_optional(output_dir / "paper_external_validation_collection_tiers.csv")
    execution = read_optional(output_dir / "paper_external_validation_execution_dashboard.csv")
    target_strategy = read_optional(output_dir / "paper_target_journal_strategy.csv")
    split_acceptance = read_optional(args.input_root / f"{args.output_prefix}_external_split_acceptance.csv")
    freeze = read_optional(output_dir / "paper_method_freeze_summary.csv")
    deployment = read_optional(output_dir / "paper_deployment_decision_recommendation_table.csv")
    conformal = read_optional(output_dir / "paper_conformal_rr_recommendation_table.csv")
    triage_summary = read_optional(output_dir / "paper_rr_physiological_triage_summary.csv")
    bilateral_summary = read_optional(output_dir / "paper_rr_bilateral_consistency_summary.csv")
    rollback_metrics = read_optional(output_dir / "paper_consensus_rollback_probe_metrics.csv")
    rollback_stats = read_optional(
        output_dir / "paper_consensus_rollback_probe_statistical_tests.csv"
    )

    claim_table = build_claim_table(
        main_results,
        acceptance,
        target_strategy,
        split_acceptance,
        rollback_metrics,
        rollback_stats,
        triage_summary,
        bilateral_summary,
    )
    text = build_manuscript_text(
        main_results,
        acceptance,
        collection,
        execution,
        target_strategy,
        split_acceptance,
        freeze,
        deployment,
        conformal,
        rollback_metrics,
        rollback_stats,
        triage_summary,
        bilateral_summary,
        claim_table,
    )

    asset_md = output_dir / "paper_algorithmic_engineering_manuscript_insert.md"
    docs_md = docs_dir / "thermal_rr_algorithmic_engineering_insert.md"
    claim_csv = output_dir / "paper_algorithmic_engineering_claim_table.csv"
    asset_md.write_text(text, encoding="utf-8")
    docs_md.write_text(text, encoding="utf-8")
    claim_table.to_csv(claim_csv, index=False)

    print(f"Saved algorithmic engineering insert: {asset_md}")
    print(f"Saved docs insert: {docs_md}")
    print(f"Saved claim table: {claim_csv}")
    print(claim_table[["claim_family", "status", "paper_location"]].to_string(index=False))


if __name__ == "__main__":
    main()
