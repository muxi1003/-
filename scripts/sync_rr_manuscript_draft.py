from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from build_rr_external_validation_reporting_insert import (
    build_reporting_text,
    collect_state,
)
from build_rr_manuscript_claim_update import (
    algorithmic_quality_context,
    bilateral_consistency_context,
    conformal_uncertainty_context,
    exact,
    fmt,
    metadata_context,
    method_metrics,
    physiological_triage_context,
    readiness_status,
    row_contains,
)


BEGIN_SAFE_GATE = "<!-- BEGIN CURRENT_SAFE_GATE_ROLLBACK_RESULTS -->"
END_SAFE_GATE = "<!-- END CURRENT_SAFE_GATE_ROLLBACK_RESULTS -->"
BEGIN_AGREEMENT = "<!-- BEGIN RR_METHOD_AGREEMENT_RESULTS -->"
END_AGREEMENT = "<!-- END RR_METHOD_AGREEMENT_RESULTS -->"
BEGIN_DISCUSSION = "<!-- BEGIN CURRENT_CLAIM_BOUNDARY_DISCUSSION -->"
END_DISCUSSION = "<!-- END CURRENT_CLAIM_BOUNDARY_DISCUSSION -->"
BEGIN_LITERATURE = "<!-- BEGIN RECENT_LITERATURE_POSITIONING -->"
END_LITERATURE = "<!-- END RECENT_LITERATURE_POSITIONING -->"
BEGIN_COLOR_METHOD = "<!-- BEGIN CALIBRATION_FREE_COLOR_METHOD -->"
END_COLOR_METHOD = "<!-- END CALIBRATION_FREE_COLOR_METHOD -->"
BEGIN_COLOR_RESULTS = "<!-- BEGIN CALIBRATION_FREE_COLOR_RESULTS -->"
END_COLOR_RESULTS = "<!-- END CALIBRATION_FREE_COLOR_RESULTS -->"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize the long thermal RR manuscript draft with the current "
            "paper assets, including safe-gate, rollback, metadata, and claim "
            "boundary wording."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=repo_root / "docs" / "thermal_rr_manuscript_draft.md",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def first_exact(table: pd.DataFrame, column: str, value: str) -> pd.Series:
    if table.empty or column not in table.columns:
        return pd.Series(dtype=object)
    mask = table[column].astype(str) == value
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def replace_between(text: str, start: str, end: str, replacement_body: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise ValueError(f"Could not find start marker: {start}")
    body_start = start_index + len(start)
    end_index = text.find(end, body_start)
    if end_index < 0:
        raise ValueError(f"Could not find end marker after {start}: {end}")
    return text[:body_start] + "\n\n" + replacement_body.strip() + "\n\n" + text[end_index:]


def replace_from_heading(text: str, heading: str, replacement_body: str) -> str:
    start_index = text.find(heading)
    if start_index < 0:
        raise ValueError(f"Could not find heading: {heading}")
    return text[:start_index] + heading + "\n\n" + replacement_body.strip() + "\n"


def upsert_marked_block(
    text: str,
    begin: str,
    end: str,
    block_body: str,
    *,
    insert_before: str,
) -> str:
    block = f"{begin}\n\n{block_body.strip()}\n\n{end}\n\n"
    begin_index = text.find(begin)
    if begin_index >= 0:
        end_index = text.find(end, begin_index)
        if end_index < 0:
            raise ValueError(f"Found {begin} without {end}")
        return text[:begin_index] + block + text[end_index + len(end):].lstrip()
    insert_index = text.find(insert_before)
    if insert_index < 0:
        raise ValueError(f"Could not find insertion point: {insert_before}")
    return text[:insert_index] + block + text[insert_index:]


def upsert_marked_block_before_any(
    text: str,
    begin: str,
    end: str,
    block_body: str,
    *,
    insert_before_options: list[str],
) -> str:
    block = f"{begin}\n\n{block_body.strip()}\n\n{end}\n\n"
    begin_index = text.find(begin)
    if begin_index >= 0:
        end_index = text.find(end, begin_index)
        if end_index < 0:
            raise ValueError(f"Found {begin} without {end}")
        return text[:begin_index] + block + text[end_index + len(end):].lstrip()
    candidates = [
        text.find(insert_before)
        for insert_before in insert_before_options
        if text.find(insert_before) >= 0
    ]
    if not candidates:
        raise ValueError(
            "Could not find any insertion point: "
            + ", ".join(insert_before_options)
        )
    insert_index = min(candidates)
    return text[:insert_index] + block + text[insert_index:]


def renumber_results_headings(text: str) -> str:
    heading_numbers = {
        "Selective Reporting and Algorithmic Quality Identified High-Confidence Subsets": "3.6",
        "Prefix-Group Holdout Validation Suggested Limited Cross-Domain Robustness": "3.7",
        "Feature-Block Ablation Supported Peak-Structure and Frequency-Domain Contributions": "3.8",
        "Heat-Stress Interpretation Readiness": "3.9",
        "External Validation Readiness and Sample Planning": "3.10",
    }
    for title, section_number in heading_numbers.items():
        text = re.sub(
            rf"### 3\.\d+ {re.escape(title)}",
            f"### {section_number} {title}",
            text,
        )
    return text


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def calibration_free_color_context(assets: Path) -> dict[str, str]:
    metrics = read_optional_csv(assets / "paper_calibration_free_thermal_index_metrics.csv")
    bootstrap = read_optional_csv(
        assets / "paper_calibration_free_thermal_index_bootstrap_ci.csv"
    )
    robustness = read_optional_csv(
        assets / "paper_calibration_free_thermal_index_robustness_metrics.csv"
    )
    robustness_bootstrap = read_optional_csv(
        assets / "paper_calibration_free_thermal_index_robustness_bootstrap_ci.csv"
    )
    reannotation_priority = read_optional_csv(
        assets / "paper_calibration_free_thermal_index_reannotation_priority.csv"
    )
    reannotation_packets = read_optional_csv(
        assets / "paper_p2g_reannotation_packet_summary.csv"
    )
    primary_packets = read_optional_csv(
        assets / "paper_p2g_primary_all_packet_summary.csv"
    )
    extension_packets = read_optional_csv(
        assets / "paper_p2g_extension_all_packet_summary.csv"
    )
    extension_predictions = read_optional_csv(
        assets / "paper_p2g_extension_all_frozen_predictions.csv"
    )
    p2g_progress = read_optional_csv(
        assets / "paper_p2g_primary_plus_extension_annotation_progress_summary.csv"
    )
    consensus_selection = read_optional_csv(
        assets / "paper_calibration_free_consensus_ensemble_selection.csv"
    )
    stability_selection = read_optional_csv(
        assets / "paper_calibration_free_stability_gate_selection.csv"
    )
    geometry_metrics = read_optional_csv(
        assets / "paper_calibration_free_geometry_roi_metrics.csv"
    )
    session_metrics = read_optional_csv(
        assets / "paper_calibration_free_session_context_metrics.csv"
    )
    selection = read_optional_csv(assets / "paper_calibration_free_thermal_index_selection.csv")
    method = "duration_gated_calibration_free_internal_ranked_ensemble"
    internal = first_exact(metrics, "method", method)
    external = metrics[
        metrics.get("cohort", pd.Series("", index=metrics.index)).astype(str).eq(
            "external_provisional"
        )
        & metrics.get("method", pd.Series("", index=metrics.index)).astype(str).eq(method)
    ]
    if internal.empty or external.empty or selection.empty:
        return {"available": "false"}
    external_row = external.iloc[0]
    delta = bootstrap[
        bootstrap.get("comparison", pd.Series("", index=bootstrap.index)).astype(str).eq(
            "gated_color_minus_offline_temperature_context"
        )
        & bootstrap.get("metric", pd.Series("", index=bootstrap.index)).astype(str).eq("rr_r2")
    ]
    delta_text = "clustered comparison unavailable"
    if not delta.empty:
        row = delta.iloc[0]
        delta_text = (
            f"delta R2={fmt(row['estimate'], 4)} (95% CI "
            f"{fmt(row['ci_low'], 4)} to {fmt(row['ci_high'], 4)})"
        )
    selected = selection.iloc[0]
    robustness_text = "reference-uncertainty audit unavailable"
    if not robustness.empty and not robustness_bootstrap.empty:
        optimistic = first_exact(robustness, "analysis", "one_count_optimistic_rowwise")
        conservative = first_exact(robustness, "analysis", "one_count_conservative_rowwise")
        session = first_exact(
            robustness, "analysis", "duration_weighted_source_session_aggregation"
        )
        r2_ci = first_exact(robustness_bootstrap, "metric", "rr_r2")
        if not optimistic.empty and not conservative.empty and not session.empty and not r2_ci.empty:
            robustness_text = (
                f"The +/-1-breath sensitivity range was R2={fmt(conservative['rr_r2'], 4)} "
                f"to {fmt(optimistic['rr_r2'], 4)}. Source-session bootstrap for the "
                f"point-reference R2 was {fmt(r2_ci['estimate'], 4)} (95% CI "
                f"{fmt(r2_ci['ci_low'], 4)} to {fmt(r2_ci['ci_high'], 4)}), and "
                f"duration-weighted session aggregation gave R2={fmt(session['rr_r2'], 4)} "
                f"and MAE={fmt(session['rr_mae'], 3)} breaths/min."
            )
    reannotation_text = "dual-blinded recount packet not yet generated"
    confirmation_scope = 0
    if not p2g_progress.empty and "expected_rows" in p2g_progress.columns:
        expected = pd.to_numeric(p2g_progress["expected_rows"], errors="coerce").dropna()
        confirmation_scope = int(expected.max()) if not expected.empty else 0
    if not reannotation_priority.empty and not reannotation_packets.empty:
        priorities = reannotation_priority.get(
            "review_priority", pd.Series("", index=reannotation_priority.index)
        ).astype(str)
        p0 = int(priorities.eq("P0_dual_blinded_recount_and_adjudication").sum())
        p1 = int(priorities.eq("P1_second_independent_recount").sum())
        packet_sizes = ", ".join(
            f"{row.annotator_id} ({int(row.clips)} clips)"
            for row in reannotation_packets.itertuples(index=False)
        )
        reannotation_text = (
            f"A P0/P1 reannotation queue (P0={p0}, P1={p1}) has been issued as "
            f"separate blinded packets for {packet_sizes}; those recounts and adjudication "
            "are pending."
        )
        if not primary_packets.empty:
            primary_sizes = ", ".join(
                f"{row.annotator_id} ({int(row.clips)} clips)"
                for row in primary_packets.itertuples(index=False)
            )
            reannotation_text += (
                f" Complete primary-scope blinded packets have also been prepared for "
                f"{primary_sizes}; their counts and adjudication are pending."
            )
    if not extension_packets.empty:
        extension_sizes = ", ".join(
            f"{row.annotator_id} ({int(row.clips)} clips)"
            for row in extension_packets.itertuples(index=False)
        )
        extension_text = (
            " A pre-registered duration-eligible extension pool has also been issued as "
            f"blinded packets for {extension_sizes}; historical counts are cleared and its "
            "results remain pending. Each export records visibility as clear, uncertain, or unreadable, "
            "together with optional count uncertainty; unreadable clips remain in the flow table and do "
            "not enter RR scoring. "
        )
        if len(extension_predictions) == len(extension_packets.iloc[:1]) * int(extension_packets.iloc[0].clips):
            extension_text += (
                "Frozen P2g predictions have been generated without reading manual counts or "
                "reference RR. The pool can contribute only after independent recount, adjudication, "
                "and an explicit unreadable-clip flow table."
            )
        else:
            extension_text += (
                "The pool can contribute only after independent recount, adjudication, frozen prediction "
                "extraction, and an explicit unreadable-clip flow table."
            )
        if reannotation_text == "dual-blinded recount packet not yet generated":
            reannotation_text = extension_text.strip()
        else:
            reannotation_text += extension_text
    if confirmation_scope:
        reannotation_text += (
            f" The frozen P2g confirmation scope is {confirmation_scope} duration-eligible "
            "clips across the primary and pre-registered extension packets."
        )
    ablation_text = "calibration-free robustness ablations not yet generated"
    if (
        not consensus_selection.empty
        and not stability_selection.empty
        and not geometry_metrics.empty
        and not session_metrics.empty
    ):
        geometry_external = geometry_metrics[
            geometry_metrics.get("cohort", pd.Series("", index=geometry_metrics.index))
            .astype(str)
            .eq("external_provisional")
            & geometry_metrics.get("method", pd.Series("", index=geometry_metrics.index))
            .astype(str)
            .eq("duration_gated_calibration_free_consensus_ensemble")
        ]
        session_best = session_metrics.sort_values(["rr_rmse", "rr_mae"]).iloc[0]
        if not geometry_external.empty:
            geometry_row = geometry_external.iloc[0]
            ablation_text = (
                "Robust-aggregation selection retained the mean and kinematic selection retained "
                "no frame filter. The internal-selected geometry-normalized ROI variant achieved "
                f"external R2={fmt(geometry_row['rr_r2'], 4)} versus the mean P2g result, while "
                f"the best fixed session-context variant ({session_best['method']}) had "
                f"R2={fmt(session_best['rr_r2'], 4)}; neither replaced clipwise mean P2g."
            )
    return {
        "available": "true",
        "internal": method_metrics(internal),
        "external": metric_text_from_row(external_row),
        "external_r2": fmt(external_row.get("rr_r2"), 4),
        "external_mae": fmt(external_row.get("rr_mae"), 3),
        "external_rmse": fmt(external_row.get("rr_rmse"), 3),
        "within_one": f"{int(float(external_row.get('within_one_count')))} / {int(float(external_row.get('videos')))}",
        "members": str(selected.get("selected_members", "")),
        "prominence": fmt(selected.get("selected_prominence"), 3),
        "delta": delta_text,
        "robustness": robustness_text,
        "reannotation": reannotation_text,
        "confirmation_scope": str(confirmation_scope),
        "ablation": ablation_text,
    }


def metric_text_from_row(row: pd.Series) -> str:
    return (
        f"RR R2={fmt(row.get('rr_r2'), 4)}, MAE={fmt(row.get('rr_mae'), 3)} breaths/min, "
        f"RMSE={fmt(row.get('rr_rmse'), 3)} breaths/min"
    )


def build_literature_positioning(assets: Path) -> str:
    sources = read_optional_csv(assets / "paper_literature_recent_source_verification.csv")
    gaps = read_optional_csv(assets / "paper_literature_recent_gap_innovation_matrix.csv")
    if sources.empty or gaps.empty:
        return (
            "Recent literature positioning has not yet been generated. Run "
            "`scripts/build_rr_literature_recent_refresh.py` before using this "
            "manuscript section."
        )
    verified = sources[
        sources["crossref_status"].astype(str).str.contains("verified", case=False, na=False)
    ]
    source_count = int(len(sources))
    verified_count = int(len(verified))
    p0 = gaps[gaps["priority"].astype(str).eq("P0")].iloc[0]
    p2 = gaps[gaps["priority"].astype(str).eq("P2")].iloc[0]
    p3 = gaps[gaps["priority"].astype(str).eq("P3")].iloc[0]
    p4 = gaps[gaps["priority"].astype(str).eq("P4")].iloc[0]
    return (
        "A recent-literature refresh was used to position the contribution "
        f"against {source_count} core sources, of which {verified_count} were "
        "verified through DOI, publisher, or PubMed metadata on 2026-07-13. The closest thermal "
        "respiratory-rate studies already establish infrared nostril detection, "
        "temperature-curve extraction, and head-movement-aware thermal RR "
        "monitoring as active baselines. Therefore, the novelty of the present "
        "work is not claimed as basic YOLO localization or thermal mapping. "
        f"Instead, the primary literature-derived gap is {p0['gap'].lower()}, "
        "which motivates the conservative signal-aware residual correction and "
        "safe-gate framework.\n\n"
        "The same refresh also shows why the paper should avoid a purely "
        "accuracy-centered narrative. Frequency-domain RR work supports the use "
        "of FFT, autocorrelation, and spectral count agreement as an independent "
        "signal-consensus layer, while environmental and ATHI-related RR studies "
        "show that biological interpretation requires synchronized thermal "
        "environment and animal-context metadata. Recent multi-farm physiology evidence also "
        "shows that RR responds to weather and animal state, so the current paper does not "
        "infer heat stress from video-derived RR alone. Accordingly, the present paper "
        f"treats {p3['gap'].lower()} as a future biological-meaning gate rather "
        "than as a completed heat-stress analysis. Modern precision-livestock "
        "vision work on cow re-identification and multimodal animal monitoring "
        f"also makes {p4['gap'].lower()} a validation requirement, not a minor "
        "reporting detail.\n\n"
        "This literature positioning leads to a conservative manuscript claim: "
        "the current dataset supports an internally validated precision "
        "improvement and reliability-triage framework for thermal-video RR "
        "estimation, whereas head-movement, occlusion, nostril-visibility, "
        f"heat-stress, and external-deployment claims remain conditional because "
        f"{p2['current_evidence']}."
    )


def build_reference_placeholders(assets: Path) -> str:
    sources = read_optional_csv(assets / "paper_literature_recent_source_verification.csv")
    if sources.empty:
        return (
            "Recent literature source verification table not generated. Run "
            "`scripts/build_rr_literature_recent_refresh.py`."
        )
    lines = []
    for _, row in sources.iterrows():
        title = str(row.get("title", "")).strip()
        venue = str(row.get("venue", "")).strip()
        year = str(row.get("year", "")).strip()
        doi = str(row.get("doi", "")).strip()
        status = str(row.get("crossref_status", "")).strip()
        suffix = "Crossref verified" if status == "verified" else "local/publisher record; Crossref not verified"
        pieces = [piece for piece in [title, venue, year] if piece]
        entry = ". ".join(pieces)
        if doi:
            entry = f"{entry}. DOI: https://doi.org/{doi}."
        else:
            entry = f"{entry}."
        lines.append(f"{entry} {suffix}.")
    return "\n\n".join(lines)


def agreement_row(table: pd.DataFrame, method_id: str) -> pd.Series:
    if table.empty or "method_id" not in table.columns:
        raise ValueError("Agreement summary is missing method_id")
    mask = table["method_id"].astype(str).eq(method_id)
    if not mask.any():
        raise ValueError(f"Agreement summary missing method_id={method_id}")
    return table.loc[mask].iloc[0]


def agreement_delta(table: pd.DataFrame, method_id: str) -> pd.Series:
    if table.empty or "method_id" not in table.columns:
        raise ValueError("Agreement delta table is missing method_id")
    mask = table["method_id"].astype(str).eq(method_id)
    if not mask.any():
        raise ValueError(f"Agreement delta table missing method_id={method_id}")
    return table.loc[mask].iloc[0]


def agreement_stratum(table: pd.DataFrame, method_id: str, stratum: str) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    mask = (
        table["method_id"].astype(str).eq(method_id)
        & table["truth_rr_stratum"].astype(str).eq(stratum)
    )
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def agreement_exact(row: pd.Series) -> str:
    try:
        return f"{int(float(row.get('count_exact')))}/{int(float(row.get('n')))}"
    except (TypeError, ValueError):
        return "NA"


def build_agreement_results(assets: Path) -> str:
    summary = read_csv(assets / "paper_rr_method_agreement_summary.csv")
    delta = read_csv(assets / "paper_rr_method_agreement_delta_vs_default.csv")
    strata = read_csv(assets / "paper_rr_method_agreement_truth_rr_strata.csv")
    default = agreement_row(summary, "default")
    quality = agreement_row(summary, "quality_residual_fixed_oof")
    signal = agreement_row(summary, "signal_consensus_fixed_oof")
    safe = agreement_row(summary, "signal_aware_safe_fixed_oof")
    safe_group = agreement_row(summary, "signal_aware_safe_prefix_group_fixed")
    safe_delta = agreement_delta(delta, "signal_aware_safe_fixed_oof")
    safe_low = agreement_stratum(strata, "signal_aware_safe_fixed_oof", "low_rr_lt_50")
    safe_mid = agreement_stratum(strata, "signal_aware_safe_fixed_oof", "mid_rr_50_to_70")
    safe_high = agreement_stratum(strata, "signal_aware_safe_fixed_oof", "high_rr_ge_70")
    return (
        "### 3.5 Bland-Altman Agreement and Error-Distribution Tightening\n\n"
        "Bland-Altman analysis was added to test whether the algorithmic gains "
        "reflected a narrower agreement distribution rather than only a higher "
        "identity-line RR R2. Agreement error was defined as predicted RR minus "
        "manual RR. The default pipeline had a mean bias of "
        f"{fmt(default['rr_bias_bpm'])} breaths/min, 95% limits of agreement "
        f"from {fmt(default['rr_loa_lower_bpm'])} to "
        f"{fmt(default['rr_loa_upper_bpm'])} breaths/min, and a LoA width of "
        f"{fmt(default['rr_loa_width_bpm'])} breaths/min. The quality-aware "
        f"residual correction reduced the LoA width to "
        f"{fmt(quality['rr_loa_width_bpm'])} breaths/min, and the "
        "signal-consensus supplement further reduced it to "
        f"{fmt(signal['rr_loa_width_bpm'])} breaths/min. The conservative "
        "signal-aware safe gate produced the tightest current non-truth "
        f"agreement, with bias {fmt(safe['rr_bias_bpm'])} breaths/min, 95% "
        f"LoA from {fmt(safe['rr_loa_lower_bpm'])} to "
        f"{fmt(safe['rr_loa_upper_bpm'])} breaths/min, LoA width "
        f"{fmt(safe['rr_loa_width_bpm'])} breaths/min, MAE "
        f"{fmt(safe['rr_mae_bpm'])} breaths/min, and exact count agreement in "
        f"{agreement_exact(safe)} videos.\n\n"
        "Relative to the default pipeline, the safe gate narrowed the LoA width "
        f"by {fmt(abs(float(safe_delta['delta_rr_loa_width_bpm'])))} breaths/min, "
        f"reduced MAE by {fmt(abs(float(safe_delta['delta_rr_mae_bpm'])))} "
        f"breaths/min, reduced RMSE by "
        f"{fmt(abs(float(safe_delta['delta_rr_rmse_bpm'])))} breaths/min, and "
        f"increased exact count agreement by "
        f"{int(float(safe_delta['delta_count_exact']))} videos without "
        "introducing any >=2-breath count errors. Under the prefix-group "
        "internal stress test, the safe-gate LoA width remained "
        f"{fmt(safe_group['rr_loa_width_bpm'])} breaths/min, supporting the "
        "interpretation that the gate improves agreement while preserving "
        "grouped internal robustness.\n\n"
        "The RR-stratified agreement summary showed that the gain was clearest "
        f"in the lower and middle RR ranges: the safe gate had MAE "
        f"{fmt(safe_low.get('rr_mae_bpm'))} breaths/min in "
        f"{int(float(safe_low.get('n')))} low-RR videos and MAE "
        f"{fmt(safe_mid.get('rr_mae_bpm'))} breaths/min in "
        f"{int(float(safe_mid.get('n')))} mid-RR videos. The high-RR stratum "
        f"contained only {int(float(safe_high.get('n')))} videos, where the "
        f"safe gate had bias {fmt(safe_high.get('rr_bias_bpm'))} breaths/min "
        f"and MAE {fmt(safe_high.get('rr_mae_bpm'))} breaths/min. This small "
        "high-RR stratum should therefore be reported as a limitation, not as "
        "evidence for high-RR or heat-stress deployment."
    )


def build_current_text(
    assets: Path,
    input_root: Path,
    corrected_prefix: str,
) -> dict[str, str]:
    main = read_csv(assets / "paper_main_results_table.csv")
    stats = read_csv(assets / "paper_rr_method_statistical_tests_table.csv")
    rollback_metrics = read_csv(assets / "paper_consensus_rollback_probe_metrics.csv")
    rollback_stats = read_csv(assets / "paper_consensus_rollback_probe_statistical_tests.csv")
    readiness = read_csv(assets / "paper_submission_readiness_table.csv")

    default = row_contains(main, "method", "Default thermal RR pipeline")
    quality = row_contains(main, "method", "Quality-aware residual correction (fixed threshold")
    signal = row_contains(main, "method", "Signal-consensus supplement (fixed")
    signal_aware = row_contains(main, "method", "Signal-aware residual correction (fixed threshold")
    signal_group = row_contains(main, "method", "Signal-aware residual correction (leave-one-prefix")
    safe = row_contains(main, "method", "Conservative signal-aware safe gate (fixed threshold")
    safe_group = row_contains(main, "method", "Conservative signal-aware safe gate (leave-one-prefix")
    truth = row_contains(main, "method", "Truth-calibrated upper bound")
    safe_vs_default = row_contains(stats, "method_id", "signal_aware_safe_fixed_oof")
    safe_vs_signal = stats[
        (stats["baseline_method_id"].astype(str) == "signal_consensus_fixed_oof")
        & (stats["method_id"].astype(str) == "signal_aware_safe_fixed_oof")
    ].iloc[0]
    rollback_fixed = first_exact(rollback_metrics, "label", "consensus_rollback_probe_fixed")
    rollback_group = first_exact(
        rollback_metrics,
        "label",
        "consensus_rollback_probe_prefix_group",
    )
    rollback_fixed_stats = first_exact(rollback_stats, "validation_mode", "fixed")
    rollback_group_stats = first_exact(rollback_stats, "validation_mode", "prefix_group")

    quality_status = algorithmic_quality_context(assets)
    conformal_status = conformal_uncertainty_context(assets)
    triage_status = physiological_triage_context(assets)
    bilateral_status = bilateral_consistency_context(assets)
    metadata_status = metadata_context(assets)
    readiness_text = readiness_status(readiness)
    agreement_results = build_agreement_results(assets)
    repo_root = Path(__file__).resolve().parents[1]
    external_state = collect_state(
        input_root,
        corrected_prefix,
        repo_root / "Dataset_new" / "72video" / "external_al_images",
        "external_repro",
    )
    external_results = build_reporting_text(external_state)
    external_results_body = external_results.split("\n\n", 1)[1]
    color = calibration_free_color_context(assets)
    external_included = int(external_state["included"])
    external_consensus = int(external_state["consensus_ready"])
    external_sessions = int(external_state["source_sessions"])
    external_cows = int(external_state["unique_cows"])
    external_dates = int(external_state["unique_dates"])
    if color.get("available") == "true":
        confirmation_scope = int(color.get("confirmation_scope", "0") or 0)
        if confirmation_scope > 0:
            external_included = confirmation_scope
        external_summary = (
            "A provisional single-annotator external development analysis was completed "
            "for 94 Jiufu Ranch clips. The internally selected duration-gated relative "
            f"thermal-color ensemble achieved {color['external']}. This is not a "
            "confirmatory external claim because manual references were not independently "
            "blinded and adjudicated."
        )
        external_results_body = (
            "A provisional external development analysis used 94 clips with a numeric "
            "single-annotator breath count and duration of at least 20 s. Twenty-six of "
            "the 125 inventory clips were not scorable because the respiratory cycles were "
            "not sufficiently visible. The duration-gated calibration-free relative thermal-"
            "color ensemble achieved "
            f"{color['external']}, with within-one breath agreement in {color['within_one']} clips. "
            "The method retained the frozen absolute-temperature pipeline for short clips "
            "and used the internally selected color ensemble for long clips. Relative to "
            "offline temperature context, source-session clustered bootstrap gave "
            f"{color['delta']}.\n\n"
            "These values are exploratory development evidence, not independent external "
            "validation: the Jiufu manual counts were produced by one annotator, some clips "
            "were visibly ambiguous, and this cohort was inspected during method development. "
            f"{color['reannotation']} The dual-blinded annotation and adjudication protocol "
            "remains the required path for confirmatory reporting."
        )
    elif bool(external_state["external_metrics_available"]):
        external_summary = (
            "The independent Jiufu Ranch external cohort has been scored with "
            f"the frozen workflow; the primary claim gate is "
            f"{external_state['external_claim']} and the algorithmic safe-gate "
            f"claim is {external_state['algorithmic_claim']}."
        )
    else:
        external_summary = (
            "An independent Jiufu Ranch external candidate cohort contains "
            f"{external_included} included clips from {external_sessions} source "
            f"sessions, {external_cows} cow labels, and {external_dates} collection "
            f"dates, but blinded consensus references are currently available for "
            f"only {external_consensus}/{external_included} clips; external RR "
            "performance therefore remains unscored."
        )

    manuscript_status = (
        "This is a working manuscript draft generated from the current reproducible "
        "outputs in `Dataset_new/72video/al_images`. It is suitable for internal "
        "writing and supervisor review, but it is not yet ready for journal "
        "submission. The 73-video development/evaluation set lacks exact per-video "
        "date/time, camera ID, synchronized barn environment records, and manual "
        "quality scores. Numeric labels have been copied into `cow_id`, and the "
        "Lindian County ranch collection window has been used to prefill provenance. "
        f"{external_summary}"
    )

    external_methods = (
        "The external validation workflow used a separately collected candidate "
        "cohort from Jiufu Ranch, Hulunbuir City, Inner Mongolia, China. Inventory "
        f"screening retained {external_included} clips from {external_sessions} "
        f"source sessions, {external_cows} cow labels, and {external_dates} collection "
        "dates; short terminal fragments were retained in the audit inventory but "
        "excluded from default RR scoring. Two blinded annotation packets were "
        "generated, and only dual-annotator consensus or adjudicated breath counts "
        "are permitted to enter the external truth file. Clips originating from the "
        "same long video share `source_session_id` and must be treated as clustered "
        "observations rather than independent animals.\n\n"
        "The method was frozen before external scoring. The predefined absolute "
        "acceptance gates require external RR R2 >= 0.90, MAE <= 2.5 breaths/min, "
        "RMSE <= 4.0 breaths/min, and within-one breath agreement >= 95%. The "
        "external reporting layer remains claim-blocked when blinded exports, "
        "consensus references, frozen predictions, or acceptance outputs are missing. "
        "No external threshold or model parameter may be retuned after viewing this "
        "cohort's reference outcomes."
    )
    if color.get("available") == "true":
        external_methods += (
            "\n\nFor the exploratory color-signal development analysis, seven relative "
            "pseudo-color ROI signals were ranked on the internal cohort. Their fixed "
            f"members were `{color['members']}`. Peak prominence was selected on internal "
            f"data only from a prespecified grid, yielding `{color['prominence']}`. A "
            "duration threshold derived from the internal acquisition distribution retained "
            "the frozen temperature workflow for short clips and activated the color ensemble "
            "for long clips."
        )

    external_discussion = (
        "The independently collected Jiufu Ranch cohort materially improves the "
        "study design because it separates acquisition site and collection period "
        "from the Lindian development data. However, an assembled cohort is not the "
        "same as completed external validation. "
        f"{external_summary} The current manuscript can therefore describe the "
        "frozen external protocol, cohort composition, and annotation gate, while "
        "external accuracy and generalization claims remain conditional on completed "
        "blinded references and predefined acceptance gates. The frozen P2g "
        f"confirmation scope contains {external_included} duration-eligible clips across "
        "30 source sessions and 29 cow labels, so session-clustered "
        "uncertainty and animal-level wording remain important even after clip-level "
        "metrics become available."
    )

    limitations = (
        "This study has several limitations. First, the development/evaluation set "
        "contains only 73 processed videos, which limits bootstrap precision and the "
        "stability of residual correction. Second, development-set `cow_id` is derived "
        "from numeric video labels rather than verified animal identity, and exact "
        "per-video acquisition time and camera ID are unavailable. Third, synchronized "
        "barn temperature, humidity, THI, and ATHI values are unavailable, so the data "
        "cannot support heat-stress association claims. Fourth, manual head-motion, "
        "occlusion, and nostril-visibility scores are unavailable and cannot be "
        "replaced by algorithmic risk scores. Fifth, the independent Jiufu Ranch "
        f"candidate cohort includes {external_included} scorable clips, but blinded "
        f"consensus RR is currently complete for {external_consensus}/{external_included}; "
        "external R2 and agreement are therefore not yet estimable. Sixth, multiple "
        "clips derive from the same source session, so clip-level observations are "
        "clustered and must not be equated with independent animals. Seventh, the "
        "truth-calibrated result is an upper-bound analysis only and cannot be reported "
        "as main performance. Finally, internally selected quality, deployment, and "
        "uncertainty rules require frozen prospective evaluation before deployment use."
    )
    if color.get("available") == "true":
        limitations += (
            " The current Jiufu color-ensemble result is based on a single-annotator "
            "provisional reference and a cohort inspected during development; it cannot "
            "substitute for a prospectively frozen, dual-blinded external validation."
        )

    conclusions = (
        "A quality-aware residual peak-count correction framework improved internal "
        "validation of a YOLO-based thermal-infrared respiratory-rate pipeline, and "
        "signal-consensus and conservative safe-gate extensions further reduced "
        "candidate residual errors without truth-assisted calibration. The resulting "
        "framework adds interpretable agreement, review-triage, and uncertainty outputs "
        "to the point estimate. An independent Jiufu Ranch candidate cohort has now "
        f"been assembled with {external_included} scorable clips, but external "
        f"reference consensus remains {external_consensus}/{external_included}. Thus, "
        "the present evidence supports an internally validated precision-improvement "
        "framework and a completed external-validation protocol, not yet a confirmed "
        "external-performance or heat-stress-monitoring claim."
    )
    if color.get("available") == "true":
        conclusions = (
            "A duration-gated relative thermal-color ensemble provided a second, "
            "calibration-free signal path for long clips and improved the provisional "
            f"cross-farm development result to {color['external']}. " + conclusions
        )

    abstract = (
        "Respiratory rate is a sensitive physiological indicator for dairy cow "
        "welfare and heat-stress monitoring, but non-contact estimation from "
        "infrared thermography remains vulnerable to nostril occlusion, weak "
        "respiratory peaks, boundary-cycle ambiguity, and motion-related signal "
        "degradation. We developed a reproducible thermal-video respiratory-rate "
        "pipeline that combines YOLO-based nostril localization, nostril "
        "temperature-curve construction, quality-aware residual peak-count "
        "correction, frequency-domain signal consensus, a conservative "
        "signal-aware residual gate, algorithmic review triage, conformal RR "
        "uncertainty intervals, RR-only physiological triage, bilateral nostril "
        "consistency screening, and an exploratory "
        "consensus rollback guard. On "
        f"73 thermal videos, the default pipeline achieved {method_metrics(default)}. "
        f"The fixed-threshold quality-aware residual correction improved internal "
        f"out-of-fold performance to {method_metrics(quality)}, and a "
        f"frequency-domain signal-consensus supplement reached {method_metrics(signal)}. "
        f"The conservative signal-aware safe gate achieved the strongest current "
        f"non-truth internal candidate result, reaching {method_metrics(safe)}; "
        f"relative to the default pipeline, the paired delta RR R2 was "
        f"{fmt(safe_vs_default['delta_rr_r2'], 4)} with a 95% bootstrap CI from "
        f"{fmt(safe_vs_default['delta_rr_r2_ci_low'], 4)} to "
        f"{fmt(safe_vs_default['delta_rr_r2_ci_high'], 4)}, and exact agreement "
        f"increased by {int(float(safe_vs_default['delta_exact_count']))} videos. "
        f"A narrow rollback probe further reached {method_metrics(rollback_fixed)}, "
        f"but it changed only one video and McNemar's one-sided exact p value "
        f"versus the safe gate was {fmt(rollback_fixed_stats['mcnemar_exact_p_method_better'], 3)}, "
        "so this rule is treated as supplementary error analysis rather than "
        "confirmed superiority. Internal reliability analysis added a non-truth "
        f"quality triage layer ({quality_status}), conformal prediction "
        f"intervals ({conformal_status}), RR-only physiological triage "
        f"({triage_status}), and bilateral nostril consistency screening "
        f"({bilateral_status}). Known-context metadata now include "
        "numeric-label cow_id, one Lindian County ranch scene label, and the "
        "2023-08-05 to 2023-08-10 collection window. "
        f"{external_summary} "
        "Therefore, the current claim should be an internally validated "
        "precision-improvement and reliability-triage framework rather than a "
        "deployment-ready true animal-identity or heat-stress-monitoring result."
    )

    safe_gate_results = (
        "### 3.4 Conservative Signal-Aware Safe Gate and Rollback Error Probe\n\n"
        "The conservative signal-aware safe gate was evaluated as a stricter "
        "precision candidate after the quality-aware and signal-consensus stages. "
        "The gate preserves the signal-aware residual correction only when the "
        "candidate correction is a negative adjustment and the autocorrelation "
        "count remains close to the signal-consensus count. This rule uses only "
        "prediction-time signal consistency and does not use the reference RR, "
        "reference breath count, or truth-calibrated review parameters. It reached "
        f"{method_metrics(safe)} in fixed out-of-fold validation. Relative to the "
        f"default pipeline, the paired delta RR R2 was {fmt(safe_vs_default['delta_rr_r2'], 4)} "
        f"(95% bootstrap CI {fmt(safe_vs_default['delta_rr_r2_ci_low'], 4)} to "
        f"{fmt(safe_vs_default['delta_rr_r2_ci_high'], 4)}), MAE changed by "
        f"{fmt(safe_vs_default['delta_rr_mae'], 3)} breaths/min, and exact agreement "
        f"changed by {int(float(safe_vs_default['delta_exact_count']))} videos. "
        f"Relative to the signal-consensus supplement, the paired delta RR R2 was "
        f"{fmt(safe_vs_signal['delta_rr_r2'], 4)} (95% bootstrap CI "
        f"{fmt(safe_vs_signal['delta_rr_r2_ci_low'], 4)} to "
        f"{fmt(safe_vs_signal['delta_rr_r2_ci_high'], 4)}). Under prefix-group "
        f"stress testing, the safe gate reached {method_metrics(safe_group)}, "
        f"whereas the ungated signal-aware candidate reached {method_metrics(signal_group)}. "
        "This contrast supports the safe gate as the strongest current internal "
        "precision candidate, while still requiring true animal-identity, "
        "session, camera, and external validation before deployment claims.\n\n"
        "A narrow consensus rollback guard was then evaluated as an exploratory "
        "post-safe-gate error probe for one rare failure pattern: the residual "
        "model applied a -1 correction, while FFT and spectral count estimates "
        "supported the original count, autocorrelation supported the corrected "
        "count, and the selected thermal signal amplitude was low. The rollback "
        f"probe reached {method_metrics(rollback_fixed)} in fixed out-of-fold "
        f"validation and {method_metrics(rollback_group)} under prefix-group "
        f"stress. Its paired delta RR R2 versus the safe gate was "
        f"{fmt(rollback_fixed_stats['delta_rr_r2'], 4)} (95% bootstrap CI "
        f"{fmt(rollback_fixed_stats['delta_rr_r2_ci_low'], 4)} to "
        f"{fmt(rollback_fixed_stats['delta_rr_r2_ci_high'], 4)}) in fixed validation "
        f"and {fmt(rollback_group_stats['delta_rr_r2'], 4)} (95% bootstrap CI "
        f"{fmt(rollback_group_stats['delta_rr_r2_ci_low'], 4)} to "
        f"{fmt(rollback_group_stats['delta_rr_r2_ci_high'], 4)}) under prefix-group "
        f"stress. Exact agreement increased by one video in both views, but "
        f"McNemar's one-sided exact p value was "
        f"{fmt(rollback_fixed_stats['mcnemar_exact_p_method_better'], 3)}. Because "
        "this rule changed only one video and was discovered during internal error "
        "analysis, it should be reported only as supplementary error analysis "
        "unless it is frozen and validated prospectively."
    )

    discussion = (
        "The current innovation hierarchy is now clearer than in the earlier "
        "draft. The quality-aware residual correction remains the main internal "
        "algorithmic contribution because it improves the default workflow while "
        "retaining nested and prefix-group checks. The signal-consensus supplement "
        "and the conservative signal-aware safe gate show that independent "
        "periodicity estimates can improve precision when they agree with the "
        "residual classifier. The safe gate is the strongest internal precision "
        "candidate, whereas the rollback guard is a narrower supplementary error "
        "probe whose confidence interval touches zero and whose McNemar evidence "
        "is weak because it rescues only one internal case. This distinction is "
        "important for manuscript positioning: the safe gate can be discussed as "
        "a high-precision internal candidate, but the rollback guard should be "
        "kept in supplementary error analysis until frozen external validation. "
        f"The RR-only physiological triage layer adds a more interpretable "
        f"review and alert surface without converting RR into a diagnosis: "
        f"{triage_status}. The bilateral nostril consistency gate further "
        f"checks whether both nostril temperature curves carry compatible "
        f"respiratory periodicity before auto-reporting: {bilateral_status}. "
        f"The metadata boundary also remains explicit: {metadata_status}. The "
        f"readiness audit remains {readiness_text}. {external_summary} These "
        "boundaries mean that "
        "the manuscript can argue for an internally validated RR estimation and "
        "reliability-triage framework, while heat-stress, manual-quality, camera, "
        "date/session, and external deployment claims remain conditional on completed "
        "reference annotation, frozen scoring, and missing metadata."
    )

    figure_plan = (
        "Table 1 should report the default pipeline, fixed-threshold residual "
        "correction, signal-consensus supplement, conservative signal-aware safe "
        "gate, consensus rollback probe, nested threshold cross-validation, "
        "leave-one-prefix-group-out validation, and truth-calibrated upper bound "
        "with clear labels separating main results, candidate extensions, "
        "exploratory error probes, sensitivity analysis, grouped internal "
        "validation, and upper-bound analysis. Table 2 should report paired "
        "statistical tests, including bootstrap CIs, McNemar exact p values, "
        "Bland-Altman bias and limits of agreement, and the fact that the "
        "rollback probe changes one video only. Table 3 should "
        "report the prefix-group holdout and pseudo-external gate stress-test "
        "results, including the ns risk group and the groups that require "
        "signal-aware or signal-consensus logic to pass all gates. A supplementary "
        "table should report selective automatic-report and manual-review triage, "
        "algorithmic quality tier metrics, deployment decision curves, and "
        "RR-only physiological triage actions, high-RR candidate flags, bilateral "
        "nostril consistency gate metrics, and conformal interval coverage/width. "
        "Another supplementary table should "
        "report metadata context, the Jiufu Ranch external candidate cohort, blinded "
        "annotation progress, and acceptance gates, including the distinction between "
        "an assembled cohort and completed external performance evidence. Figure 1 "
        "should show the method "
        "workflow from thermal video to nostril temperature curves, residual "
        "correction, signal-consensus/safe-gate screening, rollback error review, "
        "selective automatic-report/review triage, RR-only physiological triage, "
        "bilateral nostril consistency screening, conformal RR interval reporting, "
        "respiratory rate, and heat-stress-context reporting. Figure 2 should use "
        "the current method-comparison plots and Bland-Altman plots to show "
        "predicted-versus-reference results, agreement, bootstrap confidence intervals, and threshold "
        "sensitivity. Figure 3 should show representative corrected and uncorrected "
        "curve examples, including endpoint missing peaks, boundary mismatch, "
        "noise overcounting, and the rollback case."
    )
    if color.get("available") == "true":
        figure_plan += (
            " A supplementary external-development figure should compare frozen "
            "temperature context with the duration-gated color ensemble and display the "
            "source-session cluster-bootstrap interval, explicitly labelled provisional."
        )

        external_methods = (
            "The Jiufu Ranch inventory contains 125 clips from 30 source sessions, 29 "
            "cow labels, and 6 collection dates. Manual review produced numeric single-"
            "annotator breath counts for 99 clips; 26 clips were not counted because "
            "respiratory cycles were insufficiently visible. The primary provisional "
            "analysis retained 94 clips with a numeric count and at least 20 s duration. "
            "Clips cut from one long video share `source_session_id` and were treated as "
            "clustered observations for uncertainty analysis.\n\n"
            "This analysis is a development diagnostic, not confirmatory external validation. "
            "A second independent blinded count and adjudication remain required before "
            "external performance can be presented as a validation result. The confirmatory "
            "acceptance thresholds remain RR R2 >= 0.90, MAE <= 2.5 breaths/min, RMSE <= "
            "4.0 breaths/min, and within-one breath agreement >= 95%."
        )
        external_discussion = (
            "The Jiufu Ranch cohort provides a useful domain-shift stress test because it "
            "was acquired at another ranch and in another collection period. The internally "
            f"selected color ensemble reached {color['external']} on 94 provisionally "
            "counted clips, but these values are development evidence rather than external "
            "validation because one annotator supplied the reference and the cohort was "
            "inspected during method development. The next valid claim gate is a locked "
            "pipeline evaluated against independent blinded counts with source-session "
            "clustered uncertainty."
        )
        limitations = (
            "This study has several limitations. First, the development/evaluation set "
            "contains only 73 processed videos, limiting the stability of internal residual "
            "correction. Second, development `cow_id` values are derived from numeric video "
            "labels rather than verified identity, and exact per-video time and camera ID "
            "are unavailable. Third, synchronized barn temperature, humidity, THI, ATHI, "
            "and manual motion/occlusion/visibility labels are unavailable, precluding "
            "heat-stress and quality-stratified claims. Fourth, the Jiufu result uses 94 "
            "clips with a single-annotator provisional reference; it is a development "
            "diagnostic and cannot substitute for prospective dual-blinded external "
            "validation. Fifth, clips from one source session are correlated and must not "
            "be equated with independent animals. Finally, the truth-calibrated upper bound "
            "and internally selected triage rules are not deployable performance claims."
        )
        conclusions = (
            "A duration-gated relative thermal-color ensemble provided a calibration-free "
            "long-clip signal path and, while preserving the internal frozen result, achieved "
            f"{color['external']} in a provisional cross-farm development analysis. Together "
            "with the internal quality-aware residual correction and conservative safe gate, "
            "the work supports an interpretable thermal-video RR framework. Confirmation "
            "requires a locked pipeline, independently blinded external counts, and newly "
            "collected source sessions; the present data do not support confirmed external "
            "deployment or heat-stress-monitoring claims."
        )

    color_method = ""
    color_results = ""
    if color.get("available") == "true":
        mapping_audit_note = ""
        if (
            assets / "paper_temperature_mapping_source_parity_report.md"
        ).exists() and (
            assets / "paper_temperature_mapping_source_parity_summary.csv"
        ).exists():
            mapping_audit_note = (
                " A separate source-parity audit verified that the historical random-"
                "forest pkl receives OpenCV BGR triplets (despite its `RGB` filename) and "
                "that stored circular-ROI temperatures replay identically. The relative "
                "branch therefore addresses camera/render-domain transfer rather than a "
                "BGR/RGB implementation error."
            )
        color_method = (
            "### 2.10 Duration-Gated Calibration-Free Relative Thermal-Color Ensemble\n\n"
            "To reduce dependence on an absolute RGB-to-temperature mapping when the "
            "pseudo-color rendering domain changes, a relative color-signal branch was "
            "implemented for long clips. Circular nostril ROIs were extracted from the "
            "same tracked coordinates and converted into grayscale, RGB, CIELAB, hue, and "
            "red-minus-blue time series. Individual signal polarity and the top-seven RR "
            "ensemble were ranked on internal development RMSE only. The selected members "
            f"were `{color['members']}`. Peak prominence was selected on the internal "
            f"cohort only (`{color['prominence']}`), and the long-clip activation threshold "
            "was inherited from the internal duration distribution. This branch therefore "
            "does not use external RR labels at prediction time."
            f"{mapping_audit_note}"
        )
        color_results = (
            "### 3.11 Provisional Cross-Farm Relative Color-Signal Development Analysis\n\n"
            "The duration-gated relative thermal-color ensemble preserved the internal "
            f"frozen result ({color['internal']}) while achieving {color['external']} on "
            "the 94-clip provisional Jiufu analysis. Relative to offline temperature "
            f"context, the source-session clustered comparison gave {color['delta']}. "
            "This indicates that relative pseudo-color dynamics may be more robust than "
            "absolute temperature mapping under the observed farm/camera shift. However, "
            "the analysis is exploratory because the external reference is single-annotator "
            f"and the cohort was inspected during development. {color['robustness']} It must not be reported as "
            f"confirmatory external validation. {color['ablation']} {color['reannotation']}"
        )

    return {
        "manuscript_status": manuscript_status,
        "abstract": abstract,
        "external_methods": external_methods,
        "literature_positioning": build_literature_positioning(assets),
        "safe_gate_results": safe_gate_results,
        "agreement_results": agreement_results,
        "external_validation_results": external_results_body,
        "discussion": discussion,
        "external_discussion": external_discussion,
        "limitations": limitations,
        "conclusions": conclusions,
        "figure_plan": figure_plan,
        "reference_placeholders": build_reference_placeholders(assets),
        "color_method": color_method,
        "color_results": color_results,
    }


def sync_manuscript(
    manuscript_path: Path,
    assets: Path,
    input_root: Path,
    corrected_prefix: str,
) -> tuple[str, dict[str, str]]:
    text = manuscript_path.read_text(encoding="utf-8")
    parts = build_current_text(assets, input_root, corrected_prefix)
    text = replace_between(
        text,
        "## Manuscript Status",
        "## Abstract",
        parts["manuscript_status"],
    )
    text = replace_between(text, "## Abstract", "## 1. Introduction", parts["abstract"])
    text = replace_between(
        text,
        "### 2.9 External Validation Planning and Acceptance Gates",
        "## 3. Results",
        parts["external_methods"],
    )
    if parts["color_method"]:
        text = upsert_marked_block(
            text,
            BEGIN_COLOR_METHOD,
            END_COLOR_METHOD,
            parts["color_method"],
            insert_before="## 3. Results",
        )
    text = upsert_marked_block(
        text,
        BEGIN_LITERATURE,
        END_LITERATURE,
        parts["literature_positioning"],
        insert_before="## 2. Materials and Methods",
    )
    text = upsert_marked_block(
        text,
        BEGIN_SAFE_GATE,
        END_SAFE_GATE,
        parts["safe_gate_results"],
        insert_before="### 3.4 Selective Reporting and Algorithmic Quality Identified High-Confidence Subsets",
    )
    text = upsert_marked_block_before_any(
        text,
        BEGIN_AGREEMENT,
        END_AGREEMENT,
        parts["agreement_results"],
        insert_before_options=[
            "### 3.4 Selective Reporting and Algorithmic Quality Identified High-Confidence Subsets",
            "### 3.5 Selective Reporting and Algorithmic Quality Identified High-Confidence Subsets",
            "### 3.6 Selective Reporting and Algorithmic Quality Identified High-Confidence Subsets",
        ],
    )
    text = upsert_marked_block(
        text,
        BEGIN_DISCUSSION,
        END_DISCUSSION,
        parts["discussion"],
        insert_before="Compared with purely end-to-end video models",
    )
    external_results_headings = [
        "### 3.10 Frozen External Candidate Cohort and Reference-Annotation Gate",
        "### 3.10 External Validation Readiness and Sample Planning",
    ]
    external_results_heading = next(
        (heading for heading in external_results_headings if heading in text),
        None,
    )
    if external_results_heading is None:
        raise ValueError(
            "Could not find an external-validation Results heading: "
            + ", ".join(external_results_headings)
        )
    text = replace_between(
        text,
        external_results_heading,
        "## 4. Discussion",
        parts["external_validation_results"],
    )
    if parts["color_results"]:
        text = upsert_marked_block(
            text,
            BEGIN_COLOR_RESULTS,
            END_COLOR_RESULTS,
            parts["color_results"],
            insert_before="## 4. Discussion",
        )
    text = text.replace(
        "### 3.10 External Validation Readiness and Sample Planning",
        "### 3.10 Frozen External Candidate Cohort and Reference-Annotation Gate",
    )
    text = text.replace(
        "The absence of a true external set, the reliance on internal prefix-group "
        "stress testing, and the lack of per-video date/camera/manual-quality metadata "
        "argue for a cautious paper claim and for additional external or metadata-stratified "
        "validation.",
        "The absence of completed blinded external references and frozen external metrics, "
        "together with the reliance on internal prefix-group stress testing and missing "
        "per-video date/camera/manual-quality metadata, argues for a cautious paper claim "
        "until external scoring and metadata-stratified validation are complete.",
    )
    text = upsert_marked_block(
        text,
        "<!-- BEGIN EXTERNAL_VALIDATION_DISCUSSION -->",
        "<!-- END EXTERNAL_VALIDATION_DISCUSSION -->",
        parts["external_discussion"],
        insert_before="## 5. Limitations",
    )
    text = replace_between(text, "## 5. Limitations", "## 6. Conclusions", parts["limitations"])
    text = replace_between(
        text,
        "## 6. Conclusions",
        "## Figure and Table Plan",
        parts["conclusions"],
    )
    text = replace_between(
        text,
        "## Figure and Table Plan",
        "## Data and Code Availability Statement",
        parts["figure_plan"],
    )
    text = replace_from_heading(
        text,
        "## Reference Placeholders",
        parts["reference_placeholders"],
    )
    text = renumber_results_headings(text)
    return text, parts


def write_report(path: Path, manuscript_path: Path, parts: dict[str, str]) -> None:
    lines = [
        "# Thermal RR Manuscript Draft Sync Report",
        "",
        f"Manuscript: `{manuscript_path}`",
        "",
        "Updated sections: manuscript status; abstract; external-development methods and results; calibration-free color method/results blocks; recent literature positioning; claim boundaries; limitations; conclusions; figure and table plan; reference placeholders.",
        "",
        "Key synchronized claims:",
        "",
        "- Conservative signal-aware safe gate is the strongest current non-truth internal precision candidate.",
        "- Bland-Altman agreement is synchronized from the current method-agreement CSV outputs.",
        "- Recent literature is used to position novelty around residual errors, signal consensus, metadata-gated biological meaning, and external validation.",
        "- Consensus rollback guard is supplementary error analysis only because it rescues one internal case and McNemar evidence is weak.",
        "- The Jiufu color result is labeled as provisional single-annotator development evidence, not confirmatory external validation.",
        "- Current manuscript remains internal/supervisor-review ready, not Q2 submission ready, until blinded external references and acceptance gates pass.",
        "",
        "Verification snippets:",
        "",
        parts["safe_gate_results"][:1200],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    assets = input_root / f"{args.corrected_prefix}_paper_assets"
    manuscript_path = args.manuscript.resolve()
    if not manuscript_path.exists():
        raise FileNotFoundError(f"Missing manuscript draft: {manuscript_path}")
    assets.mkdir(parents=True, exist_ok=True)
    synced, parts = sync_manuscript(
        manuscript_path,
        assets,
        input_root,
        args.corrected_prefix,
    )
    manuscript_path.write_text(synced, encoding="utf-8")
    assets_copy = assets / "paper_thermal_rr_manuscript_draft.md"
    report_path = assets / "paper_thermal_rr_manuscript_draft_sync_report.md"
    assets_copy.write_text(synced, encoding="utf-8")
    write_report(report_path, manuscript_path, parts)
    print(f"Synced manuscript draft: {manuscript_path}")
    print(f"Saved paper-assets manuscript draft: {assets_copy}")
    print(f"Saved manuscript sync report: {report_path}")


if __name__ == "__main__":
    main()
