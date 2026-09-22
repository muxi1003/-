from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a Q2-or-higher innovation roadmap from the current RR metrics, "
            "readiness gates, external-validation quotas, and literature map."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def first_match(
    table: pd.DataFrame | None,
    column: str,
    contains: str,
) -> pd.Series | None:
    if table is None or table.empty or column not in table.columns:
        return None
    mask = table[column].astype(str).str.contains(contains, case=False, regex=False)
    if not mask.any():
        return None
    return table.loc[mask].iloc[0]


def exact_count(row: pd.Series | None) -> str:
    if row is None or "exact_count" not in row:
        return "NA"
    exact_value = row.get("exact_count")
    if isinstance(exact_value, str) and "/" in exact_value:
        return exact_value
    if "count_valid_videos" in row and pd.notna(row.get("count_valid_videos")):
        return f"{int(exact_value)}/{int(row['count_valid_videos'])}"
    if "videos" in row and pd.notna(row.get("videos")):
        return f"{int(exact_value)}/{int(row['videos'])}"
    return str(exact_value)


def metric_evidence(row: pd.Series | None) -> str:
    if row is None:
        return "not generated"
    return (
        f"RR R2={fmt_float(row.get('rr_r2'))}; "
        f"MAE={fmt_float(row.get('rr_mae_bpm', row.get('rr_mae')), 4)} bpm; "
        f"RMSE={fmt_float(row.get('rr_rmse_bpm', row.get('rr_rmse')), 4)} bpm; "
        f"exact={exact_count(row)}"
    )


def readiness_summary(readiness: pd.DataFrame | None) -> tuple[str, str]:
    if readiness is None or readiness.empty:
        return "not generated", "Run audit_rr_submission_readiness.py."
    q2_status = str(readiness.get("q2_submission_status", pd.Series(["unknown"])).iloc[0])
    blockers = readiness[
        readiness.get("q2_blocking", pd.Series(False, index=readiness.index)).astype(bool)
        & readiness.get("status", pd.Series("", index=readiness.index))
        .astype(str)
        .isin(["FAIL", "WARN"])
    ]
    blocker_names = "; ".join(blockers["check"].astype(str).head(8).tolist())
    if len(blockers) > 8:
        blocker_names += f"; +{len(blockers) - 8} more"
    return q2_status, blocker_names or "no Q2 blockers reported"


def metadata_evidence(output_dir: Path) -> str:
    progress = read_csv(output_dir / "paper_metadata_annotation_progress.csv")
    heat = read_csv(output_dir / "paper_heat_stress_readiness.csv")
    pieces: list[str] = []
    if progress is not None and not progress.empty:
        by_field = {str(row["field"]): row for _, row in progress.iterrows()}

        def field_text(field: str) -> str:
            row = by_field.get(field)
            if row is None:
                return f"{field}=not generated"
            nonempty = int(float(row.get("nonempty", 0)))
            total = int(float(row.get("total_videos", 0)))
            unique = int(float(row.get("unique_values", 0)))
            return f"{field}={nonempty}/{total}, unique={unique}"

        pieces.append(
            "; ".join(
                [
                    field_text("cow_id"),
                    field_text("scene_id"),
                    field_text("ambient_temperature_c"),
                    field_text("relative_humidity_percent"),
                    field_text("thi"),
                    field_text("external_test_split"),
                ]
            )
        )
        still_missing = [
            field
            for field in [
                "collection_date",
                "camera_id",
                "head_motion_score_0_3",
                "occlusion_score_0_3",
                "nostril_visibility_score_0_3",
            ]
            if int(float(by_field.get(field, {}).get("nonempty", 0))) == 0
        ]
        if still_missing:
            pieces.append("remaining missing fields: " + ", ".join(still_missing))
    if heat is not None and not heat.empty:
        thi_category = heat[heat["field"].astype(str) == "thi_category"]
        if not thi_category.empty:
            row = thi_category.iloc[0]
            pieces.append(
                "THI category readiness="
                f"{row.get('ready', 'NA')} with unique_values={row.get('unique_values', 'NA')}"
            )
    if not pieces:
        return "metadata evidence tables not generated"
    return "; ".join(pieces)


def quota_evidence(quota: pd.DataFrame | None) -> str:
    if quota is None or quota.empty:
        return "external sample quota table not generated"
    videos = quota[quota["quota_dimension"].astype(str) == "external videos"].copy()
    if videos.empty:
        return "external sample quota table lacks video rows"
    parts = [
        f"{row['collection_tier']}={row['minimum_target']} videos"
        for _, row in videos.iterrows()
    ]
    return "; ".join(parts)


def error_driven_external_evidence(output_dir: Path) -> str:
    queue = read_csv(output_dir / "paper_external_validation_error_driven_queue.csv")
    strata = read_csv(output_dir / "paper_external_validation_error_driven_strata.csv")
    if queue is None or queue.empty:
        return "error-driven external validation queue not generated"
    p0 = queue[queue["priority_rank"].astype(str).isin(["1", "2"])]
    low_risk_false_auto = queue[
        queue["planning_stratum"].astype(str)
        == "P0_low_risk_false_auto_challenge"
    ]
    strata_count = 0 if strata is None or strata.empty else len(strata)
    return (
        f"error-driven external queue rows={len(queue)}; "
        f"P0 challenge rows={len(p0)}; "
        f"low-risk false-auto prototypes={len(low_risk_false_auto)}; "
        f"coverage strata={strata_count}"
    )


def split_all_use_external_batch_evidence(output_dir: Path) -> str:
    readiness = read_csv(output_dir / "paper_external_validation_split_all_use_readiness.csv")
    sessions = read_csv(output_dir / "paper_external_validation_split_all_use_session_summary.csv")
    fieldwork = read_csv(
        output_dir / "paper_external_validation_split_all_use_fieldwork_template.csv"
    )
    if readiness is None or readiness.empty:
        return "split_all_use external batch not inventoried"
    evidence_by_check = {
        str(row["check"]): str(row["evidence"]) for _, row in readiness.iterrows()
    }
    session_count = 0 if sessions is None or sessions.empty else len(sessions)
    historical_count_rows = 0
    annotator_rows = 0
    camera_rows = 0
    if fieldwork is not None and not fieldwork.empty:
        historical_count_rows = int(
            fieldwork.get("manual_breath_count", pd.Series("", index=fieldwork.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        annotator_rows = int(
            fieldwork.get("reference_rr_annotator", pd.Series("", index=fieldwork.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        camera_rows = int(
            fieldwork.get("camera_id", pd.Series("", index=fieldwork.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
    return (
        "split_all_use candidate external batch: "
        f"{evidence_by_check.get('clip-level external rows available', 'clips=NA')}; "
        f"source_sessions={session_count}; "
        f"{evidence_by_check.get('unique cow_id labels available', 'unique_cows=NA')}; "
        f"{evidence_by_check.get('collection dates available', 'unique_dates=NA')}; "
        f"{evidence_by_check.get('104-video Q2 algorithmic external tier by clips', 'target=NA')}; "
        f"historical single-annotator counts={historical_count_rows}/{len(fieldwork) if fieldwork is not None else 0} "
        "(development-only, not confirmatory truth); "
        f"source annotator identities={annotator_rows}/{len(fieldwork) if fieldwork is not None else 0}; "
        f"camera metadata={camera_rows}/{len(fieldwork) if fieldwork is not None else 0} (optional for RR scoring)"
    )


def blinded_annotation_evidence(output_dir: Path) -> str:
    p2g_progress = read_csv(
        output_dir / "paper_p2g_primary_plus_extension_annotation_progress_summary.csv"
    )
    if p2g_progress is not None and not p2g_progress.empty:
        expected = pd.to_numeric(
            p2g_progress.get("expected_rows", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        complete = pd.to_numeric(
            p2g_progress.get("complete_required_rows", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        expected_rows = int(expected.max()) if not expected.empty else 0
        complete_rows = int(complete.min()) if not complete.empty else 0
        statuses = ", ".join(
            f"{row.annotator}={row.status} ({int(float(row.complete_required_rows))}/"
            f"{int(float(row.expected_rows))})"
            for row in p2g_progress.itertuples(index=False)
        )
        primary_packets = read_csv(output_dir / "paper_p2g_primary_all_packet_summary.csv")
        extension_packets = read_csv(output_dir / "paper_p2g_extension_all_packet_summary.csv")
        primary_size = (
            int(pd.to_numeric(primary_packets["clips"], errors="coerce").max())
            if primary_packets is not None and not primary_packets.empty and "clips" in primary_packets
            else 0
        )
        extension_size = (
            int(pd.to_numeric(extension_packets["clips"], errors="coerce").max())
            if extension_packets is not None and not extension_packets.empty and "clips" in extension_packets
            else 0
        )
        return (
            "frozen_P2g_primary_plus_extension_scope="
            f"{expected_rows} duration-eligible clips (primary={primary_size}, "
            f"pre-registered_extension={extension_size}); "
            f"dual_annotation_progress={complete_rows}/{expected_rows}; "
            f"exports={statuses}; consensus_ready=0 pending A/B exports and adjudication"
        )

    agreement = read_csv(output_dir / "paper_external_validation_breath_annotation_agreement_summary.csv")
    dashboard = read_csv(output_dir / "paper_external_validation_annotation_submission_dashboard.csv")
    worklist_a = output_dir / "paper_external_validation_breath_annotation_worklist_annotator_a.csv"
    worklist_b = output_dir / "paper_external_validation_breath_annotation_worklist_annotator_b.csv"
    packet_a = output_dir / "paper_external_validation_breath_annotation_packet_annotator_a.html"
    packet_b = output_dir / "paper_external_validation_breath_annotation_packet_annotator_b.html"
    pieces = [
        f"blind_packets={'present' if packet_a.exists() and packet_b.exists() else 'missing'}",
        f"per_annotator_worklists={'present' if worklist_a.exists() and worklist_b.exists() else 'missing'}",
    ]
    if agreement is not None and not agreement.empty:
        metrics = {
            str(row["metric"]): str(row["value"])
            for _, row in agreement.iterrows()
            if "metric" in agreement.columns and "value" in agreement.columns
        }
        pieces.append(
            "annotation_agreement="
            f"included={metrics.get('included_external_rows', 'NA')}; "
            f"complete_dual={metrics.get('complete_dual_annotation_rows', 'NA')}; "
            f"consensus_ready={metrics.get('consensus_ready_rows', 'NA')}; "
            f"needs_adjudication={metrics.get('needs_adjudication_rows', 'NA')}"
        )
    if dashboard is not None and not dashboard.empty:
        completion = dashboard[dashboard["gate"].astype(str) == "dual_annotation_completion"]
        consensus = dashboard[dashboard["gate"].astype(str) == "consensus_ready"]
        if not completion.empty:
            pieces.append(
                f"dual_annotation_completion={completion.iloc[0].get('status', 'NA')} "
                f"({completion.iloc[0].get('evidence', 'NA')})"
            )
        if not consensus.empty:
            pieces.append(
                f"consensus_gate={consensus.iloc[0].get('status', 'NA')} "
                f"({consensus.iloc[0].get('evidence', 'NA')})"
            )
    return "; ".join(pieces)


def selective_evidence(selective: pd.DataFrame | None) -> str:
    if selective is None or selective.empty:
        return "selective RR metrics not generated"
    strict = first_match(selective, "label", "strict_auto_report_subset")
    if strict is None:
        strict = first_match(selective, "label", "strict_auto")
    if strict is None:
        return "selective RR metrics generated but strict subset row missing"
    coverage = fmt_float(strict.get("coverage"), 4)
    return (
        f"strict auto subset: n={int(strict.get('videos', 0))}; "
        f"coverage={coverage}; RR R2={fmt_float(strict.get('rr_r2'))}; "
        f"MAE={fmt_float(strict.get('rr_mae', strict.get('rr_mae_bpm')), 4)} bpm; "
        f"exact={exact_count(strict)}"
    )


def algorithmic_quality_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_algorithmic_quality_tier_metrics_table.csv")
    if table is None or table.empty:
        table = read_csv(output_dir.parent / "paper_repro_algorithmic_quality_tier_metrics.csv")
    if table is None or table.empty:
        return "algorithmic quality stratification not generated"
    mask = (
        (table["method"].astype(str) == "signal_aware_safe_gate")
        & (table["tier_type"].astype(str) == "fixed_risk_tier")
        & (table["tier"].astype(str) == "low_risk_auto_candidate")
    )
    if not mask.any():
        return "algorithmic quality table generated but low-risk safe-gate row missing"
    row = table.loc[mask].iloc[0]
    return (
        "algorithmic low-risk safe-gate subset: "
        f"n={int(row['videos'])}; coverage={fmt_float(row['coverage'])}; "
        f"RR R2={fmt_float(row['rr_r2'])}; "
        f"MAE={fmt_float(row['rr_mae'], 4)} bpm; "
        f"exact={int(row['exact_count'])}/{int(row['count_valid_videos'])}"
    )


def deployment_decision_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_deployment_decision_recommendation_table.csv")
    if table is None or table.empty:
        table = read_csv(output_dir.parent / "paper_repro_deployment_decision_recommendation.csv")
    if table is None or table.empty:
        return "deployment decision curve not generated"
    row = table.iloc[0]
    return (
        f"decision curve operating point={row.get('operating_point')}; "
        f"auto n={int(row.get('auto_videos'))}; "
        f"coverage={fmt_float(row.get('auto_coverage'))}; "
        f"review_load={fmt_float(row.get('manual_review_load'))}; "
        f"RR R2={fmt_float(row.get('rr_r2'))}; "
        f"MAE={fmt_float(row.get('rr_mae'), 4)} bpm; "
        f"exact={int(row.get('exact_count'))}/{int(row.get('auto_videos'))}"
    )


def conformal_uncertainty_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_conformal_rr_recommendation_table.csv")
    if table is None or table.empty:
        table = read_csv(output_dir.parent / "paper_repro_conformal_rr_recommendation.csv")
    if table is None or table.empty:
        return "conformal RR uncertainty not generated"
    row = table.iloc[0]
    return (
        f"conformal interval={row.get('interval_variant')}; "
        f"target={fmt_float(row.get('target_coverage'))}; "
        f"coverage={fmt_float(row.get('coverage'))}; "
        f"mean_width={fmt_float(row.get('mean_width_bpm'), 3)} bpm; "
        f"low_risk_n={int(row.get('low_risk_videos'))}; "
        f"low_risk_coverage={fmt_float(row.get('low_risk_coverage'))}"
    )


def bilateral_consistency_evidence(output_dir: Path) -> str:
    table = read_csv(output_dir / "paper_rr_bilateral_consistency_summary.csv")
    if table is None or table.empty:
        return "bilateral nostril consistency summary not generated"
    row = first_match(table, "subset", "bilateral_consistent_auto_report_subset")
    if row is None:
        return "bilateral nostril consistency table generated but auto-report row missing"
    return (
        "bilateral-consistent auto-report subset: "
        f"n={int(row.get('videos', 0))}; "
        f"coverage={fmt_float(row.get('coverage'))}; "
        f"RR R2={fmt_float(row.get('rr_r2'))}; "
        f"MAE={fmt_float(row.get('rr_mae_bpm'), 4)} bpm; "
        f"RMSE={fmt_float(row.get('rr_rmse_bpm'), 4)} bpm; "
        f"exact={row.get('exact_count')}; "
        f"uses_truth_for_decision={row.get('uses_truth_for_decision')}"
    )


def method_agreement_evidence(output_dir: Path) -> str:
    summary = read_csv(output_dir / "paper_rr_method_agreement_summary.csv")
    if summary is None or summary.empty:
        return "method agreement/LoA analysis not generated"
    safe = summary[summary["method_id"].astype(str) == "signal_aware_safe_fixed_oof"]
    default = summary[summary["method_id"].astype(str) == "default"]
    if safe.empty or default.empty:
        return "method agreement table generated but default or safe-gate rows are missing"
    safe_row = safe.iloc[0]
    default_row = default.iloc[0]
    delta_loa = float(safe_row["rr_loa_width_bpm"]) - float(default_row["rr_loa_width_bpm"])
    return (
        "Bland-Altman agreement: "
        f"default LoA width={fmt_float(default_row.get('rr_loa_width_bpm'), 3)} bpm; "
        f"safe-gate LoA width={fmt_float(safe_row.get('rr_loa_width_bpm'), 3)} bpm; "
        f"delta={fmt_float(delta_loa, 3)} bpm; "
        f"safe-gate bias={fmt_float(safe_row.get('rr_bias_bpm'), 3)} bpm"
    )


def frame_motion_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_frame_motion_augmented_residual_probe_metrics.csv")
    tiers = read_csv(output_dir / "paper_frame_motion_tier_metrics.csv")
    if metrics is None or metrics.empty:
        return "frame-motion innovation probe not generated"
    probe = first_match(metrics, "label", "motion_augmented_residual_probe")
    quality = first_match(metrics, "label", "existing_quality_residual_fixed_oof")
    default = first_match(metrics, "label", "default_thermal_rr_pipeline")
    tier_text = ""
    if tiers is not None and not tiers.empty:
        high = first_match(tiers, "motion_risk_tier", "high_motion")
        if high is not None:
            tier_text = (
                f"; high-motion tier n={int(high.get('videos'))}, "
                f"default_exact_rate={fmt_float(high.get('default_exact_rate'))}, "
                f"safe_gate_exact_rate={fmt_float(high.get('safe_gate_exact_rate'))}"
            )
    return (
        "frame-motion probe: "
        f"default={metric_evidence(default)}; "
        f"quality_fixed={metric_evidence(quality)}; "
        f"motion_augmented={metric_evidence(probe)}"
        f"{tier_text}; conclusion=use for motion-scenario stratification, not as current main accuracy model"
    )


def physiology_decoder_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_physiology_sequence_decoder_metrics.csv")
    if metrics is None or metrics.empty:
        return "physiology-constrained sequence decoder not generated"
    safe = first_match(metrics, "label", "safe_gate_fixed_baseline")
    fixed = first_match(metrics, "label", "physiology_decoder_fixed")
    nested = first_match(metrics, "label", "physiology_decoder_nested")
    safe_group = first_match(metrics, "label", "safe_gate_prefix_group_baseline")
    grouped = first_match(metrics, "label", "physiology_decoder_prefix_group")
    return (
        f"safe fixed={metric_evidence(safe)}; decoder fixed={metric_evidence(fixed)}; "
        f"decoder nested={metric_evidence(nested)}; safe prefix-group="
        f"{metric_evidence(safe_group)}; decoder prefix-group={metric_evidence(grouped)}; "
        "conclusion=negative ablation: peak-sequence re-decoding degraded the safe gate "
        "and is not promoted"
    )


def multi_roi_selector_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_multi_roi_selector_metrics.csv")
    if metrics is None or metrics.empty:
        return "current-truth multi-ROI selector probe not generated"
    safe = first_match(metrics, "label", "safe_gate_baseline")
    held_video = first_match(metrics, "label", "multi_roi_selector_stratified_oof")
    grouped = first_match(metrics, "label", "multi_roi_selector_prefix_group")
    oracle = first_match(metrics, "label", "multi_roi_oracle_upper_bound")
    return (
        f"safe gate={metric_evidence(safe)}; held-video selector="
        f"{metric_evidence(held_video)}; prefix-group selector={metric_evidence(grouped)}; "
        f"multi-ROI oracle upper bound={metric_evidence(oracle)}; conclusion=feature "
        "space contains recoverable information, but the learned selector is unstable "
        "and is not promoted"
    )


def duration_windowed_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_duration_normalized_windowed_metrics.csv")
    if metrics is None or metrics.empty:
        return "duration-normalized windowed innovation not generated"
    internal_base = metrics[
        metrics["cohort"].astype(str).eq("internal_development")
        & metrics["method"].astype(str).eq("frozen_whole_clip_baseline")
    ]
    internal_offline = metrics[
        metrics["cohort"].astype(str).eq("internal_development")
        & metrics["method"].astype(str).eq(
            "duration_normalized_windowed_offline_context"
        )
    ]
    external_base = metrics[
        metrics["cohort"].astype(str).eq("external_provisional")
        & metrics["method"].astype(str).eq("frozen_whole_clip_baseline")
    ]
    external_offline = metrics[
        metrics["cohort"].astype(str).eq("external_provisional")
        & metrics["method"].astype(str).eq(
            "duration_normalized_windowed_offline_context"
        )
    ]
    external_causal = metrics[
        metrics["cohort"].astype(str).eq("external_provisional")
        & metrics["method"].astype(str).eq("duration_normalized_windowed_causal")
    ]
    if any(
        frame.empty
        for frame in [
            internal_base,
            internal_offline,
            external_base,
            external_offline,
            external_causal,
        ]
    ):
        return "duration-normalized metrics generated but expected rows are missing"
    return (
        f"internal baseline={metric_evidence(internal_base.iloc[0])}; "
        f"internal offline context={metric_evidence(internal_offline.iloc[0])}; "
        f"provisional external baseline={metric_evidence(external_base.iloc[0])}; "
        f"provisional external offline context={metric_evidence(external_offline.iloc[0])}; "
        f"provisional external causal context={metric_evidence(external_causal.iloc[0])}; "
        "conclusion=offline and causal zero-shot recovery without internal degradation, "
        "but absolute external accuracy remains below the Q2 gate"
    )


def farm_calibration_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_external_farm_calibration_metrics.csv")
    bootstrap = read_csv(output_dir / "paper_external_farm_calibration_bootstrap_ci.csv")
    if metrics is None or metrics.empty:
        return "few-shot external farm calibration probe not generated"
    session = metrics[
        metrics["validation_scheme"].astype(str).eq("source_session_group_kfold")
        & metrics["method"].astype(str).eq(
            "few_shot_nested_compact_offline_selector"
        )
    ]
    date = metrics[
        metrics["validation_scheme"].astype(str).eq("leave_one_collection_date_out")
        & metrics["method"].astype(str).eq(
            "few_shot_nested_compact_offline_selector"
        )
    ]
    cow = metrics[
        metrics["validation_scheme"].astype(str).eq("leave_one_cow_out")
        & metrics["method"].astype(str).eq(
            "few_shot_nested_compact_offline_selector"
        )
    ]
    color_session = metrics[
        metrics["validation_scheme"].astype(str).eq("source_session_group_kfold")
        & metrics["method"].astype(str).eq("few_shot_single_color_huber")
    ]
    color_date = metrics[
        metrics["validation_scheme"].astype(str).eq("leave_one_collection_date_out")
        & metrics["method"].astype(str).eq("few_shot_single_color_huber")
    ]
    color_cow = metrics[
        metrics["validation_scheme"].astype(str).eq("leave_one_cow_out")
        & metrics["method"].astype(str).eq("few_shot_single_color_huber")
    ]
    if any(
        frame.empty
        for frame in [session, date, cow, color_session, color_date, color_cow]
    ):
        return "farm calibration metrics generated but compact validation rows are missing"
    bootstrap_note = "cluster bootstrap not generated"
    if bootstrap is not None and not bootstrap.empty:
        delta = bootstrap[
            bootstrap["comparison"].astype(str).eq(
                "nested_compact_offline_minus_zero_shot_offline_context"
            )
            & bootstrap["metric"].astype(str).eq("rr_r2")
        ]
        if not delta.empty:
            row = delta.iloc[0]
            bootstrap_note = (
                f"source-session bootstrap delta R2={float(row['estimate']):.6f} "
                f"(95% CI {float(row['ci_low']):.6f} to "
                f"{float(row['ci_high']):.6f})"
            )
        color_delta = bootstrap[
            bootstrap["comparison"].astype(str).eq(
                "color_huber_minus_zero_shot_color"
            )
            & bootstrap["metric"].astype(str).eq("rr_r2")
        ]
        if not color_delta.empty:
            row = color_delta.iloc[0]
            bootstrap_note += (
                f"; color-Huber versus zero-shot color delta R2="
                f"{float(row['estimate']):.6f} (95% CI "
                f"{float(row['ci_low']):.6f} to {float(row['ci_high']):.6f})"
            )
    return (
        f"source-session nested compact={metric_evidence(session.iloc[0])}; "
        f"leave-one-date-out nested compact={metric_evidence(date.iloc[0])}; "
        f"leave-one-cow-out nested compact={metric_evidence(cow.iloc[0])}; "
        f"source-session single-color Huber={metric_evidence(color_session.iloc[0])}; "
        f"leave-one-date-out single-color Huber={metric_evidence(color_date.iloc[0])}; "
        f"leave-one-cow-out single-color Huber={metric_evidence(color_cow.iloc[0])}; "
        f"{bootstrap_note}; conclusion=farm-specific labels improve the older temperature-"
        "context adaptation point estimates, but improvement over optimized zero-shot P2g "
        "is inconsistent and statistically unresolved; this remains calibrated domain "
        "adaptation rather than independent external validation"
    )


def calibration_free_color_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_calibration_free_thermal_index_metrics.csv")
    bootstrap = read_csv(
        output_dir / "paper_calibration_free_thermal_index_bootstrap_ci.csv"
    )
    robustness = read_csv(
        output_dir / "paper_calibration_free_thermal_index_robustness_metrics.csv"
    )
    robustness_bootstrap = read_csv(
        output_dir / "paper_calibration_free_thermal_index_robustness_bootstrap_ci.csv"
    )
    if metrics is None or metrics.empty:
        return "calibration-free thermal-color probe not generated"
    method = "duration_gated_calibration_free_internal_ranked_ensemble"
    internal = metrics[
        metrics["cohort"].astype(str).eq("internal_development")
        & metrics["method"].astype(str).eq(method)
    ]
    external = metrics[
        metrics["cohort"].astype(str).eq("external_provisional")
        & metrics["method"].astype(str).eq(method)
    ]
    if internal.empty or external.empty:
        return "calibration-free metrics generated but duration-gated rows are missing"
    bootstrap_note = "cluster bootstrap not generated"
    if bootstrap is not None and not bootstrap.empty:
        delta = bootstrap[
            bootstrap["comparison"].astype(str).eq(
                "gated_color_minus_offline_temperature_context"
            )
            & bootstrap["metric"].astype(str).eq("rr_r2")
        ]
        if not delta.empty:
            row = delta.iloc[0]
            bootstrap_note = (
                f"versus offline temperature context delta R2="
                f"{float(row['estimate']):.6f} (95% CI "
                f"{float(row['ci_low']):.6f} to {float(row['ci_high']):.6f})"
            )
    robustness_note = "robustness audit not generated"
    if robustness is not None and robustness_bootstrap is not None:
        session = robustness[
            robustness["analysis"].astype(str).eq(
                "duration_weighted_source_session_aggregation"
            )
        ]
        r2_ci = robustness_bootstrap[
            robustness_bootstrap["metric"].astype(str).eq("rr_r2")
        ]
        if not session.empty and not r2_ci.empty:
            session_row = session.iloc[0]
            ci_row = r2_ci.iloc[0]
            robustness_note = (
                f"source-session R2 CI {float(ci_row['ci_low']):.6f} to "
                f"{float(ci_row['ci_high']):.6f}; duration-weighted session R2="
                f"{float(session_row['rr_r2']):.6f}"
            )
    return (
        f"internal duration-gated color ensemble={metric_evidence(internal.iloc[0])}; "
        f"provisional external duration-gated color ensemble="
        f"{metric_evidence(external.iloc[0])}; {bootstrap_note}; {robustness_note}; conclusion=relative "
        "pseudo-color fusion improves cross-farm zero-shot robustness without changing "
        "the internal frozen result"
    )


def calibration_free_reannotation_evidence(output_dir: Path) -> str:
    priority = read_csv(output_dir / "paper_calibration_free_thermal_index_reannotation_priority.csv")
    packets = read_csv(output_dir / "paper_p2g_reannotation_packet_summary.csv")
    if priority is None or priority.empty or packets is None or packets.empty:
        return "P2g blinded reannotation packet not generated"
    priorities = priority.get("review_priority", pd.Series("", index=priority.index)).astype(str)
    p0 = int(priorities.eq("P0_dual_blinded_recount_and_adjudication").sum())
    p1 = int(priorities.eq("P1_second_independent_recount").sum())
    packet_sizes = ", ".join(
        f"{row.annotator_id}={int(row.clips)}" for row in packets.itertuples(index=False)
    )
    primary_packets = read_csv(output_dir / "paper_p2g_primary_all_packet_summary.csv")
    primary_note = "complete primary packets not generated"
    if primary_packets is not None and not primary_packets.empty:
        primary_sizes = ", ".join(
            f"{row.annotator_id}={int(row.clips)}"
            for row in primary_packets.itertuples(index=False)
        )
        primary_note = f"complete primary packets prepared ({primary_sizes})"
    extension_packets = read_csv(output_dir / "paper_p2g_extension_all_packet_summary.csv")
    extension_predictions = read_csv(output_dir / "paper_p2g_extension_all_frozen_predictions.csv")
    extension_note = "pre-registered external extension packets not generated"
    if extension_packets is not None and not extension_packets.empty:
        extension_sizes = ", ".join(
            f"{row.annotator_id}={int(row.clips)}"
            for row in extension_packets.itertuples(index=False)
        )
        extension_note = (
            "pre-registered duration-eligible external extension packets prepared "
            f"({extension_sizes}); historical counts cleared; "
            f"frozen label-free P2g predictions={len(extension_predictions) if extension_predictions is not None else 0}"
        )
    return (
        f"priority audit table={len(priority)} clips; high-impact reannotation queue="
        f"{p0 + p1} clips (P0={p0}, P1={p1}); "
        f"blinded packets prepared ({packet_sizes}) without existing counts or model predictions; "
        f"{primary_note}; {extension_note}; coordinator mappings are withheld from annotators"
    )


def calibration_free_negative_ablation_evidence(output_dir: Path) -> str:
    consensus = read_csv(output_dir / "paper_calibration_free_consensus_ensemble_selection.csv")
    stability = read_csv(output_dir / "paper_calibration_free_stability_gate_selection.csv")
    geometry = read_csv(output_dir / "paper_calibration_free_geometry_roi_metrics.csv")
    session = read_csv(output_dir / "paper_calibration_free_session_context_metrics.csv")
    if any(table is None or table.empty for table in [consensus, stability, geometry, session]):
        return "calibration-free negative-ablation battery not generated"
    geometry_external = geometry[
        geometry["cohort"].astype(str).eq("external_provisional")
        & geometry["method"].astype(str).eq(
            "duration_gated_calibration_free_consensus_ensemble"
        )
    ]
    if geometry_external.empty:
        return "calibration-free negative-ablation battery missing geometry external row"
    session_best = session.sort_values(["rr_rmse", "rr_mae"]).iloc[0]
    return (
        f"robust aggregation selected={consensus.iloc[0].get('aggregation', 'NA')}; "
        f"kinematic selection={stability.iloc[0].get('rule', 'NA')}; "
        f"internal-selected geometry strategy={geometry_external.iloc[0].get('strategy', 'NA')} "
        f"external {metric_evidence(geometry_external.iloc[0])}; "
        f"best fixed session-context={session_best.get('method', 'NA')} "
        f"({metric_evidence(session_best)}); none displaced clipwise mean P2g"
    )


def consensus_rollback_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_consensus_rollback_probe_metrics.csv")
    cases = read_csv(output_dir / "paper_consensus_rollback_probe_cases.csv")
    tests = read_csv(output_dir / "paper_consensus_rollback_probe_statistical_tests.csv")
    if metrics is None or metrics.empty:
        return "consensus rollback probe not generated"
    fixed_safe = metrics[
        (metrics["validation_mode"].astype(str) == "fixed")
        & (metrics["label"].astype(str) == "signal_aware_safe_fixed_oof")
    ]
    fixed_probe = metrics[
        (metrics["validation_mode"].astype(str) == "fixed")
        & (metrics["label"].astype(str) == "consensus_rollback_probe_fixed")
    ]
    group_safe = metrics[
        (metrics["validation_mode"].astype(str) == "prefix_group")
        & (metrics["label"].astype(str) == "signal_aware_safe_prefix_group")
    ]
    group_probe = metrics[
        (metrics["validation_mode"].astype(str) == "prefix_group")
        & (metrics["label"].astype(str) == "consensus_rollback_probe_prefix_group")
    ]
    if fixed_safe.empty or fixed_probe.empty or group_safe.empty or group_probe.empty:
        return "consensus rollback probe generated but expected comparison rows are missing"
    fixed_safe_row = fixed_safe.iloc[0]
    fixed_probe_row = fixed_probe.iloc[0]
    group_safe_row = group_safe.iloc[0]
    group_probe_row = group_probe.iloc[0]
    case_ids = ""
    if cases is not None and not cases.empty:
        case_ids = "; rollback_cases=" + ",".join(cases["video_id"].astype(str).tolist())
    stats_text = ""
    if tests is not None and not tests.empty:
        stats_parts = []
        for mode in ["fixed", "prefix_group"]:
            match = tests[tests["validation_mode"].astype(str) == mode]
            if match.empty:
                continue
            row = match.iloc[0]
            stats_parts.append(
                f"{mode} paired delta RR R2={fmt_float(row.get('delta_rr_r2'))} "
                f"(CI {fmt_float(row.get('delta_rr_r2_ci_low'))} to "
                f"{fmt_float(row.get('delta_rr_r2_ci_high'))}), "
                f"delta exact={int(float(row.get('delta_exact_count', 0)))}, "
                f"McNemar p={fmt_float(row.get('mcnemar_exact_p_method_better'), 3)}"
            )
        if stats_parts:
            stats_text = "; paired statistics: " + "; ".join(stats_parts)
    return (
        "fixed safe gate: "
        f"{metric_evidence(fixed_safe_row)}; fixed rollback probe: "
        f"{metric_evidence(fixed_probe_row)}; prefix-group safe gate: "
        f"{metric_evidence(group_safe_row)}; prefix-group rollback probe: "
        f"{metric_evidence(group_probe_row)}"
        f"{case_ids}{stats_text}; conclusion=small internal +1 exact rescue, "
        "McNemar evidence is weak, exploratory until frozen external validation"
    )


def post_safe_gate_evidence(output_dir: Path) -> str:
    metrics = read_csv(output_dir / "paper_post_safe_gate_second_stage_probe_metrics_table.csv")
    if metrics is None or metrics.empty:
        metrics = read_csv(
            output_dir.parent / "paper_repro_post_safe_gate_second_stage_probe_metrics.csv"
        )
    cases = read_csv(output_dir / "paper_post_safe_gate_error_cases_table.csv")
    if cases is None or cases.empty:
        cases = read_csv(output_dir.parent / "paper_repro_post_safe_gate_error_cases.csv")
    if metrics is None or metrics.empty:
        return "post safe-gate diagnostic not generated"
    stable = int(metrics.get("passes_internal_stability_guard", pd.Series(False)).astype(bool).sum())
    point_guard = int(metrics.get("passes_metric_guard", pd.Series(False)).astype(bool).sum())
    best = metrics.sort_values(
        ["exact_count", "abs_count_error_ge2", "rr_mae"],
        ascending=[False, True, True],
    ).iloc[0]
    error_count = 0 if cases is None or cases.empty else int(len(cases))
    return (
        f"post safe-gate residual errors={error_count}; "
        f"stable second-stage probes={stable}; point-metric-only probes={point_guard}; "
        f"best exact={int(best.get('exact_count'))}/73 with "
        f">=2 errors={int(best.get('abs_count_error_ge2'))}"
    )


def bootstrap_evidence(bootstrap: pd.DataFrame | None) -> str:
    if bootstrap is None or bootstrap.empty:
        return "bootstrap CI table not generated"
    delta = first_match(bootstrap, "display_metric", "Delta RR R2")
    if delta is None:
        return "Delta RR R2 CI missing"
    return str(delta.get("estimate_with_ci", "Delta RR R2 CI missing"))


def signal_aware_ci_evidence(bootstrap: pd.DataFrame | None) -> str:
    if bootstrap is None or bootstrap.empty:
        return "signal-aware bootstrap CI not generated"
    rows = []
    for comparison in [
        "signal_aware_vs_signal_consensus",
        "signal_aware_vs_default",
    ]:
        match = bootstrap[
            (bootstrap["comparison"].astype(str) == comparison)
            & (bootstrap["metric"].astype(str) == "delta_rr_r2")
        ]
        if not match.empty:
            rows.append(f"{comparison} Delta RR R2={match.iloc[0]['estimate_with_ci']}")
    return "; ".join(rows) if rows else "signal-aware Delta RR R2 CI missing"


def safe_policy_ci_evidence(bootstrap: pd.DataFrame | None) -> str:
    if bootstrap is None or bootstrap.empty:
        return "safe-policy bootstrap CI not generated"
    rows = []
    targets = [
        ("fixed", "safe_policy_vs_signal_consensus", "delta_rr_r2"),
        ("prefix_group", "safe_policy_vs_original_signal_aware", "delta_rr_r2"),
        ("prefix_group", "safe_policy_vs_original_signal_aware", "delta_abs_count_error_ge2"),
    ]
    for mode, comparison, metric in targets:
        match = bootstrap[
            (bootstrap["validation_mode"].astype(str) == mode)
            & (bootstrap["comparison"].astype(str) == comparison)
            & (bootstrap["metric"].astype(str) == metric)
        ]
        if not match.empty:
            rows.append(f"{mode}/{comparison}/{metric}={match.iloc[0]['estimate_with_ci']}")
    return "; ".join(rows) if rows else "safe-policy CI rows missing"


def build_rows(
    main_results: pd.DataFrame | None,
    selective: pd.DataFrame | None,
    bootstrap: pd.DataFrame | None,
    signal_aware_bootstrap: pd.DataFrame | None,
    safe_policy_bootstrap: pd.DataFrame | None,
    readiness: pd.DataFrame | None,
    quota: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame:
    default = first_match(main_results, "method", "Default thermal RR pipeline")
    fixed = first_match(main_results, "method", "fixed threshold, out-of-fold")
    nested = first_match(main_results, "method", "nested threshold CV")
    signal = first_match(main_results, "method", "Signal-consensus supplement (fixed")
    signal_nested = first_match(main_results, "method", "Signal-consensus supplement (nested")
    signal_aware = first_match(
        main_results,
        "method",
        "Signal-aware residual correction (fixed threshold",
    )
    signal_aware_group = first_match(
        main_results,
        "method",
        "Signal-aware residual correction (leave-one-prefix-group-out",
    )
    safe_signal = first_match(
        main_results,
        "method",
        "Conservative signal-aware safe gate (fixed threshold",
    )
    safe_signal_group = first_match(
        main_results,
        "method",
        "Conservative signal-aware safe gate (leave-one-prefix-group-out",
    )
    truth = first_match(main_results, "method", "Truth-calibrated upper bound")
    q2_status, q2_blockers = readiness_summary(readiness)
    metadata_status = metadata_evidence(output_dir)

    fixed_evidence = metric_evidence(fixed)
    if nested is not None:
        fixed_evidence += f"; nested={fmt_float(nested.get('rr_r2'))}"
    signal_evidence = metric_evidence(signal)
    if signal_nested is not None:
        signal_evidence += f"; nested={fmt_float(signal_nested.get('rr_r2'))}"
    signal_aware_evidence = metric_evidence(signal_aware)
    if signal_aware_group is not None:
        signal_aware_evidence += (
            f"; prefix-group={fmt_float(signal_aware_group.get('rr_r2'))} "
            f"({exact_count(signal_aware_group)})"
        )
    signal_aware_evidence += f"; CI: {signal_aware_ci_evidence(signal_aware_bootstrap)}"
    safe_signal_evidence = metric_evidence(safe_signal)
    if safe_signal_group is not None:
        safe_signal_evidence += (
            f"; prefix-group={fmt_float(safe_signal_group.get('rr_r2'))} "
            f"({exact_count(safe_signal_group)})"
        )
    if signal_aware_group is not None:
        safe_signal_evidence += (
            f"; ungated prefix-group comparator={fmt_float(signal_aware_group.get('rr_r2'))} "
            f"({exact_count(signal_aware_group)})"
        )
    safe_signal_evidence += f"; CI: {safe_policy_ci_evidence(safe_policy_bootstrap)}"
    safe_signal_evidence += f"; {post_safe_gate_evidence(output_dir)}"

    rows = [
        {
            "priority": "P0",
            "roadmap_item": "quality-aware residual peak-count correction",
            "paper_role": "main internal algorithm innovation",
            "claim_now": (
                "Use as the primary internal innovation result, not as external "
                "generalization."
            ),
            "current_evidence": fixed_evidence,
            "q2_gate": (
                "Keep fixed thresholds frozen; add real cow_id/date/camera grouping "
                "and external split validation."
            ),
            "next_action": (
                "Fill metadata template, rerun metadata GroupKFold, and validate on "
                "the frozen external split."
            ),
            "acceptance_threshold": (
                "Internal: fixed OOF RR R2 >= 0.940 and nested RR R2 >= 0.935; "
                "external: no performance collapse versus default."
            ),
            "source_basis": (
                "J Therm Biol 2025 DOI 10.1016/j.jtherbio.2025.104154; "
                "MDPI Agriculture 2023 DOI 10.3390/agriculture13101939"
            ),
        },
        {
            "priority": "P1",
            "roadmap_item": "signal-consensus precision extension",
            "paper_role": "candidate precision supplement",
            "claim_now": (
                "Report as an internal candidate extension because FFT/autocorr "
                "agreement improves precision without truth calibration."
            ),
            "current_evidence": signal_evidence,
            "q2_gate": (
                "Promote to a main extension only if external paired CI for Delta "
                "RR R2 or MAE excludes no improvement."
            ),
            "next_action": (
                "Run the same frozen signal-consensus rule on the 104-video "
                "algorithmic Q2 external target, or the 209-video strong "
                "relative-improvement target."
            ),
            "acceptance_threshold": (
                f"Current Delta RR R2 bootstrap evidence: {bootstrap_evidence(bootstrap)}; "
                "external paired improvement should exclude zero for strong wording."
            ),
            "source_basis": (
                "JDS Communications 2024 FFT RR route; current "
                "rr_signal_consensus_validation.py outputs."
            ),
        },
        {
            "priority": "P2",
            "roadmap_item": "conservative signal-aware safe gate",
            "paper_role": "highest-precision internal candidate",
            "claim_now": (
                "Use as a safety-gated internal precision candidate. It improves "
                "stratified OOF metrics and removes the ungated signal-aware "
                "prefix-group large-error failure, but the grouping is still a proxy."
            ),
            "current_evidence": safe_signal_evidence,
            "q2_gate": (
                "Do not promote above the method-frozen main result unless real "
                "cow_id/session/external validation confirms the same gain."
            ),
            "next_action": (
                "Rerun the safe gate after metadata GroupKFold and external split "
                "labels are available; keep threshold-selected rows as sensitivity only."
            ),
            "acceptance_threshold": (
                "External or real-group RR R2 must exceed the signal-consensus baseline "
                "without increasing >=2-breath errors."
            ),
            "source_basis": (
                "Current rr_signal_aware_safe_policy.py outputs; ungated "
                f"signal-aware evidence retained as caution ({signal_aware_evidence})."
            ),
        },
        {
            "priority": "P2b",
            "roadmap_item": "consensus rollback guard for rare harmful residual corrections",
            "paper_role": "exploratory precision supplement after safe gate",
            "claim_now": (
                "Use only as an internal error-analysis probe: it audits rare "
                "cases where the residual model applies a -1 correction despite "
                "FFT and spectral evidence supporting the original count."
            ),
            "current_evidence": consensus_rollback_evidence(output_dir),
            "q2_gate": (
                "Do not promote above the conservative safe gate unless the "
                "rollback rule is frozen before validation and improves exact "
                "agreement under external and grouped validation without harming "
                "already-correct videos."
            ),
            "next_action": (
                "Freeze the rollback guard only as a candidate extension, include "
                "its single internal rescue case in supplementary error analysis, "
                "and test it prospectively in the 104-video algorithmic external set."
            ),
            "acceptance_threshold": (
                "External rollback guard must improve or preserve RR R2, MAE, "
                "RMSE, exact count, and >=2-breath error rate relative to the safe gate."
            ),
            "source_basis": (
                "Current consensus rollback probe; frequency-domain respiratory "
                "count estimates from signal-consensus pipeline."
            ),
        },
        {
            "priority": "P2c",
            "roadmap_item": "physiology-constrained peak-sequence decoder",
            "paper_role": "negative algorithm ablation and design boundary",
            "claim_now": (
                "Report only as a negative supplementary ablation if space permits. "
                "Do not use it as an accuracy contribution because fixed, nested, and "
                "prefix-group results all underperform the conservative safe gate."
            ),
            "current_evidence": physiology_decoder_evidence(output_dir),
            "q2_gate": (
                "Promote only if a future frozen decoder preserves zero >=2-breath "
                "errors and improves or matches safe-gate MAE under both nested and "
                "real grouped/external validation."
            ),
            "next_action": (
                "Stop tuning the same fused-curve peak sequence. Prioritize multi-ROI "
                "thermal feature extraction and a conservative ROI selector, because "
                "the current residual errors were not identifiable from sequence "
                "regularity and signal-count priors alone."
            ),
            "acceptance_threshold": (
                "Fixed, nested, and grouped RR R2/MAE must not be worse than the safe "
                "gate, with abs_count_error_ge2 remaining zero."
            ),
            "source_basis": (
                "Current dynamic-programming candidate-peak sequence experiment using "
                "peak relief, interval regularity, boundary coverage, and independent "
                "spectral/FFT/autocorrelation priors."
            ),
        },
        {
            "priority": "P2d",
            "roadmap_item": "multi-ROI thermal feature selector on current truth",
            "paper_role": "latent-information analysis and negative selector validation",
            "claim_now": (
                "Use the oracle row only to show that multiple thermal ROIs contain "
                "recoverable signal. Do not claim a deployable multi-ROI accuracy gain: "
                "the held-video selector was weaker than the safe gate and prefix-group "
                "validation introduced >=2-breath errors."
            ),
            "current_evidence": multi_roi_selector_evidence(output_dir),
            "q2_gate": (
                "A deployable selector must exceed the safe gate under both held-video "
                "and grouped/external validation while preserving zero >=2-breath errors."
            ),
            "next_action": (
                "Do not port or tune the selector as a main method yet. Complete the "
                "frozen P2g primary-plus-extension blinded external references first, then test whether a "
                "selector learned on the 73-video development set transfers without "
                "retuning."
            ),
            "acceptance_threshold": (
                "Held-video and prefix-group RR R2/MAE must improve over the safe gate, "
                "and abs_count_error_ge2 must remain zero."
            ),
            "source_basis": (
                "Precomputed 68-column multi-ROI thermal features from the same 73 "
                "videos, rescored against the current yoloV8 truth table with held-video "
                "and prefix-group validation. Source-workspace RR labels were not reused."
            ),
        },
        {
            "priority": "P2e",
            "roadmap_item": "duration-normalized offline and causal context fusion",
            "paper_role": "primary cross-duration zero-shot development innovation",
            "claim_now": (
                "Report as a development innovation that preserves the internal frozen "
                "result and substantially reduces the provisional cross-farm failure. "
                "Do not call the Jiufu result confirmatory external validation because "
                "its labels were already inspected during method development."
            ),
            "current_evidence": duration_windowed_evidence(output_dir),
            "q2_gate": (
                "Freeze window length, activation threshold, quality weighting, and "
                  "offline and causal session fusion before a newly collected or untouched "
                  "external cohort."
            ),
            "next_action": (
                  "Retain offline context for retrospective video analysis and causal context "
                  "for online deployment; never present the centered offline estimate as "
                  "real-time. Next evaluate a temperature-mapping-independent relative thermal "
                  "signal on new source sessions."
            ),
            "acceptance_threshold": (
                "New independent external RR R2 >= 0.90, MAE <= 2.5 bpm, RMSE <= "
                "4.0 bpm, with source-session clustered intervals and no internal decline."
            ),
            "source_basis": (
                "Internal acquisition-duration distribution; bilateral thermal curve "
                "quality; cattle multiscale periodicity/rolling-median literature; "
                "PPG harmonic sequential-fusion literature."
            ),
        },
        {
            "priority": "P2f",
            "roadmap_item": "few-shot farm-specific calibration with nested grouped validation",
            "paper_role": "practical target-farm adaptation supplement",
            "claim_now": (
                "Report only as source-session/date-held-out target-farm adaptation. "
                "It quantifies the benefit of a small calibration set but is not a "
                "zero-shot or independent external result."
            ),
            "current_evidence": farm_calibration_evidence(output_dir),
            "q2_gate": (
                "Predefine calibration sessions prospectively, lock the calibrator, and "
                "evaluate on subsequently collected cows/sessions from the same and a "
                "second farm."
            ),
            "next_action": (
                  "Treat one-feature color Huber as an optional farm-adaptation supplement, "
                  "not the main method: optimized zero-shot P2g is already comparable and "
                  "the paired difference is not statistically resolved."
            ),
            "acceptance_threshold": (
                "Prospective calibrated holdout RR R2 >= 0.90 and MAE <= 2.5 bpm; "
                "leave-one-date-out results must not collapse relative to session holdout."
            ),
            "source_basis": (
                  "Nested source-session GroupKFold, leave-one-collection-date-out, and "
                  "leave-one-cow-out validation using only label-independent thermal "
                  "quality features; source-session cluster bootstrap."
            ),
        },
        {
            "priority": "P2g",
            "roadmap_item": "duration-gated calibration-free relative thermal-color ensemble",
            "paper_role": "leading zero-shot cross-farm signal innovation",
            "claim_now": (
                "Report as internal-selected development evidence: seven relative color "
                "signals are ranked on the internal cohort only, and the internal-derived "
                "duration gate preserves the frozen short-video pipeline. The Jiufu result "
                "is not confirmatory because its labels have been inspected."
            ),
            "current_evidence": (
                f"{calibration_free_color_evidence(output_dir)}; "
                f"{calibration_free_reannotation_evidence(output_dir)}"
            ),
            "q2_gate": (
                "Freeze channel definitions, internal ranking, top-7 ensemble, polarity, "
                "and 21-second gate before testing newly collected source sessions."
            ),
            "next_action": (
                "Complete the prepared P0/P1 dual-blinded recounts, adjudicate disagreements, "
                "then validate the frozen gated color ensemble against consensus counts; retain "
                "the absolute temperature curve for physiological interpretation and short clips."
            ),
            "acceptance_threshold": (
                "New independent external RR R2 >= 0.90, MAE <= 2.5 bpm, RMSE <= 4.0 bpm; "
                "source-session paired CI versus offline temperature context must exclude zero."
            ),
            "source_basis": (
                "Relative pseudo-color ROI dynamics avoid dependence on farm/camera-specific "
                "absolute RGB-to-temperature calibration; signal membership and ensemble size "
                "are selected using only the 73-video internal development cohort."
            ),
        },
        {
            "priority": "P2h",
            "roadmap_item": "calibration-free robustness ablation battery",
            "paper_role": "negative cross-farm ablation and complexity boundary",
            "claim_now": (
                "Report only as a supplementary negative ablation: internally selected robust "
                "aggregation, kinematic frame filtering, source-session context, and geometry-"
                "normalized ROI scaling did not replace the clipwise mean P2g rule."
            ),
            "current_evidence": calibration_free_negative_ablation_evidence(output_dir),
            "q2_gate": (
                "Do not alter the P2g mean, top-7 channels, prominence, or duration gate unless "
                "a future independent external cohort shows a source-session paired improvement."
            ),
            "next_action": (
                "Keep the ablation outputs in the supplement, freeze the selected P2g rule, and "
                "prioritize blinded P0/P1 recounts over further label-inspected tuning."
            ),
            "acceptance_threshold": (
                "Any replacement requires a new independent external RR R2 improvement with a "
                "source-session paired 95% CI excluding zero and no internal frozen regression."
            ),
            "source_basis": (
                "Recent RR work motivates viable ROI/frame and multi-cow session designs, but "
                "the present no-label variants must demonstrate transfer rather than plausibility."
            ),
        },
        {
            "priority": "P3",
            "roadmap_item": "uncertainty-aware selective RR reporting, algorithmic quality stratification, deployment decision curve, and conformal RR intervals",
            "paper_role": "deployment and reliability supplement",
            "claim_now": (
                "Use as automatic-report versus manual-review triage and "
                "non-truth curve-quality stratification; report coverage, review "
                "load, and prediction-interval width together."
            ),
            "current_evidence": (
                f"{selective_evidence(selective)}; "
                f"{algorithmic_quality_evidence(output_dir)}; "
                f"{deployment_decision_evidence(output_dir)}; "
                f"{conformal_uncertainty_evidence(output_dir)}; "
                f"{method_agreement_evidence(output_dir)}"
            ),
            "q2_gate": (
                "Freeze the triage, risk thresholds, and interval calibration rule "
                "before external testing and report both accepted and rejected subsets."
            ),
            "next_action": (
                "Apply the strict review gate, algorithmic quality risk threshold, "
                "and conformal interval rule to external videos and audit false "
                "auto-accepts, review burden, and interval coverage."
            ),
            "acceptance_threshold": (
                "No high-confidence external auto-report should exceed +/-1 breath; "
                "interval coverage and width must be reported beside accuracy."
            ),
            "source_basis": (
                "PLF deployment requirement: quality flags and review routing are more "
                "defensible than only reporting mean accuracy."
            ),
        },
        {
            "priority": "P3b",
            "roadmap_item": "bilateral nostril signal-consistency auto-report gate",
            "paper_role": "physiological consistency reliability supplement",
            "claim_now": (
                "Use as an internal selective-reporting reliability layer: before "
                "automatic RR reporting, require left and right nostril temperature "
                "curves to show compatible respiratory periodicity. Do not describe "
                "it as external validation, disease detection, heat-stress diagnosis, "
                "or true animal-identity evidence."
            ),
            "current_evidence": bilateral_consistency_evidence(output_dir),
            "q2_gate": (
                "Freeze the bilateral thresholds before external testing and report "
                "coverage alongside accuracy; validate both accepted and rejected "
                "subsets on independent external videos."
            ),
            "next_action": (
                "Apply the same bilateral gate to the method-frozen external split, "
                "then audit false auto-accepts, rejected-but-correct cases, and "
                "left/right curve failures."
            ),
            "acceptance_threshold": (
                "The bilateral accepted subset should preserve high exact agreement "
                "without hiding external failures; rejected cases must be reported "
                "as review burden, not discarded from evaluation."
            ),
            "source_basis": (
                "Current build_rr_bilateral_consistency_gate.py outputs; the 2025 "
                "Journal of Thermal Biology head-movement RR paper motivates "
                "two-nostril curve fusion; recent thermal vital-sign work motivates "
                "tracking and signal reliability checks."
            ),
        },
        {
            "priority": "P3c",
            "roadmap_item": "frame-motion-aware head-movement proxy and external sampling strata",
            "paper_role": "motion-scenario innovation screen and external validation design",
            "claim_now": (
                "Use as an objective, non-truth frame-motion layer to describe "
                "video instability and to stratify future external validation. "
                "Do not claim manual head-motion robustness or improved main accuracy."
            ),
            "current_evidence": frame_motion_evidence(output_dir),
            "q2_gate": (
                "Promote only if the feature extraction rule is frozen and the "
                "motion-aware model beats the safe gate under nested, prefix-group, "
                "and external validation."
            ),
            "next_action": (
                "Use high-motion videos as a required external sampling stratum; "
                "collect manual head-motion/occlusion labels only if making a "
                "manual-quality robustness claim."
            ),
            "acceptance_threshold": (
                "A motion-aware residual model must exceed the conservative safe gate "
                "without increasing >=2-breath errors; otherwise keep the feature as "
                "quality-control and sampling evidence."
            ),
            "source_basis": (
                "Current frame-motion probe; head-movement-scenario thermal RR paper; "
                "thermal vital-sign tracking literature."
            ),
        },
        {
            "priority": "P4",
            "roadmap_item": "metadata-stratified robustness and heat-stress context",
            "paper_role": "Q2+ meaning gate",
            "claim_now": (
                "Use the filled numeric cow_id and Lindian collection context only. "
                "Do not claim heat-stress monitoring or quality-stratified robustness yet."
            ),
            "current_evidence": (
                f"{metadata_status}; Q2 readiness={q2_status}; blockers={q2_blockers}"
            ),
            "q2_gate": (
                "Keep numeric cow_id and collection provenance explicit; fill real collection date, "
                "camera ID, head motion, occlusion, nostril visibility, and an external "
                "split with nonzero external videos."
            ),
            "next_action": (
                "Manually score quality fields, recover acquisition date/camera if records "
                "exist, or collect a new external set; then rerun metadata quality, "
                "heat-stress context, and submission-readiness audits."
            ),
            "acceptance_threshold": (
                "Required metadata coverage should be 73/73 for internal metadata claims "
                "or complete on the external validation set; heat-stress stratification "
                "requires more than one THI category or synchronized barn measurements."
            ),
            "source_basis": (
                "Biosystems Engineering 2024 environmental RR ML; BO-RF ATHI local "
                "paper; PLF dataset survey arXiv:2406.10628."
            ),
        },
        {
            "priority": "P5",
            "roadmap_item": "frozen external validation and sample-size ladder",
            "paper_role": "Q2+ submission gate",
            "claim_now": (
                "Current work is internally manuscript-ready but not Q2+ ready until "
                "real external validation is complete. The split_all_use batch now "
                "supports a blinded dual-annotation route, but it has not been scored."
            ),
            "current_evidence": (
                f"{quota_evidence(quota)}; {error_driven_external_evidence(output_dir)}; "
                f"{split_all_use_external_batch_evidence(output_dir)}; "
                f"{blinded_annotation_evidence(output_dir)}"
            ),
            "q2_gate": (
                "Minimum smoke test: 50 external videos; algorithmic Q2 external "
                "validation: 104 videos without heat-stress/manual-quality claims; "
                "full metadata Q2 validation: 104 videos with environment and "
                "manual-quality fields; strong relative-improvement evidence: 209 videos."
            ),
            "next_action": (
                "Complete the blinded A/B breath-count annotation exports, run agreement "
                "and adjudication until consensus_ready equals included rows, then score "
                "the method-frozen external split without changing thresholds."
            ),
            "acceptance_threshold": (
                "External split validation PASS and no Q2-blocking readiness failures."
            ),
            "source_basis": (
                "Current external-validation sample plan; PLF dataset survey, outdoor "
                "livestock monitoring review, cross-farm transfer-learning evidence, "
                "and MultiCamCows identity literature support grouped validation."
            ),
        },
        {
            "priority": "Guardrail",
            "roadmap_item": "truth-calibrated upper bound",
            "paper_role": "supplementary diagnostic only",
            "claim_now": (
                "Never use as model performance; it uses reference-assisted per-video "
                "calibration."
            ),
            "current_evidence": metric_evidence(truth),
            "q2_gate": (
                "Keep it outside the abstract and main performance claim. Use only to "
                "show signal information content."
            ),
            "next_action": (
                "In manuscript tables, label this row oracle_upper_bound or upper_bound_only."
            ),
            "acceptance_threshold": (
                "Primary-performance-allowed must remain no in metric reporting tiers."
            ),
            "source_basis": "Internal leakage guard; paper_metric_reporting_tiers.csv.",
        },
        {
            "priority": "Future",
            "roadmap_item": "end-to-end video transformer or multi-camera identity fusion",
            "paper_role": "future work unless new data are collected",
            "claim_now": (
                "Do not make this the current main claim with only 73 thermal-curve "
                "videos."
            ),
            "current_evidence": metric_evidence(default),
            "q2_gate": (
                "Requires larger multi-cow, multi-session, preferably multi-camera data "
                "with identity and external splits."
            ),
            "next_action": (
                "After external validation, collect paired RGB/thermal or multi-camera "
                "data if targeting a stronger end-to-end paper."
            ),
            "acceptance_threshold": (
                "Transformer or identity model must be trained/evaluated by cow/session "
                "holdout, not random frame/video split."
            ),
            "source_basis": (
                "J Dairy Sci 2024 VideoMAE DOI 10.3168/jds.2023-24601; "
                "MultiCamCows arXiv:2410.12695."
            ),
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    roadmap: pd.DataFrame,
    q2_status: str,
    q2_blockers: str,
) -> None:
    today = date.today().isoformat()
    main_rows = roadmap[roadmap["priority"].astype(str).str.startswith("P")]
    guard_rows = roadmap[roadmap["priority"].isin(["Guardrail", "Future"])]
    text = f"""# Q2+ Innovation Roadmap for Thermal RR Manuscript

Generated: {today}

This roadmap converts the current repository evidence into manuscript decisions.
It is intentionally conservative: precision improvements are separated from
external-generalization and biological-meaning claims.

## Decision Summary

- Main internal algorithm claim: quality-aware residual peak-count correction.
- Precision extension: signal-consensus correction, currently internal only.
- Reliability supplements: uncertainty-aware selective reporting, decision-curve review routing, bilateral nostril consistency, and conformal RR intervals with coverage and width.
- Q2+ gates: real metadata, heat-stress context, and frozen external validation.
- New external candidate batch: use blinded A/B HTML packets and consensus adjudication before scoring the 109-clip frozen P2g primary-plus-extension scope (`94` primary plus `15` pre-registered duration-eligible extension clips).
- Current Q2 readiness: `{q2_status}`.
- Current blocking or caution checks: {q2_blockers}.

## Claim Roadmap

{markdown_table(main_rows, ["priority", "roadmap_item", "paper_role", "claim_now", "current_evidence", "q2_gate", "next_action"])}

## Guardrails and Future Work

{markdown_table(guard_rows, ["priority", "roadmap_item", "paper_role", "claim_now", "current_evidence", "q2_gate"])}

## How to Use RR R2 in the Paper

Use the default pipeline RR R2 as the baseline, the fixed out-of-fold quality
residual RR R2 as the main internal innovation result, the signal-consensus RR R2
as a candidate precision extension, and the conservative signal-aware safe-gate
RR R2 as the highest-precision internal candidate with an explicit proxy-group
and external-validation caution. The ungated signal-aware RR R2 should be kept
as a method-development comparator. The truth-calibrated RR R2 is an oracle
upper bound only and must not be described as deployable prediction performance.

## Minimum Next Experiment Sequence

1. Fill the metadata template for cow_id, collection date, scene/camera, THI or
   temperature/humidity, motion, occlusion, nostril visibility, and split label.
2. Rerun metadata quality, grouped validation, heat-stress context, external
   method freeze, external split validation, and submission readiness.
3. If external videos are collected, use the 50-video tier only for feasibility,
   the 104-video algorithmic tier for the current no-heat/no-manual-quality Q2
   engineering route, the 104-video full-metadata tier only for heat-stress or
   manual-quality claims, and the 209-video tier for a strong relative-improvement
   claim.
4. Keep truth-calibrated outputs in an upper-bound paragraph or supplement.
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results = read_csv(output_dir / "paper_main_results_table.csv")
    selective = read_csv(output_dir / "paper_selective_rr_metrics_table.csv")
    bootstrap = read_csv(output_dir / "paper_bootstrap_ci_table.csv")
    signal_aware_bootstrap = read_csv(
        output_dir / "paper_signal_aware_residual_bootstrap_ci_table.csv"
    )
    safe_policy_bootstrap = read_csv(
        output_dir / "paper_signal_aware_safe_policy_bootstrap_ci_table.csv"
    )
    readiness = read_csv(output_dir / "paper_submission_readiness_table.csv")
    quota = read_csv(output_dir / "paper_external_validation_quota_table.csv")

    q2_status, q2_blockers = readiness_summary(readiness)
    roadmap = build_rows(
        main_results,
        selective,
        bootstrap,
        signal_aware_bootstrap,
        safe_policy_bootstrap,
        readiness,
        quota,
        output_dir,
    )

    csv_path = output_dir / "paper_literature_q2_plus_innovation_roadmap.csv"
    md_path = output_dir / "paper_literature_q2_plus_innovation_roadmap.md"
    roadmap.to_csv(csv_path, index=False)
    write_report(md_path, roadmap, q2_status, q2_blockers)

    print(f"Saved Q2+ innovation roadmap CSV: {csv_path}")
    print(f"Saved Q2+ innovation roadmap report: {md_path}")
    print("\nRoadmap items:")
    print(roadmap[["priority", "roadmap_item", "paper_role"]].to_string(index=False))


if __name__ == "__main__":
    main()
