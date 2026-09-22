from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CROSSREF_CHECK_DATE = "2026-07-13"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a recent-literature evidence refresh for the thermal RR paper. "
            "The output maps verified sources to manuscript-safe innovation claims."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--docs-dir", type=Path, default=repo_root / "docs")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}"


def first_contains(table: pd.DataFrame | None, column: str, text: str) -> pd.Series | None:
    if table is None or table.empty or column not in table.columns:
        return None
    mask = table[column].astype(str).str.contains(text, case=False, regex=False)
    if not mask.any():
        return None
    return table.loc[mask].iloc[0]


def metric_evidence(row: pd.Series | None) -> str:
    if row is None:
        return "not generated"
    return (
        f"RR R2={fmt_float(row.get('rr_r2'))}; "
        f"MAE={fmt_float(row.get('rr_mae_bpm', row.get('rr_mae')), 3)} bpm; "
        f"RMSE={fmt_float(row.get('rr_rmse_bpm', row.get('rr_rmse')), 3)} bpm; "
        f"exact={row.get('exact_count', 'NA')}"
    )


def field_coverage(output_dir: Path, field: str) -> str:
    progress = read_csv(output_dir / "paper_metadata_annotation_progress.csv")
    if progress is None or progress.empty:
        return f"{field}=not generated"
    row = progress[progress["field"].astype(str) == field]
    if row.empty:
        return f"{field}=not generated"
    item = row.iloc[0]
    return (
        f"{field}={int(float(item.get('nonempty', 0)))}/"
        f"{int(float(item.get('total_videos', 0)))}"
    )


def unique_nonempty(table: pd.DataFrame | None, column: str, limit: int = 3) -> str:
    if table is None or table.empty or column not in table.columns:
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


def known_collection_context(output_dir: Path) -> str:
    metadata = read_csv(output_dir / "paper_metadata_template.csv")
    location = "/".join(
        [
            unique_nonempty(metadata, "collection_location_country"),
            unique_nonempty(metadata, "collection_location_province"),
            unique_nonempty(metadata, "collection_location_county"),
            unique_nonempty(metadata, "collection_site"),
        ]
    )
    return "; ".join(
        [
            f"{field_coverage(output_dir, 'cow_id')} derived from numeric video labels",
            (
                f"{field_coverage(output_dir, 'external_test_split')}; "
                f"values={unique_nonempty(metadata, 'external_test_split')}"
            ),
            f"collection_start_date={unique_nonempty(metadata, 'collection_start_date')}",
            f"collection_end_date={unique_nonempty(metadata, 'collection_end_date')}",
            f"location={location}",
            f"scene_id={unique_nonempty(metadata, 'scene_id')}",
        ]
    )


def external_workflow_evidence(output_dir: Path) -> str:
    freeze = read_csv(output_dir / "paper_method_freeze_summary.csv")
    commands = read_csv(output_dir / "paper_external_validation_run_commands.csv")
    annotation = read_csv(output_dir / "paper_external_validation_breath_annotation_agreement_summary.csv")
    annotation_dashboard = read_csv(output_dir / "paper_external_validation_annotation_submission_dashboard.csv")
    if freeze is None or freeze.empty:
        freeze_text = "method freeze not generated"
    else:
        item = freeze.iloc[0]
        freeze_id = str(item.get("method_freeze_id", ""))
        freeze_text = freeze_id[:12] + "..." if len(freeze_id) > 12 else freeze_id
        freeze_text = (
            f"freeze_status={item.get('freeze_status', 'unknown')}; "
            f"method_freeze_id={freeze_text}; "
            f"required_files={item.get('required_files_present', 'NA')}/"
            f"{item.get('required_files', 'NA')}"
        )
    if commands is None or commands.empty:
        command_text = "external run commands not generated"
    else:
        steps = commands["step"].astype(str).tolist() if "step" in commands.columns else []
        command_text = (
            f"external workflow steps={len(commands)}; "
            f"apply_frozen_postprocessors={'yes' if 'apply_frozen_postprocessors' in steps else 'no'}; "
            f"score_external_split={'yes' if 'score_external_split' in steps else 'no'}"
        )
    if annotation is None or annotation.empty:
        annotation_text = "dual annotation audit not generated"
    else:
        metrics = {str(row["metric"]): str(row["value"]) for _, row in annotation.iterrows()}
        annotation_text = (
            "blinded dual annotation="
            f"included={metrics.get('included_external_rows', 'NA')}; "
            f"complete_dual={metrics.get('complete_dual_annotation_rows', 'NA')}; "
            f"consensus_ready={metrics.get('consensus_ready_rows', 'NA')}; "
            f"needs_adjudication={metrics.get('needs_adjudication_rows', 'NA')}"
        )
    if annotation_dashboard is not None and not annotation_dashboard.empty:
        route = annotation_dashboard[
            annotation_dashboard["gate"].astype(str) == "target_journal_route"
        ]
        if not route.empty:
            annotation_text += (
                f"; target_route={route.iloc[0].get('status', 'NA')} "
                f"({route.iloc[0].get('evidence', 'NA')})"
            )
    return f"{freeze_text}; {command_text}; {annotation_text}"


def readiness_evidence(output_dir: Path) -> str:
    readiness = read_csv(output_dir / "paper_submission_readiness_table.csv")
    if readiness is None or readiness.empty:
        readiness = read_csv(output_dir.parent / "paper_repro_quality_residual_submission_readiness.csv")
    if readiness is None or readiness.empty:
        return "submission readiness audit not generated"
    status = str(readiness.get("q2_submission_status", pd.Series(["unknown"])).iloc[0])
    blockers = readiness[
        readiness.get("q2_blocking", pd.Series(False, index=readiness.index)).astype(str).str.lower()
        == "true"
    ]
    blockers = blockers[blockers.get("status", pd.Series("", index=blockers.index)).isin(["FAIL", "WARN"])]
    names = "; ".join(blockers.get("check", pd.Series(dtype=str)).astype(str).head(6).tolist())
    if len(blockers) > 6:
        names += f"; +{len(blockers) - 6} more"
    return f"Q2 status={status}; blockers={names or 'none'}"


def source_register() -> pd.DataFrame:
    rows = [
        {
            "source_id": "lin_2025_head_movement_irt_rr",
            "citation_short": "Journal of Thermal Biology, 2025",
            "title": "Respiratory rate detection of dairy cows based on infrared thermography in head movement scenarios",
            "year": 2025,
            "venue": "Journal of Thermal Biology",
            "doi": "10.1016/j.jtherbio.2025.104154",
            "url": "https://doi.org/10.1016/j.jtherbio.2025.104154",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "primary_recent_baseline",
            "innovation_pressure": "The nostril-keypoint and temperature-mapping route is already current; novelty must move to residual error handling, uncertainty, and validation.",
        },
        {
            "source_id": "zhao_2023_irt_dl_rr",
            "citation_short": "Agriculture, 2023",
            "title": "Detection of Respiratory Rate of Dairy Cows Based on Infrared Thermography and Deep Learning",
            "year": 2023,
            "venue": "Agriculture",
            "doi": "10.3390/agriculture13101939",
            "url": "https://doi.org/10.3390/agriculture13101939",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "technical_baseline",
            "innovation_pressure": "A modular thermal-RR baseline exists; the paper should emphasize peak-count residuals and reproducible validation rather than only repeating YOLO detection.",
        },
        {
            "source_id": "mantovani_2024_fft_rr",
            "citation_short": "JDS Communications, 2024",
            "title": "Predicting respiration rate in unrestrained dairy cows using image analysis and fast Fourier transform",
            "year": 2024,
            "venue": "JDS Communications",
            "doi": "10.3168/jdsc.2023-0442",
            "url": "https://doi.org/10.3168/jdsc.2023-0442",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "signal_consensus_support",
            "innovation_pressure": "Frequency-domain RR evidence supports FFT/autocorrelation agreement as an auxiliary quality signal.",
        },
        {
            "source_id": "shu_2024_multi_cow_cv_rr",
            "citation_short": "Computers and Electronics in Agriculture, 2024",
            "title": "Non-contact respiration rate measurement of multiple cows in a free-stall barn using computer vision methods",
            "year": 2024,
            "venue": "Computers and Electronics in Agriculture",
            "doi": "10.1016/j.compag.2024.108678",
            "url": "https://doi.org/10.1016/j.compag.2024.108678",
            "crossref_status": "official_publisher_verified_2026_07_13",
            "crossref_checked": "2026-07-13",
            "paper_use": "session_and_deployment_context",
            "innovation_pressure": "Multi-cow RR deployment requires explicit tracking and session provenance; temporal smoothing must still be empirically validated rather than assumed beneficial.",
        },
        {
            "source_id": "cattle_rr_roi_2025",
            "citation_short": "Computers and Electronics in Agriculture, 2025",
            "title": "Respiratory rate regression in Holstein dairy cattle with deep neural networks: An evaluation on different body regions",
            "year": 2025,
            "venue": "Computers and Electronics in Agriculture",
            "doi": "10.1016/j.compag.2025.110987",
            "url": "https://doi.org/10.1016/j.compag.2025.110987",
            "crossref_status": "official_publisher_verified_2026_07_13",
            "crossref_checked": "2026-07-13",
            "paper_use": "roi_selection_and_robustness_context",
            "innovation_pressure": "Viable frame and ROI selection are active RR research themes, but geometry-normalized ROI choices must transfer across farms before becoming a main claim.",
        },
        {
            "source_id": "wang_2024_rgb_videomae_rr",
            "citation_short": "Journal of Dairy Science, 2024",
            "title": "Learning end-to-end respiratory rate prediction of dairy cows from red, green, and blue videos",
            "year": 2024,
            "venue": "Journal of Dairy Science",
            "doi": "10.3168/jds.2023-24601",
            "url": "https://doi.org/10.3168/jds.2023-24601",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "future_end_to_end_context",
            "innovation_pressure": "End-to-end video learning is active, but the current small thermal-curve dataset is better suited to interpretable signal correction.",
        },
        {
            "source_id": "yan_2024_environmental_ml_rr",
            "citation_short": "Biosystems Engineering, 2024",
            "title": "A comparative study of machine learning models for respiration rate prediction in dairy cows: Exploring algorithms, feature engineering, and model interpretation",
            "year": 2024,
            "venue": "Biosystems Engineering",
            "doi": "10.1016/j.biosystemseng.2024.01.010",
            "url": "https://doi.org/10.1016/j.biosystemseng.2024.01.010",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "environmental_context_support",
            "innovation_pressure": "Environmental features make RR biologically meaningful; current manuscript must avoid THI/heat-stress claims until synchronized metadata exist.",
        },
        {
            "source_id": "tresoldi_2025_rr_physiology_context",
            "citation_short": "Journal of Dairy Science, 2025",
            "title": "A comprehensive study of respiration rates in dairy cattle in a Mediterranean climate",
            "year": 2025,
            "venue": "Journal of Dairy Science",
            "doi": "10.3168/jds.2024-26038",
            "url": "https://doi.org/10.3168/jds.2024-26038",
            "crossref_status": "pubmed_verified_2026_07_13",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "rr_physiology_context",
            "innovation_pressure": "RR varies with weather, life stage, breed, posture, location, production, and gestation; camera-derived RR alone must not be relabeled as heat stress without synchronized biological context.",
        },
        {
            "source_id": "reed_2026_multifarm_weather_rr",
            "citation_short": "Journal of Dairy Science, 2026",
            "title": "Unravelling grazing dairy cows' response to weather through respiration rate and drooling",
            "year": 2026,
            "venue": "Journal of Dairy Science",
            "doi": "10.3168/jds.2026-28317",
            "url": "https://doi.org/10.3168/jds.2026-28317",
            "crossref_status": "official_publisher_verified_2026_07_13",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "multifarm_physiology_context",
            "innovation_pressure": "Recent multi-farm RR evidence reinforces the value of cross-site validation and shows that environmental interpretation needs matched weather and animal-response observations.",
        },
        {
            "source_id": "fmcw_2022_radar_rr",
            "citation_short": "Computers and Electronics in Agriculture, 2022",
            "title": "Frequency modulated continuous wave radar-based system for monitoring dairy cow respiration rate",
            "year": 2022,
            "venue": "Computers and Electronics in Agriculture",
            "doi": "10.1016/j.compag.2022.106913",
            "url": "https://doi.org/10.1016/j.compag.2022.106913",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "noncontact_modality_context",
            "innovation_pressure": "Non-contact RR is broader than cameras; thermal-video claims should include quality and review triage to be deployment-relevant.",
        },
        {
            "source_id": "yu_2025_multicam_cow_reid",
            "citation_short": "Computers and Electronics in Agriculture, 2025",
            "title": "Holstein-Friesian re-identification using multiple cameras and self-supervision on a working farm",
            "year": 2025,
            "venue": "Computers and Electronics in Agriculture",
            "doi": "10.1016/j.compag.2025.110568",
            "url": "https://doi.org/10.1016/j.compag.2025.110568",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "cow_identity_validation_context",
            "innovation_pressure": "Identity-aware PLF work raises the standard for real cow_id, camera, and session grouping.",
        },
        {
            "source_id": "animalformer_2024_plf_multimodal",
            "citation_short": "CVPRW, 2024",
            "title": "AnimalFormer: Multimodal Vision Framework for Behavior-based Precision Livestock Farming",
            "year": 2024,
            "venue": "2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops",
            "doi": "10.1109/cvprw63382.2024.00795",
            "url": "https://doi.org/10.1109/cvprw63382.2024.00795",
            "crossref_status": "verified",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "multimodal_plf_context",
            "innovation_pressure": "PLF vision is moving toward multi-trait and multimodal pipelines; current work can claim multi-signal curve-quality engineering, not a full multimodal monitoring system.",
        },
        {
            "source_id": "sadeghi_2024_calves_thermal_vitals",
            "citation_short": "arXiv, 2024",
            "title": "Non-Invasive Monitoring of Vital Signs in Calves Using Thermal Imaging Technology",
            "year": 2024,
            "venue": "arXiv",
            "doi": "",
            "url": "https://arxiv.org/abs/2405.11532",
            "crossref_status": "web_verified_arxiv_2026_07_10",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "thermal_vital_sign_tracking_context",
            "innovation_pressure": "Recent calf thermal-vital-sign work uses ROI tracking and pixel-level signal processing; this supports tracking, signal reliability, and Bland-Altman style agreement checks rather than only mean accuracy.",
        },
        {
            "source_id": "bhujel_2024_plf_cv_dataset_survey",
            "citation_short": "arXiv, 2024",
            "title": "Public Computer Vision Datasets for Precision Livestock Farming: A Systematic Survey",
            "year": 2024,
            "venue": "arXiv",
            "doi": "",
            "url": "https://arxiv.org/abs/2406.10628",
            "crossref_status": "web_verified_arxiv_2026_07_10",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "dataset_and_metadata_gap_support",
            "innovation_pressure": "The survey identifies scarce high-quality annotated PLF datasets and missing contextual metadata as bottlenecks; this directly supports the external-validation and metadata gates.",
        },
        {
            "source_id": "scott_2024_outdoor_livestock_cv_review",
            "citation_short": "arXiv, 2024",
            "title": "Systematic Literature Review of Vision-Based Approaches to Outdoor Livestock Monitoring with Lessons from Wildlife Studies",
            "year": 2024,
            "venue": "arXiv",
            "doi": "",
            "url": "https://arxiv.org/abs/2410.05041",
            "crossref_status": "web_verified_arxiv_2026_07_10",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "real_world_monitoring_challenge_context",
            "innovation_pressure": "Outdoor and farm-scale vision monitoring reviews emphasize motion, occlusion, habitat variation, and 24/7 monitoring challenges; these map to the frame-motion and manual-quality gates.",
        },
        {
            "source_id": "wang_2026_cross_farm_transfer_learning",
            "citation_short": "arXiv, 2026",
            "title": "Evaluating transfer learning strategies for improving dairy cattle body weight prediction in small farms using depth-image and point-cloud data",
            "year": 2026,
            "venue": "arXiv",
            "doi": "",
            "url": "https://arxiv.org/abs/2601.01044",
            "crossref_status": "web_verified_arxiv_2026_07_10",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "cross_farm_generalization_context",
            "innovation_pressure": "Although it targets body weight, the cross-farm transfer-learning result reinforces that farm/session shifts must be tested explicitly before broad PLF generalization claims.",
        },
        {
            "source_id": "yan_2024_cn_athi_rf_rr",
            "citation_short": "Chinese agricultural engineering paper, 2024",
            "title": "Random-forest respiratory-rate prediction with hyperparameter optimization and ATHI-related features",
            "year": 2024,
            "venue": "Transactions of the Chinese Society of Agricultural Engineering",
            "doi": "10.11975/j.issn.1002-6819.202401090",
            "url": "https://doi.org/10.11975/j.issn.1002-6819.202401090",
            "crossref_status": "crossref_404_local_or_publisher_record_only",
            "crossref_checked": CROSSREF_CHECK_DATE,
            "paper_use": "local_environmental_context_support",
            "innovation_pressure": "ATHI strengthens the heat-stress story, but current environment columns are still missing and must not be fabricated.",
        },
    ]
    return pd.DataFrame(rows)


def build_gap_matrix(output_dir: Path) -> pd.DataFrame:
    main_results = read_csv(output_dir / "paper_main_results_table.csv")
    default = first_contains(main_results, "method", "Default thermal RR pipeline")
    safe_gate = first_contains(
        main_results,
        "method",
        "Conservative signal-aware safe gate (fixed threshold",
    )
    prefix_safe = first_contains(
        main_results,
        "method",
        "Conservative signal-aware safe gate (leave-one-prefix",
    )
    signal = first_contains(main_results, "method", "Signal-consensus supplement (fixed")
    truth = first_contains(main_results, "method", "Truth-calibrated upper bound")
    consensus = read_csv(output_dir / "paper_calibration_free_consensus_ensemble_selection.csv")
    stability = read_csv(output_dir / "paper_calibration_free_stability_gate_selection.csv")
    geometry = read_csv(output_dir / "paper_calibration_free_geometry_roi_metrics.csv")
    session = read_csv(output_dir / "paper_calibration_free_session_context_metrics.csv")
    track_metrics = read_csv(output_dir / "paper_calibration_free_track_stabilized_roi_metrics.csv")
    track_selection = read_csv(output_dir / "paper_calibration_free_track_stabilized_roi_selected_strategy.csv")
    track_bootstrap = read_csv(output_dir / "paper_calibration_free_track_stabilized_roi_bootstrap_ci.csv")
    mapping_calibration = read_csv(output_dir / "paper_temperature_mapping_source_parity_calibration.csv")
    coherence_metrics = read_csv(output_dir / "paper_bilateral_coherence_frequency_metrics.csv")
    p2g_session_metrics = read_csv(
        output_dir / "paper_p2g_desktop_single_annotator_provisional_session_metrics.csv"
    )
    track_session_metrics = read_csv(
        output_dir / "paper_track_stabilized_v2_desktop_single_annotator_provisional_session_metrics.csv"
    )
    weather_proxy_daily = read_csv(
        output_dir / "paper_lindian_nasa_power_daily_weather_proxy.csv"
    )
    weather_proxy_video = read_csv(
        output_dir / "paper_lindian_nasa_power_video_weather_proxy.csv"
    )

    ablation_evidence = "calibration-free ablation battery not generated"
    if consensus is not None and stability is not None and geometry is not None and session is not None:
        geometry_external = geometry[
            geometry.get("cohort", pd.Series("", index=geometry.index)).astype(str).eq("external_provisional")
            & geometry.get("method", pd.Series("", index=geometry.index)).astype(str).eq(
                "duration_gated_calibration_free_consensus_ensemble"
            )
        ]
        session_best = session.sort_values("rr_rmse").iloc[0] if not session.empty else None
        if not consensus.empty and not stability.empty and not geometry_external.empty and session_best is not None:
            ablation_evidence = (
                f"robust aggregation selected={consensus.iloc[0].get('aggregation', 'NA')}; "
                f"kinematic selection={stability.iloc[0].get('rule', 'NA')}; "
                f"geometry external={metric_evidence(geometry_external.iloc[0])}; "
                f"best session-context={session_best.get('method', 'NA')} "
                f"({metric_evidence(session_best)})"
            )

    track_evidence = "bilateral trajectory-stabilized ROI candidate not yet generated"
    if track_metrics is not None and track_selection is not None and not track_metrics.empty and not track_selection.empty:
        track = track_metrics.iloc[0]
        selected_strategy = str(track_selection.iloc[0].get("strategy", "unknown"))
        delta = track_bootstrap[
            track_bootstrap.get("metric", pd.Series("", index=track_bootstrap.index))
            .astype(str)
            .eq("rr_r2")
        ] if track_bootstrap is not None else pd.DataFrame()
        delta_text = "clustered delta unavailable"
        if not delta.empty:
            row = delta.iloc[0]
            delta_text = (
                f"P2g delta R2={float(row['estimate']):.4f} "
                f"(95% CI {float(row['ci_low']):.4f} to {float(row['ci_high']):.4f})"
            )
        track_evidence = (
            f"strategy={selected_strategy}; provisional external {metric_evidence(track)}; "
            f"{delta_text}; 109 primary-plus-extension predictions frozen without manual RR read"
        )

    mapping_evidence = "temperature-mapping source-parity audit not generated"
    if mapping_calibration is not None and not mapping_calibration.empty:
        direct = mapping_calibration[
            mapping_calibration.get("input_order", pd.Series("", index=mapping_calibration.index))
            .astype(str)
            .eq("opencv_bgr_source_parity")
        ]
        reverse = mapping_calibration[
            mapping_calibration.get("input_order", pd.Series("", index=mapping_calibration.index))
            .astype(str)
            .eq("channel_reversed_rgb_negative_control")
        ]
        if not direct.empty and not reverse.empty:
            mapping_evidence = (
                f"source-parity OpenCV BGR calibration R2={float(direct.iloc[0]['r2']):.4f}, "
                f"MAE={float(direct.iloc[0]['mae_c']):.4f} C; channel-reversed negative "
                f"control R2={float(reverse.iloc[0]['r2']):.4f}"
            )

    coherence_evidence = "bilateral coherence-frequency ablation not generated"
    if coherence_metrics is not None and not coherence_metrics.empty:
        internal_coherence = coherence_metrics[
            coherence_metrics.get("cohort", pd.Series("", index=coherence_metrics.index))
            .astype(str)
            .eq("internal_development")
            & coherence_metrics.get("method", pd.Series("", index=coherence_metrics.index))
            .astype(str)
            .eq("fixed_bilateral_coherence_frequency")
        ]
        external_coherence = coherence_metrics[
            coherence_metrics.get("cohort", pd.Series("", index=coherence_metrics.index))
            .astype(str)
            .eq("external_provisional")
            & coherence_metrics.get("method", pd.Series("", index=coherence_metrics.index))
            .astype(str)
            .eq("duration_gated_fixed_bilateral_coherence_frequency")
        ]
        if not internal_coherence.empty and not external_coherence.empty:
            coherence_evidence = (
                f"bilateral coherence-frequency negative ablation: internal "
                f"{metric_evidence(internal_coherence.iloc[0])}; provisional external gated "
                f"{metric_evidence(external_coherence.iloc[0])}"
            )

    session_evidence = "source-session endpoint not yet generated"
    if (
        p2g_session_metrics is not None
        and track_session_metrics is not None
        and not p2g_session_metrics.empty
        and not track_session_metrics.empty
    ):
        p2g_session = p2g_session_metrics.iloc[0]
        track_session = track_session_metrics.iloc[0]
        session_evidence = (
            "single-annotator time-weighted source-session endpoint: "
            f"P2g (sessions={int(p2g_session['source_sessions'])}, "
            f"R2={float(p2g_session['rr_r2']):.4f}, MAE={float(p2g_session['rr_mae']):.3f}); "
            f"track v2 (R2={float(track_session['rr_r2']):.4f}, "
            f"MAE={float(track_session['rr_mae']):.3f})"
        )

    weather_proxy_evidence = "county-day weather proxy not generated"
    if (
        weather_proxy_daily is not None
        and weather_proxy_video is not None
        and not weather_proxy_daily.empty
        and not weather_proxy_video.empty
    ):
        assigned = int(
            pd.to_numeric(
                weather_proxy_video.get(
                    "weather_proxy_t2m_c", pd.Series(np.nan, index=weather_proxy_video.index)
                ),
                errors="coerce",
            )
            .notna()
            .sum()
        )
        weather_proxy_evidence = (
            "NASA POWER/MERRA2 county-midpoint daily proxy for collection window: "
            f"days={len(weather_proxy_daily)}, T2M={float(weather_proxy_daily['weather_proxy_t2m_c'].min()):.2f}"
            f"-{float(weather_proxy_daily['weather_proxy_t2m_c'].max()):.2f} C, "
            f"RH2M={float(weather_proxy_daily['weather_proxy_rh2m_percent'].min()):.2f}"
            f"-{float(weather_proxy_daily['weather_proxy_rh2m_percent'].max()):.2f}%; "
            f"per-video dates assigned={assigned}/{len(weather_proxy_video)}"
        )

    metadata_snapshot = "; ".join(
        [
            field_coverage(output_dir, "cow_id"),
            field_coverage(output_dir, "collection_start_date"),
            field_coverage(output_dir, "collection_end_date"),
            field_coverage(output_dir, "collection_location_country"),
            field_coverage(output_dir, "collection_location_province"),
            field_coverage(output_dir, "collection_location_county"),
            field_coverage(output_dir, "collection_site"),
            field_coverage(output_dir, "external_test_split"),
            field_coverage(output_dir, "ambient_temperature_c"),
            field_coverage(output_dir, "relative_humidity_percent"),
            field_coverage(output_dir, "thi"),
            field_coverage(output_dir, "head_motion_score_0_3"),
            field_coverage(output_dir, "occlusion_score_0_3"),
            field_coverage(output_dir, "nostril_visibility_score_0_3"),
        ]
    )

    rows = [
        {
            "priority": "P0",
            "gap": "Residual breath-count errors after nostril temperature extraction",
            "supporting_sources": "lin_2025_head_movement_irt_rr; zhao_2023_irt_dl_rr; mantovani_2024_fft_rr",
            "implemented_or_next_innovation": "Keep the conservative signal-aware safe gate as the main current internal precision candidate.",
            "current_evidence": f"default: {metric_evidence(default)}; safe gate: {metric_evidence(safe_gate)}; prefix stress: {metric_evidence(prefix_safe)}",
            "paper_position": "Main internal algorithm contribution",
            "safe_claim": "The method improves internal RR precision and exact breath-count agreement under method-frozen internal evaluation.",
            "boundary": "Not external validation; report bootstrap/McNemar uncertainty and prefix-group stress separately.",
        },
        {
            "priority": "P1",
            "gap": "Peak counting alone is brittle under short, noisy, or boundary-truncated respiratory curves",
            "supporting_sources": "mantovani_2024_fft_rr; fmcw_2022_radar_rr",
            "implemented_or_next_innovation": "Use FFT/autocorrelation/spectral count agreement as a supplementary consensus feature, not a truth-assisted override.",
            "current_evidence": f"signal consensus: {metric_evidence(signal)}",
            "paper_position": "Secondary precision mechanism or supplement",
            "safe_claim": "Independent signal-count agreement supports error triage and conservative correction.",
            "boundary": "Do not promote above the safe gate until external paired evidence is available.",
        },
        {
            "priority": "P2",
            "gap": "Head movement, occlusion, and nostril visibility are central reviewer questions",
            "supporting_sources": "lin_2025_head_movement_irt_rr; sadeghi_2024_calves_thermal_vitals; animalformer_2024_plf_multimodal; scott_2024_outdoor_livestock_cv_review",
            "implemented_or_next_innovation": "Use the existing metadata annotation pack, frame-motion probe, and blinded external annotation workflow to stratify or at least triage these cases.",
            "current_evidence": metadata_snapshot,
            "paper_position": "Required Q2+ validation extension",
            "safe_claim": "The current dataset contains numeric video-label cow IDs and a motion/quality annotation workflow.",
            "boundary": "No robustness claim until head_motion, occlusion, and nostril_visibility are filled or automatically measured with verified rules.",
        },
        {
            "priority": "P2b",
            "gap": "Cross-farm calibration-free ROI and temporal choices can appear plausible but need transfer evidence",
            "supporting_sources": "lin_2025_head_movement_irt_rr; cattle_rr_roi_2025; shu_2024_multi_cow_cv_rr",
            "implemented_or_next_innovation": "Retain duration-gated mean P2g as the current main route. Robust aggregation, kinematic filtering, source-session context, and geometry-normalized ROI scaling remain audited negative ablations; the new bilateral trajectory-stabilized ROI is a separately frozen cross-domain candidate.",
            "current_evidence": f"{mapping_evidence}; {ablation_evidence}; track candidate: {track_evidence}; {session_evidence}; {coherence_evidence}",
            "paper_position": "Exploratory cross-domain candidate with a frozen extension pilot",
            "safe_claim": "Bilateral midpoint/vector stabilization increases track availability under external detector dropout and is now frozen for untouched extension recount. The frozen scorer now reports both clip-level and time-weighted source-session endpoints.",
            "boundary": "The Jiufu labels are single-annotator and development-inspected. Its provisional gain has a clustered interval spanning zero, so it cannot displace P2g or enter primary manuscript results until blinded A/B consensus or untouched external validation is complete. Source-session results are a deployment-relevant secondary endpoint, not a replacement for the complete clip-level flow. The bilateral coherence-frequency ablation failed internally and is retained only as a negative result.",
        },
        {
            "priority": "P3",
            "gap": "RR needs physiological meaning beyond an accuracy number",
            "supporting_sources": "yan_2024_environmental_ml_rr; yan_2024_cn_athi_rf_rr; tresoldi_2025_rr_physiology_context; reed_2026_multifarm_weather_rr",
            "implemented_or_next_innovation": "Use quality-aware cross-farm RR measurement as the current paper's operational contribution. Add synchronized barn temperature, humidity, THI/ATHI, posture, animal state, and collection date before any heat-stress or welfare-response claim.",
            "current_evidence": f"{known_collection_context(output_dir)}; {weather_proxy_evidence}; {metadata_snapshot}",
            "paper_position": "Biological meaning and dairy-science journal fit",
            "safe_claim": "Videos were collected at Lindian County ranch, Heilongjiang, China, from 2023-08-05 to 2023-08-10. A cached county-day NASA POWER proxy describes the collection window only.",
            "boundary": "Do not write heat-stress monitoring or THI-stratified performance from regional/source provenance alone. The current proxy is a county-midpoint daily source with 0/73 per-video date assignments, not a barn sensor or synchronized environmental measurement.",
        },
        {
            "priority": "P4",
            "gap": "External generalization now depends on identity, camera, date, and farm/session separation",
            "supporting_sources": "bhujel_2024_plf_cv_dataset_survey; scott_2024_outdoor_livestock_cv_review; yu_2025_multicam_cow_reid; shu_2024_multi_cow_cv_rr; wang_2024_rgb_videomae_rr; wang_2026_cross_farm_transfer_learning; animalformer_2024_plf_multimodal",
            "implemented_or_next_innovation": "Treat all current rows as internal; use the split_all_use Hulunbuir/Jiufu batch as an independent external set after blinded A/B manual RR consensus; apply frozen postprocessors without external-truth fitting, then score the external split.",
            "current_evidence": f"{readiness_evidence(output_dir)}; {external_workflow_evidence(output_dir)}",
            "paper_position": "Submission gate, not optional polish",
            "safe_claim": "The method is frozen and the external validation workflow is ready for manual RR consensus and scoring.",
            "boundary": "Do not label current 73 development videos as external; do not report external RR R2 until A/B consensus truth and frozen scoring outputs exist.",
        },
        {
            "priority": "P5",
            "gap": "Truth-assisted calibration can hide leakage",
            "supporting_sources": "methodological leakage guard",
            "implemented_or_next_innovation": "Keep truth-calibrated RR as an oracle upper bound only.",
            "current_evidence": f"truth-calibrated: {metric_evidence(truth)}",
            "paper_position": "Supplementary diagnostic",
            "safe_claim": "Truth-assisted performance shows remaining signal information in the curves.",
            "boundary": "Do not use truth-calibrated RR R2 in abstract, highlights, or primary performance claims.",
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


def write_report(
    report_path: Path,
    source_table: pd.DataFrame,
    gap_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = "\n".join(
        [
            "# Recent Literature Refresh For Thermal RR Innovation",
            "",
            f"Generated: {generated}",
            f"Source verification refresh date: {CROSSREF_CHECK_DATE}",
            "",
            "This refresh is evidence-gated. The table is restricted to DOI-verified core sources, local publisher-only sources, web-verified recent preprints/reviews, and current project outputs.",
            "",
            "## Source Register",
            "",
            markdown_table(
                source_table,
                [
                    "source_id",
                    "year",
                    "venue",
                    "doi",
                    "crossref_status",
                    "paper_use",
                    "innovation_pressure",
                ],
            ),
            "",
            "## Gap-To-Innovation Matrix",
            "",
            markdown_table(
                gap_matrix,
                [
                    "priority",
                    "gap",
                    "supporting_sources",
                    "implemented_or_next_innovation",
                    "current_evidence",
                    "safe_claim",
                    "boundary",
                ],
            ),
            "",
            "## Recommended Q2+ Storyline",
            "",
            "The strongest current route is an internally validated algorithm paper centered on conservative signal-aware residual correction, supported by frequency-domain agreement and explicit uncertainty/review triage. The paper should present Lindian County, Heilongjiang provenance and the current metadata audit as transparency, not as heat-stress or external-validation evidence.",
            "",
            "The Q2-or-higher route becomes substantially stronger only after adding a frozen external split with real cow/session/camera separation and synchronized environmental or quality metadata. Without that, the safe submission posture is an internal-method paper with clear external-validation and heat-stress limitations.",
            "",
            "## Current Metadata Boundary",
            "",
            f"- {known_collection_context(output_dir)}",
            f"- {field_coverage(output_dir, 'cow_id')}",
            f"- {field_coverage(output_dir, 'collection_start_date')}",
            f"- {field_coverage(output_dir, 'collection_end_date')}",
            f"- {field_coverage(output_dir, 'collection_location_country')}",
            f"- {field_coverage(output_dir, 'collection_location_province')}",
            f"- {field_coverage(output_dir, 'collection_location_county')}",
            f"- {field_coverage(output_dir, 'collection_site')}",
            f"- {field_coverage(output_dir, 'external_test_split')}",
            f"- {field_coverage(output_dir, 'ambient_temperature_c')}",
            f"- {field_coverage(output_dir, 'relative_humidity_percent')}",
            f"- {field_coverage(output_dir, 'thi')}",
            f"- {field_coverage(output_dir, 'head_motion_score_0_3')}",
            f"- {field_coverage(output_dir, 'occlusion_score_0_3')}",
            f"- {field_coverage(output_dir, 'nostril_visibility_score_0_3')}",
            "",
        ]
    )
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    source_table = source_register()
    gap_matrix = build_gap_matrix(output_dir)

    source_csv = output_dir / "paper_literature_recent_source_verification.csv"
    gap_csv = output_dir / "paper_literature_recent_gap_innovation_matrix.csv"
    report_md = output_dir / "paper_literature_recent_refresh.md"
    docs_md = args.docs_dir / "thermal_rr_literature_recent_refresh.md"

    source_table.to_csv(source_csv, index=False)
    gap_matrix.to_csv(gap_csv, index=False)
    write_report(report_md, source_table, gap_matrix, output_dir)
    docs_md.write_text(report_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved recent source verification: {source_csv}")
    print(f"Saved recent gap matrix: {gap_csv}")
    print(f"Saved recent literature refresh: {report_md}")
    print(f"Saved docs copy: {docs_md}")
    print(gap_matrix[["priority", "gap", "paper_position"]].to_string(index=False))


if __name__ == "__main__":
    main()
