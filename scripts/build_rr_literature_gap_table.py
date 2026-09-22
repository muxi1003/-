from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build a paper-ready literature gap table for the thermal RR innovation plan."
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--local-lit-root",
        type=Path,
        default=Path("E:/real/\u5b66\u4e60/\u547c\u5438\u6587\u732e"),
        help="Folder containing local respiratory-rate PDFs.",
    )
    parser.add_argument(
        "--secondary-lit-root",
        type=Path,
        default=Path("E:/real/\u53ef\u80fd\u7ed3\u5408\u6587\u732e"),
        help="Optional secondary folder containing related PDFs.",
    )
    parser.add_argument("--output-prefix", default="paper_literature")
    return parser.parse_args()


def literature_rows() -> list[dict[str, object]]:
    return [
        {
            "source_id": "jtherbio_2025_head_movement_thermal_rr",
            "year": 2025,
            "source_type": "journal_article",
            "title": "Respiratory rate detection of dairy cows based on infrared thermography in head movement scenarios",
            "venue": "Journal of Thermal Biology",
            "doi_or_url": "https://doi.org/10.1016/j.jtherbio.2025.104154",
            "local_files": "1-s2.0-S0306456525001111-main.pdf",
            "main_method": (
                "YOLOv8n-Pose nostril keypoints, random-forest RGB-to-temperature "
                "mapping, and two-nostril thermal-curve fusion."
            ),
            "reported_evidence": (
                "Reported R2 = 0.92, RMSE = 3.53 bpm, and average RR detection "
                "accuracy = 96.3% on 246 thermal videos."
            ),
            "gap_for_current_work": (
                "Already covers nostril keypoints and temperature mapping under head "
                "movement; novelty should target residual peak errors, curve quality, "
                "and validation rather than repeating localization."
            ),
            "repo_innovation_link": "P0 quality-aware residual correction; P3 signal consensus; P1 motion/occlusion metadata.",
            "paper_role": "primary_baseline_and_gap",
        },
        {
            "source_id": "agriculture_2023_yolov8_thermal_rr",
            "year": 2023,
            "source_type": "journal_article",
            "title": "Detection of Respiratory Rate of Dairy Cows Based on Infrared Thermography and Deep Learning",
            "venue": "Agriculture",
            "doi_or_url": "https://doi.org/10.3390/agriculture13101939",
            "local_files": "agriculture-13-01939-v2.pdf",
            "main_method": (
                "Infrared thermography with YOLOv8 nose detection, nostril "
                "segmentation, nostril temperature extraction, and sliding-window "
                "peak counting."
            ),
            "reported_evidence": "Reported RR accuracy = 94.58% and correlation coefficient R = 0.95 on 81 infrared videos.",
            "gap_for_current_work": (
                "Strong modular baseline, but fixed peak decisions and limited "
                "metadata validation leave room for quality-aware correction and "
                "deployment triage."
            ),
            "repo_innovation_link": "P0 residual peak-count correction; P4 selective automatic-report/manual-review triage.",
            "paper_role": "technical_baseline",
        },
        {
            "source_id": "jds_2024_rgb_videomae_rr",
            "year": 2024,
            "source_type": "journal_article",
            "title": "Learning end-to-end respiratory rate prediction of dairy cows from red, green, and blue videos",
            "venue": "Journal of Dairy Science",
            "doi_or_url": "https://doi.org/10.3168/jds.2023-24601",
            "local_files": "1-s2.0-S0022030224010300-main.pdf",
            "main_method": "End-to-end RGB VideoMAE respiratory-rate prediction with respiratory-belt reference.",
            "reported_evidence": "Reported MAE = 2.58 bpm, RMSE = 3.52 bpm, and Pearson r = 0.86 from 6 dairy cows.",
            "gap_for_current_work": (
                "Shows the potential of end-to-end video learning, but the current "
                "73-video thermal-curve dataset is better suited to interpretable "
                "small-data signal correction."
            ),
            "repo_innovation_link": "Future P5 end-to-end thermal/RGB fusion after larger grouped data collection.",
            "paper_role": "future_direction_not_main_claim",
        },
        {
            "source_id": "jdsc_2024_fft_unrestrained_rr",
            "year": 2024,
            "source_type": "journal_article",
            "title": "Predicting respiration rate in unrestrained dairy cows using image analysis and fast Fourier transform",
            "venue": "JDS Communications",
            "doi_or_url": "https://doi.org/10.3168/jdsc.2023-0442",
            "local_files": "1-s2.0-S2666910223001217-main.pdf",
            "main_method": "Training-free image analysis with FFT-based respiratory-rate estimation for unrestrained lying cows.",
            "reported_evidence": "Supports frequency-domain RR estimation as a non-contact alternative to peak counting.",
            "gap_for_current_work": (
                "Frequency methods can fail when signals are short or noisy, but they "
                "are useful as independent evidence for peak-count correction."
            ),
            "repo_innovation_link": "P3 FFT/autocorrelation/spectral signal-consensus supplement.",
            "paper_role": "supports_signal_consensus",
        },
        {
            "source_id": "biosystemseng_2024_environmental_ml_rr",
            "year": 2024,
            "source_type": "journal_article",
            "title": "A comparative study of machine learning models for respiration rate prediction in dairy cows: Exploring algorithms, feature engineering, and model interpretation",
            "venue": "Biosystems Engineering",
            "doi_or_url": "https://doi.org/10.1016/j.biosystemseng.2024.01.010",
            "local_files": "1-s2.0-S1537511024000163-main.pdf",
            "main_method": "Comparison of 20 machine-learning algorithms using production and environmental variables.",
            "reported_evidence": "CatBoost with environmental variables performed best; reported R2 = 0.676 and MAE = 7.246.",
            "gap_for_current_work": (
                "Environmental variables add biological meaning and interpretability, "
                "but indirect RR prediction is less precise than direct thermal/video RR."
            ),
            "repo_innovation_link": "P2 THI/ATHI heat-stress context; metadata-stratified performance analysis.",
            "paper_role": "supports_biological_context",
        },
        {
            "source_id": "chinese_borf_athi_rr",
            "year": 2024,
            "source_type": "journal_article_local_pdf",
            "title": "Random-forest respiratory-rate prediction with hyperparameter optimization and ATHI-related features",
            "venue": "Transactions of the Chinese Society of Agricultural Engineering",
            "doi_or_url": "https://doi.org/10.11975/j.issn.1002-6819.202401090",
            "local_files": "\u57fa\u4e8e\u8d85\u53c2\u6570\u4f18\u5316\u7b97\u6cd5\u7684\u968f\u673a\u68ee\u6797\u6a21\u578b\u9884\u6d4b\u5976\u725b\u547c\u5438\u9891\u7387.pdf",
            "main_method": "BO-RF model using ATHI, time region, milk yield, days in milk, posture, and parity.",
            "reported_evidence": "Reported BO-RF R2 = 0.614, MAE = 7.723 bpm, MAPE = 14.4%, RMSE = 9.737 bpm; ATHI was the dominant feature.",
            "gap_for_current_work": "Justifies collecting ATHI and cow-level production metadata before making heat-stress claims.",
            "repo_innovation_link": "P2 heat-stress metadata template and readiness gate.",
            "paper_role": "supports_metadata_collection",
        },
        {
            "source_id": "chinese_rr_monitoring_review",
            "year": 2019,
            "source_type": "review_local_pdf",
            "title": "Review of automatic monitoring technologies for dairy cow respiratory rate",
            "venue": "China Dairy Cattle",
            "doi_or_url": "https://doi.org/10.19556/j.0258-7033.20190130-07",
            "local_files": "\u5976\u725b\u547c\u5438\u9891\u7387\u81ea\u52a8\u76d1\u6d4b\u6280\u672f\u7814\u7a76\u8fdb\u5c55.pdf",
            "main_method": "Review of contact and non-contact RR monitoring technologies.",
            "reported_evidence": "Provides background for manual, sensor, thermal, and vision-based RR monitoring routes.",
            "gap_for_current_work": "Useful for Introduction framing, but not enough to justify the core novelty by itself.",
            "repo_innovation_link": "Introduction and related-work framing.",
            "paper_role": "background_review",
        },
        {
            "source_id": "icssas_2025_iot_cow_health",
            "year": 2025,
            "source_type": "conference_article",
            "title": "IoT-enabled device for predictive monitoring and disease management in cow",
            "venue": "ICSSAS",
            "doi_or_url": "https://doi.org/10.1109/ICSSAS66150.2025.11081385",
            "local_files": "1781525583024_25474811_725256_60_ieeeconference_IoT-EnabledDeviceforPredictiveMonitoringandDiseaseManagementinCow.pdf",
            "main_method": "IoT sensing of temperature, humidity, heart rate, and respiratory rate with cloud dashboard and machine-learning health classification.",
            "reported_evidence": "Reports controlled-condition health-status classification accuracy of 80% to 95%.",
            "gap_for_current_work": "Supports the broader value of RR as a health-management signal, but its hardware and synthetic-data setting are not the main thermal-video contribution.",
            "repo_innovation_link": "Future deployment context: combine RR confidence, environmental data, and health alerts.",
            "paper_role": "future_health_monitoring_context",
        },
        {
            "source_id": "compag_2022_fmcw_radar_rr",
            "year": 2022,
            "source_type": "journal_article",
            "title": "Frequency modulated continuous wave radar-based system for monitoring dairy cow respiration rate",
            "venue": "Computers and Electronics in Agriculture",
            "doi_or_url": "https://doi.org/10.1016/j.compag.2022.106913",
            "local_files": "",
            "main_method": "FMCW radar-based non-contact dairy-cow respiratory-rate monitoring.",
            "reported_evidence": "Shows that non-contact RR monitoring is not limited to camera or thermal modalities.",
            "gap_for_current_work": (
                "Radar can be robust to lighting but requires different hardware; thermal "
                "video needs uncertainty and quality reporting to be deployable."
            ),
            "repo_innovation_link": "Motivates quality-aware confidence and review triage for camera-based RR.",
            "paper_role": "noncontact_modality_comparison",
        },
        {
            "source_id": "arxiv_2024_plf_cv_datasets_survey",
            "year": 2024,
            "source_type": "preprint_or_survey",
            "title": "Public Computer Vision Datasets for Precision Livestock Farming: A Systematic Survey",
            "venue": "arXiv",
            "doi_or_url": "https://arxiv.org/abs/2406.10628",
            "local_files": "",
            "main_method": "Systematic survey of public computer-vision datasets for precision livestock farming.",
            "reported_evidence": (
                "Identifies limited high-quality annotated datasets, diverse environments, "
                "and contextual metadata as bottlenecks in PLF CV research."
            ),
            "gap_for_current_work": "Directly supports the need for metadata, external splits, and careful evidence boundaries.",
            "repo_innovation_link": "P1 metadata/external-validation protocol and Q2 readiness audit.",
            "paper_role": "supports_validation_gap",
        },
        {
            "source_id": "arxiv_2024_multicam_cows_reid",
            "year": 2024,
            "source_type": "preprint_dataset",
            "title": "MultiCamCows2024: A multi-view image dataset for AI-driven Holstein-Friesian cattle re-identification on a working farm",
            "venue": "arXiv",
            "doi_or_url": "https://arxiv.org/abs/2410.12695",
            "local_files": "",
            "main_method": "Multi-camera cow re-identification with supervised and self-supervised baselines.",
            "reported_evidence": "Reports a farm-scale multi-camera dataset and high single-image identification accuracy.",
            "gap_for_current_work": "Shows a route to true cow identity grouping, but does not convert the current numeric video-label cow_id into animal identity metadata.",
            "repo_innovation_link": "P1 numeric-video-label stress testing now; future P6 true cow-identity GroupKFold and identity-aware deployment.",
            "paper_role": "supports_cow_identity_validation",
        },
        {
            "source_id": "arxiv_2025_multicamera_tracking",
            "year": 2025,
            "source_type": "preprint",
            "title": "Vision transformer-based multi-camera multi-object tracking framework for dairy cow monitoring",
            "venue": "arXiv",
            "doi_or_url": "https://arxiv.org/abs/2508.01752",
            "local_files": "",
            "main_method": "Multi-camera segmentation, tracking, geometric alignment, and motion-aware association.",
            "reported_evidence": "Reports high multi-object tracking accuracy and low identity switches in dairy-cow monitoring videos.",
            "gap_for_current_work": "Relevant to future robust deployment but too large for the current 73-video thermal RR paper.",
            "repo_innovation_link": "Future P6 multi-camera tracking and identity-stable RR aggregation.",
            "paper_role": "future_deployment_direction",
        },
        {
            "source_id": "cvprw_2024_animalformer",
            "year": 2024,
            "source_type": "conference_workshop",
            "title": "AnimalFormer: Multimodal Vision Framework for Behavior-based Precision Livestock Farming",
            "venue": "CVPR Workshops",
            "doi_or_url": "https://doi.org/10.1109/CVPRW63382.2024.00795",
            "local_files": "",
            "main_method": "Multimodal vision framework combining detection, segmentation, pose, and behavior analytics.",
            "reported_evidence": "Supports the trend toward multi-trait and multimodal PLF pipelines.",
            "gap_for_current_work": "Current paper can borrow the multi-trait idea at the signal-feature level, not claim a full multimodal PLF system.",
            "repo_innovation_link": "P0/P3 multi-feature quality signal design; future multimodal extension.",
            "paper_role": "supports_multi_trait_design",
        },
    ]


def resolve_local_files(row: dict[str, object], roots: list[Path]) -> dict[str, object]:
    files = [name.strip() for name in str(row.get("local_files", "")).split(";") if name.strip()]
    found: list[str] = []
    for file_name in files:
        for root in roots:
            path = root / file_name
            if path.exists():
                found.append(str(path))
                break
    row["local_pdf_present"] = bool(files) and len(found) == len(files)
    row["resolved_local_paths"] = "; ".join(found)
    return row


def first_match(table: pd.DataFrame | None, contains: str) -> pd.Series | None:
    if table is None or table.empty or "method" not in table.columns:
        return None
    mask = table["method"].astype(str).str.contains(contains, case=False, regex=False)
    if not mask.any():
        return None
    return table[mask].iloc[0]


def metric_text(row: pd.Series | None) -> str:
    if row is None:
        return "current metric row not available"
    return (
        f"RR R2={float(row['rr_r2']):.6f}, "
        f"MAE={float(row['rr_mae_bpm']):.4f} bpm, "
        f"RMSE={float(row['rr_rmse_bpm']):.4f} bpm, "
        f"exact={row['exact_count']}"
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def metadata_gap_text(readiness: pd.DataFrame | None) -> str:
    if readiness is None or readiness.empty:
        return "submission readiness table not available"
    q2_items = readiness[
        ((readiness["status"].astype(str) == "FAIL") | (readiness["status"].astype(str) == "WARN"))
        & readiness["q2_blocking"].astype(bool)
    ]
    if q2_items.empty:
        return "no Q2 blocking or caution item currently reported"
    checks = "; ".join(q2_items["check"].astype(str).head(6))
    return f"{len(q2_items)} Q2 blocking/caution checks remain: {checks}"


def selective_text(selective_metrics: pd.DataFrame | None) -> str:
    if selective_metrics is None or selective_metrics.empty:
        return "selective-reporting metrics not available"
    strict = selective_metrics[
        selective_metrics["label"].astype(str).str.contains("strict", case=False, regex=False)
    ]
    if strict.empty:
        strict = selective_metrics.head(1)
    row = strict.iloc[0]
    parts = []
    for column in ["accepted_videos", "videos", "rr_r2", "rr_mae", "exact_count"]:
        if column in row:
            parts.append(f"{column}={row[column]}")
    return ", ".join(parts)


def build_innovation_priority_table(output_dir: Path, input_root: Path) -> pd.DataFrame:
    metric_tiers = read_csv_if_exists(output_dir / "paper_metric_reporting_tiers.csv")
    readiness = read_csv_if_exists(input_root / "paper_repro_quality_residual_submission_readiness.csv")
    selective_metrics = read_csv_if_exists(input_root / "paper_repro_selective_rr_metrics.csv")

    default = first_match(metric_tiers, "Default thermal RR pipeline")
    fixed = first_match(metric_tiers, "fixed threshold, out-of-fold")
    nested = first_match(metric_tiers, "nested threshold CV")
    signal_fixed = first_match(metric_tiers, "Signal-consensus supplement (fixed threshold")
    signal_nested = first_match(metric_tiers, "Signal-consensus supplement (nested threshold")
    truth_upper = first_match(metric_tiers, "Truth-calibrated upper bound")
    q2_gap = metadata_gap_text(readiness)

    return pd.DataFrame(
        [
            {
                "priority": "P0",
                "innovation_point": "Quality-aware residual peak-count correction",
                "literature_basis": "jtherbio_2025_head_movement_thermal_rr; agriculture_2023_yolov8_thermal_rr; cvprw_2024_animalformer",
                "current_repo_evidence": f"default baseline: {metric_text(default)}; fixed OOF: {metric_text(fixed)}; nested: {metric_text(nested)}",
                "paper_use": "main internal algorithm innovation",
                "claim_strength_now": "strong_internal_not_external",
                "q2_value": "improves precision while staying interpretable for small thermal-curve data",
                "remaining_gate": q2_gap,
                "next_experiment": "Freeze this as the main method; validate by real cow_id/date/camera grouping and external split.",
                "requires_new_data_or_metadata": "metadata_or_external_split",
            },
            {
                "priority": "P1",
                "innovation_point": "Real metadata-stratified validation and frozen external split",
                "literature_basis": "arxiv_2024_plf_cv_datasets_survey; arxiv_2024_multicam_cows_reid",
                "current_repo_evidence": q2_gap,
                "paper_use": "Q2+ submission gate, not optional polish",
                "claim_strength_now": "not_ready",
                "q2_value": "turns an internal reproduction into animal-level and scene-level generalization evidence",
                "remaining_gate": "cow_id, scene_id, regional weather proxies, and internal split labels are filled; exact collection_date, camera_id, manual quality scores, and nonzero external rows remain missing",
                "next_experiment": "Recover acquisition date/camera if records exist, keep current rows internal, collect or mark a method-frozen external set, then rerun metadata quality and frozen external validation.",
                "requires_new_data_or_metadata": "metadata_required",
            },
            {
                "priority": "P2",
                "innovation_point": "Heat-stress meaning with THI/ATHI and environmental stratification",
                "literature_basis": "biosystemseng_2024_environmental_ml_rr; chinese_borf_athi_rr",
                "current_repo_evidence": "regional NASA POWER temperature, humidity, and THI proxy values are filled for 73/73 videos, but they are constant collection-window context values rather than synchronized barn measurements.",
                "paper_use": "biological significance and journal-fit section",
                "claim_strength_now": "context_ready_stratified_data_missing",
                "q2_value": "connects RR estimation to welfare and heat-stress management instead of reporting only CV metrics",
                "remaining_gate": "THI/ATHI claims need synchronized barn sensor measurements or multiple real THI strata, plus posture/production context if available",
                "next_experiment": "Collect per-video or session-matched barn temperature and humidity, compute THI/ATHI, and report RR error by thermal-stress strata.",
                "requires_new_data_or_metadata": "metadata_required",
            },
            {
                "priority": "P3",
                "innovation_point": "FFT/autocorrelation signal-consensus precision supplement",
                "literature_basis": "jdsc_2024_fft_unrestrained_rr; agriculture_2023_yolov8_thermal_rr",
                "current_repo_evidence": f"fixed signal consensus: {metric_text(signal_fixed)}; nested signal consensus: {metric_text(signal_nested)}",
                "paper_use": "secondary precision extension or supplement",
                "claim_strength_now": "promising_internal_candidate",
                "q2_value": "uses frequency-domain agreement to reduce peak-count residual errors without truth-calibrated overrides",
                "remaining_gate": "still internal and should not replace the method-frozen main result unless externally validated",
                "next_experiment": "Keep as supplement unless external split confirms improvement over P0.",
                "requires_new_data_or_metadata": "external_validation_preferred",
            },
            {
                "priority": "P4",
                "innovation_point": "Selective automatic-report versus manual-review triage",
                "literature_basis": "compag_2022_fmcw_radar_rr; arxiv_2024_plf_cv_datasets_survey",
                "current_repo_evidence": selective_text(selective_metrics),
                "paper_use": "deployment reliability supplement",
                "claim_strength_now": "internal_triage_only",
                "q2_value": "makes the system operational by separating high-confidence automatic reports from review-required cases",
                "remaining_gate": "coverage/accuracy trade-off must be frozen and tested prospectively or externally",
                "next_experiment": "Report coverage beside accuracy and use the external split to test the review gate.",
                "requires_new_data_or_metadata": "external_validation_preferred",
            },
            {
                "priority": "P5",
                "innovation_point": "End-to-end RGB/thermal transformer fusion",
                "literature_basis": "jds_2024_rgb_videomae_rr",
                "current_repo_evidence": "not implemented for this 73-video thermal-curve dataset",
                "paper_use": "future work only for this manuscript",
                "claim_strength_now": "do_not_claim",
                "q2_value": "could be a later paper if more grouped raw videos and independent labels are collected",
                "remaining_gate": "current sample size is too small for a defensible transformer main claim",
                "next_experiment": "Collect a larger multi-session dataset before implementing this as a main contribution.",
                "requires_new_data_or_metadata": "new_video_data_required",
            },
            {
                "priority": "P6",
                "innovation_point": "Identity-aware multi-camera tracking and farm deployment",
                "literature_basis": "arxiv_2024_multicam_cows_reid; arxiv_2025_multicamera_tracking",
                "current_repo_evidence": "not implemented; cow_id is currently a video-derived numeric label, not identity tracking evidence",
                "paper_use": "future deployment direction, not current contribution",
                "claim_strength_now": "do_not_claim",
                "q2_value": "would support long-term repeated-measure monitoring and true animal-level generalization",
                "remaining_gate": "requires identity labels, multi-camera data, and tracking evaluation",
                "next_experiment": "Verify whether numeric cow_id labels map to individual cows, then collect identity labels or multi-camera data before automatic re-identification.",
                "requires_new_data_or_metadata": "identity_metadata_required",
            },
            {
                "priority": "Guardrail",
                "innovation_point": "Truth-calibrated RR is only an oracle upper bound",
                "literature_basis": "current metric reporting tiers; leakage guard in submission readiness audit",
                "current_repo_evidence": metric_text(truth_upper),
                "paper_use": "supplementary diagnostic only",
                "claim_strength_now": "upper_bound_not_method_performance",
                "q2_value": "shows the extracted curves contain respiratory information, but does not prove deployable prediction",
                "remaining_gate": "must never be used as primary performance or external-validation evidence",
                "next_experiment": "Keep out of the abstract/main result table; use only in an upper-bound paragraph if needed.",
                "requires_new_data_or_metadata": "no",
            },
        ]
    )


def write_innovation_markdown(df: pd.DataFrame, output_path: Path, search_date: str) -> None:
    columns = [
        "priority",
        "innovation_point",
        "paper_use",
        "claim_strength_now",
        "current_repo_evidence",
        "remaining_gate",
        "next_experiment",
    ]
    table = df[columns].fillna("").astype(str)
    lines = [
        "# Literature-Grounded Innovation Priority Table",
        "",
        f"Search/cross-check date: {search_date}.",
        "",
        "This file converts the literature gap table into manuscript decisions. Use it to decide what can be written as the main contribution, what should remain supplementary, and what still blocks Q2-or-higher submission.",
        "",
        "## Decision Summary",
        "",
        "- Main contribution now: P0 quality-aware residual peak-count correction.",
        "- Main Q2+ blocker: P1/P2 metadata and external validation, not another truth-assisted tuning run.",
        "- Secondary precision result: P3 signal-consensus supplement.",
        "- Deployment supplement: P4 selective report/review triage.",
        "- Do not claim as method performance: truth-calibrated RR R2.",
        "",
        "## Priority Table",
        "",
    ]
    lines.append("| " + " | ".join(table.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "source_id",
        "year",
        "venue",
        "main_method",
        "reported_evidence",
        "gap_for_current_work",
        "repo_innovation_link",
        "paper_role",
    ]
    table = df[columns].fillna("").astype(str)
    lines = [
        "# Literature-Grounded Innovation Gap Table",
        "",
        "This table is a targeted, paper-support evidence map for the thermal dairy-cow respiratory-rate workflow. It is not a formal systematic review.",
        "",
        f"Search/cross-check date: {date.today().isoformat()}.",
        "",
        "## Recommended Claim Boundary",
        "",
        "- Main claim: quality-aware residual peak-count correction and signal-consensus screening improve internal thermal-video RR validation.",
        "- Submission gate: Q2-or-higher claims still need cow_id/date/scene metadata, heat-stress context, and an external or frozen holdout split.",
        "- Upper-bound only: truth-calibrated RR results must remain an oracle/diagnostic analysis.",
        "",
        "## Evidence Table",
        "",
    ]
    lines.append("| " + " | ".join(table.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    lines.extend(
        [
            "",
            "## Source Links",
            "",
        ]
    )
    for _, row in df.iterrows():
        link = str(row.get("doi_or_url", "")).strip()
        if link:
            lines.append(f"- {row['source_id']}: {link}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / "paper_repro_quality_residual_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [args.local_lit_root.resolve(), args.secondary_lit_root.resolve()]
    rows = [resolve_local_files(row, roots) for row in literature_rows()]
    df = pd.DataFrame(rows)
    search_date = date.today().isoformat()
    df["search_crosscheck_date"] = search_date
    df["evidence_boundary"] = df["paper_role"].map(
        {
            "primary_baseline_and_gap": "main_related_work",
            "technical_baseline": "main_related_work",
            "supports_signal_consensus": "method_rationale",
            "supports_biological_context": "meaning_and_metadata",
            "supports_metadata_collection": "meaning_and_metadata",
            "future_health_monitoring_context": "future_work",
            "supports_validation_gap": "validation_rationale",
            "supports_cow_identity_validation": "validation_rationale",
            "future_direction_not_main_claim": "future_work",
            "future_deployment_direction": "future_work",
        }
    ).fillna("supporting_context")

    csv_path = output_dir / f"{args.output_prefix}_gap_table.csv"
    md_path = output_dir / f"{args.output_prefix}_gap_table.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)

    innovation_df = build_innovation_priority_table(output_dir, input_root)
    innovation_csv_path = output_dir / f"{args.output_prefix}_innovation_priority.csv"
    innovation_md_path = output_dir / f"{args.output_prefix}_innovation_priority.md"
    innovation_df.to_csv(innovation_csv_path, index=False)
    write_innovation_markdown(innovation_df, innovation_md_path, search_date)

    print(f"Saved literature gap table: {csv_path}")
    print(f"Saved literature gap report: {md_path}")
    print(f"Saved innovation priority table: {innovation_csv_path}")
    print(f"Saved innovation priority report: {innovation_md_path}")
    print(df[["source_id", "year", "paper_role", "local_pdf_present"]].to_string(index=False))


if __name__ == "__main__":
    main()
