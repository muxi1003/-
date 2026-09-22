from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a literature-to-innovation crosswalk for the thermal RR "
            "manuscript package. The output links recent dairy-cow RR and PLF "
            "literature to the current implemented evidence, remaining gates, "
            "and manuscript wording."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def first_contains(table: pd.DataFrame | None, column: str, text: str) -> pd.Series | None:
    if table is None or table.empty or column not in table.columns:
        return None
    mask = table[column].astype(str).str.contains(text, case=False, regex=False)
    if not mask.any():
        return None
    return table.loc[mask].iloc[0]


def exact(row: pd.Series | None) -> str:
    if row is None:
        return "NA"
    value = row.get("exact_count", "NA")
    if isinstance(value, str) and "/" in value:
        return value
    total = row.get("videos", row.get("count_valid_videos", "NA"))
    try:
        return f"{int(value)}/{int(total)}"
    except (TypeError, ValueError):
        return str(value)


def method_evidence(row: pd.Series | None) -> str:
    if row is None:
        return "not generated"
    return (
        f"RR R2={fmt(row.get('rr_r2'))}; "
        f"MAE={fmt(row.get('rr_mae_bpm', row.get('rr_mae')), 3)} bpm; "
        f"RMSE={fmt(row.get('rr_rmse_bpm', row.get('rr_rmse')), 3)} bpm; "
        f"exact={exact(row)}"
    )


def readiness_text(readiness: pd.DataFrame | None) -> str:
    if readiness is None or readiness.empty:
        return "readiness audit not generated"
    status = "unknown"
    if "q2_submission_status" in readiness.columns:
        status = str(readiness["q2_submission_status"].dropna().iloc[0])
    blockers = readiness[
        readiness.get("q2_blocking", pd.Series(False, index=readiness.index)).astype(bool)
        & readiness.get("status", pd.Series("", index=readiness.index))
        .astype(str)
        .isin(["FAIL", "WARN"])
    ]
    names = "; ".join(blockers["check"].astype(str).head(6).tolist())
    if len(blockers) > 6:
        names += f"; +{len(blockers) - 6} more"
    return f"Q2 status={status}; blockers={names or 'none'}"


def metadata_text(output_dir: Path) -> str:
    progress = read_csv(output_dir / "paper_metadata_annotation_progress.csv")
    heat = read_csv(output_dir / "paper_heat_stress_readiness.csv")
    pieces: list[str] = []
    if progress is not None and not progress.empty and "field" in progress.columns:
        by_field = {str(row["field"]): row for _, row in progress.iterrows()}

        def field(field_name: str) -> str:
            row = by_field.get(field_name)
            if row is None:
                return f"{field_name}=not generated"
            return (
                f"{field_name}={int(float(row.get('nonempty', 0)))}/"
                f"{int(float(row.get('total_videos', 0)))}"
            )

        pieces.append(
            ", ".join(
                [
                    field("cow_id"),
                    field("scene_id"),
                    field("ambient_temperature_c"),
                    field("relative_humidity_percent"),
                    field("thi"),
                    field("external_test_split"),
                ]
            )
        )
    if heat is not None and not heat.empty:
        thi_category = heat[heat["field"].astype(str) == "thi_category"]
        if not thi_category.empty:
            row = thi_category.iloc[0]
            pieces.append(
                "THI category not analysis-ready "
                f"(ready={row.get('ready', 'NA')}, unique={row.get('unique_values', 'NA')})"
            )
    return "; ".join(pieces) if pieces else "metadata progress not generated"


def source_register() -> pd.DataFrame:
    rows = [
        {
            "source_id": "zhao_2023_irt_dl",
            "citation_short": "Zhao et al., 2023",
            "title": "Detection of Respiratory Rate of Dairy Cows Based on Infrared Thermography and Deep Learning",
            "venue": "Agriculture",
            "year": 2023,
            "doi_or_url": "https://doi.org/10.3390/agriculture13101939",
            "source_type": "peer_reviewed_article",
            "local_pdf": "agriculture-13-01939-v2.pdf",
            "relevance": "Closest thermal-infrared dairy-cow RR baseline.",
            "limitation_for_gap": "Pipeline-level RR detection is relevant, but Q2 claims still need stronger robustness, residual-error analysis, and external validation.",
        },
        {
            "source_id": "mantovani_2024_fft",
            "citation_short": "Mantovani et al., 2024",
            "title": "Predicting respiration rate in unrestrained dairy cows using image analysis and fast Fourier transform",
            "venue": "JDS Communications",
            "year": 2024,
            "doi_or_url": "https://doi.org/10.3168/jdsc.2023-0442",
            "source_type": "peer_reviewed_article",
            "local_pdf": "1-s2.0-S2666910223001217-main.pdf",
            "relevance": "Supports frequency-domain RR estimation in unrestrained dairy cows.",
            "limitation_for_gap": "FFT-only or image-motion routes do not solve nostril thermal ROI quality and residual peak errors by themselves.",
        },
        {
            "source_id": "wang_2024_rgb_end_to_end",
            "citation_short": "Wang et al., 2024",
            "title": "Learning end-to-end respiratory rate prediction of dairy cows from red, green, and blue videos",
            "venue": "Journal of Dairy Science",
            "year": 2024,
            "doi_or_url": "https://doi.org/10.3168/jds.2023-24601",
            "source_type": "peer_reviewed_article",
            "local_pdf": "1-s2.0-S0022030224010300-main.pdf",
            "relevance": "Shows that end-to-end video learning is an active RR direction.",
            "limitation_for_gap": "RGB end-to-end models need larger, identity-aware video data; current thermal dataset is better suited for method-frozen signal processing plus residual correction.",
        },
        {
            "source_id": "yan_2024_ml_comparison",
            "citation_short": "Yan et al., 2024",
            "title": "A comparative study of machine learning models for respiration rate prediction in dairy cows",
            "venue": "Biosystems Engineering",
            "year": 2024,
            "doi_or_url": "https://doi.org/10.1016/j.biosystemseng.2024.01.010",
            "source_type": "peer_reviewed_article",
            "local_pdf": "1-s2.0-S1537511024000163-main.pdf",
            "relevance": "Supports environmental and feature-engineered RR prediction, including model interpretation.",
            "limitation_for_gap": "Environmental prediction is complementary, not a replacement for direct non-contact RR measurement; current regional THI proxies support context but not synchronized barn-level heat-stress inference.",
        },
        {
            "source_id": "sadeghi_2024_thermal_calves",
            "citation_short": "Sadeghi et al., 2024",
            "title": "Non-Invasive Monitoring of Vital Signs in Calves Using Thermal Imaging Technology",
            "venue": "arXiv",
            "year": 2024,
            "doi_or_url": "https://doi.org/10.48550/arXiv.2405.11532",
            "source_type": "preprint",
            "local_pdf": "",
            "relevance": "Recent thermal vital-sign work uses ROI tracking and signal processing for respiration.",
            "limitation_for_gap": "Calf vital-sign setting is related but not the same as dairy-cow nostril thermal RR; external validation remains needed.",
        },
        {
            "source_id": "yu_2025_multicamcows",
            "citation_short": "Yu et al., 2025",
            "title": "Holstein-Friesian Re-Identification using Multiple Cameras and Self-Supervision on a Working Farm",
            "venue": "Computers and Electronics in Agriculture / arXiv",
            "year": 2025,
            "doi_or_url": "https://doi.org/10.1016/j.compag.2025.110568",
            "source_type": "peer_reviewed_article_preprint_record",
            "local_pdf": "",
            "relevance": "Supports cow identity and multi-camera grouping as a PLF generalization requirement.",
            "limitation_for_gap": "Current cow_id values are derived from video numeric labels; stronger identity, camera, and session records are still needed for deployment claims.",
        },
        {
            "source_id": "yan_2024_bo_rf_local_cn",
            "citation_short": "Yan et al., 2024 local Chinese PDF",
            "title": "Predicting respiratory rate of dairy cows using hyperparameter-optimized random forest models",
            "venue": "Transactions of the Chinese Society of Agricultural Engineering",
            "year": 2024,
            "doi_or_url": "https://doi.org/10.11975/j.issn.1002-6819.202401090",
            "source_type": "peer_reviewed_article_local_pdf",
            "local_pdf": "hyperparameter_optimized_rf_rr_chinese.pdf",
            "relevance": "Highlights ATHI/environment features as meaningful RR context.",
            "limitation_for_gap": "The model predicts RR from environmental and production variables; current thermal paper has regional THI context but still lacks ATHI, production variables, and synchronized environmental strata.",
        },
        {
            "source_id": "li_2019_rr_monitoring_review",
            "citation_short": "Li et al., 2019 local Chinese review",
            "title": "Review of automatic monitoring technology for dairy cow respiratory rate",
            "venue": "China Dairy Cattle",
            "year": 2019,
            "doi_or_url": "https://doi.org/10.19556/j.0258-7033.20190130-07",
            "source_type": "review_local_pdf",
            "local_pdf": "automatic_monitoring_rr_review_chinese.pdf",
            "relevance": "Supports the broader need for automatic, non-contact RR monitoring.",
            "limitation_for_gap": "Review-level support is background; current contribution must be validated with present data and external testing.",
        },
    ]
    return pd.DataFrame(rows)


def build_crosswalk(
    source_table: pd.DataFrame,
    main_results: pd.DataFrame | None,
    stats: pd.DataFrame | None,
    readiness: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame:
    default = first_contains(main_results, "method", "Default thermal RR pipeline")
    quality = first_contains(main_results, "method", "Quality-aware residual correction (fixed threshold")
    signal = first_contains(main_results, "method", "Signal-consensus supplement (fixed")
    signal_aware = first_contains(main_results, "method", "Signal-aware residual correction (fixed")
    signal_group = first_contains(main_results, "method", "Signal-aware residual correction (leave-one-prefix")
    truth = first_contains(main_results, "method", "Truth-calibrated upper bound")
    signal_aware_stats = first_contains(stats, "method_id", "signal_aware_fixed_oof")

    signal_aware_delta = "not generated"
    if signal_aware_stats is not None:
        signal_aware_delta = (
            f"vs default: Delta RR R2={fmt(signal_aware_stats.get('delta_rr_r2'))}; "
            f"Delta MAE={fmt(signal_aware_stats.get('delta_rr_mae'), 3)} bpm; "
            f"Delta exact={signal_aware_stats.get('delta_exact_count')}; "
            f"McNemar one-sided p={fmt(signal_aware_stats.get('mcnemar_exact_p_method_better'), 4)}"
        )

    rows = [
        {
            "manuscript_claim": "Thermal nostril RR can be reproduced with a non-contact IRT pipeline.",
            "literature_basis": "zhao_2023_irt_dl; sadeghi_2024_thermal_calves; li_2019_rr_monitoring_review",
            "gap_or_pressure": "Thermal RR is relevant, but pipeline errors and ROI/signal quality must be explicitly handled.",
            "implemented_response": "YOLO nostril/nose localization, nostril temperature curves, respiratory-rate curve generation, and method-frozen evaluation.",
            "current_project_evidence": f"default baseline: {method_evidence(default)}",
            "recommended_paper_location": "Methods and baseline results",
            "safe_wording": "We reproduced a non-contact thermal RR pipeline and used it as the baseline for residual-error analysis.",
            "do_not_claim_yet": "Do not claim external deployment or heat-stress monitoring until metadata and external validation pass.",
        },
        {
            "manuscript_claim": "Quality-aware residual correction improves internal RR accuracy.",
            "literature_basis": "zhao_2023_irt_dl; mantovani_2024_fft; yan_2024_ml_comparison",
            "gap_or_pressure": "Prior modular or signal-processing pipelines can propagate ROI and peak-detection errors.",
            "implemented_response": "Out-of-fold residual correction using curve quality and peak context, with sensitivity and grouped stress tests.",
            "current_project_evidence": f"quality-aware fixed OOF: {method_evidence(quality)}",
            "recommended_paper_location": "Main results as internal innovation",
            "safe_wording": "The quality-aware residual correction improved the internal out-of-fold estimate over the default pipeline.",
            "do_not_claim_yet": "Do not call it externally validated until true animal identity, session metadata, and external splits are available.",
        },
        {
            "manuscript_claim": "Frequency-domain agreement is a defensible precision supplement.",
            "literature_basis": "mantovani_2024_fft; sadeghi_2024_thermal_calves",
            "gap_or_pressure": "FFT/autocorrelation evidence can stabilize peak counts, but should not be tuned with held-out truth.",
            "implemented_response": "Signal-consensus supplement combines spectral, FFT, and autocorrelation count evidence without truth calibration.",
            "current_project_evidence": f"signal-consensus fixed OOF: {method_evidence(signal)}",
            "recommended_paper_location": "Supplementary or secondary internal experiment",
            "safe_wording": "Signal-consensus features improved internal precision and motivate a signal-quality extension.",
            "do_not_claim_yet": "Do not promote above the frozen main method without external paired evidence.",
        },
        {
            "manuscript_claim": "Signal-aware residual correction is the highest-precision internal candidate.",
            "literature_basis": "mantovani_2024_fft; sadeghi_2024_thermal_calves; wang_2024_rgb_end_to_end",
            "gap_or_pressure": "Modern video/signal methods point toward richer temporal models, but small datasets are leakage-prone.",
            "implemented_response": "A second-stage signal-aware residual model uses non-truth signal features and out-of-fold predictions.",
            "current_project_evidence": (
                f"signal-aware fixed OOF: {method_evidence(signal_aware)}; "
                f"{signal_aware_delta}; prefix-group stress: {method_evidence(signal_group)}"
            ),
            "recommended_paper_location": "Internal candidate result with caution paragraph",
            "safe_wording": "The signal-aware model achieved the strongest internal precision, but its prefix-group stress test prevents broad generalization wording.",
            "do_not_claim_yet": "Do not present as the primary deployable method until true animal identity/session metadata and external validation improve.",
        },
        {
            "manuscript_claim": "Environmental and heat-stress meaning requires THI/ATHI metadata.",
            "literature_basis": "yan_2024_ml_comparison; yan_2024_bo_rf_local_cn; li_2019_rr_monitoring_review",
            "gap_or_pressure": "RR is biologically meaningful for heat stress, but direct thermal RR alone cannot prove heat-stress context.",
            "implemented_response": "Metadata template, Q2 unblock queue, Lindian County regional weather-proxy THI fields, and readiness audits separate contextual covariates from synchronized heat-stress evidence.",
            "current_project_evidence": f"{metadata_text(output_dir)}; {readiness_text(readiness)}",
            "recommended_paper_location": "Limitations and Q2 readiness plan",
            "safe_wording": "The present internal experiment estimates RR and reports regional THI context for the Lindian County collection window.",
            "do_not_claim_yet": "Do not write heat-stress monitoring, THI-stratified performance, or barn-level thermal exposure inference without synchronized or variable environmental measurements.",
        },
        {
            "manuscript_claim": "True animal-identity generalization and external validation are required for Q2+ strength.",
            "literature_basis": "yu_2025_multicamcows; wang_2024_rgb_end_to_end",
            "gap_or_pressure": "Current PLF computer-vision work increasingly emphasizes identity, multi-camera data, and reproducible splits.",
            "implemented_response": "Method-freeze manifest, numeric-video-label metadata stress tests, external split validation scripts, and Q2 unblock queue.",
            "current_project_evidence": f"{metadata_text(output_dir)}; {readiness_text(readiness)}",
            "recommended_paper_location": "Validation design and limitations",
            "safe_wording": "We froze the method and ran internal metadata-group validation using numeric video-label cow_id values.",
            "do_not_claim_yet": "Do not state cross-farm, cross-camera, date-level, or external generalization while all current split labels are internal and no true external videos are present.",
        },
        {
            "manuscript_claim": "Truth-calibrated results are only an oracle upper bound.",
            "literature_basis": "methodological leakage guard",
            "gap_or_pressure": "Reviewer risk: label-assisted calibration can inflate performance.",
            "implemented_response": "Truth-calibrated outputs are separated and marked upper_bound_only in reporting tiers.",
            "current_project_evidence": f"truth-calibrated: {method_evidence(truth)}",
            "recommended_paper_location": "Supplementary diagnostic only",
            "safe_wording": "A truth-assisted upper-bound analysis showed remaining signal information but was excluded from deployable performance claims.",
            "do_not_claim_yet": "Do not include truth-calibrated RR R2 in abstract, highlights, or main method performance.",
        },
    ]
    crosswalk = pd.DataFrame(rows)
    source_lookup = source_table.set_index("source_id")["doi_or_url"].to_dict()
    crosswalk["source_links"] = crosswalk["literature_basis"].apply(
        lambda value: "; ".join(
            source_lookup.get(item.strip(), "")
            for item in str(value).split(";")
            if source_lookup.get(item.strip(), "")
        )
    )
    return crosswalk


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("\n", " ") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_markdown(path: Path, source_table: pd.DataFrame, crosswalk: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Literature-Innovation Crosswalk For Thermal RR Manuscript",
            "",
            "This document links the current implemented evidence to local PDFs and current public literature. It is intended for the Introduction, Related Work, Discussion, and Limitations sections.",
            "",
            "## Source Register",
            "",
            markdown_table(
                source_table,
                [
                    "source_id",
                    "citation_short",
                    "year",
                    "venue",
                    "doi_or_url",
                    "relevance",
                    "limitation_for_gap",
                ],
            ),
            "",
            "## Claim Crosswalk",
            "",
            markdown_table(
                crosswalk,
                [
                    "manuscript_claim",
                    "literature_basis",
                    "gap_or_pressure",
                    "implemented_response",
                    "current_project_evidence",
                    "safe_wording",
                    "do_not_claim_yet",
                ],
            ),
            "",
            "## Manuscript Use Boundary",
            "",
            "- Use default, quality-aware residual, and fixed out-of-fold signal-aware rows as internal-validation evidence.",
            "- Keep truth-calibrated results as an oracle upper-bound supplement only.",
            "- Do not make Q2+ external, true animal-identity, heat-stress, or deployment claims until metadata and external validation gates pass.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_results = read_csv(output_dir / "paper_main_results_table.csv")
    stats = read_csv(output_dir / "paper_rr_method_statistical_tests_table.csv")
    readiness = read_csv(output_dir / "paper_submission_readiness_table.csv")

    sources = source_register()
    crosswalk = build_crosswalk(sources, main_results, stats, readiness, output_dir)
    sources_csv = output_dir / "paper_literature_source_register.csv"
    crosswalk_csv = output_dir / "paper_literature_innovation_crosswalk.csv"
    crosswalk_md = output_dir / "paper_literature_innovation_crosswalk.md"
    sources.to_csv(sources_csv, index=False)
    crosswalk.to_csv(crosswalk_csv, index=False)
    write_markdown(crosswalk_md, sources, crosswalk)
    print(f"Saved source register: {sources_csv}")
    print(f"Saved innovation crosswalk: {crosswalk_csv}")
    print(f"Saved innovation crosswalk report: {crosswalk_md}")
    print(crosswalk[["manuscript_claim", "recommended_paper_location", "current_project_evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
