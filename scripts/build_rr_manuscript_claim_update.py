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
            "Generate manuscript-ready wording updates from the current RR "
            "metrics, paired statistical tests, readiness gates, and "
            "literature-innovation crosswalk."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def row_contains(table: pd.DataFrame, column: str, text: str) -> pd.Series:
    mask = table[column].astype(str).str.contains(text, case=False, regex=False)
    if not mask.any():
        raise ValueError(f"Could not find row containing {text!r} in {column}")
    return table.loc[mask].iloc[0]


def exact(row: pd.Series) -> str:
    value = row.get("exact_count", "NA")
    if isinstance(value, str) and "/" in value:
        return value
    total = row.get("videos", row.get("count_valid_videos", "NA"))
    try:
        return f"{int(value)}/{int(total)}"
    except (TypeError, ValueError):
        return str(value)


def method_metrics(row: pd.Series) -> str:
    return (
        f"RR R2={fmt(row.get('rr_r2'), 4)}, "
        f"MAE={fmt(row.get('rr_mae_bpm', row.get('rr_mae')))} breaths/min, "
        f"RMSE={fmt(row.get('rr_rmse_bpm', row.get('rr_rmse')))} breaths/min, "
        f"exact={exact(row)}"
    )


def readiness_status(readiness: pd.DataFrame) -> str:
    status = "unknown"
    if "q2_submission_status" in readiness.columns:
        values = readiness["q2_submission_status"].dropna()
        if not values.empty:
            status = str(values.iloc[0])
    blockers = readiness[
        readiness.get("q2_blocking", pd.Series(False, index=readiness.index)).astype(bool)
        & readiness.get("status", pd.Series("", index=readiness.index))
        .astype(str)
        .isin(["FAIL", "WARN"])
    ]
    names = "; ".join(blockers["check"].astype(str).head(8).tolist())
    if len(blockers) > 8:
        names += f"; +{len(blockers) - 8} more"
    return f"{status}; blockers: {names}"


def metadata_context(assets: Path) -> str:
    progress_path = assets / "paper_metadata_annotation_progress.csv"
    heat_path = assets / "paper_heat_stress_readiness.csv"
    prefill_path = assets / "paper_metadata_annotation_context_prefill_report.md"
    parts: list[str] = []

    if progress_path.exists():
        progress = pd.read_csv(progress_path)
        coverage = {
            str(row["field"]): (
                int(float(row.get("nonempty", 0))),
                int(float(row.get("total_videos", 0))),
                int(float(row.get("unique_values", 0))),
            )
            for _, row in progress.iterrows()
        }

        def cov(field: str) -> str:
            nonempty, total, unique = coverage.get(field, (0, 0, 0))
            return f"{field}={nonempty}/{total} ({unique} unique)"

        parts.append(
            "current metadata coverage: "
            + ", ".join(
                [
                    cov("cow_id"),
                    cov("scene_id"),
                    cov("ambient_temperature_c"),
                    cov("relative_humidity_percent"),
                    cov("thi"),
                    cov("external_test_split"),
                ]
            )
        )
        missing = [
            field
            for field in [
                "collection_date",
                "camera_id",
                "head_motion_score_0_3",
                "occlusion_score_0_3",
                "nostril_visibility_score_0_3",
            ]
            if coverage.get(field, (0, 0, 0))[0] == 0
        ]
        if missing:
            parts.append("still missing: " + ", ".join(missing))

    if heat_path.exists():
        heat = pd.read_csv(heat_path)
        thi = heat[heat["field"].astype(str) == "thi_category"]
        if not thi.empty:
            row = thi.iloc[0]
            parts.append(
                "THI category is not stratified "
                f"(coverage={row.get('coverage_percent', 'NA')}%, "
                f"unique_values={row.get('unique_values', 'NA')})"
            )

    if prefill_path.exists():
        parts.append(
            "known collection context is Lindian County, Heilongjiang, China, "
            "2023-08-05 to 2023-08-10; synchronized barn temperature/humidity, "
            "THI/ATHI, and manual quality scores remain unavailable"
        )

    return "; ".join(parts) if parts else "metadata context outputs not generated"


def algorithmic_quality_context(assets: Path) -> str:
    path = assets / "paper_algorithmic_quality_tier_metrics_table.csv"
    if not path.exists():
        return "algorithmic quality stratification not generated"
    table = pd.read_csv(path)
    mask = (
        (table["method"].astype(str) == "signal_aware_safe_gate")
        & (table["tier_type"].astype(str) == "fixed_risk_tier")
        & (table["tier"].astype(str) == "low_risk_auto_candidate")
    )
    if not mask.any():
        return "algorithmic quality table generated but low-risk safe-gate row missing"
    row = table.loc[mask].iloc[0]
    return (
        "algorithmic low-risk safe-gate subset "
        f"n={int(row['videos'])}, coverage={float(row['coverage']):.3f}, "
        f"RR R2={float(row['rr_r2']):.4f}, "
        f"MAE={float(row['rr_mae']):.3f} breaths/min, "
        f"exact={int(row['exact_count'])}/{int(row['count_valid_videos'])}"
    )


def conformal_uncertainty_context(assets: Path) -> str:
    path = assets / "paper_conformal_rr_recommendation_table.csv"
    if not path.exists():
        return "conformal RR uncertainty not generated"
    table = pd.read_csv(path)
    if table.empty:
        return "conformal RR uncertainty recommendation is empty"
    row = table.iloc[0]
    return (
        f"{row.get('interval_variant')} target={fmt(row.get('target_coverage'), 2)}, "
        f"coverage={fmt(row.get('coverage'), 4)}, "
        f"mean width={fmt(row.get('mean_width_bpm'), 3)} breaths/min, "
        f"low-risk n={int(float(row.get('low_risk_videos', 0)))}, "
        f"low-risk coverage={fmt(row.get('low_risk_coverage'), 4)}"
    )


def physiological_triage_context(assets: Path) -> str:
    path = assets / "paper_rr_physiological_triage_summary.csv"
    if not path.exists():
        return "RR-only physiological triage not generated"
    table = pd.read_csv(path)
    if table.empty:
        return "RR-only physiological triage summary is empty"

    def by_label(label: str) -> pd.Series:
        row = table[table["label"].astype(str).eq(label)]
        if row.empty:
            return pd.Series(dtype=object)
        return row.iloc[0]

    auto = by_label("auto_report_subset")
    high_auto = by_label("auto_high_rr_candidate")
    upper = by_label("predicted_upper_quartile_rr_candidate")
    if auto.empty:
        return "RR-only physiological triage generated but auto-report row is missing"
    parts = [
        (
            "RR-only low-risk auto-report subset "
            f"n={int(float(auto.get('videos', 0)))}, "
            f"coverage={fmt(auto.get('coverage'), 3)}, "
            f"RR R2={fmt(auto.get('rr_r2'), 4)}, "
            f"MAE={fmt(auto.get('rr_mae'), 3)} breaths/min, "
            f"exact_rate={fmt(auto.get('exact_rate'), 3)}"
        )
    ]
    if not high_auto.empty:
        parts.append(
            "auto-report high-RR candidate subset "
            f"n={int(float(high_auto.get('videos', 0)))}, "
            f"RR R2={fmt(high_auto.get('rr_r2'), 4)}, "
            f"MAE={fmt(high_auto.get('rr_mae'), 3)} breaths/min, "
            f"exact_rate={fmt(high_auto.get('exact_rate'), 3)}"
        )
    if not upper.empty:
        parts.append(
            "all predicted upper-quartile RR candidates "
            f"n={int(float(upper.get('videos', 0)))}, "
            f"MAE={fmt(upper.get('rr_mae'), 3)} breaths/min; "
            "these remain context-review candidates, not heat-stress diagnoses"
        )
    return "; ".join(parts)


def bilateral_consistency_context(assets: Path) -> str:
    path = assets / "paper_rr_bilateral_consistency_summary.csv"
    if not path.exists():
        return "bilateral nostril consistency gate not generated"
    table = pd.read_csv(path)
    if table.empty:
        return "bilateral nostril consistency summary is empty"
    row = table[table["subset"].astype(str).eq("bilateral_consistent_auto_report_subset")]
    if row.empty:
        return "bilateral nostril consistency generated but auto-report row is missing"
    item = row.iloc[0]
    return (
        "bilateral-consistent auto-report subset "
        f"n={int(float(item.get('videos', 0)))}, "
        f"coverage={fmt(item.get('coverage'), 3)}, "
        f"RR R2={fmt(item.get('rr_r2'), 4)}, "
        f"MAE={fmt(item.get('rr_mae_bpm'), 3)} breaths/min, "
        f"exact_rate={fmt(item.get('exact_rate'), 3)}"
    )


def load_tables(input_root: Path, corrected_prefix: str) -> dict[str, pd.DataFrame]:
    assets = input_root / f"{corrected_prefix}_paper_assets"
    return {
        "main": pd.read_csv(assets / "paper_main_results_table.csv"),
        "stats": pd.read_csv(assets / "paper_rr_method_statistical_tests_table.csv"),
        "readiness": pd.read_csv(assets / "paper_submission_readiness_table.csv"),
        "crosswalk": pd.read_csv(assets / "paper_literature_innovation_crosswalk.csv"),
        "_assets_path": assets,
    }


def build_update_text(tables: dict[str, pd.DataFrame]) -> str:
    main = tables["main"]
    stats = tables["stats"]
    readiness = tables["readiness"]
    metadata_status = metadata_context(tables["_assets_path"])
    quality_status = algorithmic_quality_context(tables["_assets_path"])
    conformal_status = conformal_uncertainty_context(tables["_assets_path"])
    triage_status = physiological_triage_context(tables["_assets_path"])
    bilateral_status = bilateral_consistency_context(tables["_assets_path"])
    default = row_contains(main, "method", "Default thermal RR pipeline")
    quality = row_contains(main, "method", "Quality-aware residual correction (fixed threshold")
    signal = row_contains(main, "method", "Signal-consensus supplement (fixed")
    signal_aware = row_contains(main, "method", "Signal-aware residual correction (fixed threshold")
    signal_group = row_contains(main, "method", "Signal-aware residual correction (leave-one-prefix")
    safe_signal = row_contains(main, "method", "Conservative signal-aware safe gate (fixed threshold")
    safe_signal_group = row_contains(
        main,
        "method",
        "Conservative signal-aware safe gate (leave-one-prefix",
    )
    truth = row_contains(main, "method", "Truth-calibrated upper bound")
    signal_aware_stats = row_contains(stats, "method_id", "signal_aware_fixed_oof")
    safe_signal_stats = row_contains(stats, "method_id", "signal_aware_safe_fixed_oof")
    safe_signal_group_stats = row_contains(
        stats,
        "method_id",
        "signal_aware_safe_prefix_group_fixed",
    )
    signal_consensus_stats = row_contains(stats, "method_id", "signal_consensus_fixed_oof")

    abstract = f"""Respiratory rate is a sensitive physiological indicator for dairy cow welfare and heat-stress monitoring, but non-contact estimation from infrared thermography remains vulnerable to nostril occlusion, weak respiratory peaks, boundary-cycle ambiguity, and motion-related signal degradation. We developed a reproducible thermal-video respiratory-rate pipeline that combines YOLO-based nostril localization, nostril temperature-curve construction, quality-aware residual peak-count correction, frequency-domain signal consensus, a conservative signal-aware residual gate, algorithmic review triage, conformal RR uncertainty intervals, RR-only physiological triage, and bilateral nostril consistency screening. On 73 thermal videos, the default pipeline achieved {method_metrics(default)}. The fixed-threshold quality-aware residual correction improved internal out-of-fold performance to {method_metrics(quality)}, and a frequency-domain signal-consensus supplement reached {method_metrics(signal)}. The conservative signal-aware safe gate achieved the strongest internal precision, reaching {method_metrics(safe_signal)} and improving exact count agreement by {int(safe_signal_stats['delta_exact_count'])} videos relative to the default pipeline (McNemar one-sided exact p={fmt(safe_signal_stats['mcnemar_exact_p_method_better'], 4)}). Internal reliability analysis added a non-truth quality triage layer ({quality_status}), conformal prediction intervals ({conformal_status}), RR-only physiological triage ({triage_status}), and a bilateral consistency auto-report screen ({bilateral_status}). Therefore, we report the quality-aware correction as the main internal innovation and the safety-gated signal-aware model as a high-precision internal candidate, while reserving true animal-identity, heat-stress, and external deployment claims for future metadata-stratified and frozen external validation."""

    signal_results = f"""### Conservative signal-aware safe gate candidate

The safety-gated signal-aware model provided the strongest internal out-of-fold estimate among the tested candidates. The gate accepts only an already predicted -1 residual correction when the autocorrelation count estimate is within two breaths of the signal-consensus count, so it uses non-truth prediction-time signal consistency rather than reference labels. Relative to the default pipeline, it increased RR R2 from {fmt(default['rr_r2'])} to {fmt(safe_signal['rr_r2'])}, reduced MAE from {fmt(default['rr_mae_bpm'])} to {fmt(safe_signal['rr_mae_bpm'])} breaths/min, and increased exact breath-count agreement from {exact(default)} to {exact(safe_signal)}. The paired method table estimated Delta RR R2={fmt(safe_signal_stats['delta_rr_r2'], 4)}, Delta MAE={fmt(safe_signal_stats['delta_rr_mae'], 3)} breaths/min, and Delta exact count={int(safe_signal_stats['delta_exact_count'])}; McNemar's one-sided exact test for exact-count improvement was p={fmt(safe_signal_stats['mcnemar_exact_p_method_better'], 4)}. The ungated signal-aware model reached {method_metrics(signal_aware)} but degraded under prefix-group stress testing ({method_metrics(signal_group)}). By contrast, the safe gate reached {method_metrics(safe_signal_group)} under the same prefix-group stress test and removed the large-count-error failures. This result should be framed as internal precision and robustness evidence rather than as the primary deployable method until real cow/session/external validation is complete."""

    discussion = f"""The current evidence supports a cautious innovation hierarchy. The quality-aware residual correction is the main internally validated algorithmic contribution because it improves the frozen baseline while retaining conservative nested and prefix-group checks. The signal-consensus and conservative signal-aware extensions show that frequency-domain agreement and richer non-truth signal features can recover additional one-breath residual errors. The safe gate is the strongest internal precision candidate, and its prefix-group stress result addresses the failure mode observed in the ungated signal-aware model. The algorithmic quality stratification, conformal uncertainty interval layer, RR-only physiological triage, and bilateral nostril consistency gate add deployment-oriented reliability and interpretation evidence: {quality_status}; {conformal_status}; {triage_status}; {bilateral_status}. This distinction is important for Q2-or-higher submission: the literature supports thermal and video-based RR monitoring, FFT-based periodicity analysis, end-to-end video learning, true animal-identity grouping, and heat-stress context, but the current evidence still has to be separated by claim strength. Metadata status is now: {metadata_status}. The readiness audit remains {readiness_status(readiness)}. Consequently, the manuscript should present the current work as an internally validated precision-improvement framework with reliability-aware RR reporting, RR-only review triage, and bilateral signal-consistency screening; report numeric-video-label grouped validation as internal metadata-group evidence; describe Lindian County, Heilongjiang, China and 2023-08-05 to 2023-08-10 as collection-provenance context; and explicitly reserve deployment-ready cross-session, camera-domain, manual quality-stratified, heat-stress, and external-validation claims for real metadata or new independent videos."""

    guardrails = [
        {
            "claim": "Main performance claim",
            "use": f"Quality-aware residual correction: {method_metrics(quality)}.",
            "avoid": "Do not use truth-calibrated R2 as main performance.",
        },
        {
            "claim": "Highest internal precision",
            "use": f"Conservative signal-aware safe gate: {method_metrics(safe_signal)} with McNemar p={fmt(safe_signal_stats['mcnemar_exact_p_method_better'], 4)} for exact-count improvement versus default.",
            "avoid": "Do not call signal-aware externally validated; the safe gate still needs real cow/session and external split validation.",
        },
        {
            "claim": "Prefix-group stress result",
            "use": f"Safe gate prefix-group stress: {method_metrics(safe_signal_group)}; ungated signal-aware prefix-group result was {method_metrics(signal_group)}.",
            "avoid": f"Do not claim this is cow-level validation; prefix grouping is still a proxy. McNemar p versus default in prefix-group output was {fmt(safe_signal_group_stats['mcnemar_exact_p_method_better'], 4)}.",
        },
        {
            "claim": "Truth-calibrated upper bound",
            "use": f"Supplementary oracle diagnostic only: {method_metrics(truth)}.",
            "avoid": "Do not put this in the abstract, highlights, or main deployable method row.",
        },
        {
            "claim": "Q2+ biological meaning",
            "use": "State that cow_id is filled from the video numeric label and that Lindian County, Heilongjiang, China with the 2023-08-05 to 2023-08-10 collection window provides provenance context.",
            "avoid": "Do not claim validated heat-stress monitoring, THI-stratified performance, camera-domain robustness, manual-quality robustness, or external deployment until real date/camera/environment/quality/external metadata are available.",
        },
        {
            "claim": "Algorithmic quality triage",
            "use": f"Use as an internal deployment supplement: {quality_status}.",
            "avoid": "Do not describe the risk score as manual head-motion, occlusion, or nostril-visibility annotation, and do not call its threshold externally validated.",
        },
        {
            "claim": "Conformal RR uncertainty",
            "use": f"Use as an internal reliability supplement: {conformal_status}.",
            "avoid": "Do not describe internal conformal interval coverage as prospective or external reliability until the interval rule is frozen and externally validated.",
        },
        {
            "claim": "RR-only physiological triage",
            "use": f"Use as an internal review/alert supplement: {triage_status}.",
            "avoid": "Do not describe high-RR candidate flags as heat-stress, disease, welfare, or treatment labels without synchronized environment and animal-context metadata.",
        },
        {
            "claim": "Bilateral nostril consistency screen",
            "use": f"Use as an internal selective-reporting reliability supplement: {bilateral_status}.",
            "avoid": "Do not describe bilateral consistency as external validation, heat-stress diagnosis, welfare classification, or true animal-identity evidence.",
        },
    ]
    guardrail_lines = [
        "| claim | safe use | avoid |",
        "| --- | --- | --- |",
        *[
            f"| {row['claim']} | {row['use']} | {row['avoid']} |"
            for row in guardrails
        ],
    ]

    return "\n".join(
        [
            "# Thermal RR Manuscript Claim Update",
            "",
            "This file is generated from current paper assets. Use it to update `docs/thermal_rr_manuscript_draft.md` without copying stale metrics.",
            "",
            "## Conservative Abstract Replacement",
            "",
            abstract,
            "",
            "## Results Insert",
            "",
            signal_results,
            "",
            "## Discussion Insert",
            "",
            discussion,
            "",
            "## Claim Guardrails",
            "",
            "\n".join(guardrail_lines),
            "",
            "## Literature Crosswalk Input",
            "",
            "Use `paper_literature_innovation_crosswalk.csv` for source-to-claim mapping and `paper_literature_source_register.csv` for DOI links.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_root = args.input_root.resolve()
    assets = input_root / f"{args.corrected_prefix}_paper_assets"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    tables = load_tables(input_root, args.corrected_prefix)
    text = build_update_text(tables)
    docs_path = docs_dir / "thermal_rr_manuscript_claim_update.md"
    assets_path = assets / "paper_manuscript_claim_update.md"
    docs_path.write_text(text, encoding="utf-8")
    assets_path.write_text(text, encoding="utf-8")
    print(f"Saved manuscript claim update: {docs_path}")
    print(f"Saved paper-assets manuscript claim update: {assets_path}")


if __name__ == "__main__":
    main()
