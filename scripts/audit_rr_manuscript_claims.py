from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_output_dir = (
        default_input_root / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description="Audit manuscript claim wording against the current thermal RR evidence package."
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=repo_root / "docs" / "thermal_rr_manuscript_draft.md",
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippet_for(text: str, pattern: str, *, width: int = 120) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    start = match.start()
    end = min(len(text), match.end() + width)
    return normalize(text[start:end])


def has_pattern(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def row(
    check: str,
    status: str,
    risk: str,
    evidence: str,
    action: str,
    *,
    manuscript_section: str = "",
    claim_boundary: str = "",
) -> dict[str, object]:
    return {
        "check": check,
        "status": status,
        "risk": risk,
        "manuscript_section": manuscript_section,
        "evidence": evidence,
        "recommended_action": action,
        "claim_boundary": claim_boundary,
    }


def metric_from_table(
    table: pd.DataFrame | None,
    method_contains: str,
    metric: str,
    *,
    digits: int = 3,
) -> str:
    if table is None or table.empty or "method" not in table.columns:
        return ""
    mask = table["method"].astype(str).str.contains(
        method_contains, case=False, regex=False
    )
    if not mask.any() or metric not in table.columns:
        return ""
    value = pd.to_numeric(table.loc[mask, metric], errors="coerce").dropna()
    if value.empty:
        return ""
    return f"{float(value.iloc[0]):.{digits}f}"


def check_metric_alignment(
    text: str, main_results: pd.DataFrame | None
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    expected = [
        ("default pipeline RR R2", "Default thermal RR pipeline", "rr_r2", PASS),
        (
            "fixed residual-corrector RR R2",
            "fixed threshold, out-of-fold",
            "rr_r2",
            PASS,
        ),
        (
            "signal-consensus RR R2",
            "Signal-consensus supplement (fixed threshold",
            "rr_r2",
            PASS,
        ),
    ]
    for check_name, method_key, metric, status_if_found in expected:
        rounded = metric_from_table(main_results, method_key, metric)
        if not rounded:
            checks.append(
                row(
                    check_name,
                    WARN,
                    "asset_missing",
                    "Could not read the metric from paper_main_results_table.csv.",
                    "Regenerate paper assets before final manuscript audit.",
                    claim_boundary="Do not manually type final metric values without matching current generated outputs.",
                )
            )
            continue
        if rounded in text:
            checks.append(
                row(
                    check_name,
                    status_if_found,
                    "metric_alignment",
                    f"Manuscript contains current rounded value {rounded}.",
                    "Keep this value synchronized with paper_main_results_table.csv after reruns.",
                    claim_boundary="Metric wording is aligned with current generated assets.",
                )
            )
        else:
            checks.append(
                row(
                    check_name,
                    FAIL,
                    "metric_drift",
                    f"Current rounded value {rounded} was not found in the manuscript text.",
                    "Update the manuscript result paragraph after rebuilding paper assets.",
                    claim_boundary="A manuscript metric must match the generated result table.",
                )
            )
    return checks


def check_truth_calibrated_boundary(text: str) -> dict[str, object]:
    if "truth-calibrated" not in text.lower():
        return row(
            "truth-calibrated boundary",
            WARN,
            "missing_boundary",
            "The manuscript does not mention truth-calibrated outputs.",
            "If the upper-bound table is included, explicitly label it as truth-assisted and not main performance.",
            claim_boundary="Truth-assisted outputs can be upper-bound analysis only.",
        )
    has_upper = has_pattern(text, r"truth-calibrated.{0,160}upper[- ]bound|upper[- ]bound.{0,160}truth-calibrated")
    has_not_main = has_pattern(
        text,
        r"truth-calibrated.{0,220}(not treated as main|not.*main method performance|cannot be reported as main|upper-bound analysis only)|not.*main method performance.{0,220}truth-calibrated",
    )
    status = PASS if has_upper and has_not_main else FAIL
    return row(
        "truth-calibrated boundary",
        status,
        "truth_leakage",
        snippet_for(text, r"truth-calibrated.{0,260}"),
        "Keep truth-calibrated R2 out of the abstract's main method claim and table it only as upper-bound/oracle analysis.",
        manuscript_section="Methods, Limitations, Figure/Table plan",
        claim_boundary="Truth-calibrated outputs use reference information and cannot represent deployable performance.",
    )


def check_external_boundary(text: str, acceptance: pd.DataFrame | None) -> dict[str, object]:
    asset_fail = False
    if acceptance is not None and not acceptance.empty and "status" in acceptance.columns:
        asset_fail = acceptance["status"].astype(str).str.upper().eq(FAIL).any()
    has_missing = has_pattern(
        text,
        r"external validation data are still missing|no frozen external split|external split readiness audit currently fails|external_test_split is empty|not.*independent external validation",
    )
    has_no_claim = has_pattern(
        text,
        r"no external metrics|no external-performance claim|before claiming broad generalization|larger external",
    )
    has_provisional = has_pattern(
        text,
        r"provisional single-annotator external development|exploratory development evidence|"
        r"development diagnostic, not confirmatory external validation",
    )
    has_not_confirmatory = has_pattern(
        text,
        r"not a confirmatory external claim|not independent external validation|"
        r"cannot substitute for prospective dual-blinded external validation|"
        r"dual-blinded annotation.*required",
    )
    planning_boundary = has_missing and has_no_claim
    provisional_boundary = asset_fail and has_provisional and has_not_confirmatory
    status = PASS if planning_boundary or provisional_boundary else FAIL
    evidence = snippet_for(text, r"external.{0,260}(missing|fails|not|claim|split|validation)")
    if asset_fail:
        evidence = f"Acceptance gate file contains FAIL; manuscript evidence: {evidence}"
    action = (
        "Keep provisional single-annotator metrics labeled as development evidence and "
        "retain dual-blinded adjudication as the confirmatory gate."
        if provisional_boundary
        else "Do not write external performance claims until external split readiness and acceptance gates are PASS."
    )
    return row(
        "external validation boundary",
        status,
        "external_overclaim",
        evidence,
        action,
        manuscript_section="Status, External validation planning, Results",
        claim_boundary=(
            "Provisional single-annotator development analysis is allowed when explicitly "
            "non-confirmatory; independent external validation still requires blinded consensus."
            if provisional_boundary
            else "Current evidence supports internal validation and planning, not external validation performance."
        ),
    )


def check_pseudo_external_boundary(text: str) -> dict[str, object]:
    if "pseudo-external" not in text.lower():
        return row(
            "pseudo-external boundary",
            WARN,
            "missing_boundary",
            "The manuscript does not mention the pseudo-external stress test.",
            "If this analysis is used, define it as an internal prefix-domain stress test.",
            claim_boundary="Pseudo-external prefix groups cannot substitute for independent external testing.",
        )
    has_internal = has_pattern(
        text,
        r"pseudo-external.{0,260}(internal domain-risk analysis|internal prefix-domain|not as independent external validation|not as definitive generalization)|internal.{0,260}pseudo-external",
    )
    return row(
        "pseudo-external boundary",
        PASS if has_internal else FAIL,
        "pseudo_external_overclaim",
        snippet_for(text, r"pseudo-external.{0,320}"),
        "Keep this wording as internal domain-risk/stress-test evidence only.",
        manuscript_section="Prefix-group holdout validation",
        claim_boundary="Pseudo-external evidence is useful for risk localization but is not external validation.",
    )


def check_selective_reporting_boundary(text: str) -> dict[str, object]:
    has_coverage = has_pattern(text, r"selective.{0,260}coverage|coverage.{0,260}selective")
    has_review = has_pattern(text, r"manual[- ]review|manual review")
    has_not_prospective = has_pattern(
        text,
        r"not.*prospective reliability|prospective reliability.{0,180}thresholds are frozen|evaluated externally",
    )
    status = PASS if has_coverage and has_review and has_not_prospective else FAIL
    return row(
        "selective reporting boundary",
        status,
        "selective_subset_overclaim",
        snippet_for(text, r"selective.{0,360}"),
        "Always report selective accuracy with coverage and manual-review routing; do not present the subset as total-dataset performance.",
        manuscript_section="Selective RR reporting",
        claim_boundary="Selective reporting is a triage/uncertainty analysis, not a replacement for external validation.",
    )


def check_metadata_heat_boundary(text: str) -> dict[str, object]:
    has_legacy_empty_metadata = has_pattern(text, r"(^|[^0-9])0/73([^0-9]|$)")
    has_context_proxy = has_pattern(
        text,
        r"regional NASA POWER|regional proxy|not synchronized barn sensor|constant across the 73 videos",
    )
    has_no_assoc = has_pattern(
        text,
        r"cannot yet report associations|no heat-stress association is reported|fields are empty|cannot report THI-stratified|cannot support THI-stratified|not enough to support heat-stress inference",
    )
    status = PASS if (has_legacy_empty_metadata or has_context_proxy) and has_no_assoc else FAIL
    return row(
        "metadata and heat-stress boundary",
        status,
        "biological_overclaim",
        snippet_for(
            text,
            r"regional NASA POWER.{0,300}|cannot.{0,260}THI-stratified|not enough.{0,260}heat-stress|(^|[^0-9])0/73([^0-9]|$).{0,260}",
        ),
        "Keep heat-stress and animal-level conclusions conditional until metadata are filled and rerun.",
        manuscript_section="Heat-stress interpretation readiness",
        claim_boundary="Current data support an interface for biological interpretation, not THI/ATHI association claims.",
    )


def check_q2_readiness_boundary(text: str) -> dict[str, object]:
    has_not_ready = has_pattern(text, r"not yet ready for journal submission|before journal submission")
    risky = []
    for pattern in [
        r"submission[- ]ready",
        r"ready for journal submission",
        r"ready for q2",
        r"q2[- ]or[- ]higher submission[- ]ready",
    ]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context = normalize(text[max(0, match.start() - 80) : match.end() + 80])
            if not re.search(r"not|before|if|until", context, flags=re.IGNORECASE):
                risky.append(context)
    status = PASS if has_not_ready and not risky else FAIL
    evidence = snippet_for(text, r"not yet ready for journal submission|before journal submission")
    if risky:
        evidence = "; ".join(risky[:3])
    return row(
        "Q2 submission readiness boundary",
        status,
        "q2_overclaim",
        evidence,
        "Keep the manuscript as internal/supervisor-review ready until external and metadata gates pass.",
        manuscript_section="Manuscript status, Conclusions",
        claim_boundary="Current package is not Q2+ submission-ready without external and metadata evidence.",
    )


def check_statistical_uncertainty_boundary(text: str) -> dict[str, object]:
    has_bootstrap = has_pattern(
        text,
        r"bootstrap confidence interval|bootstrap.*confidence|bootstrap CI",
    )
    has_cross_zero = has_pattern(
        text,
        r"crossed zero|cross zero|touches zero|touched zero|includes zero|from -0\.0059 to \+0\.0411",
    )
    status = PASS if has_bootstrap and has_cross_zero else FAIL
    return row(
        "statistical uncertainty boundary",
        status,
        "effect_size_overclaim",
        snippet_for(text, r"bootstrap.{0,300}"),
        "Report the precision gain as promising/internal because the R2 delta CI crosses zero.",
        manuscript_section="Results, Discussion",
        claim_boundary="The current internal improvement is promising but not a statistically locked external gain.",
    )


def check_data_availability_boundary(text: str) -> dict[str, object]:
    has_caveat = has_pattern(
        text, r"should not claim public data or code availability|until the dataset-sharing decision is finalized"
    )
    return row(
        "data/code availability boundary",
        PASS if has_caveat else FAIL,
        "availability_overclaim",
        snippet_for(text, r"Data and Code Availability.{0,300}|public data.{0,220}"),
        "Do not claim public release until the sharing decision and anonymization state are finalized.",
        manuscript_section="Data and Code Availability",
        claim_boundary="Local reproducibility is not the same as public availability.",
    )


def check_overclaim_language(text: str) -> dict[str, object]:
    forbidden_patterns = [
        r"solved dairy-cow respiratory-rate monitoring",
        r"robust under all head-motion conditions",
        r"broad generalization has been demonstrated",
        r"external validation passed",
        r"independent external test set achieved",
        r"ready for q2 submission",
        r"deployment-ready",
    ]
    hits = []
    for pattern in forbidden_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context = normalize(text[max(0, match.start() - 80) : match.end() + 80])
            if re.search(r"not|rather than|without|until|before|avoid|do not", context, flags=re.IGNORECASE):
                continue
            hits.append(snippet_for(text, pattern, width=80))
            break
    return row(
        "overclaim language scan",
        PASS if not hits else FAIL,
        "unsupported_language",
        "No forbidden overclaim phrases found." if not hits else " | ".join(hits[:5]),
        "Rewrite any broad deployment, Q2-ready, or external-validation claims as conditional/internal evidence.",
        claim_boundary="Avoid broad deployment/generalization claims until external and metadata gates pass.",
    )


def build_audit(args: argparse.Namespace) -> pd.DataFrame:
    if not args.manuscript.exists():
        return pd.DataFrame(
            [
                row(
                    "manuscript file present",
                    FAIL,
                    "missing_file",
                    f"{args.manuscript} does not exist.",
                    "Create or point --manuscript to the current manuscript draft.",
                )
            ]
        )
    text = args.manuscript.read_text(encoding="utf-8")
    normalized = normalize(text)
    main_results = read_csv_if_exists(args.output_dir / "paper_main_results_table.csv")
    acceptance = read_csv_if_exists(
        args.output_dir / "paper_external_split_acceptance_table.csv"
    )

    rows = [
        row(
            "manuscript file present",
            PASS if len(normalized) > 1000 else WARN,
            "source_file",
            f"{args.manuscript} characters={len(normalized)}",
            "Use this file as the audited manuscript source.",
        )
    ]
    rows.extend(check_metric_alignment(normalized, main_results))
    rows.extend(
        [
            check_truth_calibrated_boundary(normalized),
            check_external_boundary(normalized, acceptance),
            check_pseudo_external_boundary(normalized),
            check_selective_reporting_boundary(normalized),
            check_metadata_heat_boundary(normalized),
            check_q2_readiness_boundary(normalized),
            check_statistical_uncertainty_boundary(normalized),
            check_data_availability_boundary(normalized),
            check_overclaim_language(normalized),
        ]
    )
    audit = pd.DataFrame(rows)
    audit.insert(0, "audited_file", str(args.manuscript))
    return audit


def write_markdown(audit: pd.DataFrame, output_path: Path) -> None:
    counts = audit["status"].value_counts().to_dict()
    if (audit["status"] == FAIL).any():
        status = "fail"
    elif (audit["status"] == WARN).any():
        status = "caution_ready"
    else:
        status = "pass"
    lines = [
        "# Manuscript Claim Audit",
        "",
        f"Overall status: `{status}`",
        "",
        f"PASS: {counts.get(PASS, 0)}; WARN: {counts.get(WARN, 0)}; FAIL: {counts.get(FAIL, 0)}",
        "",
        "| check | status | risk | evidence | recommended_action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, item in audit.iterrows():
        evidence = str(item["evidence"]).replace("|", "/")
        action = str(item["recommended_action"]).replace("|", "/")
        lines.append(
            f"| {item['check']} | {item['status']} | {item['risk']} | {evidence} | {action} |"
        )
    lines.extend(
        [
            "",
            "Use this audit as a wording guard only. A passing claim audit does not replace cow-level metadata, heat-stress metadata, or frozen external validation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    csv_path = args.output_dir / "paper_manuscript_claim_audit.csv"
    md_path = args.output_dir / "paper_manuscript_claim_audit.md"
    audit.to_csv(csv_path, index=False)
    write_markdown(audit, md_path)
    print(f"Saved manuscript claim audit: {csv_path}")
    print(f"Saved manuscript claim audit report: {md_path}")
    print(audit[["check", "status", "risk"]].to_string(index=False))


if __name__ == "__main__":
    main()
