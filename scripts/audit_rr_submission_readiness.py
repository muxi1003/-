from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Audit whether the thermal RR paper package is ready for Q2+ submission."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--min-fixed-r2", type=float, default=0.94)
    parser.add_argument("--min-nested-r2", type=float, default=0.935)
    parser.add_argument("--min-group-r2", type=float, default=0.93)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def row(
    category: str,
    check: str,
    status: str,
    evidence: str,
    action: str,
    *,
    q2_blocking: bool,
    internal_blocking: bool = False,
) -> dict[str, object]:
    return {
        "category": category,
        "check": check,
        "status": status,
        "q2_blocking": bool(q2_blocking),
        "internal_manuscript_blocking": bool(internal_blocking),
        "evidence": evidence,
        "recommended_action": action,
    }


def file_check(path: Path, label: str, *, min_bytes: int = 1, q2_blocking: bool = False) -> dict[str, object]:
    if path.exists() and path.stat().st_size >= min_bytes:
        return row(
            "assets",
            label,
            PASS,
            f"{path} exists, bytes={path.stat().st_size}",
            "No action needed.",
            q2_blocking=False,
        )
    return row(
        "assets",
        label,
        FAIL,
        f"{path} missing or smaller than {min_bytes} bytes",
        f"Regenerate {path.name} before manuscript assembly.",
        q2_blocking=q2_blocking,
        internal_blocking=True,
    )


def text_contains_check(
    path: Path,
    label: str,
    required_patterns: list[str],
    *,
    min_bytes: int = 1,
    q2_blocking: bool = False,
) -> dict[str, object]:
    if not path.exists() or path.stat().st_size < min_bytes:
        return row(
            "assets",
            label,
            FAIL,
            f"{path} missing or smaller than {min_bytes} bytes",
            f"Regenerate {path.name} before manuscript assembly.",
            q2_blocking=q2_blocking,
            internal_blocking=True,
        )
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    if missing:
        return row(
            "assets",
            label,
            FAIL,
            f"{path} exists but is missing required text markers: {missing}",
            f"Regenerate {path.name} and inspect claim wording.",
            q2_blocking=q2_blocking,
            internal_blocking=True,
        )
    return row(
        "assets",
        label,
        PASS,
        f"{path} exists, bytes={path.stat().st_size}, required text markers present",
        "No action needed.",
        q2_blocking=False,
    )


def exact_count_value(value: object) -> int:
    if isinstance(value, str) and "/" in value:
        return int(value.split("/", 1)[0])
    return int(float(value))


def find_row(table: pd.DataFrame, contains: str) -> pd.Series | None:
    mask = table["method"].astype(str).str.contains(contains, case=False, regex=False)
    if not mask.any():
        return None
    return table[mask].iloc[0]


def metric_checks(main_results: pd.DataFrame | None, args: argparse.Namespace) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if main_results is None or main_results.empty:
        return [
            row(
                "performance",
                "main results table",
                FAIL,
                "paper_main_results_table.csv missing or empty",
                "Run scripts/build_rr_paper_assets.py after refreshing metrics.",
                q2_blocking=True,
                internal_blocking=True,
            )
        ]

    default = find_row(main_results, "Default thermal RR pipeline")
    fixed = find_row(main_results, "fixed threshold")
    nested = find_row(main_results, "nested threshold CV")
    group_selected = find_row(main_results, "threshold selected on training groups")
    best_grid = find_row(main_results, "best threshold grid")
    truth_upper = find_row(main_results, "Truth-calibrated upper bound")

    if default is None or fixed is None:
        checks.append(
            row(
                "performance",
                "default and fixed-threshold rows present",
                FAIL,
                "Required default/fixed rows not found in main results table.",
                "Regenerate paper assets and inspect method labels.",
                q2_blocking=True,
                internal_blocking=True,
            )
        )
        return checks

    default_r2 = float(default["rr_r2"])
    fixed_r2 = float(fixed["rr_r2"])
    default_exact = exact_count_value(default["exact_count"])
    fixed_exact = exact_count_value(fixed["exact_count"])
    checks.append(
        row(
            "performance",
            "fixed-threshold innovation improves default baseline",
            PASS if fixed_r2 > default_r2 and fixed_exact > default_exact else FAIL,
            f"default R2={default_r2:.6f}, fixed R2={fixed_r2:.6f}; default exact={default_exact}, fixed exact={fixed_exact}",
            "Keep fixed-threshold out-of-fold result as the main internal innovation result.",
            q2_blocking=not (fixed_r2 > default_r2 and fixed_exact > default_exact),
            internal_blocking=not (fixed_r2 > default_r2 and fixed_exact > default_exact),
        )
    )
    checks.append(
        row(
            "performance",
            f"fixed-threshold R2 >= {args.min_fixed_r2:.3f}",
            PASS if fixed_r2 >= float(args.min_fixed_r2) else WARN,
            f"fixed R2={fixed_r2:.6f}",
            "Use as a target-gate for the internal algorithm claim.",
            q2_blocking=False,
        )
    )

    if nested is None:
        checks.append(
            row(
                "validation",
                "nested threshold CV present",
                FAIL,
                "Nested threshold CV row not found.",
                "Run scripts/rr_quality_residual_validation.py and rebuild paper assets.",
                q2_blocking=True,
                internal_blocking=True,
            )
        )
    else:
        nested_r2 = float(nested["rr_r2"])
        checks.append(
            row(
                "validation",
                f"nested threshold CV R2 >= {args.min_nested_r2:.3f}",
                PASS if nested_r2 >= float(args.min_nested_r2) else WARN,
                f"nested R2={nested_r2:.6f}, default R2={default_r2:.6f}",
                "Report nested CV as the conservative internal performance estimate.",
                q2_blocking=False,
            )
        )

    if group_selected is None:
        checks.append(
            row(
                "validation",
                "prefix-group holdout row present",
                FAIL,
                "Leave-one-prefix-group-out selected-threshold row not found.",
                "Run scripts/rr_quality_residual_group_validation.py and rebuild paper assets.",
                q2_blocking=True,
                internal_blocking=True,
            )
        )
    else:
        group_r2 = float(group_selected["rr_r2"])
        checks.append(
            row(
                "validation",
                f"prefix-group holdout R2 >= {args.min_group_r2:.3f}",
                PASS if group_r2 >= float(args.min_group_r2) else WARN,
                f"prefix-group selected-threshold R2={group_r2:.6f}",
                "Treat as internal grouped validation only; replace with real metadata GroupKFold.",
                q2_blocking=False,
            )
        )

    if best_grid is not None:
        paper_use = str(best_grid.get("paper_use", ""))
        checks.append(
            row(
                "leakage_guard",
                "best grid threshold is sensitivity-only",
                PASS if paper_use == "sensitivity_only" else FAIL,
                f"paper_use={paper_use}",
                "Do not report grid-selected result as the primary method estimate.",
                q2_blocking=paper_use != "sensitivity_only",
                internal_blocking=paper_use != "sensitivity_only",
            )
        )
    if truth_upper is not None:
        paper_use = str(truth_upper.get("paper_use", ""))
        checks.append(
            row(
                "leakage_guard",
                "truth-calibrated result is upper-bound only",
                PASS if paper_use == "upper_bound_only" else FAIL,
                f"paper_use={paper_use}",
                "Keep truth-assisted calibration out of the main performance claim.",
                q2_blocking=paper_use != "upper_bound_only",
                internal_blocking=paper_use != "upper_bound_only",
            )
        )
    return checks


def metric_reporting_tier_checks(metric_tiers: pd.DataFrame | None) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if metric_tiers is None or metric_tiers.empty:
        return [
            row(
                "leakage_guard",
                "metric reporting tiers table present",
                FAIL,
                "paper_metric_reporting_tiers.csv missing or empty",
                "Run scripts/build_rr_paper_assets.py to regenerate the metric reporting tiers.",
                q2_blocking=True,
                internal_blocking=True,
            )
        ]

    required_columns = {
        "method",
        "evidence_tier",
        "primary_performance_allowed",
        "recommended_location",
        "claim_boundary",
    }
    missing = sorted(required_columns - set(metric_tiers.columns))
    if missing:
        return [
            row(
                "leakage_guard",
                "metric reporting tiers schema",
                FAIL,
                f"missing columns={missing}",
                "Regenerate paper assets with the current scripts/build_rr_paper_assets.py.",
                q2_blocking=True,
                internal_blocking=True,
            )
        ]

    truth_upper = find_row(metric_tiers, "Truth-calibrated upper bound")
    if truth_upper is None:
        checks.append(
            row(
                "leakage_guard",
                "truth-calibrated tier present",
                FAIL,
                "Truth-calibrated upper-bound row not found in metric reporting tiers.",
                "Regenerate paper assets and inspect method labels.",
                q2_blocking=True,
                internal_blocking=True,
            )
        )
    else:
        primary_allowed = str(truth_upper["primary_performance_allowed"]).strip().lower()
        evidence_tier = str(truth_upper["evidence_tier"]).strip()
        safe = primary_allowed == "no" and evidence_tier == "oracle_upper_bound"
        checks.append(
            row(
                "leakage_guard",
                "truth-calibrated excluded from primary performance claim",
                PASS if safe else FAIL,
                (
                    f"primary_performance_allowed={primary_allowed}; "
                    f"evidence_tier={evidence_tier}"
                ),
                "Keep truth-calibrated R2 as a supplement/oracle upper bound only.",
                q2_blocking=not safe,
                internal_blocking=not safe,
            )
        )

    default = find_row(metric_tiers, "Default thermal RR pipeline")
    fixed = find_row(metric_tiers, "fixed threshold")
    baseline_allowed = (
        default is not None
        and str(default["primary_performance_allowed"]).strip().lower() == "yes"
    )
    innovation_allowed = (
        fixed is not None
        and str(fixed["primary_performance_allowed"]).strip().lower() == "yes"
        and str(fixed["evidence_tier"]).strip() == "primary_internal_innovation"
    )
    checks.append(
        row(
            "leakage_guard",
            "main internal metric tiers are manuscript-usable",
            PASS if baseline_allowed and innovation_allowed else FAIL,
            (
                f"default_primary_allowed={baseline_allowed}; "
                f"fixed_primary_innovation_allowed={innovation_allowed}"
            ),
            "Use default as baseline and fixed-threshold OOF correction as the internal innovation result.",
            q2_blocking=not (baseline_allowed and innovation_allowed),
            internal_blocking=not (baseline_allowed and innovation_allowed),
        )
    )

    unsafe_primary = metric_tiers[
        metric_tiers["evidence_tier"].isin(["oracle_upper_bound", "sensitivity_analysis"])
        & (metric_tiers["primary_performance_allowed"].astype(str).str.lower() != "no")
    ]
    checks.append(
        row(
            "leakage_guard",
            "oracle and sensitivity metrics are not primary claims",
            PASS if unsafe_primary.empty else FAIL,
            f"unsafe_rows={len(unsafe_primary)}",
            "Set oracle/sensitivity rows to primary_performance_allowed=no before manuscript use.",
            q2_blocking=not unsafe_primary.empty,
            internal_blocking=not unsafe_primary.empty,
        )
    )
    return checks


def safe_policy_checks(safe_policy_metrics: pd.DataFrame | None) -> list[dict[str, object]]:
    if safe_policy_metrics is None or safe_policy_metrics.empty:
        return [
            row(
                "validation",
                "signal-aware safe policy metrics present",
                FAIL,
                "paper_signal_aware_safe_policy_metrics_table.csv missing or empty",
                "Run scripts/rr_signal_aware_safe_policy.py and rebuild paper assets.",
                q2_blocking=False,
                internal_blocking=True,
            )
        ]

    def label_row(label: str) -> pd.Series | None:
        mask = safe_policy_metrics["label"].astype(str) == label
        if not mask.any():
            return None
        return safe_policy_metrics[mask].iloc[0]

    fixed_original = label_row("signal_aware_residual_fixed_oof")
    fixed_safe = label_row("signal_aware_safe_policy_fixed_oof")
    group_original = label_row("signal_aware_residual_prefix_group_fixed_threshold")
    group_safe = label_row("signal_aware_safe_policy_prefix_group_fixed_threshold")
    required = {
        "original fixed": fixed_original,
        "safe fixed": fixed_safe,
        "original prefix-group": group_original,
        "safe prefix-group": group_safe,
    }
    missing = [name for name, item in required.items() if item is None]
    if missing:
        return [
            row(
                "validation",
                "signal-aware safe policy expected rows present",
                FAIL,
                f"missing rows={missing}",
                "Regenerate safe policy metrics and inspect label names.",
                q2_blocking=False,
                internal_blocking=True,
            )
        ]

    assert fixed_original is not None
    assert fixed_safe is not None
    assert group_original is not None
    assert group_safe is not None
    fixed_gain = float(fixed_safe["rr_r2"]) - float(fixed_original["rr_r2"])
    group_gain = float(group_safe["rr_r2"]) - float(group_original["rr_r2"])
    group_ge2 = int(float(group_safe["abs_count_error_ge2"]))
    original_group_ge2 = int(float(group_original["abs_count_error_ge2"]))
    return [
        row(
            "validation",
            "safe policy improves fixed OOF signal-aware R2",
            PASS if fixed_gain > 0 else WARN,
            (
                f"safe fixed R2={float(fixed_safe['rr_r2']):.6f}; "
                f"original fixed R2={float(fixed_original['rr_r2']):.6f}; "
                f"delta={fixed_gain:.6f}"
            ),
            "Treat as an internal precision candidate pending real grouped/external validation.",
            q2_blocking=False,
        ),
        row(
            "validation",
            "safe policy removes prefix-group large count errors",
            PASS if group_ge2 == 0 and group_ge2 < original_group_ge2 else WARN,
            (
                f"safe prefix-group abs_count_error_ge2={group_ge2}; "
                f"original prefix-group abs_count_error_ge2={original_group_ge2}; "
                f"safe prefix-group R2={float(group_safe['rr_r2']):.6f}; "
                f"original prefix-group R2={float(group_original['rr_r2']):.6f}; "
                f"delta R2={group_gain:.6f}"
            ),
            "Use as proxy-group stress evidence only; still require real cow/session grouping.",
            q2_blocking=False,
        ),
    ]


def bootstrap_checks(bootstrap: pd.DataFrame | None) -> list[dict[str, object]]:
    if bootstrap is None or bootstrap.empty:
        return [
            row(
                "statistics",
                "bootstrap CI table present",
                FAIL,
                "paper_bootstrap_ci_table.csv missing or empty",
                "Run scripts/rr_quality_residual_validation.py and rebuild paper assets.",
                q2_blocking=True,
                internal_blocking=True,
            )
        ]
    checks = []
    by_metric = {str(item["display_metric"]): item for _, item in bootstrap.iterrows()}
    delta_r2 = by_metric.get("Delta RR R2")
    if delta_r2 is not None:
        ci_low = float(delta_r2["ci_low_2_5"])
        checks.append(
            row(
                "statistics",
                "Delta RR R2 bootstrap CI excludes zero",
                PASS if ci_low > 0 else WARN,
                f"Delta RR R2 CI low={ci_low:.6f}, high={float(delta_r2['ci_high_97_5']):.6f}",
                "Current CI crosses zero; strengthen with metadata/external validation or larger sample.",
                q2_blocking=ci_low <= 0,
            )
        )
    delta_exact = by_metric.get("Delta exact count")
    if delta_exact is not None:
        ci_low = float(delta_exact["ci_low_2_5"])
        checks.append(
            row(
                "statistics",
                "Delta exact-count bootstrap CI is nonnegative",
                PASS if ci_low >= 0 else WARN,
                f"Delta exact count CI low={ci_low:.3f}, high={float(delta_exact['ci_high_97_5']):.3f}",
                "Report exact-count improvement with CI.",
                q2_blocking=False,
            )
        )
    return checks


def metadata_checks(metadata: pd.DataFrame | None, heat: pd.DataFrame | None) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if metadata is None or metadata.empty:
        checks.append(
            row(
                "metadata",
                "metadata grouped validation readiness present",
                FAIL,
                "metadata readiness file missing or empty",
                "Run scripts/rr_quality_residual_metadata_group_validation.py after filling metadata.",
                q2_blocking=True,
                internal_blocking=False,
            )
        )
    else:
        cow = metadata[metadata["group_column"].astype(str) == "cow_id"]
        ready = bool(cow.iloc[0]["ready_for_group_validation"]) if not cow.empty else False
        evidence = cow.iloc[0].to_dict() if not cow.empty else {}
        checks.append(
            row(
                "metadata",
                "cow_id numeric-video-label grouping readiness",
                PASS if ready else FAIL,
                str(evidence),
                (
                    "Treat cow_id as the numeric video label for the current dataset. "
                    "Do not use animal-level wording unless a separate real-cow identity "
                    "map is available; rerun metadata group validation after any change."
                ),
                q2_blocking=not ready,
            )
        )

    if heat is None or heat.empty:
        checks.append(
            row(
                "metadata",
                "heat-stress readiness present",
                FAIL,
                "heat-stress readiness file missing or empty",
                "Run scripts/build_rr_heat_stress_context.py after filling environmental metadata.",
                q2_blocking=False,
            )
        )
        return checks

    heat_by_field = {str(item["field"]): item for _, item in heat.iterrows()}
    for field, label, action, q2_required in [
        (
            "external_test_split",
            "external split labels complete",
            "Keep current rows internal unless a method-frozen independent external set exists.",
            True,
        ),
        (
            "ambient_temperature_c",
            "ambient temperature context complete",
            "Use synchronized barn temperature for THI-stratified claims; regional proxies are context only.",
            False,
        ),
        (
            "relative_humidity_percent",
            "relative humidity context complete",
            "Use synchronized barn humidity for THI-stratified claims; regional proxies are context only.",
            False,
        ),
        (
            "thi_analysis",
            "THI analysis readiness",
            "Compute or provide THI after temperature and humidity are available.",
            False,
        ),
        (
            "head_motion_score_0_3",
            "head-motion robustness metadata complete",
            "Use manual visual scores only; if unavailable, keep this as a limitation and avoid motion-stratified claims.",
            False,
        ),
        (
            "occlusion_score_0_3",
            "occlusion robustness metadata complete",
            "Use manual visual scores only; if unavailable, keep this as a limitation and avoid occlusion-stratified claims.",
            False,
        ),
        (
            "nostril_visibility_score_0_3",
            "nostril visibility metadata complete",
            "Use manual visual scores only; if unavailable, keep this as a limitation and avoid ROI-quality stratified claims.",
            False,
        ),
    ]:
        item = heat_by_field.get(field)
        ready = bool(item["ready"]) if item is not None else False
        evidence = (
            f"coverage={float(item['coverage_percent']):.1f}%, nonmissing={int(item['nonmissing'])}/{int(item['total_videos'])}"
            if item is not None
            else "field missing from readiness table"
        )
        checks.append(
            row(
                "metadata",
                label,
                PASS if ready else FAIL,
                evidence,
                action,
                q2_blocking=bool(q2_required and not ready),
            )
        )
    return checks


def external_split_checks(external_split: pd.DataFrame | None) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if external_split is None or external_split.empty:
        checks.append(
            row(
                "external_validation",
                "frozen external split validation ready",
                FAIL,
                "external split readiness file missing or empty",
                "Run scripts/rr_external_split_validation.py after independent external rows are present.",
                q2_blocking=True,
                internal_blocking=False,
            )
        )
        return checks
    ready_row = external_split[
        external_split["check"].astype(str) == "external split validation ready"
    ]
    if ready_row.empty:
        checks.append(
            row(
                "external_validation",
                "frozen external split validation ready",
                FAIL,
                "external split readiness row missing",
                "Regenerate external split readiness with scripts/rr_external_split_validation.py.",
                q2_blocking=True,
                internal_blocking=False,
            )
        )
        return checks
    item = ready_row.iloc[0]
    ready = str(item.get("status", "")).upper() == PASS and str(
        item.get("external_ready", "")
    ).lower() in {"true", "1"}
    checks.append(
        row(
            "external_validation",
            "frozen external split validation ready",
            PASS if ready else FAIL,
            str(item.to_dict()),
            (
                "Keep the current development rows internal; add/import independent "
                "method-frozen external rows, label those rows external, then rerun "
                "scripts/rr_external_split_validation.py."
            ),
            q2_blocking=not ready,
            internal_blocking=False,
        )
    )
    return checks


def metadata_quality_checks(
    metadata_quality: pd.DataFrame | None,
    split_leakage: pd.DataFrame | None,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if metadata_quality is None or metadata_quality.empty:
        checks.append(
            row(
                "metadata_quality",
                "metadata quality claim gates generated",
                FAIL,
                "paper_metadata_quality_audit.csv missing or empty",
                "Run scripts/audit_rr_metadata_quality.py after creating or filling paper_metadata_template.csv.",
                q2_blocking=True,
                internal_blocking=False,
            )
        )
    else:
        q2_mask = metadata_quality["q2_blocking"].astype(bool)
        q2_fail = metadata_quality[q2_mask & metadata_quality["status"].isin([FAIL, WARN])]
        nonblocking_warn = metadata_quality[
            (~q2_mask) & metadata_quality["status"].eq(WARN)
        ]
        if q2_fail.empty:
            checks.append(
                row(
                    "metadata_quality",
                    "metadata quality Q2 claim gates",
                    PASS,
                    f"q2_blocking_failures=0; checks={len(metadata_quality)}",
                    "No action needed before downstream metadata-specific validation.",
                    q2_blocking=False,
                )
            )
        else:
            evidence = "; ".join(
                f"{item['check']} ({item['status']}): {item['evidence']}"
                for _, item in q2_fail.head(8).iterrows()
            )
            checks.append(
                row(
                    "metadata_quality",
                    "metadata quality Q2 claim gates",
                    FAIL,
                    f"q2_blocking_failures={len(q2_fail)}; {evidence}",
                    "Fill the required metadata fields, rerun scripts/audit_rr_metadata_quality.py, then rerun this readiness audit.",
                    q2_blocking=True,
                    internal_blocking=False,
                )
            )
        if not nonblocking_warn.empty:
            evidence = "; ".join(
                f"{item['check']}: {item['evidence']}"
                for _, item in nonblocking_warn.head(8).iterrows()
            )
            checks.append(
                row(
                    "metadata_quality",
                    "metadata quality nonblocking warnings",
                    WARN,
                    f"warnings={len(nonblocking_warn)}; {evidence}",
                    "Review these warnings before making strong Q2+ claims.",
                    q2_blocking=False,
                    internal_blocking=False,
                )
            )

    if split_leakage is None or split_leakage.empty:
        checks.append(
            row(
                "metadata_quality",
                "external split leakage audit generated",
                FAIL,
                "paper_metadata_split_leakage_audit.csv missing or empty",
                "Run scripts/audit_rr_metadata_quality.py to generate split leakage and balance screens.",
                q2_blocking=True,
                internal_blocking=False,
            )
        )
    else:
        q2_mask = split_leakage["q2_blocking"].astype(bool)
        q2_fail = split_leakage[q2_mask & split_leakage["status"].isin([FAIL, WARN])]
        warn = split_leakage[(~q2_mask) & split_leakage["status"].eq(WARN)]
        if q2_fail.empty:
            status = WARN if not warn.empty else PASS
            evidence = (
                f"q2_blocking_failures=0; warnings={len(warn)}"
                if not warn.empty
                else f"q2_blocking_failures=0; checks={len(split_leakage)}"
            )
            checks.append(
                row(
                    "metadata_quality",
                    "external split leakage screen",
                    status,
                    evidence,
                    "Review WARN rows before claiming animal/date/camera/scene-independent external validation.",
                    q2_blocking=False,
                    internal_blocking=False,
                )
            )
        else:
            evidence = "; ".join(
                f"{item['check']} ({item['status']}): {item['evidence']}"
                for _, item in q2_fail.head(8).iterrows()
            )
            checks.append(
                row(
                    "metadata_quality",
                    "external split leakage screen",
                    FAIL,
                    f"q2_blocking_failures={len(q2_fail)}; {evidence}",
                    (
                        "Add/import independent external rows before external leakage "
                        "screening; do not relabel current development rows as external."
                    ),
                    q2_blocking=True,
                    internal_blocking=False,
                )
            )
    return checks


def method_freeze_checks(method_freeze: pd.DataFrame | None) -> list[dict[str, object]]:
    if method_freeze is None or method_freeze.empty:
        return [
            row(
                "method_freeze",
                "external method freeze manifest locked",
                FAIL,
                "paper_method_freeze_summary.csv missing or empty",
                "Run scripts/freeze_rr_external_method.py before external validation and rerun this readiness audit.",
                q2_blocking=True,
                internal_blocking=False,
            )
        ]
    item = method_freeze.iloc[0]
    status = str(item.get("freeze_status", "")).strip().lower()
    required_missing = int(float(item.get("required_files_missing", 1)))
    required_present = int(float(item.get("required_files_present", 0)))
    freeze_id = str(item.get("method_freeze_id", ""))
    locked = status == "locked" and required_missing == 0 and bool(freeze_id)
    return [
        row(
            "method_freeze",
            "external method freeze manifest locked",
            PASS if locked else FAIL,
            (
                f"freeze_status={status}; freeze_id={freeze_id}; "
                f"required_present={required_present}; required_missing={required_missing}"
            ),
            "Regenerate scripts/freeze_rr_external_method.py after refreshing method assets; use this freeze ID for external validation.",
            q2_blocking=not locked,
            internal_blocking=False,
        )
    ]


def output_status(readiness: pd.DataFrame) -> tuple[str, str]:
    internal_blockers = readiness[
        (readiness["status"] == FAIL) & (readiness["internal_manuscript_blocking"].astype(bool))
    ]
    q2_blockers = readiness[
        (readiness["status"] == FAIL) & (readiness["q2_blocking"].astype(bool))
    ]
    q2_warnings = readiness[
        (readiness["status"] == WARN) & (readiness["q2_blocking"].astype(bool))
    ]
    internal = "ready" if internal_blockers.empty else "not_ready"
    q2 = "ready" if q2_blockers.empty and q2_warnings.empty else "not_ready"
    return internal, q2


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    data = table[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(value.replace("\n", " ") for value in row) + " |" for row in data.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(readiness: pd.DataFrame, output_path: Path, internal_status: str, q2_status: str) -> None:
    q2_blockers = readiness[
        ((readiness["status"] == FAIL) | (readiness["status"] == WARN))
        & readiness["q2_blocking"].astype(bool)
    ].copy()
    internal_blockers = readiness[
        (readiness["status"] == FAIL) & readiness["internal_manuscript_blocking"].astype(bool)
    ].copy()
    text = f"""# Thermal RR Submission Readiness Audit

Generated: {datetime.now().isoformat(timespec="seconds")}

## Overall Status

- Internal manuscript package: `{internal_status}`
- Q2-or-higher submission readiness: `{q2_status}`
- Internal blocking checks: {len(internal_blockers)}
- Q2 blocking or caution checks: {len(q2_blockers)}

Interpretation: `ready` for the internal manuscript package means the current code, tables, figures, and draft artifacts are present. `ready` for Q2-or-higher submission additionally requires real metadata-stratified validation, heat-stress context, and external/generalization evidence.

## Q2 Blocking Or Caution Items

{markdown_table(q2_blockers, ["category", "check", "status", "evidence", "recommended_action"]) if not q2_blockers.empty else "No Q2 blocking or caution items detected."}

## Internal Manuscript Blocking Items

{markdown_table(internal_blockers, ["category", "check", "status", "evidence", "recommended_action"]) if not internal_blockers.empty else "No internal manuscript-package blockers detected."}

## All Checks

{markdown_table(readiness, ["category", "check", "status", "q2_blocking", "internal_manuscript_blocking", "evidence", "recommended_action"])}
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    paper_assets = input_root / f"{args.corrected_prefix}_paper_assets"
    figures = input_root / f"{args.corrected_prefix}_figures"
    signal_aware_figures = input_root / f"{args.output_prefix}_signal_aware_residual_figures"
    repo_root = Path(__file__).resolve().parents[1]

    main_results = read_csv_if_exists(paper_assets / "paper_main_results_table.csv")
    metric_tiers = read_csv_if_exists(paper_assets / "paper_metric_reporting_tiers.csv")
    safe_policy_metrics = read_csv_if_exists(
        paper_assets / "paper_signal_aware_safe_policy_metrics_table.csv"
    )
    if safe_policy_metrics is None:
        safe_policy_metrics = read_csv_if_exists(
            input_root / f"{args.output_prefix}_signal_aware_safe_policy_metrics.csv"
        )
    bootstrap = read_csv_if_exists(paper_assets / "paper_bootstrap_ci_table.csv")
    metadata = read_csv_if_exists(paper_assets / "paper_metadata_group_readiness_table.csv")
    heat = read_csv_if_exists(paper_assets / "paper_heat_stress_readiness.csv")
    metadata_quality = read_csv_if_exists(paper_assets / "paper_metadata_quality_audit.csv")
    split_leakage = read_csv_if_exists(paper_assets / "paper_metadata_split_leakage_audit.csv")
    method_freeze = read_csv_if_exists(paper_assets / "paper_method_freeze_summary.csv")
    external_split = read_csv_if_exists(paper_assets / "paper_external_split_readiness_table.csv")
    if external_split is None:
        external_split = read_csv_if_exists(input_root / f"{args.output_prefix}_external_split_readiness.csv")

    checks: list[dict[str, object]] = []
    checks.extend(metric_checks(main_results, args))
    checks.extend(metric_reporting_tier_checks(metric_tiers))
    checks.extend(safe_policy_checks(safe_policy_metrics))
    checks.extend(bootstrap_checks(bootstrap))
    checks.extend(metadata_checks(metadata, heat))
    checks.extend(metadata_quality_checks(metadata_quality, split_leakage))
    checks.extend(method_freeze_checks(method_freeze))
    checks.extend(external_split_checks(external_split))
    for path, label, min_bytes in [
        (paper_assets / "paper_main_results_table.csv", "main results table", 100),
        (paper_assets / "paper_metric_reporting_tiers.csv", "metric reporting tiers table", 100),
        (
            paper_assets / "paper_rr_method_statistical_tests_table.csv",
            "paired RR statistical tests table",
            100,
        ),
        (
            paper_assets / "paper_signal_aware_residual_metrics_table.csv",
            "signal-aware residual metrics table",
            100,
        ),
        (
            paper_assets / "paper_signal_aware_safe_policy_metrics_table.csv",
            "signal-aware safe policy metrics table",
            100,
        ),
        (
            paper_assets / "paper_signal_aware_safe_policy_bootstrap_ci_table.csv",
            "signal-aware safe policy bootstrap CI table",
            100,
        ),
        (
            paper_assets / "paper_signal_aware_safe_policy_cases_table.csv",
            "signal-aware safe policy cases table",
            100,
        ),
        (paper_assets / "paper_feature_block_ablation_table.csv", "feature-block ablation table", 100),
        (paper_assets / "paper_representative_case_table.csv", "representative case table", 100),
        (
            paper_assets / "paper_literature_innovation_crosswalk.csv",
            "literature innovation crosswalk table",
            100,
        ),
        (
            paper_assets / "paper_literature_source_register.csv",
            "literature source register",
            100,
        ),
        (
            paper_assets / "paper_metadata_annotation_q2_unblock_queue.csv",
            "Q2 metadata unblock queue",
            100,
        ),
        (
            paper_assets / "paper_rr_bilateral_consistency_summary.csv",
            "bilateral nostril consistency summary",
            100,
        ),
        (
            paper_assets / "paper_rr_bilateral_consistency_report.md",
            "bilateral nostril consistency report",
            1000,
        ),
        (paper_assets / "paper_assets_summary.md", "paper assets summary", 1000),
        (
            paper_assets / "paper_literature_innovation_crosswalk.md",
            "literature innovation crosswalk report",
            1000,
        ),
        (paper_assets / "paper_manuscript_claim_update.md", "paper manuscript claim update", 1000),
        (figures / "rr_quality_residual_combined.png", "combined main figure", 10000),
        (figures / "rr_representative_cases.png", "representative case figure", 10000),
        (
            signal_aware_figures / "signal_aware_rr_prediction_scatter.png",
            "signal-aware prediction scatter figure",
            10000,
        ),
        (
            signal_aware_figures / "signal_aware_bootstrap_ci.png",
            "signal-aware bootstrap CI figure",
            10000,
        ),
        (
            signal_aware_figures / "signal_aware_case_examples.png",
            "signal-aware case examples figure",
            10000,
        ),
        (repo_root / "docs" / "thermal_rr_manuscript_draft.md", "English manuscript draft", 1000),
        (
            repo_root / "docs" / "thermal_rr_manuscript_claim_update.md",
            "manuscript claim update document",
            1000,
        ),
        (repo_root / "docs" / "thermal_rr_reproducibility_readme.md", "reproducibility README", 1000),
        (repo_root / "docs" / "thermal_rr_submission_strategy.md", "submission strategy document", 1000),
    ]:
        checks.append(file_check(path, label, min_bytes=min_bytes))
    checks.append(
        text_contains_check(
            paper_assets / "paper_manuscript_claim_update.md",
            "manuscript claim guardrails present",
            [
                "Do not use truth-calibrated R2 as main performance.",
                "Do not call signal-aware externally validated",
                "RR R2=0.9997",
                "Q2+ biological meaning",
            ],
            min_bytes=1000,
        )
    )
    checks.append(
        text_contains_check(
            paper_assets / "paper_literature_innovation_crosswalk.md",
            "literature claim crosswalk guardrails present",
            [
                "Signal-aware residual correction is the highest-precision internal candidate.",
                "Truth-calibrated results are only an oracle upper bound.",
                "Do not make Q2+ external, true animal-identity, heat-stress, or deployment claims",
            ],
            min_bytes=1000,
        )
    )

    readiness = pd.DataFrame(checks)
    internal_status, q2_status = output_status(readiness)
    readiness["internal_package_status"] = internal_status
    readiness["q2_submission_status"] = q2_status

    csv_path = input_root / f"{args.corrected_prefix}_submission_readiness.csv"
    md_path = input_root / f"{args.corrected_prefix}_submission_readiness_summary.md"
    readiness.to_csv(csv_path, index=False)
    write_report(readiness, md_path, internal_status, q2_status)

    print(f"Saved submission readiness table: {csv_path}")
    print(f"Saved submission readiness report: {md_path}")
    print(f"Internal manuscript package: {internal_status}")
    print(f"Q2-or-higher submission readiness: {q2_status}")
    print("\nQ2 blocking/caution checks:")
    blockers = readiness[
        ((readiness["status"] == FAIL) | (readiness["status"] == WARN))
        & readiness["q2_blocking"].astype(bool)
    ]
    if blockers.empty:
        print("None")
    else:
        print(blockers[["category", "check", "status", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
