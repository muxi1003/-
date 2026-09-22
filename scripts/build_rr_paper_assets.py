from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build paper-ready tables and metadata templates for RR innovation assets."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def read_csv_with_encoding_fallback(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise UnicodeError(f"Could not decode {path}: " + " | ".join(errors))


def read_inputs(input_root: Path, output_prefix: str, corrected_prefix: str) -> dict[str, pd.DataFrame]:
    files = {
        "summary": input_root / f"{output_prefix}_summary.csv",
        "metrics": input_root / f"{output_prefix}_metrics.csv",
        "truth_calibrated_metrics": input_root / f"{output_prefix}_truth_calibrated_metrics.csv",
        "validation_summary": input_root / f"{corrected_prefix}_validation_summary.csv",
        "group_metrics": input_root / f"{corrected_prefix}_group_metrics.csv",
        "group_metrics_by_prefix": input_root / f"{corrected_prefix}_group_metrics_by_prefix.csv",
        "bootstrap_ci": input_root / f"{corrected_prefix}_bootstrap_ci.csv",
        "error_taxonomy": input_root / f"{output_prefix}_error_taxonomy.csv",
        "error_taxonomy_summary": input_root / f"{output_prefix}_error_taxonomy_summary.csv",
        "feature_importance": input_root / f"{corrected_prefix}_feature_importance.csv",
    }
    data = {name: pd.read_csv(require_file(path)) for name, path in files.items()}
    optional_files = {
        "ablation_metrics": input_root / f"{corrected_prefix}_ablation_metrics.csv",
        "case_examples": input_root / f"{corrected_prefix}_case_examples.csv",
        "submission_readiness": input_root / f"{corrected_prefix}_submission_readiness.csv",
        "signal_consensus_comparison": input_root
        / f"{output_prefix}_signal_consensus_comparison_summary.csv",
        "signal_aware_residual_metrics": input_root
        / f"{output_prefix}_signal_aware_residual_metrics.csv",
        "signal_aware_residual_threshold_grid": input_root
        / f"{output_prefix}_signal_aware_residual_threshold_grid.csv",
        "signal_aware_residual_bootstrap_ci": input_root
        / f"{output_prefix}_signal_aware_residual_bootstrap_ci.csv",
        "signal_aware_safe_policy_metrics": input_root
        / f"{output_prefix}_signal_aware_safe_policy_metrics.csv",
        "signal_aware_safe_policy_bootstrap_ci": input_root
        / f"{output_prefix}_signal_aware_safe_policy_bootstrap_ci.csv",
        "signal_aware_safe_policy_cases": input_root
        / f"{output_prefix}_signal_aware_safe_policy_cases.csv",
        "rr_method_paired_predictions": input_root / f"{output_prefix}_rr_method_paired_predictions.csv",
        "rr_method_statistics": input_root / f"{output_prefix}_rr_method_statistics.csv",
        "rr_method_statistical_tests": input_root / f"{output_prefix}_rr_method_statistical_tests.csv",
        "selective_rr_metrics": input_root / f"{output_prefix}_selective_rr_metrics.csv",
        "selective_rr_threshold_grid": input_root
        / f"{output_prefix}_selective_rr_threshold_grid.csv",
        "algorithmic_quality_tier_metrics": input_root
        / f"{output_prefix}_algorithmic_quality_tier_metrics.csv",
        "algorithmic_quality_feature_summary": input_root
        / f"{output_prefix}_algorithmic_quality_feature_summary.csv",
        "algorithmic_quality_threshold_grid": input_root
        / f"{output_prefix}_algorithmic_quality_threshold_grid.csv",
        "deployment_decision_curve": input_root
        / f"{output_prefix}_deployment_decision_curve.csv",
        "deployment_decision_recommendation": input_root
        / f"{output_prefix}_deployment_decision_recommendation.csv",
        "conformal_rr_metrics": input_root / f"{output_prefix}_conformal_rr_metrics.csv",
        "conformal_rr_recommendation": input_root
        / f"{output_prefix}_conformal_rr_recommendation.csv",
        "post_safe_gate_error_cases": input_root
        / f"{output_prefix}_post_safe_gate_error_cases.csv",
        "post_safe_gate_probe_metrics": input_root
        / f"{output_prefix}_post_safe_gate_second_stage_probe_metrics.csv",
        "external_split_readiness": input_root / f"{output_prefix}_external_split_readiness.csv",
        "external_split_metrics": input_root / f"{output_prefix}_external_split_metrics.csv",
        "external_split_acceptance": input_root / f"{output_prefix}_external_split_acceptance.csv",
    }
    for name, path in optional_files.items():
        if path.exists():
            data[name] = pd.read_csv(path)
    return data


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def fmt_exact(row: pd.Series) -> str:
    exact = int(row["exact_count"])
    total = int(row["count_valid_videos"])
    return f"{exact}/{total}"


def method_label(label: str) -> str:
    mapping = {
        "baseline_default": "Default thermal RR pipeline",
        "fixed_default_threshold_quality_residual": "Quality-aware residual correction (fixed threshold, out-of-fold)",
        "best_grid_threshold_quality_residual": "Quality-aware residual correction (best threshold grid, sensitivity only)",
        "nested_threshold_cv_quality_residual": "Quality-aware residual correction (nested threshold CV)",
        "signal_consensus_supplement_fixed_threshold": "Signal-consensus supplement (fixed threshold, out-of-fold)",
        "nested_signal_consensus_supplement": "Signal-consensus supplement (nested threshold CV)",
        "group_holdout_baseline_default": "Default thermal RR pipeline",
        "group_holdout_fixed_threshold": "Quality-aware residual correction (leave-one-prefix-group-out, fixed threshold)",
        "group_holdout_train_selected_threshold": "Quality-aware residual correction (leave-one-prefix-group-out, threshold selected on training groups)",
        "group_holdout_fixed_signal_consensus_supplement": "Signal-consensus supplement (leave-one-prefix-group-out, fixed threshold)",
        "group_holdout_train_selected_signal_consensus_supplement": "Signal-consensus supplement (leave-one-prefix-group-out, threshold selected on training groups)",
        "signal_aware_residual_fixed_oof": "Signal-aware residual correction (fixed threshold, out-of-fold)",
        "signal_aware_residual_best_threshold_sensitivity": "Signal-aware residual correction (best threshold grid, sensitivity only)",
        "signal_aware_residual_prefix_group_fixed_threshold": "Signal-aware residual correction (leave-one-prefix-group-out, fixed threshold)",
        "signal_aware_safe_policy_fixed_oof": "Conservative signal-aware safe gate (fixed threshold, out-of-fold)",
        "signal_aware_safe_policy_prefix_group_fixed_threshold": "Conservative signal-aware safe gate (leave-one-prefix-group-out, fixed threshold)",
        "paper_repro_truth_calibrated": "Truth-calibrated upper bound",
    }
    return mapping.get(label, label)


def build_main_results_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    validation = data["validation_summary"].copy()
    group_metrics = data["group_metrics"].copy()
    truth_upper = data["truth_calibrated_metrics"].copy()
    rows = []
    for _, row in validation.iterrows():
        rows.append(
            {
                "method": method_label(str(row["label"])),
                "validation_setting": str(row.get("evaluation_note", "")),
                "videos": int(row["videos"]),
                "rr_r2": float(row["rr_r2"]),
                "rr_pearson_r2": float(row["rr_pearson_r2"]),
                "rr_mae_bpm": float(row["rr_mae"]),
                "rr_rmse_bpm": float(row["rr_rmse"]),
                "exact_count": fmt_exact(row),
                "within_one_count": fmt_exact(
                    pd.Series(
                        {
                            "exact_count": row["within_one_count"],
                            "count_valid_videos": row["count_valid_videos"],
                        }
                    )
                ),
                "paper_use": (
                    "main_or_supplement"
                    if str(row["label"]) != "best_grid_threshold_quality_residual"
                    else "sensitivity_only"
                ),
            }
        )
    for _, row in group_metrics.iterrows():
        if str(row["label"]) == "group_holdout_baseline_default":
            continue
        rows.append(
            {
                "method": method_label(str(row["label"])),
                "validation_setting": str(row.get("evaluation_note", "")),
                "videos": int(row["videos"]),
                "rr_r2": float(row["rr_r2"]),
                "rr_pearson_r2": float(row["rr_pearson_r2"]),
                "rr_mae_bpm": float(row["rr_mae"]),
                "rr_rmse_bpm": float(row["rr_rmse"]),
                "exact_count": fmt_exact(row),
                "within_one_count": fmt_exact(
                    pd.Series(
                        {
                            "exact_count": row["within_one_count"],
                            "count_valid_videos": row["count_valid_videos"],
                        }
                    )
                ),
                "paper_use": "internal_group_generalization",
            }
        )
    signal_consensus = data.get("signal_consensus_comparison")
    if signal_consensus is not None and not signal_consensus.empty:
        signal_paper_use = {
            "signal_consensus_supplement_fixed_threshold": "candidate_internal_extension",
            "nested_signal_consensus_supplement": "candidate_nested_validation",
            "group_holdout_fixed_signal_consensus_supplement": "candidate_internal_group_generalization",
            "group_holdout_train_selected_signal_consensus_supplement": "candidate_internal_group_generalization",
        }
        for label, paper_use in signal_paper_use.items():
            match = signal_consensus[signal_consensus["label"].astype(str) == label]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "method": method_label(str(row["label"])),
                    "validation_setting": str(row.get("evaluation_note", "")),
                    "videos": int(row["videos"]),
                    "rr_r2": float(row["rr_r2"]),
                    "rr_pearson_r2": float(row["rr_pearson_r2"]),
                    "rr_mae_bpm": float(row["rr_mae"]),
                    "rr_rmse_bpm": float(row["rr_rmse"]),
                    "exact_count": fmt_exact(row),
                    "within_one_count": fmt_exact(
                        pd.Series(
                            {
                                "exact_count": row["within_one_count"],
                                "count_valid_videos": row["count_valid_videos"],
                            }
                        )
                    ),
                    "paper_use": paper_use,
                }
            )
    signal_aware = data.get("signal_aware_residual_metrics")
    if signal_aware is not None and not signal_aware.empty:
        signal_aware_paper_use = {
            "signal_aware_residual_fixed_oof": "candidate_internal_precision",
            "signal_aware_residual_best_threshold_sensitivity": "sensitivity_only",
            "signal_aware_residual_prefix_group_fixed_threshold": "candidate_group_validation_failed",
        }
        for label, paper_use in signal_aware_paper_use.items():
            match = signal_aware[signal_aware["label"].astype(str) == label]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "method": method_label(str(row["label"])),
                    "validation_setting": str(row.get("evaluation_note", "")),
                    "videos": int(row["videos"]),
                    "rr_r2": float(row["rr_r2"]),
                    "rr_pearson_r2": float(row["rr_pearson_r2"]),
                    "rr_mae_bpm": float(row["rr_mae"]),
                    "rr_rmse_bpm": float(row["rr_rmse"]),
                    "exact_count": fmt_exact(row),
                    "within_one_count": fmt_exact(
                        pd.Series(
                            {
                                "exact_count": row["within_one_count"],
                                "count_valid_videos": row["count_valid_videos"],
                            }
                        )
                    ),
                    "paper_use": paper_use,
                }
            )
    safe_policy = data.get("signal_aware_safe_policy_metrics")
    if safe_policy is not None and not safe_policy.empty:
        safe_policy_paper_use = {
            "signal_aware_safe_policy_fixed_oof": "candidate_internal_precision_safety_gated",
            "signal_aware_safe_policy_prefix_group_fixed_threshold": "candidate_prefix_group_stress_passed_internal_only",
        }
        for label, paper_use in safe_policy_paper_use.items():
            match = safe_policy[safe_policy["label"].astype(str) == label]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "method": method_label(str(row["label"])),
                    "validation_setting": str(row.get("evaluation_note", "")),
                    "videos": int(row["videos"]),
                    "rr_r2": float(row["rr_r2"]),
                    "rr_pearson_r2": float(row["rr_pearson_r2"]),
                    "rr_mae_bpm": float(row["rr_mae"]),
                    "rr_rmse_bpm": float(row["rr_rmse"]),
                    "exact_count": fmt_exact(row),
                    "within_one_count": fmt_exact(
                        pd.Series(
                            {
                                "exact_count": row["within_one_count"],
                                "count_valid_videos": row["count_valid_videos"],
                            }
                        )
                    ),
                    "paper_use": paper_use,
                }
            )
    for _, row in truth_upper.iterrows():
        rows.append(
            {
                "method": method_label(str(row["label"])),
                "validation_setting": "truth_assisted_upper_bound_not_main_method",
                "videos": int(row["videos"]),
                "rr_r2": float(row["rr_r2"]),
                "rr_pearson_r2": float(row["rr_pearson_r2"]),
                "rr_mae_bpm": float(row["rr_mae"]),
                "rr_rmse_bpm": float(row["rr_rmse"]),
                "exact_count": fmt_exact(row),
                "within_one_count": fmt_exact(
                    pd.Series(
                        {
                            "exact_count": row["within_one_count"],
                            "count_valid_videos": row["count_valid_videos"],
                        }
                    )
                ),
                "paper_use": "upper_bound_only",
            }
        )
    return pd.DataFrame(rows)


def reporting_tier_for_result(method: str, paper_use: str) -> dict[str, str]:
    method_lower = method.lower()
    if paper_use == "upper_bound_only" or "truth-calibrated" in method_lower:
        return {
            "evidence_tier": "oracle_upper_bound",
            "main_text_allowed": "yes_with_caveat",
            "primary_performance_allowed": "no",
            "recommended_location": "supplement_or_upper_bound_paragraph",
            "leakage_risk": "high_truth_assisted",
            "claim_boundary": (
                "Use only as a truth-assisted upper bound showing that the extracted "
                "temperature curves contain respiratory information."
            ),
            "recommended_wording": (
                "Truth-assisted calibration yielded an upper-bound RR R2, but this "
                "post-hoc result used reference labels and was not used as the primary "
                "performance estimate."
            ),
        }
    if paper_use == "sensitivity_only" or "best threshold grid" in method_lower:
        return {
            "evidence_tier": "sensitivity_analysis",
            "main_text_allowed": "yes_with_caveat",
            "primary_performance_allowed": "no",
            "recommended_location": "supplement_or_sensitivity_analysis",
            "leakage_risk": "parameter_selection_on_current_set",
            "claim_boundary": (
                "Use only to show sensitivity to threshold choice; do not present as "
                "the deployable method estimate."
            ),
            "recommended_wording": (
                "A grid-selected threshold was reported as sensitivity analysis only."
            ),
        }
    if method == "Default thermal RR pipeline":
        return {
            "evidence_tier": "primary_internal_baseline",
            "main_text_allowed": "yes",
            "primary_performance_allowed": "yes",
            "recommended_location": "main_results_baseline",
            "leakage_risk": "internal_validation_only",
            "claim_boundary": (
                "Report as the current 73-video internal baseline; do not claim "
                "external generalization."
            ),
            "recommended_wording": (
                "The default thermal RR pipeline achieved the internal baseline RR R2."
            ),
        }
    if paper_use == "main_or_supplement" and "fixed threshold" in method_lower:
        return {
            "evidence_tier": "primary_internal_innovation",
            "main_text_allowed": "yes",
            "primary_performance_allowed": "yes",
            "recommended_location": "main_results_innovation",
            "leakage_risk": "internal_out_of_fold_validation",
            "claim_boundary": (
                "Use as the main internal innovation result, with the Q2 caveat that "
                "cow-level metadata and external validation are still required."
            ),
            "recommended_wording": (
                "The quality-aware residual correction improved internal out-of-fold "
                "RR R2, MAE, RMSE, and exact-count performance over the default pipeline."
            ),
        }
    if paper_use == "main_or_supplement" and "nested threshold cv" in method_lower:
        return {
            "evidence_tier": "conservative_internal_validation",
            "main_text_allowed": "yes",
            "primary_performance_allowed": "yes",
            "recommended_location": "main_results_validation",
            "leakage_risk": "internal_nested_validation",
            "claim_boundary": (
                "Use as the conservative internal validation estimate; still not an "
                "external validation claim."
            ),
            "recommended_wording": (
                "Nested threshold validation supported the internal robustness of the "
                "residual-correction strategy."
            ),
        }
    if paper_use == "internal_group_generalization":
        return {
            "evidence_tier": "internal_prefix_group_validation",
            "main_text_allowed": "yes",
            "primary_performance_allowed": "secondary",
            "recommended_location": "main_or_supplement_group_validation",
            "leakage_risk": "proxy_grouping_not_real_cow_id",
            "claim_boundary": (
                "Report as prefix-group internal validation only; replace with real "
                "cow_id/date/camera grouping before a strong generalization claim."
            ),
            "recommended_wording": (
                "Prefix-group holdout suggested internal grouped robustness, but real "
                "animal-level grouping remains required."
            ),
        }
    if paper_use == "candidate_group_validation_failed":
        return {
            "evidence_tier": "failed_proxy_group_validation",
            "main_text_allowed": "yes_as_caution",
            "primary_performance_allowed": "no",
            "recommended_location": "limitations_or_supplementary_caution",
            "leakage_risk": "proxy_grouping_not_real_cow_id",
            "claim_boundary": (
                "Use only as a caution that the high-precision candidate did not "
                "survive prefix-group stress testing."
            ),
            "recommended_wording": (
                "The signal-aware second-stage correction improved stratified "
                "out-of-fold performance but degraded under prefix-group holdout, "
                "so it was retained only as an internal candidate pending real "
                "cow-level and external validation."
            ),
        }
    if paper_use == "candidate_internal_precision_safety_gated":
        return {
            "evidence_tier": "candidate_internal_safety_gated_precision",
            "main_text_allowed": "yes_with_caveat",
            "primary_performance_allowed": "secondary",
            "recommended_location": "secondary_results_or_methods_development",
            "leakage_risk": "internal_out_of_fold_validation",
            "claim_boundary": (
                "Use as a safety-gated internal candidate that improves precision "
                "without using truth labels at prediction time; still requires true "
                "animal identity/session metadata and external validation for broader "
                "generalization wording."
            ),
            "recommended_wording": (
                "A conservative signal-aware gate improved internal out-of-fold RR "
                "precision while retaining non-truth prediction-time inputs."
            ),
        }
    if paper_use == "candidate_prefix_group_stress_passed_internal_only":
        return {
            "evidence_tier": "candidate_proxy_group_stress_passed",
            "main_text_allowed": "yes_with_caveat",
            "primary_performance_allowed": "secondary",
            "recommended_location": "secondary_results_or_robustness_analysis",
            "leakage_risk": "proxy_grouping_not_real_cow_id",
            "claim_boundary": (
                "Use as evidence that the conservative gate reduced the prefix-group "
                "failure mode; do not claim external generalization until real "
                "cow/session/external validation is complete."
            ),
            "recommended_wording": (
                "The conservative signal-aware gate removed the large prefix-group "
                "count-error failures seen in the ungated signal-aware candidate."
            ),
        }
    if paper_use.startswith("candidate_"):
        return {
            "evidence_tier": "candidate_internal_extension",
            "main_text_allowed": "yes_with_caveat",
            "primary_performance_allowed": "secondary",
            "recommended_location": "supplement_or_secondary_results",
            "leakage_risk": "internal_candidate_not_method_frozen",
            "claim_boundary": (
                "Use as a candidate extension or secondary analysis unless it is "
                "method-frozen and validated externally."
            ),
            "recommended_wording": (
                "Signal-consensus results were treated as a candidate extension that "
                "requires further grouped and external validation."
            ),
        }
    return {
        "evidence_tier": "unclassified",
        "main_text_allowed": "no",
        "primary_performance_allowed": "no",
        "recommended_location": "do_not_report_until_classified",
        "leakage_risk": "unknown",
        "claim_boundary": "Classify this result before manuscript use.",
        "recommended_wording": "Do not report this metric until its evidence tier is assigned.",
    }


def build_metric_reporting_tiers(main_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for display_order, (_, row) in enumerate(main_results.iterrows(), start=1):
        method = str(row["method"])
        tier = reporting_tier_for_result(method, str(row["paper_use"]))
        rows.append(
            {
                "display_order": display_order,
                "method": method,
                "validation_setting": str(row["validation_setting"]),
                "rr_r2": float(row["rr_r2"]),
                "rr_mae_bpm": float(row["rr_mae_bpm"]),
                "rr_rmse_bpm": float(row["rr_rmse_bpm"]),
                "exact_count": str(row["exact_count"]),
                "source_paper_use": str(row["paper_use"]),
                **tier,
            }
        )
    return pd.DataFrame(rows)


def build_group_table(group_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot_rows = []
    for group_name, group_df in group_metrics.groupby("heldout_prefix_group", sort=True):
        by_label = {str(row["label"]).split("_", 1)[1]: row for _, row in group_df.iterrows()}
        baseline = by_label["baseline_default"]
        selected = by_label["train_selected_threshold"]
        pivot_rows.append(
            {
                "heldout_prefix_group": group_name,
                "videos": int(baseline["videos"]),
                "baseline_rr_r2": float(baseline["rr_r2"]),
                "selected_threshold_rr_r2": float(selected["rr_r2"]),
                "delta_rr_r2": float(selected["rr_r2"]) - float(baseline["rr_r2"]),
                "baseline_mae_bpm": float(baseline["rr_mae"]),
                "selected_threshold_mae_bpm": float(selected["rr_mae"]),
                "delta_mae_bpm": float(selected["rr_mae"]) - float(baseline["rr_mae"]),
                "baseline_exact": fmt_exact(baseline),
                "selected_threshold_exact": fmt_exact(selected),
                "interpretation": group_interpretation(
                    group_name,
                    float(selected["rr_r2"]) - float(baseline["rr_r2"]),
                    int(selected["exact_count"]) - int(baseline["exact_count"]),
                ),
            }
        )
    return pd.DataFrame(pivot_rows)


def group_interpretation(group_name: str, delta_r2: float, delta_exact: int) -> str:
    if delta_r2 > 0 and delta_exact > 0:
        return "improved under prefix-group holdout"
    if delta_r2 < 0 or delta_exact < 0:
        return "degraded; needs conservative thresholds or external validation"
    return "unchanged; likely different residual-error pattern or no confident correction"


def build_error_table(error_summary: pd.DataFrame, total_errors: int) -> pd.DataFrame:
    table = error_summary.copy()
    table["percentage_of_errors"] = table["videos"] / total_errors * 100.0
    explanations = {
        "endpoint_or_boundary_missing_peak": "Missed breath near analysis-window boundary or long edge gap.",
        "boundary_extra_peak_or_window_mismatch": "Extra boundary peak or mismatch between analysis window and breathing cycle.",
        "noise_or_double_peak_overcount": "Short interval suggests local noise or double-counted peak.",
        "overcount_periodicity_ambiguous": "Overcount without a single dominant local signature.",
        "single_channel_or_fusion_missing_peak": "One nostril channel or fusion choice likely suppresses a valid peak.",
        "weak_noise_peak_overcount": "Detected peak has weak prominence relative to median peak.",
    }
    table["paper_interpretation"] = table["error_type"].map(explanations).fillna("")
    return table


def build_bootstrap_table(bootstrap_ci: pd.DataFrame) -> pd.DataFrame:
    keep = ["delta_rr_r2", "delta_rr_mae", "delta_rr_rmse", "delta_exact_count"]
    table = bootstrap_ci[bootstrap_ci["metric"].isin(keep)].copy()
    names = {
        "delta_rr_r2": "Delta RR R2",
        "delta_rr_mae": "Delta MAE (bpm)",
        "delta_rr_rmse": "Delta RMSE (bpm)",
        "delta_exact_count": "Delta exact count",
    }
    table["display_metric"] = table["metric"].map(names)
    table["estimate_with_ci"] = table.apply(
        lambda row: (
            f"{float(row['estimate']):.4f} "
            f"({float(row['ci_low_2_5']):.4f}, {float(row['ci_high_97_5']):.4f})"
        ),
        axis=1,
    )
    return table[
        [
            "display_metric",
            "estimate",
            "ci_low_2_5",
            "ci_high_97_5",
            "estimate_with_ci",
            "bootstrap_samples",
            "note",
        ]
    ]


def build_feature_table(feature_importance: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    table = feature_importance.head(top_n).copy()
    table["feature"] = table["feature"].str.replace("num__", "", regex=False).str.replace(
        "cat__", "", regex=False
    )
    table["paper_interpretation"] = table["feature"].map(feature_interpretation).fillna(
        "Curve-quality or pipeline-state feature used by the residual corrector."
    )
    return table


def feature_interpretation(feature: str) -> str:
    if "peak" in feature:
        return "Encodes detected or candidate breath-cycle count information."
    if "gap" in feature:
        return "Encodes endpoint or boundary-cycle ambiguity."
    if "interval" in feature:
        return "Encodes periodicity stability of the breathing curve."
    if "prominence" in feature:
        return "Encodes peak salience and weak/noisy peak risk."
    if "spectral" in feature:
        return "Encodes frequency-domain consistency of the curve."
    if "amplitude" in feature or "std" in feature:
        return "Encodes signal strength of the nostril temperature curve."
    return "Curve-quality or pipeline-state feature used by the residual corrector."


def build_ablation_table(ablation_metrics: pd.DataFrame) -> pd.DataFrame:
    keep_labels = [
        "baseline_default",
        "all_features",
        "summary_only",
        "curve_only",
        "without_peak_count_context",
        "without_endpoint_gap_context",
        "without_prominence_context",
        "without_spectral_context",
        "without_signal_strength_context",
        "without_missing_fusion_context",
    ]
    table = ablation_metrics[ablation_metrics["label"].isin(keep_labels)].copy()
    table["paper_use"] = table["label"].map(
        {
            "baseline_default": "baseline",
            "all_features": "main_internal_validation",
            "summary_only": "ablation",
            "curve_only": "ablation",
            "without_peak_count_context": "ablation",
            "without_endpoint_gap_context": "ablation",
            "without_prominence_context": "ablation",
            "without_spectral_context": "ablation",
            "without_signal_strength_context": "ablation",
            "without_missing_fusion_context": "ablation",
        }
    )
    columns = [
        "label",
        "description",
        "retained_feature_count",
        "dropped_feature_count",
        "rr_r2",
        "rr_mae",
        "rr_rmse",
        "exact_count",
        "applied_corrections",
        "delta_vs_full_rr_r2",
        "delta_vs_full_rr_mae",
        "delta_vs_full_exact_count",
        "paper_use",
    ]
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    return table[columns]


def build_metadata_template(summary: pd.DataFrame, existing_path: Path | None = None) -> pd.DataFrame:
    columns = [
        "video_id",
        "cow_id",
        "collection_date",
        "collection_start_date",
        "collection_end_date",
        "collection_time",
        "collection_location_country",
        "collection_location_province",
        "collection_location_county",
        "collection_site",
        "camera_id",
        "scene_id",
        "ambient_temperature_c",
        "relative_humidity_percent",
        "thi",
        "athi",
        "posture",
        "head_motion_score_0_3",
        "occlusion_score_0_3",
        "nostril_visibility_score_0_3",
        "operator_or_annotator",
        "external_test_split",
        "notes",
    ]
    template = pd.DataFrame({"video_id": summary["video_id"].astype(str)})
    existing = None
    if existing_path is not None and existing_path.exists():
        existing = pd.read_csv(existing_path, dtype=str, keep_default_na=False)
        if "video_id" in existing.columns:
            existing["video_id"] = existing["video_id"].astype(str)
            keep = [column for column in columns if column in existing.columns]
            template = template.merge(existing[keep], on="video_id", how="left")
    for column in columns:
        if column not in template.columns:
            template[column] = ""
    return template[columns].fillna("")


def write_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    formatted = table.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: fmt_float(value, 4))
    formatted = formatted.fillna("").astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    output_dir: Path,
    main_results: pd.DataFrame,
    metric_reporting_tiers: pd.DataFrame,
    group_table: pd.DataFrame,
    error_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    ablation_table: pd.DataFrame | None,
    case_table: pd.DataFrame | None,
    submission_readiness: pd.DataFrame | None,
    annotation_progress: pd.DataFrame | None,
    annotation_priority: pd.DataFrame | None,
    metadata_quality_audit: pd.DataFrame | None,
    metadata_quality_field_status: pd.DataFrame | None,
    metadata_split_leakage_audit: pd.DataFrame | None,
    metadata_readiness: pd.DataFrame | None,
    heat_stress_readiness: pd.DataFrame | None,
    signal_consensus: pd.DataFrame | None,
    selective_rr_metrics: pd.DataFrame | None,
    selective_rr_threshold_grid: pd.DataFrame | None,
    algorithmic_quality_tier_metrics: pd.DataFrame | None,
    algorithmic_quality_feature_summary: pd.DataFrame | None,
    algorithmic_quality_threshold_grid: pd.DataFrame | None,
    deployment_decision_curve: pd.DataFrame | None,
    deployment_decision_recommendation: pd.DataFrame | None,
    conformal_rr_metrics: pd.DataFrame | None,
    conformal_rr_recommendation: pd.DataFrame | None,
    post_safe_gate_error_cases: pd.DataFrame | None,
    post_safe_gate_probe_metrics: pd.DataFrame | None,
    external_split_readiness: pd.DataFrame | None,
    external_split_metrics: pd.DataFrame | None,
    external_split_acceptance: pd.DataFrame | None,
    literature_gap_table: pd.DataFrame | None,
    literature_innovation_priority: pd.DataFrame | None,
    literature_q2_roadmap: pd.DataFrame | None,
    literature_source_register: pd.DataFrame | None,
    literature_innovation_crosswalk: pd.DataFrame | None,
    submission_gap_action_plan: pd.DataFrame | None,
    submission_gap_field_checklist: pd.DataFrame | None,
    submission_gap_video_queue: pd.DataFrame | None,
    external_validation_sample_plan: pd.DataFrame | None,
    external_validation_acceptance: pd.DataFrame | None,
    external_validation_collection: pd.DataFrame | None,
    external_validation_quota: pd.DataFrame | None,
    external_validation_error_queue: pd.DataFrame | None,
    external_validation_error_strata: pd.DataFrame | None,
    external_fieldwork_preflight: pd.DataFrame | None,
    external_fieldwork_tier_status: pd.DataFrame | None,
    external_fieldwork_field_coverage: pd.DataFrame | None,
    external_fieldwork_row_issues: pd.DataFrame | None,
    external_validation_execution_dashboard: pd.DataFrame | None,
    external_validation_input_contract: pd.DataFrame | None,
    external_validation_execution_steps: pd.DataFrame | None,
    external_validation_acceptance_dashboard: pd.DataFrame | None,
    pseudo_external_gate_summary: pd.DataFrame | None,
    pseudo_external_gate_metrics: pd.DataFrame | None,
    manuscript_claim_audit: pd.DataFrame | None,
    method_freeze_summary: pd.DataFrame | None,
    method_freeze_parameters: pd.DataFrame | None,
    target_journal_strategy: pd.DataFrame | None,
    target_journal_gate_matrix: pd.DataFrame | None,
    claim_scope_routes: pd.DataFrame | None,
    claim_scope_allowed_claims: pd.DataFrame | None,
) -> Path:
    report = output_dir / "paper_assets_summary.md"
    if metadata_readiness is None or metadata_readiness.empty:
        metadata_section = (
            "Metadata grouped validation readiness has not been generated yet. Run "
            "`scripts/rr_quality_residual_metadata_group_validation.py` after filling "
            "`paper_metadata_template.csv`."
        )
    else:
        metadata_section = write_markdown_table(
            metadata_readiness,
            [
                "group_column",
                "total_summary_videos",
                "nonempty_values",
                "unique_groups",
                "missing_values_for_summary_videos",
                "ready_for_group_validation",
            ],
        )
    if heat_stress_readiness is None or heat_stress_readiness.empty:
        heat_stress_section = (
            "Heat-stress context readiness has not been generated yet. Run "
            "`scripts/build_rr_heat_stress_context.py` after filling environmental metadata."
        )
    else:
        heat_stress_section = write_markdown_table(
            heat_stress_readiness,
            [
                "field",
                "purpose",
                "nonmissing",
                "total_videos",
                "coverage_percent",
                "unique_values",
                "ready",
            ],
        )
    if ablation_table is None or ablation_table.empty:
        ablation_section = (
            "Feature-block ablation has not been generated yet. Run "
            "`scripts/rr_quality_residual_ablation.py` to produce "
            "`paper_repro_quality_residual_ablation_metrics.csv`."
        )
    else:
        ablation_section = write_markdown_table(
            ablation_table,
            [
                "label",
                "retained_feature_count",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "delta_vs_full_rr_r2",
                "paper_use",
            ],
        )
    if case_table is None or case_table.empty:
        case_section = (
            "Representative case figures have not been generated yet. Run "
            "`scripts/rr_quality_residual_case_figures.py` to create Figure 3 assets."
        )
    else:
        case_section = write_markdown_table(
            case_table,
            [
                "case_label",
                "video_id",
                "error_type",
                "truth_count",
                "peaks",
                "applied_adjust",
                "corrected_peaks",
                "corrected_count_error",
            ],
        )
    if submission_readiness is None or submission_readiness.empty:
        readiness_section = (
            "Submission readiness audit has not been generated yet. Run "
            "`scripts/audit_rr_submission_readiness.py` after refreshing paper assets."
        )
    else:
        important = submission_readiness[
            (submission_readiness["status"].isin(["FAIL", "WARN"]))
            | submission_readiness["q2_blocking"].astype(bool)
        ].copy()
        if important.empty:
            readiness_section = "No blocking or warning checks detected."
        else:
            readiness_section = write_markdown_table(
                important,
                [
                    "category",
                    "check",
                    "status",
                    "q2_blocking",
                    "evidence",
                    "recommended_action",
                ],
            )
    if annotation_progress is None or annotation_progress.empty:
        annotation_section = (
            "Metadata annotation progress has not been generated yet. Run "
            "`scripts/build_rr_metadata_annotation_pack.py` to create the annotation sheet."
        )
    else:
        annotation_section = write_markdown_table(
            annotation_progress,
            ["field", "nonempty", "total_videos", "coverage_percent", "unique_values"],
        )
    if annotation_priority is None or annotation_priority.empty:
        annotation_priority_section = (
            "Metadata annotation priority summary has not been generated yet. Run "
            "`scripts/build_rr_metadata_annotation_pack.py` to create priority batches."
        )
    else:
        annotation_priority_section = write_markdown_table(
            annotation_priority,
            [
                "q2_annotation_batch",
                "videos",
                "critical",
                "high",
                "medium",
                "normal",
                "max_q2_annotation_score",
                "mean_q2_annotation_score",
            ],
        )
    dashboard_path = output_dir / "paper_metadata_annotation_dashboard.html"
    if dashboard_path.exists():
        dashboard_section = (
            f"`{dashboard_path}`\n\n"
            "Use this offline dashboard to view priority batches, sample frames, curves, "
            "peak-review figures, and editable Q2 metadata fields."
        )
    else:
        dashboard_section = (
            "Metadata annotation dashboard has not been generated yet. Run "
            "`scripts/build_rr_metadata_annotation_dashboard.py` after creating the annotation sheet."
        )
    if metadata_quality_audit is None or metadata_quality_audit.empty:
        metadata_quality_section = (
            "Metadata quality audit has not been generated yet. Run "
            "`scripts/audit_rr_metadata_quality.py` after creating or filling "
            "`paper_metadata_template.csv`."
        )
    else:
        important_metadata = metadata_quality_audit[
            (metadata_quality_audit["status"].isin(["FAIL", "WARN"]))
            | metadata_quality_audit["q2_blocking"].astype(bool)
        ].copy()
        if important_metadata.empty:
            metadata_quality_section = "No metadata quality warnings or blockers detected."
        else:
            metadata_quality_section = write_markdown_table(
                important_metadata,
                [
                    "category",
                    "check",
                    "status",
                    "q2_blocking",
                    "evidence",
                    "recommended_action",
                ],
            )
    if metadata_quality_field_status is None or metadata_quality_field_status.empty:
        metadata_field_status_section = "No metadata claim field coverage table is available."
    else:
        metadata_field_status_section = write_markdown_table(
            metadata_quality_field_status,
            [
                "field",
                "status",
                "nonempty",
                "total_videos",
                "coverage_percent",
                "claim_unlocked",
            ],
        )
    if metadata_split_leakage_audit is None or metadata_split_leakage_audit.empty:
        metadata_split_leakage_section = "No split leakage audit is available."
    else:
        metadata_split_leakage_section = write_markdown_table(
            metadata_split_leakage_audit,
            ["category", "check", "status", "evidence", "recommended_action"],
        )
    if signal_consensus is None or signal_consensus.empty:
        signal_consensus_section = (
            "Signal-consensus validation has not been generated yet. Run "
            "`scripts/rr_signal_consensus_validation.py` after refreshing residual "
            "validation outputs."
        )
    else:
        signal_consensus_section = write_markdown_table(
            signal_consensus,
            [
                "validation_family",
                "label",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "within_one_count",
                "evaluation_note",
            ],
        )
    if selective_rr_metrics is None or selective_rr_metrics.empty:
        selective_rr_section = (
            "Selective RR reporting has not been generated yet. Run "
            "`scripts/rr_selective_prediction_validation.py` after refreshing signal-consensus outputs."
        )
    else:
        selective_rr_section = write_markdown_table(
            selective_rr_metrics,
            [
                "label",
                "videos",
                "coverage",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "exact_rate",
                "evaluation_note",
            ],
        )
    if selective_rr_threshold_grid is None or selective_rr_threshold_grid.empty:
        selective_rr_grid_section = "No selective RR score-threshold grid is available."
    else:
        selective_rr_grid_section = write_markdown_table(
            selective_rr_threshold_grid,
            ["label", "coverage", "rr_r2", "rr_mae", "exact_count", "exact_rate"],
        )
    if algorithmic_quality_tier_metrics is None or algorithmic_quality_tier_metrics.empty:
        algorithmic_quality_section = (
            "Algorithmic quality stratification has not been generated yet. Run "
            "`scripts/build_rr_algorithmic_quality_context.py` after refreshing "
            "the safe-gate predictions."
        )
    else:
        quality_rows = algorithmic_quality_tier_metrics[
            algorithmic_quality_tier_metrics["method"].astype(str).isin(
                ["signal_aware_safe_gate", "default_pipeline"]
            )
        ].copy()
        algorithmic_quality_section = write_markdown_table(
            quality_rows,
            [
                "method",
                "tier_type",
                "tier",
                "videos",
                "coverage",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "within_one_count",
                "evaluation_note",
            ],
        )
    if (
        algorithmic_quality_feature_summary is None
        or algorithmic_quality_feature_summary.empty
    ):
        algorithmic_quality_feature_section = (
            "No algorithmic quality feature summary is available."
        )
    else:
        feature_columns = [
            column
            for column in [
                "tier_type",
                "tier",
                "videos",
                "algorithmic_review_risk_score_mean",
                "algorithmic_signal_agreement_mean",
                "algorithmic_interval_instability_score_mean",
                "algorithmic_model_ambiguity_score_mean",
                "algorithmic_missingness_score_mean",
            ]
            if column in algorithmic_quality_feature_summary.columns
        ]
        algorithmic_quality_feature_section = write_markdown_table(
            algorithmic_quality_feature_summary,
            feature_columns,
        )
    if (
        algorithmic_quality_threshold_grid is None
        or algorithmic_quality_threshold_grid.empty
    ):
        algorithmic_quality_threshold_section = (
            "No algorithmic quality threshold grid is available."
        )
    else:
        algorithmic_quality_threshold_section = write_markdown_table(
            algorithmic_quality_threshold_grid,
            [
                "tier",
                "videos",
                "coverage",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "exact_rate",
                "evaluation_note",
            ],
        )
    if deployment_decision_curve is None or deployment_decision_curve.empty:
        deployment_decision_curve_section = (
            "Deployment decision curve has not been generated yet. Run "
            "`scripts/build_rr_deployment_decision_curve.py` after algorithmic "
            "quality stratification."
        )
    else:
        deployment_decision_curve_section = write_markdown_table(
            deployment_decision_curve,
            [
                "operating_point",
                "auto_videos",
                "auto_coverage",
                "manual_review_load",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "exact_rate",
                "abs_count_error_ge1",
                "abs_count_error_ge2",
                "review_capture_rate_ge1",
            ],
        )
    if deployment_decision_recommendation is None or deployment_decision_recommendation.empty:
        deployment_decision_recommendation_section = (
            "No deployment operating point recommendation is available."
        )
    else:
        deployment_decision_recommendation_section = write_markdown_table(
            deployment_decision_recommendation,
            [
                "operating_point",
                "auto_videos",
                "auto_coverage",
                "manual_review_load",
                "rr_r2",
                "rr_mae",
                "exact_count",
                "exact_rate",
                "selection_rule",
                "recommendation_boundary",
            ],
        )
    if conformal_rr_metrics is None or conformal_rr_metrics.empty:
        conformal_rr_metrics_section = (
            "Conformal RR uncertainty has not been generated yet. Run "
            "`scripts/build_rr_conformal_uncertainty.py` after paired method "
            "predictions and algorithmic quality scores are available."
        )
    else:
        conformal_focus = conformal_rr_metrics[
            (conformal_rr_metrics["target_coverage"].round(6) == 0.90)
            & conformal_rr_metrics["method"].astype(str).isin(
                [
                    "default_pipeline",
                    "quality_residual_fixed_oof",
                    "signal_consensus_fixed_oof",
                    "signal_aware_safe_fixed_oof",
                ]
            )
        ]
        conformal_rr_metrics_section = write_markdown_table(
            conformal_focus,
            [
                "method_label",
                "interval_variant",
                "coverage",
                "coverage_minus_target",
                "mean_width_bpm",
                "median_width_bpm",
                "low_risk_videos",
                "low_risk_coverage",
                "high_risk_coverage",
                "internal_status",
            ],
        )
    if conformal_rr_recommendation is None or conformal_rr_recommendation.empty:
        conformal_rr_recommendation_section = (
            "No conformal uncertainty recommendation is available."
        )
    else:
        conformal_rr_recommendation_section = write_markdown_table(
            conformal_rr_recommendation,
            [
                "method_label",
                "target_coverage",
                "interval_variant",
                "coverage",
                "mean_width_bpm",
                "median_width_bpm",
                "low_risk_videos",
                "low_risk_coverage",
                "low_risk_mean_width_bpm",
                "internal_status",
                "selection_rule",
            ],
        )
    if post_safe_gate_error_cases is None or post_safe_gate_error_cases.empty:
        post_safe_gate_error_section = (
            "Post safe-gate error diagnostics have not been generated yet. Run "
            "`scripts/build_rr_post_safe_gate_error_diagnostics.py` after "
            "algorithmic quality scoring."
        )
    else:
        post_safe_gate_error_section = write_markdown_table(
            post_safe_gate_error_cases,
            [
                "video_id",
                "video_prefix_group",
                "truth_count",
                "post_safe_gate_baseline_count",
                "post_safe_gate_target_adjust",
                "post_safe_gate_error_direction",
                "algorithmic_review_risk_score",
                "algorithmic_risk_tier_fixed",
                "signal_aware_safe_guard_reason",
            ],
        )
    if post_safe_gate_probe_metrics is None or post_safe_gate_probe_metrics.empty:
        post_safe_gate_probe_section = (
            "No second-stage post safe-gate probe metrics are available."
        )
    else:
        post_safe_gate_probe_focus = post_safe_gate_probe_metrics.sort_values(
            [
                "passes_internal_stability_guard",
                "passes_metric_guard",
                "exact_count",
                "abs_count_error_ge2",
                "rr_mae",
            ],
            ascending=[False, False, False, True, True],
        ).head(12)
        post_safe_gate_probe_section = write_markdown_table(
            post_safe_gate_probe_focus,
            [
                "rank_overall",
                "probe_name",
                "cv_scheme",
                "policy_label",
                "exact_count",
                "delta_exact_vs_safe_gate",
                "rr_r2",
                "rr_mae",
                "abs_count_error_ge2",
                "applied_videos",
                "passes_metric_guard",
                "passes_internal_stability_guard",
                "paper_use",
            ],
        )
    if external_split_readiness is None or external_split_readiness.empty:
        external_split_section = (
            "External split validation has not been generated yet. Run "
            "`scripts/rr_external_split_validation.py` after filling `external_test_split`."
        )
    else:
        external_split_section = write_markdown_table(
            external_split_readiness,
            [
                "check",
                "status",
                "videos",
                "evidence",
                "recommended_action",
                "external_ready",
                "method_freeze_id",
                "freeze_status",
            ],
        )
    if external_split_metrics is None or external_split_metrics.empty:
        external_split_metrics_section = (
            "No external split metrics are available because the frozen split is not ready."
        )
    else:
        external_split_metrics_section = write_markdown_table(
            external_split_metrics,
            [
                "split_scope",
                "split_label",
                "method",
                "videos",
                "rr_r2",
                "rr_mae",
                "rr_rmse",
                "exact_count",
                "paper_use",
                "method_freeze_id",
            ],
        )
    if external_split_acceptance is None or external_split_acceptance.empty:
        external_split_acceptance_section = (
            "No external split acceptance gates are available yet. Run "
            "`scripts/rr_external_split_validation.py`."
        )
    else:
        external_split_acceptance_section = write_markdown_table(
            external_split_acceptance,
            [
                "gate",
                "status",
                "method",
                "split_scope",
                "metric",
                "value",
                "threshold",
                "claim_unlocked",
                "method_freeze_id",
                "freeze_status",
            ],
        )
    if external_validation_sample_plan is None or external_validation_sample_plan.empty:
        external_validation_plan_section = (
            "External-validation sample plan has not been generated yet. Run "
            "`scripts/build_rr_external_validation_sample_plan.py` after refreshing "
            "bootstrap CIs and paper assets."
        )
    else:
        external_validation_plan_section = write_markdown_table(
            external_validation_sample_plan,
            [
                "planning_use",
                "metric",
                "current_internal_estimate",
                "target_precision_or_gate",
                "estimated_min_external_videos",
                "paper_interpretation",
            ],
        )
    if external_validation_acceptance is None or external_validation_acceptance.empty:
        external_validation_acceptance_section = "No external-validation acceptance criteria are available."
    else:
        external_validation_acceptance_section = write_markdown_table(
            external_validation_acceptance,
            [
                "claim_type",
                "method_scope",
                "metric",
                "acceptance_rule",
                "current_internal_reference",
                "manuscript_use_if_passed",
            ],
        )
    if external_validation_collection is None or external_validation_collection.empty:
        external_validation_collection_section = "No external-validation collection tiers are available."
    else:
        external_validation_collection_section = write_markdown_table(
            external_validation_collection,
            [
                "collection_tier",
                "recommended_videos",
                "required_metadata",
                "claim_allowed",
            ],
        )
    if external_validation_quota is None or external_validation_quota.empty:
        external_validation_quota_section = "No external-validation quota table is available."
    else:
        external_validation_quota_section = write_markdown_table(
            external_validation_quota,
            [
                "collection_tier",
                "quota_dimension",
                "minimum_target",
                "preferred_target",
                "q2_blocking_if_missing",
                "rationale",
            ],
        )
    if external_validation_error_queue is None or external_validation_error_queue.empty:
        external_validation_error_queue_section = (
            "Error-driven external validation queue has not been generated yet. Run "
            "`scripts/build_rr_error_driven_external_validation_queue.py` after "
            "post safe-gate diagnostics and the external validation sample plan."
        )
    else:
        external_validation_error_queue_section = write_markdown_table(
            external_validation_error_queue.head(20),
            [
                "priority_rank",
                "planning_stratum",
                "internal_prototype_video_id",
                "video_prefix_group",
                "internal_safe_gate_count_error",
                "error_direction",
                "algorithmic_review_risk_score",
                "algorithmic_risk_tier_fixed",
                "minimum_external_matches",
                "preferred_external_matches",
                "paper_use",
            ],
        )
    if external_validation_error_strata is None or external_validation_error_strata.empty:
        external_validation_error_strata_section = (
            "No error-driven external validation strata table is available."
        )
    else:
        external_validation_error_strata_section = write_markdown_table(
            external_validation_error_strata,
            [
                "quota_type",
                "stratum",
                "internal_videos",
                "internal_safe_gate_errors",
                "minimum_external_videos",
                "preferred_external_videos",
                "why_it_matters",
                "q2_claim_unlocked_if_covered",
            ],
        )
    if external_fieldwork_tier_status is None or external_fieldwork_tier_status.empty:
        external_fieldwork_tier_section = (
            "External fieldwork preflight has not been generated yet. Run "
            "`scripts/preflight_rr_external_fieldwork.py` after generating or filling "
            "the fieldwork template."
        )
    else:
        external_fieldwork_tier_section = write_markdown_table(
            external_fieldwork_tier_status,
            ["tier", "check", "status", "evidence", "recommended_action"],
        )
    if external_fieldwork_preflight is None or external_fieldwork_preflight.empty:
        external_fieldwork_preflight_section = "No external fieldwork preflight checks are available."
    else:
        important_fieldwork = external_fieldwork_preflight[
            (external_fieldwork_preflight["status"].astype(str).isin(["FAIL", "WARN"]))
            | external_fieldwork_preflight["q2_blocking"].astype(bool)
        ].copy()
        external_fieldwork_preflight_section = write_markdown_table(
            important_fieldwork,
            ["category", "check", "status", "q2_blocking", "evidence", "recommended_action"],
        )
    if external_fieldwork_field_coverage is None or external_fieldwork_field_coverage.empty:
        external_fieldwork_coverage_section = "No external fieldwork field-coverage table is available."
    else:
        external_fieldwork_coverage_section = write_markdown_table(
            external_fieldwork_field_coverage,
            ["field", "present", "nonempty", "included_rows", "coverage_percent", "q2_required"],
        )
    if external_fieldwork_row_issues is None or external_fieldwork_row_issues.empty:
        external_fieldwork_issue_section = "No row-level external fieldwork issues detected."
    else:
        external_fieldwork_issue_section = write_markdown_table(
            external_fieldwork_row_issues.head(20),
            ["row_index", "external_video_id", "issue_type", "field", "severity", "evidence"],
        )
    if (
        external_validation_execution_dashboard is None
        or external_validation_execution_dashboard.empty
    ):
        external_validation_execution_dashboard_section = (
            "External validation execution plan has not been generated yet. Run "
            "`scripts/build_rr_external_validation_execution_plan.py`."
        )
    else:
        external_validation_execution_dashboard_section = write_markdown_table(
            external_validation_execution_dashboard,
            [
                "gate_order",
                "gate",
                "status",
                "evidence",
                "next_action",
                "method_freeze_id",
            ],
        )
    if (
        external_validation_input_contract is None
        or external_validation_input_contract.empty
    ):
        external_validation_input_contract_section = (
            "No external validation input contract is available."
        )
    else:
        external_validation_input_contract_section = write_markdown_table(
            external_validation_input_contract,
            [
                "input_artifact",
                "required",
                "expected_location_or_format",
                "owner_action",
                "checked_by",
            ],
        )
    if (
        external_validation_execution_steps is None
        or external_validation_execution_steps.empty
    ):
        external_validation_execution_steps_section = (
            "No external validation execution command sequence is available."
        )
    else:
        external_validation_execution_steps_section = write_markdown_table(
            external_validation_execution_steps,
            [
                "step_order",
                "phase",
                "step",
                "can_run_now",
                "command_template",
                "success_output",
                "paper_use",
            ],
        )
    if (
        external_validation_acceptance_dashboard is None
        or external_validation_acceptance_dashboard.empty
    ):
        external_validation_acceptance_dashboard_section = (
            "No external validation acceptance dashboard is available."
        )
    else:
        external_validation_acceptance_dashboard_section = write_markdown_table(
            external_validation_acceptance_dashboard,
            [
                "acceptance_family",
                "metric_or_gate",
                "acceptance_rule",
                "current_status",
                "claim_if_passed",
            ],
        )
    if method_freeze_summary is None or method_freeze_summary.empty:
        method_freeze_section = (
            "External method freeze manifest has not been generated yet. Run "
            "`scripts/freeze_rr_external_method.py` before external validation."
        )
    else:
        method_freeze_section = write_markdown_table(
            method_freeze_summary,
            [
                "method_freeze_id",
                "freeze_status",
                "primary_method",
                "secondary_method",
                "required_files_present",
                "required_files_missing",
                "hashed_files",
            ],
        )
    if method_freeze_parameters is None or method_freeze_parameters.empty:
        method_freeze_parameters_section = "No frozen method parameter table is available."
    else:
        method_freeze_parameters_section = write_markdown_table(
            method_freeze_parameters,
            ["method_scope", "parameter", "value", "external_use"],
        )
    if target_journal_strategy is None or target_journal_strategy.empty:
        target_journal_strategy_section = (
            "Target journal strategy has not been generated yet. Run "
            "`scripts/build_rr_target_journal_strategy.py` after readiness and "
            "fieldwork-preflight outputs are refreshed."
        )
    else:
        target_journal_columns = [
            "target_journal",
            "route",
            "journal_strategy_status",
            "lead_claim_if_ready",
            "missing_must_have_gates",
            "recommended_next_action",
            "current_quartile_note",
        ]
        if "claim_exclusions" in target_journal_strategy.columns:
            target_journal_columns.insert(5, "claim_exclusions")
        target_journal_strategy_section = write_markdown_table(
            target_journal_strategy,
            target_journal_columns,
        )
    if target_journal_gate_matrix is None or target_journal_gate_matrix.empty:
        target_journal_gate_section = "No target-journal evidence gate matrix is available."
    else:
        target_journal_gate_section = write_markdown_table(
            target_journal_gate_matrix,
            ["gate", "description", "status", "evidence"],
        )
    if claim_scope_routes is None or claim_scope_routes.empty:
        claim_scope_routes_section = (
            "Claim-scope strategy has not been generated yet. Run "
            "`scripts/build_rr_claim_scope_strategy.py`."
        )
    else:
        claim_scope_routes_section = write_markdown_table(
            claim_scope_routes,
            [
                "route",
                "current_status",
                "required_next_gate",
                "main_claim_allowed_now",
                "main_claim_after_next_gate",
                "excluded_claims",
                "paper_use",
            ],
        )
    if claim_scope_allowed_claims is None or claim_scope_allowed_claims.empty:
        claim_scope_allowed_claims_section = "No claim-scope wording table is available."
    else:
        claim_scope_allowed_claims_section = write_markdown_table(
            claim_scope_allowed_claims,
            ["claim_family", "status", "recommended_wording", "must_not_say"],
        )
    if pseudo_external_gate_summary is None or pseudo_external_gate_summary.empty:
        pseudo_external_gate_section = (
            "Pseudo-external gate stress test has not been generated yet. Run "
            "`scripts/build_rr_pseudo_external_gate_stress_test.py`."
        )
    else:
        pseudo_external_gate_section = write_markdown_table(
            pseudo_external_gate_summary,
            [
                "pseudo_external_group",
                "videos",
                "baseline_rr_r2",
                "best_gate_method",
                "best_gate_pass_count",
                "best_gate_all_pass",
                "best_gate_rr_r2",
                "best_gate_mae",
                "best_gate_rmse",
                "risk_notes",
                "paper_use",
            ],
        )
    if manuscript_claim_audit is None or manuscript_claim_audit.empty:
        manuscript_claim_audit_section = (
            "Manuscript claim audit has not been generated yet. Run "
            "`scripts/audit_rr_manuscript_claims.py` after updating "
            "`docs/thermal_rr_manuscript_draft.md`."
        )
    else:
        manuscript_claim_audit_section = write_markdown_table(
            manuscript_claim_audit,
            [
                "check",
                "status",
                "risk",
                "evidence",
                "recommended_action",
            ],
        )
    if literature_gap_table is None or literature_gap_table.empty:
        literature_gap_section = (
            "Literature gap table has not been generated yet. Run "
            "`scripts/build_rr_literature_gap_table.py` to create "
            "`paper_literature_gap_table.csv` and `paper_literature_gap_table.md`."
        )
    else:
        literature_gap_section = write_markdown_table(
            literature_gap_table,
            [
                "source_id",
                "year",
                "venue",
                "gap_for_current_work",
                "repo_innovation_link",
                "paper_role",
            ],
        )
    if literature_innovation_priority is None or literature_innovation_priority.empty:
        literature_innovation_section = (
            "Literature-grounded innovation priority table has not been generated yet. "
            "Run `scripts/build_rr_literature_gap_table.py`."
        )
    else:
        literature_innovation_section = write_markdown_table(
            literature_innovation_priority,
            [
                "priority",
                "innovation_point",
                "paper_use",
                "claim_strength_now",
                "current_repo_evidence",
                "remaining_gate",
                "next_experiment",
            ],
        )
    if literature_q2_roadmap is None or literature_q2_roadmap.empty:
        literature_q2_roadmap_section = (
            "Q2+ innovation roadmap has not been generated yet. Run "
            "`scripts/build_rr_q2_plus_innovation_roadmap.py`."
        )
    else:
        literature_q2_roadmap_section = write_markdown_table(
            literature_q2_roadmap,
            [
                "priority",
                "roadmap_item",
                "paper_role",
                "claim_now",
                "current_evidence",
                "q2_gate",
                "next_action",
            ],
        )
    if literature_source_register is None or literature_source_register.empty:
        literature_source_register_section = (
            "Literature source register has not been generated yet. Run "
            "`scripts/build_rr_literature_innovation_crosswalk.py`."
        )
    else:
        literature_source_register_section = write_markdown_table(
            literature_source_register,
            [
                "source_id",
                "citation_short",
                "year",
                "venue",
                "doi_or_url",
                "relevance",
            ],
        )
    if literature_innovation_crosswalk is None or literature_innovation_crosswalk.empty:
        literature_crosswalk_section = (
            "Literature-innovation claim crosswalk has not been generated yet. Run "
            "`scripts/build_rr_literature_innovation_crosswalk.py`."
        )
    else:
        literature_crosswalk_section = write_markdown_table(
            literature_innovation_crosswalk,
            [
                "manuscript_claim",
                "literature_basis",
                "implemented_response",
                "current_project_evidence",
                "safe_wording",
                "do_not_claim_yet",
            ],
        )
    if submission_gap_action_plan is None or submission_gap_action_plan.empty:
        submission_gap_section = (
            "Q2+ submission gap action pack has not been generated yet. Run "
            "`scripts/build_rr_submission_gap_action_pack.py` after refreshing "
            "submission readiness and metadata annotation outputs."
        )
    else:
        submission_gap_section = write_markdown_table(
            submission_gap_action_plan,
            [
                "priority",
                "category",
                "check",
                "status",
                "required_fields",
                "deliverable",
                "claim_unlocked",
            ],
        )
    if submission_gap_field_checklist is None or submission_gap_field_checklist.empty:
        submission_gap_field_section = "No field-level submission gap checklist is available."
    else:
        submission_gap_field_section = write_markdown_table(
            submission_gap_field_checklist,
            [
                "fill_priority",
                "field",
                "current_nonempty",
                "total_videos",
                "coverage_percent",
                "downstream_claim_unlocked",
            ],
        )
    if submission_gap_video_queue is None or submission_gap_video_queue.empty:
        submission_gap_video_section = "No video-level submission gap queue is available."
    else:
        submission_gap_video_section = write_markdown_table(
            submission_gap_video_queue.head(20),
            [
                "video_id",
                "annotation_priority",
                "q2_annotation_score",
                "q2_annotation_batch",
                "error_type",
                "selective_action",
            ],
        )
    text = f"""# Thermal RR Paper Assets Summary

Generated from current CSV outputs. Use this file as a manuscript/results checklist, not as a replacement for external validation.

## Main Results Table

{write_markdown_table(main_results, ["method", "validation_setting", "videos", "rr_r2", "rr_mae_bpm", "rr_rmse_bpm", "exact_count", "paper_use"])}

## Metric Reporting Tiers

{write_markdown_table(metric_reporting_tiers, ["method", "rr_r2", "exact_count", "evidence_tier", "primary_performance_allowed", "recommended_location", "claim_boundary"])}

## Prefix-Group Holdout Table

{write_markdown_table(group_table, ["heldout_prefix_group", "videos", "baseline_rr_r2", "selected_threshold_rr_r2", "delta_rr_r2", "baseline_exact", "selected_threshold_exact", "interpretation"])}

## Residual Error Taxonomy

{write_markdown_table(error_table, ["error_type", "videos", "percentage_of_errors", "paper_interpretation"])}

## Bootstrap Improvement CI

{write_markdown_table(bootstrap_table, ["display_metric", "estimate_with_ci", "bootstrap_samples", "note"])}

## Top Residual-Corrector Features

{write_markdown_table(feature_table, ["feature", "importance", "paper_interpretation"])}

## Feature-Block Ablation

{ablation_section}

## Representative Curve Cases

{case_section}

## Submission Readiness Audit

{readiness_section}

## Metadata Annotation Progress

{annotation_section}

## Metadata Annotation Priority Batches

{annotation_priority_section}

## Metadata Annotation Dashboard

{dashboard_section}

## Metadata Quality And Split Leakage Audit

{metadata_quality_section}

### Metadata Claim Field Coverage

{metadata_field_status_section}

### Split Leakage And Balance Screens

{metadata_split_leakage_section}

## Metadata Grouped-Validation Readiness

{metadata_section}

## Heat-Stress Context Readiness

{heat_stress_section}

## Signal-Consensus Experiment

{signal_consensus_section}

## Selective RR Reporting and Review Triage

{selective_rr_section}

### Selective Score Threshold Sensitivity

{selective_rr_grid_section}

## Algorithmic Quality Stratification

{algorithmic_quality_section}

### Algorithmic Quality Feature Summary

{algorithmic_quality_feature_section}

### Algorithmic Risk Threshold Sensitivity

{algorithmic_quality_threshold_section}

## Deployment Decision Curve

{deployment_decision_curve_section}

### Recommended Internal Auto-Report Operating Point

{deployment_decision_recommendation_section}

## Internal Conformal RR Uncertainty

{conformal_rr_metrics_section}

### Recommended Internal RR Prediction Interval

{conformal_rr_recommendation_section}

## Post Safe-Gate Error Diagnostic

{post_safe_gate_error_section}

### Second-Stage Probe Guardrail

{post_safe_gate_probe_section}

## External Split Validation

{external_split_section}

### External Split Metrics

{external_split_metrics_section}

### External Split Acceptance Gates

{external_split_acceptance_section}

## External Validation Sample Plan

{external_validation_plan_section}

### External Validation Acceptance Criteria

{external_validation_acceptance_section}

### External Validation Collection Tiers

{external_validation_collection_section}

### External Validation Stratified Quotas

{external_validation_quota_section}

### Error-Driven External Validation Queue

{external_validation_error_queue_section}

### Error-Driven External Validation Strata

{external_validation_error_strata_section}

### External Fieldwork Preflight

{external_fieldwork_tier_section}

#### Fieldwork Blocking Checks

{external_fieldwork_preflight_section}

#### Fieldwork Required Field Coverage

{external_fieldwork_coverage_section}

#### Fieldwork Row-Level Issues

{external_fieldwork_issue_section}

### External Validation Execution Plan

{external_validation_execution_dashboard_section}

#### External Validation Input Contract

{external_validation_input_contract_section}

#### External Validation Command Sequence

{external_validation_execution_steps_section}

#### External Validation Acceptance Dashboard

{external_validation_acceptance_dashboard_section}

## Frozen External Method Manifest

{method_freeze_section}

### Frozen Method Parameters

{method_freeze_parameters_section}

## Target Journal Strategy

{target_journal_strategy_section}

### Target Journal Evidence Gates

{target_journal_gate_section}

### Claim-Scope Strategy For Missing Environment And Quality Metadata

{claim_scope_routes_section}

#### Allowed And Forbidden Claim Wording

{claim_scope_allowed_claims_section}

## Pseudo-External Gate Stress Test

{pseudo_external_gate_section}

## Manuscript Claim Audit

{manuscript_claim_audit_section}

## Literature-Grounded Innovation Gap

{literature_gap_section}

## Literature-Grounded Innovation Priorities

{literature_innovation_section}

## Q2+ Innovation Roadmap

{literature_q2_roadmap_section}

## Literature-Innovation Claim Crosswalk

### Source Register

{literature_source_register_section}

### Claim Wording Guardrails

{literature_crosswalk_section}

## Q2+ Submission Gap Action Pack

{submission_gap_section}

### Required Metadata Fields

{submission_gap_field_section}

### First Videos To Annotate

{submission_gap_video_section}

## Manuscript-Use Notes

- The default pipeline and fixed-threshold/nested residual-correction results can be reported as internal validation results.
- The signal-consensus supplement is a candidate precision extension; report it as an additional internal experiment unless external and metadata-group validation also support it.
- The signal-aware residual correction is a higher-precision internal candidate only; its prefix-group result must be reported as a caution before making generalization claims.
- The selective RR reporting results should be described as an uncertainty-aware review-prioritization analysis, not as a replacement for external validation.
- The post safe-gate second-stage probes are error diagnostics only; do not promote them to primary performance unless a frozen candidate passes true external validation.
- External split metrics should be reported only after `external split validation ready` is PASS.
- The error-driven external validation queue defines collection priorities only; it is not an external result and must be prospectively populated with new videos.
- The external fieldwork preflight is a worksheet readiness check, not a prediction-performance result.
- The external validation execution plan is a checklist and frozen-command handoff; it is not external performance evidence until independent rows are scored.
- The target journal strategy is a submission planning matrix; verify current JCR/SJR quartile and author guidelines before selecting a journal.
- The claim-scope strategy is the active guardrail when environment metadata and manual quality scores are unavailable; it narrows claims instead of fabricating those fields.
- The best grid threshold result should be reported only as threshold sensitivity because the threshold is selected on the current out-of-fold predictions.
- The truth-calibrated result is an upper-bound analysis only and must not be described as main method performance.
- Prefix groups are heuristic video-ID domains, not verified cow IDs. The current `cow_id` field is a numeric video label; replace it with true animal IDs, per-video dates, and camera/scene groups before claiming animal-level generalization.
- The metadata template in `paper_metadata_template.csv` is required to strengthen the work from RR detection toward heat-stress or health-monitoring interpretation.
- Follow `docs/thermal_rr_external_validation_protocol.md` before claiming Q2-or-higher submission readiness.
- Use `docs/thermal_rr_literature_innovation_matrix.md` to keep the innovation claim grounded in local and current PLF literature.
- Run `scripts/build_rr_heat_stress_context.py` after adding temperature/humidity or THI/ATHI metadata to generate biological interpretation tables.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(input_root, args.output_prefix, args.corrected_prefix)

    main_results = build_main_results_table(data)
    metric_reporting_tiers = build_metric_reporting_tiers(main_results)
    group_table = build_group_table(data["group_metrics_by_prefix"])
    total_errors = int(data["error_taxonomy_summary"]["videos"].sum())
    error_table = build_error_table(data["error_taxonomy_summary"], total_errors)
    bootstrap_table = build_bootstrap_table(data["bootstrap_ci"])
    feature_table = build_feature_table(data["feature_importance"])
    ablation_table = (
        build_ablation_table(data["ablation_metrics"])
        if "ablation_metrics" in data
        else None
    )
    case_table = data["case_examples"] if "case_examples" in data else None
    submission_readiness = (
        data["submission_readiness"] if "submission_readiness" in data else None
    )
    signal_consensus = (
        data["signal_consensus_comparison"]
        if "signal_consensus_comparison" in data
        else None
    )
    signal_aware_residual_metrics = (
        data["signal_aware_residual_metrics"]
        if "signal_aware_residual_metrics" in data
        else None
    )
    signal_aware_residual_threshold_grid = (
        data["signal_aware_residual_threshold_grid"]
        if "signal_aware_residual_threshold_grid" in data
        else None
    )
    signal_aware_residual_bootstrap_ci = (
        data["signal_aware_residual_bootstrap_ci"]
        if "signal_aware_residual_bootstrap_ci" in data
        else None
    )
    signal_aware_safe_policy_metrics = (
        data["signal_aware_safe_policy_metrics"]
        if "signal_aware_safe_policy_metrics" in data
        else None
    )
    signal_aware_safe_policy_bootstrap_ci = (
        data["signal_aware_safe_policy_bootstrap_ci"]
        if "signal_aware_safe_policy_bootstrap_ci" in data
        else None
    )
    signal_aware_safe_policy_cases = (
        data["signal_aware_safe_policy_cases"]
        if "signal_aware_safe_policy_cases" in data
        else None
    )
    selective_rr_metrics = (
        data["selective_rr_metrics"] if "selective_rr_metrics" in data else None
    )
    selective_rr_threshold_grid = (
        data["selective_rr_threshold_grid"]
        if "selective_rr_threshold_grid" in data
        else None
    )
    algorithmic_quality_tier_metrics = (
        data["algorithmic_quality_tier_metrics"]
        if "algorithmic_quality_tier_metrics" in data
        else None
    )
    algorithmic_quality_feature_summary = (
        data["algorithmic_quality_feature_summary"]
        if "algorithmic_quality_feature_summary" in data
        else None
    )
    algorithmic_quality_threshold_grid = (
        data["algorithmic_quality_threshold_grid"]
        if "algorithmic_quality_threshold_grid" in data
        else None
    )
    deployment_decision_curve = (
        data["deployment_decision_curve"] if "deployment_decision_curve" in data else None
    )
    deployment_decision_recommendation = (
        data["deployment_decision_recommendation"]
        if "deployment_decision_recommendation" in data
        else None
    )
    conformal_rr_metrics = (
        data["conformal_rr_metrics"] if "conformal_rr_metrics" in data else None
    )
    conformal_rr_recommendation = (
        data["conformal_rr_recommendation"]
        if "conformal_rr_recommendation" in data
        else None
    )
    post_safe_gate_error_cases = (
        data["post_safe_gate_error_cases"]
        if "post_safe_gate_error_cases" in data
        else None
    )
    post_safe_gate_probe_metrics = (
        data["post_safe_gate_probe_metrics"]
        if "post_safe_gate_probe_metrics" in data
        else None
    )
    external_split_readiness = (
        data["external_split_readiness"] if "external_split_readiness" in data else None
    )
    external_split_metrics = (
        data["external_split_metrics"] if "external_split_metrics" in data else None
    )
    external_split_acceptance = (
        data["external_split_acceptance"] if "external_split_acceptance" in data else None
    )
    literature_gap_path = output_dir / "paper_literature_gap_table.csv"
    literature_gap_table = (
        pd.read_csv(literature_gap_path) if literature_gap_path.exists() else None
    )
    literature_innovation_path = output_dir / "paper_literature_innovation_priority.csv"
    literature_innovation_priority = (
        pd.read_csv(literature_innovation_path)
        if literature_innovation_path.exists()
        else None
    )
    literature_q2_roadmap_path = output_dir / "paper_literature_q2_plus_innovation_roadmap.csv"
    literature_q2_roadmap = (
        pd.read_csv(literature_q2_roadmap_path)
        if literature_q2_roadmap_path.exists()
        else None
    )
    literature_source_register_path = output_dir / "paper_literature_source_register.csv"
    literature_source_register = (
        pd.read_csv(literature_source_register_path)
        if literature_source_register_path.exists()
        else None
    )
    literature_innovation_crosswalk_path = output_dir / "paper_literature_innovation_crosswalk.csv"
    literature_innovation_crosswalk = (
        pd.read_csv(literature_innovation_crosswalk_path)
        if literature_innovation_crosswalk_path.exists()
        else None
    )
    submission_gap_action_path = output_dir / "paper_submission_gap_action_plan.csv"
    submission_gap_action_plan = (
        pd.read_csv(submission_gap_action_path)
        if submission_gap_action_path.exists()
        else None
    )
    submission_gap_field_path = output_dir / "paper_submission_gap_field_checklist.csv"
    submission_gap_field_checklist = (
        pd.read_csv(submission_gap_field_path)
        if submission_gap_field_path.exists()
        else None
    )
    submission_gap_video_path = output_dir / "paper_submission_gap_video_queue.csv"
    submission_gap_video_queue = (
        pd.read_csv(submission_gap_video_path)
        if submission_gap_video_path.exists()
        else None
    )
    external_validation_sample_path = output_dir / "paper_external_validation_sample_plan.csv"
    external_validation_sample_plan = (
        pd.read_csv(external_validation_sample_path)
        if external_validation_sample_path.exists()
        else None
    )
    external_validation_acceptance_path = (
        output_dir / "paper_external_validation_acceptance_criteria.csv"
    )
    external_validation_acceptance = (
        pd.read_csv(external_validation_acceptance_path)
        if external_validation_acceptance_path.exists()
        else None
    )
    external_validation_collection_path = (
        output_dir / "paper_external_validation_collection_tiers.csv"
    )
    external_validation_collection = (
        pd.read_csv(external_validation_collection_path)
        if external_validation_collection_path.exists()
        else None
    )
    external_validation_quota_path = output_dir / "paper_external_validation_quota_table.csv"
    external_validation_quota = (
        pd.read_csv(external_validation_quota_path)
        if external_validation_quota_path.exists()
        else None
    )
    external_validation_error_queue_path = (
        output_dir / "paper_external_validation_error_driven_queue.csv"
    )
    external_validation_error_queue = (
        pd.read_csv(external_validation_error_queue_path)
        if external_validation_error_queue_path.exists()
        else None
    )
    external_validation_error_strata_path = (
        output_dir / "paper_external_validation_error_driven_strata.csv"
    )
    external_validation_error_strata = (
        pd.read_csv(external_validation_error_strata_path)
        if external_validation_error_strata_path.exists()
        else None
    )
    external_fieldwork_preflight_path = (
        output_dir / "paper_external_validation_fieldwork_preflight.csv"
    )
    external_fieldwork_preflight = (
        pd.read_csv(external_fieldwork_preflight_path)
        if external_fieldwork_preflight_path.exists()
        else None
    )
    external_fieldwork_tier_status_path = (
        output_dir / "paper_external_validation_fieldwork_tier_status.csv"
    )
    external_fieldwork_tier_status = (
        pd.read_csv(external_fieldwork_tier_status_path)
        if external_fieldwork_tier_status_path.exists()
        else None
    )
    external_fieldwork_field_coverage_path = (
        output_dir / "paper_external_validation_fieldwork_field_coverage.csv"
    )
    external_fieldwork_field_coverage = (
        pd.read_csv(external_fieldwork_field_coverage_path)
        if external_fieldwork_field_coverage_path.exists()
        else None
    )
    external_fieldwork_row_issues_path = (
        output_dir / "paper_external_validation_fieldwork_row_issues.csv"
    )
    external_fieldwork_row_issues = (
        pd.read_csv(external_fieldwork_row_issues_path)
        if external_fieldwork_row_issues_path.exists()
        else None
    )
    external_validation_execution_dashboard_path = (
        output_dir / "paper_external_validation_execution_dashboard.csv"
    )
    external_validation_execution_dashboard = (
        pd.read_csv(external_validation_execution_dashboard_path)
        if external_validation_execution_dashboard_path.exists()
        else None
    )
    external_validation_input_contract_path = (
        output_dir / "paper_external_validation_input_contract.csv"
    )
    external_validation_input_contract = (
        pd.read_csv(external_validation_input_contract_path)
        if external_validation_input_contract_path.exists()
        else None
    )
    external_validation_execution_steps_path = (
        output_dir / "paper_external_validation_execution_steps.csv"
    )
    external_validation_execution_steps = (
        pd.read_csv(external_validation_execution_steps_path)
        if external_validation_execution_steps_path.exists()
        else None
    )
    external_validation_acceptance_dashboard_path = (
        output_dir / "paper_external_validation_acceptance_dashboard.csv"
    )
    external_validation_acceptance_dashboard = (
        pd.read_csv(external_validation_acceptance_dashboard_path)
        if external_validation_acceptance_dashboard_path.exists()
        else None
    )
    pseudo_external_gate_summary_path = output_dir / "paper_pseudo_external_gate_stress_summary.csv"
    pseudo_external_gate_summary = (
        pd.read_csv(pseudo_external_gate_summary_path)
        if pseudo_external_gate_summary_path.exists()
        else None
    )
    pseudo_external_gate_metrics_path = output_dir / "paper_pseudo_external_gate_stress_metrics.csv"
    pseudo_external_gate_metrics = (
        pd.read_csv(pseudo_external_gate_metrics_path)
        if pseudo_external_gate_metrics_path.exists()
        else None
    )
    manuscript_claim_audit_path = output_dir / "paper_manuscript_claim_audit.csv"
    manuscript_claim_audit = (
        pd.read_csv(manuscript_claim_audit_path)
        if manuscript_claim_audit_path.exists()
        else None
    )
    method_freeze_summary_path = output_dir / "paper_method_freeze_summary.csv"
    method_freeze_summary = (
        pd.read_csv(method_freeze_summary_path)
        if method_freeze_summary_path.exists()
        else None
    )
    method_freeze_parameters_path = output_dir / "paper_method_freeze_parameters.csv"
    method_freeze_parameters = (
        pd.read_csv(method_freeze_parameters_path)
        if method_freeze_parameters_path.exists()
        else None
    )
    target_journal_strategy_path = output_dir / "paper_target_journal_strategy.csv"
    target_journal_strategy = (
        pd.read_csv(target_journal_strategy_path)
        if target_journal_strategy_path.exists()
        else None
    )
    target_journal_gate_path = output_dir / "paper_target_journal_gate_matrix.csv"
    target_journal_gate_matrix = (
        pd.read_csv(target_journal_gate_path)
        if target_journal_gate_path.exists()
        else None
    )
    claim_scope_routes_path = output_dir / "paper_claim_scope_routes.csv"
    claim_scope_routes = (
        pd.read_csv(claim_scope_routes_path)
        if claim_scope_routes_path.exists()
        else None
    )
    claim_scope_allowed_claims_path = output_dir / "paper_claim_scope_allowed_claims.csv"
    claim_scope_allowed_claims = (
        pd.read_csv(claim_scope_allowed_claims_path)
        if claim_scope_allowed_claims_path.exists()
        else None
    )
    metadata_template = build_metadata_template(
        data["summary"], output_dir / "paper_metadata_template.csv"
    )
    readiness_path = input_root / f"{args.corrected_prefix}_metadata_group_readiness.csv"
    metadata_readiness = pd.read_csv(readiness_path) if readiness_path.exists() else None
    heat_readiness_path = output_dir / "paper_heat_stress_readiness.csv"
    heat_stress_readiness = (
        pd.read_csv(heat_readiness_path) if heat_readiness_path.exists() else None
    )
    annotation_progress_path = output_dir / "paper_metadata_annotation_progress.csv"
    annotation_progress = (
        pd.read_csv(annotation_progress_path) if annotation_progress_path.exists() else None
    )
    annotation_priority_path = output_dir / "paper_metadata_annotation_priority_summary.csv"
    annotation_priority = (
        pd.read_csv(annotation_priority_path) if annotation_priority_path.exists() else None
    )
    metadata_quality_audit_path = output_dir / "paper_metadata_quality_audit.csv"
    metadata_quality_audit = (
        pd.read_csv(metadata_quality_audit_path)
        if metadata_quality_audit_path.exists()
        else None
    )
    metadata_quality_field_path = output_dir / "paper_metadata_quality_field_status.csv"
    metadata_quality_field_status = (
        pd.read_csv(metadata_quality_field_path)
        if metadata_quality_field_path.exists()
        else None
    )
    metadata_split_leakage_path = output_dir / "paper_metadata_split_leakage_audit.csv"
    metadata_split_leakage_audit = (
        pd.read_csv(metadata_split_leakage_path)
        if metadata_split_leakage_path.exists()
        else None
    )

    outputs = {
        "paper_main_results_table.csv": main_results,
        "paper_metric_reporting_tiers.csv": metric_reporting_tiers,
        "paper_prefix_group_holdout_table.csv": group_table,
        "paper_error_taxonomy_table.csv": error_table,
        "paper_bootstrap_ci_table.csv": bootstrap_table,
        "paper_feature_importance_table.csv": feature_table,
        "paper_metadata_template.csv": metadata_template,
    }
    if ablation_table is not None:
        outputs["paper_feature_block_ablation_table.csv"] = ablation_table
    if "case_examples" in data:
        outputs["paper_representative_case_table.csv"] = data["case_examples"]
    if "submission_readiness" in data:
        outputs["paper_submission_readiness_table.csv"] = data["submission_readiness"]
    if signal_consensus is not None:
        outputs["paper_signal_consensus_table.csv"] = signal_consensus
    if signal_aware_residual_metrics is not None:
        outputs["paper_signal_aware_residual_metrics_table.csv"] = (
            signal_aware_residual_metrics
        )
    if signal_aware_residual_threshold_grid is not None:
        outputs["paper_signal_aware_residual_threshold_grid_table.csv"] = (
            signal_aware_residual_threshold_grid
        )
    if signal_aware_residual_bootstrap_ci is not None:
        outputs["paper_signal_aware_residual_bootstrap_ci_table.csv"] = (
            signal_aware_residual_bootstrap_ci
        )
    if signal_aware_safe_policy_metrics is not None:
        outputs["paper_signal_aware_safe_policy_metrics_table.csv"] = (
            signal_aware_safe_policy_metrics
        )
    if signal_aware_safe_policy_bootstrap_ci is not None:
        outputs["paper_signal_aware_safe_policy_bootstrap_ci_table.csv"] = (
            signal_aware_safe_policy_bootstrap_ci
        )
    if signal_aware_safe_policy_cases is not None:
        outputs["paper_signal_aware_safe_policy_cases_table.csv"] = (
            signal_aware_safe_policy_cases
        )
    if "rr_method_paired_predictions" in data:
        outputs["paper_rr_method_paired_predictions_table.csv"] = data[
            "rr_method_paired_predictions"
        ]
    if "rr_method_statistics" in data:
        outputs["paper_rr_method_statistics_table.csv"] = data["rr_method_statistics"]
    if "rr_method_statistical_tests" in data:
        outputs["paper_rr_method_statistical_tests_table.csv"] = data[
            "rr_method_statistical_tests"
        ]
    if selective_rr_metrics is not None:
        outputs["paper_selective_rr_metrics_table.csv"] = selective_rr_metrics
    if selective_rr_threshold_grid is not None:
        outputs["paper_selective_rr_threshold_grid_table.csv"] = selective_rr_threshold_grid
    if algorithmic_quality_tier_metrics is not None:
        outputs["paper_algorithmic_quality_tier_metrics_table.csv"] = (
            algorithmic_quality_tier_metrics
        )
    if algorithmic_quality_feature_summary is not None:
        outputs["paper_algorithmic_quality_feature_summary_table.csv"] = (
            algorithmic_quality_feature_summary
        )
    if algorithmic_quality_threshold_grid is not None:
        outputs["paper_algorithmic_quality_threshold_grid_table.csv"] = (
            algorithmic_quality_threshold_grid
        )
    if deployment_decision_curve is not None:
        outputs["paper_deployment_decision_curve_table.csv"] = deployment_decision_curve
    if deployment_decision_recommendation is not None:
        outputs["paper_deployment_decision_recommendation_table.csv"] = (
            deployment_decision_recommendation
        )
    if conformal_rr_metrics is not None:
        outputs["paper_conformal_rr_metrics_table.csv"] = conformal_rr_metrics
    if conformal_rr_recommendation is not None:
        outputs["paper_conformal_rr_recommendation_table.csv"] = (
            conformal_rr_recommendation
        )
    if post_safe_gate_error_cases is not None:
        outputs["paper_post_safe_gate_error_cases_table.csv"] = (
            post_safe_gate_error_cases
        )
    if post_safe_gate_probe_metrics is not None:
        outputs["paper_post_safe_gate_second_stage_probe_metrics_table.csv"] = (
            post_safe_gate_probe_metrics
        )
    if external_split_readiness is not None:
        outputs["paper_external_split_readiness_table.csv"] = external_split_readiness
    if external_split_metrics is not None:
        outputs["paper_external_split_metrics_table.csv"] = external_split_metrics
    if external_split_acceptance is not None:
        outputs["paper_external_split_acceptance_table.csv"] = external_split_acceptance
    if metadata_readiness is not None:
        outputs["paper_metadata_group_readiness_table.csv"] = metadata_readiness
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)

    report = write_report(
        output_dir,
        main_results,
        metric_reporting_tiers,
        group_table,
        error_table,
        bootstrap_table,
        feature_table,
        ablation_table,
        case_table,
        submission_readiness,
        annotation_progress,
        annotation_priority,
        metadata_quality_audit,
        metadata_quality_field_status,
        metadata_split_leakage_audit,
        metadata_readiness,
        heat_stress_readiness,
        signal_consensus,
        selective_rr_metrics,
        selective_rr_threshold_grid,
        algorithmic_quality_tier_metrics,
        algorithmic_quality_feature_summary,
        algorithmic_quality_threshold_grid,
        deployment_decision_curve,
        deployment_decision_recommendation,
        conformal_rr_metrics,
        conformal_rr_recommendation,
        post_safe_gate_error_cases,
        post_safe_gate_probe_metrics,
        external_split_readiness,
        external_split_metrics,
        external_split_acceptance,
        literature_gap_table,
        literature_innovation_priority,
        literature_q2_roadmap,
        literature_source_register,
        literature_innovation_crosswalk,
        submission_gap_action_plan,
        submission_gap_field_checklist,
        submission_gap_video_queue,
        external_validation_sample_plan,
        external_validation_acceptance,
        external_validation_collection,
        external_validation_quota,
        external_validation_error_queue,
        external_validation_error_strata,
        external_fieldwork_preflight,
        external_fieldwork_tier_status,
        external_fieldwork_field_coverage,
        external_fieldwork_row_issues,
        external_validation_execution_dashboard,
        external_validation_input_contract,
        external_validation_execution_steps,
        external_validation_acceptance_dashboard,
        pseudo_external_gate_summary,
        pseudo_external_gate_metrics,
        manuscript_claim_audit,
        method_freeze_summary,
        method_freeze_parameters,
        target_journal_strategy,
        target_journal_gate_matrix,
        claim_scope_routes,
        claim_scope_allowed_claims,
    )

    manifest_rows = [
        {
            "asset": str(path),
            "rows": int(table.shape[0]),
            "columns": int(table.shape[1]),
        }
        for path, table in ((output_dir / name, table) for name, table in outputs.items())
    ]
    for optional_csv in sorted(output_dir.glob("paper_heat_stress_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_csv in sorted(output_dir.glob("paper_metadata_annotation_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_metadata_annotation_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_html in sorted(output_dir.glob("paper_metadata_annotation_*.html")):
        manifest_rows.append({"asset": str(optional_html), "rows": np.nan, "columns": np.nan})
    for optional_xlsx in sorted(output_dir.glob("paper_metadata_annotation_*.xlsx")):
        manifest_rows.append({"asset": str(optional_xlsx), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_metadata_context_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_csv in sorted(output_dir.glob("paper_metadata_preflight_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_metadata_preflight_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_metadata_quality_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_metadata_quality_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_metadata_split_leakage_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_metadata_import_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_metadata_import_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_html in sorted(input_root.glob(f"{args.output_prefix}_metadata_annotation_*.html")):
        manifest_rows.append({"asset": str(optional_html), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_signal_consensus*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_signal_aware_residual*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_signal_aware_residual*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_signal_aware_safe_policy*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_signal_aware_safe_policy*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    signal_aware_figure_dir = input_root / f"{args.output_prefix}_signal_aware_residual_figures"
    if signal_aware_figure_dir.exists():
        for optional_csv in sorted(signal_aware_figure_dir.glob("signal_aware_*.csv")):
            optional_table = pd.read_csv(optional_csv)
            manifest_rows.append(
                {
                    "asset": str(optional_csv),
                    "rows": int(optional_table.shape[0]),
                    "columns": int(optional_table.shape[1]),
                }
            )
        for optional_figure in sorted(
            [
                *signal_aware_figure_dir.glob("signal_aware_*.png"),
                *signal_aware_figure_dir.glob("signal_aware_*.pdf"),
            ]
        ):
            manifest_rows.append({"asset": str(optional_figure), "rows": np.nan, "columns": np.nan})
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_rr_method_statistical_tests*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_selective_rr*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_selective_rr*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_algorithmic_quality*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_algorithmic_quality*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(input_root.glob(f"{args.output_prefix}_external_split*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(input_root.glob(f"{args.output_prefix}_external_split*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_external_validation_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_external_validation_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_html in sorted(output_dir.glob("paper_external_validation_*.html")):
        manifest_rows.append({"asset": str(optional_html), "rows": np.nan, "columns": np.nan})
    for physiology_csv in sorted(output_dir.glob("paper_physiology_sequence_decoder_*.csv")):
        physiology_table = pd.read_csv(physiology_csv)
        manifest_rows.append(
            {
                "asset": str(physiology_csv),
                "rows": len(physiology_table),
                "columns": len(physiology_table.columns),
            }
        )
    for physiology_md in sorted(output_dir.glob("paper_physiology_sequence_decoder_*.md")):
        manifest_rows.append(
            {"asset": str(physiology_md), "rows": np.nan, "columns": np.nan}
        )
    for multi_roi_csv in sorted(output_dir.glob("paper_multi_roi_selector_*.csv")):
        multi_roi_table = pd.read_csv(multi_roi_csv)
        manifest_rows.append(
            {
                "asset": str(multi_roi_csv),
                "rows": len(multi_roi_table),
                "columns": len(multi_roi_table.columns),
            }
        )
    for multi_roi_md in sorted(output_dir.glob("paper_multi_roi_selector_*.md")):
        manifest_rows.append(
            {"asset": str(multi_roi_md), "rows": np.nan, "columns": np.nan}
        )
    for innovation_csv in sorted(output_dir.glob("paper_duration_normalized_windowed_*.csv")):
        innovation_table = pd.read_csv(innovation_csv)
        manifest_rows.append(
            {
                "asset": str(innovation_csv),
                "rows": len(innovation_table),
                "columns": len(innovation_table.columns),
            }
        )
    for innovation_md in sorted(output_dir.glob("paper_duration_normalized_windowed_*.md")):
        manifest_rows.append(
            {"asset": str(innovation_md), "rows": np.nan, "columns": np.nan}
        )
    for calibration_csv in sorted(output_dir.glob("paper_external_farm_calibration_*.csv")):
        calibration_table = pd.read_csv(calibration_csv)
        manifest_rows.append(
            {
                "asset": str(calibration_csv),
                "rows": len(calibration_table),
                "columns": len(calibration_table.columns),
            }
        )
    for calibration_md in sorted(output_dir.glob("paper_external_farm_calibration_*.md")):
        manifest_rows.append(
            {"asset": str(calibration_md), "rows": np.nan, "columns": np.nan}
        )
    for color_csv in sorted(output_dir.glob("paper_calibration_free_*.csv")):
        color_table = pd.read_csv(color_csv)
        manifest_rows.append(
            {
                "asset": str(color_csv),
                "rows": len(color_table),
                "columns": len(color_table.columns),
            }
        )
    for color_md in sorted(output_dir.glob("paper_calibration_free_*.md")):
        manifest_rows.append(
            {"asset": str(color_md), "rows": np.nan, "columns": np.nan}
        )
    for track_csv in sorted(output_dir.glob("paper_track_stabilized_*.csv")):
        track_table = pd.read_csv(track_csv)
        manifest_rows.append(
            {
                "asset": str(track_csv),
                "rows": len(track_table),
                "columns": len(track_table.columns),
            }
        )
    for track_md in sorted(output_dir.glob("paper_track_stabilized_*.md")):
        manifest_rows.append({"asset": str(track_md), "rows": np.nan, "columns": np.nan})
    for track_json in sorted(output_dir.glob("paper_track_stabilized_*.json")):
        manifest_rows.append({"asset": str(track_json), "rows": np.nan, "columns": np.nan})
    for mapping_csv in sorted(output_dir.glob("paper_temperature_mapping_*.csv")):
        mapping_table = pd.read_csv(mapping_csv)
        manifest_rows.append(
            {
                "asset": str(mapping_csv),
                "rows": len(mapping_table),
                "columns": len(mapping_table.columns),
            }
        )
    for mapping_md in sorted(output_dir.glob("paper_temperature_mapping_*.md")):
        manifest_rows.append(
            {"asset": str(mapping_md), "rows": np.nan, "columns": np.nan}
        )
    for weather_csv in sorted(output_dir.glob("paper_lindian_nasa_power_*.csv")):
        weather_table = pd.read_csv(weather_csv)
        manifest_rows.append(
            {
                "asset": str(weather_csv),
                "rows": len(weather_table),
                "columns": len(weather_table.columns),
            }
        )
    for weather_md in sorted(output_dir.glob("paper_lindian_nasa_power_*.md")):
        manifest_rows.append(
            {"asset": str(weather_md), "rows": np.nan, "columns": np.nan}
        )
    for weather_json in sorted(output_dir.glob("paper_lindian_nasa_power_*.json")):
        manifest_rows.append(
            {"asset": str(weather_json), "rows": np.nan, "columns": np.nan}
        )
    for coherence_csv in sorted(output_dir.glob("paper_bilateral_coherence_*.csv")):
        coherence_table = pd.read_csv(coherence_csv)
        manifest_rows.append(
            {
                "asset": str(coherence_csv),
                "rows": len(coherence_table),
                "columns": len(coherence_table.columns),
            }
        )
    for coherence_md in sorted(output_dir.glob("paper_bilateral_coherence_*.md")):
        manifest_rows.append(
            {"asset": str(coherence_md), "rows": np.nan, "columns": np.nan}
        )
    for peak_fusion_csv in sorted(output_dir.glob("paper_bounded_bilateral_peak_fusion_*.csv")):
        peak_fusion_table = pd.read_csv(peak_fusion_csv)
        manifest_rows.append(
            {
                "asset": str(peak_fusion_csv),
                "rows": len(peak_fusion_table),
                "columns": len(peak_fusion_table.columns),
            }
        )
    for peak_fusion_md in sorted(output_dir.glob("paper_bounded_bilateral_peak_fusion_*.md")):
        manifest_rows.append(
            {"asset": str(peak_fusion_md), "rows": np.nan, "columns": np.nan}
        )
    for diagnostic_csv in sorted(output_dir.glob("paper_structured_peak_diagnostic_residual_*.csv")):
        diagnostic_table = pd.read_csv(diagnostic_csv)
        manifest_rows.append(
            {
                "asset": str(diagnostic_csv),
                "rows": len(diagnostic_table),
                "columns": len(diagnostic_table.columns),
            }
        )
    for diagnostic_md in sorted(output_dir.glob("paper_structured_peak_diagnostic_residual_*.md")):
        manifest_rows.append(
            {"asset": str(diagnostic_md), "rows": np.nan, "columns": np.nan}
        )
    for lindian_csv in sorted(output_dir.glob("lindian_*.csv")):
        lindian_table = read_csv_with_encoding_fallback(lindian_csv)
        manifest_rows.append(
            {
                "asset": str(lindian_csv),
                "rows": len(lindian_table),
                "columns": len(lindian_table.columns),
            }
        )
    for lindian_md in sorted(output_dir.glob("lindian_*.md")):
        manifest_rows.append(
            {"asset": str(lindian_md), "rows": np.nan, "columns": np.nan}
        )
    for lindian49_csv in sorted(
        output_dir.glob("lindian49_butterworth_spectral_*.csv")
    ):
        lindian49_table = read_csv_with_encoding_fallback(lindian49_csv)
        manifest_rows.append(
            {
                "asset": str(lindian49_csv),
                "rows": len(lindian49_table),
                "columns": len(lindian49_table.columns),
            }
        )
    for lindian49_md in sorted(
        output_dir.glob("lindian49_butterworth_spectral_*.md")
    ):
        manifest_rows.append(
            {"asset": str(lindian49_md), "rows": np.nan, "columns": np.nan}
        )
    for lindian49_png in sorted(
        output_dir.glob("lindian49_butterworth_spectral_*.png")
    ):
        manifest_rows.append(
            {"asset": str(lindian49_png), "rows": np.nan, "columns": np.nan}
        )
    for paper73_csv in sorted(output_dir.glob("paper73_adaptive_roi_*.csv")):
        paper73_table = read_csv_with_encoding_fallback(paper73_csv)
        manifest_rows.append(
            {
                "asset": str(paper73_csv),
                "rows": len(paper73_table),
                "columns": len(paper73_table.columns),
            }
        )
    for paper73_md in sorted(output_dir.glob("paper73_adaptive_roi_*.md")):
        manifest_rows.append(
            {"asset": str(paper73_md), "rows": np.nan, "columns": np.nan}
        )
    for radius_csv in sorted(output_dir.glob("paper73_fixed_radius_*.csv")):
        radius_table = read_csv_with_encoding_fallback(radius_csv)
        manifest_rows.append(
            {
                "asset": str(radius_csv),
                "rows": len(radius_table),
                "columns": len(radius_table.columns),
            }
        )
    for radius_md in sorted(output_dir.glob("paper73_fixed_radius_*.md")):
        manifest_rows.append(
            {"asset": str(radius_md), "rows": np.nan, "columns": np.nan}
        )
    for spectral_csv in sorted(output_dir.glob("paper73_butterworth_spectral_*.csv")):
        spectral_table = read_csv_with_encoding_fallback(spectral_csv)
        manifest_rows.append(
            {
                "asset": str(spectral_csv),
                "rows": len(spectral_table),
                "columns": len(spectral_table.columns),
            }
        )
    for spectral_md in sorted(output_dir.glob("paper73_butterworth_spectral_*.md")):
        manifest_rows.append(
            {"asset": str(spectral_md), "rows": np.nan, "columns": np.nan}
        )
    for spectral_png in sorted(output_dir.glob("paper73_butterworth_spectral_*.png")):
        manifest_rows.append(
            {"asset": str(spectral_png), "rows": np.nan, "columns": np.nan}
        )
    for reannotation_csv in sorted(output_dir.glob("paper_p2g_reannotation_*.csv")):
        reannotation_table = pd.read_csv(reannotation_csv)
        manifest_rows.append(
            {
                "asset": str(reannotation_csv),
                "rows": len(reannotation_table),
                "columns": len(reannotation_table.columns),
            }
        )
    for reannotation_md in sorted(output_dir.glob("paper_p2g_reannotation_*.md")):
        manifest_rows.append(
            {"asset": str(reannotation_md), "rows": np.nan, "columns": np.nan}
        )
    for reannotation_html in sorted(output_dir.glob("paper_p2g_reannotation_*.html")):
        manifest_rows.append(
            {"asset": str(reannotation_html), "rows": np.nan, "columns": np.nan}
        )
    for p2g_csv in sorted(output_dir.glob("paper_p2g_*.csv")):
        if p2g_csv.name.startswith("paper_p2g_reannotation_"):
            continue
        p2g_table = pd.read_csv(p2g_csv)
        manifest_rows.append(
            {
                "asset": str(p2g_csv),
                "rows": len(p2g_table),
                "columns": len(p2g_table.columns),
            }
        )
    for p2g_md in sorted(output_dir.glob("paper_p2g_*.md")):
        if p2g_md.name.startswith("paper_p2g_reannotation_"):
            continue
        manifest_rows.append({"asset": str(p2g_md), "rows": np.nan, "columns": np.nan})
    for p2g_html in sorted(output_dir.glob("paper_p2g_*.html")):
        if p2g_html.name.startswith("paper_p2g_reannotation_"):
            continue
        manifest_rows.append({"asset": str(p2g_html), "rows": np.nan, "columns": np.nan})
    for p2g_json in sorted(output_dir.glob("paper_p2g_*.json")):
        manifest_rows.append({"asset": str(p2g_json), "rows": np.nan, "columns": np.nan})
    external_scoring_root = input_root.parent / "external_al_images"
    if external_scoring_root.exists():
        for external_csv in sorted(external_scoring_root.glob("external_repro_external_split_*.csv")):
            external_table = pd.read_csv(external_csv)
            manifest_rows.append(
                {
                    "asset": str(external_csv),
                    "rows": len(external_table),
                    "columns": len(external_table.columns),
                }
            )
        for external_md in sorted(external_scoring_root.glob("external_repro_external_split_*.md")):
            manifest_rows.append(
                {"asset": str(external_md), "rows": np.nan, "columns": np.nan}
            )
    for optional_csv in sorted(output_dir.glob("paper_pseudo_external_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_pseudo_external_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_manuscript_claim_audit*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_manuscript_claim_audit*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_md in sorted(output_dir.glob("paper_manuscript_claim_update*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_md in sorted(output_dir.glob("paper_thermal_rr_manuscript_draft*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_claim_scope_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_claim_scope_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_frame_motion_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_frame_motion_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_consensus_rollback_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_consensus_rollback_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_algorithmic_engineering_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_algorithmic_engineering_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_rr_physiological_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_rr_physiological_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_rr_bilateral_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_rr_bilateral_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_rr_method_agreement_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_rr_method_agreement_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    agreement_figure_dir = output_dir / "paper_rr_method_agreement_figures"
    if agreement_figure_dir.exists():
        for optional_figure in sorted(agreement_figure_dir.glob("*.png")):
            manifest_rows.append({"asset": str(optional_figure), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_method_freeze_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_json in sorted(output_dir.glob("paper_method_freeze_*.json")):
        manifest_rows.append({"asset": str(optional_json), "rows": np.nan, "columns": np.nan})
    for optional_md in sorted(output_dir.glob("paper_method_freeze_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_literature_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_literature_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_submission_gap_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_submission_gap_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    for optional_csv in sorted(output_dir.glob("paper_q2_package_refresh_*.csv")):
        optional_table = pd.read_csv(optional_csv)
        manifest_rows.append(
            {
                "asset": str(optional_csv),
                "rows": int(optional_table.shape[0]),
                "columns": int(optional_table.shape[1]),
            }
        )
    for optional_md in sorted(output_dir.glob("paper_q2_package_refresh_*.md")):
        manifest_rows.append({"asset": str(optional_md), "rows": np.nan, "columns": np.nan})
    heat_report = output_dir / "paper_heat_stress_context_summary.md"
    if heat_report.exists():
        manifest_rows.append({"asset": str(heat_report), "rows": np.nan, "columns": np.nan})
    for route_asset in [
        output_dir / "paper73_technical_route.svg",
        output_dir / "paper73_technical_route.png",
        output_dir / "paper73_technical_route_composite.svg",
        output_dir / "paper73_technical_route_composite.pdf",
        output_dir / "paper73_technical_route_composite.png",
    ]:
        if route_asset.exists():
            manifest_rows.append({"asset": str(route_asset), "rows": np.nan, "columns": np.nan})
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    for support_doc in [
        docs_dir / "thermal_rr_external_validation_protocol.md",
        docs_dir / "thermal_rr_external_validation_reporting_insert.md",
        docs_dir / "thermal_rr_literature_innovation_matrix.md",
        docs_dir / "thermal_rr_literature_recent_refresh.md",
        docs_dir / "thermal_rr_manuscript_claim_update.md",
        docs_dir / "thermal_rr_algorithmic_engineering_insert.md",
        docs_dir / "thermal_rr_method_agreement_insert.md",
        docs_dir / "thermal_rr_physiological_triage.md",
        docs_dir / "thermal_rr_bilateral_consistency_gate.md",
        docs_dir / "thermal_rr_submission_route_decision.md",
    ]:
        if support_doc.exists():
            manifest_rows.append({"asset": str(support_doc), "rows": np.nan, "columns": np.nan})
    manifest_rows.append({"asset": str(report), "rows": np.nan, "columns": np.nan})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "paper_assets_manifest.csv", index=False)

    print(f"Saved paper assets to: {output_dir}")
    print(f"Saved report: {report}")
    print("\nMain results:")
    print(main_results.to_string(index=False))
    print("\nMetadata template:")
    print(output_dir / "paper_metadata_template.csv")


if __name__ == "__main__":
    main()
