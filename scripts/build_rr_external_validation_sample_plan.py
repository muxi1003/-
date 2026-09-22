from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build external-validation sample-size and acceptance-criteria tables for Q2+ submission planning."
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def ci_half_width(row: pd.Series) -> float:
    return float(row["ci_high_97_5"] - row["ci_low_2_5"]) / 2.0


def estimate_n_for_precision(
    current_n: int, current_half_width: float, target_half_width: float, design_minimum: int = 0
) -> int:
    if target_half_width <= 0 or current_half_width <= 0:
        return max(current_n, design_minimum)
    estimated = current_n * (current_half_width / target_half_width) ** 2
    return int(max(math.ceil(estimated), design_minimum))


def estimate_n_to_exclude_zero(current_n: int, estimate: float, half_width: float) -> int | None:
    if math.isclose(float(estimate), 0.0) or half_width <= 0:
        return None
    if half_width < abs(float(estimate)):
        return current_n
    return int(math.ceil(current_n * (half_width / abs(float(estimate))) ** 2))


def row_by_metric(bootstrap: pd.DataFrame, metric: str) -> pd.Series:
    match = bootstrap[bootstrap["metric"].astype(str) == metric]
    if match.empty:
        raise ValueError(f"Missing bootstrap metric: {metric}")
    return match.iloc[0]


def build_sample_plan(bootstrap: pd.DataFrame, current_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    absolute_targets = [
        {
            "metric": "corrected_rr_r2",
            "display_metric": "External primary-method RR R2",
            "target_half_width": 0.03,
            "gate": "lower 95% CI should remain >= 0.90 for a strong absolute-performance claim",
            "design_minimum": 80,
        },
        {
            "metric": "corrected_rr_mae",
            "display_metric": "External primary-method MAE",
            "target_half_width": 0.50,
            "gate": "upper 95% CI should remain <= 2.5 bpm",
            "design_minimum": 80,
        },
        {
            "metric": "corrected_rr_rmse",
            "display_metric": "External primary-method RMSE",
            "target_half_width": 0.60,
            "gate": "upper 95% CI should remain <= 4.0 bpm",
            "design_minimum": 80,
        },
    ]
    for item in absolute_targets:
        metric_row = row_by_metric(bootstrap, str(item["metric"]))
        half_width = ci_half_width(metric_row)
        rows.append(
            {
                "planning_use": "absolute_external_performance",
                "metric": item["display_metric"],
                "current_internal_estimate": float(metric_row["estimate"]),
                "current_internal_ci_low": float(metric_row["ci_low_2_5"]),
                "current_internal_ci_high": float(metric_row["ci_high_97_5"]),
                "target_precision_or_gate": item["gate"],
                "estimated_min_external_videos": estimate_n_for_precision(
                    current_n,
                    half_width,
                    float(item["target_half_width"]),
                    int(item["design_minimum"]),
                ),
                "basis": (
                    f"CI-width scaling from current paired bootstrap n={current_n}; "
                    "planning aid, not a guarantee."
                ),
                "paper_interpretation": (
                    "Use to justify external absolute performance if the frozen external "
                    "set has complete metadata and no post-hoc threshold changes."
                ),
            }
        )

    relative_targets = [
        {
            "metric": "delta_rr_r2",
            "display_metric": "Delta RR R2 versus default",
            "gate": "95% CI lower bound > 0 if claiming statistically supported R2 improvement",
        },
        {
            "metric": "delta_rr_mae",
            "display_metric": "Delta MAE versus default",
            "gate": "95% CI upper bound < 0 if claiming statistically supported MAE reduction",
        },
        {
            "metric": "delta_rr_rmse",
            "display_metric": "Delta RMSE versus default",
            "gate": "95% CI upper bound < 0 if claiming statistically supported RMSE reduction",
        },
        {
            "metric": "delta_exact_count",
            "display_metric": "Delta exact-count agreement versus default",
            "gate": "95% CI lower bound > 0 if claiming more exact counts",
        },
    ]
    for item in relative_targets:
        metric_row = row_by_metric(bootstrap, str(item["metric"]))
        half_width = ci_half_width(metric_row)
        needed = estimate_n_to_exclude_zero(current_n, float(metric_row["estimate"]), half_width)
        rows.append(
            {
                "planning_use": "relative_improvement_evidence",
                "metric": item["display_metric"],
                "current_internal_estimate": float(metric_row["estimate"]),
                "current_internal_ci_low": float(metric_row["ci_low_2_5"]),
                "current_internal_ci_high": float(metric_row["ci_high_97_5"]),
                "target_precision_or_gate": item["gate"],
                "estimated_min_external_videos": needed if needed is not None else "",
                "basis": (
                    f"Approximate n needed for the current effect size to exclude zero; "
                    f"current half-width={half_width:.4f}."
                ),
                "paper_interpretation": (
                    "Use this only for planning. If this n is not feasible, write the "
                    "relative gain as internal/candidate evidence and emphasize external "
                    "absolute performance instead."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_acceptance_criteria(
    main_results: pd.DataFrame,
    safe_policy_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    def metric_value(method_contains: str, column: str) -> float | str:
        match = main_results[main_results["method"].astype(str).str.contains(method_contains, regex=False)]
        if match.empty:
            return ""
        return float(match.iloc[0][column])

    def safe_metric(label: str, column: str) -> float | str:
        if safe_policy_metrics is None or safe_policy_metrics.empty:
            return metric_value("Conservative signal-aware safe gate (fixed threshold", column)
        match = safe_policy_metrics[safe_policy_metrics["label"].astype(str).eq(label)]
        if match.empty:
            return ""
        return float(match.iloc[0][column])

    return pd.DataFrame(
        [
            {
                "claim_type": "algorithmic_engineering_external_performance",
                "method_scope": "conservative signal-aware safe gate, frozen before external evaluation",
                "metric": "RR R2",
                "acceptance_rule": "external RR R2 >= 0.90 and preferably lower 95% CI >= 0.90",
                "current_internal_reference": safe_metric("signal_aware_safe_policy_fixed_oof", "rr_r2"),
                "manuscript_use_if_passed": "primary algorithmic-engineering external validation table",
            },
            {
                "claim_type": "algorithmic_engineering_external_performance",
                "method_scope": "conservative signal-aware safe gate, frozen before external evaluation",
                "metric": "MAE",
                "acceptance_rule": "external MAE <= 2.5 bpm and preferably upper 95% CI <= 2.5 bpm",
                "current_internal_reference": safe_metric("signal_aware_safe_policy_fixed_oof", "rr_mae"),
                "manuscript_use_if_passed": "primary algorithmic-engineering external validation table",
            },
            {
                "claim_type": "algorithmic_engineering_external_performance",
                "method_scope": "conservative signal-aware safe gate, frozen before external evaluation",
                "metric": "RMSE",
                "acceptance_rule": "external RMSE <= 4.0 bpm and preferably upper 95% CI <= 4.0 bpm",
                "current_internal_reference": safe_metric("signal_aware_safe_policy_fixed_oof", "rr_rmse"),
                "manuscript_use_if_passed": "primary algorithmic-engineering external validation table",
            },
            {
                "claim_type": "algorithmic_engineering_external_performance",
                "method_scope": "conservative signal-aware safe gate, frozen before external evaluation",
                "metric": "exact count agreement",
                "acceptance_rule": "external exact-count agreement should not be worse than the default external pipeline; within-one agreement >= 95%",
                "current_internal_reference": safe_metric("signal_aware_safe_policy_fixed_oof", "exact_count"),
                "manuscript_use_if_passed": "primary algorithmic-engineering external validation table",
            },
            {
                "claim_type": "external_absolute_performance",
                "method_scope": "quality-aware residual correction, frozen before external evaluation",
                "metric": "RR R2",
                "acceptance_rule": "external RR R2 >= 0.90 and preferably lower 95% CI >= 0.90",
                "current_internal_reference": metric_value(
                    "Quality-aware residual correction (fixed threshold, out-of-fold)", "rr_r2"
                ),
                "manuscript_use_if_passed": "main external validation table",
            },
            {
                "claim_type": "external_absolute_performance",
                "method_scope": "quality-aware residual correction, frozen before external evaluation",
                "metric": "MAE",
                "acceptance_rule": "external MAE <= 2.5 bpm and preferably upper 95% CI <= 2.5 bpm",
                "current_internal_reference": metric_value(
                    "Quality-aware residual correction (fixed threshold, out-of-fold)", "rr_mae_bpm"
                ),
                "manuscript_use_if_passed": "main external validation table",
            },
            {
                "claim_type": "external_absolute_performance",
                "method_scope": "quality-aware residual correction, frozen before external evaluation",
                "metric": "RMSE",
                "acceptance_rule": "external RMSE <= 4.0 bpm and preferably upper 95% CI <= 4.0 bpm",
                "current_internal_reference": metric_value(
                    "Quality-aware residual correction (fixed threshold, out-of-fold)", "rr_rmse_bpm"
                ),
                "manuscript_use_if_passed": "main external validation table",
            },
            {
                "claim_type": "external_count_agreement",
                "method_scope": "quality-aware residual correction, frozen before external evaluation",
                "metric": "within-one breath agreement",
                "acceptance_rule": "within-one count agreement >= 95%",
                "current_internal_reference": "73/73",
                "manuscript_use_if_passed": "agreement and reliability statement",
            },
            {
                "claim_type": "relative_improvement",
                "method_scope": "quality-aware residual correction versus default pipeline",
                "metric": "paired delta MAE/RMSE/R2/exact count",
                "acceptance_rule": "paired external CI must support the claimed direction; otherwise report as internal improvement only",
                "current_internal_reference": "delta R2 CI crosses zero in current bootstrap",
                "manuscript_use_if_passed": "stronger innovation claim in Results and Abstract",
            },
            {
                "claim_type": "candidate_extension",
                "method_scope": "signal-consensus supplement",
                "metric": "RR R2 and exact count",
                "acceptance_rule": "treat as secondary unless thresholds are frozen before external testing and pass the same external gates",
                "current_internal_reference": metric_value(
                    "Signal-consensus supplement (fixed threshold, out-of-fold)", "rr_r2"
                ),
                "manuscript_use_if_passed": "secondary precision extension or supplementary validation",
            },
            {
                "claim_type": "leakage_guard",
                "method_scope": "truth-calibrated upper bound",
                "metric": "RR R2",
                "acceptance_rule": "never use as external performance; keep as oracle/upper-bound analysis",
                "current_internal_reference": metric_value("Truth-calibrated upper bound", "rr_r2"),
                "manuscript_use_if_passed": "supplementary diagnostic only",
            },
        ]
    )


def build_collection_plan(sample_plan: pd.DataFrame) -> pd.DataFrame:
    absolute_target_n = int(
        sample_plan[sample_plan["planning_use"] == "absolute_external_performance"][
            "estimated_min_external_videos"
        ]
        .replace("", np.nan)
        .dropna()
        .astype(int)
        .max()
    )
    r2_delta_row = sample_plan[sample_plan["metric"] == "Delta RR R2 versus default"].iloc[0]
    r2_delta_n = int(r2_delta_row["estimated_min_external_videos"])
    return pd.DataFrame(
        [
            {
                "collection_tier": "minimum_holdout_smoke_test",
                "recommended_videos": 50,
                "required_metadata": "cow_id;collection_date;camera_id;scene_id;external_test_split;manual_breath_count;manual_duration_seconds;manual_rr_bpm",
                "design_rule": "consecutive or prospectively assigned videos; no selection by prediction error",
                "claim_allowed": "external feasibility only; avoid strong relative-improvement wording",
            },
            {
                "collection_tier": "q2_algorithmic_external_validation",
                "recommended_videos": absolute_target_n,
                "required_metadata": (
                    "cow_id;collection_date;camera_id;scene_id;external_test_split;"
                    "manual_breath_count;manual_duration_seconds;manual_rr_bpm;"
                    "reference_rr_annotator"
                ),
                "design_rule": (
                    "freeze the conservative signal-aware safe gate before scoring; "
                    "sample prospectively across cows, sessions, and camera/scene groups; "
                    "do not require environment or manual quality scores unless making those claims"
                ),
                "claim_allowed": (
                    "external algorithmic RR accuracy, safe-gate behavior, algorithmic quality control, "
                    "and uncertainty reporting; no heat-stress or manual-quality stratified claims"
                ),
            },
            {
                "collection_tier": "q2_target_absolute_validation",
                "recommended_videos": absolute_target_n,
                "required_metadata": (
                    "cow_id;collection_date;camera_id;scene_id;ambient_temperature_c;"
                    "relative_humidity_percent;head_motion_score_0_3;occlusion_score_0_3;"
                    "nostril_visibility_score_0_3;external_test_split;manual_breath_count;"
                    "manual_duration_seconds;manual_rr_bpm;reference_rr_annotator"
                ),
                "design_rule": "include multiple cows, sessions, scenes, motion/occlusion strata, and THI range where possible",
                "claim_allowed": "external absolute RR performance plus metadata-stratified robustness",
            },
            {
                "collection_tier": "q2_strong_relative_improvement",
                "recommended_videos": r2_delta_n,
                "required_metadata": (
                    "same as q2_target_absolute_validation, with enough paired default and corrected predictions"
                ),
                "design_rule": "freeze all thresholds before evaluation and compare paired default versus corrected errors",
                "claim_allowed": "statistically supported relative RR R2 improvement if the external CI excludes zero",
            },
        ]
    )


def build_quota_table(collection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quota_specs = {
        "minimum_holdout_smoke_test": [
            ("external videos", 50, 50, "minimum feasibility check for frozen external pipeline", True),
            ("unique cow_id or animal_label", 5, 8, "avoid a single-label external result; use true animal IDs when available", True),
            ("collection dates or sessions", 2, 3, "reduce same-session leakage risk", True),
            ("camera_id or scene_id groups", 2, 3, "show at least basic scene/camera variation", True),
            ("manual reference RR protocol", "all videos", "all videos", "manual count and duration must be recorded for every external video", True),
        ],
        "q2_algorithmic_external_validation": [
            ("external videos", "from sample plan", ">= q2_algorithmic_external_validation recommended_videos", "target external RR R2/MAE/RMSE precision for the algorithmic engineering route", True),
            ("unique cow_id or animal_label", 8, 12, "support label diversity; animal-level wording requires true cow identities", True),
            ("collection dates or sessions", 4, 6, "support date/session robustness without heat-stress claims", True),
            ("camera_id or scene_id groups", 2, 4, "support camera/scene deployment variation", True),
            ("manual reference RR protocol", "all videos", "all videos", "manual count and duration must be recorded for every external video", True),
            ("paired default and frozen safe-gate predictions", "all videos", "all videos", "required for algorithmic external comparison", True),
            ("pre-frozen method freeze id", "one locked id", "one locked id", "prevents post-hoc external tuning", True),
            ("temperature/humidity/THI fields", "not required", "record if available", "claim-dependent only; not needed for algorithmic RR validation", False),
            ("manual quality scores", "not required", "record if available", "claim-dependent only; not needed for algorithmic RR validation", False),
        ],
        "q2_target_absolute_validation": [
            ("external videos", "from sample plan", ">= q2_target_absolute_validation recommended_videos", "target absolute RR R2/MAE/RMSE external precision", True),
            ("unique real cow_id", 8, 12, "support animal-level generalization language only when true cow identities are recorded", True),
            ("collection dates or sessions", 4, 6, "support date/session robustness checks", True),
            ("camera_id or scene_id groups", 2, 4, "support scene/camera robustness checks", True),
            ("THI or temperature+humidity coverage", "all videos", "all videos", "unlock heat-stress/welfare context", True),
            ("head motion scores", "all videos", "all videos", "unlock motion-robustness stratification", True),
            ("occlusion and nostril visibility scores", "all videos", "all videos", "unlock ROI-quality and occlusion analyses", True),
        ],
        "q2_strong_relative_improvement": [
            ("external videos", "from sample plan", ">= q2_strong_relative_improvement recommended_videos", "paired CI for improvement over default", True),
            ("unique real cow_id", 12, 20, "stronger animal-level diversity for paired comparison when true cow identities are recorded", True),
            ("collection dates or sessions", 6, 10, "stronger temporal diversity for paired comparison", True),
            ("camera_id or scene_id groups", 3, 5, "stronger deployment-domain diversity", True),
            ("paired default and corrected predictions", "all videos", "all videos", "required for relative-improvement claims", True),
            ("pre-frozen method freeze id", "one locked id", "one locked id", "prevents post-hoc external tuning", True),
        ],
    }
    for _, tier in collection.iterrows():
        tier_name = str(tier["collection_tier"])
        recommended = int(tier["recommended_videos"])
        for dimension, minimum, preferred, rationale, blocking in quota_specs.get(tier_name, []):
            min_value = recommended if minimum == "from sample plan" else minimum
            pref_value = (
                f">= {recommended}"
                if isinstance(preferred, str) and "recommended_videos" in preferred
                else preferred
            )
            rows.append(
                {
                    "collection_tier": tier_name,
                    "recommended_videos": recommended,
                    "quota_dimension": dimension,
                    "minimum_target": min_value,
                    "preferred_target": pref_value,
                    "q2_blocking_if_missing": bool(blocking),
                    "rationale": rationale,
                }
            )
    return pd.DataFrame(rows)


def build_fieldwork_template(collection: pd.DataFrame) -> pd.DataFrame:
    max_videos = int(collection["recommended_videos"].max())
    tier_thresholds = [
        (
            "minimum_holdout_smoke_test",
            int(collection.loc[collection["collection_tier"] == "minimum_holdout_smoke_test", "recommended_videos"].iloc[0]),
        ),
        (
            "q2_algorithmic_external_validation",
            int(collection.loc[collection["collection_tier"] == "q2_algorithmic_external_validation", "recommended_videos"].iloc[0]),
        ),
        (
            "q2_strong_relative_improvement",
            int(collection.loc[collection["collection_tier"] == "q2_strong_relative_improvement", "recommended_videos"].iloc[0]),
        ),
    ]
    rows: list[dict[str, object]] = []
    for index in range(1, max_videos + 1):
        tier = tier_thresholds[-1][0]
        for tier_name, threshold in tier_thresholds:
            if index <= threshold:
                tier = tier_name
                break
        rows.append(
            {
                "planned_video_index": index,
                "target_collection_tier": tier,
                "claim_dependent_environment_or_quality_required": (
                    "no_for_algorithmic_route; yes_for_heat_stress_or_manual_quality_claims"
                ),
                "external_video_id": "",
                "raw_video_path": "",
                "cow_id": "",
                "collection_date": "",
                "collection_time": "",
                "camera_id": "",
                "scene_id": "",
                "ambient_temperature_c": "",
                "relative_humidity_percent": "",
                "thi": "",
                "athi": "",
                "posture": "",
                "head_motion_score_0_3": "",
                "occlusion_score_0_3": "",
                "nostril_visibility_score_0_3": "",
                "external_test_split": "external",
                "manual_breath_count": "",
                "manual_duration_seconds": "",
                "manual_rr_bpm": "",
                "reference_rr_annotator": "",
                "reference_protocol_notes": "",
                "error_driven_planning_stratum": "",
                "matched_internal_prototype_video_id": "",
                "target_error_direction": "",
                "target_algorithmic_risk_tier": "",
                "manual_quality_reviewer": "",
                "temperature_humidity_source": "",
                "external_collection_notes": "",
                "include_in_external_validation": "yes",
                "exclusion_reason": "",
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    table = table.fillna("").astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.to_numpy():
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    sample_plan: pd.DataFrame,
    acceptance: pd.DataFrame,
    collection: pd.DataFrame,
    quota: pd.DataFrame,
) -> Path:
    report = output_dir / "paper_external_validation_sample_plan.md"
    text = f"""# External Validation Sample Plan

This plan converts the current internal bootstrap uncertainty into a practical external-validation design. The estimates are planning aids, not a substitute for the external experiment itself.

## Recommended Boundary

- For the current feasible Q2 path, prioritize `q2_algorithmic_external_validation`: external RR accuracy of a frozen safe-gate algorithm with manual RR reference, but no heat-stress or manual-quality stratified claims.
- If only a small external holdout is feasible, report external absolute performance and avoid claiming statistically proven improvement over the default workflow.
- If environment metadata and manual quality scores remain unavailable, do not use the full `q2_target_absolute_validation` tier for heat-stress or visual-quality claims.
- If the paper needs a strong relative-improvement claim for RR R2, the current small effect size requires a much larger external set than the current 73-video internal dataset.
- All external thresholds must be frozen before evaluation. Do not tune on the external split.

## Sample Size Planning Table

{write_markdown_table(sample_plan)}

## Acceptance Criteria

{write_markdown_table(acceptance)}

## Collection Tiers

{write_markdown_table(collection)}

## Stratified Collection Quotas

{write_markdown_table(quota)}

## Fieldwork Template

Use `paper_external_validation_fieldwork_template.csv` as the external collection worksheet. It has one row per planned video up to the strongest relative-improvement target and includes metadata, reference RR, inclusion, and exclusion fields.
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

    bootstrap = pd.read_csv(require_file(input_root / f"{args.corrected_prefix}_bootstrap_ci.csv"))
    main_results = pd.read_csv(require_file(output_dir / "paper_main_results_table.csv"))
    safe_policy_metrics_path = output_dir / "paper_signal_aware_safe_policy_metrics_table.csv"
    safe_policy_metrics = (
        pd.read_csv(safe_policy_metrics_path)
        if safe_policy_metrics_path.exists()
        else pd.DataFrame()
    )
    current_n = int(main_results["videos"].dropna().astype(int).max())

    sample_plan = build_sample_plan(bootstrap, current_n)
    acceptance = build_acceptance_criteria(main_results, safe_policy_metrics)
    collection = build_collection_plan(sample_plan)
    quota = build_quota_table(collection)
    fieldwork = build_fieldwork_template(collection)

    sample_path = output_dir / "paper_external_validation_sample_plan.csv"
    acceptance_path = output_dir / "paper_external_validation_acceptance_criteria.csv"
    collection_path = output_dir / "paper_external_validation_collection_tiers.csv"
    quota_path = output_dir / "paper_external_validation_quota_table.csv"
    fieldwork_path = output_dir / "paper_external_validation_fieldwork_template.csv"
    sample_plan.to_csv(sample_path, index=False)
    acceptance.to_csv(acceptance_path, index=False)
    collection.to_csv(collection_path, index=False)
    quota.to_csv(quota_path, index=False)
    fieldwork.to_csv(fieldwork_path, index=False)
    report = write_report(output_dir, sample_plan, acceptance, collection, quota)

    print(f"Saved external validation sample plan: {sample_path}")
    print(f"Saved external validation acceptance criteria: {acceptance_path}")
    print(f"Saved external validation collection tiers: {collection_path}")
    print(f"Saved external validation quota table: {quota_path}")
    print(f"Saved external validation fieldwork template: {fieldwork_path}")
    print(f"Saved external validation report: {report}")
    print("\nCollection tiers:")
    print(collection.to_string(index=False))


if __name__ == "__main__":
    main()
