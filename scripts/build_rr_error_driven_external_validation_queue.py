from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build an error-driven external validation queue from current internal "
            "safe-gate residual errors, algorithmic risk tiers, and Q2 external "
            "sample-size targets. This is a collection-planning artifact, not an "
            "external validation result."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric(data: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def prefix_group(video_id: object) -> str:
    text = str(video_id)
    match = re.match(r"^[A-Za-z]+", text)
    if match:
        return match.group(0).lower()
    return "numeric"


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.input_root / f"{args.corrected_prefix}_paper_assets"
    )


def load_risk_threshold(input_root: Path, output_prefix: str) -> float:
    recommendation = read_optional(
        input_root / f"{output_prefix}_deployment_decision_recommendation.csv"
    )
    if not recommendation.empty and "operating_point" in recommendation.columns:
        value = str(recommendation.iloc[0].get("operating_point", ""))
        if value.startswith("risk_le_"):
            try:
                return float(value.replace("risk_le_", ""))
            except ValueError:
                pass
    return 0.30


def q2_target_n(output_dir: Path) -> int:
    quota = read_optional(output_dir / "paper_external_validation_quota_table.csv")
    if quota.empty:
        return 104
    rows = quota[
        (quota["collection_tier"].astype(str) == "q2_target_absolute_validation")
        & (quota["quota_dimension"].astype(str) == "external videos")
    ]
    if rows.empty:
        return 104
    value = pd.to_numeric(rows.iloc[0].get("minimum_target"), errors="coerce")
    if pd.isna(value):
        return 104
    return int(value)


def add_error_fields(quality: pd.DataFrame) -> pd.DataFrame:
    data = quality.copy()
    if "video_prefix_group" not in data.columns:
        data["video_prefix_group"] = data["video_id"].map(prefix_group)
    safe_count = numeric(data, "signal_aware_safe_final_peaks")
    truth_count = numeric(data, "truth_count")
    count_error = safe_count - truth_count
    data["safe_gate_count_error"] = count_error
    data["safe_gate_abs_count_error"] = count_error.abs()
    data["safe_gate_exact"] = data["safe_gate_abs_count_error"] == 0
    data["post_safe_gate_target_adjust"] = (truth_count - safe_count).round().clip(-1, 1)
    data["post_safe_gate_error_direction"] = np.where(
        data["post_safe_gate_target_adjust"] > 0,
        "under_counted_needs_plus1",
        np.where(
            data["post_safe_gate_target_adjust"] < 0,
            "over_counted_needs_minus1",
            "no_residual_count_error",
        ),
    )
    return data


def priority_from_case(row: pd.Series, risk_threshold: float) -> tuple[int, str, int, int]:
    risk = float(row.get("algorithmic_review_risk_score", 0.0))
    exact = bool(row.get("safe_gate_exact", False))
    low_risk_auto = risk <= risk_threshold
    if not exact and low_risk_auto:
        return 1, "P0_low_risk_false_auto_challenge", 3, 5
    if not exact and risk >= 0.45:
        return 2, "P0_high_risk_residual_error", 3, 5
    if not exact:
        return 3, "P1_residual_error_balance", 2, 4
    if exact and risk >= 0.45:
        return 4, "P1_high_risk_correct_case_control", 2, 3
    if exact and abs(risk - risk_threshold) <= 0.05:
        return 5, "P2_decision_boundary_case", 1, 2
    return 9, "P3_background_distribution_case", 1, 1


def build_queue(
    quality: pd.DataFrame,
    error_cases: pd.DataFrame,
    risk_threshold: float,
) -> pd.DataFrame:
    data = add_error_fields(quality)
    error_ids = set(error_cases.get("video_id", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, object]] = []

    candidates = []
    candidates.extend(data[data["video_id"].astype(str).isin(error_ids)].index.tolist())
    high_risk_correct = data[
        (data["safe_gate_exact"]) & (numeric(data, "algorithmic_review_risk_score") >= 0.45)
    ].sort_values("algorithmic_review_risk_score", ascending=False)
    candidates.extend(high_risk_correct.head(12).index.tolist())
    boundary = data[
        (numeric(data, "algorithmic_review_risk_score") - risk_threshold).abs() <= 0.05
    ].sort_values("algorithmic_review_risk_score", ascending=False)
    candidates.extend(boundary.head(10).index.tolist())

    seen: set[int] = set()
    for index in candidates:
        if index in seen:
            continue
        seen.add(index)
        row = data.loc[index]
        priority_order, stratum, min_matches, preferred_matches = priority_from_case(
            row, risk_threshold
        )
        risk = float(row.get("algorithmic_review_risk_score", math.nan))
        error_direction = str(row.get("post_safe_gate_error_direction", ""))
        prefix_group = str(row.get("video_prefix_group", ""))
        rows.append(
            {
                "priority_rank": priority_order,
                "planning_stratum": stratum,
                "internal_prototype_video_id": row.get("video_id", ""),
                "video_prefix_group": prefix_group,
                "internal_truth_count": row.get("truth_count", ""),
                "internal_safe_gate_count": row.get("signal_aware_safe_final_peaks", ""),
                "internal_safe_gate_count_error": row.get("safe_gate_count_error", ""),
                "error_direction": error_direction,
                "algorithmic_review_risk_score": risk,
                "algorithmic_risk_tier_fixed": row.get("algorithmic_risk_tier_fixed", ""),
                "algorithmic_signal_agreement": row.get("algorithmic_signal_agreement", ""),
                "signal_aware_safe_guard_reason": row.get("signal_aware_safe_guard_reason", ""),
                "minimum_external_matches": min_matches,
                "preferred_external_matches": preferred_matches,
                "external_collection_rule": (
                    "Collect prospectively assigned external videos matching this "
                    "failure/risk signature; do not reuse the internal video as "
                    "external evidence and do not choose external rows after seeing "
                    "prediction errors."
                ),
                "required_metadata": (
                    "cow_id;collection_date;camera_id;scene_id;ambient_temperature_c;"
                    "relative_humidity_percent;head_motion_score_0_3;"
                    "occlusion_score_0_3;nostril_visibility_score_0_3;"
                    "manual_breath_count;manual_duration_seconds;external_test_split"
                ),
                "why_it_matters": (
                    "covers residual count direction, non-truth risk triage, and "
                    "prefix/session diversity for the frozen safe-gate method"
                ),
                "paper_use": "external_collection_priority_not_validation_result",
            }
        )
    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    return queue.sort_values(
        [
            "priority_rank",
            "algorithmic_review_risk_score",
            "internal_prototype_video_id",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def stratum_row(
    quota_type: str,
    stratum: str,
    internal_videos: int,
    internal_errors: int,
    minimum: int,
    preferred: int,
    selection_rule: str,
    why_it_matters: str,
    claim_unlocked: str,
) -> dict[str, object]:
    return {
        "quota_type": quota_type,
        "stratum": stratum,
        "internal_videos": int(internal_videos),
        "internal_safe_gate_errors": int(internal_errors),
        "minimum_external_videos": int(minimum),
        "preferred_external_videos": int(preferred),
        "overlap_allowed": True,
        "selection_rule": selection_rule,
        "why_it_matters": why_it_matters,
        "q2_claim_unlocked_if_covered": claim_unlocked,
    }


def build_strata(quality: pd.DataFrame, target_n: int, risk_threshold: float) -> pd.DataFrame:
    data = add_error_fields(quality)
    rows: list[dict[str, object]] = []

    for direction in ["over_counted_needs_minus1", "under_counted_needs_plus1"]:
        subset = data[data["post_safe_gate_error_direction"] == direction]
        rows.append(
            stratum_row(
                "residual_error_direction",
                direction,
                len(subset),
                int((subset["safe_gate_abs_count_error"] > 0).sum()),
                max(12, math.ceil(target_n * 0.12)),
                max(18, math.ceil(target_n * 0.18)),
                "prospectively collect videos expected to include similar breathing-rate and curve-shape conditions",
                "balances the two remaining +/-1 count residual directions",
                "defensible external analysis of remaining safe-gate error modes",
            )
        )

    risk = numeric(data, "algorithmic_review_risk_score")
    low_risk_errors = data[(risk <= risk_threshold) & (data["safe_gate_abs_count_error"] > 0)]
    high_risk = data[risk >= 0.45]
    rows.append(
        stratum_row(
            "deployment_triage_challenge",
            f"low_risk_auto_candidate_errors_risk_le_{risk_threshold:.2f}",
            len(low_risk_errors),
            int((low_risk_errors["safe_gate_abs_count_error"] > 0).sum()),
            max(8, len(low_risk_errors) * 3),
            max(12, len(low_risk_errors) * 5),
            "include low-risk-looking videos with independent manual RR reference",
            "stress-tests false auto-report cases that are hardest to catch by review triage",
            "credible auto-report safety claim",
        )
    )
    rows.append(
        stratum_row(
            "deployment_triage_challenge",
            "high_algorithmic_risk_cases",
            len(high_risk),
            int((high_risk["safe_gate_abs_count_error"] > 0).sum()),
            max(18, math.ceil(target_n * 0.20)),
            max(28, math.ceil(target_n * 0.30)),
            "include videos whose non-truth risk score would trigger manual review",
            "checks whether the review gate captures difficult external videos",
            "review-burden and risk-triage validation",
        )
    )

    for tier, subset in data.groupby(data.get("algorithmic_risk_tier_fixed", pd.Series("", index=data.index)).astype(str)):
        if not tier or tier == "nan":
            continue
        proportion = len(subset) / max(len(data), 1)
        rows.append(
            stratum_row(
                "algorithmic_risk_tier",
                tier,
                len(subset),
                int((subset["safe_gate_abs_count_error"] > 0).sum()),
                max(8, math.ceil(target_n * proportion * 0.75)),
                max(12, math.ceil(target_n * proportion)),
                "match the non-truth risk-tier distribution while preserving enough high-risk cases",
                "prevents external validation from only sampling easy curves",
                "risk-stratified external performance table",
            )
        )

    if "video_prefix_group" in data.columns:
        for prefix, subset in data.groupby(data["video_prefix_group"].astype(str)):
            proportion = len(subset) / max(len(data), 1)
            rows.append(
                stratum_row(
                    "current_domain_proxy",
                    f"prefix_group_{prefix}",
                    len(subset),
                    int((subset["safe_gate_abs_count_error"] > 0).sum()),
                    max(5, math.ceil(target_n * proportion * 0.5)),
                    max(8, math.ceil(target_n * proportion)),
                    "use as an internal-domain proxy only; replace with real cow/session/camera strata externally",
                    "keeps the external set from missing current failure-associated ID domains",
                    "domain-balance sensitivity analysis",
                )
            )

    rr = numeric(data, "truth_rr")
    if rr.notna().sum() >= 6:
        bands = pd.qcut(rr.rank(method="first"), q=3, labels=["low_rr", "mid_rr", "high_rr"])
        for band, subset in data.groupby(bands.astype(str)):
            rows.append(
                stratum_row(
                    "reference_rr_range",
                    band,
                    len(subset),
                    int((subset["safe_gate_abs_count_error"] > 0).sum()),
                    max(10, math.ceil(target_n / 4)),
                    max(18, math.ceil(target_n / 3)),
                    "cover low, middle, and high reference RR ranges using manual counts",
                    "guards against a narrow breathing-rate distribution in the external set",
                    "external RR-range robustness",
                )
            )
    return pd.DataFrame(rows)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if data.empty:
        return "_No rows available._"
    view = data[[column for column in columns if column in data.columns]].head(max_rows).copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else fmt_float(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    queue: pd.DataFrame,
    strata: pd.DataFrame,
    target_n: int,
    risk_threshold: float,
) -> None:
    text = f"""# Error-Driven External Validation Queue

This plan turns the current internal failure modes into external collection
targets. It does not create external validation evidence by itself.

## Design Boundary

- Q2 absolute-validation target used here: `{target_n}` external videos.
- Deployment auto-report risk threshold used here: `risk <= {risk_threshold:.2f}`.
- Current internal videos are prototypes for strata only; they must not be reused
  as external validation rows.
- External videos should be assigned prospectively, before prediction errors are
  inspected.

## Priority Queue

{markdown_table(queue, [
    'priority_rank',
    'planning_stratum',
    'internal_prototype_video_id',
    'video_prefix_group',
    'internal_safe_gate_count_error',
    'error_direction',
    'algorithmic_review_risk_score',
    'algorithmic_risk_tier_fixed',
    'minimum_external_matches',
    'preferred_external_matches',
    'paper_use',
])}

## Coverage Strata

{markdown_table(strata, [
    'quota_type',
    'stratum',
    'internal_videos',
    'internal_safe_gate_errors',
    'minimum_external_videos',
    'preferred_external_videos',
    'why_it_matters',
    'q2_claim_unlocked_if_covered',
])}

## Manuscript Use

Use this as a Methods supplement for external validation design and as an
internal-to-external failure-mode bridge. Do not report these rows as validation
performance until new external videos are collected, manually referenced, and
scored by the frozen method.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = read_required(
        args.input_root / f"{args.output_prefix}_algorithmic_quality_predictions.csv"
    )
    error_cases = read_required(
        args.input_root / f"{args.output_prefix}_post_safe_gate_error_cases.csv"
    )
    risk_threshold = load_risk_threshold(args.input_root, args.output_prefix)
    target_n = q2_target_n(output_dir)

    queue = build_queue(quality, error_cases, risk_threshold)
    strata = build_strata(quality, target_n, risk_threshold)

    queue_path = output_dir / "paper_external_validation_error_driven_queue.csv"
    strata_path = output_dir / "paper_external_validation_error_driven_strata.csv"
    report_path = output_dir / "paper_external_validation_error_driven_report.md"
    queue.to_csv(queue_path, index=False)
    strata.to_csv(strata_path, index=False)
    write_report(report_path, queue, strata, target_n, risk_threshold)

    print(f"Saved error-driven external validation queue: {queue_path}")
    print(f"Saved error-driven external validation strata: {strata_path}")
    print(f"Saved error-driven external validation report: {report_path}")
    print("\nTop priority rows:")
    print(
        queue[
            [
                "priority_rank",
                "planning_stratum",
                "internal_prototype_video_id",
                "algorithmic_review_risk_score",
                "minimum_external_matches",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
