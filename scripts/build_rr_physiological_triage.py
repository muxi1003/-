from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build an RR-only physiological triage layer from the conservative "
            "signal-aware safe-gate prediction, algorithmic risk score, and "
            "internal conformal uncertainty. This does not diagnose heat stress "
            "or disease; it produces a reproducible review/alert asset."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--interval-variant", default="risk_adaptive_fixed_scale")
    return parser.parse_args()


def output_dir_for(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def read_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    if len(valid) < 2:
        return math.nan
    truth = valid["truth"].to_numpy(dtype=float)
    pred = valid["pred"].to_numpy(dtype=float)
    sse = float(np.sum((truth - pred) ** 2))
    sst = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if sst == 0.0:
        return math.nan
    return 1.0 - sse / sst


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def load_risk_threshold(input_root: Path, output_prefix: str) -> float:
    recommendation = read_optional(input_root / f"{output_prefix}_deployment_decision_recommendation.csv")
    if not recommendation.empty and "risk_threshold" in recommendation.columns:
        value = pd.to_numeric(recommendation["risk_threshold"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    return 0.30


def load_predictions(input_root: Path, output_prefix: str, target_coverage: float, interval_variant: str) -> pd.DataFrame:
    safe = read_required(input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv")
    quality = read_optional(input_root / f"{output_prefix}_algorithmic_quality_predictions.csv")
    if not quality.empty:
        quality_columns = [
            column
            for column in [
                "video_id",
                "algorithmic_review_risk_score",
                "algorithmic_quality_score",
                "algorithmic_signal_agreement",
                "algorithmic_signal_disagreement_score",
                "algorithmic_risk_tier_fixed",
                "algorithmic_quality_score_uses_truth",
            ]
            if column in quality.columns
        ]
        if "video_id" in quality_columns:
            safe = safe.merge(
                quality[quality_columns],
                on="video_id",
                how="left",
                validate="one_to_one",
            )

    intervals = read_optional(input_root / f"{output_prefix}_conformal_rr_intervals.csv")
    if not intervals.empty:
        mask = (
            intervals.get("method", pd.Series("", index=intervals.index)).astype(str).eq("signal_aware_safe_fixed_oof")
            & np.isclose(
                pd.to_numeric(intervals.get("target_coverage", np.nan), errors="coerce"),
                float(target_coverage),
                equal_nan=False,
            )
            & intervals.get("interval_variant", pd.Series("", index=intervals.index)).astype(str).eq(interval_variant)
        )
        interval_columns = [
            column
            for column in [
                "video_id",
                "lower_rr_bpm",
                "upper_rr_bpm",
                "interval_width_bpm",
                "covered",
            ]
            if column in intervals.columns
        ]
        selected = intervals.loc[mask, interval_columns].copy()
        if not selected.empty:
            safe = safe.merge(selected, on="video_id", how="left", validate="one_to_one")
    return safe


def assign_predicted_rr_band(predicted_rr: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    valid = pd.to_numeric(predicted_rr, errors="coerce").dropna()
    if valid.empty:
        return pd.Series("unknown", index=predicted_rr.index), {
            "q25": math.nan,
            "q75": math.nan,
            "q90": math.nan,
        }
    q25, q75, q90 = [float(x) for x in np.nanquantile(valid.to_numpy(dtype=float), [0.25, 0.75, 0.90])]
    values = pd.to_numeric(predicted_rr, errors="coerce")
    bands = pd.Series("mid_rr_internal_range", index=predicted_rr.index, dtype=object)
    bands[values < q25] = "lower_quartile_rr_internal_range"
    bands[values >= q75] = "upper_quartile_rr_candidate"
    bands[values >= q90] = "top_decile_rr_candidate"
    bands[values.isna()] = "unknown"
    return bands, {"q25": q25, "q75": q75, "q90": q90}


def build_triage_table(
    predictions: pd.DataFrame,
    *,
    risk_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = predictions.copy()
    out["triage_predicted_count"] = numeric(out, "signal_aware_safe_final_peaks")
    out["triage_predicted_rr_bpm"] = numeric(out, "signal_aware_safe_final_rr_bpm")
    out["algorithmic_review_risk_score"] = numeric(out, "algorithmic_review_risk_score").fillna(1.0)
    out["algorithmic_quality_score"] = numeric(out, "algorithmic_quality_score")
    out["triage_auto_report_candidate"] = out["algorithmic_review_risk_score"] <= float(risk_threshold)
    out["triage_risk_threshold"] = float(risk_threshold)
    out["triage_rr_band"], quantiles = assign_predicted_rr_band(out["triage_predicted_rr_bpm"])

    lower = numeric(out, "lower_rr_bpm")
    pred = out["triage_predicted_rr_bpm"]
    q75 = quantiles["q75"]
    q90 = quantiles["q90"]

    action = pd.Series("manual_review_low_confidence", index=out.index, dtype=object)
    auto = out["triage_auto_report_candidate"]
    action[auto] = "auto_report_routine_rr"
    if np.isfinite(q75):
        action[auto & (lower >= q75)] = "auto_report_high_rr_candidate_context_review"
        action[(~auto) & (pred >= q75)] = "manual_review_high_rr_candidate"
    if np.isfinite(q90):
        action[auto & (pred >= q90) & ~(lower >= q75)] = "auto_report_high_rr_uncertain_context_review"
    out["triage_recommended_action"] = action
    out["triage_uses_truth_for_decision"] = False
    out["triage_claim_boundary"] = (
        "RR-only screening and review triage; not heat-stress, disease, or welfare diagnosis "
        "without synchronized environment and animal-context metadata."
    )

    selected_columns = [
        "video_id",
        "triage_predicted_count",
        "triage_predicted_rr_bpm",
        "lower_rr_bpm",
        "upper_rr_bpm",
        "interval_width_bpm",
        "algorithmic_review_risk_score",
        "algorithmic_quality_score",
        "algorithmic_signal_agreement",
        "algorithmic_signal_disagreement_score",
        "algorithmic_risk_tier_fixed",
        "triage_risk_threshold",
        "triage_auto_report_candidate",
        "triage_rr_band",
        "triage_recommended_action",
        "triage_uses_truth_for_decision",
        "truth_count",
        "truth_rr",
        "signal_aware_safe_abs_count_error",
        "signal_aware_safe_abs_rr_error",
        "triage_claim_boundary",
    ]
    existing = [column for column in selected_columns if column in out.columns]
    return out[existing].copy(), quantiles


def metric_row(data: pd.DataFrame, mask: pd.Series, label: str, selection_rule: str) -> dict[str, object]:
    subset = data.loc[mask].copy()
    truth_rr = numeric(subset, "truth_rr")
    pred_rr = numeric(subset, "triage_predicted_rr_bpm")
    truth_count = numeric(subset, "truth_count")
    pred_count = numeric(subset, "triage_predicted_count")
    rr_valid = pd.DataFrame({"truth": truth_rr, "pred": pred_rr}).dropna()
    count_valid = pd.DataFrame({"truth": truth_count, "pred": pred_count}).dropna()
    if rr_valid.empty:
        rr_mae = math.nan
        rr_rmse = math.nan
    else:
        error = rr_valid["pred"].to_numpy(dtype=float) - rr_valid["truth"].to_numpy(dtype=float)
        rr_mae = float(np.mean(np.abs(error)))
        rr_rmse = float(np.sqrt(np.mean(error**2)))
    if count_valid.empty:
        count_mae = math.nan
        exact_count = 0
        within_one_count = 0
        ge2 = 0
    else:
        count_error = count_valid["pred"].to_numpy(dtype=float) - count_valid["truth"].to_numpy(dtype=float)
        abs_count_error = np.abs(count_error)
        count_mae = float(np.mean(abs_count_error))
        exact_count = int(np.sum(abs_count_error == 0))
        within_one_count = int(np.sum(abs_count_error <= 1))
        ge2 = int(np.sum(abs_count_error >= 2))
    return {
        "label": label,
        "videos": int(len(subset)),
        "coverage": float(len(subset) / len(data)) if len(data) else math.nan,
        "rr_valid_videos": int(len(rr_valid)),
        "rr_r2": regression_r2(truth_rr, pred_rr),
        "rr_mae": rr_mae,
        "rr_rmse": rr_rmse,
        "count_valid_videos": int(len(count_valid)),
        "count_mae": count_mae,
        "exact_count": exact_count,
        "exact_rate": float(exact_count / len(count_valid)) if len(count_valid) else math.nan,
        "within_one_count": within_one_count,
        "within_one_rate": float(within_one_count / len(count_valid)) if len(count_valid) else math.nan,
        "abs_count_error_ge2": ge2,
        "selection_rule": selection_rule,
        "uses_truth_for_selection": False,
        "paper_use": "rr_only_triage_audit",
    }


def build_summary(data: pd.DataFrame, quantiles: dict[str, float]) -> pd.DataFrame:
    auto = data["triage_auto_report_candidate"].astype(bool)
    high = data["triage_rr_band"].isin(["upper_quartile_rr_candidate", "top_decile_rr_candidate"])
    top_decile = data["triage_rr_band"].eq("top_decile_rr_candidate")
    rows = [
        metric_row(data, pd.Series(True, index=data.index), "all_safe_gate_predictions", "all videos"),
        metric_row(
            data,
            auto,
            "auto_report_subset",
            "algorithmic_review_risk_score <= frozen deployment threshold",
        ),
        metric_row(
            data,
            ~auto,
            "manual_review_subset",
            "algorithmic_review_risk_score > frozen deployment threshold",
        ),
        metric_row(
            data,
            high,
            "predicted_upper_quartile_rr_candidate",
            f"predicted safe-gate RR >= internal q75 ({fmt_float(quantiles['q75'], 3)} bpm)",
        ),
        metric_row(
            data,
            auto & high,
            "auto_high_rr_candidate",
            "auto-report candidate and predicted safe-gate RR >= internal q75",
        ),
        metric_row(
            data,
            top_decile,
            "predicted_top_decile_rr_candidate",
            f"predicted safe-gate RR >= internal q90 ({fmt_float(quantiles['q90'], 3)} bpm)",
        ),
    ]
    summary = pd.DataFrame(rows)
    summary["rr_q25_bpm"] = quantiles["q25"]
    summary["rr_q75_bpm"] = quantiles["q75"]
    summary["rr_q90_bpm"] = quantiles["q90"]
    summary["claim_boundary"] = (
        "Internal RR-only screening audit. External validation and synchronized "
        "environment/quality metadata are required before heat-stress or disease claims."
    )
    return summary


def metadata_context(output_dir: Path) -> pd.DataFrame:
    progress = read_optional(output_dir / "paper_metadata_annotation_progress.csv")
    fields = [
        "collection_start_date",
        "collection_end_date",
        "collection_location_country",
        "collection_location_province",
        "collection_location_county",
        "collection_site",
        "ambient_temperature_c",
        "relative_humidity_percent",
        "thi",
        "athi",
        "head_motion_score_0_3",
        "occlusion_score_0_3",
        "nostril_visibility_score_0_3",
        "external_test_split",
    ]
    rows = []
    for field in fields:
        if progress.empty:
            rows.append({"field": field, "nonempty": 0, "total_videos": 0, "coverage_percent": math.nan})
            continue
        match = progress[progress["field"].astype(str).eq(field)]
        if match.empty:
            rows.append({"field": field, "nonempty": 0, "total_videos": 0, "coverage_percent": math.nan})
        else:
            item = match.iloc[0]
            rows.append(
                {
                    "field": field,
                    "nonempty": int(float(item.get("nonempty", 0))),
                    "total_videos": int(float(item.get("total_videos", 0))),
                    "coverage_percent": float(item.get("coverage_percent", math.nan)),
                }
            )
    context = pd.DataFrame(rows)
    context["triage_interpretation"] = np.where(
        context["field"].isin(
            [
                "ambient_temperature_c",
                "relative_humidity_percent",
                "thi",
                "athi",
                "head_motion_score_0_3",
                "occlusion_score_0_3",
                "nostril_visibility_score_0_3",
            ]
        )
        & (context["nonempty"] == 0),
        "missing_do_not_claim_stratified_physiology_or_quality",
        "available_context_or_split_field",
    )
    return context


def source_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "lin_2025_irt_head_movement_rr",
                "source_type": "thermal_rr_baseline",
                "url": "https://doi.org/10.1016/j.jtherbio.2025.104154",
                "paper_use": "Positions thermal nostril RR as current baseline; triage adds reliability and review value.",
            },
            {
                "source_id": "yan_2024_environmental_ml_rr",
                "source_type": "environmental_rr_context",
                "url": "https://doi.org/10.1016/j.biosystemseng.2024.01.010",
                "paper_use": "Supports the need for environment variables before physiological interpretation.",
            },
            {
                "source_id": "cartwright_2022_heat_challenge",
                "source_type": "heat_stress_physiology_context",
                "url": "https://arxiv.org/abs/2201.02675",
                "paper_use": "Shows RR and THI are linked in heat-challenge studies; current dataset lacks synchronized THI.",
            },
            {
                "source_id": "sadeghi_2024_calves_thermal_vitals",
                "source_type": "thermal_vital_sign_context",
                "url": "https://arxiv.org/abs/2405.11532",
                "paper_use": "Supports thermal vital-sign monitoring as a PLF direction while preserving species/age boundaries.",
            },
        ]
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(fmt_float(value))
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    triage: pd.DataFrame,
    context: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    risk_threshold: float,
    target_coverage: float,
    interval_variant: str,
) -> None:
    action_counts = (
        triage["triage_recommended_action"]
        .value_counts()
        .rename_axis("triage_recommended_action")
        .reset_index(name="videos")
    )
    text = "\n".join(
        [
            "# RR-Only Physiological Triage Layer",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            f"Risk threshold: `{risk_threshold:.2f}`",
            f"Conformal interval: `{target_coverage:.2f}` target coverage, `{interval_variant}`",
            "",
            "This asset converts the strongest current non-truth RR estimate into a review and alert workflow: automatic reporting for low algorithmic-risk videos, high-RR candidate flags from predicted RR only, and explicit context-review boundaries. It does not use reference truth for triage decisions.",
            "",
            "## Internal Audit",
            "",
            markdown_table(
                summary,
                [
                    "label",
                    "videos",
                    "coverage",
                    "rr_r2",
                    "rr_mae",
                    "rr_rmse",
                    "exact_count",
                    "exact_rate",
                    "abs_count_error_ge2",
                    "selection_rule",
                ],
            ),
            "",
            "## Triage Actions",
            "",
            markdown_table(action_counts, ["triage_recommended_action", "videos"]),
            "",
            "## Metadata Boundary",
            "",
            markdown_table(
                context,
                ["field", "nonempty", "total_videos", "coverage_percent", "triage_interpretation"],
            ),
            "",
            "## Source Register",
            "",
            markdown_table(sources, ["source_id", "source_type", "url", "paper_use"]),
            "",
            "## Manuscript Boundary",
            "",
            "Use this as an internal RR-only deployment and interpretation supplement. The current data support high-precision automatic RR reporting for a selected low-risk subset, not a heat-stress diagnosis. Heat-stress, welfare, motion, occlusion, or cow-level physiological claims still require synchronized environment metadata, manual or validated automated quality labels, and independent external validation.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = output_dir_for(input_root, args.corrected_prefix).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_threshold = load_risk_threshold(input_root, args.output_prefix)
    predictions = load_predictions(
        input_root,
        args.output_prefix,
        args.target_coverage,
        args.interval_variant,
    )
    triage, quantiles = build_triage_table(predictions, risk_threshold=risk_threshold)
    summary = build_summary(triage, quantiles)
    context = metadata_context(output_dir)
    sources = source_register()

    triage_path = output_dir / "paper_rr_physiological_triage_predictions.csv"
    summary_path = output_dir / "paper_rr_physiological_triage_summary.csv"
    context_path = output_dir / "paper_rr_physiological_triage_metadata_context.csv"
    sources_path = output_dir / "paper_rr_physiological_triage_source_register.csv"
    report_path = output_dir / "paper_rr_physiological_triage_report.md"
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "thermal_rr_physiological_triage.md"

    triage.to_csv(triage_path, index=False)
    summary.to_csv(summary_path, index=False)
    context.to_csv(context_path, index=False)
    sources.to_csv(sources_path, index=False)
    write_report(
        report_path,
        summary,
        triage,
        context,
        sources,
        risk_threshold=risk_threshold,
        target_coverage=args.target_coverage,
        interval_variant=args.interval_variant,
    )
    docs_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved physiological triage predictions: {triage_path}")
    print(f"Saved physiological triage summary: {summary_path}")
    print(f"Saved physiological triage report: {report_path}")
    print(f"Saved docs copy: {docs_path}")
    print(summary[["label", "videos", "coverage", "rr_r2", "rr_mae", "exact_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
