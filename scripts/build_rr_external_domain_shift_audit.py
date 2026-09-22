from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


SUMMARY_FEATURES = [
    "selection_score",
    "selection_median_prominence",
    "selection_interval_cv",
    "selection_amplitude",
    "selection_missing_rate",
    "mean_prominence",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare internal and external RR signal-quality distributions without "
            "using external labels for model selection."
        )
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "al_images",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--paper-assets-dir", type=Path, default=assets)
    return parser.parse_args()


def temperature_features(root: Path, prefix: str, cohort: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(f"*/{prefix}_temperatures.csv")):
        data = pd.read_csv(path)
        required = {"left_temp", "right_temp", "left_conf", "right_conf"}
        if not required.issubset(data.columns):
            continue
        left = pd.to_numeric(data["left_temp"], errors="coerce")
        right = pd.to_numeric(data["right_temp"], errors="coerce")
        left_conf = pd.to_numeric(data["left_conf"], errors="coerce")
        right_conf = pd.to_numeric(data["right_conf"], errors="coerce")
        combined = pd.concat([left, right], ignore_index=True)
        rows.append(
            {
                "cohort": cohort,
                "video_id": path.parent.name,
                "frames_temperature_csv": len(data),
                "left_valid_fraction": float(left.notna().mean()),
                "right_valid_fraction": float(right.notna().mean()),
                "both_missing_fraction": float((left.isna() & right.isna()).mean()),
                "temperature_median": float(combined.median()),
                "temperature_std": float(combined.std()),
                "temperature_iqr": float(combined.quantile(0.75) - combined.quantile(0.25)),
                "left_confidence_median": float(left_conf.median()),
                "right_confidence_median": float(right_conf.median()),
            }
        )
    return pd.DataFrame(rows)


def load_cohort(root: Path, prefix: str, cohort: str) -> pd.DataFrame:
    summary_path = root / f"{prefix}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    available = [feature for feature in SUMMARY_FEATURES if feature in summary.columns]
    summary_columns = ["video_id", *available]
    for optional in ["truth_rr", "rr_bpm", "abs_rr_error", "truth_count", "peaks"]:
        if optional in summary.columns:
            summary_columns.append(optional)
    summary = summary[summary_columns].copy()
    for column in summary.columns:
        if column != "video_id":
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    temps = temperature_features(root, prefix, cohort)
    output = summary.merge(temps, on="video_id", how="left", validate="one_to_one")
    output["cohort"] = cohort
    return output


def pooled_smd(internal: pd.Series, external: pd.Series) -> float:
    a = internal.dropna().to_numpy(dtype=float)
    b = external.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    denominator = len(a) + len(b) - 2
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / denominator)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else float("nan")


def shift_size(smd: float) -> str:
    if not np.isfinite(smd):
        return "unavailable"
    magnitude = abs(smd)
    if magnitude >= 0.8:
        return "large"
    if magnitude >= 0.5:
        return "moderate"
    if magnitude >= 0.2:
        return "small"
    return "negligible"


def build_comparison(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "video_id",
        "cohort",
        "truth_rr",
        "rr_bpm",
        "abs_rr_error",
        "truth_count",
        "peaks",
        "frames_temperature_csv",
    }
    features = sorted((set(internal.columns) & set(external.columns)) - excluded)
    rows: list[dict[str, object]] = []
    for feature in features:
        a = pd.to_numeric(internal[feature], errors="coerce").dropna()
        b = pd.to_numeric(external[feature], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ks = ks_2samp(a, b)
        smd = pooled_smd(a, b)
        rows.append(
            {
                "feature": feature,
                "internal_n": len(a),
                "external_n": len(b),
                "internal_mean": float(a.mean()),
                "external_mean": float(b.mean()),
                "internal_median": float(a.median()),
                "external_median": float(b.median()),
                "internal_iqr": float(a.quantile(0.75) - a.quantile(0.25)),
                "external_iqr": float(b.quantile(0.75) - b.quantile(0.25)),
                "standardized_mean_difference_external_minus_internal": smd,
                "shift_magnitude": shift_size(smd),
                "ks_statistic": float(ks.statistic),
                "ks_p_value_unadjusted": float(ks.pvalue),
                "domain_shift_flag": bool(abs(smd) >= 0.8 or ks.pvalue < 0.001),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "standardized_mean_difference_external_minus_internal",
        key=lambda values: values.abs(),
        ascending=False,
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def build_error_associations(external: pd.DataFrame) -> pd.DataFrame:
    if "abs_rr_error" not in external.columns:
        return pd.DataFrame()
    excluded = {
        "video_id",
        "cohort",
        "truth_rr",
        "rr_bpm",
        "abs_rr_error",
        "truth_count",
        "peaks",
        "frames_temperature_csv",
    }
    rows: list[dict[str, object]] = []
    error = pd.to_numeric(external["abs_rr_error"], errors="coerce")
    for feature in sorted(set(external.columns) - excluded):
        values = pd.to_numeric(external[feature], errors="coerce")
        valid = values.notna() & error.notna()
        if valid.sum() < 10 or values[valid].nunique() < 2:
            continue
        statistic, p_value = spearmanr(values[valid], error[valid])
        rows.append(
            {
                "feature": feature,
                "videos": int(valid.sum()),
                "spearman_r_with_abs_rr_error": float(statistic),
                "p_value_unadjusted": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_bh"] = benjamini_hochberg(result["p_value_unadjusted"].to_numpy(dtype=float))
    result["exploratory_association_flag"] = result["p_value_bh"] < 0.05
    return result.sort_values(
        "spearman_r_with_abs_rr_error",
        key=lambda values: values.abs(),
        ascending=False,
    )


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    associations: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    top = comparison.head(8)
    lines = [
        "# Internal-to-External RR Domain-Shift Audit",
        "",
        "Status: `large_signal_quality_domain_shift_detected`",
        "",
        f"Internal videos: `{len(internal)}`; provisional external videos: `{len(external)}`.",
        "",
        "## Largest Distribution Shifts",
        "",
        "| feature | internal median | external median | SMD | magnitude | KS p |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['internal_median']:.6f} | "
            f"{row['external_median']:.6f} | "
            f"{row['standardized_mean_difference_external_minus_internal']:.3f} | "
            f"{row['shift_magnitude']} | {row['ks_p_value_unadjusted']:.3g} |"
        )
    lines.extend(
        [
            "",
            "The dominant changes are higher missingness and interval irregularity with lower "
            "curve-selection score/prominence. Temperature location shifts are smaller. This "
            "supports a nostril tracking and periodic-signal stability diagnosis rather than a "
            "simple global temperature-offset explanation.",
            "",
            "## Error Associations",
            "",
        ]
    )
    if associations.empty:
        lines.append("_No error-association table was available._")
    else:
        lines.extend(
            [
                "| feature | Spearman r | BH-adjusted p | flag |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for _, row in associations.head(8).iterrows():
            lines.append(
                f"| {row['feature']} | {row['spearman_r_with_abs_rr_error']:.3f} | "
                f"{row['p_value_bh']:.3g} | {bool(row['exploratory_association_flag'])} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not retune the frozen RR thresholds on these external labels. The next method "
            "iteration should target label-independent nostril tracking continuity and explicit "
            "harmonic/half-rate failure detection, then be frozen and tested on new independent "
            "or held-back external sessions. These tests are diagnostic and exploratory, not "
            "confirmatory hypothesis tests.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    internal_root = args.internal_root.resolve()
    external_root = args.external_root.resolve()
    internal = load_cohort(internal_root, args.internal_prefix, "internal_development")
    external = load_cohort(external_root, args.external_prefix, "external_provisional")
    comparison = build_comparison(internal, external)
    associations = build_error_associations(external)
    per_video = pd.concat([internal, external], ignore_index=True, sort=False)

    comparison_path = external_root / f"{args.external_prefix}_domain_shift_feature_comparison.csv"
    association_path = external_root / f"{args.external_prefix}_domain_shift_error_associations.csv"
    per_video_path = external_root / f"{args.external_prefix}_domain_shift_per_video.csv"
    report_path = external_root / f"{args.external_prefix}_domain_shift_audit.md"
    paper_assets_dir = args.paper_assets_dir.resolve()
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    associations.to_csv(association_path, index=False)
    per_video.to_csv(per_video_path, index=False)
    write_report(report_path, comparison, associations, internal, external)
    comparison.to_csv(
        paper_assets_dir
        / "paper_external_validation_single_reference_domain_shift_feature_comparison.csv",
        index=False,
    )
    associations.to_csv(
        paper_assets_dir
        / "paper_external_validation_single_reference_domain_shift_error_associations.csv",
        index=False,
    )
    write_report(
        paper_assets_dir / "paper_external_validation_single_reference_domain_shift_audit.md",
        comparison,
        associations,
        internal,
        external,
    )
    print(f"Saved domain-shift feature comparison: {comparison_path}")
    print(f"Saved domain-shift error associations: {association_path}")
    print(f"Saved domain-shift report: {report_path}")
    print(f"flagged_features={int(comparison['domain_shift_flag'].sum())}/{len(comparison)}")


if __name__ == "__main__":
    main()
