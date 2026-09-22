from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is available in the target env.
    stats = None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Merge RR outputs with environment/scene metadata and build heat-stress "
            "context tables. The script computes THI from temperature and humidity "
            "when THI is not manually provided."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--thi-thresholds",
        nargs=3,
        type=float,
        default=[68.0, 72.0, 80.0],
        metavar=("MILD", "MODERATE", "SEVERE"),
        help=(
            "Default dairy THI category thresholds. Categories are <MILD, "
            "MILD-MODERATE, MODERATE-SEVERE, and >=SEVERE."
        ),
    )
    return parser.parse_args()


def default_metadata_csv(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets" / "paper_metadata_template.csv"


def default_output_dir(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets"


def normalize_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.isin(["", "nan", "NaN", "None", "<NA>"]))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_thi(temperature_c: pd.Series, relative_humidity_percent: pd.Series) -> pd.Series:
    """Common dairy THI formula using dry-bulb temperature in C and RH in percent."""
    temp = numeric(temperature_c)
    rh = numeric(relative_humidity_percent)
    fahrenheit = 1.8 * temp + 32.0
    return fahrenheit - ((0.55 - 0.0055 * rh) * (fahrenheit - 58.0))


def assign_thi_category(thi: pd.Series, thresholds: list[float]) -> pd.Series:
    mild, moderate, severe = thresholds
    categories = pd.Series("missing", index=thi.index, dtype="object")
    categories = categories.mask(thi.notna() & (thi < mild), "no_heat_stress")
    categories = categories.mask(thi.notna() & (thi >= mild) & (thi < moderate), "mild")
    categories = categories.mask(thi.notna() & (thi >= moderate) & (thi < severe), "moderate")
    categories = categories.mask(thi.notna() & (thi >= severe), "severe")
    return categories


def load_inputs(input_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_csv = args.metadata_csv or default_metadata_csv(input_root, args.corrected_prefix)
    predictions_csv = (
        args.predictions_csv
        or input_root / f"{args.corrected_prefix}_predictions.csv"
    )
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV does not exist: {predictions_csv}")

    metadata = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(predictions_csv)
    if "video_id" not in metadata.columns or "video_id" not in predictions.columns:
        raise ValueError("Both metadata and predictions must include video_id.")
    metadata["video_id"] = metadata["video_id"].astype(str)
    predictions["video_id"] = predictions["video_id"].astype(str)
    return metadata, predictions


def build_context_dataset(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rr_columns = [
        "video_id",
        "truth_rr",
        "rr_bpm",
        "corrected_rr_bpm",
        "truth_count",
        "peaks",
        "corrected_peaks",
        "abs_rr_error",
        "corrected_abs_rr_error",
        "residual_confidence",
        "residual_margin",
        "applied_adjust",
    ]
    available_rr_columns = [column for column in rr_columns if column in predictions.columns]
    context = metadata.merge(
        predictions[available_rr_columns],
        on="video_id",
        how="left",
        validate="one_to_one",
    )

    for column in ["ambient_temperature_c", "relative_humidity_percent", "thi", "athi"]:
        if column not in context.columns:
            context[column] = np.nan

    computed_thi = compute_thi(
        context["ambient_temperature_c"],
        context["relative_humidity_percent"],
    )
    provided_thi = numeric(context["thi"])
    context["thi_computed"] = computed_thi
    context["thi_analysis"] = provided_thi.where(provided_thi.notna(), computed_thi)
    context["thi_source"] = np.where(
        provided_thi.notna(),
        "provided_metadata",
        np.where(computed_thi.notna(), "computed_from_temperature_humidity", "missing"),
    )
    context["thi_category"] = assign_thi_category(context["thi_analysis"], thresholds)
    context["athi_analysis"] = numeric(context["athi"])
    return context


def build_readiness(context: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("cow_id", "metadata_grouping", "Current dataset uses the numeric video label; real cow-level GroupKFold requires true animal IDs."),
        ("collection_date", "metadata_grouping", "Useful for date-level GroupKFold."),
        ("scene_id", "metadata_grouping", "Useful for scene-level GroupKFold."),
        ("camera_id", "metadata_grouping", "Useful for camera-level GroupKFold."),
        ("ambient_temperature_c", "heat_stress_context", "Needed to compute THI."),
        ("relative_humidity_percent", "heat_stress_context", "Needed to compute THI."),
        ("thi_analysis", "heat_stress_context", "Primary heat-stress index for analysis."),
        ("athi_analysis", "heat_stress_context", "Optional adjusted heat-stress index."),
        ("posture", "animal_context", "Optional behavioral context."),
        ("head_motion_score_0_3", "video_quality_context", "Useful for motion robustness analysis."),
        ("occlusion_score_0_3", "video_quality_context", "Useful for occlusion robustness analysis."),
        ("nostril_visibility_score_0_3", "video_quality_context", "Useful for ROI visibility analysis."),
        ("external_test_split", "validation_design", "Required to mark external test videos."),
    ]
    rows = []
    total = len(context)
    for column, purpose, note in checks:
        if column in context.columns:
            if column in {"thi_analysis", "athi_analysis"}:
                values = numeric(context[column])
                nonmissing = int(values.notna().sum())
                unique = int(values.dropna().nunique())
            else:
                values = normalize_text(context[column])
                nonmissing = int(values.notna().sum())
                unique = int(values.dropna().nunique())
        else:
            nonmissing = 0
            unique = 0
        rows.append(
            {
                "field": column,
                "purpose": purpose,
                "nonmissing": nonmissing,
                "total_videos": total,
                "coverage_percent": (nonmissing / total * 100.0) if total else math.nan,
                "unique_values": unique,
                "ready": bool(nonmissing == total and unique >= 1),
                "note": note,
            }
        )

    thi_categories = context.loc[context["thi_category"] != "missing", "thi_category"]
    rows.append(
        {
            "field": "thi_category",
            "purpose": "heat_stress_context",
            "nonmissing": int(len(thi_categories)),
            "total_videos": total,
            "coverage_percent": (len(thi_categories) / total * 100.0) if total else math.nan,
            "unique_values": int(thi_categories.nunique()),
            "ready": bool(len(thi_categories) == total and thi_categories.nunique() >= 2),
            "note": "Needed for comparing RR across heat-stress strata.",
        }
    )
    return pd.DataFrame(rows)


def build_category_summary(context: pd.DataFrame) -> pd.DataFrame:
    if "thi_category" not in context.columns:
        return pd.DataFrame()
    valid = context[context["thi_category"] != "missing"].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "thi_category",
                "videos",
                "mean_corrected_rr_bpm",
                "sd_corrected_rr_bpm",
                "mean_truth_rr",
                "mean_thi",
            ]
        )
    return (
        valid.groupby("thi_category", dropna=False)
        .agg(
            videos=("video_id", "count"),
            mean_corrected_rr_bpm=("corrected_rr_bpm", "mean"),
            sd_corrected_rr_bpm=("corrected_rr_bpm", "std"),
            mean_truth_rr=("truth_rr", "mean"),
            mean_thi=("thi_analysis", "mean"),
            min_thi=("thi_analysis", "min"),
            max_thi=("thi_analysis", "max"),
        )
        .reset_index()
    )


def spearman_association(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    x_values = numeric(x)
    y_values = numeric(y)
    valid = x_values.notna() & y_values.notna()
    n = int(valid.sum())
    if n < 3:
        return math.nan, math.nan, n
    if stats is None:
        return float(x_values[valid].corr(y_values[valid], method="spearman")), math.nan, n
    result = stats.spearmanr(x_values[valid], y_values[valid], nan_policy="omit")
    return float(result.statistic), float(result.pvalue), n


def build_association_table(context: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        ("truth_rr", "reference_rr"),
        ("rr_bpm", "default_predicted_rr"),
        ("corrected_rr_bpm", "fixed_threshold_corrected_rr"),
    ]
    predictors = [
        ("thi_analysis", "THI"),
        ("athi_analysis", "ATHI"),
        ("head_motion_score_0_3", "Head motion score"),
        ("occlusion_score_0_3", "Occlusion score"),
        ("nostril_visibility_score_0_3", "Nostril visibility score"),
    ]
    rows = []
    for predictor, predictor_label in predictors:
        if predictor not in context.columns:
            continue
        for outcome, outcome_label in outcomes:
            if outcome not in context.columns:
                continue
            rho, p_value, n = spearman_association(context[predictor], context[outcome])
            rows.append(
                {
                    "predictor": predictor_label,
                    "outcome": outcome_label,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n": n,
                    "interpretation": (
                        "insufficient_metadata" if n < 3 else "exploratory_association"
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    formatted = formatted.fillna("").astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_summary(
    output_dir: Path,
    readiness: pd.DataFrame,
    category_summary: pd.DataFrame,
    association: pd.DataFrame,
) -> Path:
    report = output_dir / "paper_heat_stress_context_summary.md"
    text = f"""# RR Heat-Stress Context Summary

This report merges respiratory-rate outputs with the metadata template. It is designed to support the manuscript's biological interpretation after environment and scene metadata are filled.

## Readiness

{markdown_table(readiness[["field", "purpose", "nonmissing", "total_videos", "coverage_percent", "unique_values", "ready"]])}

## THI Category Summary

{markdown_table(category_summary)}

## Exploratory Associations

{markdown_table(association)}

## Notes

- THI is computed only when `ambient_temperature_c` and `relative_humidity_percent` are available and `thi` is not already provided.
- ATHI is not fabricated. Fill `athi` when using a literature-specific adjusted index.
- Association rows with n < 3 are readiness placeholders, not statistical evidence.
- Use these outputs to upgrade the paper from RR detection toward heat-stress or health-monitoring interpretation.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(input_root, args.corrected_prefix)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, predictions = load_inputs(input_root, args)
    context = build_context_dataset(metadata, predictions, list(args.thi_thresholds))
    readiness = build_readiness(context)
    category_summary = build_category_summary(context)
    association = build_association_table(context)

    outputs = {
        "paper_heat_stress_context_dataset.csv": context,
        "paper_heat_stress_readiness.csv": readiness,
        "paper_heat_stress_category_summary.csv": category_summary,
        "paper_heat_stress_association_table.csv": association,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    report = write_summary(output_dir, readiness, category_summary, association)

    print(f"Saved heat-stress context assets to: {output_dir}")
    print(f"Saved report: {report}")
    print("\nReadiness:")
    print(readiness.to_string(index=False))
    if not category_summary.empty:
        print("\nTHI category summary:")
        print(category_summary.to_string(index=False))
    if not association.empty:
        print("\nAssociations:")
        print(association.to_string(index=False))


if __name__ == "__main__":
    main()
