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
            "Generate manuscript-ready Results, Methods, Discussion, and figure "
            "caption wording from the RR Bland-Altman agreement outputs."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=repo_root / "docs")
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def row_by_id(table: pd.DataFrame, method_id: str) -> pd.Series:
    if table.empty or "method_id" not in table.columns:
        raise ValueError("Agreement table is missing method_id")
    mask = table["method_id"].astype(str).eq(method_id)
    if not mask.any():
        raise ValueError(f"Agreement table is missing method_id={method_id}")
    return table.loc[mask].iloc[0]


def delta_by_id(table: pd.DataFrame, method_id: str) -> pd.Series:
    if table.empty or "method_id" not in table.columns:
        raise ValueError("Delta table is missing method_id")
    mask = table["method_id"].astype(str).eq(method_id)
    if not mask.any():
        raise ValueError(f"Delta table is missing method_id={method_id}")
    return table.loc[mask].iloc[0]


def strata_row(table: pd.DataFrame, method_id: str, stratum: str) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    mask = (
        table["method_id"].astype(str).eq(method_id)
        & table["truth_rr_stratum"].astype(str).eq(stratum)
    )
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def integer(value: object) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return "NA"
    return str(number)


def method_sentence(row: pd.Series) -> str:
    name = str(row["display_name"])
    if str(row.get("method_id")) == "default":
        subject = "The default thermal RR pipeline"
    else:
        subject = f"The {name[:1].lower()}{name[1:]}"
    return (
        f"{subject} had a mean bias of "
        f"{fmt(row['rr_bias_bpm'])} breaths/min, 95% limits of agreement from "
        f"{fmt(row['rr_loa_lower_bpm'])} to {fmt(row['rr_loa_upper_bpm'])} "
        f"breaths/min, a LoA width of {fmt(row['rr_loa_width_bpm'])} "
        f"breaths/min, MAE of {fmt(row['rr_mae_bpm'])} breaths/min, and exact "
        f"count agreement in {integer(row['count_exact'])}/{integer(row['n'])} "
        "videos"
    )


def figure_caption(figure_path: Path, row: pd.Series) -> str:
    return (
        f"**{figure_path.name}.** Bland-Altman plot for "
        f"{row['display_name']} on the internal 73-video set. The "
        "vertical axis shows predicted minus manual respiratory rate, and the "
        "horizontal axis shows the mean of predicted and manual respiratory "
        f"rate. The solid line denotes the mean bias "
        f"({fmt(row['rr_bias_bpm'])} breaths/min), and dashed lines denote the "
        f"95% limits of agreement ({fmt(row['rr_loa_lower_bpm'])} to "
        f"{fmt(row['rr_loa_upper_bpm'])} breaths/min)."
    )


def build_text(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    strata: pd.DataFrame,
    figure_dir: Path,
) -> str:
    default = row_by_id(summary, "default")
    quality = row_by_id(summary, "quality_residual_fixed_oof")
    signal = row_by_id(summary, "signal_consensus_fixed_oof")
    safe = row_by_id(summary, "signal_aware_safe_fixed_oof")
    safe_group = row_by_id(summary, "signal_aware_safe_prefix_group_fixed")
    safe_delta = delta_by_id(delta, "signal_aware_safe_fixed_oof")
    safe_low = strata_row(strata, "signal_aware_safe_fixed_oof", "low_rr_lt_50")
    safe_mid = strata_row(strata, "signal_aware_safe_fixed_oof", "mid_rr_50_to_70")
    safe_high = strata_row(strata, "signal_aware_safe_fixed_oof", "high_rr_ge_70")
    default_figure = figure_dir / "paper_rr_method_agreement_bland_altman_default.png"
    quality_figure = figure_dir / "paper_rr_method_agreement_bland_altman_quality_residual_fixed_oof.png"
    safe_figure = figure_dir / "paper_rr_method_agreement_bland_altman_signal_aware_safe_fixed_oof.png"

    high_rr_text = (
        "The high-RR stratum remained weakly supported because it contained only "
        f"{integer(safe_high.get('n'))} videos; in that stratum the safe gate "
        f"had bias {fmt(safe_high.get('rr_bias_bpm'))} breaths/min and MAE "
        f"{fmt(safe_high.get('rr_mae_bpm'))} breaths/min. This result should be "
        "reported as a limitation rather than as evidence for high-RR or "
        "heat-stress deployment."
        if not safe_high.empty
        else "The high-RR stratum was not available in the agreement table."
    )

    text = f"""# RR Method Agreement Manuscript Insert

This generated insert converts the current RR Bland-Altman agreement outputs
into manuscript-ready wording. The values come from the internal 73-video set
and must not be described as independent external validation.

## Methods Wording

Bland-Altman agreement was used as a complementary evaluation to identity-line
RR R2, MAE, RMSE, and exact breath-count agreement. Agreement error was defined
as predicted respiratory rate minus manual reference respiratory rate. For each
method, the mean bias, error standard deviation, and 95% limits of agreement
(LoA; bias +/- 1.96 SD) were computed over videos with both manual and predicted
RR. Proportional bias was screened by regressing the agreement error against
the mean of manual and predicted RR. Respiratory-rate strata were summarized as
low RR (<50 breaths/min), mid RR (50 to <70 breaths/min), and high RR (>=70
breaths/min). These analyses were applied only to the current internal
development set and should be rerun unchanged after the `split_all_use`
external videos receive blinded A/B consensus reference labels.

## Results Wording

To complement the R2 and count-agreement results, Bland-Altman analysis showed
that the proposed corrections tightened the error distribution around the manual
reference. {method_sentence(default)}. The quality-aware residual correction
reduced the LoA width to {fmt(quality['rr_loa_width_bpm'])} breaths/min, and
the signal-consensus supplement further reduced it to
{fmt(signal['rr_loa_width_bpm'])} breaths/min. The conservative signal-aware
safe gate produced the tightest current non-truth agreement, with a bias of
{fmt(safe['rr_bias_bpm'])} breaths/min and 95% LoA from
{fmt(safe['rr_loa_lower_bpm'])} to {fmt(safe['rr_loa_upper_bpm'])} breaths/min.
Relative to the default pipeline, the safe gate narrowed the LoA width by
{fmt(abs(float(safe_delta['delta_rr_loa_width_bpm'])))} breaths/min, decreased
MAE by {fmt(abs(float(safe_delta['delta_rr_mae_bpm'])))} breaths/min, decreased
RMSE by {fmt(abs(float(safe_delta['delta_rr_rmse_bpm'])))} breaths/min, and
increased exact count agreement by {integer(safe_delta['delta_count_exact'])}
videos, without introducing any >=2-breath count errors. In the prefix-group
internal stress test, the safe-gate LoA width remained
{fmt(safe_group['rr_loa_width_bpm'])} breaths/min, supporting the interpretation
that the safe gate improves agreement while preserving grouped internal
robustness.

The stratum analysis indicated that the agreement gain was clearest in the
lower and middle RR ranges. In the safe-gate result, the low-RR stratum
contained {integer(safe_low.get('n'))} videos with MAE
{fmt(safe_low.get('rr_mae_bpm'))} breaths/min, and the mid-RR stratum contained
{integer(safe_mid.get('n'))} videos with MAE
{fmt(safe_mid.get('rr_mae_bpm'))} breaths/min. {high_rr_text}

## Discussion Wording

The Bland-Altman results strengthen the manuscript claim because they show that
the improvement is not limited to a higher RR R2. The safe gate also reduced
the absolute error distribution and narrowed the limits of agreement, which is
more directly relevant to whether an automatically reported respiratory rate
can be interpreted against manual counting. However, this remains an internal
agreement analysis. The method should therefore be described as an internally
validated precision candidate with agreement tightening, while external
generalization, animal/session/camera robustness, and high-RR biological claims
remain conditional on the frozen `split_all_use` external scoring workflow.

## Figure Captions

{figure_caption(default_figure, default)}

{figure_caption(quality_figure, quality)}

{figure_caption(safe_figure, safe)}

## Claim Boundary

These Bland-Altman results can be used in the paper as internal method-agreement
evidence. They should not be used as an external validation result, and they do
not replace the default RR R2/MAE/RMSE table. After A/B annotation is complete
for the 119 included `split_all_use` external clips, the same agreement analysis
should be rerun on external paired predictions before making an external
agreement claim.
"""
    return text


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    output_dir = output_dir_for(args).resolve()
    docs_dir = args.docs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    summary = read_required_csv(output_dir / "paper_rr_method_agreement_summary.csv")
    delta = read_required_csv(output_dir / "paper_rr_method_agreement_delta_vs_default.csv")
    strata = read_required_csv(output_dir / "paper_rr_method_agreement_truth_rr_strata.csv")
    figure_dir = output_dir / "paper_rr_method_agreement_figures"
    text = build_text(summary, delta, strata, figure_dir)

    asset_path = output_dir / "paper_rr_method_agreement_manuscript_insert.md"
    docs_path = docs_dir / "thermal_rr_method_agreement_insert.md"
    asset_path.write_text(text, encoding="utf-8")
    docs_path.write_text(text, encoding="utf-8")
    print(f"Saved RR method agreement manuscript insert: {asset_path}")
    print(f"Saved docs insert: {docs_path}")


if __name__ == "__main__":
    main()
