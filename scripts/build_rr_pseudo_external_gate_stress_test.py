from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GATES = [
    ("rr_r2_gate", "rr_r2", ">=", 0.90),
    ("rr_mae_gate", "rr_mae", "<=", 2.50),
    ("rr_rmse_gate", "rr_rmse", "<=", 4.00),
    ("within_one_gate", "within_one_rate", ">=", 0.95),
    ("exact_rate_gate", "exact_rate", ">=", 0.70),
]

METHOD_LABELS = {
    "baseline_default": "default_pipeline",
    "fixed_threshold": "quality_residual_fixed",
    "fixed_signal_consensus_supplement": "signal_consensus_fixed",
    "train_selected_threshold": "quality_residual_train_selected",
    "train_selected_signal_consensus_supplement": "signal_consensus_train_selected",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Run an internal pseudo-external acceptance-gate stress test over "
            "video-ID prefix groups. This is not a substitute for real external validation."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def normalize_label(label: str, group: str) -> str:
    prefix = f"{group}_"
    return label[len(prefix) :] if label.startswith(prefix) else label


def method_alias(short_label: str) -> str:
    return METHOD_LABELS.get(short_label, short_label)


def gate_pass(value: float, operator: str, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def build_gate_metrics(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in group_metrics.iterrows():
        group = str(row["heldout_prefix_group"])
        short_label = normalize_label(str(row["label"]), group)
        method = method_alias(short_label)
        exact_rate = float(row["exact_count"]) / float(row["count_valid_videos"])
        within_one_rate = float(row["within_one_count"]) / float(row["count_valid_videos"])
        out = {
            "pseudo_external_group": group,
            "method": method,
            "source_label": row["label"],
            "videos": int(row["videos"]),
            "rr_r2": float(row["rr_r2"]),
            "rr_mae": float(row["rr_mae"]),
            "rr_rmse": float(row["rr_rmse"]),
            "exact_count": int(row["exact_count"]),
            "count_valid_videos": int(row["count_valid_videos"]),
            "exact_rate": exact_rate,
            "within_one_count": int(row["within_one_count"]),
            "within_one_rate": within_one_rate,
            "stress_test_boundary": (
                "internal_prefix_group_only_not_real_external_validation"
            ),
        }
        for gate_name, metric, operator, threshold in GATES:
            passed = gate_pass(float(out[metric]), operator, threshold)
            out[gate_name] = "PASS" if passed else "FAIL"
        gate_columns = [gate_name for gate_name, _, _, _ in GATES]
        out["absolute_gate_pass_count"] = sum(out[col] == "PASS" for col in gate_columns)
        out["absolute_all_gates_pass"] = all(out[col] == "PASS" for col in gate_columns)
        rows.append(out)

    result = pd.DataFrame(rows)
    baseline = result[result["method"] == "default_pipeline"][
        ["pseudo_external_group", "rr_r2", "rr_mae", "rr_rmse", "exact_rate"]
    ].rename(
        columns={
            "rr_r2": "baseline_rr_r2",
            "rr_mae": "baseline_rr_mae",
            "rr_rmse": "baseline_rr_rmse",
            "exact_rate": "baseline_exact_rate",
        }
    )
    result = result.merge(baseline, on="pseudo_external_group", how="left")
    result["delta_rr_r2_vs_default"] = result["rr_r2"] - result["baseline_rr_r2"]
    result["delta_mae_vs_default"] = result["rr_mae"] - result["baseline_rr_mae"]
    result["delta_rmse_vs_default"] = result["rr_rmse"] - result["baseline_rr_rmse"]
    result["delta_exact_rate_vs_default"] = (
        result["exact_rate"] - result["baseline_exact_rate"]
    )
    result["relative_direction"] = np.where(
        (result["method"] == "default_pipeline"),
        "baseline",
        np.where(
            (result["delta_rr_r2_vs_default"] > 0)
            & (result["delta_mae_vs_default"] < 0)
            & (result["delta_rmse_vs_default"] <= 0)
            & (result["delta_exact_rate_vs_default"] >= 0),
            "improves_default_directionally",
            "does_not_consistently_improve_default",
        ),
    )
    return result


def build_group_summary(gate_metrics: pd.DataFrame) -> pd.DataFrame:
    preferred_methods = [
        "quality_residual_fixed",
        "signal_consensus_fixed",
        "quality_residual_train_selected",
        "signal_consensus_train_selected",
    ]
    rows: list[dict[str, object]] = []
    for group, group_df in gate_metrics.groupby("pseudo_external_group", sort=True):
        baseline = group_df[group_df["method"] == "default_pipeline"].iloc[0]
        best_gate_row = (
            group_df[group_df["method"].isin(preferred_methods)]
            .sort_values(["absolute_gate_pass_count", "rr_r2", "exact_rate"], ascending=False)
            .iloc[0]
        )
        fixed = group_df[group_df["method"] == "quality_residual_fixed"]
        signal = group_df[group_df["method"] == "signal_consensus_fixed"]
        fixed_row = fixed.iloc[0] if not fixed.empty else None
        signal_row = signal.iloc[0] if not signal.empty else None
        risk_notes: list[str] = []
        if bool(best_gate_row["absolute_all_gates_pass"]):
            risk_notes.append("at_least_one_method_passes_absolute_gates")
        else:
            risk_notes.append("no_method_passes_all_absolute_gates")
        if fixed_row is not None and fixed_row["relative_direction"] != "improves_default_directionally":
            risk_notes.append("quality_residual_fixed_does_not_improve_default_directionally")
        if signal_row is not None and signal_row["relative_direction"] == "improves_default_directionally":
            risk_notes.append("signal_consensus_improves_default_directionally")
        if int(baseline["videos"]) < 10:
            risk_notes.append("small_group_n_interpret_cautiously")
        rows.append(
            {
                "pseudo_external_group": group,
                "videos": int(baseline["videos"]),
                "baseline_rr_r2": float(baseline["rr_r2"]),
                "best_gate_method": best_gate_row["method"],
                "best_gate_pass_count": int(best_gate_row["absolute_gate_pass_count"]),
                "best_gate_all_pass": bool(best_gate_row["absolute_all_gates_pass"]),
                "best_gate_rr_r2": float(best_gate_row["rr_r2"]),
                "best_gate_mae": float(best_gate_row["rr_mae"]),
                "best_gate_rmse": float(best_gate_row["rr_rmse"]),
                "best_gate_exact_rate": float(best_gate_row["exact_rate"]),
                "risk_notes": ";".join(risk_notes),
                "paper_use": "internal_domain_stress_test_only",
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


def write_report(output_dir: Path, summary: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    report = output_dir / "paper_pseudo_external_gate_stress_report.md"
    display_metrics = metrics[
        metrics["method"].isin(
            [
                "default_pipeline",
                "quality_residual_fixed",
                "signal_consensus_fixed",
                "quality_residual_train_selected",
                "signal_consensus_train_selected",
            ]
        )
    ][
        [
            "pseudo_external_group",
            "method",
            "videos",
            "rr_r2",
            "rr_mae",
            "rr_rmse",
            "exact_rate",
            "absolute_all_gates_pass",
            "relative_direction",
        ]
    ]
    text = f"""# Pseudo-External Acceptance Gate Stress Test

This is an internal domain stress test using filename-derived prefix groups as pseudo-external domains. It is not real external validation and must not be written as independent test evidence.

## Group Summary

{write_markdown_table(summary)}

## Method-Level Gate Results

{write_markdown_table(display_metrics)}

## Interpretation Boundary

- Use this table to identify risky internal domains and to plan external data collection.
- Do not replace real `cow_id`, date/session, camera/scene, or external split validation with prefix groups.
- Passing pseudo-external gates is a useful stress-test signal, not a Q2+ readiness proof.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / "paper_repro_quality_residual_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    group_metrics_path = input_root / f"{args.output_prefix}_signal_consensus_group_metrics_by_prefix.csv"
    group_metrics = pd.read_csv(require_file(group_metrics_path))
    gate_metrics = build_gate_metrics(group_metrics)
    group_summary = build_group_summary(gate_metrics)

    metrics_path = output_dir / "paper_pseudo_external_gate_stress_metrics.csv"
    summary_path = output_dir / "paper_pseudo_external_gate_stress_summary.csv"
    gate_metrics.to_csv(metrics_path, index=False)
    group_summary.to_csv(summary_path, index=False)
    report = write_report(output_dir, group_summary, gate_metrics)

    print(f"Saved pseudo-external gate metrics: {metrics_path}")
    print(f"Saved pseudo-external gate summary: {summary_path}")
    print(f"Saved pseudo-external gate report: {report}")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
