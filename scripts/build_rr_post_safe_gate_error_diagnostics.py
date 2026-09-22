from __future__ import annotations

import argparse
import math
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only in missing-env runs.
    raise SystemExit(
        "Missing scikit-learn. Run this script with the project environment, "
        "for example E:\\real\\anaconda\\envs\\plant_gpu\\python.exe."
    ) from exc


CLASSES = np.array([-1, 0, 1], dtype=int)
LEAKAGE_TOKENS = (
    "truth",
    "target_adjust",
    "count_error",
    "rr_error",
    "abs_count_error",
    "abs_rr_error",
)
NON_FEATURE_COLUMNS = {
    "video_id",
    "temperature_csv",
    "curve_csv",
    "curve_png",
    "review_png",
    "status",
    "evaluation_note",
    "signal_aware_evaluation_note",
    "signal_aware_safe_evaluation_note",
    "rr_duration_source",
    "fusion_mode",
    "selected_fusion_mode",
    "peak_retune_rule",
    "analysis_window_source",
    "algorithmic_quality_score_uses_truth",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build post safe-gate error diagnostics and second-stage residual "
            "model probes. These probes are for error analysis only; they use "
            "current internal truth labels to train residual direction models and "
            "must not be reported as primary deployable performance."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--random-state", type=int, default=20260707)
    return parser.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def numeric(data: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def first_existing(data: pd.DataFrame, columns: list[str], default: float = math.nan) -> pd.Series:
    for column in columns:
        if column in data.columns:
            values = numeric(data, column)
            if not values.isna().all():
                return values
    return pd.Series(default, index=data.index, dtype=float)


def prefix_group(video_id: object) -> str:
    text = str(video_id)
    match = re.match(r"^[A-Za-z]+", text)
    if match:
        return match.group(0).lower()
    return "numeric"


def regression_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    if len(valid) < 2:
        return math.nan
    truth = valid["truth"].to_numpy(dtype=float)
    pred = valid["pred"].to_numpy(dtype=float)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if ss_tot == 0.0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def load_data(input_root: Path, output_prefix: str) -> pd.DataFrame:
    quality = read_required(input_root / f"{output_prefix}_algorithmic_quality_predictions.csv")
    paired_path = input_root / f"{output_prefix}_rr_method_paired_predictions.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        keep = [
            column
            for column in paired.columns
            if column == "video_id"
            or column.startswith("signal_aware_safe_fixed_oof")
            or column.startswith("default_")
            or column in {"truth_count", "truth_rr"}
        ]
        paired = paired[keep].copy()
        data = quality.merge(
            paired,
            on="video_id",
            how="left",
            suffixes=("", "_paired"),
            validate="one_to_one",
        )
    else:
        data = quality.copy()
    data["video_prefix_group"] = data["video_id"].map(prefix_group)
    return data


def add_post_safe_gate_targets(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    safe_count = first_existing(
        prepared,
        ["signal_aware_safe_final_peaks", "signal_aware_safe_fixed_oof_count"],
    )
    truth_count = numeric(prepared, "truth_count")
    residual = (truth_count - safe_count).round()
    prepared["post_safe_gate_target_adjust_raw"] = residual
    prepared["post_safe_gate_target_adjust"] = residual.clip(-1, 1).astype("Int64")
    prepared["post_safe_gate_target_was_clipped"] = residual.abs() > 1
    prepared["post_safe_gate_baseline_count"] = safe_count
    prepared["post_safe_gate_baseline_rr"] = first_existing(
        prepared,
        ["signal_aware_safe_final_rr_bpm", "signal_aware_safe_fixed_oof_rr"],
    )
    return prepared


def feature_columns(data: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in data.columns:
        lower = column.lower()
        if column in NON_FEATURE_COLUMNS:
            continue
        if any(token in lower for token in LEAKAGE_TOKENS):
            continue
        if lower.endswith("_paired"):
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() >= 5:
            columns.append(column)
    return columns


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_l2_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits[f"stratified_{n_splits}fold_oof"] = list(splitter.split(np.zeros(len(y)), y))
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        n_splits = min(5, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits[f"prefix_group_{n_splits}fold_oof"] = list(
            splitter.split(np.zeros(len(y)), y, groups)
        )
    return splits


def predict_oof_probabilities(
    model: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    probabilities = np.zeros((len(y), len(CLASSES)), dtype=float)
    for train_index, test_index in splits:
        y_train = y[train_index]
        if len(np.unique(y_train)) < 2:
            class_value = int(y_train[0])
            fold_proba = np.zeros((len(test_index), len(CLASSES)), dtype=float)
            fold_proba[:, np.where(CLASSES == class_value)[0][0]] = 1.0
        else:
            model.fit(x.iloc[train_index], y_train)
            raw_proba = model.predict_proba(x.iloc[test_index])
            fold_proba = np.zeros((len(test_index), len(CLASSES)), dtype=float)
            model_classes = model.named_steps["model"].classes_.astype(int)
            for class_index, class_value in enumerate(model_classes):
                destination = np.where(CLASSES == class_value)[0]
                if len(destination):
                    fold_proba[:, destination[0]] = raw_proba[:, class_index]
        probabilities[test_index, :] = fold_proba
    predicted = CLASSES[np.argmax(probabilities, axis=1)]
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = sorted_probabilities[:, -1]
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return pd.DataFrame(
        {
            "prob_adjust_minus1": probabilities[:, 0],
            "prob_adjust_0": probabilities[:, 1],
            "prob_adjust_plus1": probabilities[:, 2],
            "predicted_adjust": predicted,
            "second_stage_confidence": confidence,
            "second_stage_margin": margin,
        },
        index=x.index,
    )


def prediction_duration(data: pd.DataFrame) -> pd.Series:
    return first_existing(
        data,
        [
            "rr_duration_seconds",
            "duration_seconds",
            "truth_duration_seconds",
            "raw_duration_seconds",
        ],
    )


def metric_values(
    data: pd.DataFrame,
    final_count: pd.Series,
    baseline_abs_count_error: pd.Series,
    applied_adjust: pd.Series,
) -> dict[str, object]:
    truth_count = numeric(data, "truth_count")
    truth_rr = numeric(data, "truth_rr")
    duration = prediction_duration(data)
    final_rr = final_count * 60.0 / duration.replace(0.0, np.nan)
    fallback_rr = numeric(data, "post_safe_gate_baseline_rr")
    final_rr = final_rr.where(np.isfinite(final_rr), fallback_rr)
    count_error = final_count - truth_count
    abs_count_error = count_error.abs()
    rr_error = final_rr - truth_rr
    applied = applied_adjust != 0
    adjusted_abs = abs_count_error
    beneficial = applied & (adjusted_abs < baseline_abs_count_error)
    harmful = applied & (adjusted_abs > baseline_abs_count_error)
    neutral = applied & (adjusted_abs == baseline_abs_count_error)
    return {
        "videos": int(len(data)),
        "rr_r2": regression_r2(truth_rr, final_rr),
        "rr_mae": float(rr_error.abs().mean()),
        "rr_rmse": float(np.sqrt(np.mean(np.square(rr_error.dropna())))),
        "count_mae": float(abs_count_error.mean()),
        "exact_count": int((abs_count_error == 0).sum()),
        "exact_rate": float((abs_count_error == 0).mean()),
        "within_one_count": int((abs_count_error <= 1).sum()),
        "abs_count_error_ge1": int((abs_count_error >= 1).sum()),
        "abs_count_error_ge2": int((abs_count_error >= 2).sum()),
        "applied_videos": int(applied.sum()),
        "beneficial_adjustments": int(beneficial.sum()),
        "harmful_adjustments": int(harmful.sum()),
        "neutral_adjustments": int(neutral.sum()),
    }


def evaluate_policy_grid(
    data: pd.DataFrame,
    probabilities: pd.DataFrame,
    model_name: str,
    cv_scheme: str,
    baseline_metrics: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    safe_count = numeric(data, "post_safe_gate_baseline_count")
    baseline_abs_count_error = (
        safe_count - numeric(data, "truth_count")
    ).abs()
    thresholds = [(0.0, 0.0)]
    thresholds.extend(
        product([0.40, 0.50, 0.60, 0.70, 0.80], [0.05, 0.10, 0.20, 0.30])
    )
    for confidence_threshold, margin_threshold in thresholds:
        predicted_adjust = probabilities["predicted_adjust"].astype(int)
        apply_mask = (
            (predicted_adjust != 0)
            & (probabilities["second_stage_confidence"] >= confidence_threshold)
            & (probabilities["second_stage_margin"] >= margin_threshold)
        )
        applied_adjust = predicted_adjust.where(apply_mask, 0).astype(int)
        final_count = safe_count + applied_adjust
        values = metric_values(data, final_count, baseline_abs_count_error, applied_adjust)
        passes_metric_guard = (
            values["exact_count"] > int(baseline_metrics["exact_count"])
            and values["abs_count_error_ge2"] == 0
            and values["rr_mae"] <= float(baseline_metrics["rr_mae"])
        )
        rows.append(
            {
                "probe_name": model_name,
                "cv_scheme": cv_scheme,
                "confidence_threshold": float(confidence_threshold),
                "margin_threshold": float(margin_threshold),
                "policy_label": (
                    "apply_all_predicted_nonzero"
                    if confidence_threshold == 0.0 and margin_threshold == 0.0
                    else f"conf_ge_{confidence_threshold:.2f}_margin_ge_{margin_threshold:.2f}"
                ),
                **values,
                "delta_exact_vs_safe_gate": int(values["exact_count"])
                - int(baseline_metrics["exact_count"]),
                "delta_rr_mae_vs_safe_gate": float(values["rr_mae"])
                - float(baseline_metrics["rr_mae"]),
                "delta_rr_r2_vs_safe_gate": float(values["rr_r2"])
                - float(baseline_metrics["rr_r2"]),
                "passes_metric_guard": bool(passes_metric_guard),
                "paper_use": "diagnostic_probe_not_manuscript_primary",
                "promotion_boundary": (
                    "Do not promote from this table alone: the residual target and "
                    "policy threshold are selected on the current internal set. "
                    "Freeze any candidate and validate on a true external split first."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_error_cases(data: pd.DataFrame) -> pd.DataFrame:
    error_mask = numeric(data, "post_safe_gate_baseline_count") != numeric(data, "truth_count")
    cases = data.loc[error_mask].copy()
    cases["post_safe_gate_error_direction"] = np.where(
        cases["post_safe_gate_target_adjust"].astype(float) > 0,
        "under_counted_needs_plus1",
        "over_counted_needs_minus1",
    )
    columns = [
        "video_id",
        "video_prefix_group",
        "truth_count",
        "post_safe_gate_baseline_count",
        "post_safe_gate_target_adjust",
        "post_safe_gate_error_direction",
        "truth_rr",
        "post_safe_gate_baseline_rr",
        "signal_aware_safe_rr_error",
        "algorithmic_review_risk_score",
        "algorithmic_risk_tier_fixed",
        "algorithmic_signal_agreement",
        "algorithmic_interval_instability_score",
        "algorithmic_model_ambiguity_score",
        "algorithmic_missingness_score",
        "algorithmic_guard_discordance_score",
        "signal_aware_safe_guard_reason",
        "spectral_count_estimate",
        "fft_count_estimate",
        "autocorr_count_estimate",
        "signal_consensus_adjust",
        "signal_aware_safe_source_applied_adjust",
        "signal_aware_safe_applied_adjust",
    ]
    present = [column for column in columns if column in cases.columns]
    return cases[present].sort_values(
        ["post_safe_gate_target_adjust", "algorithmic_review_risk_score", "video_id"],
        ascending=[True, False, True],
    )


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if data.empty:
        return "_No rows available._"
    view = data[[column for column in columns if column in data.columns]].head(max_rows)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    baseline_metrics: dict[str, object],
    error_cases: pd.DataFrame,
    probe_metrics: pd.DataFrame,
    feature_count: int,
) -> None:
    best = probe_metrics.sort_values(
        ["exact_count", "abs_count_error_ge2", "rr_mae", "applied_videos"],
        ascending=[False, True, True, True],
    ).head(12)
    stable = probe_metrics[
        probe_metrics["passes_internal_stability_guard"].astype(bool)
    ].copy()
    point_only = probe_metrics[
        probe_metrics["passes_metric_guard"].astype(bool)
        & ~probe_metrics["passes_internal_stability_guard"].astype(bool)
    ].copy()
    if stable.empty:
        conclusion = (
            "No second-stage residual probe passed the stricter internal stability "
            "guard, which requires improvement over the frozen safe-gate baseline "
            "without >=2-count errors and with prefix-group stress support. "
            "The correct action is to keep the conservative safe gate as the "
            "point-estimate candidate and use these residual cases to guide "
            "external sampling and manual quality review, not to add another "
            "tuned correction layer."
        )
        if not point_only.empty:
            best_point = point_only.sort_values(
                ["exact_count", "rr_mae"],
                ascending=[False, True],
            ).iloc[0]
            conclusion += (
                " A point-metric-only probe did improve exact count to "
                f"{int(best_point['exact_count'])}/73, but it did not satisfy the "
                "prefix-group stability guard."
            )
    else:
        conclusion = (
            "At least one probe passes the stricter internal stability guard. It is "
            "still not a manuscript-primary result until the candidate is frozen and "
            "evaluated on a true external split."
        )
    text = f"""# Post Safe-Gate Error Diagnostics

This diagnostic uses current internal reference counts to inspect the residual
errors left after the conservative signal-aware safe gate. It is not a deployable
method claim.

## Safe-Gate Baseline

| metric | value |
| --- | ---: |
| videos | {baseline_metrics['videos']} |
| RR R2 | {baseline_metrics['rr_r2']:.6f} |
| RR MAE bpm | {baseline_metrics['rr_mae']:.6f} |
| RR RMSE bpm | {baseline_metrics['rr_rmse']:.6f} |
| exact count | {baseline_metrics['exact_count']} |
| abs count error >= 2 | {baseline_metrics['abs_count_error_ge2']} |

## Remaining Error Cases

{markdown_table(error_cases, [
    'video_id',
    'video_prefix_group',
    'truth_count',
    'post_safe_gate_baseline_count',
    'post_safe_gate_target_adjust',
    'post_safe_gate_error_direction',
    'algorithmic_review_risk_score',
    'algorithmic_risk_tier_fixed',
    'signal_aware_safe_guard_reason',
])}

## Second-Stage Probe Summary

Feature count after leakage filtering: {feature_count}

{markdown_table(best, [
    'rank_overall',
    'probe_name',
    'cv_scheme',
    'policy_label',
    'exact_count',
    'delta_exact_vs_safe_gate',
    'rr_r2',
    'rr_mae',
    'abs_count_error_ge2',
    'applied_videos',
    'beneficial_adjustments',
    'harmful_adjustments',
    'passes_metric_guard',
    'passes_internal_stability_guard',
])}

## Interpretation

{conclusion}

Guardrail: every row in the probe table is marked
`diagnostic_probe_not_manuscript_primary` because the residual target is derived
from current internal truth labels and policy thresholds are screened on this
same 73-video set.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    data = add_post_safe_gate_targets(load_data(input_root, args.output_prefix))
    data = data.dropna(subset=["post_safe_gate_target_adjust"]).reset_index(drop=True)
    y = data["post_safe_gate_target_adjust"].astype(int).to_numpy()
    features = feature_columns(data)
    if not features:
        raise ValueError("No usable non-leakage numeric features were available.")

    safe_count = numeric(data, "post_safe_gate_baseline_count")
    baseline_abs = (safe_count - numeric(data, "truth_count")).abs()
    baseline_metrics = metric_values(
        data,
        safe_count,
        baseline_abs,
        pd.Series(0, index=data.index, dtype=int),
    )
    error_cases = build_error_cases(data)

    x = data[features].copy()
    groups = data["video_prefix_group"].astype(str).to_numpy()
    split_map = cv_splits(y, groups, args.random_state)
    if not split_map:
        raise ValueError("Not enough class or group diversity for OOF diagnostics.")

    metrics: list[pd.DataFrame] = []
    for cv_scheme, splits in split_map.items():
        for model_name, model in build_models(args.random_state).items():
            probabilities = predict_oof_probabilities(model, x, y, splits)
            metrics.append(
                evaluate_policy_grid(
                    data,
                    probabilities,
                    model_name,
                    cv_scheme,
                    baseline_metrics,
                )
            )

    probe_metrics = pd.concat(metrics, ignore_index=True)
    guard_by_policy = (
        probe_metrics.groupby(["probe_name", "policy_label"])["passes_metric_guard"]
        .agg(["sum", "count"])
        .reset_index()
    )
    guard_by_policy["passes_internal_stability_guard"] = (
        guard_by_policy["sum"] == guard_by_policy["count"]
    ) & (guard_by_policy["count"] >= 2)
    probe_metrics = probe_metrics.merge(
        guard_by_policy[
            ["probe_name", "policy_label", "passes_internal_stability_guard"]
        ],
        on=["probe_name", "policy_label"],
        how="left",
        validate="many_to_one",
    )
    probe_metrics["passes_internal_stability_guard"] = (
        probe_metrics["passes_internal_stability_guard"].fillna(False).astype(bool)
    )
    probe_metrics = probe_metrics.sort_values(
        [
            "passes_internal_stability_guard",
            "passes_metric_guard",
            "exact_count",
            "abs_count_error_ge2",
            "rr_mae",
            "applied_videos",
            "probe_name",
            "cv_scheme",
        ],
        ascending=[False, False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    probe_metrics.insert(0, "rank_overall", np.arange(1, len(probe_metrics) + 1))
    probe_metrics["baseline_exact_count"] = int(baseline_metrics["exact_count"])
    probe_metrics["baseline_rr_r2"] = float(baseline_metrics["rr_r2"])
    probe_metrics["baseline_rr_mae"] = float(baseline_metrics["rr_mae"])
    probe_metrics["feature_count"] = len(features)
    probe_metrics["leakage_filter"] = (
        "numeric features only; excludes truth, target_adjust, count_error, "
        "rr_error, abs_count_error, abs_rr_error, IDs, paths, and status fields"
    )

    error_cases_path = input_root / f"{args.output_prefix}_post_safe_gate_error_cases.csv"
    probe_metrics_path = (
        input_root / f"{args.output_prefix}_post_safe_gate_second_stage_probe_metrics.csv"
    )
    report_path = input_root / f"{args.output_prefix}_post_safe_gate_error_diagnostics.md"
    error_cases.to_csv(error_cases_path, index=False)
    probe_metrics.to_csv(probe_metrics_path, index=False)
    write_report(report_path, baseline_metrics, error_cases, probe_metrics, len(features))

    print(f"Saved post safe-gate error cases: {error_cases_path}")
    print(f"Saved second-stage probe metrics: {probe_metrics_path}")
    print(f"Saved diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
