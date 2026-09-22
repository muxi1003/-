from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "baseline_rr_bpm",
    "windowed_rr_bpm",
    "windowed_offline_context_rr_bpm",
    "windowed_causal_rr_bpm",
    "raw_curve_duration_seconds",
    "windowed_windows",
    "windowed_quality_mean",
    "windowed_rr_std",
    "selection_score",
    "selection_interval_cv",
    "selection_amplitude",
    "selection_missing_rate",
    "mean_prominence",
    "missing_both",
    "frames",
]

COMPACT_FEATURE_SETS = {
    "offline_context_only": ["windowed_offline_context_rr_bpm"],
    "causal_context_only": ["windowed_causal_rr_bpm"],
    "rr_consensus_offline": [
        "baseline_rr_bpm",
        "windowed_rr_bpm",
        "windowed_offline_context_rr_bpm",
    ],
    "rr_consensus_causal": [
        "baseline_rr_bpm",
        "windowed_rr_bpm",
        "windowed_causal_rr_bpm",
    ],
    "rr_duration_offline": [
        "baseline_rr_bpm",
        "windowed_offline_context_rr_bpm",
        "raw_curve_duration_seconds",
    ],
    "rr_duration_causal": [
        "baseline_rr_bpm",
        "windowed_causal_rr_bpm",
        "raw_curve_duration_seconds",
    ],
    "rr_stability_offline": [
        "baseline_rr_bpm",
        "windowed_offline_context_rr_bpm",
        "windowed_rr_std",
        "windowed_quality_mean",
    ],
    "rr_stability_causal": [
        "baseline_rr_bpm",
        "windowed_causal_rr_bpm",
        "windowed_rr_std",
        "windowed_quality_mean",
    ],
    "rr_selection_quality_offline": [
        "baseline_rr_bpm",
        "windowed_offline_context_rr_bpm",
        "selection_score",
        "selection_interval_cv",
        "selection_missing_rate",
    ],
    "rr_selection_quality_causal": [
        "baseline_rr_bpm",
        "windowed_causal_rr_bpm",
        "selection_score",
        "selection_interval_cv",
        "selection_missing_rate",
    ],
    "compact_signal_quality_offline": [
        "baseline_rr_bpm",
        "windowed_rr_bpm",
        "windowed_offline_context_rr_bpm",
        "windowed_quality_mean",
        "windowed_rr_std",
        "selection_score",
        "selection_interval_cv",
        "selection_missing_rate",
        "mean_prominence",
    ],
}

OFFLINE_COMPACT_FEATURE_SETS = {
    name: columns
    for name, columns in COMPACT_FEATURE_SETS.items()
    if "offline" in name
}
CAUSAL_COMPACT_FEATURE_SETS = {
    name: columns
    for name, columns in COMPACT_FEATURE_SETS.items()
    if "causal" in name
}


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
            "Evaluate few-shot external farm calibration using nested source-session "
            "and leave-one-date-out validation."
        )
    )
    parser.add_argument("--assets-dir", type=Path, default=assets)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo_root / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=20260711)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def huber_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                HuberRegressor(
                    epsilon=1.35,
                    alpha=1.0,
                    max_iter=2000,
                ),
            ),
        ]
    )


def compact_candidates(
    feature_sets: dict[str, list[str]],
) -> list[tuple[str, str, list[str], Pipeline]]:
    candidates: list[tuple[str, str, list[str], Pipeline]] = []
    for feature_set, columns in feature_sets.items():
        for alpha in [0.1, 1.0, 3.0, 10.0, 30.0]:
            candidates.append(
                (feature_set, f"ridge_{alpha:g}", columns, ridge_pipeline(alpha))
            )
        candidates.append((feature_set, "huber", columns, huber_pipeline()))
    return candidates


def load_data(assets: Path, external_root: Path) -> pd.DataFrame:
    predictions = pd.read_csv(assets / "paper_duration_normalized_windowed_predictions.csv")
    predictions = predictions[predictions["cohort"].eq("external_provisional")].copy()
    summary = pd.read_csv(external_root / "external_repro_single_reference_summary.csv")
    reference = pd.read_csv(assets / "paper_external_validation_single_reference_rows.csv")
    columns = [
        "video_id",
        "selection_score",
        "selection_interval_cv",
        "selection_amplitude",
        "selection_missing_rate",
        "mean_prominence",
        "missing_both",
        "frames",
    ]
    data = predictions.merge(summary[columns], on="video_id", how="left", validate="one_to_one")
    data = data.merge(
        reference[["external_video_id", "collection_date", "cow_id"]],
        left_on="video_id",
        right_on="external_video_id",
        how="left",
        validate="one_to_one",
    )
    color = pd.read_csv(
        assets / "paper_calibration_free_thermal_index_gated_predictions.csv"
    )
    color = color[color["cohort"].astype(str).eq("external_provisional")][
        ["video_id", "duration_gated_rr_bpm"]
    ].rename(columns={"duration_gated_rr_bpm": "color_rr_bpm"})
    data = data.merge(color, on="video_id", how="left", validate="one_to_one")
    for column in FEATURE_COLUMNS + ["truth_rr", "color_rr_bpm"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def inner_select_compact_candidate(
    data: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    max_folds: int,
    feature_sets: dict[str, list[str]],
) -> tuple[str, str, list[str], Pipeline, pd.DataFrame]:
    folds = min(int(max_folds), len(np.unique(groups)))
    if folds < 2:
        feature_set = sorted(feature_sets)[0]
        columns = feature_sets[feature_set]
        return feature_set, "huber", columns, huber_pipeline(), pd.DataFrame()
    splitter = GroupKFold(n_splits=folds)
    rows: list[dict[str, object]] = []
    candidate_lookup: dict[tuple[str, str], tuple[list[str], Pipeline]] = {}
    for feature_set, model_name, columns, model in compact_candidates(feature_sets):
        fold_mae: list[float] = []
        for train_index, validation_index in splitter.split(data, y, groups):
            fitted = model
            fitted.fit(data.iloc[train_index][columns], y[train_index])
            prediction = fitted.predict(data.iloc[validation_index][columns])
            fold_mae.append(mean_absolute_error(y[validation_index], prediction))
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "features": ";".join(columns),
                "feature_count": len(columns),
                "inner_mae": float(np.mean(fold_mae)),
                "inner_folds": folds,
            }
        )
        candidate_lookup[(feature_set, model_name)] = (columns, model)
    table = pd.DataFrame(rows).sort_values(
        ["inner_mae", "feature_count", "feature_set", "model"]
    )
    selected = table.iloc[0]
    columns, model = candidate_lookup[(selected["feature_set"], selected["model"])]
    return str(selected["feature_set"]), str(selected["model"]), columns, model, table


def inner_select_alpha(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    max_folds: int,
) -> tuple[float, pd.DataFrame]:
    unique_groups = np.unique(groups)
    folds = min(int(max_folds), len(unique_groups))
    alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    if folds < 2:
        return 10.0, pd.DataFrame(
            [{"alpha": alpha, "inner_mae": math.nan, "inner_folds": folds} for alpha in alphas]
        )
    splitter = GroupKFold(n_splits=folds)
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        fold_mae: list[float] = []
        for train_index, validation_index in splitter.split(X, y, groups):
            model = ridge_pipeline(alpha)
            model.fit(X.iloc[train_index], y[train_index])
            prediction = model.predict(X.iloc[validation_index])
            fold_mae.append(mean_absolute_error(y[validation_index], prediction))
        rows.append(
            {
                "alpha": alpha,
                "inner_mae": float(np.mean(fold_mae)),
                "inner_folds": folds,
            }
        )
    table = pd.DataFrame(rows)
    selected = float(table.sort_values(["inner_mae", "alpha"]).iloc[0]["alpha"])
    return selected, table


def outer_predictions(
    data: pd.DataFrame,
    split_groups: pd.Series,
    validation_scheme: str,
    outer_folds: int,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data[FEATURE_COLUMNS]
    y = data["truth_rr"].to_numpy(dtype=float)
    groups = split_groups.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if validation_scheme == "source_session_group_kfold":
        splitter = GroupKFold(n_splits=min(int(outer_folds), len(unique_groups)))
    elif validation_scheme == "leave_one_collection_date_out":
        splitter = LeaveOneGroupOut()
    elif validation_scheme == "leave_one_cow_out":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(validation_scheme)

    ridge_prediction = np.full(len(data), np.nan, dtype=float)
    huber_prediction = np.full(len(data), np.nan, dtype=float)
    offline_compact_prediction = np.full(len(data), np.nan, dtype=float)
    causal_compact_prediction = np.full(len(data), np.nan, dtype=float)
    color_huber_prediction = np.full(len(data), np.nan, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)
    selection_rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = groups[train_index]
        selected_alpha, inner_table = inner_select_alpha(
            X.iloc[train_index].reset_index(drop=True),
            y[train_index],
            train_groups,
            max_folds=inner_folds,
        )
        ridge = ridge_pipeline(selected_alpha)
        huber = huber_pipeline()
        ridge.fit(X.iloc[train_index], y[train_index])
        huber.fit(X.iloc[train_index], y[train_index])
        offline_set, offline_model_name, offline_columns, offline_model, offline_table = (
            inner_select_compact_candidate(
                data.iloc[train_index].reset_index(drop=True),
                y[train_index],
                train_groups,
                max_folds=inner_folds,
                feature_sets=OFFLINE_COMPACT_FEATURE_SETS,
            )
        )
        causal_set, causal_model_name, causal_columns, causal_model, causal_table = (
            inner_select_compact_candidate(
                data.iloc[train_index].reset_index(drop=True),
                y[train_index],
                train_groups,
                max_folds=inner_folds,
                feature_sets=CAUSAL_COMPACT_FEATURE_SETS,
            )
        )
        offline_model.fit(data.iloc[train_index][offline_columns], y[train_index])
        causal_model.fit(data.iloc[train_index][causal_columns], y[train_index])
        color_huber = huber_pipeline()
        color_huber.fit(data.iloc[train_index][["color_rr_bpm"]], y[train_index])
        ridge_prediction[test_index] = ridge.predict(X.iloc[test_index])
        huber_prediction[test_index] = huber.predict(X.iloc[test_index])
        offline_compact_prediction[test_index] = offline_model.predict(
            data.iloc[test_index][offline_columns]
        )
        causal_compact_prediction[test_index] = causal_model.predict(
            data.iloc[test_index][causal_columns]
        )
        color_huber_prediction[test_index] = color_huber.predict(
            data.iloc[test_index][["color_rr_bpm"]]
        )
        fold_ids[test_index] = fold
        heldout = sorted(np.unique(groups[test_index]).tolist())
        selection_rows.append(
            {
                "validation_scheme": validation_scheme,
                "outer_fold": fold,
                "train_videos": len(train_index),
                "test_videos": len(test_index),
                "train_groups": len(np.unique(groups[train_index])),
                "test_groups": len(heldout),
                "heldout_groups": ";".join(heldout),
                "selected_ridge_alpha": selected_alpha,
                "selected_offline_feature_set": offline_set,
                "selected_offline_model": offline_model_name,
                "selected_offline_features": ";".join(offline_columns),
                "selected_offline_inner_mae": float(offline_table.iloc[0]["inner_mae"]),
                "selected_causal_feature_set": causal_set,
                "selected_causal_model": causal_model_name,
                "selected_causal_features": ";".join(causal_columns),
                "selected_causal_inner_mae": float(causal_table.iloc[0]["inner_mae"]),
                "inner_alpha_scores": ";".join(
                    f"{row.alpha:g}:{row.inner_mae:.6f}"
                    for row in inner_table.itertuples(index=False)
                ),
            }
        )
    output = data[
        [
            "video_id",
            "source_session_id",
            "collection_date",
            "cow_id",
            "truth_rr",
            "truth_count",
            "truth_duration_seconds",
            "baseline_rr_bpm",
            "windowed_offline_context_rr_bpm",
            "windowed_causal_rr_bpm",
            "windowed_sequential_rr_bpm",
            "color_rr_bpm",
        ]
    ].copy()
    output["validation_scheme"] = validation_scheme
    output["outer_fold"] = fold_ids
    output["nested_ridge_rr_bpm"] = ridge_prediction
    output["fixed_huber_rr_bpm"] = huber_prediction
    output["nested_compact_offline_rr_bpm"] = offline_compact_prediction
    output["nested_compact_causal_rr_bpm"] = causal_compact_prediction
    output["nested_compact_rr_bpm"] = offline_compact_prediction
    output["color_huber_rr_bpm"] = color_huber_prediction
    output["ridge_huber_ensemble_rr_bpm"] = 0.75 * ridge_prediction + 0.25 * huber_prediction
    return output, pd.DataFrame(selection_rows)


def metric_row(
    validation_scheme: str,
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    truth_count: np.ndarray,
    duration: np.ndarray,
    note: str,
) -> dict[str, object]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    y = truth[valid]
    p = prediction[valid]
    error = p - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    predicted_count = np.rint(prediction * duration / 60.0)
    count_valid = np.isfinite(truth_count) & np.isfinite(predicted_count)
    count_error = np.abs(predicted_count[count_valid] - truth_count[count_valid])
    group_unit = {
        "source_session_group_kfold": "source_session_id",
        "leave_one_collection_date_out": "collection_date",
        "leave_one_cow_out": "cow_id",
    }.get(validation_scheme, "validation_group")
    return {
        "validation_scheme": validation_scheme,
        "method": method,
        "videos": int(valid.sum()),
        "groups": group_unit,
        "rr_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "rr_mae": float(np.mean(np.abs(error))),
        "rr_rmse": float(np.sqrt(np.mean(error**2))),
        "count_mae": float(np.mean(count_error)),
        "exact_count": int(np.isclose(count_error, 0).sum()),
        "within_one_count": int((count_error <= 1 + 1e-9).sum()),
        "evaluation_note": note,
    }


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme, data in predictions.groupby("validation_scheme", sort=False):
        truth = data["truth_rr"].to_numpy(dtype=float)
        truth_count = data["truth_count"].to_numpy(dtype=float)
        duration = data["truth_duration_seconds"].to_numpy(dtype=float)
        for method, column, note in [
            (
                "zero_shot_frozen_baseline",
                "baseline_rr_bpm",
                "no_target_farm_labels",
            ),
            (
                "zero_shot_duration_normalized_offline_context",
                "windowed_offline_context_rr_bpm",
                "no_target_farm_labels_offline_uses_future_same_session_clips",
            ),
            (
                "zero_shot_duration_normalized_causal",
                "windowed_causal_rr_bpm",
                "no_target_farm_labels_current_and_prior_clips_only",
            ),
            (
                "zero_shot_duration_gated_calibration_free_color",
                "color_rr_bpm",
                "no_target_farm_labels_internal_selected_color_ensemble",
            ),
            (
                "few_shot_nested_ridge",
                "nested_ridge_rr_bpm",
                "target_farm_training_sessions_only_nested_group_validation",
            ),
            (
                "few_shot_fixed_huber",
                "fixed_huber_rr_bpm",
                "target_farm_training_sessions_only_group_validation",
            ),
            (
                "few_shot_nested_compact_offline_selector",
                "nested_compact_offline_rr_bpm",
                "offline_feature_set_and_regularization_selected_inside_each_training_fold",
            ),
            (
                "few_shot_nested_compact_causal_selector",
                "nested_compact_causal_rr_bpm",
                "causal_feature_set_and_regularization_selected_inside_each_training_fold",
            ),
            (
                "few_shot_single_color_huber",
                "color_huber_rr_bpm",
                "one_color_rr_feature_fixed_huber_group_validation",
            ),
            (
                "few_shot_ridge_huber_ensemble",
                "ridge_huber_ensemble_rr_bpm",
                "fixed_75_25_ensemble_of_group_validated_calibrators",
            ),
        ]:
            rows.append(
                metric_row(
                    scheme,
                    method,
                    truth,
                    data[column].to_numpy(dtype=float),
                    truth_count,
                    duration,
                    note,
                )
            )
    return pd.DataFrame(rows)


def bootstrap_metric_deltas(
    predictions: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    data = predictions[
        predictions["validation_scheme"].eq("source_session_group_kfold")
    ].copy()
    groups = sorted(data["source_session_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    comparisons = {
        "nested_compact_offline_minus_zero_shot_offline_context": (
            "nested_compact_offline_rr_bpm",
            "windowed_offline_context_rr_bpm",
        ),
        "nested_compact_causal_minus_zero_shot_causal": (
            "nested_compact_causal_rr_bpm",
            "windowed_causal_rr_bpm",
        ),
        "nested_compact_offline_minus_fixed_huber": (
            "nested_compact_offline_rr_bpm",
            "fixed_huber_rr_bpm",
        ),
        "color_huber_minus_zero_shot_color": (
            "color_huber_rr_bpm",
            "color_rr_bpm",
        ),
        "color_huber_minus_nested_compact_offline": (
            "color_huber_rr_bpm",
            "nested_compact_offline_rr_bpm",
        ),
    }

    def values(frame: pd.DataFrame, column: str) -> dict[str, float]:
        truth = frame["truth_rr"].to_numpy(dtype=float)
        prediction = frame[column].to_numpy(dtype=float)
        error = prediction - truth
        denominator = float(np.sum((truth - np.mean(truth)) ** 2))
        return {
            "rr_r2": float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else math.nan,
            "rr_mae": float(np.mean(np.abs(error))),
            "rr_rmse": float(np.sqrt(np.mean(error**2))),
        }

    replicate_values = {
        comparison: {metric: [] for metric in ["rr_r2", "rr_mae", "rr_rmse"]}
        for comparison in comparisons
    }
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        replicate = pd.concat(
            [data[data["source_session_id"].astype(str).eq(group)] for group in sampled],
            ignore_index=True,
        )
        for comparison, (candidate, reference) in comparisons.items():
            candidate_values = values(replicate, candidate)
            reference_values = values(replicate, reference)
            for metric in replicate_values[comparison]:
                replicate_values[comparison][metric].append(
                    candidate_values[metric] - reference_values[metric]
                )
    rows: list[dict[str, object]] = []
    for comparison, (candidate, reference) in comparisons.items():
        candidate_values = values(data, candidate)
        reference_values = values(data, reference)
        for metric, samples in replicate_values[comparison].items():
            finite = np.asarray(samples, dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": candidate_values[metric] - reference_values[metric],
                    "ci_low": float(np.quantile(finite, 0.025)),
                    "ci_high": float(np.quantile(finite, 0.975)),
                    "resamples": len(finite),
                    "cluster_unit": "source_session_id",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "# Few-Shot External Farm Calibration Probe",
        "",
        "Status: `development_adaptation_candidate_not_external_validation`",
        "",
        "The zero-shot duration-normalized offline and causal estimators are compared with calibration "
        "models trained only on other target-farm groups. Ridge regularization and a "
        "predefined low-dimensional feature set are selected inside each outer training "
        "fold. Cow ID, date, and source-session "
        "identity are used only for splitting, never as model features.",
        "",
        "| validation | method | R2 | MAE (bpm) | RMSE (bpm) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['validation_scheme']} | {row['method']} | {row['rr_r2']:.6f} | "
            f"{row['rr_mae']:.6f} | {row['rr_rmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Outer folds audited: `{len(selections)}`.",
            "",
            "## Cluster-Bootstrap Evidence",
            "",
            "Source-session clusters were resampled so clips from one long video were "
            "never treated as independent bootstrap units.",
            "",
            "| comparison | metric | estimate | 95% CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['estimate']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "The nested compact selector is compared separately with zero-shot offline-context "
            "and causal estimators. Its incremental advantage "
            "over fixed Huber is not statistically resolved, so the compact selector and "
            "Huber should be described as comparable low-capacity adaptation variants.",
            "The single-color Huber has the strongest point estimates among supervised "
            "calibrators, but it does not consistently exceed the optimized zero-shot color "
            "estimator across session, date, and cow holdouts. Its source-session bootstrap "
            "intervals versus both zero-shot color and nested offline compact include zero. "
            "It is therefore an optional parsimonious adaptation, not a superior main method.",
            "",
            "## Claim Boundary",
            "",
            "These are out-of-fold target-farm adaptation estimates, not independent "
            "external validation. They quantify the potential value of a small farm-specific "
            "calibration set. A future study must predefine calibration sessions and evaluate "
            "the frozen calibrator on newly collected animals/sessions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    assets = args.assets_dir.resolve()
    data = load_data(assets, args.external_root.resolve())
    session_predictions, session_selections = outer_predictions(
        data,
        data["source_session_id"],
        "source_session_group_kfold",
        args.outer_folds,
        args.inner_folds,
    )
    date_predictions, date_selections = outer_predictions(
        data,
        data["collection_date"],
        "leave_one_collection_date_out",
        args.outer_folds,
        args.inner_folds,
    )
    cow_predictions, cow_selections = outer_predictions(
        data,
        data["cow_id"],
        "leave_one_cow_out",
        args.outer_folds,
        args.inner_folds,
    )
    predictions = pd.concat(
        [session_predictions, date_predictions, cow_predictions], ignore_index=True
    )
    selections = pd.concat(
        [session_selections, date_selections, cow_selections], ignore_index=True
    )
    metrics = build_metrics(predictions)
    bootstrap = bootstrap_metric_deltas(
        predictions, args.bootstrap_resamples, args.random_state
    )
    predictions_path = assets / "paper_external_farm_calibration_predictions.csv"
    selections_path = assets / "paper_external_farm_calibration_fold_selections.csv"
    metrics_path = assets / "paper_external_farm_calibration_metrics.csv"
    report_path = assets / "paper_external_farm_calibration_report.md"
    bootstrap_path = assets / "paper_external_farm_calibration_bootstrap_ci.csv"
    predictions.to_csv(predictions_path, index=False)
    selections.to_csv(selections_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    write_report(report_path, metrics, selections, bootstrap)
    print(f"Saved farm calibration predictions: {predictions_path}")
    print(f"Saved farm calibration metrics: {metrics_path}")
    print(f"Saved farm calibration report: {report_path}")
    print(f"Saved farm calibration bootstrap CI: {bootstrap_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
