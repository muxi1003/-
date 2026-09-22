from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rr_quality_residual_validation import metric_dict


EXTERNAL_SPLITS = {"external", "holdout", "test"}
INTERNAL_SPLITS = {"internal", "train", "training", "validation", "val"}

PRIMARY_METHOD = "quality_residual_fixed"
DEFAULT_METHOD = "default_pipeline"
ALGORITHMIC_ENGINEERING_METHOD = "signal_aware_safe_gate_fixed"
ALGORITHMIC_ENGINEERING_MIN_VIDEOS = 104
EXTERNAL_SCOPE = "external_pool"

ABSOLUTE_ACCEPTANCE_RULES = [
    {
        "gate": "external RR R2 >= 0.90",
        "metric": "rr_r2",
        "operator": ">=",
        "threshold": 0.90,
        "claim_unlocked": "external absolute RR performance",
    },
    {
        "gate": "external MAE <= 2.5 bpm",
        "metric": "rr_mae",
        "operator": "<=",
        "threshold": 2.50,
        "claim_unlocked": "main external validation table",
    },
    {
        "gate": "external RMSE <= 4.0 bpm",
        "metric": "rr_rmse",
        "operator": "<=",
        "threshold": 4.00,
        "claim_unlocked": "main external validation table",
    },
    {
        "gate": "external within-one breath agreement >= 95%",
        "metric": "within_one_rate",
        "operator": ">=",
        "threshold": 0.95,
        "claim_unlocked": "agreement and reliability statement",
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen external/internal RR split metrics after external_test_split "
            "metadata is filled. Empty split metadata produces a readiness report but no "
            "external performance claim."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--method-freeze-csv", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--cluster-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--cluster-bootstrap-seed", type=int, default=20260710)
    return parser.parse_args()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def normalize_split(value: object) -> str:
    text = normalize_text(value).lower().replace("-", "_")
    if not text:
        return "missing"
    if text in EXTERNAL_SPLITS:
        return "external_pool"
    if text in INTERNAL_SPLITS:
        return "internal_pool"
    return "other_pool"


def default_metadata_csv(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets" / "paper_metadata_template.csv"


def default_method_freeze_csv(input_root: Path, corrected_prefix: str) -> Path:
    return input_root / f"{corrected_prefix}_paper_assets" / "paper_method_freeze_summary.csv"


def load_method_freeze(method_freeze_csv: Path) -> dict[str, object]:
    if not method_freeze_csv.exists():
        return {
            "method_freeze_id": "",
            "freeze_status": "missing",
            "method_freeze_locked": False,
            "method_freeze_csv": str(method_freeze_csv),
        }
    freeze = pd.read_csv(method_freeze_csv, dtype=str, keep_default_na=False)
    if freeze.empty:
        return {
            "method_freeze_id": "",
            "freeze_status": "empty",
            "method_freeze_locked": False,
            "method_freeze_csv": str(method_freeze_csv),
        }
    item = freeze.iloc[0]
    freeze_id = str(item.get("method_freeze_id", ""))
    freeze_status = str(item.get("freeze_status", ""))
    try:
        required_missing = int(float(item.get("required_files_missing", 1)))
    except (TypeError, ValueError):
        required_missing = 1
    locked = bool(freeze_id and freeze_status == "locked" and required_missing == 0)
    return {
        "method_freeze_id": freeze_id,
        "freeze_status": freeze_status,
        "method_freeze_locked": locked,
        "method_freeze_csv": str(method_freeze_csv),
    }


def metric_for_subset(
    label: str,
    subset: pd.DataFrame,
    rr_col: str,
    count_col: str,
    evaluation_note: str,
) -> dict[str, object]:
    row = metric_dict(
        label,
        pd.to_numeric(subset["truth_rr"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(subset[rr_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(subset["truth_count"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(subset[count_col], errors="coerce").to_numpy(dtype=float),
        evaluation_note=evaluation_note,
    )
    return row


def load_prediction_table(input_root: Path, output_prefix: str, corrected_prefix: str) -> pd.DataFrame:
    summary_path = input_root / f"{output_prefix}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    base = pd.read_csv(summary_path, dtype=str, keep_default_na=False)

    for path, columns in [
        (
            input_root / f"{corrected_prefix}_predictions.csv",
            ["video_id", "corrected_rr_bpm", "corrected_peaks"],
        ),
        (
            input_root / f"{output_prefix}_signal_consensus_predictions.csv",
            ["video_id", "signal_consensus_rr_bpm", "signal_consensus_peaks"],
        ),
        (
            input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv",
            ["video_id", "signal_aware_safe_final_rr_bpm", "signal_aware_safe_final_peaks"],
        ),
        (
            input_root / f"{output_prefix}_selective_rr_predictions.csv",
            [
                "video_id",
                "strict_auto_accept",
                "score_auto_accept",
                "selective_action",
                "selective_review_score",
            ],
        ),
    ]:
        extra = read_optional_csv(path)
        if extra.empty:
            continue
        available = [column for column in columns if column in extra.columns]
        base = base.merge(extra[available], on="video_id", how="left", validate="one_to_one")
    return base


def merge_metadata(predictions: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        metadata = pd.DataFrame({"video_id": predictions["video_id"].astype(str)})
        metadata["external_test_split"] = ""
    else:
        metadata = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
        if "video_id" not in metadata.columns:
            raise ValueError(f"Metadata CSV is missing required column: video_id: {metadata_csv}")
        if "external_test_split" not in metadata.columns:
            metadata["external_test_split"] = ""
    for column in ["cow_id", "source_session_id", "collection_date", "scene_id"]:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata["video_id"] = metadata["video_id"].astype(str)
    metadata = metadata[
        [
            "video_id",
            "external_test_split",
            "cow_id",
            "source_session_id",
            "collection_date",
            "scene_id",
        ]
    ].drop_duplicates("video_id")
    merged = predictions.merge(metadata, on="video_id", how="left", validate="one_to_one")
    merged["external_test_split"] = merged["external_test_split"].fillna("")
    merged["split_pool"] = merged["external_test_split"].map(normalize_split)
    merged["split_label"] = merged["external_test_split"].map(lambda value: normalize_text(value).lower())
    merged.loc[merged["split_label"] == "", "split_label"] = "missing"
    return merged


def build_readiness(predictions: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    total = int(len(predictions))
    split_values = predictions["external_test_split"].map(normalize_text)
    nonmissing = int((split_values != "").sum())
    external_n = int((predictions["split_pool"] == "external_pool").sum())
    internal_n = int((predictions["split_pool"] == "internal_pool").sum())
    missing_n = int((predictions["split_pool"] == "missing").sum())
    other_n = int((predictions["split_pool"] == "other_pool").sum())
    split_fill_action = (
        "Current split labels are complete but all rows are internal; keep these "
        "development rows internal and add/import independent method-frozen external rows."
        if nonmissing == total and external_n == 0
        else "Fill external_test_split for every video."
    )
    external_row_action = (
        "Collect or import independent external/holdout/test videos and label those "
        "new rows external; do not relabel current development videos as external."
    )
    rows = [
        {
            "check": "metadata file exists",
            "status": "PASS" if metadata_csv.exists() else "FAIL",
            "videos": total,
            "evidence": str(metadata_csv),
            "recommended_action": "Create/fill paper_metadata_template.csv.",
            "external_ready": False,
        },
        {
            "check": "external_test_split nonempty",
            "status": "PASS" if nonmissing == total else "FAIL",
            "videos": nonmissing,
            "evidence": f"nonmissing={nonmissing}/{total}",
            "recommended_action": split_fill_action,
            "external_ready": False,
        },
        {
            "check": "external rows present",
            "status": "PASS" if external_n > 0 else "FAIL",
            "videos": external_n,
            "evidence": f"external/holdout/test rows={external_n}",
            "recommended_action": external_row_action,
            "external_ready": False,
        },
        {
            "check": "internal rows present",
            "status": "PASS" if internal_n > 0 else "WARN",
            "videos": internal_n,
            "evidence": f"internal/train/validation rows={internal_n}",
            "recommended_action": "Keep threshold/model-development videos separate from the external subset.",
            "external_ready": False,
        },
        {
            "check": "unrecognized split labels absent",
            "status": "PASS" if other_n == 0 else "WARN",
            "videos": other_n,
            "evidence": f"other split labels={other_n}; missing={missing_n}",
            "recommended_action": "Use internal, train, validation, external, holdout, or test labels.",
            "external_ready": False,
        },
    ]
    ready = bool(nonmissing == total and external_n > 0 and other_n == 0)
    rows.append(
        {
            "check": "external split validation ready",
            "status": "PASS" if ready else "FAIL",
            "videos": external_n,
            "evidence": (
                f"ready={ready}; total={total}; internal={internal_n}; external={external_n}; "
                f"missing={missing_n}; other={other_n}"
            ),
            "recommended_action": (
                "Run this script again after independent external rows are present "
                "and split labels are complete."
            ),
            "external_ready": ready,
        }
    )
    return pd.DataFrame(rows)


def available_methods(predictions: pd.DataFrame) -> list[dict[str, str]]:
    methods = [
        {
            "method": "default_pipeline",
            "rr_col": "rr_bpm",
            "count_col": "peaks",
            "paper_use": "baseline",
        }
    ]
    if {"corrected_rr_bpm", "corrected_peaks"}.issubset(predictions.columns):
        methods.append(
            {
                "method": "quality_residual_fixed",
                "rr_col": "corrected_rr_bpm",
                "count_col": "corrected_peaks",
                "paper_use": "main_candidate",
            }
        )
    if {"signal_consensus_rr_bpm", "signal_consensus_peaks"}.issubset(predictions.columns):
        methods.append(
            {
                "method": "signal_consensus_fixed",
                "rr_col": "signal_consensus_rr_bpm",
                "count_col": "signal_consensus_peaks",
                "paper_use": "candidate_extension",
            }
        )
    if {"signal_aware_safe_final_rr_bpm", "signal_aware_safe_final_peaks"}.issubset(
        predictions.columns
    ):
        methods.append(
            {
                "method": "signal_aware_safe_gate_fixed",
                "rr_col": "signal_aware_safe_final_rr_bpm",
                "count_col": "signal_aware_safe_final_peaks",
                "paper_use": "secondary_safety_gated_candidate",
            }
        )
    return methods


def build_metrics(
    predictions: pd.DataFrame,
    readiness: pd.DataFrame,
    method_freeze: dict[str, object],
) -> pd.DataFrame:
    ready_row = readiness[readiness["check"] == "external split validation ready"].iloc[0]
    if not bool(ready_row["external_ready"]):
        return pd.DataFrame(
            columns=[
                "split_scope",
                "split_label",
                "method",
                "paper_use",
                "label",
                "videos",
                "rr_valid_videos",
                "rr_r2",
                "rr_pearson_r2",
                "rr_mae",
                "rr_rmse",
                "count_valid_videos",
                "count_mae",
                "exact_count",
                "within_one_count",
                "abs_count_error_ge2",
                "evaluation_note",
                "method_freeze_id",
                "freeze_status",
                "method_freeze_locked",
            ]
        )

    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, pd.Series]] = [
        ("all_labeled", "all_labeled", predictions["split_pool"] != "missing"),
        ("internal_pool", "internal_pool", predictions["split_pool"] == "internal_pool"),
        ("external_pool", "external_pool", predictions["split_pool"] == "external_pool"),
    ]
    for split_label in sorted(predictions["split_label"].unique()):
        if split_label == "missing":
            continue
        scopes.append(
            (
                "raw_split_label",
                split_label,
                predictions["split_label"].astype(str) == split_label,
            )
        )

    for split_scope, split_label, mask in scopes:
        subset = predictions[mask].copy()
        if subset.empty:
            continue
        for method in available_methods(predictions):
            row = metric_for_subset(
                f"{method['method']}__{split_label}",
                subset,
                method["rr_col"],
                method["count_col"],
                evaluation_note=f"{split_scope}:{split_label}",
            )
            row.update(
                {
                    "split_scope": split_scope,
                    "split_label": split_label,
                    "method": method["method"],
                    "paper_use": method["paper_use"],
                    "method_freeze_id": method_freeze["method_freeze_id"],
                    "freeze_status": method_freeze["freeze_status"],
                    "method_freeze_locked": method_freeze["method_freeze_locked"],
                }
            )
            rows.append(row)

        if "strict_auto_accept" in subset.columns and {
            "signal_consensus_rr_bpm",
            "signal_consensus_peaks",
        }.issubset(subset.columns):
            strict_mask = subset["strict_auto_accept"].astype(str).str.lower().isin(["true", "1", "yes"])
            strict_subset = subset[strict_mask]
            if not strict_subset.empty:
                row = metric_for_subset(
                    f"signal_consensus_strict_auto__{split_label}",
                    strict_subset,
                    "signal_consensus_rr_bpm",
                    "signal_consensus_peaks",
                    evaluation_note=f"{split_scope}:{split_label}:strict_auto_subset",
                )
                row.update(
                    {
                        "split_scope": split_scope,
                        "split_label": split_label,
                        "method": "signal_consensus_strict_auto",
                        "paper_use": "selective_reporting_subset",
                        "method_freeze_id": method_freeze["method_freeze_id"],
                        "freeze_status": method_freeze["freeze_status"],
                        "method_freeze_locked": method_freeze["method_freeze_locked"],
                    }
                )
                rows.append(row)

    columns = [
        "split_scope",
        "split_label",
        "method",
        "paper_use",
        "label",
        "videos",
        "rr_valid_videos",
        "rr_r2",
        "rr_pearson_r2",
        "rr_mae",
        "rr_rmse",
        "count_valid_videos",
        "count_mae",
        "exact_count",
        "within_one_count",
        "abs_count_error_ge2",
        "evaluation_note",
        "method_freeze_id",
        "freeze_status",
        "method_freeze_locked",
    ]
    metrics = pd.DataFrame(rows)
    return metrics[columns] if not metrics.empty else pd.DataFrame(columns=columns)


CLUSTER_BOOTSTRAP_COLUMNS = [
    "method",
    "paper_use",
    "split_scope",
    "cluster_variable",
    "clusters",
    "videos",
    "resamples_requested",
    "valid_rr_r2_resamples",
    "rr_r2",
    "rr_r2_ci_low",
    "rr_r2_ci_high",
    "rr_mae",
    "rr_mae_ci_low",
    "rr_mae_ci_high",
    "rr_rmse",
    "rr_rmse_ci_low",
    "rr_rmse_ci_high",
    "exact_rate",
    "exact_rate_ci_low",
    "exact_rate_ci_high",
    "within_one_rate",
    "within_one_rate_ci_low",
    "within_one_rate_ci_high",
    "method_freeze_id",
    "freeze_status",
    "method_freeze_locked",
    "evaluation_note",
]


def rr_count_statistics(
    truth_rr: np.ndarray,
    predicted_rr: np.ndarray,
    truth_count: np.ndarray,
    predicted_count: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    rr_truth = truth_rr[indices]
    rr_predicted = predicted_rr[indices]
    rr_valid = np.isfinite(rr_truth) & np.isfinite(rr_predicted)
    if rr_valid.any():
        errors = rr_predicted[rr_valid] - rr_truth[rr_valid]
        sse = float(np.sum(errors**2))
        centered = rr_truth[rr_valid] - float(np.mean(rr_truth[rr_valid]))
        sst = float(np.sum(centered**2))
        rr_r2 = 1.0 - sse / sst if sst > 0 else math.nan
        rr_mae = float(np.mean(np.abs(errors)))
        rr_rmse = float(np.sqrt(np.mean(errors**2)))
    else:
        rr_r2 = rr_mae = rr_rmse = math.nan
    count_truth = truth_count[indices]
    count_predicted = predicted_count[indices]
    count_valid = np.isfinite(count_truth) & np.isfinite(count_predicted)
    if count_valid.any():
        count_error = np.abs(count_predicted[count_valid] - count_truth[count_valid])
        exact_rate = float(np.mean(count_error == 0))
        within_one_rate = float(np.mean(count_error <= 1))
    else:
        exact_rate = within_one_rate = math.nan
    return {
        "rr_r2": rr_r2,
        "rr_mae": rr_mae,
        "rr_rmse": rr_rmse,
        "exact_rate": exact_rate,
        "within_one_rate": within_one_rate,
    }


def percentile_interval(values: list[float]) -> tuple[float, float, int]:
    valid = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if valid.size == 0:
        return math.nan, math.nan, 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), int(valid.size)


def cluster_bootstrap_row(
    subset: pd.DataFrame,
    method: dict[str, str],
    cluster_variable: str,
    *,
    resamples: int,
    seed: int,
    method_freeze: dict[str, object],
) -> dict[str, object] | None:
    if cluster_variable not in subset.columns:
        return None
    working = subset.copy()
    working[cluster_variable] = working[cluster_variable].map(normalize_text)
    working = working[working[cluster_variable].ne("")].reset_index(drop=True)
    clusters = sorted(working[cluster_variable].unique())
    if len(clusters) < 2 or working.empty:
        return None
    truth_rr = pd.to_numeric(working["truth_rr"], errors="coerce").to_numpy(dtype=float)
    predicted_rr = pd.to_numeric(working[method["rr_col"]], errors="coerce").to_numpy(dtype=float)
    truth_count = pd.to_numeric(working["truth_count"], errors="coerce").to_numpy(dtype=float)
    predicted_count = pd.to_numeric(working[method["count_col"]], errors="coerce").to_numpy(dtype=float)
    cluster_indices = {
        cluster: np.flatnonzero(working[cluster_variable].to_numpy() == cluster)
        for cluster in clusters
    }
    all_indices = np.arange(len(working), dtype=int)
    point = rr_count_statistics(
        truth_rr,
        predicted_rr,
        truth_count,
        predicted_count,
        all_indices,
    )
    rng = np.random.default_rng(seed)
    sampled_metrics = {key: [] for key in point}
    for _ in range(max(int(resamples), 0)):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = np.concatenate(
            [cluster_indices[str(cluster)] for cluster in sampled_clusters]
        )
        values = rr_count_statistics(
            truth_rr,
            predicted_rr,
            truth_count,
            predicted_count,
            sampled_indices,
        )
        for key, value in values.items():
            sampled_metrics[key].append(value)
    intervals = {
        key: percentile_interval(values) for key, values in sampled_metrics.items()
    }
    return {
        "method": method["method"],
        "paper_use": method["paper_use"],
        "split_scope": EXTERNAL_SCOPE,
        "cluster_variable": cluster_variable,
        "clusters": len(clusters),
        "videos": len(working),
        "resamples_requested": int(resamples),
        "valid_rr_r2_resamples": intervals["rr_r2"][2],
        "rr_r2": point["rr_r2"],
        "rr_r2_ci_low": intervals["rr_r2"][0],
        "rr_r2_ci_high": intervals["rr_r2"][1],
        "rr_mae": point["rr_mae"],
        "rr_mae_ci_low": intervals["rr_mae"][0],
        "rr_mae_ci_high": intervals["rr_mae"][1],
        "rr_rmse": point["rr_rmse"],
        "rr_rmse_ci_low": intervals["rr_rmse"][0],
        "rr_rmse_ci_high": intervals["rr_rmse"][1],
        "exact_rate": point["exact_rate"],
        "exact_rate_ci_low": intervals["exact_rate"][0],
        "exact_rate_ci_high": intervals["exact_rate"][1],
        "within_one_rate": point["within_one_rate"],
        "within_one_rate_ci_low": intervals["within_one_rate"][0],
        "within_one_rate_ci_high": intervals["within_one_rate"][1],
        "method_freeze_id": method_freeze["method_freeze_id"],
        "freeze_status": method_freeze["freeze_status"],
        "method_freeze_locked": method_freeze["method_freeze_locked"],
        "evaluation_note": (
            "Percentile 95% CI from cluster bootstrap; clusters sampled with "
            "replacement and all clips from each sampled cluster retained."
        ),
    }


def build_cluster_bootstrap(
    predictions: pd.DataFrame,
    readiness: pd.DataFrame,
    method_freeze: dict[str, object],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    ready_row = readiness[readiness["check"] == "external split validation ready"].iloc[0]
    if not bool(ready_row["external_ready"]):
        return pd.DataFrame(columns=CLUSTER_BOOTSTRAP_COLUMNS)
    external = predictions[predictions["split_pool"].astype(str).eq(EXTERNAL_SCOPE)].copy()
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(available_methods(predictions)):
        for cluster_index, cluster_variable in enumerate(["source_session_id", "cow_id"]):
            row = cluster_bootstrap_row(
                external,
                method,
                cluster_variable,
                resamples=resamples,
                seed=seed + method_index * 100 + cluster_index,
                method_freeze=method_freeze,
            )
            if row is not None:
                rows.append(row)
    return (
        pd.DataFrame(rows, columns=CLUSTER_BOOTSTRAP_COLUMNS)
        if rows
        else pd.DataFrame(columns=CLUSTER_BOOTSTRAP_COLUMNS)
    )


def pass_threshold(value: float, operator: str, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def row_value(row: pd.Series, metric: str) -> float:
    if metric == "within_one_rate":
        valid = float(row["count_valid_videos"])
        if valid <= 0:
            return math.nan
        return float(row["within_one_count"]) / valid
    return float(row[metric])


def absolute_acceptance_rows(
    method_row: pd.Series,
    *,
    method: str,
    gate_prefix: str,
    claim_unlocked: str,
    method_freeze: dict[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    statuses: list[str] = []
    freeze_locked = bool(method_freeze["method_freeze_locked"])
    for rule in ABSOLUTE_ACCEPTANCE_RULES:
        value = row_value(method_row, str(rule["metric"]))
        passed = pass_threshold(value, str(rule["operator"]), float(rule["threshold"]))
        status = "PASS" if passed else "FAIL"
        statuses.append(status)
        rows.append(
            {
                "gate": f"{gate_prefix}: {rule['gate']}",
                "status": status,
                "method": method,
                "split_scope": EXTERNAL_SCOPE,
                "metric": rule["metric"],
                "value": value,
                "threshold": f"{rule['operator']} {rule['threshold']}",
                "evidence": (
                    f"external videos={int(method_row['videos'])}; "
                    f"{rule['metric']}={value:.6f}"
                ),
                "claim_unlocked": claim_unlocked if passed else "",
                "recommended_action": (
                    "Report as frozen external validation for this method."
                    if passed
                    else "Do not claim this external gate; inspect failure cases or collect more data."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
    return rows, statuses


def read_required_relative_n(input_root: Path, corrected_prefix: str) -> int | None:
    sample_path = (
        input_root
        / f"{corrected_prefix}_paper_assets"
        / "paper_external_validation_sample_plan.csv"
    )
    if not sample_path.exists():
        return None
    sample = pd.read_csv(sample_path)
    match = sample[sample["metric"].astype(str) == "Delta RR R2 versus default"]
    if match.empty:
        return None
    try:
        return int(float(match.iloc[0]["estimated_min_external_videos"]))
    except (TypeError, ValueError):
        return None


def build_acceptance(
    readiness: pd.DataFrame,
    metrics: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    input_root: Path,
    corrected_prefix: str,
    method_freeze: dict[str, object],
) -> pd.DataFrame:
    ready = bool(
        readiness.loc[
            readiness["check"] == "external split validation ready", "external_ready"
        ].iloc[0]
    )
    rows: list[dict[str, object]] = []
    freeze_locked = bool(method_freeze["method_freeze_locked"])
    rows.append(
        {
            "gate": "external method freeze locked",
            "status": "PASS" if freeze_locked else "FAIL",
            "method": PRIMARY_METHOD,
            "split_scope": EXTERNAL_SCOPE,
            "metric": "method_freeze",
            "value": math.nan,
            "threshold": "freeze_status == locked",
            "evidence": (
                f"method_freeze_id={method_freeze['method_freeze_id']}; "
                f"freeze_status={method_freeze['freeze_status']}; "
                f"method_freeze_locked={freeze_locked}"
            ),
            "claim_unlocked": "frozen external validation method" if freeze_locked else "",
            "recommended_action": (
                "Use this freeze ID in external validation reporting."
                if freeze_locked
                else "Run scripts/freeze_rr_external_method.py before external validation."
            ),
            "method_freeze_id": method_freeze["method_freeze_id"],
            "freeze_status": method_freeze["freeze_status"],
            "method_freeze_locked": freeze_locked,
        }
    )
    if not ready:
        rows.append(
            {
                "gate": "external split validation ready",
                "status": "FAIL",
                "method": "",
                "split_scope": EXTERNAL_SCOPE,
                "metric": "",
                "value": math.nan,
                "threshold": "",
                "evidence": "external_test_split is not fully labeled or no external rows are present",
                "claim_unlocked": "",
                "recommended_action": "Fill external_test_split before reporting external metrics.",
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        rows.append(
            {
                "gate": "external performance claim allowed",
                "status": "FAIL",
                "method": PRIMARY_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "",
                "value": math.nan,
                "threshold": "",
                "evidence": (
                    "no frozen external metrics were produced; "
                    f"method_freeze_id={method_freeze['method_freeze_id']}; "
                    f"freeze_status={method_freeze['freeze_status']}"
                ),
                "claim_unlocked": "",
                "recommended_action": "Rerun this script after the split labels are ready.",
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        rows.append(
            {
                "gate": "algorithmic engineering external claim allowed",
                "status": "FAIL",
                "method": ALGORITHMIC_ENGINEERING_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "",
                "value": math.nan,
                "threshold": (
                    f"external_n >= {ALGORITHMIC_ENGINEERING_MIN_VIDEOS} + "
                    "method frozen + all algorithmic absolute gates PASS"
                ),
                "evidence": (
                    "no frozen external metrics were produced for the "
                    "algorithmic engineering route"
                ),
                "claim_unlocked": "",
                "recommended_action": (
                    "Collect/import independent external videos, fill manual RR "
                    "truth, and rerun this script before using safe-gate as the "
                    "algorithmic Q2 route."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        return pd.DataFrame(rows)

    external_primary = metrics[
        (metrics["split_scope"].astype(str) == EXTERNAL_SCOPE)
        & (metrics["method"].astype(str) == PRIMARY_METHOD)
    ]
    if external_primary.empty:
        rows.append(
            {
                "gate": "primary external method metrics present",
                "status": "FAIL",
                "method": PRIMARY_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "",
                "value": math.nan,
                "threshold": "",
                "evidence": f"No {PRIMARY_METHOD} metrics found for external pool.",
                "claim_unlocked": "",
                "recommended_action": "Ensure corrected predictions exist before external evaluation.",
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        return pd.DataFrame(rows)

    primary_row = external_primary.iloc[0]
    absolute_gate_statuses: list[str] = []
    for rule in ABSOLUTE_ACCEPTANCE_RULES:
        value = row_value(primary_row, str(rule["metric"]))
        passed = pass_threshold(value, str(rule["operator"]), float(rule["threshold"]))
        absolute_gate_statuses.append("PASS" if passed else "FAIL")
        rows.append(
            {
                "gate": rule["gate"],
                "status": "PASS" if passed else "FAIL",
                "method": PRIMARY_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": rule["metric"],
                "value": value,
                "threshold": f"{rule['operator']} {rule['threshold']}",
                "evidence": (
                    f"external videos={int(primary_row['videos'])}; "
                    f"{rule['metric']}={value:.6f}"
                ),
                "claim_unlocked": rule["claim_unlocked"] if passed else "",
                "recommended_action": (
                    "Report as external absolute validation."
                    if passed
                    else "Do not claim this external gate; inspect failure cases or collect more data."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )

    primary_cluster = cluster_bootstrap[
        cluster_bootstrap["method"].astype(str).eq(PRIMARY_METHOD)
        & cluster_bootstrap["cluster_variable"].astype(str).eq("source_session_id")
    ] if not cluster_bootstrap.empty else pd.DataFrame()
    primary_cluster_pass = not primary_cluster.empty
    rows.append(
        {
            "gate": "source-session clustered uncertainty available",
            "status": "PASS" if primary_cluster_pass else "FAIL",
            "method": PRIMARY_METHOD,
            "split_scope": EXTERNAL_SCOPE,
            "metric": "cluster_bootstrap",
            "value": (
                float(primary_cluster.iloc[0]["clusters"])
                if primary_cluster_pass
                else math.nan
            ),
            "threshold": "source_session_id clusters >= 2 and cluster bootstrap produced",
            "evidence": (
                f"source_sessions={int(primary_cluster.iloc[0]['clusters'])}; "
                f"resamples={int(primary_cluster.iloc[0]['resamples_requested'])}"
                if primary_cluster_pass
                else "source-session clustered confidence intervals are missing"
            ),
            "claim_unlocked": "cluster-aware external uncertainty" if primary_cluster_pass else "",
            "recommended_action": (
                "Report clip-level estimates with source-session clustered confidence intervals."
                if primary_cluster_pass
                else "Populate source_session_id and regenerate clustered confidence intervals."
            ),
            "method_freeze_id": method_freeze["method_freeze_id"],
            "freeze_status": method_freeze["freeze_status"],
            "method_freeze_locked": freeze_locked,
        }
    )
    absolute_allowed = ready and freeze_locked and primary_cluster_pass and all(
        status == "PASS" for status in absolute_gate_statuses
    )
    rows.append(
        {
            "gate": "external performance claim allowed",
            "status": "PASS" if absolute_allowed else "FAIL",
            "method": PRIMARY_METHOD,
            "split_scope": EXTERNAL_SCOPE,
            "metric": "absolute_gate_bundle",
            "value": math.nan,
            "threshold": (
                "split ready + method frozen + all absolute gates PASS + "
                "source-session clustered uncertainty available"
            ),
            "evidence": (
                f"split_ready={ready}; method_freeze_locked={freeze_locked}; "
                f"absolute_gate_statuses={','.join(absolute_gate_statuses)}; "
                f"source_session_cluster_bootstrap={primary_cluster_pass}; "
                f"method_freeze_id={method_freeze['method_freeze_id']}"
            ),
            "claim_unlocked": "external absolute RR performance" if absolute_allowed else "",
            "recommended_action": (
                "External absolute-performance claim can be written with this freeze ID."
                if absolute_allowed
                else "Do not claim external performance until split, freeze, and absolute gates all pass."
            ),
            "method_freeze_id": method_freeze["method_freeze_id"],
            "freeze_status": method_freeze["freeze_status"],
            "method_freeze_locked": freeze_locked,
        }
    )

    external_default = metrics[
        (metrics["split_scope"].astype(str) == EXTERNAL_SCOPE)
        & (metrics["method"].astype(str) == DEFAULT_METHOD)
    ]
    external_algorithmic = metrics[
        (metrics["split_scope"].astype(str) == EXTERNAL_SCOPE)
        & (metrics["method"].astype(str) == ALGORITHMIC_ENGINEERING_METHOD)
    ]
    algorithmic_allowed = False
    if external_algorithmic.empty:
        rows.append(
            {
                "gate": "algorithmic engineering method metrics present",
                "status": "FAIL",
                "method": ALGORITHMIC_ENGINEERING_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "",
                "value": math.nan,
                "threshold": "safe-gate external metrics present",
                "evidence": f"No {ALGORITHMIC_ENGINEERING_METHOD} metrics found for external pool.",
                "claim_unlocked": "",
                "recommended_action": (
                    "Ensure signal-aware safe-gate predictions exist before using "
                    "the algorithmic engineering Q2 route."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
    else:
        algorithmic_row = external_algorithmic.iloc[0]
        algorithmic_rows, algorithmic_statuses = absolute_acceptance_rows(
            algorithmic_row,
            method=ALGORITHMIC_ENGINEERING_METHOD,
            gate_prefix="algorithmic engineering",
            claim_unlocked="algorithmic-engineering external RR performance",
            method_freeze=method_freeze,
        )
        rows.extend(algorithmic_rows)
        algorithmic_n = int(algorithmic_row["videos"])
        sample_pass = algorithmic_n >= ALGORITHMIC_ENGINEERING_MIN_VIDEOS
        rows.append(
            {
                "gate": "algorithmic engineering external sample size",
                "status": "PASS" if sample_pass else "FAIL",
                "method": ALGORITHMIC_ENGINEERING_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "videos",
                "value": float(algorithmic_n),
                "threshold": f">= {ALGORITHMIC_ENGINEERING_MIN_VIDEOS}",
                "evidence": (
                    f"external_n={algorithmic_n}; required_n={ALGORITHMIC_ENGINEERING_MIN_VIDEOS}"
                ),
                "claim_unlocked": "algorithmic Q2 external validation sample size"
                if sample_pass
                else "",
                "recommended_action": (
                    "Sample-size gate supports the algorithmic Q2 route."
                    if sample_pass
                    else "Collect more independent external videos before using the algorithmic Q2 route."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        if external_default.empty:
            exact_pass = False
            exact_evidence = "default external metrics are missing"
            exact_status = "FAIL"
            exact_value = math.nan
        else:
            default_row_for_algorithmic = external_default.iloc[0]
            exact_value = float(algorithmic_row["exact_count"]) - float(
                default_row_for_algorithmic["exact_count"]
            )
            exact_pass = exact_value >= 0
            exact_status = "PASS" if exact_pass else "FAIL"
            exact_evidence = (
                f"algorithmic_exact={int(algorithmic_row['exact_count'])}/"
                f"{int(algorithmic_row['count_valid_videos'])}; "
                f"default_exact={int(default_row_for_algorithmic['exact_count'])}/"
                f"{int(default_row_for_algorithmic['count_valid_videos'])}; "
                f"delta_exact={exact_value:.0f}"
            )
        rows.append(
            {
                "gate": "algorithmic exact count not worse than default",
                "status": exact_status,
                "method": f"{ALGORITHMIC_ENGINEERING_METHOD}_vs_{DEFAULT_METHOD}",
                "split_scope": EXTERNAL_SCOPE,
                "metric": "delta_exact_count",
                "value": exact_value,
                "threshold": ">= 0",
                "evidence": exact_evidence,
                "claim_unlocked": "safe-gate exact-count agreement not worse than default"
                if exact_pass
                else "",
                "recommended_action": (
                    "Exact-count comparison supports safe-gate external reporting."
                    if exact_pass
                    else "Keep safe-gate as secondary unless exact-count comparison is acceptable."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        algorithmic_cluster = cluster_bootstrap[
            cluster_bootstrap["method"].astype(str).eq(ALGORITHMIC_ENGINEERING_METHOD)
            & cluster_bootstrap["cluster_variable"].astype(str).eq("source_session_id")
        ] if not cluster_bootstrap.empty else pd.DataFrame()
        algorithmic_cluster_pass = not algorithmic_cluster.empty
        rows.append(
            {
                "gate": "algorithmic engineering source-session uncertainty available",
                "status": "PASS" if algorithmic_cluster_pass else "FAIL",
                "method": ALGORITHMIC_ENGINEERING_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "cluster_bootstrap",
                "value": (
                    float(algorithmic_cluster.iloc[0]["clusters"])
                    if algorithmic_cluster_pass
                    else math.nan
                ),
                "threshold": "source_session_id clusters >= 2 and cluster bootstrap produced",
                "evidence": (
                    f"source_sessions={int(algorithmic_cluster.iloc[0]['clusters'])}; "
                    f"resamples={int(algorithmic_cluster.iloc[0]['resamples_requested'])}"
                    if algorithmic_cluster_pass
                    else "safe-gate source-session confidence intervals are missing"
                ),
                "claim_unlocked": "cluster-aware algorithmic uncertainty"
                if algorithmic_cluster_pass
                else "",
                "recommended_action": (
                    "Report safe-gate estimates with clustered confidence intervals."
                    if algorithmic_cluster_pass
                    else "Populate source_session_id and regenerate safe-gate clustered confidence intervals."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
        algorithmic_allowed = bool(
            ready
            and freeze_locked
            and sample_pass
            and exact_pass
            and algorithmic_cluster_pass
            and all(status == "PASS" for status in algorithmic_statuses)
        )
        rows.append(
            {
                "gate": "algorithmic engineering external claim allowed",
                "status": "PASS" if algorithmic_allowed else "FAIL",
                "method": ALGORITHMIC_ENGINEERING_METHOD,
                "split_scope": EXTERNAL_SCOPE,
                "metric": "algorithmic_gate_bundle",
                "value": math.nan,
                "threshold": (
                    f"external_n >= {ALGORITHMIC_ENGINEERING_MIN_VIDEOS} + "
                    "method frozen + all algorithmic gates PASS + exact not worse than "
                    "default + source-session clustered uncertainty available"
                ),
                "evidence": (
                    f"split_ready={ready}; method_freeze_locked={freeze_locked}; "
                    f"sample_pass={sample_pass}; exact_not_worse={exact_pass}; "
                    f"source_session_cluster_bootstrap={algorithmic_cluster_pass}; "
                    f"algorithmic_gate_statuses={','.join(algorithmic_statuses)}"
                ),
                "claim_unlocked": "algorithmic-engineering Q2 external RR validation"
                if algorithmic_allowed
                else "",
                "recommended_action": (
                    "Safe-gate can be written as the algorithmic engineering external result."
                    if algorithmic_allowed
                    else "Do not use safe-gate as the Q2 engineering external result until this bundle passes."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
    required_relative_n = read_required_relative_n(input_root, corrected_prefix)
    external_n = int(primary_row["videos"])
    if external_default.empty:
        rows.append(
            {
                "gate": "paired relative-improvement evidence",
                "status": "WARN",
                "method": f"{PRIMARY_METHOD}_vs_{DEFAULT_METHOD}",
                "split_scope": EXTERNAL_SCOPE,
                "metric": "delta_rr_r2",
                "value": math.nan,
                "threshold": f"recommended videos >= {required_relative_n}"
                if required_relative_n is not None
                else "recommended n unavailable",
                "evidence": "default external metrics are missing",
                "claim_unlocked": "",
                "recommended_action": "Keep relative-improvement wording internal/candidate only.",
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )
    else:
        default_row = external_default.iloc[0]
        delta_r2 = float(primary_row["rr_r2"]) - float(default_row["rr_r2"])
        delta_mae = float(primary_row["rr_mae"]) - float(default_row["rr_mae"])
        enough_n = required_relative_n is not None and external_n >= required_relative_n
        direction_ok = delta_r2 > 0 and delta_mae < 0
        status = "PASS" if enough_n and direction_ok else "WARN"
        rows.append(
            {
                "gate": "paired relative-improvement evidence",
                "status": status,
                "method": f"{PRIMARY_METHOD}_vs_{DEFAULT_METHOD}",
                "split_scope": EXTERNAL_SCOPE,
                "metric": "delta_rr_r2;delta_rr_mae",
                "value": delta_r2,
                "threshold": f"recommended videos >= {required_relative_n}"
                if required_relative_n is not None
                else "recommended n unavailable",
                "evidence": (
                    f"external_n={external_n}; delta_rr_r2={delta_r2:.6f}; "
                    f"delta_mae={delta_mae:.6f}"
                ),
                "claim_unlocked": "statistically supported relative improvement"
                if status == "PASS"
                else "",
                "recommended_action": (
                    "Relative-improvement wording can be considered if paired CI also supports it."
                    if status == "PASS"
                    else "Write relative improvement as internal/candidate evidence unless paired external CI is adequate."
                ),
                "method_freeze_id": method_freeze["method_freeze_id"],
                "freeze_status": method_freeze["freeze_status"],
                "method_freeze_locked": freeze_locked,
            }
        )

    rows.append(
        {
            "gate": "truth-calibrated excluded from external performance",
            "status": "PASS",
            "method": "truth_calibrated",
            "split_scope": EXTERNAL_SCOPE,
            "metric": "leakage_guard",
            "value": math.nan,
            "threshold": "not allowed as main method",
            "evidence": "external metrics are computed from default/corrected/signal-consensus predictions only",
            "claim_unlocked": "upper-bound diagnostic only",
            "recommended_action": "Keep truth-calibrated RR out of external performance claims.",
            "method_freeze_id": method_freeze["method_freeze_id"],
            "freeze_status": method_freeze["freeze_status"],
            "method_freeze_locked": freeze_locked,
        }
    )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    readiness: pd.DataFrame,
    metrics: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    acceptance: pd.DataFrame,
    predictions_path: Path,
    method_freeze: dict[str, object],
) -> None:
    ready = bool(
        readiness.loc[
            readiness["check"] == "external split validation ready", "external_ready"
        ].iloc[0]
    )
    lines = [
        "# External Split Validation Report",
        "",
        f"Predictions with split labels: `{predictions_path}`",
        "",
        f"Method freeze ID: `{method_freeze['method_freeze_id']}`",
        "",
        f"Method freeze status: `{method_freeze['freeze_status']}`",
        "",
        f"External split validation ready: `{ready}`",
        "",
        "## Readiness",
        "",
        "| check | status | videos | evidence | recommended_action |",
        "|---|---|---:|---|---|",
    ]
    for _, row in readiness.iterrows():
        lines.append(
            "| {check} | {status} | {videos} | {evidence} | {recommended_action} |".format(
                check=row["check"],
                status=row["status"],
                videos=row["videos"],
                evidence=str(row["evidence"]).replace("|", "/"),
                recommended_action=str(row["recommended_action"]).replace("|", "/"),
            )
            )
    lines.extend(["", "## Metrics", ""])
    if metrics.empty:
        lines.append(
            "No external split metrics were produced because `external_test_split` is not ready."
        )
    else:
        lines.append("| split | method | freeze_id | videos | rr_r2 | rr_mae | rr_rmse | exact_count |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in metrics.iterrows():
            lines.append(
                "| {split} | {method} | {freeze_id} | {videos} | {rr_r2:.4f} | {rr_mae:.4f} | {rr_rmse:.4f} | {exact}/{valid} |".format(
                    split=row["split_label"],
                    method=row["method"],
                    freeze_id=row.get("method_freeze_id", ""),
                    videos=int(row["videos"]),
                    rr_r2=float(row["rr_r2"]) if np.isfinite(float(row["rr_r2"])) else math.nan,
                    rr_mae=float(row["rr_mae"]) if np.isfinite(float(row["rr_mae"])) else math.nan,
                    rr_rmse=float(row["rr_rmse"]) if np.isfinite(float(row["rr_rmse"])) else math.nan,
                    exact=int(row["exact_count"]),
                    valid=int(row["count_valid_videos"]),
                )
            )
    lines.extend(["", "## Clustered Uncertainty", ""])
    if cluster_bootstrap.empty:
        lines.append(
            "No clustered confidence intervals were produced because external metrics "
            "or source-session/cow grouping metadata are not ready."
        )
    else:
        lines.append(
            "| method | cluster | clusters | videos | rr_r2 (95% CI) | rr_mae (95% CI) | exact_rate (95% CI) |"
        )
        lines.append("|---|---|---:|---:|---|---|---|")
        for _, row in cluster_bootstrap.iterrows():
            lines.append(
                "| {method} | {cluster} | {clusters} | {videos} | {r2:.4f} "
                "({r2_low:.4f}, {r2_high:.4f}) | {mae:.4f} ({mae_low:.4f}, "
                "{mae_high:.4f}) | {exact:.4f} ({exact_low:.4f}, {exact_high:.4f}) |".format(
                    method=row["method"],
                    cluster=row["cluster_variable"],
                    clusters=int(row["clusters"]),
                    videos=int(row["videos"]),
                    r2=float(row["rr_r2"]),
                    r2_low=float(row["rr_r2_ci_low"]),
                    r2_high=float(row["rr_r2_ci_high"]),
                    mae=float(row["rr_mae"]),
                    mae_low=float(row["rr_mae_ci_low"]),
                    mae_high=float(row["rr_mae_ci_high"]),
                    exact=float(row["exact_rate"]),
                    exact_low=float(row["exact_rate_ci_low"]),
                    exact_high=float(row["exact_rate_ci_high"]),
                )
            )
    lines.extend(["", "## Acceptance Gates", ""])
    if acceptance.empty:
        lines.append("No acceptance gates were generated.")
    else:
        lines.append("| gate | status | method | freeze_id | value | threshold | evidence |")
        lines.append("|---|---|---|---|---:|---|---|")
        for _, row in acceptance.iterrows():
            value = row.get("value", math.nan)
            try:
                value_text = "" if not np.isfinite(float(value)) else f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_text = ""
            lines.append(
                "| {gate} | {status} | {method} | {freeze_id} | {value} | {threshold} | {evidence} |".format(
                    gate=str(row["gate"]).replace("|", "/"),
                    status=row["status"],
                    method=str(row["method"]).replace("|", "/"),
                    freeze_id=str(row.get("method_freeze_id", "")).replace("|", "/"),
                    value=value_text,
                    threshold=str(row["threshold"]).replace("|", "/"),
                    evidence=str(row["evidence"]).replace("|", "/"),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Only rows marked as `external`, `holdout`, or `test` in `external_test_split` should be described as frozen external-test evidence. These metrics remain invalid for Q2+ claims until the split labels are filled before final threshold/model selection.",
            "",
            "The `algorithmic engineering external claim allowed` gate applies to the conservative signal-aware safe gate and supports only the no-heat/no-manual-quality engineering route. It does not unlock THI, heat-stress, head-motion, occlusion, or nostril-visibility stratified claims.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    metadata_csv = args.metadata_csv or default_metadata_csv(input_root, args.corrected_prefix)
    method_freeze_csv = args.method_freeze_csv or default_method_freeze_csv(
        input_root, args.corrected_prefix
    )
    method_freeze = load_method_freeze(method_freeze_csv)

    predictions = load_prediction_table(input_root, args.output_prefix, args.corrected_prefix)
    predictions = merge_metadata(predictions, metadata_csv)
    predictions["method_freeze_id"] = method_freeze["method_freeze_id"]
    predictions["freeze_status"] = method_freeze["freeze_status"]
    predictions["method_freeze_locked"] = method_freeze["method_freeze_locked"]
    readiness = build_readiness(predictions, metadata_csv)
    readiness["method_freeze_id"] = method_freeze["method_freeze_id"]
    readiness["freeze_status"] = method_freeze["freeze_status"]
    readiness["method_freeze_locked"] = method_freeze["method_freeze_locked"]
    metrics = build_metrics(predictions, readiness, method_freeze)
    cluster_bootstrap = build_cluster_bootstrap(
        predictions,
        readiness,
        method_freeze,
        resamples=args.cluster_bootstrap_resamples,
        seed=args.cluster_bootstrap_seed,
    )
    acceptance = build_acceptance(
        readiness,
        metrics,
        cluster_bootstrap,
        input_root,
        args.corrected_prefix,
        method_freeze,
    )

    predictions_path = input_root / f"{args.output_prefix}_external_split_predictions.csv"
    readiness_path = input_root / f"{args.output_prefix}_external_split_readiness.csv"
    metrics_path = input_root / f"{args.output_prefix}_external_split_metrics.csv"
    cluster_bootstrap_path = (
        input_root / f"{args.output_prefix}_external_split_cluster_bootstrap.csv"
    )
    acceptance_path = input_root / f"{args.output_prefix}_external_split_acceptance.csv"
    report_path = input_root / f"{args.output_prefix}_external_split_report.md"

    predictions.to_csv(predictions_path, index=False)
    readiness.to_csv(readiness_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    cluster_bootstrap.to_csv(cluster_bootstrap_path, index=False)
    acceptance.to_csv(acceptance_path, index=False)
    write_report(
        report_path,
        readiness,
        metrics,
        cluster_bootstrap,
        acceptance,
        predictions_path,
        method_freeze,
    )

    print(f"Saved external split predictions: {predictions_path}")
    print(f"Saved external split readiness: {readiness_path}")
    print(f"Saved external split metrics: {metrics_path}")
    print(f"Saved external split cluster bootstrap: {cluster_bootstrap_path}")
    print(f"Saved external split acceptance: {acceptance_path}")
    print(f"Saved external split report: {report_path}")
    print(readiness.to_string(index=False))
    if metrics.empty:
        print("No metrics produced because external split validation is not ready.")
    else:
        print(metrics.to_string(index=False))
    print("\nAcceptance gates:")
    print(acceptance.to_string(index=False))


if __name__ == "__main__":
    main()
