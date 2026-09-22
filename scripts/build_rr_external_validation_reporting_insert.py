from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXTERNAL_METHODS = [
    ("default_pipeline", "Default pipeline"),
    ("quality_residual_fixed", "Quality-aware residual correction"),
    ("signal_aware_safe_gate_fixed", "Conservative signal-aware safe gate"),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a manuscript-ready external-validation status/results insert. "
            "The output remains claim-blocked until blinded reference annotations "
            "and frozen external metrics are available."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--external-output-prefix", default="external_repro")
    parser.add_argument("--external-input-root", type=Path, default=None)
    return parser.parse_args()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def number(value: object, default: float = math.nan) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else default


def integer(value: object, default: int = 0) -> int:
    parsed = number(value)
    return int(parsed) if np.isfinite(parsed) else default


def metric_value(table: pd.DataFrame, metric: str, default: object = "") -> object:
    if table.empty or not {"metric", "value"}.issubset(table.columns):
        return default
    match = table[table["metric"].astype(str).eq(metric)]
    return match.iloc[0]["value"] if not match.empty else default


def readiness_value(table: pd.DataFrame, check: str, column: str = "evidence") -> str:
    if table.empty or "check" not in table.columns or column not in table.columns:
        return ""
    match = table[table["check"].astype(str).eq(check)]
    return clean(match.iloc[0][column]) if not match.empty else ""


def progress_row(table: pd.DataFrame, annotator: str) -> pd.Series:
    if table.empty or "annotator" not in table.columns:
        return pd.Series(dtype=object)
    match = table[table["annotator"].astype(str).str.lower().eq(annotator)]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def acceptance_status(table: pd.DataFrame, gate: str) -> str:
    if table.empty or not {"gate", "status"}.issubset(table.columns):
        return "NOT_EVALUATED"
    match = table[table["gate"].astype(str).eq(gate)]
    return clean(match.iloc[0]["status"]) if not match.empty else "NOT_EVALUATED"


def external_metric_row(metrics: pd.DataFrame, method: str) -> pd.Series:
    if metrics.empty or not {"split_scope", "method"}.issubset(metrics.columns):
        return pd.Series(dtype=object)
    match = metrics[
        metrics["split_scope"].astype(str).eq("external_pool")
        & metrics["method"].astype(str).eq(method)
    ]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def cluster_metric_row(cluster_metrics: pd.DataFrame, method: str) -> pd.Series:
    if cluster_metrics.empty or not {"method", "cluster_variable"}.issubset(
        cluster_metrics.columns
    ):
        return pd.Series(dtype=object)
    match = cluster_metrics[
        cluster_metrics["method"].astype(str).eq(method)
        & cluster_metrics["cluster_variable"].astype(str).eq("source_session_id")
    ]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def agreement_from_predictions(
    predictions: pd.DataFrame,
    method: str,
) -> tuple[float, float, float, int]:
    prediction_columns = {
        "default_pipeline": "rr_bpm",
        "quality_residual_fixed": "corrected_rr_bpm",
        "signal_aware_safe_gate_fixed": "signal_aware_safe_final_rr_bpm",
    }
    rr_column = prediction_columns.get(method, "")
    if predictions.empty or rr_column not in predictions.columns or "truth_rr" not in predictions.columns:
        return math.nan, math.nan, math.nan, 0
    subset = predictions.copy()
    if "split_pool" in subset.columns:
        subset = subset[subset["split_pool"].astype(str).eq("external_pool")]
    truth = pd.to_numeric(subset["truth_rr"], errors="coerce")
    predicted = pd.to_numeric(subset[rr_column], errors="coerce")
    valid = truth.notna() & predicted.notna()
    errors = (predicted[valid] - truth[valid]).to_numpy(dtype=float)
    if errors.size == 0:
        return math.nan, math.nan, math.nan, 0
    bias = float(np.mean(errors))
    sd = float(np.std(errors, ddof=1)) if errors.size > 1 else math.nan
    lower = bias - 1.96 * sd if np.isfinite(sd) else math.nan
    upper = bias + 1.96 * sd if np.isfinite(sd) else math.nan
    return bias, lower, upper, int(errors.size)


def fmt(value: object, digits: int = 3) -> str:
    parsed = number(value)
    return f"{parsed:.{digits}f}" if np.isfinite(parsed) else "NA"


def collect_state(
    input_root: Path,
    corrected_prefix: str,
    external_input_root: Path,
    external_output_prefix: str,
) -> dict[str, object]:
    assets = input_root / f"{corrected_prefix}_paper_assets"
    inventory_readiness = read_optional_csv(
        assets / "paper_external_validation_split_all_use_readiness.csv"
    )
    inventory = read_optional_csv(
        assets / "paper_external_validation_split_all_use_inventory.csv"
    )
    sessions = read_optional_csv(
        assets / "paper_external_validation_split_all_use_session_summary.csv"
    )
    progress = read_optional_csv(
        assets / "paper_external_validation_annotation_progress_summary.csv"
    )
    agreement = read_optional_csv(
        assets / "paper_external_validation_breath_annotation_agreement_summary.csv"
    )
    metrics = read_optional_csv(
        external_input_root / f"{external_output_prefix}_external_split_metrics.csv"
    )
    cluster_bootstrap = read_optional_csv(
        external_input_root
        / f"{external_output_prefix}_external_split_cluster_bootstrap.csv"
    )
    acceptance = read_optional_csv(
        external_input_root / f"{external_output_prefix}_external_split_acceptance.csv"
    )
    predictions = read_optional_csv(
        external_input_root / f"{external_output_prefix}_external_split_predictions.csv"
    )

    included = integer(metric_value(agreement, "included_external_rows"))
    if included == 0 and not inventory.empty:
        if "include_default" in inventory.columns:
            included = int(
                inventory["include_default"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
            )
        else:
            included = len(inventory)
    total_clips = len(inventory)
    if total_clips == 0:
        evidence = readiness_value(inventory_readiness, "clip-level external rows available")
        total_clips = integer(evidence.split("=")[-1]) if "=" in evidence else 0
    short_excluded = max(total_clips - included, 0)
    source_sessions = len(sessions)
    unique_cows = sessions["cow_id"].nunique() if "cow_id" in sessions.columns else 0
    unique_dates = sessions["collection_date"].nunique() if "collection_date" in sessions.columns else 0

    a_row = progress_row(progress, "a")
    b_row = progress_row(progress, "b")
    complete_dual = integer(metric_value(agreement, "complete_dual_annotation_rows"))
    consensus_ready = integer(metric_value(agreement, "consensus_ready_rows"))
    needs_adjudication = integer(metric_value(agreement, "needs_adjudication_rows"), included)
    external_metrics_available = any(
        not external_metric_row(metrics, method).empty for method, _ in EXTERNAL_METHODS
    )
    external_claim = acceptance_status(acceptance, "external performance claim allowed")
    algorithmic_claim = acceptance_status(
        acceptance, "algorithmic engineering external claim allowed"
    )

    if external_metrics_available:
        status = "external_scored_claim_allowed" if "PASS" in {external_claim, algorithmic_claim} else "external_scored_gate_failed"
        next_action = (
            "Report only methods whose frozen external acceptance bundle passed."
            if "PASS" in {external_claim, algorithmic_claim}
            else "Inspect external failure modes; do not tune the frozen method on this test cohort."
        )
    elif included > 0 and consensus_ready >= included and included > 0:
        status = "references_ready_scoring_pending"
        next_action = "Run scripts/run_rr_external_validation_after_annotation.py to produce frozen external predictions and metrics."
    elif included > 0:
        status = "candidate_cohort_ready_reference_labels_pending"
        next_action = (
            "Complete and export both blinded annotation CSVs, adjudicate disagreements, "
            "then run scripts/run_rr_external_validation_after_annotation.py."
        )
    else:
        status = "external_candidate_cohort_missing"
        next_action = "Build the external candidate inventory before external scoring."

    return {
        "assets": assets,
        "external_input_root": external_input_root,
        "inventory_readiness": inventory_readiness,
        "progress": progress,
        "agreement": agreement,
        "metrics": metrics,
        "cluster_bootstrap": cluster_bootstrap,
        "acceptance": acceptance,
        "predictions": predictions,
        "total_clips": total_clips,
        "included": included,
        "short_excluded": short_excluded,
        "source_sessions": source_sessions,
        "unique_cows": int(unique_cows),
        "unique_dates": int(unique_dates),
        "annotator_a_status": clean(a_row.get("status", "MISSING")),
        "annotator_b_status": clean(b_row.get("status", "MISSING")),
        "annotator_a_complete": integer(a_row.get("complete_required_rows", 0)),
        "annotator_b_complete": integer(b_row.get("complete_required_rows", 0)),
        "complete_dual": complete_dual,
        "consensus_ready": consensus_ready,
        "needs_adjudication": needs_adjudication,
        "external_metrics_available": external_metrics_available,
        "cluster_bootstrap_available": not cluster_bootstrap.empty,
        "external_claim": external_claim,
        "algorithmic_claim": algorithmic_claim,
        "status": status,
        "next_action": next_action,
    }


def build_reporting_text(state: dict[str, object]) -> str:
    included = int(state["included"])
    total = int(state["total_clips"])
    sessions = int(state["source_sessions"])
    cows = int(state["unique_cows"])
    dates = int(state["unique_dates"])
    metrics = state["metrics"]
    cluster_bootstrap = state["cluster_bootstrap"]
    predictions = state["predictions"]

    cohort_paragraph = (
        "### 3.10 Frozen External Candidate Cohort and Reference-Annotation Gate\n\n"
        "An independently collected external candidate cohort was assembled from "
        "thermal videos acquired at Jiufu Ranch, Hulunbuir City, Inner Mongolia, "
        f"China. The inventory contains {total} clips, of which {included} were "
        f"retained for default 30-s respiratory-rate scoring and {int(state['short_excluded'])} "
        f"short terminal fragments were excluded. The retained clips represent {sessions} "
        f"source recording sessions, {cows} cow labels, and {dates} collection dates. "
        "Because clips cut from the same long video are correlated, source_session_id, "
        "rather than clip ID alone, must be used for leakage control and clustered "
        "uncertainty analysis; the clip count must not be described as an equal number "
        "of independent animals."
    )

    if not bool(state["external_metrics_available"]):
        annotation_paragraph = (
            "Blinded manual breath-count packets were prepared for two annotators, "
            f"each expecting {included} rows. At the current reporting snapshot, "
            f"annotator A has {int(state['annotator_a_complete'])}/{included} complete rows "
            f"(status: {state['annotator_a_status']}) and annotator B has "
            f"{int(state['annotator_b_complete'])}/{included} complete rows "
            f"(status: {state['annotator_b_status']}). Dual-annotator reference counts "
            f"are complete for {int(state['complete_dual'])}/{included} clips, consensus "
            f"RR is ready for {int(state['consensus_ready'])}/{included}, and "
            f"{int(state['needs_adjudication'])} clips remain unresolved. Therefore, no "
            "external RR R2, MAE, RMSE, exact-count agreement, or Bland-Altman limits "
            "can currently be reported. The external cohort exists and exceeds the "
            "104-clip algorithmic planning tier, but the reference-outcome gate remains "
            "closed; this is preparation evidence, not external performance evidence."
        )
        gate_paragraph = (
            "After both blinded exports and adjudication are complete, the frozen runner "
            "will merge consensus references, generate predictions without retuning, and "
            "evaluate the default, quality-aware, and safe-gate methods. An external "
            "absolute-performance statement is permitted only when the method freeze is "
            "locked and the predefined gates are met: RR R2 at least 0.90, MAE no greater "
            "than 2.5 breaths/min, RMSE no greater than 4.0 breaths/min, and within-one "
            "breath agreement at least 95%. Until then, the manuscript may state that an "
            "independent candidate cohort and blinded annotation protocol were established, "
            "but it must not claim external accuracy or generalization."
        )
        return "\n\n".join([cohort_paragraph, annotation_paragraph, gate_paragraph])

    method_sentences: list[str] = []
    for method, label in EXTERNAL_METHODS:
        row = external_metric_row(metrics, method)
        if row.empty:
            continue
        valid_n = integer(row.get("rr_valid_videos", row.get("videos", 0)))
        exact = integer(row.get("exact_count", 0))
        bias, lower, upper, agreement_n = agreement_from_predictions(predictions, method)
        agreement_text = (
            f", with Bland-Altman bias {fmt(bias)} breaths/min and 95% limits "
            f"from {fmt(lower)} to {fmt(upper)} breaths/min (n={agreement_n})"
            if agreement_n > 0
            else ""
        )
        clustered = cluster_metric_row(cluster_bootstrap, method)
        cluster_text = (
            f"; the source-session cluster-bootstrap 95% CI was "
            f"{fmt(clustered.get('rr_r2_ci_low'), 4)} to "
            f"{fmt(clustered.get('rr_r2_ci_high'), 4)} for RR R2 and "
            f"{fmt(clustered.get('rr_mae_ci_low'))} to "
            f"{fmt(clustered.get('rr_mae_ci_high'))} breaths/min for MAE"
            if not clustered.empty
            else ""
        )
        method_sentences.append(
            f"{label} achieved external RR R2={fmt(row.get('rr_r2'), 4)}, "
            f"MAE={fmt(row.get('rr_mae'))} breaths/min, RMSE={fmt(row.get('rr_rmse'))} "
            f"breaths/min, and exact count agreement in {exact}/{valid_n} clips"
            f"{agreement_text}{cluster_text}."
        )
    results_paragraph = " ".join(method_sentences)
    gate_paragraph = (
        f"The primary external-performance claim gate was {state['external_claim']}, "
        f"and the algorithmic-engineering safe-gate claim was {state['algorithmic_claim']}. "
        "Only a PASS gate tied to the locked freeze ID permits the corresponding external "
        "claim; a failed gate is reported as a prospective-validation result and must not "
        "be repaired by tuning on this external cohort."
    )
    return "\n\n".join([cohort_paragraph, results_paragraph, gate_paragraph])


def status_table(state: dict[str, object]) -> pd.DataFrame:
    keys = [
        "status",
        "total_clips",
        "included",
        "short_excluded",
        "source_sessions",
        "unique_cows",
        "unique_dates",
        "annotator_a_status",
        "annotator_b_status",
        "annotator_a_complete",
        "annotator_b_complete",
        "complete_dual",
        "consensus_ready",
        "needs_adjudication",
        "external_metrics_available",
        "cluster_bootstrap_available",
        "external_claim",
        "algorithmic_claim",
        "next_action",
    ]
    return pd.DataFrame([{"field": key, "value": state[key]} for key in keys])


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_root = args.input_root.resolve()
    external_input_root = (
        args.external_input_root.resolve()
        if args.external_input_root is not None
        else repo_root / "Dataset_new" / "72video" / "external_al_images"
    )
    state = collect_state(
        input_root,
        args.corrected_prefix,
        external_input_root,
        args.external_output_prefix,
    )
    text = build_reporting_text(state)
    assets = Path(state["assets"])
    assets.mkdir(parents=True, exist_ok=True)
    report_path = assets / "paper_external_validation_reporting_insert.md"
    status_path = assets / "paper_external_validation_reporting_status.csv"
    docs_path = repo_root / "docs" / "thermal_rr_external_validation_reporting_insert.md"
    report_path.write_text(text + "\n", encoding="utf-8")
    docs_path.write_text(text + "\n", encoding="utf-8")
    status_table(state).to_csv(status_path, index=False)
    print(f"Saved external validation reporting insert: {report_path}")
    print(f"Saved external validation reporting status: {status_path}")
    print(f"Saved docs copy: {docs_path}")
    print(f"Status: {state['status']}")
    print(f"Next action: {state['next_action']}")


if __name__ == "__main__":
    main()
