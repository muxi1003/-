from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PRIMARY_METHOD = "quality_aware_residual_fixed_threshold"
SECONDARY_METHOD = "conservative_signal_aware_safe_gate_candidate"
TERTIARY_METHOD = "signal_consensus_supplement_candidate"
RELIABILITY_METHOD = "bilateral_nostril_consistency_reliability_gate"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_output_dir = (
        default_input_root / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current thermal RR method package before external validation "
            "by hashing method scripts, parameters, and key result/protocol assets."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_row(base_dir: Path, path: Path, scope: str, role: str, required: bool) -> dict[str, object]:
    exists = path.exists()
    try:
        relative = str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        relative = str(path.resolve())
    return {
        "scope": scope,
        "role": role,
        "required": bool(required),
        "path": str(path),
        "relative_path": relative,
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
        "sha256": sha256_file(path) if exists and path.is_file() else "",
        "modified_time_utc": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if exists
            else ""
        ),
    }


def regex_default(source: str, option: str, fallback: object) -> object:
    pattern = rf'parser\.add_argument\("{re.escape(option)}".*?default=([^,\)]+)'
    match = re.search(pattern, source, flags=re.DOTALL)
    if not match:
        return fallback
    value = match.group(1).strip()
    if value.startswith(("'", '"')):
        return value.strip("'\"")
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_defaults(script_path: Path, defaults: dict[str, object]) -> dict[str, object]:
    if not script_path.exists():
        return defaults.copy()
    source = script_path.read_text(encoding="utf-8")
    return {
        name: regex_default(source, option, fallback)
        for name, (option, fallback) in defaults.items()
    }


def build_parameters(repo_root: Path) -> pd.DataFrame:
    residual_defaults = load_defaults(
        repo_root / "scripts" / "rr_quality_residual_corrector.py",
        {
            "confidence_threshold": ("--confidence-threshold", 0.54),
            "margin_threshold": ("--margin-threshold", 0.26),
            "folds": ("--folds", 5),
            "cv_random_state": ("--cv-random-state", 42),
            "model_random_state": ("--model-random-state", 4),
            "n_estimators": ("--n-estimators", 200),
            "max_depth": ("--max-depth", 3),
            "min_samples_leaf": ("--min-samples-leaf", 4),
        },
    )
    validation_defaults = load_defaults(
        repo_root / "scripts" / "rr_quality_residual_validation.py",
        {
            "validation_confidence_threshold": ("--confidence-threshold", 0.54),
            "validation_margin_threshold": ("--margin-threshold", 0.26),
            "validation_folds": ("--folds", 5),
            "validation_cv_random_state": ("--cv-random-state", 42),
        },
    )
    signal_defaults = load_defaults(
        repo_root / "scripts" / "rr_signal_consensus_validation.py",
        {
            "supplement_confidence": ("--supplement-confidence", 0.50),
            "supplement_margin": ("--supplement-margin", 0.10),
            "min_signal_votes": ("--min-signal-votes", 2),
            "min_supplement_interval_cv": ("--min-supplement-interval-cv", 0.09),
        },
    )
    safe_gate_defaults = load_defaults(
        repo_root / "scripts" / "rr_signal_aware_safe_policy.py",
        {
            "max_autocorr_delta": ("--max-autocorr-delta", 2.0),
            "bootstrap_samples": ("--bootstrap-samples", 5000),
            "bootstrap_random_state": ("--bootstrap-random-state", 20260707),
            "source_name": ("--source-name", "signal_aware_residual"),
        },
    )
    bilateral_defaults = load_defaults(
        repo_root / "scripts" / "build_rr_bilateral_consistency_gate.py",
        {
            "min_bilateral_corr": ("--min-bilateral-corr", 0.20),
            "max_fft_count_delta": ("--max-fft-count-delta", 1.0),
            "min_amplitude_ratio": ("--min-amplitude-ratio", 0.10),
            "max_correlation_lag": ("--max-correlation-lag", 8),
        },
    )
    rows: list[dict[str, object]] = []
    for name, value in residual_defaults.items():
        rows.append(
            {
                "method_scope": PRIMARY_METHOD,
                "parameter": name,
                "value": value,
                "source": "scripts/rr_quality_residual_corrector.py default",
                "external_use": "primary frozen method",
            }
        )
    for name, value in validation_defaults.items():
        rows.append(
            {
                "method_scope": PRIMARY_METHOD,
                "parameter": name,
                "value": value,
                "source": "scripts/rr_quality_residual_validation.py default",
                "external_use": "primary frozen validation threshold",
            }
        )
    for name, value in signal_defaults.items():
        rows.append(
            {
                "method_scope": TERTIARY_METHOD,
                "parameter": name,
                "value": value,
                "source": "scripts/rr_signal_consensus_validation.py default",
                "external_use": "tertiary candidate extension only",
            }
        )
    for name, value in safe_gate_defaults.items():
        rows.append(
            {
                "method_scope": SECONDARY_METHOD,
                "parameter": name,
                "value": value,
                "source": "scripts/rr_signal_aware_safe_policy.py default",
                "external_use": "secondary safety-gated candidate only",
            }
        )
    for name, value in bilateral_defaults.items():
        rows.append(
            {
                "method_scope": RELIABILITY_METHOD,
                "parameter": name,
                "value": value,
                "source": "scripts/build_rr_bilateral_consistency_gate.py default",
                "external_use": "internal reliability gate; freeze before external selective-reporting evaluation",
            }
        )
    rows.extend(
        [
            {
                "method_scope": PRIMARY_METHOD,
                "parameter": "external_primary_claim",
                "value": "absolute performance only unless paired external CI supports relative improvement",
                "source": "paper_external_validation_acceptance_criteria.csv",
                "external_use": "claim boundary",
            },
            {
                "method_scope": "truth_calibrated_upper_bound",
                "parameter": "external_use",
                "value": "never use as external performance",
                "source": "paper_main_results_table.csv",
                "external_use": "leakage guard",
            },
        ]
    )
    return pd.DataFrame(rows)


def metric_rows(main_results: pd.DataFrame | None) -> list[dict[str, object]]:
    if main_results is None or main_results.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, item in main_results.iterrows():
        rows.append(
            {
                "method": item.get("method", ""),
                "validation_setting": item.get("validation_setting", ""),
                "rr_r2": item.get("rr_r2", ""),
                "rr_mae_bpm": item.get("rr_mae_bpm", ""),
                "rr_rmse_bpm": item.get("rr_rmse_bpm", ""),
                "exact_count": item.get("exact_count", ""),
                "paper_use": item.get("paper_use", ""),
            }
        )
    return rows


def freeze_id_for(
    manifest: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: list[dict[str, object]],
) -> str:
    payload = {
        "manifest": manifest[
            ["scope", "role", "required", "relative_path", "exists", "bytes", "sha256"]
        ].sort_values(["scope", "role", "relative_path"]).to_dict(orient="records"),
        "parameters": parameters.sort_values(
            ["method_scope", "parameter"]
        ).to_dict(orient="records"),
        "metrics": sorted(metrics, key=lambda row: str(row.get("method", ""))),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("\n", " ").replace("|", "/") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    parameters: pd.DataFrame,
    output_path: Path,
) -> None:
    row = summary.iloc[0].to_dict()
    missing_required = manifest[manifest["required"].astype(bool) & ~manifest["exists"].astype(bool)]
    text = [
        "# Frozen External Method Manifest",
        "",
        f"Freeze status: `{row['freeze_status']}`",
        "",
        f"Method freeze ID: `{row['method_freeze_id']}`",
        "",
        f"Primary frozen method: `{row['primary_method']}`",
        "",
        f"Secondary candidate method: `{row['secondary_method']}`",
        "",
        f"Tertiary candidate method: `{row.get('tertiary_method', '')}`",
        "",
        f"Reliability gate: `{row.get('reliability_method', '')}`",
        "",
        f"Required files present: `{row['required_files_present']}`",
        "",
        "This freeze manifest is intended to be generated before external validation. If any script, parameter, or locked result asset changes, regenerate the manifest and treat external validation claims as tied to the new freeze ID.",
        "",
        "## Missing Required Files",
        "",
        "No required files are missing."
        if missing_required.empty
        else markdown_table(missing_required, ["scope", "role", "path"]),
        "",
        "## Frozen Parameters",
        "",
        markdown_table(parameters, ["method_scope", "parameter", "value", "external_use"]),
        "",
        "## Hashed Files",
        "",
        markdown_table(
            manifest,
            ["scope", "role", "required", "relative_path", "exists", "bytes", "sha256"],
        ),
    ]
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        ("source_code", "default RR pipeline", repo_root / "scripts" / "paper_repro_rr.py", True),
        ("source_code", "quality residual corrector", repo_root / "scripts" / "rr_quality_residual_corrector.py", True),
        ("source_code", "quality residual validation", repo_root / "scripts" / "rr_quality_residual_validation.py", True),
        ("source_code", "signal consensus candidate", repo_root / "scripts" / "rr_signal_consensus_validation.py", True),
        ("source_code", "signal-aware residual candidate", repo_root / "scripts" / "rr_signal_aware_residual_validation.py", True),
        ("source_code", "conservative signal-aware safe gate", repo_root / "scripts" / "rr_signal_aware_safe_policy.py", True),
        ("source_code", "RR-only physiological triage", repo_root / "scripts" / "build_rr_physiological_triage.py", True),
        ("source_code", "bilateral nostril consistency reliability gate", repo_root / "scripts" / "build_rr_bilateral_consistency_gate.py", True),
        ("source_code", "frozen external postprocessor applier", repo_root / "scripts" / "apply_rr_frozen_external_postprocessors.py", True),
        ("source_code", "external split validation", repo_root / "scripts" / "rr_external_split_validation.py", True),
        ("source_code", "metadata quality audit", repo_root / "scripts" / "audit_rr_metadata_quality.py", True),
        ("source_code", "submission readiness audit", repo_root / "scripts" / "audit_rr_submission_readiness.py", True),
        ("source_code", "method freeze generator", repo_root / "scripts" / "freeze_rr_external_method.py", True),
    ]
    result_files = [
        ("result_asset", "default summary", input_root / f"{args.output_prefix}_summary.csv", True),
        ("result_asset", "default metrics", input_root / f"{args.output_prefix}_metrics.csv", True),
        ("result_asset", "quality residual predictions", input_root / f"{args.corrected_prefix}_predictions.csv", True),
        ("result_asset", "quality residual validation summary", input_root / f"{args.corrected_prefix}_validation_summary.csv", True),
        ("result_asset", "quality residual bootstrap CI", input_root / f"{args.corrected_prefix}_bootstrap_ci.csv", True),
        ("result_asset", "quality residual group metrics", input_root / f"{args.corrected_prefix}_group_metrics.csv", True),
        ("result_asset", "signal-aware safe gate predictions", input_root / f"{args.output_prefix}_signal_aware_safe_policy_predictions.csv", True),
        ("result_asset", "signal-aware safe gate group predictions", input_root / f"{args.output_prefix}_signal_aware_safe_policy_group_predictions.csv", True),
        ("result_asset", "signal-aware safe gate metrics", input_root / f"{args.output_prefix}_signal_aware_safe_policy_metrics.csv", True),
        ("result_asset", "signal-aware safe gate bootstrap CI", input_root / f"{args.output_prefix}_signal_aware_safe_policy_bootstrap_ci.csv", True),
        ("paper_asset", "main results table", output_dir / "paper_main_results_table.csv", True),
        ("paper_asset", "safe gate metrics table", output_dir / "paper_signal_aware_safe_policy_metrics_table.csv", True),
        ("paper_asset", "external acceptance criteria", output_dir / "paper_external_validation_acceptance_criteria.csv", True),
        ("paper_asset", "external sample plan", output_dir / "paper_external_validation_sample_plan.csv", False),
        ("paper_asset", "metadata quality audit", output_dir / "paper_metadata_quality_audit.csv", False),
        ("paper_asset", "manuscript claim audit", output_dir / "paper_manuscript_claim_audit.csv", False),
        ("paper_asset", "RR-only physiological triage summary", output_dir / "paper_rr_physiological_triage_summary.csv", True),
        ("paper_asset", "RR-only physiological triage report", output_dir / "paper_rr_physiological_triage_report.md", True),
        ("paper_asset", "bilateral nostril consistency summary", output_dir / "paper_rr_bilateral_consistency_summary.csv", True),
        ("paper_asset", "bilateral nostril consistency report", output_dir / "paper_rr_bilateral_consistency_report.md", True),
        ("protocol_doc", "external validation protocol", repo_root / "docs" / "thermal_rr_external_validation_protocol.md", True),
        ("protocol_doc", "submission strategy", repo_root / "docs" / "thermal_rr_submission_strategy.md", True),
    ]
    manifest = pd.DataFrame(
        [
            file_row(repo_root, path, scope, role, required)
            for scope, role, path, required in [*source_files, *result_files]
        ]
    )
    parameters = build_parameters(repo_root)
    main_results_path = output_dir / "paper_main_results_table.csv"
    main_results = pd.read_csv(main_results_path) if main_results_path.exists() else None
    metrics = metric_rows(main_results)
    freeze_id = freeze_id_for(manifest, parameters, metrics)
    missing_required = manifest[manifest["required"].astype(bool) & ~manifest["exists"].astype(bool)]
    freeze_status = "locked" if missing_required.empty else "incomplete"
    summary = pd.DataFrame(
        [
            {
                "method_freeze_id": freeze_id,
                "freeze_status": freeze_status,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "primary_method": PRIMARY_METHOD,
                "secondary_method": SECONDARY_METHOD,
                "tertiary_method": TERTIARY_METHOD,
                "reliability_method": RELIABILITY_METHOD,
                "required_files": int(manifest["required"].astype(bool).sum()),
                "required_files_present": int(
                    (manifest["required"].astype(bool) & manifest["exists"].astype(bool)).sum()
                ),
                "required_files_missing": int(len(missing_required)),
                "hashed_files": int(manifest["sha256"].astype(str).ne("").sum()),
                "external_claim_boundary": (
                    "Use the primary frozen method for external absolute-performance "
                    "validation; treat the conservative signal-aware safe gate and "
                    "signal consensus as secondary/tertiary candidates unless explicitly "
                    "externally passed; never use truth-calibrated outputs as external "
                    "performance."
                ),
            }
        ]
    )

    summary_path = output_dir / "paper_method_freeze_summary.csv"
    manifest_path = output_dir / "paper_method_freeze_manifest.csv"
    parameter_path = output_dir / "paper_method_freeze_parameters.csv"
    json_path = output_dir / "paper_method_freeze_summary.json"
    report_path = output_dir / "paper_method_freeze_report.md"

    summary.to_csv(summary_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    parameters.to_csv(parameter_path, index=False)
    json_path.write_text(
        json.dumps(summary.iloc[0].to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary, manifest, parameters, report_path)

    print(f"Saved method freeze summary: {summary_path}")
    print(f"Saved method freeze manifest: {manifest_path}")
    print(f"Saved method freeze parameters: {parameter_path}")
    print(f"Saved method freeze report: {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
