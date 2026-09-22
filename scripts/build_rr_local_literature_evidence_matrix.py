from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SOURCE_OVERRIDES: dict[str, dict[str, str]] = {
    "1-s2.0-S0306456525001111-main.pdf": {
        "source_id": "lin_2025_head_movement_thermal_rr",
        "year": "2025",
        "venue_or_type": "Journal of Thermal Biology",
        "paper_title": "Low-resolution thermal RR estimation under head movement",
        "method_family": "YOLOv8n-Pose nostril keypoints, RF temperature mapping, two-nostril fusion",
        "literature_gap": "Nostril localization and temperature mapping are already current; novelty must move downstream to residual peak errors, quality gates, uncertainty, and external validation.",
        "current_project_response": "Conservative signal-aware safe gate, bilateral consistency gate, selective reporting, and frozen external validation workflow.",
        "manuscript_role": "primary recent baseline and reproduction anchor",
        "claim_boundary": "Do not present YOLO/keypoint mapping alone as the new contribution.",
    },
    "agriculture-13-01939-v2.pdf": {
        "source_id": "zhao_2023_irt_deep_learning_rr",
        "year": "2023",
        "venue_or_type": "Agriculture",
        "paper_title": "Detection of respiratory rate of dairy cows based on infrared thermography and deep learning",
        "method_family": "IRT nose detection, nostril segmentation, temperature curves, sliding-window peak detection",
        "literature_gap": "A strong modular IRT baseline exists, but peak counting remains vulnerable to noisy, missing, and boundary-truncated cycles.",
        "current_project_response": "Quality-aware residual peak-count correction and signal-consensus checks.",
        "manuscript_role": "technical baseline",
        "claim_boundary": "Do not claim novelty from simply repeating thermal nostril detection and peak counting.",
    },
    "1-s2.0-S2666910223001217-main.pdf": {
        "source_id": "mantovani_2024_fft_image_rr",
        "year": "2024",
        "venue_or_type": "JDS Communications",
        "paper_title": "Predicting respiration rate in unrestrained dairy cows using image analysis and fast Fourier transform",
        "method_family": "Image-analysis respiratory signal plus FFT count estimate",
        "literature_gap": "Frequency-domain RR can support a training-free sanity check, but should not replace validated thermal curve extraction in the current dataset.",
        "current_project_response": "FFT/autocorrelation agreement is used as a supplementary signal-consensus feature and safe-gate criterion.",
        "manuscript_role": "signal-consensus support",
        "claim_boundary": "Keep FFT evidence as auxiliary agreement unless external paired evidence proves superiority.",
    },
    "1-s2.0-S0022030224010300-main.pdf": {
        "source_id": "wang_2024_rgb_videomae_rr",
        "year": "2024",
        "venue_or_type": "Journal of Dairy Science",
        "paper_title": "Learning end-to-end respiratory rate prediction of dairy cows from RGB videos",
        "method_family": "End-to-end RGB video transformer with respiratory-belt reference",
        "literature_gap": "End-to-end video learning is active, but it needs larger, grouped, labeled video datasets.",
        "current_project_response": "Keep transformer fusion as future work; use interpretable thermal-curve correction for the current 73-video internal set.",
        "manuscript_role": "future-model context",
        "claim_boundary": "Do not reposition the current paper as an end-to-end transformer paper.",
    },
    "1-s2.0-S1537511024000163-main.pdf": {
        "source_id": "yan_2024_environmental_ml_rr",
        "year": "2024",
        "venue_or_type": "Biosystems Engineering",
        "paper_title": "Comparative machine-learning models for dairy-cow respiration-rate prediction",
        "method_family": "Production and environmental features with model comparison and interpretation",
        "literature_gap": "Environmental and production metadata give RR biological meaning, but they are not available per video in the current internal set.",
        "current_project_response": "Separate algorithmic RR validation from heat-stress/THI claims; require real ambient temperature and humidity for those claims.",
        "manuscript_role": "biological context and metadata gate",
        "claim_boundary": "Do not infer THI or heat-stress strata from location/date provenance alone.",
    },
    "\u57fa\u4e8e\u8d85\u53c2\u6570\u4f18\u5316\u7b97\u6cd5\u7684\u968f\u673a\u68ee\u6797\u6a21\u578b\u9884\u6d4b\u5976\u725b\u547c\u5438\u9891\u7387.pdf": {
        "source_id": "yan_2024_cn_athi_rf_rr",
        "year": "2024",
        "venue_or_type": "Transactions of the CSAE",
        "paper_title": "Predicting respiratory rate of dairy cows using hyperparameter-optimized random forest models",
        "method_family": "ATHI, time region, milk yield, DIM, posture, parity with optimized RF",
        "literature_gap": "ATHI and production factors explain RR variation and strengthen welfare interpretation.",
        "current_project_response": "Use as justification for future THI/ATHI collection; keep current external batch as algorithmic validation first.",
        "manuscript_role": "local environmental-context support",
        "claim_boundary": "Do not use regional weather proxies as per-video ATHI evidence.",
    },
    "\u5976\u725b\u547c\u5438\u9891\u7387\u81ea\u52a8\u76d1\u6d4b\u6280\u672f\u7814\u7a76\u8fdb\u5c55.pdf": {
        "source_id": "li_2019_rr_monitoring_review",
        "year": "2019",
        "venue_or_type": "Review",
        "paper_title": "Research progress in automatic monitoring technology of dairy-cow respiratory rate",
        "method_family": "Contact and non-contact RR monitoring review",
        "literature_gap": "Continuous, non-contact RR monitoring remains valuable because manual counting is labor-intensive and not continuous.",
        "current_project_response": "Frame thermal-video RR as a non-contact PLF monitoring workflow with review triage.",
        "manuscript_role": "introduction framing",
        "claim_boundary": "Use for motivation, not as evidence of current algorithm performance.",
    },
    "pdfviewer.pdf": {
        "source_id": "early_cattle_rr_heat_exchange",
        "year": "unknown",
        "venue_or_type": "Older cattle RR physiology paper",
        "paper_title": "Respiratory heat transfer and automated RR monitoring context",
        "method_family": "Physiological RR response to heat and early automated measurement motivation",
        "literature_gap": "RR is a meaningful physiological response, but heat-stress claims require real environmental exposure data.",
        "current_project_response": "Report RR-only review alerts separately from heat-stress or welfare diagnoses.",
        "manuscript_role": "physiological motivation",
        "claim_boundary": "Do not treat high RR alone as a diagnosed heat-stress label.",
    },
    "1781525583024_25474811_725256_60_ieeeconference_IoT-EnabledDeviceforPredictiveMonitoringandDiseaseManagementinCow.pdf": {
        "source_id": "iot_cow_health_monitoring",
        "year": "unknown",
        "venue_or_type": "IoT conference paper",
        "paper_title": "IoT-enabled predictive monitoring and disease management in cows",
        "method_family": "Multisensor IoT dashboard with threshold or ML health status",
        "literature_gap": "Deployment systems need real-time provenance, sensor context, and decision outputs.",
        "current_project_response": "Use algorithmic quality tiers, external validation packet, and review routing instead of claiming a disease-management system.",
        "manuscript_role": "deployment context only",
        "claim_boundary": "Do not claim disease diagnosis from RR-only thermal video.",
    },
    "bioconf_isgp2022_01007.pdf": {
        "source_id": "cattle_physiology_values",
        "year": "2022",
        "venue_or_type": "Conference physiology reference",
        "paper_title": "Physiology values of breath, pulse, and body temperature of cattle",
        "method_family": "Reference physiology values",
        "literature_gap": "Normal-value context helps interpret RR, but breed, age, environment, and measurement protocol matter.",
        "current_project_response": "Keep RR values as measured prediction targets; avoid health classification without protocol-specific thresholds.",
        "manuscript_role": "physiology background",
        "claim_boundary": "Do not convert RR prediction into health status classification without labels.",
    },
    "Engineering_Village_detailed_6-15-2026_131345602.pdf": {
        "source_id": "engineering_village_record_1",
        "year": "2026",
        "venue_or_type": "Database record",
        "paper_title": "Engineering Village search record",
        "method_family": "Bibliographic metadata",
        "literature_gap": "Search records are not primary evidence.",
        "current_project_response": "Use only as a pointer to source discovery, not as manuscript evidence.",
        "manuscript_role": "exclude from main evidence",
        "claim_boundary": "Do not cite database-export PDFs as if they were full research papers.",
    },
    "Engineering_Village_detailed_6-15-2026_132440939.pdf": {
        "source_id": "engineering_village_record_2",
        "year": "2026",
        "venue_or_type": "Database record",
        "paper_title": "Engineering Village search record",
        "method_family": "Bibliographic metadata",
        "literature_gap": "Search records are not primary evidence.",
        "current_project_response": "Use only as a pointer to source discovery, not as manuscript evidence.",
        "manuscript_role": "exclude from main evidence",
        "claim_boundary": "Do not cite database-export PDFs as if they were full research papers.",
    },
    "\u57fa\u4e8eIATEFF-YOLO\u7684\u591c\u95f4\u5976\u725b\u722c\u8de8\u884c\u4e3a\u68c0\u6d4b.pdf": {
        "source_id": "iateff_yolo_mounting_behavior",
        "year": "",
        "venue_or_type": "Unrelated cattle behavior detection paper",
        "paper_title": "Nighttime dairy-cow mounting behavior detection based on IATEFF-YOLO",
        "method_family": "YOLO behavior detection, not respiratory-rate estimation",
        "literature_gap": "This source is outside the respiratory-rate evidence chain.",
        "current_project_response": "Exclude from the RR manuscript evidence matrix unless discussing generic YOLO detection context.",
        "manuscript_role": "exclude from main evidence",
        "claim_boundary": "Do not cite as respiratory-rate, thermal nostril, or RR validation evidence.",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_roots = [
        Path("E:/real") / "\u5b66\u4e60" / "\u547c\u5438\u6587\u732e",
        Path("E:/real") / "\u53ef\u80fd\u7ed3\u5408\u6587\u732e",
    ]
    parser = argparse.ArgumentParser(
        description=(
            "Build a local-literature evidence matrix that connects reviewed "
            "respiratory-rate PDFs to the current thermal RR innovation claims."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--literature-root",
        type=Path,
        action="append",
        default=default_roots,
        help="Folder containing local literature PDFs. Can be repeated.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def normalize_space(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def extract_pdf_text(path: Path, pages: int = 2) -> tuple[int, str, str]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyMuPDF/fitz is required for local PDF extraction") from exc
    with fitz.open(path) as doc:
        page_count = int(doc.page_count)
        title = str((doc.metadata or {}).get("title") or "").strip()
        text_parts = []
        for index in range(min(pages, page_count)):
            text_parts.append(doc.load_page(index).get_text("text"))
    return page_count, title, normalize_space("\n".join(text_parts))


def detect_doi(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0).rstrip(".,;") if match else ""


def read_metric_snapshot(output_dir: Path) -> dict[str, str]:
    result = {
        "internal_best_evidence": "not generated",
        "external_batch_evidence": "not generated",
        "external_missing_for_scoring": "not generated",
    }
    main = output_dir / "paper_main_results_table.csv"
    if main.exists():
        table = pd.read_csv(main)
        match = table[table["method"].astype(str).str.contains("safe gate", case=False, na=False)]
        if not match.empty:
            row = match.iloc[0]
            result["internal_best_evidence"] = (
                f"safe gate RR R2={float(row.get('rr_r2', 0)):.4f}; "
                f"MAE={float(row.get('rr_mae_bpm', 0)):.3f} bpm; "
                f"RMSE={float(row.get('rr_rmse_bpm', 0)):.3f} bpm; "
                f"exact={row.get('exact_count', '')}"
            )
    readiness = output_dir / "paper_external_validation_split_all_use_readiness.csv"
    if readiness.exists():
        table = pd.read_csv(readiness)
        evidence = {
            str(row["check"]): str(row["evidence"]) for _, row in table.iterrows()
        }
        result["external_batch_evidence"] = "; ".join(
            value
            for value in [
                evidence.get("default included clips after duration screen", ""),
                evidence.get("source long-video/session rows available", ""),
                evidence.get("unique cow_id labels available", ""),
                evidence.get("collection dates available", ""),
                evidence.get("104-video Q2 algorithmic external tier by clips", ""),
            ]
            if value
        )
    coverage = output_dir / "paper_external_validation_fieldwork_field_coverage.csv"
    if coverage.exists():
        table = pd.read_csv(coverage)
        blockers = []
        for field in ["manual_breath_count", "reference_rr_annotator", "camera_id"]:
            row = table[table["field"].astype(str).eq(field)]
            if not row.empty and int(float(row.iloc[0].get("nonempty", 0))) == 0:
                blockers.append(field)
        result["external_missing_for_scoring"] = ", ".join(blockers) if blockers else "none"
    return result


def iter_pdf_paths(roots: list[Path]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.pdf")):
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def build_matrix(paths: list[Path], output_dir: Path) -> pd.DataFrame:
    metrics = read_metric_snapshot(output_dir)
    rows: list[dict[str, object]] = []
    for path in paths:
        page_count, meta_title, first_text = extract_pdf_text(path)
        override = SOURCE_OVERRIDES.get(path.name, {})
        source_id = override.get("source_id", path.stem)
        row = {
            "source_id": source_id,
            "file_name": path.name,
            "file_path": str(path),
            "pages": page_count,
            "bytes": path.stat().st_size,
            "year": override.get("year", ""),
            "venue_or_type": override.get("venue_or_type", ""),
            "paper_title": override.get("paper_title", meta_title or first_text[:120]),
            "metadata_title": meta_title,
            "detected_doi": detect_doi(first_text),
            "method_family": override.get("method_family", "needs_manual_review"),
            "literature_gap": override.get("literature_gap", "needs_manual_review"),
            "current_project_response": override.get(
                "current_project_response", "needs_manual_review"
            ),
            "manuscript_role": override.get("manuscript_role", "needs_manual_review"),
            "claim_boundary": override.get("claim_boundary", "needs_manual_review"),
            "current_internal_evidence": metrics["internal_best_evidence"],
            "external_batch_evidence": metrics["external_batch_evidence"],
            "external_missing_for_scoring": metrics["external_missing_for_scoring"],
            "text_preview": first_text[:500],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(data: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if data.empty:
        return "_No rows._"
    view = data[columns].head(max_rows).fillna("").astype(str)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, matrix: pd.DataFrame, output_dir: Path) -> None:
    roles = matrix["manuscript_role"].value_counts().rename_axis("role").reset_index(name="sources")
    text = f"""# Local Literature Evidence Matrix

This report links local respiratory-rate PDFs to the current thermal RR
innovation route. It is a claim-control document: use it to write the
Introduction, Methods motivation, and Discussion boundaries without overstating
the current evidence.

## Current Project Evidence Snapshot

- Internal best non-truth method: `{matrix['current_internal_evidence'].iloc[0] if not matrix.empty else 'not generated'}`
- New external candidate batch: `{matrix['external_batch_evidence'].iloc[0] if not matrix.empty else 'not generated'}`
- External scoring blockers: `{matrix['external_missing_for_scoring'].iloc[0] if not matrix.empty else 'not generated'}`

## Manuscript Roles

{markdown_table(roles, ['role', 'sources'])}

## Source-To-Innovation Matrix

{markdown_table(matrix, ['source_id', 'year', 'venue_or_type', 'method_family', 'literature_gap', 'current_project_response', 'manuscript_role', 'claim_boundary'])}

## Recommended Paper Position

The defensible Q2-or-higher route is not another YOLO-only reproduction. The
paper should be framed as a method-frozen thermal-video RR estimation workflow
with signal-aware residual correction, selective reporting, bilateral
periodicity consistency, uncertainty reporting, and an independent external
validation protocol. The new `split_all_use` batch can satisfy the 104-video
algorithmic external-validation size target after manual breath-count labels,
annotator, and camera ID are filled.

Heat-stress, THI/ATHI, disease, and welfare claims remain out of scope until
real synchronized environment or animal-context labels are available.

Generated assets live in `{output_dir}`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [root.resolve() for root in args.literature_root]
    paths = iter_pdf_paths(roots)
    matrix = build_matrix(paths, output_dir)
    csv_path = output_dir / "paper_literature_local_evidence_matrix.csv"
    md_path = output_dir / "paper_literature_local_evidence_matrix.md"
    matrix.to_csv(csv_path, index=False)
    write_report(md_path, matrix, output_dir)
    print(f"Saved local literature evidence matrix: {csv_path}")
    print(f"Saved local literature evidence report: {md_path}")
    print(f"Scanned PDFs={len(paths)}; curated={int(matrix['source_id'].ne(matrix['file_name']).sum()) if not matrix.empty else 0}")


if __name__ == "__main__":
    main()
