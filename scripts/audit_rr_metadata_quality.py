from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

METADATA_COLUMNS = [
    "video_id",
    "cow_id",
    "collection_date",
    "collection_start_date",
    "collection_end_date",
    "collection_time",
    "collection_location_country",
    "collection_location_province",
    "collection_location_county",
    "collection_site",
    "camera_id",
    "scene_id",
    "ambient_temperature_c",
    "relative_humidity_percent",
    "thi",
    "athi",
    "posture",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
    "operator_or_annotator",
    "external_test_split",
    "notes",
]

FIELD_CLAIMS = {
    "cow_id": "numeric video label copied from video_id; not an animal identity unless independently mapped to real cows",
    "collection_date": "date/session leakage control and date-level grouped validation",
    "collection_start_date": "known collection-window provenance; not a substitute for per-video acquisition date",
    "collection_end_date": "known collection-window provenance; not a substitute for per-video acquisition date",
    "collection_location_country": "collection provenance",
    "collection_location_province": "collection provenance",
    "collection_location_county": "collection provenance",
    "collection_site": "collection provenance",
    "camera_id": "camera-domain robustness analysis",
    "scene_id": "scene/barn-domain robustness analysis",
    "ambient_temperature_c": "claim-specific THI calculation and heat-stress interpretation",
    "relative_humidity_percent": "claim-specific THI calculation and heat-stress interpretation",
    "thi": "claim-specific THI-stratified respiratory-rate and error analysis",
    "athi": "claim-specific adjusted-THI stratified analysis if the formula is defined",
    "head_motion_score_0_3": "claim-specific head-motion robustness analysis",
    "occlusion_score_0_3": "claim-specific occlusion robustness analysis",
    "nostril_visibility_score_0_3": "claim-specific ROI-quality robustness analysis",
    "external_test_split": "split provenance labels; frozen external validation only if independent external rows exist",
}

SCORE_COLUMNS = [
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
]

NUMERIC_RANGES = {
    "ambient_temperature_c": (-30.0, 60.0),
    "relative_humidity_percent": (0.0, 100.0),
    "thi": (0.0, 120.0),
    "athi": (0.0, 140.0),
}

EXTERNAL_SPLITS = {"external", "holdout", "test"}
INTERNAL_SPLITS = {"internal", "train", "training", "validation", "val"}
VALID_SPLITS = EXTERNAL_SPLITS | INTERNAL_SPLITS | {""}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    default_output_dir = (
        default_input_root / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Audit metadata quality, claim unlocks, and external split leakage risk "
            "for the thermal RR manuscript package."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--min-cow-groups", type=int, default=5)
    parser.add_argument("--min-external-videos", type=int, default=50)
    parser.add_argument("--q2-external-videos", type=int, default=104)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def normalize_split(value: object) -> str:
    text = normalize_text(value).lower().replace("-", "_")
    if text in EXTERNAL_SPLITS:
        return "external"
    if text in INTERNAL_SPLITS:
        return "internal"
    if not text:
        return "missing"
    return "other"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_summary(input_root: Path, output_prefix: str) -> pd.DataFrame:
    path = input_root / f"{output_prefix}_summary.csv"
    summary = read_csv(path)
    if "video_id" not in summary.columns:
        raise ValueError(f"Summary CSV is missing video_id: {path}")
    return summary


def load_metadata(metadata_csv: Path, summary: pd.DataFrame) -> pd.DataFrame:
    if metadata_csv.exists():
        metadata = read_csv(metadata_csv)
    else:
        metadata = pd.DataFrame({"video_id": summary["video_id"].astype(str)})
    if "video_id" not in metadata.columns:
        raise ValueError(f"Metadata CSV is missing video_id: {metadata_csv}")
    metadata["video_id"] = metadata["video_id"].astype(str)
    for column in METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata[METADATA_COLUMNS].fillna("").map(normalize_text)
    ordered = pd.DataFrame({"video_id": summary["video_id"].astype(str)})
    merged = ordered.merge(metadata, on="video_id", how="left", validate="one_to_one")
    for column in METADATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    return merged[METADATA_COLUMNS].fillna("").map(normalize_text)


def audit_row(
    category: str,
    check: str,
    status: str,
    evidence: str,
    action: str,
    *,
    q2_blocking: bool,
    claim_unlocked: str = "",
) -> dict[str, object]:
    return {
        "category": category,
        "check": check,
        "status": status,
        "q2_blocking": bool(q2_blocking),
        "evidence": evidence,
        "recommended_action": action,
        "claim_unlocked": claim_unlocked,
    }


def field_status(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(metadata)
    for field, claim in FIELD_CLAIMS.items():
        values = metadata[field].map(normalize_text) if field in metadata.columns else pd.Series([""] * total)
        nonempty = int(values.ne("").sum())
        coverage = nonempty / max(total, 1) * 100.0
        if coverage == 100.0:
            status = PASS
        elif coverage == 0.0:
            status = FAIL
        else:
            status = WARN
        rows.append(
            {
                "field": field,
                "status": status,
                "nonempty": nonempty,
                "total_videos": total,
                "coverage_percent": coverage,
                "unique_values": int(values[values.ne("")].nunique()),
                "claim_unlocked": claim,
            }
        )
    return pd.DataFrame(rows)


def validate_values(metadata: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in SCORE_COLUMNS:
        invalid = []
        values = metadata[column].map(normalize_text)
        for video_id, value in zip(metadata["video_id"], values):
            if value == "":
                continue
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(number) or float(number) not in {0.0, 1.0, 2.0, 3.0}:
                invalid.append(str(video_id))
        rows.append(
            audit_row(
                "value_quality",
                f"{column} valid 0-3 score",
                PASS if not invalid else FAIL,
                f"invalid={len(invalid)}; videos={';'.join(invalid[:20])}",
                f"Use only integer values 0, 1, 2, or 3 for {column}.",
                q2_blocking=bool(invalid),
                claim_unlocked=FIELD_CLAIMS[column],
            )
        )

    for column, (low, high) in NUMERIC_RANGES.items():
        invalid = []
        values = metadata[column].map(normalize_text)
        numeric = pd.to_numeric(values.mask(values.eq(""), np.nan), errors="coerce")
        for video_id, value, number in zip(metadata["video_id"], values, numeric):
            if value == "":
                continue
            if pd.isna(number) or float(number) < low or float(number) > high:
                invalid.append(str(video_id))
        rows.append(
            audit_row(
                "value_quality",
                f"{column} numeric range",
                PASS if not invalid else FAIL,
                f"allowed=[{low}, {high}], invalid={len(invalid)}; videos={';'.join(invalid[:20])}",
                f"Correct out-of-range or nonnumeric values for {column}.",
                q2_blocking=bool(invalid),
                claim_unlocked=FIELD_CLAIMS.get(column, ""),
            )
        )
    for column, regex, example, q2_blocking in [
        ("collection_date", DATE_RE, "YYYY-MM-DD", True),
        ("collection_start_date", DATE_RE, "YYYY-MM-DD", False),
        ("collection_end_date", DATE_RE, "YYYY-MM-DD", False),
        ("collection_time", TIME_RE, "HH:MM or HH:MM:SS", False),
    ]:
        invalid = []
        values = metadata[column].map(normalize_text)
        for video_id, value in zip(metadata["video_id"], values):
            if value and not regex.match(value):
                invalid.append(str(video_id))
        rows.append(
            audit_row(
                "value_quality",
                f"{column} format",
                PASS if not invalid else FAIL,
                f"expected={example}, invalid={len(invalid)}; videos={';'.join(invalid[:20])}",
                f"Use {example} for {column}.",
                q2_blocking=bool(invalid) and q2_blocking,
                claim_unlocked=FIELD_CLAIMS.get(column, ""),
            )
        )
    return rows


def structure_checks(metadata: pd.DataFrame, summary: pd.DataFrame, metadata_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = int(len(summary))
    rows.append(
        audit_row(
            "metadata_structure",
            "metadata file exists",
            PASS if metadata_csv.exists() else FAIL,
            str(metadata_csv),
            "Create paper_metadata_template.csv or run scripts/build_rr_paper_assets.py.",
            q2_blocking=not metadata_csv.exists(),
        )
    )
    duplicates = metadata["video_id"][metadata["video_id"].duplicated()].unique().tolist()
    rows.append(
        audit_row(
            "metadata_structure",
            "one metadata row per summary video",
            PASS if len(metadata) == total and not duplicates else FAIL,
            f"metadata_rows={len(metadata)}, summary_rows={total}, duplicate_ids={';'.join(duplicates[:20])}",
            "Keep exactly one metadata row for every processed video.",
            q2_blocking=bool(len(metadata) != total or duplicates),
        )
    )
    return rows


def claim_readiness_checks(
    metadata: pd.DataFrame,
    fields: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = int(len(metadata))
    cow = metadata["cow_id"].map(normalize_text)
    cow_nonempty = int(cow.ne("").sum())
    cow_label_groups = int(cow[cow.ne("")].nunique())
    cow_ready = cow_nonempty == total and cow_label_groups >= 2
    rows.append(
        audit_row(
            "claim_readiness",
            "cow_id video-label grouping complete",
            PASS if cow_ready else FAIL,
            f"nonempty={cow_nonempty}/{total}, unique_video_labels={cow_label_groups}",
            (
                "Use cow_id only as the numeric video label in the current dataset; "
                "do not use animal-level wording unless a separate real-cow identity map exists."
            ),
            q2_blocking=not cow_ready,
            claim_unlocked=FIELD_CLAIMS["cow_id"],
        )
    )
    rows.append(
        audit_row(
            "claim_readiness",
            f"video-label groups >= {args.min_cow_groups} for label-group stress testing",
            PASS if cow_label_groups >= args.min_cow_groups else WARN,
            f"unique_video_labels={cow_label_groups}, target={args.min_cow_groups}",
            (
                "This supports a video-label grouped stress test only. For Q2+ animal-level "
                "evidence, collect or recover true cow identities."
            ),
            q2_blocking=False,
            claim_unlocked="video-label grouped stress-test design",
        )
    )

    temp = metadata["ambient_temperature_c"].map(normalize_text)
    rh = metadata["relative_humidity_percent"].map(normalize_text)
    thi = metadata["thi"].map(normalize_text)
    heat_ready = (temp.ne("") & rh.ne("")) | thi.ne("")
    rows.append(
        audit_row(
            "claim_readiness",
            "THI or temperature+humidity ready",
            PASS if bool(heat_ready.all()) else FAIL,
            f"ready={int(heat_ready.sum())}/{total}",
            (
                "Fill synchronized barn temperature/humidity or THI only if heat-stress "
                "or THI-stratified claims will be made; otherwise keep this as a limitation."
            ),
            q2_blocking=False,
            claim_unlocked="heat-stress interpretation and THI-stratified analysis",
        )
    )

    for column in SCORE_COLUMNS:
        nonempty = int(metadata[column].map(normalize_text).ne("").sum())
        ready = nonempty == total
        rows.append(
            audit_row(
                "claim_readiness",
                f"{column} robustness stratification ready",
                PASS if ready else FAIL,
                f"nonempty={nonempty}/{total}",
                (
                    f"Score {column} for all videos only if robustness stratification "
                    "will be claimed; leave blank and report as unavailable otherwise."
                ),
                q2_blocking=False,
                claim_unlocked=FIELD_CLAIMS[column],
            )
        )

    split_values = metadata["external_test_split"].map(normalize_text)
    split_norm = split_values.map(normalize_split)
    split_nonempty = int(split_values.ne("").sum())
    external_n = int(split_norm.eq("external").sum())
    internal_n = int(split_norm.eq("internal").sum())
    other_n = int(split_norm.eq("other").sum())
    split_ready = split_nonempty == total and other_n == 0
    rows.append(
        audit_row(
            "claim_readiness",
            "external split labels ready",
            PASS if split_ready else FAIL,
            f"nonempty={split_nonempty}/{total}, internal={internal_n}, external={external_n}, other={other_n}",
            (
                "Current rows can be internal for provenance; independent external/holdout "
                "rows are required before external validation claims."
            ),
            q2_blocking=not split_ready,
            claim_unlocked=FIELD_CLAIMS["external_test_split"],
        )
    )
    rows.append(
        audit_row(
            "claim_readiness",
            f"external videos >= {args.min_external_videos} minimum feasibility tier",
            PASS if external_n >= args.min_external_videos else WARN,
            f"external={external_n}, target={args.min_external_videos}",
            "Collect or mark enough frozen external videos for a credible feasibility tier.",
            q2_blocking=False,
            claim_unlocked="minimum external feasibility evidence",
        )
    )
    rows.append(
        audit_row(
            "claim_readiness",
            f"external videos >= {args.q2_external_videos} Q2 absolute-performance target",
            PASS if external_n >= args.q2_external_videos else WARN,
            f"external={external_n}, target={args.q2_external_videos}",
            "Use the external sample plan to reach the Q2-oriented absolute-performance target.",
            q2_blocking=False,
            claim_unlocked="Q2-oriented external absolute-performance claim",
        )
    )
    return rows


def split_leakage_checks(metadata: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    merged = metadata.merge(summary, on="video_id", how="left", validate="one_to_one")
    merged["split_pool"] = merged["external_test_split"].map(normalize_split)
    rows: list[dict[str, object]] = []
    external = merged[merged["split_pool"] == "external"].copy()
    internal = merged[merged["split_pool"] == "internal"].copy()
    if external.empty:
        rows.append(
            audit_row(
                "split_leakage",
                "external split leakage screen evaluable",
                WARN,
                "external rows=0",
                (
                    "Add/import independent external rows before evaluating leakage "
                    "or balance risk; do not relabel current development rows as external."
                ),
                q2_blocking=False,
                claim_unlocked="frozen external validation claim",
            )
        )
        return pd.DataFrame(rows)

    rows.append(
        audit_row(
            "split_leakage",
            "external split leakage screen evaluable",
            PASS,
            f"external rows={len(external)}, internal rows={len(internal)}",
            "Review the following balance checks before reporting external metrics.",
            q2_blocking=False,
            claim_unlocked="external split auditability",
        )
    )

    for field, claim in [
        ("cow_id", "video-label split transparency; animal-independent validation only with real cow IDs"),
        ("collection_date", "date/session-independent external validation"),
        ("camera_id", "camera-independent external validation"),
        ("scene_id", "scene-independent external validation"),
    ]:
        ext_values = set(external[field].map(normalize_text)) - {""}
        int_values = set(internal[field].map(normalize_text)) - {""}
        if not ext_values:
            status = WARN
            evidence = "external values missing"
        else:
            overlap = sorted(ext_values & int_values)
            status = PASS if not overlap else WARN
            evidence = f"external_unique={len(ext_values)}, overlap_with_internal={len(overlap)}; overlap={';'.join(overlap[:20])}"
        rows.append(
            audit_row(
                "split_leakage",
                f"{field} external/internal overlap screen",
                status,
                evidence,
                f"If claiming {claim}, keep external {field} independent from development data or explain the design.",
                q2_blocking=False,
                claim_unlocked=claim,
            )
        )

    for metric in ["abs_count_error", "truth_rr"]:
        if metric not in merged.columns or internal.empty:
            continue
        ext_metric = pd.to_numeric(external[metric], errors="coerce").dropna()
        int_metric = pd.to_numeric(internal[metric], errors="coerce").dropna()
        if ext_metric.empty or int_metric.empty:
            continue
        delta = float(ext_metric.mean() - int_metric.mean())
        status = PASS if abs(delta) <= (0.25 if metric == "abs_count_error" else 5.0) else WARN
        rows.append(
            audit_row(
                "split_balance",
                f"{metric} external/internal balance screen",
                status,
                f"external_mean={ext_metric.mean():.4f}, internal_mean={int_metric.mean():.4f}, delta={delta:.4f}",
                "External split should be frozen by collection design; large outcome imbalance needs transparent explanation.",
                q2_blocking=False,
                claim_unlocked="external split balance transparency",
            )
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ").replace("|", "/") for value in row) + " |"
        for row in table.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    audit: pd.DataFrame,
    fields: pd.DataFrame,
    split_audit: pd.DataFrame,
    output_path: Path,
) -> None:
    q2_fail = audit[audit["q2_blocking"].astype(bool) & audit["status"].eq(FAIL)]
    if q2_fail.empty:
        q2_status = "metadata_claim_ready"
    else:
        q2_status = "metadata_claim_not_ready"
    text = [
        "# Metadata Quality And Split Leakage Audit",
        "",
        f"Q2 metadata claim status: `{q2_status}`",
        "",
        "This audit checks whether filled metadata can support video-label grouped stress tests, heat-stress interpretation, robustness stratification, and frozen external-validation claims. The current `cow_id` is a numeric video label, not confirmed animal identity. It does not replace running the downstream validation scripts.",
        "",
        "## Q2 Blocking Checks",
        "",
    ]
    if q2_fail.empty:
        text.append("No Q2-blocking metadata checks failed.")
    else:
        text.append(
            markdown_table(
                q2_fail,
                ["category", "check", "status", "evidence", "recommended_action"],
            )
        )
    text.extend(
        [
            "",
            "## Field Claim Coverage",
            "",
            markdown_table(
                fields,
                [
                    "field",
                    "status",
                    "nonempty",
                    "total_videos",
                    "coverage_percent",
                    "claim_unlocked",
                ],
            ),
            "",
            "## Split Leakage And Balance Screens",
            "",
            markdown_table(
                split_audit,
                ["category", "check", "status", "evidence", "recommended_action"],
            ),
            "",
            "## All Metadata Checks",
            "",
            markdown_table(
                audit,
                [
                    "category",
                    "check",
                    "status",
                    "q2_blocking",
                    "evidence",
                    "claim_unlocked",
                ],
            ),
        ]
    )
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else output_dir / "paper_metadata_template.csv"
    )

    summary = load_summary(input_root, args.output_prefix)
    metadata = load_metadata(metadata_csv, summary)
    fields = field_status(metadata)
    audit_rows = []
    audit_rows.extend(structure_checks(metadata, summary, metadata_csv))
    audit_rows.extend(validate_values(metadata))
    audit_rows.extend(claim_readiness_checks(metadata, fields, args))
    audit = pd.DataFrame(audit_rows)
    split_audit = split_leakage_checks(metadata, summary)

    audit_path = output_dir / "paper_metadata_quality_audit.csv"
    field_path = output_dir / "paper_metadata_quality_field_status.csv"
    split_path = output_dir / "paper_metadata_split_leakage_audit.csv"
    report_path = output_dir / "paper_metadata_quality_audit.md"
    audit.to_csv(audit_path, index=False)
    fields.to_csv(field_path, index=False)
    split_audit.to_csv(split_path, index=False)
    write_report(audit, fields, split_audit, report_path)

    print(f"Saved metadata quality audit: {audit_path}")
    print(f"Saved metadata field status: {field_path}")
    print(f"Saved split leakage audit: {split_path}")
    print(f"Saved metadata quality report: {report_path}")
    print(audit[["category", "check", "status", "q2_blocking"]].to_string(index=False))


if __name__ == "__main__":
    main()
