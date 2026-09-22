from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd


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

Q2_REQUIRED_FIELDS = [
    "cow_id",
    "collection_start_date",
    "collection_end_date",
    "collection_location_country",
    "collection_location_province",
    "collection_location_county",
    "collection_site",
    "scene_id",
    "external_test_split",
]

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

FIELD_CLAIMS = {
    "cow_id": "numeric video label copied from video_id; animal-level generalization only if labels map to real cows",
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

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
EXTERNAL_SPLITS = {"external", "holdout", "test"}
INTERNAL_SPLITS = {"internal", "train", "training", "validation", "val"}
ALLOWED_SPLITS = EXTERNAL_SPLITS | INTERNAL_SPLITS | {""}

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Preflight-check hand-filled RR metadata before import, grouped validation, "
            "and external split evaluation."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Filled metadata CSV to check. Defaults to the paper-assets fill template.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any Q2-blocking preflight check fails.",
    )
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
    if text == "":
        return ""
    return "other"


def default_metadata_csv(input_root: Path, corrected_prefix: str) -> Path:
    paper_assets = input_root / f"{corrected_prefix}_paper_assets"
    candidates = [
        paper_assets / "paper_metadata_annotation_fill_template.csv",
        paper_assets / "paper_metadata_template.csv",
        paper_assets / "paper_metadata_annotation_sheet.csv",
        input_root / "paper_repro_metadata_annotation_sheet.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_summary(input_root: Path, output_prefix: str) -> pd.DataFrame:
    summary_path = input_root / f"{output_prefix}_summary.csv"
    summary = read_csv(summary_path)
    if "video_id" not in summary.columns:
        raise ValueError(f"Summary CSV is missing video_id: {summary_path}")
    summary["video_id"] = summary["video_id"].map(normalize_text)
    return summary[summary["video_id"].ne("")].copy()


def load_metadata(metadata_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata = read_csv(metadata_csv)
    original_columns = list(metadata.columns)
    if "video_id" in metadata.columns:
        metadata["video_id"] = metadata["video_id"].map(normalize_text)
    else:
        metadata["video_id"] = ""
    for column in METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata[METADATA_COLUMNS].fillna("").map(normalize_text)
    return metadata, original_columns


def add_check(
    rows: list[dict[str, object]],
    category: str,
    check: str,
    status: str,
    q2_blocking: bool,
    affected_rows: int,
    evidence: str,
    recommended_action: str,
) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "status": status,
            "q2_blocking": bool(q2_blocking),
            "affected_rows": int(affected_rows),
            "evidence": evidence,
            "recommended_action": recommended_action,
        }
    )


def unique_by_video(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata[metadata["video_id"].ne("")].drop_duplicates("video_id", keep="first")


def structure_checks(
    metadata: pd.DataFrame,
    original_columns: list[str],
    summary: pd.DataFrame,
    metadata_csv: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_ids = set(summary["video_id"].astype(str))
    present_ids = set(metadata["video_id"].astype(str)) - {""}
    missing_columns = [column for column in METADATA_COLUMNS if column not in original_columns]
    empty_video_ids = int(metadata["video_id"].eq("").sum())
    duplicate_ids = metadata.loc[
        metadata["video_id"].ne("") & metadata["video_id"].duplicated(), "video_id"
    ].unique()
    missing_videos = sorted(expected_ids - present_ids)
    unknown_videos = sorted(present_ids - expected_ids)

    add_check(
        rows,
        "structure",
        "metadata CSV exists",
        PASS if metadata_csv.exists() else FAIL,
        q2_blocking=not metadata_csv.exists(),
        affected_rows=0 if metadata_csv.exists() else 1,
        evidence=str(metadata_csv),
        recommended_action="Create the metadata CSV from the paper metadata template.",
    )
    add_check(
        rows,
        "structure",
        "required metadata columns present",
        PASS if not missing_columns else FAIL,
        q2_blocking=bool(missing_columns),
        affected_rows=len(missing_columns),
        evidence="missing=" + ";".join(missing_columns[:20]),
        recommended_action="Keep the template columns unchanged before import.",
    )
    add_check(
        rows,
        "structure",
        "video_id values are nonempty",
        PASS if empty_video_ids == 0 else FAIL,
        q2_blocking=empty_video_ids > 0,
        affected_rows=empty_video_ids,
        evidence=f"empty_video_id_rows={empty_video_ids}",
        recommended_action="Do not edit or blank out video_id values.",
    )
    add_check(
        rows,
        "structure",
        "video_id values are unique",
        PASS if len(duplicate_ids) == 0 else FAIL,
        q2_blocking=len(duplicate_ids) > 0,
        affected_rows=len(duplicate_ids),
        evidence="duplicates=" + ";".join(map(str, duplicate_ids[:20])),
        recommended_action="Keep exactly one row per video_id.",
    )
    add_check(
        rows,
        "structure",
        "all summary videos covered",
        PASS if not missing_videos and not unknown_videos else FAIL,
        q2_blocking=bool(missing_videos or unknown_videos),
        affected_rows=len(missing_videos) + len(unknown_videos),
        evidence=(
            f"expected={len(expected_ids)}, present={len(present_ids)}, "
            f"missing={len(missing_videos)}, unknown={len(unknown_videos)}"
        ),
        recommended_action="Use the current paper metadata template and keep all 73 video rows.",
    )
    return rows


def field_coverage(metadata: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    merged = summary[["video_id"]].merge(unique_by_video(metadata), on="video_id", how="left")
    rows = []
    total = int(len(merged))
    for column in METADATA_COLUMNS:
        if column == "video_id":
            continue
        values = merged[column].map(normalize_text) if column in merged.columns else pd.Series([""] * total)
        nonempty = int(values.ne("").sum())
        coverage = nonempty / max(total, 1) * 100.0
        q2_required = column in Q2_REQUIRED_FIELDS
        if coverage == 100.0:
            status = PASS
        elif coverage == 0.0 and q2_required:
            status = FAIL
        else:
            status = WARN
        rows.append(
            {
                "field": column,
                "status": status,
                "q2_required": q2_required,
                "nonempty": nonempty,
                "total_videos": total,
                "coverage_percent": coverage,
                "unique_values": int(values[values.ne("")].nunique()),
                "claim_unlocked": FIELD_CLAIMS.get(column, ""),
            }
        )
    return pd.DataFrame(rows)


def invalid_value_map(metadata: pd.DataFrame) -> dict[str, list[str]]:
    invalid: dict[str, list[str]] = {}

    def mark(video_id: object, field: str) -> None:
        key = normalize_text(video_id)
        if key:
            invalid.setdefault(key, []).append(field)

    for column in SCORE_COLUMNS:
        for _, row in metadata.iterrows():
            value = normalize_text(row[column])
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                mark(row["video_id"], column)
                continue
            if number not in {0.0, 1.0, 2.0, 3.0}:
                mark(row["video_id"], column)

    for column, (low, high) in NUMERIC_RANGES.items():
        for _, row in metadata.iterrows():
            value = normalize_text(row[column])
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                mark(row["video_id"], column)
                continue
            if not (low <= number <= high):
                mark(row["video_id"], column)

    for _, row in metadata.iterrows():
        split = normalize_text(row["external_test_split"]).lower().replace("-", "_")
        if split not in ALLOWED_SPLITS:
            mark(row["video_id"], "external_test_split")
        for date_column in [
            "collection_date",
            "collection_start_date",
            "collection_end_date",
        ]:
            date_value = normalize_text(row[date_column])
            if date_value and not DATE_RE.match(date_value):
                mark(row["video_id"], date_column)
        time_value = normalize_text(row["collection_time"])
        if time_value and not TIME_RE.match(time_value):
            mark(row["video_id"], "collection_time")

    return invalid


def value_and_claim_checks(
    metadata: pd.DataFrame,
    fields: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    merged = summary[["video_id"]].merge(unique_by_video(metadata), on="video_id", how="left")
    total = int(len(merged))

    for column in SCORE_COLUMNS:
        invalid = []
        for _, row in merged.iterrows():
            value = normalize_text(row[column])
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                invalid.append(str(row["video_id"]))
                continue
            if number not in {0.0, 1.0, 2.0, 3.0}:
                invalid.append(str(row["video_id"]))
        add_check(
            rows,
            "values",
            f"{column} values in 0-3",
            PASS if not invalid else FAIL,
            q2_blocking=bool(invalid),
            affected_rows=len(invalid),
            evidence="invalid_video_ids=" + ";".join(invalid[:20]),
            recommended_action=f"Use integer scores 0, 1, 2, or 3 for {column}.",
        )

    for column, (low, high) in NUMERIC_RANGES.items():
        invalid = []
        for _, row in merged.iterrows():
            value = normalize_text(row[column])
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                invalid.append(str(row["video_id"]))
                continue
            if not (low <= number <= high):
                invalid.append(str(row["video_id"]))
        add_check(
            rows,
            "values",
            f"{column} numeric range",
            PASS if not invalid else FAIL,
            q2_blocking=bool(invalid),
            affected_rows=len(invalid),
            evidence=f"range=[{low}, {high}], invalid_video_ids=" + ";".join(invalid[:20]),
            recommended_action=f"Correct out-of-range or nonnumeric values for {column}.",
        )

    for column, regex, example, q2_blocking in [
        ("collection_date", DATE_RE, "YYYY-MM-DD", True),
        ("collection_start_date", DATE_RE, "YYYY-MM-DD", False),
        ("collection_end_date", DATE_RE, "YYYY-MM-DD", False),
        ("collection_time", TIME_RE, "HH:MM or HH:MM:SS", False),
    ]:
        invalid = [
            str(row["video_id"])
            for _, row in merged.iterrows()
            if normalize_text(row[column]) and not regex.match(normalize_text(row[column]))
        ]
        add_check(
            rows,
            "values",
            f"{column} format",
            PASS if not invalid else FAIL,
            q2_blocking=bool(invalid) and q2_blocking,
            affected_rows=len(invalid),
            evidence="invalid_video_ids=" + ";".join(invalid[:20]),
            recommended_action=f"Use {example} for {column}.",
        )

    split_values = merged["external_test_split"].map(normalize_text)
    split_norm = split_values.map(normalize_split)
    invalid_splits = [
        str(video_id)
        for video_id, value in zip(merged["video_id"], split_values)
        if value.lower().replace("-", "_") not in ALLOWED_SPLITS
    ]
    internal_n = int(split_norm.eq("internal").sum())
    external_n = int(split_norm.eq("external").sum())
    other_n = int(split_norm.eq("other").sum())
    split_ready = split_values.ne("").all() and internal_n > 0 and not invalid_splits
    add_check(
        rows,
        "values",
        "external_test_split controlled values",
        PASS if not invalid_splits else FAIL,
        q2_blocking=bool(invalid_splits),
        affected_rows=len(invalid_splits),
        evidence="invalid_video_ids=" + ";".join(invalid_splits[:20]),
        recommended_action="Use internal/train/validation or external/holdout/test controlled values.",
    )
    add_check(
        rows,
        "claim_readiness",
        "external split labels ready",
        PASS if split_ready else FAIL,
        q2_blocking=not split_ready,
        affected_rows=int(split_values.eq("").sum()) + other_n,
        evidence=(
            f"nonempty={int(split_values.ne('').sum())}/{total}, "
            f"internal={internal_n}, external={external_n}, other={other_n}"
        ),
        recommended_action="Current rows may remain internal for provenance; independent external/holdout rows are required before external validation claims.",
    )

    cow = merged["cow_id"].map(normalize_text)
    cow_nonempty = int(cow.ne("").sum())
    cow_label_groups = int(cow[cow.ne("")].nunique())
    cow_ready = cow_nonempty == total and cow_label_groups >= 2
    add_check(
        rows,
        "claim_readiness",
        "cow_id video-label grouping complete",
        PASS if cow_ready else FAIL,
        q2_blocking=not cow_ready,
        affected_rows=int(cow.eq("").sum()),
        evidence=f"nonempty={cow_nonempty}/{total}, unique_video_labels={cow_label_groups}",
        recommended_action=(
            "Use cow_id only as the current numeric video label; verify a separate "
            "real animal identity map before animal-level wording."
        ),
    )
    if cow_nonempty == total:
        add_check(
            rows,
            "claim_readiness",
            "cow_id video-label diversity screen",
            PASS if cow_label_groups >= 5 else WARN,
            q2_blocking=False,
            affected_rows=max(0, 5 - cow_label_groups),
            evidence=f"unique_video_labels={cow_label_groups}",
            recommended_action=(
                "This is only a video-label diversity screen. For stronger Q2 "
                "animal-level wording, recover true cow identities or collect external "
                "videos with real cow IDs."
            ),
        )

    for column, claim in [
        ("collection_date", "date/session leakage control and date-level grouped validation"),
        ("camera_id", "camera-domain robustness analysis"),
        ("scene_id", "scene/barn-domain robustness analysis"),
    ]:
        values = merged[column].map(normalize_text)
        nonempty = int(values.ne("").sum())
        ready = nonempty == total
        add_check(
            rows,
            "claim_readiness",
            f"{column} complete",
            PASS if ready else FAIL,
            q2_blocking=False if column in {"collection_date", "camera_id"} else not ready,
            affected_rows=total - nonempty,
            evidence=f"nonempty={nonempty}/{total}, unique={int(values[values.ne('')].nunique())}",
            recommended_action=f"Fill real {column} for every video before claiming {claim}.",
        )

    temp = merged["ambient_temperature_c"].map(normalize_text)
    rh = merged["relative_humidity_percent"].map(normalize_text)
    thi = merged["thi"].map(normalize_text)
    heat_ready = (temp.ne("") & rh.ne("")) | thi.ne("")
    add_check(
        rows,
        "claim_readiness",
        "THI or temperature+humidity ready",
        PASS if bool(heat_ready.all()) else FAIL,
        q2_blocking=False,
        affected_rows=int((~heat_ready).sum()),
        evidence=f"ready={int(heat_ready.sum())}/{total}",
        recommended_action="Use real synchronized barn temperature/humidity or THI only; leave blank and report as unavailable if records do not exist.",
    )

    for column in SCORE_COLUMNS:
        row = fields[fields["field"].eq(column)].iloc[0]
        ready = int(row["nonempty"]) == total
        add_check(
            rows,
            "claim_readiness",
            f"{column} complete",
            PASS if ready else FAIL,
            q2_blocking=False,
            affected_rows=total - int(row["nonempty"]),
            evidence=f"nonempty={int(row['nonempty'])}/{total}",
            recommended_action=f"Use manual visual scores for {column} only; leave blank and avoid robustness claims if scores cannot be produced.",
        )

    return rows


def method_specs(input_root: Path, output_prefix: str, corrected_prefix: str) -> list[dict[str, str]]:
    return [
        {
            "method": "default",
            "path": str(input_root / f"{output_prefix}_summary.csv"),
            "rr_col": "rr_bpm",
            "abs_count_col": "abs_count_error",
        },
        {
            "method": "quality_residual_fixed_oof",
            "path": str(input_root / f"{corrected_prefix}_predictions.csv"),
            "rr_col": "corrected_rr_bpm",
            "abs_count_col": "corrected_abs_count_error",
        },
        {
            "method": "signal_aware_safe_gate",
            "path": str(input_root / f"{output_prefix}_signal_aware_safe_policy_predictions.csv"),
            "rr_col": "signal_aware_safe_final_rr_bpm",
            "abs_count_col": "signal_aware_safe_abs_count_error",
        },
    ]


def compute_r2(truth: pd.Series, pred: pd.Series) -> float:
    values = pd.DataFrame({"truth": pd.to_numeric(truth, errors="coerce"), "pred": pd.to_numeric(pred, errors="coerce")})
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    sst = float(((values["truth"] - values["truth"].mean()) ** 2).sum())
    if sst == 0.0:
        return np.nan
    sse = float(((values["truth"] - values["pred"]) ** 2).sum())
    return 1.0 - sse / sst


def split_risk_table(
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    input_root: Path,
    output_prefix: str,
    corrected_prefix: str,
) -> pd.DataFrame:
    split_table = summary[["video_id"]].merge(unique_by_video(metadata), on="video_id", how="left")
    split_table["split_pool"] = split_table["external_test_split"].map(normalize_split)
    rows: list[dict[str, object]] = []

    for spec in method_specs(input_root, output_prefix, corrected_prefix):
        path = Path(spec["path"])
        if not path.exists():
            rows.append(
                {
                    "method": spec["method"],
                    "status": WARN,
                    "split_pool": "all",
                    "videos": 0,
                    "rr_r2": np.nan,
                    "mean_abs_count_error": np.nan,
                    "exact_count_rate": np.nan,
                    "risk_note": f"Prediction file missing: {path}",
                }
            )
            continue
        predictions = pd.read_csv(path)
        needed = {"video_id", "truth_rr", spec["rr_col"], spec["abs_count_col"]}
        if not needed.issubset(predictions.columns):
            rows.append(
                {
                    "method": spec["method"],
                    "status": WARN,
                    "split_pool": "all",
                    "videos": 0,
                    "rr_r2": np.nan,
                    "mean_abs_count_error": np.nan,
                    "exact_count_rate": np.nan,
                    "risk_note": "Prediction columns missing: " + ";".join(sorted(needed - set(predictions.columns))),
                }
            )
            continue
        merged = split_table[["video_id", "split_pool"]].merge(
            predictions[["video_id", "truth_rr", spec["rr_col"], spec["abs_count_col"]]],
            on="video_id",
            how="left",
        )
        for pool in ["internal", "external"]:
            subset = merged[merged["split_pool"].eq(pool)].copy()
            abs_count = pd.to_numeric(subset[spec["abs_count_col"]], errors="coerce")
            videos = int(len(subset))
            if videos == 0:
                risk_note = "split pool is empty"
                status = FAIL if pool == "external" else WARN
            elif videos < max(10, int(round(len(split_table) * 0.2))):
                risk_note = "split pool is small; confidence intervals will be weak"
                status = WARN
            else:
                risk_note = "split pool size is usable for a first external screen"
                status = PASS
            rows.append(
                {
                    "method": spec["method"],
                    "status": status,
                    "split_pool": pool,
                    "videos": videos,
                    "rr_r2": compute_r2(subset["truth_rr"], subset[spec["rr_col"]]),
                    "mean_abs_count_error": float(abs_count.mean()) if abs_count.notna().any() else np.nan,
                    "exact_count_rate": float(abs_count.eq(0).mean()) if videos else np.nan,
                    "risk_note": risk_note,
                }
            )

        external = merged[merged["split_pool"].eq("external")]
        internal = merged[merged["split_pool"].eq("internal")]
        if not external.empty and not internal.empty:
            ext_abs = pd.to_numeric(external[spec["abs_count_col"]], errors="coerce")
            int_abs = pd.to_numeric(internal[spec["abs_count_col"]], errors="coerce")
            exact_delta = float(ext_abs.eq(0).mean() - int_abs.eq(0).mean())
            mae_delta = float(ext_abs.mean() - int_abs.mean())
            status = WARN if abs(exact_delta) > 0.25 or abs(mae_delta) > 1.0 else PASS
            rows.append(
                {
                    "method": spec["method"],
                    "status": status,
                    "split_pool": "external_minus_internal",
                    "videos": int(len(external) + len(internal)),
                    "rr_r2": np.nan,
                    "mean_abs_count_error": mae_delta,
                    "exact_count_rate": exact_delta,
                    "risk_note": (
                        "Large performance imbalance; verify external split was fixed before error review."
                        if status == WARN
                        else "No large outcome imbalance detected in the filled split labels."
                    ),
                }
            )
    return pd.DataFrame(rows)


def split_design_checks(metadata: pd.DataFrame, summary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    merged = summary[["video_id"]].merge(unique_by_video(metadata), on="video_id", how="left")
    merged["split_pool"] = merged["external_test_split"].map(normalize_split)
    external = merged[merged["split_pool"].eq("external")].copy()
    internal = merged[merged["split_pool"].eq("internal")].copy()

    if external.empty:
        add_check(
            rows,
            "split_design",
            "external split has rows",
            WARN,
            q2_blocking=False,
            affected_rows=0,
            evidence="external rows=0",
            recommended_action=(
                "Keep current development rows internal. Add method-frozen independent "
                "external/holdout rows before making external-validation claims."
            ),
        )
        return rows

    add_check(
        rows,
        "split_design",
        "external split has rows",
        PASS,
        q2_blocking=False,
        affected_rows=int(len(external)),
        evidence=f"external rows={len(external)}, internal rows={len(internal)}",
        recommended_action="Keep the external split frozen after this point.",
    )

    for field in ["cow_id", "collection_date", "camera_id", "scene_id"]:
        ext_values = set(external[field].map(normalize_text)) - {""}
        int_values = set(internal[field].map(normalize_text)) - {""}
        if not ext_values:
            status = WARN
            evidence = "external values missing"
        elif len(ext_values) == 1:
            status = WARN
            evidence = f"external_unique=1, value={next(iter(ext_values))}"
        else:
            overlap = ext_values & int_values
            status = PASS
            evidence = (
                f"external_unique={len(ext_values)}, internal_unique={len(int_values)}, "
                f"overlap={len(overlap)}"
            )
        add_check(
            rows,
            "split_design",
            f"{field} external diversity screen",
            status,
            q2_blocking=False,
            affected_rows=0 if status == PASS else int(len(external)),
            evidence=evidence,
            recommended_action=(
                f"If claiming {field}-independent validation, keep external {field} "
                "separate from development data or state the limitation."
            ),
        )
    return rows


def video_issue_table(metadata: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    merged = summary[["video_id"]].merge(unique_by_video(metadata), on="video_id", how="left")
    invalid = invalid_value_map(metadata)
    rows = []
    for _, row in merged.iterrows():
        video_id = str(row["video_id"])
        missing = [
            column
            for column in Q2_REQUIRED_FIELDS
            if normalize_text(row.get(column, "")) == ""
        ]
        invalid_fields = sorted(set(invalid.get(video_id, [])))
        split_pool = normalize_split(row.get("external_test_split", ""))
        rows.append(
            {
                "video_id": video_id,
                "missing_q2_fields": ";".join(missing),
                "invalid_fields": ";".join(invalid_fields),
                "split_pool": split_pool,
                "q2_ready": not missing and not invalid_fields and split_pool in {"internal", "external"},
                "issue_count": len(missing) + len(invalid_fields) + (0 if split_pool in {"internal", "external"} else 1),
            }
        )
    issues = pd.DataFrame(rows)
    return issues.sort_values(["q2_ready", "issue_count", "video_id"], ascending=[True, False, True])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 25) -> str:
    table = df[columns].head(max_rows).copy()
    formatted = table.fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    checks: pd.DataFrame,
    fields: pd.DataFrame,
    issues: pd.DataFrame,
    split_risk: pd.DataFrame,
    report_path: Path,
    metadata_csv: Path,
    strict: bool,
) -> None:
    blocking = checks[checks["q2_blocking"].astype(bool) & checks["status"].eq(FAIL)]
    warnings = checks[checks["status"].eq(WARN)]
    status = "ready" if blocking.empty else "not_ready"
    lines = [
        "# RR Metadata Preflight Report",
        "",
        f"Metadata CSV: `{metadata_csv}`",
        f"Strict mode: `{strict}`",
        f"Q2 metadata preflight status: `{status}`",
        f"Q2-blocking failures: `{len(blocking)}`",
        f"Warnings: `{len(warnings)}`",
        "",
        "This preflight validates the hand-filled metadata before import. It does not create the external split and does not tune any model from the split results.",
        "",
        "## Blocking Checks",
        "",
    ]
    if blocking.empty:
        lines.append("No Q2-blocking failures detected.")
    else:
        lines.append(
            markdown_table(
                blocking,
                ["category", "check", "status", "affected_rows", "evidence", "recommended_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Field Coverage",
            "",
            markdown_table(
                fields,
                ["field", "status", "nonempty", "total_videos", "coverage_percent", "claim_unlocked"],
            ),
            "",
            "## Highest Priority Video Issues",
            "",
            markdown_table(
                issues[issues["q2_ready"].eq(False)],
                ["video_id", "missing_q2_fields", "invalid_fields", "split_pool", "issue_count"],
            ),
            "",
            "## Split Risk Screen",
            "",
        ]
    )
    if split_risk.empty:
        lines.append("No split risk table was generated.")
    else:
        lines.append(
            markdown_table(
                split_risk,
                ["method", "status", "split_pool", "videos", "rr_r2", "mean_abs_count_error", "exact_count_rate", "risk_note"],
            )
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else default_metadata_csv(input_root, args.corrected_prefix).resolve()
    )

    summary = load_summary(input_root, args.output_prefix)
    metadata, original_columns = load_metadata(metadata_csv)
    fields = field_coverage(metadata, summary)
    check_rows: list[dict[str, object]] = []
    check_rows.extend(structure_checks(metadata, original_columns, summary, metadata_csv))
    check_rows.extend(value_and_claim_checks(metadata, fields, summary))
    check_rows.extend(split_design_checks(metadata, summary))
    checks = pd.DataFrame(check_rows)
    issues = video_issue_table(metadata, summary)
    split_risk = split_risk_table(
        metadata,
        summary,
        input_root,
        args.output_prefix,
        args.corrected_prefix,
    )

    checks_path = output_dir / "paper_metadata_preflight_checks.csv"
    fields_path = output_dir / "paper_metadata_preflight_field_coverage.csv"
    issues_path = output_dir / "paper_metadata_preflight_video_issues.csv"
    split_risk_path = output_dir / "paper_metadata_preflight_split_risk.csv"
    report_path = output_dir / "paper_metadata_preflight_report.md"
    checks.to_csv(checks_path, index=False)
    fields.to_csv(fields_path, index=False)
    issues.to_csv(issues_path, index=False)
    split_risk.to_csv(split_risk_path, index=False)
    write_report(checks, fields, issues, split_risk, report_path, metadata_csv, bool(args.strict))

    blocking = checks[checks["q2_blocking"].astype(bool) & checks["status"].eq(FAIL)]
    print(f"Saved metadata preflight checks: {checks_path}")
    print(f"Saved metadata preflight field coverage: {fields_path}")
    print(f"Saved metadata preflight video issues: {issues_path}")
    print(f"Saved metadata preflight split risk: {split_risk_path}")
    print(f"Saved metadata preflight report: {report_path}")
    print(f"Q2 metadata preflight status: {'ready' if blocking.empty else 'not_ready'}")
    if not blocking.empty:
        print(blocking[["category", "check", "status", "affected_rows", "evidence"]].to_string(index=False))
    if args.strict and not blocking.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
