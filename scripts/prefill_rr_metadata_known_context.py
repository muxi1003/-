from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.request

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

ENVIRONMENT_COLUMNS = [
    "ambient_temperature_c",
    "relative_humidity_percent",
    "thi",
    "athi",
]

DEFAULT_LATITUDE = 47.18
DEFAULT_LONGITUDE = 124.87
DEFAULT_START = "20230805"
DEFAULT_END = "20230810"
DEFAULT_COUNTRY = "China"
DEFAULT_PROVINCE = "Heilongjiang"
DEFAULT_COUNTY = "Lindian County"
DEFAULT_SITE = "Lindian County ranch"
DEFAULT_SCENE_ID = "CN_HLJ_Lindian_ranch_2023-08-05_to_2023-08-10"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Prefill defensible RR metadata from known collection context without "
            "inventing unavailable per-video labels."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--start-date", default=DEFAULT_START, help="YYYYMMDD")
    parser.add_argument("--end-date", default=DEFAULT_END, help="YYYYMMDD")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--write-fill-template", action="store_true")
    parser.add_argument("--write-metadata-template", action="store_true")
    parser.add_argument(
        "--fill-weather-proxy-into-metadata",
        action="store_true",
        help=(
            "Write NASA POWER regional proxy means into the metadata environment "
            "columns. Default is off because these are not synchronized barn-sensor "
            "measurements and must not unlock THI-stratified manuscript claims."
        ),
    )
    parser.add_argument(
        "--no-weather-fetch",
        action="store_true",
        help="Skip NASA POWER fetch and only fill non-weather context fields.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def default_metadata_csv(input_root: Path, corrected_prefix: str) -> Path:
    assets = input_root / f"{corrected_prefix}_paper_assets"
    fill_template = assets / "paper_metadata_annotation_fill_template.csv"
    metadata_template = assets / "paper_metadata_template.csv"
    return fill_template if fill_template.exists() else metadata_template


def read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "video_id" not in data.columns:
        raise ValueError(f"Metadata CSV is missing video_id: {path}")
    for column in METADATA_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data[METADATA_COLUMNS].fillna("").map(normalize_text)


def cow_id_from_video_id(video_id: str) -> str:
    digits = re.sub(r"[^0-9]+", "", str(video_id))
    return digits or str(video_id)


def date_arg_to_iso(value: str) -> str:
    text = normalize_text(value)
    if re.match(r"^\d{8}$", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def nasa_power_url(latitude: float, longitude: float, start: str, end: str) -> str:
    return (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M&community=AG&longitude={longitude:.4f}"
        f"&latitude={latitude:.4f}&start={start}&end={end}&format=JSON"
    )


def cattle_thi(temp_c: float, rh_percent: float) -> float:
    temp_f = 1.8 * temp_c + 32.0
    return temp_f - (0.55 - 0.0055 * rh_percent) * (1.8 * temp_c - 26.0)


def fetch_weather_proxy(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float | str]]:
    url = nasa_power_url(args.latitude, args.longitude, args.start_date, args.end_date)
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    params = payload["properties"]["parameter"]
    rows = []
    for date_key in sorted(params["T2M"]):
        temp = float(params["T2M"][date_key])
        rh = float(params["RH2M"][date_key])
        rows.append(
            {
                "date": f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}",
                "ambient_temperature_c_proxy": temp,
                "relative_humidity_percent_proxy": rh,
                "thi_proxy": cattle_thi(temp, rh),
                "source": "NASA POWER AG daily T2M/RH2M regional proxy",
                "source_url": url,
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "ambient_temperature_c_proxy": float(daily["ambient_temperature_c_proxy"].mean()),
        "relative_humidity_percent_proxy": float(daily["relative_humidity_percent_proxy"].mean()),
        "thi_proxy": float(daily["thi_proxy"].mean()),
        "source_url": url,
        "weather_days": int(len(daily)),
    }
    return daily, summary


def append_note(existing: str, note: str) -> str:
    existing = normalize_text(existing)
    if note in existing:
        return existing
    if not existing:
        return note
    return f"{existing} | {note}"


def clean_previous_prefill_notes(existing: str) -> str:
    text = normalize_text(existing)
    if not text:
        return ""
    parts = [part.strip() for part in text.split("|")]
    keep = []
    for part in parts:
        if not part:
            continue
        lower = part.lower()
        if "known context prefill:" in lower:
            continue
        if "nasa_power_source=" in lower or "nasa power" in lower:
            continue
        keep.append(part)
    return " | ".join(keep)


def clear_previous_proxy_environment(filled: pd.DataFrame) -> pd.DataFrame:
    notes = filled["notes"].astype(str)
    proxy_mask = notes.str.contains(
        r"NASA_POWER_source=|NASA POWER|regional daily proxy|regional proxy",
        case=False,
        na=False,
        regex=True,
    )
    if proxy_mask.any():
        for column in ENVIRONMENT_COLUMNS:
            filled.loc[proxy_mask, column] = ""
    return filled


def prefill_metadata(
    metadata: pd.DataFrame,
    args: argparse.Namespace,
    weather_summary: dict[str, float | str] | None,
) -> pd.DataFrame:
    filled = metadata.copy()
    filled["cow_id"] = filled["video_id"].map(cow_id_from_video_id)
    filled["collection_start_date"] = date_arg_to_iso(args.start_date)
    filled["collection_end_date"] = date_arg_to_iso(args.end_date)
    filled["collection_location_country"] = DEFAULT_COUNTRY
    filled["collection_location_province"] = DEFAULT_PROVINCE
    filled["collection_location_county"] = DEFAULT_COUNTY
    filled["collection_site"] = DEFAULT_SITE
    filled["scene_id"] = args.scene_id
    filled["external_test_split"] = "internal"

    if not args.fill_weather_proxy_into_metadata:
        filled = clear_previous_proxy_environment(filled)
    filled["notes"] = filled["notes"].map(clean_previous_prefill_notes)

    note_parts = [
        "Known context prefill: current 73 videos treated as internal/development data, not true external validation.",
        "Collection context from user: Lindian County ranch, Heilongjiang, China, 2023-08-05 to 2023-08-10.",
        "cow_id derived from numeric video label.",
        "Exact per-video collection_date/time, camera_id, barn environment measurements, and manual quality scores remain unavailable.",
    ]
    if weather_summary is not None and args.fill_weather_proxy_into_metadata:
        temp = float(weather_summary["ambient_temperature_c_proxy"])
        rh = float(weather_summary["relative_humidity_percent_proxy"])
        thi = float(weather_summary["thi_proxy"])
        filled["ambient_temperature_c"] = f"{temp:.2f}"
        filled["relative_humidity_percent"] = f"{rh:.2f}"
        filled["thi"] = f"{thi:.2f}"
        note_parts.append(
            "Environment fields are NASA POWER regional daily proxy means, not barn sensor measurements."
        )
        note_parts.append(f"NASA_POWER_source={weather_summary['source_url']}")
    elif weather_summary is not None:
        note_parts.append(
            "Regional NASA POWER weather proxy was generated as a separate context table only; environment metadata fields were left blank because synchronized barn measurements are unavailable."
        )
    note = " ".join(note_parts)
    filled["notes"] = filled["notes"].map(lambda value: append_note(value, note))
    return filled[METADATA_COLUMNS]


def write_report(
    output_path: Path,
    metadata_csv: Path,
    output_csv: Path,
    weather_daily: pd.DataFrame | None,
    weather_summary: dict[str, float | str] | None,
    filled: pd.DataFrame,
) -> None:
    cow_groups = int(filled["cow_id"].map(normalize_text).nunique())
    lines = [
        "# RR Metadata Known-Context Prefill",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Input metadata CSV: `{metadata_csv}`",
        f"Output metadata CSV: `{output_csv}`",
        f"Rows: `{len(filled)}`",
        f"Unique cow_id values after numeric extraction: `{cow_groups}`",
        "",
        "## What Was Filled",
        "",
        "- `cow_id`: numeric part of `video_id`, per user clarification.",
        "- `collection_start_date` and `collection_end_date`: known collection window from user-provided provenance.",
        "- `collection_location_country`, `collection_location_province`, `collection_location_county`, and `collection_site`: known Lindian County ranch collection location.",
        "- `scene_id`: Lindian County ranch collection context.",
        "- `external_test_split`: `internal` for all current videos because the existing 73-video dataset is not an independent external validation set.",
        "- `notes`: Lindian County, Heilongjiang, China collection window of 2023-08-05 to 2023-08-10, provenance, and remaining unavailable fields.",
    ]
    if weather_summary is None:
        lines.append("- Environment fields were not filled because weather fetching was disabled.")
    else:
        lines.extend(
            [
                "- Weather proxy: NASA POWER daily regional proxy mean across the collection window was written as context only.",
                "- `ambient_temperature_c`, `relative_humidity_percent`, `thi`, `athi`: left blank unless `--fill-weather-proxy-into-metadata` is explicitly used.",
                "",
                "## Weather Proxy Summary",
                "",
                f"- Days: `{weather_summary['weather_days']}`",
                f"- Mean temperature: `{float(weather_summary['ambient_temperature_c_proxy']):.2f} C`",
                f"- Mean relative humidity: `{float(weather_summary['relative_humidity_percent_proxy']):.2f}%`",
                f"- Mean cattle THI proxy: `{float(weather_summary['thi_proxy']):.2f}`",
                f"- Source URL: `{weather_summary['source_url']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Not Filled",
            "",
            "- `collection_date`: exact per-video date is still unavailable; the known collection window is recorded in notes.",
            "- `collection_time`: unavailable.",
            "- `camera_id`: unavailable.",
            "- `ambient_temperature_c`, `relative_humidity_percent`, `thi`, `athi`: actual synchronized barn measurements are unavailable; regional weather proxy is context only.",
            "- `head_motion_score_0_3`, `occlusion_score_0_3`, `nostril_visibility_score_0_3`: unavailable manual labels; do not fabricate these.",
            "",
            "## Claim Boundary",
            "",
            "This prefill improves provenance completeness but does not create a true external validation set, synchronized heat-stress metadata, or manual quality-stratified claims.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    assets = input_root / f"{args.corrected_prefix}_paper_assets"
    assets.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else default_metadata_csv(input_root, args.corrected_prefix).resolve()
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else assets / "paper_metadata_annotation_context_prefill.csv"
    )

    metadata = read_metadata(metadata_csv)
    weather_daily = None
    weather_summary = None
    if not args.no_weather_fetch:
        weather_daily, weather_summary = fetch_weather_proxy(args)
        weather_daily.to_csv(assets / "paper_metadata_context_weather_proxy.csv", index=False)

    filled = prefill_metadata(metadata, args, weather_summary)
    filled.to_csv(output_csv, index=False)
    if args.write_fill_template:
        filled.to_csv(assets / "paper_metadata_annotation_fill_template.csv", index=False)
    if args.write_metadata_template:
        filled.to_csv(assets / "paper_metadata_template.csv", index=False)

    report_path = assets / "paper_metadata_annotation_context_prefill_report.md"
    write_report(report_path, metadata_csv, output_csv, weather_daily, weather_summary, filled)

    print(f"Saved context-prefilled metadata: {output_csv}")
    if weather_daily is not None:
        print(f"Saved weather proxy: {assets / 'paper_metadata_context_weather_proxy.csv'}")
    print(f"Saved context prefill report: {report_path}")
    print(f"write_fill_template={bool(args.write_fill_template)}")
    print(f"write_metadata_template={bool(args.write_metadata_template)}")


if __name__ == "__main__":
    main()
