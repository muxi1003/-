from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = (
        repo
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build a clearly bounded Lindian County daily NASA POWER weather proxy. "
            "This never fills barn-measured temperature, humidity, or THI fields."
        )
    )
    parser.add_argument(
        "--cached-json",
        type=Path,
        default=assets / "paper_lindian_nasa_power_20230805_20230810_raw.json",
    )
    parser.add_argument(
        "--metadata-csv", type=Path, default=assets / "paper_metadata_template.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=assets)
    return parser.parse_args()


def numeric(values: object) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def compute_thi(temperature_c: pd.Series, relative_humidity_percent: pd.Series) -> pd.Series:
    fahrenheit = 1.8 * numeric(temperature_c) + 32.0
    humidity = numeric(relative_humidity_percent)
    return fahrenheit - ((0.55 - 0.0055 * humidity) * (fahrenheit - 58.0))


def daily_table(payload: dict[str, object]) -> pd.DataFrame:
    data = payload.get("data", payload)
    parameters = data["properties"]["parameter"]
    required = ["T2M", "RH2M", "T2M_MAX", "T2M_MIN"]
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"NASA POWER cache missing parameters: {missing}")
    dates = sorted(parameters["T2M"])
    rows = []
    for date in dates:
        row = {
            "collection_date": pd.to_datetime(date, format="%Y%m%d").date().isoformat(),
            "weather_proxy_t2m_c": parameters["T2M"].get(date),
            "weather_proxy_rh2m_percent": parameters["RH2M"].get(date),
            "weather_proxy_t2m_max_c": parameters["T2M_MAX"].get(date),
            "weather_proxy_t2m_min_c": parameters["T2M_MIN"].get(date),
            "weather_proxy_source": "NASA_POWER_MERRA2_daily_county_midpoint",
            "weather_proxy_temporal_resolution": "daily_local_solar_time",
            "weather_proxy_spatial_boundary": "Lindian_County_extent_midpoint_not_ranch_GPS",
        }
        rows.append(row)
    output = pd.DataFrame(rows)
    output["weather_proxy_thi"] = compute_thi(
        output["weather_proxy_t2m_c"], output["weather_proxy_rh2m_percent"]
    )
    return output


def video_assignment(metadata: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    required = {"video_id", "collection_date", "collection_start_date", "collection_end_date"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata template missing columns: {sorted(missing)}")
    table = metadata[
        ["video_id", "collection_date", "collection_start_date", "collection_end_date"]
    ].copy()
    collection_date = pd.to_datetime(table["collection_date"], errors="coerce").dt.date
    table["collection_date"] = collection_date.astype("string")
    daily_for_merge = daily.drop(columns=["weather_proxy_source", "weather_proxy_temporal_resolution", "weather_proxy_spatial_boundary"])
    table = table.merge(daily_for_merge, on="collection_date", how="left", validate="many_to_one")
    assigned = table["weather_proxy_t2m_c"].notna()
    table["weather_proxy_assignment_status"] = np.where(
        assigned,
        "matched_per_video_collection_date",
        "unassigned_collection_date_missing_or_outside_cached_window",
    )
    table["weather_proxy_use_boundary"] = (
        "Descriptive county-day proxy only; not barn sensor data and not eligible for RR-THI inference."
    )
    return table


def write_report(path: Path, payload: dict[str, object], daily: pd.DataFrame, video: pd.DataFrame) -> None:
    location = payload.get("proxy_location", {})
    properties = payload.get("data", payload)["properties"]
    assigned = int(video["weather_proxy_t2m_c"].notna().sum())
    lines = [
        "# Lindian Collection-Window NASA POWER Weather Proxy",
        "",
        "Status: `descriptive_county_day_proxy_not_barn_measurement`",
        "",
        f"Source: `{payload.get('source_name', 'NASA POWER Daily API')}`.",
        f"Cached request: `{payload.get('source_url', '')}`.",
        f"Proxy point: latitude `{location.get('latitude')}`, longitude `{location.get('longitude')}`.",
        f"Location boundary: {location.get('description', '')}",
        f"Source time standard: `{properties['header'].get('time_standard', 'unknown')}`.",
        "",
        "## Daily Collection-Window Conditions",
        "",
        "| date | T2M C | RH2M % | T2M min C | T2M max C | proxy THI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in daily.itertuples(index=False):
        lines.append(
            f"| {row.collection_date} | {row.weather_proxy_t2m_c:.2f} | "
            f"{row.weather_proxy_rh2m_percent:.2f} | {row.weather_proxy_t2m_min_c:.2f} | "
            f"{row.weather_proxy_t2m_max_c:.2f} | {row.weather_proxy_thi:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Assignment Boundary",
            "",
            f"The metadata table has exact per-video collection dates for `{assigned}/{len(video)}` videos.",
            "The current 73-video metadata records only the 2023-08-05 to 2023-08-10 "
            "collection window, so no proxy day is assigned to individual videos. Do not copy "
            "these values into `ambient_temperature_c`, `relative_humidity_percent`, or `thi`; "
            "do not perform RR-THI association, THI strata, or heat-stress claims from this proxy.",
            "",
            "The proxy THI uses the existing project formula: (1.8*T_C+32) - "
            "(0.55-0.0055*RH)*(1.8*T_C+32-58).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.cached_json.read_text(encoding="utf-8"))
    metadata = pd.read_csv(args.metadata_csv, dtype=str, keep_default_na=False)
    daily = daily_table(payload)
    video = video_assignment(metadata, daily)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "paper_lindian_nasa_power_daily_weather_proxy.csv"
    video_path = output_dir / "paper_lindian_nasa_power_video_weather_proxy.csv"
    report_path = output_dir / "paper_lindian_nasa_power_weather_proxy.md"
    daily.to_csv(daily_path, index=False)
    video.to_csv(video_path, index=False)
    write_report(report_path, payload, daily, video)
    print(f"Saved daily weather proxy: {daily_path}")
    print(f"Saved video proxy assignment: {video_path}")
    print(f"Saved proxy report: {report_path}")
    print(f"per_video_dates_assigned={int(video['weather_proxy_t2m_c'].notna().sum())}/{len(video)}")


if __name__ == "__main__":
    main()
