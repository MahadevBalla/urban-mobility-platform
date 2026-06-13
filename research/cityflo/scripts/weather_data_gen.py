import math
import json
import time
from pathlib import Path

import pandas as pd
import requests

# =========================================================
# USER SETTINGS
# =========================================================
start_date = "2022-01-01"
end_date = "2026-05-14"
grid_spacing_km = 10  # start with 10 km; later try 5 km
timezone = "Asia/Kolkata"
sleep_between_calls = 8  # seconds between successful calls
max_retries = 8  # retries for 429 / transient errors

# Mumbai urban + suburban bounding box
lat_min, lat_max = 18.88, 19.30
lon_min, lon_max = 72.77, 73.05

output_dir = Path(r"E:\DATA\WeatherData\mumbai_openmeteo_10km_grid_data")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# VARIABLE GROUPS
# =========================================================
hourly_groups = {
    "core": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "pressure_msl",
        "surface_pressure",
        "precipitation",
        "rain",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "weather_code",
        "vapour_pressure_deficit",
    ],
    "radiation": [
        "shortwave_radiation",
        "direct_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "sunshine_duration",
        "et0_fao_evapotranspiration",
    ],
    "soil": [
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
    ],
}

daily_vars = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "sunrise",
    "sunset",
    "sunshine_duration",
    "daylight_duration",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


# =========================================================
# GRID GENERATION
# =========================================================
def km_to_lat_deg(km):
    return km / 111.32


def km_to_lon_deg(km, latitude):
    return km / (111.32 * math.cos(math.radians(latitude)))


def generate_grid(lat_min, lat_max, lon_min, lon_max, spacing_km):
    points = []
    lat_step = km_to_lat_deg(spacing_km)
    lat = lat_min
    point_id = 1

    while lat <= lat_max + 1e-9:
        lon_step = km_to_lon_deg(spacing_km, lat)
        lon = lon_min
        while lon <= lon_max + 1e-9:
            points.append(
                {
                    "grid_id": f"G{point_id:03d}",
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                }
            )
            point_id += 1
            lon += lon_step
        lat += lat_step

    return pd.DataFrame(points)


grid_df = generate_grid(lat_min, lat_max, lon_min, lon_max, grid_spacing_km)
grid_path = output_dir / f"mumbai_grid_{grid_spacing_km}km_points.csv"
grid_df.to_csv(grid_path, index=False)


# =========================================================
# HALF-MONTH WINDOWS
# =========================================================
def generate_half_month_windows(start_date_str, end_date_str):
    start_ts = pd.to_datetime(start_date_str)
    end_ts = pd.to_datetime(end_date_str)

    month_starts = pd.date_range(
        start=pd.Timestamp(start_ts.year, start_ts.month, 1), end=end_ts, freq="MS"
    )

    windows = []

    for month_start in month_starts:
        year = month_start.year
        month = month_start.month
        days_in_month = month_start.days_in_month

        half1_start = pd.Timestamp(year, month, 1).date()
        half1_end = pd.Timestamp(year, month, min(15, days_in_month)).date()

        half2_start = (
            pd.Timestamp(year, month, 16).date() if days_in_month >= 16 else None
        )
        half2_end = (
            pd.Timestamp(year, month, days_in_month).date()
            if days_in_month >= 16
            else None
        )

        candidate_windows = [
            ("H1", half1_start, half1_end),
        ]

        if half2_start is not None and half2_end is not None:
            candidate_windows.append(("H2", half2_start, half2_end))

        for half_tag, win_start, win_end in candidate_windows:
            actual_start = max(win_start, start_ts.date())
            actual_end = min(win_end, end_ts.date())

            if actual_start <= actual_end:
                windows.append(
                    {
                        "period_tag": f"{month_start.strftime('%Y_%m')}_{half_tag}",
                        "month_tag": month_start.strftime("%Y_%m"),
                        "half_tag": half_tag,
                        "start_date": actual_start.strftime("%Y-%m-%d"),
                        "end_date": actual_end.strftime("%Y-%m-%d"),
                    }
                )

    return windows


time_windows = generate_half_month_windows(start_date, end_date)

# =========================================================
# REQUEST WITH RETRY / BACKOFF
# =========================================================
session = requests.Session()


def fetch_openmeteo_one_location(
    lat, lon, req_start_date, req_end_date, hourly_vars=None, daily_vars=None
):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": req_start_date,
        "end_date": req_end_date,
        "timezone": timezone,
    }

    if hourly_vars:
        params["hourly"] = ",".join(hourly_vars)
    if daily_vars:
        params["daily"] = ",".join(daily_vars)

    wait = 65

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=300)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                print(
                    f"429 hit. Waiting {wait} sec before retry {attempt}/{max_retries} ..."
                )
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue

            if 500 <= response.status_code < 600:
                print(
                    f"Server error {response.status_code}. Waiting {wait} sec before retry {attempt}/{max_retries} ..."
                )
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue

            print("FAILED URL:", response.url)
            print("RESPONSE:", response.text)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(
                f"Request error: {e}. Waiting {wait} sec before retry {attempt}/{max_retries} ..."
            )
            time.sleep(wait)
            wait = min(wait * 2, 300)

    raise RuntimeError(f"Failed after {max_retries} retries.")


# =========================================================
# RESUME SUPPORT
# =========================================================
progress_file = output_dir / "completed_jobs.txt"
if progress_file.exists():
    completed_jobs = set(progress_file.read_text(encoding="utf-8").splitlines())
else:
    completed_jobs = set()


def mark_job_complete(job_key):
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(job_key + "\n")
    completed_jobs.add(job_key)


# =========================================================
# METADATA SUPPORT
# =========================================================
metadata_file = output_dir / "metadata_summary.csv"
if metadata_file.exists():
    metadata_df_existing = pd.read_csv(metadata_file)
    metadata_rows = metadata_df_existing.to_dict("records")
else:
    metadata_rows = []

metadata_seen = {
    (
        row.get("grid_id"),
        row.get("period_tag"),
        row.get("requested_latitude"),
        row.get("requested_longitude"),
    )
    for row in metadata_rows
}

# =========================================================
# LOOP OVER GRID POINTS AND HALF-MONTH PERIODS
# =========================================================
for idx, row in grid_df.iterrows():
    grid_id = row["grid_id"]
    lat = row["latitude"]
    lon = row["longitude"]

    print(f"\nProcessing {grid_id} ({idx + 1}/{len(grid_df)}) at {lat}, {lon}")

    point_dir = output_dir / grid_id
    point_dir.mkdir(parents=True, exist_ok=True)

    for win in time_windows:
        period_tag = win["period_tag"]  # e.g. 2022_01_H1
        month_tag = win["month_tag"]  # e.g. 2022_01
        half_tag = win["half_tag"]  # H1 / H2
        win_start = win["start_date"]
        win_end = win["end_date"]

        print(f"  Period: {period_tag} ({win_start} to {win_end})")

        # -------------------------------------------------
        # DAILY
        # -------------------------------------------------
        daily_job = f"{grid_id}|daily|{period_tag}"

        if daily_job in completed_jobs:
            print(f"    Skipping daily {period_tag} (already completed)")
        else:
            try:
                daily_data = fetch_openmeteo_one_location(
                    lat=lat,
                    lon=lon,
                    req_start_date=win_start,
                    req_end_date=win_end,
                    daily_vars=daily_vars,
                )

                with open(
                    point_dir / f"{grid_id}_daily_{period_tag}_raw.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(daily_data, f, indent=2)

                if "daily" in daily_data:
                    df_daily = pd.DataFrame(daily_data["daily"])
                    df_daily["grid_id"] = grid_id
                    df_daily["requested_latitude"] = lat
                    df_daily["requested_longitude"] = lon
                    df_daily["month_tag"] = month_tag
                    df_daily["half_tag"] = half_tag
                    df_daily["period_tag"] = period_tag
                    df_daily.to_csv(
                        point_dir / f"{grid_id}_daily_{period_tag}.csv", index=False
                    )

                meta_key = (grid_id, period_tag, lat, lon)
                if meta_key not in metadata_seen:
                    metadata_rows.append(
                        {
                            "grid_id": grid_id,
                            "period_tag": period_tag,
                            "month_tag": month_tag,
                            "half_tag": half_tag,
                            "requested_latitude": lat,
                            "requested_longitude": lon,
                            "api_latitude": daily_data.get("latitude"),
                            "api_longitude": daily_data.get("longitude"),
                            "elevation": daily_data.get("elevation"),
                            "timezone": daily_data.get("timezone"),
                            "timezone_abbreviation": daily_data.get(
                                "timezone_abbreviation"
                            ),
                            "utc_offset_seconds": daily_data.get("utc_offset_seconds"),
                        }
                    )
                    metadata_seen.add(meta_key)

                mark_job_complete(daily_job)
                print(f"    Saved daily {period_tag}")
                time.sleep(sleep_between_calls)

            except Exception as e:
                print(f"    Daily failed for {grid_id}, {period_tag}: {e}")
                continue

        # -------------------------------------------------
        # HOURLY GROUPS
        # -------------------------------------------------
        for group_name, vars_list in hourly_groups.items():
            hourly_job = f"{grid_id}|hourly|{group_name}|{period_tag}"

            if hourly_job in completed_jobs:
                print(
                    f"    Skipping hourly {group_name} {period_tag} (already completed)"
                )
                continue

            try:
                hourly_data = fetch_openmeteo_one_location(
                    lat=lat,
                    lon=lon,
                    req_start_date=win_start,
                    req_end_date=win_end,
                    hourly_vars=vars_list,
                )

                with open(
                    point_dir / f"{grid_id}_hourly_{group_name}_{period_tag}_raw.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(hourly_data, f, indent=2)

                if "hourly" in hourly_data:
                    df_hourly = pd.DataFrame(hourly_data["hourly"])
                    df_hourly["grid_id"] = grid_id
                    df_hourly["requested_latitude"] = lat
                    df_hourly["requested_longitude"] = lon
                    df_hourly["month_tag"] = month_tag
                    df_hourly["half_tag"] = half_tag
                    df_hourly["period_tag"] = period_tag
                    df_hourly.to_csv(
                        point_dir / f"{grid_id}_hourly_{group_name}_{period_tag}.csv",
                        index=False,
                    )

                mark_job_complete(hourly_job)
                print(f"    Saved hourly {group_name} {period_tag}")
                time.sleep(sleep_between_calls)

            except Exception as e:
                print(
                    f"    Hourly group {group_name} failed for {grid_id}, {period_tag}: {e}"
                )
                break

# =========================================================
# SAVE METADATA SUMMARY
# =========================================================
if metadata_rows:
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates()
    metadata_df.to_csv(metadata_file, index=False)

print("\nFinished.")
