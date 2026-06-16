# Dataset — Cityflo Mobility Analysis

Four data assets underpin the pipeline: legacy GPS location logs, a stops reference file, a trips schedule file, and an Open-Meteo weather grid. This document covers schemas, confirmed field semantics, EDA findings, and the cleaning decisions made before any pipeline script runs.

## GPS Location Data (Legacy Format)

The legacy export covers September 2021 through January 2024, split across four file groups by snapshot date. Each group may have multiple part files representing a single continuous DB export. The current pipeline uses only the `before_2022-10-22` group (three parts, ~17 GB total), which covers September 2021 through October 2022.

Files have no header row. Columns are positional.

| Position | Field | Type | Description |
| --- | --- | --- | --- |
| 1 | `id` | int64 | Unique record ID, sequential |
| 2 | `lat` | float64 | WGS84 latitude |
| 3 | `lng` | float64 | WGS84 longitude |
| 4 | `created` | string | DB ingestion timestamp (UTC) — dropped |
| 5 | `vehicle_id` | int64 | Internal vehicle identifier |
| 6 | `timestamp` | string | GPS event time (UTC) — the temporal reference for all analysis |
| 7 | `speed` | float64 | Speed in km/h; contains sentinel error codes above 120 |
| 8 | `bearing` | float64 | Heading in degrees — dropped (distribution inconsistent with full 0–360 range) |
| 9 | `source` | int32 | GPS hardware type: 1, 2 = OBD device; 4 = standalone tracker |
| 10 | `c10` | string | Always null — dropped |
| 11 | `c11` | string | Undocumented, low cardinality — dropped |
| 12 | `deviation_s` | float64 | Seconds between GPS event time and DB ingestion |
| 13 | `c13` | string | 23.1% null, undocumented — dropped |
| 14 | `c14` | string | Near-constant value — dropped |

`timestamp` (position 6) is the actual GPS fix time and is used for all temporal analysis. `created` (position 4) is when the record was written to the database; the difference between the two is `deviation_s`.

`speed` contains confirmed sentinel values — 602, 699.999, 800, 1000, 1200 — that represent GPS hardware error states. Any value above 120 km/h is treated as a sentinel and nulled before downstream filtering.

There is no `ride_date` column in the legacy format. It is derived from `timestamp` after converting to IST (UTC+5:30).

### EDA-confirmed filter parameters

From `notebooks/02_gps_data_audit.ipynb`, applied to a 500k-row sample:

| Filter | Threshold | Rows removed |
| --- | --- | --- |
| Valid timestamp | non-null | 0.03% |
| Mumbai bbox | lat ∈ [18.8894, 19.3274], lng ∈ [72.7692, 73.1165] | 0.003% |
| Temporal deviation | \|deviation_s\| ≤ 300 | 0.12% |
| GPS jump (inter-ping haversine speed) | ≤ 120 km/h | ~0.04% of transitions |

Overall retention across all filters is approximately 99.1%. The deviation threshold of 300 seconds was chosen conservatively — the p99 of the distribution is well below this value.

The snap rate against `stops_clean.csv` using Haversine nearest-neighbour:

| Threshold | Pings within threshold |
| --- | --- |
| 200 m | 68% |
| 300 m | 82% |
| 500 m | 90% |

The pipeline uses 300 m as the snap threshold (`SNAP_THRESHOLD_M` in `config.py`).

Inter-ping gap distribution (within same vehicle):

| Percentile | Gap |
| --- | --- |
| p90 | ~3 min |
| p95 | ~8 min |
| p99 | ~18 min |

The gap distribution is bimodal — normal within-trip variation falls below ~10 minutes, with inter-trip idle periods clustering above 20 minutes. The segmentation threshold is set at 20 minutes (`GAP_THRESHOLD_MIN`), which captures 99.9% of transitions as within-segment.

## Stops Reference Data

`stops_clean.csv` is the canonical stops file used by all pipeline scripts. It is derived from the raw `stops.csv` through the cleaning steps documented in `notebooks/01_reference_data_audit.ipynb`. Do not use `stops.csv` directly.

| Field | Type | Description |
| --- | --- | --- |
| `stop_id` | int | Unique stop identifier; referenced in `trip_route` |
| `stop_name` | string | Human-readable stop name |
| `lat` | float | WGS84 latitude |
| `lng` | float | WGS84 longitude |
| `stop_category` | string | `Morning` (AM inbound pickup) or `Evening` (PM outbound pickup) |

`stop_category` reflects the primary commute direction a stop serves. Morning stops are pickup points in residential areas for the AM inbound run; Evening stops are pickup points in business districts for the PM outbound run. This field is used to construct separate AM and PM demand surfaces.

Raw `stops.csv` contained 112 stops with invalid coordinates (zero values, transposed lat/lng, or coordinates outside India). Of these, 90 were repaired from a supplemental coordinate file and 9 were corrected manually. The remaining 22 invalid stops were confirmed to not appear in any trip route in `trips_clean.csv` and are retained in the file but excluded from the BallTree snap by the coordinate validity filter.

Total: 6,115 stops. Valid for snapping: 6,093.

## Trips Schedule Data

`trips_clean.csv` is the canonical trips file. It is derived from `trips.csv` through cleaning in `notebooks/01_reference_data_audit.ipynb`.

| Field | Type | Description |
| --- | --- | --- |
| `trip_id` | int | Unique trip instance ID |
| `trip_route` | string | Ordered `(stop_id, HH:MM:SS)` tuples as a Python list-of-tuples string |
| `trip_date` | date | Scheduled date (YYYY-MM-DD) |

`trip_route` is stored as a Python `repr` of a list of tuples, for example:

```md
[(142, '19:30:00'), (152, '19:30:00'), (5301, '19:34:00'), ...]
```

Parse with `ast.literal_eval`. Do not use `json.loads` or regex — the format is a Python literal, not JSON. Scheduled times are in IST.

`trips.csv` has no `vehicle_id` column. There is no direct join key between GPS pings and scheduled trips. Route inference (`05_route_inference.py`) bridges this gap by matching observed GPS stop sequences against route templates.

`trip_date` values range from December 2025 to July 2026 — these are future scheduled dates from when the file was exported, not historical operation dates. The stop sequences and scheduled times remain valid for template extraction. Schedule adherence computed in `07_reliability.py` should therefore be interpreted as deviation from representative operational schedules rather than from historically confirmed timetables.

### Cleaning summary

| Metric | Count |
| --- | --- |
| Raw trips | 185,452 |
| Trips dropped (stop references missing valid coordinates) | 50,327 |
| Trips in `trips_clean.csv` | 135,125 |
| Unique route templates | 641 |
| Unique stop IDs referenced | 3,119 |

## Weather Data

Hourly weather re-analysis from Open-Meteo (ERA5), fetched for a 10 km grid over the Mumbai metropolitan region.

| Property | Value |
| --- | --- |
| Grid points | 15 (G001–G015) |
| Spatial coverage | lat ∈ [18.88, 19.30], lng ∈ [72.77, 73.05] |
| Temporal coverage | 2022-01-01 to 2026-05-14 |
| Temporal resolution | Hourly |
| Grid spacing | ~10 km |

Grid point coordinates are in `data/raw/weather/mumbai_grid_10km_points.csv`. Each grid point has its own subdirectory with CSV files split by variable group and half-month period.

**File naming:** `G001_hourly_core_2022_01_H1.csv` — grid point, variable group, year, month, half (`H1` = days 1–15, `H2` = days 16–end).

**Variable groups:**

- `core` (19 variables): temperature, humidity, dew point, apparent temperature, pressure, precipitation, rain, cloud cover (total/low/mid/high), wind speed/direction (10m/100m), wind gusts, weather code, vapour pressure deficit
- `radiation` (6 variables): shortwave/direct/diffuse radiation, DNI, sunshine duration, ET₀
- `soil` (8 variables): soil temperature and moisture at four depth levels (0–7 cm, 7–28 cm, 28–100 cm, 100–255 cm)
- `daily` (19 variables): daily summaries of the above plus sunrise/sunset times

All timestamps are in IST (Open-Meteo was queried with `timezone=Asia/Kolkata`).

The fetch script (`scripts/weather_data_gen.py`) has built-in resume support via `data/raw/weather/completed_jobs.txt`. If the fetch is interrupted, re-running the script will skip already-completed jobs.
