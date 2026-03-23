# Google Routes OD Data Collection (Mumbai)

This folder contains scripts to collect real-time travel time and distance data between Mumbai ward zones using the Google Routes API.

## Files

- `extract_zones.py` — reads Mumbai ward boundaries (KML) and outputs a centroid CSV with zone area
- `get_data.py` — main collection script, runs continuously and logs travel times every 5 minutes
- `mumbai-wards.kml` — source ward boundary file (from OpenCity)
- `.env.example` — copy this to `.env` and fill in your API key

## Setup

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Copy `.env.example` to `.env` in this folder and fill in your key:

```bash
cp .env.example .env
```

`.env` should look like:

```txt
GOOGLE_MAPS_API_KEY=your_api_key_here
```

Make sure **Routes API** is enabled in your Google Cloud project. If you plan to use road snapping, enable **Roads API** too (it's off by default).

## Usage

### Step 1: Generate zones (run once)

```bash
python scripts/data_collection/extract_zones.py
```

This reads `mumbai-wards.kml` and creates `mumbai_zones.csv` with one row per ward.

### Step 2: Start collecting

```bash
python scripts/data_collection/get_data.py
```

The script runs in a loop — it collects one batch of OD pairs, writes results to CSV, then sleeps until the next 5-minute mark. Stop it anytime with `Ctrl+C`. It's safe to restart — it appends to the existing CSV and won't duplicate the header.

## Output

### `mumbai_zones.csv`

| Column | Description |
| --- | --- |
| `zone_id` | Ward name (e.g. A, B, K/E, R/N) |
| `lat` | Centroid latitude |
| `lng` | Centroid longitude |
| `area_km2` | Ward area in km² (used for intra-zonal estimates) |

### `google_routes_realtime.csv`

One row per route per OD pair per batch.

| Column | Description |
| --- | --- |
| `origin_id`, `dest_id` | Sub-point IDs (e.g. `A_0`, `K/E_2`) |
| `origin_zone`, `dest_zone` | Parent ward names |
| `origin_lat/lng`, `dest_lat/lng` | Coordinates used for the API call |
| `capture_time_utc` | Batch timestamp in UTC |
| `capture_time_ist` | Same timestamp in IST |
| `route_rank` | 1 = fastest, 2/3 = alternatives |
| `route_label` | `DEFAULT_ROUTE`, `DEFAULT_ROUTE_ALTERNATE`, or `INTRA_ZONAL` |
| `distance_m` | Distance in metres |
| `duration_s` | Traffic-aware travel time in seconds |
| `is_imputed` | `True` if estimated via Wardrop formula, `False` if from API |

## What's changed from v1

The original script used a single centroid per zone and sent every OD pair directly to the API. This caused two problems: intra-zonal trips (same origin and destination ward) returned zero distance and time, and all trips looked like they started/ended at the exact same point regardless of where in the ward someone actually was.

The current version fixes both:

- **Sub-points per zone** — each ward centroid is expanded into a few random points scattered within the ward boundary. The scatter radius is derived from the ward area so smaller wards get tighter clusters.
- **Intra-zonal imputation** — same-ward OD pairs are no longer sent to the API. Instead, travel time and distance are estimated using Wardrop's formula (`avg trip length ≈ 0.52 × √area_km²`) at a conservative 20 km/h. These rows are flagged with `is_imputed = True`.
- **Road snapping** — optional, disabled by default. Set `SNAP_TO_ROADS = True` in `get_data.py` if you want sub-points snapped to the nearest road before routing (needs Roads API enabled).
- **OD sampling** — pairs are now picked by shuffling all possible zone combinations and slicing, which is faster and avoids any edge cases with the previous random-loop approach.

## Key settings

All of these are at the top of `get_data.py`:

| Setting | Default | What it controls |
| --- | --- | --- |
| `STEP_MINUTES` | 5 | How often a batch runs |
| `MAX_OD_PAIRS` | 50 | OD pairs per batch (main cost lever) |
| `POINTS_PER_ZONE` | 3 | Sub-points generated per ward |
| `MAX_ROUTES_PER_OD` | 0 | 0 = keep all returned routes (up to ~4) |
| `SNAP_TO_ROADS` | False | Road snapping via Roads API |
| `INTRA_ZONAL_SPEED_KMH` | 20 | Speed used in Wardrop estimates |

With defaults (50 pairs, ~2.5 routes average), each batch makes roughly 125 API calls. Running 16 hours a day gives around 24,000 calls/day which is within the free tier.

## Notes

- Skip overnight hours (roughly 10 PM to 6 AM IST) to save quota
- The script stops automatically on quota or auth errors (HTTP 403/429) to protect the key.
- `RANDOM_SEED = 42` keeps sub-point placement and pair selection consistent across runs.

## Source

Mumbai ward boundaries: <https://data.opencity.in/dataset/mumbai-wards-map/resource/0318c3e8-1530-4bf4-b29b-7281573dee8a>
