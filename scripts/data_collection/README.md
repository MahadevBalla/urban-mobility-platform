# Google Routes OD Data Collection (Mumbai)

This folder contains scripts to:
1. Extract zone centroids from Mumbai ward polygons
2. Collect travel time and distance data between zones using Google Routes API

---

## Files

- `extract_zones.py`  
  Converts Mumbai ward KML → centroid CSV

- `get_data.py`  
  Collects travel time + distance between OD pairs using Google Routes API

- `mumbai-wards.kml`  
  Source ward boundaries (from OpenCity dataset)

---

## Output

### 1. Zone file
`mumbai_zones.csv`

Columns:
- `zone_id`
- `lat`
- `lng`

---

### 2. Routes data
`google_routes_5min_sample.csv`

Columns:
- `origin_id`, `dest_id`
- `origin_lat`, `origin_lng`
- `dest_lat`, `dest_lng`
- `capture_time_utc`, `capture_time_ist`
- `route_rank`
- `route_label`
- `distance_m`
- `duration_s`

---

## Setup

### 1. Install dependencies from the project root

```bash
pip install -r requirements.txt
```

### 2. Set API key

Create a .env file in this directory (refer [.env.example](./.env.example)):

```bash
GOOGLE_MAPS_API_KEY=your_api_key_here
```

Make sure **Routes API is enabled** in your Google Cloud project.

## Usage

### Step 1: Generate zones

```bash
python scripts/data_collection/extract_zones.py
```

This creates: `mumbai_zones.csv`

### Step 2: Run data collection

```bash
python scripts/data_collection/get_data.py
```

- Runs continuously
- Collects data every 5 minutes
- Appends results to CSV

Stop manually using: `Ctrl + C`

## Notes

- The script uses zone centroids as origin/destination points
- Each OD pair may return multiple routes (if available)
- Data is collected using live traffic conditions

## Limitations (current version)

- Uses only one point per zone (centroid)
- Intra-zonal trips (same origin and destination) may return zero distance/time
- No road snapping (points may not lie exactly on roads)

## Data control

To reduce API usage:

- Lower `MAX_OD_PAIRS`
- Increase `STEP_MINUTES`

## Source

Mumbai ward boundaries: <https://data.opencity.in/dataset/mumbai-wards-map/resource/0318c3e8-1530-4bf4-b29b-7281573dee8a>

