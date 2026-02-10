# Module-by-Module Guide

This guide provides detailed documentation for each Python module in the system.

## 1. Data Ingestion (`src/data_ingestion/`)

### `telecom_loader.py`

**Class: `TelecomDataLoader`**

Handles loading of raw telecom data from CSV files.

```python
class TelecomDataLoader:
    def __init__(self, config=None)
    def load_cdr(self, path: str) -> pd.DataFrame
    def load_xdr(self, path: str) -> pd.DataFrame
    def load_4g(self, path: str) -> pd.DataFrame
    def load_5g(self, path: str) -> pd.DataFrame
    def load_all(self, directory: str) -> Dict[str, pd.DataFrame]
```

**Input File Formats:**

| Data Type | Required Columns | Optional Columns |
| --- | --- | --- |
| CDR | imsi, timestamp, cell_id | call_type, duration, tac |
| XDR | imsi, timestamp, cell_id | latitude, longitude, bytes, tac |
| 4G | imsi, timestamp, tac | enodeb_id, rsrp, rsrq |
| 5G | imsi, timestamp, tac | gnodeb_id, ss_rsrp |

**Example:**

```python
loader = TelecomDataLoader()
cdr = loader.load_cdr("data/cdr_data.csv")
xdr = loader.load_xdr("data/xdr_data.csv")

# Or load all at once
all_data = loader.load_all("data/")
# Returns: {'cdr': df, 'xdr': df, '4g': df, '5g': df}
```

### `cell_tower_loader.py`

**Class: `CellTowerLoader`**

Manages cell tower location data.

```python
class CellTowerLoader:
    def __init__(self, config=None)
    def load_towers(self, path: str) -> pd.DataFrame
    def infer_from_xdr(self, xdr_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]
    def add_locations(self, df: pd.DataFrame, locations: Dict) -> pd.DataFrame
```

**Tower Inference Algorithm:**
When explicit tower locations are unavailable, infer from XDR GPS data:

1. Group XDR records by `cell_id`
2. For each cell:
   - Calculate median latitude and longitude
   - Remove outliers (> 3 standard deviations)
   - Store as cell centroid
3. Also aggregate by TAC for zone-level locations

**Example:**

```python
loader = CellTowerLoader()

# If you have tower database
towers = loader.load_towers("data/towers.csv")

# If not, infer from XDR
cell_locations = loader.infer_from_xdr(xdr_df)
tac_locations = loader.infer_tac_locations(xdr_df)

# Add coordinates to records
enriched_df = loader.add_locations(telecom_df, cell_locations)
```

### `zone_loader.py`

**Class: `ZoneLoader`**

Defines spatial zones for OD aggregation.

```python
class ZoneLoader:
    def __init__(self, config=None)
    def create_tac_zones(self, df: pd.DataFrame) -> gpd.GeoDataFrame
    def create_grid_zones(self, bounds, cell_size: float) -> gpd.GeoDataFrame
    def load_zones(self, path: str) -> gpd.GeoDataFrame
    def assign_zones(self, df: pd.DataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame
    def get_zone_populations(self) -> Dict[str, int]
    def get_all_zone_ids(self) -> List[str]
```

**Zone Types:**

- **TAC Zones**: Based on Tracking Area Codes (coarse, network-defined)
- **Grid Zones**: Regular square grid (customizable resolution)
- **Custom Zones**: Load from GeoJSON/Shapefile (administrative boundaries)

**Example:**

```python
loader = ZoneLoader()

# TAC-based zones
zones = loader.create_tac_zones(telecom_df)

# Grid-based zones (500m cells)
zones = loader.create_grid_zones(
    bounds=(min_lat, min_lon, max_lat, max_lon),
    cell_size=500
)

# Custom zones
zones = loader.load_zones("data/wards.geojson")

# Assign records to zones
df = loader.assign_zones(telecom_df, zones)
```

## 2. Preprocessing (`src/preprocessing/`)

### `telecom_preprocessor.py`

**Class: `TelecomPreprocessor`**

Cleans and standardizes telecom data.

```python
class TelecomPreprocessor:
    def __init__(self, config=None)
    def process(self, cdr_df=None, xdr_df=None, ...) -> pd.DataFrame
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame
    def get_user_summary(self, df: pd.DataFrame) -> pd.DataFrame
    def filter_ping_pong(self, df, time_threshold_s=300) -> pd.DataFrame
    def remove_ping_pong(self, df) -> pd.DataFrame
```

**Processing Pipeline:**

1. **Process CDR**: Standardize columns, parse timestamps
2. **Process XDR**: Standardize columns, validate coordinates
3. **Merge Sources**: Combine all data with unified schema
4. **Remove Duplicates**: Drop exact duplicate records
5. **Temporal Filter**: Keep records in study period
6. **Spatial Filter**: Keep records in study area (optional)

**Ping-Pong Filtering:**

```md
What is ping-pong?
──────────────────
When a stationary user's phone oscillates between cell towers:
  Cell A → Cell B → Cell A → Cell B (within seconds)

This creates false movement. We detect and remove it.

Detection:
- Pattern: A-B-A-B-A (alternating cells)
- Rapid: < 5 minutes between switches
- Repeated: > 3 oscillations

Result: Middle observations marked as ping-pong
```

**Example:**

```python
preprocessor = TelecomPreprocessor()

# Full pipeline
clean_df = preprocessor.process(
    cdr_df=cdr_data,
    xdr_df=xdr_data,
    network_4g_df=lte_data
)

# Add time period, day type, etc.
clean_df = preprocessor.add_derived_features(clean_df)

# Remove ping-pong
clean_df = preprocessor.remove_ping_pong(clean_df)

# Get per-user statistics
user_stats = preprocessor.get_user_summary(clean_df)
# Columns: imsi, first_seen, last_seen, record_count,
#          unique_cells, active_days, avg_daily_records
```

### `user_filter.py`

**Class: `UserFilter`**

Filters users based on data quality criteria.

```python
class UserFilter:
    def __init__(self, config=None)
    def filter_users(self, df: pd.DataFrame) -> pd.DataFrame
    def get_filter_stats(self, df: pd.DataFrame) -> Dict
```

**Filter Criteria:**

| Criterion | Default | Purpose |
| --- | --- | --- |
| min_records | 10 | Enough data for analysis |
| min_days | 3 | Observation period |
| min_daily_records | 2 | Active phone usage |
| max_daily_records | 1000 | Filter bots/M2M devices |

**Example:**

```python
user_filter = UserFilter()

# Filter users
filtered_df = user_filter.filter_users(telecom_df)

# Check statistics
stats = user_filter.get_filter_stats(telecom_df)
print(f"Kept {stats['valid_users']} of {stats['total_users']} users")
```

## 3. Stay Detection (`src/stay_detection/`)

### `stay_detector.py`

**Class: `StayPointDetector`**

Detects meaningful locations from trajectory data.

```python
class StayPointDetector:
    def __init__(self, config=None, distance_threshold=500,
                 time_threshold=1800, grid_cell_size=300, min_visits=2)
    def detect(self, df: pd.DataFrame) -> pd.DataFrame
```

**Algorithm Phases:**

```md
PHASE 1: Candidate Extraction (Zheng-Xie Algorithm)
───────────────────────────────────────────────────
Input: Time-ordered observations for one user

anchor = first_point
cluster = [anchor]

for each subsequent point:
    if distance(anchor, point) <= threshold:
        cluster.append(point)
    else:
        if time_span(cluster) >= time_threshold:
            create_candidate_stay(cluster)
        anchor = point
        cluster = [anchor]

PHASE 2: Progressive Threshold Relaxation
─────────────────────────────────────────
If no candidates found:
1. Try 1.5x distance threshold
2. Try 0.5x time threshold
3. Try both relaxed
4. Fall back to cell-based grouping

PHASE 3: Grid Consolidation
───────────────────────────
Multiple candidates at same location (different days)
are merged using grid-based clustering.

Grid cell = floor(lat/cell_size), floor(lon/cell_size)
Merge candidates in same grid cell.

PHASE 4: Metrics Calculation
────────────────────────────
For each stay:
- visit_count: Distinct visits (separated by time)
- total_duration: Sum of all visit durations
- observation_count: Total observations
- location_confidence: Signal-quality weighted score
```

**Signal Quality Weighting:**

```python
# Signal strength affects position confidence
# Strong signal = more reliable location

signal_dbm = observation.get('signal_strength', -85)

# Map to weight (0.1 to 1.0)
#   -120 dBm (weak)   → 0.1
#   -85 dBm (medium)  → 0.5
#   -50 dBm (strong)  → 1.0
weight = max(0.1, min(1.0, (signal_dbm + 120) / 70))

# Weighted centroid
latitude = sum(lat_i * weight_i) / sum(weight_i)
longitude = sum(lon_i * weight_i) / sum(weight_i)
```

**Output DataFrame:**

| Column | Description |
| --- | --- |
| stay_id | Unique identifier |
| user_id | User (IMSI) |
| latitude, longitude | Centroid coordinates |
| cell_id | Most common cell |
| tac | Tracking Area Code |
| first_seen, last_seen | Time range |
| visit_count | Number of distinct visits |
| total_duration | Seconds spent at location |
| observation_count | Number of observations |
| location_confidence | 0-1 confidence score |

### `home_work_inference.py`

**Class: `HomeWorkInference`**

Infers home and work locations from stay patterns.

```python
class HomeWorkInference:
    def __init__(self, config=None)
    def infer(self, stay_points_df, observations_df=None) -> pd.DataFrame
    def get_home_work_summary(self, stay_points_df) -> pd.DataFrame
```

**Home Detection Algorithm:**

```md
Scoring Components (weighted):
─────────────────────────────
┌──────────────────────┬────────┬─────────────────────────────┐
│ Component            │ Weight │ Logic                       │
├──────────────────────┼────────┼─────────────────────────────┤
│ Weekday nights       │ 0.6    │ Present 8PM-7AM on weekdays │
│ Weekend presence     │ 0.3    │ Present on Sat/Sun          │
│ Early morning (3-6AM)│ 0.1    │ Very high confidence home   │
└──────────────────────┴────────┴─────────────────────────────┘

For each stay:
1. Count weekday nights present at stay
2. Count weekend days present at stay
3. Count early mornings present at stay
4. Calculate weighted score
5. Stay with highest score = home

Fallback (no temporal data):
- Use stay with longest total duration
- Weight by visit regularity
```

**Work Detection Algorithm:**

```md
Criteria:
1. NOT the home location
2. >= 500m from home (min_distance_from_home)
3. Present during work hours (7AM-8PM)
4. Weekdays only
5. >= 3 visits per week average (min_weekly_visits)

Scoring:
- Fraction of workdays present at location
- Stay with highest work-hour presence = work
```

**Output:**
Adds `location_type` column to stay_points_df:

- `'home'` - Inferred home location
- `'work'` - Inferred work location
- `'other'` - All other stays

## 4. Trip Generation (`src/trip_generation/`)

### `trip_generator.py`

**Class: `TripGenerator`**

Extracts trips from stay point sequences.

```python
class TripGenerator:
    def __init__(self, config=None)
    def generate(self, stay_points_df, observations_df=None) -> pd.DataFrame
    def validate_activity_chains(self, trips_df) -> pd.DataFrame
    def filter_incomplete_chains(self, trips_df, keep_partial=True) -> pd.DataFrame
    def get_trip_table(self, trips_df, group_by=None) -> pd.DataFrame
```

**Trip Extraction:**

```md
Method 1: From Observations (more accurate)
───────────────────────────────────────────
1. Assign each observation to nearest stay
2. Group observations by effective day (starting 3 AM)
3. Track current stay location
4. When observation at different stay:
   → Trip detected from previous stay to new stay
   → Departure = last observation at origin
   → Arrival = first observation at destination

Method 2: From Stay Timestamps (fallback)
─────────────────────────────────────────
1. Order stays by first_seen timestamp
2. Consecutive stays with different stay_id = trip
3. Departure = last_seen of origin stay
4. Arrival = first_seen of destination stay
```

**Trip Purpose Logic:**

```python
def determine_purpose(origin_type, dest_type):
    if (origin == 'home' and dest == 'work') or \
       (origin == 'work' and dest == 'home'):
        return 'HBW'  # Home-Based Work
    elif origin == 'home' or dest == 'home':
        return 'HBO'  # Home-Based Other
    else:
        return 'NHB'  # Non-Home Based
```

**Activity Chain Validation:**

```md
A valid activity chain has:
1. Spatial continuity: dest(trip_n) == origin(trip_n+1)
2. Home anchoring: Starts and/or ends at home

Example of valid chain:
  Trip 1: Home → Work      (origin = Home)
  Trip 2: Work → Lunch     (origin matches prev dest ✓)
  Trip 3: Lunch → Work     (origin matches prev dest ✓)
  Trip 4: Work → Home      (origin matches prev dest ✓, ends at home ✓)

Added columns:
- chain_id: Identifier for the activity chain
- chain_valid: True if spatially continuous
- home_anchored: True if starts/ends at home
- chain_complete: True if valid AND home-anchored
```

**Output Columns:**

| Column | Description |
| --- | --- |
| trip_id | Unique identifier (user_id + sequence) |
| user_id | User (IMSI) |
| origin_stay, destination_stay | Stay point IDs |
| origin_lat, origin_lon | Origin coordinates |
| dest_lat, dest_lon | Destination coordinates |
| origin_tac, dest_tac | Zone codes |
| origin_type, dest_type | home/work/other |
| trip_purpose | HBW/HBO/NHB |
| departure_time | Estimated departure |
| arrival_time | Arrival timestamp |
| time_period | AM_PEAK/MIDDAY/PM_PEAK/EVENING/NIGHT |
| day_type | weekday/weekend |
| distance_m | Trip distance in meters |
| duration_s | Trip duration in seconds |

### `trip_expander.py`

**Class: `TripExpander`**

Expands observed trips to population level.

```python
class TripExpander:
    def __init__(self, config=None, market_share=0.35,
                 expected_daily_trips=3.0)
    def expand(self, trips_df, user_stats, zone_populations=None,
               home_zones=None) -> pd.DataFrame
    def validate_trip_rates(self, trips_df, population,
                           observation_days=1) -> Dict
    def calibrate_expansion(self, trips_df, population,
                           target_rate=3.0) -> pd.DataFrame
    def get_expansion_summary(self, trips_df) -> Dict
```

**Expansion Methodology (Toole et al. 2015):**

```md
STAGE 1: USER-LEVEL EXPANSION
═════════════════════════════
Problem: Telecom only captures trips with phone events.
         Many trips don't generate events (short trips, no usage).

Solution: Compare observed rate to expected rate.

observed_daily_rate = trips_observed / active_days
user_factor = expected_daily_trips / observed_daily_rate

Example:
  User observed: 4 trips over 10 days
  observed_daily_rate = 4/10 = 0.4 trips/day
  expected_daily_trips = 3.0 (NHTS benchmark)
  user_factor = 3.0 / 0.4 = 7.5

Safeguards:
  - min_observed_rate = 0.5 (floor to prevent extreme factors)
  - max_expansion_factor = 20.0 (cap on any single factor)


STAGE 2: POPULATION-LEVEL EXPANSION
═══════════════════════════════════
Problem: We only observe one carrier's subscribers.
         Other carriers' subscribers are not in our data.

Solution: Scale by market penetration.

observed_users_in_zone = count of users with home in zone
estimated_total_users = observed_users / market_share
zone_factor = zone_population / estimated_total_users

Example:
  Zone population = 50,000
  Observed users = 1,750
  Market share = 35%
  Estimated total users = 1,750 / 0.35 = 5,000
  zone_factor = 50,000 / 5,000 = 10


COMBINED EXPANSION
══════════════════
expansion_factor = user_factor × zone_factor

Each observed trip contributes 'expansion_factor' to the OD matrix.
```

**Trip Rate Validation:**

```python
# Validate against NHTS benchmarks
result = expander.validate_trip_rates(
    trips_df,
    population=100000,
    observation_days=7,
    expected_rate_range=(2.5, 3.5)
)

# Result:
{
    'valid': True/False,
    'status': 'within_range' / 'under_estimated' / 'over_estimated',
    'observed_trip_rate': 2.8,
    'calibration_factor': 1.0
}
```

## 5. OD Matrix (`src/od_matrix/`)

### `od_generator.py`

**Class: `ODMatrixGenerator`**

Generates Origin-Destination matrices.

```python
class ODMatrixGenerator:
    def __init__(self, zone_loader=None, config=None)
    def generate(self, trips_df, zone_col_origin='origin_tac',
                zone_col_dest='dest_tac', weight_col='expanded_trips') -> pd.DataFrame
    def generate_by_purpose(self, trips_df) -> Dict[str, pd.DataFrame]
    def generate_by_time_period(self, trips_df) -> Dict[str, pd.DataFrame]
    def estimate_intra_zone_trips(self, od_matrix, zone_populations=None,
                                  intra_zone_rate=0.30) -> pd.DataFrame
    def to_matrix_form(self, od_df) -> Tuple[pd.DataFrame, List[str]]
    def to_csv(self, od_matrix, path)
    def compare_matrices(self, matrix_a, matrix_b) -> Dict
    def get_summary_statistics(self, od_matrix) -> Dict
```

**OD Matrix Generation:**

```md
Input: Expanded trips with origin_zone, dest_zone, expansion_factor

Process:
1. Group trips by (origin_zone, destination_zone)
2. Sum expansion factors → flow
3. Count trips → observed_trips
4. Calculate mean distance → avg_distance_m

Output: DataFrame with columns:
  - origin: Origin zone ID
  - destination: Destination zone ID
  - flow: Expanded trip count
  - observed_trips: Raw trip count
  - avg_distance_m: Average distance
```

**Intra-Zone Trip Estimation:**

```md
Problem: Telecom data misses intra-zone trips because:
1. Short trips may not trigger phone events
2. Movement within same cell/TAC is invisible
3. Research shows 20-40% of trips are intra-zone

Methods:
┌────────────┬───────────────────────────────────────────────────┐
│ Method     │ Description                                       │
├────────────┼───────────────────────────────────────────────────┤
│ proportion │ Scale observed intra-zone to reach target %       │
│            │ factor = target_rate / observed_rate              │
├────────────┼───────────────────────────────────────────────────┤
│ population │ Estimate from zone population:                    │
│            │ intra_ii = pop_i × daily_trips × intra_rate       │
├────────────┼───────────────────────────────────────────────────┤
│ gravity    │ Use gravity model for diagonal (future)           │
└────────────┴───────────────────────────────────────────────────┘

Example:
  Observed intra-zone: 10% of trips
  Target: 30%
  Scale factor = 30% / 10% = 3
  New diagonal = old diagonal × 3
```

**Matrix Comparison (for validation):**

```python
# Compare telecom OD with survey OD
stats = generator.compare_matrices(telecom_od, survey_od)

# Returns:
{
    'correlation': 0.85,        # Pearson correlation
    'rmse': 150.0,              # Root mean square error
    'mae': 100.0,               # Mean absolute error
    'total_flow_telecom': 50000,
    'total_flow_survey': 48000,
    'total_flow_ratio': 1.04,
    'common_pairs': 500,        # Both have flow
    'pairs_only_telecom': 50,   # Only in telecom
    'pairs_only_survey': 30     # Only in survey
}
```

## 6. Utilities (`src/utils/`)

### `config.py`

Configuration management with YAML support.

```python
from src.utils.config import Config, get_config

config = Config("config/default.yaml")
value = config.get('section.subsection.key', default_value)
value = config.stay_detection.get('distance_threshold', 500)
```

### `geo_utils.py`

Geographic utility functions.

```python
from src.utils.geo_utils import (
    haversine_distance,      # Distance in meters
    calculate_centroid,      # (lat, lon) centroid
    lat_lon_to_grid_cell,    # Grid cell assignment
    point_in_polygon         # Point-in-polygon test
)

dist = haversine_distance(lat1, lon1, lat2, lon2)
center = calculate_centroid([(lat1, lon1), (lat2, lon2)])
cell = lat_lon_to_grid_cell(lat, lon, cell_size, origin)
inside = point_in_polygon(lat, lon, polygon_coords)
```

### `time_utils.py`

Temporal utility functions.

```python
from src.utils.time_utils import (
    get_time_period,         # → 'AM_PEAK', 'MIDDAY', etc.
    get_day_type,            # → 'weekday' or 'weekend'
    get_effective_day,       # Day starting at custom hour
    is_home_time,            # Check if home hours
    is_work_time,            # Check if work hours
    is_weekday               # Check if weekday
)

period = get_time_period(timestamp)  # 'PM_PEAK'
day_type = get_day_type(timestamp)   # 'weekday'
eff_day = get_effective_day(timestamp, day_start_hour=3)
```

### `logger.py`

Logging utilities.

```python
from src.utils.logger import setup_logger, ProgressLogger

logger = setup_logger(__name__)
logger.info("Processing...")

# Progress logging for long operations
progress = ProgressLogger(total=1000, logger=logger)
for i, item in enumerate(items):
    progress.update(i)
```

## Summary Table

| Module | Class | Primary Method | Purpose |
| --- | --- | --- | --- |
| telecom_loader | TelecomDataLoader | load_all() | Load raw data |
| cell_tower_loader | CellTowerLoader | infer_from_xdr() | Get cell locations |
| zone_loader | ZoneLoader | create_tac_zones() | Define zones |
| telecom_preprocessor | TelecomPreprocessor | process() | Clean data |
| user_filter | UserFilter | filter_users() | Filter users |
| stay_detector | StayPointDetector | detect() | Find stays |
| home_work_inference | HomeWorkInference | infer() | Find home/work |
| trip_generator | TripGenerator | generate() | Extract trips |
| trip_expander | TripExpander | expand() | Scale to population |
| od_generator | ODMatrixGenerator | generate() | Create OD matrix |
