# Telecom Travel Demand Model - System Documentation

## Overview

This system generates Origin-Destination (OD) travel demand matrices from telecom data (CDR, XDR, 4G, 5G records). It follows the methodology established by:

- **Toole et al. (2015)** - "The path most traveled: Travel demand estimation using big data resources"
- **Alexander et al. (2015)** - "Origin-destination trips by purpose and time of day inferred from mobile phone data"
- **Zheng & Xie (2011)** - Stay point detection algorithm

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  CDR Data    │   XDR Data    │   4G Data    │   5G Data    │  Cell Towers  │
│  (calls/SMS) │  (data usage) │  (LTE logs)  │  (NR logs)   │  (locations)  │
└──────┬───────┴───────┬───────┴──────┬───────┴──────┬───────┴───────┬────────┘
       │               │              │              │               │
       └───────────────┴──────────────┴──────────────┴───────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                 │
│                        (src/data_ingestion/)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  telecom_loader.py  │  cell_tower_loader.py  │  zone_loader.py              │
└─────────────────────┴────────────────────────┴──────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING LAYER                                  │
│                          (src/preprocessing/)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  telecom_preprocessor.py  │  user_filter.py  │  ping-pong filtering         │
└───────────────────────────┴──────────────────┴──────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAY DETECTION LAYER                                  │
│                         (src/stay_detection/)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  stay_detector.py  │  home_work_inference.py  │  signal quality weighting   │
└────────────────────┴──────────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRIP GENERATION LAYER                                 │
│                         (src/trip_generation/)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  trip_generator.py  │  trip_expander.py  │  activity chain validation       │
└─────────────────────┴────────────────────┴──────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OD MATRIX LAYER                                     │
│                           (src/od_matrix/)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  od_generator.py  │  intra-zone estimation  │  matrix comparison            │
└───────────────────┴─────────────────────────┴───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  OD Matrix (CSV)  │  Trip Tables  │  Validation Reports  │  Visualizations  │
└───────────────────┴───────────────┴─────────────────────┴───────────────────┘
```

---

## Module Documentation

### 1. Data Ingestion Layer (`src/data_ingestion/`)

#### 1.1 `telecom_loader.py` - TelecomDataLoader

**Purpose:** Load and parse raw telecom data from various sources.

**Supported Data Types:**
| Type | Description | Key Fields |
|------|-------------|------------|
| CDR | Call Detail Records | IMSI, timestamp, cell_id, call_type |
| XDR | Data Usage Records | IMSI, timestamp, cell_id, latitude, longitude |
| 4G | LTE Network Logs | IMSI, timestamp, TAC, eNodeB |
| 5G | NR Network Logs | IMSI, timestamp, TAC, gNodeB |

**Key Methods:**
```python
loader = TelecomDataLoader(config)

# Load individual data types
cdr_df = loader.load_cdr("path/to/cdr.csv")
xdr_df = loader.load_xdr("path/to/xdr.csv")

# Load all available data
all_data = loader.load_all(data_directory)
```

**Output DataFrame Columns:**
- `imsi`: Anonymized subscriber identifier
- `timestamp`: Event timestamp (datetime)
- `cell_id`: Cell tower identifier
- `tac`: Tracking Area Code (for zone mapping)
- `latitude`, `longitude`: Coordinates (if available)
- `event_type`: Type of telecom event

---

#### 1.2 `cell_tower_loader.py` - CellTowerLoader

**Purpose:** Load cell tower locations or infer them from XDR data.

**Key Methods:**
```python
loader = CellTowerLoader(config)

# Load from explicit tower database
towers = loader.load_towers("path/to/towers.csv")

# Infer locations from XDR (when tower DB unavailable)
towers = loader.infer_from_xdr(xdr_df)

# Add coordinates to telecom records
enriched_df = loader.add_locations(telecom_df, towers)
```

**Location Inference Algorithm:**
1. Group XDR records by `cell_id`
2. Calculate median latitude/longitude for each cell
3. Filter outliers (>3 std from median)
4. Return cell → (lat, lon) mapping

---

#### 1.3 `zone_loader.py` - ZoneLoader

**Purpose:** Define spatial zones for OD matrix aggregation.

**Zone Types:**
| Type | Description | Use Case |
|------|-------------|----------|
| TAC | Tracking Area Code zones | Coarse regional analysis |
| LAC | Location Area Code zones | Medium granularity |
| Grid | Regular grid cells | Fine-grained analysis |
| Custom | User-defined polygons | Administrative boundaries |

**Key Methods:**
```python
loader = ZoneLoader(config)

# Create TAC-based zones from data
zones = loader.create_tac_zones(telecom_df)

# Load custom zone definitions
zones = loader.load_zones("path/to/zones.geojson")

# Assign observations to zones
df = loader.assign_zones(telecom_df, zones)
```

---

### 2. Preprocessing Layer (`src/preprocessing/`)

#### 2.1 `telecom_preprocessor.py` - TelecomPreprocessor

**Purpose:** Clean, standardize, and prepare telecom data for analysis.

**Processing Steps:**
1. **Duplicate Removal** - Remove exact duplicate records
2. **Temporal Filtering** - Filter to study period
3. **Spatial Filtering** - Filter to study area
4. **Data Standardization** - Normalize column names and formats
5. **Multi-source Merging** - Combine CDR, XDR, 4G, 5G data
6. **Ping-Pong Filtering** - Remove cell oscillation artifacts

**Key Methods:**
```python
preprocessor = TelecomPreprocessor(config)

# Full preprocessing pipeline
clean_df = preprocessor.process(
    cdr_df=cdr_data,
    xdr_df=xdr_data,
    network_4g_df=lte_data,
    network_5g_df=nr_data
)

# Filter ping-pong movements
clean_df = preprocessor.filter_ping_pong(df, time_threshold_s=300)
clean_df = preprocessor.remove_ping_pong(df)

# Get user statistics
user_stats = preprocessor.get_user_summary(clean_df)
```

**Ping-Pong Detection:**
```
Ping-pong pattern: A → B → A → B → A (rapid oscillation)
                   |<--- < 5 min --->|

Detection criteria:
- Same cells alternating (A-B-A pattern)
- Time between switches < threshold (default 300s)
- More than 3 consecutive oscillations
```

---

#### 2.2 `user_filter.py` - UserFilter

**Purpose:** Filter users based on data quality criteria.

**Filter Criteria:**
| Criterion | Default | Description |
|-----------|---------|-------------|
| min_records | 10 | Minimum total observations |
| min_days | 3 | Minimum days with activity |
| min_daily_records | 2 | Average daily observations |
| max_daily_records | 1000 | Maximum (filters bots/M2M) |

**Key Methods:**
```python
user_filter = UserFilter(config)

# Filter users
valid_df = user_filter.filter_users(telecom_df)

# Get filter statistics
stats = user_filter.get_filter_stats(telecom_df)
```

---

### 3. Stay Detection Layer (`src/stay_detection/`)

#### 3.1 `stay_detector.py` - StayPointDetector

**Purpose:** Identify meaningful locations where users spend significant time.

**Algorithm (Zheng-Xie with enhancements):**

```
Phase 1: Candidate Extraction
─────────────────────────────
For each user's time-ordered observations:
1. Start with first observation as anchor
2. Add consecutive observations within distance_threshold
3. If time span ≥ time_threshold → candidate stay
4. Move to next unprocessed observation
5. Repeat until all observations processed

Phase 2: Progressive Threshold Relaxation (if no stays found)
────────────────────────────────────────────────────────────
1. Try with 1.5x distance threshold
2. Try with 0.5x time threshold
3. Try with both relaxed
4. Fall back to cell-based grouping (last resort)

Phase 3: Grid-based Consolidation
─────────────────────────────────
1. Assign stays to grid cells (default 300m)
2. Merge stays in same grid cell
3. Calculate consolidated centroid

Phase 4: Metrics Calculation
────────────────────────────
For each stay:
- visit_count: Number of distinct visits
- total_duration: Cumulative time at location
- observation_count: Total observations
- location_confidence: Signal-weighted confidence score
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| distance_threshold | 500m | Max distance for same location |
| time_threshold | 1800s | Min time to qualify as stay (30 min) |
| grid_cell_size | 300m | Consolidation grid resolution |
| min_visits | 2 | Minimum visits to keep stay |

**Signal Quality Weighting:**
```python
# Signal strength mapped to weight (0.1 to 1.0)
# -120 dBm (weak) → 0.1 weight
# -50 dBm (strong) → 1.0 weight
weight = max(0.1, min(1.0, (signal + 120) / 70))

# Weighted centroid calculation
lat = Σ(lat_i × weight_i) / Σ(weight_i)
lon = Σ(lon_i × weight_i) / Σ(weight_i)
```

**Key Methods:**
```python
detector = StayPointDetector(
    distance_threshold=500,
    time_threshold=1800,
    grid_cell_size=300
)

# Detect stay points
stay_points = detector.detect(telecom_df)

# Output columns:
# stay_id, user_id, latitude, longitude, cell_id, tac,
# first_seen, last_seen, visit_count, total_duration,
# observation_count, location_confidence
```

---

#### 3.2 `home_work_inference.py` - HomeWorkInference

**Purpose:** Infer home and work locations from stay point patterns.

**Methodology (Alexander et al. 2015):**

```
HOME DETECTION
══════════════
Scoring components (weighted combination):
┌─────────────────────────────────────────────────────────┐
│ Component              │ Weight │ Logic                 │
├────────────────────────┼────────┼───────────────────────┤
│ Weekday nights         │  0.6   │ Present 8PM-7AM       │
│ Weekend presence       │  0.3   │ Present on Sat/Sun    │
│ Early morning (3-6AM)  │  0.1   │ High-confidence home  │
└────────────────────────┴────────┴───────────────────────┘

Home = Stay with highest combined score

WORK DETECTION
══════════════
Criteria:
1. Not the home location
2. ≥ 500m from home
3. Visited during work hours (7AM-8PM)
4. ≥ 3 visits per week on average
5. Weekdays only

Work = Non-home stay with highest work-hour presence score
```

**Key Methods:**
```python
inference = HomeWorkInference(config)

# Infer home/work for all users
stay_points = inference.infer(stay_points_df, observations_df)

# Output: stay_points with 'location_type' column
# Values: 'home', 'work', 'other'

# Get summary
summary = inference.get_home_work_summary(stay_points)
```

---

### 4. Trip Generation Layer (`src/trip_generation/`)

#### 4.1 `trip_generator.py` - TripGenerator

**Purpose:** Extract trips from stay point sequences and assign trip purposes.

**Trip Definition:**
```
A trip is movement between two different stay points.

Stay A ────────────────────────► Stay B
  │                                 │
  │ departure_time                  │ arrival_time
  │ (last obs at A)                 │ (first obs at B)
  │                                 │
  └─────────── Trip ────────────────┘
              │
              ├── distance_m
              ├── duration_s
              ├── trip_purpose (HBW/HBO/NHB)
              └── time_period (AM_PEAK/PM_PEAK/etc.)
```

**Trip Purpose Classification:**
| Origin | Destination | Purpose | Description |
|--------|-------------|---------|-------------|
| Home | Work | HBW | Home-Based Work |
| Work | Home | HBW | Home-Based Work |
| Home | Other | HBO | Home-Based Other |
| Other | Home | HBO | Home-Based Other |
| Other | Other | NHB | Non-Home Based |
| Work | Other | NHB | Non-Home Based |
| Other | Work | NHB | Non-Home Based |

**Activity Chain Validation:**
```
Valid chain: Trip destinations connect to next trip origins
─────────────────────────────────────────────────────────
Trip 1: Home → Work     ✓ (dest = next origin)
Trip 2: Work → Shop     ✓ (dest = next origin)
Trip 3: Shop → Home     ✓ (ends at home)

Invalid chain:
─────────────
Trip 1: Home → Work
Trip 2: Gym → Home      ✗ (gap: Work ≠ Gym)
```

**Key Methods:**
```python
generator = TripGenerator(config)

# Generate trips
trips_df = generator.generate(stay_points_df, observations_df)

# Validate activity chains
trips_df = generator.validate_activity_chains(trips_df)

# Filter incomplete chains
trips_df = generator.filter_incomplete_chains(trips_df, keep_partial=True)

# Get trip summary table
trip_table = generator.get_trip_table(trips_df)
```

**Output Columns:**
- `trip_id`, `user_id`
- `origin_stay`, `destination_stay`
- `origin_lat`, `origin_lon`, `dest_lat`, `dest_lon`
- `origin_tac`, `dest_tac`
- `origin_type`, `dest_type` (home/work/other)
- `trip_purpose` (HBW/HBO/NHB)
- `departure_time`, `arrival_time`
- `time_period` (AM_PEAK/MIDDAY/PM_PEAK/EVENING/NIGHT)
- `day_type` (weekday/weekend)
- `distance_m`, `duration_s`
- `chain_id`, `chain_valid`, `home_anchored`

---

#### 4.2 `trip_expander.py` - TripExpander

**Purpose:** Expand observed trips to represent full population.

**Two-Stage Expansion (Toole et al. 2015):**

```
STAGE 1: USER-LEVEL EXPANSION
═════════════════════════════
Accounts for under-observation of trips in telecom data.
Not all trips generate phone events.

Formula:
user_factor = expected_daily_trips / observed_daily_rate

Where:
- expected_daily_trips = 3.0 (NHTS average)
- observed_daily_rate = trips_counted / active_days

Example:
- User has 2 observed trips over 5 days
- observed_daily_rate = 2/5 = 0.4 trips/day
- user_factor = 3.0 / 0.4 = 7.5

Safeguards:
- min_observed_rate = 0.5 (prevents extreme factors)
- max_expansion_factor = 20.0 (caps outliers)


STAGE 2: POPULATION-LEVEL EXPANSION
════════════════════════════════════
Accounts for carrier market penetration.
We only observe subscribers of one carrier.

Formula:
zone_factor = zone_population / (observed_users / market_share)

Example:
- Zone population = 100,000
- Observed users from zone = 3,500
- Market share = 35%
- Estimated total users = 3,500 / 0.35 = 10,000
- zone_factor = 100,000 / 10,000 = 10


COMBINED EXPANSION
══════════════════
expansion_factor = user_factor × zone_factor
expanded_trips = 1 × expansion_factor (per observed trip)
```

**Trip Rate Validation:**
```python
# Expected daily trip rates (NHTS benchmarks)
Expected range: 2.5 - 3.5 trips/person/day

# Validation
result = expander.validate_trip_rates(
    trips_df,
    population=100000,
    observation_days=7,
    expected_rate_range=(2.5, 3.5)
)

# Calibration (if needed)
trips_df = expander.calibrate_expansion(
    trips_df,
    population=100000,
    target_rate=3.0
)
```

**Key Methods:**
```python
expander = TripExpander(
    market_share=0.35,
    expected_daily_trips=3.0
)

# Expand trips
expanded_df = expander.expand(
    trips_df,
    user_stats,
    zone_populations,  # Optional: Dict[zone_id, population]
    home_zones         # Optional: Dict[user_id, home_zone_id]
)

# Get expansion summary
summary = expander.get_expansion_summary(expanded_df)
```

---

### 5. OD Matrix Layer (`src/od_matrix/`)

#### 5.1 `od_generator.py` - ODMatrixGenerator

**Purpose:** Generate Origin-Destination matrices at various granularities.

**OD Matrix Structure:**
```
         │ Zone A │ Zone B │ Zone C │ ...
─────────┼────────┼────────┼────────┼────
Zone A   │  T_AA  │  T_AB  │  T_AC  │
Zone B   │  T_BA  │  T_BB  │  T_BC  │
Zone C   │  T_CA  │  T_CB  │  T_CC  │
...      │        │        │        │

T_ij = Expanded trips from zone i to zone j
Diagonal (T_ii) = Intra-zone trips
```

**Intra-Zone Trip Estimation:**
```
Problem: Telecom data under-counts intra-zone trips because:
1. Short trips may not generate phone events
2. Movement within same cell/TAC is invisible
3. Typically 20-40% of trips are intra-zone (Toole et al.)

Solution: Estimate and adjust diagonal of OD matrix

Methods:
┌─────────────┬──────────────────────────────────────────┐
│ Method      │ Description                              │
├─────────────┼──────────────────────────────────────────┤
│ proportion  │ Scale observed intra-zone to target %   │
│ population  │ Estimate from zone population × rate    │
│ gravity     │ Use gravity model for diagonal          │
└─────────────┴──────────────────────────────────────────┘
```

**Key Methods:**
```python
generator = ODMatrixGenerator(zone_loader, config)

# Generate OD matrix
od_matrix = generator.generate(
    trips_df,
    zone_col_origin='origin_tac',
    zone_col_dest='dest_tac',
    weight_col='expanded_trips'
)

# Generate by purpose/time period
od_by_purpose = generator.generate_by_purpose(trips_df)  # {'HBW': df, 'HBO': df, 'NHB': df}
od_by_time = generator.generate_by_time_period(trips_df) # {'AM_PEAK': df, ...}

# Estimate intra-zone trips
od_matrix = generator.estimate_intra_zone_trips(
    od_matrix,
    zone_populations,
    intra_zone_rate=0.30
)

# Convert to square matrix form
matrix, zones = generator.to_matrix_form(od_matrix)

# Save to CSV
generator.to_csv(od_matrix, "output/od_matrix.csv")

# Compare with survey data
comparison = generator.compare_matrices(telecom_od, survey_od)
```

**Output Columns:**
- `origin`: Origin zone ID
- `destination`: Destination zone ID
- `flow`: Expanded trip count
- `observed_trips`: Raw observed trips
- `avg_distance_m`: Average trip distance

---

### 6. Utility Modules (`src/utils/`)

#### 6.1 `config.py` - Configuration Management

**Purpose:** Centralized configuration with defaults and validation.

**Usage:**
```python
from src.utils.config import Config, get_config

# Load from file
config = Config("config/default.yaml")

# Access nested values
distance = config.stay_detection.get('distance_threshold', 500)
market_share = config.get('od_matrix.expansion.market_share', 0.35)
```

#### 6.2 `geo_utils.py` - Geographic Utilities

**Key Functions:**
```python
from src.utils.geo_utils import (
    haversine_distance,      # Distance between two lat/lon points
    calculate_centroid,      # Centroid of point set
    lat_lon_to_grid_cell,    # Assign point to grid cell
    point_in_polygon         # Check if point in polygon
)

# Example
dist = haversine_distance(lat1, lon1, lat2, lon2)  # Returns meters
```

#### 6.3 `time_utils.py` - Temporal Utilities

**Key Functions:**
```python
from src.utils.time_utils import (
    get_time_period,         # Classify into AM_PEAK, MIDDAY, etc.
    get_day_type,            # weekday/weekend
    get_effective_day,       # Day starting at 3 AM
    is_home_time,            # Check if timestamp is home hours
    is_work_time             # Check if timestamp is work hours
)

# Time periods
# AM_PEAK:  6:00 - 9:00
# MIDDAY:   9:00 - 16:00
# PM_PEAK:  16:00 - 19:00
# EVENING:  19:00 - 22:00
# NIGHT:    22:00 - 6:00
```

---

## Configuration Reference

### Default Configuration (`config/default.yaml`)

```yaml
# Stay Detection
stay_detection:
  distance_threshold: 500      # meters
  time_threshold: 1800         # seconds (30 min)
  grid_cell_size: 300          # meters
  min_visits: 2

# Home/Work Inference
home_work_inference:
  home:
    start_hour: 20             # 8 PM
    end_hour: 7                # 7 AM
    min_frequency: 0.5
  work:
    start_hour: 7
    end_hour: 20
    min_distance_from_home: 500
    min_weekly_visits: 3

# Trip Generation
trip_generation:
  day_start_hour: 3            # Day boundary
  min_trip_distance: 200       # meters
  max_trip_distance: 100000    # meters
  max_trip_duration: 14400     # seconds (4 hours)
  departure_time_method: conditional_probability

# OD Matrix
od_matrix:
  precision: 2                 # Decimal places for flow
  expansion:
    market_share: 0.35         # Carrier market share
    expected_daily_trips: 3.0  # NHTS average
    min_observed_rate: 0.5     # Min trips/day to prevent extreme factors
    max_expansion_factor: 20.0 # Cap on expansion
    apply_vehicle_rate: false
    default_vehicle_rate: 0.3

# Preprocessing
preprocessing:
  temporal_filter:
    enabled: true
  spatial_filter:
    enabled: false
  user_filter:
    min_records: 10
    min_days: 3
```

---

## Data Flow Example

### Complete Pipeline Execution

```python
from src.data_ingestion import TelecomDataLoader, CellTowerLoader, ZoneLoader
from src.preprocessing import TelecomPreprocessor
from src.stay_detection import StayPointDetector, HomeWorkInference
from src.trip_generation import TripGenerator, TripExpander
from src.od_matrix import ODMatrixGenerator

# 1. Load Data
loader = TelecomDataLoader()
cdr_df = loader.load_cdr("data/cdr.csv")
xdr_df = loader.load_xdr("data/xdr.csv")

# 2. Infer Cell Locations
cell_loader = CellTowerLoader()
cell_locations = cell_loader.infer_from_xdr(xdr_df)

# 3. Preprocess
preprocessor = TelecomPreprocessor()
clean_df = preprocessor.process(cdr_df=cdr_df, xdr_df=xdr_df)
clean_df = cell_loader.add_locations(clean_df, cell_locations)
clean_df = preprocessor.remove_ping_pong(clean_df)
user_stats = preprocessor.get_user_summary(clean_df)

# 4. Create Zones
zone_loader = ZoneLoader()
zones = zone_loader.create_tac_zones(clean_df)

# 5. Detect Stay Points
detector = StayPointDetector(distance_threshold=500, time_threshold=1800)
stay_points = detector.detect(clean_df)

# 6. Infer Home/Work
inference = HomeWorkInference()
stay_points = inference.infer(stay_points, clean_df)

# 7. Generate Trips
generator = TripGenerator()
trips = generator.generate(stay_points, clean_df)
trips = generator.validate_activity_chains(trips)

# 8. Expand Trips
expander = TripExpander(market_share=0.35)
expanded = expander.expand(trips, user_stats)

# 9. Generate OD Matrix
od_gen = ODMatrixGenerator(zone_loader)
od_matrix = od_gen.generate(expanded)
od_matrix = od_gen.estimate_intra_zone_trips(od_matrix, intra_zone_rate=0.30)

# 10. Export
od_gen.to_csv(od_matrix, "output/od_matrix.csv")
```

---

## Quality Metrics & Validation

### Expected Benchmarks

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Daily trip rate | 2.5 - 3.5 trips/person | NHTS |
| Intra-zone trips | 20% - 40% of total | Toole et al. |
| HBW trips | 15% - 25% of total | Travel surveys |
| Home detection rate | > 90% of users | Alexander et al. |
| Work detection rate | 40% - 60% of employed | Alexander et al. |

### Validation Methods

1. **Trip Rate Validation**
   ```python
   result = expander.validate_trip_rates(trips, population, days)
   # Check result['observed_trip_rate'] is in expected range
   ```

2. **OD Matrix Comparison**
   ```python
   stats = od_gen.compare_matrices(telecom_od, survey_od)
   # Check stats['correlation'] > 0.7
   # Check stats['rmse'] is acceptable
   ```

3. **Activity Chain Completeness**
   ```python
   trips = generator.validate_activity_chains(trips)
   complete_pct = trips.groupby('chain_id')['chain_complete'].first().mean()
   # Expect > 60% complete chains
   ```

---

## Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| 0 stay points detected | Sparse data, strict thresholds | Relaxed thresholds are auto-applied; check data quality |
| 0 work locations | Short observation period | Need 1+ week of data for work detection |
| High expansion factors | Low observation rate | Check user_stats; may need more data |
| Missing distances | No coordinates | Ensure cell tower locations are loaded |
| NaN in OD matrix | Missing zone assignments | Check TAC column exists in stays |

### Debug Logging

```python
import logging
logging.getLogger('src').setLevel(logging.DEBUG)
```

---

## References

1. Toole, J.L., et al. (2015). "The path most traveled: Travel demand estimation using big data resources." Transportation Research Part C.

2. Alexander, L., et al. (2015). "Origin-destination trips by purpose and time of day inferred from mobile phone data." Transportation Research Part C.

3. Zheng, Y., & Xie, X. (2011). "Learning travel recommendations from user-generated GPS traces." ACM TIST.

4. Calabrese, F., et al. (2011). "Estimating Origin-Destination flows using mobile phone location data." IEEE Pervasive Computing.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-08 | Initial release with methodological fixes |

---

*Generated for Telecom Travel Demand Model*
