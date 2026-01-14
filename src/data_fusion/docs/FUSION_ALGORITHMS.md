# Fusion Algorithms

**Directory**: `fusion_algorithms/`

## Overview

This directory contains four different approaches to reconstruct vehicle trajectories from sensor data. Each algorithm combines different data sources with OpenStreetMap road network data.

```
fusion_algorithms/
├── base_fusion.py           # Abstract base class
├── gps_osm_fusion.py        # GPS + OSM
├── gtfs_osm_fusion.py       # GTFS + OSM
├── gps_gtfs_osm_fusion.py   # GPS + GTFS + OSM (Tri-source)
└── cdr_osm_fusion.py        # CDR + OSM
```

---

## Base Fusion Class

**File**: `base_fusion.py`

### Purpose

Provides common functionality inherited by all fusion algorithms:
- Map matching (snapping points to road network)
- Trajectory interpolation
- Speed calculation
- Trajectory smoothing

### Key Data Classes

```python
@dataclass
class ReconstructedPoint:
    """A point in the reconstructed trajectory."""
    timestamp: datetime
    latitude: float
    longitude: float
    speed_mps: float
    matched_edge_id: Optional[str]  # Road segment ID
    confidence: float               # 0-1, how sure we are
    source: str                     # Which data contributed
```

```python
@dataclass
class ReconstructedTrajectory:
    """A complete reconstructed trajectory."""
    trip_id: str
    vehicle_id: str
    points: List[ReconstructedPoint]
    processing_time_ms: float
    algorithm: str
```

### Core Methods

#### `map_match_point(lat, lon, search_radius_m)`

Snaps a GPS point to the nearest road segment.

```
Before Map Matching          After Map Matching
─────────────────────        ─────────────────────
                                    Road
    ○ GPS point              ═══════●═══════════
                                    ↑
    (off the road)           Snapped to road
```

#### `interpolate_between_points(p1, p2, interval_seconds)`

Fills gaps between two points with interpolated positions.

```
Original (gap)               After Interpolation
─────────────────────        ─────────────────────
●─────────────────●          ●──○──○──○──○──○──●
t=0              t=60        t=0  10  20  30... 60
```

#### `smooth_trajectory(trajectory, window_size)`

Applies moving average to reduce noise.

```
Before Smoothing             After Smoothing
─────────────────────        ─────────────────────
    ○
   / \    ○                  ────────────────────
  ○   \  / \                 Smooth curve
       ○    ○
```

---

## GPS + OSM Fusion

**File**: `gps_osm_fusion.py`

### How It Works

```
Input: Raw GPS points (noisy, with gaps)
Output: Smooth trajectory aligned to roads

Step 1: Map Matching
────────────────────────────────────────
GPS points:    ○  ○    ○  ○     ○
               │  │    │  │     │
Roads:    ═════●══●════●══●═════●═══════
               ↑  ↑    ↑  ↑     ↑
          Snapped to nearest road segment

Step 2: Gap Interpolation
────────────────────────────────────────
Before:   ●──────────────────●  (60 sec gap)
After:    ●──○──○──○──○──○──●  (1 sec intervals)
          All interpolated points also map-matched

Step 3: Smoothing
────────────────────────────────────────
Moving average filter reduces remaining noise
```

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_radius_m` | 50 m | Max distance for map matching |
| `interpolation_interval_s` | 1.0 sec | Interval for filling gaps |
| `max_gap_seconds` | 60 sec | Maximum gap to interpolate |
| `smooth_window` | 5 | Moving average window size |

### Usage

```python
from src.data_fusion.fusion_algorithms import GPSOSMFusion

fusion = GPSOSMFusion(
    road_network=osm_roads,  # Optional GeoDataFrame
    search_radius_m=50.0
)

trajectories = fusion.fuse(gps_data=gps_dataframe)
```

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Real-time position data | Cannot handle large gaps |
| High accuracy when GPS works | No schedule context |
| Simple and fast | Fails during dropout |

---

## GTFS + OSM Fusion

**File**: `gtfs_osm_fusion.py`

### How It Works

```
Input: GTFS schedule (stops + times)
Output: Trajectory based on scheduled positions

Step 1: Parse Stop Times
────────────────────────────────────────
Stop A (08:00) ──────────> Stop B (08:05)
     │                          │
     ▼                          ▼
  Position A               Position B

Step 2: Interpolate Between Stops
────────────────────────────────────────
08:00    08:01    08:02    08:03    08:04    08:05
  ●────────○────────○────────○────────○────────●
Stop A                                      Stop B

Each ○ is interpolated based on:
- Linear position between stops
- Estimated speed from distance/time
- Map-matched to road network
```

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interpolation_interval_s` | 1.0 sec | Points per second |
| `schedule_deviation_factor` | 0.0 | Add randomness to schedule |
| `dwell_time_s` | 30 sec | Default stop dwell time |

### Usage

```python
from src.data_fusion.fusion_algorithms import GTFSOSMFusion

fusion = GTFSOSMFusion()

trajectories = fusion.fuse(
    stops_df=gtfs_stops,
    stop_times_df=gtfs_stop_times,
    base_date=datetime(2025, 1, 1)  # Important for timestamp alignment
)
```

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Works without real-time data | Assumes schedule adherence |
| 100% temporal coverage | Only accurate at stops |
| No sensor hardware needed | Cannot capture deviations |

---

## GPS + GTFS + OSM Fusion (Tri-Source) ⭐

**File**: `gps_gtfs_osm_fusion.py`

### How It Works

This is the **recommended algorithm** - it combines the best of GPS and GTFS.

```
Fusion Strategy
────────────────────────────────────────

Case 1: GPS Available
─────────────────────
Use GPS (high confidence), map-match to road

Case 2: GPS Near GTFS Stop
─────────────────────────
Fuse both sources with weighted average:
  position = 0.7 × GPS + 0.3 × GTFS_stop
  confidence = 0.95 (very high)

Case 3: GPS Gap (dropout)
─────────────────────────
Fall back to GTFS-based interpolation:
- Find GTFS events during gap
- Interpolate between them
- Lower confidence (0.6-0.8)

Visual Timeline:
────────────────────────────────────────
GPS:    ●──●──●──●──────────────●──●──●
                  ↑ gap        ↑
GTFS:           ◆              ◆
                Stop A       Stop B

Result: ●──●──●──●──◇──◇──◇──◇──●──●──●
                    ↑ GTFS-based ↑
```

### Confidence Scoring

| Source | Confidence | Description |
|--------|------------|-------------|
| GPS + GTFS match | 0.95 | Both sources agree |
| GPS only | 0.7-0.9 | Depends on match distance |
| GTFS stop | 0.8-0.9 | At scheduled stop |
| GTFS interpolated | 0.6 | Between stops |
| Linear fallback | 0.3 | No GTFS during gap |

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gps_search_radius_m` | 50 m | GPS map matching radius |
| `gtfs_search_radius_m` | 100 m | GTFS stop matching radius |
| `max_gps_gap_seconds` | 120 sec | When to use GTFS fallback |
| `gps_weight` | 0.7 | Weight for GPS in fusion |
| `gtfs_weight` | 0.3 | Weight for GTFS in fusion |

### Usage

```python
from src.data_fusion.fusion_algorithms import GPSGTFSOSMFusion

fusion = GPSGTFSOSMFusion(
    gps_weight=0.7,
    gtfs_weight=0.3,
    max_gps_gap_seconds=120
)

trajectories = fusion.fuse(
    gps_data=gps_df,
    stops_df=gtfs_stops,
    stop_times_df=gtfs_stop_times,
    base_date=base_date
)
```

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Best accuracy overall | More complex |
| Handles GPS dropouts | Requires both data sources |
| Confidence scoring | Slightly slower |
| Robust to gaps | |

---

## CDR + OSM Fusion

**File**: `cdr_osm_fusion.py`

### How It Works

```
Input: Cell tower connection events (very sparse)
Output: Coarse trajectory estimate

Challenge: Cell towers have ~300m accuracy
────────────────────────────────────────

Tower A (300m radius)    Tower B (300m radius)
    ┌───────────┐           ┌───────────┐
   /             \         /             \
  /    Vehicle    \       /               \
 /    somewhere    \     /    somewhere    \
 \    in here     /     \    in here      /
  \              /       \               /
   \            /         \             /
    └──────────┘           └───────────┘

Solution:
1. Use tower centroid as position estimate
2. If multiple towers, triangulate
3. Map-match to road network
4. Interpolate between events
```

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tower_accuracy_m` | 200 m | Assumed tower accuracy |
| `use_tower_triangulation` | True | Use multiple towers |
| `interpolation_interval_s` | 1.0 sec | Fill between events |

### Usage

```python
from src.data_fusion.fusion_algorithms import CDROSMFusion

fusion = CDROSMFusion(cell_towers=cell_tower_df)

trajectories = fusion.fuse(
    cdr_data=cdr_events_df,
    cell_towers=cell_tower_df
)
```

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Works with basic mobile data | Very low accuracy (~200m) |
| Low cost | Sparse data points |
| Good coverage | Cannot capture detail |

---

## Algorithm Comparison Summary

| Algorithm | Accuracy | Coverage | Complexity | Best Use Case |
|-----------|----------|----------|------------|---------------|
| GPS+OSM | High | Variable | Low | High-quality GPS data |
| GTFS+OSM | Low | High | Low | No real-time data |
| **GPS+GTFS+OSM** | **Highest** | **High** | Medium | **Transit vehicles** |
| CDR+OSM | Very Low | Medium | Medium | No GPS available |

## Recommendation

For transit vehicles with GPS but potential gaps, use **GPS+GTFS+OSM (Tri-source fusion)**. It provides:
- Best accuracy when GPS works
- Graceful degradation during gaps
- Confidence scoring for quality assessment
