# Sensor Simulator

**File**: `sensor_simulator.py`

## Purpose

The Sensor Simulator takes perfect ground truth trajectories and **degrades them** to mimic real-world sensor data. This creates realistic test data with known errors, allowing us to evaluate how well fusion algorithms recover the true trajectory.

## Why Simulate Sensors?

Real sensor data has many issues:

- **GPS**: Position noise, signal dropout, multipath errors
- **GTFS**: Schedule deviations, missing data
- **CDR**: Very sparse, coarse positioning

By simulating these characteristics, we can:

1. **Control the noise level** - Test algorithms under different conditions
2. **Know the exact error** - Compare reconstructed vs. true position
3. **Reproduce results** - Same seed = same "random" noise

## Sensor Types Simulated

### 1. GPS Probe Data

Simulates vehicle GPS trackers with realistic errors.

#### Error Sources

```md
Real GPS Position Issues
────────────────────────────────────────
┌─────────────────────────────────────┐
│                                     │
│    × Multipath                      │
│      (reflection off buildings)     │
│                                     │
│         ●───────● True path         │
│        /         \                  │
│   ○···○···○···○···○  GPS readings   │
│   │   │   │   │                     │
│   Noise (±8m typical)               │
│                                     │
│         ○ ← Dropout (missing)       │
│                                     │
└─────────────────────────────────────┘
```

#### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `sample_interval` | 5.0 sec | Time between GPS samples |
| `noise_std_meters` | 8.0 m | Standard deviation of position noise |
| `dropout_rate` | 0.1 (10%) | Probability of missing a sample |
| `multipath_rate` | 0.02 (2%) | Probability of large jump error |
| `multipath_jump_meters` | 30.0 m | Size of multipath error |

#### How It Works

```python
def simulate_gps(self, trips, sample_interval=5.0, noise_std_meters=8.0,
                 dropout_rate=0.1, multipath_rate=0.02):
    for each trip:
        for each sample_time (every 5 seconds):
            # 1. Random dropout
            if random() < dropout_rate:
                skip this sample  # GPS signal lost

            # 2. Find true position at this time
            true_point = find_closest_ground_truth_point(sample_time)

            # 3. Add Gaussian noise
            noise_lat = normal(0, noise_std_meters / 111000)
            noise_lon = normal(0, noise_std_meters / 111000)

            # 4. Occasional multipath error (big jump)
            if random() < multipath_rate:
                add large random offset (30m)

            # 5. Add timing jitter (±1 second)
            # 6. Add speed noise (±10%)
            # 7. Add heading noise (±5 degrees)

            record GPS point
```

#### Output Columns

| Column | Description |
| --- | --- |
| `vehicle_id` | Vehicle identifier |
| `trip_id` | Trip identifier |
| `timestamp` | Sample time (with jitter) |
| `latitude` | Noisy latitude |
| `longitude` | Noisy longitude |
| `speed_kmh` | Noisy speed |
| `heading` | Noisy heading |
| `hdop` | Simulated dilution of precision |
| `num_satellites` | Simulated satellite count |

### 2. GTFS Schedule Data

Generates standard GTFS transit feed from ground truth.

#### GTFS Files Generated

| File | Contents |
| --- | --- |
| `agency.txt` | Transit agency info |
| `stops.txt` | Stop locations |
| `routes.txt` | Route definitions |
| `trips.txt` | Individual trips |
| `stop_times.txt` | Arrival/departure times |
| `calendar.txt` | Service schedule |

#### Schedule Noise

The simulator adds **schedule deviation** to make it realistic:

- Actual arrival times deviate from scheduled by ±60 seconds
- This mimics real-world traffic delays

```python
def simulate_gtfs(self, generator, trips, schedule_noise_seconds=60.0):
    # For each stop arrival in ground truth:
    actual_arrival = trip.stop_arrivals[stop_id]

    # Schedule is "planned" time (actual + random noise)
    noise = uniform(-schedule_noise_seconds, +schedule_noise_seconds)
    scheduled_arrival = actual_arrival + noise
```

#### Output Structure

```python
gtfs_data = {
    'agency': DataFrame,      # Agency info
    'stops': DataFrame,       # Stop locations
    'routes': DataFrame,      # Route definitions
    'trips': DataFrame,       # Trip metadata
    'stop_times': DataFrame,  # Arrival/departure times
    'calendar': DataFrame     # Service days
}
```

### 3. CDR (Call Detail Records)

Simulates cell tower connection data from mobile devices.

#### How CDR Works

```md
Cell Tower Coverage
────────────────────────────────────────
     Tower A            Tower B
        │                  │
        ▼                  ▼
    ┌───────┐          ┌───────┐
   /         \        /         \
  /   ●───────●──────●───────●   \
 /    Vehicle path through cells  \
└─────────────────────────────────┘
     300m radius      300m radius

CDR Events recorded when:
- Making/receiving calls
- Sending SMS
- Data connection
- Tower handover (moving between cells)
```

#### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `events_per_hour` | 8.0 | Average CDR events per hour |
| `include_handovers` | True | Include tower handover events |
| `num_towers` | 15 | Cell towers along route |
| `coverage_radius_m` | 300 m | Tower coverage radius |

#### How It Works

```python
def simulate_cdr(self, trips, events_per_hour=8.0, include_handovers=True):
    # 1. Generate random event times during trip
    num_events = trip_duration_hours * events_per_hour
    event_times = random times during trip

    for event_time in event_times:
        # 2. Find vehicle's true position
        true_position = find_ground_truth_point(event_time)

        # 3. Find nearest cell tower
        nearest_tower = find_nearest_tower(true_position)

        # 4. Record CDR event
        record(timestamp, tower_id, tower_location, event_type)

    # 5. Add handover events (tower changes during movement)
    if include_handovers:
        detect_tower_changes_along_path()
```

#### Output Columns

| Column | Description |
| --- | --- |
| `user_id` | User/device identifier |
| `trip_id` | Associated trip |
| `timestamp` | Event time |
| `tower_id` | Connected cell tower |
| `cell_tower_lat` | Tower latitude |
| `cell_tower_lon` | Tower longitude |
| `event_type` | 'call', 'sms', 'data', 'handover' |

### 4. Cell Tower Generation

Creates realistic cell tower placement along the route.

```python
def generate_cell_towers(self, generator, num_towers=15, coverage_radius_m=300):
    # Space towers evenly along route
    route_length = generator.route_length
    tower_spacing = route_length / (num_towers - 1)

    for i in range(num_towers):
        # Position along route
        distance = i * tower_spacing
        point = get_point_at_distance(distance)

        # Offset from road (towers aren't on the road)
        offset = random offset (50-150m perpendicular)

        tower = CellTower(
            tower_id=f"TOWER_{i}",
            latitude=point.y + offset,
            longitude=point.x + offset,
            radius_m=coverage_radius_m
        )
```

## Usage Examples

### Complete Simulation Pipeline

```python
from src.data_fusion import GroundTruthGenerator, SensorSimulator

# 1. Generate ground truth
generator = GroundTruthGenerator()
trips = generator.generate_trips(num_trips=3)

# 2. Initialize simulator
simulator = SensorSimulator(seed=42)  # Reproducible randomness

# 3. Simulate GPS with noise
gps_data = simulator.simulate_gps(
    trips=trips,
    sample_interval=5.0,      # Every 5 seconds
    noise_std_meters=8.0,     # ±8m noise
    dropout_rate=0.1          # 10% missing
)

# 4. Generate cell towers
cell_towers = simulator.generate_cell_towers(
    generator=generator,
    num_towers=15
)

# 5. Simulate GTFS schedule
gtfs_data = simulator.simulate_gtfs(
    generator=generator,
    trips=trips
)

# 6. Simulate CDR events
cdr_data, tower_df = simulator.simulate_cdr(
    trips=trips,
    events_per_hour=8.0
)
```

### Testing Different Noise Levels

```python
# Low noise scenario
gps_clean = simulator.simulate_gps(trips, noise_std_meters=3.0, dropout_rate=0.02)

# High noise scenario
gps_noisy = simulator.simulate_gps(trips, noise_std_meters=15.0, dropout_rate=0.3)

# Urban canyon (lots of multipath)
gps_urban = simulator.simulate_gps(trips, multipath_rate=0.1, multipath_jump_meters=50)
```

## Design Decisions

1. **Configurable parameters**: Test algorithms under various conditions
2. **Reproducible with seed**: Same seed = identical "random" data
3. **Realistic error models**: Based on actual GPS/CDR characteristics
4. **Separate simulation functions**: Can simulate each sensor independently
