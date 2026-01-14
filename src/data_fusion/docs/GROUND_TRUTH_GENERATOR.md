# Ground Truth Generator

**File**: `ground_truth_generator.py`

## Purpose

The Ground Truth Generator creates **perfect vehicle trajectories** that serve as the "answer key" for evaluating fusion algorithms. These trajectories represent what we would observe if we had a perfect sensor with zero noise.

## Why Do We Need This?

To evaluate how well a fusion algorithm works, we need to know the **correct answer**. In real-world scenarios, we never know the exact true position of a vehicle. By generating synthetic ground truth:

1. We know the **exact position** at every moment
2. We can **control the scenario** (route, speed, stops)
3. We can **measure exact errors** against the truth

## Key Components

### Data Classes

```python
@dataclass
class Stop:
    """Represents a bus stop along the route."""
    stop_id: str              # Unique identifier (e.g., "STOP_001")
    name: str                 # Human-readable name
    location: Point           # Shapely Point geometry
    distance_along_route: float  # Meters from route start
    dwell_time: float         # Seconds spent at stop (default: 30)
```

```python
@dataclass
class GroundTruthPoint:
    """A single point in the ground truth trajectory."""
    timestamp: datetime       # Exact time
    latitude: float           # Exact latitude
    longitude: float          # Exact longitude
    speed_mps: float          # Speed in meters/second
    distance_traveled: float  # Cumulative distance from start
    heading: float            # Direction (0-360 degrees)
    at_stop: Optional[str]    # Stop ID if currently at a stop
```

```python
@dataclass
class GroundTruthTrip:
    """A complete ground truth trip."""
    trip_id: str
    vehicle_id: str
    route_id: str
    start_time: datetime
    end_time: datetime
    points: List[GroundTruthPoint]
    stop_arrivals: Dict[str, datetime]   # stop_id -> arrival time
    stop_departures: Dict[str, datetime] # stop_id -> departure time
```

## How It Works

### 1. Route Generation

The generator creates a default bus route in **Mumbai (Bandra area)** with 8 stops spanning approximately 3km:

```
Bandra Station → Hill Road → Turner Road → Linking Road →
SV Road Junction → Khar Road → Santacruz Link → Terminal
```

The route is represented as a `LineString` geometry with coordinates.

### 2. Speed Profile Simulation

Real buses don't travel at constant speed. The generator simulates realistic speed profiles:

```
Speed Profile Between Stops
────────────────────────────────────────────────────────
     ┌─────────────────────────────┐
     │      Cruising (25 km/h)     │
    /│                             │\
   / │                             │ \
  /  │                             │  \
 /   │                             │   \
/    │                             │    \
Accel│                             │Decel
─────┴─────────────────────────────┴─────
Stop A                            Stop B
```

Parameters:
- **Acceleration**: 1.2 m/s² (gentle start)
- **Cruising Speed**: 25 km/h average, 45 km/h max
- **Deceleration**: 1.5 m/s² (gentle stop)

### 3. Stop Dwell Times

At each stop, the bus waits for passengers:
- Base dwell time: 30 seconds
- Random variation: ±10-20 seconds
- Creates realistic schedule deviations

### 4. Point Generation

Points are generated at **1-second intervals** along the trajectory:

```python
def generate_trip(self, trip_id, vehicle_id, route_id, start_time, ...):
    # For each segment between stops:
    #   1. Calculate travel time based on distance and speed
    #   2. Generate acceleration phase points
    #   3. Generate cruising phase points
    #   4. Generate deceleration phase points
    #   5. Generate dwell points at stop
```

## Usage

### Basic Usage

```python
from src.data_fusion import GroundTruthGenerator

# Create generator with default Mumbai route
generator = GroundTruthGenerator(
    avg_speed_kmh=25.0,    # Average speed between stops
    max_speed_kmh=45.0     # Maximum cruising speed
)

# Generate multiple trips (different start times)
trips = generator.generate_trips(
    num_trips=3,
    route_id="ROUTE_001",
    headway_minutes=15     # 15 minutes between trips
)

# Convert to DataFrame for analysis
for trip in trips:
    df = trip.to_dataframe()
    print(f"Trip {trip.trip_id}: {len(df)} points")
```

### Custom Route

```python
from shapely.geometry import LineString, Point

# Define custom route
custom_route = LineString([
    (lon1, lat1), (lon2, lat2), (lon3, lat3), ...
])

# Define stops
custom_stops = [
    Stop("S1", "Start", Point(lon1, lat1), 0, dwell_time=20),
    Stop("S2", "Middle", Point(lon2, lat2), 500, dwell_time=30),
    Stop("S3", "End", Point(lon3, lat3), 1000, dwell_time=20),
]

generator = GroundTruthGenerator(
    route_line=custom_route,
    stops=custom_stops
)
```

## Output Format

### DataFrame Columns

| Column | Type | Description |
|--------|------|-------------|
| `trip_id` | str | Trip identifier |
| `vehicle_id` | str | Vehicle identifier |
| `route_id` | str | Route identifier |
| `timestamp` | datetime | Exact timestamp |
| `latitude` | float | Latitude (WGS84) |
| `longitude` | float | Longitude (WGS84) |
| `speed_mps` | float | Speed in m/s |
| `speed_kmh` | float | Speed in km/h |
| `distance_traveled` | float | Cumulative distance (m) |
| `heading` | float | Direction (degrees) |
| `at_stop` | str/None | Stop ID if at stop |

### Example Output

```csv
trip_id,vehicle_id,timestamp,latitude,longitude,speed_mps,at_stop
trip_1,bus_1,2025-01-01 08:00:00,19.0596,72.8296,0.0,STOP_001
trip_1,bus_1,2025-01-01 08:00:01,19.0596,72.8296,0.0,STOP_001
...
trip_1,bus_1,2025-01-01 08:00:30,19.0596,72.8296,0.0,STOP_001
trip_1,bus_1,2025-01-01 08:00:31,19.0597,72.8297,1.2,None
trip_1,bus_1,2025-01-01 08:00:32,19.0598,72.8298,2.4,None
...
```

## Key Methods

### `generate_trip()`
Generates a single trip with specified parameters.

### `generate_trips()`
Generates multiple trips with regular headway (time spacing).

### `to_dataframe()` / `to_geodataframe()`
Convert trip to pandas DataFrame or GeoPandas GeoDataFrame.

### `save_ground_truth()`
Save to CSV, GeoJSON, or other formats.

## Design Decisions

1. **1-second intervals**: Provides high-resolution truth for accurate error measurement
2. **Realistic speed profiles**: Acceleration/deceleration makes trajectories believable
3. **Stop dwell variation**: Simulates real-world schedule adherence issues
4. **Mumbai location**: Uses actual coordinates for realistic map matching tests
