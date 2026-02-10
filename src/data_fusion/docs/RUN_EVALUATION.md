# Run Evaluation Script

**File**: `run_evaluation.py`

## Purpose

The main entry point for running the complete data fusion evaluation pipeline. This script orchestrates all components to produce comparison results and reports.

## Quick Start

```bash
# Activate virtual environment first
cd E:\urban_transit_tool
venv\Scripts\activate

# Run default evaluation (3 trips)
python src/data_fusion/run_evaluation.py

# Run with more trips
python src/data_fusion/run_evaluation.py --trips 5

# Launch interactive dashboard
python src/data_fusion/run_evaluation.py --dashboard

# Custom output directory
python src/data_fusion/run_evaluation.py --output ./my_results
```

## Command Line Options

| Option | Default | Description |
| --- | --- | --- |
| `--trips` | 3 | Number of trips to generate |
| `--output` | `./output` | Directory for results |
| `--dashboard` | False | Launch Streamlit dashboard |
| `--quiet` | False | Suppress verbose output |

## Pipeline Steps

The script executes these steps in order:

```md
┌─────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [1/5] Generate Ground Truth                            │
│        └─▶ Create perfect trajectories                  │
│                                                         │
│  [2/5] Simulate Sensor Data                             │
│        ├─▶ GPS with noise/dropout                       │
│        ├─▶ GTFS schedule                                │
│        ├─▶ Cell towers                                  │
│        └─▶ CDR events                                   │
│                                                         │
│  [3/5] Run Fusion Algorithms                            │
│        ├─▶ GPS+OSM                                      │
│        ├─▶ GTFS+OSM                                     │
│        ├─▶ GPS+GTFS+OSM                                 │
│        └─▶ CDR+OSM                                      │
│                                                         │
│  [4/5] Calculate Metrics                                │
│        └─▶ Compare all algorithms to ground truth       │
│                                                         │
│  [5/5] Generate Reports                                 │
│        ├─▶ Text, Markdown, JSON, CSV                    │
│        └─▶ Interactive charts                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Detailed Step Breakdown

### Step 1: Generate Ground Truth

```python
gt_generator = GroundTruthGenerator(
    avg_speed_kmh=25.0,
    max_speed_kmh=45.0
)

ground_truth_trips = gt_generator.generate_trips(
    num_trips=num_trips,
    route_id="ROUTE_001",
    headway_minutes=15
)
```

**Output**: List of `GroundTruthTrip` objects with perfect 1-second resolution trajectories.

### Step 2: Simulate Sensor Data

```python
simulator = SensorSimulator(seed=42)

# GPS with realistic errors
gps_data = simulator.simulate_gps(
    trips=ground_truth_trips,
    sample_interval=5.0,        # Every 5 seconds
    noise_std_meters=8.0,       # ±8m position error
    dropout_rate=0.1,           # 10% missing samples
    multipath_rate=0.02         # 2% large errors
)

# Cell towers along route
cell_towers = simulator.generate_cell_towers(
    generator=gt_generator,
    num_towers=15,
    coverage_radius_m=300.0
)

# GTFS schedule
gtfs_data = simulator.simulate_gtfs(
    generator=gt_generator,
    trips=ground_truth_trips,
    schedule_noise_seconds=60.0
)

# CDR events
cdr_data, cell_tower_df = simulator.simulate_cdr(
    trips=ground_truth_trips,
    events_per_hour=8.0
)
```

**Outputs**:

- `gps_data`: DataFrame with noisy GPS points
- `gtfs_data`: Dict with stops, stop_times, etc.
- `cdr_data`: DataFrame with CDR events
- `cell_tower_df`: DataFrame with tower locations

### Step 3: Run Fusion Algorithms

```python
algorithms = {
    'GPS+OSM': GPSOSMFusion(),
    'GTFS+OSM': GTFSOSMFusion(),
    'GPS+GTFS+OSM': GPSGTFSOSMFusion(),
    'CDR+OSM': CDROSMFusion(cell_towers=cell_tower_df)
}

reconstructed = {}
for name, algo in algorithms.items():
    trajectories = algo.fuse(...)  # Algorithm-specific inputs
    reconstructed[name] = pd.concat([t.to_dataframe() for t in trajectories])
```

**Output**: Dict mapping algorithm name to reconstructed trajectory DataFrame.

### Step 4: Calculate Metrics

```python
metrics_calc = FusionMetrics()
comparison_df = metrics_calc.compare_algorithms(
    ground_truth_df,
    reconstructed
)
```

**Output**: DataFrame with columns:

- `algorithm`
- `spatial_rmse_m`, `spatial_mae_m`, `max_spatial_error_m`, `p95_spatial_error_m`
- `temporal_mae_s`, `temporal_coverage`
- `speed_rmse_mps`, `speed_mae_mps`
- `coverage_rate`, `matched_points_rate`
- `avg_confidence`, `confidence_weighted_rmse`
- `processing_time_ms`, `points_per_second`
- `quality_score`

### Step 5: Generate Reports

```python
report_gen = ReportGenerator(output_dir=str(output_dir))
saved_files = report_gen.save_reports(comparison_df)

# Also save raw data
ground_truth_df.to_csv(output_dir / 'ground_truth.csv')
for algo_name, recon_df in reconstructed.items():
    recon_df.to_csv(output_dir / f'reconstructed_{algo_name}.csv')

# Generate visualization
charts = AccuracyCharts()
fig = charts.create_summary_dashboard(comparison_df)
fig.write_html(output_dir / 'comparison_dashboard.html')
```

## Output Files

After running, the `output/` directory contains:

```md
output/
├── fusion_evaluation_YYYYMMDD_HHMMSS.txt   # Text report
├── fusion_evaluation_YYYYMMDD_HHMMSS.md    # Markdown report
├── fusion_evaluation_YYYYMMDD_HHMMSS.json  # JSON data
├── fusion_evaluation_YYYYMMDD_HHMMSS.csv   # Metrics table
├── comparison_dashboard.html               # Interactive charts
├── ground_truth.csv                        # Perfect trajectory
├── reconstructed_gps_osm.csv              # GPS+OSM output
├── reconstructed_gtfs_osm.csv             # GTFS+OSM output
├── reconstructed_gps_gtfs_osm.csv         # Tri-source output
└── reconstructed_cdr_osm.csv              # CDR+OSM output
```

## Sample Output

```md
============================================================
DATA FUSION ALGORITHM EVALUATION
============================================================
Configuration:
  - Trips: 3
  - Output: E:\urban_transit_tool\src\data_fusion\output

[1/5] Generating ground truth trajectories...
  Generated 2258 ground truth points across 3 trips

[2/5] Simulating sensor data...
  GPS points: 462
  GTFS stops: 8
  CDR events: 37
  Cell towers: 15

[3/5] Running fusion algorithms...
  Running GPS+OSM...
    -> 2357 points reconstructed
  Running GTFS+OSM...
    -> 1512 points reconstructed
  Running GPS+GTFS+OSM...
    -> 2357 points reconstructed
  Running CDR+OSM...
    -> 2127 points reconstructed

[4/5] Calculating metrics...

============================================================
RESULTS SUMMARY
============================================================
   algorithm  spatial_rmse_m  coverage_rate  quality_score
GPS+GTFS+OSM           32.30         1.0000          0.652
     GPS+OSM           32.54         1.0000          0.652
    GTFS+OSM          205.20         0.7533          0.398
     CDR+OSM          225.36         0.9229          0.363

RECOMMENDED: GPS+GTFS+OSM
  Quality Score: 0.652
  Spatial RMSE: 32.30 m
  Coverage: 100.0%

[5/5] Generating reports...
Reports saved to E:\urban_transit_tool\src\data_fusion\output

============================================================
EVALUATION COMPLETE
============================================================
```

## Customization

### Modify Sensor Parameters

Edit in `run_evaluation.py`:

```python
# More GPS noise
gps_data = simulator.simulate_gps(
    trips=ground_truth_trips,
    noise_std_meters=15.0,  # Was 8.0
    dropout_rate=0.3,       # Was 0.1
)
```

### Add New Algorithm

1. Create `fusion_algorithms/my_fusion.py`
2. Import in `run_evaluation.py`:

   ```python
   from src.data_fusion.fusion_algorithms import MyFusion
   ```

3. Add to algorithms dict:

   ```python
   algorithms['MyFusion'] = MyFusion()
   ```

### Change Evaluation Location

Modify `GroundTruthGenerator` to use different coordinates:

```python
# Custom route (e.g., Delhi instead of Mumbai)
custom_route = LineString([
    (77.2090, 28.6139),  # Delhi coordinates
    ...
])
generator = GroundTruthGenerator(route_line=custom_route, stops=custom_stops)
```

## Troubleshooting

### Common Errors

| Error | Solution |
| --- | --- |
| `ModuleNotFoundError: geopandas` | Activate venv: `venv\Scripts\activate` |
| `TypeError: unexpected argument` | Check function signatures match |
| `0% coverage for GTFS` | Pass correct `base_date` parameter |
| `CDR 'list' has no iterrows` | Use `cell_tower_df` not `cell_towers` |

### Debug Mode

Add print statements or use:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
