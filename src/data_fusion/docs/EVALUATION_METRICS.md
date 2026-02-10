# Evaluation Metrics

**Directory**: `evaluation/`

## Overview

The evaluation module measures how well fusion algorithms reconstruct trajectories compared to ground truth.

```md
evaluation/
├── metrics.py           # Calculate accuracy metrics
├── comparator.py        # Run & compare all algorithms
└── report_generator.py  # Generate comparison reports
```

## Metrics Calculator

**File**: `metrics.py`

### How Evaluation Works

```md
Evaluation Process
────────────────────────────────────────

Ground Truth:    ●──●──●──●──●──●──●──●──●
                 │  │  │  │  │  │  │  │  │
                 ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
Reconstructed:   ○  ○  ○     ○  ○     ○  ○
                 │  │  │     │  │     │  │
                 ▼  ▼  ▼     ▼  ▼     ▼  ▼
Errors:          ε₁ ε₂ ε₃    ε₄ ε₅    ε₆ ε₇

1. Match points by timestamp (within threshold)
2. Calculate spatial distance for each pair
3. Aggregate into metrics (RMSE, MAE, etc.)
```

### Metrics Calculated

#### Spatial Metrics

| Metric | Formula | Description |
| --- | --- | --- |
| **RMSE** | √(Σεᵢ²/n) | Root Mean Square Error - penalizes large errors |
| **MAE** | Σ\|εᵢ\|/n | Mean Absolute Error - average error |
| **Max Error** | max(εᵢ) | Worst case error |
| **P95 Error** | 95th percentile | Error for 95% of points |

```python
# Spatial RMSE calculation
errors = []
for gt_point, recon_point in matched_pairs:
    distance = haversine(gt_point, recon_point)  # meters
    errors.append(distance)

rmse = sqrt(mean(errors²))
mae = mean(|errors|)
```

#### Temporal Metrics

| Metric | Description |
| --- | --- |
| **Temporal MAE** | Average time difference between matched points |
| **Temporal Coverage** | Ratio of reconstructed duration to ground truth duration |

#### Speed Metrics

| Metric | Description |
| --- | --- |
| **Speed RMSE** | Error in speed estimation (m/s) |
| **Speed MAE** | Average speed error |

#### Coverage Metrics

| Metric | Description |
| --- | --- |
| **Coverage Rate** | % of ground truth points matched to reconstructed |
| **Matched Points Rate** | % of points successfully map-matched to roads |

#### Confidence Metrics

| Metric | Description |
| --- | --- |
| **Avg Confidence** | Mean confidence score (0-1) |
| **Confidence-Weighted RMSE** | RMSE weighted by inverse confidence |

### Quality Score

A **composite score** (0-1) combining multiple metrics:

```python
quality_score = (
    0.40 × spatial_score +      # Accuracy matters most
    0.25 × coverage_score +     # Good coverage important
    0.20 × confidence_score +   # Algorithm's self-assessment
    0.15 × speed_score          # Processing efficiency
)

where:
    spatial_score = exp(-RMSE / 50)  # Exponential decay
    coverage_score = coverage_rate
    confidence_score = avg_confidence
    speed_score = min(1, points_per_second / 10000)
```

### Usage

```python
from src.data_fusion.evaluation import FusionMetrics

metrics = FusionMetrics(
    spatial_threshold_m=50.0,    # Max distance for matching
    temporal_threshold_s=5.0     # Max time difference
)

# Evaluate single algorithm
result = metrics.evaluate(
    ground_truth=gt_df,
    reconstructed=recon_df,
    algorithm_name="GPS+OSM"
)

print(f"RMSE: {result.spatial_rmse_m:.2f} m")
print(f"Coverage: {result.coverage_rate:.1%}")
print(f"Quality Score: {result.quality_score:.3f}")
```

### MetricsResult Object

```python
@dataclass
class MetricsResult:
    algorithm_name: str

    # Spatial
    spatial_rmse_m: float
    spatial_mae_m: float
    max_spatial_error_m: float
    p95_spatial_error_m: float

    # Temporal
    temporal_mae_s: float
    temporal_coverage: float

    # Speed
    speed_rmse_mps: float
    speed_mae_mps: float

    # Coverage
    coverage_rate: float
    matched_points_rate: float

    # Confidence
    avg_confidence: float
    confidence_weighted_rmse: float

    # Processing
    processing_time_ms: float
    points_per_second: float

    # Composite
    quality_score: float
```

## Algorithm Comparator

**File**: `comparator.py`

### Purpose

Runs all fusion algorithms on the same data and compares results.

### How It Works

```md
Comparison Pipeline
────────────────────────────────────────

                    ┌─────────────────┐
                    │  Ground Truth   │
                    │  + Sensor Data  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ GPS+OSM  │     │ GTFS+OSM │     │ GPS+GTFS │ ...
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         ▼                ▼                ▼
    Trajectory 1    Trajectory 2     Trajectory 3
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    Metrics    │
                  │  Calculation  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Comparison   │
                  │    Table      │
                  └───────────────┘
```

### Usage

```python
from src.data_fusion.evaluation import FusionComparator

comparator = FusionComparator(road_network=osm_roads)

# Run all algorithms
comparison_df = comparator.run_comparison(
    ground_truth=gt_df,
    gps_data=gps_df,
    gtfs_stops=stops_df,
    gtfs_stop_times=stop_times_df,
    cdr_data=cdr_df,
    cell_towers=towers_df,
    algorithms_to_run=['GPS+OSM', 'GTFS+OSM', 'GPS+GTFS+OSM', 'CDR+OSM'],
    verbose=True
)

# Get best algorithm
best = comparator.get_best_algorithm()
print(f"Recommended: {best}")

# Get rankings
rankings = comparator.get_ranking()
```

### Sparsity Analysis

Test how algorithms perform with increasing data dropout:

```python
sparsity_results = comparator.run_sparsity_analysis(
    ground_truth=gt_df,
    gps_data=gps_df,
    gtfs_stops=stops_df,
    gtfs_stop_times=stop_times_df,
    cdr_data=cdr_df,
    cell_towers=towers_df,
    sparsity_levels=[0.1, 0.3, 0.5, 0.7]  # 10% to 70% dropout
)
```

## Report Generator

**File**: `report_generator.py`

### Purpose

Generates human-readable reports from comparison results.

### Output Formats

| Format | Use Case |
| --- | --- |
| `.txt` | Copy/paste into emails |
| `.md` | Documentation, GitHub |
| `.json` | Programmatic analysis |
| `.csv` | Open in Excel |

### Usage

```python
from src.data_fusion.evaluation import ReportGenerator

report_gen = ReportGenerator(output_dir="./output")

# Generate all formats
saved_files = report_gen.save_reports(
    comparison_df=comparison_df,
    sparsity_df=sparsity_df,
    prefix="fusion_evaluation"
)
# Creates:
#   fusion_evaluation_20250130_120000.txt
#   fusion_evaluation_20250130_120000.md
#   fusion_evaluation_20250130_120000.json
#   fusion_evaluation_20250130_120000.csv

# Or generate individual formats
text_report = report_gen.generate_summary(comparison_df)
md_report = report_gen.generate_markdown_report(comparison_df)
json_report = report_gen.generate_json_report(comparison_df)
```

### Sample Text Report

```md
============================================================
DATA FUSION ALGORITHM COMPARISON REPORT
Generated: 2025-01-30 12:00:00
============================================================

RECOMMENDED ALGORITHM
----------------------------------------
  GPS+GTFS+OSM
  Quality Score: 0.652
  Spatial RMSE: 32.30 m
  Coverage Rate: 100.0%

ALGORITHM COMPARISON
----------------------------------------
Algorithm            RMSE(m)    Coverage   Score
----------------------------------------
GPS+GTFS+OSM         32.30      100.0%     0.652
GPS+OSM              32.54      100.0%     0.652
GTFS+OSM             205.20     75.3%      0.398
CDR+OSM              225.36     92.3%      0.363

============================================================
RECOMMENDATION
============================================================
Based on the evaluation, GPS+GTFS+OSM is recommended.

Justification:
- Combines real-time GPS accuracy with schedule-based gap filling
- Achieves best balance of accuracy and coverage
- Handles GPS dropouts gracefully using GTFS fallback
============================================================
```

## Interpreting Results

### Good vs Bad Results

| Metric | Good | Acceptable | Poor |
| --- | --- | --- | --- |
| RMSE | < 20m | 20-50m | > 50m |
| Coverage | > 95% | 80-95% | < 80% |
| Quality Score | > 0.7 | 0.5-0.7 | < 0.5 |
| P95 Error | < 50m | 50-100m | > 100m |

### What Affects Results?

| Factor | Impact on RMSE |
| --- | --- |
| GPS noise ↑ | RMSE ↑ |
| GPS dropout ↑ | Coverage ↓, RMSE ↑ |
| Road network quality | Affects map matching |
| Schedule adherence | Affects GTFS methods |

### Comparing Algorithms Fairly

1. **Same ground truth** - All algorithms process the same trips
2. **Same sensor data** - Same GPS noise, dropout patterns
3. **Same metrics** - Calculated identically for all
4. **Multiple runs** - Vary parameters to ensure robustness
