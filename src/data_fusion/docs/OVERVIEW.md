# Data Fusion Module - Overview

## Purpose

This module provides a complete framework for evaluating different **multi-source data fusion** approaches for reconstructing vehicle trajectories. It answers the question: *"Which combination of data sources gives the most accurate vehicle position reconstruction?"*

## The Problem

Transit agencies have access to multiple data sources, each with different characteristics:

| Data Source | Accuracy | Frequency | Coverage | Cost |
|-------------|----------|-----------|----------|------|
| GPS Probes | ~8-15m | 1-5 sec | Variable (dropout) | Medium |
| GTFS Schedule | Stop-level only | Per-stop | 100% at stops | Free |
| Cell Tower (CDR) | ~200-500m | Sparse events | Good | Low |
| Road Network (OSM) | High | Static | Complete | Free |

**Challenge**: How do we combine these to get the best trajectory reconstruction?

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FUSION EVALUATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │Ground Truth  │───▶│   Sensor     │───▶│   Fusion     │       │
│  │ Generator    │    │  Simulator   │    │  Algorithms  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │ Perfect           │ Degraded          │ Reconstructed  │
│         │ Trajectory        │ Data              │ Trajectory     │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────┐                              │
│                    │  Evaluation  │                              │
│                    │   Metrics    │                              │
│                    └──────────────┘                              │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────┐                              │
│                    │    Report    │                              │
│                    │  Generation  │                              │
│                    └──────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Evaluation Methodology

### Controlled Comparison Approach

1. **Generate Ground Truth**: Create perfect vehicle trajectories with exact positions every second
2. **Simulate Sensors**: Degrade the perfect data to mimic real sensor characteristics
3. **Apply Fusion**: Run each fusion algorithm on the degraded data
4. **Compare to Truth**: Calculate error metrics against the ground truth

This approach ensures a fair comparison because:
- All algorithms process the **same underlying trip**
- Sensor noise is **controlled and reproducible**
- We know the **exact correct answer**

## Fusion Algorithms Compared

### 1. GPS + OSM (Baseline)
- Map-matches GPS points to road network
- Interpolates between GPS samples
- **Strength**: Real-time accuracy when GPS available
- **Weakness**: Fails during GPS dropout

### 2. GTFS + OSM (Schedule-Based)
- Uses transit schedule for stop arrival times
- Interpolates positions between stops
- **Strength**: Works without real-time data
- **Weakness**: Assumes schedule adherence

### 3. GPS + GTFS + OSM (Tri-Source) ⭐ Recommended
- Combines GPS with schedule information
- Uses GTFS to fill GPS gaps
- **Strength**: Best of both worlds
- **Weakness**: More complex

### 4. CDR + OSM (Low-Precision)
- Uses cell tower connections
- Very coarse positioning (~200-500m)
- **Strength**: Works with basic mobile data
- **Weakness**: Low accuracy

## Directory Structure

```
data_fusion/
├── docs/                      # Documentation (you are here)
│   ├── OVERVIEW.md
│   ├── GROUND_TRUTH_GENERATOR.md
│   ├── SENSOR_SIMULATOR.md
│   ├── FUSION_ALGORITHMS.md
│   ├── EVALUATION_METRICS.md
│   └── VISUALIZATION.md
│
├── fusion_algorithms/         # Core fusion implementations
│   ├── base_fusion.py         # Abstract base class
│   ├── gps_osm_fusion.py      # GPS + OSM
│   ├── gtfs_osm_fusion.py     # GTFS + OSM
│   ├── gps_gtfs_osm_fusion.py # Tri-source fusion
│   └── cdr_osm_fusion.py      # CDR + OSM
│
├── evaluation/                # Metrics and comparison
│   ├── metrics.py             # RMSE, MAE, coverage
│   ├── comparator.py          # Run all algorithms
│   └── report_generator.py    # Generate reports
│
├── visualization/             # Charts and dashboard
│   ├── trajectory_map.py      # Folium maps
│   ├── accuracy_charts.py     # Plotly charts
│   └── fusion_dashboard.py    # Streamlit dashboard
│
├── ground_truth_generator.py  # Generate perfect trajectories
├── sensor_simulator.py        # Simulate sensor data
├── run_evaluation.py          # Main entry point
└── EVALUATION_FRAMEWORK.md    # Methodology documentation
```

## Quick Start

### Command Line
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows

# Run evaluation
python src/data_fusion/run_evaluation.py

# Launch interactive dashboard
python src/data_fusion/run_evaluation.py --dashboard
```

### Output Files
- `ground_truth.csv` - Perfect trajectory (the answer key)
- `reconstructed_*.csv` - Each algorithm's output
- `fusion_evaluation_*.md` - Comparison report
- `comparison_dashboard.html` - Interactive charts

## Key Metrics

| Metric | What it Measures | Good Value |
|--------|------------------|------------|
| Spatial RMSE | Position error (meters) | < 50m |
| Coverage Rate | % of ground truth matched | > 90% |
| Quality Score | Composite score (0-1) | > 0.6 |
| P95 Error | 95th percentile error | < 100m |

## Expected Results

With default settings (8m GPS noise, 10% dropout):

| Algorithm | RMSE | Coverage | Quality |
|-----------|------|----------|---------|
| GPS+GTFS+OSM | ~32m | 100% | 0.65 |
| GPS+OSM | ~33m | 100% | 0.65 |
| GTFS+OSM | ~205m | 75% | 0.40 |
| CDR+OSM | ~225m | 92% | 0.36 |

**Recommendation**: GPS+GTFS+OSM provides the best balance of accuracy and robustness.
