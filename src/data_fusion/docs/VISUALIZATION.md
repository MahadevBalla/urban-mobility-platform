# Visualization Module

**Directory**: `visualization/`

## Overview

The visualization module provides tools for visual analysis and presentation of fusion results.

```
visualization/
├── trajectory_map.py      # Interactive Folium maps
├── accuracy_charts.py     # Plotly comparison charts
└── fusion_dashboard.py    # Streamlit interactive dashboard
```

---

## Trajectory Map

**File**: `trajectory_map.py`

### Purpose

Creates interactive maps showing ground truth vs reconstructed trajectories using Folium (Leaflet.js wrapper).

### Features

- **Multi-trajectory overlay**: Show all algorithms on one map
- **Color coding**: Different colors per algorithm
- **Point markers**: Click for details (time, speed, confidence)
- **Error indicators**: Lines showing position errors
- **Layer control**: Toggle algorithms on/off

### Visual Example

```
Map View
────────────────────────────────────────
┌─────────────────────────────────────┐
│  🟢 Ground Truth (green, thick)     │
│  🔵 GPS+OSM (blue)                  │
│  🟣 GTFS+OSM (purple)               │
│  🔴 GPS+GTFS+OSM (red)              │
│  🟠 CDR+OSM (orange)                │
│                                     │
│     🟢━━━━━━━━━━━━━━━━━🟢           │
│    🔵─ ─ ─ ─ ─ ─ ─ ─ ─🔵            │
│   🟣· · · · · · · · · ·🟣           │
│                                     │
│  [Layer Control] ☑ Ground Truth     │
│                  ☑ GPS+OSM          │
│                  ☐ GTFS+OSM         │
└─────────────────────────────────────┘
```

### Usage

```python
from src.data_fusion.visualization import TrajectoryMap

# Create map centered on trajectory
traj_map = TrajectoryMap(
    center=(19.07, 72.84),  # Mumbai
    zoom_start=14
)

# Create comparison map
map_obj = traj_map.create_comparison_map(
    ground_truth=gt_df,
    algorithm_results={
        'GPS+OSM': gps_osm_df,
        'GTFS+OSM': gtfs_osm_df,
        'GPS+GTFS+OSM': tri_source_df
    },
    show_points=True,    # Show individual points
    show_errors=True     # Show error lines
)

# Save to HTML
traj_map.save_map("trajectory_comparison.html")
```

### Additional Map Types

```python
# Heatmap of point density
heatmap = traj_map.create_heatmap(
    trajectory=recon_df,
    value_column='confidence',  # Intensity by confidence
    name="Confidence Heatmap"
)

# Error distribution heatmap
error_map = traj_map.create_error_heatmap(
    ground_truth=gt_df,
    reconstructed=recon_df,
    algo_name="GPS+OSM"
)
```

---

## Accuracy Charts

**File**: `accuracy_charts.py`

### Purpose

Creates interactive Plotly charts for comparing algorithm performance.

### Chart Types

#### 1. Bar Chart Comparison

```python
charts = AccuracyCharts()

fig = charts.create_comparison_bar_chart(
    comparison_df=comparison_df,
    metric='spatial_rmse_m',
    title='Spatial RMSE by Algorithm'
)
```

```
Spatial RMSE (m)
│
│  ████
│  ████  ████
│  ████  ████
│  ████  ████                    ████████
│  ████  ████                    ████████  ████████
│  ████  ████                    ████████  ████████
└──GPS+OSM──GPS+GTFS+OSM──────GTFS+OSM────CDR+OSM──
      32m        32m              205m        225m
```

#### 2. Radar Chart (Multi-Metric)

```python
fig = charts.create_multi_metric_radar(
    comparison_df=comparison_df,
    metrics=['spatial_rmse_m', 'coverage_rate', 'avg_confidence', 'quality_score']
)
```

```
            RMSE (inv)
               ▲
              /│\
             / │ \
            /  │  \
Coverage ◀─────●─────▶ Confidence
            \  │  /
             \ │ /
              \│/
               ▼
           Quality

Each algorithm draws a polygon
Larger area = better overall
```

#### 3. Sparsity Line Chart

```python
fig = charts.create_sparsity_line_chart(
    sparsity_df=sparsity_results
)
```

```
RMSE (m)
│
│                              CDR+OSM ●─────●
│                    GTFS+OSM ●────●─────●
│  GPS+OSM ●────●─────●─────●
│  GPS+GTFS+OSM ●────●─────●
│
└────10%────30%────50%────70%───── Data Dropout
```

#### 4. Quality Score Gauge

```python
fig = charts.create_quality_score_gauge(
    algorithm="GPS+GTFS+OSM",
    score=0.652
)
```

```
         GPS+GTFS+OSM
        Quality Score

     0      50      100
     │       │       │
     ├───────┼───────┤
     │░░░░░░░█████   │
     │  Poor │Good│  │
     └───────┴───────┘
           65.2
```

#### 5. Error Distribution Box Plot

```python
error_data = {
    'GPS+OSM': [list of errors],
    'GTFS+OSM': [list of errors],
    ...
}
fig = charts.create_error_distribution_box(error_data)
```

#### 6. Summary Dashboard

```python
# Multi-panel dashboard combining all charts
fig = charts.create_summary_dashboard(
    comparison_df=comparison_df,
    sparsity_df=sparsity_results
)
```

### Saving Charts

```python
charts.save_chart(fig, "chart.html", format='html')  # Interactive
charts.save_chart(fig, "chart.png", format='png')    # Static image
charts.save_chart(fig, "chart.svg", format='svg')    # Vector
charts.save_chart(fig, "chart.json", format='json')  # Data
```

---

## Fusion Dashboard

**File**: `fusion_dashboard.py`

### Purpose

Interactive Streamlit web application for running and visualizing the complete evaluation pipeline.

### Features

| Feature | Description |
|---------|-------------|
| **Configuration Sidebar** | Adjust parameters with sliders |
| **Live Evaluation** | Run algorithms with one click |
| **Comparison Tab** | View metrics table |
| **Map Tab** | Interactive trajectory map |
| **Charts Tab** | Visual comparisons |
| **Report Tab** | Download reports |

### Running the Dashboard

```bash
# From project root
python src/data_fusion/run_evaluation.py --dashboard

# Or directly
streamlit run src/data_fusion/visualization/fusion_dashboard.py
```

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  🚌 Data Fusion Algorithm Evaluation                     │
├──────────────┬──────────────────────────────────────────┤
│ Configuration│                                          │
│              │  [Comparison] [Map View] [Charts] [Report]│
│ Data Settings│  ─────────────────────────────────────── │
│ ─────────────│                                          │
│ Trips: [3]   │  Algorithm Comparison                    │
│ Stops: [8]   │                                          │
│ Points/s:[1] │  Best: GPS+GTFS+OSM                      │
│              │  Quality: 0.652  RMSE: 32.30m            │
│ Sensor Noise │                                          │
│ ─────────────│  ┌──────────────────────────────────┐   │
│ GPS noise:[8]│  │ Algorithm   │ RMSE │ Coverage    │   │
│ Dropout:[0.1]│  ├─────────────┼──────┼─────────────┤   │
│              │  │ GPS+GTFS+OSM│ 32.30│ 100%        │   │
│ Algorithms   │  │ GPS+OSM     │ 32.54│ 100%        │   │
│ ─────────────│  │ GTFS+OSM    │205.20│ 75%         │   │
│ ☑ GPS+OSM    │  │ CDR+OSM     │225.36│ 92%         │   │
│ ☑ GTFS+OSM   │  └──────────────────────────────────┘   │
│ ☑ GPS+GTFS   │                                          │
│ ☑ CDR+OSM    │                                          │
│              │                                          │
│ [Run Eval]   │                                          │
└──────────────┴──────────────────────────────────────────┘
```

### Tabs Overview

#### Comparison Tab
- Summary metrics (Best algorithm, quality score, RMSE, coverage)
- Detailed metrics table with highlighting
- All algorithms compared side-by-side

#### Map View Tab
- Interactive Folium map
- Ground truth in green
- Each algorithm in different color
- Click points for details

#### Charts Tab
- RMSE bar chart
- Coverage bar chart
- Multi-metric radar chart

#### Report Tab
- Text summary preview
- Download buttons:
  - 📥 Text Report (.txt)
  - 📥 Markdown Report (.md)
  - 📥 JSON Data (.json)

### Customization

```python
# In fusion_dashboard.py, modify sidebar options:

# Data Settings
num_trips = st.slider("Number of trips", 1, 10, 3)
num_stops = st.slider("Stops per trip", 3, 15, 8)

# Sensor Noise
gps_noise_m = st.slider("GPS noise (m)", 0, 30, 8)
gps_dropout_rate = st.slider("GPS dropout rate", 0.0, 0.5, 0.1)
```

---

## Color Scheme

Consistent colors across all visualizations:

| Algorithm | Color | Hex |
|-----------|-------|-----|
| Ground Truth | Green | `#2ECC71` |
| GPS+OSM | Blue | `#3498DB` |
| GTFS+OSM | Purple | `#9B59B6` |
| GPS+GTFS+OSM | Red | `#E74C3C` |
| CDR+OSM | Orange | `#F39C12` |

---

## Output Files

### From Dashboard/Runner

| File | Description |
|------|-------------|
| `comparison_dashboard.html` | Interactive Plotly multi-chart |
| `trajectory_map.html` | Folium map (if saved) |

### Generated Charts

All charts can be exported as:
- `.html` - Interactive (recommended)
- `.png` - Static image
- `.svg` - Vector graphic
- `.json` - Raw Plotly data
