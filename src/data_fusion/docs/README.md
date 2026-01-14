# Data Fusion Documentation

## Table of Contents

| Document | Description |
|----------|-------------|
| [OVERVIEW.md](OVERVIEW.md) | High-level architecture and methodology |
| [GROUND_TRUTH_GENERATOR.md](GROUND_TRUTH_GENERATOR.md) | Creating perfect test trajectories |
| [SENSOR_SIMULATOR.md](SENSOR_SIMULATOR.md) | Simulating GPS, GTFS, and CDR data |
| [FUSION_ALGORITHMS.md](FUSION_ALGORITHMS.md) | All four fusion approaches explained |
| [EVALUATION_METRICS.md](EVALUATION_METRICS.md) | Metrics, comparator, and reports |
| [VISUALIZATION.md](VISUALIZATION.md) | Maps, charts, and dashboard |
| [RUN_EVALUATION.md](RUN_EVALUATION.md) | Running the complete pipeline |

## Quick Reference

### Running the Evaluation

```bash
# Command line
python src/data_fusion/run_evaluation.py

# Interactive dashboard
python src/data_fusion/run_evaluation.py --dashboard
```

### Algorithm Comparison

| Algorithm | RMSE | Best For |
|-----------|------|----------|
| GPS+GTFS+OSM ⭐ | ~32m | Transit with GPS gaps |
| GPS+OSM | ~33m | Reliable GPS data |
| GTFS+OSM | ~205m | No real-time data |
| CDR+OSM | ~225m | No GPS available |

### Key Files

```
data_fusion/
├── run_evaluation.py          # ← Start here
├── ground_truth_generator.py  # Perfect trajectories
├── sensor_simulator.py        # Simulated sensor data
├── fusion_algorithms/         # 4 fusion methods
├── evaluation/                # Metrics & reports
└── visualization/             # Charts & dashboard
```

## Reading Order

1. **[OVERVIEW.md](OVERVIEW.md)** - Understand the problem and solution
2. **[GROUND_TRUTH_GENERATOR.md](GROUND_TRUTH_GENERATOR.md)** - How test data is created
3. **[SENSOR_SIMULATOR.md](SENSOR_SIMULATOR.md)** - How sensor errors are simulated
4. **[FUSION_ALGORITHMS.md](FUSION_ALGORITHMS.md)** - How each algorithm works
5. **[EVALUATION_METRICS.md](EVALUATION_METRICS.md)** - How we measure success
6. **[VISUALIZATION.md](VISUALIZATION.md)** - How to visualize results
7. **[RUN_EVALUATION.md](RUN_EVALUATION.md)** - How to run everything
