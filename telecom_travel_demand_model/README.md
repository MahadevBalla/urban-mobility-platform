# Telecom-Based Travel Demand Estimation Model

A comprehensive Python framework for estimating travel demand using multi-source telecom data (CDR, 4G LTE, 5G NR, XDR) with data fusion capabilities. Designed for Indian telecom data context.

## Overview

This model implements and extends the methodology from Toole et al. (2015) "The path most traveled: Travel demand estimation using big data resources" with enhancements for modern telecom data including 4G/5G network metrics and extended data records.

## Key Features

- **Multi-Source Data Fusion**: Combines CDR, 4G, 5G, and XDR data for improved accuracy
- **Stay Point Detection**: Spatiotemporal clustering to identify meaningful locations
- **Home/Work Inference**: Temporal pattern analysis for location classification
- **Trip Generation**: Extract trips with purpose classification (HBW, HBO, NHB)
- **OD Matrix Creation**: Generate origin-destination matrices at configurable spatial granularity
- **Trip Expansion**: Scale observed trips using census/population data
- **Network Quality Integration**: Use signal quality metrics for location confidence weighting
- **Modular Architecture**: Easy to extend and customize

## Project Structure

```
telecom_travel_demand_model/
├── src/
│   ├── data_ingestion/       # Data loaders for various telecom formats
│   ├── preprocessing/        # Data cleaning, standardization, filtering
│   ├── stay_detection/       # Stay point extraction algorithms
│   ├── trip_generation/      # Trip inference and classification
│   ├── od_matrix/            # OD matrix generation and expansion
│   ├── data_fusion/          # Multi-source data fusion logic
│   ├── network_analysis/     # Road network and assignment (future)
│   ├── visualization/        # Plotting and mapping utilities
│   └── utils/                # Common utilities and helpers
├── tests/                    # Unit and integration tests
├── config/                   # Configuration files
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks for exploration
└── data/
    ├── raw/                  # Raw input data
    ├── processed/            # Intermediate processed data
    └── outputs/              # Final outputs (OD matrices, etc.)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_ingestion import TelecomDataLoader
from src.preprocessing import TelecomPreprocessor
from src.stay_detection import StayPointDetector
from src.trip_generation import TripGenerator
from src.od_matrix import ODMatrixGenerator

# Load data
loader = TelecomDataLoader(config_path="config/config.yaml")
cdr_data = loader.load_cdr("data/raw/cdr_data.csv")
xdr_data = loader.load_xdr("data/raw/xdr_data.csv")

# Preprocess
preprocessor = TelecomPreprocessor()
clean_data = preprocessor.process(cdr_data, xdr_data)

# Detect stay points
detector = StayPointDetector(distance_threshold=500, time_threshold=1800)
stay_points = detector.detect(clean_data)

# Infer home/work locations
stay_points = detector.infer_home_work(stay_points)

# Generate trips
trip_gen = TripGenerator()
trips = trip_gen.generate(stay_points)

# Create OD matrix
od_gen = ODMatrixGenerator(zone_definition="tac")
od_matrix = od_gen.generate(trips)
```

## Configuration

See `config/config.yaml` for all configurable parameters including:
- Stay detection thresholds
- Home/work inference time windows
- Trip filtering criteria
- Expansion factors
- Zone definitions

## Data Requirements

### CDR Data
- IMSI/MSISDN (subscriber identifier)
- Timestamp
- Cell ID / LAC / TAC
- Call type (optional)

### XDR Data (Optional but recommended)
- Location coordinates (lat/lon)
- Session information
- RAT type (LTE/NR)

### 4G/5G Network Data (Optional)
- Cell locations
- Signal quality metrics (RSRP, RSRQ, SINR)

### Zone Definition
- TAC boundaries or custom zone polygons
- Census/population data for expansion

## Documentation

Detailed documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [SYSTEM_DOCUMENTATION.md](docs/SYSTEM_DOCUMENTATION.md) | Complete system architecture and methodology |
| [MODULE_GUIDE.md](docs/MODULE_GUIDE.md) | Detailed module-by-module documentation |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Quick reference card for common tasks |

## Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  CDR / XDR   │───▶│ Preprocess   │───▶│    Stay      │───▶│  Home/Work   │
│  4G / 5G     │    │  & Clean     │    │  Detection   │    │  Inference   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐            │
                    │  OD Matrix   │◀───│    Trip      │◀───────────┘
                    │  Generation  │    │  Expansion   │
                    └──────────────┘    └──────────────┘
```

## Key Algorithms

### Stay Point Detection (Zheng-Xie)
- Groups consecutive observations within distance threshold
- Requires minimum time threshold to qualify as stay
- Uses signal-quality weighted centroids
- Progressive threshold relaxation for sparse data

### Home/Work Inference (Alexander et al.)
- Home: Highest score for weekday nights (8PM-7AM) + weekend presence
- Work: Non-home location with regular weekday visits (7AM-8PM)

### Trip Expansion (Toole et al.)
- User-level: `expected_daily_trips / observed_daily_rate`
- Population-level: `zone_population / (observed_users / market_share)`

### Intra-Zone Estimation
- Adjusts OD matrix diagonal for under-counted short trips
- Typical target: 20-40% of total trips

## Expected Metrics

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Daily trip rate | 2.5 - 3.5 trips/person | NHTS |
| Intra-zone trips | 20% - 40% of total | Toole et al. |
| Home detection | > 90% of users | Alexander et al. |

## References

1. Toole, J.L., et al. (2015). "The path most traveled: Travel demand estimation using big data resources." Transportation Research Part C, 58, 162-177.

2. Alexander, L.P., et al. (2015). "Origin-destination trips by purpose and time of day inferred from mobile phone data." Transportation Research Part C.

3. Zheng, Y., & Xie, X. (2011). "Learning travel recommendations from user-generated GPS traces." ACM TIST.

4. Calabrese, F., et al. (2011). "Estimating Origin-Destination flows using mobile phone location data." IEEE Pervasive Computing.

## License

MIT License
