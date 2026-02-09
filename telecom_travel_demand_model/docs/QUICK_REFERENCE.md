# Quick Reference Guide

## Pipeline at a Glance

```
CDR/XDR Data → Preprocess → Stay Detection → Home/Work → Trips → Expand → OD Matrix
```

## Module Summary

| Module | File | Purpose |
|--------|------|---------|
| **Data Loading** | `telecom_loader.py` | Load CDR, XDR, 4G, 5G data |
| **Cell Towers** | `cell_tower_loader.py` | Load/infer cell locations |
| **Zones** | `zone_loader.py` | Define spatial zones (TAC/grid) |
| **Preprocessing** | `telecom_preprocessor.py` | Clean, merge, filter data |
| **User Filter** | `user_filter.py` | Filter low-quality users |
| **Stay Detection** | `stay_detector.py` | Find meaningful locations |
| **Home/Work** | `home_work_inference.py` | Infer home/work from patterns |
| **Trip Generation** | `trip_generator.py` | Extract trips from stays |
| **Trip Expansion** | `trip_expander.py` | Scale to population level |
| **OD Matrix** | `od_generator.py` | Generate OD matrices |

## Key Parameters

### Stay Detection
```python
StayPointDetector(
    distance_threshold=500,   # meters - max dist for same location
    time_threshold=1800,      # seconds - min time to be a stay (30 min)
    grid_cell_size=300,       # meters - consolidation grid
    min_visits=2              # minimum visits to keep
)
```

### Trip Expansion
```python
TripExpander(
    market_share=0.35,        # carrier market share (35%)
    expected_daily_trips=3.0  # NHTS average trips/person/day
)
```

### Home/Work Hours
```
Home hours: 8 PM - 7 AM (weekday nights)
Work hours: 7 AM - 8 PM (weekdays)
```

## Trip Purpose Codes

| Code | Meaning | Definition |
|------|---------|------------|
| HBW | Home-Based Work | home ↔ work |
| HBO | Home-Based Other | home ↔ other |
| NHB | Non-Home Based | other ↔ other |

## Time Periods

| Period | Hours |
|--------|-------|
| AM_PEAK | 6:00 - 9:00 |
| MIDDAY | 9:00 - 16:00 |
| PM_PEAK | 16:00 - 19:00 |
| EVENING | 19:00 - 22:00 |
| NIGHT | 22:00 - 6:00 |

## Expansion Formula

```
User Factor = expected_daily_trips / observed_daily_rate
Zone Factor = population / (observed_users / market_share)
Expansion Factor = User Factor × Zone Factor
```

## Expected Metrics

| Metric | Expected |
|--------|----------|
| Trip rate | 2.5-3.5 trips/person/day |
| Intra-zone | 20-40% of trips |
| Home detection | >90% of users |
| Work detection | 40-60% of employed users |

## Quick Start Code

```python
# Minimal example
from src.data_ingestion import TelecomDataLoader, CellTowerLoader, ZoneLoader
from src.preprocessing import TelecomPreprocessor
from src.stay_detection import StayPointDetector, HomeWorkInference
from src.trip_generation import TripGenerator, TripExpander
from src.od_matrix import ODMatrixGenerator

# Load
loader = TelecomDataLoader()
data = loader.load_all("data/")

# Preprocess
prep = TelecomPreprocessor()
clean = prep.process(cdr_df=data['cdr'], xdr_df=data['xdr'])

# Stay points
detector = StayPointDetector()
stays = detector.detect(clean)

# Home/work
hw = HomeWorkInference()
stays = hw.infer(stays, clean)

# Trips
gen = TripGenerator()
trips = gen.generate(stays, clean)

# Expand
exp = TripExpander(market_share=0.35)
trips = exp.expand(trips, prep.get_user_summary(clean))

# OD Matrix
od = ODMatrixGenerator()
matrix = od.generate(trips)
od.to_csv(matrix, "od_matrix.csv")
```

## File Structure

```
telecom_travel_demand_model/
├── src/
│   ├── data_ingestion/      # Data loading
│   ├── preprocessing/        # Data cleaning
│   ├── stay_detection/       # Location detection
│   ├── trip_generation/      # Trip extraction
│   ├── od_matrix/           # OD generation
│   ├── data_fusion/         # Multi-source fusion
│   ├── network_analysis/    # Route assignment
│   ├── visualization/       # Plotting
│   └── utils/               # Helpers
├── config/                  # Configuration files
├── examples/                # Example scripts
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── data/                    # Data directory
```
