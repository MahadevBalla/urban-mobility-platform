"""
config.py — Single source of truth for all pipeline parameters.

Every value here is justified by EDA findings in:
  notebooks/01_reference_data_audit.ipynb
  notebooks/02_gps_data_audit.ipynb

Import in every script:
    from config import GPS_FILES, STUDY_START, STUDY_END, SNAP_THRESHOLD_M, ...
"""

from pathlib import Path
import polars as pl

# Repository root
ROOT = Path(__file__).resolve().parent.parent

# Directory layout
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"  # scratch parquet between steps
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"
TABLES_DIR = OUTPUTS / "tables"

for _d in [DATA_PROCESSED, DATA_INTERIM, FIGURES, MODELS_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# GPS raw files — 14-column legacy CSV, NO header
GPS_FILES = [
    DATA_RAW / "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756",
    DATA_RAW / "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756_part2",
    DATA_RAW / "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756_part3",
]

# Reference data
STOPS_FILE = DATA_PROCESSED / "stops_clean.csv"
TRIPS_FILE = DATA_PROCESSED / "trips_clean.csv"
WEATHER_DIR = DATA_RAW / "weather"  # G001/ … G015/ subdirectories
WARD_KML = DATA_RAW / "mumbai_wards.kml"  # if available

# Study window
STUDY_START = "2021-09-01"
STUDY_END = "2022-10-22"

# GPS schema
LEGACY_COLS = [
    "id",  # pos  1  unique int, sequential
    "lat",  # pos  2  WGS84 latitude
    "lng",  # pos  3  WGS84 longitude
    "created",  # pos  4  DB ingestion timestamp UTC  (dropped after parsing)
    "vehicle_id",  # pos  5  internal vehicle ID
    "timestamp",  # pos  6  actual GPS event timestamp UTC  ← parsed to datetime and used for all temporal filters
    "speed",  # pos  7  speed in km/h
    "bearing",  # pos  8  heading/bearing field; Direction of travel
    "source",  # pos  9  GPS source
    "c10",  # pos 10  ALWAYS NULL → dropped
    "c11",  # pos 11  undocumented low-cardinality field; excluded from pipeline
    "deviation_s",  # pos 12  GPS vs DB timestamp delta, seconds
    "c13",  # pos 13  23.1% null, undocumented → dropped
    "c14",  # pos 14  near-constant "1" → dropped
]

LEGACY_DTYPES = {
    "id": pl.Int64,
    "lat": pl.Float64,
    "lng": pl.Float64,
    "created": pl.Utf8,
    "vehicle_id": pl.Int64,
    "timestamp": pl.Utf8,
    "speed": pl.Float64,
    "bearing": pl.Float64,
    "source": pl.Int32,
    "c10": pl.Utf8,
    "c11": pl.Utf8,
    "deviation_s": pl.Float64,
    "c13": pl.Utf8,
    "c14": pl.Utf8,
}

# Columns to drop immediately after read
LEGACY_DROP_COLS = ["created", "c10", "c11", "c13", "c14", "bearing"]

# GPS quality filters
# Tight Mumbai metropolitan bbox
MUMBAI_BBOX = {
    "lat_min": 18.8894,
    "lat_max": 19.3274,
    "lng_min": 72.7692,
    "lng_max": 73.1165,
}
MUMBAI_BUFFER_DEG = 0.15

# Speed: null-out anything > 120 km/h before ANY numeric filter
# Sentinels observed during EDA: 602, 699.999, 800, 1000, 1200
SPEED_MAX_KMH = 120

# GPS jump filter: same threshold as sentinel — inter-ping computed speed > 120
GPS_JUMP_MAX_KMH = 120

# Temporal deviation: GPS event time vs DB ingestion time
# |deviation| > 300s removes only 0.09% of rows
DEVIATION_MAX_S = 300

# Trip segmentation
# 20-min gap threshold: only 0.1% of transitions exceed this
GAP_THRESHOLD_MIN = 20
MIN_PINGS_PER_SEG = 5
MIN_DURATION_MIN = 5

# Stop snapping
# 68% of pings within 200m, 82% within 300m <- from 02_gps_data_audit.ipynb EDA
SNAP_THRESHOLD_M = 200
EARTH_R_M = 6_371_000.0  # for haversine distance calculations

# Route inference
ROUTE_MIN_OBS_STOPS = 4
ROUTE_MIN_CONFIDENCE = 0.20  # minimum combined score to assign template
ROUTE_HIGH_CONFIDENCE = 0.50  # threshold for schedule adherence use
ROUTE_TRIP_WINDOW_MIN = 45  # max delta between seg start and scheduled trip start
DEFAULT_MIN_SHARED_STOPS = 2  # minimum shared stops for candidate templates in route inference and OD matching
DEFAULT_TOP_N_CANDIDATES = 15  # max candidate templates to consider in route inference and OD matching (after shared stop filter)
DEFAULT_TRIP_ASSIGN_MIN_CONF = 0.60  # minimum confidence to assign a route template to a trip
DEFAULT_TRIP_ASSIGN_MIN_OVERLAP = 0.60  # minimum overlap (shared stops / template stops) to assign a route template to a trip
DEFAULT_VALIDATION_RANDOM_SEED = 42  # for reproducibility of train/val splits and random sampling in validation analyses

# OD matrix
OD_TIER1_MIN_CONF = 0.30  # route-template OD minimum confidence
OD_TIME_BIN_MINUTES = 30
OD_MIN_DURATION_MIN = 2
OD_MIN_PINGS = 3

# Reliability
HEADWAY_MAX_MIN = 120  # exclude overnight gaps
ON_TIME_WINDOW_MIN = 3  # ±3 min = on-time
LATE_THRESHOLD_MIN = 5
EARLY_THRESHOLD_MIN = -2
BUNCHING_PCT = 0.25  # headway < 25% of mean = bunching

# Weather (Open-Meteo 10km grid)
WEATHER_IDW_K = 4  # k nearest grid points for interpolation
WEATHER_IDW_POWER = 2  # inverse-distance weighting exponent
WEATHER_JOIN_TOL_MIN = 60  # temporal join tolerance (minutes)
WEATHER_TRANSPORT_VARS = [
    "precipitation",
    "rain",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
    "weather_code",
    "vapour_pressure_deficit",
    "shortwave_radiation",
    "sunshine_duration",
    "soil_moisture_0_to_7cm",  # monsoon waterlogging proxy
]

# H3
H3_RESOLUTION = 8

# Models
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # fast on CPU and GPU
    "device": "cuda",  # switch to "cpu" if no GPU
    "random_state": 42,
}

STGNN_PARAMS = {
    "hidden_dim": 64,
    "n_layers": 2,
    "seq_len": 12,  # 12 × 30min = 6h lookback
    "pred_horizon": 4,  # predict next 4 × 30min
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "dropout": 0.1,
    "weight_decay": 1e-4,
}

# Processed artefact paths (outputs of each pipeline script)
PINGS_CLEAN = DATA_PROCESSED / "pings_clean.parquet"
PINGS_SEGMENTED = DATA_PROCESSED / "pings_segmented.parquet"
PINGS_SNAPPED = DATA_PROCESSED / "pings_snapped.parquet"
ROUTE_CATALOG = DATA_PROCESSED / "route_catalog.parquet"
ROUTE_CATALOG_JSON = DATA_PROCESSED / "route_catalog.json"
SEGMENTS_INFERRED = DATA_PROCESSED / "segments_inferred.parquet"
OD_TIER1 = DATA_PROCESSED / "od_tier1.parquet"
OD_TIER2 = DATA_PROCESSED / "od_tier2.parquet"
OD_AGG = DATA_PROCESSED / "od_agg.parquet"
LOAD_FACTOR = DATA_PROCESSED / "load_factor.parquet"
HEADWAY_STATS = DATA_PROCESSED / "headway_stats.parquet"
SCHED_ADHERENCE = DATA_PROCESSED / "schedule_adherence_stats.parquet"
STOP_VISITS = DATA_PROCESSED / "stop_visits.parquet"
WEATHER_MASTER = DATA_PROCESSED / "weather_grid_master.parquet"
WEATHER_STOPS = DATA_PROCESSED / "weather_stop_hourly.parquet"
FEATURES_MASTER = DATA_PROCESSED / "features_master.parquet"
WARD_OD = DATA_PROCESSED / "ward_od.parquet"
MODE_SHIFT = DATA_PROCESSED / "mode_shift_scores.parquet"
CO2_SAVINGS = DATA_PROCESSED / "co2_savings.parquet"
SERVICE_GAPS = DATA_PROCESSED / "service_gaps.parquet"
SERVICE_SUPPLY = DATA_PROCESSED / "service_supply.parquet"
SERVICE_FREQUENCY = DATA_PROCESSED / "service_frequency.parquet"
