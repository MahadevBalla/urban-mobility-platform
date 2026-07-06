# Methodology — Cityflo Mobility Demand Pipeline

This document covers the pipeline architecture, per-script design decisions, configuration parameters, and methodological limitations. The companion document `dataset.md` covers raw data schemas and EDA findings.

## Pipeline Overview

The pipeline takes raw GPS pings (from either the legacy or current-format export) and reference schedule data, and produces a model-ready feature table alongside OD matrices, reliability statistics, ward-level aggregations, model predictions, evaluation artefacts, and policy-facing outputs.

```md
Raw GPS CSVs                        Raw GPS CSVs
(legacy, 14-col, no header)         (current format, header on first file per group)
        │                                   │
        ▼ 01_1_ingest_legacy                ▼ 01_1_ingest_current
        │                                   │
        └───────────────────┬───────────────┘
                            ▼ 01_2_finalize_pings  →  01_3_merge_buckets
                      pings_clean.parquet
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼ 02_route_catalog                    │
  route_catalog.parquet                       │
        │                                     │
        ▼ 03_trip_segmentation                │
  pings_segmented.parquet                     │
        │                                     │
        ▼ 04_stop_snapping                    │
  pings_snapped.parquet                       │
        │                                     │
        ▼ 05_route_inference ─────────────────┘
  segments_inferred.parquet
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼ 06_od_matrix                             ▼ 07_reliability
  od_agg.parquet                             headway_stats.parquet
  service_supply.parquet                     schedule_adherence_stats.parquet
  service_frequency.parquet                  stop_visits.parquet
        │
        ▼ 08_weather_consolidate
  weather_stop_hourly.parquet
        │
        ▼ 09_feature_engineering
  features_master.parquet
        │
        ├───────────────────────┬──────────────────────────────┐
        │                       │                              │
        ▼ 10_ward_aggregation   ▼ 11_model_nb                  ▼ 12_model_xgboost
  ward_od.parquet              nb_predictions.parquet        xgboost_predictions.parquet
                               nb_metrics.csv                xgboost_metrics.csv
        │
        └───────────────────────┬──────────────────────────────┘
                                │
                                ▼ 13_model_stgnn
                        stgnn_predictions.parquet
                        stgnn_metrics.csv
                                │
                                ▼ 14_analysis_reporting
                        figures/, model_comparison.csv,
                        network_summary.json
                                │
                                ▼ 15_policy_outputs
                        mode_shift_scores.parquet
                        co2_savings.parquet
                        service_gaps.parquet
```

Scripts 02 and 03–05 can run after `pings_clean.parquet` is ready. Scripts 06 and 07 can then run independently. The model scripts 11, 12, and 13 can also run independently once `features_master.parquet` exists.

All parameters are centralised in `scripts/config.py`. Scripts import from config rather than hardcoding any threshold.

## Script Design

### 01 — GPS Ingestion (two source formats, four scripts)

Ingestion now draws from two GPS export formats that differ in source table, file layout, and schema; both are merged into a single cleaned ping stream before any downstream stage runs.

**01_1_ingest_legacy.py** reads one legacy GPS file for one bucket (`vehicle_id % bucket_count` partitioning, as before), applies all quality filters in order (timestamp validity → study window → IST conversion → coordinate validity + bbox → deviation filter → speed sentinel null-out → column drops → temporal derived columns), and writes intermediate parquet files. Deduplication and the GPS jump filter are deferred, since both require the full sorted vehicle history across all input files.

**01_1_ingest_current.py** reads the current-format GPS export (`CURRENT_DATA_DIR`, October 2024 – March 2026), which uses a different source table (`vehiclelocationlog`) and a different file layout: a header row appears only on the first file of each month/half-month group, with header-less continuation files (`_part2`, `_part3`, ...) for exports too large for one file. The script reads each group's header from its primary file rather than assuming a fixed column order, and reuses it for that group's continuation files. `ride_date` as provided at source is discarded and regenerated from `timestamp` after IST conversion, so both formats derive this field identically rather than trusting two possibly-different upstream conventions. Columns not used downstream (`meta_data`, `created`, the raw `ride_date`) are dropped, and `deviation_in_seconds` is renamed to `deviation_s` to match the legacy field name, so both formats emit the same canonical schema. This script applies the same quality-filter logic as `01_1_ingest_legacy.py`; it accepts `--study_start` and `--study_end` arguments. These default to the legacy study window defined in `config.py`, and the script emits a warning if those defaults are used because they do not cover the October 2024–March 2026 dataset.

No EDA equivalent to `notebooks/02_gps_data_audit.ipynb` has been run on the current-format export yet. The legacy filter thresholds (`SPEED_MAX_KMH`, `GPS_JUMP_MAX_KMH`, `DEVIATION_MAX_S`, etc.) are applied to it as-is, on the assumption that GPS hardware behaviour is broadly similar across formats; this has not been independently confirmed, and filter yield/retention figures for the current-format data are not yet known.

**01_2_finalize_pings.py** now collects per-bucket intermediate parquet files from *both* ingestion scripts (`before*_bucket{N}.parquet` for legacy, `*.csv_bucket{N}.parquet` for current-format) before deduplicating on `(vehicle_id, ts_utc)` and applying the GPS jump filter. This script's output directory is now resolved from `config.DATA_PROCESSED` rather than a path hardcoded relative to the working directory, so it correctly follows the same root as every other script regardless of where it is launched from.

**01_3_merge_buckets.py** concatenates finalised bucket parquets into `pings_clean.parquet` using `scan_parquet` → `sink_parquet`, unchanged from before; since its input is already merged and deduplicated per bucket, it does not need to distinguish between formats.

### 02 — Route Catalog

Reads `trips_clean.csv` and extracts the set of unique route templates. A template is defined by its ordered stop sequence — multiple `trip_id` rows sharing the same stop order belong to the same template regardless of scheduled times or dates.

For each template, the catalog stores the median scheduled arrival time per stop across all runs of that template. The output `route_catalog.json` is a human-readable copy of the catalog for debugging and manual inspection, while the parquet is used downstream.

### 03 — Trip Segmentation

Assigns a `segment_id` to each ping within a vehicle trajectory. A new segment begins when the inter-ping gap exceeds `GAP_THRESHOLD_MIN`. Segmentation is partitioned by `vehicle_id` only, not by date, so trajectories are not artificially split at midnight.

After segmentation, segments with fewer than `MIN_PINGS_PER_SEG` pings or shorter than `MIN_DURATION_MIN` minutes are removed. These short fragments usually reflect incomplete GPS activity rather than service runs.

### 04 — Stop Snapping

Snaps each GPS ping to the nearest bus stop using a BallTree with Haversine metric. Pings within `SNAP_THRESHOLD_M` metres (200 m by default) receive a `snapped_stop_id`; pings beyond the threshold receive -1 and are retained for completeness.

Pings with null or non-finite coordinates are short-circuited before the BallTree query and assigned `-1` directly. This stage is a global nearest-stop snap, not yet route-constrained.

### 05 — Route Inference

Matches each GPS segment's observed stop sequence against the route catalog to assign a `template_id` and, where timing information is sufficient, attempt to resolve a corresponding `candidate_trip_id` for high-confidence template matches.

Candidate generation uses an inverted index (`stop_id → template_ids`) to avoid scoring all templates for every segment. Candidates are ranked by Longest Common Subsequence Similarity (LCSS) over the ordered stop-ID sequences, with coverage and a template-length penalty as tie-breakers; overlap, direction, and endpoint scores are retained as diagnostics only. The `match_margin` field stores the gap between the top two matches.

For segments where `match_confidence ≥ ROUTE_HIGH_CONFIDENCE`, the script selects a specific scheduled trip instance on the matched template by estimating a per-trip timing offset from shared timestamped stops and scoring candidates by the residual once that offset is removed, falling back to a start-time-only comparison when fewer than two timestamped stops are shared. The optional `--run_validation` mode hides part of a known route's stop sequence (independently at random, or as a contiguous block/prefix/suffix) and checks whether the correct template is recovered.

### 06 — OD Matrix

Constructs origin-destination pairs at the vehicle-run level from matched segments. All output counts represent vehicle runs, not individual passengers.

**Tier 1 OD** is used for segments with `match_confidence ≥ OD_TIER1_MIN_CONF`, where origin and destination are the terminal stops of the matched route template. **Tier 2 OD** falls back to the first and last snapped stop for unmatched or lower-confidence segments. Both tiers additionally require a minimum ping count and segment duration (`OD_MIN_PINGS`, `OD_MIN_DURATION_MIN`) to exclude degenerate near-instantaneous segments.

The final `od_agg.parquet` enriches OD pairs with stop metadata, coordinates, `trip_distance_km`, temporal fields, and segment-level summary measures. The script also writes `service_supply.parquet` and `service_frequency.parquet` as derived supply-side tables.

### 07 — Reliability

This script produces two families of service quality metrics.

**Headway statistics** collapse contiguous pings from the same vehicle at the same stop into a single stop-arrival event, then compute headway distributions by stop, month, day type, monsoon flag, and period. It outputs metrics including mean headway, headway CV, an uncapped headway reliability score, and bunching counts; a (stop, day type, period, month, monsoon) stratum is only reported once it has at least 10 headway observations. The reliability score itself is not clipped to `[0,1]` at this stage — it can go negative at heavily-bunched stops — and is instead clipped where it is consumed downstream (script 15).

**Schedule adherence** compares inferred stop arrivals against scheduled stop times for high-confidence route matches with an assigned `candidate_trip_id`. Delay is computed in minutes, and observations with extremely large absolute delay are excluded as likely trip-ID mismatches.

### 08 — Weather Consolidation

Loads all half-month weather CSVs per grid point, merges hourly variable groups on `time`, and concatenates all 15 grid points into a master grid-level parquet.

Stop-level weather is interpolated from the weather grid using inverse-distance weighting with the `WEATHER_IDW_K` nearest grid points and exponent `WEATHER_IDW_POWER`. Weights are normalised at each timestamp over whichever neighbours have valid observations. A self-test comparing the vectorised interpolation against a naive per-cell reference on a random subsample is available via `--self-test`. Derived transport-facing weather features are then added, including rolling precipitation totals, `log_precip`, rain flags, heat stress, weather severity, soil saturation indicators, and strong-wind flags.

### 09 — Feature Engineering

Assembles the model-ready feature table from OD, reliability, weather, and spatial features.

The script keeps all OD-level fields from `od_agg.parquet` via `od.*`, then adds temporal features (`hour`, `minute_of_hour`, `dow`, `day_of_year`, cyclical encodings, weekend, monsoon-season indicators, and `is_peak`). Reliability fields from script 07 are joined at the origin stop level. The weather join is implemented natively in DuckDB, selecting whichever of the floor-hour or ceiling-hour observation is nearer to each 30-minute bin; this is only correct if the weather grid is exactly hourly, an assumption checked with an explicit assertion rather than presumed silently.

Lag and rolling statistics are computed in DuckDB over `(origin_stop_id, dest_stop_id)` ordered by `time_bin_30min`. Because `od_agg.parquet` has one row per OD pair only for bins where a run was actually observed (no zero-filling of unserved bins), these lag/rolling features are computed over the sequence of *observed* bins for that OD pair rather than a fixed wall-clock interval — "previous bin" means the previous observed record, and "same hour yesterday" or "trailing 24-hour mean" can span more than their nominal interval for sparsely-served OD pairs. Spatially, the script assigns `origin_h3` and `dest_h3`, and computes `dist_cbd_km` as the Haversine distance from the origin stop to Nariman Point.

A deliberate design choice in the current version is that train/validation/test split logic is not embedded in this script. The feature table is modelling-agnostic, and split boundaries are applied later inside the model scripts.

### 10 — Ward Aggregation

Assigns each stop to a Mumbai ward using a point-in-polygon join against the ward KML boundaries. Stops that fall outside all ward polygons are assigned to the nearest ward centroid as a fallback.

Ward-level OD is then aggregated from `od_agg.parquet` by summing `trip_count` across all OD pairs whose origin and destination stops belong to the corresponding wards.

### 11 — Negative Binomial Model

Fits a count-regression baseline using `statsmodels.GLM` with a Negative Binomial family. The dispersion parameter (α) is estimated from the training fold via a method-of-moments calculation on the variance-to-mean relationship, rather than left at a library default, before being passed into the fitted model. The model uses a temporal train/validation/test split defined in `config.py`, and evaluates forecast quality on held-out data using MAE, RMSE, sMAPE, R², and Pearson correlation, against a lag-1 persistence baseline and a training-fold historical-mean baseline.

This script serves as the interpretable count-model baseline against which the more flexible machine learning models are compared.

### 12 — XGBoost Model

Trains a gradient-boosted tree regressor using the `XGB_PARAMS` configuration block, with a required/optional feature split so the script degrades gracefully if some optional features (mostly weather-derived) are absent. `TimeSeriesSplit` cross-validation is used to assess the robustness of this fixed configuration, not to select or tune it; the reported model is trained once on the fixed configuration, with a temporally-held-out (not randomly sampled) slice used for early stopping. The script stores prediction tables, model artefacts, metrics, and SHAP-based interpretability outputs.

Compared with the Negative Binomial baseline, this model is intended to capture nonlinear interactions between temporal, weather, reliability, and spatial features.

### 13 — ST-GNN Model

Trains a spatio-temporal graph neural network for demand forecasting on the H3 graph.

The graph is built from active `origin_h3` cells, with edges between cells sharing a k-ring=1 neighbourhood. Edge weights use a Gaussian kernel of Haversine distance between H3 centroids, followed by symmetric degree normalisation.

On the temporal side, the model forms sequences of length `seq_len` over the full time index, then splits sequences by target timestamp into train, validation, and test, so every validation and test sequence retains full historical context rather than restarting a warm-up window per fold. The architecture combines two GCN layers for spatial message passing with multi-head self-attention over the time axis, and is trained with Huber loss plus early stopping.

### 14 — Analysis and Reporting

Generates post-pipeline visualisations and summary tables from the processed data and available model outputs.

This script produces EDA dashboards, vehicle trajectory maps, OD heatmaps, origin/destination H3 choropleths, peak-vs-offpeak spatial flow maps, per-model evaluation figures, `model_comparison.csv`, and `network_summary.json`. It auto-detects which prediction files exist and only generates dashboards for those models.

### 15 — Policy Outputs

Transforms modelling and operations outputs into policy-facing summary artefacts.

The script writes `mode_shift_scores.parquet`, `co2_savings.parquet`, and `service_gaps.parquet`. Headway reliability from script 07 is clipped to `[0,1]` here before being used in the mode-shift score, guarding against the negative values the uncapped statistic can take at heavily-bunched stops. CO₂ savings use a displacement factor (`CAR_DISPLACEMENT_FACTOR`, currently 1.0) representing the number of private-car trips assumed displaced per observed bus run; the bus and car emission factors are not the same kind of quantity (`BUS_EMISSION_KG_PER_KM` is a per-passenger figure assuming a representative seat-fill level, `CAR_EMISSION_KG_PER_KM` is a per-vehicle fleet average), so the resulting figure is best read as a per-run, fill-assumption-embedded displacement estimate rather than a literal passenger-level or vehicle-level emissions comparison. This stage is meant to bridge the technical forecasting pipeline with planning or sustainability-oriented downstream use.

### Supporting scripts

**`weather_data_gen.py`** is a utility script for generating weather downloads. It is not part of the main processing chain but supports the raw weather data preparation workflow.

**`dry_run.py`** runs the full pipeline end-to-end against a small, subsampled copy of the data in an isolated sandbox directory, for local smoke-testing before a full HPC run.

## SLURM Workflow

The `research/cityflo/slurm/` directory contains one `.slurm` script for each numbered pipeline stage from 01 through 15, plus `run_pipeline.sh` for orchestration.

The ingestion stage is parallelised with SLURM arrays for bucketed processing. Downstream stages are mostly single-job submissions with resource requests tailored to each script. `run_pipeline.sh` encodes the dependency chain, so it is the safest way to submit the full pipeline on an HPC cluster.

## Configuration Reference

All parameters are defined in `scripts/config.py`. The table below documents the main parameters currently used by the pipeline.

| Parameter | Value | Use |
| --- | --- | --- |
| `STUDY_START` | `"2021-09-01"` | Start of GPS study window (legacy dataset) |
| `STUDY_END` | `"2022-10-22"` | End of GPS study window (legacy dataset) |
| `MODEL_TRAIN_END` | `"2022-07-31"` | Inclusive end of training period |
| `MODEL_VALID_END` | `"2022-08-31"` | Inclusive end of validation period |
| `MODEL_TEST_START` | `"2022-09-01"` | Start of held-out test period |
| `SPEED_MAX_KMH` | 120 | Speed sentinel / sanity threshold |
| `GPS_JUMP_MAX_KMH` | 120 | Maximum implied speed between consecutive pings |
| `DEVIATION_MAX_S` | 300 | GPS vs DB timestamp deviation cap |
| `MUMBAI_BUFFER_DEG` | 0.15 | Buffer applied when loading reference stops |
| `GAP_THRESHOLD_MIN` | 20 | New-segment threshold |
| `MIN_PINGS_PER_SEG` | 5 | Minimum pings required for a valid segment |
| `MIN_DURATION_MIN` | 5 | Minimum duration for a valid segment |
| `SNAP_THRESHOLD_M` | 200 | Maximum stop-snap radius |
| `ROUTE_MIN_OBS_STOPS` | 4 | Minimum observed stops for route matching |
| `ROUTE_MIN_CONFIDENCE` | 0.20 | Minimum LCSS similarity to assign a template |
| `ROUTE_HIGH_CONFIDENCE` | 0.30 | High-confidence threshold for trip-instance assignment |
| `ROUTE_TRIP_WINDOW_MIN` | 45 | Fallback start-time-only acceptance window |
| `ROUTE_MAX_TRIP_RESIDUAL_MIN` | 8.0 | Offset-corrected residual acceptance bound |
| `OD_TIER1_MIN_CONF` | 0.30 | Minimum confidence for Tier 1 route-template OD |
| `OD_TIME_BIN_MINUTES` | 30 | OD aggregation bin width |
| `OD_MIN_DURATION_MIN` | 2 | Minimum OD segment duration |
| `OD_MIN_PINGS` | 3 | Minimum pings for an OD segment |
| `HEADWAY_MAX_MIN` | 120 | Maximum headway retained for reliability stats |
| `ON_TIME_WINDOW_MIN` | 3 | On-time threshold |
| `LATE_THRESHOLD_MIN` | 5 | Late threshold |
| `EARLY_THRESHOLD_MIN` | -2 | Early threshold |
| `BUNCHING_PCT` | 0.25 | Bunching threshold as fraction of mean headway |
| `WEATHER_IDW_K` | 4 | Number of nearest grid points for interpolation |
| `WEATHER_IDW_POWER` | 2 | IDW distance exponent |
| `WEATHER_JOIN_TOL_MIN` | 60 | Feature-weather join tolerance in minutes |
| `H3_RESOLUTION` | 8 | H3 resolution used for spatial aggregation |
| `XGB_PARAMS` | dict | XGBoost hyper-parameters |
| `STGNN_PARAMS` | dict | ST-GNN hyper-parameters |
| `MODE_SHIFT_WEIGHTS` | dict | Weights for policy mode-shift scoring (load 0.40, regularity 0.35, reliability 0.25) |
| `CAR_DISPLACEMENT_FACTOR` | 1.0 | Assumed private-car trips displaced per bus run |
| `BUS_EMISSION_KG_PER_KM` | 0.030 | Bus per-passenger emissions assumption (~40% seat fill) |
| `CAR_EMISSION_KG_PER_KM` | 0.171 | Car per-vehicle fleet-average emissions assumption |
| `TIER1_THRESHOLD` | 0.70 | Higher policy scoring threshold |
| `TIER2_THRESHOLD` | 0.50 | Lower policy scoring threshold |

## Methodological Limitations

**Vehicle demand vs passenger demand.** The pipeline measures vehicle-run activity, not passenger boardings or alightings. OD `trip_count` values represent vehicle trips between stop pairs, not ridership.

**Coverage scope.** Cityflo serves a specific commuter segment in Mumbai. Results should not be interpreted as representative of all Mumbai travel demand without external scaling or validation.

**Destination inference remains proxy-based.** For Tier 2 OD, the destination is the last snapped stop before the segment ends. For Tier 1 OD, the destination is the matched template terminal. Neither directly observes passenger alighting.

**Route-template matching depends on observed stop coverage.** GPS segments containing only a small number of snapped stops may receive low template-match confidence despite representing valid trips, reducing the number of segments eligible for schedule-level analyses.

**Schedule adherence is based on current reference schedules.** The schedule reference file used by the pipeline is not from the same period as the historical GPS observations, so adherence metrics should be interpreted as deviation from representative schedules rather than as exact historical timetable adherence.

**Feature engineering and modelling are intentionally separated.** The current feature table is model-ready but split-agnostic. This is cleaner for experimentation, but it means leakage-sensitive aggregate features must be computed carefully inside model scripts if introduced later.

**Lag and rolling demand features are computed over observed bins, not wall-clock intervals.** Because `od_agg.parquet` only contains rows for bins with an observed run, lag-1/lag-day/lag-week/rolling-24h features (script 09) reflect the previous *observed* record(s) for an OD pair rather than a literal 30-minute/24-hour/7-day interval. This is more pronounced for sparsely-served OD pairs.

**Current-format GPS data has not been independently audited.** The quality-filter thresholds in this document were derived from EDA on the legacy export only (`notebooks/02_gps_data_audit.ipynb`). The current-format export (October 2024 – March 2026) reuses the same thresholds on the assumption that GPS hardware behaviour is broadly similar across formats; this has not yet been confirmed via an equivalent audit, and filter yield, snap rate, and inter-ping gap characteristics for the current-format data are not yet known.
