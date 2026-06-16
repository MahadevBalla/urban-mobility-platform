# Methodology — Cityflo Mobility Demand Pipeline

This document covers the pipeline architecture, per-script design decisions, configuration parameters, and methodological limitations. The companion document `dataset.md` covers raw data schemas and EDA findings.

## Pipeline Overview

The pipeline takes raw GPS pings and reference schedule data and produces a model-ready feature table alongside OD matrices, reliability statistics, and ward-level aggregations.

```md
Raw GPS CSVs (14-col, no header)
        │
        ▼ 01_1_ingest_legacy  →  01_2_finalize_pings  →  01_3_merge_buckets
        │
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
        ▼ 10_ward_aggregation
  ward_od.parquet
        │
        ├─────────────────────┐
        │                     │
        ▼ 11_model_nb         ▼ 12_model_xgboost
```

Scripts 02 and 03–05 can run in parallel once `pings_clean.parquet` is ready. Everything else is sequential.

All parameters are centralised in `scripts/config.py`. Scripts import from config rather than hardcoding any threshold.

## Script Design

### 01 — GPS Ingestion (three scripts)

Ingestion is split across three scripts to allow SLURM parallelism on the HPC. The full 17 GB GPS dataset is partitioned by `vehicle_id % bucket_count` so each job processes a non-overlapping subset of vehicles.

**01_1_ingest_legacy.py** reads one GPS file for one bucket, applies all quality filters in order (timestamp validity → study window → IST conversion → coordinate validity + bbox → deviation filter → speed sentinel null-out → column drops → temporal derived columns), and writes intermediate parquet files. Deduplication and the GPS jump filter are deferred because both require the full sorted vehicle history across all input files.

**01_2_finalize_pings.py** reads all intermediate parquets for one bucket, deduplicates on `(vehicle_id, ts_utc)`, then applies the GPS jump filter. The jump filter computes inter-ping haversine speed using a Polars expression chain (no numpy roundtrip) and removes pings where the implied speed exceeds `GPS_JUMP_MAX_KMH`. The haversine is computed fully within the Polars lazy graph so intermediate arrays are never materialised in Python.

**01_3_merge_buckets.py** concatenates finalized bucket parquets into `pings_clean.parquet` using `scan_parquet` → `sink_parquet` so the merge never loads the full dataset into memory.

### 02 — Route Catalog

Reads `trips_clean.csv` and extracts the set of unique route templates. A template is defined by its ordered stop sequence — multiple `trip_id` rows sharing the same stop order belong to the same template regardless of scheduled times or dates.

For each template, the catalog stores the median scheduled arrival time per stop, computed using `statistics.median` across all runs of that template. Median is used rather than mean because outlier runs (shortened trips, deadheads) are common in transit schedule exports and would skew mean arrival times.

The output `route_catalog.json` is a human-readable copy of the catalog for debugging and manual inspection. The parquet is used by all downstream scripts.

### 03 — Trip Segmentation

Assigns a `segment_id` to each ping within a vehicle trajectory. A new segment begins when the inter-ping gap exceeds `GAP_THRESHOLD_MIN` (20 minutes). Segmentation is partitioned by `vehicle_id` only, not by `(vehicle_id, ride_date)` — vehicle trajectories are continuous physical paths that should not be split at midnight.

`segment_id` restarts at 0 for each vehicle and is not globally unique. Downstream code uses `(vehicle_id, segment_id)` as the segment key.

After segmentation, segments with fewer than `MIN_PINGS_PER_SEG` (5) pings or shorter than `MIN_DURATION_MIN` (5 minutes) are removed. These micro-segments represent brief GPS activity that does not correspond to service runs.

The implementation uses DuckDB window functions (`SUM(is_new::INTEGER) OVER (PARTITION BY vehicle_id ORDER BY timestamp_ist)`) which avoids materialising per-vehicle Python DataFrames.

### 04 — Stop Snapping

Snaps each GPS ping to the nearest bus stop using a BallTree with Haversine metric. Pings within `SNAP_THRESHOLD_M` (300 m) receive a `snapped_stop_id`; pings beyond the threshold receive `snapped_stop_id = -1` and are retained for completeness. Downstream scripts filter on `snapped_stop_id != -1` where stop-level data is required.

Pings with null or non-finite coordinates are short-circuited before the BallTree query and assigned `snapped_stop_id = -1` directly. The BallTree is built from `stops_clean.csv` after excluding stops with invalid coordinates or coordinates outside the Mumbai bbox buffer.

Processing streams through PyArrow `iter_batches` at 2M rows per chunk to keep memory bounded.

This is a global snap — the nearest stop across the entire catalog, not constrained to a specific route's stops. Route-constrained validation happens in script 05 through the template matching logic.

### 05 — Route Inference

Matches each GPS segment's observed stop sequence against the 641 route templates to assign a `template_id` and, for high-confidence matches, a `candidate_trip_id`.

**Candidate generation.** For each segment, an inverted index (`stop_id → {template_ids}`) limits the matching space to templates that share at least two stops with the observed sequence. This typically reduces candidates from 641 to ~5–30 per segment.

**Scoring.** Each candidate is scored on six metrics and ranked lexicographically:

| Metric | Definition |
| --- | --- |
| `overlap_score` | `\|obs ∩ template\| / \|obs\|` — precision |
| `order_score` | LCS length ÷ observed length |
| `direction_score` | Monotonicity ratio of matched template positions |
| `endpoint_score` | Whether observed first/last stops fall near template terminals |
| `coverage_score` | `\|obs ∩ template\| / \|template\|` — recall |
| `confidence` | `min(overlap_score, order_score, max(0, direction_score))` |

Ranking uses lexicographic order on `(overlap_score, order_score, direction_score, endpoint_score, coverage_score, -template_len_penalty)` with no weighted averaging. The `match_margin` column records the confidence gap between the best and second-best candidate — low margin indicates an ambiguous match.

LCS-based order scoring is used over a simple subsequence check because it handles partial observations: a segment that started GPS recording mid-route still scores correctly as long as the observed stops appear in the right relative order.

**Trip ID assignment.** For segments where `match_confidence ≥ ROUTE_HIGH_CONFIDENCE` (0.50), the script searches `trips_clean.csv` for the closest scheduled run of the matched template on the segment's date. The search compares the segment's start time (minutes from midnight IST) against each candidate trip's first-stop scheduled time, accounting for midnight rollover. A match is assigned if the closest trip falls within `ROUTE_TRIP_WINDOW_MIN` (45 minutes).

**Validation.** The `--run_validation` flag enables a leave-some-out accuracy check: for a sample of trips in `trips_clean.csv`, 40% of stops are randomly dropped to simulate partial GPS observation, and the algorithm attempts to recover the correct template. This produces a top-1 accuracy metric that characterises matching performance independently of the GPS data.

### 06 — OD Matrix

Constructs origin-destination pairs at the vehicle-run level from matched segments. All output counts represent vehicle runs, not individual passengers.

**Tier 1** (route-template OD): for segments where `match_confidence ≥ OD_TIER1_MIN_CONF` (0.30), origin and destination are the terminal stops of the matched route template — not the first/last GPS-snapped stop. This reduces truncation bias from partial GPS coverage at trip start or end.

**Tier 2** (first/last-snap OD): for unmatched segments, origin = first snapped stop chronologically, destination = last snapped stop. This is noisier but provides coverage for unmatched segments.

The 30-minute time bin is computed via epoch arithmetic (`(epoch(ts)::BIGINT / 1800) * 1800`), which avoids timezone handling edge cases in `DATE_TRUNC`.

Quality filters (`ping_count ≥ OD_MIN_PINGS`, `trip_dur_min ≥ OD_MIN_DURATION_MIN`) are applied after the group-by aggregation using a CTE — not in a WHERE clause on window functions, which is invalid SQL.

`od_agg` joins stop metadata (name, category, coordinates) and computes trip distance using the equirectangular approximation. `service_supply` and `service_frequency` are derived tables showing estimated headways per OD pair.

### 07 — Reliability

**Headway statistics.** Raw pings at a stop are collapsed into discrete stop-arrival events before headway computation. Contiguous pings from the same vehicle at the same stop are grouped into one arrival event using a LAG-based visit boundary detection. Headways are then computed as time differences between consecutive arrivals at each stop on each day.

Bunching is computed by joining against a pre-computed per-stratum mean headway (in a separate CTE), then flagging arrivals where `headway < BUNCHING_PCT × stratum_mean`. This avoids the invalid SQL pattern of referencing a window function result inside a grouped aggregate.

**Schedule adherence.** For segments with an assigned `candidate_trip_id` and `match_confidence ≥ ROUTE_HIGH_CONFIDENCE` (0.50), the scheduled arrival time for each snapped stop is looked up directly from `trips_clean.csv` using `(trip_id, stop_id)` as the join key — not from the template's median schedule. This provides trip-specific scheduled times rather than an averaged reference.

Delay is `actual_arrival_min_ist - scheduled_arrival_min_ist`. Positive = late; negative = early. Observations with `|delay| ≥ 90 minutes` are excluded as likely trip_id misassignments.

`trips_clean.csv` contains schedules from December 2025–July 2026 while GPS observations are from September 2021–October 2022. Schedule adherence metrics therefore reflect deviation from representative operational schedules, not historically verified timetables.

### 08 — Weather Consolidation

Loads all half-month CSV files per grid point, merges the three hourly variable groups on the `time` column, and concatenates across all 15 grid points into `weather_grid_master.parquet`.

Stop-level weather is interpolated from the 15 grid points using inverse-distance weighting (IDW): for each stop, the k=4 nearest grid points are identified, weights are computed as `1/distance²`, and each weather variable is interpolated as the weighted mean of the k values.

After interpolation, derived features are added: rolling precipitation sums (3h, 6h, 24h), heat index (Steadman approximation, applied where T > 27°C and RH > 40%), WMO weather code severity binning, soil saturation flags, and a strong-wind flag (gusts > 40 km/h).

### 09 — Feature Engineering

Assembles the model-ready feature table from OD, reliability, weather, and H3 spatial assignments.

DuckDB handles the OD × reliability join and all lag/rolling window features. Lag features (`lag_1`, `lag_2`, `lag_48`, `lag_336` bins) and rolling windows (48-bin mean and std) are computed by window functions over `(origin_stop_id, dest_stop_id)` ordered by `time_bin_30min`. These look strictly backward — LAG references only past observations in the time-sorted sequence.

The weather join uses `pandas.merge_asof` with a 60-minute tolerance, matching each 30-minute OD bin to the nearest hourly weather observation at the origin stop.

H3 cell assignment (resolution 8) is applied to origin and destination stop coordinates. Null coordinates receive `None` rather than raising.

The `split` column (`"train"` / `"test"`) is assigned based on `MODEL_TRAIN_END`.

### 10 — Ward Aggregation

Assigns each stop to a Mumbai ward via point-in-polygon join against the ward KML boundaries using GeoPandas. Stops that fall outside all ward polygons are assigned to the nearest ward centroid using a BallTree fallback. The `stop_ward_map.csv` output records each stop's ward assignment and the method used (polygon or centroid).

Ward-level OD is aggregated from `od_agg.parquet` by summing `trip_count` across all OD pairs where origin and destination stop fall within the respective wards.

### 11 — Negative Binomial Model

Uses `statsmodels.NegativeBinomial` directly (not `GLM` with NB family) so that the dispersion parameter `alpha` is estimated jointly with the regression coefficients rather than fixed.

MAPE is computed only over non-zero `trip_count` observations since MAPE is undefined at zero.

### 12 — XGBoost Model

Uses `TimeSeriesSplit` for cross-validation, respecting temporal ordering across all folds. The final model trains on `split == "train"` and evaluates on `split == "test"`.

SHAP values are computed from a sample of the test set using `TreeExplainer`.

## Configuration Reference

All parameters are defined in `scripts/config.py`. The table below documents each parameter's value and the EDA finding that justifies it.

| Parameter | Value | Justification |
| --- | --- | --- |
| `STUDY_START` | `"2021-09-01"` | First date in the `before_2022-10-22` GPS files |
| `STUDY_END` | `"2022-10-22"` | Last date in the `before_2022-10-22` GPS files |
| `SPEED_MAX_KMH` | 120 | Confirmed by project supervisor; sentinel cluster confirmed at 602+ in EDA |
| `GPS_JUMP_MAX_KMH` | 120 | Same threshold; catches teleportation errors the reported speed field misses |
| `DEVIATION_MAX_S` | 300 | Removes 0.12% of rows; p99 of deviation distribution is well below this |
| `MUMBAI_BBOX` | lat [18.8894, 19.3274], lng [72.7692, 73.1165] | Tight bbox from EDA spatial distribution of valid pings |
| `MUMBAI_BUFFER_DEG` | 0.15 | Added to bbox when loading stops for BallTree |
| `GAP_THRESHOLD_MIN` | 20 | 0.1% of inter-ping transitions exceed this; bimodal gap distribution |
| `MIN_PINGS_PER_SEG` | 5 | Minimum below which a segment cannot reliably produce a stop sequence |
| `MIN_DURATION_MIN` | 5 | Filters GPS noise and idle-engine events |
| `SNAP_THRESHOLD_M` | 300 | 82% ping snap rate at 300 m; CDF plateaus beyond this |
| `ROUTE_MIN_OBS_STOPS` | 4 | With 641 templates, ≥4 confirmed stops substantially reduces ambiguous matches |
| `ROUTE_MIN_CONFIDENCE` | 0.20 | Below this, no template is assigned |
| `ROUTE_HIGH_CONFIDENCE` | 0.50 | Required for trip_id assignment and schedule adherence |
| `ROUTE_TRIP_WINDOW_MIN` | 45 | Max gap between segment start time and nearest scheduled trip departure |
| `OD_TIER1_MIN_CONF` | 0.30 | Minimum confidence for route-template OD assignment |
| `OD_MIN_DURATION_MIN` | 2 | Filters segments too short to represent a service run |
| `OD_MIN_PINGS` | 3 | Minimum pings for a segment to produce an OD pair |
| `HEADWAY_MAX_MIN` | 120 | Excludes overnight service gaps from headway statistics |
| `ON_TIME_WINDOW_MIN` | 3 | ±3 minutes on-time window (standard transit operations definition) |
| `LATE_THRESHOLD_MIN` | 5 | Delay > 5 minutes = late |
| `EARLY_THRESHOLD_MIN` | -2 | Delay < -2 minutes = early |
| `BUNCHING_PCT` | 0.25 | Headway < 25% of stratum mean = bunching event |
| `WEATHER_IDW_K` | 4 | Number of nearest grid points for IDW interpolation |
| `WEATHER_IDW_POWER` | 2 | IDW distance weighting exponent |
| `H3_RESOLUTION` | 8 | H3 resolution 8 gives ~460 m² cells, appropriate for stop density |
| `MODEL_TRAIN_END` | `"2022-07-31"` | Train/test split boundary |
| `MODEL_TEST_START` | `"2022-08-01"` | First date of held-out test period |

## Methodological Limitations

**Vehicle demand vs passenger demand.** The pipeline measures vehicle-run activity, not passenger boardings or alightings. OD `trip_count` values represent vehicle runs between stop pairs, not ridership. This distinction applies throughout — the load factor proxy, frequency estimates, and all derived metrics are vehicle-level.

**Coverage scope.** Cityflo serves a specific commuter segment in Mumbai. Results are not representative of Mumbai's overall travel demand without external modal share scaling.

**Destination inference.** For Tier 2 OD, destination is inferred from the last snapped stop before a gap, not from a confirmed alighting event. For Tier 1 OD, destination is the route terminal from the matched template. Both are proxies.

**Schedule adherence mismatch.** `trips_clean.csv` covers December 2025–July 2026; GPS observations are from September 2021–October 2022. Schedule adherence reflects deviation from representative rather than historically confirmed timetables.
