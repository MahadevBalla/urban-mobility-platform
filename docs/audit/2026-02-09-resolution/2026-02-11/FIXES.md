## FIXES - Round 1 (2026-02-18)

### Issues Addressed: 7.1, 7.1b (to_matrix_form)

### Issue 7.1 - O(N²) Memory Explosion in `_add_zero_flows`

**File:** `src/od_matrix/od_generator.py`  
**Function:** `_add_zero_flows`  
**Severity:** CRITICAL

#### Root Cause

Nested Python `for` loop over all zone pairs:

```python
# OLD - O(N²) list appends
all_pairs = []
for origin in zones:
    for dest in zones:
        all_pairs.append({"origin": origin, "destination": dest})
all_pairs_df = pd.DataFrame(all_pairs)
```

At 5,000 zones × 4 time periods × 3 purposes → 300,000,000 rows.

#### Fix Applied

```python
# NEW - single vectorized operation, O(N²) rows but no Python loop overhead
idx = pd.MultiIndex.from_product([zones, zones], names=["origin", "destination"])
all_pairs_df = idx.to_frame(index=False)
```

Plus a **hard zone-count guard** (`SPARSE_ZONE_THRESHOLD = 500`):

```python
if n > SPARSE_ZONE_THRESHOLD:
    logger.error("include_zero_flows refused: zone count exceeds threshold")
    return od_matrix   # returns non-zero flows only - safe fallback
```

### Issue 7.1b - Dense Pivot in `to_matrix_form`

**File:** `src/od_matrix/od_generator.py`  
**Function:** `to_matrix_form`  
**Severity:** HIGH

#### Root Cause

```python
# OLD - always densifies to N×N DataFrame
matrix = od_df.pivot_table(..., fill_value=0)
matrix = matrix.reindex(index=zones, columns=zones, fill_value=0)
```

#### Fix Applied

- `to_dense_matrix()` → dense pivot, raises `ValueError` if `n > 500`
- `to_sparse_matrix()` → new primary method, returns `scipy.sparse.csr_matrix`

#### Config Changes (config.yaml)

Added to `od_matrix` section:

```yaml
sparse_zone_threshold: 500    # mirrors SPARSE_ZONE_THRESHOLD in od_generator.py
expected_daily_trips: null    # null triggers NHTS warning (Audit Issue 1.1)
```

Added to `general.processing`:

```yaml
mode: "chunked"               # pre-wires pipeline for Round 1 - 8.1 fix
chunk_size_users: 1000
intermediate_format: "parquet"
intermediate_dir: "data/intermediate"
```

Added to `preprocessing`:

```yaml
ping_pong_filter: "aba"       # pre-wires for Round 2 - Issue 3.2 fix
```

Added to `trip_generation`:

```yaml
chain_continuity_tolerance_m: 0   # pre-wires for Round 6 - Issue 6.2 fix
departure_time_beta_morning: [2, 4]  # [alpha, beta] - placeholder, not calibrated
departure_time_beta_evening: [4, 2]  # [alpha, beta] - placeholder, not calibrated
```

### Issue 8.1 - Sequential In-Memory Pipeline

**File:** `src/pipeline.py`  
**Severity:** CRITICAL  

#### Root Cause

`preprocessed` DataFrame (all users, all records) was kept in `self._results["preprocessed"]` from step 2 through step 7. Steps 5-7 each hold their own output DataFrames simultaneously, meaning peak RAM = preprocessed + stay_points + trips all live at once.

#### Fix Applied

Two-mode execution via `config.general.processing.mode`:

- `full`: original in-memory behaviour, unchanged (default for sample config)
- `chunked`: after step 4, `preprocessed` is written to `config.general.processing.intermediate_dir` as parquet and deleted from memory. Steps 5 and 7 load `chunk_size_users` users at a time via parquet filter pushdown. Peak RAM reduces from O(total records) to O(chunk_size * avg_records_per_user).

No changes required to child classes - StayPointDetector, TripGenerator, and HomeWorkInference already iterate per-user internally.

## FIXES - Round 2 (2026-02-19 → 2026-02-20)

### Issues Addressed: 2.1, 2.2, 3.2

### Issue 2.1 - Winner-Takes-All Data Fusion

**File:** `src/data_fusion/multi_source_fusion.py`  
**Severity:** CRITICAL  

#### Root Cause

`_resolve_conflicts` used `idxmax()` on `location_confidence` for every time bucket, discarding all secondary source records entirely. A 5G record with valid `cell_id` and signal quality at the same timestamp as an XDR record was silently dropped - its cell identity and signal data lost.

#### Fix Applied

Replaced `_weighted_average_fusion` (which still fell back to `idxmax()` for all non-coordinate fields) with `_ivw_fusion`: a proper Inverse Variance Weighting estimator treating `location_confidence` as a precision proxy (1/σ²).

For each user and temporal alignment window:

- **Coordinates:** IVW over all records with valid GPS (`Σ w_i·lat_i / Σ w_i`)
- **cell_id:** taken from highest-confidence record that has a cell_id (not dropped)
- **fused_confidence:** mean confidence across contributing sources (heuristic reliability score)
- **sources:** comma-joined provenance string (e.g. `"xdr_coordinates,network_5g"`)
- **Timestamp:** confidence-weighted mean within bucket

Single-source buckets produce identical output to the original - no regression.

`weighted_average` is now the default `conflict_resolution` in config.

**Deferred - Round 3:** Kalman Filter trajectory smoothing (sequential noise reduction on the ordered location sequence) belongs in `stay_detector.py` alongside the DBSCAN refactor (Issue 4.1), not here. `strategy: "kalman_filter"` in config is pre-wired for that round.

Reference: IVW estimator - Hartung et al. (2008)

### Issue 2.2 - XDR Centroid Oversimplification

**File:** `src/data_ingestion/cell_tower_loader.py`, `src/utils/geo_utils.py`  
**Severity:** HIGH  

#### Root Cause

`infer_from_xdr` reduced all GPS observations for a cell to a single `(lat_mean, lon_mean)` point plus a radius derived from standard deviation. This destroys any directional information in the point cloud - a sectorized tower with three antenna lobes collapses to one centroid near the tower base, which is neither the tower location nor any sector center.

#### Fix Applied

Introduced `CellRecord` dataclass to carry both the centroid (backward-compat) and a polygon geometry per cell. Cell geometry is now built according to `config.cell_towers.location_method`:

- **`convex_hull`** (new default): `scipy.spatial.ConvexHull` on the raw XDR GPS point cloud per `cell_id`. Preserves the spatial extent and directional bias of the point cloud deterministically. Falls back to a circular polygon when GPS sample count < `config.cell_towers.min_hull_samples` (default 5). Collinear degenerate case buffered via `shapely` (local import, not a hard dep).

- **`centroid`** (legacy): original behaviour unchanged, available via config.

- **`sector_model`** (pre-wired): analytically exact wedge polygon from azimuth + beamwidth. Requires operator NMS/BSS antenna metadata not present in raw XDR. `infer_from_xdr` logs a warning and falls back to `convex_hull`. Activates automatically via `load_from_file()` once antenna data is available.

Backward compatibility preserved: `get_cell_location(cell_id)` still returns `(lat, lon, radius)`. New API: `get_cell_record(cell_id)` returns full `CellRecord` with `geometry` and `geometry_type`.

`add_locations_to_df` now only fills rows with missing coordinates (was overwriting all rows including valid GPS - silent correctness bug).

**Downstream note:** stay_detector, trip_generator, home_work_inference all consume `(lat, lon)` centroid only. Polygon geometry is stored for future zone-containment queries; integration planned for Round 3 `zone_loader` refactor.

Added to `geo_utils.py`: `build_convex_hull_polygon`, `build_sector_polygon`, `_circle_polygon`.

Config changes:

```yaml
cell_towers:
  location_method: "convex_hull"   # was "centroid"
  min_hull_samples: 5
```

### Issue 3.2 - Ping-Pong Filtering Logic

**File:** `src/preprocessing/telecom_preprocessor.py`  
**Severity:** MEDIUM  

#### Root Cause

The previous implementation used a strict ABA oscillation rule (A→B→A within a fixed time window) with two limitations:

1. Required ≥3 oscillations before flagging, allowing single A→B→A round-trips to pass.
2. Executed before `add_locations_to_df()` in the pipeline, meaning CDR rows lacked coordinates and velocity-based validation was not feasible.

No speed-based plausibility check was implemented.

#### Resolution

Ping-pong filtering was refactored to support two config-driven methods via `preprocessing.ping_pong_filter`:

##### 1. `aba` (corrected)

Flags the middle record of any A→B→A oscillation occurring within `ping_pong_time_threshold` seconds.

Used as fallback when coordinate data is unavailable.

##### 2. `velocity` (default)

Flags records where implied travel speed between consecutive observations exceeds a physically plausible threshold.

Source-aware thresholds:

- `source="xdr"` with `has_coordinates=True` → `gps_max_speed_ms` (42 m/s)
- `source="cdr"` or centroid-derived coordinates → `centroid_max_speed_ms` (80 m/s)

Pairs where `dt < min_time_gap_s` are skipped (not flagged) to avoid unreliable speed estimation at sub-resolution timestamps.

Per-user fallback to ABA occurs automatically if coordinates are missing.

#### Pipeline Wiring Fix

`remove_ping_pong()` is now called in `pipeline.py` Step 4 immediately after `add_locations_to_df()`:

```python
updated = self.cell_loader.add_locations_to_df(preprocessed)
updated = self.preprocessor.remove_ping_pong(updated)
```

This ensures centroid coordinates are available before velocity filtering.

#### Configuration Additions

```yaml
ping_pong_filter: "velocity"
ping_pong_time_threshold: 300

ping_pong_velocity:
  gps_max_speed_ms: 42.0
  centroid_max_speed_ms: 80.0
  min_time_gap_s: 5
```

## FIXES - Round 3 (2026-02-26)

### Issues Addressed: 4.1

### Issue 4.1 – Fixed-Grid Stay Consolidation

**File:** `src/stay_detection/stay_detector.py`
**Severity:** HIGH

#### Root Cause

Stay consolidation previously relied on a fixed spatial grid (`grid_cell_size = 300m`).
Candidate stay points were snapped to grid cells and merged if they fell within the same cell.

This approach introduces two structural weaknesses:

1. **Boundary Splitting**
  A single real-world location near a grid boundary may be split into multiple stay points depending on which side of the grid cell the observation falls.

2. **False Merging in Dense Areas**
  Distinct activity locations within the same grid cell (e.g., two nearby buildings) are merged into one stay, reducing spatial fidelity.

Both effects silently distort downstream:

- Home/work inference
- Trip generation
- OD estimation

No explicit error is raised — the distortion propagates.

#### Fix Applied

Replaced fixed-grid consolidation with **density-based clustering (HDBSCAN)** using geographic distance.

Key changes:

```python
clusterer = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="haversine",
    algorithm="ball_tree",
    cluster_selection_method=method,
    copy=False,
)

labels = clusterer.fit_predict(coords_rad)  # coordinates converted to radians
```

#### Why HDBSCAN

- Clusters points based on spatial density rather than arbitrary cell alignment.
- Preserves distinct nearby locations if they form separate dense groups.
- Avoids artificial splitting caused by grid boundaries.
- Handles variable density across suburban vs urban regions.
- Noise points (`label == -1`) are retained as singleton stays rather than dropped.

#### Coordinate-Less Fallback

For users without coordinate data (e.g., CDR-only records):

- Consolidation falls back to `cell_id` grouping.
- No regression in behavior for non-GPS cases.

#### Configuration Changes

New consolidation block under `stay_detection`:

```yaml
stay_detection:
  consolidation:
    method: "hdbscan"               # default (was "grid")
    min_cluster_size: 2
    min_samples: 1
    cluster_selection_method: "eom"
    metric: "haversine"
    grid_cell_size: 300             # retained for explicit fallback
```

- `grid` remains available via config.
- All clustering parameters are now externally configurable.
- `copy=False` explicitly set to avoid future sklearn default change warning.

#### Impact

- Removes grid-boundary artifacts.
- Preserves spatial diversity in dense areas.
- Produces more behaviorally consistent stay sets.
- Improves structural robustness of downstream home/work and trip inference.

Downstream modules continue to consume `(lat, lon)` centroids.

## FIXES - Round 4 (2026-03-04)

### Issues Addressed: 6.1

### Issue 6.1 – Hardcoded Beta Departure Time Parameters

**File:** `src/trip_generation/trip_generator.py`, `src/utils/time_utils.py`  
**Severity:** HIGH  

#### Root Cause

Departure time estimation used a `conditional_probability` approach where the Beta distribution parameters controlling the departure offset were hardcoded:

- random_offset = random.betavariate(2, 4)
- random_offset = random.betavariate(4, 2)

These values implicitly assumed fixed temporal behavior patterns without calibration.

Hardcoding distribution parameters creates two problems:

1. **Statistical Bias**

- Departure distributions become fixed regardless of city context, telecom sampling rate, or behavioral patterns.

2. **Calibration Impossibility**

- Model users cannot adjust the departure distribution without modifying source code.
- This violates the principle that behavioral parameters should be **externally configurable and empirically calibrated**.

#### Fix Applied

- Beta parameters are now configurable via `config.yaml`.

- TripGenerator loads the parameters during initialization:

  ```python
  self.beta_morning = tuple(
  trip_config.get("departure_time_beta_morning", [2, 4])
  )

  self.beta_evening = tuple(
  trip_config.get("departure_time_beta_evening", [4, 2])
  )
  ```

- `generate_departure_time_distribution()` now accepts these parameters:

  ```python
  generate_departure_time_distribution(
  departure_obs_time,
  arrival_obs_time,
  self.departure_method,
  beta_morning=self.beta_morning,
  beta_evening=self.beta_evening,
  )
  ```

- The time utility now uses the provided parameters:

  ```python
  alpha_m, beta_m = beta_morning
  alpha_e, beta_e = beta_evening

  random_offset = random.betavariate(alpha_m, beta_m)
  ```

#### Configuration

- Added to `config.yaml`:

  ```yaml
  trip_generation:
  departure_time_beta_morning: [2, 4]
  departure_time_beta_evening: [4, 2]
  ```

- These remain **placeholder defaults** consistent with the previous implementation.

#### Calibration Requirement

- These parameters should ideally be calibrated using **high-frequency trajectory data** (e.g., XDR GPS observations) to match observed departure-time distributions.

- The current defaults preserve backward compatibility while enabling empirical calibration.

#### Impact

- Removes hardcoded behavioral parameters from source code.
- Enables dataset-specific calibration without code modification.
- Improves scientific defensibility of departure-time estimation.

## FIXES - Round 5 (2026-03-05)

### Issue 3.1 – Aggressive User Filtering

**File:** `src/preprocessing/user_filter.py`
**Severity:** HIGH

#### Root Cause

- User filtering previously applied a behavioural threshold:

  ```python
  valid_users = valid_users[
      valid_users["avg_daily_records"] >= self.min_daily_trips
  ]
  ```

- `avg_daily_records` was used as a proxy for trip frequency, and users below the threshold were removed.

- However, telecom activity frequency is **not a reliable proxy for mobility behaviour**. Low-phone-usage users (e.g. prepaid users, shared devices, informal workers) may still have normal travel patterns.

- Filtering these users introduces **systematic sampling bias**, over-representing high-phone-usage individuals.

- At the same time, the pipeline’s expansion module already corrects sampling bias via activity-based weighting:

  ```python
  user_factor = expected_daily_trips / observed_daily_rate
  ```

- Removing low-activity users before expansion prevents this correction from operating.

#### Fix Applied

- The behavioural filtering rule was removed.

- `UserFilter` now applies **data-quality filters only**:

  - `min_records_per_user` – ensures sufficient observations for stay inference
  - `min_active_days` – ensures temporal coverage for home/work inference
  - `max_records_per_user` – removes anomalous high-frequency SIMs

- The following filter was removed: `avg_daily_records >= min_daily_trips`

- Low-activity users are now retained so that expansion weighting can correct for under-representation.

#### Code Changes

- Removed the filtering block:

  ```python
  if self.min_daily_trips > 0:
      valid_users = valid_users[
          valid_users["avg_daily_records"] >= self.min_daily_trips
      ]
  ```

- `min_daily_trips` is now deprecated and ignored.

#### Configuration Changes

- Removed from `config.yaml`:

  ```yaml
  preprocessing:
    min_daily_trips: 2.5
  ```

- Remaining filters:

  ```yaml
  preprocessing:
    min_records_per_user: 10
    min_active_days: 3
    max_records_per_user: 100000
  ```

#### Impact

- Eliminates behavioural filtering based on telecom activity
- Retains low-phone-usage users in the sample
- Allows downstream expansion weighting to correct sampling bias
- No changes required in downstream modules
