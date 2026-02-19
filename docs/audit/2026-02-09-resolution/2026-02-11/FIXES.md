# FIXES - Round 1 (2026-02-18)

## Issues Addressed: 7.1, 7.1b (to_matrix_form)

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
