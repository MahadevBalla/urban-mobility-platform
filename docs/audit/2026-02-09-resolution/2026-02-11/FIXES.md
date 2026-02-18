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

### Config Changes (config.yaml)

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
