# Telecom Travel Demand Model Audit

**Date:** 2026-02-10

## 1. Configuration & Assumptions

### 1.1 **Issue:** Reliance on NHTS (National Household Travel Survey) defaults (3.0 trips/day) for Indian context [SEVERITY: HIGH]

- **File:** `config/config.yaml`
- **Function / Section:** `general.expected_daily_trips` / `trip_generation.trip_expander`
- **Why This Is Wrong:** NHTS is a US-centric benchmark. Travel behavior in Indian metros (MMR) differs significantly due to mixed land use, trip chaining complexity, and informal transit. Trip rate calibration must be context-specific; fixed rates introduce systematic scaling bias.
- **Impact:**
  - **On Expansion Factors:** Directly skews the `user_factor` calculation, leading to systematic over- or under-estimation of total travel demand.
- **Proposed Fix:**
  - **Configuration Change:** Deprecate default. Require calibration against local travel surveys (e.g., MMRDA, RITES) or Census of India migration data.
  - **Validation Requirement:** Add a strict warning if `expected_daily_trips` is not overridden.

### 1.2 **Issue:** Hardcoded time windows (Work: 07:00-20:00, Home: 20:00-07:00) [SEVERITY: MEDIUM]

- **File:** `config/config.yaml`
- **Function / Section:** `home_work_inference.home/work`
- **Why This Is Wrong:** Presumes a standard 9-5 workday structure. Fails to account for:
  - Night shift workers (BPOs, security, industrial).
  - Informal sector workers with irregular hours.
  - Early morning market traders.
- **Impact:**
  - **On Home/Work Inference:** Misclassifies workplaces as "Home" for night-shift workers.
  - **On Policy Decision Making:** Under-represents movement patterns of service/industrial labor force.
- **Proposed Fix:**
  - **Algorithmic Correction:** Implement clustering-based time profiling rather than fixed windows to *discover* primary anchor times per user.

## 2. Data Ingestion & Fusion

### 2.1 **Issue:** Naive "winner-takes-all" fusion (`conflict_resolution="highest_priority"`) [SEVERITY: CRITICAL]

- **File:** `src/data_fusion/multi_source_fusion.py`
- **Function / Section:** `_resolve_conflicts`
- **Why This Is Wrong:** Discards valid secondary data. If a specific timestamp has both an XDR (GPS) and a 5G (Network) record, it drops the 5G data entirely. True uncertainty modeling requires fusing these inputs (e.g., Inverse Variance Weighting or Kalman Filter) to reduce error radius, not just picking one.
- **Impact:**
  - **Invalidity:** False sense of precision. Suboptimal uncertainty handling; ignores multi-sensor error structure.
- **Proposed Fix:**
  - **Algorithmic Correction:** Implement a probabilistic fusion method (e.g., EKF or particle filter) or at minimum `weighted_average` as default, using signal quality as inverse variance proxy.

### 2.2 **Issue:** Reduction of XDR GPS distribution to a single mean/std centroid (`lat_mean`, `lon_mean`) [SEVERITY: HIGH]

- **File:** `src/data_ingestion/cell_tower_loader.py`
- **Function / Section:** `infer_from_xdr`
- **Why This Is Wrong:** XDR data often reveals sectorization (directional antennas). Averaging GPS points to a simple centroid destroys the sector geometry, effectively downgrading precise XDR data to omni-directional cell accuracy. Acceptable for MVP; sector/polygon modeling remains architectural enhancement.
- **Impact:**
  - **On OD Accuracy:** Increases localization error for trips starting/ending at cell edges.
- **Proposed Fix:**
  - **Architectural Refactor:** Store cell locations as `Polygon` or `Sector` (azimuth + beamwidth) rather than `Point` + `Radius`.

## 3. Preprocessing & User Filtering

### 3.1 **Issue:** Aggressive filtering based on `min_daily_trips` (default 2.5) and `min_active_days` [SEVERITY: HIGH]

- **File:** `src/preprocessing/user_filter.py`
- **Function / Section:** `filter_users`
- **Why This Is Wrong:**
  - **Socio-economic Bias:** Users with lower phone usage (shared devices, pay-per-use, low income) are systematically removed.
  - **Real Micro-mobility:** Short trips often don't generate CDRs. Filtering users who appear "sedentary" removes genuine short-distance travelers. Filtering is necessary for noise reduction; however, expansion must account for representativeness.
- **Impact:**
  - **On Representativeness:** The resulting sample skews towards white-collar, high-mobility users, introducing systematic sampling bias that must be corrected via weighting.
- **Proposed Fix:**
  - **Algorithmic Correction:** Use stratified weighting instead of hard filtering. Assign higher expansion weights to retained low-activity users rather than dropping them.

### 3.2 **Issue:** Simplistic A-B-A oscillation check [SEVERITY: MEDIUM]

- **File:** `src/preprocessing/telecom_preprocessor.py`
- **Function / Section:** `filter_ping_pong`
- **Why This Is Wrong:**
  - **Telecom Specific Pitfall:** Signal handover often involves complex patterns (A-B-C-B-A) or rapid switching between >2 towers in dense urban canyons. The current logic only detects immediate binary switching. More complex multi-tower oscillation patterns not yet modeled.
- **Impact:**
  - **On Trip Generation:**  Falsely identifies signal jitter as "short trips", inflating intra-zone trip counts.
- **Proposed Fix:**
  - **Algorithmic Correction:** Implement a velocity-based filter or Minimum Description Length (MDL) path simplification.

## 4. Stay Point Detection

### 4.1 **Issue:** Grid-based consolidation with fixed `grid_cell_size` (default 300m) [SEVERITY: HIGH]

- **File:** `src/stay_detection/stay_detector.py`
- **Function / Section:** `_consolidate_stays`
- **Why This Is Wrong:**
  - **Spatial Inconsistency:** Grid boundaries are arbitrary. A true stay point at a grid edge gets split or mis-assigned. 300m is too coarse for precise OD (e.g., distinct bus stops) but too fine for rural towers (5km radius).
- **Impact:**
  - **On Stay Detection:** Artificial fragmentation of locations.
- **Proposed Fix:**
  - **Algorithmic Correction:** Use density-based clustering (DBSCAN/OPTICS) on the candidate centroids instead of a fixed grid.

### 4.2 **Issue:** `location_confidence` formula is heuristic and arbitrary (`point_score * 0.4 + avg_weight * 0.4...`) [SEVERITY: HIGH]

- **File:** `src/stay_detection/stay_detector.py`
- **Function / Section:** `_create_stay_from_points`
- **Why This Is Wrong:** These weights (0.4, 0.2) have no statistical basis derived from ground truth. They are "magic numbers".
- **Impact:**
  - **Scientific Invalidity:** Confidence scores are not calibrated probabilities, making them useless for downstream uncertainty propagation.
- **Proposed Fix:**
  - **Removal of Unjustified Logic:** Replace magic weights with a data-driven variance estimator or remove the confidence score if uncalibrated.

## 5. Home–Work Inference

### 5.1 **Issue:** Single work location assumption [SEVERITY: MEDIUM]

- **File:** `src/stay_detection/home_work_inference.py`
- **Function / Section:** `_infer_work`
- **Why This Is Wrong:**
  - **Modern Mobility:** Gig workers, delivery personnel, and sales agents have *no* fixed work location or *multiple* sites. The code forces a single "Work" label or returns None. Advanced behavioral extension needed.
- **Impact:**
  - **On Policy Decision Making:** Completely blinds the model to the mobility needs of the gig economy.
- **Proposed Fix:**
  - **Architectural Refactor:** Allow `work_locations` to be a list/distribution. Classify "Roaming Worker" as a distinct profile.

## 6. Trip Generation & Activity Chains

### 6.1 **Issue:** `departure_time` estimation using `conditional_probability` with hardcoded Beta values [SEVERITY: HIGH]

- **File:** `src/trip_generation/trip_generator.py`
- **Function / Section:** `_create_trip`
- **Why This Is Wrong:**
  - **Statistical Bias:** The Beta parameters (2, 4) for morning and (4, 2) for evening are hardcoded guesses. They force a specific distribution curve that may not match local reality.
- **Impact:**
  - **On Time-of-Day Analysis:** Artificial smoothing of peak hours.
- **Proposed Fix:**
  - **Validation Requirement:** parameters must be learned from a subset of users with high-frequency XDR/GPS data, not hardcoded.

### 6.2 **Issue:** Validation logic (`spatial continuity`) relies on absolute stay ID equality [SEVERITY: MEDIUM]

- **File:** `src/trip_generation/trip_generator.py`
- **Function / Section:** `validate_activity_chains`
- **Why This Is Wrong:**
  - **Telecom Uncertainty:** Origin of Trip N might be "Tower A", Dest of Trip N-1 might be "Tower B" (adjacent), yet physically be the same location. Strict equality fails to account for tower handover jitter.
- **Impact:**
  - **On Data Loss:** Valid chains are flagged as invalid/incomplete due to minor spatial noise.
- **Proposed Fix:**
  - **Algorithmic Correction:** Allow fuzzy matching for chain continuity (e.g., distance < 200m).

## 7. Trip Expansion & OD Matrix Audit

### 7.1 **Issue:** `O(N^2)` memory explosion risk. Creates all possible zone pairs `all_pairs` [SEVERITY: CRITICAL]

- **File:** `src/od_matrix/od_generator.py`
- **Function / Section:** `_add_zero_flows`
- **Why This Is Wrong:**
  - **Scalability Failure:** For a city with 5,000 TACs/Zones, this creates 25,000,000 rows *per time period/purpose*. This will crash memory on standard instances.
- **Impact:**
  - **System Failure:** The pipeline is not production-safe for city-scale deployment.
- **Proposed Fix:**
  - **Architectural Refactor:** Use sparse matrix formats (scipy.sparse) or store only non-zero flows. Do not densify the matrix in Pandas.

### 7.2 **Issue:** Linear scaling `expected / observed`. and `min_observed_rate` threshold [SEVERITY: HIGH]

- **File:** `src/trip_generation/trip_expander.py`
- **Function / Section:** `_expand_user_level`
- **Why This Is Wrong:**
  - **Statistical Bias:** Assumes missing trips are missing *at random*. In reality, short trips are missing more often. Linear expansion simply multiplies observed long trips, ignoring the structural difference of missing short trips.
- **Impact:**
  - **On OD Accuracy:** Over-estimates long-distance travel, under-estimates intra-neighborhood mobility.
- **Proposed Fix:**
  - **Algorithmic Correction:** Use distance-dependent expansion factors (inverse probability weighting based on trip distance detection probability).

## 8. Scalability & Engineering

### 8.1 **Issue:** Sequential, in-memory processing (`pd.concat`) [SEVERITY: CRITICAL]

- **File:** `src/pipeline.py`
- **Function / Section:** General Architecture
- **Why This Is Wrong:**
  - **Scalability Failure:** The pipeline loads *all* data into memory. Telecom data volume (large scale) requires streaming or distributed processing (Spark/Dask).
- **Impact:**
  - **Usability:** Cannot run on distributed-scale yet, only medium/large datasets.
- **Proposed Fix:**
  - **Architectural Refactor:** Rewrite using PySpark or Dask. At minimum, implement chunk-based processing for the `preprocessing` and `trip_generation` stages.

## 9. Validation & Scientific Defensibility

### 9.1 **Issue:** Lack of ground-truth integration hooks [SEVERITY: HIGH]

- **File:** General
- **Function / Section:** Validation
- **Why This Is Wrong:**
  - **Scientific Defensibility:** The system outputs numbers but provides no mechanism to compare against:
    - Traffic counts (screen lines).
    - Public transit ridership tickets.
    - Census flow data.
- **Impact:**
  - **Policy Risk:** Outputs are "black box" estimates that cannot be defended to a transportation authority.
- **Proposed Fix:**
  - **Architectural Refactor:** Add a `ValidationModule` that ingests external constraint data and computes RMSE/GEH statistics automatically.
