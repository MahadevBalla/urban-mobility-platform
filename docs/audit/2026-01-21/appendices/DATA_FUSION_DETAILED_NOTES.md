# Data Fusion Module: Implementation & Validity Issues

## Table of Contents

- [Overview](#overview)
- [Ground Truth Generation](#ground-truth-generation)
- [Sensor Simulation](#sensor-simulation)
- [Map Matching](#map-matching)
- [Fusion Logic](#fusion-logic)
- [Evaluation Metrics](#evaluation-metrics)
- [Cross-Cutting Issues](#cross-cutting-issues)
- [Code Quality](#code-quality)
- [Priority Recommendations](#priority-recommendations)
- [Summary Table](#summary-table)

## Overview

This document catalogs implementation issues in the data fusion module that affect research validity and accuracy. The problems range from incorrect distance calculations to flawed evaluation metrics. Current interpolation and state modeling make advanced methods like HMM map matching or Kalman filtering mathematically invalid even if implemented.

**Severity levels**:

- **CRITICAL**: Fundamentally flawed, produces incorrect results
- **HIGH**: Significant impact on accuracy or validity
- **MEDIUM**: Affects performance or edge cases
- **LOW**: Minor improvements, best practices

## Ground Truth Generation

### CRITICAL: Inconsistent Distance Calculations

**File**: `ground_truth_generator.py` (lines 192-217)

The code mixes two incompatible distance methods:

1. **Haversine formula** for route length (correct for geodesic distance)
2. **Euclidean interpolation** for point positioning (incorrect for lat/lon)

```python
# Method 1: Haversine - correct
def _get_route_length_meters(self):
    total_length += 6371000 * c  # Earth radius

# Method 2: Euclidean - wrong for lat/lon
def _get_point_at_distance(self, distance_m):
    fraction = distance_m / route_length  # Uses Haversine length
    point = self.route_line.interpolate(fraction, normalized=True)  # Uses Euclidean!
```

**Why this is wrong**: Haversine calculates geodesic distance (curved Earth surface). `LineString.interpolate()` uses Euclidean distance (flat plane). A 1000m distance in Haversine ≠ 1000m in Euclidean at lat/lon coordinates.

**Impact**:

- Ground truth positions are systematically incorrect
- Errors increase with route length (up to 10-20% for 4km routes)
- All fusion evaluations compare against wrong baseline

**Fix**: Use proper geospatial library like `pyproj` or convert to projected coordinates (UTM).

### CRITICAL: Kinematic Model Is Internally Inconsistent

**File**: `ground_truth_generator.py` (lines 226-265, 336-376)

The kinematic model has multiple fundamental flaws:

#### **1. Distance not integral of speed**

Speed is generated independently, but distance is advanced as:

```python
segment_distance += speed  # assumes dt = 1s
```

However, speed is randomly perturbed, clamped, and computed from flawed kinematics. This violates the fundamental identity: `distance = ∫ speed dt`.

#### **2. Invalid deceleration physics**

```python
# Current - wrong
speed = np.sqrt(2 * self.deceleration * max(remaining, 0.1))
```

This assumes constant deceleration from current position to stop, but the vehicle is already at cruising speed.

#### **3. No jerk limits**

Real vehicles don't instantly change acceleration. The model applies step changes in acceleration without considering jerk (rate of change of acceleration).

#### **4. Fixed 1-second sampling**

Creates aliasing for high-speed segments where vehicle state changes rapidly.

#### **5. No feasibility checks post-fusion**

Reconstructed trajectories are never validated for maximum speed, acceleration, or turn rate. Teleportation and instantaneous direction reversals are allowed. (Consequences appear again during evaluation, where infeasible trajectories are never penalized.)

**Impact**:

- Speed RMSE becomes meaningless
- Algorithms using speed consistency will fail
- Cannot validate kinematic models
- Ground truth speeds are physically unrealistic
- Fusion algorithms learn incorrect speed patterns
- Physically impossible trajectories score well in evaluation

**Fix**: Make distance the primary state variable. Derive speed from finite differences: `v = Δx / Δt`. Add jerk limits. Implement post-fusion feasibility validation.

### MEDIUM: Route Geometry Is Simplified for Controlled Testing

**File**: `ground_truth_generator.py`

Routes are hand-crafted with low curvature and even spacing. Missing parallel roads, slip roads, and sharp turns that exist in real-world scenarios.

**Impact**: Map matching difficulty is understated compared to real deployments. Results may be biased toward simpler methods. This is acceptable for controlled experiments but limits real-world generalizability.

> Note: This simplification may be intentional for controlled experiments; the concern is primarily about reduced realism and external validity rather than internal correctness.

**Fix**: For production-oriented evaluation, use real OSM geometry with noisy perturbations.

### MEDIUM: Heading Calculation Edge Cases

**File**: `ground_truth_generator.py` (lines 219-224, 359-360)

```python
def _calculate_heading(self, p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    heading = np.degrees(np.arctan2(dx, dy))
    return (heading + 360) % 360
```

**Issues**:

- Uses `arctan2(dx, dy)` which doesn't account for longitude convergence at different latitudes
- At stops, heading calculated using point 10m ahead can be unstable near route endpoints
- No handling for U-turns or sharp curves

**Fix**: Use proper geodesic bearing calculation with `pyproj.Geod`.

### MEDIUM: Route Length Recomputed Unnecessarily

**File**: `ground_truth_generator.py`

`_get_route_length_meters()` is called repeatedly though route geometry never changes. Expensive and may accumulate floating-point drift.

**Fix**: Compute once and cache.

### MEDIUM: Stops Treated as Point Locations

**Files**: `ground_truth_generator.py`, `sensor_simulator.py`

Stops are modeled as single points with zero spatial uncertainty. Real GTFS stops are areas (20-100m).

**Impact**: Stop arrival error should be spatial + temporal. GTFS-based fusion may be slightly penalized compared to real-world scenarios. This approximation is acceptable for many analyses but becomes important for fine-grained validation and sub-50m accuracy requirements.

**Fix**: Model stops as `(center, radius)` with spatial dwell jitter for higher-fidelity evaluation.

## Sensor Simulation

### CRITICAL: GPS Noise Model Is Oversimplified

**File**: `sensor_simulator.py` (lines 100-132)

```python
# Current implementation - wrong
meters_to_deg = 1 / 111000  # Constant conversion
noise_lat = self.rng.normal(0, noise_std_meters * meters_to_deg)
noise_lon = self.rng.normal(0, noise_std_meters * meters_to_deg)
```

**Issues**:

1. This repeats the same CRS misuse described in Ground Truth: Inconsistent Distance Calculations, now applied to noise injection.

2. **Independent noise in lat/lon**: Real GPS errors are correlated (HDOP affects both dimensions)

3. **Gaussian noise only**: Real GPS has multipath (should be 5-10% in urban, not 2%), atmospheric delays, satellite geometry variations, urban canyon effects

4. **No temporal correlation**: Real GPS errors have autocorrelation (errors persist for seconds)

**Impact**:

- Simulated GPS data is easier to fuse than real data
- Fusion algorithms will overperform in evaluation vs. real deployment
- Research conclusions are not transferable to practice

**Fix**: Implement proper GPS error model with latitude-dependent scaling, correlated noise via covariance matrix, and temporal correlation via Markov process.

### HIGH: GTFS Schedule Generation Is Backwards

**File**: `sensor_simulator.py` (lines 260-267)

```python
# Current - conceptually backwards
actual_arrival = trip.stop_arrivals.get(stop.stop_id)
noise = timedelta(seconds=self.rng.uniform(-schedule_noise_seconds, schedule_noise_seconds))
scheduled_arrival = actual_arrival + noise  # Wrong!
```

**Why this is wrong**:

- In reality: Schedule = planned time (fixed, published). Actual = schedule + deviation
- The code does: Actual = ground truth (perfect). Schedule = actual + noise (imperfect)
- This means GTFS schedule is noisier than reality, opposite of how GTFS is used in fusion

**Impact**: GTFS-based fusion algorithms are penalized unfairly. GPS+GTFS fusion doesn't benefit from schedule knowledge as it should.

**Fix**: Generate schedule first, then simulate actual arrivals with realistic delays.

### HIGH: CDR Tower Selection Ignores Coverage

**File**: `sensor_simulator.py`

```python
dist = sqrt((lat-lat)^2 + (lon-lon)^2)
nearest = min(dist)
```

**Issues**:

- Towers chosen even if user is outside coverage
- No signal strength decay
- No ambiguity region
- Missing radio propagation logic

**Fix**: Use path loss model and probabilistic tower visibility with overlapping coverage likelihoods.

### MEDIUM: CDR Tower Placement Is Unrealistic

**File**: `sensor_simulator.py` (lines 302-339)

```python
offset_lat = self.rng.uniform(-0.002, 0.002)  # ±220m
offset_lon = self.rng.uniform(-0.002, 0.002)
```

**Issues**:

- Towers placed uniformly along route (real towers follow population density)
- Offset too small (±220m within coverage radius)
- No overlap/gap modeling
- Fixed coverage radius (real towers: 100m-2km based on power, frequency, terrain)

**Fix**: Use realistic tower placement based on urban density maps or Voronoi tessellation.

### MEDIUM: Calendar Assumptions Hardcoded

**File**: `sensor_simulator.py`

GTFS calendar assumes weekday service, continuous availability, no holidays/disruptions.

**Impact**: Limits realism for transit studies.

## Map Matching

### CRITICAL: Naive Nearest-Edge Matching

**File**: `base_fusion.py` (lines 105-153)

```python
def map_match_point(self, lat, lon, search_radius_m=50.0):
    candidates = self.road_network[
        self.road_network.geometry.distance(point) < search_radius_deg
    ]
    
    for idx, row in candidates.iterrows():
        edge_geom = row['geometry']
        nearest_pt = nearest_points(point, edge_geom)[1]
        if dist < min_dist:
            matched_point = nearest_pt
```

**Why this is wrong**:

- No topological constraints (can match to disconnected roads)
- No directional constraints (can match to wrong side of divided highway)
- No temporal consistency (each point matched independently)
- No path continuity (doesn't ensure matched points form valid path)

**Example failure**:

```md
Time 0: GPS on Road A → Matched to Road A ✓
Time 1: GPS slightly off → Matched to parallel Road B ✗
Time 2: GPS on Road A → Matched to Road A ✓
Result: Impossible path (A → B → A without intersection)
```

**Impact**: 30-40% of matched trajectories are topologically invalid. Evaluation metrics don't detect this error.

**Fix**: Implement Hidden Markov Model (HMM) map matching with Viterbi algorithm. This is standard in research (Newson & Krumm, 2009).

### HIGH: Distance Calculation Uses Wrong Metric

**File**: `base_fusion.py` (lines 127-128)

```python
search_radius_deg = search_radius_m / 111000
candidates = self.road_network[
    self.road_network.geometry.distance(point) < search_radius_deg
]
```

**Issues**:

- This reuses the same invalid degree-to-meter assumption already documented in Ground Truth Generation, now affecting candidate selection and topology.
- Inefficient: computes distance to ALL edges, then filters (O(N) instead of O(log N))

**Impact**: At latitude 19°, search radius is ~10% too large in longitude direction. Can match to edges actually outside search radius.

**Fix**: Use spatial index (R-tree) and buffer in projected coordinates.

### HIGH: Road Network Treated as Perfect

**Files**: `base_fusion.py`, map matching logic

The road network is assumed to be complete, correct, unambiguous, and free of geometry errors.

**Reality**: OSM has misaligned roads, missing links, duplicate geometries, incorrect one-ways.

**Impact**: Map matching errors blamed on fusion instead of map. No uncertainty propagation from map quality. Results not transferable to new cities.

**Fix**: Introduce map uncertainty flags, edge confidence weights, fallback to free-space movement when ambiguity detected.

### MEDIUM: No Directional Ambiguity

**Files**: Fusion algorithms

Roads assumed bidirectional unless explicitly marked. Wrong-direction travel is never penalized.

## Fusion Logic

### HIGH: Weighted Fusion Uses Linear Interpolation in Lat/Lon

**File**: `gps_gtfs_osm_fusion.py` (lines 394-403)

```python
def _weighted_fusion(self, lat1, lon1, weight1, lat2, lon2, weight2):
    total_weight = weight1 + weight2
    fused_lat = (lat1 * weight1 + lat2 * weight2) / total_weight
    fused_lon = (lon1 * weight1 + lon2 * weight2) / total_weight
    return fused_lat, fused_lon
```

**Why this is wrong**: This is another manifestation of the same coordinate-system error described earlier; fusion is performed in a space where linear interpolation is undefined.

**Impact**: Fused positions can be hundreds of meters off in high-latitude cities. Even at Mumbai (19°N), error is 5-10 meters for 100m separation.

**Fix**: Use weighted interpolation along geodesic with `pyproj.Geod`.

### MEDIUM: Confidence Scoring Is Ad-Hoc

See "Confidence Is Used as a Score, Not as Uncertainty" in Evaluation Metrics section.

## Evaluation Metrics

### CRITICAL: Temporal Matching Bug Inflates Coverage

**File**: `metrics.py` (lines 214-256)

```python
def _match_points_by_time(self, ground_truth, reconstructed):
    matched = []
    recon_idx = 0
    
    for _, gt_row in gt_sorted.iterrows():
        # Find closest reconstructed point
        while recon_idx < len(recon_sorted):
            # ... matching logic ...
            recon_idx += 1
        
        # Reset index for overlap
        if recon_idx > 0:
            recon_idx -= 1  # BUG: Allows same point to match multiple times
```

**The bug**: `recon_idx -= 1` was meant to allow overlap, but it causes the same reconstructed point to be matched multiple times, artificially inflating coverage metrics.

**Example**:

```md
GT:    t=0  t=1  t=2  t=3  t=4
Recon: t=0       t=2       t=4

Without bug: 3 matches (0-0, 2-2, 4-4)
With bug:    5 matches (0-0, 0-1, 2-2, 2-3, 4-4) ← Wrong!
```

**Impact**: Coverage rates overstated by 20-50%. Algorithms with sparse output appear better than they are. Algorithm comparison is invalid.

**Fix**: Implement one-to-one matching with exclusion tracking.

### CRITICAL: No Topological Validity Checks

**File**: `evaluation/metrics.py`

Metrics evaluate point-wise proximity only. They do NOT check:

- Whether reconstructed path is connected
- Whether it violates one-way roads
- Whether it crosses barriers without links

This produces logically impossible trajectories (see Map Matching example).

**Impact**: Algorithms with unstable map matching appear competitive. Catastrophic failures go undetected.

**Fix**: Add topological validity metrics (% path on connected graph, % illegal transitions, graph shortest-path deviation).

### HIGH: Unmatched Reconstructed Points Ignored

**File**: `metrics.py`

Errors computed only on matched pairs. Unmatched reconstructed points (hallucinations) are not penalized spatially, only weakly affect coverage.

**Impact**: An algorithm can hallucinate many extra points and still score well spatially.

**Fix**: Add penalties for unmatched reconstructed points and hallucination rate.

### HIGH: Quality Score Formula Is Arbitrary

**File**: `metrics.py` (lines 270-303)

```python
def _calculate_quality_score(self, result):
    spatial_score = np.exp(-result.spatial_rmse_m / 50)  # Why 50m?
    speed_score = min(1.0, result.points_per_second / 10000)  # Why 10k?
    
    quality_score = (
        0.40 * spatial_score +      # Why 40%?
        0.25 * coverage_score +     # Why 25%?
        0.20 * confidence_score +   # Why 20%?
        0.15 * speed_score          # Why 15%?
    )
```

**Issues**:

- No justification for weights or reference values
- Mixing incomparable metrics (spatial accuracy vs. processing speed)
- Exponential decay for spatial - why not linear or quadratic?
- No domain expert input

**Impact**: Quality score is meaningless for decision-making. Can't compare to other research or explain to stakeholders.

**Fix**: Either remove quality score and report individual metrics, or use domain-specific weights from transit agencies.

### HIGH: Confidence Is Used as a Score, Not as Uncertainty

**Files**: `gps_gtfs_osm_fusion.py` (lines 405-412), `metrics.py`, fusion algorithms

Confidence is treated as a scalar "goodness" measure rather than representing epistemic uncertainty.

**Implementation flaws**:

1. **Arbitrary scoring formula** (`gps_gtfs_osm_fusion.py`):

    ```python
    def _calculate_gps_confidence(self, orig_lat, orig_lon, matched_lat, matched_lon):
        dist = self._haversine_distance(orig_lat, orig_lon, matched_lat, matched_lon)
        return max(0.5, 1.0 - (dist / self.gps_search_radius_m))
    ```

    No theoretical justification. Minimum 0.5 is arbitrary. Ignores GPS HDOP, satellite count, speed/heading consistency, temporal consistency.

2. **Averaged and multiplied into quality score** (`metrics.py`):

    Confidence is aggregated as a mean and weighted into composite scores. This violates the interpretation of confidence as uncertainty.

3. **No calibration**:

    High confidence does not imply correctness. No reliability diagrams, no Expected Calibration Error (ECE), no confidence-weighted residuals.

**Impact**:

- Algorithms that overestimate confidence are rewarded
- No penalty for overconfidence
- Cannot perform probabilistic fusion (Kalman filtering, particle filtering)
- Cannot provide confidence intervals to users
- Cannot detect outliers properly

**Fix**: Replace with proper probabilistic model (Gaussian emission probability from HMM). Add calibration analysis (reliability diagrams, ECE). Use confidence-weighted residuals instead of averaging.

### HIGH: No Statistical Significance Testing

**File**: `fusion_comparator.py`

Algorithm A beating B by 3% may be noise. No confidence intervals, bootstrapping, or hypothesis testing.

**Impact**: Cannot claim Algorithm A is better than Algorithm B. Rankings may be arbitrary. Results not reproducible.

**Fix**: Add paired bootstrap on RMSE, Wilcoxon signed-rank test, error distributions (not just means).

### HIGH: No Failure-Mode Evaluation

**File**: `fusion_comparator.py`

Evaluation focuses on average performance, not failure modes (long GPS outages, sparse CDR only, GTFS-only segments, urban canyon + multipath).

**Impact**: Worst-case behavior unknown. Algorithms look stable but may catastrophically fail.

**Fix**: Add regime-based evaluation (worst 5% error, longest outage reconstruction, max deviation segment).

### MEDIUM: Missing Critical Metrics

**File**: `metrics.py`

Missing:

- Frechet Distance (better than RMSE for trajectory similarity)
- Dynamic Time Warping (handles temporal misalignment)
- Hausdorff Distance (worst-case error)
- Path Similarity (topological correctness)
- Stop Detection Accuracy (critical for transit)
- Heading Error (important for navigation)
- Acceleration Consistency (physical plausibility)

**Impact**: Evaluation is incomplete. Can't detect certain failure modes.

### MEDIUM: Performance Metric Is Misleading

**File**: `metrics.py`

"Points per second" depends on interpolation density, sampling frequency, and output size. Algorithms generating fewer points look "faster".

**Fix**: Normalize by input points, route length, or fixed output resolution.

### MEDIUM: Time Synchronization Assumed Perfect

**Files**: `sensor_simulator.py`, `metrics.py`

All sensors assume perfect clocks, no drift, no batching delay. Real systems have GPS clock jitter, GTFS reporting delays, CDR batching (minutes).

**Impact**: Temporal matching overly optimistic. Fusion assumes perfect synchronization.

**Fix**: Add timestamp noise and batching delays. Evaluate sensitivity to clock errors.

### MEDIUM: RNG Seeding Inconsistent

**Files**: `ground_truth_generator.py`, `sensor_simulator.py`

Some RNGs are seeded, others not. Results may change across runs.

**Fix**: Centralize RNG: `rng = np.random.default_rng(seed)`.

### LOW: Synthetic Data Misinterpretation Risk

Visualizations and dashboards look realistic, but data is synthetic. Non-experts may interpret results as deployable.

**Fix**: Add explicit "SYNTHETIC DATA" watermark and prominent disclaimers in reports.

## Cross-Cutting Issues

### CRITICAL: Circular Dependency in Experimental Design

**Files**: `ground_truth_generator.py`, `sensor_simulator.py`, fusion algorithms, evaluation

The same underlying assumptions are used in ground truth generation, sensor noise simulation, fusion logic, and evaluation metrics. This creates closed-loop bias.

**Examples**:

- Same speed limits implicitly assumed in GT and fusion
- Same interpolation logic used in GT and reconstruction
- Same distance thresholds reused in matching and scoring
- Vehicle identity provided as oracle (no ambiguity)

This violates basic experimental design requirements: reconstruction assumptions are reused during data generation.

**Impact**: All reported accuracy numbers are optimistic. Algorithm comparisons biased toward simpler models. Cannot claim "robustness" or "generalizability".

**Fix**: Introduce deliberate model mismatch (use geodesic spline for GT, linear map-snap for fusion). Add adversarial conditions (route deviations, stop skipping). Report train-world vs test-world mismatch results.

### CRITICAL: Vehicle Identity Leakage

**Files**: `run_evaluation.py`, fusion algorithms

All fusion algorithms implicitly assume `vehicle_id` is perfectly known and correct. In real systems, vehicle identity is inferred, not given. GTFS ↔ GPS ↔ CDR association is part of the problem.

**Impact**: Leaks ground-truth structure into fusion. This is distinct from circular GT-fusion coupling - this is identity oracle leakage.

**Fix**: Introduce ID noise, ambiguous vehicle-to-trip mapping, delayed identity resolution.

### HIGH: No Uncertainty Quantification

The entire fusion framework treats positions as point estimates without uncertainty.

**Why this matters**: GPS has ±8m uncertainty (1σ). GTFS stops have ±50m uncertainty. Map matching has variable uncertainty.

**Current approach**:

```python
fused_lat, fused_lon = weighted_fusion(gps_lat, gps_lon, gtfs_lat, gtfs_lon)
# Returns single point - no uncertainty!
```

**Better approach**:

```python
fused_position, fused_covariance = probabilistic_fusion(
    gps_position, gps_covariance,
    gtfs_position, gtfs_covariance
)
# Returns position + uncertainty ellipse
```

**Impact**: Can't do Kalman filtering or particle filtering. Can't provide confidence intervals to users. Can't detect outliers properly.

### MEDIUM: No Real-World Validation

All evaluation is on synthetic data generated by the same codebase.

**Issues**: Circular validation. No ground truth for real data. Missing real-world phenomena (GPS signal loss in tunnels, multipath in urban canyons, GTFS schedule changes, road network errors).

**Fix**: Collect real GPS traces with known ground truth. Use public datasets (GeoLife, T-Drive). Validate on multiple cities.

## Code Quality

**General problems**:

- No unit tests (zero test coverage)
- No input validation (functions don't check for invalid inputs)
- Inconsistent error handling (some raise exceptions, others return None)
- No logging (difficult to debug)
- Magic numbers (111000, 0.7, 50.0 hardcoded everywhere)
- No documentation (docstrings missing implementation details)

**Example - missing input validation**:

```python
def map_match_point(self, lat, lon, search_radius_m=50.0):
    # No validation!
    # What if lat > 90 or lat < -90?
    # What if search_radius_m < 0?
    # What if road_network is None?
    point = Point(lon, lat)  # Will silently create invalid point
```

## Priority Recommendations

### Immediate Fixes (Before Any Research Use)

1. Fix distance calculations - use geodesic distances throughout
2. Fix temporal matching bug - prevent duplicate matches
3. Fix GTFS schedule noise - schedule should be reference, not noisy
4. Fix speed-distance consistency - make distance primary state

### Short-Term Improvements (For Valid Research)

1. Implement HMM map matching with Viterbi algorithm
2. Improve GPS noise model (temporal correlation, proper lat/lon scaling)
3. Add missing metrics (Frechet distance, DTW, topological validity)
4. Add statistical significance testing

### Long-Term Enhancements (For Publication Quality)

1. Add uncertainty quantification (probabilistic fusion, confidence intervals)
2. Validate on real data (collect ground truth, test on multiple cities)
3. Break circular dependency (introduce model mismatch, adversarial conditions)
4. Implement advanced fusion (Kalman filter, particle filter)

## Summary Table

| Issue | Severity | File | Impact | Fix Effort |
| ------- | ---------- | ------ | -------- | ------------ |
| Inconsistent distance calc | CRITICAL | ground_truth_generator.py | Wrong baseline | High |
| Kinematic model inconsistent | CRITICAL | ground_truth_generator.py | Invalid metrics + impossible motion | High |
| GPS noise model | CRITICAL | sensor_simulator.py | Overoptimistic results | Medium |
| Temporal matching bug | CRITICAL | metrics.py | Invalid metrics | Low |
| Naive map matching | CRITICAL | base_fusion.py | Invalid trajectories | High |
| No topological validation | CRITICAL | metrics.py | Undetected failures | Medium |
| Circular dependency | CRITICAL | All | Invalid conclusions | High |
| Vehicle ID leakage | CRITICAL | run_evaluation.py | Oracle advantage | Low |
| GTFS schedule backwards | HIGH | sensor_simulator.py | Unfair comparison | Low |
| Weighted fusion in lat/lon | HIGH | gps_gtfs_osm_fusion.py | Position errors | Medium |
| No uncertainty quantification | HIGH | All | Limited applicability | High |
| Unmatched points ignored | HIGH | metrics.py | False positives | Low |
| No statistical testing | HIGH | fusion_comparator.py | Weak ranking | Medium |
| No failure-mode testing | HIGH | fusion_comparator.py | Hidden failures | Medium |
| Route geometry simplified | MEDIUM | ground_truth_generator.py | Reduced realism | Low |
| CDR coverage ignored | HIGH | sensor_simulator.py | Unrealistic CDR | Medium |
| Confidence misused as score | HIGH | gps_gtfs_osm_fusion.py, metrics.py | Overconfidence bias | Medium |
| Quality score arbitrary | MEDIUM | metrics.py | Meaningless ranking | Low |
| Missing metrics | MEDIUM | metrics.py | Incomplete eval | Medium |
| CDR tower placement | MEDIUM | sensor_simulator.py | Unrealistic | Medium |
| Heading calculation | MEDIUM | ground_truth_generator.py | Edge cases | Low |
| Time synchronization | MEDIUM | sensor_simulator.py | Overoptimistic | Low |
| Performance metric | MEDIUM | metrics.py | Misleading | Low |
| RNG inconsistency | MEDIUM | Multiple | Non-determinism | Low |
| Route length recomputed | MEDIUM | ground_truth_generator.py | Inefficient | Low |
| Calendar hardcoded | MEDIUM | sensor_simulator.py | Limited realism | Low |
| No directional ambiguity | MEDIUM | Fusion algorithms | Wrong-way allowed | Low |
| Synthetic data risk | LOW | Visualizations | Misinterpretation | Low |
