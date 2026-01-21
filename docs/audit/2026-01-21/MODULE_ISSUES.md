# Module-Level Issues

This document lists concrete implementation issues by module, including impact and corrective direction.

Severity reflects impact on correctness and validity, not code style.

Detailed file-level analyses and proposed refactors are documented in [appendices/](./appendices/).

## A. Data Fusion

### A1. Ground Truth Generation

- **Issues**

  - Distance computed geodesically, but interpolation done in lat/lon
  - Speed generated independently from distance
  - Distance advanced as `distance += speed`, violating kinematics
  - Stops modeled as exact points with zero spatial extent
  - Routes simplified for controlled testing (reduces realism)

- **Impact**

  - Distance–speed inconsistency
  - Invalid speed RMSE
  - Kinematic models cannot be validated

- **Direction**

  - Make distance the primary state
  - Derive speed from finite differences
  - Perform interpolation in a metric CRS

### A2. Sensor Simulation

- **GPS**

  - Constant degree-to-meter scaling
  - Independent Gaussian noise
  - No temporal correlation or multipath

- **GTFS**

  - Schedule generated as noisy version of actual
  - Real-world causality is reversed

- **CDR**

  - Tower coverage and signal decay ignored

- **Impact**

  - Fusion performance overstated
  - GTFS contribution underestimated

### A3. Map Matching

- **Issues**

  - Nearest-edge matching per point
  - No topology, direction, or temporal constraints
  - Euclidean distance in lat/lon

- **Impact**

  - Disconnected or illegal paths score well
  - Errors not detected by metrics

- **Direction**

  - Topology-aware matching (e.g., HMM-style)
  - Metric distance computation

### A4. Evaluation

- **Issues**

  - Temporal matching allows reuse of reconstructed points
  - No penalties for unmatched or hallucinated points
  - No topological or kinematic validation
  - No statistical significance testing

- **Impact**

  - Coverage inflated by 20–50%
  - Rankings unreliable

## B. Zone Generation

### B1. Geometry and Area

- **Issues**

  - Areas computed in WGS84
  - Same CRS misuse as fusion module

- **Impact**

  - Population proxies meaningless
  - Density-based decisions invalid

### B2. Feature Engineering

- **Issues**

  - Road lengths computed inefficiently and incorrectly
  - Employment weights uncalibrated

- **Impact**

  - Slow execution
  - Unreliable zone attributes

### B3. Region Merging

- **Issues**

  - Cosine similarity used for spatial features
  - No compactness constraint
  - Unbounded BFS growth

- **Impact**

  - Heterogeneous and poorly shaped zones

### B4. Skim Computation

- **Issues**

  - O(N²) shortest-path computation
  - Incorrect fallback distance approximation

- **Impact**

  - Poor scalability
  - Inaccurate skims

## C. Testing and Validation Gaps

Currently missing:

- real-data validation
- cross-city tests
- failure-mode testing
- uncertainty propagation
- unit and integration tests

This limits confidence in all reported results.
