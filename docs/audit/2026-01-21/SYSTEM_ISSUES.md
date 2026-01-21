# System-Level Issues

This document describes cross-cutting issues that affect multiple modules and invalidate current results if left unresolved.

## 1. Coordinate System Misuse

Across both data fusion and zone generation, geographic coordinates (latitude/longitude) are treated as planar Cartesian coordinates.

Observed patterns:

- Constant `111 km / degree` conversion used everywhere
- Euclidean distance computed directly in lat/lon
- Areas computed on WGS84 geometries
- Linear interpolation performed on unprojected geometries

Why this is incorrect:

- Latitude/longitude are angular units, not linear distances
- Longitude distance varies with latitude (`≈ 111 × cos(latitude) km`)
- Euclidean formulas have no geometric meaning on the sphere

Impact:

- Distance errors of ~5–10% at Mumbai latitudes
- Area errors of several orders of magnitude (square degrees vs m²)
- Incorrect population densities, road lengths, skims, and metrics

This issue propagates through:

- ground truth generation
- sensor simulation
- map matching
- evaluation metrics
- zone feature engineering

**All spatial results contain systematic 5-10% distance errors that accumulate in downstream analyses.** While not catastrophic for all purposes, these errors compromise the validity of quantitative comparisons and absolute distance-based metrics.

## 2. Synthetic Benchmark Limitations

The system generates synthetic data and evaluates fusion algorithms using the same underlying assumptions - a characteristic common to most synthetic benchmarks. This provides useful relative comparisons but limits external validity.

Examples:

- Identical interpolation logic in ground truth and reconstruction
- Shared speed limits and distance thresholds
- Perfect vehicle identity assumed throughout
- Perfect time synchronization assumed

Effect:

- Algorithms are rewarded for matching the generator, not reality
- Performance metrics are optimistically biased
- Relative rankings may not hold on real data

This synthetic evaluation setup provides useful relative comparisons but should be validated with real-world data before making production deployment or comparative superiority claims.

> Note: While population synthesis and census-based constraints exist elsewhere in the repository, they are not currently used to constrain trajectory generation, fusion logic, or evaluation, and therefore do not break this circular dependency.

## 3. Evaluation Blind Spots

Current evaluation focuses on point-wise spatial proximity. This fails to detect structural failures such as:

- topologically invalid paths
- physically impossible motion
- wrong-way travel or barrier crossings

Metrics also lack:

- penalties for hallucinated points
- worst-case or failure-mode analysis
- statistical significance testing

As a result, reported performance can appear strong even when reconstructed trajectories are implausible.

## 4. Current Suitability

In its current integrated form, the system is suitable for:

- demonstrations
- exploratory prototyping
- visualization mockups

It is **not suitable** for:

- research conclusions
- algorithm comparison claims
- planning or policy analysis

Results should be treated as illustrative only.
