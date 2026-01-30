# Zone Generation Module – Post-Resolution Audit

## Overview

This document outlines the remaining methodological and structural issues identified in the Zone Generation Module following the resolution of the initial audit. While the system demonstrates sound engineering and correctness in coordinate handling, several fundamental research validity and scalability issues remain. These issues must be addressed to meet the standards for peer-reviewed publication or a PhD defense.

## Summary Table

| ID | Issue | Severity | Category |
| --- | --- | --- | --- |
| C1 | Population Proxy Has No Scientific Basis | CRITICAL | Research Validity |
| C2 | Employment Proxy Weights Are Arbitrary | CRITICAL | Research Validity |
| C3 | Barrier Detection Ignores Crossing Infrastructure | CRITICAL | Algorithmic Design |
| C4 | No Validation Against Ground Truth | CRITICAL | Validation |
| C5 | Place-Based OSM Geocoding Fails When Polygon Geometry Is Unavailable | CRITICAL | Input Robustness |
| M1 | Quadratic Adjacency Graph Construction | HIGH | Scalability |
| M2 | Network Distance Matrix is O(N² · E log V) | HIGH | Scalability |
| M3 | Feature Normalization Breaks with Outliers | HIGH | Algorithmic Design |
| M4 | Land Use Classification is Overly Simplistic | HIGH | Research Validity |
| M5 | Centroid Weighting is Not Implemented | HIGH | Algorithmic Design |
| m3 | Test Coverage Gaps | MEDIUM | Validation |
| m1 | Magic Numbers Not Documented | LOW | Code Quality |
| m2 | Inconsistent Error Handling | LOW | Code Quality |

## 1. Research Validity & Methodology

### CRITICAL: Population Proxy Has No Scientific Basis

**Category**: Research Validity

- #### Description

  The current implementation estimates population based on a linear relationship between floor area and building levels (`area_m2 * levels`). It uses a single calculation for all building types and lacks occupancy rates, vacancy adjustments, or person-per-m² coefficients.

- #### Why This Matters

  Population is the primary input driving the zone merging algorithm. An invalid population proxy results in structurally incorrect zones, rendering the "population-balanced" claim invalid.

- #### Evidence

  `feature_engineer.py:250-253`: Calculation of `proxy_capacity` uses raw area and levels without differentiation.

- #### Impact on Claims

  Invalidates claims about population homogeneity and makes generated zones incomparable to census-based TAZs. Peer review would likely reject the methodology without calibration.

### CRITICAL: Employment Proxy Weights Are Arbitrary

**Category**: Research Validity

- #### Description

  The employment estimation uses hardcoded weights (e.g., office=250, commercial=150) that lack empirical justification or citation. These weights are multiplied by building area without normalization.

- #### Why This Matters

  Employment density drives Central Business District (CBD) identification, which triggers different merging rules. Arbitrary weights lead to arbitrary CBD boundaries and incorrect zone structures.

- #### Evidence

  `feature_engineer.py:262-270`: Hardcoded `employment_weights` dictionary without citation.

- #### Impact on Claims

  Research validity fails as there is no basis for the numbers used. Reviewers will question the origin of these parameters.

### HIGH: Land Use Classification is Overly Simplistic

**Category**: Research Validity

- #### Description

  Land use classification relies on relative quantiles (e.g., `pop > pop.quantile(0.5)`), causing the classification to change depending on the study area. It does not account for POI diversity or mixed-use entropy.

- #### Why This Matters

  A cell could be classified as "commercial" in a downtown area but "low_density" in a suburb, and merging them can result in correct individual classifications becoming incorrect ("mixed").

- #### Evidence

  `feature_engineer.py:336-373`: Conditional logic using relative quantiles for classification.

- #### Impact on Claims

  The instability of classification across different study areas undermines the reproducibility and robustness of the zoning logic.

## 2. Algorithmic Design & Correctness

### CRITICAL: Barrier Detection Ignores Crossing Infrastructure

**Category**: Algorithmic Design

- #### Description

  The system treats all motorways and railways as absolute barriers, ignoring bridges, tunnels, and pedestrian crossings. It applies a fixed 50m buffer without transportation literature justification.

- #### Why This Matters

  TAZ boundaries should reflect travel impedance, not just physical presence. Treating permeable infrastructure as absolute barriers leads to over-fragmentation, especially in grid-cities with frequent crossings.

- #### Evidence

  `barrier_detector.py:231-326`: Implementation lacks logic to check for bridges, tunnels, or permeability.

- #### Impact on Claims

  The resulting zones do not match standard planning practice. For example, a city like Manhattan would be incorrectly fragmented along its waterfront highways.

### HIGH: Place-Based OSM Geocoding Fails When Polygon Geometry Is Unavailable

**Category**: Input Robustness

- #### Description

  - The zone generation pipeline relies on OpenStreetMap place-name geocoding to obtain a polygon or multipolygon boundary.  
  - For several valid inputs (e.g., "Connaught Place, Delhi, India"), Nominatim returns a geocoding result that does **not contain a valid Polygon or
  MultiPolygon geometry**.
  - As a result, `osmnx.geocode_to_gdf()` raises a TypeError and the pipeline terminates before zone generation can begin.

- #### Why This Matters

  - The pipeline assumes a polygonal study boundary for grid generation, network extraction, and barrier detection.  
  - When a place is represented in OSM without a polygon boundary, place-name-based geocoding becomes unreliable and prevents zone generation for otherwise valid study areas.

- #### Evidence

  Observed during zone generation for inputs such as:
  - "Mumbai, Maharashtra, India", "Connaught Place, Delhi, India"
  
    > “Nominatim did not geocode query to a geometry of type (Multi)Polygon”

- #### Impact on Claims

  Limits the robustness of the system to inputs with well-defined polygon boundaries, reducing applicability across diverse urban contexts.

### HIGH: Feature Normalization Breaks with Outliers

**Category**: Algorithmic Design

- #### Description

  The region merging process uses min-max normalization, which is highly sensitive to outliers. A single cell with extreme values (e.g., an airport or university) can compress the range for all other cells, making the distance metric meaningless.

- #### Why This Matters

  If one cell has 100x the population of others, the normalized values for the majority of cells will become indistinguishable (e.g., crowding near 0.0), destroying the effectiveness of the similarity metric.

- #### Evidence

  `region_merger.py:194-200`: Min-max normalization logic without outlier handling.

- #### Impact on Claims

  The merging algorithm may fail to correctly group similar cells in the presence of density outliers.

### HIGH: Centroid Weighting is Not Implemented

**Category**: Algorithmic Design

- #### Description

  The code explicitly notes that activity-weighted centroids are not implemented and defaults to geometric centroids. This ignores the internal distribution of activity within a zone.

- #### Why This Matters

  Geometric centroids do not reflect the true center of activity, especially in zones with uneven development. This introduces systematic bias into skim matrices and trip distribution models.

- #### Evidence

  `centroid_connector.py:80-86`: Comments admit that `centroid = zone.geometry.centroid` is used instead of a weighted approach.

- #### Impact on Claims

  Skim matrices derived from these centroids will be inaccurate, affecting downstream transport modeling steps.

## 3. Validation & Testing

### CRITICAL: No Validation Against Ground Truth

**Category**: Validation

#### Description

  The pipeline lacks comparison against official TAZs from planning organizations or census tracts. Current validation metrics (Polsby-Popper, CV) are purely geometric and do not assessing functional travel behavior homogeneity.

#### Why This Matters

  Without empirical validation, there is no evidence that the generated zones are suitable for travel demand modeling. Geometric compactness is not a proxy for functional correctness.

#### Evidence

  Entire pipeline lacks a benchmarking step against ground truth data.

#### Impact on Claims

  The system cannot be claimed as a valid alternative to manual zoning without empirical proof. This is a primary failure point for a PhD defense.

### MEDIUM: Test Coverage Gaps

**Category**: Validation

#### Description

  While unit test coverage is good, there are missing integration tests for the full pipeline with real OSM data. There are also no tests for H3 version compatibility, large-scale performance, or edge cases like island zones.

#### Why This Matters

  The absence of integration and stress tests leaves the system vulnerable to failures in real-world scenarios or when scaling up to larger cities.

#### Evidence

  Identified gaps in the test suite regarding integration and edge cases.

#### Impact on Claims

  Reduces confidence in the system's robustness and production readiness.

## 4. Scalability & Performance

### HIGH: Quadratic Adjacency Graph Construction

**Category**: Scalability

#### Description

  The adjacency graph construction uses a nested loop with costly geometry operations (`touches` or `intersects`), leading to O(N·k·T) complexity.

#### Why This Matters

  For large datasets (e.g., 50,000 cells), this operation becomes a significant bottleneck, taking minutes to run and blocking city-scale analysis.

#### Evidence

  `region_merger.py:134-164`: Nested loops for spatial relationship checking.

#### Impact on Claims

  Limits the tool's applicability to neighborhood or district-scale analysis, preventing full-city deployments.

### HIGH: Network Distance Matrix is O(N² · E log V)

**Category**: Scalability

#### Description

  The skim computer calculates shortest paths independently for all zone pairs or unique nodes, resulting in high computational cost for large numbers of zones.

#### Why This Matters

  Calculating network-based skims for thousands of zones becomes impractical (taking hours), forcing a fallback to Euclidean distance which is less accurate.

#### Evidence

  `skim_computer.py:156-203`: Loop structure for Dijkstra path calculations.

#### Impact on Claims

  Prevents the effective use of network-based skims for larger cities, a key requirement for accurate transport modeling.

## 5. Code Quality & Maintenance

### LOW: Magic Numbers Not Documented

**Category**: Code Quality

#### Description

  Several key parameters (e.g., `target_hex_count=7500`, `sliver_area_fraction=0.05`) are hardcoded without documentation explaining their rationale.

#### Why This Matters

  Lack of documentation makes it difficult for future researchers to understand the basis for these choices or how to adjust them for different contexts.

#### Evidence

  `hex_grid.py:55`, `config.py:47`, `config.py:41`.

#### Impact on Claims

  Reduces the reproducibility and transparency of the system configuration.

### LOW: Intentional Error Suppression for API Resilience (m2)

**Category**: Code Quality / Reliability

#### Description

The module employs broad `try-except` blocks around OSM extraction methods (e.g., `osm_network.py`), which catch exceptions and return empty DataFrames rather than propagating errors.

#### Why This Matters

**Benefit**: This design prevents total pipeline failure during long-running batch processes when individual API calls fail due to timeouts or rate limits (common with public OSM/Overpass APIs).
**Downside**: It can lead to "silent failures" where zones are generated with missing features (e.g., no roads or buildings) without alerting the user, potentially compromising downstream analysis if logs are ignored.

#### Evidence

`osm_network.py`: Consistent pattern of `try: ... except Exception: return empty_gdf` across extraction methods.

#### Impact on Claims

Does not invalidate research claims but requires strict log monitoring during data generation to ensure data completeness.
