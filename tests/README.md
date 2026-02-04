# Zone Generation Test Suite — Documentation

## Overview

This directory contains the verification and regression test suite for the **Zone Generation Module** of the Urban Mobility Platform.

The purpose of this test suite is to check that the core logic of zone generation behaves as expected under controlled conditions. The tests focus on correctness, internal consistency, and protection against common implementation errors (especially spatial and CRS-related issues).

All tests are offline, deterministic, and suitable for continuous integration (CI) environments.

## Table of Contents

- [Overview](#overview)
- [Scope of the Test Suite](#scope-of-the-test-suite)
- [Design Principles](#design-principles)
- [Running the Tests](#running-the-tests)
- [pytest Configuration](#pytest-configuration-pytestini)
- [Test File Overview](#test-file-overview)
- [Notes and Limitations](#notes-and-limitations)

## Scope of the Test Suite

The test suite aims to:

- Check algorithmic correctness for individual components
- Detect CRS and unit-related bugs (degrees vs meters)
- Ensure hard constraints are enforced consistently
- Verify that the full pipeline can execute without runtime errors
- Catch regressions introduced by refactoring or dependency changes

The tests are intentionally run on **small synthetic geometries** rather than real cities.

## Design Principles

### 1. Early Failure on Invalid Inputs

The code is expected to fail clearly when given invalid inputs, such as:

- Missing or invalid CRS
- Empty or malformed GeoDataFrames
- Invalid configuration values
- Negative or nonsensical feature values

The tests check that such cases raise errors instead of silently continuing.

### 2. Logic Tested in Isolation

The test suite avoids dependence on external systems:

- No live OpenStreetMap queries
- No network access
- No database access
- No file I/O beyond temporary test outputs

External dependencies are mocked where required.  
The goal is to test **algorithmic logic**, not infrastructure reliability.

### 3. Explicit Constraints from Urban Planning Logic

Several tests encode constraints that are expected to hold for zone generation, such as:

- Zones should be spatially connected
- Zones should not cross major barriers
- Zones should satisfy basic compactness requirements
- Cells with highly dissimilar features should not be merged
- Population and area limits should be respected

These are treated as **hard checks** in the test suite.

### 4. Careful Handling of CRS and Units

Spatial computations are a common source of subtle bugs.  
The test suite explicitly checks that:

- Areas, lengths, and distances are computed in projected CRS
- Geographic CRS (lat/lon) is not used for metric calculations
- CRS is preserved or transformed intentionally
- Unit regressions are detected early

CRS-related checks are concentrated in dedicated tests.

## Running the Tests

From the repository root, run:

```bash
pytest
```

The test suite is configured to:

- Produce deterministic results
- Fail immediately on errors
- Treat deprecation warnings as errors
- Use short, readable tracebacks

## pytest Configuration (`pytest.ini`)

```ini
[pytest]
minversion = 7.0

testpaths =
    tests

pythonpath =
    .

addopts =
    -ra
    --strict-markers
    --strict-config
    --disable-warnings
    --maxfail=1
    --tb=short

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks integration tests
    db: tests that require a database
    network: tests that require network access

filterwarnings =
    error::DeprecationWarning
    error::PendingDeprecationWarning
    ignore::UserWarning
```

## Test File Overview

### `test_pipeline_smoke.py`

Checks that the complete zone generation pipeline can run end-to-end without crashing.

- All major stages are executed
- Heavy dependencies are mocked
- No numerical or planning assertions are made

This test is intended as a basic sanity check.

---

### `test_crs_regressions.py`

Contains tests specifically focused on CRS and unit correctness.

Examples include:

- Ensuring areas are positive and computed in metric CRS
- Verifying buffer operations use meters
- Checking distance outputs are in reasonable units

This file exists to prevent accidental reintroduction of CRS-related bugs.

---

### `test_hex_grid.py`

Tests the hexagonal grid generation logic.

Coverage includes:

- Automatic resolution selection
- Resolution monotonicity
- Hexagon generation and identifiers
- Area computation
- Neighborhood and distance consistency
- Error handling for invalid inputs

---

### `test_barrier_detector.py`

Tests detection and handling of physical barriers.

Coverage includes:

- Identification of road, rail, and water barriers
- Metric buffering of barriers
- Splitting grid cells along barriers
- Sliver filtering behavior
- Tagging cells by barrier proximity

These tests ensure that barriers are handled consistently.

---

### `test_feature_engineer.py`

Tests feature computation used later in zone merging.

Coverage includes:

- Road network metrics
- Distance-to-feature calculations
- Building-based population proxies
- POI-based employment proxies
- Land-use classification
- Special flags (e.g., CBD indicators)

The focus is on correctness and numerical stability.

---

### `test_osm_network_extractor.py`

Tests OSM extraction logic with all network access mocked.

Coverage includes:

- Boundary handling
- Road and POI classification logic
- Geometry coercion (e.g., polygons to centroids)
- Area and level parsing
- Output schema of extracted data

---

### `test_centroid_connector.py`

Tests centroid generation and connector logic.

Coverage includes:

- Input validation
- CRS preservation
- Geometric and activity-weighted centroid computation
- Connector length limits
- Transit stop linking

These tests ensure centroids and connectors remain spatially valid.

---

### `test_region_merger.py`

Tests the zone merging (region growing) logic.

Key checks include:

- Every cell is assigned to a zone
- Zone IDs are consistent
- Barriers are respected
- Compactness constraints are enforced
- Invalid feature values are rejected
- The algorithm terminates reliably

This file focuses on correctness of the core merging logic.

---

### `test_skim_computer.py`

Tests origin–destination (OD) skim computation.

Coverage includes:

- Euclidean distance matrix properties
- Network fallback behavior
- Travel time computation across modes
- Generalized cost matrix structure
- Shape, symmetry, and diagonal checks

The goal is to ensure skims are well-formed and interpretable.

---

### `test_zone_validator.py`

Tests the zone validation logic used for post-generation checks.

Coverage includes:

- Schema validation
- Population homogeneity (CV)
- Compactness statistics
- Area and population constraints
- Geometric connectivity
- Barrier violations
- Routing connectivity using skim matrices
- Overall pass/fail outcomes

This file checks that validation results are internally consistent.

## Notes and Limitations

This test suite does **not** attempt to:

- Validate results against real-world ground truth
- Measure performance on large cities
- Assess traffic realism or behavioral accuracy
- Replace empirical evaluation or calibration

Its role is to support correct implementation and safe iteration during development.
