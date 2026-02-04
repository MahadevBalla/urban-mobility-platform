# Zone Generation Module – Fixes Implemented

This document lists code changes that address specific items from the audit.

## Audit Issues

### C1: Population Proxy Has No Scientific Basis

- **Original issue**: Population estimated as `area × floors` without empirical basis.
- **What was changed**:
  - Code now uses `POPULATION_MODEL` with `m2_per_person` inputs.
  - Added logic to infer building type from OSM tags.
- **Files involved**: `feature_engineer.py`, `config.py`
- **Why this fix is correct**: The code now follows the coefficient-based population model referenced in the audit.

### C2: Employment Proxy Weights Are Arbitrary

- **Original issue**: Employment weights were hardcoded without normalization.
- **What was changed**:
  - Implemented area normalization for employment density.
  - Added z-score normalization logic.
- **Files involved**: `feature_engineer.py`, `config.py`
- **Why this fix is correct**: The calculation now includes the normalization step requested in the audit.

### C3: Barrier Detection Ignores Crossing Infrastructure

- **Original issue**: Major roads were treated as absolute barriers regardless of crossings.
- **What was changed**:
  - Added code to identify bridges, tunnels, and crossing nodes.
  - Implemented a permeability score calculation.
- **Files involved**: `barrier_detector.py`, `config.py`
- **Why this fix is correct**: The code now includes logic to filter barriers based on crossing features.

### C5: Place-Based OSM Geocoding Fails When Polygon Geometry Is Unavailable

- **Original issue**: Some valid place-name inputs could not be geocoded to a Polygon or MultiPolygon, causing OSMnx to raise an exception and halt the pipeline.
- **What was changed**:
  - Added explicit bounding-box (bbox) input support as an alternative boundary definition.
  - Updated the zone generation pipeline to accept bbox-defined study areas.

- **Files involved**: `osm_network.py`, `zone_generator.py`, `app.py`, `config.py`

- **Why this fix is correct**: Bounding boxes provide a reliable and explicit spatial boundary when polygonal geometries are unavailable or inconsistent in OpenStreetMap.

### M2: Network Distance Matrix is O(N² · E log V)

- **Original issue**: Distance computation was slow for large networks.
- **What was changed**:
  - Implemented parallel execution for Dijkstra paths.
- **Files involved**: `skim_computer.py`, `zone_generator.py`
- **Why this fix is correct**: The implementation now uses the `joblib` parallel backend.

### M3: Feature Normalization Breaks with Outliers

- **Original issue**: Min-max normalization was sensitive to outliers.
- **What was changed**:
  - Replaced min-max logic with z-score calculation.
  - Added value clipping.
- **Files involved**: `region_merger.py`
- **Why this fix is correct**: The code now uses the z-score method specified in the audit.

### M4: Land Use Classification is Overly Simplistic

- **Original issue**: Classification relied on relative quantiles.
- **What was changed**:
  - Replaced quantile logic with absolute threshold checks.
- **Files involved**: `feature_engineer.py`
- **Why this fix is correct**: The classification logic now uses the fixed thresholds defined in `config.py`.

### M5: Centroid Weighting is Not Implemented

- **Original issue**: Centroids were calculated geometrically.
- **What was changed**:
  - Added `_activity_weighted_centroid` function.
  - Code extracts building areas to use as weights.
- **Files involved**: `centroid_connector.py`
- **Why this fix is correct**: The centroid calculation now inputs building area weights.

### m1: Magic Numbers Not Documented

- **Original issue**: Parameters were hardcoded in logic files.
- **What was changed**:
  - Parameters moved to `ZoneGenConfig` class.
- **Files involved**: `config.py`
- **Why this fix is correct**: Configuration values are now centralized.

### m3: Test Coverage Gaps

- **Original issue**: Missing validation tests.
- **What was changed**:
  - Added validation checks to test files.
- **Files involved**: `tests/*.py`
- **Why this fix is correct**: Tests now include asserts for the new features.

## Other Recent Changes (Not in Audit Issue List)

The following changes were made but are not mapped to an issue in `NEW_ISSUES_IDENTIFIED.md`:

- **C7**: Set parallel backend to "threading".
- **C8**: Renamed `proxy_employment` columns.
- **C9**: Changed default skim method to network-based.
