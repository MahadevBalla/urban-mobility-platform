# Zone Generation Module – Fixes Implemented

This document describes the fixes implemented in response to the 2026-01-21 audit of the zone generation module.

## 1. CRS and Metric Correctness

### ZG-1: Fixed Degree-to-Meter Conversions

**Original Issue**: Distance calculations used a constant 111 km/degree conversion, ignoring latitude effects on longitude.

**What Was Changed**:

- Replaced all constant degree-to-meter conversions with proper geodesic calculations using `pyproj.Geod`
- All area and buffer operations now performed in projected CRS (UTM or EPSG:6933 fallback)
- Single CRS estimation cached and reused per module

**Files Modified**:

- `feature_engineer.py`
- `skim_computer.py`
- `hex_grid.py`
- `barrier_detector.py`
- `region_merger.py`

**Why This Is Correct**: Geodesic calculations account for Earth's curvature and latitude-dependent longitude scaling, ensuring accurate distances and areas.

### ZG-2: Fixed Building Area Calculation

**Original Issue**: Building areas calculated in WGS84 (lat/lon), yielding square degrees instead of square meters.

**What Was Changed**:

- Buildings now projected to UTM before area calculation
- Fallback to EPSG:6933 if UTM estimation fails
- Areas correctly computed in m²

**Files Modified**:

- `osm_network.py`

**Why This Is Correct**: Projected coordinate systems use meters as units, giving correct area measurements.

### ZG-5: Fixed Buffer Distance Distortion

**Original Issue**: Barrier buffering used Web Mercator (EPSG:3857), causing distance distortion at different latitudes.

**What Was Changed**:

- Buffering now performed in local UTM CRS
- Fallback to EPSG:6933 (equal-area projection)
- Buffer widths now consistent regardless of latitude

**Files Modified**:

- `barrier_detector.py`

**Why This Is Correct**: UTM provides accurate metric distances for local areas, ensuring 50m buffer is actually 50m.

## 2. Grid Generation and Geometry Handling

### ZG-3: Fixed Area Recalculation After Clipping

**Original Issue**: Hexagon areas not recalculated after clipping to boundary, causing incorrect values for boundary zones.

**What Was Changed**:

- Areas now recomputed after overlay operation
- Computation performed in metric CRS
- Added assertion to prevent silent CRS errors

**Files Modified**:

- `hex_grid.py`

**Why This Is Correct**: Clipping changes geometry shape, so areas must be recalculated to match actual geometry.

### ZG-4: Improved H3 Resolution Selection

**Original Issue**: Resolution selection based on arbitrary area thresholds (300, 1500, 5000 km²).

**What Was Changed**:

- Resolution now selected to achieve target hexagon count
- Calculation based on study area size and H3 resolution-area relationship
- Target count configurable per city

**Files Modified**:

- `hex_grid.py`

**Why This Is Correct**: Selecting resolution based on desired hex count ensures consistent grid density across different study areas.

### ZG-6: Fixed Sliver Removal Threshold

**Original Issue**: Sliver polygons filtered using absolute threshold (0.01 km²) instead of relative to hex size.

**What Was Changed**:

- Threshold now relative to median cell area (5% of median)
- Resolution-invariant filtering
- Works correctly across H3 resolutions 7–10

**Files Modified**:

- `barrier_detector.py`

**Why This Is Correct**: Relative threshold adapts to grid resolution, preventing removal of valid cells in fine grids or retention of slivers in coarse grids.

## 3. Feature Engineering

### ZG-7: Fixed Road Length Calculation

**Original Issue**: Road length calculation projected cells 6 times (once per road class), allowed polygon geometries (length = perimeter, incorrect), and used Web Mercator.

**What Was Changed**:

- Single projection to UTM performed once
- Geometries filtered to LineStrings only before length calculation
- Correct aggregation with zero-fill for cells without roads
- No repeated copies or projections

**Files Modified**:

- `feature_engineer.py`

**Why This Is Correct**: Single projection reduces memory usage and processing time. LineString filtering ensures only road segments are measured, not polygon perimeters.

### ZG-8: Made Employment Weights Configurable

**Original Issue**: Employment proxy weights (office=10, commercial=5, etc.) were hardcoded with no justification.

**What Was Changed**:

- Weights now injectable as parameter
- Default heuristic values documented and logged
- Architecture supports future calibration with census data

**Files Modified**:

- `feature_engineer.py`

**Why This Is Correct**: Parameterized weights allow future calibration without code changes. Current defaults are documented as heuristic placeholders.

## 4. Region Merging

### ZG-9: Replaced Cosine Similarity with Euclidean Distance

**Original Issue**: Cosine similarity measures angle between vectors, not magnitude. Cells with very different populations but proportional characteristics had perfect similarity.

**What Was Changed**:

- Replaced with Euclidean distance in normalized feature space
- Land-use-aware thresholds (tighter for residential, looser for CBD)
- Magnitude-sensitive metric prevents merging dissimilar cells

**Files Modified**:

- `region_merger.py`

**Why This Is Correct**: Euclidean distance considers both direction and magnitude, preventing merging of cells that differ significantly in absolute values.

### ZG-10: Replaced BFS Queue with Priority Queue

**Original Issue**: Region growing used unordered FIFO queue that could grow unbounded and processed neighbors in arbitrary order.

**What Was Changed**:

- Replaced with priority queue (heapq-based)
- Candidates sorted by similarity to region centroid
- Best-first growth ensures bounded frontier

**Files Modified**:

- `region_merger.py`

**Why This Is Correct**: Priority queue processes most similar cells first, leading to more homogeneous zones and better performance.

### ZG-11: Added Compactness Constraint

**Original Issue**: Region growing had no geometric compactness check, allowing long thin zones.

**What Was Changed**:

- Polsby-Popper compactness metric implemented
- Threshold (0.2) prevents pathological shapes
- Computed efficiently on merged geometry
- Skipped for small regions to avoid numerical instability

**Files Modified**:

- `region_merger.py`

**Why This Is Correct**: Compactness constraint ensures zones are reasonably circular, following transportation planning best practices.

## 5. Skim Matrix Computation

### ZG-12: Optimized Network Distance Calculation

**Original Issue**: Shortest path computed independently for all N² zone pairs, taking hours for 500+ zones.

**What Was Changed**:

- Single-source Dijkstra run once per unique centroid node
- Results cached and reused for all destination zones
- Complexity reduced from O(N² × E log V) to O(N × E log V)

**Files Modified**:

- `skim_computer.py`

**Why This Is Correct**: Single-source Dijkstra from each origin computes all destinations at once, eliminating redundant path calculations.

### ZG-13: Fixed Euclidean Fallback Formula

**Original Issue**: Fallback distance for unconnected zones used Euclidean distance in lat/lon with constant degree-to-meter scaling.

**What Was Changed**:

- Fallback now uses geodesic distance via `pyproj.Geod`
- Detour factor configurable and documented
- Dimensionally correct everywhere

**Files Modified**:

- `skim_computer.py`

**Why This Is Correct**: Geodesic distance provides accurate great-circle distance on Earth's surface.

### ZG-14: Added Distance-Dependent Speed Model

**Original Issue**: Travel time computed as distance/speed with constant speed, ignoring congestion and stops.

**What Was Changed**:

- Distance-decay speed model: `v(d) = v_max × (1 - exp(-d/d₀))`
- Short trips correctly slower (accounts for stops, acceleration)
- Mode-specific parameters (drive, transit, walk)

**Files Modified**:

- `skim_computer.py`

**Why This Is Correct**: Distance-decay model captures the reality that short trips have lower average speeds due to stops and acceleration.

## 6. Validation Logic

### ZG-15: Added Zone Validation Module

**Original Issue**: Generated zones never validated against transportation planning standards.

**What Was Changed**:

- Created comprehensive validation module with 8 checks:
  - Population homogeneity (coefficient of variation)
  - Compactness (Polsby-Popper)
  - Size constraints (min/max population and area)
  - Topological connectivity (adjacency graph)
  - Routing connectivity (skim matrix reachability)
  - Barrier respect (spatial intersection test)
  - Schema validation (required columns)
- Hard pass/fail gate before zone acceptance
- Validation applied to both fresh generations and database-cached zones

**Files Modified**:

- `zone_validator.py` (new file)
- `zone_generator.py`

**Why This Is Correct**: Validation ensures generated zones meet minimum quality standards before use in transportation models.

### ZG-16: Added Multi-Modal Network Support

**Original Issue**: Skim matrices computed only for driving mode.

**What Was Changed**:

- Multi-modal skim computation implemented:
  - Drive mode with congestion proxy
  - Transit mode with schedule-based speeds
  - Walk mode with pedestrian speeds
  - Mode-specific impedance functions
- Generalized cost skims computed per mode

**Files Modified**:

- `skim_computer.py`

**Why This Is Correct**: Multi-modal support enables analysis of transit-oriented cities and mode choice modeling.
