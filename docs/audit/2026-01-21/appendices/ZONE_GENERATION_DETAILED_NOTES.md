# Zone Generation Module: Implementation & Validity Issues

## Table of Contents

- [Overview](#overview)
- [Cross-Module Distance Issues](#cross-module-distance-issues)
- [OSM Network Extraction](#osm-network-extraction)
- [H3 Grid Generation](#h3-grid-generation)
- [Barrier Detection](#barrier-detection)
- [Feature Engineering](#feature-engineering)
- [Region Merging](#region-merging)
- [Skim Matrix Computation](#skim-matrix-computation)
- [Cross-Cutting Issues](#cross-cutting-issues)
- [Code Quality](#code-quality)
- [Priority Recommendations](#priority-recommendations)
- [Summary Table](#summary-table)
- [Testing Recommendations](#testing-recommendations)

## Overview

This document catalogs implementation issues in the zone generation module that affect the quality and validity of generated Traffic Analysis Zones (TAZs). Issues range from incorrect distance calculations to flawed merging algorithms and unrealistic assumptions. While less critical than data fusion issues, these problems still compromise the usability of generated zones for transportation planning.

**Severity levels**:

- **CRITICAL**: Fundamentally flawed, produces incorrect results
- **HIGH**: Significant impact on zone quality
- **MEDIUM**: Affects performance or edge cases
- **LOW**: Minor improvements, best practices

## Cross-Module Distance Issues

### CRITICAL: Inconsistent Degree-to-Meter Conversions

**Files**: Multiple throughout module

The code uses constant 111 km/degree conversion everywhere, ignoring latitude effects:

```python
# feature_engineer.py, line 123
self.cells_gdf['distance_to_station_m'] = distances * 111000  # Wrong!

# skim_computer.py, line 58
dist_matrix_km = dist_matrix * 111  # Wrong!

# skim_computer.py, line 173
euclidean = ... * 111  # Wrong!
```

**Why this is wrong**:

- 1° latitude ≈ 111 km everywhere (correct)
- 1° longitude ≈ 111 km × cos(latitude) (varies by latitude)
- At Mumbai (19°N): 1° lon ≈ 105 km, not 111 km
- At higher latitudes, error is even worse

**Impact**:

- Distance calculations are 5-10% off in longitude direction
- Skim matrices have systematic errors
- Zone sizes and feature calculations are incorrect
- Affects all downstream analyses

**Fix**: Use proper geodesic calculations or projected coordinates:

```python
# Option 1: Use pyproj for geodesic distance
from pyproj import Geod
geod = Geod(ellps='WGS84')
distance_m = geod.inv(lon1, lat1, lon2, lat2)[2]

# Option 2: Convert to projected CRS
gdf_projected = gdf.to_crs("EPSG:3857")  # Web Mercator
distances_m = gdf_projected.geometry.distance(point)
```

## OSM Network Extraction

### MEDIUM: Building Area Calculated in Wrong CRS

**File**: `osm_network.py` (line 216)

```python
# Calculate building area
buildings_gdf['area_m2'] = buildings_gdf.geometry.area
```

This calculates area in WGS84 (lat/lon) coordinates, yielding square degrees, not square meters.

**Impact**:

- Building areas are completely wrong (off by 10,000x)
- `proxy_capacity` (area × levels) is meaningless
- Population proxies are invalid
- Zone classification based on wrong data

**Fix**:

```python
# Project to metric CRS first
buildings_projected = buildings_gdf.to_crs("EPSG:3857")
buildings_gdf['area_m2'] = buildings_projected.geometry.area
```

### LOW: Inefficient POI Classification

**File**: `osm_network.py` (lines 308-322)

The `_classify_poi` function checks columns sequentially with multiple if/elif statements. Inefficient and doesn't handle priority correctly.

**Better approach**: Use priority-ordered checks with early returns.

## H3 Grid Generation

### HIGH: Area Calculation After Clipping Is Incorrect

**File**: `hex_grid.py` (lines 82-84, 198-199)

```python
# Clip to boundary
hex_gdf = gpd.overlay(hex_gdf, self.boundary_gdf, how='intersection')

# Later...
gdf_projected = gdf.to_crs("EPSG:3857")
gdf['area_km2'] = gdf_projected.geometry.area / 1_000_000
```

**Issues**:

- After clipping, hexagons at boundary are no longer perfect hexagons
- Area calculation happens before clipping in `_hexagons_to_geodataframe`
- Clipping changes geometry, but area isn't recalculated
- Result: area values don't match actual geometry

**Impact**:

- Zone areas incorrect for boundary zones
- Population density calculations wrong
- Affects zone merging decisions

**Fix**: Recalculate areas after clipping:

```python
def generate_hexagons(self, resolution=None):
    # ... existing code ...
    
    # Clip to boundary
    hex_gdf = gpd.overlay(hex_gdf, self.boundary_gdf, how='intersection')
    
    # RECALCULATE areas after clipping
    hex_projected = hex_gdf.to_crs("EPSG:3857")
    hex_gdf['area_km2'] = hex_projected.geometry.area / 1_000_000
    
    logger.info(f"After clipping: {len(hex_gdf)} hexagons")
    return hex_gdf
```

### MEDIUM: Resolution Selection Is Arbitrary

**File**: `hex_grid.py` (lines 30-56)

```python
# Resolution selection based on area
if area_km2 < 300:  # Small city
    resolution = 9  # ~0.1 km² per hex
elif area_km2 < 1500:  # Medium city
    resolution = 8  # ~0.7 km² per hex
# ...
```

**Issues**:

- No justification for thresholds (300, 1500, 5000 km²)
- No consideration of urban density (dense Mumbai vs sprawling Phoenix)
- No consideration of target zone count
- Comment says "Targeting ~5,000-10,000 hexagons" but doesn't calculate this

**Better approach**: Calculate resolution to achieve target hex count:

```python
def auto_select_resolution(self, target_hex_count=7500):
    """Select H3 resolution to achieve target hex count."""
    area_km2 = self.boundary_gdf.to_crs("EPSG:3857").geometry.area.iloc[0] / 1_000_000
    
    # Calculate target area per hex
    target_area_per_hex = area_km2 / target_hex_count
    
    # H3 resolution areas (approximate, km²)
    h3_areas = {6: 36.1, 7: 5.16, 8: 0.737, 9: 0.105, 10: 0.015}
    
    # Find closest resolution
    best_res = min(h3_areas.keys(),
                   key=lambda r: abs(h3_areas[r] - target_area_per_hex))
    
    logger.info(f"Area: {area_km2:.2f} km², Target: {target_hex_count} hexes")
    logger.info(f"Selected resolution {best_res} (~{h3_areas[best_res]:.3f} km²/hex)")
    
    return best_res
```

## Barrier Detection

### HIGH: Buffer Distance Doesn't Account for Latitude

**File**: `barrier_detector.py` (lines 83-114)

```python
def buffer_corridors(self, corridors_gdf, buffer_distance=50):
    # Convert to projected CRS for accurate buffering (meters)
    corridors_projected = corridors_gdf.to_crs("EPSG:3857")
    
    # Buffer
    corridors_projected['geometry'] = corridors_projected.buffer(buffer_distance)
    
    # Convert back to WGS84
    buffered = corridors_projected.to_crs("EPSG:4326")
```

**Issues**:

- Web Mercator (EPSG:3857) distorts distances at high latitudes
- A 50m buffer in Mumbai is different from 50m in Stockholm using Web Mercator
- Should use local UTM zone for accurate metric buffering

**Impact**: Barriers are too wide/narrow depending on latitude. Affects grid splitting accuracy.

**Fix**: Use local UTM zone:

```python
def buffer_corridors(self, corridors_gdf, buffer_distance=50):
    # Determine appropriate UTM zone
    centroid = corridors_gdf.unary_union.centroid
    utm_crs = corridors_gdf.estimate_utm_crs()  # GeoPandas 0.7+
    
    # Project to UTM
    corridors_projected = corridors_gdf.to_crs(utm_crs)
    
    # Buffer (now accurate)
    corridors_projected['geometry'] = corridors_projected.buffer(buffer_distance)
    
    # Convert back
    buffered = corridors_projected.to_crs("EPSG:4326")
    return buffered
```

### MEDIUM: Sliver Removal Threshold Is Hardcoded

**File**: `barrier_detector.py` (line 240)

```python
# Filter out very small slivers (< 0.01 km²)
split_gdf = split_gdf[split_gdf['area_km2'] > 0.01]
```

**Issues**:

- 0.01 km² = 10,000 m² is arbitrary
- Should be relative to original hex size, not absolute
- In high-resolution grids (res=10), 0.01 km² might be larger than the hexes

**Better approach**:

```python
# Filter slivers < 5% of median cell area
median_area = split_gdf['area_km2'].median()
min_area = median_area * 0.05
split_gdf = split_gdf[split_gdf['area_km2'] > min_area]
logger.info(f"Removed slivers < {min_area:.4f} km²")
```

## Feature Engineering

### CRITICAL: Road Length Calculation Is Inefficient and Incorrect

**File**: `feature_engineer.py` (lines 56-106)

The road length calculation is extremely inefficient with potential correctness issues:

```python
for road_class in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'local']:
    # ...
    # Project to metric CRS
    cells_projected = self.cells_gdf.copy()  # ← COPIES ENTIRE GDF 6 TIMES!
    cells_projected['_cell_id'] = cells_projected.index
    cells_projected = cells_projected.to_crs("EPSG:3857")
    
    roads_projected = class_roads.to_crs("EPSG:3857")
    
    # Intersect roads with cells
    intersections = gpd.overlay(
        cells_projected[['_cell_id', 'geometry']],
        roads_projected[['geometry']],
        how='intersection',
        keep_geom_type=False  # ← Can create points/polygons, not just lines!
    )
    
    # Calculate lengths
    intersections['length_m'] = intersections.geometry.length  # ← WRONG for non-LineStrings!
```

**Issues**:

- Copies entire cells GDF 6 times (once per road class) - huge memory waste
- Projects to Web Mercator - distance distortion at high latitudes
- `keep_geom_type=False` allows points and polygons
  - Points have length=0 (correct)
  - Polygons have length=perimeter (WRONG - should be 0 or error)
- No error handling for geometry collection types

**Impact**: Slow performance (6x unnecessary projections), incorrect road lengths if overlay creates polygons, high memory usage.

**Fix**: Project once, process all road classes together, filter to LineStrings only:

```python
def compute_network_metrics(self):
    logger.info("  Computing network metrics...")
    
    if 'roads' not in self.osm_data or self.osm_data['roads'].empty:
        return
    
    roads = self.osm_data['roads']
    
    # Project ONCE
    cells_projected = self.cells_gdf.copy()
    cells_projected['_cell_id'] = cells_projected.index
    utm_crs = cells_projected.estimate_utm_crs()
    cells_projected = cells_projected.to_crs(utm_crs)
    
    roads_projected = roads.to_crs(utm_crs)
    
    # Process all road classes together
    for road_class in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'local']:
        class_roads = roads_projected[roads_projected['road_class'] == road_class]
        
        if class_roads.empty:
            self.cells_gdf[f'{road_class}_length_m'] = 0
            continue
        
        # Intersect
        intersections = gpd.overlay(
            cells_projected[['_cell_id', 'geometry']],
            class_roads[['geometry']],
            how='intersection',
            keep_geom_type=True  # Keep only LineStrings
        )
        
        # Filter to LineStrings only
        intersections = intersections[intersections.geometry.type == 'LineString']
        
        if not intersections.empty:
            intersections['length_m'] = intersections.geometry.length
            length_by_cell = intersections.groupby('_cell_id')['length_m'].sum()
            self.cells_gdf[f'{road_class}_length_m'] = length_by_cell.reindex(
                self.cells_gdf.index, fill_value=0
            )
        else:
            self.cells_gdf[f'{road_class}_length_m'] = 0
```

### HIGH: Employment Weights Are Arbitrary

**File**: `feature_engineer.py` (lines 234-248)

```python
# Weight: office=10, commercial=5, industrial=8, education=3, healthcare=2
office_weight = 10
commercial_weight = 5
industrial_weight = 8
education_weight = 3
healthcare_weight = 2
```

**Issues**:

- No justification for these weights
- No calibration to actual employment data
- No regional variation (office jobs in Mumbai vs manufacturing city)
- Missing categories (retail, hospitality, government)

**Impact**: Employment proxies don't correlate with reality. Zone classification unreliable. Can't use for trip generation models.

**Fix**: Calibrate weights using census data or use ITE-based estimates:

```python
def compute_poi_proxies(self, employment_weights=None):
    """
    Compute POI-based employment proxies.
    
    Args:
        employment_weights: Dict of POI type -> employment factor
                          Defaults to ITE-based estimates
    """
    if employment_weights is None:
        # Based on ITE Trip Generation Manual
        employment_weights = {
            'office': 250,  # employees per 1000 m² GFA
            'commercial': 150,  # retail
            'industrial': 100,  # manufacturing
            'education': 50,  # staff per facility
            'healthcare': 200  # hospital staff
        }
    
    # ... rest of implementation
```

## Region Merging

### HIGH: Cosine Similarity for Spatial Features Is Inappropriate

**File**: `region_merger.py` (lines 121-146, 248-263)

```python
def _get_feature_vectors(self):
    feature_cols = [
        'proxy_population',
        'proxy_employment',
        'total_building_area_m2',
        'avg_building_levels'
    ]
    
    # Normalize to [0, 1]
    features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-6)
    return features_norm

def _can_merge(self, ...):
    # ...
    similarity = cosine_similarity(seed_features, candidate_features)[0][0]
```

**Why this is wrong**: Cosine similarity measures angle between vectors, not magnitude. Appropriate for text (word frequencies) but inappropriate for spatial features.

**Example**:

```md
Cell A: population=1000, employment=100
Cell B: population=10000, employment=1000
Cosine similarity = 1.0 (perfect match!)
But these cells are VERY different in absolute terms!
```

**Impact**: Merges cells with very different characteristics. Creates heterogeneous zones. Defeats purpose of similarity-based merging.

**Fix**: Use Euclidean distance in normalized feature space:

```python
def _can_merge(self, seed_idx, candidate_idx, region_cells, feature_vectors):
    # ...
    
    # Euclidean distance in normalized feature space
    seed_features = feature_vectors[seed_idx]
    candidate_features = feature_vectors[candidate_idx]
    
    distance = np.linalg.norm(seed_features - candidate_features)
    
    # Lower threshold for CBD, higher for residential
    if self.cells_gdf.loc[seed_idx, 'is_cbd']:
        max_distance = 0.5  # More heterogeneous allowed
    elif seed_landuse == 'residential':
        max_distance = 0.2  # More homogeneous required
    else:
        max_distance = 0.3
    
    if distance > max_distance:
        return False
    
    return True
```

### MEDIUM: BFS Queue Can Grow Unbounded

**File**: `region_merger.py` (lines 172-200)

```python
while queue and region_population < target_pop:
    candidate_idx = queue.popleft()
    
    # ...
    
    # Add neighbors to queue
    for neighbor_idx in adjacency[candidate_idx]:
        if neighbor_idx not in visited:
            queue.append(neighbor_idx)  # ← Can add many neighbors
```

**Issues**:

- No limit on queue size - can grow to thousands of cells
- No priority - processes neighbors in arbitrary order
- Inefficient for large grids

**Better approach**: Use priority queue based on similarity to seed:

```python
import heapq

def _grow_region(self, seed_idx, adjacency, feature_vectors):
    # ...
    
    # Priority queue: (negative_similarity, cell_idx)
    # Negative because heapq is min-heap
    pq = []
    
    for neighbor_idx in adjacency[seed_idx]:
        similarity = cosine_similarity(
            feature_vectors[seed_idx].reshape(1, -1),
            feature_vectors[neighbor_idx].reshape(1, -1)
        )[0][0]
        heapq.heappush(pq, (-similarity, neighbor_idx))
    
    visited = set([seed_idx])
    
    while pq and region_population < target_pop:
        neg_sim, candidate_idx = heapq.heappop(pq)
        
        if candidate_idx in visited:
            continue
        
        visited.add(candidate_idx)
        
        # ... rest of merging logic
```

### MEDIUM: No Compactness Constraint

**File**: `region_merger.py`

The region growing algorithm has no compactness constraint. Can create long, thin zones that snake through the city.

**Impact**: Zones can be non-compact (high perimeter-to-area ratio). Violates transportation planning best practices. Makes zones less useful for analysis.

**Fix**: Add compactness check using Polsby-Popper metric:

```python
def _can_merge(self, seed_idx, candidate_idx, region_cells, feature_vectors):
    # ... existing checks ...
    
    # Compactness check
    if len(region_cells) > 3:  # Only check after a few cells
        # Calculate current region compactness
        region_geoms = [self.cells_gdf.loc[idx, 'geometry'] for idx in region_cells]
        region_union = unary_union(region_geoms)
        
        # Polsby-Popper compactness: 4π * area / perimeter²
        # Perfect circle = 1.0, lower = less compact
        area = region_union.area
        perimeter = region_union.length
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        
        # Require minimum compactness
        if compactness < 0.2:  # Very non-compact
            return False
    
    return True
```

## Skim Matrix Computation

### CRITICAL: Network Distance Calculation Is O(N²) and Slow

**File**: `skim_computer.py` (lines 136-179)

```python
for i in range(n_zones):
    for j in range(n_zones):
        # ...
        length = nx.shortest_path_length(
            self.network_graph,
            origin_node,
            dest_node,
            weight='length'
        )
```

**Issues**:

- O(N²) shortest path calculations - for 100 zones, that's 10,000 paths
- No caching - recalculates paths even if already computed
- Extremely slow for large zone systems (500+ zones)
- No parallelization

**Impact**: Takes hours for large cities. Impractical for real-world use.

**Fix**: Use all-pairs shortest path algorithms:

```python
def compute_network_distance_matrix(self, sample_size=None):
    logger.info("Computing network distance matrix...")
    
    # ... setup code ...
    
    # Use all-pairs shortest path (much faster)
    logger.info("Computing all-pairs shortest paths...")
    
    # Johnson's algorithm (works for directed graphs)
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(
        self.network_graph,
        weight='length'
    ))
    
    # Build matrix
    dist_matrix = np.zeros((n_zones, n_zones))
    
    for i, origin_node in enumerate(nearest_nodes):
        if origin_node is None:
            continue
        
        for j, dest_node in enumerate(nearest_nodes):
            if dest_node is None:
                continue
            
            if i == j:
                dist_matrix[i, j] = 0
            else:
                try:
                    length = all_lengths[origin_node][dest_node]
                    dist_matrix[i, j] = length / 1000  # Convert to km
                except KeyError:
                    # No path - use Euclidean * detour factor
                    dist_matrix[i, j] = self._euclidean_fallback(i, j)
    
    # ... rest of code
```

**Performance**:

- Before: O(N² × E log V) ≈ hours for 500 zones
- After: O(N × E log V) ≈ minutes for 500 zones

### HIGH: Euclidean Fallback Uses Wrong Formula

**File**: `skim_computer.py` (lines 170-175)

```python
# No path exists, use Euclidean distance * 1.3 (detour factor)
euclidean = np.sqrt(
    (centroids_sample.iloc[i].geometry.x - centroids_sample.iloc[j].geometry.x) ** 2 +
    (centroids_sample.iloc[i].geometry.y - centroids_sample.iloc[j].geometry.y) ** 2
) * 111
dist_matrix[i, j] = euclidean * 1.3
```

**Issues**:

- Euclidean distance in lat/lon (same issue as before)
- Constant 1.3 detour factor is arbitrary
  - Urban areas: 1.2-1.4
  - Suburban: 1.3-1.6
  - Rural: 1.5-2.0
- Should use geodesic distance

**Fix**:

```python
from pyproj import Geod

def _euclidean_fallback(self, i, j, detour_factor=1.4):
    """Calculate fallback distance when no network path exists."""
    geod = Geod(ellps='WGS84')
    
    c_i = self.centroids_sample.iloc[i].geometry
    c_j = self.centroids_sample.iloc[j].geometry
    
    # Geodesic distance
    _, _, distance_m = geod.inv(c_i.x, c_i.y, c_j.x, c_j.y)
    
    # Apply detour factor
    return (distance_m / 1000) * detour_factor
```

### MEDIUM: Travel Time Doesn't Account for Congestion

**File**: `skim_computer.py` (lines 193-216)

```python
def compute_travel_time_matrix(self, distance_matrix, avg_speed_kmh=30):
    # Time = Distance / Speed * 60
    time_matrix = (distance_matrix / avg_speed_kmh) * 60
```

**Issues**:

- Constant speed - no congestion effects
- No time-of-day variation
- No distance-speed relationship (short trips are slower per km due to stops)

**Better approach**: Implement speed model with congestion:

```python
def compute_travel_time_matrix(
    self,
    distance_matrix,
    time_period='peak',
    speed_model='bpr'
):
    """
    Compute travel time with congestion effects.
    
    Args:
        distance_matrix: Distance matrix (km)
        time_period: 'peak', 'offpeak', or 'night'
        speed_model: 'constant', 'bpr', or 'distance_decay'
    """
    if speed_model == 'constant':
        speeds = {'peak': 25, 'offpeak': 35, 'night': 45}
        avg_speed = speeds.get(time_period, 30)
        return (distance_matrix / avg_speed) * 60
    
    elif speed_model == 'distance_decay':
        # Shorter trips have lower average speed (more stops)
        # v(d) = v_max * (1 - exp(-d/d_0))
        v_max = 40  # km/h
        d_0 = 5  # km
        
        speeds = v_max * (1 - np.exp(-distance_matrix / d_0))
        speeds = np.maximum(speeds, 10)  # Minimum 10 km/h
        
        return (distance_matrix / speeds) * 60
    
    # ... BPR function implementation
```

## Cross-Cutting Issues

### HIGH: No Validation of Generated Zones

The module generates zones but never validates if they meet transportation planning standards.

**Missing validations**:

- Zone size homogeneity (are zones roughly similar in population/area?)
- Compactness (are zones compact or long and thin?)
- Connectivity (can you route between all zone pairs?)
- Minimum zone size (any zones too small to be useful?)
- Maximum zone size (any zones too large, >20,000 population?)
- Barrier respect (do zones cross major barriers?)

**Fix**: Add validation module:

```python
class ZoneValidator:
    """Validate generated zones against planning standards."""
    
    def validate_zones(self, zones_gdf):
        """Run all validation checks."""
        results = {
            'population_cv': self._check_population_homogeneity(zones_gdf),
            'compactness_mean': self._check_compactness(zones_gdf),
            'connectivity': self._check_connectivity(zones_gdf),
            'size_violations': self._check_size_constraints(zones_gdf)
        }
        
        return results
    
    def _check_population_homogeneity(self, zones_gdf):
        """Check coefficient of variation in population."""
        pop = zones_gdf['proxy_population']
        cv = pop.std() / pop.mean()
        
        # Good: CV < 0.5, Acceptable: CV < 1.0, Poor: CV > 1.0
        return cv
    
    def _check_compactness(self, zones_gdf):
        """Check Polsby-Popper compactness."""
        zones_projected = zones_gdf.to_crs("EPSG:3857")
        
        compactness = []
        for _, zone in zones_projected.iterrows():
            area = zone.geometry.area
            perimeter = zone.geometry.length
            pp = (4 * np.pi * area) / (perimeter ** 2)
            compactness.append(pp)
        
        return np.mean(compactness)
    
    # ... more validation methods
```

### MEDIUM: No Support for Multi-Modal Networks

The skim matrix computation only considers driving. Real transportation planning needs:

- Transit (bus, metro, rail)
- Walking
- Cycling
- Multi-modal (drive to transit)

**Impact**: Can't use zones for multi-modal trip distribution. Limited applicability to transit-oriented cities.

**Fix**: Extend skim computation to support multiple modes with proper impedance functions.

## Code Quality

**General problems**:

- No input validation (functions don't check for invalid inputs)
- Inconsistent error handling (some raise exceptions, others return empty GeoDataFrames)
- No logging levels (everything is INFO, should use DEBUG for details)
- Magic numbers everywhere (50, 0.01, 111000, 1.3, etc.)
- No configuration (all parameters hardcoded)
- No unit tests (zero test coverage)

**Example - missing input validation**:

```python
def __init__(self, cells_gdf, osm_data):
    # No validation!
    # What if cells_gdf is empty?
    # What if osm_data is missing required keys?
    # What if CRS doesn't match?
    self.cells_gdf = cells_gdf.copy()
    self.osm_data = osm_data
```

**Should be**:

```python
def __init__(self, cells_gdf, osm_data):
    if cells_gdf.empty:
        raise ValueError("cells_gdf cannot be empty")
    
    if not isinstance(osm_data, dict):
        raise TypeError("osm_data must be a dictionary")
    
    required_keys = ['roads', 'buildings', 'pois']
    missing = [k for k in required_keys if k not in osm_data]
    if missing:
        logger.warning(f"Missing OSM data: {missing}")
    
    if cells_gdf.crs is None:
        raise ValueError("cells_gdf must have a CRS defined")
    
    self.cells_gdf = cells_gdf.copy()
    self.osm_data = osm_data
```

## Priority Recommendations

### Immediate Fixes (Before Production Use)

1. **Fix all degree-to-meter conversions** - use geodesic calculations or projected CRS. Affects all distance-based features.

2. **Fix building area calculation** - project to metric CRS before calculating area. Critical for population proxies.

3. **Optimize skim matrix computation** - use all-pairs shortest path. Essential for scalability.

### Short-Term Improvements (For Reliable Results)

1. **Fix road length calculation** - project once, filter geometry types. Major performance improvement.

2. **Replace cosine similarity** - use Euclidean distance in normalized space. Better zone homogeneity.

3. **Add zone validation** - implement quality checks. Ensure zones meet standards.

### Long-Term Enhancements (For Research Quality)

1. **Calibrate employment weights** - use census data or ITE rates. Make weights configurable.

2. **Add compactness constraints** - implement in region merging. Improve zone quality.

3. **Support multi-modal networks** - extend skim computation. Enable transit planning.

## Summary Table

| Issue | Severity | File | Impact | Fix Effort |
| ------- | ---------- | ------ | -------- | ------------ |
| Degree-to-meter conversion | CRITICAL | Multiple | Wrong distances everywhere | Medium |
| Road length calculation | CRITICAL | feature_engineer.py | Slow + incorrect | Medium |
| O(N²) skim computation | CRITICAL | skim_computer.py | Impractical for large cities | High |
| Area after clipping | HIGH | hex_grid.py | Wrong zone sizes | Low |
| Cosine similarity for merging | HIGH | region_merger.py | Poor zone quality | Medium |
| Buffer in Web Mercator | HIGH | barrier_detector.py | Inaccurate barriers | Low |
| No zone validation | HIGH | All | Unknown quality | Medium |
| Arbitrary employment weights | HIGH | feature_engineer.py | Unrealistic proxies | Low |
| Euclidean fallback wrong | HIGH | skim_computer.py | Wrong distances | Low |
| Building area in wrong CRS | MEDIUM | osm_network.py | Wrong population proxies | Low |
| Resolution selection arbitrary | MEDIUM | hex_grid.py | Suboptimal hex count | Medium |
| Sliver threshold hardcoded | MEDIUM | barrier_detector.py | Inconsistent filtering | Low |
| BFS queue unbounded | MEDIUM | region_merger.py | Inefficient | Medium |
| No compactness constraint | MEDIUM | region_merger.py | Non-compact zones | Medium |
| Travel time no congestion | MEDIUM | skim_computer.py | Unrealistic times | Medium |
| No multi-modal support | MEDIUM | skim_computer.py | Limited applicability | High |
| POI classification inefficient | LOW | osm_network.py | Minor performance | Low |

## Testing Recommendations

**Unit tests needed**:

```python
# test_hex_grid.py
def test_area_calculation_after_clipping():
    """Verify areas match geometry after clipping."""

def test_resolution_selection():
    """Verify resolution selection produces expected hex count."""

# test_feature_engineer.py
def test_road_length_accuracy():
    """Verify road lengths match manual calculation."""

def test_distance_calculations():
    """Verify distances use proper geodesic formulas."""

# test_region_merger.py
def test_zone_homogeneity():
    """Verify merged zones have similar characteristics."""

def test_compactness():
    """Verify zones meet minimum compactness threshold."""

# test_skim_computer.py
def test_skim_matrix_symmetry():
    """Verify distance matrix is symmetric."""

def test_network_vs_euclidean():
    """Verify network distances >= Euclidean distances."""
```

**Integration tests needed**:

```python
def test_end_to_end_pipeline():
    """Test complete pipeline from OSM to skim matrices."""

def test_zone_count_target():
    """Verify final zone count is close to target."""

def test_zone_validation():
    """Verify all zones pass quality checks."""
```
