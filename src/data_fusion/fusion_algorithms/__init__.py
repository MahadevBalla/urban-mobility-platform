"""
Fusion Algorithms Package

Contains implementations of different data fusion combinations:
- GPS + OSM: Basic map matching
- GTFS + OSM: Schedule-based trajectory
- GPS + GTFS + OSM: Tri-source fusion (recommended)
- CDR + OSM: Low-precision mobile data fusion
"""

from .base_fusion import BaseFusionAlgorithm
from .gps_osm_fusion import GPSOSMFusion
from .gtfs_osm_fusion import GTFSOSMFusion
from .gps_gtfs_osm_fusion import GPSGTFSOSMFusion
from .cdr_osm_fusion import CDROSMFusion

__all__ = [
    'BaseFusionAlgorithm',
    'GPSOSMFusion',
    'GTFSOSMFusion',
    'GPSGTFSOSMFusion',
    'CDROSMFusion',
]
