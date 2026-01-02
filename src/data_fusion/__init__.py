"""
Data Fusion Module for Urban Mobility Platform

This module provides tools for evaluating and implementing multi-source
data fusion for vehicle trajectory reconstruction.

Combinations evaluated:
1. GPS + OSM (baseline map matching)
2. GTFS + OSM (schedule-based)
3. GPS + GTFS + OSM (tri-source fusion)
4. CDR + Cell Towers + OSM (low-precision)

Usage:
    from src.data_fusion import GroundTruthGenerator, SensorSimulator
    from src.data_fusion.fusion_algorithms import GPSGTFSOSMFusion
    from src.data_fusion.evaluation import FusionComparator
"""

from .ground_truth_generator import GroundTruthGenerator
from .sensor_simulator import SensorSimulator

__all__ = [
    'GroundTruthGenerator',
    'SensorSimulator',
]

__version__ = '1.0.0'
