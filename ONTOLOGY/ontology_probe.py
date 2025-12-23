"""GPS Probe Data Ontology

Entities for GPS probe/floating car data from:
- TomTom Traffic
- HERE Technologies
- INRIX
- StreetLight Data

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ontology_base import (
    TransportEntity, SpatialEntity, TemporalEntity,
    TimeOfDay, DayType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ProbeProvider(Enum):
    """GPS probe data providers"""
    TOMTOM = "tomtom"
    HERE = "here"
    INRIX = "inrix"
    STREETLIGHT = "streetlight"
    OTHER = "other"


class VehicleType(Enum):
    """Vehicle classification"""
    PASSENGER_CAR = "passenger_car"
    COMMERCIAL_VEHICLE = "commercial_vehicle"
    TAXI = "taxi"
    BUS = "bus"
    TRUCK_LIGHT = "truck_light"
    TRUCK_HEAVY = "truck_heavy"
    MOTORCYCLE = "motorcycle"


class CongestionLevel(Enum):
    """Traffic congestion levels"""
    FREE_FLOW = "free_flow"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class ProbeDataProvider(TransportEntity):
    """GPS probe data provider organization"""
    provider_id: str = ""
    provider_name: Optional[ProbeProvider] = None
    data_license_type: str = ""  # e.g., "academic", "commercial"
    
    # Coverage
    coverage_region: Optional[str] = None  # WKT MultiPolygon
    update_frequency_seconds: int = 60
    
    # Data characteristics
    typical_sample_size: Optional[int] = None
    penetration_rate_percent: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "ProbeDataProvider"


@dataclass
class ProbeVehicle(TransportEntity):
    """Probe vehicle (anonymized)"""
    probe_vehicle_id: str = ""  # Hashed identifier
    data_provider_id: str = ""  # FK to ProbeDataProvider
    
    # Vehicle characteristics
    vehicle_type: VehicleType = VehicleType.PASSENGER_CAR
    
    # Activity period
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_traces: Optional[int] = None

    def __post_init__(self):
        self.entity_type = "ProbeVehicle"


@dataclass
class GPSTracePoint(SpatialEntity, TemporalEntity):
    """Individual GPS trace point"""
    trace_id: str = ""
    probe_vehicle_id: str = ""  # FK to ProbeVehicle
    
    # Spatial (geometry in SpatialEntity, WGS84)
    altitude_m: Optional[float] = None
    
    # Kinematic
    speed_kmh: Optional[float] = None
    heading_degrees: Optional[float] = None  # 0-360, 0=North
    acceleration_mps2: Optional[float] = None
    
    # GPS quality
    accuracy_m: Optional[float] = None  # Horizontal accuracy
    hdop: Optional[float] = None  # Horizontal Dilution of Precision
    num_satellites: Optional[int] = None
    
    # Context
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.entity_type = "GPSTracePoint"


@dataclass
class MatchedTrace(SpatialEntity):
    """GPS trace matched to road network"""
    matched_trace_id: str = ""
    original_trace_id: str = ""  # FK to GPSTracePoint
    road_segment_id: str = ""  # FK to RoadSegment
    
    # Matched location
    offset_from_segment_start_m: float = 0.0
    matched_location: Any = None  # Point geometry on road segment
    
    # Matching quality
    matching_confidence: float = 0.0  # 0-1
    perpendicular_distance_m: Optional[float] = None  # Distance from road
    
    # Attributes
    speed_kmh: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.entity_type = "MatchedTrace"


@dataclass
class SpeedObservation(TemporalEntity):
    """Speed observation on road segment"""
    observation_id: str = ""
    road_segment_id: str = ""  # FK to RoadSegment
    
    # Speed data
    average_speed_kmh: float = 0.0
    percentile_15_speed_kmh: Optional[float] = None
    percentile_50_speed_kmh: Optional[float] = None  # Median
    percentile_85_speed_kmh: Optional[float] = None
    std_speed_kmh: Optional[float] = None
    
    # Reference speeds
    free_flow_speed_kmh: Optional[float] = None
    speed_limit_kmh: Optional[float] = None
    
    # Sample
    sample_size: int = 0  # Number of probes
    
    # Congestion
    congestion_level: Optional[CongestionLevel] = None
    level_of_service: Optional[str] = None  # A-F scale
    
    # Temporal
    timestamp: Optional[datetime] = None
    time_of_day: Optional[TimeOfDay] = None
    day_type: Optional[DayType] = None

    def __post_init__(self):
        self.entity_type = "SpeedObservation"


@dataclass
class TravelTimeSegment(TemporalEntity):
    """Travel time on road segment"""
    segment_id: str = ""
    road_segment_id: str = ""  # FK to RoadSegment
    
    # Time period
    time_period: str = ""  # e.g., "AM_PEAK_07:00-09:00"
    day_type: Optional[DayType] = None
    
    # Travel time (seconds)
    historical_travel_time_seconds: float = 0.0
    real_time_travel_time_seconds: Optional[float] = None
    free_flow_travel_time_seconds: Optional[float] = None
    
    # Confidence
    confidence_interval_95_lower: Optional[float] = None
    confidence_interval_95_upper: Optional[float] = None
    sample_size: Optional[int] = None
    
    # Derived metrics
    congestion_ratio: Optional[float] = None  # real_time / free_flow
    delay_seconds: Optional[float] = None  # real_time - free_flow

    def __post_init__(self):
        self.entity_type = "TravelTimeSegment"


@dataclass
class TrafficFlow(TemporalEntity):
    """Traffic flow on road segment"""
    flow_id: str = ""
    road_segment_id: str = ""  # FK to RoadSegment
    
    # Flow metrics
    volume_veh_per_hour: int = 0
    speed_kmh: float = 0.0
    density_veh_per_km: Optional[float] = None
    
    # Temporal
    timestamp: Optional[datetime] = None
    time_period_minutes: int = 15  # Aggregation period
    
    # Quality
    sample_size: int = 0
    confidence: float = 0.8

    def __post_init__(self):
        self.entity_type = "TrafficFlow"


@dataclass
class ODTrajectory(TransportEntity):
    """Complete vehicle trajectory (trip)"""
    trajectory_id: str = ""
    probe_vehicle_id: str = ""  # FK to ProbeVehicle
    
    # Spatial
    origin_zone_id: Optional[str] = None
    destination_zone_id: Optional[str] = None
    trace_points: List[str] = field(default_factory=list)  # List of trace IDs
    
    # Temporal
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    
    # Metrics
    total_distance_km: float = 0.0
    average_speed_kmh: float = 0.0
    max_speed_kmh: Optional[float] = None
    num_stops: Optional[int] = None
    
    # Route
    road_segments_used: List[str] = field(default_factory=list)
    route_geometry: Optional[Any] = None  # LineString


if __name__ == "__main__":
    # Example usage
    provider = ProbeDataProvider(
        provider_id="PROV_001",
        entity_id="PROV_001",
        provider_name=ProbeProvider.TOMTOM,
        data_license_type="academic",
        update_frequency_seconds=60,
        penetration_rate_percent=5.0
    )
    
    trace = GPSTracePoint(
        trace_id="TRACE_001",
        entity_id="TRACE_001",
        probe_vehicle_id="VEH_001",
        geometry=None,
        centroid=(77.2090, 28.6139),
        speed_kmh=45.5,
        heading_degrees=90,
        accuracy_m=10.0,
        timestamp=datetime(2023, 10, 15, 8, 30, 15)
    )
    
    speed_obs = SpeedObservation(
        observation_id="SPEED_001",
        entity_id="SPEED_001",
        road_segment_id="SEG_001",
        average_speed_kmh=32.5,
        free_flow_speed_kmh=60.0,
        congestion_level=CongestionLevel.MODERATE,
        sample_size=25,
        timestamp=datetime(2023, 10, 15, 8, 30)
    )
    
    print(f"Provider: {provider.provider_name.value}")
    print(f"Speed observation: {speed_obs.average_speed_kmh} km/h (congestion: {speed_obs.congestion_level.value})")
