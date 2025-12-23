"""GTFS Data Ontology

Entities for public transit data:
- GTFS Static (schedules)
- GTFS Realtime (vehicle positions, trip updates, alerts)

Standards: GTFS Specification v2.0+

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, time
from enum import Enum

from ontology_base import (
    TransportEntity, SpatialEntity, TemporalEntity,
    GTFSRouteType, DayType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class LocationType(Enum):
    """GTFS location types"""
    STOP = 0  # Stop or platform
    STATION = 1  # Physical structure/area with stops
    ENTRANCE_EXIT = 2  # Entrance/exit to station
    GENERIC_NODE = 3  # Location within station
    BOARDING_AREA = 4  # Specific boarding location


class WheelchairAccessible(Enum):
    """Wheelchair accessibility"""
    UNKNOWN = 0
    ACCESSIBLE = 1
    NOT_ACCESSIBLE = 2


class BikesAllowed(Enum):
    """Bicycle allowance"""
    UNKNOWN = 0
    ALLOWED = 1
    NOT_ALLOWED = 2


class PickupDropoffType(Enum):
    """Pickup/dropoff availability"""
    REGULAR = 0
    NOT_AVAILABLE = 1
    PHONE_AGENCY = 2
    COORDINATE_WITH_DRIVER = 3


class Timepoint(Enum):
    """Time accuracy"""
    APPROXIMATE = 0
    EXACT = 1


class OccupancyStatus(Enum):
    """Vehicle occupancy (GTFS-RT)"""
    EMPTY = "EMPTY"
    MANY_SEATS = "MANY_SEATS_AVAILABLE"
    FEW_SEATS = "FEW_SEATS_AVAILABLE"
    STANDING_ROOM = "STANDING_ROOM_ONLY"
    CRUSHED = "CRUSHED_STANDING_ROOM_ONLY"
    FULL = "FULL"
    NOT_ACCEPTING = "NOT_ACCEPTING_PASSENGERS"


class CongestionLevel(Enum):
    """Traffic congestion (GTFS-RT)"""
    UNKNOWN = "UNKNOWN_CONGESTION_LEVEL"
    SMOOTH = "RUNNING_SMOOTHLY"
    STOP_AND_GO = "STOP_AND_GO"
    CONGESTION = "CONGESTION"
    SEVERE = "SEVERE_CONGESTION"


class VehicleStopStatus(Enum):
    """Vehicle status at stop (GTFS-RT)"""
    INCOMING = "INCOMING_AT"
    STOPPED = "STOPPED_AT"
    IN_TRANSIT = "IN_TRANSIT_TO"


class ScheduleRelationship(Enum):
    """Schedule relationship (GTFS-RT)"""
    SCHEDULED = "SCHEDULED"
    ADDED = "ADDED"
    UNSCHEDULED = "UNSCHEDULED"
    CANCELED = "CANCELED"
    SKIPPED = "SKIPPED"


class AlertCause(Enum):
    """Service alert causes"""
    UNKNOWN_CAUSE = "UNKNOWN_CAUSE"
    OTHER_CAUSE = "OTHER_CAUSE"
    TECHNICAL_PROBLEM = "TECHNICAL_PROBLEM"
    STRIKE = "STRIKE"
    DEMONSTRATION = "DEMONSTRATION"
    ACCIDENT = "ACCIDENT"
    HOLIDAY = "HOLIDAY"
    WEATHER = "WEATHER"
    MAINTENANCE = "MAINTENANCE"
    CONSTRUCTION = "CONSTRUCTION"
    POLICE_ACTIVITY = "POLICE_ACTIVITY"
    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"


class AlertEffect(Enum):
    """Service alert effects"""
    NO_SERVICE = "NO_SERVICE"
    REDUCED_SERVICE = "REDUCED_SERVICE"
    SIGNIFICANT_DELAYS = "SIGNIFICANT_DELAYS"
    DETOUR = "DETOUR"
    ADDITIONAL_SERVICE = "ADDITIONAL_SERVICE"
    MODIFIED_SERVICE = "MODIFIED_SERVICE"
    OTHER_EFFECT = "OTHER_EFFECT"
    UNKNOWN_EFFECT = "UNKNOWN_EFFECT"
    STOP_MOVED = "STOP_MOVED"


# ============================================================================
# GTFS STATIC ENTITIES
# ============================================================================

@dataclass
class Agency(TransportEntity):
    """Transit agency"""
    agency_id: str = ""
    agency_name: str = ""
    agency_url: str = ""
    agency_timezone: str = ""
    agency_lang: Optional[str] = None
    agency_phone: Optional[str] = None
    agency_fare_url: Optional[str] = None
    agency_email: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "Agency"


@dataclass
class Stop(SpatialEntity):
    """Transit stop or station"""
    stop_id: str = ""
    stop_code: Optional[str] = None
    stop_name: Optional[str] = None
    stop_desc: Optional[str] = None
    zone_id: Optional[str] = None  # Fare zone
    stop_url: Optional[str] = None
    location_type: LocationType = LocationType.STOP
    parent_station: Optional[str] = None
    stop_timezone: Optional[str] = None
    wheelchair_accessible: WheelchairAccessible = WheelchairAccessible.UNKNOWN
    level_id: Optional[str] = None
    platform_code: Optional[str] = None
    
    # Link to TAZ
    taz_id: Optional[str] = None
    
    # Link to OSM
    osm_node_id: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "Stop"


@dataclass
class Route(TransportEntity):
    """Transit route"""
    route_id: str = ""
    agency_id: Optional[str] = None
    route_short_name: Optional[str] = None
    route_long_name: Optional[str] = None
    route_desc: Optional[str] = None
    route_type: GTFSRouteType = GTFSRouteType.BUS
    route_url: Optional[str] = None
    route_color: Optional[str] = None
    route_text_color: Optional[str] = None
    route_sort_order: Optional[int] = None
    continuous_pickup: Optional[PickupDropoffType] = None
    continuous_drop_off: Optional[PickupDropoffType] = None

    def __post_init__(self):
        self.entity_type = "Route"


@dataclass
class Trip(TransportEntity):
    """Transit trip"""
    trip_id: str = ""
    route_id: str = ""
    service_id: str = ""
    trip_headsign: Optional[str] = None
    trip_short_name: Optional[str] = None
    direction_id: Optional[int] = None  # 0 or 1
    block_id: Optional[str] = None
    shape_id: Optional[str] = None
    wheelchair_accessible: WheelchairAccessible = WheelchairAccessible.UNKNOWN
    bikes_allowed: BikesAllowed = BikesAllowed.UNKNOWN

    def __post_init__(self):
        self.entity_type = "Trip"


@dataclass
class StopTime(TransportEntity):
    """Stop time in trip sequence"""
    trip_id: str = ""
    arrival_time: Optional[time] = None  # Can exceed 24 hours
    departure_time: Optional[time] = None
    stop_id: str = ""
    stop_sequence: int = 0
    stop_headsign: Optional[str] = None
    pickup_type: PickupDropoffType = PickupDropoffType.REGULAR
    drop_off_type: PickupDropoffType = PickupDropoffType.REGULAR
    continuous_pickup: Optional[PickupDropoffType] = None
    continuous_drop_off: Optional[PickupDropoffType] = None
    shape_dist_traveled: Optional[float] = None
    timepoint: Timepoint = Timepoint.EXACT

    def __post_init__(self):
        self.entity_type = "StopTime"
        # Create composite ID
        self.entity_id = f"{self.trip_id}_{self.stop_sequence}"


@dataclass
class Calendar(TransportEntity):
    """Service calendar"""
    service_id: str = ""
    monday: bool = False
    tuesday: bool = False
    wednesday: bool = False
    thursday: bool = False
    friday: bool = False
    saturday: bool = False
    sunday: bool = False
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def __post_init__(self):
        self.entity_type = "Calendar"


@dataclass
class CalendarDate(TransportEntity):
    """Calendar exceptions"""
    service_id: str = ""
    date: Optional[date] = None
    exception_type: int = 1  # 1=added, 2=removed

    def __post_init__(self):
        self.entity_type = "CalendarDate"
        self.entity_id = f"{self.service_id}_{self.date}"


@dataclass
class Shape(SpatialEntity):
    """Route shape"""
    shape_id: str = ""
    points: List[Tuple[float, float, int, float]] = field(default_factory=list)
    # Each point: (lat, lon, sequence, dist_traveled)

    def __post_init__(self):
        self.entity_type = "Shape"


# ============================================================================
# GTFS REALTIME ENTITIES
# ============================================================================

@dataclass
class VehiclePosition(SpatialEntity, TemporalEntity):
    """Real-time vehicle position"""
    vehicle_id: str = ""
    trip_id: Optional[str] = None
    route_id: Optional[str] = None
    
    # Position (geometry in SpatialEntity)
    bearing: Optional[float] = None  # 0-360 degrees
    speed_mps: Optional[float] = None  # meters per second
    
    # Stop relationship
    current_stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None
    current_status: Optional[VehicleStopStatus] = None
    
    # Congestion & occupancy
    congestion_level: Optional[CongestionLevel] = None
    occupancy_status: Optional[OccupancyStatus] = None
    
    # Timing
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.entity_type = "VehiclePosition"


@dataclass
class TripUpdate(TemporalEntity):
    """Real-time trip update"""
    trip_id: str = ""
    vehicle_id: Optional[str] = None
    route_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    delay_seconds: Optional[int] = None
    schedule_relationship: ScheduleRelationship = ScheduleRelationship.SCHEDULED
    
    # Stop time updates
    stop_time_updates: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.entity_type = "TripUpdate"


@dataclass
class StopTimeUpdate:
    """Real-time stop time update"""
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None
    arrival_delay_seconds: Optional[int] = None
    arrival_time: Optional[datetime] = None
    arrival_uncertainty: Optional[int] = None
    departure_delay_seconds: Optional[int] = None
    departure_time: Optional[datetime] = None
    departure_uncertainty: Optional[int] = None
    schedule_relationship: ScheduleRelationship = ScheduleRelationship.SCHEDULED


@dataclass
class ServiceAlert(TemporalEntity):
    """Service alert"""
    alert_id: str = ""
    
    # Affected entities
    affected_routes: List[str] = field(default_factory=list)
    affected_stops: List[str] = field(default_factory=list)
    affected_trips: List[str] = field(default_factory=list)
    
    # Alert content
    header_text: str = ""
    description_text: Optional[str] = None
    url: Optional[str] = None
    
    # Cause and effect
    cause: AlertCause = AlertCause.UNKNOWN_CAUSE
    effect: AlertEffect = AlertEffect.UNKNOWN_EFFECT
    
    # Active period
    active_period_start: Optional[datetime] = None
    active_period_end: Optional[datetime] = None
    
    # Severity
    severity_level: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "ServiceAlert"


if __name__ == "__main__":
    # Example usage
    agency = Agency(
        agency_id="DMRC",
        entity_id="DMRC",
        agency_name="Delhi Metro Rail Corporation",
        agency_url="https://www.delhimetrorail.com",
        agency_timezone="Asia/Kolkata",
        agency_lang="en"
    )
    
    stop = Stop(
        stop_id="STOP_001",
        entity_id="STOP_001",
        stop_name="Rajiv Chowk",
        geometry=None,
        centroid=(77.2090, 28.6328),
        location_type=LocationType.STATION,
        wheelchair_accessible=WheelchairAccessible.ACCESSIBLE,
        taz_id="TAZ_001"
    )
    
    route = Route(
        route_id="BLUE_LINE",
        entity_id="BLUE_LINE",
        agency_id="DMRC",
        route_short_name="Blue Line",
        route_long_name="Dwarka Sector 21 - Noida Electronic City / Vaishali",
        route_type=GTFSRouteType.SUBWAY,
        route_color="0000FF"
    )
    
    vehicle_pos = VehiclePosition(
        vehicle_id="VEH_001",
        entity_id="VP_001",
        trip_id="TRIP_001",
        route_id="BLUE_LINE",
        geometry=None,
        centroid=(77.2090, 28.6328),
        bearing=90.0,
        speed_mps=20.0,
        current_status=VehicleStopStatus.IN_TRANSIT,
        occupancy_status=OccupancyStatus.MANY_SEATS,
        timestamp=datetime.now()
    )
    
    print(f"Agency: {agency.agency_name}")
    print(f"Stop: {stop.stop_name}")
    print(f"Route: {route.route_long_name}")
    print(f"Vehicle at {vehicle_pos.speed_mps} m/s, occupancy: {vehicle_pos.occupancy_status.value}")
