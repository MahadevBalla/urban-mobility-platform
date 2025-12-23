"""Transport Data Ontology - Base Classes

Core abstract classes and enumerations for transport data integration.
Aligned with Transmodel, GTFS, ActivitySim, and MATSim standards.

Version: 1.0.0
Created: November 2025
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, date, time
from decimal import Decimal
import json


# ============================================================================
# ENUMERATIONS
# ============================================================================

class GeographyType(Enum):
    """Geographic unit classifications"""
    STATE = "state"
    COUNTY = "county"
    TRACT = "tract"
    BLOCK_GROUP = "block_group"
    PUMA = "puma"
    TAZ = "taz"  # Traffic Analysis Zone
    H3_HEX = "h3_hex"
    CUSTOM = "custom"


class Sex(Enum):
    """Biological sex categories"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class EmploymentStatus(Enum):
    """Employment status classifications"""
    EMPLOYED_FULL_TIME = "employed_full_time"
    EMPLOYED_PART_TIME = "employed_part_time"
    UNEMPLOYED = "unemployed"
    NOT_IN_LABOR_FORCE = "not_in_labor_force"
    STUDENT = "student"
    RETIRED = "retired"


class EducationLevel(Enum):
    """Education attainment levels"""
    LESS_THAN_HS = "less_than_high_school"
    HS_DIPLOMA = "high_school_diploma"
    SOME_COLLEGE = "some_college"
    ASSOCIATES = "associates_degree"
    BACHELORS = "bachelors_degree"
    MASTERS = "masters_degree"
    DOCTORAL = "doctoral_degree"


class TripPurpose(Enum):
    """Trip purposes (NHTS/HTS aligned)"""
    HOME = "home"
    WORK = "work"
    SCHOOL = "school"
    SHOPPING = "shopping"
    SOCIAL_RECREATION = "social_recreation"
    EAT_MEAL = "eat_meal"
    PERSONAL_BUSINESS = "personal_business"
    ESCORT = "escort"  # Accompanying others
    MEDICAL = "medical"
    OTHER = "other"


class TransportMode(Enum):
    """Transport modes (Transmodel aligned)"""
    WALK = "walk"
    BICYCLE = "bicycle"
    EBIKE = "ebike"
    ESCOOTER = "escooter"
    CAR_DRIVER = "car_driver"
    CAR_PASSENGER = "car_passenger"
    MOTORCYCLE = "motorcycle"
    TAXI = "taxi"
    RIDESHARE = "rideshare"  # Uber, Lyft, etc.
    BUS = "bus"
    METRO = "metro"
    TRAM = "tram"
    RAIL = "rail"
    FERRY = "ferry"
    SCHOOL_BUS = "school_bus"
    AUTO_RICKSHAW = "auto_rickshaw"  # India-specific
    OTHER = "other"


class GTFSRouteType(Enum):
    """GTFS route type codes"""
    TRAM = 0
    SUBWAY = 1
    RAIL = 2
    BUS = 3
    FERRY = 4
    CABLE_TRAM = 5
    AERIAL_LIFT = 6
    FUNICULAR = 7
    TROLLEYBUS = 11
    MONORAIL = 12


class HighwayClass(Enum):
    """OSM highway classifications"""
    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    SERVICE = "service"
    CYCLEWAY = "cycleway"
    FOOTWAY = "footway"
    PATH = "path"


class LandUseType(Enum):
    """Land use categories (LBCS aligned)"""
    RESIDENTIAL_SINGLE = "residential_single_family"
    RESIDENTIAL_MULTI = "residential_multi_family"
    COMMERCIAL_RETAIL = "commercial_retail"
    COMMERCIAL_OFFICE = "commercial_office"
    INDUSTRIAL_LIGHT = "industrial_light"
    INDUSTRIAL_HEAVY = "industrial_heavy"
    INSTITUTIONAL_SCHOOL = "institutional_school"
    INSTITUTIONAL_HOSPITAL = "institutional_hospital"
    INSTITUTIONAL_GOVERNMENT = "institutional_government"
    RECREATIONAL = "recreational"
    AGRICULTURAL = "agricultural"
    VACANT = "vacant"
    MIXED_USE = "mixed_use"
    TRANSPORTATION = "transportation"


class IncidentSeverity(Enum):
    """Traffic incident severity levels"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class TimeOfDay(Enum):
    """Time of day periods for analysis"""
    EARLY_MORNING = "early_morning"  # 5-7 AM
    AM_PEAK = "am_peak"  # 7-10 AM
    MIDDAY = "midday"  # 10 AM - 3 PM
    PM_PEAK = "pm_peak"  # 3-7 PM
    EVENING = "evening"  # 7-10 PM
    NIGHT = "night"  # 10 PM - 5 AM


class DayType(Enum):
    """Day type classifications"""
    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    HOLIDAY = "holiday"


class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # 0.2 - 0.5
    VERY_LOW = "very_low"  # < 0.2
    UNKNOWN = "unknown"


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

@dataclass
class TransportEntity:
    """Abstract base class for all ontology entities"""
    entity_id: str
    entity_type: str
    data_source: Optional[str] = None
    data_quality_score: Optional[float] = None  # 0-1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (datetime, date, time)):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SpatialEntity(TransportEntity):
    """Base for entities with spatial attributes"""
    geometry: Any = None  # Shapely geometry object (Point, LineString, Polygon)
    srid: int = 4326  # WGS84 default
    centroid: Optional[Tuple[float, float]] = None  # (lon, lat)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)

    def get_wkt(self) -> str:
        """Get Well-Known Text representation"""
        if self.geometry:
            return self.geometry.wkt
        return None

    def get_geojson(self) -> Dict[str, Any]:
        """Get GeoJSON representation"""
        if self.geometry:
            return {
                "type": "Feature",
                "geometry": self.geometry.__geo_interface__,
                "properties": self.to_dict()
            }
        return None


@dataclass
class TemporalEntity(TransportEntity):
    """Base for entities with temporal validity"""
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    is_current: bool = True
    temporal_resolution: Optional[str] = None  # e.g., "15min", "1hour", "1day"

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if entity is valid at given timestamp"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True


@dataclass
class AgentEntity(TransportEntity):
    """Base for agents (people, vehicles, organizations)"""
    agent_name: Optional[str] = None
    agent_category: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityEntity(TransportEntity):
    """Base for activities (trips, tours)"""
    origin_id: Optional[str] = None
    destination_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None


@dataclass
class InfrastructureEntity(SpatialEntity):
    """Base for infrastructure (facilities, sensors)"""
    facility_name: Optional[str] = None
    capacity: Optional[int] = None
    operator: Optional[str] = None
    status: Optional[str] = "operational"


# ============================================================================
# RELATIONSHIPS
# ============================================================================

@dataclass
class Relationship:
    """Relationship between two entities"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: Optional[float] = None  # 0-1 confidence
    created_at: datetime = field(default_factory=datetime.utcnow)


class RelationshipType(Enum):
    """Standard relationship types"""
    IS_A = "is_a"  # Inheritance
    HAS_A = "has_a"  # Composition
    PART_OF = "part_of"  # Aggregation
    LOCATED_IN = "located_in"  # Spatial containment
    CONNECTED_TO = "connected_to"  # Network connectivity
    SERVED_BY = "served_by"  # Service relationship
    USES = "uses"  # Usage relationship
    PRODUCES = "produces"  # Production
    CONSUMES = "consumes"  # Consumption
    FLOWS_TO = "flows_to"  # Flow/movement


# ============================================================================
# COMMON VALUE OBJECTS
# ============================================================================

@dataclass
class TimePeriod:
    """Time period definition"""
    period_id: str
    start_time: time
    end_time: time
    day_types: List[DayType]
    time_of_day: TimeOfDay


@dataclass
class Location:
    """Generic location"""
    location_id: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    zone_id: Optional[str] = None
    place_name: Optional[str] = None


@dataclass
class Zone:
    """Traffic Analysis Zone"""
    zone_id: str
    area_km2: float
    geometry: Any = None  # Polygon
    zone_name: Optional[str] = None
    population: Optional[int] = None
    employment: Optional[int] = None
    centroid: Optional[Tuple[float, float]] = None
    land_use_mix: Dict[LandUseType, float] = field(default_factory=dict)


@dataclass
class ODPair:
    """Origin-Destination Pair"""
    origin_zone_id: str
    destination_zone_id: str
    trip_count: int
    time_period: Optional[TimePeriod] = None
    average_travel_time_minutes: Optional[float] = None
    distance_km: Optional[float] = None
    mode_split: Dict[TransportMode, float] = field(default_factory=dict)


if __name__ == "__main__":
    # Example usage
    zone = Zone(
        zone_id="TAZ_001",
        area_km2=2.5,
        geometry=None,
        zone_name="Central Business District",
        population=15000,
        employment=25000
    )
    print(f"Created zone: {zone.zone_id} with {zone.population} residents")
