"""Mobile Phone OD Data Ontology

Entities for mobile phone-based origin-destination data:
- CDR (Call Detail Records)
- Mobile network positioning
- Inferred trips and OD matrices

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from ontology_base import (
    TransportEntity, SpatialEntity, ActivityEntity,
    TimeOfDay, DayType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class CDREventType(Enum):
    """CDR event types"""
    CALL_OUTGOING = "call_outgoing"
    CALL_INCOMING = "call_incoming"
    SMS_OUTGOING = "sms_outgoing"
    SMS_INCOMING = "sms_incoming"
    DATA_SESSION = "data_session"
    LOCATION_UPDATE = "location_update"


class NetworkTechnology(Enum):
    """Mobile network technology"""
    GSM_2G = "2G"
    UMTS_3G = "3G"
    LTE_4G = "4G"
    NR_5G = "5G"


class DeviceType(Enum):
    """Mobile device types"""
    SMARTPHONE = "smartphone"
    FEATURE_PHONE = "feature_phone"
    TABLET = "tablet"
    IOT_DEVICE = "iot_device"


class SubscriptionType(Enum):
    """Subscription types"""
    PREPAID = "prepaid"
    POSTPAID = "postpaid"


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class MobileOperator(TransportEntity):
    """Mobile network operator"""
    operator_id: str = ""
    operator_name: str = ""
    country_code: str = ""
    network_technology: Optional[NetworkTechnology] = None
    market_share_percent: Optional[float] = None
    
    # Coverage
    coverage_area: Optional[str] = None  # WKT polygon
    num_cell_towers: Optional[int] = None

    def __post_init__(self):
        self.entity_type = "MobileOperator"


@dataclass
class CellTower(SpatialEntity):
    """Cell tower / base station"""
    tower_id: str = ""
    operator_id: str = ""  # FK to MobileOperator
    
    # Location details (geometry in SpatialEntity)
    elevation_m: Optional[float] = None
    azimuth_degrees: Optional[float] = None  # Antenna direction
    coverage_radius_m: Optional[float] = None
    
    # Network
    technology: NetworkTechnology = NetworkTechnology.LTE_4G
    cell_id: Optional[str] = None
    lac_code: Optional[str] = None  # Location Area Code
    
    # Administrative
    zone_id: Optional[str] = None  # FK to TAZ
    deployment_date: Optional[datetime] = None
    status: str = "active"

    def __post_init__(self):
        self.entity_type = "CellTower"


@dataclass
class AnonymizedUser(TransportEntity):
    """Anonymized mobile user"""
    anon_user_id: str = ""  # Hashed/encrypted identifier
    operator_id: str = ""  # FK to MobileOperator
    
    # Device
    device_type: DeviceType = DeviceType.SMARTPHONE
    subscription_type: SubscriptionType = SubscriptionType.PREPAID
    
    # Inferred characteristics (from usage patterns)
    estimated_age_group: Optional[str] = None  # e.g., "25-34"
    estimated_gender: Optional[str] = None  # M/F/Unknown
    
    # Home/work locations (most common towers)
    home_tower_id: Optional[str] = None
    home_zone_id: Optional[str] = None
    work_tower_id: Optional[str] = None
    work_zone_id: Optional[str] = None
    
    # Activity profile
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_events: Optional[int] = None

    def __post_init__(self):
        self.entity_type = "AnonymizedUser"


@dataclass
class CDREvent(TransportEntity):
    """Call Detail Record event"""
    event_id: str = ""
    anon_user_id: str = ""  # FK to AnonymizedUser
    tower_id: str = ""  # FK to CellTower
    
    # Event details
    timestamp: Optional[datetime] = None
    event_type: Optional[CDREventType] = None
    
    # Type-specific attributes
    duration_seconds: Optional[float] = None  # For calls
    data_volume_mb: Optional[float] = None  # For data sessions
    
    # Other party (if applicable, anonymized)
    other_party_anon_id: Optional[str] = None
    
    # Signal quality
    signal_strength: Optional[int] = None  # dBm

    def __post_init__(self):
        self.entity_type = "CDREvent"


@dataclass
class StayPoint(SpatialEntity):
    """Detected stay point (dwell location)"""
    stay_point_id: str = ""
    anon_user_id: str = ""  # FK to AnonymizedUser
    
    # Location
    tower_id: str = ""  # FK to CellTower
    zone_id: Optional[str] = None
    
    # Temporal
    arrival_time: Optional[datetime] = None
    departure_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    
    # Classification
    is_home: bool = False
    is_work: bool = False
    activity_type: Optional[str] = None
    
    # Events during stay
    num_events: int = 0
    event_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.entity_type = "StayPoint"


@dataclass
class InferredTrip(ActivityEntity):
    """Trip inferred from CDR data"""
    inferred_trip_id: str = ""
    anon_user_id: str = ""  # FK to AnonymizedUser
    
    # Origin
    origin_tower_id: str = ""
    origin_zone_id: Optional[str] = None
    origin_stay_point_id: Optional[str] = None
    
    # Destination
    destination_tower_id: str = ""
    destination_zone_id: Optional[str] = None
    destination_stay_point_id: Optional[str] = None
    
    # Temporal
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    travel_time_minutes: float = 0.0
    time_of_day: Optional[TimeOfDay] = None
    day_of_week: Optional[DayType] = None
    
    # Confidence
    confidence_score: float = 0.0  # 0-1
    num_supporting_events: int = 0
    
    # Attributes
    euclidean_distance_km: Optional[float] = None
    network_distance_km: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "InferredTrip"


@dataclass
class ODMatrix_Mobile(TransportEntity):
    """OD matrix derived from mobile phone data"""
    od_matrix_id: str = ""
    
    # Temporal scope
    date: Optional[datetime] = None
    time_period: Optional[str] = None  # e.g., "AM_PEAK"
    day_type: Optional[DayType] = None
    
    # Spatial scope
    origin_zone_id: str = ""
    destination_zone_id: str = ""
    
    # Values
    trip_count: int = 0
    average_travel_time_minutes: Optional[float] = None
    std_travel_time_minutes: Optional[float] = None
    
    # Sample information
    sample_size: int = 0  # Number of mobile users
    expansion_factor: float = 1.0  # To total population
    market_penetration_rate: Optional[float] = None
    
    # Quality
    confidence_interval_95: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.entity_type = "ODMatrix_Mobile"


@dataclass
class MobilityPattern:
    """Aggregate mobility pattern from mobile data"""
    pattern_id: str
    zone_id: str
    date: datetime
    time_of_day: TimeOfDay
    
    # Metrics
    unique_users_present: int
    users_arriving: int
    users_departing: int
    net_inflow: int  # arrivals - departures
    
    # Origins/destinations
    top_origin_zones: Dict[str, int] = field(default_factory=dict)
    top_destination_zones: Dict[str, int] = field(default_factory=dict)


if __name__ == "__main__":
    # Example usage
    operator = MobileOperator(
        operator_id="OP_001",
        entity_id="OP_001",
        operator_name="Bharti Airtel",
        country_code="IN",
        network_technology=NetworkTechnology.LTE_4G,
        market_share_percent=32.5
    )
    
    tower = CellTower(
        tower_id="TOWER_001",
        entity_id="TOWER_001",
        operator_id="OP_001",
        geometry=None,
        centroid=(77.2090, 28.6139),  # Delhi coordinates
        coverage_radius_m=500,
        zone_id="TAZ_001"
    )
    
    inferred_trip = InferredTrip(
        inferred_trip_id="MTRIP_001",
        entity_id="MTRIP_001",
        anon_user_id="USER_ABC123",
        origin_tower_id="TOWER_001",
        destination_tower_id="TOWER_045",
        origin_zone_id="TAZ_001",
        destination_zone_id="TAZ_045",
        departure_time=datetime(2023, 10, 15, 8, 30),
        arrival_time=datetime(2023, 10, 15, 9, 15),
        travel_time_minutes=45.0,
        time_of_day=TimeOfDay.AM_PEAK,
        day_of_week=DayType.WEEKDAY,
        confidence_score=0.85
    )
    
    print(f"Mobile operator: {operator.operator_name} ({operator.market_share_percent}% market share)")
    print(f"Inferred trip from zone {inferred_trip.origin_zone_id} to {inferred_trip.destination_zone_id}")
    print(f"Confidence: {inferred_trip.confidence_score:.2f}")
