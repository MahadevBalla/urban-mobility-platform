"""
Transport Data Ontology Package

A comprehensive ontology for integrating 14 diverse urban mobility data sources
for 4-step travel demand modeling.

Version: 1.0.0
Created: November 2025
Institution: IIT Bombay Research Internship
"""

__version__ = "1.0.0"
__author__ = "IIT Bombay Research Team"
__description__ = "Transport Data Ontology for 4-Step Travel Demand Modeling"

# Base classes and enumerations
from .ontology_base import (
    # Abstract base classes
    TransportEntity,
    SpatialEntity,
    TemporalEntity,
    AgentEntity,
    ActivityEntity,
    InfrastructureEntity,
    
    # Enumerations
    GeographyType,
    Sex,
    EmploymentStatus,
    EducationLevel,
    TripPurpose,
    TransportMode,
    GTFSRouteType,
    HighwayClass,
    LandUseType,
    IncidentSeverity,
    TimeOfDay,
    DayType,
    DataQuality,
    
    # Value objects
    Relationship,
    RelationshipType,
    TimePeriod,
    Location,
    Zone,
    ODPair,
)

# Census data
from .ontology_census import (
    Race,
    Ethnicity,
    HouseholdType,
    DwellingType,
    Tenure,
    IncomeBracket,
    GeographicUnit,
    Household,
    Person,
    SyntheticPopulation,
    ControlTotal,
)

# Household travel surveys
from .ontology_hts import (
    WeatherCondition,
    TransitPass,
    ParkingType as HTSParkingType,
    PickupDropoffType as HTSPickupDropoffType,
    Survey,
    SurveyHousehold,
    SurveyPerson,
    Trip,
    Tour,
    Activity,
    TripChain,
)

# Mobile phone OD data
from .ontology_mobile import (
    CDREventType,
    NetworkTechnology,
    DeviceType,
    SubscriptionType,
    MobileOperator,
    CellTower,
    AnonymizedUser,
    CDREvent,
    StayPoint,
    InferredTrip,
    ODMatrix_Mobile,
    MobilityPattern,
)

# GPS probe data
from .ontology_probe import (
    ProbeProvider,
    VehicleType,
    CongestionLevel as ProbeCongestionLevel,
    ProbeDataProvider,
    ProbeVehicle,
    GPSTracePoint,
    MatchedTrace,
    SpeedObservation,
    TravelTimeSegment,
    TrafficFlow,
    ODTrajectory,
)

# GTFS data
from .ontology_gtfs import (
    LocationType,
    WheelchairAccessible,
    BikesAllowed,
    PickupDropoffType as GTFSPickupDropoffType,
    Timepoint,
    OccupancyStatus,
    CongestionLevel as GTFSCongestionLevel,
    VehicleStopStatus,
    ScheduleRelationship,
    AlertCause,
    AlertEffect,
    Agency,
    Stop,
    Route,
    Trip as GTFSTrip,
    StopTime,
    Calendar,
    CalendarDate,
    Shape,
    VehiclePosition,
    TripUpdate,
    StopTimeUpdate,
    ServiceAlert,
)

# OpenStreetMap data
from .ontology_osm import (
    BuildingType,
    AmenityType,
    ShopType,
    SurfaceType,
    OSMNode,
    OSMWay,
    RoadSegment,
    Building,
    PointOfInterest,
    TransitStop,
    Barrier,
    Intersection,
)

# Module metadata
MODULES = {
    'base': 'Core ontology base classes and enumerations',
    'census': 'Census and demographic data',
    'hts': 'Household travel surveys',
    'mobile': 'Mobile phone OD data',
    'probe': 'GPS probe/floating car data',
    'gtfs': 'Public transit GTFS data',
    'osm': 'OpenStreetMap data',
}

PENDING_MODULES = {
    'ticketing': 'Automated fare collection (AFC)',
    'tmc': 'Traffic message channel incidents',
    'landuse': 'Land use and parcel data',
    'parking': 'Parking facility data',
    'traffic': 'Traffic count data',
    'airquality': 'Air quality measurements',
    'accidents': 'Crash/accident data',
    'integrator': 'Cross-source data integration',
}

def get_version():
    """Get ontology version"""
    return __version__

def list_modules():
    """List available ontology modules"""
    return MODULES

def list_pending_modules():
    """List modules to be implemented"""
    return PENDING_MODULES

def get_entity_count():
    """Get count of entity classes"""
    import inspect
    from . import ontology_base, ontology_census, ontology_hts
    from . import ontology_mobile, ontology_probe, ontology_gtfs, ontology_osm
    
    modules = [
        ontology_base, ontology_census, ontology_hts,
        ontology_mobile, ontology_probe, ontology_gtfs, ontology_osm
    ]
    
    count = 0
    for module in modules:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, '__dataclass_fields__'):
                count += 1
    
    return count

def get_enum_count():
    """Get count of enumerations"""
    import inspect
    from enum import Enum
    from . import ontology_base, ontology_census, ontology_hts
    from . import ontology_mobile, ontology_probe, ontology_gtfs, ontology_osm
    
    modules = [
        ontology_base, ontology_census, ontology_hts,
        ontology_mobile, ontology_probe, ontology_gtfs, ontology_osm
    ]
    
    count = 0
    for module in modules:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Enum) and obj != Enum:
                count += 1
    
    return count

def print_summary():
    """Print ontology summary"""
    print(f"Transport Data Ontology v{__version__}")
    print(f"=" * 50)
    print(f"\nImplemented Modules: {len(MODULES)}")
    for module, description in MODULES.items():
        print(f"  - {module}: {description}")
    
    print(f"\nPending Modules: {len(PENDING_MODULES)}")
    for module, description in PENDING_MODULES.items():
        print(f"  - {module}: {description}")
    
    print(f"\nStatistics:")
    print(f"  Entity Classes: {get_entity_count()}")
    print(f"  Enumerations: {get_enum_count()}")
    print(f"\nStandards Aligned:")
    print(f"  - Census PUMS, ACS")
    print(f"  - NHTS, Regional HTS")
    print(f"  - GTFS Static & Realtime")
    print(f"  - OpenStreetMap Data Model")
    print(f"  - Transmodel, ActivitySim, MATSim")

__all__ = [
    # Version info
    '__version__',
    'get_version',
    'list_modules',
    'print_summary',
    
    # Base classes
    'TransportEntity',
    'SpatialEntity',
    'TemporalEntity',
    'AgentEntity',
    'ActivityEntity',
    'InfrastructureEntity',
    
    # Common enumerations
    'GeographyType',
    'Sex',
    'EmploymentStatus',
    'EducationLevel',
    'TripPurpose',
    'TransportMode',
    'TimeOfDay',
    'DayType',
    
    # Value objects
    'Zone',
    'ODPair',
    'Location',
    'TimePeriod',
    
    # Census
    'GeographicUnit',
    'Household',
    'Person',
    
    # HTS
    'Survey',
    'Trip',
    'Tour',
    'Activity',
    
    # Mobile
    'CellTower',
    'InferredTrip',
    'ODMatrix_Mobile',
    
    # Probe
    'GPSTracePoint',
    'SpeedObservation',
    'TravelTimeSegment',
    
    # GTFS
    'Agency',
    'Stop',
    'Route',
    'VehiclePosition',
    
    # OSM
    'RoadSegment',
    'Building',
    'PointOfInterest',
]

if __name__ == "__main__":
    print_summary()
