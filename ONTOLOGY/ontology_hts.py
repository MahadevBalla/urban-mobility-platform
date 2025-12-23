"""Household Travel Survey (HTS) Ontology

Entities for travel surveys aligned with:
- NHTS (National Household Travel Survey)
- Regional household travel surveys
- ActivitySim input format

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time
from enum import Enum

from ontology_base import (
    TransportEntity, ActivityEntity, AgentEntity,
    TripPurpose, TransportMode, TimeOfDay, DayType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class WeatherCondition(Enum):
    """Weather during travel"""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    OVERCAST = "overcast"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    FOG = "fog"


class TransitPass(Enum):
    """Transit pass ownership"""
    NONE = "none"
    MONTHLY = "monthly"
    ANNUAL = "annual"
    EMPLOYER_PROVIDED = "employer_provided"
    STUDENT = "student"
    SENIOR = "senior"


class ParkingType(Enum):
    """Parking type used"""
    FREE_LOT = "free_lot"
    FREE_STREET = "free_street"
    PAID_LOT = "paid_lot"
    PAID_GARAGE = "paid_garage"
    PAID_STREET = "paid_street"
    NONE = "none"


class PickupDropoffType(Enum):
    """GTFS-aligned pickup/dropoff types"""
    REGULAR = "regular"
    NOT_AVAILABLE = "not_available"
    PHONE_AGENCY = "phone_agency"
    COORDINATE_WITH_DRIVER = "coordinate_with_driver"


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class Survey(TransportEntity):
    """Travel survey metadata"""
    survey_id: str = ""
    survey_name: str = ""
    survey_year: int = 0
    survey_region: str = ""
    survey_type: str = "household_travel_survey"
    
    # Methodology
    sampling_methodology: Optional[str] = None
    sample_size_households: Optional[int] = None
    sample_size_persons: Optional[int] = None
    response_rate: Optional[float] = None
    
    # Time coverage
    survey_start_date: Optional[date] = None
    survey_end_date: Optional[date] = None
    
    # Geographic coverage
    geographic_extent: Optional[str] = None  # WKT or description

    def __post_init__(self):
        self.entity_type = "Survey"


@dataclass
class SurveyHousehold(AgentEntity):
    """Household in travel survey (extends Census Household)"""
    survey_household_id: str = ""
    survey_id: str = ""  # FK to Survey
    
    # Interview details
    interview_date: Optional[date] = None
    day_of_week: Optional[DayType] = None
    is_weekend: bool = False
    is_holiday: bool = False
    weather_condition: Optional[WeatherCondition] = None
    
    # All Census Household attributes can be included
    household_size: Optional[int] = None
    num_workers: Optional[int] = None
    household_income: Optional[float] = None
    vehicles_available: Optional[int] = None
    zone_id: Optional[str] = None
    
    # Survey weights
    household_weight: Optional[float] = None
    expansion_factor: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "SurveyHousehold"


@dataclass
class SurveyPerson(AgentEntity):
    """Person in travel survey (extends Census Person)"""
    survey_person_id: str = ""
    survey_household_id: str = ""  # FK to SurveyHousehold
    person_num: int = 1
    
    # Demographics
    age: Optional[int] = None
    sex: Optional[str] = None
    employment_status: Optional[str] = None
    education_level: Optional[str] = None
    
    # Transportation resources
    has_drivers_license: Optional[bool] = None
    has_transit_pass: Optional[TransitPass] = None
    has_bicycle: Optional[bool] = None
    smartphone_ownership: Optional[bool] = None
    
    # Usual behavior
    usual_mode_to_work: Optional[TransportMode] = None
    usual_departure_time: Optional[time] = None
    work_from_home_days_per_week: Optional[int] = None
    
    # Survey metadata
    completed_diary: bool = True
    person_weight: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "SurveyPerson"


@dataclass
class Trip(ActivityEntity):
    """Individual trip from travel survey"""
    trip_id: str = ""
    survey_person_id: str = ""  # FK to SurveyPerson
    trip_num: int = 0  # Sequence number within person's day
    
    # Spatial
    origin_location: Optional[Dict[str, float]] = None  # {lat, lon}
    destination_location: Optional[Dict[str, float]] = None
    origin_zone_id: Optional[str] = None
    destination_zone_id: Optional[str] = None
    origin_purpose: Optional[TripPurpose] = None
    destination_purpose: TripPurpose = TripPurpose.OTHER
    
    # Temporal
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    travel_time_minutes: Optional[float] = None
    time_of_day: Optional[TimeOfDay] = None
    
    # Travel characteristics
    mode: Optional[TransportMode] = None
    distance_km: Optional[float] = None
    distance_miles: Optional[float] = None
    
    # Mode-specific attributes
    vehicle_occupancy: Optional[int] = None  # For car trips
    transit_route: Optional[str] = None
    transit_line: Optional[str] = None
    transit_boarding_stop: Optional[str] = None
    transit_alighting_stop: Optional[str] = None
    num_transfers: Optional[int] = None
    
    # Costs
    parking_type: Optional[ParkingType] = None
    parking_cost: Optional[float] = None
    parking_duration_minutes: Optional[float] = None
    transit_fare: Optional[float] = None
    toll_cost: Optional[float] = None
    
    # Context
    trip_purpose: TripPurpose = TripPurpose.OTHER
    activity_duration_at_destination: Optional[float] = None
    
    # Quality
    is_imputed: bool = False
    geocode_quality: Optional[str] = None
    trip_weight: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "Trip"


@dataclass
class Tour(TransportEntity):
    """Tour (chain of trips with return to origin)"""
    tour_id: str = ""
    survey_person_id: str = ""  # FK to SurveyPerson
    tour_num: int = 0  # Sequence number within person's day
    
    # Tour characteristics
    tour_purpose: Optional[TripPurpose] = None  # Primary purpose
    anchor_location: Optional[Dict[str, float]] = None  # Work/school location
    origin_location: Optional[Dict[str, float]] = None  # Usually home
    
    # Temporal
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    
    # Structure
    num_trips: int = 0
    num_intermediate_stops: int = 0
    trip_ids: List[str] = field(default_factory=list)  # Ordered list
    
    # Mode
    primary_mode: Optional[TransportMode] = None
    is_multimodal: bool = False
    
    # Spatial
    total_distance_km: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "Tour"


@dataclass
class Activity(TransportEntity):
    """Activity episode at a location"""
    activity_id: str = ""
    survey_person_id: str = ""  # FK to SurveyPerson
    activity_num: int = 0
    
    # Location
    location: Optional[Dict[str, float]] = None  # {lat, lon}
    zone_id: Optional[str] = None
    place_name: Optional[str] = None
    
    # Activity type
    activity_type: Optional[TripPurpose] = None
    activity_category: Optional[str] = None  # More detailed
    
    # Temporal
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    
    # Context
    with_household_members: Optional[List[str]] = field(default_factory=list)
    with_non_household: Optional[int] = None
    
    # Trips
    preceding_trip_id: Optional[str] = None
    following_trip_id: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "Activity"


@dataclass
class TripChain:
    """Complete trip chain for a person on survey day"""
    chain_id: str
    survey_person_id: str
    survey_date: date
    
    # Trips in order
    trips: List[Trip] = field(default_factory=list)
    tours: List[Tour] = field(default_factory=list)
    activities: List[Activity] = field(default_factory=list)
    
    # Summary statistics
    total_trips: int = 0
    total_distance_km: Optional[float] = None
    total_travel_time_minutes: Optional[float] = None
    modes_used: List[TransportMode] = field(default_factory=list)


if __name__ == "__main__":
    # Example usage
    survey = Survey(
        survey_id="SURVEY_2023",
        entity_id="SURVEY_2023",
        survey_name="Delhi Metro Region Travel Survey",
        survey_year=2023,
        survey_region="Delhi NCR",
        sample_size_households=5000
    )
    
    trip = Trip(
        trip_id="TRIP_001",
        entity_id="TRIP_001",
        survey_person_id="P_001",
        trip_num=1,
        origin_zone_id="TAZ_001",
        destination_zone_id="TAZ_045",
        departure_time=datetime(2023, 10, 15, 8, 30),
        arrival_time=datetime(2023, 10, 15, 9, 15),
        travel_time_minutes=45.0,
        mode=TransportMode.METRO,
        trip_purpose=TripPurpose.WORK,
        distance_km=15.5
    )
    
    print(f"Created {survey.survey_name} with {survey.sample_size_households} households")
    print(f"Trip {trip.trip_id}: {trip.mode.value} from zone {trip.origin_zone_id} to {trip.destination_zone_id}")
