"""Census Data Ontology

Entities for Census and demographic data aligned with:
- Census PUMS (Public Use Microdata Sample)
- ACS (American Community Survey)
- PopulationSim data structures

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import date
from enum import Enum

from ontology_base import (
    TransportEntity, SpatialEntity, AgentEntity,
    Sex, EmploymentStatus, EducationLevel, GeographyType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class Race(Enum):
    """Race categories (Census aligned)"""
    WHITE = "white"
    BLACK = "black_or_african_american"
    ASIAN = "asian"
    PACIFIC_ISLANDER = "native_hawaiian_or_pacific_islander"
    NATIVE_AMERICAN = "american_indian_or_alaska_native"
    TWO_OR_MORE = "two_or_more_races"
    OTHER = "some_other_race"


class Ethnicity(Enum):
    """Ethnicity categories"""
    HISPANIC = "hispanic_or_latino"
    NON_HISPANIC = "not_hispanic_or_latino"


class HouseholdType(Enum):
    """Household composition types"""
    SINGLE_PERSON = "single_person"
    MARRIED_COUPLE = "married_couple_no_children"
    MARRIED_WITH_CHILDREN = "married_couple_with_children"
    SINGLE_PARENT = "single_parent"
    NON_FAMILY = "non_family"
    OTHER = "other"


class DwellingType(Enum):
    """Dwelling unit types"""
    SINGLE_FAMILY_DETACHED = "single_family_detached"
    SINGLE_FAMILY_ATTACHED = "single_family_attached"
    APARTMENT_2_4_UNITS = "apartment_2_to_4_units"
    APARTMENT_5_PLUS_UNITS = "apartment_5_plus_units"
    MOBILE_HOME = "mobile_home"
    OTHER = "other"


class Tenure(Enum):
    """Housing tenure"""
    OWNER_OCCUPIED = "owner_occupied"
    RENTER_OCCUPIED = "renter_occupied"


class IncomeBracket(Enum):
    """Household income brackets (USD annual)"""
    LESS_THAN_15K = "less_than_15000"
    FROM_15K_TO_25K = "15000_to_24999"
    FROM_25K_TO_35K = "25000_to_34999"
    FROM_35K_TO_50K = "35000_to_49999"
    FROM_50K_TO_75K = "50000_to_74999"
    FROM_75K_TO_100K = "75000_to_99999"
    FROM_100K_TO_150K = "100000_to_149999"
    FROM_150K_TO_200K = "150000_to_199999"
    ABOVE_200K = "200000_or_more"


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class GeographicUnit(SpatialEntity):
    """Census geographic unit (PUMA, Tract, Block Group, etc.)"""
    geography_id: str = ""  # FIPS code, PUMA code, etc. (required in practice)
    geography_type: Optional[GeographyType] = None
    geography_name: str = ""
    parent_geography_id: Optional[str] = None
    population_total: Optional[int] = None
    households_total: Optional[int] = None
    area_land_km2: Optional[float] = None
    area_water_km2: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "GeographicUnit"


@dataclass
class Household(AgentEntity):
    """Census household unit"""
    household_id: str = ""  # Unique identifier (required in practice)
    serial_no: Optional[str] = None  # Census serial number
    geography_id: Optional[str] = None  # FK to GeographicUnit
    zone_id: Optional[str] = None  # FK to TAZ
    
    # Demographics
    household_size: int = 1
    num_adults: Optional[int] = None
    num_children: Optional[int] = None
    num_workers: Optional[int] = None
    num_students: Optional[int] = None
    
    # Housing
    dwelling_type: Optional[DwellingType] = None
    tenure: Optional[Tenure] = None
    year_built: Optional[int] = None
    num_bedrooms: Optional[int] = None
    num_rooms: Optional[int] = None
    
    # Economic
    household_income: Optional[float] = None
    income_bracket: Optional[IncomeBracket] = None
    
    # Transportation
    vehicles_available: int = 0
    bicycles_available: Optional[int] = None
    
    # Weights
    survey_weight: Optional[float] = None
    expansion_factor: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "Household"


@dataclass
class Person(AgentEntity):
    """Census person/individual"""
    person_id: str = ""  # Unique identifier (required in practice)
    household_id: str = ""  # FK to Household (required in practice)
    age: int = 0  # Age in years (required in practice)
    sex: Optional[Sex] = None
    person_num: int = 1  # Person number within household
    serial_no: Optional[str] = None  # Census serial number
    race: Optional[Race] = None
    ethnicity: Optional[Ethnicity] = None
    relationship_to_householder: Optional[str] = None
    marital_status: Optional[str] = None
    
    # Education
    education_level: Optional[EducationLevel] = None
    currently_enrolled: bool = False
    school_location_zone_id: Optional[str] = None
    
    # Employment
    employment_status: Optional[EmploymentStatus] = None
    occupation_code: Optional[str] = None  # SOC code
    industry_code: Optional[str] = None  # NAICS code
    work_location_zone_id: Optional[str] = None
    hours_worked_per_week: Optional[int] = None
    weeks_worked_per_year: Optional[int] = None
    
    # Income
    personal_income: Optional[float] = None
    earnings: Optional[float] = None
    
    # Transportation
    has_drivers_license: Optional[bool] = None
    commute_mode: Optional[str] = None
    commute_time_minutes: Optional[int] = None
    
    # Weights
    person_weight: Optional[float] = None

    def __post_init__(self):
        self.entity_type = "Person"


@dataclass
class SyntheticPopulation:
    """Synthetic population metadata (from PopulationSim)"""
    population_id: str
    geography_id: str
    year: int
    total_persons: int
    total_households: int
    
    # Control totals
    controls_applied: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    goodness_of_fit: Optional[float] = None
    convergence_achieved: bool = False
    
    # Provenance
    creation_method: str = "population_sim"
    seed_source: Optional[str] = None  # e.g., "ACS_2020_PUMS"
    created_date: Optional[date] = None


@dataclass
class ControlTotal:
    """Population synthesis control total"""
    control_id: str
    geography_id: str
    geography_type: GeographyType
    
    # Control specification
    control_name: str
    control_category: str  # e.g., "age_sex", "household_income"
    target_value: int
    importance_weight: float = 1.0
    
    # Actual values
    achieved_value: Optional[int] = None
    absolute_error: Optional[int] = None
    relative_error: Optional[float] = None


if __name__ == "__main__":
    # Example usage
    hh = Household(
        household_id="HH_001",
        entity_id="HH_001",
        household_size=4,
        num_workers=2,
        household_income=75000.0,
        income_bracket=IncomeBracket.FROM_50K_TO_75K,
        vehicles_available=2,
        tenure=Tenure.OWNER_OCCUPIED
    )
    
    person = Person(
        person_id="P_001",
        household_id="HH_001",
        entity_id="P_001",
        age=35,
        sex=Sex.MALE,
        education_level=EducationLevel.BACHELORS,
        employment_status=EmploymentStatus.EMPLOYED_FULL_TIME,
        has_drivers_license=True
    )
    
    print(f"Created household {hh.household_id} with {hh.household_size} persons")
    print(f"Person {person.person_id}: {person.age} years, {person.sex.value}")
