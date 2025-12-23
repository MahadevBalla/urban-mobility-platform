"""OpenStreetMap Data Ontology

Entities for OSM data:
- Road network
- Buildings
- Points of Interest
- Land use

Standards: OSM data model, OSMnx

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ontology_base import (
    TransportEntity, SpatialEntity,
    HighwayClass, LandUseType
)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class BuildingType(Enum):
    """OSM building types"""
    RESIDENTIAL = "residential"
    APARTMENTS = "apartments"
    HOUSE = "house"
    DETACHED = "detached"
    COMMERCIAL = "commercial"
    RETAIL = "retail"
    OFFICE = "office"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    UNIVERSITY = "university"
    PUBLIC = "public"
    TRANSPORTATION = "transportation"
    RELIGIOUS = "church"


class AmenityType(Enum):
    """OSM amenity types"""
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAR = "bar"
    SCHOOL = "school"
    UNIVERSITY = "university"
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    PHARMACY = "pharmacy"
    BANK = "bank"
    POST_OFFICE = "post_office"
    POLICE = "police"
    FIRE_STATION = "fire_station"
    PARKING = "parking"
    FUEL = "fuel"
    PLACE_OF_WORSHIP = "place_of_worship"


class ShopType(Enum):
    """OSM shop types"""
    SUPERMARKET = "supermarket"
    CONVENIENCE = "convenience"
    MALL = "mall"
    DEPARTMENT_STORE = "department_store"
    CLOTHES = "clothes"
    ELECTRONICS = "electronics"
    BAKERY = "bakery"
    BUTCHER = "butcher"
    GREENGROCER = "greengrocer"


class SurfaceType(Enum):
    """Road surface types"""
    PAVED = "paved"
    ASPHALT = "asphalt"
    CONCRETE = "concrete"
    UNPAVED = "unpaved"
    GRAVEL = "gravel"
    DIRT = "dirt"
    GROUND = "ground"


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class OSMNode(SpatialEntity):
    """OSM node (point)"""
    osm_node_id: int = 0  # OSM ID
    version: Optional[int] = None
    timestamp: Optional[datetime] = None
    changeset_id: Optional[int] = None
    user_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.entity_type = "OSMNode"
        self.entity_id = f"node_{self.osm_node_id}"


@dataclass
class OSMWay(SpatialEntity):
    """OSM way (line or polygon)"""
    osm_way_id: int = 0  # OSM ID
    nodes: List[int] = field(default_factory=list)  # Ordered node IDs
    version: Optional[int] = None
    timestamp: Optional[datetime] = None
    changeset_id: Optional[int] = None
    user_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Processed geometry (LineString or Polygon)
    is_closed: bool = False

    def __post_init__(self):
        self.entity_type = "OSMWay"
        self.entity_id = f"way_{self.osm_way_id}"


@dataclass
class RoadSegment(SpatialEntity):
    """Processed road segment for routing"""
    segment_id: str = ""
    osm_way_id: int = 0  # FK to OSMWay
    from_node_id: int = 0  # FK to OSMNode
    to_node_id: int = 0  # FK to OSMNode
    
    # Road classification
    highway_class: Optional[HighwayClass] = None
    name: Optional[str] = None
    ref: Optional[str] = None  # Route number (e.g., "NH1")
    
    # Physical attributes
    length_m: float = 0.0
    lanes: Optional[int] = None
    width_m: Optional[float] = None
    surface: Optional[SurfaceType] = None
    
    # Traffic rules
    oneway: bool = False
    maxspeed_kmh: Optional[int] = None
    access: Optional[str] = "yes"
    
    # Structure
    bridge: bool = False
    tunnel: bool = False
    
    # For routing
    free_flow_speed_kmh: Optional[float] = None
    capacity_veh_per_hour: Optional[int] = None
    forward_cost: Optional[float] = None  # Travel time in forward direction
    backward_cost: Optional[float] = None  # Travel time in backward direction
    
    # FHWA functional classification
    functional_class: Optional[int] = None  # 1-7

    def __post_init__(self):
        self.entity_type = "RoadSegment"


@dataclass
class Building(SpatialEntity):
    """OSM building"""
    building_id: str = ""
    osm_way_id: int = 0  # FK to OSMWay
    
    # Building type
    building_type: Optional[BuildingType] = None
    building_levels: Optional[int] = None
    height_m: Optional[float] = None
    
    # Area calculations
    footprint_area_m2: Optional[float] = None
    total_floor_area_m2: Optional[float] = None  # footprint × levels
    
    # Address
    addr_housenumber: Optional[str] = None
    addr_street: Optional[str] = None
    addr_city: Optional[str] = None
    addr_postcode: Optional[str] = None
    
    # Attribution
    name: Optional[str] = None
    
    # Link to zone
    taz_id: Optional[str] = None
    zone_id: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "Building"
        # Calculate total floor area if not provided
        if self.total_floor_area_m2 is None and self.footprint_area_m2 and self.building_levels:
            self.total_floor_area_m2 = self.footprint_area_m2 * self.building_levels


@dataclass
class PointOfInterest(SpatialEntity):
    """OSM Point of Interest"""
    poi_id: str = ""
    osm_node_id: Optional[int] = None
    osm_way_id: Optional[int] = None  # If POI is area
    
    # POI classification
    amenity: Optional[AmenityType] = None
    shop: Optional[ShopType] = None
    leisure: Optional[str] = None
    tourism: Optional[str] = None
    
    # Details
    name: Optional[str] = None
    brand: Optional[str] = None
    operator: Optional[str] = None
    
    # Operating info
    opening_hours: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    
    # Food-specific
    cuisine: Optional[str] = None
    
    # Link to zone
    taz_id: Optional[str] = None
    zone_id: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "PointOfInterest"


@dataclass
class TransitStop(SpatialEntity):
    """OSM transit stop"""
    stop_id: str = ""
    osm_node_id: int = 0  # FK to OSMNode
    
    # Stop type
    public_transport: str = ""  # "stop_position", "platform"
    highway: Optional[str] = "bus_stop"
    railway: Optional[str] = None  # "stop", "halt", "station"
    
    # Details
    stop_name: Optional[str] = None
    ref: Optional[str] = None  # Stop code
    operator: Optional[str] = None
    network: Optional[str] = None
    
    # Amenities
    shelter: Optional[bool] = None
    bench: Optional[bool] = None
    bin: Optional[bool] = None
    
    # Link to GTFS
    gtfs_stop_id: Optional[str] = None
    
    # Link to zone
    taz_id: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "TransitStop"


@dataclass
class Barrier(SpatialEntity):
    """Major barrier (river, rail corridor, etc.)"""
    barrier_id: str = ""
    barrier_type: str = ""  # "waterway", "railway", "motorway"
    osm_way_id: Optional[int] = None
    
    # Attributes
    name: Optional[str] = None
    width_m: Optional[float] = None
    
    # Buffered geometry for zone splitting
    buffer_distance_m: float = 50.0
    buffered_geometry: Optional[Any] = None

    def __post_init__(self):
        self.entity_type = "Barrier"


@dataclass
class Intersection(SpatialEntity):
    """Road intersection/junction"""
    intersection_id: str = ""
    osm_node_id: int = 0  # FK to OSMNode
    
    # Connected roads
    connected_segments: List[str] = field(default_factory=list)
    num_connections: int = 0
    
    # Traffic control
    has_traffic_signal: bool = False
    has_stop_sign: bool = False
    has_roundabout: bool = False
    
    # Classification
    is_motorway_junction: bool = False
    junction_type: Optional[str] = None

    def __post_init__(self):
        self.entity_type = "Intersection"


if __name__ == "__main__":
    # Example usage
    road = RoadSegment(
        segment_id="SEG_001",
        entity_id="SEG_001",
        osm_way_id=123456789,
        from_node_id=1001,
        to_node_id=1002,
        geometry=None,
        highway_class=HighwayClass.PRIMARY,
        name="Outer Ring Road",
        length_m=500.0,
        lanes=4,
        maxspeed_kmh=80,
        oneway=False,
        free_flow_speed_kmh=60.0,
        capacity_veh_per_hour=1800
    )
    
    building = Building(
        building_id="BLDG_001",
        entity_id="BLDG_001",
        osm_way_id=987654321,
        geometry=None,
        building_type=BuildingType.RESIDENTIAL,
        building_levels=5,
        footprint_area_m2=400.0,
        taz_id="TAZ_001"
    )
    
    poi = PointOfInterest(
        poi_id="POI_001",
        entity_id="POI_001",
        osm_node_id=555666777,
        geometry=None,
        centroid=(77.2090, 28.6139),
        amenity=AmenityType.RESTAURANT,
        name="Karim's Restaurant",
        cuisine="indian",
        opening_hours="Mo-Su 11:00-23:00",
        taz_id="TAZ_001"
    )
    
    print(f"Road: {road.name}, {road.length_m}m, {road.lanes} lanes")
    print(f"Building: {building.building_levels} floors, {building.total_floor_area_m2}m² total area")
    print(f"POI: {poi.name}, type: {poi.amenity.value}")
