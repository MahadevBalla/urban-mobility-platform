# Transport Data Ontology

**Version:** 1.0.0  
**Created:** November 2025  
**Project:** 4-Step Travel Demand Modeling Platform  
**IIT Bombay Research Internship**

---

## Overview

This folder contains a comprehensive transport data ontology designed for multi-source data integration in urban mobility analysis and 4-step travel demand modeling. The ontology provides semantic consistency across 14 diverse data sources.

## Standards Alignment

This ontology aligns with international standards:

- **Transmodel** - CEN European Reference Data Model for Public Transport
- **GTFS/GTFS-RT** - General Transit Feed Specification (Static & Realtime)
- **NeTEx** - Network Timetable Exchange
- **ActivitySim/PopulationSim** - Activity-based travel demand model structures
- **MATSim** - Multi-Agent Transport Simulation data model
- **LBCS** - Land-Based Classification Standards
- **APDS** - Alliance for Parking Data Standards
- **ISO 14819** - Traffic Message Channel (TMC/ALERT-C)
- **Census PUMS** - Public Use Microdata Sample
- **OSM** - OpenStreetMap data model

## Data Sources Covered

1. **Census Data** - Demographics, socioeconomics
2. **Household Travel Surveys (HTS)** - Trip diaries, tours, activities
3. **Mobile Phone OD Data** - CDR-based origin-destination matrices
4. **GPS Probe Data** - TomTom/Here vehicle traces
5. **OpenStreetMap** - Road network, buildings, POIs
6. **GTFS** - Public transit schedules and real-time data
7. **Ticketing Data (AFC)** - Automated fare collection
8. **TMC Data** - Traffic incidents and events
9. **Land Use Data** - Parcels, zoning, classifications
10. **Parking Data** - Facilities, occupancy, pricing
11. **Traffic Counts** - ATR, volume, speed data
12. **Passenger Counts** - Transit boarding/alighting
13. **Air Quality Data** - Pollutant measurements
14. **Accident Data** - Crash records, severity

## Folder Structure

```
ONTOLOGY/
├── README.md                          # This file
├── ONTOLOGY_SPECIFICATION.md          # Complete ontology documentation
├── ENTITY_RELATIONSHIPS.md            # ER diagrams and relationships
├── DATA_LINEAGE.md                    # Data flow and transformations
│
├── ontology_base.py                   # Base classes and enumerations
├── ontology_census.py                 # Census data entities
├── ontology_hts.py                    # Household travel survey entities
├── ontology_mobile.py                 # Mobile phone OD entities
├── ontology_probe.py                  # GPS probe data entities
├── ontology_osm.py                    # OpenStreetMap entities
├── ontology_gtfs.py                   # GTFS entities
├── ontology_ticketing.py              # AFC ticketing entities
├── ontology_tmc.py                    # Traffic message channel entities
├── ontology_landuse.py                # Land use entities
├── ontology_parking.py                # Parking data entities
├── ontology_traffic.py                # Traffic count entities
├── ontology_airquality.py             # Air quality entities
├── ontology_accidents.py              # Accident/crash entities
│
└── ontology_integrator.py             # Cross-source integration logic
```

## Core Concepts

### Entity Hierarchy

```
TransportEntity (Abstract)
├── SpatialEntity (has geometry)
├── TemporalEntity (has time validity)
├── AgentEntity (people, vehicles, organizations)
├── ActivityEntity (trips, tours, activities)
└── InfrastructureEntity (facilities, sensors)
```

### Key Design Principles

1. **Semantic Consistency** - Common vocabulary across data sources
2. **Extensibility** - Easy to add new data sources
3. **Traceability** - Data lineage and provenance tracking
4. **Interoperability** - Standards-aligned for data exchange
5. **Quality Awareness** - Quality scores and confidence intervals

## Usage

### Python Implementation

```python
from ONTOLOGY.ontology_base import TransportEntity, SpatialEntity
from ONTOLOGY.ontology_census import Person, Household
from ONTOLOGY.ontology_hts import Trip, Tour
from ONTOLOGY.ontology_integrator import OntologyIntegrator

# Create integrator
integrator = OntologyIntegrator()

# Load data from multiple sources
integrator.load_census_data('census.csv')
integrator.load_hts_data('survey.csv')
integrator.load_gtfs_data('gtfs/')

# Semantic integration
integrated_trips = integrator.integrate_trips(
    sources=['hts', 'mobile', 'afc'],
    resolution='zone'
)
```

### Data Fusion Example

```python
# Resolve conflicts between data sources
from ONTOLOGY.ontology_integrator import DataFusion

fusion = DataFusion()

# Mobile OD says 1000 trips from Zone A to B
# HTS says 850 trips from Zone A to B
# AFC says 920 trips from Zone A to B

fused_value = fusion.weighted_average(
    values=[1000, 850, 920],
    weights=[0.4, 0.4, 0.2],  # Based on data quality
    confidence_intervals=[(950, 1050), (800, 900), (900, 940)]
)
```

## For Researchers

### Adding a New Data Source

1. Create `ontology_newsource.py`
2. Define entities inheriting from base classes
3. Specify relationships to existing entities
4. Implement transformation to common format
5. Update `ontology_integrator.py`

### Validation

The ontology includes validators for:
- Data type consistency
- Referential integrity
- Temporal validity
- Spatial containment
- Enumeration constraints

## References

- **Transmodel**: https://transmodel-cen.eu/
- **GTFS Specification**: https://gtfs.org/
- **ActivitySim Documentation**: https://activitysim.github.io/
- **MATSim**: https://matsim.org/
- **LBCS**: https://www.planning.org/lbcs/
