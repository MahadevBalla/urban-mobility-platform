# Urban Mobility Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791.svg)](https://www.postgresql.org/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.3-5CAE58.svg)](https://postgis.net/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An open-source platform for automated Traffic Analysis Zone (TAZ) generation and travel demand modeling.**

Generate production-ready traffic zones for any city worldwide using OpenStreetMap data, with an interactive web dashboard, PostgreSQL caching, and a comprehensive transport data ontology.

![Dashboard Preview](docs/images/dashboard_preview.png)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Zone Generation Pipeline](#zone-generation-pipeline)
- [Services & Ports](#services--ports)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Transport Data Ontology](#transport-data-ontology)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

| Feature | Description |
|---------|-------------|
| **Automated Zone Generation** | 8-step pipeline creates TAZ-like zones from OSM data |
| **Global Coverage** | Works for any city with OpenStreetMap data |
| **Smart Barrier Detection** | Zones respect highways, railways, and rivers |
| **Proxy Demand Estimation** | Population/employment from building footprints & POIs |
| **Database Caching** | Sub-2-second loading via PostgreSQL + PostGIS |
| **Interactive Dashboard** | Streamlit web UI with maps, statistics, exports |
| **Transport Ontology** | Standardized schemas for 14 data sources |
| **Docker Ready** | One-command deployment |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/urban-mobility-platform.git
cd urban-mobility-platform

# Copy environment file
cp .env.example .env

# Start all services (PostgreSQL + Streamlit App)
docker-compose up -d

# Access the dashboard
# Open: http://localhost:8501
```

### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/urban-mobility-platform.git
cd urban-mobility-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Open: http://localhost:8501
```

> **Note:** For database features, you'll need PostgreSQL running. See [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md) for details.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         URBAN MOBILITY PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   STREAMLIT     │    │   POSTGRESQL    │    │   TRANSPORT     │     │
│  │   Dashboard     │◄──►│   + PostGIS     │    │   Ontology      │     │
│  │   (Port 8501)   │    │   (Port 5432)   │    │  (14 Sources)   │     │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────┘     │
│           │                      │                                       │
│           │                      │  ┌─────────────────┐                 │
│           │                      └──│   pgAdmin       │                 │
│           │                         │   (Port 5051)   │                 │
│           │                         └─────────────────┘                 │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ZONE GENERATION ENGINE                        │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  1. OSM Extract    2. H3 Grid    3. Barrier Split    4. Features │   │
│  │        ↓               ↓              ↓                  ↓       │   │
│  │  5. Region Merge   6. Centroids  7. Skim Matrices   8. Export    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Zone Generation Pipeline

The platform uses an **8-step automated pipeline**:

| Step | Module | Description |
|:----:|--------|-------------|
| 1 | `osm_network.py` | Extract roads, rail, water, buildings, POIs from OpenStreetMap |
| 2 | `hex_grid.py` | Generate H3 hexagonal grid (auto-resolution 6-9 based on area) |
| 3 | `barrier_detector.py` | Identify major corridors and split grid along barriers |
| 4 | `feature_engineer.py` | Compute proxy population, employment, land-use classification |
| 5 | `region_merger.py` | Merge cells into zones using region-growing algorithm |
| 6 | `centroid_connector.py` | Generate activity-weighted zone centroids |
| 7 | `skim_computer.py` | Compute distance, time, and cost matrices |
| 8 | `zone_generator.py` | Orchestrate pipeline, save to files and database |

---

## Services & Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Streamlit Dashboard** | 8501 | http://localhost:8501 | Main web interface |
| **PostgreSQL + PostGIS** | 5432 | `localhost:5432` | Spatial database |
| **pgAdmin** | 5051 | http://localhost:5051 | Database management UI |

### Default Credentials

| Service | Username/Email | Password |
|---------|---------------|----------|
| PostgreSQL | `urban_admin` | `urban_transit_2024` |
| pgAdmin | `admin@example.com` | `admin` |

---

## Output Files

After zone generation, the following files are created:

| File | Format | Description |
|------|--------|-------------|
| `zones.geojson` | GeoJSON | Zone polygons with all attributes |
| `centroids.geojson` | GeoJSON | Zone centroid points |
| `zones_summary.csv` | CSV | Zone attributes in tabular format |
| `skim_distance_km.csv` | CSV | Zone-to-zone distance matrix (km) |
| `skim_time_drive_min.csv` | CSV | Driving time matrix (minutes) |
| `skim_time_transit_min.csv` | CSV | Transit time matrix (minutes) |
| `skim_time_walk_min.csv` | CSV | Walking time matrix (minutes) |
| `skim_cost_drive.csv` | CSV | Driving cost matrix |

---

## Project Structure

```
urban-mobility-platform/
│
├── app.py                          # Streamlit web dashboard
├── docker-compose.yml              # Docker services orchestration
├── Dockerfile                      # Application container
├── requirements.txt                # Python dependencies
├── database_schema.sql             # PostgreSQL + PostGIS schema
├── .env.example                    # Environment variables template
│
├── src/
│   ├── zone_generation/            # Core zone generation engine
│   │   ├── zone_generator.py       # Main pipeline orchestrator
│   │   ├── osm_network.py          # OpenStreetMap data extraction
│   │   ├── hex_grid.py             # H3 hexagonal grid generation
│   │   ├── barrier_detector.py     # Barrier detection & grid splitting
│   │   ├── feature_engineer.py     # Feature computation (pop, emp, land-use)
│   │   ├── region_merger.py        # Region-growing zone merging
│   │   ├── centroid_connector.py   # Centroid & connector generation
│   │   └── skim_computer.py        # Skim matrix computation
│   │
│   └── database/                   # Database layer
│       ├── postgres_connector.py   # PostgreSQL connection management
│       └── zone_manager.py         # Zone CRUD operations & caching
│
├── ONTOLOGY/                       # Transport data ontology (v1.0)
│   ├── ontology_base.py            # Abstract classes & 30+ enumerations
│   ├── ontology_census.py          # Census data (Person, Household)
│   ├── ontology_hts.py             # Household travel surveys (Trip, Tour)
│   ├── ontology_mobile.py          # Mobile phone OD data (CDR, StayPoint)
│   ├── ontology_probe.py           # GPS probe data (Trace, Speed)
│   ├── ontology_gtfs.py            # GTFS transit (Agency, Route, Stop)
│   ├── ontology_osm.py             # OpenStreetMap (Road, Building, POI)
│   ├── transport_ontology.ttl      # RDF/TTL semantic export
│   └── README.md                   # Ontology documentation
│
├── data/                           # Data storage
│   ├── raw/                        # Raw input data
│   └── processed/                  # Processed data
│
├── research/                       # Research materials
│   ├── matsim/                     # MATSim simulation examples
│   └── populationsim/              # Population synthesis examples
│
├── docs/                           # Additional documentation
│
├── SETUP_AND_USAGE.md              # Detailed setup & usage guide
├── IMPLEMENTATION_PLAN_V2.md       # Future development roadmap
└── LICENSE                         # MIT License
```

---

## Transport Data Ontology

The platform includes a comprehensive **Transport Data Ontology** supporting 14 urban mobility data sources:

### Implemented Modules (7/14)

| Module | Data Source | Key Entities |
|--------|-------------|--------------|
| `ontology_base` | Core Framework | 30+ enumerations, abstract classes |
| `ontology_census` | Census Data | Person, Household, SyntheticPopulation |
| `ontology_hts` | Travel Surveys | Trip, Tour, Activity, TripChain |
| `ontology_mobile` | Mobile Phone OD | CellTower, CDREvent, ODMatrix |
| `ontology_probe` | GPS Probe | GPSTrace, SpeedObservation, TrafficFlow |
| `ontology_gtfs` | GTFS Transit | Agency, Route, Stop, Trip, GTFS-RT |
| `ontology_osm` | OpenStreetMap | RoadSegment, Building, POI |

### Planned Modules (7/14)

- Ticketing/AFC Data
- Traffic Message Channel (TMC)
- Land Use/Parcel Data
- Parking Facility Data
- Traffic Count Data
- Air Quality Data
- Accident/Crash Data

See [ONTOLOGY/README.md](ONTOLOGY/README.md) for complete documentation.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12+ |
| **Web Framework** | Streamlit |
| **Database** | PostgreSQL 15, PostGIS 3.3 |
| **Containerization** | Docker, Docker Compose |
| **Geospatial** | GeoPandas, Shapely, OSMnx, H3, Folium |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, HDBSCAN |
| **Visualization** | Plotly, Matplotlib |
| **Graph Analysis** | NetworkX |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[SETUP_AND_USAGE.md](SETUP_AND_USAGE.md)** | Complete setup, configuration, and usage guide |
| **[ONTOLOGY/README.md](ONTOLOGY/README.md)** | Transport data ontology documentation |
| **[IMPLEMENTATION_PLAN_V2.md](IMPLEMENTATION_PLAN_V2.md)** | Future development roadmap (4-step model, GenAI) |

---

## Use Cases

- **Urban Planning** - Generate TAZs for transport master plans
- **Travel Demand Modeling** - Input zones for 4-step models
- **Accessibility Analysis** - Zone-based accessibility metrics
- **Transit Planning** - Service coverage and connectivity analysis
- **Academic Research** - Transport modeling studies and publications

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/urban-mobility-platform.git
cd urban-mobility-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development database
docker-compose up -d postgres

# Run tests
python test_zone_generation.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **IIT Bombay** - Research support and guidance
- **OpenStreetMap Contributors** - Geographic data foundation
- **Uber H3** - Hexagonal grid system
- **MATSim Community** - Multi-agent transport simulation reference

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/urban-mobility-platform/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/urban-mobility-platform/discussions)

---

<p align="center">
  <b>Built with passion for open-source urban mobility research</b>
</p>
