# Setup and Usage Guide

**Complete guide for setting up, configuring, and using the Urban Mobility Platform.**

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation Methods](#2-installation-methods)
   - [Docker Installation (Recommended)](#option-a-docker-installation-recommended)
   - [Local Installation](#option-b-local-installation)
   - [Hybrid Installation](#option-c-hybrid-installation)
3. [Configuration](#3-configuration)
4. [Starting the Services](#4-starting-the-services)
5. [Using the Dashboard](#5-using-the-dashboard)
6. [Database Management with pgAdmin](#6-database-management-with-pgadmin)
7. [Programmatic Usage](#7-programmatic-usage)
8. [Output Files Reference](#8-output-files-reference)
9. [Environment Variables](#9-environment-variables)
10. [Docker Commands Reference](#10-docker-commands-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Advanced Configuration](#12-advanced-configuration)
13. [Performance Optimization](#13-performance-optimization)
14. [Backup and Restore](#14-backup-and-restore)

---

## 1. Prerequisites

### Required Software

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| **Python** | 3.12+ | Core runtime | [python.org](https://www.python.org/downloads/) |
| **Docker Desktop** | 24.0+ | Container runtime | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Git** | 2.40+ | Version control | [git-scm.com](https://git-scm.com/downloads) |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free | 50 GB free |
| **CPU** | 4 cores | 8 cores |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Latest stable |

### Verify Prerequisites

```bash
# Check Python version
python --version  # Should show 3.12.x or higher

# Check Docker version
docker --version  # Should show 24.x or higher

# Check Docker Compose version
docker-compose --version  # Should show 2.x or higher

# Check Git version
git --version  # Should show 2.40.x or higher
```

---

## 2. Installation Methods

### Option A: Docker Installation (Recommended)

This is the easiest method - everything runs in containers.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/urban-mobility-platform.git
cd urban-mobility-platform
```

#### Step 2: Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# (Optional) Edit the file to customize passwords
# Default values work out of the box
```

#### Step 3: Start All Services

```bash
# Start PostgreSQL + Streamlit App
docker-compose up -d

# Wait for services to initialize (about 30 seconds)
# Check status
docker-compose ps
```

#### Step 4: Verify Services

```bash
# Check all containers are running
docker ps

# Expected output:
# CONTAINER ID   IMAGE                    STATUS          PORTS
# xxxxxxxxxxxx   urban-mobility-platform  Up 30 seconds   0.0.0.0:8501->8501/tcp
# xxxxxxxxxxxx   postgis/postgis:15-3.3   Up 30 seconds   0.0.0.0:5432->5432/tcp
```

#### Step 5: Access the Dashboard

Open your browser and navigate to: **http://localhost:8501**

---

### Option B: Local Installation

Run Python locally while using Docker only for the database.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/urban-mobility-platform.git
cd urban-mobility-platform
```

#### Step 2: Create Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Start Database (Docker)

```bash
# Start only PostgreSQL
docker-compose up -d postgres

# Wait for database to be ready
docker-compose logs -f postgres
# Press Ctrl+C when you see "database system is ready to accept connections"
```

#### Step 5: Set Environment Variables

**Windows (Command Prompt):**
```cmd
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=urban_transit_db
set DB_USER=urban_admin
set DB_PASSWORD=urban_transit_2024
```

**Windows (PowerShell):**
```powershell
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="urban_transit_db"
$env:DB_USER="urban_admin"
$env:DB_PASSWORD="urban_transit_2024"
```

**Linux/macOS:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=urban_transit_db
export DB_USER=urban_admin
export DB_PASSWORD=urban_transit_2024
```

#### Step 6: Run the Dashboard

```bash
streamlit run app.py
```

Open your browser: **http://localhost:8501**

---

### Option C: Hybrid Installation

Run everything locally without Docker (requires manual PostgreSQL setup).

#### Step 1: Install PostgreSQL + PostGIS

**Windows:**
1. Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. During installation, select "PostGIS" from Stack Builder

**macOS:**
```bash
brew install postgresql@15 postgis
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql-15 postgresql-15-postgis-3
sudo systemctl start postgresql
```

#### Step 2: Create Database and User

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Run these SQL commands:
CREATE USER urban_admin WITH PASSWORD 'urban_transit_2024';
CREATE DATABASE urban_transit_db OWNER urban_admin;
\c urban_transit_db
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
\q
```

#### Step 3: Initialize Schema

```bash
# Apply the database schema
psql -U urban_admin -d urban_transit_db -f database_schema.sql
```

#### Step 4: Follow Local Installation Steps

Continue from [Option B, Step 2](#step-2-create-virtual-environment).

---

## 3. Configuration

### Environment Variables File (.env)

Create a `.env` file in the project root with the following contents:

```bash
# ===========================================
# Urban Mobility Platform - Configuration
# ===========================================

# Database Configuration
DB_HOST=postgres              # Use 'localhost' for local installation
DB_PORT=5432
DB_NAME=urban_transit_db
DB_USER=urban_admin
DB_PASSWORD=urban_transit_2024

# pgAdmin Configuration
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=admin

# Application Settings
ENVIRONMENT=production        # Options: development, production
PYTHONUNBUFFERED=1
```

### Zone Generation Parameters

Configure zone generation in the dashboard or programmatically:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `target_population` | 5000 | 1000-10000 | Target proxy population per zone |
| `buffer_distance` | 50 | 20-100 | Barrier buffer distance (meters) |
| `hex_resolution` | Auto | 6-9 | H3 hexagonal grid resolution |

---

## 4. Starting the Services

### Start All Services (Docker)

```bash
# Start PostgreSQL and Streamlit App
docker-compose up -d

# Start with pgAdmin (database management UI)
docker-compose --profile tools up -d
```

### Start Individual Services

```bash
# Start only database
docker-compose up -d postgres

# Start only the app (requires database)
docker-compose up -d app

# Start pgAdmin
docker-compose --profile tools up -d pgadmin
```

### View Service Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f postgres
docker-compose logs -f pgadmin
```

### Stop Services

```bash
# Stop all services (keeps data)
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v

# Stop specific service
docker-compose stop app
```

---

## 5. Using the Dashboard

### Accessing the Dashboard

Open your browser and navigate to: **http://localhost:8501**

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                        URBAN MOBILITY PLATFORM                       │
├─────────────────┬───────────────────────────────────────────────────┤
│                 │                                                    │
│   SIDEBAR       │              MAIN CONTENT AREA                     │
│                 │                                                    │
│  ┌───────────┐  │  ┌─────────────────────────────────────────────┐  │
│  │ City      │  │  │                                             │  │
│  │ Selection │  │  │            MAP VIEW / STATISTICS            │  │
│  └───────────┘  │  │                                             │  │
│                 │  │                                             │  │
│  ┌───────────┐  │  └─────────────────────────────────────────────┘  │
│  │ Parameters│  │                                                    │
│  │ - Target  │  │  ┌──────┬──────┬──────┬──────┐                    │
│  │   Pop     │  │  │ Map  │Stats │Details│Export│    TABS           │
│  │ - H3 Res  │  │  └──────┴──────┴──────┴──────┘                    │
│  └───────────┘  │                                                    │
│                 │                                                    │
│  ┌───────────┐  │                                                    │
│  │ Generate  │  │                                                    │
│  │ Button    │  │                                                    │
│  └───────────┘  │                                                    │
│                 │                                                    │
│  ┌───────────┐  │                                                    │
│  │ Database  │  │                                                    │
│  │ Status    │  │                                                    │
│  └───────────┘  │                                                    │
│                 │                                                    │
└─────────────────┴───────────────────────────────────────────────────┘
```

### Step-by-Step Usage

#### 1. Select a City

**Quick Select (Pre-configured Cities):**
- Bandra, Mumbai, India
- Manhattan, New York, USA
- Westminster, London, UK
- Shibuya, Tokyo, Japan
- And more...

**Custom Location:**
- Enter any city name (e.g., "Berlin, Germany")
- Uses OpenStreetMap Nominatim for geocoding

#### 2. Configure Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| **Target Population** | Proxy population per zone | 3000-5000 for urban, 5000-8000 for suburban |
| **H3 Resolution** | Grid cell size | Auto (recommended) or 7-8 for most cities |

#### 3. Check Database Status

The sidebar shows database status:
- **✅ Cached** - Zones already exist, instant loading
- **🔄 New** - Will generate fresh zones (~10-15 min)
- **⚠️ Unavailable** - Database not connected

#### 4. Generate or Load Zones

- **If Cached:** Click "Load from Database" (instant)
- **If New:** Click "Generate Zones" (10-15 minutes)

#### 5. Explore Results

**Map Tab:**
- Interactive map with zone polygons
- Color-coded by land use
- Click zones for details
- Zoom/pan controls

**Statistics Tab:**
- Land use distribution (pie chart)
- Zone size histogram
- Population/employment distribution

**Details Tab:**
- Full zone attribute table
- Sortable columns
- Search/filter functionality

**Export Tab:**
- Download GeoJSON files
- Download CSV files
- Download skim matrices

---

## 6. Database Management with pgAdmin

### Starting pgAdmin

```bash
# Start pgAdmin (runs on port 5051)
docker-compose --profile tools up -d pgadmin
```

### Accessing pgAdmin

1. Open browser: **http://localhost:5051**
2. Login credentials:
   - **Email:** `admin@example.com`
   - **Password:** `admin`

### Connecting to the Database

1. Right-click "Servers" → "Register" → "Server"
2. **General Tab:**
   - Name: `Urban Transit DB`
3. **Connection Tab:**
   - Host: `postgres` (or `localhost` if running locally)
   - Port: `5432`
   - Database: `urban_transit_db`
   - Username: `urban_admin`
   - Password: `urban_transit_2024`
4. Click "Save"

### Database Tables

Navigate to: **Servers → Urban Transit DB → Databases → urban_transit_db → Schemas → public → Tables**

| Table | Description |
|-------|-------------|
| `cities` | City metadata and boundaries |
| `zone_generations` | Generation run metadata |
| `zones` | Individual zone geometries and attributes |
| `skim_matrices` | Zone-to-zone travel matrices |
| `connectors` | Network connector geometries |

### Useful SQL Queries

**List all cities with zone counts:**
```sql
SELECT c.place_name, COUNT(z.zone_id) as num_zones,
       MAX(g.created_at) as last_generated
FROM cities c
JOIN zone_generations g ON c.city_id = g.city_id
JOIN zones z ON g.generation_id = z.generation_id
GROUP BY c.place_name
ORDER BY last_generated DESC;
```

**Get zones for a specific city:**
```sql
SELECT z.zone_id, z.area_km2, z.proxy_population,
       z.proxy_employment, z.dominant_landuse
FROM zones z
JOIN zone_generations g ON z.generation_id = g.generation_id
JOIN cities c ON g.city_id = c.city_id
WHERE c.place_name = 'Bandra, Mumbai, India'
ORDER BY z.zone_id;
```

**Export zones as GeoJSON:**
```sql
SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(zone_geometry)::json,
            'properties', json_build_object(
                'zone_id', zone_id,
                'area_km2', area_km2,
                'population', proxy_population,
                'employment', proxy_employment,
                'landuse', dominant_landuse
            )
        )
    )
)
FROM zones
WHERE generation_id = 1;
```

---

## 7. Programmatic Usage

### Basic Usage

```python
from src.zone_generation.zone_generator import AutomatedZoneGenerator

# Create generator
generator = AutomatedZoneGenerator(
    place_name="Mumbai, India",
    target_population=5000,
    buffer_distance=50,
    output_dir="./output_mumbai"
)

# Generate zones
results = generator.generate_zones()

# Access results
print(f"Generated {results['num_zones']} zones")
print(f"Total area: {results['total_area_km2']:.2f} km²")
print(f"Total population proxy: {results['total_proxy_population']}")
```

### Custom Boundary

```python
from shapely.geometry import box, Polygon

# Using bounding box
boundary = box(72.8, 19.0, 72.9, 19.1)  # lon_min, lat_min, lon_max, lat_max

# Using custom polygon
boundary = Polygon([
    (72.82, 19.05),
    (72.85, 19.08),
    (72.88, 19.06),
    (72.86, 19.03),
    (72.82, 19.05)
])

generator = AutomatedZoneGenerator(
    boundary_polygon=boundary,
    target_population=5000,
    output_dir="./output_custom"
)
```

### Access Generated Data

```python
import geopandas as gpd
import pandas as pd

# Load zones
zones = gpd.read_file("./output_mumbai/zones.geojson")

# Load centroids
centroids = gpd.read_file("./output_mumbai/centroids.geojson")

# Load skim matrix
skim_distance = pd.read_csv("./output_mumbai/skim_distance_km.csv", index_col=0)
skim_time = pd.read_csv("./output_mumbai/skim_time_drive_min.csv", index_col=0)

# Explore zones
print(zones.columns)
print(zones.describe())

# Get specific zone
zone = zones[zones['zone_id'] == 'TAZ_0001']
print(zone)
```

### Using Database Cache

```python
from src.database.zone_manager import ZoneManager

# Initialize zone manager
manager = ZoneManager()

# Check if zones exist
exists = manager.check_zones_exist(
    place_name="Mumbai, India",
    target_population=5000,
    buffer_distance=50
)

if exists:
    # Load from cache
    zones_gdf, centroids_gdf, metadata = manager.load_zone_generation(
        place_name="Mumbai, India"
    )
    print(f"Loaded {len(zones_gdf)} zones from cache")
else:
    # Generate new zones
    generator = AutomatedZoneGenerator(...)
    results = generator.generate_zones()

    # Save to cache
    manager.save_zone_generation(
        place_name="Mumbai, India",
        zones_gdf=results['zones'],
        centroids_gdf=results['centroids'],
        params={'target_population': 5000, 'buffer_distance': 50}
    )
```

---

## 8. Output Files Reference

### zones.geojson

GeoJSON file containing zone polygons with attributes.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[72.83, 19.05], [72.84, 19.05], ...]]
      },
      "properties": {
        "zone_id": "TAZ_0001",
        "area_km2": 1.234,
        "proxy_population": 4567,
        "proxy_employment": 1234,
        "dominant_landuse": "commercial",
        "is_cbd": true,
        "is_special_generator": false,
        "total_building_area_m2": 50000,
        "avg_building_levels": 3.5,
        "poi_commercial_count": 150,
        "poi_education_count": 5,
        "poi_healthcare_count": 3
      }
    }
  ]
}
```

### Zone Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `zone_id` | string | Unique identifier (TAZ_0001, TAZ_0002, ...) |
| `area_km2` | float | Zone area in square kilometers |
| `proxy_population` | integer | Estimated population (from building data) |
| `proxy_employment` | integer | Estimated employment (from POI data) |
| `dominant_landuse` | string | Primary land use (residential/commercial/industrial/mixed) |
| `is_cbd` | boolean | Whether zone is in Central Business District |
| `is_special_generator` | boolean | Whether zone is a special trip generator |
| `total_building_area_m2` | float | Total building footprint area |
| `avg_building_levels` | float | Average building height in levels |
| `poi_*_count` | integer | Point of interest counts by category |

### Skim Matrix Format

CSV files with zone-to-zone matrices:

```csv
,TAZ_0001,TAZ_0002,TAZ_0003,...
TAZ_0001,0.00,1.23,2.45,...
TAZ_0002,1.23,0.00,1.56,...
TAZ_0003,2.45,1.56,0.00,...
```

| Matrix | Unit | Description |
|--------|------|-------------|
| `skim_distance_km.csv` | kilometers | Euclidean/network distance |
| `skim_time_drive_min.csv` | minutes | Driving time (30 km/h avg) |
| `skim_time_transit_min.csv` | minutes | Transit time (includes wait) |
| `skim_time_walk_min.csv` | minutes | Walking time (5 km/h) |
| `skim_cost_drive.csv` | currency | Driving cost estimate |

---

## 9. Environment Variables

### Complete Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `postgres` | Database hostname |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `urban_transit_db` | Database name |
| `DB_USER` | `urban_admin` | Database username |
| `DB_PASSWORD` | `urban_transit_2024` | Database password |
| `PGADMIN_DEFAULT_EMAIL` | `admin@example.com` | pgAdmin login email |
| `PGADMIN_DEFAULT_PASSWORD` | `admin` | pgAdmin login password |
| `ENVIRONMENT` | `production` | Environment mode |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |

### Setting Environment Variables

**Using .env file (recommended):**
```bash
# Create .env file
cp .env.example .env
# Edit values as needed
```

**Command line (Windows):**
```cmd
set DB_PASSWORD=your_password
```

**Command line (Linux/macOS):**
```bash
export DB_PASSWORD=your_password
```

---

## 10. Docker Commands Reference

### Service Management

```bash
# Start all services
docker-compose up -d

# Start with pgAdmin
docker-compose --profile tools up -d

# Stop all services
docker-compose down

# Stop and remove volumes (reset database)
docker-compose down -v

# Restart specific service
docker-compose restart app

# View running containers
docker-compose ps
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs app
docker-compose logs postgres
docker-compose logs pgadmin

# View last 100 lines
docker-compose logs --tail=100 app
```

### Container Access

```bash
# Open shell in app container
docker exec -it urban_transit_app bash

# Open PostgreSQL CLI
docker exec -it urban_transit_db psql -U urban_admin -d urban_transit_db

# Run Python command in container
docker exec -it urban_transit_app python -c "print('Hello')"
```

### Building and Updating

```bash
# Rebuild containers
docker-compose build

# Rebuild without cache
docker-compose build --no-cache

# Pull latest images
docker-compose pull

# Update and restart
docker-compose pull && docker-compose up -d
```

### Resource Management

```bash
# View container resource usage
docker stats

# Clean up unused resources
docker system prune

# Remove unused volumes
docker volume prune

# Remove unused images
docker image prune
```

---

## 11. Troubleshooting

### Database Connection Issues

**Problem:** `password authentication failed for user "urban_admin"`

**Solution:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

**Problem:** `connection refused` or `could not connect to server`

**Solution:**
```bash
# Check if database is running
docker-compose ps

# Check database logs
docker-compose logs postgres

# Wait for database to be ready
docker-compose up -d postgres
sleep 30
docker-compose up -d app
```

### Port Conflicts

**Problem:** `port 5432 is already in use`

**Solution:**
```bash
# Find process using port
# Windows:
netstat -ano | findstr :5432
# Linux/macOS:
lsof -i :5432

# Kill the process or change port in docker-compose.yml
```

**Change port in docker-compose.yml:**
```yaml
postgres:
  ports:
    - "5433:5432"  # Use 5433 externally
```

### pgAdmin Issues

**Problem:** pgAdmin won't start or keeps restarting

**Solution:**
```bash
# Reset pgAdmin data
docker-compose --profile tools down
docker volume rm urban_transit_pgadmin_data
docker-compose --profile tools up -d pgadmin
```

**Problem:** Can't connect to database from pgAdmin

**Solution:**
- Use hostname `postgres` (not `localhost`) when inside Docker
- Use hostname `localhost` when running pgAdmin outside Docker

### Zone Generation Issues

**Problem:** Zone generation fails with memory error

**Solution:**
- Reduce area size (use smaller boundary)
- Increase H3 resolution (smaller cells = less memory)
- Increase Docker memory limit

**Problem:** OSM data extraction timeout

**Solution:**
```python
# Use smaller area or retry
import osmnx as ox
ox.settings.timeout = 300  # Increase timeout to 5 minutes
```

### Dashboard Issues

**Problem:** Dashboard shows "Database unavailable"

**Solution:**
```bash
# Verify environment variables are set
echo $DB_PASSWORD  # Linux/macOS
echo %DB_PASSWORD%  # Windows

# Set if missing
export DB_PASSWORD=urban_transit_2024
streamlit run app.py
```

**Problem:** Map not loading

**Solution:**
- Clear browser cache
- Check browser console for errors
- Ensure zones were generated successfully

---

## 12. Advanced Configuration

### Custom H3 Resolution

```python
# Resolution guide:
# 6 = ~36 km² cells (large cities, regional)
# 7 = ~5 km² cells (city-wide)
# 8 = ~0.7 km² cells (district-level)
# 9 = ~0.1 km² cells (neighborhood-level)

generator = AutomatedZoneGenerator(
    place_name="City",
    hex_resolution=8,  # Override auto-selection
    target_population=5000
)
```

### Custom Barrier Detection

```python
# Modify barrier types in barrier_detector.py
MAJOR_ROAD_TYPES = ['motorway', 'trunk', 'primary']
RAIL_TYPES = ['rail', 'subway', 'light_rail']
WATER_TYPES = ['river', 'canal', 'stream']

# Adjust buffer distance
generator = AutomatedZoneGenerator(
    place_name="City",
    buffer_distance=30  # Smaller buffer = fewer splits
)
```

### Custom Land Use Classification

```python
# In feature_engineer.py, modify POI signatures:
LANDUSE_SIGNATURES = {
    'residential': {'residential': 0.8, 'commercial': 0.1, 'education': 0.1},
    'commercial': {'commercial': 0.6, 'office': 0.3, 'retail': 0.1},
    'industrial': {'industrial': 0.7, 'warehouse': 0.2, 'office': 0.1},
    'mixed': {'residential': 0.3, 'commercial': 0.3, 'office': 0.4}
}
```

### Database Connection Pooling

```python
# In postgres_connector.py
from psycopg2 import pool

connection_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
```

---

## 13. Performance Optimization

### Zone Generation Speed

| Optimization | Impact | How |
|--------------|--------|-----|
| Use cached OSM data | 2-3x faster | OSMnx caches automatically |
| Higher H3 resolution | Slower but more detail | Use auto or 7-8 |
| Smaller boundary | Much faster | Use district boundaries |
| SSD storage | 2x faster I/O | Store data on SSD |

### Database Performance

```sql
-- Add indexes for common queries
CREATE INDEX idx_zones_city ON zones USING btree(generation_id);
CREATE INDEX idx_zones_geom ON zones USING gist(zone_geometry);

-- Analyze tables for query optimization
ANALYZE zones;
ANALYZE skim_matrices;
```

### Docker Resource Limits

```yaml
# In docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
  postgres:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## 14. Backup and Restore

### Database Backup

```bash
# Backup entire database
docker exec urban_transit_db pg_dump -U urban_admin urban_transit_db > backup.sql

# Backup specific tables
docker exec urban_transit_db pg_dump -U urban_admin -t zones -t cities urban_transit_db > zones_backup.sql

# Compressed backup
docker exec urban_transit_db pg_dump -U urban_admin urban_transit_db | gzip > backup.sql.gz
```

### Database Restore

```bash
# Restore from backup
cat backup.sql | docker exec -i urban_transit_db psql -U urban_admin -d urban_transit_db

# Restore compressed backup
gunzip -c backup.sql.gz | docker exec -i urban_transit_db psql -U urban_admin -d urban_transit_db
```

### Export Zones

```bash
# Export all zones as GeoJSON
docker exec urban_transit_app python -c "
from src.database.zone_manager import ZoneManager
import json

manager = ZoneManager()
cities = manager.list_available_cities()

for city in cities:
    zones, centroids, meta = manager.load_zone_generation(city)
    zones.to_file(f'export_{city.replace(\", \", \"_\")}.geojson', driver='GeoJSON')
    print(f'Exported {city}')
"
```

### Volume Backup

```bash
# Backup Docker volumes
docker run --rm -v urban_transit_postgres_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/postgres_data_backup.tar.gz /data

# Restore Docker volumes
docker run --rm -v urban_transit_postgres_data:/data -v $(pwd):/backup alpine \
    tar xzf /backup/postgres_data_backup.tar.gz -C /
```

---

## Quick Reference Card

### URLs

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| pgAdmin | http://localhost:5051 |
| PostgreSQL | localhost:5432 |

### Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| PostgreSQL | `urban_admin` | `urban_transit_2024` |
| pgAdmin | `admin@example.com` | `admin` |

### Essential Commands

```bash
# Start everything
docker-compose --profile tools up -d

# Stop everything
docker-compose down

# View logs
docker-compose logs -f

# Reset database
docker-compose down -v && docker-compose up -d

# Access database CLI
docker exec -it urban_transit_db psql -U urban_admin -d urban_transit_db
```

---

**Need help?** Open an issue on GitHub or check the [Troubleshooting](#11-troubleshooting) section.
