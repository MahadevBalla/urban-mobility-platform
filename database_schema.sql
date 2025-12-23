-- Urban Transit Tool - Database Schema
-- PostgreSQL + PostGIS schema for zone generation caching

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Table 1: Cities
-- Stores metadata about processed cities
CREATE TABLE IF NOT EXISTS cities (
    city_id SERIAL PRIMARY KEY,
    place_name VARCHAR(255) UNIQUE NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,  -- e.g., "bandra_mumbai_india"
    boundary_geom GEOMETRY(POLYGON, 4326),  -- City boundary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for cities table
CREATE INDEX IF NOT EXISTS idx_cities_place_name ON cities(place_name);
CREATE INDEX IF NOT EXISTS idx_cities_normalized ON cities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_cities_boundary ON cities USING GIST(boundary_geom);

-- Table 2: Zone Generations
-- Stores metadata about each zone generation run
CREATE TABLE IF NOT EXISTS zone_generations (
    generation_id SERIAL PRIMARY KEY,
    city_id INTEGER REFERENCES cities(city_id) ON DELETE CASCADE,
    target_population INTEGER NOT NULL,
    buffer_distance FLOAT NOT NULL,
    hex_resolution INTEGER,
    num_zones INTEGER NOT NULL,
    total_area_km2 FLOAT NOT NULL,
    total_proxy_population INTEGER DEFAULT 0,
    total_proxy_employment INTEGER DEFAULT 0,
    processing_time_seconds FLOAT,
    osm_extraction_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_current BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_generation UNIQUE(city_id, target_population, buffer_distance, hex_resolution)
);

-- Indexes for zone_generations table
CREATE INDEX IF NOT EXISTS idx_generation_city ON zone_generations(city_id);
CREATE INDEX IF NOT EXISTS idx_generation_current ON zone_generations(city_id, is_current);
CREATE INDEX IF NOT EXISTS idx_generation_created ON zone_generations(created_at DESC);

-- Table 3: Zones
-- Stores individual zone data with geometries and attributes
CREATE TABLE IF NOT EXISTS zones (
    zone_pk SERIAL PRIMARY KEY,
    generation_id INTEGER REFERENCES zone_generations(generation_id) ON DELETE CASCADE,
    zone_id VARCHAR(50) NOT NULL,
    zone_geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    centroid_geometry GEOMETRY(POINT, 4326) NOT NULL,
    area_km2 FLOAT NOT NULL,
    proxy_population INTEGER DEFAULT 0,
    proxy_employment INTEGER DEFAULT 0,
    dominant_landuse VARCHAR(50),
    is_cbd BOOLEAN DEFAULT FALSE,
    is_special_generator BOOLEAN DEFAULT FALSE,
    CONSTRAINT unique_zone UNIQUE(generation_id, zone_id)
);

-- Indexes for zones table
CREATE INDEX IF NOT EXISTS idx_zones_generation ON zones(generation_id);
CREATE INDEX IF NOT EXISTS idx_zones_zone_id ON zones(zone_id);
CREATE INDEX IF NOT EXISTS idx_zones_geometry ON zones USING GIST(zone_geometry);
CREATE INDEX IF NOT EXISTS idx_zones_centroid ON zones USING GIST(centroid_geometry);
CREATE INDEX IF NOT EXISTS idx_zones_landuse ON zones(dominant_landuse);
CREATE INDEX IF NOT EXISTS idx_zones_cbd ON zones(is_cbd);

-- Table 4: Skim Matrices
-- Stores zone-to-zone skim matrices (distance, time, cost)
CREATE TABLE IF NOT EXISTS skim_matrices (
    skim_id SERIAL PRIMARY KEY,
    generation_id INTEGER REFERENCES zone_generations(generation_id) ON DELETE CASCADE,
    origin_zone_id VARCHAR(50) NOT NULL,
    destination_zone_id VARCHAR(50) NOT NULL,
    distance_km FLOAT,
    time_drive_min FLOAT,
    time_transit_min FLOAT,
    time_walk_min FLOAT,
    cost_drive FLOAT,
    CONSTRAINT unique_skim_pair UNIQUE(generation_id, origin_zone_id, destination_zone_id)
);

-- Indexes for skim_matrices table
CREATE INDEX IF NOT EXISTS idx_skim_generation ON skim_matrices(generation_id);
CREATE INDEX IF NOT EXISTS idx_skim_origin ON skim_matrices(origin_zone_id);
CREATE INDEX IF NOT EXISTS idx_skim_destination ON skim_matrices(destination_zone_id);
CREATE INDEX IF NOT EXISTS idx_skim_od_pair ON skim_matrices(generation_id, origin_zone_id, destination_zone_id);

-- Table 5: Connectors (Optional)
-- Stores network connectors if generated
CREATE TABLE IF NOT EXISTS connectors (
    connector_id SERIAL PRIMARY KEY,
    zone_pk INTEGER REFERENCES zones(zone_pk) ON DELETE CASCADE,
    connector_geometry GEOMETRY(LINESTRING, 4326),
    connector_type VARCHAR(50),
    length_m FLOAT
);

-- Indexes for connectors table
CREATE INDEX IF NOT EXISTS idx_connectors_zone ON connectors(zone_pk);
CREATE INDEX IF NOT EXISTS idx_connectors_geometry ON connectors USING GIST(connector_geometry);
CREATE INDEX IF NOT EXISTS idx_connectors_type ON connectors(connector_type);

-- Helper function: Get latest generation for a city
CREATE OR REPLACE FUNCTION get_latest_generation(p_place_name VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    v_generation_id INTEGER;
BEGIN
    SELECT g.generation_id INTO v_generation_id
    FROM zone_generations g
    JOIN cities c ON g.city_id = c.city_id
    WHERE c.place_name = p_place_name
      AND g.is_current = TRUE
    ORDER BY g.created_at DESC
    LIMIT 1;

    RETURN v_generation_id;
END;
$$ LANGUAGE plpgsql;

-- Helper function: Mark generation as current (and unmark others)
CREATE OR REPLACE FUNCTION set_current_generation(p_generation_id INTEGER)
RETURNS VOID AS $$
DECLARE
    v_city_id INTEGER;
BEGIN
    -- Get city_id for this generation
    SELECT city_id INTO v_city_id
    FROM zone_generations
    WHERE generation_id = p_generation_id;

    -- Unmark all other generations for this city
    UPDATE zone_generations
    SET is_current = FALSE
    WHERE city_id = v_city_id;

    -- Mark this generation as current
    UPDATE zone_generations
    SET is_current = TRUE
    WHERE generation_id = p_generation_id;
END;
$$ LANGUAGE plpgsql;

-- Create view for easy zone querying with city info
CREATE OR REPLACE VIEW zones_with_city AS
SELECT
    z.zone_pk,
    z.zone_id,
    z.zone_geometry,
    z.centroid_geometry,
    z.area_km2,
    z.proxy_population,
    z.proxy_employment,
    z.dominant_landuse,
    z.is_cbd,
    z.is_special_generator,
    g.generation_id,
    g.target_population,
    g.buffer_distance,
    g.created_at as generation_date,
    c.place_name,
    c.normalized_name
FROM zones z
JOIN zone_generations g ON z.generation_id = g.generation_id
JOIN cities c ON g.city_id = c.city_id;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO urban_admin;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO urban_admin;

-- Initial setup complete message
DO $$
BEGIN
    RAISE NOTICE 'Urban Transit Tool database schema created successfully!';
    RAISE NOTICE 'Tables: cities, zone_generations, zones, skim_matrices, connectors';
    RAISE NOTICE 'View: zones_with_city';
    RAISE NOTICE 'Functions: get_latest_generation, set_current_generation';
END $$;
