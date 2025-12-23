"""
Interactive Zone Generation Dashboard
Streamlit app for generating and visualizing TAZ zones
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time

from src.zone_generation.zone_generator import AutomatedZoneGenerator

# Page config
st.set_page_config(
    page_title="Urban Transit Project",
    page_icon="🗺️",
    layout="wide"
)

# Title
st.title("🗺️ Traffic Analysis Zone Dashboard")
st.markdown("Generate TAZ-like zones for any city in the world -Himangshu Shekhar")

# Sidebar for inputs
st.sidebar.header("⚙️ Configuration")

# Popular cities for quick selection (use neighborhoods for faster testing)
popular_cities = {
    "Bandra, Mumbai": "Bandra, Mumbai, India",
    "Manhattan, NYC": "Manhattan, New York, USA",
    "Westminster, London": "Westminster, London, United Kingdom",
    "Shibuya, Tokyo": "Shibuya, Tokyo, Japan",
    "1st Arrondissement, Paris": "1st Arrondissement, Paris, France",
    "Central Area, Singapore": "Central Area, Singapore",
    "Connaught Place, Delhi": "Connaught Place, Delhi, India",
    "Custom": "Custom"
}

city_selection = st.sidebar.selectbox(
    "Select a City",
    options=list(popular_cities.keys()),
    index=0
)

if city_selection == "Custom":
    place_name = st.sidebar.text_input(
        "Enter Place Name",
        value="Bandra, Mumbai, India",
        help="Any location searchable on OpenStreetMap (e.g., 'Manhattan, New York', 'Shibuya, Tokyo')"
    )
    st.sidebar.warning("⚠️ Use neighborhoods/districts, not entire cities! Large areas take 10+ minutes.")
else:
    place_name = popular_cities[city_selection]

# Parameters
st.sidebar.subheader("📊 Parameters")

target_population = st.sidebar.slider(
    "Target Population per Zone",
    min_value=1000,
    max_value=10000,
    value=3000,
    step=500,
    help="Ideal population for each zone (used for merging cells)"
)

hex_resolution = st.sidebar.selectbox(
    "H3 Resolution",
    options=[None, 6, 7, 8, 9, 10],
    index=0,
    help="H3 hexagon resolution (lower = larger hexagons). Auto-selects if None."
)

# Database status check
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Database Status")

try:
    from src.database import ZoneManager
    zone_manager = ZoneManager()

    # Check if zones exist in database
    generation_id = zone_manager.check_zones_exist(
        place_name=place_name,
        target_population=target_population,
        buffer_distance=50,  # Default buffer distance
        hex_resolution=hex_resolution
    )

    if generation_id:
        st.sidebar.success("✅ Cached zones available")
        st.sidebar.caption("Will load from database (~2s)")
    else:
        st.sidebar.info("🔄 New generation required")
        st.sidebar.caption("Will generate fresh (~10-15 min)")

except Exception as e:
    st.sidebar.warning("⚠️ Database unavailable")
    st.sidebar.caption("Will use file-based caching")
    zone_manager = None

# Load Cached Zones section
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Load Cached Zones")

if zone_manager:
    try:
        # Get list of available cities from database
        available_cities = zone_manager.list_available_cities()

        if available_cities and len(available_cities) > 0:
            # Create options for selectbox
            city_options = {}
            for city in available_cities:
                city_name = city['place_name']
                num_gens = city['num_generations']
                city_options[f"{city_name} ({num_gens} generation{'s' if num_gens != 1 else ''})"] = city_name

            # Dropdown to select cached city
            selected_cached = st.sidebar.selectbox(
                "Select from database:",
                options=["-- Select a city --"] + list(city_options.keys()),
                key="cached_city_selector"
            )

            # If user selected a city, show generation options
            if selected_cached != "-- Select a city --":
                selected_city_name = city_options[selected_cached]

                # Get all generations for this city
                query = """
                SELECT
                    g.generation_id,
                    g.target_population,
                    g.buffer_distance,
                    g.hex_resolution,
                    g.num_zones,
                    g.created_at,
                    g.is_current
                FROM zone_generations g
                JOIN cities c ON g.city_id = c.city_id
                WHERE c.place_name = %s
                ORDER BY g.created_at DESC;
                """
                generations = zone_manager.db.execute_query(query, (selected_city_name,))

                if generations:
                    # Show generation details
                    st.sidebar.caption(f"**{len(generations)} generation(s) available:**")

                    gen_options = {}
                    for gen in generations:
                        gen_id = gen['generation_id']
                        created = gen['created_at'].strftime('%Y-%m-%d %H:%M')
                        num_zones = gen['num_zones']
                        target_pop = gen['target_population']
                        current = " ⭐ Current" if gen['is_current'] else ""

                        gen_label = f"Gen #{gen_id}: {num_zones} zones (pop={target_pop}) - {created}{current}"
                        gen_options[gen_label] = gen_id

                    selected_gen_label = st.sidebar.selectbox(
                        "Select generation:",
                        options=list(gen_options.keys()),
                        key="gen_selector"
                    )

                    selected_gen_id = gen_options[selected_gen_label]

                    # Load button
                    if st.sidebar.button("📂 Load Cached Zones", use_container_width=True):
                        with st.spinner('Loading zones from database...'):
                            try:
                                # Clear existing session state first
                                st.session_state.zones_gdf = None
                                st.session_state.centroids_gdf = None
                                st.session_state.results = None

                                cached_data = zone_manager.load_zone_generation(selected_gen_id)

                                if cached_data:
                                    # Store in session state
                                    st.session_state.zones_gdf = cached_data['zones_gdf']
                                    st.session_state.centroids_gdf = cached_data['centroids_gdf']

                                    # Create results dict for compatibility
                                    metadata = cached_data['metadata']

                                    # Create temporary output directory and save files
                                    import os
                                    output_dir = Path(f"./output_cache_{selected_gen_id}")
                                    output_dir.mkdir(exist_ok=True)

                                    # Save GeoJSON files for download
                                    cached_data['zones_gdf'].to_file(output_dir / 'zones.geojson', driver='GeoJSON')
                                    cached_data['centroids_gdf'].to_file(output_dir / 'centroids.geojson', driver='GeoJSON')

                                    # Create zones summary CSV
                                    zones_df = cached_data['zones_gdf'].drop(columns=['geometry'])
                                    zones_df.to_csv(output_dir / 'zones_summary.csv', index=False)

                                    # Save skim matrices if available (skip for very large datasets)
                                    num_zones = metadata['num_zones']
                                    skim_matrices = cached_data.get('skim_matrices', {})

                                    if num_zones < 5000:  # Only export skim CSVs for smaller datasets
                                        if 'distance_km' in skim_matrices:
                                            skim_matrices['distance_km'].to_csv(output_dir / 'skim_distance_km.csv')
                                        if 'time_drive_min' in skim_matrices:
                                            skim_matrices['time_drive_min'].to_csv(output_dir / 'skim_time_drive_min.csv')
                                        if 'time_transit_min' in skim_matrices:
                                            skim_matrices['time_transit_min'].to_csv(output_dir / 'skim_time_transit_min.csv')
                                        if 'time_walk_min' in skim_matrices:
                                            skim_matrices['time_walk_min'].to_csv(output_dir / 'skim_time_walk_min.csv')
                                        if 'cost_drive' in skim_matrices:
                                            skim_matrices['cost_drive'].to_csv(output_dir / 'skim_cost_drive.csv')
                                    else:
                                        # For large datasets, create a note file instead
                                        with open(output_dir / 'LARGE_DATASET_NOTE.txt', 'w') as f:
                                            f.write(f"This dataset has {num_zones} zones.\n")
                                            f.write(f"Skim matrices are available in the database but not exported to CSV due to size.\n")
                                            f.write(f"Matrix would have {num_zones * num_zones:,} entries.\n")
                                            f.write(f"\nTo export skim matrices, query the database directly:\n")
                                            f.write(f"SELECT * FROM skim_matrices WHERE generation_id = {selected_gen_id};\n")

                                    st.session_state.results = {
                                        'num_zones': metadata['num_zones'],
                                        'total_area_km2': metadata['total_area_km2'],
                                        'avg_proxy_population': metadata['total_proxy_population'] / metadata['num_zones'] if metadata['num_zones'] > 0 else 0,
                                        'avg_proxy_employment': metadata['total_proxy_employment'] / metadata['num_zones'] if metadata['num_zones'] > 0 else 0,
                                        'output_dir': str(output_dir)
                                    }

                                    st.sidebar.success(f"✅ Loaded {metadata['num_zones']} zones!")
                                    st.rerun()
                                else:
                                    st.sidebar.error("Failed to load zones from database")
                            except Exception as e:
                                st.sidebar.error(f"Error loading zones: {str(e)}")
                                import traceback
                                st.sidebar.code(traceback.format_exc(), language="python")
        else:
            st.sidebar.info("No cached zones available yet")
    except Exception as e:
        st.sidebar.warning(f"Could not fetch cached cities")
else:
    st.sidebar.info("Database not connected")

# Generate button
st.sidebar.markdown("---")
generate_button = st.sidebar.button("🚀 Generate Zones", type="primary", use_container_width=True)

# Session state for storing results
if 'zones_gdf' not in st.session_state:
    st.session_state.zones_gdf = None
if 'centroids_gdf' not in st.session_state:
    st.session_state.centroids_gdf = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Main content
if generate_button:
    with st.spinner(f'🔄 Generating zones for {place_name}...'):
        try:
            # Create generator
            generator = AutomatedZoneGenerator(
                place_name=place_name,
                target_population=target_population,
                output_dir=f"./output_{place_name.replace(',', '').replace(' ', '_').lower()}",
                hex_resolution=hex_resolution if hex_resolution else None
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Generate zones
            start_time = time.time()

            status_text.text("⏳ Step 1/8: Extracting OSM data...")
            progress_bar.progress(12)

            results = generator.generate_zones()

            progress_bar.progress(100)
            elapsed_time = time.time() - start_time

            status_text.text(f"✅ Complete in {elapsed_time:.1f} seconds!")

            # Load results
            output_dir = Path(results['output_dir'])
            zones_gdf = gpd.read_file(output_dir / 'zones.geojson')
            centroids_gdf = gpd.read_file(output_dir / 'centroids.geojson')

            # Store in session state
            st.session_state.zones_gdf = zones_gdf
            st.session_state.centroids_gdf = centroids_gdf
            st.session_state.results = results

            st.success(f"✅ Successfully generated {results['num_zones']} zones!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display results if available
if st.session_state.zones_gdf is not None:
    zones_gdf = st.session_state.zones_gdf
    centroids_gdf = st.session_state.centroids_gdf
    results = st.session_state.results

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Zones", results['num_zones'])
    with col2:
        st.metric("Total Area", f"{results['total_area_km2']:.2f} km²")
    with col3:
        st.metric("Avg Population/Zone", f"{results['avg_proxy_population']:.0f}")
    with col4:
        st.metric("Avg Employment/Zone", f"{results.get('avg_proxy_employment', 0):.0f}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📊 Statistics", "📋 Zone Details", "📥 Download"])

    with tab1:
        st.subheader("Interactive Zone Map")

        # Create folium map
        center_lat = zones_gdf.geometry.centroid.y.mean()
        center_lon = zones_gdf.geometry.centroid.x.mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )

        # Add zones
        folium.GeoJson(
            zones_gdf,
            name='Zones',
            style_function=lambda x: {
                'fillColor': '#3388ff',
                'color': '#0066cc',
                'weight': 2,
                'fillOpacity': 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['zone_id', 'dominant_landuse', 'proxy_population', 'proxy_employment', 'area_km2'],
                aliases=['Zone ID:', 'Land Use:', 'Population:', 'Employment:', 'Area (km²):'],
                localize=True
            )
        ).add_to(m)

        # Add centroids
        for idx, row in centroids_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                popup=f"Zone {row['zone_id']}"
            ).add_to(m)

        # Display map
        st_folium(m, width=1400, height=600)

    with tab2:
        st.subheader("Zone Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Land use distribution
            st.write("**Land Use Distribution**")
            land_use_counts = zones_gdf['dominant_landuse'].value_counts()
            fig = px.pie(
                values=land_use_counts.values,
                names=land_use_counts.index,
                title="Zones by Land Use Type"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Zone size distribution
            st.write("**Zone Size Distribution**")
            fig = px.histogram(
                zones_gdf,
                x='area_km2',
                nbins=20,
                title="Distribution of Zone Areas",
                labels={'area_km2': 'Area (km²)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Population vs Employment scatter
        st.write("**Population vs Employment**")
        fig = px.scatter(
            zones_gdf,
            x='proxy_population',
            y='proxy_employment',
            color='dominant_landuse',
            size='area_km2',
            hover_data=['zone_id'],
            title="Population vs Employment by Land Use",
            labels={
                'proxy_population': 'Proxy Population',
                'proxy_employment': 'Proxy Employment',
                'dominant_landuse': 'Land Use'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Zone Details Table")

        # Display zone attributes
        display_cols = [
            'zone_id', 'dominant_landuse', 'area_km2',
            'proxy_population', 'proxy_employment',
            'is_cbd', 'is_special_generator'
        ]

        available_cols = [col for col in display_cols if col in zones_gdf.columns]

        st.dataframe(
            zones_gdf[available_cols].sort_values('zone_id'),
            use_container_width=True,
            height=400
        )

        # Summary statistics
        st.write("**Summary Statistics**")
        st.dataframe(
            zones_gdf[['area_km2', 'proxy_population', 'proxy_employment']].describe(),
            use_container_width=True
        )

    with tab4:
        st.subheader("Download Generated Files")

        output_dir = Path(results['output_dir'])

        col1, col2 = st.columns(2)

        with col1:
            st.write("**GeoJSON Files**")

            # Zones GeoJSON
            with open(output_dir / 'zones.geojson', 'r') as f:
                st.download_button(
                    label="📥 Download Zones (GeoJSON)",
                    data=f.read(),
                    file_name="zones.geojson",
                    mime="application/json"
                )

            # Centroids GeoJSON
            with open(output_dir / 'centroids.geojson', 'r') as f:
                st.download_button(
                    label="📥 Download Centroids (GeoJSON)",
                    data=f.read(),
                    file_name="centroids.geojson",
                    mime="application/json"
                )

        with col2:
            st.write("**CSV Files**")

            # Zone summary
            if (output_dir / 'zones_summary.csv').exists():
                zones_summary = pd.read_csv(output_dir / 'zones_summary.csv')
                st.download_button(
                    label="📥 Download Zone Summary (CSV)",
                    data=zones_summary.to_csv(index=False),
                    file_name="zones_summary.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Zone summary CSV not yet available")

            # Distance skim
            if (output_dir / 'skim_distance_km.csv').exists():
                distance_skim = pd.read_csv(output_dir / 'skim_distance_km.csv')
                st.download_button(
                    label="📥 Download Distance Matrix (CSV)",
                    data=distance_skim.to_csv(index=False),
                    file_name="skim_distance_km.csv",
                    mime="text/csv"
                )
            else:
                st.info("⏳ Distance matrix is being prepared for large datasets...")
                st.caption("For datasets with >1000 zones, matrix export may take a minute. Check the output folder later.")

        st.info(f"💡 All files are also saved in: `{output_dir}`")

        # Show what files are available
        if output_dir.exists():
            available_files = list(output_dir.glob('*'))
            if available_files:
                st.caption(f"✓ {len(available_files)} file(s) ready in output folder")

else:
    # Welcome message
    st.info("On the panel, Select a city and click 'Generate Zones' to start!")

    st.markdown("""
    ### How It Works:

    1. **Select a city** from the dropdown or enter a custom location
    2. **Adjust parameters** like target population and resolution
    3. **Click Generate** to create zones using OpenStreetMap data
    4. **Visualize** zones on an interactive map
    5. **Download** GeoJSON and CSV files for use in transport models

    ### Current Features:

    - Works for **any city in the world** 🌍
    - Uses only **OpenStreetMap** data (no Census required)
    - Generates **TAZ-like zones** suitable for four-step models
    - Includes **skim matrices** for distance, time, and cost
    - Classifies zones by **land use** (residential, commercial, mixed, etc.)
    - Identifies **CBD** and **special generators**
    - **Database caching** - Previously generated zones load instantly (<2s) 💾
    - **Docker-based** PostgreSQL + PostGIS backend for portability

    ### Next Steps According to Plan:

    After generating zones, you can:
    - Import into **QGIS**, **ArcGIS**, or other GIS software
    - Use skim matrices in **PTV Visum**, **EMME**, or custom models
    - Calibrate with Census/survey data when available
    - Integrate with GPS/CDR/GTFS transit data
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🎓 IIT Bombay Research Project**")
st.sidebar.markdown("Urban Transit Planning Tool")
