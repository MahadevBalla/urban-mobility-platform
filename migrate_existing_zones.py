"""
Migration Script for Existing Zone Generations
Imports existing output_* folders into the PostgreSQL database
"""

import os
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import ZoneManager, DatabaseConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_output_folder_name(folder_name: str) -> Optional[Dict]:
    """
    Parse output folder name to extract place name and parameters

    Args:
        folder_name: Output folder name (e.g., "output_bandra_mumbai_india")

    Returns:
        Dictionary with parsed parameters or None if invalid
    """
    if not folder_name.startswith("output_"):
        return None

    # Extract place name (remove "output_" prefix)
    place_part = folder_name[7:]  # Remove "output_"

    # Convert underscores back to spaces and commas
    # This is a best-effort reconstruction
    parts = place_part.split('_')

    # Try to reconstruct city name
    # Common patterns: city_country or neighborhood_city_country
    if len(parts) >= 2:
        place_name = ' '.join(parts).title()
        # Add commas after first word (assuming neighborhood, city format)
        if len(parts) >= 3:
            place_name = f"{parts[0].title()}, {parts[1].title()}, {parts[2].title()}"
        elif len(parts) == 2:
            place_name = f"{parts[0].title()}, {parts[1].title()}"
    else:
        place_name = place_part.title()

    return {
        'place_name': place_name,
        'original_folder': folder_name
    }


def load_zone_generation_from_folder(folder_path: Path) -> Optional[Dict]:
    """
    Load zone generation data from an output folder

    Args:
        folder_path: Path to output folder

    Returns:
        Dictionary with zone data or None if invalid
    """
    try:
        # Check required files
        zones_file = folder_path / 'zones.geojson'
        centroids_file = folder_path / 'centroids.geojson'

        if not zones_file.exists() or not centroids_file.exists():
            logger.warning(f"Missing required files in {folder_path.name}")
            return None

        # Load zones and centroids
        zones_gdf = gpd.read_file(zones_file)
        centroids_gdf = gpd.read_file(centroids_file)

        # Convert numpy types to Python types
        for col in zones_gdf.columns:
            if col != 'geometry' and zones_gdf[col].dtype.name.startswith('int'):
                zones_gdf[col] = zones_gdf[col].astype(int)
            elif col != 'geometry' and zones_gdf[col].dtype.name.startswith('float'):
                zones_gdf[col] = zones_gdf[col].astype(float)

        # Load connectors if available
        connectors_file = folder_path / 'connectors.geojson'
        connectors_gdf = None
        if connectors_file.exists():
            connectors_gdf = gpd.read_file(connectors_file)

        # Load skim matrices
        skim_matrices = {}
        for skim_file in folder_path.glob('skim_*.csv'):
            skim_name = skim_file.stem.replace('skim_', '')
            skim_matrices[skim_name] = pd.read_csv(skim_file, index_col=0)

        # Extract parameters from zone data
        # Try to infer target population from average
        avg_population = float(zones_gdf['proxy_population'].mean())
        target_population = int(round(avg_population / 1000) * 1000)  # Round to nearest 1000

        return {
            'zones_gdf': zones_gdf,
            'centroids_gdf': centroids_gdf,
            'connectors_gdf': connectors_gdf,
            'skim_matrices': skim_matrices,
            'target_population': int(target_population),
            'buffer_distance': float(50.0),  # Default value (cannot infer from files)
            'hex_resolution': None  # Cannot infer from files
        }

    except Exception as e:
        logger.error(f"Failed to load {folder_path.name}: {e}")
        return None


def migrate_folder(folder_path: Path, zone_manager: ZoneManager, dry_run: bool = False) -> bool:
    """
    Migrate a single output folder to database

    Args:
        folder_path: Path to output folder
        zone_manager: ZoneManager instance
        dry_run: If True, only simulate migration

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Migrating: {folder_path.name}")
    logger.info(f"{'='*60}")

    # Parse folder name
    folder_info = parse_output_folder_name(folder_path.name)
    if not folder_info:
        logger.warning(f"Could not parse folder name: {folder_path.name}")
        return False

    place_name = folder_info['place_name']
    logger.info(f"Detected place name: {place_name}")

    # Load zone data
    zone_data = load_zone_generation_from_folder(folder_path)
    if not zone_data:
        logger.error(f"Failed to load zone data from {folder_path.name}")
        return False

    logger.info(f"Loaded {len(zone_data['zones_gdf'])} zones")
    logger.info(f"Target population: {zone_data['target_population']}")
    logger.info(f"Skim matrices: {list(zone_data['skim_matrices'].keys())}")

    # Check if already exists in database
    existing_id = zone_manager.check_zones_exist(
        place_name=place_name,
        target_population=zone_data['target_population'],
        buffer_distance=zone_data['buffer_distance'],
        hex_resolution=zone_data['hex_resolution']
    )

    if existing_id:
        logger.warning(f"Zones for {place_name} already exist in database (generation_id={existing_id})")
        logger.info("Skipping to avoid duplicates")
        return True

    if dry_run:
        logger.info("[DRY RUN] Would import to database")
        return True

    # Import to database
    try:
        generation_id = zone_manager.save_zone_generation(
            place_name=place_name,
            zones_gdf=zone_data['zones_gdf'],
            centroids_gdf=zone_data['centroids_gdf'],
            skim_matrices=zone_data['skim_matrices'],
            connectors_gdf=zone_data['connectors_gdf'],
            generation_params={
                'target_population': zone_data['target_population'],
                'buffer_distance': zone_data['buffer_distance'],
                'hex_resolution': zone_data['hex_resolution']
            },
            processing_time=0  # Unknown for migrated data
        )

        logger.info(f"✓ Successfully imported (generation_id={generation_id})")
        return True

    except Exception as e:
        logger.error(f"Failed to import to database: {e}", exc_info=True)
        return False


def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate existing zone generation folders to PostgreSQL database"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simulate migration without actually importing to database"
    )
    parser.add_argument(
        '--folder',
        type=str,
        help="Migrate specific folder only (e.g., output_bandra_mumbai_india)"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("ZONE GENERATION MIGRATION TOOL")
    logger.info("="*60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made to database")

    # Initialize database connection
    try:
        zone_manager = ZoneManager()
        logger.info("✓ Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.error("Make sure Docker services are running: docker-compose up -d")
        return 1

    # Find output folders
    current_dir = Path.cwd()

    if args.folder:
        # Migrate specific folder
        folder_path = current_dir / args.folder
        if not folder_path.exists():
            logger.error(f"Folder not found: {args.folder}")
            return 1

        output_folders = [folder_path]
    else:
        # Find all output folders
        output_folders = [f for f in current_dir.iterdir()
                         if f.is_dir() and f.name.startswith('output_')]

    if not output_folders:
        logger.warning("No output folders found!")
        logger.info("Output folders should be named like: output_cityname")
        return 0

    logger.info(f"\nFound {len(output_folders)} output folder(s):")
    for folder in output_folders:
        logger.info(f"  - {folder.name}")

    # Migrate each folder
    logger.info("\nStarting migration...\n")

    successful = 0
    failed = 0

    for folder in output_folders:
        if migrate_folder(folder, zone_manager, dry_run=args.dry_run):
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(output_folders)}")

    if args.dry_run:
        logger.info("\nThis was a dry run. Run without --dry-run to actually import.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
