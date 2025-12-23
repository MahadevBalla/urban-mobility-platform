"""
Test Script for Automated Zone Generation
Run this to test the complete zone generation pipeline
"""

import sys
sys.path.append('src')

from zone_generation.zone_generator import AutomatedZoneGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "=" * 70)
    print("AUTOMATED ZONE GENERATION - TEST SCRIPT")
    print("=" * 70)

    # Test with a small area first
    print("\nTest 1: Small area (Bandra, Mumbai)")
    print("-" * 70)

    try:
        generator = AutomatedZoneGenerator(
            place_name="Bandra, Mumbai, India",
            target_population=3000,
            buffer_distance=30,
            output_dir="./output_bandra_test"
        )

        results = generator.generate_zones()

        print("\n" + "=" * 70)
        print("TEST 1 - SUCCESS")
        print("=" * 70)
        print(f"✅ Zones created: {results['num_zones']}")
        print(f"✅ Total area: {results['total_area_km2']:.2f} km²")
        print(f"✅ Proxy population: {results['total_proxy_population']:.0f}")
        print(f"✅ Proxy employment: {results['total_proxy_employment']:.0f}")
        print(f"✅ Output directory: {results['output_dir']}")

        print(f"\n📊 Land use distribution:")
        for land_use, count in results['land_use_distribution'].items():
            print(f"   {land_use}: {count} zones")

        print(f"\n🏙️  CBD zones: {results['num_cbd_zones']}")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)

        print("\n📁 Check the output directory for:")
        print("   - zones.geojson (zone polygons)")
        print("   - centroids.geojson (zone centroids)")
        print("   - connectors.geojson (network connectors)")
        print("   - skim_*.csv (distance/time/cost matrices)")
        print("   - zones_summary.csv (zone attributes)")

        print("\n💡 Next steps:")
        print("   1. Visualize zones in QGIS or your preferred GIS tool")
        print("   2. Try different cities by changing place_name")
        print("   3. Adjust target_population for different zone sizes")
        print("   4. Calibrate with Census data when available")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
