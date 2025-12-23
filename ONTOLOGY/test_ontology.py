"""
Quick test script for Transport Data Ontology

Run this to verify ontology is working correctly.

Usage:
    cd e:/urban_transit_tool
    python ONTOLOGY/test_ontology.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Testing Transport Data Ontology")
print("=" * 60)

# Test 1: Import base module
print("\n[1/5] Testing base module import...")
try:
    from ONTOLOGY.ontology_base import TransportEntity, Zone, TransportMode
    print("✅ Base module imported successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Import all modules
print("\n[2/5] Testing all module imports...")
try:
    from ONTOLOGY import ontology_census, ontology_hts, ontology_mobile
    from ONTOLOGY import ontology_probe, ontology_gtfs, ontology_osm
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Create entities
print("\n[3/5] Testing entity creation...")
try:
    from ONTOLOGY.ontology_base import Zone, Sex
    from ONTOLOGY.ontology_census import Person, Household
    from ONTOLOGY.ontology_hts import Trip, TripPurpose
    
    zone = Zone(
        zone_id="TAZ_001",
        area_km2=2.5,
        geometry=None,
        zone_name="Test Zone",
        population=10000
    )
    
    household = Household(
        entity_id="HH_001",
        entity_type="Household",
        household_id="HH_001",
        zone_id=zone.zone_id,
        household_size=3
    )
    
    person = Person(
        entity_id="P_001",
        entity_type="Person",
        person_id="P_001",
        household_id=household.household_id,
        age=35,
        sex=Sex.MALE
    )
    
    trip = Trip(
        entity_id="TRIP_001",
        entity_type="Trip",
        trip_id="TRIP_001",
        survey_person_id=person.person_id,
        origin_zone_id=zone.zone_id,
        destination_zone_id="TAZ_045",
        mode=TransportMode.METRO,
        trip_purpose=TripPurpose.WORK
    )
    
    print("✅ Entities created successfully")
    print(f"   Zone: {zone.zone_name} ({zone.population} people)")
    print(f"   Household: {household.household_size} persons")
    print(f"   Person: {person.age} years, {person.sex.value}")
    print(f"   Trip: {trip.mode.value} for {trip.trip_purpose.value}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test serialization
print("\n[4/5] Testing serialization...")
try:
    person_dict = person.to_dict()
    person_json = person.to_json()
    print("✅ Serialization working")
    print(f"   to_dict() returned {len(person_dict)} fields")
    print(f"   to_json() length: {len(person_json)} chars")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Print summary
print("\n[5/5] Testing module summary...")
try:
    from ONTOLOGY import print_summary
    print()
    print_summary()
    print("\n✅ Summary printed successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\nOntology is ready to use!")
print("\nNext steps:")
print("  1. I have provided documentation in README.md")
print("  2. Check ONTOLOGY/QUICK_START_GUIDE.md for examples")
print("  3. We can possibly start integrating with data sources")
print("\nExample usage:")
print("  from ONTOLOGY import Zone, Person, Trip")
print("  zone = Zone(zone_id='TAZ_001', area_km2=2.5, population=10000)")
