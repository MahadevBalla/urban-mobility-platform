"""
Ground Truth Generator

Generates perfect vehicle trajectories for evaluation purposes.
These trajectories serve as the gold standard against which all
fusion algorithms are compared.

Features:
- Creates realistic bus route on road network
- Generates exact positions at 1-second intervals
- Records precise arrival/departure times at stops
- Simulates realistic speed profiles (acceleration, cruising, deceleration)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import substring
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Stop:
    """Represents a bus stop along the route."""
    stop_id: str
    name: str
    location: Point
    distance_along_route: float  # meters from start
    dwell_time: float = 30.0  # seconds


@dataclass
class GroundTruthPoint:
    """A single point in the ground truth trajectory."""
    timestamp: datetime
    latitude: float
    longitude: float
    speed_mps: float  # meters per second
    distance_traveled: float  # total distance from start
    heading: float  # degrees, 0 = north
    at_stop: Optional[str] = None  # stop_id if at a stop


@dataclass
class GroundTruthTrip:
    """A complete ground truth trip."""
    trip_id: str
    vehicle_id: str
    route_id: str
    start_time: datetime
    end_time: datetime
    points: List[GroundTruthPoint] = field(default_factory=list)
    stop_arrivals: Dict[str, datetime] = field(default_factory=dict)
    stop_departures: Dict[str, datetime] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trip points to DataFrame."""
        records = []
        for p in self.points:
            records.append({
                'trip_id': self.trip_id,
                'vehicle_id': self.vehicle_id,
                'route_id': self.route_id,
                'timestamp': p.timestamp,
                'latitude': p.latitude,
                'longitude': p.longitude,
                'speed_mps': p.speed_mps,
                'speed_kmh': p.speed_mps * 3.6,
                'distance_traveled': p.distance_traveled,
                'heading': p.heading,
                'at_stop': p.at_stop
            })
        return pd.DataFrame(records)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert trip points to GeoDataFrame."""
        df = self.to_dataframe()
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        return gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')


class GroundTruthGenerator:
    """
    Generates ground truth vehicle trajectories for fusion evaluation.

    The generator creates realistic bus trajectories along a defined route,
    with proper speed profiles, stop dwell times, and precise timing.
    """

    def __init__(
        self,
        route_line: LineString = None,
        stops: List[Stop] = None,
        avg_speed_kmh: float = 25.0,
        max_speed_kmh: float = 45.0,
        acceleration: float = 1.2,  # m/s²
        deceleration: float = 1.5,  # m/s²
        seed: int = 42
    ):
        """
        Initialize the ground truth generator.

        Args:
            route_line: LineString representing the route path
            stops: List of Stop objects along the route
            avg_speed_kmh: Average speed between stops
            max_speed_kmh: Maximum cruising speed
            acceleration: Acceleration rate in m/s²
            deceleration: Deceleration rate in m/s²
            seed: Random seed for reproducibility
        """
        self.route_line = route_line
        self.stops = stops or []
        self.avg_speed_kmh = avg_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.rng = np.random.RandomState(seed)

        # Convert speeds to m/s
        self.avg_speed_mps = avg_speed_kmh / 3.6
        self.max_speed_mps = max_speed_kmh / 3.6

        # If no route provided, generate a default one
        if self.route_line is None:
            self._generate_default_route()

    def _generate_default_route(self):
        """Generate a default bus route in Mumbai (Bandra area)."""
        # Create a realistic bus route path (approximately 4km)
        # Coordinates are in Bandra West, Mumbai
        route_coords = [
            (72.8296, 19.0596),  # Start: Bandra Station
            (72.8310, 19.0610),
            (72.8325, 19.0625),
            (72.8340, 19.0640),
            (72.8355, 19.0655),  # Stop 1
            (72.8370, 19.0670),
            (72.8385, 19.0680),
            (72.8400, 19.0690),
            (72.8415, 19.0700),  # Stop 2
            (72.8430, 19.0710),
            (72.8445, 19.0720),
            (72.8460, 19.0730),
            (72.8475, 19.0740),  # Stop 3
            (72.8490, 19.0750),
            (72.8505, 19.0760),
            (72.8520, 19.0770),
            (72.8535, 19.0780),  # Stop 4
            (72.8550, 19.0790),
            (72.8565, 19.0800),
            (72.8580, 19.0810),
            (72.8595, 19.0820),  # Stop 5
            (72.8610, 19.0830),
            (72.8625, 19.0840),
            (72.8640, 19.0850),  # End: Terminal
        ]

        self.route_line = LineString(route_coords)

        # Generate stops along the route
        route_length = self._get_route_length_meters()
        stop_distances = [0, 500, 1000, 1500, 2000, 2500, 3000, route_length]
        stop_names = [
            "Bandra Station",
            "Hill Road Junction",
            "Turner Road",
            "Linking Road",
            "SV Road Junction",
            "Khar Road",
            "Santacruz Link",
            "Terminal"
        ]

        self.stops = []
        for i, (dist, name) in enumerate(zip(stop_distances, stop_names)):
            # Get point at distance along route
            point = self._get_point_at_distance(dist)
            stop = Stop(
                stop_id=f"STOP_{i+1:03d}",
                name=name,
                location=point,
                distance_along_route=dist,
                dwell_time=30 + self.rng.uniform(-10, 20)  # 20-50 seconds
            )
            self.stops.append(stop)

    def _get_route_length_meters(self) -> float:
        """Get route length in meters (approximate using degree to meter conversion)."""
        # Simple approximation: 1 degree ≈ 111km at equator, adjust for latitude
        coords = list(self.route_line.coords)
        total_length = 0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            # Haversine approximation
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            total_length += 6371000 * c  # Earth radius in meters
        return total_length

    def _get_point_at_distance(self, distance_m: float) -> Point:
        """Get point at given distance along route."""
        route_length = self._get_route_length_meters()
        if distance_m >= route_length:
            return Point(self.route_line.coords[-1])

        # Normalize distance to route length ratio
        fraction = distance_m / route_length
        point = self.route_line.interpolate(fraction, normalized=True)
        return point

    def _calculate_heading(self, p1: Point, p2: Point) -> float:
        """Calculate heading from p1 to p2 in degrees (0 = north)."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        heading = np.degrees(np.arctan2(dx, dy))
        return (heading + 360) % 360

    def _generate_speed_profile(
        self,
        segment_length: float,
        dwell_at_end: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Generate realistic speed profile for a segment.

        Returns list of (distance, speed) tuples representing
        acceleration, cruising, and deceleration phases.
        """
        speeds = []

        # Calculate distances for each phase
        accel_time = self.max_speed_mps / self.acceleration
        accel_dist = 0.5 * self.acceleration * accel_time**2

        decel_time = self.max_speed_mps / self.deceleration
        decel_dist = 0.5 * self.deceleration * decel_time**2

        cruise_dist = segment_length - accel_dist - decel_dist

        if cruise_dist < 0:
            # Segment too short for full acceleration/deceleration
            # Use triangular profile
            max_achievable_speed = np.sqrt(
                2 * segment_length * self.acceleration * self.deceleration /
                (self.acceleration + self.deceleration)
            )
            accel_dist = max_achievable_speed**2 / (2 * self.acceleration)
            decel_dist = segment_length - accel_dist
            cruise_dist = 0

        return {
            'accel_dist': accel_dist,
            'cruise_dist': cruise_dist,
            'decel_dist': decel_dist,
            'max_speed': min(self.max_speed_mps,
                           np.sqrt(2 * accel_dist * self.acceleration))
        }

    def generate_trip(
        self,
        trip_id: str,
        vehicle_id: str,
        route_id: str,
        start_time: datetime,
        schedule_deviation_seconds: float = 0,
        speed_variation: float = 0.1
    ) -> GroundTruthTrip:
        """
        Generate a single ground truth trip.

        Args:
            trip_id: Unique trip identifier
            vehicle_id: Vehicle identifier
            route_id: Route identifier
            start_time: Trip start time
            schedule_deviation_seconds: Deviation from schedule (positive = late)
            speed_variation: Random speed variation factor (0.1 = ±10%)

        Returns:
            GroundTruthTrip with all trajectory points
        """
        trip = GroundTruthTrip(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            route_id=route_id,
            start_time=start_time,
            end_time=start_time  # Will be updated
        )

        current_time = start_time + timedelta(seconds=schedule_deviation_seconds)
        current_distance = 0
        route_length = self._get_route_length_meters()

        # Process each segment between stops
        for i in range(len(self.stops) - 1):
            start_stop = self.stops[i]
            end_stop = self.stops[i + 1]
            segment_length = end_stop.distance_along_route - start_stop.distance_along_route

            # Record arrival at start stop (if not first stop)
            if i == 0:
                trip.stop_arrivals[start_stop.stop_id] = current_time

            # Dwell at stop
            dwell_time = start_stop.dwell_time * (1 + self.rng.uniform(-0.2, 0.3))

            # Add points during dwell
            dwell_end = current_time + timedelta(seconds=dwell_time)
            while current_time < dwell_end:
                point = GroundTruthPoint(
                    timestamp=current_time,
                    latitude=start_stop.location.y,
                    longitude=start_stop.location.x,
                    speed_mps=0,
                    distance_traveled=current_distance,
                    heading=self._calculate_heading(start_stop.location, end_stop.location),
                    at_stop=start_stop.stop_id
                )
                trip.points.append(point)
                current_time += timedelta(seconds=1)

            trip.stop_departures[start_stop.stop_id] = current_time

            # Generate speed profile for segment
            profile = self._generate_speed_profile(segment_length)

            # Move through segment
            segment_distance = 0
            while segment_distance < segment_length:
                # Determine current speed based on position in segment
                if segment_distance < profile['accel_dist']:
                    # Acceleration phase
                    speed = np.sqrt(2 * self.acceleration * max(segment_distance, 0.1))
                elif segment_distance < profile['accel_dist'] + profile['cruise_dist']:
                    # Cruising phase
                    speed = profile['max_speed']
                else:
                    # Deceleration phase
                    remaining = segment_length - segment_distance
                    speed = np.sqrt(2 * self.deceleration * max(remaining, 0.1))

                # Apply random variation
                speed *= (1 + self.rng.uniform(-speed_variation, speed_variation))
                speed = max(0.5, min(speed, self.max_speed_mps))  # Clamp

                # Calculate position
                total_distance = current_distance + segment_distance
                position = self._get_point_at_distance(total_distance)

                # Calculate heading
                next_pos = self._get_point_at_distance(total_distance + 10)
                heading = self._calculate_heading(position, next_pos)

                # Create point
                point = GroundTruthPoint(
                    timestamp=current_time,
                    latitude=position.y,
                    longitude=position.x,
                    speed_mps=speed,
                    distance_traveled=total_distance,
                    heading=heading,
                    at_stop=None
                )
                trip.points.append(point)

                # Advance time and distance
                current_time += timedelta(seconds=1)
                segment_distance += speed  # distance = speed * 1 second

            current_distance += segment_length

            # Record arrival at end stop
            trip.stop_arrivals[end_stop.stop_id] = current_time

        # Handle final stop
        final_stop = self.stops[-1]
        final_dwell = final_stop.dwell_time
        dwell_end = current_time + timedelta(seconds=final_dwell)

        while current_time < dwell_end:
            point = GroundTruthPoint(
                timestamp=current_time,
                latitude=final_stop.location.y,
                longitude=final_stop.location.x,
                speed_mps=0,
                distance_traveled=current_distance,
                heading=0,
                at_stop=final_stop.stop_id
            )
            trip.points.append(point)
            current_time += timedelta(seconds=1)

        trip.stop_departures[final_stop.stop_id] = current_time
        trip.end_time = current_time

        return trip

    def generate_trips(
        self,
        num_trips: int = 10,
        route_id: str = "ROUTE_001",
        start_date: datetime = None,
        headway_minutes: float = 15,
        num_vehicles: int = 3
    ) -> List[GroundTruthTrip]:
        """
        Generate multiple ground truth trips.

        Args:
            num_trips: Number of trips to generate
            route_id: Route identifier
            start_date: Start date/time for first trip
            headway_minutes: Time between consecutive trips
            num_vehicles: Number of vehicles to rotate

        Returns:
            List of GroundTruthTrip objects
        """
        if start_date is None:
            start_date = datetime(2025, 1, 15, 7, 0, 0)  # 7:00 AM

        trips = []
        current_time = start_date

        for i in range(num_trips):
            vehicle_id = f"V{(i % num_vehicles) + 1:03d}"
            trip_id = f"TRIP_{i+1:04d}"

            # Add some random schedule deviation (-60 to +180 seconds)
            deviation = self.rng.uniform(-60, 180)

            trip = self.generate_trip(
                trip_id=trip_id,
                vehicle_id=vehicle_id,
                route_id=route_id,
                start_time=current_time,
                schedule_deviation_seconds=deviation,
                speed_variation=0.15
            )
            trips.append(trip)

            # Next trip starts after headway
            current_time += timedelta(minutes=headway_minutes)

        return trips

    def get_route_as_geodataframe(self) -> gpd.GeoDataFrame:
        """Get the route as a GeoDataFrame."""
        return gpd.GeoDataFrame(
            {'route_id': ['ROUTE_001'], 'geometry': [self.route_line]},
            crs='EPSG:4326'
        )

    def get_stops_as_geodataframe(self) -> gpd.GeoDataFrame:
        """Get stops as a GeoDataFrame."""
        records = []
        for stop in self.stops:
            records.append({
                'stop_id': stop.stop_id,
                'stop_name': stop.name,
                'distance_along_route': stop.distance_along_route,
                'dwell_time': stop.dwell_time,
                'geometry': stop.location
            })
        return gpd.GeoDataFrame(records, crs='EPSG:4326')

    def save_ground_truth(
        self,
        trips: List[GroundTruthTrip],
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save ground truth data to files.

        Args:
            trips: List of ground truth trips
            output_dir: Output directory path

        Returns:
            Dictionary of output file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        output_files = {}

        # Save all trips as single CSV
        all_points = pd.concat([trip.to_dataframe() for trip in trips], ignore_index=True)
        trajectory_path = os.path.join(output_dir, 'ground_truth_trajectories.csv')
        all_points.to_csv(trajectory_path, index=False)
        output_files['trajectories'] = trajectory_path

        # Save stop arrivals/departures
        arrival_records = []
        for trip in trips:
            for stop_id, arrival_time in trip.stop_arrivals.items():
                departure_time = trip.stop_departures.get(stop_id)
                arrival_records.append({
                    'trip_id': trip.trip_id,
                    'vehicle_id': trip.vehicle_id,
                    'stop_id': stop_id,
                    'arrival_time': arrival_time,
                    'departure_time': departure_time
                })

        arrivals_df = pd.DataFrame(arrival_records)
        arrivals_path = os.path.join(output_dir, 'ground_truth_stop_times.csv')
        arrivals_df.to_csv(arrivals_path, index=False)
        output_files['stop_times'] = arrivals_path

        # Save route geometry
        route_gdf = self.get_route_as_geodataframe()
        route_path = os.path.join(output_dir, 'route.geojson')
        route_gdf.to_file(route_path, driver='GeoJSON')
        output_files['route'] = route_path

        # Save stops
        stops_gdf = self.get_stops_as_geodataframe()
        stops_path = os.path.join(output_dir, 'stops.geojson')
        stops_gdf.to_file(stops_path, driver='GeoJSON')
        output_files['stops'] = stops_path

        # Save trip summary
        summary_records = []
        for trip in trips:
            summary_records.append({
                'trip_id': trip.trip_id,
                'vehicle_id': trip.vehicle_id,
                'route_id': trip.route_id,
                'start_time': trip.start_time,
                'end_time': trip.end_time,
                'duration_minutes': (trip.end_time - trip.start_time).total_seconds() / 60,
                'num_points': len(trip.points)
            })

        summary_df = pd.DataFrame(summary_records)
        summary_path = os.path.join(output_dir, 'trip_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        output_files['summary'] = summary_path

        print(f"Ground truth data saved to {output_dir}")
        print(f"  - Trajectories: {len(all_points)} points")
        print(f"  - Trips: {len(trips)}")
        print(f"  - Stops: {len(self.stops)}")

        return output_files


# Convenience function
def generate_sample_ground_truth(output_dir: str = None) -> Tuple[List[GroundTruthTrip], GroundTruthGenerator]:
    """
    Generate sample ground truth data for testing.

    Returns:
        Tuple of (list of trips, generator instance)
    """
    generator = GroundTruthGenerator(seed=42)
    trips = generator.generate_trips(
        num_trips=15,
        route_id="ROUTE_001",
        start_date=datetime(2025, 1, 15, 7, 0, 0),
        headway_minutes=12,
        num_vehicles=5
    )

    if output_dir:
        generator.save_ground_truth(trips, output_dir)

    return trips, generator


if __name__ == "__main__":
    # Test the generator
    output_path = "src/data_fusion/data/synthetic/ground_truth"
    trips, generator = generate_sample_ground_truth(output_path)

    print(f"\nGenerated {len(trips)} trips")
    print(f"Route length: {generator._get_route_length_meters():.0f} meters")
    print(f"Number of stops: {len(generator.stops)}")

    # Print sample trip info
    sample_trip = trips[0]
    print(f"\nSample trip: {sample_trip.trip_id}")
    print(f"  Vehicle: {sample_trip.vehicle_id}")
    print(f"  Duration: {(sample_trip.end_time - sample_trip.start_time).total_seconds() / 60:.1f} minutes")
    print(f"  Points: {len(sample_trip.points)}")
