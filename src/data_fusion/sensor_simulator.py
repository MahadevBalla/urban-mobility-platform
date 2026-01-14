"""
Sensor Data Simulator

Simulates realistic sensor data from ground truth trajectories:
- GPS Probe Data: Position with noise, dropouts, varying frequency
- GTFS Schedule: Transit schedule with realistic deviations
- CDR Events: Sparse cell tower events for mobile data

Each sensor type has configurable parameters to simulate different
data quality levels for robust evaluation.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import os

from .ground_truth_generator import GroundTruthTrip, GroundTruthGenerator, Stop


@dataclass
class GPSPoint:
    """Simulated GPS point with noise."""
    vehicle_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_kmh: float
    heading: float
    hdop: float  # Horizontal dilution of precision
    num_satellites: int


@dataclass
class CDREvent:
    """Simulated Call Detail Record event."""
    user_id: str
    timestamp: datetime
    cell_tower_id: str
    cell_tower_lat: float
    cell_tower_lon: float
    event_type: str  # 'call', 'sms', 'data'
    duration_seconds: Optional[float] = None


@dataclass
class CellTower:
    """Cell tower definition."""
    tower_id: str
    latitude: float
    longitude: float
    radius_m: float  # Coverage radius


class SensorSimulator:
    """
    Simulates sensor data from ground truth trajectories.

    Supports simulation of:
    - GPS probe data with configurable noise and dropout
    - GTFS schedule data
    - CDR (Call Detail Records) with cell tower triangulation
    """

    def __init__(self, seed: int = 42):
        """Initialize simulator with random seed."""
        self.rng = np.random.RandomState(seed)
        self.cell_towers: List[CellTower] = []

    # ==================== GPS SIMULATION ====================

    def simulate_gps(
        self,
        trips: List[GroundTruthTrip],
        sample_interval: float = 5.0,
        noise_std_meters: float = 8.0,
        dropout_rate: float = 0.1,
        multipath_rate: float = 0.02,
        multipath_jump_meters: float = 30.0
    ) -> pd.DataFrame:
        """
        Simulate GPS probe data from ground truth.

        Args:
            trips: List of ground truth trips
            sample_interval: Seconds between GPS samples
            noise_std_meters: Standard deviation of position noise
            dropout_rate: Probability of missing a sample
            multipath_rate: Probability of multipath error (large jump)
            multipath_jump_meters: Size of multipath error

        Returns:
            DataFrame with simulated GPS points
        """
        gps_records = []

        # Conversion factor: meters to degrees (approximate)
        meters_to_deg = 1 / 111000  # At equator, rough approximation

        for trip in trips:
            # Sample points at specified interval
            sample_times = []
            current_time = trip.start_time

            while current_time <= trip.end_time:
                sample_times.append(current_time)
                current_time += timedelta(seconds=sample_interval)

            for sample_time in sample_times:
                # Check for dropout
                if self.rng.random() < dropout_rate:
                    continue

                # Find closest ground truth point
                gt_point = self._find_closest_point(trip, sample_time)
                if gt_point is None:
                    continue

                # Apply noise
                noise_lat = self.rng.normal(0, noise_std_meters * meters_to_deg)
                noise_lon = self.rng.normal(0, noise_std_meters * meters_to_deg)

                # Apply multipath error occasionally
                if self.rng.random() < multipath_rate:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    noise_lat += multipath_jump_meters * meters_to_deg * np.cos(angle)
                    noise_lon += multipath_jump_meters * meters_to_deg * np.sin(angle)

                # Add timing jitter (±1 second)
                time_jitter = timedelta(seconds=self.rng.uniform(-1, 1))

                # Speed noise (±10%)
                speed_noise = 1 + self.rng.uniform(-0.1, 0.1)

                # Heading noise (±5 degrees)
                heading_noise = self.rng.uniform(-5, 5)

                gps_point = GPSPoint(
                    vehicle_id=trip.vehicle_id,
                    timestamp=sample_time + time_jitter,
                    latitude=gt_point.latitude + noise_lat,
                    longitude=gt_point.longitude + noise_lon,
                    speed_kmh=gt_point.speed_mps * 3.6 * speed_noise,
                    heading=(gt_point.heading + heading_noise) % 360,
                    hdop=self.rng.uniform(0.8, 2.5),
                    num_satellites=self.rng.randint(6, 12)
                )

                gps_records.append({
                    'vehicle_id': gps_point.vehicle_id,
                    'trip_id': trip.trip_id,
                    'timestamp': gps_point.timestamp,
                    'latitude': gps_point.latitude,
                    'longitude': gps_point.longitude,
                    'speed_kmh': gps_point.speed_kmh,
                    'heading': gps_point.heading,
                    'hdop': gps_point.hdop,
                    'num_satellites': gps_point.num_satellites
                })

        df = pd.DataFrame(gps_records)
        df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
        return df

    def _find_closest_point(self, trip: GroundTruthTrip, target_time: datetime):
        """Find the ground truth point closest to target time."""
        min_diff = None
        closest_point = None

        for point in trip.points:
            diff = abs((point.timestamp - target_time).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest_point = point

        return closest_point

    # ==================== GTFS SIMULATION ====================

    def simulate_gtfs(
        self,
        generator: GroundTruthGenerator,
        trips: List[GroundTruthTrip],
        schedule_noise_seconds: float = 60.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Simulate GTFS feed from ground truth.

        Creates standard GTFS files:
        - agency.txt
        - stops.txt
        - routes.txt
        - trips.txt
        - stop_times.txt
        - calendar.txt

        Args:
            generator: Ground truth generator with route/stop info
            trips: Ground truth trips (used to derive schedule)
            schedule_noise_seconds: Random deviation in schedule times

        Returns:
            Dictionary of DataFrames for each GTFS file
        """
        gtfs = {}

        # agency.txt
        gtfs['agency'] = pd.DataFrame([{
            'agency_id': 'AGENCY_001',
            'agency_name': 'City Transit Authority',
            'agency_url': 'http://citytransit.example.com',
            'agency_timezone': 'Asia/Kolkata'
        }])

        # stops.txt
        stops_records = []
        for stop in generator.stops:
            stops_records.append({
                'stop_id': stop.stop_id,
                'stop_name': stop.name,
                'stop_lat': stop.location.y,
                'stop_lon': stop.location.x,
                'location_type': 0,  # Stop
                'wheelchair_boarding': 1
            })
        gtfs['stops'] = pd.DataFrame(stops_records)

        # routes.txt
        gtfs['routes'] = pd.DataFrame([{
            'route_id': 'ROUTE_001',
            'agency_id': 'AGENCY_001',
            'route_short_name': 'R1',
            'route_long_name': 'Bandra - Santacruz Express',
            'route_type': 3,  # Bus
            'route_color': '0000FF',
            'route_text_color': 'FFFFFF'
        }])

        # trips.txt
        trips_records = []
        for trip in trips:
            trips_records.append({
                'route_id': trip.route_id,
                'service_id': 'WEEKDAY',
                'trip_id': trip.trip_id,
                'trip_headsign': 'Santacruz Terminal',
                'direction_id': 0,
                'block_id': trip.vehicle_id
            })
        gtfs['trips'] = pd.DataFrame(trips_records)

        # stop_times.txt
        stop_times_records = []
        for trip in trips:
            for i, stop in enumerate(generator.stops):
                # Get actual arrival time and add some schedule noise
                actual_arrival = trip.stop_arrivals.get(stop.stop_id)
                if actual_arrival:
                    # Schedule is the "planned" time (actual + random deviation)
                    noise = timedelta(seconds=self.rng.uniform(
                        -schedule_noise_seconds, schedule_noise_seconds
                    ))
                    scheduled_arrival = actual_arrival + noise

                    # Departure is arrival + dwell time
                    scheduled_departure = scheduled_arrival + timedelta(seconds=stop.dwell_time)

                    stop_times_records.append({
                        'trip_id': trip.trip_id,
                        'arrival_time': scheduled_arrival.strftime('%H:%M:%S'),
                        'departure_time': scheduled_departure.strftime('%H:%M:%S'),
                        'stop_id': stop.stop_id,
                        'stop_sequence': i + 1,
                        'pickup_type': 0,
                        'drop_off_type': 0
                    })

        gtfs['stop_times'] = pd.DataFrame(stop_times_records)

        # calendar.txt
        gtfs['calendar'] = pd.DataFrame([{
            'service_id': 'WEEKDAY',
            'monday': 1,
            'tuesday': 1,
            'wednesday': 1,
            'thursday': 1,
            'friday': 1,
            'saturday': 0,
            'sunday': 0,
            'start_date': '20250101',
            'end_date': '20251231'
        }])

        return gtfs

    # ==================== CDR SIMULATION ====================

    def generate_cell_towers(
        self,
        generator: GroundTruthGenerator,
        num_towers: int = 15,
        coverage_radius_m: float = 300.0
    ) -> List[CellTower]:
        """
        Generate cell towers along the route.

        Args:
            generator: Ground truth generator with route info
            num_towers: Number of cell towers to generate
            coverage_radius_m: Coverage radius of each tower

        Returns:
            List of CellTower objects
        """
        route_length = generator._get_route_length_meters()
        tower_spacing = route_length / (num_towers - 1)

        self.cell_towers = []
        for i in range(num_towers):
            distance = i * tower_spacing
            point = generator._get_point_at_distance(distance)

            # Add some offset from route (towers aren't exactly on the road)
            offset_lat = self.rng.uniform(-0.002, 0.002)
            offset_lon = self.rng.uniform(-0.002, 0.002)

            tower = CellTower(
                tower_id=f"TOWER_{i+1:03d}",
                latitude=point.y + offset_lat,
                longitude=point.x + offset_lon,
                radius_m=coverage_radius_m * self.rng.uniform(0.8, 1.2)
            )
            self.cell_towers.append(tower)

        return self.cell_towers

    def simulate_cdr(
        self,
        trips: List[GroundTruthTrip],
        events_per_hour: float = 8.0,
        include_handovers: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate CDR (Call Detail Records) from ground truth.

        CDR data is much sparser than GPS and only records events
        at cell tower locations.

        Args:
            trips: Ground truth trips
            events_per_hour: Average number of events per hour
            include_handovers: Include cell tower handover events

        Returns:
            Tuple of (CDR events DataFrame, Cell towers DataFrame)
        """
        if not self.cell_towers:
            raise ValueError("Cell towers not generated. Call generate_cell_towers first.")

        cdr_records = []

        for trip in trips:
            # User ID is derived from vehicle ID (simulating driver's phone)
            user_id = f"USER_{trip.vehicle_id}"

            # Calculate trip duration
            trip_duration_hours = (trip.end_time - trip.start_time).total_seconds() / 3600
            expected_events = int(events_per_hour * trip_duration_hours)

            # Generate random event times
            event_times = []
            for _ in range(expected_events):
                random_offset = timedelta(
                    seconds=self.rng.uniform(0, trip_duration_hours * 3600)
                )
                event_time = trip.start_time + random_offset
                event_times.append(event_time)

            event_times.sort()

            for event_time in event_times:
                # Find position at event time
                gt_point = self._find_closest_point(trip, event_time)
                if gt_point is None:
                    continue

                # Find nearest cell tower
                nearest_tower = self._find_nearest_tower(
                    gt_point.latitude, gt_point.longitude
                )

                # Event type
                event_type = self.rng.choice(['call', 'sms', 'data'], p=[0.2, 0.1, 0.7])
                duration = None
                if event_type == 'call':
                    duration = self.rng.exponential(120)  # Average 2 min call

                cdr_records.append({
                    'user_id': user_id,
                    'trip_id': trip.trip_id,
                    'timestamp': event_time,
                    'cell_tower_id': nearest_tower.tower_id,
                    'cell_tower_lat': nearest_tower.latitude,
                    'cell_tower_lon': nearest_tower.longitude,
                    'event_type': event_type,
                    'duration_seconds': duration
                })

            # Add handover events (when moving between tower coverage areas)
            if include_handovers:
                handover_events = self._generate_handovers(trip)
                cdr_records.extend(handover_events)

        cdr_df = pd.DataFrame(cdr_records)
        cdr_df = cdr_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Cell towers DataFrame
        towers_df = pd.DataFrame([{
            'tower_id': t.tower_id,
            'latitude': t.latitude,
            'longitude': t.longitude,
            'radius_m': t.radius_m
        } for t in self.cell_towers])

        return cdr_df, towers_df

    def _find_nearest_tower(self, lat: float, lon: float) -> CellTower:
        """Find the nearest cell tower to a location."""
        min_dist = float('inf')
        nearest = self.cell_towers[0]

        for tower in self.cell_towers:
            dist = np.sqrt((tower.latitude - lat)**2 + (tower.longitude - lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = tower

        return nearest

    def _generate_handovers(self, trip: GroundTruthTrip) -> List[Dict]:
        """Generate cell tower handover events during a trip."""
        handovers = []
        last_tower = None

        # Sample every 30 seconds to detect tower changes
        sample_times = []
        current_time = trip.start_time
        while current_time <= trip.end_time:
            sample_times.append(current_time)
            current_time += timedelta(seconds=30)

        for sample_time in sample_times:
            gt_point = self._find_closest_point(trip, sample_time)
            if gt_point is None:
                continue

            current_tower = self._find_nearest_tower(
                gt_point.latitude, gt_point.longitude
            )

            if last_tower is not None and current_tower.tower_id != last_tower.tower_id:
                # Handover occurred
                handovers.append({
                    'user_id': f"USER_{trip.vehicle_id}",
                    'trip_id': trip.trip_id,
                    'timestamp': sample_time,
                    'cell_tower_id': current_tower.tower_id,
                    'cell_tower_lat': current_tower.latitude,
                    'cell_tower_lon': current_tower.longitude,
                    'event_type': 'handover',
                    'duration_seconds': None
                })

            last_tower = current_tower

        return handovers

    # ==================== ROAD NETWORK ====================

    def generate_road_network(
        self,
        generator: GroundTruthGenerator
    ) -> gpd.GeoDataFrame:
        """
        Generate a simplified road network from the route.

        In a real scenario, this would come from OSM.
        Here we create a simplified network for evaluation.

        Args:
            generator: Ground truth generator

        Returns:
            GeoDataFrame with road segments
        """
        from shapely.geometry import LineString

        # Create road segments from route
        route_coords = list(generator.route_line.coords)
        segments = []

        for i in range(len(route_coords) - 1):
            segment_line = LineString([route_coords[i], route_coords[i + 1]])

            # Determine road type based on position
            if i < 3 or i > len(route_coords) - 4:
                highway_type = 'secondary'
                lanes = 2
                maxspeed = 40
            else:
                highway_type = 'primary'
                lanes = 4
                maxspeed = 50

            segments.append({
                'edge_id': f"EDGE_{i+1:04d}",
                'from_node': f"NODE_{i+1:04d}",
                'to_node': f"NODE_{i+2:04d}",
                'name': f"Main Road Section {i+1}",
                'highway': highway_type,
                'lanes': lanes,
                'maxspeed': maxspeed,
                'oneway': False,
                'length_m': segment_line.length * 111000,  # Approximate
                'geometry': segment_line
            })

        return gpd.GeoDataFrame(segments, crs='EPSG:4326')

    # ==================== SAVE DATA ====================

    def save_all_sensor_data(
        self,
        output_dir: str,
        gps_data: pd.DataFrame,
        gtfs_data: Dict[str, pd.DataFrame],
        cdr_data: pd.DataFrame,
        towers_data: pd.DataFrame,
        road_network: gpd.GeoDataFrame
    ) -> Dict[str, str]:
        """
        Save all simulated sensor data to files.

        Args:
            output_dir: Output directory
            gps_data: GPS DataFrame
            gtfs_data: Dictionary of GTFS DataFrames
            cdr_data: CDR DataFrame
            towers_data: Cell towers DataFrame
            road_network: Road network GeoDataFrame

        Returns:
            Dictionary of output file paths
        """
        output_files = {}

        # GPS data
        gps_dir = os.path.join(output_dir, 'gps_traces')
        os.makedirs(gps_dir, exist_ok=True)
        gps_path = os.path.join(gps_dir, 'gps_traces.csv')
        gps_data.to_csv(gps_path, index=False)
        output_files['gps'] = gps_path

        # GTFS data
        gtfs_dir = os.path.join(output_dir, 'gtfs')
        os.makedirs(gtfs_dir, exist_ok=True)
        for name, df in gtfs_data.items():
            path = os.path.join(gtfs_dir, f'{name}.txt')
            df.to_csv(path, index=False)
            output_files[f'gtfs_{name}'] = path

        # CDR data
        cdr_dir = os.path.join(output_dir, 'cdr_events')
        os.makedirs(cdr_dir, exist_ok=True)
        cdr_path = os.path.join(cdr_dir, 'cdr_events.csv')
        cdr_data.to_csv(cdr_path, index=False)
        output_files['cdr'] = cdr_path

        towers_path = os.path.join(cdr_dir, 'cell_towers.csv')
        towers_data.to_csv(towers_path, index=False)
        output_files['towers'] = towers_path

        # Road network
        network_dir = os.path.join(output_dir, 'road_network')
        os.makedirs(network_dir, exist_ok=True)
        network_path = os.path.join(network_dir, 'road_network.geojson')
        road_network.to_file(network_path, driver='GeoJSON')
        output_files['road_network'] = network_path

        print(f"Sensor data saved to {output_dir}")
        print(f"  - GPS points: {len(gps_data)}")
        print(f"  - GTFS files: {len(gtfs_data)}")
        print(f"  - CDR events: {len(cdr_data)}")
        print(f"  - Cell towers: {len(towers_data)}")
        print(f"  - Road segments: {len(road_network)}")

        return output_files


def generate_all_synthetic_data(output_dir: str = None, seed: int = 42):
    """
    Generate complete synthetic dataset for fusion evaluation.

    Args:
        output_dir: Output directory (default: src/data_fusion/data/synthetic)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with all generated data
    """
    from .ground_truth_generator import GroundTruthGenerator

    if output_dir is None:
        output_dir = "src/data_fusion/data/synthetic"

    # Generate ground truth
    print("Generating ground truth...")
    generator = GroundTruthGenerator(seed=seed)
    trips = generator.generate_trips(
        num_trips=15,
        route_id="ROUTE_001",
        start_date=datetime(2025, 1, 15, 7, 0, 0),
        headway_minutes=12,
        num_vehicles=5
    )

    gt_dir = os.path.join(output_dir, 'ground_truth')
    generator.save_ground_truth(trips, gt_dir)

    # Simulate sensor data
    print("\nSimulating sensor data...")
    simulator = SensorSimulator(seed=seed)

    # GPS simulation
    print("  - Simulating GPS traces...")
    gps_data = simulator.simulate_gps(
        trips,
        sample_interval=5.0,
        noise_std_meters=8.0,
        dropout_rate=0.1
    )

    # GTFS simulation
    print("  - Simulating GTFS schedule...")
    gtfs_data = simulator.simulate_gtfs(
        generator, trips,
        schedule_noise_seconds=60.0
    )

    # CDR simulation
    print("  - Generating cell towers...")
    simulator.generate_cell_towers(generator, num_towers=15)

    print("  - Simulating CDR events...")
    cdr_data, towers_data = simulator.simulate_cdr(
        trips,
        events_per_hour=8.0
    )

    # Road network
    print("  - Generating road network...")
    road_network = simulator.generate_road_network(generator)

    # Save all data
    print("\nSaving data...")
    simulator.save_all_sensor_data(
        output_dir,
        gps_data,
        gtfs_data,
        cdr_data,
        towers_data,
        road_network
    )

    return {
        'ground_truth': trips,
        'generator': generator,
        'gps_data': gps_data,
        'gtfs_data': gtfs_data,
        'cdr_data': cdr_data,
        'towers_data': towers_data,
        'road_network': road_network
    }


if __name__ == "__main__":
    data = generate_all_synthetic_data()
    print("\n✓ Synthetic data generation complete!")
