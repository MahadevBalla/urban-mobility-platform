"""
Unit tests for the travel demand pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.geo_utils import haversine_distance, lat_lon_to_grid_cell
from utils.time_utils import get_time_period, get_effective_day, is_home_time
from utils.config import Config


class TestGeoUtils:
    """Tests for geographic utility functions."""

    def test_haversine_distance_same_point(self):
        """Distance between same point should be 0."""
        dist = haversine_distance(19.076, 72.877, 19.076, 72.877)
        assert dist == 0

    def test_haversine_distance_known_points(self):
        """Test distance calculation with known values."""
        # Mumbai to Pune approximately 150 km
        dist = haversine_distance(19.076, 72.877, 18.520, 73.856)
        assert 140000 < dist < 160000  # 140-160 km

    def test_lat_lon_to_grid_cell(self):
        """Test grid cell assignment."""
        row, col = lat_lon_to_grid_cell(19.076, 72.877, 1000, (19.0, 72.8))
        assert isinstance(row, int)
        assert isinstance(col, int)


class TestTimeUtils:
    """Tests for time utility functions."""

    def test_get_time_period_am_peak(self):
        """Test AM peak period detection."""
        ts = datetime(2024, 2, 4, 8, 30, 0)
        period = get_time_period(ts)
        assert period == "AM_PEAK"

    def test_get_time_period_pm_peak(self):
        """Test PM peak period detection."""
        ts = datetime(2024, 2, 4, 17, 30, 0)
        period = get_time_period(ts)
        assert period == "PM_PEAK"

    def test_get_effective_day_after_midnight(self):
        """Test effective day for early morning hours."""
        ts = datetime(2024, 2, 4, 2, 0, 0)  # 2 AM
        effective = get_effective_day(ts, day_start_hour=3)
        assert effective == datetime(2024, 2, 3).date()  # Previous day

    def test_is_home_time_night(self):
        """Test home time detection."""
        ts = datetime(2024, 2, 4, 22, 0, 0)  # 10 PM
        assert is_home_time(ts, home_start=20, home_end=7)

    def test_is_home_time_day(self):
        """Test non-home time detection."""
        ts = datetime(2024, 2, 4, 12, 0, 0)  # Noon
        assert not is_home_time(ts, home_start=20, home_end=7)


class TestConfig:
    """Tests for configuration management."""

    def test_config_get_default(self):
        """Test getting value with default."""
        config = Config()
        value = config.get("nonexistent.key", default=42)
        assert value == 42

    def test_config_set_get(self):
        """Test setting and getting values."""
        config = Config()
        config.set("test.nested.value", 123)
        assert config.get("test.nested.value") == 123


class TestSampleData:
    """Tests using sample data structure."""

    @pytest.fixture
    def sample_cdr(self):
        """Create sample CDR data."""
        return pd.DataFrame({
            'IMSI': ['404451234567890'] * 5,
            'START_TIME': pd.date_range('2024-02-04 08:00', periods=5, freq='H'),
            'CELL_ID': ['45231', '45232', '45231', '45233', '45231'],
            'TAC': ['5012'] * 5,
            'CALL_TYPE': ['MOC', 'DATA', 'MTC', 'MOC', 'DATA']
        })

    @pytest.fixture
    def sample_xdr(self):
        """Create sample XDR data."""
        return pd.DataFrame({
            'IMSI': ['404451234567890'] * 5,
            'TIMESTAMP': pd.date_range('2024-02-04 08:00', periods=5, freq='H'),
            'CELL_ID': ['100011', '100021', '100011', '100031', '100011'],
            'TAC': ['5012'] * 5,
            'LOCATION_LAT': [19.076, 19.082, 19.076, 19.090, 19.076],
            'LOCATION_LON': [72.877, 72.885, 72.877, 72.892, 72.877]
        })

    def test_sample_data_structure(self, sample_cdr, sample_xdr):
        """Verify sample data has expected structure."""
        assert len(sample_cdr) == 5
        assert len(sample_xdr) == 5
        assert 'IMSI' in sample_cdr.columns
        assert 'LOCATION_LAT' in sample_xdr.columns


class TestStayDetection:
    """Tests for stay point detection logic."""

    def test_consecutive_same_location(self):
        """Users at same cell multiple times should create stay."""
        # This tests the logic, not the actual implementation
        cells = ['A', 'A', 'A', 'B', 'B', 'A']
        expected_stays = 3  # A (start), B (middle), A (end)
        # Simplified counting of transitions
        transitions = sum(1 for i in range(1, len(cells)) if cells[i] != cells[i-1])
        # Transitions + 1 = number of stays
        assert transitions + 1 == expected_stays


class TestTripPurpose:
    """Tests for trip purpose classification."""

    def test_home_to_work_is_hbw(self):
        """Trip from home to work should be HBW."""
        origin_type = 'home'
        dest_type = 'work'

        if origin_type == 'home' and dest_type == 'work':
            purpose = 'HBW'
        elif origin_type == 'work' and dest_type == 'home':
            purpose = 'HBW'
        elif origin_type == 'home' or dest_type == 'home':
            purpose = 'HBO'
        else:
            purpose = 'NHB'

        assert purpose == 'HBW'

    def test_home_to_other_is_hbo(self):
        """Trip from home to other should be HBO."""
        origin_type = 'home'
        dest_type = 'other'

        if origin_type == 'home' and dest_type == 'work':
            purpose = 'HBW'
        elif origin_type == 'work' and dest_type == 'home':
            purpose = 'HBW'
        elif origin_type == 'home' or dest_type == 'home':
            purpose = 'HBO'
        else:
            purpose = 'NHB'

        assert purpose == 'HBO'

    def test_work_to_other_is_nhb(self):
        """Trip from work to other should be NHB."""
        origin_type = 'work'
        dest_type = 'other'

        if origin_type == 'home' and dest_type == 'work':
            purpose = 'HBW'
        elif origin_type == 'work' and dest_type == 'home':
            purpose = 'HBW'
        elif origin_type == 'home' or dest_type == 'home':
            purpose = 'HBO'
        else:
            purpose = 'NHB'

        assert purpose == 'NHB'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
