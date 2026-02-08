"""
Telecom data loader for CDR, XDR, and network data.

Handles loading and basic validation of various telecom data formats
commonly used in Indian telecom operators.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TelecomDataLoader:
    """
    Loader for telecom data from various sources.

    Supports:
    - CDR (Call Detail Records)
    - XDR (Extended Data Records)
    - 4G LTE network data
    - 5G NR network data

    Example:
        >>> loader = TelecomDataLoader()
        >>> cdr_df = loader.load_cdr("data/raw/cdr_data.csv")
        >>> xdr_df = loader.load_xdr("data/raw/xdr_data.csv")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the telecom data loader.

        Args:
            config: Configuration object. Uses global config if not provided.
        """
        self.config = config or get_config()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that required configuration exists."""
        required_keys = ["data_sources.cdr", "data_sources.xdr"]
        for key in required_keys:
            if self.config.get(key) is None:
                logger.warning(f"Configuration key '{key}' not found")

    def load_cdr(
        self,
        path: Union[str, Path],
        parse_dates: bool = True,
        sample_fraction: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load Call Detail Records (CDR) data.

        Args:
            path: Path to CDR CSV file or directory containing CDR files.
            parse_dates: Whether to parse timestamp columns.
            sample_fraction: Optional fraction of data to sample (0-1).

        Returns:
            DataFrame with CDR data.
        """
        logger.info(f"Loading CDR data from {path}")
        path = Path(path)

        if path.is_dir():
            # Load all matching files
            pattern = self.config.get("data_sources.cdr.file_pattern", "cdr_*.csv")
            files = list(path.glob(pattern))
            logger.info(f"Found {len(files)} CDR files matching pattern '{pattern}'")

            dfs = []
            for f in files:
                df = self._load_single_cdr(f, parse_dates)
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
        else:
            df = self._load_single_cdr(path, parse_dates)

        # Optional sampling
        if sample_fraction is not None and 0 < sample_fraction < 1:
            df = df.sample(frac=sample_fraction, random_state=42)
            logger.info(f"Sampled {len(df)} records ({sample_fraction*100:.1f}%)")

        logger.info(f"Loaded {len(df)} CDR records")
        return df

    def _load_single_cdr(self, path: Path, parse_dates: bool) -> pd.DataFrame:
        """Load a single CDR file."""
        # Define column mapping for standardization
        column_mapping = {
            'RECORD_ID': 'record_id',
            'IMSI': 'imsi',
            'MSISDN': 'msisdn',
            'IMEI': 'imei',
            'CALL_TYPE': 'call_type',
            'START_TIME': 'timestamp',
            'END_TIME': 'end_time',
            'DURATION_SEC': 'duration',
            'CELL_ID': 'cell_id',
            'LAC': 'lac',
            'TAC': 'tac',
            'SERVING_NETWORK': 'plmn',
            'ROAMING_FLAG': 'roaming',
            'TERMINATION_CAUSE': 'termination_cause',
            'CHARGE_AMOUNT': 'charge',
            'SERVICE_TYPE': 'service_type'
        }

        date_cols = ['START_TIME', 'END_TIME'] if parse_dates else None

        df = pd.read_csv(
            path,
            parse_dates=date_cols,
            dtype={
                'IMSI': str,
                'MSISDN': str,
                'IMEI': str,
                'CELL_ID': str,
                'LAC': str,
                'TAC': str
            }
        )

        # Standardize column names
        df = df.rename(columns=column_mapping)

        # Validate required columns
        required = self.config.get("data_sources.cdr.required_columns", [])
        required_std = [column_mapping.get(c, c.lower()) for c in required]

        missing = set(required_std) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required CDR columns: {missing}")

        return df

    def load_xdr(
        self,
        path: Union[str, Path],
        parse_dates: bool = True,
        sample_fraction: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load Extended Data Records (XDR) data.

        XDR data is particularly valuable as it often contains actual
        geographic coordinates from GPS or enhanced cell positioning.

        Args:
            path: Path to XDR CSV file or directory.
            parse_dates: Whether to parse timestamp columns.
            sample_fraction: Optional fraction of data to sample.

        Returns:
            DataFrame with XDR data.
        """
        logger.info(f"Loading XDR data from {path}")
        path = Path(path)

        if path.is_dir():
            pattern = self.config.get("data_sources.xdr.file_pattern", "xdr_*.csv")
            files = list(path.glob(pattern))
            logger.info(f"Found {len(files)} XDR files")

            dfs = []
            for f in files:
                df = self._load_single_xdr(f, parse_dates)
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
        else:
            df = self._load_single_xdr(path, parse_dates)

        if sample_fraction is not None and 0 < sample_fraction < 1:
            df = df.sample(frac=sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} XDR records")
        return df

    def _load_single_xdr(self, path: Path, parse_dates: bool) -> pd.DataFrame:
        """Load a single XDR file."""
        column_mapping = {
            'RECORD_TYPE': 'record_type',
            'TIMESTAMP': 'timestamp',
            'IMSI': 'imsi',
            'IMEI': 'imei',
            'MSISDN': 'msisdn',
            'EVENT_TYPE': 'event_type',
            'RAT_TYPE': 'rat_type',
            'CELL_ID': 'cell_id',
            'ENODEB_ID': 'enodeb_id',
            'GNODEB_ID': 'gnodeb_id',
            'TAC': 'tac',
            'LAC': 'lac',
            'SESSION_ID': 'session_id',
            'BYTES_UPLINK': 'bytes_ul',
            'BYTES_DOWNLINK': 'bytes_dl',
            'LOCATION_LAT': 'latitude',
            'LOCATION_LON': 'longitude',
            'APPLICATION_ID': 'app_id'
        }

        date_cols = ['TIMESTAMP'] if parse_dates else None

        df = pd.read_csv(
            path,
            parse_dates=date_cols,
            dtype={
                'IMSI': str,
                'MSISDN': str,
                'IMEI': str,
                'CELL_ID': str,
                'TAC': str,
                'LAC': str
            }
        )

        df = df.rename(columns=column_mapping)
        return df

    def load_network_4g(
        self,
        path: Union[str, Path],
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load 4G LTE network performance data.

        This data provides cell-level metrics including signal quality
        which can be used for location confidence weighting.

        Args:
            path: Path to 4G network data file.
            parse_dates: Whether to parse timestamp columns.

        Returns:
            DataFrame with 4G network data.
        """
        logger.info(f"Loading 4G network data from {path}")

        column_mapping = {
            'TIMESTAMP': 'timestamp',
            'ENODEB_ID': 'enodeb_id',
            'CELL_ID': 'cell_id',
            'EARFCN': 'earfcn',
            'PCI': 'pci',
            'TAC': 'tac',
            'PLMN': 'plmn',
            'IMSI': 'imsi',
            'IMEI': 'imei',
            'RSRP_DBM': 'rsrp',
            'RSRQ_DB': 'rsrq',
            'SINR_DB': 'sinr',
            'CQI': 'cqi',
            'THROUGHPUT_DL_MBPS': 'throughput_dl',
            'THROUGHPUT_UL_MBPS': 'throughput_ul',
            'ACTIVE_USERS': 'active_users',
            'LATENCY_MS': 'latency'
        }

        date_cols = ['TIMESTAMP'] if parse_dates else None

        df = pd.read_csv(
            path,
            parse_dates=date_cols,
            dtype={'IMSI': str, 'IMEI': str, 'CELL_ID': str, 'TAC': str}
        )

        df = df.rename(columns=column_mapping)
        logger.info(f"Loaded {len(df)} 4G network records")
        return df

    def load_network_5g(
        self,
        path: Union[str, Path],
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load 5G NR network performance data.

        Args:
            path: Path to 5G network data file.
            parse_dates: Whether to parse timestamp columns.

        Returns:
            DataFrame with 5G network data.
        """
        logger.info(f"Loading 5G network data from {path}")

        column_mapping = {
            'TIMESTAMP': 'timestamp',
            'GNODEB_ID': 'gnodeb_id',
            'NCI': 'nci',
            'NR_ARFCN': 'nr_arfcn',
            'SSB_FREQUENCY_MHZ': 'frequency',
            'PCI': 'pci',
            'TAC': 'tac',
            'PLMN': 'plmn',
            'IMSI': 'imsi',
            'IMEI': 'imei',
            'SS_RSRP_DBM': 'rsrp',
            'SS_RSRQ_DB': 'rsrq',
            'SS_SINR_DB': 'sinr',
            'CQI': 'cqi',
            'THROUGHPUT_DL_GBPS': 'throughput_dl',
            'THROUGHPUT_UL_GBPS': 'throughput_ul',
            'ACTIVE_USERS': 'active_users',
            'LATENCY_MS': 'latency',
            'NETWORK_SLICE_ID': 'slice_id',
            'BEAM_ID': 'beam_id',
            'NR_BAND': 'band',
            'BANDWIDTH_MHZ': 'bandwidth'
        }

        date_cols = ['TIMESTAMP'] if parse_dates else None

        df = pd.read_csv(
            path,
            parse_dates=date_cols,
            dtype={'IMSI': str, 'IMEI': str, 'NCI': str, 'TAC': str}
        )

        df = df.rename(columns=column_mapping)
        logger.info(f"Loaded {len(df)} 5G network records")
        return df

    def load_all(
        self,
        data_dir: Union[str, Path],
        sample_fraction: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available data sources from a directory.

        Args:
            data_dir: Directory containing data files.
            sample_fraction: Optional fraction to sample.

        Returns:
            Dictionary mapping source name to DataFrame.
        """
        data_dir = Path(data_dir)
        result = {}

        # CDR
        cdr_files = list(data_dir.glob("cdr*.csv"))
        if cdr_files:
            result['cdr'] = self.load_cdr(cdr_files[0], sample_fraction=sample_fraction)

        # XDR
        xdr_files = list(data_dir.glob("xdr*.csv"))
        if xdr_files:
            result['xdr'] = self.load_xdr(xdr_files[0], sample_fraction=sample_fraction)

        # 4G
        network_4g_files = list(data_dir.glob("4g*.csv"))
        if network_4g_files:
            result['network_4g'] = self.load_network_4g(network_4g_files[0])

        # 5G
        network_5g_files = list(data_dir.glob("5g*.csv"))
        if network_5g_files:
            result['network_5g'] = self.load_network_5g(network_5g_files[0])

        logger.info(f"Loaded data sources: {list(result.keys())}")
        return result
