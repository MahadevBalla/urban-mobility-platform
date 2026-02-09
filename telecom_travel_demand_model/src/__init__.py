"""
Telecom-Based Travel Demand Estimation Model

A comprehensive framework for estimating travel demand using multi-source
telecom data (CDR, 4G LTE, 5G NR, XDR) with data fusion capabilities.

Based on methodology from:
- Toole et al. (2015) "The path most traveled: Travel demand estimation using big data resources"
- Alexander et al. (2015) "Validation of origin-destination trips"

Extended for Indian telecom context with 4G/5G network data integration.
"""

__version__ = "1.0.0"
__author__ = "Urban Transit Tool Team"

from src.utils.config import Config
from src.utils.logger import setup_logger

# Package-level logger
logger = setup_logger(__name__)
