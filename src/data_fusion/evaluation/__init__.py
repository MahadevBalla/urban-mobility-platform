"""
Evaluation Package

Contains tools for evaluating and comparing fusion algorithms:
- Metrics: RMSE, MAE, coverage calculations
- Comparator: Run and compare all fusion methods
- ReportGenerator: Generate comparison reports
"""

from .metrics import FusionMetrics
from .comparator import FusionComparator
from .report_generator import ReportGenerator

__all__ = [
    'FusionMetrics',
    'FusionComparator',
    'ReportGenerator',
]
