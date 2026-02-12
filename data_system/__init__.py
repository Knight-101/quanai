"""
Data System Module

This module handles all data fetching and feature engineering operations
for the quantitative trading system.
"""

from .derivative_data_fetcher import PerpetualDataFetcher
from .data_manager import DataManager
from .drive_adapter import DriveAdapter

try:
    from .feature_engine import DerivativesFeatureEngine
except Exception:
    DerivativesFeatureEngine = None

__all__ = ['PerpetualDataFetcher', 'DerivativesFeatureEngine', 'DataManager', 'DriveAdapter']
