"""
Spot Trading Module

This module provides functionality for spot trading,
including environments, training, and backtesting.
"""

from .trading_env import MultiCryptoEnv, CurriculumTradingWrapper
from .train import ICMFeatureExtractor

__all__ = ['MultiCryptoEnv', 'CurriculumTradingWrapper', 'ICMFeatureExtractor'] 