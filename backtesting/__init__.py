"""
Institutional-grade backtesting module for cryptocurrency trading models.

This module provides tools for backtesting trading models with a focus on
eliminating bias, analyzing market regimes, and providing comprehensive metrics.
"""

from .institutional_backtester import InstitutionalBacktester

__all__ = ['InstitutionalBacktester'] 