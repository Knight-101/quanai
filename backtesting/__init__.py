"""
Institutional Backtesting Package

This package provides robust backtesting capabilities for crypto trading models,
specifically designed to be compatible with the existing codebase.
"""

from .institutional_backtester import InstitutionalBacktester
from .regime_analyzer import RegimeAnalyzer, MarketRegime

__all__ = ['InstitutionalBacktester', 'RegimeAnalyzer', 'MarketRegime'] 