"""
Optimized Backtesting Module

This module provides optimized backtesting capabilities for reinforcement learning trading models,
with special focus on observation space compatibility between training and backtesting.

Key Features:
- Observation space adaptation for model compatibility
- Market regime analysis
- Comprehensive performance metrics
- Visualization capabilities
"""

from .institutional_backtester import InstitutionalBacktester, run_institutional_backtest
from .regime_analyzer import RegimeAnalyzer, MarketRegime, RegimePeriod
from .metrics import calculate_metrics, calculate_returns, calculate_regime_metrics, calculate_drawdowns
from .visualization import create_performance_charts, create_regime_comparison_chart, create_regime_transition_chart

__all__ = [
    # Main backtest classes
    "InstitutionalBacktester",
    "run_institutional_backtest",
    
    # Regime analysis
    "RegimeAnalyzer",
    "MarketRegime",
    "RegimePeriod",
    
    # Metrics
    "calculate_metrics",
    "calculate_returns",
    "calculate_regime_metrics",
    "calculate_drawdowns",
    
    # Visualization
    "create_performance_charts",
    "create_regime_comparison_chart",
    "create_regime_transition_chart"
] 