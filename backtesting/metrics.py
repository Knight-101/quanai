"""
Performance Metrics Module

This module provides a comprehensive set of metrics for evaluating
trading strategy performance in backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    Calculate returns from a series of portfolio values.
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Array of returns
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return np.array([])
        
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate annualized Sharpe ratio
    sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
    target_return: float = 0.0
) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        target_return: Minimum acceptable return
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate downside deviation (standard deviation of negative returns only)
    downside_returns = excess_returns[excess_returns < target_return]
    
    if len(downside_returns) == 0:
        # If no downside returns, use a small value to avoid division by zero
        downside_deviation = 1e-10
    else:
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
        
    # Calculate annualized Sortino ratio
    sortino = np.mean(excess_returns) / (downside_deviation + 1e-10) * np.sqrt(periods_per_year)
    
    return sortino


def calculate_drawdowns(portfolio_values: List[float]) -> Tuple[List[float], float, int]:
    """
    Calculate drawdowns from a series of portfolio values.
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Tuple containing (drawdown series, maximum drawdown, maximum drawdown duration)
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return [], 0.0, 0
        
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown in percent
    drawdown = (values - running_max) / running_max
    
    # Find maximum drawdown and its duration
    max_drawdown = np.min(drawdown)
    
    # Find drawdown duration
    max_duration = 0
    current_duration = 0
    in_drawdown = False
    
    for i in range(1, len(values)):
        if values[i] < running_max[i]:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
                current_duration = 0
            current_duration += 1
        else:
            # Not in drawdown
            if in_drawdown:
                in_drawdown = False
                max_duration = max(max_duration, current_duration)
                current_duration = 0
    
    # In case we are still in a drawdown
    if in_drawdown:
        max_duration = max(max_duration, current_duration)
    
    return drawdown.tolist(), max_drawdown, max_duration


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate from a list of trades.
    
    Args:
        trades: List of trade dictionaries with PnL information
        
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
        
    # Count winning trades
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    
    # Calculate win rate
    win_rate = winning_trades / len(trades)
    
    return win_rate


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate profit factor from a list of trades.
    
    Args:
        trades: List of trade dictionaries with PnL information
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
        
    # Sum up winning and losing trades
    gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    # Calculate profit factor
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return profit_factor


def calculate_expectancy(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate expectancy from a list of trades.
    
    Args:
        trades: List of trade dictionaries with PnL information
        
    Returns:
        Expectancy (average profit/loss per trade)
    """
    if not trades:
        return 0.0
        
    # Calculate total PnL
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    
    # Calculate expectancy
    expectancy = total_pnl / len(trades)
    
    return expectancy


def calculate_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    window_size: int = None
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level for VaR
        window_size: Size of rolling window, or None for full history
        
    Returns:
        Value at Risk at the specified confidence level
    """
    if len(returns) == 0:
        return 0.0
        
    if window_size is not None and window_size < len(returns):
        # Use the most recent window
        returns = returns[-window_size:]
        
    # Calculate VaR
    var = -np.percentile(returns, 100 * (1 - confidence_level))
    
    return var


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    window_size: int = None
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level for Expected Shortfall
        window_size: Size of rolling window, or None for full history
        
    Returns:
        Expected Shortfall at the specified confidence level
    """
    if len(returns) == 0:
        return 0.0
        
    if window_size is not None and window_size < len(returns):
        # Use the most recent window
        returns = returns[-window_size:]
        
    # Calculate VaR
    var = calculate_value_at_risk(returns, confidence_level)
    
    # Calculate Expected Shortfall
    if var != 0:
        es = -np.mean(returns[returns <= -var])
    else:
        es = 0.0
        
    return es


def calculate_calmar_ratio(
    returns: np.ndarray,
    portfolio_values: List[float],
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Array of returns
        portfolio_values: List of portfolio values over time
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0 or len(portfolio_values) < 2:
        return 0.0
        
    # Calculate annualized return
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calculate maximum drawdown
    _, max_drawdown, _ = calculate_drawdowns(portfolio_values)
    
    # Calculate Calmar ratio
    if abs(max_drawdown) < 1e-10:
        # If no drawdown, use a small value to avoid division by zero
        calmar = annualized_return / 1e-10
    else:
        calmar = annualized_return / abs(max_drawdown)
        
    return calmar


def calculate_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate the Omega ratio.
    
    Args:
        returns: Array of returns
        threshold: Return threshold
        periods_per_year: Number of periods in a year
        
    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return 0.0
        
    # Adjust threshold to the same frequency as returns
    adjusted_threshold = threshold / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - adjusted_threshold
    
    # Calculate positive and negative sums
    positive_sum = np.sum(excess_returns[excess_returns > 0])
    negative_sum = np.abs(np.sum(excess_returns[excess_returns < 0]))
    
    # Calculate Omega ratio
    if negative_sum < 1e-10:
        # If no negative excess returns, use a small value to avoid division by zero
        omega = positive_sum / 1e-10
    else:
        omega = positive_sum / negative_sum
        
    return omega


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate return distribution kurtosis.
    
    Args:
        returns: Array of returns
        
    Returns:
        Kurtosis
    """
    if len(returns) < 4:  # Need at least 4 data points for kurtosis
        return 0.0
        
    return stats.kurtosis(returns)


def calculate_skew(returns: np.ndarray) -> float:
    """
    Calculate return distribution skewness.
    
    Args:
        returns: Array of returns
        
    Returns:
        Skewness
    """
    if len(returns) < 3:  # Need at least 3 data points for skewness
        return 0.0
        
    return stats.skew(returns)


def calculate_average_leverage(leverages: List[float]) -> float:
    """
    Calculate average leverage.
    
    Args:
        leverages: List of leverage values over time
        
    Returns:
        Average leverage
    """
    if not leverages:
        return 0.0
        
    return np.mean(leverages)


def calculate_max_leverage(leverages: List[float]) -> float:
    """
    Calculate maximum leverage.
    
    Args:
        leverages: List of leverage values over time
        
    Returns:
        Maximum leverage
    """
    if not leverages:
        return 0.0
        
    return np.max(leverages)


def calculate_volatility(
    returns: np.ndarray,
    periods_per_year: int = 365
) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
        
    # Calculate annualized volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    return volatility


def calculate_downside_volatility(
    returns: np.ndarray,
    threshold: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate downside volatility.
    
    Args:
        returns: Array of returns
        threshold: Return threshold
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized downside volatility
    """
    if len(returns) == 0:
        return 0.0
        
    # Adjust threshold to the same frequency as returns
    adjusted_threshold = threshold / periods_per_year
    
    # Calculate downside volatility
    downside_returns = returns[returns < adjusted_threshold]
    
    if len(downside_returns) == 0:
        return 0.0
        
    downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year)
    
    return downside_volatility


def calculate_best_worst_periods(
    returns: np.ndarray,
    periods: List[int] = [1, 5, 20, 60]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate best and worst returns over different periods.
    
    Args:
        returns: Array of returns
        periods: List of period lengths to analyze
        
    Returns:
        Dictionary with best and worst returns for each period
    """
    if len(returns) == 0:
        return {
            'best': {period: 0.0 for period in periods},
            'worst': {period: 0.0 for period in periods}
        }
        
    result = {
        'best': {},
        'worst': {}
    }
    
    # Calculate rolling returns for each period
    for period in periods:
        if period > len(returns):
            # Skip periods longer than available data
            result['best'][period] = 0.0
            result['worst'][period] = 0.0
            continue
            
        # Calculate rolling returns
        rolling_returns = np.array([
            np.prod(1 + returns[i:i+period]) - 1
            for i in range(len(returns) - period + 1)
        ])
        
        # Find best and worst returns
        result['best'][period] = np.max(rolling_returns)
        result['worst'][period] = np.min(rolling_returns)
        
    return result


def calculate_recovery_time(
    portfolio_values: List[float],
    timestamps: Optional[List] = None
) -> int:
    """
    Calculate average recovery time from drawdowns.
    
    Args:
        portfolio_values: List of portfolio values over time
        timestamps: List of timestamps corresponding to portfolio values
        
    Returns:
        Average recovery time in periods (or days if timestamps provided)
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return 0
        
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate drawdowns and peaks
    running_max = np.maximum.accumulate(values)
    is_peak = np.append([True], running_max[1:] > running_max[:-1])
    peak_indices = np.where(is_peak)[0]
    
    # If no peaks, return 0
    if len(peak_indices) == 0:
        return 0
        
    recovery_times = []
    
    for peak_idx in peak_indices:
        peak_value = values[peak_idx]
        
        # Find next time value exceeds peak
        recovery_idx = None
        for i in range(peak_idx + 1, len(values)):
            if values[i] >= peak_value:
                recovery_idx = i
                break
                
        if recovery_idx is not None:
            if timestamps is not None:
                # Calculate recovery time in days
                recovery_time = (timestamps[recovery_idx] - timestamps[peak_idx]).days
            else:
                # Calculate recovery time in periods
                recovery_time = recovery_idx - peak_idx
                
            recovery_times.append(recovery_time)
    
    # Calculate average recovery time
    if recovery_times:
        avg_recovery_time = np.mean(recovery_times)
    else:
        avg_recovery_time = 0
        
    return avg_recovery_time


def calculate_maximum_consecutive_wins_losses(trades: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate maximum consecutive wins and losses.
    
    Args:
        trades: List of trade dictionaries with PnL information
        
    Returns:
        Dictionary with maximum consecutive wins and losses
    """
    if not trades:
        return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for trade in trades:
        pnl = trade.get('pnl', 0)
        
        if pnl > 0:
            # Winning trade
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        elif pnl < 0:
            # Losing trade
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
    return {
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }


def calculate_profit_loss_ratio(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate average profit/loss ratio.
    
    Args:
        trades: List of trade dictionaries with PnL information
        
    Returns:
        Average profit/loss ratio
    """
    if not trades:
        return 0.0
        
    # Get winning and losing trades
    winning_trades = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0]
    losing_trades = [abs(trade.get('pnl', 0)) for trade in trades if trade.get('pnl', 0) < 0]
    
    # Calculate average profit and loss
    avg_profit = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else float('inf')
    
    # Calculate profit/loss ratio
    if avg_loss == 0:
        return float('inf') if avg_profit > 0 else 0.0
        
    profit_loss_ratio = avg_profit / avg_loss
    
    return profit_loss_ratio


def calculate_trade_statistics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive trade statistics.
    
    Args:
        trades: List of trade dictionaries with PnL and other information
        
    Returns:
        Dictionary with trade statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_trade_pnl': 0.0,
            'avg_win_pnl': 0.0,
            'avg_loss_pnl': 0.0,
            'max_win_pnl': 0.0,
            'max_loss_pnl': 0.0,
            'profit_loss_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_period': 0.0,
            'max_holding_period': 0,
            'min_holding_period': 0,
            'avg_leverage': 0.0,
            'max_leverage': 0.0
        }
        
    # Calculate basic trade statistics
    total_trades = len(trades)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    expectancy = calculate_expectancy(trades)
    
    # Calculate average trade PnL
    avg_trade_pnl = np.mean([trade.get('pnl', 0) for trade in trades])
    
    # Calculate average winning and losing trade PnL
    winning_trades = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0]
    losing_trades = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0]
    
    avg_win_pnl = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss_pnl = np.mean(losing_trades) if losing_trades else 0.0
    
    # Calculate maximum winning and losing trade PnL
    max_win_pnl = np.max(winning_trades) if winning_trades else 0.0
    max_loss_pnl = np.min(losing_trades) if losing_trades else 0.0
    
    # Calculate profit/loss ratio
    profit_loss_ratio = calculate_profit_loss_ratio(trades)
    
    # Calculate maximum consecutive wins and losses
    consecutive_stats = calculate_maximum_consecutive_wins_losses(trades)
    
    # Calculate holding period statistics
    holding_periods = [trade.get('holding_period', 0) for trade in trades]
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
    max_holding_period = np.max(holding_periods) if holding_periods else 0
    min_holding_period = np.min(holding_periods) if holding_periods else 0
    
    # Calculate leverage statistics
    leverages = [trade.get('leverage', 0) for trade in trades]
    
    avg_leverage = np.mean(leverages) if leverages else 0.0
    max_leverage = np.max(leverages) if leverages else 0.0
    
    # Compile all statistics
    stats = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_trade_pnl': avg_trade_pnl,
        'avg_win_pnl': avg_win_pnl,
        'avg_loss_pnl': avg_loss_pnl,
        'max_win_pnl': max_win_pnl,
        'max_loss_pnl': max_loss_pnl,
        'profit_loss_ratio': profit_loss_ratio,
        'max_consecutive_wins': consecutive_stats['max_consecutive_wins'],
        'max_consecutive_losses': consecutive_stats['max_consecutive_losses'],
        'avg_holding_period': avg_holding_period,
        'max_holding_period': max_holding_period,
        'min_holding_period': min_holding_period,
        'avg_leverage': avg_leverage,
        'max_leverage': max_leverage
    }
    
    return stats


def calculate_performance_metrics(
    portfolio_values: List[float],
    timestamps: List = None,
    trades: List[Dict[str, Any]] = None,
    leverages: List[float] = None,
    benchmark_returns: np.ndarray = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        portfolio_values: List of portfolio values over time
        timestamps: List of timestamps corresponding to portfolio values
        trades: List of trade dictionaries with PnL and other information
        leverages: List of leverage values over time
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary with performance metrics
    """
    if not portfolio_values or len(portfolio_values) < 2:
        logger.warning("Insufficient portfolio values for metric calculation")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'calmar_ratio': 0.0,
            'volatility': 0.0,
            'downside_volatility': 0.0,
            'value_at_risk': 0.0,
            'expected_shortfall': 0.0,
            'omega_ratio': 0.0,
            'kurtosis': 0.0,
            'skew': 0.0,
            'recovery_time': 0,
            'avg_leverage': 0.0,
            'max_leverage': 0.0
        }
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Calculate total return
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    
    # Calculate annualized return
    period_count = len(portfolio_values) - 1
    if timestamps:
        # Calculate years between first and last timestamp
        time_diff = timestamps[-1] - timestamps[0]
        years = time_diff.total_seconds() / (365.25 * 24 * 60 * 60)
        annualized_return = (1 + total_return) ** (1 / max(years, 1e-10)) - 1
    else:
        # Calculate using periods per year
        annualized_return = (1 + total_return) ** (periods_per_year / max(period_count, 1e-10)) - 1
    
    # Calculate volatility metrics
    volatility = calculate_volatility(returns, periods_per_year)
    downside_volatility = calculate_downside_volatility(returns, 0.0, periods_per_year)
    
    # Calculate risk-adjusted return metrics
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    # Calculate drawdown metrics
    drawdowns, max_drawdown, max_drawdown_duration = calculate_drawdowns(portfolio_values)
    calmar_ratio = calculate_calmar_ratio(returns, portfolio_values, periods_per_year)
    recovery_time = calculate_recovery_time(portfolio_values, timestamps)
    
    # Calculate distribution metrics
    value_at_risk = calculate_value_at_risk(returns, 0.95)
    expected_shortfall = calculate_expected_shortfall(returns, 0.95)
    omega_ratio = calculate_omega_ratio(returns, risk_free_rate / periods_per_year, periods_per_year)
    kurtosis = calculate_kurtosis(returns)
    skew = calculate_skew(returns)
    
    # Calculate leverage metrics
    avg_leverage = calculate_average_leverage(leverages) if leverages else 0.0
    max_leverage = calculate_max_leverage(leverages) if leverages else 0.0
    
    # Calculate trade statistics if available
    trade_stats = calculate_trade_statistics(trades) if trades else {}
    
    # Calculate benchmark comparison metrics if benchmark returns available
    benchmark_metrics = {}
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Calculate alpha and beta
        if len(returns) > 0 and len(benchmark_returns) >= len(returns):
            benchmark_returns_aligned = benchmark_returns[-len(returns):]
            
            # Calculate beta (covariance / variance)
            covariance = np.cov(returns, benchmark_returns_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_returns_aligned)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = 0.0
                
            # Calculate alpha (annualized)
            benchmark_mean = np.mean(benchmark_returns_aligned)
            alpha = np.mean(returns) - (beta * benchmark_mean)
            alpha_annualized = alpha * periods_per_year
            
            # Calculate information ratio
            tracking_error = np.std(returns - benchmark_returns_aligned)
            information_ratio = (np.mean(returns) - benchmark_mean) / (tracking_error + 1e-10) * np.sqrt(periods_per_year)
            
            benchmark_metrics = {
                'alpha': alpha_annualized,
                'beta': beta,
                'tracking_error': tracking_error * np.sqrt(periods_per_year),
                'information_ratio': information_ratio
            }
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'calmar_ratio': calmar_ratio,
        'volatility': volatility,
        'downside_volatility': downside_volatility,
        'value_at_risk': value_at_risk,
        'expected_shortfall': expected_shortfall,
        'omega_ratio': omega_ratio,
        'kurtosis': kurtosis,
        'skew': skew,
        'recovery_time': recovery_time,
        'avg_leverage': avg_leverage,
        'max_leverage': max_leverage,
        **trade_stats,
        **benchmark_metrics
    }
    
    return metrics 