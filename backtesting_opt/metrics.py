import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_returns(portfolio_values: List[float]) -> List[float]:
    """
    Calculate returns from a list of portfolio values.
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        List of returns between consecutive portfolio values
    """
    returns = []
    
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] == 0:
            returns.append(0.0)
        else:
            returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
            
    return returns

def calculate_metrics(
    portfolio_values: List[float],
    returns: List[float],
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.02,
    trades: List[Dict] = None,
    timestamps: List = None,
    leverages: List[float] = None
) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        portfolio_values: List of portfolio values over time
        returns: List of returns between consecutive portfolio values
        initial_capital: Initial capital amount
        risk_free_rate: Annual risk-free rate (as decimal)
        trades: List of trade dictionaries
        timestamps: List of timestamps for portfolio values
        leverages: List of leverage values over time
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic portfolio metrics
    if len(portfolio_values) > 1:
        # Total return
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        metrics['total_return'] = total_return
        
        # Calculate time period
        if timestamps and len(timestamps) >= 2:
            try:
                if isinstance(timestamps[0], str):
                    start_date = pd.to_datetime(timestamps[0])
                    end_date = pd.to_datetime(timestamps[-1])
                else:
                    start_date = timestamps[0]
                    end_date = timestamps[-1]
                    
                # Calculate trading days and years
                if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
                    days = (end_date - start_date).days
                    years = days / 365.0
                    trading_days = len(portfolio_values)
                    
                    metrics['days'] = days
                    metrics['years'] = years
                    metrics['trading_days'] = trading_days
                    
                    # Annualized return
                    if years > 0:
                        metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1
                    else:
                        metrics['annualized_return'] = total_return
            except Exception as e:
                logger.error(f"Error calculating time-based metrics: {str(e)}")
                metrics['annualized_return'] = total_return
        else:
            # If no timestamps, use number of data points as proxy for days
            trading_days = len(portfolio_values)
            years = trading_days / 252.0  # Approximation
            
            metrics['trading_days'] = trading_days
            metrics['years'] = years
            
            # Annualized return
            if years > 0:
                metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1
            else:
                metrics['annualized_return'] = total_return
        
        # Convert returns to numpy array for calculations
        returns_array = np.array(returns)
        
        # Return metrics
        metrics['mean_return'] = float(np.mean(returns_array))
        metrics['std_return'] = float(np.std(returns_array))
        
        # Risk metrics
        if metrics['std_return'] > 0:
            # Sharpe ratio
            excess_returns = metrics['mean_return'] * 252 - risk_free_rate
            metrics['sharpe_ratio'] = excess_returns / (metrics['std_return'] * np.sqrt(252))
            
            # Sortino ratio (downside risk)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_deviation = float(np.std(downside_returns))
                metrics['sortino_ratio'] = excess_returns / (downside_deviation * np.sqrt(252))
            else:
                metrics['sortino_ratio'] = float('inf')  # No downside
        else:
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
            
        # Calculate drawdown
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        metrics['max_drawdown'] = float(min(drawdown))
        metrics['current_drawdown'] = float(drawdown[-1])
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = metrics.get('annualized_return', total_return) / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = float('inf')  # No drawdown
            
    else:
        # Not enough data
        metrics['total_return'] = 0.0
        metrics['annualized_return'] = 0.0
        metrics['mean_return'] = 0.0
        metrics['std_return'] = 0.0
        metrics['sharpe_ratio'] = 0.0
        metrics['sortino_ratio'] = 0.0
        metrics['max_drawdown'] = 0.0
        metrics['current_drawdown'] = 0.0
        metrics['calmar_ratio'] = 0.0
    
    # Trade metrics
    if trades:
        profitable_trades = [t for t in trades if t.get('realized_pnl', t.get('pnl', 0)) > 0]
        unprofitable_trades = [t for t in trades if t.get('realized_pnl', t.get('pnl', 0)) <= 0]
        
        metrics['total_trades'] = len(trades)
        metrics['profitable_trades'] = len(profitable_trades)
        metrics['unprofitable_trades'] = len(unprofitable_trades)
        
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = len(profitable_trades) / len(trades)
        else:
            metrics['win_rate'] = 0.0
            
        # Calculate profit metrics
        if profitable_trades:
            total_profit = sum(t.get('realized_pnl', t.get('pnl', 0)) for t in profitable_trades)
            avg_profit = total_profit / len(profitable_trades)
            max_profit = max(t.get('realized_pnl', t.get('pnl', 0)) for t in profitable_trades)
            
            metrics['total_profit'] = float(total_profit)
            metrics['avg_profit'] = float(avg_profit)
            metrics['max_profit'] = float(max_profit)
        else:
            metrics['total_profit'] = 0.0
            metrics['avg_profit'] = 0.0
            metrics['max_profit'] = 0.0
            
        # Calculate loss metrics
        if unprofitable_trades:
            total_loss = sum(t.get('realized_pnl', t.get('pnl', 0)) for t in unprofitable_trades)
            avg_loss = total_loss / len(unprofitable_trades)
            max_loss = min(t.get('realized_pnl', t.get('pnl', 0)) for t in unprofitable_trades)
            
            metrics['total_loss'] = float(total_loss)
            metrics['avg_loss'] = float(avg_loss)
            metrics['max_loss'] = float(max_loss)
        else:
            metrics['total_loss'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['max_loss'] = 0.0
            
        # Profit factor
        if metrics['total_loss'] != 0:
            metrics['profit_factor'] = abs(metrics['total_profit'] / metrics['total_loss'])
        else:
            metrics['profit_factor'] = float('inf')  # No losses
            
        # Average trade metrics
        if metrics['total_trades'] > 0:
            metrics['avg_trade_pnl'] = (metrics['total_profit'] + metrics['total_loss']) / metrics['total_trades']
        else:
            metrics['avg_trade_pnl'] = 0.0
            
        # Calculate expectancy
        if metrics['win_rate'] > 0 and metrics['avg_profit'] > 0 and metrics['avg_loss'] < 0:
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_profit'] + 
                                     (1 - metrics['win_rate']) * metrics['avg_loss'])
        else:
            metrics['expectancy'] = 0.0
    else:
        # No trades
        metrics['total_trades'] = 0
        metrics['profitable_trades'] = 0
        metrics['unprofitable_trades'] = 0
        metrics['win_rate'] = 0.0
        metrics['total_profit'] = 0.0
        metrics['total_loss'] = 0.0
        metrics['avg_profit'] = 0.0
        metrics['avg_loss'] = 0.0
        metrics['max_profit'] = 0.0
        metrics['max_loss'] = 0.0
        metrics['profit_factor'] = 0.0
        metrics['avg_trade_pnl'] = 0.0
        metrics['expectancy'] = 0.0
        
    # Leverage metrics
    if leverages:
        metrics['avg_leverage'] = float(np.mean(leverages))
        metrics['max_leverage'] = float(np.max(leverages))
        metrics['min_leverage'] = float(np.min(leverages))
        
        # Calculate leverage adjusted metrics
        if metrics['avg_leverage'] > 0:
            metrics['return_to_leverage_ratio'] = metrics['total_return'] / metrics['avg_leverage']
        else:
            metrics['return_to_leverage_ratio'] = 0.0
            
        # Leverage consistency (standard deviation of leverage)
        metrics['leverage_consistency'] = float(np.std(leverages))
    else:
        metrics['avg_leverage'] = 0.0
        metrics['max_leverage'] = 0.0
        metrics['min_leverage'] = 0.0
        metrics['return_to_leverage_ratio'] = 0.0
        metrics['leverage_consistency'] = 0.0
        
    # Recovery factor
    if metrics['max_drawdown'] != 0:
        metrics['recovery_factor'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['recovery_factor'] = float('inf')
        
    return metrics

def calculate_regime_metrics(
    returns: List[float], 
    regime_periods: List, 
    timestamps: List
) -> Dict:
    """
    Calculate performance metrics for each market regime.
    
    Args:
        returns: List of returns
        regime_periods: List of regime periods
        timestamps: List of timestamps for returns
        
    Returns:
        Dictionary of regime-specific metrics
    """
    regime_metrics = {}
    
    # Create DataFrame with returns and timestamps
    if not timestamps or len(timestamps) != len(returns) + 1:
        logger.warning("Timestamps don't match returns length for regime metrics")
        return regime_metrics
        
    # Ensure timestamps are datetime objects
    if isinstance(timestamps[0], str):
        timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
    # Create returns DataFrame
    returns_df = pd.DataFrame({
        'timestamp': timestamps[1:],  # Skip first timestamp
        'return': returns
    })
    returns_df.set_index('timestamp', inplace=True)
    
    # Calculate metrics for each regime
    for period in regime_periods:
        try:
            # Get returns in this period
            period_returns = returns_df.loc[period.start_date:period.end_date]['return']
            
            if len(period_returns) == 0:
                continue
                
            # Calculate metrics
            total_return = (1 + period_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(period_returns)) - 1 if len(period_returns) > 0 else 0
            sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252) if np.std(period_returns) > 0 else 0
            
            # Store metrics
            regime_name = period.regime.value
            if regime_name not in regime_metrics:
                regime_metrics[regime_name] = {
                    'total_return': [],
                    'annualized_return': [],
                    'sharpe_ratio': [],
                    'max_drawdown': [],
                    'periods': 0,
                    'days': 0
                }
                
            # Update metrics
            regime_metrics[regime_name]['total_return'].append(total_return)
            regime_metrics[regime_name]['annualized_return'].append(annualized_return)
            regime_metrics[regime_name]['sharpe_ratio'].append(sharpe)
            
            # Calculate drawdown
            cumulative = (1 + period_returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            regime_metrics[regime_name]['max_drawdown'].append(max_drawdown)
            
            # Update counters
            regime_metrics[regime_name]['periods'] += 1
            regime_metrics[regime_name]['days'] += (period.end_date - period.start_date).days
            
        except Exception as e:
            logger.error(f"Error calculating regime metrics for {period.regime.value}: {str(e)}")
    
    # Calculate averages
    for regime, metrics in regime_metrics.items():
        metrics['avg_total_return'] = np.mean(metrics['total_return']) if metrics['total_return'] else 0
        metrics['avg_annualized_return'] = np.mean(metrics['annualized_return']) if metrics['annualized_return'] else 0
        metrics['avg_sharpe_ratio'] = np.mean(metrics['sharpe_ratio']) if metrics['sharpe_ratio'] else 0
        metrics['avg_max_drawdown'] = np.mean(metrics['max_drawdown']) if metrics['max_drawdown'] else 0
        
    return regime_metrics

def calculate_drawdowns(portfolio_values: List[float]) -> List[Dict]:
    """
    Calculate and identify significant drawdown periods.
    
    Args:
        portfolio_values: List of portfolio values
        
    Returns:
        List of drawdown periods with start/end indices and magnitude
    """
    if len(portfolio_values) < 2:
        return []
        
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate drawdowns
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    
    # Identify drawdown periods
    in_drawdown = False
    start_idx = 0
    drawdown_periods = []
    
    for i in range(1, len(drawdown)):
        # Start of drawdown
        if not in_drawdown and drawdown[i] < 0:
            in_drawdown = True
            start_idx = i - 1  # Previous peak
            
        # End of drawdown (recovery)
        elif in_drawdown and values[i] >= peak[start_idx]:
            in_drawdown = False
            
            # Calculate metrics for this drawdown
            max_dd_idx = start_idx + np.argmin(drawdown[start_idx:i])
            max_dd = drawdown[max_dd_idx]
            recovery_time = i - max_dd_idx
            
            # Only record significant drawdowns (e.g., > 1%)
            if max_dd < -0.01:
                drawdown_periods.append({
                    'start_idx': int(start_idx),
                    'max_dd_idx': int(max_dd_idx),
                    'end_idx': int(i),
                    'max_drawdown': float(max_dd),
                    'recovery_time': int(recovery_time),
                    'drawdown_length': int(max_dd_idx - start_idx)
                })
    
    # Handle drawdown at the end of the series
    if in_drawdown:
        max_dd_idx = start_idx + np.argmin(drawdown[start_idx:])
        max_dd = drawdown[max_dd_idx]
        
        if max_dd < -0.01:
            drawdown_periods.append({
                'start_idx': int(start_idx),
                'max_dd_idx': int(max_dd_idx),
                'end_idx': int(len(drawdown) - 1),
                'max_drawdown': float(max_dd),
                'recovery_time': None,  # No recovery yet
                'drawdown_length': int(max_dd_idx - start_idx)
            })
            
    return drawdown_periods 