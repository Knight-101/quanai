"""
Visualization Module for Backtesting Results

This module provides utilities for visualizing backtesting results
with a focus on performance metrics, equity curves, and trade analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
import logging
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
from .regime_analyzer import MarketRegime

# Set up logging
logger = logging.getLogger(__name__)

# Set Seaborn style for all plots
sns.set(style="whitegrid")


def plot_equity_curve(
    portfolio_values: List[float],
    timestamps: List = None,
    drawdowns: List[float] = None,
    benchmark_values: List[float] = None,
    trades: List[Dict] = None,
    regime_periods: List = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None
) -> Figure:
    """
    Plot equity curve with drawdowns and trades.
    
    Args:
        portfolio_values: List of portfolio values
        timestamps: List of timestamps
        drawdowns: List of drawdown values
        benchmark_values: List of benchmark values for comparison
        trades: List of trade dictionaries
        regime_periods: List of regime period objects
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Handle timestamps
    x_values = timestamps if timestamps else range(len(portfolio_values))
    
    # Plot equity curve
    ax1.plot(x_values, portfolio_values, label="Portfolio", linewidth=2, color="royalblue")
    
    # Plot benchmark if provided
    if benchmark_values:
        if len(benchmark_values) < len(portfolio_values):
            # Pad benchmark values if necessary
            benchmark_values = [benchmark_values[0]] * (len(portfolio_values) - len(benchmark_values)) + benchmark_values
        elif len(benchmark_values) > len(portfolio_values):
            # Trim benchmark values if necessary
            benchmark_values = benchmark_values[-len(portfolio_values):]
            
        ax1.plot(x_values, benchmark_values, label="Benchmark", linewidth=1.5, color="gray", alpha=0.7)
    
    # Color background by market regime if provided
    if regime_periods:
        for period in regime_periods:
            start_idx = timestamps.index(period.start_date) if timestamps else 0
            end_idx = timestamps.index(period.end_date) if timestamps else len(portfolio_values) - 1
            
            # Set color based on regime
            if period.regime == MarketRegime.BULL:
                color = "lightgreen"
            elif period.regime == MarketRegime.BEAR:
                color = "lightcoral"
            elif period.regime == MarketRegime.HIGH_VOL:
                color = "lightyellow"
            elif period.regime == MarketRegime.CRISIS:
                color = "lightpink"
            else:
                color = "lightgray"
                
            # Add background color for this regime period
            ax1.axvspan(x_values[start_idx], x_values[end_idx], color=color, alpha=0.3)
    
    # Plot trades if provided
    if trades:
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                # Find timestamps or indices
                if timestamps:
                    entry_idx = timestamps.index(trade['entry_time']) if trade['entry_time'] in timestamps else 0
                    exit_idx = timestamps.index(trade['exit_time']) if trade['exit_time'] in timestamps else len(portfolio_values) - 1
                else:
                    entry_idx = trade.get('entry_idx', 0)
                    exit_idx = trade.get('exit_idx', len(portfolio_values) - 1)
                
                # Determine color based on PnL
                color = "green" if trade.get('pnl', 0) > 0 else "red"
                
                # Plot entry and exit points
                ax1.scatter(x_values[entry_idx], portfolio_values[entry_idx], color=color, marker="^", s=50)
                ax1.scatter(x_values[exit_idx], portfolio_values[exit_idx], color=color, marker="v", s=50)
    
    # Plot drawdowns if provided
    if drawdowns:
        ax2.fill_between(x_values, 0, drawdowns, color="red", alpha=0.3, label="Drawdown")
        ax2.set_ylim([min(drawdowns) - 0.01, 0.01])
        ax2.set_ylabel("Drawdown", fontsize=10)
    
    # Format dates on x-axis if timestamps provided
    if timestamps and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.gcf().autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.gcf().autofmt_xdate()
        
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Add labels and title
    ax1.set_ylabel("Portfolio Value", fontsize=12)
    ax1.set_title(title, fontsize=14)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Equity curve saved to {save_path}")
    
    return fig


def plot_returns_histogram(
    returns: np.ndarray,
    benchmark_returns: np.ndarray = None,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    save_path: str = None
) -> Figure:
    """
    Plot histogram of returns with normal distribution overlay.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot strategy returns
    sns.histplot(returns, bins=bins, kde=True, stat="density", color="royalblue", alpha=0.7, label="Strategy Returns", ax=ax)
    
    # Plot benchmark returns if provided
    if benchmark_returns is not None:
        sns.histplot(benchmark_returns, bins=bins, kde=True, stat="density", color="gray", alpha=0.5, label="Benchmark Returns", ax=ax)
    
    # Add vertical lines for key values
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    plt.axvline(x=np.mean(returns), color="royalblue", linestyle="-", alpha=0.8, label=f"Mean: {np.mean(returns):.4f}")
    
    # Add reference normal distribution
    if len(returns) > 2:
        x = np.linspace(min(returns), max(returns), 1000)
        y = np.exp(-(x - np.mean(returns))**2 / (2 * np.var(returns))) / np.sqrt(2 * np.pi * np.var(returns))
        plt.plot(x, y, color="red", alpha=0.5, label="Normal Distribution")
    
    # Format plot
    plt.title(title, fontsize=14)
    plt.xlabel("Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Returns histogram saved to {save_path}")
    
    return fig


def plot_rolling_metrics(
    returns: np.ndarray,
    timestamps: List = None,
    window: int = 30,
    metrics: List[str] = ["return", "volatility", "sharpe"],
    title: str = "Rolling Metrics",
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None
) -> Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        returns: Array of returns
        timestamps: List of timestamps
        window: Rolling window size
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    if len(returns) < window:
        logger.warning(f"Not enough returns for rolling window of {window}")
        return None
        
    # Calculate rolling metrics
    rolling_data = {}
    
    if "return" in metrics:
        rolling_return = np.array([np.mean(returns[i:i+window]) * 252 for i in range(len(returns) - window + 1)])
        rolling_data["Rolling Annual Return"] = rolling_return
        
    if "volatility" in metrics:
        rolling_vol = np.array([np.std(returns[i:i+window]) * np.sqrt(252) for i in range(len(returns) - window + 1)])
        rolling_data["Rolling Annual Volatility"] = rolling_vol
        
    if "sharpe" in metrics:
        rolling_sharpe = np.array([np.mean(returns[i:i+window]) / (np.std(returns[i:i+window]) + 1e-10) * np.sqrt(252) for i in range(len(returns) - window + 1)])
        rolling_data["Rolling Sharpe Ratio"] = rolling_sharpe
        
    if "drawdown" in metrics:
        rolling_drawdown = []
        for i in range(len(returns) - window + 1):
            window_returns = returns[i:i+window]
            cum_returns = np.cumprod(1 + window_returns) - 1
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / (peak + 1)
            rolling_drawdown.append(np.min(drawdown))
        rolling_data["Rolling Max Drawdown"] = np.array(rolling_drawdown)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-axis values
    x_values = timestamps[window-1:] if timestamps else range(window-1, len(returns))
    
    # Plot each metric
    for name, values in rolling_data.items():
        ax.plot(x_values, values, label=name, linewidth=2)
    
    # Format plot
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format dates on x-axis if timestamps provided
    if timestamps and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.gcf().autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.gcf().autofmt_xdate()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Rolling metrics saved to {save_path}")
    
    return fig


def plot_drawdowns(
    portfolio_values: List[float],
    timestamps: List = None,
    top_n: int = 5,
    title: str = "Top Drawdowns",
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None
) -> Figure:
    """
    Plot top N drawdowns.
    
    Args:
        portfolio_values: List of portfolio values
        timestamps: List of timestamps
        top_n: Number of top drawdowns to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    if len(portfolio_values) < 2:
        logger.warning("Not enough portfolio values for drawdown analysis")
        return None
        
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown in percent
    drawdown = (values - running_max) / running_max
    
    # Find drawdown episodes
    episodes = []
    in_drawdown = False
    start_idx = 0
    current_max = 0
    
    for i in range(len(drawdown)):
        if drawdown[i] < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = i
            current_max = values[i-1] if i > 0 else values[i]
        elif drawdown[i] == 0 and in_drawdown:
            # End of drawdown
            in_drawdown = False
            
            # Calculate drawdown depth and duration
            depth = np.min(drawdown[start_idx:i])
            duration = i - start_idx
            recovery = values[i] / current_max - 1  # Should be close to 0
            
            episodes.append({
                'start_idx': start_idx,
                'end_idx': i,
                'depth': depth,
                'duration': duration,
                'recovery': recovery
            })
    
    # If still in drawdown at the end, add it as an episode
    if in_drawdown:
        depth = np.min(drawdown[start_idx:])
        duration = len(drawdown) - start_idx
        recovery = values[-1] / current_max - 1
        
        episodes.append({
            'start_idx': start_idx,
            'end_idx': len(drawdown) - 1,
            'depth': depth,
            'duration': duration,
            'recovery': recovery
        })
    
    # Sort episodes by depth
    episodes.sort(key=lambda x: x['depth'])
    
    # Take top N episodes
    top_episodes = episodes[:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-axis values
    x_values = timestamps if timestamps else range(len(portfolio_values))
    
    # Plot portfolio equity curve
    ax.plot(x_values, portfolio_values, color="royalblue", alpha=0.7, label="Portfolio")
    
    # Plot each top drawdown
    colors = plt.cm.rainbow(np.linspace(0, 1, len(top_episodes)))
    
    for i, episode in enumerate(top_episodes):
        start_idx = episode['start_idx']
        end_idx = episode['end_idx']
        
        # Draw rectangle highlight for this drawdown period
        if timestamps:
            ax.axvspan(timestamps[start_idx], timestamps[end_idx], alpha=0.2, color=colors[i], 
                      label=f"DD {i+1}: {episode['depth']:.1%}, {episode['duration']} periods")
        else:
            ax.axvspan(start_idx, end_idx, alpha=0.2, color=colors[i], 
                      label=f"DD {i+1}: {episode['depth']:.1%}, {episode['duration']} periods")
    
    # Format plot
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    # Format dates on x-axis if timestamps provided
    if timestamps and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.gcf().autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.gcf().autofmt_xdate()
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Top drawdowns saved to {save_path}")
    
    return fig


def create_performance_tearsheet(
    metrics: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 12),
    save_path: str = None
) -> Figure:
    """
    Create a performance tearsheet with key metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Organize metrics into categories
    categories = {
        'Returns': [
            ('Total Return', f"{metrics.get('total_return', 0):.2%}"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"),
            ('Monthly Return', f"{metrics.get('monthly_return', 0):.2%}"),
            ('Positive Months', f"{metrics.get('positive_months', 0):.1%}")
        ],
        'Risk': [
            ('Volatility', f"{metrics.get('volatility', 0):.2%}"),
            ('Downside Volatility', f"{metrics.get('downside_volatility', 0):.2%}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Longest Drawdown', f"{metrics.get('max_drawdown_duration', 0)} periods"),
            ('Average Recovery Time', f"{metrics.get('recovery_time', 0)} periods")
        ],
        'Risk-Adjusted Returns': [
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
            ('Omega Ratio', f"{metrics.get('omega_ratio', 0):.2f}")
        ],
        'Trading Statistics': [
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2%}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Average Trade', f"${metrics.get('avg_trade_pnl', 0):.2f}"),
            ('Avg Holding Period', f"{metrics.get('avg_holding_period', 0):.1f} periods")
        ],
        'Risk Metrics': [
            ('Value at Risk (95%)', f"{metrics.get('value_at_risk', 0):.2%}"),
            ('Expected Shortfall', f"{metrics.get('expected_shortfall', 0):.2%}"),
            ('Average Leverage', f"{metrics.get('avg_leverage', 0):.2f}"),
            ('Maximum Leverage', f"{metrics.get('max_leverage', 0):.2f}")
        ]
    }
    
    # Build the table data
    table_data = []
    
    # Add a title row
    table_data.append(['', 'Performance Summary', ''])
    
    # Add category data
    for category, category_metrics in categories.items():
        # Add category header
        table_data.append([category, '', ''])
        
        # Add metrics
        for name, value in category_metrics:
            table_data.append(['', name, value])
        
        # Add empty row after category
        table_data.append(['', '', ''])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colWidths=[0.2, 0.5, 0.3],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style specific cells
    for i, row in enumerate(table_data):
        if row[0] and not row[1] and not row[2]:
            # Category header
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#d9e5ff')
                cell.set_text_props(weight='bold')
        elif i == 0:
            # Title row
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#b5ceff')
                cell.set_text_props(weight='bold', fontsize=12)
        elif not row[0] and row[1] and row[2]:
            # Metric row
            value = row[2]
            if '%' in value:
                # Parse the percentage value
                try:
                    value_float = float(value.strip('%')) / 100
                    if value_float < 0:
                        color = '#ffcccc'  # Light red for negative values
                    else:
                        color = '#ccffcc'  # Light green for positive values
                    table[(i, 2)].set_facecolor(color)
                except:
                    pass
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance tearsheet saved to {save_path}")
    
    return fig


def create_visualization_suite(
    results: Dict[str, Any],
    output_dir: str = "backtest_results",
    prefix: str = "",
    create_tearsheet: bool = True
):
    """
    Create a comprehensive suite of visualizations for backtest results.
    
    Args:
        results: Dictionary of backtest results
        output_dir: Directory to save visualization files
        prefix: Prefix for file names
        create_tearsheet: Whether to create a performance tearsheet
        
    Returns:
        Dictionary of created figure paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract data
    portfolio_values = results.get('portfolio_values', [])
    timestamps = results.get('timestamps', None)
    drawdowns = results.get('drawdowns', None)
    returns = results.get('returns', None)
    trades = results.get('trades', None)
    metrics = results.get('metrics', {})
    regime_periods = results.get('regime_periods', None)
    leverages = results.get('leverages', None)
    benchmark_values = results.get('benchmark_values', None)
    benchmark_returns = results.get('benchmark_returns', None)
    
    # Initialize figures dict
    figures = {}
    
    # 1. Plot equity curve
    if portfolio_values:
        equity_curve_path = os.path.join(output_dir, f"{prefix}equity_curve.png")
        fig = plot_equity_curve(
            portfolio_values=portfolio_values,
            timestamps=timestamps,
            drawdowns=drawdowns,
            benchmark_values=benchmark_values,
            trades=trades,
            regime_periods=regime_periods,
            title="Portfolio Equity Curve",
            save_path=equity_curve_path
        )
        figures['equity_curve'] = equity_curve_path
    
    # 2. Plot returns histogram
    if returns is not None:
        returns_hist_path = os.path.join(output_dir, f"{prefix}returns_distribution.png")
        fig = plot_returns_histogram(
            returns=returns,
            benchmark_returns=benchmark_returns,
            title="Returns Distribution",
            save_path=returns_hist_path
        )
        figures['returns_histogram'] = returns_hist_path
    
    # 3. Plot rolling metrics
    if returns is not None and len(returns) > 30:
        rolling_metrics_path = os.path.join(output_dir, f"{prefix}rolling_metrics.png")
        fig = plot_rolling_metrics(
            returns=returns,
            timestamps=timestamps,
            window=30,
            metrics=["return", "volatility", "sharpe"],
            title="30-Day Rolling Metrics",
            save_path=rolling_metrics_path
        )
        figures['rolling_metrics'] = rolling_metrics_path
    
    # 4. Plot top drawdowns
    if portfolio_values:
        drawdowns_path = os.path.join(output_dir, f"{prefix}top_drawdowns.png")
        fig = plot_drawdowns(
            portfolio_values=portfolio_values,
            timestamps=timestamps,
            top_n=5,
            title="Top 5 Drawdowns",
            save_path=drawdowns_path
        )
        figures['top_drawdowns'] = drawdowns_path
    
    # 5. Create performance tearsheet
    if create_tearsheet and metrics:
        tearsheet_path = os.path.join(output_dir, f"{prefix}performance_tearsheet.png")
        fig = create_performance_tearsheet(
            metrics=metrics,
            save_path=tearsheet_path
        )
        figures['performance_tearsheet'] = tearsheet_path
    
    logger.info(f"Created {len(figures)} visualization figures in {output_dir}")
    
    return figures 