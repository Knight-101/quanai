import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import os

# Configure logging
logger = logging.getLogger(__name__)

def currency_formatter(x, pos):
    """Format values as currency"""
    if abs(x) >= 1e6:
        return '${:.1f}M'.format(x * 1e-6)
    elif abs(x) >= 1e3:
        return '${:.1f}K'.format(x * 1e-3)
    else:
        return '${:.2f}'.format(x)

def pct_formatter(x, pos):
    """Format values as percentages"""
    return '{:.1f}%'.format(x * 100)

def create_performance_charts(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 22),
    regime_periods: Optional[List] = None
):
    """
    Create comprehensive performance charts from backtest results.
    
    Args:
        results: Dictionary of backtest results
        output_path: Path to save the charts
        show: Whether to display the charts
        figsize: Figure size (width, height)
        regime_periods: List of regime periods
    """
    try:
        # Extract data from results
        portfolio_values = results.get('portfolio_values', [])
        timestamps = results.get('timestamps', [])
        returns = results.get('returns', [])
        drawdowns = results.get('drawdowns', [])
        leverages = results.get('leverages', [])
        trades = results.get('trades', [])
        metrics = results.get('metrics', {})
        
        # Check if we have enough data to plot
        if len(portfolio_values) < 2:
            logger.warning("Not enough data to create performance charts")
            return
            
        # Create timestamps if not provided
        if not timestamps or len(timestamps) != len(portfolio_values):
            logger.warning("Creating synthetic timestamps")
            timestamps = pd.date_range(
                start=datetime.now() - pd.Timedelta(days=len(portfolio_values) - 1),
                periods=len(portfolio_values),
                freq='D'
            )
            
        # Convert timestamps to datetime if they are strings
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
            
        # Create figure with subplots
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = gridspec.GridSpec(5, 2, height_ratios=[3, 2, 2, 1, 2])
        
        # 1. Portfolio value chart
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(timestamps, portfolio_values, linewidth=2, color='#1f77b4')
        ax1.set_title('Portfolio Value', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Add market regimes if provided
        if regime_periods:
            # Set background color based on regime
            regime_colors = {
                'trending_up': '#d0f0c0',    # Light green
                'trending_down': '#ffd0d0',  # Light red
                'range_bound': '#f0f0d0',    # Light yellow
                'volatile': '#d0d0f0',       # Light blue
                'crisis': '#ff9090',         # Reddish
                'recovery': '#90ee90',       # Greenish
                'normal': '#f0f0f0'          # Light gray
            }
            
            for period in regime_periods:
                # Only show if in range
                if period.end_date < timestamps[0] or period.start_date > timestamps[-1]:
                    continue
                    
                # Adjust start/end to be within range
                start_date = max(period.start_date, timestamps[0])
                end_date = min(period.end_date, timestamps[-1])
                
                # Get color for this regime
                color = regime_colors.get(period.regime.value, '#f0f0f0')
                
                # Add background color
                ax1.axvspan(start_date, end_date, alpha=0.3, color=color)
                
                # Add regime label at the top of the span
                # Find the middle datetime for the label
                mid_date = start_date + (end_date - start_date) / 2
                
                # Add label
                ax1.text(
                    mid_date, ax1.get_ylim()[1] * 0.95,
                    period.regime.value.replace('_', ' ').title(),
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=8
                )
                
        # Customize grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Returns chart
        ax2 = plt.subplot(gs[1, 0])
        ax2.plot(timestamps[1:], returns, linewidth=1, color='#2ca02c', alpha=0.8)
        ax2.set_title('Daily Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        
        # Add horizontal line at 0
        ax2.axhline(y=0, color='#d62728', linestyle='-', alpha=0.6)
        
        # Customize grid
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Drawdown chart
        if drawdowns and len(drawdowns) == len(portfolio_values):
            ax3 = plt.subplot(gs[1, 1])
            ax3.fill_between(timestamps, 0, drawdowns, color='#d62728', alpha=0.3)
            ax3.plot(timestamps, drawdowns, linewidth=1, color='#d62728')
            ax3.set_title('Drawdowns', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Drawdown')
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
            
            # Set y-limits
            max_dd = min(drawdowns) if drawdowns else 0
            ax3.set_ylim(1.1 * max_dd, 0.005)
            
            # Customize grid
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
        # 4. Leverage chart
        if leverages and len(leverages) > 0:
            ax4 = plt.subplot(gs[2, 0])
            ax4.plot(timestamps[1:], leverages, linewidth=1, color='#ff7f0e')
            ax4.set_title('Leverage', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Leverage')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # Set y-limits
            max_lev = max(leverages) if leverages else 1
            ax4.set_ylim(0, max_lev * 1.1)
            
            # Customize grid
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
        # 5. Trade positions chart
        if trades and len(trades) > 0:
            # Group trades by asset
            trade_assets = sorted(set(t['asset'] for t in trades if 'asset' in t))
            
            # Check if trades have timestamp
            has_timestamps = all('timestamp' in t for t in trades)
            
            if trade_assets and has_timestamps:
                # Create a dictionary to hold asset positions over time
                positions_by_asset = {asset: [] for asset in trade_assets}
                position_timestamps = []
                
                # Start with initial positions
                current_positions = {asset: 0 for asset in trade_assets}
                
                # Sort trades by timestamp
                sorted_trades = sorted(trades, key=lambda x: x['timestamp'])
                
                # Track portfolio value at each trade
                for trade in sorted_trades:
                    # Update position
                    asset = trade['asset']
                    size = trade.get('size', 0)
                    current_positions[asset] += size
                    
                    # Store all positions at this timestamp
                    position_timestamps.append(trade['timestamp'])
                    for a, pos in current_positions.items():
                        positions_by_asset[a].append(pos)
                
                # Plot positions
                ax5 = plt.subplot(gs[2, 1])
                
                # Plot each asset's position
                for asset in trade_assets:
                    if len(position_timestamps) == len(positions_by_asset[asset]):
                        ax5.step(position_timestamps, positions_by_asset[asset], 
                                where='post', label=asset)
                
                ax5.set_title('Asset Positions', fontsize=12, fontweight='bold')
                ax5.set_ylabel('Position Size')
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax5.legend(loc='upper left')
                
                # Add horizontal line at 0
                ax5.axhline(y=0, color='#7f7f7f', linestyle='-', alpha=0.6)
                
                # Customize grid
                ax5.grid(True, linestyle='--', alpha=0.7)
                ax5.spines['top'].set_visible(False)
                ax5.spines['right'].set_visible(False)
            
        # 6. Metrics summary
        ax6 = plt.subplot(gs[3, :])
        ax6.axis('off')
        
        # Divide metrics into two columns
        metrics_text = [
            f"Total Return: {metrics.get('total_return', 0):.2%}",
            f"Annualized Return: {metrics.get('annualized_return', 0):.2%}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}",
            f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Avg Trade: ${metrics.get('avg_trade_pnl', 0):.2f}",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Avg Leverage: {metrics.get('avg_leverage', 0):.2f}",
            f"Days: {metrics.get('days', 0)}"
        ]
        
        half = len(metrics_text) // 2
        
        # Create the left column
        summary_left = '\n'.join(metrics_text[:half])
        ax6.text(0.25, 0.5, summary_left, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=11,
                 fontweight='bold')
                 
        # Create the right column
        summary_right = '\n'.join(metrics_text[half:])
        ax6.text(0.75, 0.5, summary_right, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=11,
                 fontweight='bold')
                 
        # 7. Asset performance comparison
        if trades and len(trades) > 0 and trade_assets:
            # Calculate performance by asset
            asset_metrics = {}
            
            for asset in trade_assets:
                asset_trades = [t for t in trades if t.get('asset') == asset]
                
                if not asset_trades:
                    continue
                    
                # Calculate asset metrics
                profitable_trades = [t for t in asset_trades if t.get('realized_pnl', t.get('pnl', 0)) > 0]
                total_pnl = sum(t.get('realized_pnl', t.get('pnl', 0)) for t in asset_trades)
                
                asset_metrics[asset] = {
                    'total_pnl': total_pnl,
                    'trades': len(asset_trades),
                    'profitable': len(profitable_trades),
                    'win_rate': len(profitable_trades) / len(asset_trades) if asset_trades else 0,
                    'avg_pnl': total_pnl / len(asset_trades) if asset_trades else 0
                }
            
            # Plot asset comparison
            ax7 = plt.subplot(gs[4, :])
            
            assets = list(asset_metrics.keys())
            x = np.arange(len(assets))
            width = 0.35
            
            # Plot total PnL by asset
            total_pnl = [asset_metrics[a]['total_pnl'] for a in assets]
            win_rates = [asset_metrics[a]['win_rate'] for a in assets]
            
            bars = ax7.bar(x, total_pnl, width, label='Total PnL')
            
            # Color bars based on profit/loss
            for i, bar in enumerate(bars):
                if total_pnl[i] >= 0:
                    bar.set_color('#2ca02c')  # Green for profit
                else:
                    bar.set_color('#d62728')  # Red for loss
            
            # Add win rate as text labels
            for i, rate in enumerate(win_rates):
                ax7.text(i, total_pnl[i] * 1.05 if total_pnl[i] >= 0 else total_pnl[i] * 0.95, 
                        f"Win: {rate:.1%}", 
                        ha='center', va='bottom' if total_pnl[i] >= 0 else 'top',
                        fontsize=9)
            
            ax7.set_title('Performance by Asset', fontsize=12, fontweight='bold')
            ax7.set_ylabel('PnL')
            ax7.set_xticks(x)
            ax7.set_xticklabels(assets)
            ax7.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
            
            # Add horizontal line at 0
            ax7.axhline(y=0, color='#7f7f7f', linestyle='-', alpha=0.6)
            
            # Customize grid
            ax7.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)
        
        # Adjust layout and save/show
        plt.tight_layout()
        
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error creating performance charts: {str(e)}")
        import traceback
        traceback.print_exc()

def create_regime_comparison_chart(
    regime_metrics: Dict,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create a chart comparing performance across different market regimes.
    
    Args:
        regime_metrics: Dictionary of regime-specific metrics
        output_path: Path to save the chart
        show: Whether to display the chart
        figsize: Figure size (width, height)
    """
    try:
        # Check if we have regime metrics
        if not regime_metrics:
            logger.warning("No regime metrics to plot")
            return
            
        # Create figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract data
        regimes = list(regime_metrics.keys())
        returns = [regime_metrics[r]['avg_total_return'] for r in regimes]
        sharpes = [regime_metrics[r]['avg_sharpe_ratio'] for r in regimes]
        periods = [regime_metrics[r]['periods'] for r in regimes]
        days = [regime_metrics[r]['days'] for r in regimes]
        
        # Calculate period percentages for pie chart
        total_days = sum(days)
        day_pcts = [d / total_days for d in days] if total_days > 0 else [0] * len(days)
        
        # Bar chart of returns
        x = np.arange(len(regimes))
        width = 0.35
        
        # Plot returns
        bars = ax1.bar(x, returns, width)
        
        # Color bars based on return
        for i, bar in enumerate(bars):
            if returns[i] >= 0:
                bar.set_color('#2ca02c')  # Green for profit
            else:
                bar.set_color('#d62728')  # Red for loss
                
        # Add sharpe ratio as text
        for i, sharpe in enumerate(sharpes):
            ax1.text(i, returns[i] * 1.05 if returns[i] >= 0 else returns[i] * 0.95, 
                    f"Sharpe: {sharpe:.2f}", 
                    ha='center', va='bottom' if returns[i] >= 0 else 'top',
                    fontsize=9)
        
        ax1.set_title('Returns by Market Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Return')
        ax1.set_xticks(x)
        ax1.set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45, ha='right')
        ax1.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        
        # Add horizontal line at 0
        ax1.axhline(y=0, color='#7f7f7f', linestyle='-', alpha=0.6)
        
        # Customize grid
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Pie chart of time spent in each regime
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        wedges, texts, autotexts = ax2.pie(
            day_pcts, 
            labels=[f"{r.replace('_', ' ').title()}" for r in regimes],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(regimes)]
        )
        
        # Make text easier to read
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            
        ax2.set_title('Time Spent in Each Regime', fontsize=12, fontweight='bold')
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add period counts as legend
        legend_labels = [f"{r.replace('_', ' ').title()}: {p} periods, {d} days" 
                         for r, p, d in zip(regimes, periods, days)]
        ax2.legend(wedges, legend_labels, 
                  title="Regime Distribution",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Adjust layout and save/show
        plt.tight_layout()
        
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error creating regime comparison chart: {str(e)}")
        import traceback
        traceback.print_exc()

def create_regime_transition_chart(
    regime_periods: List,
    output_path: Optional[str] = None, 
    show: bool = False,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create a chart showing regime transitions over time.
    
    Args:
        regime_periods: List of regime periods
        output_path: Path to save the chart
        show: Whether to display the chart
        figsize: Figure size (width, height)
    """
    try:
        # Check if we have regime periods
        if not regime_periods:
            logger.warning("No regime periods to plot")
            return
            
        # Create figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a mapping of regime values to numeric values for plotting
        regimes = sorted(set(period.regime.value for period in regime_periods))
        regime_to_num = {regime: i for i, regime in enumerate(regimes)}
        
        # Sort periods by start date
        sorted_periods = sorted(regime_periods, key=lambda x: x.start_date)
        
        # Create lists for start dates, end dates, and regime values
        start_dates = [period.start_date for period in sorted_periods]
        end_dates = [period.end_date for period in sorted_periods]
        regime_nums = [regime_to_num[period.regime.value] for period in sorted_periods]
        
        # Define colors for regimes
        regime_colors = {
            'trending_up': '#2ca02c',    # Green
            'trending_down': '#d62728',  # Red
            'range_bound': '#ff7f0e',    # Orange
            'volatile': '#1f77b4',       # Blue
            'crisis': '#7f7f7f',         # Gray
            'recovery': '#9467bd',       # Purple
            'normal': '#8c564b'          # Brown
        }
        
        # Plot regime transitions
        for i, period in enumerate(sorted_periods):
            # Get start and end dates
            start = period.start_date
            end = period.end_date
            
            # Get color for this regime
            color = regime_colors.get(period.regime.value, '#e377c2')
            
            # Plot horizontal line
            ax.plot([start, end], [regime_to_num[period.regime.value]] * 2, 
                   linewidth=10, color=color, solid_capstyle='butt')
            
            # Add label at the start of the regime
            if i == 0 or period.regime != sorted_periods[i-1].regime:
                ax.text(start, regime_to_num[period.regime.value] + 0.1, 
                       period.regime.value.replace('_', ' ').title(),
                       fontsize=9, va='bottom')
        
        # Set y-ticks and labels
        ax.set_yticks(list(range(len(regimes))))
        ax.set_yticklabels([r.replace('_', ' ').title() for r in regimes])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Set labels and title
        ax.set_title('Market Regime Transitions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Regime', fontsize=12)
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        legend_handles = [plt.Line2D([0], [0], color=regime_colors.get(r, '#e377c2'), 
                                    linewidth=5) for r in regimes]
        ax.legend(legend_handles, [r.replace('_', ' ').title() for r in regimes], 
                 loc='best', title='Market Regimes')
        
        # Adjust layout and save/show
        plt.tight_layout()
        
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error creating regime transition chart: {str(e)}")
        import traceback
        traceback.print_exc() 