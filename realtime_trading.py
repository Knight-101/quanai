#!/usr/bin/env python3
"""
Real-time Paper Trading Module for RL Trading Bot

This module provides a robust, production-grade real-time paper trading system
that loads a trained RL model and executes paper trades based on real-time market data.
It includes:
- Real-time data collection every minute
- Precise signal generation using the trained model
- Comprehensive position management and logging
- PnL calculation and tracking
- Web API endpoints for monitoring
"""

import os
import sys
import time
import json
import logging
import asyncio
import argparse
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import yaml
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from pathlib import Path
import traceback
import threading
import signal
import websockets
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Local imports
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
from data_system.feature_engine import DerivativesFeatureEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('realtime_trading')

# Add DailyReportGenerator class
class DailyReportGenerator:
    """
    Generates comprehensive daily trading reports with performance metrics.
    
    Includes:
    - PnL metrics (daily, cumulative)
    - Risk metrics (Sharpe, Sortino, drawdown)
    - Trading statistics (win rate, avg trade, etc.)
    - Symbol-specific performance
    """
    
    def __init__(self, reports_dir: str, risk_free_rate: float = 0.02/365):
        """
        Initialize the report generator.
        
        Args:
            reports_dir: Directory to save reports
            risk_free_rate: Daily risk-free rate for Sharpe calculation
        """
        self.reports_dir = reports_dir
        self.risk_free_rate = risk_free_rate
        self.daily_stats = {}
        self.all_days_stats = {}
        self.current_date = date.today()
        
        # Ensure reports directory exists
        os.makedirs(self.reports_dir, exist_ok=True)
        # Create subdirectories
        os.makedirs(os.path.join(self.reports_dir, 'daily'), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'trades'), exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('report_generator')
    
    def process_day_trades(self, trades: List[Dict], positions: Dict, 
                          pnl_history: List[Dict], start_balance: float,
                          symbols: List[str], day: Optional[date] = None):
        """
        Process all trades for a specific day and generate performance metrics.
        
        Args:
            trades: List of all trades executed
            positions: Current positions
            pnl_history: PnL history with timestamps
            start_balance: Starting balance for the day
            symbols: List of trading symbols
            day: Specific day to process (defaults to today)
        
        Returns:
            Dict with daily performance metrics
        """
        if day is None:
            day = date.today()
            
        day_str = day.strftime('%Y-%m-%d')
        self.logger.info(f"Generating daily report for {day_str}")
        
        # Filter trades for the specified day
        day_trades = []
        for trade in trades:
            trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00') 
                                              if trade['timestamp'].endswith('Z') 
                                              else trade['timestamp'])
            if trade_time.date() == day:
                day_trades.append(trade)
        
        # Filter PnL history for the specified day
        day_pnl = []
        for pnl_record in pnl_history:
            pnl_time = datetime.fromisoformat(pnl_record['timestamp'].replace('Z', '+00:00') 
                                             if pnl_record['timestamp'].endswith('Z') 
                                             else pnl_record['timestamp'])
            if pnl_time.date() == day:
                day_pnl.append(pnl_record)
        
        # Initialize performance metrics
        metrics = {
            'date': day_str,
            'total_trades': len(day_trades),
            'trades_by_symbol': {symbol: 0 for symbol in symbols},
            'pnl': {
                'realized': 0.0,
                'unrealized': 0.0,
                'total': 0.0,
                'by_symbol': {symbol: 0.0 for symbol in symbols}
            },
            'balance': {
                'start': start_balance,
                'end': pnl_history[-1]['balance'] if pnl_history else start_balance,
                'change': 0.0,
                'change_pct': 0.0
            },
            'sharpe': 0.0,
            'sortino': 0.0,
            'drawdown': {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'current_drawdown': 0.0,
                'current_drawdown_pct': 0.0
            },
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'trade_details': day_trades,
            'hourly_performance': {}
        }
        
        # Calculate trade statistics
        if day_trades:
            # Count trades by symbol
            for trade in day_trades:
                if 'symbol' in trade:
                    symbol = trade['symbol']
                    metrics['trades_by_symbol'][symbol] = metrics['trades_by_symbol'].get(symbol, 0) + 1
            
            # Calculate realized PnL from trades
            total_realized_pnl = 0.0
            realized_by_symbol = {symbol: 0.0 for symbol in symbols}
            win_trades = []
            loss_trades = []
            
            for trade in day_trades:
                # Sum up realized PnL
                if 'pnl' in trade:
                    pnl = trade['pnl']
                    total_realized_pnl += pnl
                    
                    # Track by symbol
                    if 'symbol' in trade:
                        symbol = trade['symbol']
                        realized_by_symbol[symbol] = realized_by_symbol.get(symbol, 0.0) + pnl
                    
                    # Track win/loss trades
                    if pnl > 0:
                        win_trades.append(pnl)
                    elif pnl < 0:
                        loss_trades.append(pnl)
            
            # Calculate win rate and related metrics
            if win_trades or loss_trades:
                metrics['win_rate'] = len(win_trades) / (len(win_trades) + len(loss_trades)) if (len(win_trades) + len(loss_trades)) > 0 else 0.0
                metrics['avg_win'] = sum(win_trades) / len(win_trades) if win_trades else 0.0
                metrics['avg_loss'] = sum(loss_trades) / len(loss_trades) if loss_trades else 0.0
                metrics['best_trade'] = max(win_trades) if win_trades else 0.0
                metrics['worst_trade'] = min(loss_trades) if loss_trades else 0.0
                metrics['profit_factor'] = abs(sum(win_trades) / sum(loss_trades)) if loss_trades and sum(loss_trades) != 0 else float('inf')
            
            metrics['pnl']['realized'] = total_realized_pnl
            for symbol in symbols:
                metrics['pnl']['by_symbol'][symbol] = realized_by_symbol.get(symbol, 0.0)
        
        # Calculate current unrealized PnL
        total_unrealized_pnl = 0.0
        for symbol, position in positions.items():
            if isinstance(position, dict) and 'unrealized_pnl' in position:
                total_unrealized_pnl += position['unrealized_pnl']
        
        metrics['pnl']['unrealized'] = total_unrealized_pnl
        metrics['pnl']['total'] = metrics['pnl']['realized'] + metrics['pnl']['unrealized']
        
        # Calculate balance changes
        if day_pnl:
            start_value = day_pnl[0]['total_value'] if day_pnl else start_balance
            end_value = day_pnl[-1]['total_value'] if day_pnl else (start_balance + total_realized_pnl + total_unrealized_pnl)
            
            metrics['balance']['start'] = start_value
            metrics['balance']['end'] = end_value
            metrics['balance']['change'] = end_value - start_value
            metrics['balance']['change_pct'] = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0.0
        
        # Calculate drawdown
        if day_pnl:
            # Convert PnL history to DataFrame for easier analysis
            pnl_df = pd.DataFrame(day_pnl)
            pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
            pnl_df = pnl_df.set_index('timestamp')
            
            # Calculate running maximum
            pnl_df['peak'] = pnl_df['total_value'].cummax()
            
            # Calculate drawdown in dollars and percentage
            pnl_df['drawdown'] = pnl_df['peak'] - pnl_df['total_value']
            pnl_df['drawdown_pct'] = (pnl_df['drawdown'] / pnl_df['peak']) * 100
            
            # Get maximum drawdown
            metrics['drawdown']['max_drawdown'] = pnl_df['drawdown'].max()
            metrics['drawdown']['max_drawdown_pct'] = pnl_df['drawdown_pct'].max()
            
            # Get current drawdown
            metrics['drawdown']['current_drawdown'] = pnl_df['drawdown'].iloc[-1]
            metrics['drawdown']['current_drawdown_pct'] = pnl_df['drawdown_pct'].iloc[-1]
            
            # Calculate hourly performance
            pnl_df['hour'] = pnl_df.index.hour
            hourly_pnl = pnl_df.groupby('hour')['total_value'].last() - pnl_df.groupby('hour')['total_value'].first()
            metrics['hourly_performance'] = hourly_pnl.to_dict()
            
            # Calculate returns for Sharpe/Sortino calculation
            pnl_df['returns'] = pnl_df['total_value'].pct_change()
            
            # Drop NaN values (first row will have NaN return)
            pnl_df = pnl_df.dropna(subset=['returns'])
            
            if len(pnl_df) > 1:
                # Calculate Sharpe ratio (annualized)
                excess_returns = pnl_df['returns'] - self.risk_free_rate
                sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
                metrics['sharpe'] = sharpe * (252 ** 0.5)  # Annualized
                
                # Calculate Sortino ratio (downside risk only)
                downside_returns = pnl_df[pnl_df['returns'] < 0]['returns']
                sortino = excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
                metrics['sortino'] = sortino * (252 ** 0.5)  # Annualized
        
        # Store the metrics
        self.daily_stats[day_str] = metrics
        return metrics
    
    def generate_daily_report(self, day: Optional[date] = None):
        """
        Generate a detailed daily report and save it to disk.
        
        Args:
            day: Day to generate report for (defaults to today)
        
        Returns:
            Path to the generated report
        """
        if day is None:
            day = date.today()
            
        day_str = day.strftime('%Y-%m-%d')
        
        if day_str not in self.daily_stats:
            self.logger.warning(f"No data available for {day_str}, cannot generate report")
            return None
        
        # Get the day's metrics
        metrics = self.daily_stats[day_str]
        
        # Create report file path
        report_path = os.path.join(self.reports_dir, 'daily', f"report_{day_str}.json")
        
        # Save full metrics to JSON
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate HTML report
        html_report_path = os.path.join(self.reports_dir, 'daily', f"report_{day_str}.html")
        self._generate_html_report(metrics, html_report_path)
        
        # Generate charts
        charts_path = os.path.join(self.reports_dir, 'charts', f"performance_{day_str}.png")
        self._generate_performance_chart(metrics, charts_path)
        
        # Save trades detail to CSV
        trades_path = os.path.join(self.reports_dir, 'trades', f"trades_{day_str}.csv")
        if metrics['trade_details']:
            pd.DataFrame(metrics['trade_details']).to_csv(trades_path, index=False)
        
        self.logger.info(f"Daily report generated at {report_path}")
        return report_path
    
    def _generate_html_report(self, metrics: Dict, output_path: str):
        """Generate an HTML report from the metrics."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Report {metrics['date']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
                .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-title {{ font-weight: bold; margin-bottom: 10px; color: #555; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading Report - {metrics['date']}</h1>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-title">PnL (Total)</div>
                        <div class="metric-value {('positive' if metrics['pnl']['total'] >= 0 else 'negative')}">${metrics['pnl']['total']:.2f}</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">PnL (%)</div>
                        <div class="metric-value {('positive' if metrics['balance']['change_pct'] >= 0 else 'negative')}">{metrics['balance']['change_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Sharpe Ratio</div>
                        <div class="metric-value">{metrics['sharpe']:.2f}</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Sortino Ratio</div>
                        <div class="metric-value">{metrics['sortino']:.2f}</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Win Rate</div>
                        <div class="metric-value">{metrics['win_rate']*100:.2f}%</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Max Drawdown</div>
                        <div class="metric-value negative">{metrics['drawdown']['max_drawdown_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Total Trades</div>
                        <div class="metric-value">{metrics['total_trades']}</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-title">Profit Factor</div>
                        <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                    </div>
                </div>
                
                <h2>PnL by Symbol</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>PnL</th>
                        <th>Trades</th>
                    </tr>
        """
        
        # Add rows for each symbol
        for symbol, pnl in metrics['pnl']['by_symbol'].items():
            trades_count = metrics['trades_by_symbol'].get(symbol, 0)
            html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{('positive' if pnl >= 0 else 'negative')}">${pnl:.2f}</td>
                        <td>{trades_count}</td>
                    </tr>
            """
            
        html += f"""
                </table>
                
                <h2>Trade Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Average Win</td>
                        <td class="positive">${metrics['avg_win']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">${metrics['avg_loss']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Best Trade</td>
                        <td class="positive">${metrics['best_trade']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Worst Trade</td>
                        <td class="negative">${metrics['worst_trade']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Starting Balance</td>
                        <td>${metrics['balance']['start']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Ending Balance</td>
                        <td>${metrics['balance']['end']:.2f}</td>
                    </tr>
                </table>
                
                <h2>Hourly Performance</h2>
                <table>
                    <tr>
                        <th>Hour</th>
                        <th>PnL</th>
                    </tr>
        """
        
        # Add rows for hourly performance
        for hour, hour_pnl in metrics['hourly_performance'].items():
            html += f"""
                    <tr>
                        <td>{hour}:00</td>
                        <td class="{('positive' if hour_pnl >= 0 else 'negative')}">${hour_pnl:.2f}</td>
                    </tr>
            """
            
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_performance_chart(self, metrics: Dict, output_path: str):
        """Generate performance charts and save to disk."""
        # Only generate chart if we have hourly performance data
        if not metrics['hourly_performance']:
            return
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot PnL by hour
            hours = list(metrics['hourly_performance'].keys())
            pnl_values = list(metrics['hourly_performance'].values())
            
            plt.bar(hours, pnl_values, color=['g' if p >= 0 else 'r' for p in pnl_values])
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f"Hourly PnL - {metrics['date']}")
            plt.xlabel("Hour")
            plt.ylabel("PnL ($)")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating performance chart: {str(e)}")
    
    def update_all_time_stats(self):
        """Update all-time statistics across all days."""
        if not self.daily_stats:
            return
        
        # Aggregate data across all days
        all_time_metrics = {
            'total_days': len(self.daily_stats),
            'total_trades': 0,
            'winning_days': 0,
            'losing_days': 0,
            'best_day': {
                'date': None,
                'pnl': 0.0
            },
            'worst_day': {
                'date': None,
                'pnl': 0.0
            },
            'cumulative_pnl': 0.0,
            'avg_daily_pnl': 0.0,
            'total_fees': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'daily_returns': [],
            'pnl_by_symbol': {},
            'trades_by_symbol': {}
        }
        
        # Calculate daily returns for each day
        winning_days = 0
        losing_days = 0
        all_daily_returns = []
        best_pnl = float('-inf')
        worst_pnl = float('inf')
        
        for day_str, day_stats in sorted(self.daily_stats.items()):
            # Count winning and losing days
            daily_pnl = day_stats['pnl']['total']
            all_time_metrics['total_trades'] += day_stats['total_trades']
            
            if daily_pnl > 0:
                winning_days += 1
            elif daily_pnl < 0:
                losing_days += 1
            
            # Track best and worst days
            if daily_pnl > best_pnl:
                best_pnl = daily_pnl
                all_time_metrics['best_day']['date'] = day_str
                all_time_metrics['best_day']['pnl'] = best_pnl
            
            if daily_pnl < worst_pnl:
                worst_pnl = daily_pnl
                all_time_metrics['worst_day']['date'] = day_str
                all_time_metrics['worst_day']['pnl'] = worst_pnl
            
            # Accumulate returns
            daily_return = day_stats['balance']['change_pct'] / 100  # Convert percentage to decimal
            all_daily_returns.append(daily_return)
            
            # Accumulate PnL
            all_time_metrics['cumulative_pnl'] += daily_pnl
            
            # Track fees
            daily_fees = sum(trade.get('cost', 0) for trade in day_stats['trade_details'] if 'cost' in trade)
            all_time_metrics['total_fees'] += daily_fees
            
            # Track PnL and trades by symbol
            for symbol, symbol_pnl in day_stats['pnl']['by_symbol'].items():
                if symbol not in all_time_metrics['pnl_by_symbol']:
                    all_time_metrics['pnl_by_symbol'][symbol] = 0.0
                    all_time_metrics['trades_by_symbol'][symbol] = 0
                
                all_time_metrics['pnl_by_symbol'][symbol] += symbol_pnl
                all_time_metrics['trades_by_symbol'][symbol] += day_stats['trades_by_symbol'].get(symbol, 0)
        
        # Update counts
        all_time_metrics['winning_days'] = winning_days
        all_time_metrics['losing_days'] = losing_days
        
        # Calculate win rate
        all_time_metrics['win_rate'] = winning_days / len(self.daily_stats) if self.daily_stats else 0.0
        
        # Calculate average daily PnL
        all_time_metrics['avg_daily_pnl'] = all_time_metrics['cumulative_pnl'] / len(self.daily_stats) if self.daily_stats else 0.0
        
        # Calculate Sharpe and Sortino ratios
        if all_daily_returns:
            returns_array = np.array(all_daily_returns)
            excess_returns = returns_array - self.risk_free_rate
            
            if len(excess_returns) > 1:
                returns_std = np.std(excess_returns, ddof=1)
                returns_mean = np.mean(excess_returns)
                
                # Sharpe ratio (annualized)
                if returns_std > 0:
                    all_time_metrics['sharpe_ratio'] = (returns_mean / returns_std) * np.sqrt(252)
                
                # Sortino ratio (annualized)
                downside_returns = excess_returns[excess_returns < 0]
                if len(downside_returns) > 1:
                    downside_std = np.std(downside_returns, ddof=1)
                    if downside_std > 0:
                        all_time_metrics['sortino_ratio'] = (returns_mean / downside_std) * np.sqrt(252)
        
        # Calculate maximum drawdown across all days
        # Convert daily stats to a DataFrame for cumulative metrics
        dates = []
        daily_values = []
        
        for day_str, day_stats in sorted(self.daily_stats.items()):
            dates.append(day_str)
            daily_values.append(day_stats['balance']['end'])
        
        if daily_values:
            # Create DataFrame with daily values
            df = pd.DataFrame({'value': daily_values}, index=dates)
            
            # Calculate running maximum
            df['peak'] = df['value'].cummax()
            
            # Calculate drawdown in dollars and percentage
            df['drawdown'] = df['peak'] - df['value']
            df['drawdown_pct'] = (df['drawdown'] / df['peak']) * 100
            
            # Get maximum drawdown
            all_time_metrics['max_drawdown'] = df['drawdown'].max()
            all_time_metrics['max_drawdown_pct'] = df['drawdown_pct'].max()
        
        # Calculate profit factor across all trades
        win_pnl = sum(day_stats['pnl']['total'] for day_stats in self.daily_stats.values() 
                    if day_stats['pnl']['total'] > 0)
        loss_pnl = abs(sum(day_stats['pnl']['total'] for day_stats in self.daily_stats.values() 
                         if day_stats['pnl']['total'] < 0))
        
        if loss_pnl > 0:
            all_time_metrics['profit_factor'] = win_pnl / loss_pnl
        else:
            all_time_metrics['profit_factor'] = float('inf') if win_pnl > 0 else 0.0
        
        # Store the updated all-time metrics
        self.all_days_stats = all_time_metrics
        
        # Save to disk
        summary_path = os.path.join(self.reports_dir, 'all_time_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_time_metrics, f, indent=2)
        
        # Generate summary chart
        chart_path = os.path.join(self.reports_dir, 'charts', 'all_time_performance.png')
        self._generate_all_time_chart(chart_path)
        
        return all_time_metrics
    
    def _generate_all_time_chart(self, output_path: str):
        """Generate all-time performance chart."""
        if not self.daily_stats:
            return
        
        try:
            # Extract dates and daily PnL values
            dates = []
            daily_pnl = []
            cumulative_pnl = []
            running_pnl = 0
            
            for day_str, day_stats in sorted(self.daily_stats.items()):
                dates.append(day_str)
                daily_pnl.append(day_stats['pnl']['total'])
                running_pnl += day_stats['pnl']['total']
                cumulative_pnl.append(running_pnl)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot daily PnL
            ax1.bar(dates, daily_pnl, color=['g' if p >= 0 else 'r' for p in daily_pnl])
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.set_title('Daily PnL')
            ax1.set_ylabel('PnL ($)')
            ax1.grid(axis='y', alpha=0.3)
            
            # Format x-axis for better readability
            if len(dates) > 10:
                ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot cumulative PnL
            ax2.plot(dates, cumulative_pnl, color='b', marker='o', markersize=4)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_title('Cumulative PnL')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative PnL ($)')
            ax2.grid(alpha=0.3)
            
            # Format x-axis for better readability
            if len(dates) > 10:
                ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            # Generate PnL by symbol chart
            symbol_chart_path = os.path.join(os.path.dirname(output_path), 'symbol_performance.png')
            self._generate_symbol_performance_chart(symbol_chart_path)
            
        except Exception as e:
            self.logger.error(f"Error generating all-time performance chart: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _generate_symbol_performance_chart(self, output_path: str):
        """Generate symbol performance chart."""
        if not self.all_days_stats or not self.all_days_stats.get('pnl_by_symbol'):
            return
        
        try:
            # Extract symbols and PnL values
            symbols = list(self.all_days_stats['pnl_by_symbol'].keys())
            pnl_values = list(self.all_days_stats['pnl_by_symbol'].values())
            
            # Sort by PnL
            sorted_data = sorted(zip(symbols, pnl_values), key=lambda x: x[1], reverse=True)
            sorted_symbols, sorted_pnl = zip(*sorted_data) if sorted_data else ([], [])
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(sorted_symbols, sorted_pnl, color=['g' if p >= 0 else 'r' for p in sorted_pnl])
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.,
                        height + (5 if height > 0 else -15),
                        f'${height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top')
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title('PnL by Symbol')
            plt.xlabel('Symbol')
            plt.ylabel('PnL ($)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating symbol performance chart: {str(e)}")
            self.logger.error(traceback.format_exc())

class RealTimeTrader:
    """
    Real-time trading system that loads a trained RL model and executes paper trades
    based on real-time market data.
    """
    def __init__(
        self,
        model_path: str,
        env_path: str = None,
        config_path: str = 'config/prod_config.yaml',
        initial_balance: float = 10000.0,
        historical_data_path: str = None,
        max_leverage: float = 20.0,
        websocket_port: int = 8765,
        save_trades_path: str = 'data/trades',
        backfill_days: int = 5,
        force_timeframe: str = None,  # Add parameter to force a specific timeframe
        resample_data: bool = True,   # Add parameter to enable/disable resampling
        enable_reports: bool = True   # New parameter to enable/disable reporting
    ):
        """
        Initialize the real-time trading system.
        
        Args:
            model_path: Path to the trained RL model
            env_path: Path to the saved trading environment (optional)
            config_path: Path to the configuration file
            initial_balance: Initial account balance for paper trading
            historical_data_path: Path to historical data for backfilling (optional)
            max_leverage: Maximum allowed leverage
            websocket_port: Port for websocket server
            save_trades_path: Directory to save trade logs
            backfill_days: Number of days of historical data to backfill
            force_timeframe: Force a specific timeframe for API calls (e.g., '5m')
            resample_data: Whether to resample 1m data to match training timeframe
            enable_reports: Whether to enable detailed reporting
        """
        self.model_path = model_path
        self.env_path = env_path
        self.config_path = config_path
        self.initial_balance = initial_balance
        self.historical_data_path = historical_data_path
        self.max_leverage = max_leverage
        self.websocket_port = websocket_port
        self.save_trades_path = save_trades_path
        self.backfill_days = backfill_days
        self.force_timeframe = force_timeframe
        self.resample_data = resample_data
        self.enable_reports = enable_reports
        
        # Create necessary directories
        os.makedirs(self.save_trades_path, exist_ok=True)
        
        # Create reports directory if enabled
        if self.enable_reports:
            self.reports_path = os.path.join(self.save_trades_path, 'reports')
            os.makedirs(self.reports_path, exist_ok=True)
            
            # Initialize report generator
            self.report_generator = DailyReportGenerator(self.reports_path)
            
            # Track last report generation time
            self.last_report_time = None
        
        # Load configuration
        self.config = self._load_config()
        self.symbols = self.config['trading']['symbols']
        
        # Determine timeframe handling
        # Default to 1m for live trading if not forced
        self.api_timeframe = force_timeframe or '1m'
        # Training timeframe from config (default to 5m which is typically used in training)
        self.training_timeframe = self.config['trading'].get('timeframe', '5m')
        
        # Log timeframe information
        logger.info(f"API data timeframe: {self.api_timeframe}")
        logger.info(f"Model training timeframe: {self.training_timeframe}")
        if self.resample_data and self.api_timeframe != self.training_timeframe:
            logger.info(f"Will resample {self.api_timeframe} data to {self.training_timeframe}")
        
        # Transaction costs from config
        self.commission = self.config['trading'].get('commission', 0.0004)
        
        # State variables
        self.is_running = False
        self.last_trade_time = {symbol: None for symbol in self.symbols}
        self.connected_clients = set()
        self.websocket_server = None
        
        # Initialize data structures
        self.market_data = {}  # Stores latest market data
        self.positions = {}  # Current positions
        self.trades = []  # Trade history
        self.pnl_history = []  # PnL history
        self.total_pnl = 0.0  # Total PnL
        self.balance = initial_balance  # Current balance
        self.total_value = initial_balance  # Current total value (balance + unrealized PnL)
        
        # Setup feature engine
        self.feature_engine = DerivativesFeatureEngine(
            volatility_window=self.config['feature_engineering']['volatility_window'],
            n_components=self.config['feature_engineering']['n_components']
        )
        
        # Initialize positions
        self._initialize_positions()
        
        # Initialize exchange client (for data only)
        self.exchange = ccxt_async.binance({
            'options': {
                'defaultType': 'future',
                'defaultMarket': 'linear',
                'defaultMarginMode': 'cross'
            },
            'enableRateLimit': True
        })
        
        # Binance-specific symbol mappings
        self.symbol_mappings = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        
        # Initialize asset data
        self.asset_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        
        # Initialize model and environment to None, will load later
        self.model = None
        self.env = None
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Initialize last observation time
        self.last_observation_time = datetime.now()
        
        # Run state
        self.shutting_down = False
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _initialize_positions(self):
        """Initialize position tracking structure."""
        self.positions = {
            symbol: {
                'size': 0.0,
                'entry_price': 0.0,
                'current_price': 0.0,
                'leverage': 0.0,
                'direction': 0,  # -1=short, 0=none, 1=long
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'timestamp': None,
                'value': 0.0,
                'liquidation_price': 0.0
            } for symbol in self.symbols
        }
    
    async def _setup_websocket_server(self):
        """Set up websocket server for real-time data streaming."""
        async def handler(websocket, path):
            logger.info(f"Client connected: {websocket.remote_address}")
            self.connected_clients.add(websocket)
            try:
                async for message in websocket:
                    # Handle client commands
                    command = json.loads(message)
                    if command.get('action') == 'get_status':
                        await websocket.send(json.dumps(self.get_status()))
                    elif command.get('action') == 'get_trades':
                        await websocket.send(json.dumps({'trades': self.trades[-100:]}))
                    elif command.get('action') == 'get_positions':
                        await websocket.send(json.dumps({'positions': self.positions}))
            except Exception as e:
                logger.error(f"Websocket error: {str(e)}")
            finally:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected: {websocket.remote_address}")
        
        self.websocket_server = await websockets.serve(handler, "0.0.0.0", self.websocket_port)
        logger.info(f"Websocket server started on port {self.websocket_port}")
    
    async def _broadcast_update(self, data):
        """Broadcast updates to all connected clients."""
        if not self.connected_clients:
            return
        
        message = json.dumps(data)
        websockets_to_remove = set()
        
        for websocket in self.connected_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                websockets_to_remove.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {websocket.remote_address}: {str(e)}")
                websockets_to_remove.add(websocket)
        
        # Remove closed connections
        self.connected_clients -= websockets_to_remove
    
    async def setup(self):
        """Set up the real-time trading system."""
        logger.info("Setting up real-time trading system...")
        
        # Load the RL model
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = PPO.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize risk engine with identical settings to those used in training
        # These settings should match what's in main_opt.py for consistency
        risk_limits = RiskLimits(
            account_max_leverage=self.max_leverage * 0.8,  # 80% of max to provide buffer
            position_max_leverage=self.max_leverage,
            max_drawdown_pct=self.config['risk_management']['limits'].get('max_drawdown', 0.2),
            position_concentration=self.config['risk_management']['limits'].get('position_concentration', 0.33),
            daily_loss_limit_pct=self.config['risk_management']['limits'].get('daily_loss_limit', 0.15)
        )
        
        self.risk_engine = InstitutionalRiskEngine(
            initial_balance=self.initial_balance,
            risk_limits=risk_limits,
            use_dynamic_limits=self.config['risk_management'].get('use_dynamic_limits', True),
            use_vol_scaling=self.config['risk_management'].get('use_vol_scaling', True)
        )
        
        # Load historical data or start with empty data
        if self.historical_data_path and os.path.exists(self.historical_data_path):
            await self._load_historical_data()
        else:
            # Backfill some historical data (important to match training data format)
            await self._backfill_data()
        
        # Create environment for inference - MUST MATCH TRAINING ENVIRONMENT
        await self._create_environment()
        
        # Setup websocket server
        await self._setup_websocket_server()
        
        logger.info("Real-time trading system setup complete")
    
    async def _load_historical_data(self):
        """Load historical data from file."""
        try:
            logger.info(f"Loading historical data from {self.historical_data_path}")
            
            # Load data based on file extension
            if self.historical_data_path.endswith('.parquet'):
                self.historical_df = pd.read_parquet(self.historical_data_path)
            elif self.historical_data_path.endswith('.csv'):
                self.historical_df = pd.read_csv(self.historical_data_path)
                # Convert timestamp to datetime if it's a string
                if isinstance(self.historical_df.index[0], str):
                    self.historical_df.index = pd.to_datetime(self.historical_df.index)
            else:
                raise ValueError(f"Unsupported file format: {self.historical_data_path}")
            
            # Check if we have a MultiIndex DataFrame
            if not isinstance(self.historical_df.columns, pd.MultiIndex):
                # Convert to MultiIndex if not already
                assets = self.symbols
                
                # Create empty dictionary to hold DataFrames for each asset
                asset_dfs = {}
                
                # Check if there are separate columns for each asset
                for asset in assets:
                    asset_cols = [col for col in self.historical_df.columns if asset in col]
                    if asset_cols:
                        # Extract asset data
                        asset_df = self.historical_df[asset_cols]
                        # Rename columns to remove asset prefix
                        asset_df.columns = [col.replace(f"{asset}_", "") for col in asset_df.columns]
                        asset_dfs[asset] = asset_df
                
                # If we couldn't find asset-specific columns, assume data is for all assets
                if not asset_dfs:
                    for asset in assets:
                        asset_dfs[asset] = self.historical_df.copy()
                
                # Create MultiIndex DataFrame
                dfs = []
                for asset, df in asset_dfs.items():
                    # Create MultiIndex columns
                    df.columns = pd.MultiIndex.from_product([[asset], df.columns])
                    dfs.append(df)
                
                self.historical_df = pd.concat(dfs, axis=1)
            
            # Make sure DataFrame is sorted by time
            self.historical_df = self.historical_df.sort_index()
            
            # Filter to recent data
            if self.backfill_days > 0:
                cutoff_date = datetime.now() - timedelta(days=self.backfill_days)
                self.historical_df = self.historical_df[self.historical_df.index >= cutoff_date]
            
            logger.info(f"Loaded historical data with shape: {self.historical_df.shape}")
            
            # Process data through feature engine
            self.historical_df = self.feature_engine.engineer_features({'binance': {sym: self.historical_df[sym] for sym in self.symbols}})
            
            logger.info(f"Processed historical data with shape: {self.historical_df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty DataFrames
            self.historical_df = pd.DataFrame()
            await self._backfill_data()
    
    async def _backfill_data(self):
        """Backfill data by fetching historical data from exchange."""
        logger.info(f"Backfilling {self.backfill_days} days of historical data...")
        
        try:
            # Load markets
            await self.exchange.load_markets()
            
            # Fetch historical data for each symbol
            backfill_data = {}
            for symbol in self.symbols:
                exchange_symbol = self.symbol_mappings.get(symbol, symbol)
                since = int((datetime.now() - timedelta(days=self.backfill_days)).timestamp() * 1000)
                
                all_ohlcv = []
                while True:
                    ohlcv = await self.exchange.fetch_ohlcv(exchange_symbol, self.timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    if len(ohlcv) < 1000:
                        break
                    since = ohlcv[-1][0] + 1
                    await asyncio.sleep(self.exchange.rateLimit / 1000)
                
                if not all_ohlcv:
                    logger.warning(f"No historical data found for {symbol}")
                    continue
                
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add funding rate if available
                try:
                    funding_since = since
                    all_funding = []
                    while True:
                        funding = await self.exchange.fetch_funding_rate_history(exchange_symbol, since=funding_since, limit=1000)
                        if not funding:
                            break
                        all_funding.extend(funding)
                        if len(funding) < 1000:
                            break
                        funding_since = funding[-1]['timestamp'] + 1
                        await asyncio.sleep(self.exchange.rateLimit / 1000)
                    
                    if all_funding:
                        funding_df = pd.DataFrame(all_funding)
                        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                        funding_df.set_index('timestamp', inplace=True)
                        df['funding_rate'] = funding_df['fundingRate']
                        df['funding_rate'] = df['funding_rate'].ffill()
                    else:
                        df['funding_rate'] = 0
                except Exception as e:
                    logger.warning(f"Could not fetch funding data for {symbol}: {str(e)}")
                    df['funding_rate'] = 0
                
                # Add other necessary columns
                df['bid_depth'] = 0
                df['ask_depth'] = 0
                
                backfill_data[symbol] = df
            
            # Create MultiIndex DataFrame
            dfs = []
            for symbol, df in backfill_data.items():
                # Create MultiIndex columns
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                dfs.append(df)
            
            if dfs:
                self.historical_df = pd.concat(dfs, axis=1)
                self.historical_df = self.historical_df.sort_index()
                
                # Process through feature engine
                self.historical_df = self.feature_engine.engineer_features({'binance': backfill_data})
                
                logger.info(f"Backfilled historical data with shape: {self.historical_df.shape}")
            else:
                logger.warning("Could not backfill data for any symbol")
                self.historical_df = pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error backfilling historical data: {str(e)}")
            logger.error(traceback.format_exc())
            self.historical_df = pd.DataFrame()
    
    async def _create_environment(self):
        """Create trading environment for inference that matches training environment."""
        logger.info("Creating trading environment for inference...")
        
        try:
            # Load environment with normalization stats if provided
            if self.env_path and os.path.exists(self.env_path):
                logger.info(f"Loading environment from: {self.env_path}")
                
                # First create base environment with identical settings to training
                # These settings should exactly match those used in InstitutionalPerpetualEnv during training
                base_env = InstitutionalPerpetualEnv(
                    df=self.historical_df,
                    assets=self.symbols,
                    window_size=100,  # Should match training window size
                    max_leverage=self.max_leverage,
                    commission=self.commission,
                    risk_engine=self.risk_engine,
                    initial_balance=self.initial_balance,
                    funding_fee_multiplier=self.config.get('trading', {}).get('funding_fee_multiplier', 1.0),
                    base_features=self.config.get('feature_engineering', {}).get('base_features', ['open', 'high', 'low', 'close', 'volume']),
                    tech_features=self.config.get('feature_engineering', {}).get('tech_features', [
                        'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
                        'atr', 'sma_10', 'ema_10', 'obv'
                    ]),
                    risk_free_rate=self.config.get('trading', {}).get('risk_free_rate', 0.02),
                    max_drawdown=self.config.get('risk_management', {}).get('limits', {}).get('max_drawdown', 0.2),
                    maintenance_margin=self.config.get('trading', {}).get('maintenance_margin', 0.05),
                    max_steps=10000,  # Large value for inference
                    verbose=False
                )
                
                # Wrap it in a DummyVecEnv
                vec_env = DummyVecEnv([lambda: base_env])
                
                # Load environment with normalization (CRITICAL for proper inference)
                self.env = VecNormalize.load(self.env_path, vec_env)
                
                # Reset training mode and reward normalization for inference
                self.env.training = False
                self.env.norm_reward = False
                
                logger.info("Environment loaded successfully with normalization stats")
            else:
                logger.warning("No environment file provided, creating new environment (NOT RECOMMENDED)")
                logger.warning("Performance may not match training without proper normalization stats")
                
                # Create a fresh environment - try to match training config
                base_env = InstitutionalPerpetualEnv(
                    df=self.historical_df,
                    assets=self.symbols,
                    window_size=100,  # Should match training window size
                    max_leverage=self.max_leverage,
                    commission=self.commission,
                    risk_engine=self.risk_engine,
                    initial_balance=self.initial_balance,
                    funding_fee_multiplier=self.config.get('trading', {}).get('funding_fee_multiplier', 1.0),
                    base_features=self.config.get('feature_engineering', {}).get('base_features', ['open', 'high', 'low', 'close', 'volume']),
                    tech_features=self.config.get('feature_engineering', {}).get('tech_features', [
                        'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
                        'atr', 'sma_10', 'ema_10', 'obv'
                    ]),
                    risk_free_rate=self.config.get('trading', {}).get('risk_free_rate', 0.02),
                    max_drawdown=self.config.get('risk_management', {}).get('limits', {}).get('max_drawdown', 0.2),
                    maintenance_margin=self.config.get('trading', {}).get('maintenance_margin', 0.05),
                    max_steps=10000,  # Large value for inference
                    verbose=False
                )
                
                # Wrap it in DummyVecEnv and VecNormalize
                self.env = VecNormalize(DummyVecEnv([lambda: base_env]))
                
                # Reset training mode for inference
                self.env.training = False
                self.env.norm_reward = False
                
                logger.info("New environment created successfully")
        
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _update_environment_data(self, new_data: pd.DataFrame):
        """Update environment with new data for inference."""
        with self.data_lock:
            try:
                # Get the underlying environment from the vectorized wrapper
                base_env = self.env.envs[0]
                
                # Update the environment's DataFrame
                base_env.df = pd.concat([base_env.df.iloc[:-1], new_data])
                
                # Update current step if needed
                base_env.current_step = len(base_env.df) - 1
                
                logger.debug("Environment data updated successfully")
            except Exception as e:
                logger.error(f"Error updating environment data: {str(e)}")
                logger.error(traceback.format_exc())
    
    async def _fetch_latest_data(self):
        """Fetch latest market data from exchange."""
        logger.info("Fetching latest market data...")
        
        try:
            latest_data = {}
            
            for symbol in self.symbols:
                exchange_symbol = self.symbol_mappings.get(symbol, symbol)
                
                # Fetch latest OHLCV candle using the API timeframe (typically 1m)
                ohlcv = await self.exchange.fetch_ohlcv(exchange_symbol, self.api_timeframe, limit=2)
                if not ohlcv or len(ohlcv) < 2:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    continue
                
                # Use the completed candle for decision making
                completed_candle = ohlcv[0]
                
                # Create DataFrame with single candle
                df = pd.DataFrame([completed_candle], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Get latest price for PnL calculation (current candle)
                current_price = float(ohlcv[1][4])  # Close price of current candle
                
                # Update position's current price
                self.positions[symbol]['current_price'] = current_price
                
                # Fetch latest funding rate
                try:
                    funding = await self.exchange.fetch_funding_rate(exchange_symbol)
                    df['funding_rate'] = funding['fundingRate'] if funding else 0
                except Exception as e:
                    logger.warning(f"Could not fetch funding data for {symbol}: {str(e)}")
                    df['funding_rate'] = 0
                
                # Add other necessary columns
                df['bid_depth'] = 0
                df['ask_depth'] = 0
                
                # Store dataframe
                latest_data[symbol] = df
                
                # Update last_prices for PnL calculation
                self.market_data[symbol] = {
                    'price': current_price,
                    'timestamp': df.index[0],
                    'ohlcv': {
                        'open': float(df['open'].iloc[0]),
                        'high': float(df['high'].iloc[0]),
                        'low': float(df['low'].iloc[0]),
                        'close': float(df['close'].iloc[0]),
                        'volume': float(df['volume'].iloc[0])
                    },
                    'funding_rate': float(df['funding_rate'].iloc[0])
                }
            
            # Create MultiIndex DataFrame for environment update
            dfs = []
            for symbol, df in latest_data.items():
                # Create MultiIndex columns
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, axis=1)
                
                # Check if we need to resample to training timeframe
                if self.resample_data and self.api_timeframe != self.training_timeframe:
                    # Fetch additional historical data for proper resampling
                    need_additional_data = True
                    # Log the resampling attempt
                    logger.info(f"Attempting to resample from {self.api_timeframe} to {self.training_timeframe}")
                    
                    # If we have historical data, use it for resampling context
                    if hasattr(self, 'historical_df') and not self.historical_df.empty:
                        # Get required number of bars for resampling to the training timeframe
                        if self.api_timeframe == '1m' and self.training_timeframe == '5m':
                            bars_needed = 5
                        elif self.api_timeframe == '1m' and self.training_timeframe == '15m':
                            bars_needed = 15
                        elif self.api_timeframe == '1m' and self.training_timeframe == '30m':
                            bars_needed = 30
                        elif self.api_timeframe == '1m' and self.training_timeframe == '1h':
                            bars_needed = 60
                        else:
                            bars_needed = 10  # Default fallback
                            
                        # Try to fetch additional bars for proper resampling
                        additional_bars = {}
                        for symbol in self.symbols:
                            exchange_symbol = self.symbol_mappings.get(symbol, symbol)
                            
                            try:
                                # Fetch additional bars for proper resampling
                                ohlcv_history = await self.exchange.fetch_ohlcv(
                                    exchange_symbol, 
                                    self.api_timeframe, 
                                    limit=bars_needed
                                )
                                
                                if ohlcv_history and len(ohlcv_history) >= bars_needed:
                                    df_hist = pd.DataFrame(
                                        ohlcv_history, 
                                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                    )
                                    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='ms')
                                    df_hist.set_index('timestamp', inplace=True)
                                    
                                    # Add missing columns that might be needed
                                    df_hist['funding_rate'] = latest_data[symbol]['funding_rate'].iloc[0] if 'funding_rate' in latest_data[symbol] else 0
                                    df_hist['bid_depth'] = 0
                                    df_hist['ask_depth'] = 0
                                    
                                    # Store historical bars
                                    additional_bars[symbol] = df_hist
                                else:
                                    logger.warning(f"Could not fetch enough historical bars for {symbol} resampling")
                            except Exception as e:
                                logger.error(f"Error fetching historical bars for {symbol}: {str(e)}")
                        
                        # If we got additional bars for all symbols, create combined historical data
                        if len(additional_bars) == len(self.symbols):
                            # Create MultiIndex DataFrame with historical data
                            hist_dfs = []
                            for symbol, df in additional_bars.items():
                                # Create MultiIndex columns
                                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                                hist_dfs.append(df)
                                
                            if hist_dfs:
                                historical_combined = pd.concat(hist_dfs, axis=1)
                                historical_combined = historical_combined.sort_index()
                                
                                # Resample to training timeframe
                                resampled = self._resample_ohlcv(historical_combined, self.training_timeframe)
                                
                                if not resampled.empty:
                                    # Process through feature engine
                                    processed_df = self.feature_engine.engineer_features(
                                        {'binance': {sym: additional_bars[sym] for sym in self.symbols}}
                                    )
                                    
                                    logger.info(f"Successfully resampled data to {self.training_timeframe}")
                                    return processed_df
                                else:
                                    logger.warning(f"Resampling produced empty DataFrame")
                    
                    # If we're here, either no historical data or resampling failed
                    logger.warning(f"Could not properly resample data, using raw {self.api_timeframe} data")
                
                # Process through feature engine for additional features
                processed_df = self.feature_engine.engineer_features({'binance': latest_data})
                
                logger.info(f"Fetched and processed latest data with timestamp: {processed_df.index[0]}")
                
                return processed_df
            else:
                logger.warning("Could not fetch latest data for any symbol")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _resample_ohlcv(self, df, target_timeframe):
        """
        Resample OHLCV data to a different timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '5m', '15m')
            
        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {target_timeframe}")
        
        try:
            # Convert the target_timeframe string to pandas offset alias
            # Map common trading timeframes
            timeframe_map = {
                '1m': '1min',
                '3m': '3min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1H',
                '2h': '2H',
                '4h': '4H',
                '6h': '6H',
                '8h': '8H',
                '12h': '12H',
                '1d': '1D',
            }
            
            if target_timeframe not in timeframe_map:
                logger.error(f"Unsupported target timeframe: {target_timeframe}")
                return pd.DataFrame()
            
            pandas_timeframe = timeframe_map[target_timeframe]
            
            # Create an empty DataFrame to store the resampled data
            resampled_dfs = []
            
            # Process each symbol separately
            for symbol in self.symbols:
                # Extract columns for this symbol
                symbol_cols = [col for col in df.columns if col[0] == symbol]
                if not symbol_cols:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Get the data for this symbol
                symbol_data = df[symbol_cols].copy()
                
                # Get the column names without the symbol prefix
                col_names = [col[1] for col in symbol_cols]
                
                # Flatten the MultiIndex columns temporarily for resampling
                symbol_data.columns = col_names
                
                # Resample OHLCV data
                resampled = pd.DataFrame()
                
                if 'open' in col_names and 'high' in col_names and 'low' in col_names and 'close' in col_names and 'volume' in col_names:
                    # Resample OHLCV data using proper aggregation methods
                    ohlcv_resampled = pd.DataFrame()
                    ohlcv_resampled['open'] = symbol_data['open'].resample(pandas_timeframe).first()
                    ohlcv_resampled['high'] = symbol_data['high'].resample(pandas_timeframe).max()
                    ohlcv_resampled['low'] = symbol_data['low'].resample(pandas_timeframe).min()
                    ohlcv_resampled['close'] = symbol_data['close'].resample(pandas_timeframe).last()
                    ohlcv_resampled['volume'] = symbol_data['volume'].resample(pandas_timeframe).sum()
                    
                    # Add to the main resampled DataFrame
                    resampled = ohlcv_resampled
                
                # Process other columns if they exist
                for col in col_names:
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        # For other columns, use mean as the aggregation method
                        # This is a reasonable default for most technical indicators
                        if col in symbol_data.columns:
                            resampled[col] = symbol_data[col].resample(pandas_timeframe).mean()
                
                # Restore the MultiIndex columns
                resampled.columns = pd.MultiIndex.from_product([[symbol], resampled.columns])
                
                # Add to the list of resampled DataFrames
                resampled_dfs.append(resampled)
            
            # Combine all resampled DataFrames
            if resampled_dfs:
                combined_resampled = pd.concat(resampled_dfs, axis=1)
                return combined_resampled
            else:
                logger.warning("No data was resampled")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    async def generate_signals(self, new_data: pd.DataFrame):
        """Generate trading signals using the RL model."""
        logger.info("Generating trading signals...")
        
        try:
            # Update environment with new data
            await self._update_environment_data(new_data)
            
            # Reset the environment to incorporate the new data
            obs = self.env.reset()
            
            # Get model prediction (action)
            # IMPORTANT: Use deterministic=False during trading to match training behavior
            # This allows the model to explore and consider uncertainty in its decision
            action, _ = self.model.predict(obs, deterministic=False)
            
            # Log the raw action vector
            logger.info(f"Raw action vector: {action}")
            
            # Process action into trading signals
            signals = {}
            for i, symbol in enumerate(self.symbols):
                # Get action value for this asset (-1 to 1 range typically)
                action_value = float(action[i])
                
                # Convert to trading signal
                if abs(action_value) < 0.2:  # Small threshold to avoid tiny positions
                    signal = 0  # No trade (hold)
                    signal_type = "HOLD"
                elif action_value > 0:
                    signal = action_value  # Long position
                    signal_type = "LONG"
                else:
                    signal = action_value  # Short position
                    signal_type = "SHORT"
                
                # Calculate target leverage from signal strength
                target_leverage = abs(action_value) * self.max_leverage
                
                signals[symbol] = {
                    'signal': action_value,
                    'signal_type': signal_type,
                    'target_leverage': target_leverage,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Signal for {symbol}: {signal_type} with leverage {target_leverage:.2f}x")
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    async def execute_trades(self, signals: Dict):
        """Execute paper trades based on generated signals."""
        logger.info("Executing paper trades based on signals...")
        
        trades_executed = []
        current_time = datetime.now()
        
        try:
            for symbol, signal_data in signals.items():
                signal = signal_data['signal']
                target_leverage = signal_data['target_leverage']
                signal_type = signal_data['signal_type']
                
                # Get current position
                position = self.positions[symbol]
                current_price = self.market_data[symbol]['price']
                
                # Determine if we need to execute a trade
                execute_trade = False
                trade_direction = 0
                
                if signal_type == "HOLD":
                    # No trade needed
                    pass
                elif signal_type == "LONG":
                    if position['direction'] <= 0:  # No position or short position
                        # Close existing short position if any
                        if position['direction'] < 0:
                            await self._close_position(symbol)
                        
                        # Open new long position
                        execute_trade = True
                        trade_direction = 1
                    elif position['leverage'] != target_leverage:
                        # Adjust leverage of existing long position
                        execute_trade = True
                        trade_direction = 1
                elif signal_type == "SHORT":
                    if position['direction'] >= 0:  # No position or long position
                        # Close existing long position if any
                        if position['direction'] > 0:
                            await self._close_position(symbol)
                        
                        # Open new short position
                        execute_trade = True
                        trade_direction = -1
                    elif position['leverage'] != target_leverage:
                        # Adjust leverage of existing short position
                        execute_trade = True
                        trade_direction = -1
                
                # Execute the trade if needed
                if execute_trade:
                    # Calculate position size based on target leverage
                    portfolio_value = self.balance + sum(p['unrealized_pnl'] for p in self.positions.values())
                    position_value = portfolio_value * target_leverage / len(self.symbols)
                    position_size = position_value / current_price
                    
                    # Apply direction
                    position_size *= trade_direction
                    
                    # Calculate transaction cost
                    transaction_cost = abs(position_value) * self.commission
                    
                    # Update position
                    self.positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'current_price': current_price,
                        'leverage': target_leverage,
                        'direction': trade_direction,
                        'unrealized_pnl': 0.0,
                        'realized_pnl': position.get('realized_pnl', 0.0),
                        'timestamp': current_time,
                        'value': position_value,
                        'liquidation_price': self._calculate_liquidation_price(
                            current_price,
                            trade_direction,
                            target_leverage
                        )
                    }
                    
                    # Record the trade
                    trade = {
                        'symbol': symbol,
                        'timestamp': current_time.isoformat(),
                        'action': 'BUY' if trade_direction > 0 else 'SELL',
                        'price': current_price,
                        'size': position_size,
                        'value': position_value,
                        'leverage': target_leverage,
                        'cost': transaction_cost,
                        'signal': signal
                    }
                    
                    self.trades.append(trade)
                    trades_executed.append(trade)
                    
                    # Update balance for transaction costs
                    self.balance -= transaction_cost
                    
                    logger.info(f"Executed {trade['action']} trade for {symbol} at price {current_price}, "
                               f"size {position_size:.6f}, value ${position_value:.2f}, leverage {target_leverage:.2f}x")
                
                # Update last trade time
                self.last_trade_time[symbol] = current_time
            
            # Update PnL after trades
            await self._update_pnl()
            
            return trades_executed
        
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def _close_position(self, symbol: str):
        """Close an existing position and realize PnL."""
        position = self.positions[symbol]
        
        if position['size'] == 0:
            logger.debug(f"No position to close for {symbol}")
            return
        
        current_price = self.market_data[symbol]['price']
        position_size = position['size']
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate PnL
        if direction > 0:  # Long position
            pnl = position_size * (current_price - entry_price)
        else:  # Short position
            pnl = position_size * (entry_price - current_price)
        
        # Calculate transaction cost
        position_value = abs(position_size * current_price)
        transaction_cost = position_value * self.commission
        
        # Update realized PnL
        realized_pnl = pnl - transaction_cost
        
        # Update position
        position['realized_pnl'] += realized_pnl
        position['size'] = 0
        position['entry_price'] = 0
        position['leverage'] = 0
        position['direction'] = 0
        position['unrealized_pnl'] = 0
        position['value'] = 0
        position['liquidation_price'] = 0
        
        # Update balance
        self.balance += realized_pnl
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'action': 'CLOSE',
            'price': current_price,
            'size': position_size,
            'value': position_value,
            'pnl': realized_pnl,
            'cost': transaction_cost
        }
        
        self.trades.append(trade)
        
        logger.info(f"Closed position for {symbol} at price {current_price}, "
                   f"realized PnL: ${realized_pnl:.2f}")
    
    def _calculate_liquidation_price(self, price: float, direction: int, leverage: float) -> float:
        """
        Calculate liquidation price for a position.
        
        For long positions: liquidation_price = entry_price * (1 - mm / leverage)
        For short positions: liquidation_price = entry_price * (1 + mm / leverage)
        
        Where mm is the maintenance margin (e.g., 0.05 for 5%).
        """
        maintenance_margin = 0.05  # 5% maintenance margin
        
        if direction > 0:  # Long position
            liquidation_price = price * (1 - maintenance_margin / leverage)
        elif direction < 0:  # Short position
            liquidation_price = price * (1 + maintenance_margin / leverage)
        else:  # No position
            liquidation_price = 0
        
        return liquidation_price
    
    async def _update_pnl(self):
        """Update unrealized PnL for all positions."""
        logger.debug("Updating PnL...")
        
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if position['size'] == 0:
                continue
            
            current_price = self.market_data[symbol]['price']
            position_size = position['size']
            entry_price = position['entry_price']
            direction = position['direction']
            
            # Calculate unrealized PnL
            if direction > 0:  # Long position
                unrealized_pnl = position_size * (current_price - entry_price)
            else:  # Short position
                unrealized_pnl = position_size * (entry_price - current_price)
            
            # Update position
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
            # Add to total
            total_unrealized_pnl += unrealized_pnl
        
        # Update total value
        self.total_value = self.balance + total_unrealized_pnl
        
        # Record PnL history
        self.pnl_history.append({
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'unrealized_pnl': total_unrealized_pnl,
            'total_value': self.total_value
        })
        
        # Keep only the last 10000 PnL records (about 1 week at 1-minute intervals)
        if len(self.pnl_history) > 10000:
            self.pnl_history = self.pnl_history[-10000:]
        
        logger.debug(f"Updated PnL - Balance: ${self.balance:.2f}, "
                    f"Unrealized PnL: ${total_unrealized_pnl:.2f}, "
                    f"Total Value: ${self.total_value:.2f}")
    
    async def run_trading_loop(self):
        """Run the main trading loop."""
        logger.info("Starting real-time trading loop...")
        self.is_running = True
        
        try:
            while self.is_running and not self.shutting_down:
                current_time = datetime.now()
                
                # Only fetch new data if it's a new minute
                if (current_time - self.last_observation_time).total_seconds() >= 60:
                    logger.info(f"Fetching data at {current_time}")
                    
                    # Fetch latest data
                    new_data = await self._fetch_latest_data()
                    
                    if new_data is not None and not new_data.empty:
                        # Generate signals
                        signals = await self.generate_signals(new_data)
                        
                        # Execute trades based on signals
                        trades = await self.execute_trades(signals)
                        
                        # Save trade data
                        self._save_trade_data()
                        
                        # Broadcast update to connected clients
                        await self._broadcast_update({
                            'type': 'update',
                            'timestamp': current_time.isoformat(),
                            'positions': self.positions,
                            'balance': self.balance,
                            'total_value': self.total_value,
                            'trades': trades
                        })
                        
                        # Check if we need to generate a daily report
                        if self.enable_reports:
                            # Generate daily reports at midnight or when the day changes
                            if (self.last_report_time is None or 
                                current_time.date() > self.last_report_time.date()):
                                
                                # If we have a previous day to report on
                                if self.last_report_time is not None:
                                    previous_day = self.last_report_time.date()
                                    await self._generate_daily_report(previous_day)
                                
                                # Update last report time
                                self.last_report_time = current_time
                        
                        # Update last observation time
                        self.last_observation_time = current_time
                    else:
                        logger.warning("No data fetched, skipping this iteration")
                
                # Sleep until next minute
                seconds_to_next_minute = 60 - datetime.now().second
                await asyncio.sleep(seconds_to_next_minute)
        
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Generate final report for the current day
            if self.enable_reports and self.last_report_time is not None:
                current_day = datetime.now().date()
                await self._generate_daily_report(current_day)
            
            logger.info("Trading loop stopped")
            self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current trading status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'balance': self.balance,
            'total_value': self.total_value,
            'positions': self.positions,
            'market_data': self.market_data,
            'last_trade_time': {k: v.isoformat() if v else None for k, v in self.last_trade_time.items()},
            'pnl_history': self.pnl_history[-100:]  # Return last 100 PnL records
        }
    
    def _save_trade_data(self):
        """Save trade data to file."""
        try:
            # Save trades
            trades_file = os.path.join(self.save_trades_path, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
            
            # Save positions
            positions_file = os.path.join(self.save_trades_path, 'positions.json')
            with open(positions_file, 'w') as f:
                # Convert any datetime objects to ISO format strings
                positions_json = {}
                for symbol, position in self.positions.items():
                    position_copy = position.copy()
                    if position_copy.get('timestamp') and isinstance(position_copy['timestamp'], datetime):
                        position_copy['timestamp'] = position_copy['timestamp'].isoformat()
                    positions_json[symbol] = position_copy
                json.dump(positions_json, f, indent=2)
            
            # Save PnL history
            pnl_file = os.path.join(self.save_trades_path, 'pnl_history.json')
            with open(pnl_file, 'w') as f:
                json.dump(self.pnl_history, f, indent=2)
            
            # Save current status
            status_file = os.path.join(self.save_trades_path, 'status.json')
            with open(status_file, 'w') as f:
                status = self.get_status()
                # Convert any datetime objects to ISO format strings
                if status.get('last_trade_time'):
                    status['last_trade_time'] = {k: v for k, v in status['last_trade_time'].items()}
                json.dump(status, f, indent=2)
            
            logger.debug("Trade data saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving trade data: {str(e)}")
    
    async def shutdown(self):
        """Gracefully shutdown the trading system."""
        logger.info("Shutting down real-time trading system...")
        
        self.shutting_down = True
        self.is_running = False
        
        # Close all positions
        for symbol in self.symbols:
            await self._close_position(symbol)
        
        # Save final state
        self._save_trade_data()
        
        # Generate final report if reporting is enabled
        if self.enable_reports:
            current_day = datetime.now().date()
            await self._generate_daily_report(current_day)
        
        # Close exchange connection
        await self.exchange.close()
        
        # Close websocket server if active
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("Real-time trading system shutdown complete")

    async def _generate_daily_report(self, day: date):
        """Generate a daily trading report."""
        if not self.enable_reports:
            return
        
        logger.info(f"Generating daily report for {day}")
        
        try:
            # Compute start balance for the day
            # Use previous day's end balance or initial balance
            day_str = day.strftime('%Y-%m-%d')
            
            # Check if we have existing reports to determine start balance
            all_reports = self.report_generator.daily_stats
            previous_days = [d for d in all_reports.keys() if d < day_str]
            
            if previous_days:
                latest_previous_day = max(previous_days)
                start_balance = all_reports[latest_previous_day]['balance']['end']
            else:
                start_balance = self.initial_balance
            
            # Process the day's trades
            day_metrics = self.report_generator.process_day_trades(
                trades=self.trades,
                positions=self.positions,
                pnl_history=self.pnl_history,
                start_balance=start_balance,
                symbols=self.symbols,
                day=day
            )
            
            # Generate the report file
            report_path = self.report_generator.generate_daily_report(day)
            
            # Update all-time statistics
            all_time_stats = self.report_generator.update_all_time_stats()
            
            logger.info(f"Daily report generated at {report_path}")
            
            return report_path
        
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            logger.error(traceback.format_exc())
            return None

async def main():
    """Main function to set up and run the real-time trader."""
    parser = argparse.ArgumentParser(description='Real-time RL Paper Trading')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained RL model')
    parser.add_argument('--env', type=str, help='Path to the saved trading environment')
    parser.add_argument('--config', type=str, default='config/prod_config.yaml', help='Path to configuration file')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance for paper trading')
    parser.add_argument('--historical-data', type=str, help='Path to historical data file')
    parser.add_argument('--max-leverage', type=float, default=20.0, help='Maximum allowed leverage')
    parser.add_argument('--websocket-port', type=int, default=8765, help='Websocket server port')
    parser.add_argument('--save-path', type=str, default='data/trades', help='Directory to save trade logs')
    parser.add_argument('--backfill-days', type=int, default=5, help='Number of days to backfill data')
    
    # Add new timeframe control parameters
    parser.add_argument('--force-timeframe', type=str, help='Force a specific timeframe for API calls (e.g., "5m")')
    parser.add_argument('--disable-resampling', action='store_true', help='Disable automatic resampling of 1m to 5m data')
    
    # Add reporting control
    parser.add_argument('--disable-reports', action='store_true', help='Disable detailed daily reports generation')
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    trader = None
    
    def handle_shutdown(sig, frame):
        nonlocal trader
        if trader:
            logger.info(f"Received shutdown signal: {sig}")
            if not trader.shutting_down:
                asyncio.create_task(trader.shutdown())
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Initialize trader
    trader = RealTimeTrader(
        model_path=args.model,
        env_path=args.env,
        config_path=args.config,
        initial_balance=args.balance,
        historical_data_path=args.historical_data,
        max_leverage=args.max_leverage,
        websocket_port=args.websocket_port,
        save_trades_path=args.save_path,
        backfill_days=args.backfill_days,
        force_timeframe=args.force_timeframe,
        resample_data=not args.disable_resampling,
        enable_reports=not args.disable_reports
    )
    
    # Set up trader
    await trader.setup()
    
    # Run trading loop
    await trader.run_trading_loop()
    
    # Ensure proper shutdown
    await trader.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 