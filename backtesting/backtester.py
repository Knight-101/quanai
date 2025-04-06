"""
Institutional Backtester Module

This module provides the core InstitutionalBacktester class for conducting
comprehensive backtests on trading models.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("institutional_backtester")

# Import internal modules 
from .regime_analyzer import RegimeAnalyzer, MarketRegime, RegimePeriod
from .metrics import calculate_performance_metrics, calculate_trade_statistics
from .visualization import create_visualization_suite


class InstitutionalBacktester:
    """
    Institutional-grade backtesting system for cryptocurrency trading models.
    
    This class provides a comprehensive framework for backtesting trading models,
    with features for regime analysis, walk-forward validation, and detailed metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str = None,
        data_df: pd.DataFrame = None,
        assets: List[str] = None,
        initial_capital: float = 100000.0,
        start_date: str = None,
        end_date: str = None,
        risk_free_rate: float = 0.02,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        max_leverage: float = 10.0,
        output_dir: str = "backtest_results",
        verbose: bool = True,
        benchmark: str = None,
        window_size: int = 100,
        regime_analysis: bool = True
    ):
        """
        Initialize the institutional backtester.
        
        Args:
            model_path: Path to the trained model
            data_path: Path to market data file (parquet, csv, pickle)
            data_df: DataFrame with market data (alternative to data_path)
            assets: List of assets to trade
            initial_capital: Initial capital for backtesting
            start_date: Start date for backtesting (format: YYYY-MM-DD)
            end_date: End date for backtesting (format: YYYY-MM-DD)
            risk_free_rate: Annual risk-free rate
            commission: Trading commission
            slippage: Trading slippage
            max_leverage: Maximum allowed leverage
            output_dir: Directory for output files
            verbose: Whether to print progress information
            benchmark: Symbol for benchmark comparison
            window_size: Number of data points for observation window
            regime_analysis: Whether to analyze market regimes
        """
        # Store configuration
        self.model_path = model_path
        self.data_path = data_path
        self.data_df = data_df
        self.assets = assets
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.commission = commission
        self.slippage = slippage
        self.max_leverage = max_leverage
        self.output_dir = output_dir
        self.verbose = verbose
        self.benchmark = benchmark
        self.window_size = window_size
        self.regime_analysis = regime_analysis
        
        # Initialize other attributes
        self.model = None
        self.env = None
        self.data = None
        self.regime_analyzer = None
        self.regimes = None
        
        # Set up logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log initialization
        logger.info(f"Backtester initialized with model: {model_path}")
    
    def _setup_logging(self):
        """Configure logging based on verbosity setting"""
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
    
    def load_data(self):
        """Load and preprocess market data"""
        if self.data is not None:
            logger.info("Data already loaded")
            return self.data
            
        if self.data_df is not None:
            logger.info("Using provided DataFrame")
            data = self.data_df
        elif self.data_path:
            logger.info(f"Loading data from {self.data_path}")
            
            # Determine file format and load accordingly
            file_ext = os.path.splitext(self.data_path)[1].lower()
            if file_ext == '.parquet':
                data = pd.read_parquet(self.data_path)
            elif file_ext == '.csv':
                data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            elif file_ext in ['.pkl', '.pickle']:
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        else:
            raise ValueError("Either data_path or data_df must be provided")
        
        # Filter by date range if provided
        if self.start_date or self.end_date:
            logger.info(f"Filtering data by date range: {self.start_date} to {self.end_date}")
            if isinstance(data.index, pd.DatetimeIndex):
                if self.start_date:
                    data = data[data.index >= self.start_date]
                if self.end_date:
                    data = data[data.index <= self.end_date]
            else:
                logger.warning("Data index is not DatetimeIndex, cannot filter by date")
        
        # Detect assets if not provided
        if self.assets is None and isinstance(data.columns, pd.MultiIndex):
            self.assets = list(data.columns.get_level_values(0).unique())
            logger.info(f"Detected assets: {self.assets}")
        
        self.data = data
        logger.info(f"Loaded data with shape: {data.shape}")
        
        return data
    
    def load_model(self):
        """Load the trained model"""
        if self.model is not None:
            logger.info("Model already loaded")
            return self.model
            
        try:
            logger.info(f"Loading model from {self.model_path}")
            # Load model using stable-baselines3
            self.model = PPO.load(self.model_path)
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _create_environment(self):
        """Create a backtesting environment similar to the training environment"""
        try:
            # Import locally to avoid circular imports
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from trading_env.institutional_perp_env import InstitutionalPerpetualEnv

            # Check if data is loaded
            if self.data is None:
                self.load_data()
            
            # Create environment with evaluation settings
            logger.info("Creating backtesting environment")
            env = InstitutionalPerpetualEnv(
                df=self.data,
                assets=self.assets,
                window_size=self.window_size,
                max_leverage=self.max_leverage,
                commission=self.commission,
                initial_balance=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
                verbose=False  # Disable verbose mode for backtesting
            )
            
            # Wrap in DummyVecEnv as required by stable-baselines3
            vec_env = DummyVecEnv([lambda: env])
            
            # If we have a normalization file, load it
            norm_path = os.path.join(os.path.dirname(self.model_path), "vec_normalize.pkl")
            if os.path.exists(norm_path):
                logger.info(f"Loading normalization from {norm_path}")
                vec_env = VecNormalize.load(norm_path, vec_env)
                
                # Disable training-time features
                vec_env.norm_reward = False  # Do not normalize rewards during evaluation
                vec_env.training = False     # Do not update normalization statistics
            
            self.env = vec_env
            return vec_env
        except ImportError:
            logger.error("Could not import InstitutionalPerpetualEnv. Make sure trading_env module is available.")
            raise
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise
    
    def analyze_market_regimes(self):
        """Analyze market data to identify different market regimes"""
        if not self.regime_analysis:
            logger.info("Market regime analysis is disabled")
            return None
            
        if self.regimes is not None:
            logger.info("Market regimes already analyzed")
            return self.regimes
            
        # Check if data is loaded
        if self.data is None:
            self.load_data()
            
        logger.info("Analyzing market regimes")
        
        # Create regime analyzer
        self.regime_analyzer = RegimeAnalyzer(
            volatility_window=20,
            trend_window=50,
            high_vol_threshold=0.8,
            low_vol_threshold=0.2,
            trend_threshold=0.6,
            crisis_threshold=0.95
        )
        
        # Analyze market data
        self.regimes = self.regime_analyzer.analyze_market_data(
            df=self.data,
            price_col='close',
            asset_level=True if self.assets and len(self.assets) > 1 else False
        )
        
        # Log identified regimes
        if self.regimes:
            for asset, regime_periods in self.regimes.items():
                logger.info(f"Identified {len(regime_periods)} regime periods for {asset}")
                for i, period in enumerate(regime_periods[:3]):  # Log first 3 periods
                    logger.info(f"  Regime {i+1}: {period.regime.value} from {period.start_date.date()} to {period.end_date.date()}")
                if len(regime_periods) > 3:
                    logger.info(f"  ... and {len(regime_periods) - 3} more periods")
        
        return self.regimes
    
    def run_backtest(self):
        """
        Run a full backtest on the loaded model and data.
        
        Returns:
            Dict containing backtest results
        """
        logger.info("Starting backtest")
        start_time = time.time()
        
        # Load model and data if not already loaded
        if self.model is None:
            self.load_model()
        
        if self.data is None:
            self.load_data()
        
        # Create environment if not already created
        if self.env is None:
            self._create_environment()
        
        # Run the backtest simulation
        results = self._run_backtest_simulation()
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        results.update({'metrics': metrics})
        
        # Save results
        self._save_results(results, 'backtest_results')
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Log completion
        duration = time.time() - start_time
        logger.info(f"Backtest completed in {duration:.2f} seconds")
        
        return results 
    
    def _run_backtest_simulation(self):
        """
        Run the actual backtest simulation.
        
        Returns:
            Dict containing raw backtest results
        """
        logger.info("Running backtest simulation")
        
        # Reset the environment and get initial observation
        obs = self.env.reset()
        
        # Initialize result tracking
        timestamps = []
        portfolio_values = []
        returns = []
        positions = []
        leverages = []
        trades = []
        actions = []
        observations = []
        rewards = []
        
        # Get the base env (unwrap from VecEnv)
        if hasattr(self.env, 'venv'):
            base_env = self.env.venv.envs[0]
        else:
            base_env = self.env.envs[0]
        
        # Track regime periods if enabled
        regime_periods = None
        if self.regime_analysis and self.regimes:
            # Use the market-level regimes or first asset's regimes
            if 'MARKET' in self.regimes:
                regime_periods = self.regimes['MARKET']
            elif self.assets and self.assets[0] in self.regimes:
                regime_periods = self.regimes[self.assets[0]]
        
        # Run simulation
        done = False
        total_steps = len(self.data) - self.window_size if hasattr(self.data, '__len__') else 1000
        
        with tqdm(total=total_steps, disable=not self.verbose) as pbar:
            while not done:
                # Predict action using the model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute the action
                next_obs, reward, done, info = self.env.step(action)
                
                # Get step info from the base environment
                step_info = info[0] if isinstance(info, list) else info
                
                # Track results
                if hasattr(base_env, 'current_time'):
                    timestamps.append(base_env.current_time)
                elif hasattr(base_env, 'current_step'):
                    timestamps.append(base_env.current_step)
                else:
                    timestamps.append(len(portfolio_values))
                
                # Track portfolio value
                if 'portfolio_value' in step_info:
                    portfolio_values.append(step_info['portfolio_value'])
                    
                    # Calculate returns
                    if len(portfolio_values) > 1:
                        ret = portfolio_values[-1] / portfolio_values[-2] - 1
                        returns.append(ret)
                    else:
                        returns.append(0.0)
                
                # Track positions
                if 'positions' in step_info:
                    positions.append(step_info['positions'])
                
                # Track leverage
                if 'leverage' in step_info:
                    leverages.append(step_info['leverage'])
                elif 'current_leverage' in step_info:
                    leverages.append(step_info['current_leverage'])
                
                # Track trades
                if 'trades' in step_info and step_info['trades']:
                    for trade in step_info['trades']:
                        trade_info = trade.copy()
                        trade_info['step'] = len(portfolio_values) - 1
                        trade_info['timestamp'] = timestamps[-1]
                        trades.append(trade_info)
                
                # Track actions, observations, and rewards
                actions.append(action)
                observations.append(obs)
                rewards.append(reward)
                
                # Update observation
                obs = next_obs
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Portfolio: ${portfolio_values[-1]:.2f}")
                
                # Break if done
                if done:
                    break
        
        # Process drawdowns
        drawdowns = self._calculate_drawdowns(portfolio_values)
        
        # Process benchmark data if available
        benchmark_values = None
        benchmark_returns = None
        if self.benchmark and self.data is not None:
            if isinstance(self.data.columns, pd.MultiIndex):
                benchmark_col = (self.benchmark, 'close')
                if benchmark_col in self.data.columns:
                    benchmark_series = self.data[benchmark_col].iloc[self.window_size:]
                    if len(benchmark_series) >= len(portfolio_values):
                        benchmark_series = benchmark_series[:len(portfolio_values)]
                        # Normalize to start at same value as portfolio
                        benchmark_values = benchmark_series / benchmark_series.iloc[0] * portfolio_values[0]
                        benchmark_returns = benchmark_values.pct_change().fillna(0).values
            else:
                # Attempt to find benchmark in flat columns
                if self.benchmark in self.data.columns:
                    benchmark_series = self.data[self.benchmark].iloc[self.window_size:]
                    if len(benchmark_series) >= len(portfolio_values):
                        benchmark_series = benchmark_series[:len(portfolio_values)]
                        # Normalize to start at same value as portfolio
                        benchmark_values = benchmark_series / benchmark_series.iloc[0] * portfolio_values[0]
                        benchmark_returns = benchmark_values.pct_change().fillna(0).values
        
        # Compile results
        results = {
            'timestamps': timestamps,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'positions': positions,
            'leverages': leverages,
            'trades': trades,
            'actions': actions,
            'observations': observations,
            'rewards': rewards,
            'drawdowns': drawdowns,
            'regime_periods': regime_periods,
            'benchmark_values': benchmark_values,
            'benchmark_returns': benchmark_returns
        }
        
        logger.info(f"Backtest simulation completed with {len(portfolio_values)} steps")
        
        return results
    
    def _calculate_drawdowns(self, portfolio_values):
        """Calculate drawdowns from portfolio values"""
        if not portfolio_values:
            return []
            
        # Convert to numpy array
        values = np.array(portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max
        
        return drawdown.tolist()
    
    def _calculate_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        logger.info("Calculating performance metrics")
        
        # Extract required data
        portfolio_values = results.get('portfolio_values', [])
        timestamps = results.get('timestamps', None)
        returns = results.get('returns', [])
        trades = results.get('trades', [])
        leverages = results.get('leverages', [])
        benchmark_returns = results.get('benchmark_returns', None)
        
        # Convert returns to numpy array
        returns = np.array(returns)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            portfolio_values=portfolio_values,
            timestamps=timestamps,
            trades=trades,
            leverages=leverages,
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.risk_free_rate
        )
        
        # Calculate trade statistics if we have trades
        if trades:
            trade_metrics = calculate_trade_statistics(trades)
            metrics.update(trade_metrics)
        
        # Log key metrics
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        if trades:
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        return metrics
    
    def _create_visualizations(self, results):
        """Create visualizations for backtest results"""
        logger.info("Creating visualizations")
        
        # Create and save visualization suite
        figures = create_visualization_suite(
            results=results,
            output_dir=self.output_dir,
            prefix="",
            create_tearsheet=True
        )
        
        logger.info(f"Created {len(figures)} visualization figures")
        
        return figures
    
    def _save_results(self, results, prefix=""):
        """Save backtest results to disk"""
        logger.info("Saving backtest results")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract data to save
        metrics = results.get('metrics', {})
        portfolio_values = results.get('portfolio_values', [])
        returns = results.get('returns', [])
        timestamps = results.get('timestamps', [])
        trades = results.get('trades', [])
        
        # Save metrics to JSON
        metrics_file = os.path.join(self.output_dir, f"{prefix}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: str(x) if isinstance(x, (datetime, np.float32, np.float64, np.int64)) else x)
        
        # Save portfolio values to CSV
        if portfolio_values and timestamps:
            portfolio_file = os.path.join(self.output_dir, f"{prefix}_portfolio.csv")
            portfolio_df = pd.DataFrame({
                'timestamp': timestamps,
                'value': portfolio_values,
                'return': [0] + returns if returns else [0] * len(portfolio_values)
            })
            portfolio_df.to_csv(portfolio_file, index=False)
        
        # Save trades to CSV
        if trades:
            trades_file = os.path.join(self.output_dir, f"{prefix}_trades.csv")
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
        
        return self.output_dir
    
    def save_results(self, results, prefix=""):
        """Public method to save results"""
        return self._save_results(results, prefix)
        
    def create_performance_plots(self, results, save_to_file=True):
        """Create and optionally save performance plots"""
        if save_to_file:
            return self._create_visualizations(results)
        else:
            # Just create the plots but don't save them
            portfolio_values = results.get('portfolio_values', [])
            timestamps = results.get('timestamps', [])
            returns = results.get('returns', [])
            
            plt.figure(figsize=(12, 8))
            plt.plot(timestamps, portfolio_values)
            plt.title('Portfolio Value')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            if returns:
                plt.figure(figsize=(12, 8))
                plt.hist(returns, bins=50, alpha=0.75)
                plt.axvline(np.mean(returns), color='r', linestyle='dashed', linewidth=1)
                plt.title('Returns Distribution')
                plt.xlabel('Return')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            
            return None
    
    def run_regime_analysis(self):
        """
        Run regime-specific backtests and analysis.
        
        Returns:
            Dict containing regime analysis results
        """
        if not self.regime_analysis:
            logger.warning("Market regime analysis is disabled")
            return None
            
        logger.info("Running regime analysis")
        
        # Make sure regimes are analyzed
        if self.regimes is None:
            self.analyze_market_regimes()
        
        if not self.regimes:
            logger.warning("No market regimes identified, skipping regime analysis")
            return None
        
        # Get primary regime periods
        regime_key = 'MARKET' if 'MARKET' in self.regimes else (self.assets[0] if self.assets else None)
        if not regime_key or regime_key not in self.regimes:
            logger.warning("Could not find suitable regime periods for analysis")
            return None
            
        regime_periods = self.regimes[regime_key]
        
        # Get baseline backtest results if not already run
        baseline_results = self.run_backtest()
        
        # Group portfolio values and returns by regime
        regime_performance = {}
        regime_metrics = {}
        
        # Get the timestamps and portfolio values
        timestamps = baseline_results.get('timestamps', [])
        portfolio_values = baseline_results.get('portfolio_values', [])
        returns = baseline_results.get('returns', [])
        
        if not timestamps or not portfolio_values:
            logger.warning("No timestamp or portfolio value data for regime analysis")
            return None
        
        # Convert timestamps to consistent format
        if timestamps and not isinstance(timestamps[0], (datetime, pd.Timestamp)):
            # Convert step indices to corresponding timestamps from the data
            if self.data is not None and isinstance(self.data.index, pd.DatetimeIndex):
                timestamps = [self.data.index[self.window_size + i] for i in range(len(timestamps)) if self.window_size + i < len(self.data.index)]
                if len(timestamps) != len(portfolio_values):
                    # Truncate the longer list to match the shorter one
                    min_length = min(len(timestamps), len(portfolio_values))
                    timestamps = timestamps[:min_length]
                    portfolio_values = portfolio_values[:min_length]
                    returns = returns[:min_length] if returns else []
        
        # Create timestamp index for efficient lookup
        timestamps_idx = {pd.Timestamp(t) if not isinstance(t, (datetime, pd.Timestamp)) else t: i 
                          for i, t in enumerate(timestamps)}
        
        # Analyze each regime
        for period in regime_periods:
            regime_name = period.regime.value
            
            # Find start and end indices
            start_ts = period.start_date
            end_ts = period.end_date
            
            start_idx = None
            end_idx = None
            
            # Find closest timestamps if exact match not found
            if start_ts not in timestamps_idx:
                closest_ts = min(timestamps_idx.keys(), key=lambda x: abs((x - start_ts).total_seconds()))
                start_idx = timestamps_idx[closest_ts]
            else:
                start_idx = timestamps_idx[start_ts]
                
            if end_ts not in timestamps_idx:
                closest_ts = min(timestamps_idx.keys(), key=lambda x: abs((x - end_ts).total_seconds()))
                end_idx = timestamps_idx[closest_ts]
            else:
                end_idx = timestamps_idx[end_ts]
            
            # Skip if invalid indices
            if start_idx is None or end_idx is None or start_idx >= end_idx:
                continue
                
            # Extract regime-specific data
            regime_timestamps = timestamps[start_idx:end_idx+1]
            regime_portfolio = portfolio_values[start_idx:end_idx+1]
            regime_returns = returns[start_idx:end_idx+1] if returns else []
            
            # Store in results
            regime_performance[regime_name] = {
                'timestamps': regime_timestamps,
                'portfolio_values': regime_portfolio,
                'returns': regime_returns,
                'start_date': start_ts,
                'end_date': end_ts
            }
            
            # Calculate regime-specific metrics
            if regime_returns:
                # Calculate key metrics
                total_return = regime_portfolio[-1] / regime_portfolio[0] - 1
                annualized_return = total_return / (len(regime_returns) / 252)
                volatility = np.std(regime_returns) * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Calculate drawdowns
                drawdowns = self._calculate_drawdowns(regime_portfolio)
                max_drawdown = min(drawdowns) if drawdowns else 0
                
                # Store metrics
                regime_metrics[regime_name] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'num_days': len(regime_returns),
                    'start_date': start_ts,
                    'end_date': end_ts
                }
                
                logger.info(f"Regime {regime_name}: Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Period: {len(regime_returns)} days")
        
        # Compile results
        regime_results = {
            'regime_performance': regime_performance,
            'regime_metrics': regime_metrics,
            'regime_periods': regime_periods
        }
        
        # Save results
        self._save_results({
            'regime_metrics': regime_metrics,
            'regime_periods': [{'regime': r.regime.value, 'start_date': r.start_date, 'end_date': r.end_date} 
                               for r in regime_periods]
        }, 'regime_analysis')
        
        return regime_results
    
    def create_regime_plots(self, regime_results, save_to_file=True):
        """Create regime analysis visualizations"""
        if not regime_results or 'regime_performance' not in regime_results:
            logger.warning("No regime performance data for visualization")
            return None
            
        regime_performance = regime_results.get('regime_performance', {})
        regime_metrics = regime_results.get('regime_metrics', {})
        
        if not regime_performance or not regime_metrics:
            logger.warning("Empty regime performance or metrics")
            return None
            
        # Create output directory if needed
        if save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Plot regime performance comparison
        plt.figure(figsize=(12, 8))
        
        # Sort regimes by performance
        sorted_regimes = sorted(regime_metrics.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        # Bar chart of returns
        regime_names = [r[0] for r in sorted_regimes]
        returns = [r[1]['total_return'] for r in sorted_regimes]
        sharpes = [r[1]['sharpe_ratio'] for r in sorted_regimes]
        
        x = range(len(regime_names))
        
        # Create figure with secondary y-axis
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot bars for returns
        bars = ax1.bar(x, [r * 100 for r in returns], alpha=0.7)
        
        # Color bars based on positive/negative returns
        for i, ret in enumerate(returns):
            bars[i].set_color('green' if ret >= 0 else 'red')
            
        ax1.set_xlabel('Market Regime')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Performance by Market Regime')
        
        # Create secondary y-axis for Sharpe
        ax2 = ax1.twinx()
        ax2.plot(x, sharpes, 'o-', color='blue', linewidth=2, markersize=8)
        ax2.set_ylabel('Sharpe Ratio', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Set x-ticks
        plt.xticks(x, regime_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig(os.path.join(self.output_dir, 'regime_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        # Create individual regime equity curves
        for regime_name, performance in regime_performance.items():
            plt.figure(figsize=(12, 6))
            
            timestamps = performance.get('timestamps', [])
            portfolio_values = performance.get('portfolio_values', [])
            
            if not timestamps or not portfolio_values:
                continue
                
            plt.plot(timestamps, portfolio_values, linewidth=2)
            plt.title(f"Performance in {regime_name} Regime")
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            
            # Add metrics annotation
            metrics = regime_metrics.get(regime_name, {})
            if metrics:
                text = f"Return: {metrics['total_return']:.2%}\n"
                text += f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
                text += f"Max DD: {metrics['max_drawdown']:.2%}\n"
                text += f"Days: {metrics['num_days']}"
                
                plt.annotate(text, xy=(0.02, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                            va='top', ha='left', fontsize=10)
            
            plt.tight_layout()
            
            if save_to_file:
                plt.savefig(os.path.join(self.output_dir, f'regime_{regime_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        return True
    
    def run_walk_forward_validation(
        self,
        windows: int = 3,
        window_size: int = None,
        step_size: int = None,
        min_window_size: int = 30
    ):
        """
        Run walk-forward validation to test model robustness across time periods.
        
        Args:
            windows: Number of windows to use (alternative to window_size)
            window_size: Size of each window in days (if not specified, will divide data into `windows` equal parts)
            step_size: Step size between windows in days (if not specified, will use non-overlapping windows)
            min_window_size: Minimum window size in days
            
        Returns:
            Dict containing walk-forward validation results
        """
        logger.info("Running walk-forward validation")
        
        # Load data if not already loaded
        if self.data is None:
            self.load_data()
        
        # Get data date range
        if isinstance(self.data.index, pd.DatetimeIndex):
            data_start = self.data.index.min()
            data_end = self.data.index.max()
            total_days = (data_end - data_start).days
            
            if window_size is None:
                # Divide the data into `windows` approximately equal parts
                window_size = total_days // windows
                
            if step_size is None:
                # Use non-overlapping windows
                step_size = window_size
                
            logger.info(f"Total data span: {total_days} days, using window size: {window_size} days, step size: {step_size} days")
            
            # Generate window start and end dates
            window_starts = []
            window_ends = []
            
            current_start = data_start
            while current_start + timedelta(days=min_window_size) <= data_end:
                window_end = min(current_start + timedelta(days=window_size), data_end)
                
                window_starts.append(current_start)
                window_ends.append(window_end)
                
                # Move to next window start
                current_start = current_start + timedelta(days=step_size)
                
                # Break if we've generated the requested number of windows
                if len(window_starts) >= windows:
                    break
                    
            logger.info(f"Generated {len(window_starts)} walk-forward windows")
        else:
            # Non-datetime index, use integer indices
            total_steps = len(self.data)
            
            if window_size is None:
                # Divide the data into `windows` approximately equal parts
                window_size = total_steps // windows
                
            if step_size is None:
                # Use non-overlapping windows
                step_size = window_size
                
            logger.info(f"Total data length: {total_steps} steps, using window size: {window_size} steps, step size: {step_size} steps")
            
            # Generate window start and end indices
            window_starts = []
            window_ends = []
            
            current_start = 0
            while current_start + min_window_size <= total_steps:
                window_end = min(current_start + window_size, total_steps)
                
                window_starts.append(current_start)
                window_ends.append(window_end)
                
                # Move to next window start
                current_start = current_start + step_size
                
                # Break if we've generated the requested number of windows
                if len(window_starts) >= windows:
                    break
                    
            logger.info(f"Generated {len(window_starts)} walk-forward windows")
        
        # Run backtests for each window
        window_results = []
        window_metrics = []
        
        for i, (start, end) in enumerate(zip(window_starts, window_ends)):
            logger.info(f"Running walk-forward window {i+1}/{len(window_starts)}: {start} to {end}")
            
            # Save the original date range
            original_start = self.start_date
            original_end = self.end_date
            
            try:
                # Set window date range
                self.start_date = start if isinstance(start, str) else start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else start
                self.end_date = end if isinstance(end, str) else end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else end
                
                # Reset environment and data
                self.env = None
                self.data = None
                
                # Run backtest for this window
                results = self.run_backtest()
                
                # Store window results
                window_results.append({
                    'window_index': i,
                    'start': start,
                    'end': end,
                    'results': results
                })
                
                # Store window metrics
                metrics = results.get('metrics', {})
                metrics['window_index'] = i
                metrics['start'] = start
                metrics['end'] = end
                window_metrics.append(metrics)
                
                logger.info(f"Window {i+1} results: Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}")
            finally:
                # Restore original date range
                self.start_date = original_start
                self.end_date = original_end
        
        # Calculate aggregated metrics across all windows
        if window_metrics:
            # Calculate averages and standard deviations
            total_returns = [m.get('total_return', 0) for m in window_metrics]
            sharpe_ratios = [m.get('sharpe_ratio', 0) for m in window_metrics]
            max_drawdowns = [m.get('max_drawdown', 0) for m in window_metrics]
            
            aggregated_metrics = {
                'avg_total_return': sum(total_returns) / len(total_returns),
                'std_total_return': np.std(total_returns),
                'avg_sharpe_ratio': sum(sharpe_ratios) / len(sharpe_ratios),
                'std_sharpe_ratio': np.std(sharpe_ratios),
                'avg_max_drawdown': sum(max_drawdowns) / len(max_drawdowns),
                'std_max_drawdown': np.std(max_drawdowns),
                'num_windows': len(window_metrics),
                'window_size': window_size,
                'step_size': step_size
            }
            
            logger.info(f"Aggregated metrics across {len(window_metrics)} windows:")
            logger.info(f"  Avg Return: {aggregated_metrics['avg_total_return']:.2%} (±{aggregated_metrics['std_total_return']:.2%})")
            logger.info(f"  Avg Sharpe: {aggregated_metrics['avg_sharpe_ratio']:.2f} (±{aggregated_metrics['std_sharpe_ratio']:.2f})")
            logger.info(f"  Avg Max DD: {aggregated_metrics['avg_max_drawdown']:.2%} (±{aggregated_metrics['std_max_drawdown']:.2%})")
        else:
            aggregated_metrics = {}
        
        # Compile results
        wf_results = {
            'window_results': window_results,
            'window_metrics': window_metrics,
            'aggregated_metrics': aggregated_metrics
        }
        
        # Save results
        self._save_results({
            'window_metrics': window_metrics,
            'aggregated_metrics': aggregated_metrics
        }, 'walk_forward')
        
        # Create visualizations
        self._create_walk_forward_plot(wf_results)
        
        return wf_results
    
    def _create_walk_forward_plot(self, walk_forward_results):
        """Create visualization for walk-forward validation results"""
        if not walk_forward_results or 'window_metrics' not in walk_forward_results:
            logger.warning("No walk-forward metrics for visualization")
            return None
            
        window_metrics = walk_forward_results.get('window_metrics', [])
        
        if not window_metrics:
            logger.warning("Empty window metrics")
            return None
            
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract data for plotting
        window_indices = [m.get('window_index', i) for i, m in enumerate(window_metrics)]
        returns = [m.get('total_return', 0) for m in window_metrics]
        sharpes = [m.get('sharpe_ratio', 0) for m in window_metrics]
        drawdowns = [m.get('max_drawdown', 0) for m in window_metrics]
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot returns
        bars1 = ax1.bar(window_indices, [r * 100 for r in returns], alpha=0.7)
        for i, ret in enumerate(returns):
            bars1[i].set_color('green' if ret >= 0 else 'red')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Walk-Forward Validation Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot Sharpe ratios
        bars2 = ax2.bar(window_indices, sharpes, alpha=0.7, color='blue')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot drawdowns
        bars3 = ax3.bar(window_indices, [d * 100 for d in drawdowns], alpha=0.7, color='red')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_xlabel('Window Index')
        ax3.grid(True, alpha=0.3)
        
        # Get aggregated metrics
        agg_metrics = walk_forward_results.get('aggregated_metrics', {})
        
        # Add average lines
        if agg_metrics:
            ax1.axhline(y=agg_metrics.get('avg_total_return', 0) * 100, color='blue', linestyle='--', 
                      label=f"Avg: {agg_metrics.get('avg_total_return', 0):.2%}")
            ax2.axhline(y=agg_metrics.get('avg_sharpe_ratio', 0), color='green', linestyle='--',
                      label=f"Avg: {agg_metrics.get('avg_sharpe_ratio', 0):.2f}")
            ax3.axhline(y=agg_metrics.get('avg_max_drawdown', 0) * 100, color='purple', linestyle='--',
                      label=f"Avg: {agg_metrics.get('avg_max_drawdown', 0):.2%}")
            
            # Add legends
            ax1.legend()
            ax2.legend()
            ax3.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'walk_forward_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return True


# Utility function for easier usage
def run_institutional_backtest(
    model_path: str,
    data_path: str = None,
    data_df: pd.DataFrame = None,
    assets: List[str] = None,
    initial_capital: float = 10000.0,
    start_date: str = None,
    end_date: str = None,
    output_dir: str = "backtest_results",
    regime_analysis: bool = True,
    walk_forward: bool = False,
    **kwargs
):
    """
    Run an institutional-grade backtest with a single function call.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to market data file (parquet, csv, pickle)
        data_df: DataFrame with market data (alternative to data_path)
        assets: List of assets to trade
        initial_capital: Initial capital for backtesting
        start_date: Start date for backtesting (format: YYYY-MM-DD)
        end_date: End date for backtesting (format: YYYY-MM-DD)
        output_dir: Directory for output files
        regime_analysis: Whether to analyze market regimes
        walk_forward: Whether to run walk-forward validation
        **kwargs: Additional arguments for InstitutionalBacktester
        
    Returns:
        Dict containing backtest results
    """
    # Initialize backtester
    backtester = InstitutionalBacktester(
        model_path=model_path,
        data_path=data_path,
        data_df=data_df,
        assets=assets,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        regime_analysis=regime_analysis,
        **kwargs
    )
    
    # Run standard backtest
    results = backtester.run_backtest()
    
    # Run regime analysis if enabled
    if regime_analysis:
        regime_results = backtester.run_regime_analysis()
        results['regime_results'] = regime_results
    
    # Run walk-forward validation if enabled
    if walk_forward:
        wf_results = backtester.run_walk_forward_validation()
        results['walk_forward_results'] = wf_results
    
    return results 