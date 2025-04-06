"""
Institutional Backtester Module

This module provides the main backtesting functionality for evaluating
trading models with institutional-grade metrics and analysis.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import json
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import traceback
from pathlib import Path
import time
from tqdm import tqdm

# Import local modules
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
from .regime_analyzer import RegimeAnalyzer, MarketRegime, RegimePeriod
from .metrics import calculate_performance_metrics, calculate_returns
from .visualization import create_visualization_suite

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class InstitutionalBacktester:
    """
    Institutional-grade backtester for evaluating trading models.
    
    This class provides a comprehensive backtesting framework with:
    - Market regime analysis
    - Walk-forward testing
    - Detailed performance metrics
    - Trade analysis
    - Visualization tools
    
    It integrates with the existing trading environment and risk engine
    to provide a robust evaluation of trading models.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str = None,
        data_df: pd.DataFrame = None,
        assets: List[str] = None,
        initial_capital: float = 10000.0,
        start_date: str = None,
        end_date: str = None,
        risk_free_rate: float = 0.02,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        max_leverage: float = 10.0,
        output_dir: str = "backtest_results",
        verbose: bool = True,
        benchmark_symbol: str = None,
        regime_analysis: bool = True
    ):
        """
        Initialize the backtester.
        
        Args:
            model_path: Path to the trained model
            data_path: Path to the market data (can be parquet, csv, pickle)
            data_df: Direct dataframe input (alternative to data_path)
            assets: List of assets to trade
            initial_capital: Initial capital for backtesting
            start_date: Start date for backtesting (format: YYYY-MM-DD)
            end_date: End date for backtesting (format: YYYY-MM-DD)
            risk_free_rate: Annual risk-free rate
            commission: Trading commission
            slippage: Trading slippage
            max_leverage: Maximum allowed leverage
            output_dir: Directory for output files
            verbose: Whether to display detailed logs
            benchmark_symbol: Symbol for benchmark comparison
            regime_analysis: Whether to perform market regime analysis
        """
        self.model_path = model_path
        self.data_path = data_path
        self.data_df = data_df
        self.assets = assets
        self.initial_capital = initial_capital
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None
        self.risk_free_rate = risk_free_rate
        self.commission = commission
        self.slippage = slippage
        self.max_leverage = max_leverage
        self.output_dir = output_dir
        self.verbose = verbose
        self.benchmark_symbol = benchmark_symbol
        self.regime_analysis = regime_analysis
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up file logging
        self._setup_logging()
        
        # Internal state
        self.model = None
        self.env = None
        self.env_path = None
        self.data = None
        self.regime_analyzer = RegimeAnalyzer() if regime_analysis else None
        self.regime_periods = []
        self.backtest_results = {}
        
        logger.info(f"Initialized InstitutionalBacktester with model: {model_path}")
        
    def _setup_logging(self):
        """Set up file logging"""
        log_path = os.path.join(self.output_dir, "backtest.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def load_data(self):
        """Load market data for backtesting"""
        if self.data_df is not None:
            logger.info("Using provided DataFrame")
            self.data = self.data_df
            return

        if not self.data_path:
            raise ValueError("Either data_path or data_df must be provided")
            
        logger.info(f"Loading data from: {self.data_path}")
        
        # Load data based on file extension
        file_ext = os.path.splitext(self.data_path)[1].lower()
        try:
            if file_ext == '.parquet':
                self.data = pd.read_parquet(self.data_path)
            elif file_ext == '.csv':
                self.data = pd.read_csv(self.data_path, parse_dates=True, index_col=0)
            elif file_ext in ['.pkl', '.pickle']:
                with open(self.data_path, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
        # Apply date filters if provided
        if self.start_date or self.end_date:
            logger.info(f"Filtering data from {self.start_date} to {self.end_date}")
            if 'timestamp' in self.data.index.names or isinstance(self.data.index, pd.DatetimeIndex):
                # Filter by index
                mask = pd.Series(True, index=self.data.index)
                if self.start_date:
                    mask = mask & (self.data.index >= self.start_date)
                if self.end_date:
                    mask = mask & (self.data.index <= self.end_date)
                self.data = self.data.loc[mask]
            else:
                # Try to filter by a date column
                for col in ['timestamp', 'date', 'time']:
                    if col in self.data.columns:
                        if self.start_date:
                            self.data = self.data[self.data[col] >= self.start_date]
                        if self.end_date:
                            self.data = self.data[self.data[col] <= self.end_date]
                        break
        
        # Set assets if not provided
        if not self.assets and isinstance(self.data.columns, pd.MultiIndex):
            self.assets = list(self.data.columns.get_level_values(0).unique())
            logger.info(f"Detected assets: {self.assets}")
            
        logger.info(f"Loaded data with shape: {self.data.shape}")
        
    def load_model(self):
        """Load the trained model and environment"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Check for environment path
        model_dir = os.path.dirname(self.model_path)
        potential_env_paths = [
            self.model_path.replace('model', 'env') + '.pkl',
            os.path.join(model_dir, "vec_normalize.pkl"),
            os.path.join(model_dir, "best_env.pkl"),
            os.path.join(model_dir, "final_env.pkl")
        ]
        
        # Find first existing environment path
        for env_path in potential_env_paths:
            if os.path.exists(env_path):
                self.env_path = env_path
                logger.info(f"Found environment at: {env_path}")
                break
                
        # Load the model
        try:
            self.model = PPO.load(self.model_path)
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def _create_environment(self):
        """Create a fresh environment for backtesting"""
        logger.info("Creating backtesting environment")
        
        # Configure risk engine
        risk_limits = RiskLimits(
            account_max_leverage=self.max_leverage * 0.8,
            position_max_leverage=self.max_leverage,
            max_drawdown_pct=0.3
        )
        
        risk_engine = InstitutionalRiskEngine(
            initial_balance=self.initial_capital,
            risk_limits=risk_limits,
            use_dynamic_limits=True,
            use_vol_scaling=True
        )
        
        # Setup environment
        env = InstitutionalPerpetualEnv(
            df=self.data,
            assets=self.assets,
            window_size=100,  # Default window size
            max_leverage=self.max_leverage,
            commission=self.commission,
            risk_engine=risk_engine,
            risk_free_rate=self.risk_free_rate,
            initial_balance=self.initial_capital,
            verbose=False,  # Disable verbose logging during backtesting
            training_mode=False  # Not in training mode
        )
        
        # Wrap environment
        env = DummyVecEnv([lambda: env])
        
        # If we have a saved environment, try to load normalization stats
        if self.env_path and os.path.exists(self.env_path):
            try:
                logger.info(f"Loading environment normalization from: {self.env_path}")
                env = VecNormalize.load(self.env_path, env)
                # Disable training mode features
                env.training = False
                env.norm_reward = False
            except Exception as e:
                logger.warning(f"Could not load environment normalization: {str(e)}")
                # Create a new VecNormalize wrapper
                env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
        else:
            # Create a new VecNormalize wrapper
            env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
            
        logger.info("Environment created successfully")
        return env
    
    def analyze_market_regimes(self):
        """Analyze market regimes in the data"""
        if not self.regime_analysis:
            logger.info("Market regime analysis disabled")
            return
            
        if not self.regime_analyzer:
            self.regime_analyzer = RegimeAnalyzer()
            
        logger.info("Analyzing market regimes")
        
        # Analyze at market level (not asset-specific)
        regime_periods_by_asset = self.regime_analyzer.analyze_market_data(
            df=self.data, 
            price_col='close',
            asset_level=False
        )
        
        # Store regime periods
        self.regime_periods = regime_periods_by_asset.get('MARKET', [])
        
        # Log identified regimes
        for period in self.regime_periods:
            logger.info(f"Regime: {period.regime.value} from {period.start_date.date()} to {period.end_date.date()}")
            
        # Save regime info to file
        regime_info = [
            {
                'regime': period.regime.value,
                'start_date': period.start_date.strftime('%Y-%m-%d'),
                'end_date': period.end_date.strftime('%Y-%m-%d'),
                'description': period.description
            }
            for period in self.regime_periods
        ]
        
        with open(os.path.join(self.output_dir, 'market_regimes.json'), 'w') as f:
            json.dump(regime_info, f, indent=2)
            
        logger.info(f"Identified {len(self.regime_periods)} market regime periods")
        
    def run_backtest(self):
        """Run full backtest with the trained model"""
        # Make sure data is loaded
        if self.data is None:
            self.load_data()
            
        # Make sure model is loaded
        if self.model is None:
            self.load_model()
            
        # Create environment
        self.env = self._create_environment()
        
        # Analyze market regimes
        if self.regime_analysis:
            self.analyze_market_regimes()
            
        # Run the backtest
        logger.info("Starting backtest")
        start_time = time.time()
        
        results = self._run_backtest_simulation()
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Backtest completed in {duration:.2f} seconds")
        
        # Calculate performance metrics
        self._calculate_metrics(results)
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Save results
        self._save_results()
        
        return self.backtest_results
        
    def _run_backtest_simulation(self):
        """Run the backtest simulation with the trained model"""
        # Reset the environment
        obs = self.env.reset()
        
        # Initialize tracking variables
        done = False
        portfolio_values = []
        timestamps = []
        returns = []
        actions = []
        rewards = []
        trades = []
        positions_history = []
        leverages = []
        
        # Get unwrapped environment to access more information
        unwrapped_env = self.env.unwrapped.envs[0]
        
        # Track initial portfolio value
        portfolio_values.append(unwrapped_env.initial_balance)
        if hasattr(unwrapped_env, 'df') and 'timestamp' in unwrapped_env.df.index.names:
            timestamps.append(unwrapped_env.df.index[0])
            
        # Main simulation loop
        pbar = tqdm(total=len(unwrapped_env.df) - unwrapped_env.window_size, desc="Backtesting", disable=not self.verbose)
        
        while not done:
            # Get prediction from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Update progress bar
            pbar.update(1)
            
            # Extract information from info
            if 'info' in info:
                step_info = info['info'][0]  # For vectorized environments
            else:
                step_info = info[0] if isinstance(info, list) else info
                
            # Store data
            portfolio_value = step_info.get('portfolio_value', unwrapped_env.balance)
            portfolio_values.append(portfolio_value)
            
            if len(portfolio_values) > 1:
                step_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
                returns.append(step_return)
                
            actions.append(action)
            rewards.append(reward)
            
            # Get timestamp if available
            current_timestamp = None
            if hasattr(unwrapped_env, 'current_step') and unwrapped_env.current_step < len(unwrapped_env.df):
                if isinstance(unwrapped_env.df.index, pd.DatetimeIndex):
                    current_timestamp = unwrapped_env.df.index[unwrapped_env.current_step]
                elif 'timestamp' in unwrapped_env.df.index.names:
                    current_timestamp = unwrapped_env.df.index[unwrapped_env.current_step][0]  # For MultiIndex
                    
            if current_timestamp:
                timestamps.append(current_timestamp)
                
            # Track trades
            new_trades = step_info.get('trades', [])
            for trade in new_trades:
                trade['portfolio_value'] = portfolio_value
                if current_timestamp:
                    trade['timestamp'] = current_timestamp
                trades.append(trade)
                
            # Track positions
            position_snapshot = {
                'step': unwrapped_env.current_step,
                'timestamp': current_timestamp,
                'portfolio_value': portfolio_value,
                'positions': {}
            }
            
            # Get position details
            for asset, position in unwrapped_env.positions.items():
                position_snapshot['positions'][asset] = {
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'mark_price': position.get('mark_price', position.get('last_price', 0)),
                    'unrealized_pnl': position.get('unrealized_pnl', 0),
                    'direction': position.get('direction', 0),
                    'leverage': position.get('leverage', 0)
                }
                
            positions_history.append(position_snapshot)
            
            # Track leverage
            total_leverage = sum(abs(pos.get('leverage', 0)) for pos in unwrapped_env.positions.values())
            leverages.append(total_leverage)
            
            # Update observation
            obs = next_obs
            
        pbar.close()
        
        # Compile results
        results = {
            'portfolio_values': portfolio_values,
            'timestamps': timestamps,
            'returns': returns,
            'actions': actions,
            'rewards': rewards,
            'trades': trades,
            'positions_history': positions_history,
            'leverages': leverages
        }
        
        # Calculate drawdowns
        if len(portfolio_values) > 1:
            values = np.array(portfolio_values)
            peak = np.maximum.accumulate(values)
            drawdowns = (values - peak) / peak
            results['drawdowns'] = drawdowns.tolist()
        
        logger.info(f"Simulation completed with {len(portfolio_values)} steps and {len(trades)} trades")
        
        return results
        
    def _calculate_metrics(self, results):
        """Calculate performance metrics from backtest results"""
        logger.info("Calculating performance metrics")
        
        portfolio_values = results.get('portfolio_values', [])
        timestamps = results.get('timestamps', [])
        returns = results.get('returns', [])
        trades = results.get('trades', [])
        leverages = results.get('leverages', [])
        
        # Handle empty or very short results
        if len(portfolio_values) < 2:
            logger.warning("Insufficient data for metric calculation")
            self.backtest_results = results
            self.backtest_results['metrics'] = {}
            return
            
        # Calculate returns if not provided
        if not returns and len(portfolio_values) > 1:
            returns = calculate_returns(portfolio_values)
            results['returns'] = returns
            
        # Calculate benchmark returns if a benchmark is provided
        benchmark_returns = None
        if self.benchmark_symbol and self.benchmark_symbol in self.assets:
            # Extract benchmark prices
            if isinstance(self.data.columns, pd.MultiIndex):
                benchmark_prices = self.data.xs(self.benchmark_symbol, level=0)['close']
                benchmark_prices = benchmark_prices.values if hasattr(benchmark_prices, 'values') else benchmark_prices
            else:
                # Handle single asset
                benchmark_prices = self.data['close'].values
                
            # Calculate benchmark returns
            benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
            results['benchmark_returns'] = benchmark_returns
            
            # Calculate benchmark values (starting from initial capital)
            benchmark_values = [self.initial_capital]
            for ret in benchmark_returns:
                benchmark_values.append(benchmark_values[-1] * (1 + ret))
            results['benchmark_values'] = benchmark_values
                
        # Calculate comprehensive metrics
        metrics = calculate_performance_metrics(
            portfolio_values=portfolio_values,
            timestamps=timestamps,
            trades=trades,
            leverages=leverages,
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.risk_free_rate
        )
        
        # Add to results
        results['metrics'] = metrics
        
        # Update class attribute with results
        self.backtest_results = results
        
        # Log key metrics
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        
    def _create_visualizations(self, results):
        """Create visualizations from backtest results"""
        logger.info("Creating visualizations")
        
        # Add regime periods to results
        results['regime_periods'] = self.regime_periods
        
        # Create visualization suite
        figures = create_visualization_suite(
            results=results,
            output_dir=self.output_dir,
            create_tearsheet=True
        )
        
        # Add figure paths to results
        results['figures'] = figures
        
    def _save_results(self):
        """Save backtest results to files"""
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save metrics to JSON
        metrics_path = os.path.join(self.output_dir, 'backtest_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.backtest_results['metrics'], f, indent=2)
            
        # Save trades to CSV
        trades_path = os.path.join(self.output_dir, 'backtest_trades.csv')
        if self.backtest_results.get('trades'):
            trades_df = pd.DataFrame(self.backtest_results['trades'])
            trades_df.to_csv(trades_path, index=False)
            
        # Save portfolio values to CSV
        portfolio_path = os.path.join(self.output_dir, 'backtest_portfolio.csv')
        portfolio_df = pd.DataFrame({
            'timestamp': self.backtest_results.get('timestamps', range(len(self.backtest_results['portfolio_values']))),
            'portfolio_value': self.backtest_results['portfolio_values'],
            'return': [0] + self.backtest_results.get('returns', []),
            'drawdown': self.backtest_results.get('drawdowns', [0] * len(self.backtest_results['portfolio_values']))
        })
        portfolio_df.to_csv(portfolio_path, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
        
    def run_regime_backtest(self):
        """Run separate backtests for each identified market regime"""
        if not self.regime_analysis:
            logger.info("Market regime analysis disabled, skipping regime backtest")
            return {}
            
        # Make sure we have identified regimes
        if not self.regime_periods:
            self.analyze_market_regimes()
            
        if not self.regime_periods:
            logger.warning("No market regimes identified, skipping regime backtest")
            return {}
            
        # Run a backtest for each regime
        regime_results = {}
        
        for period in self.regime_periods:
            regime_name = period.regime.value
            logger.info(f"Running backtest for regime: {regime_name} from {period.start_date.date()} to {period.end_date.date()}")
            
            # Create a copy of the backtester with regime-specific dates
            regime_tester = InstitutionalBacktester(
                model_path=self.model_path,
                data_df=self.data,
                assets=self.assets,
                initial_capital=self.initial_capital,
                start_date=period.start_date,
                end_date=period.end_date,
                risk_free_rate=self.risk_free_rate,
                commission=self.commission,
                slippage=self.slippage,
                max_leverage=self.max_leverage,
                output_dir=os.path.join(self.output_dir, f"regime_{regime_name}"),
                verbose=self.verbose,
                benchmark_symbol=self.benchmark_symbol,
                regime_analysis=False  # Disable nested regime analysis
            )
            
            # Run backtest for this regime
            regime_results[regime_name] = regime_tester.run_backtest()
            
        # Compile regime performance comparison
        regime_comparison = {
            regime: {
                'total_return': results['metrics'].get('total_return', 0),
                'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
                'max_drawdown': results['metrics'].get('max_drawdown', 0),
                'win_rate': results['metrics'].get('win_rate', 0),
                'avg_trade_pnl': results['metrics'].get('avg_trade_pnl', 0),
                'total_trades': results['metrics'].get('total_trades', 0)
            }
            for regime, results in regime_results.items()
        }
        
        # Save regime comparison to file
        comparison_path = os.path.join(self.output_dir, 'regime_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(regime_comparison, f, indent=2)
            
        logger.info(f"Regime comparison saved to {comparison_path}")
        
        return regime_results
        
    def run_walk_forward_validation(
        self,
        window_size: int = 60,
        step_size: int = 30,
        min_window_size: int = 30
    ):
        """
        Run walk-forward validation to test model robustness.
        
        Args:
            window_size: Size of each test window in days
            step_size: Size of step between windows in days
            min_window_size: Minimum window size to run a test
            
        Returns:
            Dictionary of walk-forward results
        """
        # Make sure data is loaded
        if self.data is None:
            self.load_data()
            
        # Make sure model is loaded
        if self.model is None:
            self.load_model()
            
        # Get date range from data
        if isinstance(self.data.index, pd.DatetimeIndex):
            start_date = self.data.index.min()
            end_date = self.data.index.max()
        elif 'timestamp' in self.data.index.names:
            # For MultiIndex with timestamp
            start_date = self.data.index.get_level_values('timestamp').min()
            end_date = self.data.index.get_level_values('timestamp').max()
        else:
            logger.warning("Cannot identify timestamps in data, using row indices for walk-forward validation")
            start_date = 0
            end_date = len(self.data) - 1
            
        logger.info(f"Running walk-forward validation from {start_date} to {end_date}")
        
        # Generate windows
        windows = []
        
        if isinstance(start_date, (pd.Timestamp, datetime)):
            # Date-based windows
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + pd.Timedelta(days=window_size), end_date)
                if (current_end - current_start).days >= min_window_size:
                    windows.append((current_start, current_end))
                current_start += pd.Timedelta(days=step_size)
        else:
            # Index-based windows
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + window_size, end_date)
                if current_end - current_start >= min_window_size:
                    windows.append((current_start, current_end))
                current_start += step_size
                
        logger.info(f"Created {len(windows)} test windows for walk-forward validation")
        
        # Run backtest for each window
        walk_forward_results = {}
        
        for i, (window_start, window_end) in enumerate(windows):
            window_name = f"window_{i+1}"
            logger.info(f"Running backtest for {window_name}: {window_start} to {window_end}")
            
            # Create a copy of the backtester with window-specific dates
            window_tester = InstitutionalBacktester(
                model_path=self.model_path,
                data_df=self.data,
                assets=self.assets,
                initial_capital=self.initial_capital,
                start_date=window_start,
                end_date=window_end,
                risk_free_rate=self.risk_free_rate,
                commission=self.commission,
                slippage=self.slippage,
                max_leverage=self.max_leverage,
                output_dir=os.path.join(self.output_dir, f"walkforward_{window_name}"),
                verbose=False,  # Disable verbose output for sub-tests
                benchmark_symbol=self.benchmark_symbol,
                regime_analysis=False  # Disable nested regime analysis
            )
            
            # Run backtest for this window
            window_results = window_tester.run_backtest()
            
            # Store key metrics
            walk_forward_results[window_name] = {
                'start_date': window_start,
                'end_date': window_end,
                'total_return': window_results['metrics'].get('total_return', 0),
                'sharpe_ratio': window_results['metrics'].get('sharpe_ratio', 0),
                'max_drawdown': window_results['metrics'].get('max_drawdown', 0),
                'win_rate': window_results['metrics'].get('win_rate', 0),
                'num_trades': window_results['metrics'].get('total_trades', 0)
            }
            
        # Calculate walk-forward statistics
        returns = [window['total_return'] for window in walk_forward_results.values()]
        sharpes = [window['sharpe_ratio'] for window in walk_forward_results.values()]
        drawdowns = [window['max_drawdown'] for window in walk_forward_results.values()]
        win_rates = [window['win_rate'] for window in walk_forward_results.values()]
        
        walk_forward_stats = {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': min(returns),
            'max_return': max(returns),
            'avg_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'avg_drawdown': np.mean(drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'consistency': len([r for r in returns if r > 0]) / len(returns) if returns else 0
        }
        
        # Save walk-forward results to file
        results_path = os.path.join(self.output_dir, 'walkforward_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'windows': walk_forward_results,
                'stats': walk_forward_stats
            }, f, indent=2)
            
        logger.info(f"Walk-forward results saved to {results_path}")
        
        # Create walk-forward visualization
        self._create_walk_forward_plot(walk_forward_results)
        
        return {
            'windows': walk_forward_results,
            'stats': walk_forward_stats
        }
        
    def _create_walk_forward_plot(self, walk_forward_results):
        """Create visualization of walk-forward validation results"""
        windows = list(walk_forward_results.keys())
        returns = [walk_forward_results[w]['total_return'] for w in windows]
        sharpes = [walk_forward_results[w]['sharpe_ratio'] for w in windows]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot returns
        ax1.bar(windows, returns, color=['green' if r > 0 else 'red' for r in returns])
        ax1.set_ylabel('Total Return')
        ax1.set_title('Walk-Forward Validation Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot Sharpe ratios
        ax2.bar(windows, sharpes, color=['green' if s > 0 else 'red' for s in sharpes])
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xlabel('Test Window')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'walkforward_performance.png')
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Walk-forward visualization saved to {fig_path}")


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
    walk_forward: bool = False
):
    """
    Run a complete institutional-grade backtest.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the market data (can be parquet, csv, pickle)
        data_df: Direct dataframe input (alternative to data_path)
        assets: List of assets to trade
        initial_capital: Initial capital for backtesting
        start_date: Start date for backtesting (format: YYYY-MM-DD)
        end_date: End date for backtesting (format: YYYY-MM-DD)
        output_dir: Directory for output files
        regime_analysis: Whether to perform market regime analysis
        walk_forward: Whether to perform walk-forward validation
        
    Returns:
        Dictionary of backtest results
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
        regime_analysis=regime_analysis
    )
    
    # Run main backtest
    results = backtester.run_backtest()
    
    # Run regime-specific backtests if enabled
    if regime_analysis:
        regime_results = backtester.run_regime_backtest()
        results['regime_results'] = regime_results
        
    # Run walk-forward validation if enabled
    if walk_forward:
        wf_results = backtester.run_walk_forward_validation()
        results['walk_forward_results'] = wf_results
        
    return results 