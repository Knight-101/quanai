import sys
import os

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
import yaml
import warnings
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from data_system.data_manager import DataManager
from data_system.feature_engine import DerivativesFeatureEngine
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting/backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InstitutionalBacktester:
    """
    Institutional-grade backtester for cryptocurrency trading models with 
    a focus on eliminating bias, analyzing market regimes, and providing 
    comprehensive performance metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: Optional[str] = None,
        output_dir: str = "results/backtest",
        initial_capital: float = 10000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        regime_analysis: bool = False,
        walk_forward: bool = False,
        config_path: str = 'config/prod_config.yaml'
    ):
        """
        Initialize the institutional backtester.
        
        Args:
            model_path: Path to the trained model to test
            data_path: Path to the data file (if None, will use data from data directory)
            output_dir: Directory to save backtest results
            initial_capital: Initial capital for backtesting
            start_date: Start date for backtesting (format: 'YYYY-MM-DD')
            end_date: End date for backtesting (format: 'YYYY-MM-DD')
            assets: List of assets to backtest (if None, will use all available)
            regime_analysis: Whether to perform market regime analysis
            walk_forward: Whether to perform walk-forward validation
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets
        self.regime_analysis = regime_analysis
        self.walk_forward = walk_forward
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize data manager
        self.data_manager = DataManager(base_path=self.config['data']['cache_dir'])
        
        # Initialize feature engine
        self.feature_engine = DerivativesFeatureEngine(
            volatility_window=self.config['feature_engineering']['volatility_window'],
            n_components=self.config['feature_engineering']['n_components']
        )
        
        # Initialize risk engine
        self.risk_engine = InstitutionalRiskEngine(
            risk_limits=RiskLimits(**self.config['risk_management']['limits'])
        )
        
        # Store backtest results
        self.results = None
        self.portfolio_history = None
        self.trade_history = None
        self.market_regimes = None
        self.regime_performance = None
        self.walkforward_results = None
        
        # Load model and environment
        self.model = None
        self.env = None
        self.data = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data for backtesting.
        
        Returns:
            DataFrame with processed data ready for backtesting
        """
        logger.info("Loading data for backtesting...")
        
        if self.data_path and os.path.exists(self.data_path):
            # Load from specified path
            try:
                if self.data_path.endswith('.parquet'):
                    data = pd.read_parquet(self.data_path)
                elif self.data_path.endswith('.csv'):
                    data = pd.read_csv(self.data_path)
                else:
                    raise ValueError(f"Unsupported file format: {self.data_path}")
                    
                logger.info(f"Loaded data from {self.data_path} with shape {data.shape}")
                self.data = data
                return data
            except Exception as e:
                logger.error(f"Error loading data from {self.data_path}: {str(e)}")
                raise
        else:
            # Load from data directory using DataManager
            logger.info("No specific data path provided. Loading from data directory...")
            
            # If assets not specified, use all from config
            if not self.assets:
                self.assets = self.config['trading']['symbols']
                logger.info(f"Using assets from config: {self.assets}")
            
            # First try to load feature data
            try:
                feature_data = self.data_manager.load_feature_data('base_features')
                if feature_data is not None and len(feature_data) > 0:
                    logger.info(f"Loaded feature data with shape {feature_data.shape}")
                    
                    # Apply date filtering if specified
                    if self.start_date or self.end_date:
                        feature_data = self._filter_data_by_date(feature_data)
                    
                    self.data = feature_data
                    return feature_data
            except Exception as e:
                logger.warning(f"Could not load feature data: {str(e)}")
            
            # If feature data not available, load and process market data
            logger.info("Feature data not available. Loading and processing market data...")
            raw_data = self._load_market_data()
            if raw_data:
                processed_data = self._process_market_data(raw_data)
                self.data = processed_data
                return processed_data
            else:
                raise ValueError("No data available for backtesting")
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by date range"""
        if not hasattr(data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot filter by date.")
            return data
            
        # Get the datetime index if it exists
        try:
            if 'datetime' in data.index.names:
                date_idx = data.index.get_level_values('datetime')
            else:
                date_idx = data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(date_idx):
                    logger.warning("First index level is not datetime. Cannot filter by date.")
                    return data
        except Exception as e:
            logger.warning(f"Error accessing index for date filtering: {str(e)}")
            return data
            
        # Apply date filters
        if self.start_date:
            start_date = pd.to_datetime(self.start_date)
            data = data[date_idx >= start_date]
            logger.info(f"Filtered data from {start_date}")
            
        if self.end_date:
            end_date = pd.to_datetime(self.end_date)
            data = data[date_idx <= end_date]
            logger.info(f"Filtered data until {end_date}")
            
        logger.info(f"Data after date filtering: {data.shape}")
        return data
    
    def _load_market_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load market data from data directory"""
        all_data = {exchange: {} for exchange in self.config['data']['exchanges']}
        has_all_data = True
        
        for exchange in self.config['data']['exchanges']:
            logger.info(f"Loading market data for exchange: {exchange}")
            for symbol in self.assets:
                # Convert symbols to format used in storage
                clean_symbol = symbol.replace('/', '').replace(':', '')
                logger.info(f"Loading data for {exchange}_{clean_symbol}_{self.config['data']['timeframe']}")
                
                # Load data
                data = self.data_manager.load_market_data(
                    exchange=exchange,
                    symbol=clean_symbol,
                    timeframe=self.config['data']['timeframe'],
                    start_time=self.start_date,
                    end_time=self.end_date,
                    data_type='perpetual'
                )
                
                # Check if we have data
                if data is None or len(data) == 0:
                    logger.warning(f"No data found for {exchange}_{clean_symbol}")
                    has_all_data = False
                    break
                else:
                    logger.info(f"Loaded data for {exchange}_{clean_symbol}: {len(data)} points")
                    all_data[exchange][clean_symbol] = data
                
            if not has_all_data:
                break
                
        if has_all_data:
            return all_data
        else:
            logger.error("Incomplete market data. Cannot proceed with backtesting.")
            return None
            
    def _process_market_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Process raw market data into features"""
        # Similar process as in TradingSystem._format_data_for_training
        logger.info("Processing market data into features...")
        
        all_dfs = []
        
        for exchange, exchange_data in raw_data.items():
            for symbol, symbol_data in exchange_data.items():
                # Ensure the data has a datetime index
                if not pd.api.types.is_datetime64_any_dtype(symbol_data.index):
                    symbol_data.index = pd.to_datetime(symbol_data.index)
                
                # Add exchange and symbol as columns
                symbol_data['exchange'] = exchange
                symbol_data['symbol'] = symbol
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in symbol_data.columns]
                if missing_cols:
                    logger.warning(f"Missing columns for {exchange}_{symbol}: {missing_cols}")
                    continue
                
                # Generate features
                try:
                    logger.info(f"Generating features for {exchange}_{symbol}")
                    feature_df = self.feature_engine.generate_all_features(symbol_data)
                    
                    # Set multi-level index with datetime and symbol
                    feature_df['datetime'] = feature_df.index
                    feature_df.set_index(['datetime', 'symbol'], inplace=True)
                    
                    all_dfs.append(feature_df)
                except Exception as e:
                    logger.error(f"Error generating features for {exchange}_{symbol}: {str(e)}")
                    continue
        
        if not all_dfs:
            logger.error("No features generated. Cannot proceed with backtesting.")
            return None
            
        # Combine all data
        combined_df = pd.concat(all_dfs)
        logger.info(f"Combined feature data shape: {combined_df.shape}")
        
        # Save processed data for future use
        self.data_manager.save_feature_data(
            data=combined_df,
            feature_set='backtest_features',
            metadata={
                'feature_config': self.config['feature_engineering'],
                'exchanges': self.config['data']['exchanges'],
                'symbols': self.assets,
                'timeframe': self.config['data']['timeframe']
            }
        )
        
        return combined_df
    
    def load_model(self) -> PPO:
        """
        Load the trained model for backtesting.
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load the model
            model = PPO.load(self.model_path)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            
            # Look for env file alongside model
            env_path = None
            if os.path.isdir(self.model_path):
                # Try common env file patterns in the directory
                env_patterns = [
                    os.path.join(self.model_path, "vec_normalize.pkl"),
                    os.path.join(self.model_path, "env.pkl"),
                    os.path.join(self.model_path, "vec_env.pkl")
                ]
                for pattern in env_patterns:
                    if os.path.exists(pattern):
                        env_path = pattern
                        break
            else:
                # Check if there's a corresponding .pkl file in the same directory
                model_dir = os.path.dirname(self.model_path)
                model_name = os.path.splitext(os.path.basename(self.model_path))[0]
                env_patterns = [
                    os.path.join(model_dir, f"{model_name}_env.pkl"),
                    os.path.join(model_dir, "vec_normalize.pkl"),
                    os.path.join(model_dir, "env.pkl")
                ]
                for pattern in env_patterns:
                    if os.path.exists(pattern):
                        env_path = pattern
                        break
            
            if env_path:
                logger.info(f"Found environment file at {env_path}")
                self.env_path = env_path
            else:
                logger.warning("No environment file found. Will create a new environment.")
                self.env_path = None
            
            self.model = model
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_environment(self, data: pd.DataFrame) -> VecNormalize:
        """
        Prepare the environment for backtesting.
        
        Args:
            data: Processed data to use in the environment
            
        Returns:
            Prepared VecNormalize environment
        """
        logger.info("Preparing environment for backtesting...")
        
        # Get list of assets
        if not self.assets and data is not None:
            if hasattr(data.index, 'get_level_values') and 'symbol' in data.index.names:
                self.assets = data.index.get_level_values('symbol').unique().tolist()
            else:
                # Try to infer assets from columns
                potential_assets = []
                for col in data.columns:
                    if isinstance(col, tuple) and len(col) > 0:
                        potential_assets.append(col[0])
                if potential_assets:
                    self.assets = list(set(potential_assets))
        
        logger.info(f"Using assets: {self.assets}")
        
        # Configure features
        base_features = ['open', 'high', 'low', 'close', 'volume']
        tech_features = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr_14', 'adx_14', 'cci_14',
            'market_regime', 'hurst_exponent', 'volatility_regime'
        ]
        
        # Create environment
        logger.info("Creating environment with training_mode=False for backtesting")
        env = InstitutionalPerpetualEnv(
            df=data,
            assets=self.assets,
            initial_balance=self.initial_capital,
            max_drawdown=self.config['risk_management']['limits']['max_drawdown'],
            window_size=self.config['model']['window_size'],
            max_leverage=self.config['trading']['max_leverage'],
            commission=self.config['trading']['commission'],
            funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
            base_features=base_features,
            tech_features=tech_features,
            risk_engine=self.risk_engine,
            risk_free_rate=self.config['trading']['risk_free_rate'],
            verbose=False,
            training_mode=False  # Important: Disable training mode for backtesting
        )
        
        # Wrap with DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
        
        # Either load existing VecNormalize or create a new one
        if self.env_path and os.path.exists(self.env_path):
            try:
                logger.info(f"Loading normalization statistics from {self.env_path}")
                vec_norm_env = VecNormalize.load(self.env_path, vec_env)
                
                # Disable training-related features for backtesting
                vec_norm_env.training = False  # No updates to normalization stats
                vec_norm_env.norm_reward = False  # Use raw rewards for evaluation
                logger.info("Environment loaded with existing normalization statistics")
            except Exception as e:
                logger.warning(f"Error loading environment, creating new one: {str(e)}")
                vec_norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        else:
            logger.info("Creating new environment with default normalization")
            vec_norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        
        self.env = vec_norm_env
        return vec_norm_env 

    def run_backtest(self, n_eval_episodes: int = 1) -> Dict[str, Any]:
        """
        Run the backtest with the loaded model and environment.
        
        Args:
            n_eval_episodes: Number of evaluation episodes to run
            
        Returns:
            Dictionary of backtest results
        """
        logger.info(f"Starting backtest with {n_eval_episodes} episodes...")
        
        # Load data and model if not already loaded
        if self.data is None:
            self.data = self.load_data()
        
        if self.model is None:
            self.model = self.load_model()
            
        if self.env is None:
            self.env = self.prepare_environment(self.data)
        
        # Initialize results storage
        all_portfolio_values = []
        all_returns = []
        all_trades = []
        all_positions = []
        all_drawdowns = []
        all_leverage = []
        all_rewards = []
        
        # Run episodes
        for episode in range(n_eval_episodes):
            logger.info(f"Running episode {episode+1}/{n_eval_episodes}")
            
            # Reset environment
            obs = self.env.reset()
            
            # Initialize episode tracking
            done = False
            episode_reward = 0
            step_count = 0
            episode_portfolio = [self.initial_capital]
            episode_returns = []
            episode_trades = []
            episode_positions = []
            episode_drawdowns = []
            episode_leverage = []
            episode_rewards = []
            
            # Run episode until done
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Extract info from environment
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]  # Unwrap from VecEnv
                
                # Track reward
                episode_reward += reward
                episode_rewards.append(reward)
                
                # Track portfolio value
                portfolio_value = info.get('portfolio_value', episode_portfolio[-1])
                episode_portfolio.append(portfolio_value)
                
                # Track returns
                if len(episode_portfolio) >= 2:
                    returns = (episode_portfolio[-1] / episode_portfolio[-2]) - 1
                    episode_returns.append(returns)
                
                # Track positions and trades
                if 'positions' in info:
                    positions = info['positions']
                    episode_positions.append(positions)
                
                if 'trades' in info:
                    trades = info['trades']
                    if trades:
                        for trade in trades:
                            trade['timestamp'] = info.get('timestamp', step_count)
                            episode_trades.append(trade)
                
                # Track risk metrics
                if 'risk_metrics' in info:
                    risk_metrics = info['risk_metrics']
                    
                    if 'current_drawdown' in risk_metrics:
                        episode_drawdowns.append(risk_metrics['current_drawdown'])
                    elif 'max_drawdown' in risk_metrics:
                        episode_drawdowns.append(risk_metrics['max_drawdown'])
                    
                    if 'leverage_utilization' in risk_metrics:
                        episode_leverage.append(risk_metrics['leverage_utilization'])
                
                step_count += 1
                
                # Check for excessive steps to prevent infinite loops
                if step_count > 10000:
                    logger.warning(f"Ending episode after {step_count} steps to prevent infinite loop")
                    break
            
            logger.info(f"Episode {episode+1} completed with {step_count} steps")
            logger.info(f"Final portfolio value: ${episode_portfolio[-1]:.2f}")
            logger.info(f"Number of trades: {len(episode_trades)}")
            
            # Store episode results
            all_portfolio_values.append(episode_portfolio)
            all_returns.extend(episode_returns)
            all_trades.extend(episode_trades)
            all_positions.append(episode_positions)
            all_drawdowns.extend(episode_drawdowns)
            all_leverage.extend(episode_leverage)
            all_rewards.extend(episode_rewards)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            portfolio_values=all_portfolio_values,
            returns=all_returns,
            trades=all_trades,
            drawdowns=all_drawdowns,
            leverage=all_leverage,
            rewards=all_rewards
        )
        
        # Store results
        self.results = metrics
        self.portfolio_history = all_portfolio_values
        self.trade_history = all_trades
        
        # Save results
        self.save_results()
        
        # If requested, run additional analyses
        if self.regime_analysis:
            self.run_regime_analysis()
            
        if self.walk_forward:
            self.run_walk_forward_validation()
        
        return metrics
    
    def calculate_metrics(
        self,
        portfolio_values: List[List[float]],
        returns: List[float],
        trades: List[Dict],
        drawdowns: List[float],
        leverage: List[float],
        rewards: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Returns:
            Dictionary of calculated metrics
        """
        logger.info("Calculating performance metrics...")
        
        metrics = {}
        
        # Handle empty results
        if not portfolio_values or all(len(pv) <= 1 for pv in portfolio_values):
            logger.warning("No portfolio values available for metrics calculation")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
                "avg_trade_return": 0.0,
                "avg_leverage": 0.0
            }
        
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in portfolio_values:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Overall performance metrics
        initial_value = combined_portfolio[0]
        final_value = combined_portfolio[-1]
        
        # Total return
        total_return = (final_value / initial_value) - 1
        metrics["total_return"] = total_return
        
        # Annualized return (assuming 252 trading days per year)
        if len(combined_portfolio) > 1:
            days = len(combined_portfolio) / 288  # Assuming 5-minute data (288 bars per day)
            annual_return = (1 + total_return) ** (252 / days) - 1
            metrics["annual_return"] = annual_return
        else:
            metrics["annual_return"] = total_return
        
        # Sharpe ratio (assuming risk-free rate from config)
        risk_free_rate = self.config['trading']['risk_free_rate']
        if returns and len(returns) > 1:
            returns_array = np.array(returns)
            returns_mean = np.mean(returns_array)
            returns_std = np.std(returns_array)
            
            if returns_std > 0:
                sharpe = (returns_mean - risk_free_rate / 252) / returns_std * np.sqrt(252)
                metrics["sharpe_ratio"] = sharpe
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0
        
        # Sortino ratio (downside risk only)
        if returns and len(returns) > 1:
            returns_array = np.array(returns)
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = (np.mean(returns_array) - risk_free_rate / 252) / downside_std * np.sqrt(252)
                    metrics["sortino_ratio"] = sortino
                else:
                    metrics["sortino_ratio"] = np.inf
            else:
                metrics["sortino_ratio"] = np.inf
        else:
            metrics["sortino_ratio"] = 0.0
        
        # Maximum drawdown
        if drawdowns and len(drawdowns) > 0:
            max_dd = max(drawdowns)
            metrics["max_drawdown"] = max_dd
        else:
            # Calculate from portfolio values
            rolling_max = np.maximum.accumulate(combined_portfolio)
            drawdowns = (rolling_max - combined_portfolio) / rolling_max
            max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            metrics["max_drawdown"] = max_dd
        
        # Calmar ratio (annualized return / max drawdown)
        if metrics["max_drawdown"] > 0:
            calmar = metrics["annual_return"] / metrics["max_drawdown"]
            metrics["calmar_ratio"] = calmar
        else:
            metrics["calmar_ratio"] = np.inf
        
        # Trade-specific metrics
        if trades and len(trades) > 0:
            metrics["trade_count"] = len(trades)
            
            # Calculate returns for each trade
            trade_returns = []
            profitable_trades = 0
            
            for trade in trades:
                if 'pnl' in trade and 'cost' in trade:
                    trade_return = trade['pnl'] - trade['cost']
                    trade_returns.append(trade_return)
                    
                    if trade_return > 0:
                        profitable_trades += 1
            
            # Win rate
            if trade_returns:
                metrics["win_rate"] = profitable_trades / len(trade_returns)
                metrics["avg_trade_return"] = np.mean(trade_returns)
                metrics["avg_trade_duration"] = np.mean([trade.get('duration', 0) for trade in trades])
            else:
                metrics["win_rate"] = 0.0
                metrics["avg_trade_return"] = 0.0
                metrics["avg_trade_duration"] = 0.0
        else:
            metrics["trade_count"] = 0
            metrics["win_rate"] = 0.0
            metrics["avg_trade_return"] = 0.0
            metrics["avg_trade_duration"] = 0.0
        
        # Leverage metrics
        if leverage and len(leverage) > 0:
            metrics["avg_leverage"] = np.mean(leverage)
            metrics["max_leverage"] = np.max(leverage)
        else:
            metrics["avg_leverage"] = 0.0
            metrics["max_leverage"] = 0.0
        
        # Reward metrics
        if rewards and len(rewards) > 0:
            metrics["mean_reward"] = np.mean(rewards)
            metrics["reward_sharpe"] = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0.0
        else:
            metrics["mean_reward"] = 0.0
            metrics["reward_sharpe"] = 0.0
        
        # Additional metrics
        metrics["profit_factor"] = self._calculate_profit_factor(trades)
        metrics["recovery_factor"] = self._calculate_recovery_factor(combined_portfolio, metrics["max_drawdown"])
        metrics["ulcer_index"] = self._calculate_ulcer_index(combined_portfolio)
        
        logger.info("Metrics calculation completed")
        return metrics
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0
            
        gross_profit = 0.0
        gross_loss = 0.0
        
        for trade in trades:
            if 'pnl' in trade and 'cost' in trade:
                profit = trade['pnl'] - trade['cost']
                if profit > 0:
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    def _calculate_recovery_factor(self, portfolio: List[float], max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if max_drawdown == 0 or len(portfolio) < 2:
            return np.inf
            
        total_return = (portfolio[-1] / portfolio[0]) - 1
        return total_return / max_drawdown
    
    def _calculate_ulcer_index(self, portfolio: List[float]) -> float:
        """Calculate ulcer index (square root of average squared drawdown)"""
        if len(portfolio) < 2:
            return 0.0
            
        # Calculate percentage drawdowns
        rolling_max = np.maximum.accumulate(portfolio)
        drawdowns = (rolling_max - portfolio) / rolling_max
        
        # Calculate ulcer index
        squared_drawdowns = np.square(drawdowns)
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    def save_results(self) -> None:
        """Save backtest results to output directory"""
        logger.info(f"Saving backtest results to {self.output_dir}")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if self.results:
            metrics_file = os.path.join(self.output_dir, "backtest_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.results, f, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Save portfolio history
        if self.portfolio_history:
            # Combine portfolio values from multiple episodes
            combined_portfolio = []
            timestamps = []
            step = 0
            
            for pv in self.portfolio_history:
                if len(pv) > 0:
                    if not combined_portfolio:
                        combined_portfolio = pv
                        timestamps = list(range(len(pv)))
                    else:
                        # Continue from the last value of the previous episode
                        scale_factor = combined_portfolio[-1] / pv[0]
                        combined_portfolio.extend([v * scale_factor for v in pv[1:]])
                        timestamps.extend([t + step for t in range(1, len(pv))])
                
                step = len(combined_portfolio)
            
            # Create portfolio DataFrame
            portfolio_df = pd.DataFrame({
                'timestamp': timestamps,
                'portfolio_value': combined_portfolio
            })
            
            # Calculate returns and drawdowns
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['rolling_max'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['rolling_max'] - portfolio_df['portfolio_value']) / portfolio_df['rolling_max']
            
            # Save to CSV
            portfolio_file = os.path.join(self.output_dir, "backtest_portfolio.csv")
            portfolio_df.to_csv(portfolio_file, index=False)
            logger.info(f"Saved portfolio history to {portfolio_file}")
        
        # Save trade history
        if self.trade_history:
            # Create trades DataFrame
            trades_df = pd.DataFrame(self.trade_history)
            
            # Save to CSV
            trades_file = os.path.join(self.output_dir, "backtest_trades.csv")
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trade history to {trades_file}")
        
        # Save market regime analysis
        if self.market_regimes:
            regime_file = os.path.join(self.output_dir, "market_regimes.json")
            with open(regime_file, 'w') as f:
                json.dump(self.market_regimes, f, indent=4)
            logger.info(f"Saved market regime analysis to {regime_file}")
        
        if self.regime_performance:
            regime_perf_file = os.path.join(self.output_dir, "regime_comparison.json")
            with open(regime_perf_file, 'w') as f:
                json.dump(self.regime_performance, f, indent=4)
            logger.info(f"Saved regime performance comparison to {regime_perf_file}")
        
        # Save walkforward results
        if self.walkforward_results:
            wf_file = os.path.join(self.output_dir, "walkforward_results.json")
            with open(wf_file, 'w') as f:
                json.dump(self.walkforward_results, f, indent=4)
            logger.info(f"Saved walk-forward validation results to {wf_file}")
        
        logger.info("Backtest results saved successfully")

    def run_regime_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes.
        
        Returns:
            Dictionary of regime analysis results
        """
        logger.info("Running market regime analysis...")
        
        if self.data is None:
            logger.error("No data available for regime analysis. Run backtest first.")
            return None
            
        # Identify market regimes
        regimes = self._identify_market_regimes()
        self.market_regimes = regimes
        
        if not regimes:
            logger.warning("No market regimes identified")
            return None
            
        # Run backtest for each regime
        regime_performance = {}
        
        for regime_name, regime_data in regimes.items():
            logger.info(f"Running backtest for regime: {regime_name}")
            
            # Filter data for this regime
            regime_start = regime_data['start_date']
            regime_end = regime_data['end_date']
            
            # Create a temporary backtester with the regime date range
            regime_backtester = InstitutionalBacktester(
                model_path=self.model_path,
                output_dir=os.path.join(self.output_dir, f"regime_{regime_name}"),
                initial_capital=self.initial_capital,
                start_date=regime_start,
                end_date=regime_end,
                assets=self.assets,
                config_path=os.path.join(self.config['data']['cache_dir'], 'config/prod_config.yaml')
            )
            
            # Use the same model and data
            regime_backtester.model = self.model
            
            # Filter data for this regime
            regime_data_filtered = self._filter_data_by_date_range(self.data, regime_start, regime_end)
            
            if regime_data_filtered is not None and len(regime_data_filtered) > 0:
                # Run backtest on regime data
                regime_backtester.data = regime_data_filtered
                regime_backtester.env = regime_backtester.prepare_environment(regime_data_filtered)
                
                regime_metrics = regime_backtester.run_backtest(n_eval_episodes=1)
                
                # Store regime performance
                regime_performance[regime_name] = {
                    'metrics': regime_metrics,
                    'period': f"{regime_start} to {regime_end}",
                    'description': regime_data['description'],
                    'characteristics': regime_data['characteristics']
                }
            else:
                logger.warning(f"No data available for regime {regime_name}")
        
        # Compare performance across regimes
        if regime_performance:
            self.regime_performance = regime_performance
            logger.info("Regime analysis completed")
            return regime_performance
        else:
            logger.warning("No regime performance data collected")
            return None
    
    def _identify_market_regimes(self) -> Dict[str, Dict]:
        """
        Identify different market regimes in the data.
        
        Returns:
            Dictionary of market regimes with their characteristics
        """
        logger.info("Identifying market regimes...")
        
        # Ensure data has datetime index
        if not hasattr(self.data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot identify regimes by date.")
            return {}
            
        # Get the datetime index
        try:
            if 'datetime' in self.data.index.names:
                dates = self.data.index.get_level_values('datetime')
            else:
                dates = self.data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(dates):
                    logger.warning("First index level is not datetime. Cannot identify regimes by date.")
                    return {}
        except Exception as e:
            logger.warning(f"Error accessing index for regime identification: {str(e)}")
            return {}
        
        # Get unique dates and sort
        unique_dates = sorted(pd.to_datetime(dates.unique()))
        
        if len(unique_dates) < 30:
            logger.warning("Not enough data points to identify regimes")
            return {}
        
        # Method 1: Use market_regime feature if available
        if self._has_market_regime_feature():
            logger.info("Using market_regime feature for regime identification")
            regimes = self._identify_regimes_from_feature()
            if regimes:
                return regimes
        
        # Method 2: Identify based on volatility and trend
        logger.info("Identifying regimes based on volatility and trend patterns")
        
        # Calculate rolling statistics by resampling to daily
        try:
            # Get a representative asset for regime detection
            if self.assets and len(self.assets) > 0:
                primary_asset = self.assets[0]
            else:
                # Try to infer primary asset from the data
                primary_asset = None
                for col in self.data.columns:
                    if isinstance(col, tuple) and len(col) > 0:
                        primary_asset = col[0]
                        break
                
                if not primary_asset:
                    logger.warning("Could not identify a primary asset for regime detection")
                    return {}
            
            # Extract price data for the primary asset
            price_data = self._extract_price_series(primary_asset)
            
            if price_data is None or len(price_data) < 30:
                logger.warning(f"Not enough price data for {primary_asset} to identify regimes")
                return {}
            
            # Resample to daily
            daily_data = price_data.resample('D').last().dropna()
            
            # Calculate returns
            returns = daily_data.pct_change().dropna()
            
            # Calculate volatility (20-day rolling std)
            volatility = returns.rolling(window=20).std().dropna()
            
            # Calculate trend (20-day rolling mean of returns)
            trend = returns.rolling(window=20).mean().dropna()
            
            # Define regime thresholds
            high_vol_threshold = volatility.quantile(0.7)
            low_vol_threshold = volatility.quantile(0.3)
            up_trend_threshold = trend.quantile(0.6)
            down_trend_threshold = trend.quantile(0.4)
            
            # Identify regime periods
            regimes = {}
            
            # 1. Bull market (high trend, moderate volatility)
            bull_periods = self._find_consecutive_periods(
                (trend > up_trend_threshold) & (volatility < high_vol_threshold)
            )
            
            # 2. Bear market (negative trend, can be high volatility)
            bear_periods = self._find_consecutive_periods(
                trend < down_trend_threshold
            )
            
            # 3. High volatility/Crisis periods
            crisis_periods = self._find_consecutive_periods(
                volatility > high_vol_threshold
            )
            
            # 4. Low volatility/Range-bound periods
            sideways_periods = self._find_consecutive_periods(
                (volatility < low_vol_threshold) & 
                (trend >= down_trend_threshold) & 
                (trend <= up_trend_threshold)
            )
            
            # Format regimes dictionary
            for i, period in enumerate(bull_periods):
                regime_id = f"bull_{i+1}"
                regimes[regime_id] = {
                    'name': f"Bull Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Uptrend with moderate volatility from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'positive',
                        'volatility': 'moderate',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean()
                    }
                }
            
            for i, period in enumerate(bear_periods):
                regime_id = f"bear_{i+1}"
                regimes[regime_id] = {
                    'name': f"Bear Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Downtrend from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'negative',
                        'volatility': 'high',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean()
                    }
                }
            
            for i, period in enumerate(crisis_periods):
                regime_id = f"crisis_{i+1}"
                regimes[regime_id] = {
                    'name': f"High Volatility Period {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"High volatility period from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'mixed',
                        'volatility': 'very high',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean()
                    }
                }
                
            for i, period in enumerate(sideways_periods):
                regime_id = f"sideways_{i+1}"
                regimes[regime_id] = {
                    'name': f"Sideways Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Low volatility range-bound market from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'neutral',
                        'volatility': 'low',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean()
                    }
                }
            
            logger.info(f"Identified {len(regimes)} market regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"Error in regime identification: {str(e)}")
            return {}
    
    def _has_market_regime_feature(self) -> bool:
        """Check if the data has a market_regime feature"""
        for col in self.data.columns:
            if isinstance(col, tuple):
                if len(col) > 1 and col[1] == 'market_regime':
                    return True
            elif col == 'market_regime':
                return True
        return False
    
    def _identify_regimes_from_feature(self) -> Dict[str, Dict]:
        """Identify regimes based on the market_regime feature"""
        regime_feature = None
        
        # Find the market_regime column
        for col in self.data.columns:
            if isinstance(col, tuple):
                if len(col) > 1 and col[1] == 'market_regime':
                    regime_feature = col
                    break
            elif col == 'market_regime':
                regime_feature = col
                break
        
        if regime_feature is None:
            return {}
        
        # Get unique regime values
        regime_values = self.data[regime_feature].dropna().unique()
        
        regimes = {}
        for regime_value in regime_values:
            # Get rows with this regime value
            regime_mask = self.data[regime_feature] == regime_value
            regime_rows = self.data[regime_mask]
            
            if len(regime_rows) == 0:
                continue
            
            # Get start and end dates
            if hasattr(regime_rows.index, 'get_level_values'):
                if 'datetime' in regime_rows.index.names:
                    dates = regime_rows.index.get_level_values('datetime')
                else:
                    dates = regime_rows.index.get_level_values(0)
            else:
                # Try using a datetime column if index is not datetime
                if 'datetime' in regime_rows.columns:
                    dates = regime_rows['datetime']
                else:
                    continue
            
            dates = pd.to_datetime(dates)
            start_date = min(dates).strftime('%Y-%m-%d')
            end_date = max(dates).strftime('%Y-%m-%d')
            
            # Map numeric regime values to names
            regime_names = {
                0: "Sideways",
                1: "Bull",
                2: "Bear",
                3: "Crisis",
                4: "Recovery"
            }
            
            regime_name = regime_names.get(regime_value, f"Regime {regime_value}")
            
            regime_id = f"{regime_name.lower()}_{regime_value}"
            regimes[regime_id] = {
                'name': regime_name,
                'start_date': start_date,
                'end_date': end_date,
                'description': f"{regime_name} market from {start_date} to {end_date}",
                'characteristics': {
                    'regime_value': int(regime_value) if isinstance(regime_value, (int, float)) else str(regime_value),
                    'duration_days': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days,
                    'data_points': len(regime_rows)
                }
            }
        
        return regimes
    
    def _extract_price_series(self, asset: str) -> pd.Series:
        """Extract closing price series for a specific asset"""
        # Try different ways to get the price data
        price_data = None
        
        # Try as multi-index column
        for col in self.data.columns:
            if isinstance(col, tuple) and col[0] == asset and col[1] == 'close':
                price_data = self.data[col]
                break
        
        # If not found, try as individual column
        if price_data is None:
            if f"{asset}_close" in self.data.columns:
                price_data = self.data[f"{asset}_close"]
            elif "close" in self.data.columns:
                # If there's only one asset, the 'close' column might be used directly
                price_data = self.data["close"]
        
        if price_data is None:
            logger.warning(f"Could not find price data for asset {asset}")
            return None
        
        # Reset and set index for easier time-based operations
        if hasattr(self.data.index, 'get_level_values'):
            if 'datetime' in self.data.index.names:
                price_data = price_data.reset_index().set_index('datetime')
            else:
                # Assume first level is datetime
                temp_index = self.data.index.get_level_values(0)
                if pd.api.types.is_datetime64_any_dtype(temp_index):
                    price_data = pd.Series(price_data.values, index=temp_index)
        
        return price_data
    
    def _find_consecutive_periods(self, condition_mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find consecutive periods where a condition is True"""
        periods = []
        
        if len(condition_mask) == 0:
            return periods
            
        # Get dates where condition is True
        dates = condition_mask.index[condition_mask]
        
        if len(dates) == 0:
            return periods
            
        # Find edges of consecutive periods
        date_groups = []
        current_group = [dates[0]]
        
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 7:  # Allow gaps of up to 7 days
                current_group.append(dates[i])
            else:
                if len(current_group) >= 5:  # Minimum 5 days for a period
                    date_groups.append(current_group)
                current_group = [dates[i]]
        
        if len(current_group) >= 5:
            date_groups.append(current_group)
        
        # Convert groups to periods
        for group in date_groups:
            periods.append((group[0], group[-1]))
        
        return periods
    
    def _filter_data_by_date_range(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date range"""
        if not hasattr(data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot filter by date.")
            return data
            
        # Get the datetime index
        try:
            if 'datetime' in data.index.names:
                date_idx = data.index.get_level_values('datetime')
            else:
                date_idx = data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(date_idx):
                    logger.warning("First index level is not datetime. Cannot filter by date.")
                    return data
        except Exception as e:
            logger.warning(f"Error accessing index for date filtering: {str(e)}")
            return data
            
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data
        filtered_data = data[(date_idx >= start_date) & (date_idx <= end_date)]
        logger.info(f"Filtered data from {start_date} to {end_date}: {filtered_data.shape}")
        
        return filtered_data
    
    def run_walk_forward_validation(self, window_size: int = 60, step_size: int = 30) -> Dict[str, Any]:
        """
        Run walk-forward validation to test model robustness across time periods.
        
        Args:
            window_size: Number of days for each window
            step_size: Number of days to step forward for each iteration
            
        Returns:
            Dictionary of walk-forward validation results
        """
        logger.info(f"Running walk-forward validation with {window_size}-day windows and {step_size}-day steps...")
        
        if self.data is None:
            logger.error("No data available for walk-forward validation. Run backtest first.")
            return None
            
        # Get unique dates
        if hasattr(self.data.index, 'get_level_values'):
            if 'datetime' in self.data.index.names:
                dates = self.data.index.get_level_values('datetime')
            else:
                dates = self.data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(dates):
                    logger.warning("First index level is not datetime. Cannot run walk-forward validation.")
                    return None
        else:
            logger.warning("Data index does not have multiple levels. Cannot run walk-forward validation.")
            return None
            
        # Convert to dates and get unique values
        unique_dates = pd.to_datetime(dates.unique()).date
        unique_dates = sorted(unique_dates)
        
        if len(unique_dates) < window_size:
            logger.warning("Not enough data for walk-forward validation")
            return None
            
        # Create windows
        windows = []
        start_idx = 0
        
        while start_idx < len(unique_dates):
            if start_idx + window_size > len(unique_dates):
                break
                
            start_date = unique_dates[start_idx]
            end_date = unique_dates[min(start_idx + window_size - 1, len(unique_dates) - 1)]
            
            windows.append((start_date, end_date))
            start_idx += step_size
        
        if len(windows) == 0:
            logger.warning("No valid windows created for walk-forward validation")
            return None
            
        logger.info(f"Created {len(windows)} windows for walk-forward validation")
        
        # Run backtest for each window
        wf_results = []
        
        for i, (start_date, end_date) in enumerate(windows):
            logger.info(f"Running window {i+1}/{len(windows)}: {start_date} to {end_date}")
            
            # Create a temporary backtester for this window
            window_backtester = InstitutionalBacktester(
                model_path=self.model_path,
                output_dir=os.path.join(self.output_dir, f"window_{i+1}"),
                initial_capital=self.initial_capital,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                assets=self.assets,
                config_path=os.path.join(self.config['data']['cache_dir'], 'config/prod_config.yaml')
            )
            
            # Use the same model
            window_backtester.model = self.model
            
            # Filter data for this window
            window_data = self._filter_data_by_date_range(
                self.data, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if window_data is not None and len(window_data) > 0:
                # Run backtest on window data
                window_backtester.data = window_data
                window_backtester.env = window_backtester.prepare_environment(window_data)
                
                window_metrics = window_backtester.run_backtest(n_eval_episodes=1)
                
                # Store window results
                wf_results.append({
                    'window': i+1,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'metrics': window_metrics
                })
            else:
                logger.warning(f"No data available for window {i+1}")
        
        # Aggregate results
        if wf_results:
            # Calculate aggregate metrics
            agg_metrics = {
                'total_return': np.mean([r['metrics'].get('total_return', 0) for r in wf_results]),
                'sharpe_ratio': np.mean([r['metrics'].get('sharpe_ratio', 0) for r in wf_results]),
                'max_drawdown': np.mean([r['metrics'].get('max_drawdown', 0) for r in wf_results]),
                'win_rate': np.mean([r['metrics'].get('win_rate', 0) for r in wf_results]),
                'trade_count': np.mean([r['metrics'].get('trade_count', 0) for r in wf_results]),
                'avg_trade_return': np.mean([r['metrics'].get('avg_trade_return', 0) for r in wf_results]),
            }
            
            # Calculate robustness metrics
            returns = [r['metrics'].get('total_return', 0) for r in wf_results]
            sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in wf_results]
            drawdowns = [r['metrics'].get('max_drawdown', 0) for r in wf_results]
            
            robustness = {
                'return_std': np.std(returns),
                'return_min': np.min(returns),
                'return_max': np.max(returns),
                'sharpe_std': np.std(sharpes),
                'sharpe_min': np.min(sharpes),
                'sharpe_max': np.max(sharpes),
                'drawdown_std': np.std(drawdowns),
                'drawdown_min': np.min(drawdowns),
                'drawdown_max': np.max(drawdowns),
                'profit_windows': sum(1 for r in returns if r > 0),
                'loss_windows': sum(1 for r in returns if r <= 0),
                'profit_ratio': sum(1 for r in returns if r > 0) / len(returns) if len(returns) > 0 else 0
            }
            
            walkforward_results = {
                'window_results': wf_results,
                'aggregate_metrics': agg_metrics,
                'robustness_metrics': robustness,
                'window_parameters': {
                    'window_size_days': window_size,
                    'step_size_days': step_size,
                    'total_windows': len(wf_results)
                }
            }
            
            self.walkforward_results = walkforward_results
            logger.info("Walk-forward validation completed")
            return walkforward_results
        else:
            logger.warning("No walk-forward validation results collected")
            return None 

    def create_visualizations(self) -> None:
        """Create visualizations of backtest results"""
        logger.info("Creating backtest visualizations...")
        
        if not self.portfolio_history or not self.results:
            logger.warning("No backtest results available for visualization")
            return
            
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # Create equity curve with drawdowns
        self._create_equity_curve()
        
        # Create returns distribution
        self._create_returns_distribution()
        
        # Create rolling metrics
        self._create_rolling_metrics()
        
        # Create underwater chart (drawdowns)
        self._create_drawdown_chart()
        
        # Create regime performance comparison if available
        if self.regime_performance:
            self._create_regime_comparison()
            
        # Create walk-forward analysis if available
        if self.walkforward_results:
            self._create_walkforward_analysis()
            
        # Create performance tearsheet
        self._create_performance_tearsheet()
        
        logger.info("Visualization creation completed")
    
    def _create_equity_curve(self) -> None:
        """Create equity curve with drawdowns"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'timestamp': range(len(combined_portfolio)),
            'portfolio_value': combined_portfolio
        })
        
        # Calculate returns and drawdowns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['rolling_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['rolling_max'] - portfolio_df['portfolio_value']) / portfolio_df['rolling_max']
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio Value', color='blue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        
        # Plot drawdowns
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.fill_between(portfolio_df['timestamp'], 0, portfolio_df['drawdown'], color='red', alpha=0.5, label='Drawdown')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdowns')
        ax2.legend()
        
        # Invert y-axis for drawdowns (0 at top)
        ax2.invert_yaxis()
        
        # Add key performance metrics as text
        performance_text = (
            f"Total Return: {self.results.get('total_return', 0):.2%}\n"
            f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {self.results.get('max_drawdown', 0):.2%}\n"
            f"Win Rate: {self.results.get('win_rate', 0):.2%}\n"
            f"# Trades: {self.results.get('trade_count', 0)}"
        )
        
        plt.figtext(0.01, 0.01, performance_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        equity_curve_file = os.path.join(self.output_dir, "equity_curve.png")
        plt.savefig(equity_curve_file, dpi=150)
        plt.close()
        
        logger.info(f"Created equity curve visualization: {equity_curve_file}")
    
    def _create_returns_distribution(self) -> None:
        """Create returns distribution visualization"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Calculate returns
        portfolio_series = pd.Series(combined_portfolio)
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) == 0:
            logger.warning("No returns available for distribution visualization")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot returns distribution
        ax = plt.subplot(1, 1, 1)
        sns.histplot(returns, bins=50, kde=True, ax=ax, color='blue')
        ax.axvline(x=0, color='red', linestyle='--', label='Zero Return')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        
        # Calculate and display statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        stats_text = (
            f"Mean: {mean_return:.4f}\n"
            f"Std Dev: {std_return:.4f}\n"
            f"Skew: {skew:.4f}\n"
            f"Kurtosis: {kurtosis:.4f}\n"
            f"Pos/Neg: {(returns > 0).sum()}/{(returns < 0).sum()}"
        )
        
        plt.figtext(0.01, 0.01, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        returns_dist_file = os.path.join(self.output_dir, "returns_distribution.png")
        plt.savefig(returns_dist_file, dpi=150)
        plt.close()
        
        logger.info(f"Created returns distribution visualization: {returns_dist_file}")
    
    def _create_rolling_metrics(self) -> None:
        """Create visualization of rolling performance metrics"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio
        })
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate rolling metrics
        window_size = min(20, len(portfolio_df) // 10)  # Dynamic window size
        if window_size < 2:
            logger.warning("Not enough data points for rolling metrics visualization")
            return
            
        portfolio_df['rolling_return'] = portfolio_df['returns'].rolling(window=window_size).mean()
        portfolio_df['rolling_volatility'] = portfolio_df['returns'].rolling(window=window_size).std()
        portfolio_df['rolling_sharpe'] = portfolio_df['rolling_return'] / portfolio_df['rolling_volatility']
        portfolio_df['rolling_drawdown'] = ((portfolio_df['portfolio_value'].rolling(window=window_size).max() - 
                                             portfolio_df['portfolio_value']) / 
                                            portfolio_df['portfolio_value'].rolling(window=window_size).max())
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot rolling return
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(portfolio_df['rolling_return'], color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Return')
        ax1.set_title(f'Rolling {window_size}-Period Return')
        
        # Plot rolling volatility
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(portfolio_df['rolling_volatility'], color='orange')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Volatility')
        ax2.set_title(f'Rolling {window_size}-Period Volatility')
        
        # Plot rolling Sharpe
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(portfolio_df['rolling_sharpe'], color='green')
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title(f'Rolling {window_size}-Period Sharpe Ratio')
        
        plt.tight_layout()
        
        # Save figure
        rolling_metrics_file = os.path.join(self.output_dir, "rolling_metrics.png")
        plt.savefig(rolling_metrics_file, dpi=150)
        plt.close()
        
        logger.info(f"Created rolling metrics visualization: {rolling_metrics_file}")
    
    def _create_drawdown_chart(self) -> None:
        """Create underwater chart (drawdowns)"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio,
            'timestamp': range(len(combined_portfolio))
        })
        
        # Calculate drawdowns
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['portfolio_value']) / portfolio_df['peak']
        
        # Find top drawdowns
        drawdown_periods = []
        current_dd = None
        
        for i, row in portfolio_df.iterrows():
            if row['drawdown'] == 0 and current_dd is not None:
                # End of drawdown
                drawdown_periods.append({
                    'start': current_dd['start'],
                    'end': i,
                    'depth': current_dd['depth'],
                    'length': i - current_dd['start'],
                    'recovery': i - current_dd['peak']
                })
                current_dd = None
            elif row['drawdown'] > 0:
                if current_dd is None:
                    # Start of new drawdown
                    current_dd = {
                        'start': i,
                        'peak': i,
                        'depth': row['drawdown']
                    }
                elif row['drawdown'] > current_dd['depth']:
                    # New max drawdown
                    current_dd['peak'] = i
                    current_dd['depth'] = row['drawdown']
        
        # Add last drawdown if it hasn't recovered
        if current_dd is not None:
            drawdown_periods.append({
                'start': current_dd['start'],
                'end': len(portfolio_df) - 1,
                'depth': current_dd['depth'],
                'length': len(portfolio_df) - 1 - current_dd['start'],
                'recovery': 'ongoing'
            })
        
        # Sort by depth
        drawdown_periods = sorted(drawdown_periods, key=lambda x: x['depth'], reverse=True)
        
        # Take top 5
        top_drawdowns = drawdown_periods[:5]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot underwater chart
        ax1 = plt.subplot(2, 1, 1)
        ax1.fill_between(portfolio_df['timestamp'], 0, portfolio_df['drawdown'], color='red', alpha=0.5)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Drawdown')
        ax1.set_title('Underwater Chart')
        ax1.invert_yaxis()  # 0 at top
        
        # Plot top drawdowns as table
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('off')
        
        if top_drawdowns:
            table_data = [
                ['Rank', 'Start', 'End', 'Depth', 'Length', 'Recovery']
            ]
            
            for i, dd in enumerate(top_drawdowns, 1):
                table_data.append([
                    f"{i}",
                    f"{dd['start']}",
                    f"{dd['end']}",
                    f"{dd['depth']:.2%}",
                    f"{dd['length']} steps",
                    f"{dd['recovery']} steps" if isinstance(dd['recovery'], (int, float)) else dd['recovery']
                ])
            
            ax2.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.1, 0.15, 0.15, 0.15, 0.2, 0.2])
            ax2.set_title('Top 5 Drawdowns')
        
        plt.tight_layout()
        
        # Save figure
        drawdown_file = os.path.join(self.output_dir, "top_drawdowns.png")
        plt.savefig(drawdown_file, dpi=150)
        plt.close()
        
        logger.info(f"Created drawdown visualization: {drawdown_file}")
    
    def _create_regime_comparison(self) -> None:
        """Create visualization of performance across market regimes"""
        if not self.regime_performance:
            logger.warning("No regime performance data available for visualization")
            return
        
        # Extract metrics by regime
        regimes = list(self.regime_performance.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'trade_count']
        
        metric_values = {metric: [] for metric in metrics}
        regime_names = []
        
        for regime, data in self.regime_performance.items():
            regime_names.append(regime)
            
            for metric in metrics:
                value = data['metrics'].get(metric, 0)
                metric_values[metric].append(value)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot metrics by regime
        for i, metric in enumerate(metrics):
            ax = plt.subplot(len(metrics), 1, i+1)
            bars = ax.bar(regime_names, metric_values[metric])
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                if 'return' in metric or 'rate' in metric or 'drawdown' in metric:
                    # Format as percentage
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2%}', ha='center', va='bottom')
                else:
                    # Format as number
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            if i == 0:
                ax.set_title('Performance by Market Regime')
            
            if i == len(metrics) - 1:
                ax.set_xlabel('Regime')
        
        plt.tight_layout()
        
        # Save figure
        regime_file = os.path.join(self.output_dir, "regime_comparison.png")
        plt.savefig(regime_file, dpi=150)
        plt.close()
        
        logger.info(f"Created regime comparison visualization: {regime_file}")
    
    def _create_walkforward_analysis(self) -> None:
        """Create visualization of walk-forward validation results"""
        if not self.walkforward_results or 'window_results' not in self.walkforward_results:
            logger.warning("No walk-forward validation results available for visualization")
            return
        
        window_results = self.walkforward_results['window_results']
        
        if not window_results:
            logger.warning("Empty walk-forward validation results")
            return
        
        # Extract data
        window_numbers = [r['window'] for r in window_results]
        returns = [r['metrics'].get('total_return', 0) for r in window_results]
        sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in window_results]
        drawdowns = [r['metrics'].get('max_drawdown', 0) for r in window_results]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot returns by window
        ax1 = plt.subplot(3, 1, 1)
        ax1.bar(window_numbers, returns, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return')
        ax1.set_title('Returns by Window')
        
        # Add horizontal line for average return
        avg_return = np.mean(returns)
        ax1.axhline(y=avg_return, color='green', linestyle='-', label=f'Avg: {avg_return:.2%}')
        ax1.legend()
        
        # Plot Sharpe ratios by window
        ax2 = plt.subplot(3, 1, 2)
        ax2.bar(window_numbers, sharpes, color='orange')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio by Window')
        
        # Add horizontal line for average Sharpe
        avg_sharpe = np.mean(sharpes)
        ax2.axhline(y=avg_sharpe, color='green', linestyle='-', label=f'Avg: {avg_sharpe:.2f}')
        ax2.legend()
        
        # Plot drawdowns by window
        ax3 = plt.subplot(3, 1, 3)
        ax3.bar(window_numbers, drawdowns, color='red')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Max Drawdown')
        ax3.set_title('Max Drawdown by Window')
        
        # Add horizontal line for average drawdown
        avg_drawdown = np.mean(drawdowns)
        ax3.axhline(y=avg_drawdown, color='green', linestyle='-', label=f'Avg: {avg_drawdown:.2%}')
        ax3.legend()
        
        plt.tight_layout()
        
        # Save figure
        wf_file = os.path.join(self.output_dir, "walkforward_analysis.png")
        plt.savefig(wf_file, dpi=150)
        plt.close()
        
        logger.info(f"Created walk-forward analysis visualization: {wf_file}")
    
    def _create_performance_tearsheet(self) -> None:
        """Create a comprehensive performance tearsheet"""
        if not self.results:
            logger.warning("No results available for performance tearsheet")
            return
        
        # Create figure
        plt.figure(figsize=(12, 15))
        
        # Extract key metrics
        metrics = self.results
        
        # Title
        plt.suptitle('Performance Tearsheet', fontsize=16, y=0.98)
        
        # Overall performance metrics
        plt.subplot(5, 1, 1)
        plt.axis('off')
        
        perf_text = (
            "Overall Performance Metrics\n\n"
            f"Total Return: {metrics.get('total_return', 0):.2%}\n"
            f"Annual Return: {metrics.get('annual_return', 0):.2%}\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
            f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}"
        )
        
        plt.text(0.5, 0.5, perf_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Trade metrics
        plt.subplot(5, 1, 2)
        plt.axis('off')
        
        trade_text = (
            "Trade Metrics\n\n"
            f"Number of Trades: {metrics.get('trade_count', 0)}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Avg Trade Return: {metrics.get('avg_trade_return', 0):.2%}\n"
            f"Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} steps\n"
            f"Avg Leverage: {metrics.get('avg_leverage', 0):.2f}x"
        )
        
        plt.text(0.5, 0.5, trade_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Risk metrics
        plt.subplot(5, 1, 3)
        plt.axis('off')
        
        risk_text = (
            "Risk Metrics\n\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"Recovery Factor: {metrics.get('recovery_factor', 0):.2f}\n"
            f"Ulcer Index: {metrics.get('ulcer_index', 0):.4f}\n"
            f"Max Leverage: {metrics.get('max_leverage', 0):.2f}x\n"
        )
        
        plt.text(0.5, 0.5, risk_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Equity curve
        plt.subplot(5, 1, 4)
        
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        plt.plot(combined_portfolio)
        plt.title('Equity Curve')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        
        # Drawdown chart
        plt.subplot(5, 1, 5)
        
        # Calculate drawdowns
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio
        })
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['portfolio_value']) / portfolio_df['peak']
        
        plt.fill_between(range(len(portfolio_df)), 0, portfolio_df['drawdown'], color='red', alpha=0.5)
        plt.title('Drawdowns')
        plt.xlabel('Time Step')
        plt.ylabel('Drawdown')
        plt.gca().invert_yaxis()  # 0 at top
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save figure
        tearsheet_file = os.path.join(self.output_dir, "performance_tearsheet.png")
        plt.savefig(tearsheet_file, dpi=150)
        plt.close()
        
        logger.info(f"Created performance tearsheet: {tearsheet_file}") 