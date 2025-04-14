import os
import time
import json
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
from .regime_analyzer import RegimeAnalyzer
from .metrics import calculate_returns, calculate_metrics
from .visualization import create_performance_charts

# Configure logging
logger = logging.getLogger(__name__)

class InstitutionalBacktester:
    """
    Institutional-grade backtester for RL trading agents.
    
    This backtester is designed to work with models trained using the InstitutionalPerpetualEnv
    environment. It ensures compatibility between trained models and backtesting data by 
    properly handling observation spaces and feature dimensions.
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
            data_path: Path to the data file (CSV or Parquet)
            data_df: DataFrame containing market data (alternative to data_path)
            assets: List of asset symbols to trade
            initial_capital: Initial capital for backtesting
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            risk_free_rate: Annual risk-free rate for performance metrics
            commission: Trading commission rate
            slippage: Slippage rate for execution
            max_leverage: Maximum allowed leverage
            output_dir: Directory for saving backtest results
            verbose: Whether to display detailed progress and information
            benchmark_symbol: Symbol to use as benchmark (if any)
            regime_analysis: Whether to perform market regime analysis
        """
        # Store parameters
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
        self.benchmark_symbol = benchmark_symbol
        self.regime_analysis = regime_analysis
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize objects
        self.data = None
        self.model = None
        self.env = None
        self.regime_analyzer = None
        self.regime_periods = []
        self.backtest_results = {}
        
        # Check for environment normalization file
        self.env_path = os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')
        
        # Setup logging
        self._setup_logging()
        
        # Log initialization
        logger.info(f"Initialized InstitutionalBacktester with model: {model_path}")
        
    def _setup_logging(self):
        """Configure logging to file and console"""
        log_file = os.path.join(self.output_dir, "backtest.log")
        
        # Add a file handler to the logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Set log level
        logger.setLevel(logging.INFO)
        
    def load_data(self):
        """Load data from file or use provided DataFrame"""
        try:
            # If DataFrame was provided directly, use it
            if self.data_df is not None:
                logger.info("Using provided DataFrame")
                self.data = self.data_df
                
                # Find unique assets if not provided
                if not self.assets and isinstance(self.data.columns, pd.MultiIndex):
                    self.assets = list(self.data.columns.get_level_values(0).unique())
                    logger.info(f"Extracted assets from data: {self.assets}")
                    
                # Apply date filtering if needed
                if self.start_date or self.end_date:
                    self._filter_data_by_date()
                    
                return self.data
                
            # Otherwise load data from file
            if not self.data_path or not os.path.exists(self.data_path):
                raise ValueError(f"Data file not found at {self.data_path}")
                
            # Load data based on file extension
            ext = os.path.splitext(self.data_path)[1].lower()
            if ext == '.parquet':
                logger.info(f"Loading data from parquet file: {self.data_path}")
                self.data = pd.read_parquet(self.data_path)
            elif ext == '.csv':
                logger.info(f"Loading data from CSV file: {self.data_path}")
                self.data = pd.read_csv(self.data_path)
                
                # Convert timestamp column to datetime if it exists
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data.set_index('timestamp', inplace=True)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            # Apply date filtering
            if self.start_date or self.end_date:
                self._filter_data_by_date()
                
            # Find unique assets if not provided
            if not self.assets and isinstance(self.data.columns, pd.MultiIndex):
                self.assets = list(self.data.columns.get_level_values(0).unique())
                logger.info(f"Extracted assets from data: {self.assets}")
                
            logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            return self.data
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _filter_data_by_date(self):
        """Filter data by date range"""
        try:
            # Check if we have a datetime index
            if not isinstance(self.data.index, pd.DatetimeIndex):
                logger.warning("Cannot filter by date: index is not DatetimeIndex")
                return
                
            # Apply filters
            if self.start_date:
                start_date = pd.to_datetime(self.start_date)
                self.data = self.data[self.data.index >= start_date]
                
            if self.end_date:
                end_date = pd.to_datetime(self.end_date)
                self.data = self.data[self.data.index <= end_date]
                
            logger.info(f"Data filtered to shape: {self.data.shape}")
        
        except Exception as e:
            logger.error(f"Error filtering data by date: {str(e)}")
            
    def load_model(self):
        """Load the trained model"""
        try:
            from stable_baselines3 import PPO
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Check if environment normalization file exists
            if os.path.exists(self.env_path):
                logger.info(f"Found environment at: {self.env_path}")
                
            # Load the model
            self.model = PPO.load(self.model_path)
            
            # Get model observation size
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'observation_space'):
                self.model_obs_shape = self.model.policy.observation_space.shape
                logger.info(f"Model expects observation shape: {self.model_obs_shape}")
            
            logger.info("Model loaded successfully")
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _analyze_model_requirements(self):
        """Analyze the model to determine observation space requirements"""
        if not self.model:
            self.load_model()
            
        # Check observation space
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'observation_space'):
            obs_shape = self.model.policy.observation_space.shape
            logger.info(f"Model requires observation shape: {obs_shape}")
            
            # Get the exact observation dimension - critical for compatibility
            self.observation_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape[0] * obs_shape[1]
            logger.info(f"Model requires {self.observation_dim} features total")
            
            # Check action space
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'action_space'):
                action_shape = self.model.policy.action_space.shape
                logger.info(f"Model action space shape: {action_shape}")
                
                # Get number of assets from action space
                self.num_actions = action_shape[0] if len(action_shape) > 0 else 1
                
            return self.observation_dim
        else:
            logger.warning("Could not determine model observation requirements")
            return None
            
    def _adapt_data_to_model(self):
        """Adapt data to match model requirements"""
        if not hasattr(self, 'observation_dim'):
            self._analyze_model_requirements()
            
        if not self.data is not None:
            self.load_data()
            
        if not isinstance(self.data.columns, pd.MultiIndex):
            logger.error("Data must have MultiIndex columns with (asset, feature) structure")
            return False
            
        # Get current data dimensions
        assets = list(self.data.columns.get_level_values(0).unique())
        num_assets = len(assets)
        features_per_asset = len(self.data.xs(assets[0], axis=1, level=0).columns)
        
        logger.info(f"Data has {num_assets} assets with {features_per_asset} features each")
        logger.info(f"Total features in data: {num_assets * features_per_asset}")
        
        # Calculate features needed per asset to match model
        if hasattr(self, 'observation_dim'):
            # Model observation space may include portfolio features
            # Assuming standard portfolio feature count from InstitutionalPerpetualEnv
            # Each asset has 3 portfolio features + 3 global features
            portfolio_features_per_asset = 3
            global_features = 3
            
            # Calculate core market features needed per asset
            market_features_per_asset = (self.observation_dim - (portfolio_features_per_asset * num_assets) - global_features) // num_assets
            
            logger.info(f"Model requires approximately {market_features_per_asset} market features per asset")
            
            # If we have too few features, pad the data
            if features_per_asset < market_features_per_asset:
                logger.info(f"Data has fewer features than model expects. Adding padding features")
                # This will be handled in the environment
            
            # If we have too many features, note this (will be handled in environment)
            elif features_per_asset > market_features_per_asset:
                logger.info(f"Data has more features than model expects. Some features will be ignored")
                
        return True
        
    def _create_environment(self):
        """Create a fresh environment for backtesting"""
        from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
        
        logger.info("Creating backtesting environment")
        
        # Make sure model is loaded to get observation requirements
        if not self.model:
            self.load_model()
            
        # Make sure data is prepared
        if self.data is None:
            self.load_data()
            
        # Analyze model requirements and adapt data
        self._adapt_data_to_model()
            
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
            verbose=False  # Disable verbose logging during backtesting
        )
        
        # Verify environment observation dimension matches model
        if hasattr(self, 'observation_dim'):
            actual_obs_shape = env.observation_space.shape
            actual_dim = actual_obs_shape[0]
            
            logger.info(f"Environment observation shape: {actual_obs_shape}")
            logger.info(f"Model requires shape with dimension: {self.observation_dim}")
            
            if actual_dim != self.observation_dim:
                # Try to fix the environment's observation space
                logger.warning(f"Observation space mismatch: env has {actual_dim}, model expects {self.observation_dim}")
                
                from gymnasium import spaces
                # Replace environment's observation space with the model's expected shape
                # This is a critical fix to ensure compatibility
                env.observation_space = spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.observation_dim,), 
                    dtype=np.float32
                )
                
                # Monkey patch the _get_observation method to ensure correct shape
                original_get_obs = env._get_observation
                
                def patched_get_observation():
                    """Patched observation function to match model's expected shape"""
                    try:
                        # Get original observation
                        obs = original_get_obs()
                        
                        # Check if shapes match
                        if len(obs) == self.observation_dim:
                            return obs
                            
                        # If too short, pad with zeros
                        if len(obs) < self.observation_dim:
                            logger.debug(f"Padding observation from {len(obs)} to {self.observation_dim}")
                            padding = np.zeros(self.observation_dim - len(obs), dtype=np.float32)
                            return np.concatenate([obs, padding])
                            
                        # If too long, truncate
                        if len(obs) > self.observation_dim:
                            logger.debug(f"Truncating observation from {len(obs)} to {self.observation_dim}")
                            return obs[:self.observation_dim]
                            
                    except Exception as e:
                        logger.error(f"Error in patched observation function: {str(e)}")
                        return np.zeros(self.observation_dim, dtype=np.float32)
                        
                # Replace the method
                env._get_observation = patched_get_observation
                logger.info(f"Patched environment observation function to ensure shape: {self.observation_dim}")
                
        # Wrap environment
        env = DummyVecEnv([lambda: env])
        
        # If we have a saved environment, try to load normalization stats
        if self.env_path and os.path.exists(self.env_path):
            try:
                logger.info(f"Loading environment normalization from: {self.env_path}")
                
                # First create a new VecNormalize wrapper that matches our new environment
                temp_normalize = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
                
                # Now try to load the saved normalization statistics
                with open(self.env_path, "rb") as f:
                    saved_env = pickle.load(f)
                    
                # Check if shapes match before copying statistics
                saved_obs_rms = getattr(saved_env, "obs_rms", None)
                if saved_obs_rms is not None:
                    saved_mean_shape = saved_obs_rms.mean.shape[0]
                    current_mean_shape = temp_normalize.obs_rms.mean.shape[0]
                    
                    if saved_mean_shape == current_mean_shape:
                        # Copy normalization statistics
                        temp_normalize.obs_rms.mean = saved_env.obs_rms.mean
                        temp_normalize.obs_rms.var = saved_env.obs_rms.var
                        temp_normalize.obs_rms.count = saved_env.obs_rms.count
                        logger.info("Successfully copied normalization statistics")
                    else:
                        logger.warning(f"Cannot copy normalization stats: shape mismatch ({saved_mean_shape} vs {current_mean_shape})")
                
                # Use our prepared environment
                env = temp_normalize
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
        from .regime_analyzer import RegimeAnalyzer
        
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
        logger.info(f"Identified {len(self.regime_periods)} market regime periods")
        
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
        if hasattr(unwrapped_env, 'df') and isinstance(unwrapped_env.df.index, pd.DatetimeIndex):
            timestamps.append(unwrapped_env.df.index[0])
            
        # Main simulation loop
        pbar = tqdm(total=len(unwrapped_env.df) - unwrapped_env.window_size, desc="Backtesting", disable=not self.verbose)
        
        while not done:
            try:
                # Get prediction from model - wrapped in try/except to catch shape mismatch issues
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
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
                
            except Exception as e:
                logger.error(f"Error during backtest step: {str(e)}")
                import traceback
                traceback.print_exc()
                done = True
                
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
        from .metrics import calculate_metrics
        
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
            
        # Create timestamp index for time-based metrics
        if timestamps and len(timestamps) == len(portfolio_values):
            # Convert timestamps to pandas DatetimeIndex if they are strings
            if isinstance(timestamps[0], str):
                timestamps = pd.to_datetime(timestamps)
                
            results['timestamp_index'] = timestamps
            
        # Calculate metrics
        metrics = calculate_metrics(
            portfolio_values=portfolio_values,
            returns=returns,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            trades=trades,
            timestamps=timestamps,
            leverages=leverages
        )
        
        # Add metrics to results
        results['metrics'] = metrics
        
        # Store complete results
        self.backtest_results = results
        
        # Display key metrics
        if self.verbose:
            print("\n====== BACKTEST RESULTS ======")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print("==============================")
            
    def _create_visualizations(self, results):
        """Create visualization charts"""
        from .visualization import create_performance_charts
        
        logger.info("Creating performance visualizations")
        
        # Save charts to output directory
        chart_path = os.path.join(self.output_dir, 'performance_charts.png')
        
        try:
            create_performance_charts(
                results=results,
                output_path=chart_path,
                show=False,
                regime_periods=self.regime_periods if self.regime_analysis else None
            )
            logger.info(f"Performance charts saved to: {chart_path}")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            
    def _save_results(self):
        """Save backtest results to files"""
        logger.info("Saving backtest results")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save backtest information
        backtest_info = {
            'model_path': self.model_path,
            'assets': self.assets,
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'max_leverage': self.max_leverage,
            'risk_free_rate': self.risk_free_rate,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'benchmark_symbol': self.benchmark_symbol,
            'regime_analysis': self.regime_analysis,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save backtest info
        with open(os.path.join(self.output_dir, 'backtest_info.json'), 'w') as f:
            json.dump(backtest_info, f, indent=2)
            
        # Save metrics
        if 'metrics' in self.backtest_results:
            metrics = self.backtest_results['metrics']
            with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                # Convert any numpy types to native Python types
                metrics_json = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.integer, np.floating, np.bool_)):
                        metrics_json[k] = v.item()
                    else:
                        metrics_json[k] = v
                json.dump(metrics_json, f, indent=2)
                
        # Save trades
        if 'trades' in self.backtest_results and self.backtest_results['trades']:
            trades_df = pd.DataFrame(self.backtest_results['trades'])
            trades_df.to_csv(os.path.join(self.output_dir, 'trades.csv'), index=False)
            
        # Save portfolio values
        if 'portfolio_values' in self.backtest_results and 'timestamps' in self.backtest_results:
            portfolio_df = pd.DataFrame({
                'timestamp': self.backtest_results['timestamps'],
                'portfolio_value': self.backtest_results['portfolio_values']
            })
            portfolio_df.to_csv(os.path.join(self.output_dir, 'portfolio_values.csv'), index=False)
            
        logger.info(f"Backtest results saved to: {self.output_dir}")

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
    Run institutional-grade backtest with a trained model.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data file (CSV or Parquet)
        data_df: DataFrame containing market data (alternative to data_path)
        assets: List of asset symbols to trade
        initial_capital: Initial capital for backtesting
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        output_dir: Directory for saving backtest results
        regime_analysis: Whether to perform market regime analysis
        walk_forward: Whether to perform walk-forward validation
        
    Returns:
        Dictionary with backtest results
    """
    # Create backtester
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
    
    # Run backtest
    if walk_forward:
        return backtester.run_walk_forward_validation()
    else:
        return backtester.run_backtest() 