#!/usr/bin/env python3
"""
Real-time Trading Monitor

This script implements a real-time trading monitor that:
1. Fetches constant price data for BTC, ETH, and SOL on 5m timeframes
2. Executes trades via the RL model similar to training/backtesting
3. Logs all trades to a file
4. Logs current positions every 5 minutes
5. Logs market commentary every 5 minutes
6. Handles synchronization between data sources and ensures error-free operation

This is production-ready code for an AI quant crypto hedge fund.
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
import yaml
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import threading
import signal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import ccxt.async_support as ccxt_async
import ta

# Local imports - modify paths as needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_llm.model import TradingLLM
from trading_llm.chatbot import MarketChatbot
from trading_llm.inference import MarketCommentaryGenerator, extract_trading_signals
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/realtime_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('realtime_trading')

class RealTimeTradeMonitor:
    """
    Real-time trading monitor with constant data feeds, automatic trading,
    position logging, and market commentary generation.
    """
    def __init__(
        self,
        rl_model_path: str,
        llm_model_path: str,
        llm_base_model: Optional[str] = None,
        config_path: str = 'config/realtime_config.yaml',
        log_dir: str = 'logs',
        timeframe: str = '5m',
        symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        initial_balance: float = 10000.0,
        max_leverage: float = 20.0
    ):
        """
        Initialize the real-time trading monitor.
        
        Args:
            rl_model_path: Path to the trained RL model
            llm_model_path: Path to the LLM model for commentary
            llm_base_model: Base model for LLM if using LoRA
            config_path: Path to configuration file
            log_dir: Directory for logs
            timeframe: Timeframe for price data
            symbols: List of symbols to trade
            initial_balance: Initial account balance
            max_leverage: Maximum allowed leverage
        """
        self.rl_model_path = rl_model_path
        self.llm_model_path = llm_model_path
        self.llm_base_model = llm_base_model
        self.config_path = config_path
        self.log_dir = log_dir
        self.timeframe = timeframe
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/trades", exist_ok=True)
        os.makedirs(f"{log_dir}/positions", exist_ok=True)
        os.makedirs(f"{log_dir}/commentary", exist_ok=True)
        
        # Setup log files
        self.trade_log_file = f"{log_dir}/trades/trades.json"
        self.position_log_file = f"{log_dir}/positions/positions.json"
        self.commentary_log_file = f"{log_dir}/commentary/commentary.json"
        
        # Load configuration
        self.config = self._load_config()
        
        # State variables
        self.is_running = False
        self.shutting_down = False
        
        # Initialize market data storage
        self.market_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        
        # Initialize positions
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
                'value': 0.0
            } for symbol in self.symbols
        }
        
        # Trade history
        self.trades = []
        
        # Balance and PnL tracking
        self.balance = initial_balance
        self.total_value = initial_balance
        
        # Timing trackers
        self.last_position_log_time = datetime.now()
        self.last_commentary_time = datetime.now()
        
        # Add timestamp tracking for signal generation
        self.last_signal_time = {}
        for symbol in self.symbols:
            self.last_signal_time[symbol] = datetime.min
        self.last_candle_times = {}
        for symbol in self.symbols:
            self.last_candle_times[symbol] = datetime.min
        
        # Data lock for thread safety
        self.data_lock = threading.Lock()
        
        # Exchange client for data fetching
        self.exchange = None
        
        # Models
        self.rl_model = None
        self.llm_model = None
        self.chatbot = None
        self.commentary_generator = None
        
        # Transaction costs from config
        self.commission = self.config.get('trading', {}).get('commission', 0.0004)
        
        # Signal threshold from config
        self.signal_threshold = self.config.get('trading', {}).get('signal_threshold', 0.1)
        
        # Initialize trading environment for consistent leverage calculations
        try:
            from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
            
            # First, try to inspect the class signature to understand the proper parameters
            try:
                from inspect import signature
                env_signature = signature(InstitutionalPerpetualEnv.__init__)
                logger.info(f"Environment constructor signature: {env_signature}")
                # This will help us see the actual parameter names
            except Exception as e:
                logger.warning(f"Could not inspect environment signature: {e}")
            
            # Create a DataFrame with the correct MultiIndex columns structure and one row of data
            tuples = [(symbol, feat) for symbol in symbols for feat in ['open', 'high', 'low', 'close', 'volume']]
            columns = pd.MultiIndex.from_tuples(tuples, names=['asset', 'feature'])
            
            # Create a single row of data (all 1.0 values)
            data = np.ones((1, len(tuples)))
            
            # Create DataFrame with index and data
            empty_df = pd.DataFrame(data, columns=columns, index=pd.DatetimeIndex([pd.Timestamp.now()]))
            
            # Set up default last_prices dictionary for market prices
            last_prices = {symbol: 1000.0 for symbol in symbols}
            
            # Create environment config
            env_config = {
                'df': empty_df,
                'assets': symbols,
                'initial_balance': initial_balance,
                'max_leverage': max_leverage,
                'commission': self.commission,
                'window_size': 100
            }
            
            self.trading_env = InstitutionalPerpetualEnv(**env_config)
            
            # Initialize last_prices in the environment
            self.trading_env.last_prices = last_prices
            
            # Create update_market_price method if it doesn't exist
            if not hasattr(self.trading_env, 'update_market_price'):
                def update_market_price(self_env, asset, price):
                    """Update the mark price for an asset"""
                    self_env.last_prices[asset] = float(price)
                    logger.debug(f"Updated market price for {asset}: {price}")
                
                # Add the method to the environment instance
                import types
                self.trading_env.update_market_price = types.MethodType(update_market_price, self.trading_env)
            
            logger.info("Initialized trading environment for leverage calculations")
            
            # Store risk limits separately to be used by the risk engine
            risk_limits = {
                'max_position_value_usd': 0.8 * self.initial_balance,
                'max_portfolio_leverage': self.max_leverage * 0.9,
                'max_portfolio_drawdown_pct': 0.3,
                'max_single_order_value_usd': 0.5 * self.initial_balance,
                'max_portfolio_value_usd': 2.0 * self.initial_balance,
                'min_available_balance_pct': 0.1,
            }
            
            # Add any additional configuration from self.config
            if 'trading_env' in self.config:
                for key, value in self.config['trading_env'].items():
                    if key in ['df', 'assets', 'window_size', 'max_leverage', 
                              'commission', 'funding_fee_multiplier', 'base_features',
                              'tech_features', 'risk_free_rate', 'initial_balance',
                              'max_drawdown', 'maintenance_margin', 'max_steps',
                              'max_no_trade_steps', 'enforce_min_leverage', 'verbose',
                              'training_mode']:
                        env_config[key] = value
            
            # Log config
            logger.info("Initializing institutional perpetual environment with config:")
            for key, value in env_config.items():
                logger.info(f"  {key}: {value}")
            
            logger.info(f"Risk limits (to be applied separately): {risk_limits}")
            
            # Try importing the class with error handling to see details of the class requirements
            try:
                from inspect import signature
                env_signature = signature(InstitutionalPerpetualEnv.__init__)
                logger.info(f"Environment constructor signature: {env_signature}")
            except Exception as e:
                logger.warning(f"Could not inspect environment signature: {e}")
            
            # Initialize the environment
            self.trading_env = InstitutionalPerpetualEnv(**env_config)
            
            # Initialize risk engine explicitly with risk limits
            if hasattr(self.trading_env, 'initialize_risk_engine'):
                try:
                    self.trading_env.initialize_risk_engine(risk_limits=risk_limits)
                    logger.info("Risk engine initialized with custom limits")
                except Exception as e:
                    logger.error(f"Error initializing risk engine with params: {e}")
                    logger.warning("Attempting to initialize risk engine without parameters")
                    self.trading_env.initialize_risk_engine()
            
            logger.info("Initialized trading environment for position management and risk assessment")
            
        except Exception as e:
            logger.error(f"Error initializing trading environment: {str(e)}")
            logger.warning("Will use simplified leverage calculation instead")
            self.trading_env = None
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            # Return default config if file not found
            return {
                'trading': {
                    'commission': 0.0004,
                    'signal_threshold': 0.1,
                    'trade_cooldown_minutes': 5
                },
                'data': {
                    'backfill_periods': 1000
                }
            }
    
    def setup(self):
        """
        Set up the trading environment and prepare for trading.
        """
        try:
            self.logger.info("Setting up the trading environment")
            
            # Create necessary directories if they don't exist
            os.makedirs(self.config['market_data_dir'], exist_ok=True)
            os.makedirs(self.config['log_dir'], exist_ok=True)
            
            # Backfill initial data if needed
            if self.config.get('backfill_initial_data', True):
                self._backfill_initial_data()
            
            # Load market data from files if available
            self.market_data = {}
            for timeframe in self.config.get('timeframes', ['5m']):
                df = self._load_data_from_files(timeframe)
                if not df.empty:
                    self.market_data[timeframe] = df
                    self.logger.info(f"Loaded market data for timeframe {timeframe} with shape {df.shape}")
                else:
                    self.logger.warning(f"No market data loaded for timeframe {timeframe}")
            
            # Initialize the trading environment
            timeframe = self.config.get('timeframe', '5m')
            
            if timeframe not in self.market_data or self.market_data[timeframe].empty:
                raise ValueError(f"No market data available for timeframe {timeframe}")
            
            df = self.market_data[timeframe]
            
            # Set up environment configuration
            env_config = {
                # Risk management parameters
                'max_position_value_usd': self.config.get('max_position_value_usd', 1000),
                'max_portfolio_leverage': self.config.get('max_portfolio_leverage', 1.0),
                'max_portfolio_drawdown_pct': self.config.get('max_portfolio_drawdown_pct', 5.0),
                'position_sizing_method': self.config.get('position_sizing_method', 'fixed_notional'),
                'initial_capital': self.config.get('initial_capital', 10000),
                
                # Trade execution settings
                'use_market_orders': self.config.get('use_market_orders', True),
                'market_order_slippage_pct': self.config.get('market_order_slippage_pct', 0.05),
                'timeframe': timeframe,
                'trading_symbols': self.symbols,
                
                # ML model settings if enabled
                'use_ml_model': self.config.get('use_ml_model', False),
                'ml_model_path': self.config.get('ml_model_path', None),
                'feature_config': self.config.get('feature_config', {}),
                
                # General settings
                'verbose': self.config.get('verbose', 1)
            }
            
            self.logger.info(f"Environment config: {json.dumps(env_config, indent=2)}")
            
            # Initialize the environment with our market data
            try:
                self.env = InstitutionalPerpetualEnv(df=df, **env_config)
                self.logger.info("Environment initialized successfully")
                
                # Store initial step position for future reference
                self.initial_step = self.env.current_step
                self.logger.info(f"Initial environment step set to {self.initial_step}")
                
            except Exception as e:
                self.logger.error(f"Error initializing environment: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
            
            # Initialize the trading client if real trading is enabled
            if self.config.get('real_trading', False):
                self._init_trading_client()
            
            # We're ready to start trading
            self.logger.info("Trading environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _backfill_initial_data(self, timeframe):
        """
        Backfill initial market data from exchange or load from files if available.
        
        Args:
            timeframe (str): Timeframe to backfill data for (e.g., '5m')
        """
        self.logger.info(f"Backfilling initial {timeframe} data")
        
        # Try to load existing data first
        data_dir = os.path.join('data', 'market_data', timeframe)
        combined_file = os.path.join(data_dir, f'combined_{timeframe}.parquet')
        
        try:
            if os.path.exists(combined_file):
                self.logger.info(f"Loading existing combined data from {combined_file}")
                all_data = pd.read_parquet(combined_file)
                
                # Check if the data has MultiIndex columns
                if not isinstance(all_data.columns, pd.MultiIndex):
                    self.logger.warning("Converting loaded data to MultiIndex format")
                    # Create appropriate MultiIndex columns
                    columns = all_data.columns
                    unique_symbols = list(set([col.split('_')[0] for col in columns if '_' in col]))
                    
                    # Extract base columns like 'open', 'high', etc.
                    base_columns = set()
                    for col in columns:
                        parts = col.split('_')
                        if parts[0] in self.symbols:
                            base_columns.add('_'.join(parts[1:]))
                    
                    # Create MultiIndex columns
                    mi_columns = pd.MultiIndex.from_product([unique_symbols, list(base_columns)])
                    new_df = pd.DataFrame(index=all_data.index, columns=mi_columns)
                    
                    # Copy data to the new DataFrame
                    for col in columns:
                        parts = col.split('_')
                        if parts[0] in self.symbols:
                            symbol = parts[0]
                            feature = '_'.join(parts[1:])
                            new_df[(symbol, feature)] = all_data[col]
                    
                    all_data = new_df
                
                # Log the latest price for each symbol
                for symbol in self.symbols:
                    if (symbol, 'close') in all_data.columns:
                        latest_close = all_data[(symbol, 'close')].iloc[-1]
                        self.logger.info(f"Latest {symbol} close price: {latest_close}")
                    else:
                        self.logger.warning(f"Column ({symbol}, 'close') not found in data")
                
                return all_data
        except Exception as e:
            self.logger.error(f"Error loading existing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Continue to fetch data from exchange
        
        # If we couldn't load data or it doesn't exist, fetch from exchange
        self.logger.info("No existing data found or error loading. Fetching from exchange.")
        
        # Create a dictionary to store data for each symbol
        all_symbol_data = {}
        
        # Fetch data for each symbol
        for symbol in self.symbols:
            try:
                # Get OHLCV data
                self.logger.info(f"Fetching {symbol} {timeframe} data from exchange")
                since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)  # Last 30 days
                
                # Use ccxt to fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe,
                    since=since, 
                    limit=1000  # Adjust based on exchange limits
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                df = self._calculate_indicators(df, symbol)
                
                # Store in dictionary
                all_symbol_data[symbol] = df
                
                self.logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # Combine data for all symbols into a MultiIndex DataFrame
        if all_symbol_data:
            # Create an empty DataFrame to store combined data
            all_data = None
            
            for symbol, df in all_symbol_data.items():
                # Create a MultiIndex DataFrame for each symbol
                columns = pd.MultiIndex.from_product([[symbol], df.columns])
                symbol_df = pd.DataFrame(df.values, index=df.index, columns=columns)
                
                if all_data is None:
                    all_data = symbol_df
                else:
                    # Merge with existing data
                    all_data = all_data.join(symbol_df, how='outer')
            
            # Sort by timestamp and fill NaN values
            if all_data is not None:
                all_data.sort_index(inplace=True)
                all_data.fillna(method='ffill', inplace=True)
                
                # Save the data to files
                self._save_market_data_to_files(all_data, timeframe)
                
                return all_data
        
        return None
    
    def _save_market_data_to_files(self, df, timeframe):
        """
        Save market data to files with proper MultiIndex format.
        
        Args:
            df: DataFrame with MultiIndex columns to save
            timeframe: Timeframe of the data
        """
        try:
            # Ensure the data directory exists
            data_dir = os.path.join(self.config['market_data_dir'], timeframe)
            os.makedirs(data_dir, exist_ok=True)
            
            # Verify we have a MultiIndex DataFrame
            if not isinstance(df.columns, pd.MultiIndex):
                self.logger.warning("DataFrame does not have MultiIndex columns - attempting to fix")
                
                # Try to convert to MultiIndex if possible
                columns = df.columns
                if any('_' in col for col in columns):
                    tuples = [col.split('_', 1) for col in columns]
                    df.columns = pd.MultiIndex.from_tuples(tuples)
                else:
                    # If no underscore pattern exists, we can't auto-fix
                    self.logger.error("Cannot convert DataFrame to MultiIndex format - columns lack pattern")
                    return False
            
            # Save the combined MultiIndex data
            multi_index_file = os.path.join(data_dir, "multi_index_data.parquet")
            self.logger.info(f"Saving MultiIndex data to {multi_index_file}")
            df.to_parquet(multi_index_file)
            
            # Also save as CSV for easier inspection
            csv_file = os.path.join(data_dir, "multi_index_data.csv")
            df.to_csv(csv_file)
            
            # Also save individual files per symbol
            symbols = list(set([col[0] for col in df.columns]))
            for symbol in symbols:
                symbol_data = df.loc[:, (symbol, slice(None))]
                
                if not symbol_data.empty:
                    # Create a clean filename
                    clean_symbol = symbol.replace('/', '')
                    symbol_file = os.path.join(data_dir, f"{clean_symbol}.parquet")
                    self.logger.info(f"Saving data for {symbol} to {symbol_file}")
                    symbol_data.to_parquet(symbol_file)
                    
                    # Also save as CSV for easier inspection
                    csv_symbol_file = os.path.join(data_dir, f"{clean_symbol}.csv")
                    symbol_data.to_csv(csv_symbol_file)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving market data to files: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_data_from_files(self, timeframe=None):
        """
        Load market data from files for the specified timeframe.
        
        Args:
            timeframe: Specific timeframe to load, or None for all timeframes
        
        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        if timeframe is None:
            timeframes = self.config.get('timeframes', ['5m'])
        else:
            timeframes = [timeframe]
        
        success = False
        
        for tf in timeframes:
            data_dir = os.path.join(self.config['market_data_dir'], tf)
            
            # Check if directory exists
            if not os.path.exists(data_dir):
                self.logger.warning(f"Data directory does not exist: {data_dir}")
                continue
            
            # Try to load multi-index data file first
            multi_index_file = os.path.join(data_dir, "multi_index_data.parquet")
            if os.path.exists(multi_index_file):
                try:
                    self.logger.info(f"Loading multi-index data from {multi_index_file}")
                    df = pd.read_parquet(multi_index_file)
                    
                    # Ensure the DataFrame has MultiIndex columns
                    if not isinstance(df.columns, pd.MultiIndex):
                        self.logger.warning(f"File {multi_index_file} doesn't have MultiIndex columns - attempting to fix")
                        
                        # Try to convert to MultiIndex if possible
                        columns = df.columns
                        tuples = [col if isinstance(col, tuple) else (col.split('_')[0], '_'.join(col.split('_')[1:])) 
                                  for col in columns]
                        df.columns = pd.MultiIndex.from_tuples(tuples)
                    
                    # Ensure the index is a DateTimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # Sort by timestamp
                    df = df.sort_index()
                    
                    # Store in market_data dictionary
                    self.market_data[tf] = df
                    self.logger.info(f"Successfully loaded {len(df)} rows for timeframe {tf}")
                    
                    # Log info about loaded data
                    symbols = list(set([col[0] for col in df.columns]))
                    self.logger.info(f"Loaded data for symbols: {symbols}")
                    for symbol in symbols:
                        if (symbol, 'close') in df.columns:
                            self.logger.info(f"Latest price for {symbol}: {df[(symbol, 'close')].iloc[-1]}")
                    
                    success = True
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading multi-index data: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
            # If multi-index file doesn't exist, try to load individual symbol files
            self.logger.info(f"No multi-index file found, trying to load individual symbol files")
            
            combined_data = None
            for symbol in self.symbols:
                symbol_file = os.path.join(data_dir, f"{symbol}.parquet")
                
                if not os.path.exists(symbol_file):
                    self.logger.warning(f"No data file found for {symbol} at {symbol_file}")
                    continue
                
                try:
                    self.logger.info(f"Loading data for {symbol} from {symbol_file}")
                    symbol_df = pd.read_parquet(symbol_file)
                    
                    # Check if it already has a MultiIndex
                    if not isinstance(symbol_df.columns, pd.MultiIndex):
                        self.logger.warning(f"DataFrame for {symbol} doesn't have MultiIndex columns - attempting to fix")
                        
                        # Determine which format the data is in
                        if any('_' in col for col in symbol_df.columns):
                            # Format is like "BTC_close"
                            new_columns = []
                            for col in symbol_df.columns:
                                parts = col.split('_', 1)
                                if len(parts) == 2:
                                    new_columns.append((parts[0], parts[1]))
                                else:
                                    new_columns.append((symbol, col))
                            symbol_df.columns = pd.MultiIndex.from_tuples(new_columns)
                        else:
                            # Standard columns - construct MultiIndex
                            symbol_df.columns = pd.MultiIndex.from_product(
                                [[symbol], symbol_df.columns],
                                names=['asset', 'feature']
                            )
                    
                    # Ensure the index is a DateTimeIndex
                    if not isinstance(symbol_df.index, pd.DatetimeIndex):
                        symbol_df.index = pd.to_datetime(symbol_df.index)
                    
                    # Sort by timestamp
                    symbol_df = symbol_df.sort_index()
                    
                    # Combine data
                    if combined_data is None:
                        combined_data = symbol_df
                    else:
                        # Merge on index (timestamp)
                        combined_data = combined_data.join(symbol_df, how='outer')
                    
                    success = True
                except Exception as e:
                    self.logger.error(f"Error loading data for {symbol}: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
            if combined_data is not None and not combined_data.empty:
                # Handle NaN values
                combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
                
                # Store in market_data dictionary
                self.market_data[tf] = combined_data
                self.logger.info(f"Combined {len(combined_data)} rows for timeframe {tf}")
                
                # Save the combined data as multi-index file for future use
                self._save_market_data_to_files(combined_data, tf)
            else:
                self.logger.warning(f"No data loaded for timeframe {tf}")
        
        return success
    
    def _calculate_indicators(self, df, symbol):
        """
        Calculate technical indicators for a given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data
            symbol (str): Symbol for which indicators are being calculated
            
        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        try:
            self.logger.info(f"Calculating indicators for {symbol}")
            
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Check if we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found in DataFrame for {symbol}")
                    return df
            
            # Moving Averages
            sma_periods = [7, 20, 50, 100, 200]
            for period in sma_periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            ema_periods = [9, 20, 50, 200]
            for period in ema_periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Bollinger Bands (20, 2)
            period = 20
            std_dev = 2
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            df['bb_std'] = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
            
            # RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr_14'] = true_range.rolling(14).mean()
            
            # Price relative to moving averages
            df['price_to_sma_50'] = df['close'] / df['sma_50'] - 1
            df['price_to_sma_200'] = df['close'] / df['sma_200'] - 1
            
            # Volatility
            df['volatility_30'] = df['close'].pct_change().rolling(30).std() * (252**0.5)  # Annualized
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Fill NaN values
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            
            self.logger.info(f"Calculated {len(df.columns) - 5} indicators for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return the original DataFrame if there was an error
            return df
    
    async def _fetch_latest_data(self):
        """Fetch the latest market data from the exchange."""
        try:
            new_data_available = False
            market_conditions_changed = False
            
            for symbol in self.symbols:
                # Determine timeframe in seconds
                tf_seconds = self._timeframe_to_seconds(self.timeframe)
                
                # Fetch latest candles
                logger.debug(f"Fetching latest data for {symbol}")
                candles = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    limit=10  # Just get a few recent candles
                )
                
                if not candles:
                    logger.warning(f"No data received for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Get the timestamp of the most recent complete candle
                # The last candle is often incomplete, so use the second-to-last if available
                if len(df) > 1:
                    newest_complete_candle_time = df['timestamp'].iloc[-2]
                else:
                    newest_complete_candle_time = df['timestamp'].iloc[-1]
                
                # Check if this is a new candle we haven't processed yet
                if symbol in self.last_candle_times:
                    previous_candle_time = self.last_candle_times[symbol]
                    if newest_complete_candle_time > previous_candle_time:
                        logger.info(f"New candle available for {symbol}: {newest_complete_candle_time}")
                        self.last_candle_times[symbol] = newest_complete_candle_time
                        new_data_available = True
                        market_conditions_changed = True
                else:
                    # First time seeing this symbol
                    self.last_candle_times[symbol] = newest_complete_candle_time
                    new_data_available = True
                    market_conditions_changed = True
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Ensure DataFrame is sorted by time
                df = df.sort_index()
                
                # Update market data
                if symbol in self.market_data and not self.market_data[symbol].empty:
                    # Get only new data
                    last_timestamp = self.market_data[symbol].index[-1]
                    new_df = df[df.index > last_timestamp]
                    
                    if not new_df.empty:
                        # Calculate indicators
                        new_df = self._calculate_indicators(new_df, symbol)
                        
                        # Get column names and data values
                        columns = new_df.columns
                        values = new_df.values
                        
                        # Create MultiIndex columns
                        multi_columns = pd.MultiIndex.from_product(
                            [[symbol], columns],
                            names=['asset', 'feature']
                        )
                        
                        # Create DataFrame with proper MultiIndex columns
                        new_multi_df = pd.DataFrame(
                            values, 
                            index=new_df.index,
                            columns=multi_columns
                        )
                        
                        # Concat with existing data
                        self.market_data[symbol] = pd.concat([self.market_data[symbol], new_multi_df])
                        logger.debug(f"Added {len(new_df)} new rows for {symbol}")
                else:
                    # Calculate indicators
                    df = self._calculate_indicators(df, symbol)
                    
                    # Get column names and data values
                    columns = df.columns
                    values = df.values
                    
                    # Create MultiIndex columns
                    multi_columns = pd.MultiIndex.from_product(
                        [[symbol], columns],
                        names=['asset', 'feature']
                    )
                    
                    # Create DataFrame with proper MultiIndex columns
                    multi_df = pd.DataFrame(
                        values, 
                        index=df.index,
                        columns=multi_columns
                    )
                    
                    self.market_data[symbol] = multi_df
                    logger.debug(f"Initialized market data for {symbol} with {len(df)} rows")
                
                # Update the trading environment with new data if available
                if new_data_available and self.trading_env is not None:
                    try:
                        # Create a combined DataFrame with the latest data for all symbols
                        combined_data = pd.concat([self.market_data[symbol] for symbol in self.symbols 
                                                  if not self.market_data[symbol].empty], axis=1)
                        
                        if combined_data is not None and not combined_data.empty:
                            # CRITICAL FIX: Record current position within the DataFrame before updating
                            current_position = self.trading_env.current_step
                            rows_before = len(self.trading_env.df) if hasattr(self.trading_env, 'df') and self.trading_env.df is not None else 0
                            
                            # Set the DataFrame in the environment
                            self.trading_env.df = combined_data
                            logger.debug(f"Updated trading environment with new market data ({len(combined_data)} rows)")
                            
                            # CRITICAL FIX: Adjust current_step to point to the correct position in the new DataFrame
                            # If the DataFrame grew, adjust current_step accordingly to maintain the same position
                            rows_after = len(combined_data)
                            if rows_after > rows_before and rows_before > 0:
                                # Update current_step to maintain position (keep at the same percentage point)
                                # This ensures we don't go out of bounds after updating the DataFrame
                                self.trading_env.current_step = min(rows_after - 1, current_position + (rows_after - rows_before))
                                logger.debug(f"Adjusted current_step from {current_position} to {self.trading_env.current_step} due to DataFrame growth")
                            
                            # Always ensure current_step is valid
                            self.trading_env.current_step = min(len(combined_data) - 1, self.trading_env.current_step)
                            
                            # Initialize last_prices in the environment if not already set
                            if not hasattr(self.trading_env, 'last_prices') or self.trading_env.last_prices is None:
                                self.trading_env.last_prices = {}
                            
                            # Update market prices for each symbol
                            for symbol in self.symbols:
                                if not self.market_data[symbol].empty:
                                    latest_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
                                    
                                    # Update last_prices first
                                    self.trading_env.last_prices[symbol] = float(latest_price)
                                    
                                    # Make sure the symbol exists in positions
                                    if not hasattr(self.trading_env, 'positions'):
                                        self.trading_env.positions = {}
                                        
                                    if symbol not in self.trading_env.positions:
                                        self.trading_env.positions[symbol] = {
                                            'size': 0, 
                                            'entry_price': 0,
                                            'last_price': float(latest_price),
                                            'funding_accrued': 0,
                                            'leverage': 0.0,
                                            'direction': 0
                                        }
                                    
                                    # Update market price if method exists
                                    if hasattr(self.trading_env, 'update_market_price'):
                                        self.trading_env.update_market_price(symbol, latest_price)
                        
                        logger.debug(f"Updated trading environment with new data for all symbols")
                    except Exception as e:
                        logger.error(f"Error updating environment with new data: {str(e)}")
                        logger.error(traceback.format_exc())
            
            # If market conditions have changed, update the environment's market condition analysis
            if market_conditions_changed and self.trading_env is not None:
                try:
                    # Update market conditions in the environment
                    self.trading_env.update_market_conditions()
                    
                    # Get market condition information
                    market_info = self.trading_env.get_market_conditions()
                    if market_info:
                        for symbol, conditions in market_info.items():
                            # Log market regime 
                            if 'regime' in conditions:
                                logger.info(f"Market conditions for {symbol}: Regime={conditions['regime']}, "
                                           f"Volatility={conditions.get('volatility', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error updating market conditions: {str(e)}")
            
            return new_data_available
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _timeframe_to_seconds(self, timeframe):
        """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
        try:
            # Extract the number and unit
            number = int(''.join(filter(str.isdigit, timeframe)))
            unit = ''.join(filter(str.isalpha, timeframe))
            
            # Convert to seconds
            if unit.lower() == 'm':
                return number * 60
            elif unit.lower() == 'h':
                return number * 60 * 60
            elif unit.lower() == 'd':
                return number * 24 * 60 * 60
            else:
                logger.warning(f"Unknown timeframe unit: {unit}, defaulting to 300 seconds (5m)")
                return 300
                
        except Exception as e:
            logger.error(f"Error converting timeframe to seconds: {str(e)}")
            return 300  # Default to 5 minutes
    
    def _prepare_observation(self, symbol: str = None) -> np.ndarray:
        """
        Prepare observation for the RL model.
        
        If symbol is None, prepare a combined observation of all assets.
        Otherwise, prepare observation for a specific asset.
        
        The RL model expects a flattened vector of shape (78,).
        """
        try:
            with self.data_lock:
                # Check if market data is available for all symbols
                if any(self.market_data[s].empty for s in self.symbols):
                    missing_symbols = [s for s in self.symbols if self.market_data[s].empty]
                    logger.warning(f"No market data available for symbols: {missing_symbols}")
                    return None
                
                # Get the latest data for all symbols
                combined_features = []
                
                # For each symbol, extract the key features
                for sym in self.symbols:
                    df = self.market_data[sym]
                    
                    # Use a shorter lookback of 10 candles instead of 30
                    # This is because we have 3 assets x 26 features = 78 (as expected by the model)
                    window = df.tail(10).copy()
                    
                    # Key price features relative to last close
                    close_price = window[(sym, 'close')].iloc[-1]
                    
                    # Extract normalized price features (just the most recent values)
                    normalized_features = []
                    
                    # Price features (normalized by last close)
                    for col in ['open', 'high', 'low', 'close']:
                        if (sym, col) in window.columns:
                            # Only use the most recent value
                            normalized_features.append(window[(sym, col)].iloc[-1] / close_price)
                    
                    # Volume (normalized by mean)
                    vol_mean = window[(sym, 'volume')].mean()
                    if vol_mean > 0:
                        normalized_features.append(window[(sym, 'volume')].iloc[-1] / vol_mean)
                    else:
                        normalized_features.append(0.0)
                    
                    # Technical indicators
                    for indicator in ['sma_10', 'sma_20', 'ema_12', 'ema_26']:
                        if (sym, indicator) in window.columns:
                            normalized_features.append(window[(sym, indicator)].iloc[-1] / close_price)
                    
                    # MACD components
                    if (sym, 'macd') in window.columns and (sym, 'macd_signal') in window.columns:
                        # These are differences, so use absolute normalization
                        normalized_features.append(window[(sym, 'macd')].iloc[-1] / (0.01 * close_price))
                        normalized_features.append(window[(sym, 'macd_signal')].iloc[-1] / (0.01 * close_price))
                        if (sym, 'macd_hist') in window.columns:
                            normalized_features.append(window[(sym, 'macd_hist')].iloc[-1] / (0.01 * close_price))
                    
                    # RSI (already 0-100, so normalize to 0-1)
                    if (sym, 'rsi') in window.columns:
                        normalized_features.append(window[(sym, 'rsi')].iloc[-1] / 100.0)
                    
                    # Bollinger Bands
                    for band in ['bb_upper', 'bb_middle', 'bb_lower']:
                        if (sym, band) in window.columns:
                            normalized_features.append(window[(sym, band)].iloc[-1] / close_price)
                    
                    # ATR (normalize by close price)
                    if (sym, 'atr') in window.columns:
                        normalized_features.append(window[(sym, 'atr')].iloc[-1] / close_price)
                    
                    # Recent price changes (short-term momentum)
                    if len(window) >= 3:
                        # 1-candle change
                        price_change_1 = window[(sym, 'close')].iloc[-1] / window[(sym, 'close')].iloc[-2] - 1
                        normalized_features.append(price_change_1)
                        
                        # 3-candle change
                        price_change_3 = window[(sym, 'close')].iloc[-1] / window[(sym, 'close')].iloc[-min(len(window), 4)] - 1
                        normalized_features.append(price_change_3)
                    else:
                        normalized_features.extend([0.0, 0.0])
                    
                    # Current position state
                    position = self.positions[sym]
                    normalized_features.append(float(position['direction']))  # -1 (short), 0 (none), or 1 (long)
                    normalized_features.append(position['leverage'] / self.max_leverage)  # Normalized leverage
                    
                    # Convert any NaN values to 0
                    normalized_features = np.nan_to_num(normalized_features, nan=0.0)
                    
                    # Add to combined features
                    combined_features.extend(normalized_features)
                
                # Ensure we have exactly 78 features by padding or truncating
                if len(combined_features) > 78:
                    logger.warning(f"Truncating observation from {len(combined_features)} to 78 features")
                    combined_features = combined_features[:78]
                elif len(combined_features) < 78:
                    logger.warning(f"Padding observation from {len(combined_features)} to 78 features")
                    combined_features.extend([0.0] * (78 - len(combined_features)))
                
                # Convert to numpy array and ensure correct shape
                observation = np.array(combined_features, dtype=np.float32)
                
                logger.debug(f"Prepared observation with shape {observation.shape}")
                return observation
            
        except Exception as e:
            logger.error(f"Error preparing observation: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def generate_signals(self):
        """Generate trading signals using the RL model."""
        try:
            # Prepare a combined observation for all assets
            observation = self._prepare_observation()
            
            if observation is None:
                logger.warning("Skipping signal generation: No valid observation")
                return {}
            
            # Check observation shape
            if observation.shape != (78,):
                logger.warning(f"Unexpected observation shape: {observation.shape}, expected (78,)")
                return {}
                
            # Generate action using RL model
            actions, _ = self.rl_model.predict(observation, deterministic=False)
            
            # In the case of multi-asset models, the actions might come as a vector
            # Let's assume actions for BTC, ETH, SOL in that order
            signals = {}
            
            # If actions is a single value, it's likely a model trained on one asset at a time
            if isinstance(actions, (int, float)) or (isinstance(actions, np.ndarray) and actions.size == 1):
                # Use the single action for all assets (this is a fallback)
                action_value = float(actions)
                logger.warning(f"Single action value {action_value} for all assets - check model training")
                
                for i, symbol in enumerate(self.symbols):
                    direction = np.sign(action_value)
                    strength = abs(action_value)
                    
                    signals[symbol] = self._create_signal_dict(symbol, direction, strength)
            
            # If actions is a vector with multiple values (one per asset)
            elif isinstance(actions, np.ndarray) and actions.size == len(self.symbols):
                for i, symbol in enumerate(self.symbols):
                    action_value = float(actions[i])
                    direction = np.sign(action_value)
                    strength = abs(action_value)
                    
                    signals[symbol] = self._create_signal_dict(symbol, direction, strength)
            
            # If actions is a vector but doesn't match our symbol count
            elif isinstance(actions, np.ndarray):
                logger.warning(f"Action shape {actions.shape} doesn't match symbol count {len(self.symbols)}")
                
                # Try to use the first few actions
                for i, symbol in enumerate(self.symbols):
                    if i < len(actions):
                        action_value = float(actions[i])
                        direction = np.sign(action_value)
                        strength = abs(action_value)
                        
                        signals[symbol] = self._create_signal_dict(symbol, direction, strength)
            
            # Log the signals
            for symbol, signal in signals.items():
                logger.info(f"Generated signal for {symbol}: {signal['action']}, action: {signal['raw_action']:.4f}, leverage: {signal['leverage']:.2f}x")
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _create_signal_dict(self, symbol, direction, strength):
        """Create a standardized signal dictionary."""
        # Get current price
        current_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
        
        # Calculate raw action
        raw_action = direction * strength
        
        # Calculate leverage using the same function as during training
        if hasattr(self, 'trading_env') and self.trading_env is not None:
            try:
                # Use the exact same leverage calculation as in training
                target_leverage = self.trading_env._get_target_leverage(raw_action)
                logger.debug(f"Used environment leverage calculation: {raw_action} -> {target_leverage}")
            except Exception as e:
                logger.error(f"Error using environment leverage calculation: {str(e)}")
                # Fallback to simplified calculation
                target_leverage = self.max_leverage * np.tanh(abs(raw_action)) * direction
                logger.debug(f"Used fallback leverage calculation: {raw_action} -> {target_leverage}")
        else:
            # Fallback to simplified calculation if environment not available
            target_leverage = self.max_leverage * np.tanh(abs(raw_action)) * direction
        
        # Determine decision based on direction and strength
        if strength < self.signal_threshold:
            decision = "HOLD"
            target_leverage = 0.0
        elif direction > 0:
            if strength > 0.6:
                decision = "BUY (Strong)"
            elif strength > 0.3:
                decision = "BUY (Moderate)"
            else:
                decision = "BUY (Light)"
        else:
            if strength > 0.6:
                decision = "SELL (Strong)"
            elif strength > 0.3:
                decision = "SELL (Moderate)"
            else:
                decision = "SELL (Light)"
        
        # Create and return the signal dictionary
        return {
            'timestamp': datetime.now().isoformat(),
            'action': decision,
            'raw_action': raw_action,
            'direction': int(direction),
            'strength': float(strength),
            'leverage': float(target_leverage),
            'confidence': float(strength),
            'price': float(current_price)
        }
    
    async def execute_trades(self, signals: Dict):
        """Execute trades based on generated signals using the trading environment."""
        try:
            trades_executed = []
            current_time = datetime.now()
            
            # If we have the trading environment, use it for sophisticated position management
            if self.trading_env is not None:
                logger.info("Using institutional environment for trade execution and risk management")
                
                # Get current environment state before actions
                pre_state = self.trading_env.get_state()
                
                # Convert signals to actions for each symbol
                actions = {}
                for symbol, signal in signals.items():
                    # Get the raw action (direction * strength)
                    raw_action = signal['raw_action']
                    
                    # Log the action being sent to the environment
                    logger.info(f"Sending action {raw_action:.4f} to environment for {symbol}")
                    
                    actions[symbol] = raw_action
                
                # Execute the actions in the environment
                # This is the main call that applies the action to the environment
                obs, reward, done, info = self.trading_env.step(actions)
                
                # Get the new environment state after actions
                post_state = self.trading_env.get_state()
                
                # Log results from the step
                if 'rewards' in info:
                    for symbol, r in info['rewards'].items():
                        logger.info(f"Reward for {symbol}: {r:.4f}")
                
                # Record trades for each symbol where position changed
                for symbol in self.symbols:
                    pre_position = pre_state['positions'].get(symbol, {'size': 0, 'direction': 0})
                    post_position = post_state['positions'].get(symbol, {'size': 0, 'direction': 0})
                    
                    # Check if the position changed
                    if (pre_position.get('size', 0) != post_position.get('size', 0) or 
                        pre_position.get('direction', 0) != post_position.get('direction', 0)):
                        
                        # Get current price from our market data
                        current_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
                        
                        # Determine action type
                        if pre_position.get('size', 0) == 0 and post_position.get('size', 0) != 0:
                            # New position
                            action_type = 'BUY' if post_position.get('direction', 0) > 0 else 'SELL'
                        elif pre_position.get('size', 0) != 0 and post_position.get('size', 0) == 0:
                            # Closed position
                            action_type = 'CLOSE'
                        elif (pre_position.get('direction', 0) * post_position.get('direction', 0) < 0 and 
                              pre_position.get('size', 0) != 0 and post_position.get('size', 0) != 0):
                            # Direction changed
                            action_type = 'REVERSE to ' + ('BUY' if post_position.get('direction', 0) > 0 else 'SELL')
                        elif pre_position.get('size', 0) < post_position.get('size', 0):
                            # Increased position size
                            action_type = 'INCREASE ' + ('LONG' if post_position.get('direction', 0) > 0 else 'SHORT')
                        elif pre_position.get('size', 0) > post_position.get('size', 0):
                            # Decreased position size
                            action_type = 'DECREASE ' + ('LONG' if post_position.get('direction', 0) > 0 else 'SHORT')
                        else:
                            # Some other change
                            action_type = 'ADJUST'
                        
                        # Record the trade
                        trade = {
                            'symbol': symbol,
                            'timestamp': current_time.isoformat(),
                            'action': action_type,
                            'price': current_price,
                            'pre_size': pre_position.get('size', 0),
                            'post_size': post_position.get('size', 0),
                            'size_delta': post_position.get('size', 0) - pre_position.get('size', 0),
                            'pre_value': pre_position.get('value', 0),
                            'post_value': post_position.get('value', 0),
                            'pre_leverage': pre_position.get('leverage', 0),
                            'post_leverage': post_position.get('leverage', 0),
                            'direction': post_position.get('direction', 0),
                            'cost': info.get('costs', {}).get(symbol, 0),
                            'raw_signal': signals[symbol]['raw_action'] if symbol in signals else 0
                        }
                        
                        self.trades.append(trade)
                        trades_executed.append(trade)
                        
                        logger.info(f"Executed {action_type} for {symbol} at {current_price}, "
                                  f"size {post_position.get('size', 0):.6f}, "
                                  f"value ${post_position.get('value', 0):.2f}, "
                                  f"leverage {post_position.get('leverage', 0):.2f}x")
                
                # Update our internal positions to match the environment
                self.positions = post_state['positions']
                self.balance = post_state['balance']
                self.total_value = post_state['total_value']
                
                # Log any risk management actions that occurred
                if 'risk_actions' in info and info['risk_actions']:
                    for action in info['risk_actions']:
                        logger.warning(f"Risk management action: {action}")
                
                # Check for liquidations
                if 'liquidations' in info and info['liquidations']:
                    for symbol, liquidation in info['liquidations'].items():
                        logger.critical(f"LIQUIDATION on {symbol}: {liquidation}")
                
                # Log portfolio state after trades
                logger.info(f"Portfolio after trades - Balance: ${self.balance:.2f}, "
                          f"Total value: ${self.total_value:.2f}")
                for symbol, position in self.positions.items():
                    if position['size'] != 0:
                        logger.info(f"Position {symbol}: Size={position['size']:.6f}, "
                                  f"Direction={position['direction']}, Leverage={position['leverage']:.2f}x")
            
            # Fallback to our own simplified position management if environment not available
            else:
                logger.warning("Trading environment not available, using simplified position management")
                for symbol, signal in signals.items():
                    # Get signal data
                    decision = signal['action']
                    trade_direction = signal['direction']
                    target_leverage = signal['leverage']
                    current_price = signal['price']
                    
                    # Skip if HOLD
                    if decision == "HOLD" or abs(trade_direction) < 0.1:
                        continue
                    
                    # Get current position
                    position = self.positions[symbol]
                    
                    # Determine if we need to execute a trade
                    execute_trade = False
                    
                    # If no position, execute if we have a signal
                    if position['size'] == 0 and abs(trade_direction) > 0:
                        execute_trade = True
                    
                    # If position exists, execute if direction changes or significant leverage change
                    elif position['size'] != 0:
                        # Different direction than current position
                        if np.sign(position['direction']) != np.sign(trade_direction) and abs(trade_direction) > 0:
                            execute_trade = True
                        # Same direction but significant leverage change
                        elif (np.sign(position['direction']) == np.sign(trade_direction) and
                              abs(position['leverage'] - target_leverage) > 0.5 * self.max_leverage):
                            execute_trade = True
                    
                    # Execute the trade if needed
                    if execute_trade:
                        # Close existing position if direction changes
                        if position['size'] != 0 and np.sign(position['direction']) != np.sign(trade_direction):
                            await self._close_position(symbol)
                        
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
                            'value': position_value
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
                            'signal': signal['raw_action']
                        }
                        
                        self.trades.append(trade)
                        trades_executed.append(trade)
                        
                        # Update balance for transaction costs
                        self.balance -= transaction_cost
                        
                        logger.info(f"Executed {trade['action']} trade for {symbol} at price {current_price}, "
                                  f"size {position_size:.6f}, value ${position_value:.2f}, leverage {target_leverage:.2f}x")
            
            # Update PnL after trades
            await self._update_pnl()
            
            # Log trades if any were executed
            if trades_executed:
                self._save_trade_data()
            
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
        
        # FIXED: Using correct MultiIndex access pattern
        current_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
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
    
    async def _update_pnl(self):
        """Update unrealized PnL for all positions."""
        try:
            # If we have the trading environment, use it for accurate PnL calculation
            if self.trading_env is not None:
                # Update the environment with latest market data
                for symbol in self.symbols:
                    if not self.market_data[symbol].empty:
                        # FIXED: Using correct MultiIndex access pattern
                        latest_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
                        
                        # Update market price in the environment
                        self.trading_env.update_market_price(symbol, latest_price)
                
                # Recalculate PnL
                self.trading_env.calculate_pnl()
                
                # Get updated state
                state = self.trading_env.get_state()
                
                # Update our internal state to match
                self.positions = state['positions']
                self.balance = state['balance']
                self.total_value = state['total_value']
                
                # Also check funding rates if applicable (for perpetual futures)
                if hasattr(self.trading_env, 'apply_funding_rates'):
                    try:
                        funding_impacts = self.trading_env.apply_funding_rates()
                        if funding_impacts:
                            for symbol, impact in funding_impacts.items():
                                if abs(impact) > 0.01:  # Only log non-trivial impacts
                                    logger.info(f"Funding rate impact for {symbol}: ${impact:.2f}")
                    except Exception as e:
                        logger.error(f"Error applying funding rates: {str(e)}")
                
                # Perform risk checks
                risk_info = self.trading_env.check_risk_limits()
                
                # If any risk violations occurred, log them
                if risk_info['violations']:
                    for violation in risk_info['violations']:
                        logger.warning(f"Risk violation: {violation}")
                
                # Check for liquidations
                liquidations = self.trading_env.check_liquidations()
                if liquidations:
                    for symbol, liquidation in liquidations.items():
                        logger.critical(f"LIQUIDATION on {symbol}: {liquidation}")
                        
                        # If position was liquidated, ensure our internal state reflects this
                        if symbol in self.positions:
                            self.positions[symbol]['size'] = 0
                            self.positions[symbol]['direction'] = 0
                            self.positions[symbol]['leverage'] = 0
                            self.positions[symbol]['value'] = 0
                            
                            # Record the liquidation as a trade
                            trade = {
                                'symbol': symbol,
                                'timestamp': datetime.now().isoformat(),
                                'action': 'LIQUIDATION',
                                'price': self.market_data[symbol][(symbol, 'close')].iloc[-1],
                                'size': 0,
                                'value': 0,
                                'leverage': 0,
                                'pnl': liquidation.get('loss', 0)
                            }
                            self.trades.append(trade)
                            self._save_trade_data()
                
                # Also check for auto-close conditions (underwater positions)
                if hasattr(self.trading_env, 'check_underwater_positions'):
                    try:
                        underwater = self.trading_env.check_underwater_positions()
                        if underwater:
                            for symbol, position_info in underwater.items():
                                logger.warning(f"Position underwater for {symbol}: {position_info}")
                    except Exception as e:
                        logger.error(f"Error checking underwater positions: {str(e)}")
                
                logger.debug(f"Updated PnL via environment - Balance: ${self.balance:.2f}, "
                           f"Total Value: ${self.total_value:.2f}")
                
                return
            
            # Fallback to manual PnL calculation if environment not available
            logger.debug("Using fallback PnL calculation")
            
            total_unrealized_pnl = 0.0
            
            for symbol, position in self.positions.items():
                if position['size'] == 0:
                    continue
                
                # FIXED: Using correct MultiIndex access pattern
                current_price = self.market_data[symbol][(symbol, 'close')].iloc[-1]
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
            
            logger.debug(f"Updated PnL - Balance: ${self.balance:.2f}, "
                       f"Unrealized PnL: ${total_unrealized_pnl:.2f}, "
                       f"Total Value: ${self.total_value:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating PnL: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _generate_market_commentary(self):
        """Generate market commentary for all assets."""
        if self.commentary_generator is None:
            logger.warning("Commentary generator not available, skipping commentary")
            return {}
        
        try:
            commentary = {}
            
            # Generate individual commentary for each asset
            for symbol in self.symbols:
                if self.market_data[symbol].empty:
                    logger.warning(f"No market data available for {symbol}, skipping commentary")
                    continue
                
                asset_commentary = self.commentary_generator.generate_daily_commentary(
                    symbol=symbol,
                    ohlcv_data=self.market_data[symbol],
                    lookback_days=min(30, len(self.market_data[symbol])),
                    max_tokens=500,
                    temperature=0.7
                )
                
                commentary[symbol] = asset_commentary
                logger.info(f"Generated commentary for {symbol}: {len(asset_commentary)} chars")
            
            # Save commentary to file
            self._save_commentary(commentary)
            
            return commentary
        
        except Exception as e:
            logger.error(f"Error generating market commentary: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _update_chatbot(self):
        """Update the chatbot with latest market data and signals."""
        if self.chatbot is None:
            return
        
        try:
            # Update market data for chatbot
            for symbol in self.symbols:
                if not self.market_data[symbol].empty:
                    self.chatbot.update_market_data(self.market_data[symbol])
            
            # Update trading signals
            signals = {}
            for symbol, position in self.positions.items():
                if position['size'] != 0:
                    signals[symbol] = {
                        'action': 'BUY' if position['direction'] > 0 else 'SELL',
                        'confidence': abs(position['direction']),
                        'size': position['size'],
                        'entry_price': position['entry_price'],
                        'current_price': position['current_price'],
                        'pnl': position['unrealized_pnl']
                    }
            
            self.chatbot.update_trading_signals(signals)
            
            # Update portfolio performance
            performance = {
                'balance': self.balance,
                'total_value': self.total_value,
                'unrealized_pnl': self.total_value - self.balance,
                'positions': len([p for p in self.positions.values() if p['size'] != 0])
            }
            
            self.chatbot.update_portfolio_performance(performance)
            
        except Exception as e:
            logger.error(f"Error updating chatbot: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_trade_data(self):
        """Save trade data to log file."""
        try:
            # Check if the file exists
            existing_trades = []
            if os.path.exists(self.trade_log_file):
                try:
                    with open(self.trade_log_file, 'r') as f:
                        existing_data = json.load(f)
                        if 'trades' in existing_data:
                            existing_trades = existing_data['trades']
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Could not read existing trades from {self.trade_log_file}, starting fresh")
            
            # Get only new trades to append (assuming trades list is chronological)
            if existing_trades:
                # Find trades not already in the file based on timestamp
                last_timestamp = existing_trades[-1].get('timestamp', '')
                new_trades = [trade for trade in self.trades if trade.get('timestamp', '') > last_timestamp]
                
                # Append new trades to existing ones
                combined_trades = existing_trades + new_trades
            else:
                combined_trades = self.trades
            
            # Write back the combined list
            with open(self.trade_log_file, 'w') as f:
                json.dump({'trades': combined_trades}, f, indent=2)
            
            logger.debug(f"Saved {len(self.trades)} trades to {self.trade_log_file}")
        except Exception as e:
            logger.error(f"Error saving trade data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_position_data(self):
        """Save current positions to log file."""
        try:
            # Convert positions to JSON-serializable format
            serializable_positions = {}
            for symbol, position in self.positions.items():
                # Create a copy of the position dictionary
                pos_copy = position.copy()
                
                # Convert datetime to string if present
                if isinstance(pos_copy.get('timestamp'), datetime):
                    pos_copy['timestamp'] = pos_copy['timestamp'].isoformat()
                
                serializable_positions[symbol] = pos_copy
            
            position_data = {
                'timestamp': datetime.now().isoformat(),
                'balance': float(self.balance),
                'total_value': float(self.total_value),
                'positions': serializable_positions
            }
            
            with open(self.position_log_file, 'w') as f:
                json.dump(position_data, f, indent=2)
            
            logger.debug(f"Saved position data to {self.position_log_file}")
            
            # Also append to historical file
            historical_file = f"{self.log_dir}/positions/positions_{datetime.now().strftime('%Y-%m-%d')}.json"
            
            # Check if file exists, if not create it with empty list
            if not os.path.exists(historical_file):
                with open(historical_file, 'w') as f:
                    json.dump([], f)
            
            # Read existing data
            with open(historical_file, 'r') as f:
                try:
                    historical_data = json.load(f)
                except json.JSONDecodeError:
                    historical_data = []
            
            # Append new data
            historical_data.append(position_data)
            
            # Write updated data
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving position data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_commentary(self, commentary):
        """Save market commentary to log file."""
        try:
            current_time = datetime.now()
            commentary_data = {
                'timestamp': current_time.isoformat(),
                'commentary': commentary
            }
            
            # Check if file exists and read existing data
            existing_commentaries = []
            if os.path.exists(self.commentary_log_file):
                try:
                    with open(self.commentary_log_file, 'r') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            existing_commentaries = existing_data
                        elif isinstance(existing_data, dict) and 'commentaries' in existing_data:
                            existing_commentaries = existing_data['commentaries']
                        else:
                            # If it's a single commentary, convert to list
                            existing_commentaries = [existing_data]
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Could not read existing commentaries from {self.commentary_log_file}, starting fresh")
            
            # Append new commentary
            existing_commentaries.append(commentary_data)
            
            # Keep only the most recent 24 hours of commentaries
            cutoff_time = (current_time - timedelta(hours=24)).isoformat()
            recent_commentaries = [
                c for c in existing_commentaries 
                if isinstance(c, dict) and c.get('timestamp', '') >= cutoff_time
            ]
            
            # Write back combined commentaries
            with open(self.commentary_log_file, 'w') as f:
                json.dump({'commentaries': recent_commentaries}, f, indent=2)
            
            logger.debug(f"Saved commentary to {self.commentary_log_file}")
            
            # Also append to historical file
            historical_file = f"{self.log_dir}/commentary/commentary_{current_time.strftime('%Y-%m-%d')}.json"
            
            # Check if file exists, if not create it with empty list
            if not os.path.exists(historical_file):
                os.makedirs(os.path.dirname(historical_file), exist_ok=True)
                with open(historical_file, 'w') as f:
                    json.dump([], f)
            
            # Read existing data
            with open(historical_file, 'r') as f:
                try:
                    historical_data = json.load(f)
                except json.JSONDecodeError:
                    historical_data = []
            
            # Append new data
            historical_data.append(commentary_data)
            
            # Write updated data
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving commentary: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def run_trading_loop(self):
        """Run the main trading loop."""
        logger.info("Starting trading loop")
        
        self.is_running = True
        
        try:
            while self.is_running and not self.shutting_down:
                # Fetch latest data
                new_data_available = await self._fetch_latest_data()
                
                # Generate signals only when new candle data is available
                if new_data_available:
                    # Update chatbot with latest data
                    self._update_chatbot()
                    
                    # Check if enough time has passed since last signal generation
                    now = datetime.now()
                    min_signal_interval = timedelta(seconds=self._timeframe_to_seconds(self.timeframe) * 0.9)
                    
                    should_generate_signals = all(
                        (now - self.last_signal_time[symbol]) > min_signal_interval
                        for symbol in self.symbols
                    )
                    
                    if should_generate_signals:
                        logger.info("Generating new trading signals based on latest data")
                        signals = await self.generate_signals()
                        
                        # Update signal timestamps
                        for symbol in self.symbols:
                            self.last_signal_time[symbol] = now
                        
                        # Execute trades based on signals
                        if signals:
                            await self.execute_trades(signals)
                        else:
                            logger.warning("Received empty or invalid trading signals")
                    else:
                        logger.debug("Skipping signal generation - not enough time elapsed since last signals")
                else:
                    logger.debug("No new candle data available, skipping signal generation")
                
                # IMPORTANT: Always update PnL even if no new data, to catch price movements and liquidations
                await self._update_pnl()
                
                # Check margin requirements and liquidation risk even when not trading
                if self.trading_env is not None:
                    # Perform periodic risk assessment
                    risk_info = self.trading_env.check_risk_limits()
                    margin_info = self.trading_env.get_margin_info() if hasattr(self.trading_env, 'get_margin_info') else None
                    
                    # Log margin info for active positions
                    if margin_info:
                        for symbol, info in margin_info.items():
                            if self.positions.get(symbol, {}).get('size', 0) != 0:
                                margin_ratio = info.get('margin_ratio', 0)
                                if margin_ratio > 0.8:  # High margin utilization
                                    logger.warning(f"HIGH MARGIN UTILIZATION for {symbol}: {margin_ratio:.2%}")
                                elif margin_ratio > 0.5:  # Moderate margin utilization
                                    logger.info(f"Margin utilization for {symbol}: {margin_ratio:.2%}")
                
                # Generate market commentary periodically
                now = datetime.now()
                if (now - self.last_commentary_time).total_seconds() > 5 * 60:  # Every 5 minutes
                    await self._generate_market_commentary()
                    self.last_commentary_time = now
                
                # Log positions periodically
                if (now - self.last_position_log_time).total_seconds() > 5 * 60:  # Every 5 minutes
                    self._save_position_data()
                    self.last_position_log_time = now
                
                # Sleep for a bit before next update
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.is_running = False
            logger.info("Trading loop stopped")
    
    async def shutdown(self):
        """Shut down the trading monitor."""
        logger.info("Shutting down trading monitor")
        self.shutting_down = True
        self.is_running = False
        
        try:
            # Close all positions - use trading environment if available
            if self.trading_env is not None:
                # Create a dictionary of "close" actions for all symbols
                close_actions = {symbol: 0 for symbol in self.symbols}
                
                # Execute the close actions
                logger.info("Closing all positions via trading environment")
                obs, reward, done, info = self.trading_env.step(close_actions)
                
                # Get the final state
                final_state = self.trading_env.get_state()
                self.positions = final_state['positions']
                self.balance = final_state['balance']
                self.total_value = final_state['total_value']
            else:
                # Use our own close position function
                for symbol in self.symbols:
                    if self.positions[symbol]['size'] != 0:
                        await self._close_position(symbol)
            
            # Save final data
            self._save_trade_data()
            self._save_position_data()
            
            # Close exchange client
            if self.exchange:
                await self.exchange.close()
            
            logger.info("Trading monitor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            logger.error(traceback.format_exc())


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Real-time Trading Monitor')
    
    parser.add_argument('--rl-model', type=str, required=True,
                      help='Path to the trained RL model')
    
    parser.add_argument('--llm-model', type=str, required=True,
                      help='Path to the LLM model for commentary')
    
    parser.add_argument('--llm-base-model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                      help='Base model for LLM if using LoRA')
    
    parser.add_argument('--config', type=str, default='config/realtime_config.yaml',
                      help='Path to configuration file')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory for logs')
    
    parser.add_argument('--timeframe', type=str, default='5m',
                      help='Timeframe for price data')
    
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT',
                      help='Comma-separated list of symbols to trade')
    
    parser.add_argument('--balance', type=float, default=10000.0,
                      help='Initial account balance')
    
    parser.add_argument('--max-leverage', type=float, default=20.0,
                      help='Maximum allowed leverage')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Create logs directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger.info(f"Initializing real-time trading monitor for {symbols}")
    logger.info(f"Using RL model: {args.rl_model}")
    logger.info(f"Using LLM model: {args.llm_model}")
    logger.info(f"Initial balance: ${args.balance}")
    
    # Initialize trading monitor
    monitor = RealTimeTradeMonitor(
        rl_model_path=args.rl_model,
        llm_model_path=args.llm_model,
        llm_base_model=args.llm_base_model,
        config_path=args.config,
        log_dir=args.log_dir,
        timeframe=args.timeframe,
        symbols=symbols,
        initial_balance=args.balance,
        max_leverage=args.max_leverage
    )
    
    # Handle shutdown signals
    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown")
        monitor.shutting_down = True
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        # Setup monitor
        await monitor.setup()
        
        # Run trading loop
        await monitor.run_trading_loop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Ensure proper shutdown
        await monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 