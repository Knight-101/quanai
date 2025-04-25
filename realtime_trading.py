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
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import threading
import signal
import websockets
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

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
        backfill_days: int = 5
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
        
        # Create necessary directories
        os.makedirs(self.save_trades_path, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        self.symbols = self.config['trading']['symbols']
        self.timeframe = '1m'  # Using 1-minute data for real-time trading
        
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
        
        # Initialize trading environment with risk engine
        risk_limits = RiskLimits(
            account_max_leverage=self.max_leverage * 0.8,
            position_max_leverage=self.max_leverage,
            max_drawdown_pct=self.config['risk_management']['limits']['max_drawdown'],
            position_concentration=self.config['risk_management']['limits']['position_concentration'],
            daily_loss_limit_pct=0.15
        )
        
        self.risk_engine = InstitutionalRiskEngine(
            initial_balance=self.initial_balance,
            risk_limits=risk_limits,
            use_dynamic_limits=True,
            use_vol_scaling=True
        )
        
        # Load historical data or start with empty data
        if self.historical_data_path and os.path.exists(self.historical_data_path):
            await self._load_historical_data()
        else:
            # Backfill some historical data
            await self._backfill_data()
        
        # Create environment for inference
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
        """Create trading environment for inference."""
        logger.info("Creating trading environment for inference...")
        
        try:
            # Load environment with normalization stats if provided
            if self.env_path and os.path.exists(self.env_path):
                logger.info(f"Loading environment from: {self.env_path}")
                
                # First create base environment
                base_env = InstitutionalPerpetualEnv(
                    df=self.historical_df,
                    assets=self.symbols,
                    window_size=100,
                    max_leverage=self.max_leverage,
                    commission=self.commission,
                    risk_engine=self.risk_engine,
                    initial_balance=self.initial_balance,
                    verbose=False
                )
                
                # Wrap it in a DummyVecEnv
                vec_env = DummyVecEnv([lambda: base_env])
                
                # Load environment with normalization
                self.env = VecNormalize.load(self.env_path, vec_env)
                
                # Reset training mode and reward normalization for inference
                self.env.training = False
                self.env.norm_reward = False
                
                logger.info("Environment loaded successfully with normalization stats")
            else:
                logger.info("No environment file provided, creating new environment")
                
                # Create a fresh environment
                base_env = InstitutionalPerpetualEnv(
                    df=self.historical_df,
                    assets=self.symbols,
                    window_size=100,
                    max_leverage=self.max_leverage,
                    commission=self.commission,
                    risk_engine=self.risk_engine,
                    initial_balance=self.initial_balance,
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
                
                # Fetch latest OHLCV candle
                ohlcv = await self.exchange.fetch_ohlcv(exchange_symbol, self.timeframe, limit=2)
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
    
    async def generate_signals(self, new_data: pd.DataFrame):
        """Generate trading signals using the RL model."""
        logger.info("Generating trading signals...")
        
        try:
            # Update environment with new data
            await self._update_environment_data(new_data)
            
            # Reset the environment to incorporate the new data
            obs = self.env.reset()
            
            # Get model prediction (action)
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
        
        # Close exchange connection
        await self.exchange.close()
        
        # Close websocket server if active
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("Real-time trading system shutdown complete")

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
        backfill_days=args.backfill_days
    )
    
    # Set up trader
    await trader.setup()
    
    # Run trading loop
    await trader.run_trading_loop()
    
    # Ensure proper shutdown
    await trader.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 