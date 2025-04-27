#!/usr/bin/env python3
"""
Real-time Trading Monitor

This script implements a real-time trading monitor that:
1. Fetches constant price data for BTC, ETH, and SOL on 5m timeframes
2. Executes trades via the RL model similar to training/backtesting
3. Logs all trades to a file
4. Logs current positions every 15 minutes
5. Logs market commentary every 15 minutes
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
        max_leverage: float = 10.0
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
                    'trade_cooldown_minutes': 15
                },
                'data': {
                    'backfill_periods': 1000
                }
            }
    
    async def setup(self):
        """Set up the real-time trading monitor."""
        logger.info("Setting up real-time trading monitor...")
        
        # Initialize exchange
        self.exchange = ccxt_async.binance({
            'options': {
                'defaultType': 'future',
                'defaultMarket': 'linear',
                'defaultMarginMode': 'cross'
            },
            'enableRateLimit': True
        })
        
        # Load the RL model
        try:
            logger.info(f"Loading RL model from {self.rl_model_path}")
            self.rl_model = PPO.load(self.rl_model_path)
            logger.info("RL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading RL model: {str(e)}")
            raise
        
        # Load the LLM model
        try:
            logger.info(f"Loading LLM model from {self.llm_model_path}")
            self.llm_model = TradingLLM.load_from_pretrained(
                model_dir=self.llm_model_path,
                base_model=self.llm_base_model
            )
            logger.info("LLM model loaded successfully")
            
            # Initialize chatbot
            self.chatbot = MarketChatbot(
                llm_model=self.llm_model,
                asset_names=self.symbols
            )
            
            # Initialize commentary generator
            self.commentary_generator = MarketCommentaryGenerator(
                llm_model_path=self.llm_model_path,
                llm_base_model=self.llm_base_model
            )
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            logger.warning("Will continue without LLM capabilities")
        
        # Backfill initial data
        await self._backfill_initial_data()
        
        logger.info("Real-time trading monitor setup complete")
    
    async def _backfill_initial_data(self):
        """Backfill initial historical data."""
        logger.info("Backfilling initial historical data...")
        
        try:
            # Load markets
            await self.exchange.load_markets()
            
            # Get backfill periods from config
            backfill_periods = self.config.get('data', {}).get('backfill_periods', 1000)
            
            # Fetch data for each symbol
            for symbol in self.symbols:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, 
                    self.timeframe, 
                    limit=backfill_periods
                )
                
                if not ohlcv:
                    logger.warning(f"No historical data found for {symbol}")
                    continue
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate basic technical indicators
                df = self._calculate_indicators(df)
                
                self.market_data[symbol] = df
                logger.info(f"Backfilled {len(df)} periods for {symbol}")
            
            logger.info("Initial data backfill complete")
        except Exception as e:
            logger.error(f"Error backfilling initial data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a DataFrame."""
        try:
            # Copy to avoid modifying the original
            df = df.copy()
            
            # Simple Moving Averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            std_dev = df['close'].rolling(window=20).std()
            df['bb_middle'] = df['sma_20']
            df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
            df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Fill NaN values
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return df
    
    async def _fetch_latest_data(self):
        """Fetch the latest data for all symbols."""
        try:
            for symbol in self.symbols:
                # Fetch recent OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, 
                    self.timeframe, 
                    limit=2  # Get current and previous candle
                )
                
                if not ohlcv or len(ohlcv) < 2:
                    logger.warning(f"Insufficient data received for {symbol}")
                    continue
                
                # Convert to DataFrame
                new_data = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
                new_data.set_index('timestamp', inplace=True)
                
                with self.data_lock:
                    # Check if we already have this candle
                    if (self.market_data[symbol].empty or 
                        new_data.index[-1] > self.market_data[symbol].index[-1]):
                        
                        # Append new data to existing data
                        combined_data = pd.concat([self.market_data[symbol], new_data])
                        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                        combined_data = combined_data.sort_index()
                        
                        # Recalculate indicators
                        combined_data = self._calculate_indicators(combined_data)
                        
                        # Update market data
                        self.market_data[symbol] = combined_data
                        
                        logger.debug(f"Updated market data for {symbol}, new timestamp: {new_data.index[-1]}")
                    else:
                        logger.debug(f"No new data for {symbol}, current latest: {self.market_data[symbol].index[-1]}")
            
            return True
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _prepare_observation(self, symbol: str) -> np.ndarray:
        """Prepare observation for the RL model."""
        try:
            with self.data_lock:
                if self.market_data[symbol].empty:
                    logger.warning(f"No market data available for {symbol}")
                    return None
                
                # Get the latest data
                df = self.market_data[symbol]
                
                # Use the last 30 candles for the observation
                window = df.tail(30).copy()
                
                # Extract features used by the RL model
                features = []
                
                # Price and volume features
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in window.columns:
                        # Normalize by the last close price
                        if col != 'volume':
                            norm_factor = window['close'].iloc[-1]
                            features.append(window[col].values / norm_factor)
                        else:
                            # Normalize volume by its own mean to handle different scales
                            norm_factor = window['volume'].mean()
                            if norm_factor > 0:
                                features.append(window['volume'].values / norm_factor)
                            else:
                                features.append(window['volume'].values)
                
                # Technical indicators
                for indicator in ['sma_10', 'sma_20', 'ema_12', 'ema_26', 
                                 'macd', 'macd_signal', 'rsi', 
                                 'bb_upper', 'bb_middle', 'bb_lower', 'atr']:
                    if indicator in window.columns:
                        if indicator in ['sma_10', 'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_middle', 'bb_lower']:
                            # Normalize price-based indicators
                            norm_factor = window['close'].iloc[-1]
                            features.append(window[indicator].values / norm_factor)
                        elif indicator == 'rsi':
                            # RSI is already normalized
                            features.append(window[indicator].values / 100)
                        else:
                            # Other indicators
                            features.append(window[indicator].values)
                
                # Stack features
                observation = np.column_stack(features)
                
                return observation
            
        except Exception as e:
            logger.error(f"Error preparing observation for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def generate_signals(self):
        """Generate trading signals using the RL model."""
        try:
            signals = {}
            
            for symbol in self.symbols:
                # Prepare observation
                observation = self._prepare_observation(symbol)
                
                if observation is None:
                    logger.warning(f"Skipping signal generation for {symbol}: No valid observation")
                    continue
                
                # Generate action using RL model
                action, _ = self.rl_model.predict(observation, deterministic=True)
                
                # Extract the raw action value
                action_value = float(action[0])
                
                # Determine direction and strength
                direction = np.sign(action_value)
                strength = abs(action_value)
                
                # Convert to trading signal
                if strength < self.signal_threshold:
                    decision = "HOLD"
                    leverage = 0
                elif direction > 0:
                    if strength > 0.6:
                        decision = "BUY (Strong)"
                        leverage = self.max_leverage * 0.8
                    elif strength > 0.3:
                        decision = "BUY (Moderate)"
                        leverage = self.max_leverage * 0.5
                    else:
                        decision = "BUY (Light)"
                        leverage = self.max_leverage * 0.3
                else:
                    if strength > 0.6:
                        decision = "SELL (Strong)"
                        leverage = self.max_leverage * 0.8
                    elif strength > 0.3:
                        decision = "SELL (Moderate)"
                        leverage = self.max_leverage * 0.5
                    else:
                        decision = "SELL (Light)"
                        leverage = self.max_leverage * 0.3
                
                # Store signal data
                signals[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'action': decision,
                    'raw_action': action_value,
                    'direction': int(direction),
                    'strength': float(strength),
                    'leverage': float(leverage),
                    'confidence': float(strength),
                    'price': float(self.market_data[symbol]['close'].iloc[-1])
                }
                
                logger.info(f"Generated signal for {symbol}: {decision}, action: {action_value:.4f}, leverage: {leverage:.2f}x")
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    async def execute_trades(self, signals: Dict):
        """Execute trades based on generated signals."""
        try:
            trades_executed = []
            current_time = datetime.now()
            
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
        
        current_price = self.market_data[symbol]['close'].iloc[-1]
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
        logger.debug("Updating PnL...")
        
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if position['size'] == 0:
                continue
            
            current_price = self.market_data[symbol]['close'].iloc[-1]
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
            
            # Generate overall market summary
            if len(self.symbols) > 1:
                market_summary = self.commentary_generator.generate_market_summary(
                    symbols=self.symbols,
                    ohlcv_dict=self.market_data,
                    max_tokens=750,
                    temperature=0.7
                )
                
                commentary['market_summary'] = market_summary
                logger.info(f"Generated market summary: {len(market_summary)} chars")
            
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
            with open(self.trade_log_file, 'w') as f:
                json.dump({'trades': self.trades}, f, indent=2)
            logger.debug(f"Saved {len(self.trades)} trades to {self.trade_log_file}")
        except Exception as e:
            logger.error(f"Error saving trade data: {str(e)}")
    
    def _save_position_data(self):
        """Save current positions to log file."""
        try:
            position_data = {
                'timestamp': datetime.now().isoformat(),
                'balance': self.balance,
                'total_value': self.total_value,
                'positions': self.positions
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
    
    def _save_commentary(self, commentary):
        """Save market commentary to log file."""
        try:
            commentary_data = {
                'timestamp': datetime.now().isoformat(),
                'commentary': commentary
            }
            
            with open(self.commentary_log_file, 'w') as f:
                json.dump(commentary_data, f, indent=2)
            
            logger.debug(f"Saved commentary to {self.commentary_log_file}")
            
            # Also append to historical file
            historical_file = f"{self.log_dir}/commentary/commentary_{datetime.now().strftime('%Y-%m-%d')}.json"
            
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
            historical_data.append(commentary_data)
            
            # Write updated data
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving commentary: {str(e)}")
    
    async def run_trading_loop(self):
        """Run the main trading loop."""
        self.is_running = True
        logger.info("Starting trading loop")
        
        while self.is_running and not self.shutting_down:
            try:
                # Fetch latest data
                data_updated = await self._fetch_latest_data()
                
                if data_updated:
                    # Generate signals
                    signals = await self.generate_signals()
                    
                    # Execute trades
                    if signals:
                        await self.execute_trades(signals)
                    
                    # Update chatbot with latest data
                    self._update_chatbot()
                
                # Check if we need to log positions (every 15 minutes)
                if (datetime.now() - self.last_position_log_time).total_seconds() >= 900:  # 15 minutes
                    logger.info("Logging current positions")
                    self._save_position_data()
                    self.last_position_log_time = datetime.now()
                
                # Check if we need to generate commentary (every 15 minutes)
                if (datetime.now() - self.last_commentary_time).total_seconds() >= 900:  # 15 minutes
                    logger.info("Generating market commentary")
                    await self._generate_market_commentary()
                    self.last_commentary_time = datetime.now()
                
                # Sleep between updates
                await asyncio.sleep(30)  # Check for new data every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Sleep longer on error
    
    async def shutdown(self):
        """Shut down the trading monitor."""
        logger.info("Shutting down trading monitor")
        self.shutting_down = True
        self.is_running = False
        
        try:
            # Close all positions
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