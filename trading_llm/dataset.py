"""
Dataset utilities for Trading LLM training
"""

import os
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from stable_baselines3 import PPO
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.model_selection import train_test_split
import sys

logger = logging.getLogger(__name__)


class TechnicalIndicatorProcessor:
    """Calculate technical indicators for market data"""
    
    def __init__(self):
        """Initialize the technical indicator processor"""
        pass
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for a DataFrame of market data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of technical indicators
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Calculate indicators
        indicators = {}
        
        # Trend indicators
        indicators['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator().iloc[-1]
        indicators['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator().iloc[-1]
        indicators['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator().iloc[-1]
        indicators['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator().iloc[-1]
        
        # MACD
        macd = MACD(close=df['close'])
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_diff'] = macd.macd_diff().iloc[-1]
        
        # Momentum indicators
        indicators['rsi'] = RSIIndicator(close=df['close']).rsi().iloc[-1]
        
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # Volatility indicators
        bb = BollingerBands(close=df['close'])
        indicators['bb_high'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_low'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_width'] = (indicators['bb_high'] - indicators['bb_low']) / df['close'].iloc[-1]
        
        indicators['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range().iloc[-1]
        
        # Volume indicators
        indicators['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume().iloc[-1]
        
        # Price action indicators
        indicators['price'] = df['close'].iloc[-1]
        indicators['price_change_1d'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0
        indicators['price_change_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
        indicators['price_change_20d'] = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
        
        # Higher-level patterns and signals
        indicators['trend'] = self._determine_trend(df)
        indicators['volatility'] = self._calculate_volatility(df)
        indicators['support_resistance'] = self._identify_support_resistance(df)
        indicators['patterns'] = self._detect_candlestick_patterns(df)
        
        return indicators
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """
        Determine the current market trend
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Trend classification
        """
        # Calculate short and long term EMAs
        ema_short = EMAIndicator(close=df['close'], window=20).ema_indicator()
        ema_long = EMAIndicator(close=df['close'], window=50).ema_indicator()
        
        # Get the most recent values
        last_ema_short = ema_short.iloc[-1]
        last_ema_long = ema_long.iloc[-1]
        
        # Calculate the slope of the short EMA (rate of change)
        ema_short_slope = (ema_short.iloc[-1] / ema_short.iloc[-5] - 1) * 100 if len(ema_short) > 5 else 0
        
        # Determine trend based on EMA relationship and slope
        if last_ema_short > last_ema_long and ema_short_slope > 0.5:
            return "strong_uptrend"
        elif last_ema_short > last_ema_long:
            return "uptrend"
        elif last_ema_short < last_ema_long and ema_short_slope < -0.5:
            return "strong_downtrend"
        elif last_ema_short < last_ema_long:
            return "downtrend"
        else:
            return "sideways"
    
    def _calculate_volatility(self, df: pd.DataFrame) -> str:
        """
        Calculate market volatility level
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Volatility classification
        """
        # Calculate ATR as percentage of price
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        atr_percent = atr.average_true_range().iloc[-1] / df['close'].iloc[-1] * 100
        
        # Classification based on ATR percentage
        if atr_percent > 5:
            return "extreme"
        elif atr_percent > 3:
            return "high"
        elif atr_percent > 1.5:
            return "moderate"
        else:
            return "low"
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Identify key support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of support and resistance levels
        """
        # Simple implementation looking at recent price extremes
        last_20_days = df.iloc[-20:] if len(df) >= 20 else df
        
        # Find support levels (recent lows)
        support = last_20_days['low'].nsmallest(2).tolist()
        
        # Find resistance levels (recent highs)
        resistance = last_20_days['high'].nlargest(2).tolist()
        
        current_price = df['close'].iloc[-1]
        
        # Filter levels that are close to current price
        support = [level for level in support if level < current_price * 0.99]
        resistance = [level for level in resistance if level > current_price * 1.01]
        
        return {
            "support": support[:2],  # Take up to 2 support levels
            "resistance": resistance[:2],  # Take up to 2 resistance levels
            "current_price": current_price
        }
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect common candlestick patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Ensure we have enough data
        if len(df) < 3:
            return patterns
        
        # Get the last 3 candles
        last_3 = df.iloc[-3:].reset_index(drop=True)
        
        # Doji: open and close are very close
        last_candle = last_3.iloc[-1]
        if abs(last_candle['open'] - last_candle['close']) < 0.1 * (last_candle['high'] - last_candle['low']):
            patterns.append("doji")
        
        # Hammer: small body at the top, long lower shadow, little or no upper shadow
        if (last_candle['high'] - max(last_candle['open'], last_candle['close'])) < 0.1 * (last_candle['high'] - last_candle['low']) and \
           (min(last_candle['open'], last_candle['close']) - last_candle['low']) > 0.6 * (last_candle['high'] - last_candle['low']):
            patterns.append("hammer")
        
        # Engulfing pattern
        if len(last_3) >= 2:
            prev_candle = last_3.iloc[-2]
            # Bullish engulfing
            if prev_candle['close'] < prev_candle['open'] and \
               last_candle['close'] > last_candle['open'] and \
               last_candle['close'] > prev_candle['open'] and \
               last_candle['open'] < prev_candle['close']:
                patterns.append("bullish_engulfing")
            # Bearish engulfing
            elif prev_candle['close'] > prev_candle['open'] and \
                 last_candle['close'] < last_candle['open'] and \
                 last_candle['close'] < prev_candle['open'] and \
                 last_candle['open'] > prev_candle['close']:
                patterns.append("bearish_engulfing")
        
        # Morning star (bullish reversal)
        if len(last_3) >= 3:
            first = last_3.iloc[0]
            middle = last_3.iloc[1]
            last = last_3.iloc[2]
            
            if first['close'] < first['open'] and \
               abs(middle['close'] - middle['open']) < 0.3 * abs(first['close'] - first['open']) and \
               last['close'] > last['open'] and \
               last['close'] > (first['open'] + first['close']) / 2:
                patterns.append("morning_star")
        
        return patterns


class TradingDatasetGenerator:
    """Generate dataset for training Trading LLM from RL model"""
    
    def __init__(
        self,
        rl_model_path: str,
        market_data_path: str,
        output_path: str = "generated_data",
        output_dir: str = None,
        lookback_window: int = 50,
        samples_per_symbol: int = 100,
        template_path: str = None
    ):
        """
        Initialize the dataset generator
        
        Args:
            rl_model_path: Path to RL model
            market_data_path: Path to market data CSV or parquet file
            output_path: Path to output directory or filename (deprecated)
            output_dir: Path to output directory (preferred)
            lookback_window: Window size for market data
            samples_per_symbol: Number of samples to generate per symbol
            template_path: Path to custom templates
        """
        self.rl_model_path = rl_model_path
        self.market_data_path = market_data_path
        self.output_path = output_path
        self.output_dir = output_dir or output_path  # Use output_dir if provided, else fall back to output_path
        self.lookback_window = lookback_window
        self.samples_per_symbol = samples_per_symbol
        
        # Load templates (with fallback to default templates)
        self.templates = self._load_templates(template_path)
        
        # Add assets list
        self.assets = []
        
        # Initialize technical indicator processor
        self.indicator_processor = TechnicalIndicatorProcessor()
        
        logger.info(f"Initialized TradingDatasetGenerator with model: {rl_model_path}")
        logger.info(f"Market data: {market_data_path}")
        logger.info(f"Output path: {self.output_dir}")
        logger.info(f"Lookback window: {lookback_window}")
        logger.info(f"Samples per symbol: {samples_per_symbol}")
    
    def _load_templates(self, template_path: Optional[str] = None) -> List[str]:
        """
        Load explanation templates from file or use defaults
        
        Args:
            template_path: Path to template file (JSON)
            
        Returns:
            List of template strings
        """
        default_templates = [
            "Based on the current price of {recent_price:.2f}, the model recommends a {action} with {leverage:.2f}x leverage. {technical_indicators}",
            "Market analysis suggests a {action} signal at {recent_price:.2f} with {leverage:.2f}x leverage. {technical_indicators}",
            "The trading algorithm has identified a {action} opportunity at price {recent_price:.2f}, recommending {leverage:.2f}x leverage. {technical_indicators}",
            "Technical analysis indicates a {action} position with {leverage:.2f}x leverage at the current price of {recent_price:.2f}. {technical_indicators}",
            "With the current price at {recent_price:.2f}, the model's {action} signal suggests using {leverage:.2f}x leverage. {technical_indicators}"
        ]
        
        if not template_path:
            return default_templates
        
        try:
            with open(template_path, 'r') as f:
                templates = json.load(f)
                
            if not templates or not isinstance(templates, list):
                logger.warning(f"Invalid template format in {template_path}, using defaults")
                return default_templates
                
            return templates
            
        except Exception as e:
            logger.warning(f"Failed to load templates from {template_path}: {e}. Using defaults")
            return default_templates
    
    def _load_market_data(self) -> pd.DataFrame:
        """
        Load market data from file
        
        Returns:
            DataFrame with market data
        """
        try:
            logger.info(f"Loading market data from {self.market_data_path}")
            extension = os.path.splitext(self.market_data_path)[1].lower()
            
            if extension == '.csv':
                df = pd.read_csv(self.market_data_path)
            elif extension in ['.parquet', '.pq']:
                df = pd.read_parquet(self.market_data_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Loaded market data with {len(df)} rows")
            return df
        
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            raise
    
    def _load_rl_model(self) -> Any:
        """
        Load the RL model
        
        Returns:
            Loaded RL model
        """
        try:
            logger.info(f"Loading RL model from {self.rl_model_path}")
            model = PPO.load(self.rl_model_path)
            logger.info("RL model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading RL model: {str(e)}")
            raise
    
    def generate_dataset(
        self,
        policy_model_path: str = None,
        env_config_path: str = None,
        lookback_window: int = None,
        num_samples: int = 100,
        window_size: int = None,
        val_split: float = 0.2,
        min_action_threshold: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a dataset of explanations for RL trading decisions
        
        Args:
            policy_model_path: Path to the trained RL policy model
            env_config_path: Path to the environment configuration file
            lookback_window: Legacy parameter for lookback window (use window_size instead)
            num_samples: Number of samples to generate per trading symbol
            window_size: Number of candles to look back for context
            val_split: Validation split ratio (0.0-1.0)
            min_action_threshold: Threshold for filtering neutral actions
            
        Returns:
            Training and validation DataFrames with explanations
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {self.output_dir}")
        
        # Handle parameter compatibility
        lookback = window_size if window_size is not None else lookback_window if lookback_window is not None else self.lookback_window
        policy_path = policy_model_path if policy_model_path is not None else self.rl_model_path
        
        logger.info(f"Generating dataset with {num_samples} samples per symbol using model: {policy_path}")
        
        # Load the market data
        all_data = []
        
        # Handle both directory of CSV files and single parquet/csv file
        if os.path.isdir(self.market_data_path):
            # Directory with multiple files
            for file in os.listdir(self.market_data_path):
                if file.endswith(".csv"):
                    symbol = file.split(".")[0]
                    file_path = os.path.join(self.market_data_path, file)
                    try:
                        market_data = pd.read_csv(file_path)
                        # Add symbol column if not exists
                        if 'symbol' not in market_data.columns:
                            market_data['symbol'] = symbol
                        all_data.extend(self._process_symbol_data(
                            symbol, market_data, policy_path, env_config_path, 
                            lookback, num_samples, min_action_threshold
                        ))
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
        else:
            # Single file (parquet or csv)
            extension = os.path.splitext(self.market_data_path)[1].lower()
            try:
                if extension == '.csv':
                    market_data = pd.read_csv(self.market_data_path)
                elif extension in ['.parquet', '.pq']:
                    market_data = pd.read_parquet(self.market_data_path)
                else:
                    raise ValueError(f"Unsupported file format: {extension}")
                
                # If 'symbol' column exists, process each symbol separately
                if 'symbol' in market_data.columns:
                    logger.info(f"Processing multi-symbol data with {market_data['symbol'].nunique()} unique symbols")
                    for symbol, group_data in market_data.groupby('symbol'):
                        # Make a copy to avoid SettingWithCopyWarning
                        group_df = group_data.copy().reset_index(drop=True)
                        # Convert symbol to string to avoid errors
                        symbol = str(symbol)
                        logger.info(f"Processing symbol: {symbol} with {len(group_df)} rows")
                        
                        try:
                            symbol_samples = self._process_symbol_data(
                                symbol, group_df, policy_path, env_config_path, 
                                lookback, num_samples, min_action_threshold
                            )
                            all_data.extend(symbol_samples)
                        except Exception as e:
                            logger.error(f"Error processing grouped symbol {symbol}: {e}")
                else:
                    # Treat as single symbol
                    symbol = os.path.basename(self.market_data_path).split('.')[0]
                    # Add symbol column for consistency
                    if 'symbol' not in market_data.columns:
                        market_data['symbol'] = symbol
                    logger.info(f"Processing single-symbol data: {symbol} with {len(market_data)} rows")
                    all_data.extend(self._process_symbol_data(
                        symbol, market_data, policy_path, env_config_path, 
                        lookback, num_samples, min_action_threshold
                    ))
            except Exception as e:
                logger.error(f"Error processing file {self.market_data_path}: {e}")
        
        if not all_data:
            logger.error("No data was generated. Check file paths and market data format.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        
        # Save datasets
        train_path = os.path.join(self.output_dir, "train_data.json")
        val_path = os.path.join(self.output_dir, "val_data.json")
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        
        train_df.to_json(train_path, orient='records')
        val_df.to_json(val_path, orient='records')
        
        logger.info(f"Dataset generation complete. Training: {len(train_df)} samples, Validation: {len(val_df)} samples")
        logger.info(f"Saved to {train_path} and {val_path}")
        
        return train_df, val_df
    
    def _process_symbol_data(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        policy_path: str,
        env_config_path: str,
        lookback: int,
        num_samples: int,
        min_action_threshold: float
    ) -> List[Dict]:
        """
        Process data for a single symbol
        
        Args:
            symbol: Symbol name
            market_data: DataFrame with market data
            policy_path: Path to policy model
            env_config_path: Path to environment config
            lookback: Lookback window size
            num_samples: Number of samples to generate
            min_action_threshold: Threshold for filtering neutral actions
            
        Returns:
            List of samples for this symbol
        """
        symbol_samples = []
        
        try:
            # Preserve non-numeric columns (we'll exclude them from calculations)
            non_numeric_columns = ['symbol', 'asset', 'timestamp']
            preserved_columns = {}
            for col in non_numeric_columns:
                if col in market_data.columns:
                    preserved_columns[col] = market_data[col].copy()
            
            # Ensure required OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                logger.error(f"Missing required columns in data for {symbol}. Found: {market_data.columns.tolist()}")
                return []
            
            # Convert OHLCV columns to numeric types to prevent issues
            for col in market_data.columns:
                if col not in non_numeric_columns:
                    try:
                        market_data[col] = pd.to_numeric(market_data[col], errors='coerce').fillna(0).astype(np.float64)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric: {e}")
                
            # Calculate technical indicators
            try:
                market_data = self._calculate_technical_indicators(market_data)
            except Exception as e:
                logger.error(f"Failed to calculate technical indicators for {symbol}: {e}")
                # Continue without tech indicators - they'll be handled in _prepare_observation
            
            # Restore non-numeric columns
            for col, values in preserved_columns.items():
                market_data[col] = values
            
            # Ensure the data is sorted by timestamp
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp')
            else:
                # Use index as time if no timestamp column
                market_data = market_data.reset_index()
            
            # Initialize environment config (if available)
            env_config = {}
            if env_config_path and os.path.exists(env_config_path):
                try:
                    with open(env_config_path, 'r') as f:
                        env_config = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load environment config: {e}")
            
            # Load model
            try:
                logger.info(f"Loading policy model from {policy_path}")
                model = PPO.load(policy_path)
                
                # FIXED: Extract assets list from environment if possible
                if hasattr(model, 'env') and hasattr(model.env, 'get_attr'):
                    try:
                        # For vectorized environments
                        self.assets = model.env.get_attr('assets')[0]
                        logger.info(f"Extracted assets from model environment: {self.assets}")
                    except:
                        # Fallback - create assets list from symbol
                        self.assets = [symbol]
                else:
                    # Fallback - create assets list from symbol
                    self.assets = [symbol]
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return []
            
            # Generate explanations for random time windows
            max_window = len(market_data) - lookback
            
            if max_window <= 0:
                logger.warning(f"Not enough data for symbol {symbol}, skipping. Need at least {lookback} rows, found {len(market_data)}")
                return []
            
            # CHANGED: Sample from different market periods to ensure more diverse data
            # Divide the dataset into time segments and sample from each to ensure diversity
            # This helps prevent biased sampling from periods where the model consistently gives similar signals
            logger.info(f"Sampling from different time periods for {symbol} to ensure diverse signals")
            
            # Number of segments - use at least 5 segments to get different market regimes
            num_segments = min(10, max(5, num_samples // 50))
            segment_size = max_window // num_segments
            
            # Create a list of indices from different time segments
            start_indices = []
            samples_per_segment = min(num_samples * 3 // num_segments + 5, segment_size)
            
            # ENHANCED: Calculate volatility for the entire dataset to identify high/low volatility periods
            # Uses a simple rolling standard deviation of returns as volatility measure
            logger.info(f"Calculating volatility to ensure sampling from both calm and volatile periods")
            
            # Calculate returns (avoid look-ahead bias by ensuring we use data that would be available at decision time)
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().fillna(0)
                # Calculate rolling volatility 
                window_size = min(20, lookback // 2)  # Use smaller window for volatility calc
                rolling_volatility = returns.rolling(window=window_size).std().fillna(0)
                
                # Identify high and low volatility periods
                volatility_threshold_high = np.percentile(rolling_volatility, 75)  # Top 25% = high volatility
                volatility_threshold_low = np.percentile(rolling_volatility, 25)   # Bottom 25% = low volatility
                
                high_volatility_indices = np.where(rolling_volatility >= volatility_threshold_high)[0]
                low_volatility_indices = np.where(rolling_volatility <= volatility_threshold_low)[0]
                
                # Filter indices to ensure they are valid start indices (not too close to end)
                high_volatility_indices = [i for i in high_volatility_indices if i < max_window]
                low_volatility_indices = [i for i in low_volatility_indices if i < max_window]
                
                # Sample some high and low volatility periods
                volatility_samples_target = min(num_samples // 4, 50)  # Target 25% of samples from volatility-based selection
                
                if len(high_volatility_indices) > 0:
                    high_vol_samples = random.sample(
                        list(high_volatility_indices), 
                        min(volatility_samples_target, len(high_volatility_indices))
                    )
                    start_indices.extend(high_vol_samples)
                    logger.info(f"Added {len(high_vol_samples)} samples from high volatility periods")
                
                if len(low_volatility_indices) > 0:
                    low_vol_samples = random.sample(
                        list(low_volatility_indices), 
                        min(volatility_samples_target, len(low_volatility_indices))
                    )
                    start_indices.extend(low_vol_samples)
                    logger.info(f"Added {len(low_vol_samples)} samples from low volatility periods")
                
                # ADDED: Trend-based sampling to capture different market regimes (uptrend, downtrend, sideways)
                logger.info(f"Identifying market trends to ensure sampling from different market regimes")
                
                # Calculate short and long term moving averages to identify trends
                short_window = min(10, lookback // 4)
                long_window = min(30, lookback // 2)
                
                # Calculate smoothed price using moving averages
                short_ma = market_data['close'].rolling(window=short_window).mean().fillna(method='bfill')
                long_ma = market_data['close'].rolling(window=long_window).mean().fillna(method='bfill')
                
                # Identify trend periods (uptrend, downtrend, sideways)
                uptrend_mask = (short_ma > long_ma) & (market_data['close'].pct_change(20).fillna(0) > 0.01)
                downtrend_mask = (short_ma < long_ma) & (market_data['close'].pct_change(20).fillna(0) < -0.01)
                sideways_mask = (~uptrend_mask) & (~downtrend_mask)
                
                uptrend_indices = np.where(uptrend_mask)[0]
                downtrend_indices = np.where(downtrend_mask)[0]
                sideways_indices = np.where(sideways_mask)[0]
                
                # Filter indices to ensure they are valid start indices
                uptrend_indices = [i for i in uptrend_indices if i < max_window]
                downtrend_indices = [i for i in downtrend_indices if i < max_window]
                sideways_indices = [i for i in sideways_indices if i < max_window]
                
                # Sample from each trend type
                trend_samples_target = min(num_samples // 6, 30)  # Target ~15% of samples from each trend
                
                if len(uptrend_indices) > 0:
                    uptrend_samples = random.sample(
                        list(uptrend_indices),
                        min(trend_samples_target, len(uptrend_indices))
                    )
                    start_indices.extend(uptrend_samples)
                    logger.info(f"Added {len(uptrend_samples)} samples from uptrend periods")
                
                if len(downtrend_indices) > 0:
                    downtrend_samples = random.sample(
                        list(downtrend_indices),
                        min(trend_samples_target, len(downtrend_indices))
                    )
                    start_indices.extend(downtrend_samples)
                    logger.info(f"Added {len(downtrend_samples)} samples from downtrend periods")
                
                if len(sideways_indices) > 0:
                    sideways_samples = random.sample(
                        list(sideways_indices),
                        min(trend_samples_target, len(sideways_indices))
                    )
                    start_indices.extend(sideways_samples)
                    logger.info(f"Added {len(sideways_samples)} samples from sideways periods")
            
            # Continue with regular time-segment based sampling
            for i in range(num_segments):
                segment_start = i * segment_size
                segment_end = segment_start + segment_size
                
                # Ensure we don't go beyond dataset bounds
                if segment_end > max_window:
                    segment_end = max_window
                
                # Sample indices from this segment
                if segment_end > segment_start:
                    segment_samples = random.sample(
                        range(segment_start, segment_end),
                        min(samples_per_segment, segment_end - segment_start)
                    )
                    start_indices.extend(segment_samples)
            
            # Remove duplicates while preserving order
            start_indices = list(dict.fromkeys(start_indices))
            
            logger.info(f"Sampled {len(start_indices)} indices across {num_segments} time segments and volatility regimes")
            
            # Now do an initial pass to collect model behavior across different time periods
            # This helps understand the model's behavior distribution
            logger.info(f"Analyzing model behavior distribution across the dataset...")
            action_distribution = {"long": 0, "short": 0, "neutral": 0}
            
            # Sample every N points to get a quick distribution overview
            sampling_interval = max(1, len(market_data) // 1000)
            for i in range(0, len(market_data) - lookback, sampling_interval):
                try:
                    # Get window of market data
                    test_window = market_data.iloc[i:i + lookback].copy().reset_index(drop=True)
                    
                    # Create OHLCV-only window for observation preparation
                    test_obs_window = test_window.copy()
                    for col in non_numeric_columns:
                        if col in test_obs_window.columns:
                            test_obs_window = test_obs_window.drop(columns=[col])
                    
                    # Prepare observation from window
                    test_observation = self._prepare_observation(test_obs_window)
                    test_observation = np.nan_to_num(test_observation, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Get model prediction
                    test_action, _ = model.predict(test_observation, deterministic=False)
                    test_action = np.nan_to_num(test_action, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Extract action value
                    test_action_value = test_action[0] if isinstance(test_action, np.ndarray) and len(test_action) > 0 else test_action
                    
                    # Categorize the action
                    if test_action_value > 0.2:
                        action_distribution["long"] += 1
                    elif test_action_value < -0.2:
                        action_distribution["short"] += 1
                    else:
                        action_distribution["neutral"] += 1
                        
                except Exception as e:
                    # Skip errors in test sampling
                    pass
            
            # Log the action distribution
            total_samples = sum(action_distribution.values())
            if total_samples > 0:
                pct_long = action_distribution["long"] / total_samples * 100
                pct_short = action_distribution["short"] / total_samples * 100
                pct_neutral = action_distribution["neutral"] / total_samples * 100
                
                logger.info(f"Model action distribution for {symbol}:")
                logger.info(f"  Long: {action_distribution['long']} ({pct_long:.1f}%)")
                logger.info(f"  Short: {action_distribution['short']} ({pct_short:.1f}%)")
                logger.info(f"  Neutral: {action_distribution['neutral']} ({pct_neutral:.1f}%)")
            
            # Tracking for action diversity
            long_samples = 0
            short_samples = 0
            hold_samples = 0
            target_per_category = max(num_samples // 3, 10)  # Aim for balanced categories
            successful_samples = 0
            processed_indices = set()  # Track already used indices
            
            # First pass to collect actions and their counts
            action_values = []
            for start_idx in start_indices:
                try:
                    # Skip if already processed
                    if start_idx in processed_indices:
                        continue
                    
                    # Get window of market data
                    window = market_data.iloc[start_idx:start_idx + lookback].copy().reset_index(drop=True)
                    
                    # Create OHLCV-only window for observation preparation
                    observation_window = window.copy()
                    # Remove non-numeric columns for observation calculation
                    for col in non_numeric_columns:
                        if col in observation_window.columns:
                            observation_window = observation_window.drop(columns=[col])
                    
                    # Prepare observation from window
                    try:
                        observation = self._prepare_observation(observation_window)
                        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        logger.error(f"Failed to prepare observation for {symbol} at index {start_idx}: {e}")
                        continue
                    
                    # Get model prediction
                    try:
                        action, _ = model.predict(observation, deterministic=False)
                        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        logger.error(f"Failed to predict action for {symbol} at index {start_idx}: {e}")
                        continue
                    
                    # FIXED: Process multi-asset actions properly
                    # The model returns one action per asset, so we need to handle all of them
                    if isinstance(action, np.ndarray) and len(action) > 1:
                        # Multiple assets case
                        asset_actions = []
                        
                        # Process each asset action
                        for i, action_value in enumerate(action):
                            asset_name = self.assets[i] if i < len(self.assets) else f"Asset_{i}"
                            
                            # Ensure the action is a clean float and handle any NaN/Inf values
                            action_value = float(np.nan_to_num(action_value, nan=0.0, posinf=1.0, neginf=-1.0))
                            
                            # Skip if action is too close to neutral
                            if abs(action_value) < min_action_threshold:
                                continue
                                
                            # Calculate direction following perp_env logic
                            direction = np.sign(action_value) if abs(action_value) > 1e-8 else 0
                            
                            # Calculate leverage using the same method as in perp_env
                            env_max_leverage = 10.0
                            signal_threshold = 0.1  # Same as in perp_env
                            
                            if abs(action_value) < signal_threshold:
                                leverage = 0.0  # No position for very small actions
                            else:
                                # Get signal strength
                                signal_strength = abs(action_value)
                            
                                # Ensure signal strength is in valid range
                                signal_strength = max(signal_threshold, min(1.0, signal_strength))
                                
                                # Calculate leverage using the quadratic scaling for smoother leverage curve
                                min_leverage = 1.0
                                leverage = min_leverage + (env_max_leverage - min_leverage) * (signal_strength ** 1.5)
                                
                                # Ensure leverage is between min_leverage and env_max_leverage
                                leverage = max(min_leverage, min(leverage, env_max_leverage))
                            
                            # Apply direction to leverage
                            signed_leverage = leverage * direction
                            
                            # Store processed action
                            asset_actions.append({
                                "asset": asset_name,
                                "action_value": action_value,
                                "direction": direction,
                                "leverage": leverage,
                                "signed_leverage": signed_leverage
                            })
                            
                        # Skip if no significant actions
                        if not asset_actions:
                            continue
                            
                        # Use the first significant action for category counting
                        # This helps maintain the balancing system
                        primary_action = asset_actions[0]
                        action_value = primary_action["action_value"]
                        direction = primary_action["direction"]
                        leverage = primary_action["leverage"]
                        signed_leverage = primary_action["signed_leverage"]
                        
                        # Format multi-asset explanation
                        multi_asset_explanation = True
                            
                    else:
                        # Single asset case - use existing code
                        multi_asset_explanation = False
                        
                        # Extract action value
                        action_value = action[0] if isinstance(action, np.ndarray) and len(action) > 0 else action
                        
                        # FIXED: Ensure raw_action is the unmodified signal from the model
                        raw_action = float(action_value)
                        
                        # Ensure the action is a clean float and handle any NaN/Inf values
                        action_value = float(np.nan_to_num(action_value, nan=0.0, posinf=1.0, neginf=-1.0))
                        
                        # Debug the raw action value from model
                        logger.warning(f"Raw model action value for {symbol}: {action_value}")
                        
                        # Skip if action is too close to neutral
                        if abs(action_value) < min_action_threshold:
                            continue
                        
                        # Calculate direction following perp_env logic
                        # In perp_env, direction = np.sign(signal) if abs(signal) > 1e-8 else 0
                        direction = np.sign(action_value) if abs(action_value) > 1e-8 else 0
                        
                        # Calculate leverage using the same method as in perp_env
                        # Using _get_target_leverage equivalent implementation
                        env_max_leverage = 10.0
                        signal_threshold = 0.1  # Same as in perp_env
                        
                        if abs(action_value) < signal_threshold:
                            leverage = 0.0  # No position for very small actions
                        else:
                            # Get signal strength
                            signal_strength = abs(action_value)
                        
                            # Ensure signal strength is in valid range
                            signal_strength = max(signal_threshold, min(1.0, signal_strength))
                            
                            # Calculate leverage using the quadratic scaling for smoother leverage curve
                            min_leverage = 1.0
                            leverage = min_leverage + (env_max_leverage - min_leverage) * (signal_strength ** 1.5)
                            
                            # Ensure leverage is between min_leverage and env_max_leverage
                            leverage = max(min_leverage, min(leverage, env_max_leverage))
                        
                        # IMPORTANT: Apply direction to leverage like in perp_env
                        # In perp_env, they use: target_value = direction * effective_leverage * portfolio_value
                        # So the direction is combined with leverage
                        signed_leverage = leverage * direction
                    
                    # Categorize the action for balancing
                    action_category = "long" if action_value > 0 else "short" if action_value < 0 else "hold"
                    
                    # Skip this window if we're trying to balance categories and already have enough of this type
                    if action_category == "long" and long_samples >= target_per_category:
                        # Skip if we already have enough long samples
                        continue
                    elif action_category == "short" and short_samples >= target_per_category:
                        # Skip if we already have enough short samples
                        continue
                    elif action_category == "hold" and hold_samples >= target_per_category:
                        # Skip if we already have enough hold samples
                        continue
                    
                    # Calculate direction and leverage
                    direction = 1 if action_value > 0 else -1 if action_value < 0 else 0
                    
                    # Calculate leverage using the same method as in perp_env
                    # Using _get_target_leverage equivalent implementation
                    env_max_leverage = 10.0
                    signal_threshold = 0.1  # Same as in perp_env
                    
                    if abs(action_value) < signal_threshold:
                        leverage = 0.0  # No position for very small actions
                    else:
                        # Get signal strength
                        signal_strength = abs(action_value)
                        
                        # Ensure signal strength is in valid range
                        signal_strength = max(signal_threshold, min(1.0, signal_strength))
                        
                        # Calculate leverage using the quadratic scaling for smoother leverage curve
                        min_leverage = 1.0
                        leverage = min_leverage + (env_max_leverage - min_leverage) * (signal_strength ** 1.5)
                        
                        # Ensure leverage is between min_leverage and env_max_leverage
                        leverage = max(min_leverage, min(leverage, env_max_leverage))
                    
                    # Format action to include direction and strength
                    formatted_action = [float(direction), float(action_value), float(signed_leverage)]
                    
                    # Extract technical indicators for the last candle
                    tech_indicators = {}
                    skip_columns = set(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'index', 'symbol', 'asset'])
                    for col in window.columns:
                        if col not in skip_columns:
                            try:
                                value = window.iloc[-1][col]
                                if isinstance(value, (int, float, np.number)):
                                    tech_indicators[col] = float(value)
                            except Exception:
                                pass
                    
                    # Generate explanation based on whether we have multi-asset actions
                    try:
                        if multi_asset_explanation:
                            # Generate multi-asset explanation
                            assets_explanations = []
                            
                            # Process each asset with significant action
                            for asset_action in asset_actions:
                                asset_name = asset_action["asset"]
                                asset_action_value = asset_action["action_value"]
                                asset_leverage = asset_action["leverage"]
                                
                                # Generate explanation for this asset
                                asset_explanation = self._generate_explanation(
                                    observation,
                                    asset_action_value,
                                    tech_indicators,
                                    asset_leverage
                                )
                                
                                # Add to list with asset name
                                assets_explanations.append(f"{asset_name}: {asset_explanation}")
                            
                            # Combine explanations
                            explanation = "\n\nMulti-Asset Trading Decision:\n" + "\n\n".join(assets_explanations)
                            
                            # Format action including all assets
                            formatted_action = [
                                [asset_action["asset"], 
                                 float(asset_action["direction"]), 
                                 float(asset_action["action_value"]), 
                                 float(asset_action["signed_leverage"])]
                                for asset_action in asset_actions
                            ]
                        else:
                            # Single asset explanation - original code
                            explanation = self._generate_explanation(
                                observation, 
                                action_value,  # Pass raw action value instead of array
                                tech_indicators,
                                leverage  # Pass calculated leverage directly
                            )
                            
                            # Format action to include direction and strength
                            formatted_action = [float(direction), float(action_value), float(signed_leverage)]
                    except Exception as e:
                        logger.error(f"Failed to generate explanation for {symbol} at index {start_idx}: {e}")
                        if multi_asset_explanation:
                            # Default multi-asset explanation
                            assets_descriptions = []
                            for asset_action in asset_actions:
                                asset_name = asset_action["asset"]
                                action_desc = "buy" if asset_action["direction"] > 0 else "sell"
                                assets_descriptions.append(f"{asset_name}: The model suggests to {action_desc} based on current market conditions.")
                            explanation = "\n\n".join(assets_descriptions)
                            
                            # Format action including all assets
                            formatted_action = [
                                [asset_action["asset"], 
                                 float(asset_action["direction"]), 
                                 float(asset_action["action_value"]), 
                                 float(asset_action["signed_leverage"])]
                                for asset_action in asset_actions
                            ]
                        else:
                            # Default single-asset explanation
                            explanation = f"The model suggests to {'buy' if action_value > 0 else 'sell'} based on current market conditions."
                            formatted_action = [float(direction), float(action_value), float(signed_leverage)]
                    
                    # Create sample with formatted prompts
                    if multi_asset_explanation:
                        # Multi-asset prompt
                        assets_list = ", ".join([asset_action["asset"] for asset_action in asset_actions])
                        prompt = f"Given the market data for multiple assets ({assets_list}), explain the multi-asset trading decision: "
                    else:
                        # Single-asset prompt
                        prompt = f"Given the market data for {symbol}, explain the trading decision: "
                    
                    # Timestamp extraction
                    start_timestamp = None
                    end_timestamp = None
                    if 'timestamp' in window.columns:
                        # Get both start and end timestamps for the window
                        start_timestamp = window.iloc[0]['timestamp']
                        end_timestamp = window.iloc[-1]['timestamp']
                        timestamp = window.iloc[-1]['timestamp']
                    else:
                        timestamp = f"{symbol}_{start_idx}_{start_idx + lookback}"
                    
                    # Convert arrays to lists
                    try:
                        observation_list = observation.tolist()
                    except (AttributeError, TypeError):
                        observation_list = [float(x) for x in observation]
                    
                    # Create sample
                    sample = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'observation': observation_list,
                        'action': formatted_action,
                        'explanation': explanation,
                        'prompt': prompt,
                        # Add detailed period information for debugging
                        'window_info': {
                            'start_idx': start_idx,
                            'end_idx': start_idx + lookback,
                            'start_timestamp': start_timestamp,
                            'end_timestamp': end_timestamp,
                            'window_size': lookback,
                            'price_start': float(window.iloc[0]['close']),
                            'price_end': float(window.iloc[-1]['close']),
                        }
                    }
                    
                    # Add multi-asset information if applicable
                    if multi_asset_explanation:
                        sample['is_multi_asset'] = True
                        sample['assets'] = [asset_action["asset"] for asset_action in asset_actions]
                        
                        # Add additional action details in a more structured format
                        sample['actions_by_asset'] = {
                            asset_action["asset"]: {
                                "action_value": float(asset_action["action_value"]),
                                "direction": float(asset_action["direction"]),
                                "leverage": float(asset_action["leverage"]),
                                "signed_leverage": float(asset_action["signed_leverage"])
                            }
                            for asset_action in asset_actions
                        }
                    else:
                        sample['is_multi_asset'] = False
                        sample['window_info']['action_value'] = float(action_value)
                        sample['window_info']['leverage'] = float(leverage)
                    
                    symbol_samples.append(sample)
                    processed_indices.add(start_idx)  # Mark this index as processed
                    successful_samples += 1
                    
                    # Enhanced logging with timestamp info for better debugging
                    time_info = f"[{start_timestamp} to {end_timestamp}]" if start_timestamp and end_timestamp else f"[idx:{start_idx}-{start_idx+lookback}]"
                    price_change = ((float(window.iloc[-1]['close']) / float(window.iloc[0]['close'])) - 1.0) * 100
                    
                    # Log information based on single or multi-asset
                    if multi_asset_explanation:
                        # Create a summary of actions for all assets
                        actions_summary = ", ".join([
                            f"{a['asset']}: {'LONG' if a['direction'] > 0 else 'SHORT'} ({a['action_value']:.2f})"
                            for a in asset_actions
                        ])
                        
                        log_message = f"MULTI-ASSET SAMPLE | #{successful_samples} | {time_info} | price change: {price_change:.2f}% | actions: {actions_summary}"
                    else:
                        # Single asset logging (original format)
                        log_message = f"SAMPLE DATA | #{successful_samples} | {symbol} | {time_info} | price change: {price_change:.2f}% | action: {action_value:.4f} | leverage: {leverage:.2f}x | direction: {'LONG' if action_value > 0 else 'SHORT'}"
                    
                    # Update category counters
                    if action_category == "long":
                        long_samples += 1
                    elif action_category == "short":
                        short_samples += 1
                    else:
                        hold_samples += 1
                    
                    # Periodically log progress
                    if successful_samples % 50 == 0:
                        logger.info(f"Generated {successful_samples} samples for {symbol} so far... (long: {long_samples}, short: {short_samples}, hold: {hold_samples})")
                        
                except Exception as e:
                    logger.error(f"Error processing window for {symbol} at index {start_idx}: {e}")
            
            logger.info(f"Generated {len(symbol_samples)} samples for {symbol} with action distribution: long: {long_samples}, short: {short_samples}, hold: {hold_samples}")
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
        
        return symbol_samples
    
    def _prepare_observation(self, window: pd.DataFrame) -> np.ndarray:
        """
        Convert market data window to observation format required by the RL model
        
        Args:
            window: DataFrame containing market data for a time window
            
        Returns:
            Flat observation array with 78 features compatible with training format
        """
        # Expected 78 features total (13 features × 6 time periods)
        
        # Basic OHLCV features
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Technical indicator columns that should be included
        indicator_columns = [
            'SMA_10', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 
            'BB_upper', 'BB_middle', 'BB_lower'
        ]
        
        # Ensure all required OHLCV columns exist
        missing_ohlcv = [col for col in ohlcv_columns if col not in window.columns]
        if missing_ohlcv:
            raise ValueError(f"Missing required OHLCV columns: {missing_ohlcv}")
        
        # Ensure OHLCV columns are numeric (float64)
        for col in ohlcv_columns:
            window[col] = pd.to_numeric(window[col], errors='coerce').astype(np.float64)
        
        # Calculate technical indicators if any are missing
        missing_indicators = [col for col in indicator_columns if col not in window.columns]
        if missing_indicators:
            logger.warning(f"Missing technical indicators: {missing_indicators}, calculating...")
            try:
                window = self._calculate_technical_indicators(window)
            except Exception as e:
                logger.error(f"Failed to calculate technical indicators: {e}")
                # Create dummy indicators with zeros
                for col in missing_indicators:
                    if col not in window.columns:
                        window[col] = 0.0
        
        # Ensure indicator columns are numeric (float64)
        for col in indicator_columns:
            if col in window.columns:
                window[col] = pd.to_numeric(window[col], errors='coerce').fillna(0.0).astype(np.float64)
            else:
                window[col] = np.zeros(len(window), dtype=np.float64)
        
        # Number of time periods to include (78 features total / 13 features per period = 6 periods)
        required_periods = 6
        
        # Ensure we have enough periods in the window
        if len(window) < required_periods:
            logger.warning(f"Window size ({len(window)}) is less than required periods ({required_periods})")
            # Pad with repeated first row if needed
            padding = required_periods - len(window)
            if padding > 0:
                padding_data = pd.concat([window.iloc[[0]]] * padding, ignore_index=True)
                window = pd.concat([padding_data, window], ignore_index=True)
        
        # Get the most recent periods
        recent_window = window.iloc[-required_periods:].copy()
        
        # Extract features and normalize
        features_list = []
        
        for i in range(required_periods):
            period_data = recent_window.iloc[i]
            
            # Get OHLCV and normalize - ensure float64 type
            try:
                ohlcv = np.array([float(period_data[col]) for col in ohlcv_columns], dtype=np.float64)
                ohlcv_normalized = self._normalize_ohlcv(ohlcv, float(recent_window['close'].iloc[-1]))
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting OHLCV to float: {e}")
                # Use zeros as fallback
                ohlcv_normalized = np.zeros(len(ohlcv_columns), dtype=np.float64)
            
            # Get technical indicators and normalize - ensure float64 type
            indicator_values = []
            for col in indicator_columns:
                try:
                    value = float(period_data[col])
                    # Check for NaN/Inf
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    indicator_values.append(value)
                except (KeyError, ValueError, TypeError):
                    # Use zero as fallback
                    indicator_values.append(0.0)
            
            indicators = np.array(indicator_values, dtype=np.float64)
            indicators_normalized = self._normalize_indicators(indicators)
            
            # Combine all features for this period - ensure float64 type
            period_features = np.concatenate([ohlcv_normalized, indicators_normalized]).astype(np.float64)
            features_list.append(period_features)
        
        # Flatten all features into one array - ensure float64 type
        try:
            flat_observation = np.concatenate(features_list).astype(np.float64)
        except (ValueError, TypeError) as e:
            logger.error(f"Error when concatenating features: {e}")
            # Return zeros as fallback
            expected_features = (len(ohlcv_columns) + len(indicator_columns)) * required_periods
            return np.zeros(expected_features, dtype=np.float64)
        
        # Verify we have the expected number of features
        # 13 features per period × 6 periods = 78
        expected_features = (len(ohlcv_columns) + len(indicator_columns)) * required_periods
        if len(flat_observation) != expected_features:
            logger.warning(f"Expected {expected_features} features, got {len(flat_observation)}. Padding with zeros.")
            # Pad with zeros if necessary
            if len(flat_observation) < expected_features:
                padding = np.zeros(expected_features - len(flat_observation), dtype=np.float64)
                flat_observation = np.concatenate([flat_observation, padding])
            else:
                # Truncate if too long
                flat_observation = flat_observation[:expected_features]
        
        # Final check to ensure float64 type
        if flat_observation.dtype != np.float64:
            flat_observation = flat_observation.astype(np.float64)
            
        # Check for NaN/Inf values and replace with zeros
        flat_observation = np.nan_to_num(flat_observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        return flat_observation
    
    def _normalize_ohlcv(self, ohlcv: np.ndarray, reference_price: float) -> np.ndarray:
        """
        Normalize OHLCV data relative to a reference price
        
        Args:
            ohlcv: OHLCV values
            reference_price: Price to normalize against (usually latest close)
            
        Returns:
            Normalized OHLCV values
        """
        try:
            # Ensure input is float64
            ohlcv = ohlcv.astype(np.float64)
            reference_price = float(reference_price)
            
            if reference_price <= 0:
                reference_price = 1.0  # Avoid division by zero
                
            # Normalize OHLC relative to reference price
            ohlc = ohlcv[:4]
            volume = ohlcv[4]
            
            normalized_ohlc = ohlc / reference_price - 1.0
            
            # Log normalize volume if positive
            if volume > 0:
                normalized_volume = np.log(volume) / 10.0
            else:
                normalized_volume = 0.0
            
            result = np.append(normalized_ohlc, normalized_volume).astype(np.float64)
            
            # Replace any NaN/Inf values
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            return result
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error in normalize_ohlcv: {e}")
            return np.zeros(5, dtype=np.float64)  # Return zeros as fallback
    
    def _normalize_indicators(self, indicators: np.ndarray) -> np.ndarray:
        """
        Normalize technical indicators
        
        Args:
            indicators: Array of technical indicator values
            
        Returns:
            Normalized indicator values
        """
        try:
            # Ensure input is float64
            indicators = indicators.astype(np.float64)
            
            # Apply appropriate normalization to each indicator
            # This should match the normalization used during training
            normalized = np.zeros_like(indicators, dtype=np.float64)
            
            # RSI is already normalized (0-100)
            rsi_idx = 2  # Index of RSI in indicators array
            if 0 <= rsi_idx < len(indicators):
                normalized[rsi_idx] = indicators[rsi_idx] / 100.0
            
            # SMAs and Bollinger Bands are price-based
            for i in [0, 1, 4, 5, 6]:  # Indices of SMA_10, SMA_20, BB_upper, BB_middle, BB_lower
                if i < len(indicators) and not np.isnan(indicators[i]) and indicators[i] != 0:
                    normalized[i] = 0.0  # Placeholder, will be normalized with OHLC
            
            # MACD indicators can be both positive and negative
            for i in [3, 4]:  # Indices of MACD, MACD_signal
                if i < len(indicators) and not np.isnan(indicators[i]):
                    # Scale MACD to a reasonable range
                    normalized[i] = np.clip(indicators[i] / 100.0, -1, 1)
            
            # Replace any NaNs/Infs
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            return normalized.astype(np.float64)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error in normalize_indicators: {e}")
            return np.zeros(len(indicators), dtype=np.float64)  # Return zeros as fallback
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the market data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Identify columns that should not be converted to numeric
            non_numeric_columns = ['symbol', 'asset']
            
            # Ensure OHLCV columns are numeric
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(np.float64)
            
            # Add basic technical indicators
            try:
                # Moving averages
                data['SMA_10'] = SMAIndicator(data['close'], window=10).sma_indicator().astype(np.float64)
                data['SMA_20'] = SMAIndicator(data['close'], window=20).sma_indicator().astype(np.float64)
                
                # Relative Strength Index
                data['RSI_14'] = RSIIndicator(data['close'], window=14).rsi().astype(np.float64)
                
                # MACD
                macd_indicator = MACD(data['close'])
                data['MACD'] = macd_indicator.macd().astype(np.float64)
                data['MACD_signal'] = macd_indicator.macd_signal().astype(np.float64)
                
                # Bollinger Bands
                bollinger = BollingerBands(data['close'])
                data['BB_upper'] = bollinger.bollinger_hband().astype(np.float64)
                data['BB_middle'] = bollinger.bollinger_mavg().astype(np.float64)
                data['BB_lower'] = bollinger.bollinger_lband().astype(np.float64)
            except Exception as e:
                logger.error(f"Error calculating specific indicator: {e}")
                # If any indicator calculation fails, create empty ones
                for indicator in ['SMA_10', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 
                                'BB_upper', 'BB_middle', 'BB_lower']:
                    if indicator not in data.columns:
                        data[indicator] = np.zeros(len(data), dtype=np.float64)
            
            # Fill NaN values that may be created by indicators
            for col in data.columns:
                # Skip non-numeric columns
                if col in non_numeric_columns:
                    continue
                # Skip OHLCV columns that were already handled
                if col not in ohlcv_columns:
                    # Update to avoid deprecated warning by using bfill() and ffill() directly
                    data[col] = data[col].bfill().ffill().fillna(0).astype(np.float64)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return original data with empty indicator columns
            for indicator in ['SMA_10', 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal', 
                            'BB_upper', 'BB_middle', 'BB_lower']:
                if indicator not in df.columns:
                    df[indicator] = np.zeros(len(df), dtype=np.float64)
            return df
    
    def _generate_explanation(
        self, 
        observation: np.ndarray, 
        action: float, 
        technical_indicators: Optional[Dict[str, float]] = None,
        leverage: float = 1.0
    ) -> str:
        """
        Generate a natural language explanation for a trading decision
        
        Args:
            observation: Observation array from the environment
            action: Action value from the RL policy (values in range [-1,1] that determine direction and signal strength)
            technical_indicators: Dictionary of technical indicators
            leverage: Calculated leverage for the trading decision (not used in explanation generation)
            
        Returns:
            Natural language explanation of the trading decision with in-depth analysis
        """
        try:
            # Determine action type and direction based on signal value
            # In the trading environment, the sign of the action indicates direction:
            # - Positive values = Long position
            # - Negative values = Short position
            # - Near zero values = Hold/no position
            signal_strength = abs(action)
            direction = 1 if action > 0 else (-1 if action < 0 else 0)
            
            # Determine trade decision based on signal strength threshold (matches the environment's threshold)
            signal_threshold = 0.1  # Same as environment's default signal_threshold
            
            if signal_strength < signal_threshold:
                action_type = "hold"
                # No direction for hold
                direction = 0
            else:
                action_type = "buy" if direction > 0 else "sell"
            
            # Format technical indicators for explanation
            tech_indicators_str = ""
            if technical_indicators:
                # Format selected indicators
                indicator_items = []
                for indicator, value in technical_indicators.items():
                    # Format the indicator value (round floats to 2 decimal places)
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    indicator_items.append(f"{indicator}: {formatted_value}")
                
                tech_indicators_str = ", ".join(indicator_items)
            
            # Get indicator values with safe defaults
            rsi_val = technical_indicators.get('RSI_14', 50.0) if technical_indicators else 50.0
            macd_val = technical_indicators.get('MACD', 0.0) if technical_indicators else 0.0
            macd_signal_val = technical_indicators.get('MACD_signal', 0.0) if technical_indicators else 0.0
            macd_hist_val = technical_indicators.get('MACD_hist', 0.0) if technical_indicators else 0.0
            sma20_val = technical_indicators.get('SMA_20', 0.0) if technical_indicators else 0.0
            sma50_val = technical_indicators.get('SMA_50', 0.0) if technical_indicators else 0.0
            bb_upper = technical_indicators.get('BB_upper', 0.0) if technical_indicators else 0.0
            bb_lower = technical_indicators.get('BB_lower', 0.0) if technical_indicators else 0.0
            price = technical_indicators.get('close', 0.0) if technical_indicators else 0.0
            volume = technical_indicators.get('volume', 0.0) if technical_indicators else 0.0
            trend = technical_indicators.get('trend', 'unknown') if technical_indicators else 'unknown'
            
            # Create explanation templates based on action type with in-depth analysis
            # Focus on WHY a position should be taken based on technical analysis
            templates = {
                "buy": [
                    # Comprehensive technical analysis for LONG positions
                    "After analyzing the current market conditions, I'm recommending a LONG position. The RSI at {rsi:.2f} indicates bullish momentum that hasn't yet reached overbought territory. The MACD line ({macd:.2f}) is above the signal line ({macd_signal:.2f}), confirming positive momentum. Additionally, price has formed support above the SMA-20 ({sma20:.2f}), suggesting a continued uptrend. Volume analysis shows increasing buying pressure, further supporting this bullish outlook.",
                    
                    "My analysis points to a strong buying opportunity. The market is showing clear bullish signals with the RSI at {rsi:.2f} indicating strengthening momentum. MACD at {macd:.2f} versus signal line at {macd_signal:.2f} shows a positive crossover, typically a reliable buy signal. Price action is constructive with recent candles closing above both SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), confirming the uptrend. The Bollinger Bands show expansion with price pushing toward the upper band ({bb_upper:.2f}), indicating potential for continued upside movement.",
                    
                    "I recommend going LONG based on multiple confluence factors. First, the market structure shows higher lows forming an ascending pattern. The RSI at {rsi:.2f} indicates bullish momentum without being overextended. The MACD histogram ({macd_hist:.2f}) is positive and expanding, showing accelerating upward momentum. Price is trading above key moving averages with SMA-20 ({sma20:.2f}) crossing above SMA-50 ({sma50:.2f}), a classic bullish signal. Market is in a confirmed {trend} trend, with price finding support at the previous resistance levels, suggesting further upside potential.",
                    
                    "Technical analysis strongly favors a LONG position. The price has broken above a key resistance level and is now retesting it as support. The RSI reading of {rsi:.2f} shows bullish momentum that isn't yet overextended, while the MACD ({macd:.2f}) shows a positive crossover, confirming bullish momentum. Price is trading above both the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), with the shorter-term average trending upward. The recent price action shows strong buying pressure with bullish candlestick patterns and above-average volume, indicating institutional interest in pushing prices higher.",
                    
                    "Market analysis indicates it's time to go LONG. We're seeing a bullish momentum shift with the RSI at {rsi:.2f} showing strength without being overbought. The MACD indicator is bullish with the MACD line ({macd:.2f}) above the signal line ({macd_signal:.2f}) and the histogram ({macd_hist:.2f}) expanding positively. Price is trading comfortably above the SMA-20 ({sma20:.2f}) which is now sloping upward. Additionally, price has just bounced off the lower Bollinger Band ({bb_lower:.2f}) and is heading toward the middle band, indicating positive mean reversion with upside potential. The overall {trend} trend remains intact, supporting a bullish outlook.",
                    
                    "Based on comprehensive chart analysis, I recommend taking a LONG position. The market is forming a bullish continuation pattern with the RSI at {rsi:.2f} showing momentum acceleration. MACD analysis shows the MACD line ({macd:.2f}) above the signal line ({macd_signal:.2f}) with a widening histogram ({macd_hist:.2f}), confirming increasing bullish momentum. Price is trading above both the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), with the moving averages aligned in a bullish configuration. Volume analysis shows accumulation on up-moves and decreased volume on pullbacks, a classic sign of bullish continuation."
                ],
                "sell": [
                    # Comprehensive technical analysis for SHORT positions
                    "After analyzing the current market conditions, I'm recommending a SHORT position. The RSI at {rsi:.2f} indicates bearish momentum and possible overbought conditions. The MACD line ({macd:.2f}) is below the signal line ({macd_signal:.2f}), confirming negative momentum. Additionally, price has broken below the SMA-20 ({sma20:.2f}), suggesting a potential trend reversal. Volume analysis shows increasing selling pressure, further supporting this bearish outlook.",
                    
                    "My analysis points to a strong selling opportunity. The market is showing clear bearish signals with the RSI at {rsi:.2f} indicating weakening momentum. MACD at {macd:.2f} versus signal line at {macd_signal:.2f} shows a negative crossover, typically a reliable sell signal. Price action is deteriorating with recent candles closing below both SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), confirming the downtrend. The Bollinger Bands show expansion with price pushing toward the lower band ({bb_lower:.2f}), indicating potential for continued downside movement.",
                    
                    "I recommend going SHORT based on multiple confluence factors. First, the market structure shows lower highs forming a descending pattern. The RSI at {rsi:.2f} indicates bearish momentum and possible divergence from price. The MACD histogram ({macd_hist:.2f}) is negative and expanding, showing accelerating downward momentum. Price is trading below key moving averages with SMA-20 ({sma20:.2f}) crossing below SMA-50 ({sma50:.2f}), a classic bearish signal. Market is in a confirmed {trend} trend, with price finding resistance at the previous support levels, suggesting further downside potential.",
                    
                    "Technical analysis strongly favors a SHORT position. The price has broken below a key support level and is now retesting it as resistance. The RSI reading of {rsi:.2f} shows bearish momentum that's accelerating, while the MACD ({macd:.2f}) shows a negative crossover, confirming bearish momentum. Price is trading below both the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), with the shorter-term average trending downward. The recent price action shows strong selling pressure with bearish candlestick patterns and above-average volume, indicating institutional interest in pushing prices lower.",
                    
                    "Market analysis indicates it's time to go SHORT. We're seeing a bearish momentum shift with the RSI at {rsi:.2f} showing weakness and potential for continued decline. The MACD indicator is bearish with the MACD line ({macd:.2f}) below the signal line ({macd_signal:.2f}) and the histogram ({macd_hist:.2f}) expanding negatively. Price is trading uncomfortably below the SMA-20 ({sma20:.2f}) which is now sloping downward. Additionally, price has just rejected the upper Bollinger Band ({bb_upper:.2f}) and is heading toward the middle band, indicating negative mean reversion with downside potential. The overall {trend} trend has shifted bearish, supporting a negative outlook.",
                    
                    "Based on comprehensive chart analysis, I recommend taking a SHORT position. The market is forming a bearish continuation pattern with the RSI at {rsi:.2f} showing momentum deterioration. MACD analysis shows the MACD line ({macd:.2f}) below the signal line ({macd_signal:.2f}) with a widening histogram ({macd_hist:.2f}), confirming increasing bearish momentum. Price is trading below both the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), with the moving averages aligned in a bearish configuration. Volume analysis shows distribution on down-moves and decreased volume on bounces, a classic sign of bearish continuation."
                ],
                "hold": [
                    # Comprehensive analysis for HOLD positions
                    "After careful analysis of current market conditions, I recommend maintaining a NEUTRAL position. The RSI at {rsi:.2f} is in mid-range, showing no clear directional momentum. The MACD line ({macd:.2f}) and signal line ({macd_signal:.2f}) are close together, indicating indecision in the market. Price is oscillating around the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}), showing a lack of clear trend. With these mixed signals, it's prudent to wait for a more definitive setup before taking a directional position.",
                    
                    "Based on technical analysis, I suggest staying NEUTRAL at this time. The market is showing conflicting signals with the RSI at {rsi:.2f} in the middle range. The MACD histogram ({macd_hist:.2f}) is flat, showing minimal momentum in either direction. Price is trading between key support and resistance levels without clear breakout signals. The Bollinger Bands are contracting, indicating decreasing volatility and potential for a range-bound market in the near term. Better to preserve capital and wait for a higher-probability setup to emerge.",
                    
                    "My analysis indicates a HOLD position is appropriate. The market is in a consolidation phase with the RSI at {rsi:.2f} showing neither overbought nor oversold conditions. MACD analysis shows the MACD line ({macd:.2f}) and signal line ({macd_signal:.2f}) are intertwined without clear direction. Price is moving sideways between established support and resistance levels, with the SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}) flattening out. The current {trend} lacks conviction, and volume is below average, suggesting a period of accumulation or distribution before the next directional move. Better to conserve capital and wait for clarity.",
                    
                    "Technical indicators suggest maintaining a NEUTRAL stance. The market lacks directional conviction with the RSI at {rsi:.2f} hovering in the middle range. The MACD shows minimal momentum with the MACD line ({macd:.2f}) and signal line ({macd_signal:.2f}) moving close together. Price action is characterized by smaller-range candles with price trading within a defined range. The moving averages SMA-20 ({sma20:.2f}) and SMA-50 ({sma50:.2f}) are flattening and converging, typical of a consolidating market. With this technical picture, it's prudent to stand aside until a clearer trading opportunity presents itself."
                ]
            }
            
            # Select a random template for the action type
            template = random.choice(templates[action_type])
            
            # Format template with actual values
            explanation = template.format(
                rsi=rsi_val,
                macd=macd_val,
                macd_signal=macd_signal_val,
                macd_hist=macd_hist_val,
                sma20=sma20_val,
                sma50=sma50_val,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                price=price,
                volume=volume,
                trend=trend
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"The model suggests to {action_type} based on the current market conditions."


class TradingTextDataset(Dataset):
    """PyTorch dataset for Trading LLM training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        is_train: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to data file (JSON format)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            is_train: Whether this is a training dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index
        
        Args:
            idx: Index
            
        Returns:
            Tokenized item
        """
        item = self.data[idx]
        
        # Combine prompt and explanation
        text = item['prompt'] + item['explanation'] + "</s>"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        
        # Create labels (for causal LM training)
        encoded['labels'] = encoded['input_ids'].clone()
        
        # For training, mask the prompt part in loss calculation
        if self.is_train:
            prompt_encoded = self.tokenizer(
                item['prompt'],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_len = (prompt_encoded['input_ids'].squeeze(0) != self.tokenizer.pad_token_id).sum()
            
            # Set prompt tokens to -100 in labels (ignored in loss)
            encoded['labels'][:prompt_len] = -100
        
        return encoded


def create_dataloaders(
    train_data_path: str,
    val_data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TradingTextDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=True
    )
    
    val_dataset = TradingTextDataset(
        data_path=val_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=False
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader 