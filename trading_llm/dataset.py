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
            rl_model_path: Path to trained RL model
            market_data_path: Path to market data (replacing data_path)
            output_path: Path to output dataset (deprecated, use output_dir instead)
            output_dir: Directory to save generated datasets
            lookback_window: Number of periods to include in each window
            samples_per_symbol: Number of samples to generate per symbol
            template_path: Path to custom explanation templates file
        """
        self.rl_model_path = rl_model_path
        self.data_path = market_data_path  # Store internally as data_path for compatibility
        
        # Handle output directory (maintain backward compatibility)
        self.output_path = output_dir if output_dir is not None else output_path
        
        self.lookback_window = lookback_window
        self.samples_per_symbol = samples_per_symbol
        
        # Load templates
        self.templates = self._load_templates(template_path)
        
        logger.info(f"Initialized dataset generator with {len(self.templates)} templates")
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Load RL model
        self.rl_model = self._load_rl_model()
        
        # Initialize technical indicator processor
        self.indicator_processor = TechnicalIndicatorProcessor()
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
    
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
            logger.info(f"Loading market data from {self.data_path}")
            extension = os.path.splitext(self.data_path)[1].lower()
            
            if extension == '.csv':
                df = pd.read_csv(self.data_path)
            elif extension in ['.parquet', '.pq']:
                df = pd.read_parquet(self.data_path)
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
        # Handle parameter compatibility
        lookback = window_size if window_size is not None else lookback_window if lookback_window is not None else self.lookback_window
        policy_path = policy_model_path if policy_model_path is not None else self.rl_model_path
        
        logger.info(f"Generating dataset with {num_samples} samples per symbol using model: {policy_path}")
        
        # Load the market data
        all_data = []
        
        # Handle both directory of CSV files and single parquet/csv file
        if os.path.isdir(self.data_path):
            # Directory with multiple files
            for file in os.listdir(self.data_path):
                if file.endswith(".csv"):
                    symbol = file.split(".")[0]
                    file_path = os.path.join(self.data_path, file)
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
            extension = os.path.splitext(self.data_path)[1].lower()
            try:
                if extension == '.csv':
                    market_data = pd.read_csv(self.data_path)
                elif extension in ['.parquet', '.pq']:
                    market_data = pd.read_parquet(self.data_path)
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
                    symbol = os.path.basename(self.data_path).split('.')[0]
                    # Add symbol column for consistency
                    if 'symbol' not in market_data.columns:
                        market_data['symbol'] = symbol
                    logger.info(f"Processing single-symbol data: {symbol} with {len(market_data)} rows")
                    all_data.extend(self._process_symbol_data(
                        symbol, market_data, policy_path, env_config_path, 
                        lookback, num_samples, min_action_threshold
                    ))
            except Exception as e:
                logger.error(f"Error processing file {self.data_path}: {e}")
        
        if not all_data:
            logger.error("No data was generated. Check file paths and market data format.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        
        # Save datasets
        train_path = os.path.join(self.output_path, "train_data.json")
        val_path = os.path.join(self.output_path, "val_data.json")
        
        train_df.to_json(train_path, orient='records')
        val_df.to_json(val_path, orient='records')
        
        logger.info(f"Dataset generation complete. Training: {len(train_df)} samples, Validation: {len(val_df)} samples")
        
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
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return []
            
            # Generate explanations for random time windows
            max_window = len(market_data) - lookback
            
            if max_window <= 0:
                logger.warning(f"Not enough data for symbol {symbol}, skipping. Need at least {lookback} rows, found {len(market_data)}")
                return []
            
            # Sample random windows for this symbol
            sample_count = min(num_samples, max_window)
            start_indices = random.sample(range(max_window), sample_count)
            
            successful_samples = 0
            for start_idx in start_indices:
                try:
                    # Get window of market data and create a copy to avoid SettingWithCopyWarning
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
                        
                        # Verify observation is a proper numeric array
                        if not isinstance(observation, np.ndarray) or observation.dtype == np.dtype('O'):
                            logger.error(f"Invalid observation array type: {type(observation)}, dtype: {observation.dtype}")
                            # Convert to float64 explicitly
                            observation = np.array(observation, dtype=np.float64)
                        
                        # Verify no NaN or Inf values
                        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        logger.error(f"Failed to prepare observation for {symbol} at index {start_idx}: {e}")
                        continue
                    
                    # Get model prediction with type safety
                    try:
                        action, _ = model.predict(observation, deterministic=True)
                        
                        # Verify action is a proper numeric array and convert if needed
                        if not isinstance(action, np.ndarray) or action.dtype == np.dtype('O'):
                            logger.warning(f"Invalid action array type: {type(action)}, dtype: {action.dtype}")
                            # Convert to float64 explicitly
                            action = np.array(action, dtype=np.float64)
                            
                        # Verify no NaN or Inf values
                        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    except Exception as e:
                        logger.error(f"Failed to predict action for {symbol} at index {start_idx}: {e}")
                        continue
                    
                    # Skip if action is too close to neutral
                    action_value = action[0] if isinstance(action, np.ndarray) and len(action) > 0 else 0.0
                    if abs(action_value) < min_action_threshold:
                        continue
                    
                    # Extract technical indicators for the last candle (safely)
                    tech_indicators = {}
                    skip_columns = set(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'index', 'symbol', 'asset'])
                    for col in window.columns:
                        if col not in skip_columns:
                            try:
                                value = window.iloc[-1][col]
                                # Ensure value is a simple numeric type
                                if isinstance(value, (int, float, np.number)):
                                    tech_indicators[col] = float(value)
                            except Exception:
                                pass
                    
                    # Generate explanation
                    try:
                        explanation = self._generate_explanation(
                            observation, 
                            action, 
                            tech_indicators
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate explanation for {symbol} at index {start_idx}: {e}")
                        explanation = f"The model suggests to {'buy' if action_value > 0 else 'sell'} based on current market conditions."
                    
                    # Create sample with formatted prompts
                    prompt = f"Given the market data for {symbol}, explain the trading decision: "
                    
                    # Timestamp extraction
                    if 'timestamp' in window.columns:
                        timestamp = window.iloc[-1]['timestamp']
                    else:
                        timestamp = f"{symbol}_{start_idx}_{start_idx + lookback}"
                    
                    # Convert arrays to lists safely
                    try:
                        observation_list = observation.tolist()
                    except (AttributeError, TypeError):
                        observation_list = [float(x) for x in observation]
                        
                    try:
                        action_list = action.tolist()
                    except (AttributeError, TypeError):
                        action_list = [float(x) for x in action]
                    
                    sample = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'observation': observation_list,
                        'action': action_list,
                        'explanation': explanation,
                        'prompt': prompt
                    }
                    
                    symbol_samples.append(sample)
                    successful_samples += 1
                    
                    # Periodically log progress
                    if successful_samples % 50 == 0:
                        logger.info(f"Generated {successful_samples} samples for {symbol} so far...")
                        
                except Exception as e:
                    logger.error(f"Error processing window for {symbol} at index {start_idx}: {e}")
            
            logger.info(f"Generated {len(symbol_samples)} samples for {symbol}")
            
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
        action: np.ndarray, 
        technical_indicators: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate a natural language explanation for a trading decision
        
        Args:
            observation: Flat observation array from the environment
            action: Action array from the RL policy (containing leverage values)
            technical_indicators: Dictionary of technical indicators
            
        Returns:
            Natural language explanation of the trading decision
        """
        try:
            # Determine action type based on leverage value
            # Assuming action is a leverage value where:
            # - Positive values = Long position
            # - Negative values = Short position
            # - Near zero values = Hold/no position
            action_value = action[0] if isinstance(action, np.ndarray) else action
            
            if action_value > 0.1:
                action_type = "buy"
            elif action_value < -0.1:
                action_type = "sell"
            else:
                action_type = "hold"
            
            # Format technical indicators for explanation
            tech_indicators_str = ""
            if technical_indicators:
                # Select just a few important indicators to include in the explanation
                important_indicators = {}
                indicator_preference = ['RSI_14', 'MACD', 'SMA_10', 'SMA_20', 'BB_upper', 'BB_lower']
                
                # First add preferred indicators if they exist
                for ind in indicator_preference:
                    if ind in technical_indicators:
                        important_indicators[ind] = technical_indicators[ind]
                
                # If we don't have enough, add some other ones
                if len(important_indicators) < 3:
                    for ind, val in technical_indicators.items():
                        if ind not in important_indicators and len(important_indicators) < 3:
                            important_indicators[ind] = val
                
                # Format selected indicators
                indicator_items = []
                for indicator, value in important_indicators.items():
                    # Format the indicator value (round floats to 2 decimal places)
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    indicator_items.append(f"{indicator}: {formatted_value}")
                
                tech_indicators_str = ", ".join(indicator_items)
            
            # Create explanation templates based on action type
            templates = {
                "buy": [
                    "Based on the recent price movement and {indicators}, I'm going long with a leverage of {leverage:.2f}x. The market seems to be in an uptrend.",
                    "The technical indicators {indicators} suggest bullish momentum. I'm taking a long position with {leverage:.2f}x leverage.",
                    "I'm entering a long position with {leverage:.2f}x leverage as the {indicators} indicate potential upside.",
                    "The market conditions look favorable for a long position. Using {leverage:.2f}x leverage based on {indicators}."
                ],
                "sell": [
                    "Based on the recent price movement and {indicators}, I'm going short with a leverage of {leverage:.2f}x. The market seems to be in a downtrend.",
                    "The technical indicators {indicators} suggest bearish momentum. I'm taking a short position with {leverage:.2f}x leverage.",
                    "I'm entering a short position with {leverage:.2f}x leverage as the {indicators} indicate potential downside.",
                    "The market conditions look unfavorable. Using {leverage:.2f}x leverage to short based on {indicators}."
                ],
                "hold": [
                    "Based on the current market conditions and {indicators}, I'm holding my position as there's no clear directional signal.",
                    "The technical indicators {indicators} are mixed. I'm maintaining a neutral position for now.",
                    "I'm staying neutral as the {indicators} don't provide a strong directional bias.",
                    "Market conditions don't favor a strong directional bet right now based on {indicators}. Maintaining a neutral stance."
                ]
            }
            
            # Select a random template for the action type
            template = random.choice(templates[action_type])
            
            # Format template with actual values
            explanation = template.format(
                leverage=abs(action_value),
                indicators=tech_indicators_str or "current market conditions"
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