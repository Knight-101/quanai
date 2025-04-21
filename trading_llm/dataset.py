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
        output_dir: str,
        template_path: Optional[str] = None
    ):
        """
        Initialize the dataset generator
        
        Args:
            rl_model_path: Path to the trained RL model
            market_data_path: Path to market data
            output_dir: Directory to save generated dataset
            template_path: Path to explanation templates
        """
        self.rl_model_path = rl_model_path
        self.market_data_path = market_data_path
        self.output_dir = output_dir
        self.template_path = template_path
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Load RL model
        self.rl_model = self._load_rl_model()
        
        # Load templates if provided
        self.templates = self._load_templates()
        
        # Initialize technical indicator processor
        self.indicator_processor = TechnicalIndicatorProcessor()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
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
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Load explanation templates
        
        Returns:
            Dictionary of templates
        """
        templates = {
            "uptrend": "The market is in an uptrend with {price_change_5d:.2f}% price increase over the past 5 days. {additional_context}",
            "downtrend": "The market is in a downtrend with {price_change_5d:.2f}% price decrease over the past 5 days. {additional_context}",
            "overbought": "Technical indicators suggest the market is overbought with RSI at {rsi:.2f}. {additional_context}",
            "oversold": "Technical indicators suggest the market is oversold with RSI at {rsi:.2f}. {additional_context}",
            "support": "The price is testing a support level at {support}. {additional_context}",
            "resistance": "The price is testing a resistance level at {resistance}. {additional_context}",
            "bullish_crossover": "A bullish signal has formed with the {crossover_type} crossing upward. {additional_context}",
            "bearish_crossover": "A bearish signal has formed with the {crossover_type} crossing downward. {additional_context}",
            "high_volatility": "The market is showing high volatility with ATR at {atr:.2f}. {additional_context}",
            "low_volatility": "The market is showing low volatility with ATR at {atr:.2f}. {additional_context}",
            "breakout": "A breakout is occurring with price moving above {breakout_level}. {additional_context}",
            "breakdown": "A breakdown is occurring with price moving below {breakdown_level}. {additional_context}",
            "consolidation": "The market is in a consolidation phase between {support} and {resistance}. {additional_context}",
            "reversal": "A potential reversal pattern has formed with {pattern}. {additional_context}",
        }
        
        # Override with custom templates if provided
        if self.template_path:
            try:
                with open(self.template_path, 'r') as f:
                    custom_templates = json.load(f)
                templates.update(custom_templates)
                logger.info(f"Loaded {len(custom_templates)} custom templates")
            except Exception as e:
                logger.warning(f"Error loading custom templates: {str(e)}")
        
        return templates
    
    def generate_dataset(
        self,
        num_samples: int = 10000,
        window_size: int = 100,
        val_split: float = 0.1,
        min_action_threshold: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate dataset for training and validation
        
        Args:
            num_samples: Number of samples to generate
            window_size: Size of the market data window
            val_split: Fraction of data to use for validation
            min_action_threshold: Minimum absolute action value to consider
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        logger.info(f"Generating dataset with {num_samples} samples")
        
        all_samples = []
        skipped_neutral = 0
        
        # Ensure we have enough data
        if len(self.market_data) < window_size + 1:
            raise ValueError(f"Not enough market data. Have {len(self.market_data)} rows, need at least {window_size + 1}")
        
        pbar = tqdm(total=num_samples, desc="Generating samples")
        
        while len(all_samples) < num_samples:
            # Sample a random window
            max_idx = len(self.market_data) - window_size
            start_idx = np.random.randint(0, max_idx)
            end_idx = start_idx + window_size
            
            window = self.market_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Prepare observation for RL model
            obs = self._prepare_observation(window)
            
            # Get RL model action
            action, _states = self.rl_model.predict(obs, deterministic=True)
            
            # Skip if action is too close to neutral
            if abs(action) < min_action_threshold:
                skipped_neutral += 1
                continue
            
            # Calculate technical indicators
            indicators = self.indicator_processor.calculate_indicators(window)
            
            # Generate prompt and explanation
            prompt, explanation = self._generate_explanation(window, action, indicators)
            
            # Create sample
            sample = {
                "market_data": window.iloc[-5:].to_dict('records'),  # Save last 5 days
                "action": float(action),
                "indicators": indicators,
                "prompt": prompt,
                "explanation": explanation
            }
            
            all_samples.append(sample)
            pbar.update(1)
        
        pbar.close()
        
        logger.info(f"Generated {len(all_samples)} samples (skipped {skipped_neutral} neutral actions)")
        
        # Split into training and validation
        random.shuffle(all_samples)
        val_size = int(num_samples * val_split)
        train_samples = all_samples[val_size:]
        val_samples = all_samples[:val_size]
        
        # Save samples
        train_path = os.path.join(self.output_dir, "train_data.json")
        val_path = os.path.join(self.output_dir, "val_data.json")
        
        with open(train_path, 'w') as f:
            json.dump(train_samples, f, indent=2)
        
        with open(val_path, 'w') as f:
            json.dump(val_samples, f, indent=2)
            
        logger.info(f"Saved {len(train_samples)} training samples to {train_path}")
        logger.info(f"Saved {len(val_samples)} validation samples to {val_path}")
        
        return train_samples, val_samples
    
    def _prepare_observation(self, window: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare observation for RL model
        
        Args:
            window: Market data window
            
        Returns:
            Observation dictionary
        """
        # This implementation depends on your RL model's observation space
        # Here's a simple implementation assuming the model takes OHLCV data
        
        # Extract OHLCV data
        ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize data
        for i in range(4):  # Normalize OHLC
            mean = ohlcv[:, i].mean()
            std = ohlcv[:, i].std()
            if std > 0:
                ohlcv[:, i] = (ohlcv[:, i] - mean) / std
        
        # Normalize volume separately
        vol_mean = ohlcv[:, 4].mean()
        vol_std = ohlcv[:, 4].std()
        if vol_std > 0:
            ohlcv[:, 4] = (ohlcv[:, 4] - vol_mean) / vol_std
        
        # Create observation dictionary
        # Adapt this to match your RL model's expected input format
        obs = {
            "market": ohlcv,
        }
        
        return obs
    
    def _generate_explanation(
        self,
        window: pd.DataFrame,
        action: float,
        indicators: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generate prompt and explanation for a sample
        
        Args:
            window: Market data window
            action: RL model action
            indicators: Technical indicators
            
        Returns:
            Tuple of (prompt, explanation)
        """
        # Create market context
        market_context = {
            "price": indicators['price'],
            "price_change": indicators['price_change_1d'],
            "volume": window['volume'].iloc[-1],
            "volatility": indicators['atr']
        }
        
        # Format prompt
        prompt = self._format_prompt(market_context, action, indicators)
        
        # Generate explanation
        explanation = self._create_explanation(action, indicators)
        
        return prompt, explanation
    
    def _format_prompt(
        self,
        market_context: Dict[str, Any],
        action: float,
        indicators: Dict[str, Any]
    ) -> str:
        """
        Format prompt for model input
        
        Args:
            market_context: Market context dictionary
            action: RL model action
            indicators: Technical indicators
            
        Returns:
            Formatted prompt
        """
        # Determine action description
        if action > 0.6:
            action_desc = "enter a strong long position"
        elif action > 0.2:
            action_desc = "enter a moderate long position"
        elif action > 0:
            action_desc = "enter a small long position"
        elif action < -0.6:
            action_desc = "enter a strong short position"
        elif action < -0.2:
            action_desc = "enter a moderate short position"
        elif action < 0:
            action_desc = "enter a small short position"
        else:
            action_desc = "maintain a neutral position"
        
        # Format indicators
        indicators_text = "Technical indicators:\n"
        key_indicators = ['rsi', 'macd', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'bb_width', 'atr']
        for name in key_indicators:
            if name in indicators and isinstance(indicators[name], (int, float)):
                indicators_text += f"- {name}: {indicators[name]:.2f}\n"
        
        # Add trend and patterns
        indicators_text += f"- trend: {indicators.get('trend', 'unknown')}\n"
        
        patterns = indicators.get('patterns', [])
        if patterns:
            indicators_text += f"- patterns: {', '.join(patterns)}\n"
        
        # Create the prompt
        prompt = f"""<s>[INST] I'm analyzing the market for trading opportunities.

Here's the current market situation:
- Price: {market_context['price']:.2f}
- Recent price change: {market_context['price_change']:.2f}%
- Volume: {market_context['volume']:.2f}
- Market volatility: {market_context['volatility']:.2f}
{indicators_text}

The trading algorithm decided to {action_desc}.

Can you explain why this might be a good decision given the current market conditions? Give a concise explanation based on technical analysis and price action. [/INST]
"""
        return prompt
    
    def _create_explanation(self, action: float, indicators: Dict[str, Any]) -> str:
        """
        Create explanation text
        
        Args:
            action: RL model action
            indicators: Technical indicators
            
        Returns:
            Explanation text
        """
        # Determine the market condition based on indicators
        conditions = []
        
        # Trend condition
        if indicators['trend'] in ['strong_uptrend', 'uptrend']:
            conditions.append('uptrend')
        elif indicators['trend'] in ['strong_downtrend', 'downtrend']:
            conditions.append('downtrend')
        
        # RSI condition
        if indicators['rsi'] > 70:
            conditions.append('overbought')
        elif indicators['rsi'] < 30:
            conditions.append('oversold')
        
        # Support/Resistance
        support_resistance = indicators.get('support_resistance', {})
        support = support_resistance.get('support', [])
        resistance = support_resistance.get('resistance', [])
        current_price = support_resistance.get('current_price', indicators['price'])
        
        if support and min(support) > 0 and current_price < min(support) * 1.03:
            conditions.append('support')
        if resistance and current_price > max(resistance) * 0.97:
            conditions.append('resistance')
        
        # MACD crossover
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_diff'] > 0:
            conditions.append('bullish_crossover')
        elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_diff'] < 0:
            conditions.append('bearish_crossover')
        
        # Volatility
        if indicators['atr'] > 0.02 * indicators['price']:
            conditions.append('high_volatility')
        elif indicators['atr'] < 0.005 * indicators['price']:
            conditions.append('low_volatility')
        
        # No clear condition, use price action
        if not conditions:
            if action > 0:
                conditions.append('bullish_crossover')
            elif action < 0:
                conditions.append('bearish_crossover')
        
        # Select a condition that matches the action direction
        matching_conditions = []
        
        if action > 0:  # Long position
            bullish_conditions = ['uptrend', 'oversold', 'support', 'bullish_crossover', 'breakout']
            matching_conditions = [c for c in conditions if c in bullish_conditions]
        elif action < 0:  # Short position
            bearish_conditions = ['downtrend', 'overbought', 'resistance', 'bearish_crossover', 'breakdown']
            matching_conditions = [c for c in conditions if c in bearish_conditions]
        
        # If no matching condition, use any condition
        if not matching_conditions and conditions:
            matching_conditions = [conditions[0]]
        elif not matching_conditions:
            matching_conditions = ['consolidation']  # Default
        
        # Select a template based on condition
        selected_condition = random.choice(matching_conditions)
        template = self.templates.get(selected_condition, self.templates['uptrend'])
        
        # Format template with indicator values
        additional_context = ""
        patterns = indicators.get('patterns', [])
        
        if patterns:
            additional_context += f"There is a {', '.join(patterns)} pattern forming. "
        
        if 'macd' in indicators and 'macd_signal' in indicators:
            if indicators['macd'] > indicators['macd_signal']:
                additional_context += "MACD is above the signal line, suggesting bullish momentum. "
            else:
                additional_context += "MACD is below the signal line, suggesting bearish momentum. "
        
        # Format the template with indicator values
        context_vars = {
            "price_change_5d": indicators.get('price_change_5d', 0),
            "rsi": indicators.get('rsi', 50),
            "atr": indicators.get('atr', 0),
            "support": str(support[0]) if support else "N/A",
            "resistance": str(resistance[0]) if resistance else "N/A",
            "crossover_type": "MACD",
            "breakout_level": str(resistance[0]) if resistance else "recent high",
            "breakdown_level": str(support[0]) if support else "recent low",
            "pattern": patterns[0] if patterns else "price action",
            "additional_context": additional_context
        }
        
        try:
            explanation = template.format(**context_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            explanation = template.format(additional_context=additional_context)
        
        # Add action justification
        if action > 0.6:
            explanation += "\n\nThis strong bullish signal justifies entering a significant long position to capitalize on the expected upward movement."
        elif action > 0.2:
            explanation += "\n\nThese indicators suggest a moderate bullish bias, making a long position with reasonable size appropriate."
        elif action > 0:
            explanation += "\n\nWhile not overwhelmingly bullish, the technical setup supports a small long position to test the market direction."
        elif action < -0.6:
            explanation += "\n\nThis strong bearish signal justifies entering a significant short position to capitalize on the expected downward movement."
        elif action < -0.2:
            explanation += "\n\nThese indicators suggest a moderate bearish bias, making a short position with reasonable size appropriate."
        elif action < 0:
            explanation += "\n\nWhile not overwhelmingly bearish, the technical setup supports a small short position to test the market direction."
        
        return explanation


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