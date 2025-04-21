"""
Inference module for combining RL trading with LLM explanations
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from stable_baselines3 import PPO
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from trading_llm.model import TradingLLM
from trading_llm.dataset import TechnicalIndicatorProcessor

logger = logging.getLogger(__name__)


class RLStateExtractor:
    """Extract state information from RL model and observations"""
    
    def __init__(self, rl_model_path: str):
        """
        Initialize the RL state extractor
        
        Args:
            rl_model_path: Path to the trained RL model
        """
        self.rl_model_path = rl_model_path
        self.model = self._load_model()
        self.indicator_processor = TechnicalIndicatorProcessor()
    
    def _load_model(self) -> PPO:
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
    
    def predict(self, observation: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict action using RL model
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Tuple of (action, extra_data)
        """
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Extract action probabilities if available
            extra_data = {}
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "get_distribution"):
                distribution = self.model.policy.get_distribution(observation)
                if hasattr(distribution, "mean") and hasattr(distribution, "stddev"):
                    extra_data["action_mean"] = distribution.mean.detach().cpu().numpy()
                    extra_data["action_stddev"] = distribution.stddev.detach().cpu().numpy()
            
            return action, extra_data
        except Exception as e:
            logger.error(f"Error predicting with RL model: {str(e)}")
            raise
    
    def extract_features(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Extract features from observation
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract market features
        if "market" in observation:
            market_data = observation["market"]
            
            if isinstance(market_data, np.ndarray) and len(market_data.shape) >= 2:
                # Last row of OHLCV data
                last_row = market_data[-1]
                
                # Store basic price info
                if market_data.shape[1] >= 4:  # Has OHLC
                    features["price"] = last_row[3]  # Close price
                    features["price_change"] = (market_data[-1, 3] / market_data[-2, 3] - 1) * 100 if market_data.shape[0] > 1 else 0
                
                # Store volume info if available
                if market_data.shape[1] >= 5:  # Has volume
                    features["volume"] = last_row[4]
        
        return features
    
    def prepare_market_context(
        self,
        observation: Dict[str, np.ndarray],
        raw_ohlcv: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Prepare market context for explanation
        
        Args:
            observation: Observation dictionary
            raw_ohlcv: Optional dataframe with OHLCV data (for indicators)
            
        Returns:
            Dictionary with market context
        """
        # Start with basic features from observation
        context = self.extract_features(observation)
        
        # If raw OHLCV data provided, calculate technical indicators
        if raw_ohlcv is not None:
            indicators = self.indicator_processor.calculate_indicators(raw_ohlcv)
            context.update(indicators)
        
        return context


class RLLMExplainer:
    """Combined RL model with LLM for explaining trading decisions"""
    
    def __init__(
        self,
        rl_model_path: str,
        llm_model_path: str,
        llm_base_model: Optional[str] = None
    ):
        """
        Initialize the explainer
        
        Args:
            rl_model_path: Path to the trained RL model
            llm_model_path: Path to the fine-tuned LLM model
            llm_base_model: Optional base model name if llm_model_path contains LoRA adapters
        """
        self.rl_state_extractor = RLStateExtractor(rl_model_path)
        
        # Load LLM model
        logger.info(f"Loading LLM model from {llm_model_path}")
        self.llm = TradingLLM.load_from_pretrained(
            model_dir=llm_model_path,
            base_model=llm_base_model
        )
        logger.info("LLM model loaded successfully")
    
    def explain_decision(
        self,
        observation: Dict[str, np.ndarray],
        raw_ohlcv: Optional[pd.DataFrame] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a prediction with explanation
        
        Args:
            observation: Observation dictionary
            raw_ohlcv: Optional dataframe with OHLCV data (for indicators)
            max_tokens: Maximum tokens for explanation
            temperature: Temperature for generation
            
        Returns:
            Dictionary with prediction and explanation
        """
        try:
            # Predict action using RL model
            action, extra_data = self.rl_state_extractor.predict(observation)
            
            # Prepare market context
            market_context = self.rl_state_extractor.prepare_market_context(observation, raw_ohlcv)
            
            # Calculate indicators if raw_ohlcv data provided
            indicators = None
            if raw_ohlcv is not None:
                indicators = self.rl_state_extractor.indicator_processor.calculate_indicators(raw_ohlcv)
            
            # Format prompt for LLM
            prompt = self.llm.format_prompt(
                market_context=market_context,
                action=action,
                indicators=indicators
            )
            
            # Generate explanation
            explanation = self.llm.generate_explanation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            # Package result
            result = {
                "action": float(action),
                "explanation": explanation,
                "market_context": market_context
            }
            
            # Add extra data if available
            if extra_data:
                result.update(extra_data)
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "action": 0.0,
                "explanation": f"Error generating explanation: {str(e)}",
                "market_context": {}
            }
    
    def batch_explain(
        self,
        observations: List[Dict[str, np.ndarray]],
        raw_ohlcvs: Optional[List[pd.DataFrame]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions with explanations for a batch of observations
        
        Args:
            observations: List of observation dictionaries
            raw_ohlcvs: Optional list of dataframes with OHLCV data
            max_tokens: Maximum tokens for explanation
            temperature: Temperature for generation
            
        Returns:
            List of dictionaries with predictions and explanations
        """
        results = []
        
        # Process each observation
        for i, observation in enumerate(observations):
            raw_ohlcv = raw_ohlcvs[i] if raw_ohlcvs and i < len(raw_ohlcvs) else None
            result = self.explain_decision(
                observation=observation,
                raw_ohlcv=raw_ohlcv,
                max_tokens=max_tokens,
                temperature=temperature
            )
            results.append(result)
        
        return results


class MarketCommentaryGenerator:
    """Generate market commentary from historical data"""
    
    def __init__(
        self,
        llm_model_path: str,
        llm_base_model: Optional[str] = None
    ):
        """
        Initialize the market commentary generator
        
        Args:
            llm_model_path: Path to the fine-tuned LLM model
            llm_base_model: Optional base model name if llm_model_path contains LoRA adapters
        """
        # Load LLM model
        logger.info(f"Loading LLM model from {llm_model_path}")
        self.llm = TradingLLM.load_from_pretrained(
            model_dir=llm_model_path,
            base_model=llm_base_model
        )
        logger.info("LLM model loaded successfully")
        
        # Initialize indicator processor
        self.indicator_processor = TechnicalIndicatorProcessor()
    
    def generate_daily_commentary(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        lookback_days: int = 30,
        max_tokens: int = 750,
        temperature: float = 0.75
    ) -> str:
        """
        Generate daily market commentary
        
        Args:
            symbol: Market symbol
            ohlcv_data: DataFrame with OHLCV data
            lookback_days: Number of days to analyze
            max_tokens: Maximum tokens for commentary
            temperature: Temperature for generation
            
        Returns:
            Generated market commentary
        """
        # Ensure sufficient data
        if len(ohlcv_data) < lookback_days:
            return f"Insufficient data for {symbol}. Need at least {lookback_days} days, got {len(ohlcv_data)}."
        
        # Get relevant window of data
        window = ohlcv_data.iloc[-lookback_days:].copy()
        
        # Calculate indicators
        indicators = self.indicator_processor.calculate_indicators(window)
        
        # Create market context
        market_context = {
            "symbol": symbol,
            "price": window['close'].iloc[-1],
            "price_change": (window['close'].iloc[-1] / window['close'].iloc[-2] - 1) * 100,
            "price_change_1w": (window['close'].iloc[-1] / window['close'].iloc[-6] - 1) * 100 if len(window) >= 6 else 0,
            "price_change_1m": (window['close'].iloc[-1] / window['close'].iloc[-22] - 1) * 100 if len(window) >= 22 else 0,
            "volume": window['volume'].iloc[-1],
            "avg_volume": window['volume'].iloc[-5:].mean()
        }
        
        # Extract key price points
        market_context["day_high"] = window['high'].iloc[-1]
        market_context["day_low"] = window['low'].iloc[-1]
        market_context["day_open"] = window['open'].iloc[-1]
        market_context["day_close"] = window['close'].iloc[-1]
        
        # Build prompt
        prompt = f"""<s>[INST] Generate a professional market commentary for {symbol} based on the following data:

Price Information:
- Current price: {market_context['price']:.2f}
- Daily change: {market_context['price_change']:.2f}%
- Weekly change: {market_context['price_change_1w']:.2f}%
- Monthly change: {market_context['price_change_1m']:.2f}%
- Day high/low: {market_context['day_high']:.2f}/{market_context['day_low']:.2f}

Technical Indicators:
- RSI(14): {indicators['rsi']:.2f}
- MACD: {indicators['macd']:.4f}
- MACD Signal: {indicators['macd_signal']:.4f}
- EMA(12): {indicators['ema_12']:.2f}
- EMA(26): {indicators['ema_26']:.2f}
- Bollinger Bands Width: {indicators['bb_width']:.4f}
- ATR: {indicators['atr']:.4f}

Market Patterns and Trend:
- Current trend: {indicators['trend']}
- Volatility: {indicators['volatility']}
- Detected patterns: {', '.join(indicators['patterns']) if indicators['patterns'] else 'None'}
- Support levels: {indicators['support_resistance'].get('support', ['None'])[0] if indicators['support_resistance'].get('support') else 'None'}
- Resistance levels: {indicators['support_resistance'].get('resistance', ['None'])[0] if indicators['support_resistance'].get('resistance') else 'None'}

Write a detailed but concise market commentary that a professional trader would find useful. Include an analysis of price action, technical indicators, and potential future scenarios. [/INST]
"""
        
        # Generate commentary
        commentary = self.llm.generate_explanation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.15
        )
        
        return commentary
    
    def generate_market_summary(
        self,
        symbols: List[str],
        ohlcv_dict: Dict[str, pd.DataFrame],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate market summary for multiple symbols
        
        Args:
            symbols: List of market symbols
            ohlcv_dict: Dictionary mapping symbols to OHLCV dataframes
            max_tokens: Maximum tokens for summary
            temperature: Temperature for generation
            
        Returns:
            Generated market summary
        """
        # Build market context
        markets_data = []
        
        for symbol in symbols:
            if symbol not in ohlcv_dict:
                logger.warning(f"No data for {symbol}, skipping")
                continue
                
            ohlcv = ohlcv_dict[symbol]
            
            # Calculate percent change
            pct_change = (ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-2] - 1) * 100 if len(ohlcv) > 1 else 0
            
            # Calculate RSI
            rsi = RSIIndicator(close=ohlcv['close']).rsi().iloc[-1]
            
            # Calculate trend
            trend = self.indicator_processor._determine_trend(ohlcv)
            
            # Store market data
            markets_data.append({
                "symbol": symbol,
                "price": ohlcv['close'].iloc[-1],
                "change": pct_change,
                "rsi": rsi,
                "trend": trend
            })
        
        # Sort by absolute change
        markets_data.sort(key=lambda x: abs(x['change']), reverse=True)
        
        # Build market summary prompt
        markets_text = "\n".join([
            f"- {m['symbol']}: ${m['price']:.2f} ({m['change']:+.2f}%), RSI: {m['rsi']:.1f}, Trend: {m['trend']}"
            for m in markets_data
        ])
        
        # Calculate overall market stats
        avg_change = sum(m['change'] for m in markets_data) / len(markets_data) if markets_data else 0
        avg_rsi = sum(m['rsi'] for m in markets_data) / len(markets_data) if markets_data else 0
        
        # Identify noteworthy price movements
        gainers = [m for m in markets_data if m['change'] > 2.0]
        losers = [m for m in markets_data if m['change'] < -2.0]
        
        gainers_text = ", ".join([f"{m['symbol']} (+{m['change']:.1f}%)" for m in gainers[:3]])
        losers_text = ", ".join([f"{m['symbol']} ({m['change']:.1f}%)" for m in losers[:3]])
        
        # Build prompt
        prompt = f"""<s>[INST] Generate a comprehensive market summary based on the following data:

Market Overview:
- Number of markets: {len(markets_data)}
- Average price change: {avg_change:.2f}%
- Average RSI: {avg_rsi:.1f}
- Top gainers: {gainers_text if gainers else "None"}
- Top losers: {losers_text if losers else "None"}

Individual Markets:
{markets_text}

Write a comprehensive market summary that covers the overall market sentiment, noteworthy price movements, technical patterns, and potential trading opportunities. The summary should be professional, concise, and actionable. [/INST]
"""
        
        # Generate summary
        summary = self.llm.generate_explanation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        return summary


def extract_trading_signals(
    rl_model,
    market_data: pd.DataFrame,
    normalize_data: bool = True,
    action_threshold: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract trading signals and positions from the RL model for market data.
    
    Args:
        rl_model: Trained reinforcement learning model (from stable_baselines3)
        market_data: Pandas DataFrame with market data (OHLCV)
        normalize_data: Whether to normalize input data
        action_threshold: Threshold for considering an action significant
        
    Returns:
        Dictionary with trading signals for each timestamp
    """
    import numpy as np
    from gymnasium import spaces
    
    logger.info("Extracting trading signals from RL model")
    
    # Prepare market data
    data = market_data.copy()
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')
    
    # Make sure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column {col} not found in market data")
    
    # Normalize data if requested
    if normalize_data:
        for col in required_cols:
            if col == 'volume':
                # Log transform volume
                data[col] = np.log(data[col] + 1)
            
            # Standardize
            mean, std = data[col].mean(), data[col].std()
            data[col] = (data[col] - mean) / (std + 1e-8)
    
    # Extract observations from market data
    observations = []
    timestamps = []
    
    for i in range(len(data) - 1):  # -1 because we need the next price for position calculation
        # Get relevant window
        window = data.iloc[max(0, i-30):i+1]  # Use up to 30 previous values
        
        # Create observation
        if isinstance(rl_model.observation_space, spaces.Dict):
            # If the model uses a Dict observation space, format accordingly
            obs = {
                "market": window[required_cols].values,
            }
        else:
            # Otherwise, use a flat array
            obs = window[required_cols].values.flatten()
        
        observations.append(obs)
        timestamps.append(data.index[i] if not data.index.equals(pd.RangeIndex(len(data))) else i)
    
    # Get actions from model
    actions = []
    for obs in observations:
        action, _ = rl_model.predict(obs, deterministic=True)
        actions.append(float(action))
    
    # Convert actions to trading signals
    signals = {}
    for i, (timestamp, action) in enumerate(zip(timestamps, actions)):
        # Calculate position (1 for long, -1 for short, 0 for neutral)
        if action > action_threshold:
            position = 1  # Long
        elif action < -action_threshold:
            position = -1  # Short
        else:
            position = 0  # Neutral
        
        # Calculate next return (for evaluation)
        next_close = data['close'].iloc[i+1]
        current_close = data['close'].iloc[i]
        next_return = (next_close - current_close) / current_close
        
        # Calculate confidence score (scale action to 0-1 range)
        action_max = max(abs(action), 1.0)  # Prevent division by zero
        confidence = (action + action_max) / (2 * action_max)
        
        # Format signal
        signal = {
            'action': 'buy' if action > action_threshold else ('sell' if action < -action_threshold else 'hold'),
            'action_value': float(action),
            'confidence': float(confidence),
            'position': position,
            'next_return': float(next_return),
        }
        
        signals[str(timestamp)] = signal
    
    logger.info(f"Extracted {len(signals)} trading signals")
    return signals 