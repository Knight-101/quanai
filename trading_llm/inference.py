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
    
    def predict(self, observation: Union[Dict[str, np.ndarray], np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict action using RL model
        
        Args:
            observation: Observation dictionary or array
            
        Returns:
            Tuple of (action, extra_data)
        """
        try:
            # Convert numpy array to tensor if needed
            from gymnasium import spaces
            
            # Determine device that the model is using
            model_device = next(self.model.policy.parameters()).device
            logger.info(f"Model is on device: {model_device}")
            
            if isinstance(observation, np.ndarray) and isinstance(self.model.observation_space, spaces.Box):
                # Convert numpy array to torch tensor and move to correct device
                observation = torch.tensor(observation, dtype=torch.float32).to(model_device)
            elif isinstance(observation, dict) and isinstance(self.model.observation_space, spaces.Dict):
                # Convert all numpy arrays in the dict to torch tensors on correct device
                for key, value in observation.items():
                    if isinstance(value, np.ndarray):
                        observation[key] = torch.tensor(value, dtype=torch.float32).to(model_device)
            
            # Use no_grad to save memory
            with torch.no_grad():
                action, _states = self.model.predict(observation, deterministic=False)
            
            # Move action to CPU before returning
            if isinstance(action, torch.Tensor):
                action = action.cpu()  # Move to CPU but keep as tensor
            
            # Extract action probabilities if available
            extra_data = {}
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "get_distribution"):
                distribution = self.model.policy.get_distribution(observation)
                if hasattr(distribution, "mean") and hasattr(distribution, "stddev"):
                    # Move tensors to CPU
                    mean = distribution.mean.detach().cpu()
                    stddev = distribution.stddev.detach().cpu()
                    extra_data["action_mean"] = mean
                    extra_data["action_stddev"] = stddev
            
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
            
            # Ensure any cuda tensors are moved to CPU
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
            # Process any tensors in extra_data
            if extra_data:
                for key, value in extra_data.items():
                    if isinstance(value, torch.Tensor):
                        extra_data[key] = value.cpu().numpy()
            
            # Prepare market context
            market_context = self.rl_state_extractor.prepare_market_context(observation, raw_ohlcv)
            
            # Calculate indicators if raw_ohlcv data provided
            indicators = None
            if raw_ohlcv is not None:
                indicators = self.rl_state_extractor.indicator_processor.calculate_indicators(raw_ohlcv)
            
            # Ensure action is a scalar
            action_value = float(action) if isinstance(action, (np.ndarray, list)) else float(action)
            
            # Format prompt for LLM
            prompt = self.llm.format_prompt(
                market_context=market_context,
                action=action_value,
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

    def explain_trading_decisions(
        self,
        market_data_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_samples: int = 20  # Limit number of samples to prevent memory issues
    ) -> List[Dict[str, Any]]:
        """
        Generate trading decision explanations from market data file
        
        Args:
            market_data_path: Path to the market data file
            max_tokens: Maximum tokens for explanation
            temperature: Temperature for generation
            max_samples: Maximum number of samples to process (to prevent memory issues)
            
        Returns:
            List of dictionaries with predictions and explanations
        """
        try:
            # Load market data
            if market_data_path.endswith('.csv'):
                market_data = pd.read_csv(market_data_path)
            elif market_data_path.endswith('.parquet'):
                market_data = pd.read_parquet(market_data_path)
            else:
                raise ValueError(f"Unsupported market data format: {market_data_path}")
                
            # Convert column names to lowercase
            market_data.columns = market_data.columns.str.lower()
            
            # Prepare market data using same approach as training
            from gymnasium import spaces
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Check required columns
            for col in required_cols:
                if col not in market_data.columns:
                    raise ValueError(f"Required column {col} not found in market data")
            
            # Prepare and normalize data similar to extract_trading_signals
            data = market_data.copy()
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Normalize data (same as in extract_trading_signals)
            for col in required_cols:
                if col == 'volume':
                    # Log transform volume
                    data[col] = np.log(data[col] + 1)
                
                # Standardize
                mean, std = data[col].mean(), data[col].std()
                data[col] = (data[col] - mean) / (std + 1e-8)
            
            # Detect observation space type
            is_box_space = isinstance(self.rl_state_extractor.model.observation_space, spaces.Box)
            
            # For Box space, get the exact expected shape
            if is_box_space:
                expected_shape = self.rl_state_extractor.model.observation_space.shape
                expected_size = expected_shape[0] if expected_shape else 78  # Default to 78 if shape is not available
                
                # Calculate exactly how many rows we need for the expected size
                rows_needed = expected_size // len(required_cols)
                if expected_size % len(required_cols) != 0:
                    rows_needed += 1  # Round up if there's a remainder
                
                logger.info(f"Model expects observation shape {expected_shape}, using {rows_needed} rows")
            else:
                # For Dict space, use default window size
                rows_needed = 30
            
            # Generate observations from data
            observations = []
            indices = []
            
            # Process with stride to limit number of explanations
            stride = max(30, len(data) // max_samples)  # Ensure we don't process too many samples
            
            # Determine model device for tensor creation
            import torch
            model_device = next(self.rl_state_extractor.model.policy.parameters()).device
            
            # Create observations with appropriate format
            for i in range(rows_needed, len(data), stride):
                # Get window of data, using exactly the number of rows needed for the expected shape
                window = data.iloc[i-rows_needed:i]
                
                # Create observation based on space type
                if is_box_space:
                    # Flatten and ensure exact shape match
                    flattened = window[required_cols].values.flatten()
                    
                    # Adjust size if needed to match exactly what model expects
                    if len(flattened) != expected_size:
                        logger.info(f"Adjusting observation from {len(flattened)} to {expected_size} elements")
                        if len(flattened) > expected_size:
                            # Truncate if too large
                            flattened = flattened[:expected_size]
                        else:
                            # Pad with zeros if too small
                            padding = np.zeros(expected_size - len(flattened))
                            flattened = np.concatenate([flattened, padding])
                    
                    # Create tensor directly on correct device
                    obs = torch.tensor(flattened, dtype=torch.float32).to(model_device)
                else:
                    # Use dict for Dict spaces
                    obs = {
                        "market": torch.tensor(window[required_cols].values, dtype=torch.float32).to(model_device)
                    }
                
                observations.append(obs)
                indices.append(i)
                
                # Limit number of samples to prevent memory issues
                if len(observations) >= max_samples:
                    logger.info(f"Limiting to {max_samples} samples to prevent memory issues")
                    break
            
            # Generate explanations for observations
            explanations = []
            
            for i, (obs, idx) in enumerate(zip(observations, indices)):
                try:
                    # Predict action using RL model
                    with torch.no_grad():  # Use no_grad to reduce memory usage
                        action, extra_data = self.rl_state_extractor.predict(obs)
                    
                    # Move tensors to CPU and convert to numpy to free GPU memory
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                        
                    # Also ensure any values in extra_data are moved to CPU
                    if extra_data and isinstance(extra_data, dict):
                        for key, value in extra_data.items():
                            if isinstance(value, torch.Tensor):
                                extra_data[key] = value.cpu().numpy()
                    
                    # Get original (non-normalized) window for context
                    window_start = max(0, idx - rows_needed)
                    raw_ohlcv = market_data.iloc[window_start:idx]
                    
                    # Create market context using original data
                    temp_obs = {"market": raw_ohlcv[required_cols].values}
                    market_context = self.rl_state_extractor.prepare_market_context(temp_obs, raw_ohlcv)
                    
                    # Calculate indicators on raw data
                    indicators = self.rl_state_extractor.indicator_processor.calculate_indicators(raw_ohlcv)
                    
                    # Ensure action is a scalar for prompt formatting
                    action_value = float(action) if isinstance(action, (np.ndarray, list)) else float(action)
                    
                    # Format prompt
                    prompt = self.llm.format_prompt(
                        market_context=market_context,
                        action=action_value,
                        indicators=indicators
                    )
                    
                    # Generate explanation
                    expl_text = self.llm.generate_explanation(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    # Get timestamp for the explanation
                    timestamp = data.index[idx] if hasattr(data.index, 'values') and not data.index.equals(pd.RangeIndex(len(data))) else idx
                    
                    # Ensure action is a float for action_text formatting
                    action_float = float(action_value) if isinstance(action_value, (np.ndarray, list)) else float(action_value)
                    
                    # Format action text
                    action_text = "BUY" if action_float > 0.2 else ("SELL" if action_float < -0.2 else "HOLD")
                    
                    # Format explanation
                    explanation = f"Timestamp: {timestamp}\n"
                    explanation += f"Action: {action_text} (score: {action_float:.4f})\n"
                    explanation += f"Explanation: {expl_text}\n"
                    
                    explanations.append(explanation)
                    
                    # Free memory
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    # Handle individual observation errors
                    logger.error(f"Error processing observation {i}: {str(e)}")
                    timestamp = data.index[idx] if hasattr(data.index, 'values') and not data.index.equals(pd.RangeIndex(len(data))) else idx
                    explanations.append(f"Timestamp: {timestamp}\nAction: HOLD (score: 0.0000)\nExplanation: Error: {str(e)}\n")
                    
                    # Free memory on error
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error in explain_trading_decisions: {str(e)}")
            return [f"Error processing market data: {str(e)}"]


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
        
        # Format support/resistance and patterns for display
        support_levels = indicators['support_resistance'].get('support', ['N/A'])[0] if indicators['support_resistance'].get('support') else 'N/A'
        resistance_levels = indicators['support_resistance'].get('resistance', ['N/A'])[0] if indicators['support_resistance'].get('resistance') else 'N/A'
        patterns = ', '.join(indicators['patterns']) if indicators['patterns'] else 'None detected'
        
        # Try a much simpler prompt for testing
        simple_prompt = f"""<s>[INST]
Write a market analysis for {symbol} at price ${market_context['price']:.2f} with RSI {indicators['rsi']:.2f}.
[/INST]"""

        # Generate commentary with simple prompt and safer parameters
        logger.info(f"Generating commentary for {symbol} with temperature {temperature}")
        try:
            # Try simple prompt first
            logger.info("Attempting generation with simple prompt")
            commentary = self.llm.generate_explanation(
                prompt=simple_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,  # Higher top_p
                repetition_penalty=1.03  # Lower penalty
            )
            
            if not commentary or len(commentary) < 50:
                logger.warning("Simple prompt failed, trying full prompt with adjusted parameters")
                
                # Build full prompt with clear instructions
                full_prompt = f"""<s>[INST]
You are a professional market analyst with expertise in cryptocurrency markets. Generate a detailed market commentary for {symbol} based on the following data:

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
- Support level: {support_levels}
- Resistance level: {resistance_levels}

Write a concise market analysis.
[/INST]"""
                
                # Try with different parameters
                commentary = self.llm.generate_explanation(
                    prompt=full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.9,  # Higher temperature for more creativity
                    top_p=0.92,
                    repetition_penalty=1.02  # Lower penalty
                )
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            commentary = ""
        
        # Check if commentary is still empty or very short
        if not commentary or len(commentary) < 50:
            logger.warning(f"Still generated empty or very short commentary for {symbol}, using fallback text")
            commentary = f"""Analysis for {symbol}:

Current price is ${market_context['price']:.2f} with a {market_context['price_change']:.2f}% daily change. 
The RSI is at {indicators['rsi']:.2f}, suggesting {'overbought conditions' if indicators['rsi'] > 70 else 'oversold conditions' if indicators['rsi'] < 30 else 'neutral momentum'}.
The current trend appears to be {indicators['trend'].lower()} with {indicators['volatility'].lower()} volatility.

Key levels to watch: Support at {support_levels}, resistance at {resistance_levels}."""
        
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
        
        # Overall market direction
        market_direction = "bullish" if avg_change > 1.0 else "bearish" if avg_change < -1.0 else "neutral"
        
        # Build prompt with clear structure and instructions
        prompt = f"""<s>[INST]
You are a professional market analyst specializing in cryptocurrency markets. Generate a comprehensive market summary based on the following data:

Market Overview:
- Number of markets analyzed: {len(markets_data)}
- Average price change: {avg_change:.2f}%
- Overall market direction: {market_direction}
- Average RSI: {avg_rsi:.1f}
- Top gainers: {gainers_text if gainers else "None"}
- Top losers: {losers_text if losers else "None"}

Individual Markets:
{markets_text}

Your task:
1. Begin with an executive summary of the overall market conditions
2. Highlight the most significant market movements and explain possible reasons
3. Identify any market-wide patterns or correlations
4. Focus on standout assets (both positive and negative performers)
5. Conclude with an outlook based on the technical data provided

Write a professional, data-driven market summary that would be valuable for traders and investors. Be specific and reference the data provided.
[/INST]"""
        
        # Generate summary
        logger.info(f"Generating market summary for {len(markets_data)} symbols with temperature {temperature}")
        summary = self.llm.generate_explanation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Check if summary is empty or very short
        if not summary or len(summary) < 100:
            logger.warning("Generated empty or very short market summary, using fallback text")
            summary = f"""Market Summary:

The overall market shows an average change of {avg_change:.2f}%, with an average RSI of {avg_rsi:.1f}, indicating a generally {market_direction} sentiment.

Notable performers:
- Gainers: {gainers_text if gainers else "None above 2%"}
- Losers: {losers_text if losers else "None below -2%"}

The market currently displays {market_direction} characteristics with most assets showing {'positive' if avg_change > 0 else 'negative'} price action.
"""
        
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
    import torch
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
    
    # Determine model device
    model_device = next(rl_model.policy.parameters()).device
    logger.info(f"Model is on device: {model_device}")
    
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
                "market": torch.tensor(window[required_cols].values, dtype=torch.float32).to(model_device),
            }
        else:
            # Otherwise, use a flat array
            obs = torch.tensor(window[required_cols].values.flatten(), dtype=torch.float32).to(model_device)
        
        observations.append(obs)
        timestamps.append(data.index[i] if not data.index.equals(pd.RangeIndex(len(data))) else i)
    
    # Get actions from model
    actions = []
    
    # Use smaller batches to prevent memory issues
    batch_size = 32
    
    for i in range(0, len(observations), batch_size):
        batch_obs = observations[i:i+batch_size]
        batch_actions = []
        
        for obs in batch_obs:
            with torch.no_grad():  # Use no_grad to reduce memory usage
                action, _ = rl_model.predict(obs, deterministic=False)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                batch_actions.append(float(action))
                
            # Clear GPU memory after each prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        actions.extend(batch_actions)
    
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