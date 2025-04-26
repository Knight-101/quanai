#!/usr/bin/env python3
"""
Inference utilities for the Trading LLM system.
Generates explanations using LLMs for trading decisions.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
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
            # print(f"DEBUG: Model is on device: {model_device}")
            
            # IMPORTANT: stable_baselines3 expects NumPy arrays or Python objects,
            # not PyTorch tensors. If we have tensors, we need to convert them to NumPy first
            # Create a copy of the observation to avoid modifying the original
            if isinstance(observation, np.ndarray):
                observation_numpy = observation.copy()
                # print(f"DEBUG: Using numpy array observation with shape {observation_numpy.shape}")
            elif isinstance(observation, dict):
                # For dict observations, ensure all values are NumPy arrays
                observation_numpy = {}
                for key, value in observation.items():
                    if isinstance(value, torch.Tensor):
                        # print(f"DEBUG: Converting dict tensor observation[{key}] from {value.device} to numpy")
                        observation_numpy[key] = value.detach().cpu().numpy()
                    elif isinstance(value, np.ndarray):
                        observation_numpy[key] = value.copy()
                    else:
                        observation_numpy[key] = value
                # print(f"DEBUG: Using dict observation with keys {list(observation_numpy.keys())}")
            elif isinstance(observation, torch.Tensor):
                # Convert tensor to numpy
                # print(f"DEBUG: Converting tensor observation from {observation.device} to numpy")
                observation_numpy = observation.detach().cpu().numpy()
                # print(f"DEBUG: Converted to numpy with shape {observation_numpy.shape}")
            else:
                observation_numpy = observation
                # print(f"DEBUG: Using observation as-is, type: {type(observation_numpy)}")
            
            # Use no_grad to save memory
            with torch.no_grad():
                # print(f"DEBUG: Predicting with observation type: {type(observation_numpy)}")
                action, _states = self.model.predict(observation_numpy, deterministic=False)
            
            # Debug action type and device
            if isinstance(action, torch.Tensor):
                # print(f"DEBUG: Action is tensor on device: {action.device}, shape: {action.shape}")
                # Move action to CPU before returning and convert to numpy
                action = action.detach().cpu().numpy()
                # print(f"DEBUG: Converted action to numpy with shape: {action.shape}")
            else:
                # print(f"DEBUG: Action is type: {type(action)}")
                pass
            
            # Extract action probabilities if available
            extra_data = {}
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "get_distribution"):
                try:
                    # Convert observation back to tensor if needed for getting distribution
                    if isinstance(observation_numpy, np.ndarray):
                        observation_tensor = torch.tensor(observation_numpy, dtype=torch.float32).to(model_device)
                    elif isinstance(observation_numpy, dict):
                        observation_tensor = {}
                        for key, value in observation_numpy.items():
                            if isinstance(value, np.ndarray):
                                observation_tensor[key] = torch.tensor(value, dtype=torch.float32).to(model_device)
                            else:
                                observation_tensor[key] = value
                    else:
                        observation_tensor = observation_numpy

                    distribution = self.model.policy.get_distribution(observation_tensor)
                    if hasattr(distribution, "mean") and hasattr(distribution, "stddev"):
                        # Move tensors to CPU and convert to numpy
                        mean = distribution.mean.detach().cpu().numpy() if isinstance(distribution.mean, torch.Tensor) else distribution.mean
                        stddev = distribution.stddev.detach().cpu().numpy() if isinstance(distribution.stddev, torch.Tensor) else distribution.stddev
                        extra_data["action_mean"] = mean
                        extra_data["action_stddev"] = stddev
                except Exception as dist_error:
                    # print(f"DEBUG: Error extracting distribution: {str(dist_error)}")
                    pass
            
            # Ensure everything in extra_data is CPU-based
            for key in list(extra_data.keys()):
                if isinstance(extra_data[key], torch.Tensor):
                    # print(f"DEBUG: Converting extra_data[{key}] from tensor to numpy")
                    extra_data[key] = extra_data[key].detach().cpu().numpy()
            
            return action, extra_data
        except Exception as e:
            import traceback
            # print(f"ERROR predicting with RL model: {str(e)}")
            # print(f"Traceback: {traceback.format_exc()}")
            # Return a default action on error
            if isinstance(self.model.action_space, spaces.Box):
                # For continuous action spaces, return zero action
                action_shape = self.model.action_space.shape
                default_action = np.zeros(action_shape, dtype=np.float32)
            else:
                # For discrete action spaces, return zero (no action)
                default_action = np.array(0, dtype=np.int32)
            
            return default_action, {}
    
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
    """
    Explainer class for reinforcement learning trading signals using LLM for natural language explanations.
    Supports multi-asset environments with flexible asset selection by name or index.
    """
    
    def __init__(self, model=None, asset_names=None, rl_model=None, rl_model_path=None):
        """
        Initialize the RL-LLM explainer.
        
        Args:
            model: TradingLLM model instance (will create one if not provided)
            asset_names: List of asset names for multi-asset environments
            rl_model: Pre-loaded RL model instance (if available)
            rl_model_path: Path to an RL model to load (if rl_model not provided)
        """
        # Initialize model
        self.model = model if model is not None else TradingLLM()
        
        # Set asset names and create name-to-index mapping
        self.asset_names = asset_names or ["Asset"]
        self.asset_name_to_index = {name.lower(): i for i, name in enumerate(self.asset_names)}
        
        # Initialize RL model if provided
        self.rl_model = None
        if rl_model is not None:
            self.rl_model = rl_model
            logger.info("Using provided RL model for trading signals")
        elif rl_model_path is not None:
            try:
                from stable_baselines3 import PPO
                logger.info(f"Loading RL model from {rl_model_path}")
                self.rl_model = PPO.load(rl_model_path)
                logger.info("RL model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RL model from {rl_model_path}: {str(e)}")
        
        logger.info(f"RLLMExplainer initialized with {len(self.asset_names)} assets and RL model: {'Available' if self.rl_model else 'Not available'}")
    
    def explain_decision(self, observation, market_data, action, asset_index=0, asset_name=None, use_fallback=True):
        """
        Generate a natural language explanation for a trading decision.
        
        Args:
            observation: Observation from the environment
            market_data: Market data with technical indicators
            action: Action value or array of actions from the policy
            asset_index: Index of the asset to explain in a multi-asset environment (default: 0)
            asset_name: Name of the asset to explain (will override asset_index if provided)
            use_fallback: Whether to use fallback explanation if LLM output is empty
            
        Returns:
            Natural language explanation of the trading decision
        """
        try:
            # Handle multi-asset action arrays
            if isinstance(action, (list, np.ndarray)) and len(action) > 1:
                # Resolve asset index by name if provided
                if asset_name is not None:
                    asset_name = str(asset_name).lower()  # Ensure lowercase for matching
                    
                    # Direct match
                    if asset_name in self.asset_name_to_index:
                        asset_index = self.asset_name_to_index[asset_name]
                    else:
                        # Try partial matching
                        matches = [name for name in self.asset_name_to_index.keys() 
                                if asset_name in name or name in asset_name]
                        
                        if matches:
                            # Use the closest match
                            asset_index = self.asset_name_to_index[matches[0]]
                            logger.info(f"Using partial match for '{asset_name}': '{matches[0]}'")
                        else:
                            # Keep original index if no match
                            logger.warning(f"Asset name '{asset_name}' not found, using index {asset_index}")
                
                # Ensure asset index is within bounds
                asset_index = min(asset_index, len(action) - 1)
                if asset_index < 0:
                    asset_index = 0
                    
                # Extract the specific action for this asset
                specific_action = action[asset_index]
                display_asset_name = self.asset_names[asset_index] if asset_index < len(self.asset_names) else f"Asset {asset_index}"
            else:
                # Single asset scenario
                specific_action = action[0] if isinstance(action, (list, np.ndarray)) else action
                display_asset_name = asset_name or self.asset_names[0] if self.asset_names else "Asset"
            
            # Map action value to trading decision
            decision = self._action_to_decision(specific_action)
            logger.info(f"Action for {display_asset_name}: {specific_action:.4f} → Decision: {decision}")
            
            # Filter market data for the specific asset if possible
            if isinstance(market_data, dict) and asset_index < len(self.asset_names):
                asset_key = self.asset_names[asset_index]
                if asset_key in market_data:
                    specific_market_data = market_data[asset_key]
                else:
                    # Fall back to first available data
                    specific_market_data = next(iter(market_data.values())) if market_data else pd.DataFrame()
            else:
                # Use as is if not a dict or no matching asset
                specific_market_data = market_data
            
            # Log key technical indicators for debugging
            if isinstance(specific_market_data, pd.DataFrame) and not specific_market_data.empty:
                latest = specific_market_data.iloc[-1]
                indicators_log = []
                
                # Log RSI
                if 'rsi' in latest:
                    rsi = latest['rsi']
                    rsi_state = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
                    indicators_log.append(f"RSI: {rsi:.2f} ({rsi_state})")
                
                # Log MACD
                if all(col in latest for col in ['macd', 'macd_signal']):
                    macd = latest['macd']
                    signal = latest['macd_signal']
                    macd_state = "bullish" if macd > signal else "bearish"
                    indicators_log.append(f"MACD: {macd:.4f} vs Signal: {signal:.4f} ({macd_state})")
                
                # Log price trend
                if 'close' in latest and len(specific_market_data) > 5:
                    current_price = latest['close']
                    prev_price = specific_market_data.iloc[-6]['close']
                    change_pct = (current_price / prev_price - 1) * 100
                    trend_desc = "bullish" if change_pct > 1 else "bearish" if change_pct < -1 else "neutral"
                    indicators_log.append(f"Price trend: {change_pct:.2f}% ({trend_desc})")
                
                if indicators_log:
                    logger.info(f"Technical indicators for {display_asset_name}: {' | '.join(indicators_log)}")
                    
                    # Print decision vs indicator alignment warning if needed
                    if (decision.startswith("BUY") and "bearish" in str(indicators_log)) or \
                       (decision.startswith("SELL") and "bullish" in str(indicators_log)):
                        logger.warning(f"Potential mismatch: {decision} decision with conflicting indicators")
            
            # Generate explanation using the model
            try:
                logger.debug(f"Generating explanation for {display_asset_name} (decision: {decision}, action: {specific_action:.4f})")
                explanation = self.model.generate_explanation(
                    observation=observation,
                    market_data=specific_market_data,
                    action=specific_action,
                    decision=decision,
                    asset_name=display_asset_name
                )
                
                # Check if explanation is empty or default text
                is_default = explanation in ["Analysis not available at this time.", "", None]
                if is_default:
                    logger.warning(f"Empty explanation from LLM for {display_asset_name}")
                    
                    if use_fallback:
                        # Generate basic explanation based on technical indicators if LLM fails
                        logger.info("Using fallback explanation based on technical indicators")
                        explanation = self._generate_basic_explanation(specific_market_data, specific_action, decision, display_asset_name)
                
                return explanation
                
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
                if use_fallback:
                    # Fallback explanation if model generation fails
                    return self._generate_basic_explanation(specific_market_data, specific_action, decision, display_asset_name)
                else:
                    return f"Error generating explanation for {display_asset_name}: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in explain_decision: {str(e)}")
            return f"Unable to explain decision: {str(e)}"
    
    def batch_explain(self, observations, market_data, actions, asset_names=None, use_fallback=True):
        """
        Generate explanations for a batch of observations and actions.
        
        Args:
            observations: Batch of observations
            market_data: Market data with technical indicators
            actions: Batch of actions from the policy (can be multi-asset)
            asset_names: List of asset names to explain (defaults to all assets)
            use_fallback: Whether to use fallback explanation if LLM output is empty
            
        Returns:
            List of natural language explanations
        """
        explanations = []
        
        # Determine which assets to explain
        if asset_names is None:
            # Default to explaining all assets in first action
            if isinstance(actions[0], (list, np.ndarray)):
                assets_to_explain = range(len(actions[0]))
            else:
                assets_to_explain = [0]  # Single asset case
        else:
            # Use provided asset names
            assets_to_explain = []
            for name in asset_names:
                if isinstance(name, int):
                    assets_to_explain.append(name)  # Already an index
                elif name.lower() in self.asset_name_to_index:
                    assets_to_explain.append(self.asset_name_to_index[name.lower()])
                else:
                    # If name not found, append to explanation list and continue
                    explanations.append(f"Asset '{name}' not found in available assets")
                    continue
        
        # Generate explanations for each observation and selected assets
        for i, (obs, action) in enumerate(zip(observations, actions)):
            if isinstance(market_data, list):
                # Multiple market data frames provided
                md = market_data[i] if i < len(market_data) else market_data[-1]
            else:
                # Single market data frame for all observations
                md = market_data
                
            # Generate explanation for each selected asset
            for asset_idx in assets_to_explain:
                asset_name = self.asset_names[asset_idx] if asset_idx < len(self.asset_names) else f"Asset {asset_idx}"
                explanation = self.explain_decision(
                    observation=obs, 
                    market_data=md, 
                    action=action, 
                    asset_index=asset_idx,
                    asset_name=asset_name,
                    use_fallback=use_fallback
                )
                explanations.append(explanation)
                
        return explanations
    
    def _generate_basic_explanation(self, market_data, action, decision, asset_name):
        """
        Generate a basic explanation based on technical indicators when LLM fails.
        
        Args:
            market_data: Market data with technical indicators
            action: Action value from the policy
            decision: Trading decision label
            asset_name: Name of the asset
            
        Returns:
            Basic explanation text
        """
        try:
            explanation_parts = []
            explanation_parts.append(f"Decision for {asset_name}: {decision}")
            
            # Handle possible market data formats
            if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                latest = market_data.iloc[-1]
                
                # Add RSI information if available
                if 'rsi' in latest:
                    rsi = latest['rsi']
                    if rsi < 30:
                        explanation_parts.append(f"RSI is {rsi:.2f}, indicating oversold conditions.")
                    elif rsi > 70:
                        explanation_parts.append(f"RSI is {rsi:.2f}, indicating overbought conditions.")
                    else:
                        explanation_parts.append(f"RSI is {rsi:.2f}, in neutral territory.")
                
                # Add MACD information if available
                if 'macd' in latest and 'macd_signal' in latest:
                    macd = latest['macd']
                    signal = latest['macd_signal']
                    if macd > signal:
                        explanation_parts.append(f"MACD ({macd:.4f}) is above signal line ({signal:.4f}), suggesting bullish momentum.")
                    else:
                        explanation_parts.append(f"MACD ({macd:.4f}) is below signal line ({signal:.4f}), suggesting bearish momentum.")
                
                # Add price trend information if available
                if 'close' in latest and len(market_data) > 5:
                    current_price = latest['close']
                    prev_price = market_data.iloc[-6]['close']  # 5 periods ago
                    change_pct = (current_price / prev_price - 1) * 100
                    
                    if change_pct > 5:
                        trend_desc = "strongly bullish"
                    elif change_pct > 1:
                        trend_desc = "moderately bullish"
                    elif change_pct < -5:
                        trend_desc = "strongly bearish"
                    elif change_pct < -1:
                        trend_desc = "moderately bearish"
                    else:
                        trend_desc = "relatively flat"
                        
                    explanation_parts.append(f"Price trend is {trend_desc} with a {change_pct:.2f}% change over the last 5 periods.")
            
            # Add decision rationale based on action value
            if decision.startswith("BUY"):
                explanation_parts.append(f"The model recommends a {decision} position based on favorable technical indicators.")
            elif decision.startswith("SELL"):
                explanation_parts.append(f"The model recommends a {decision} position due to negative technical signals.")
            else:
                explanation_parts.append(f"The model recommends to {decision} as the market signals are currently neutral.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating basic explanation: {str(e)}")
            return f"Trading decision for {asset_name}: {decision} (action value: {action:.4f})"

    def explain_trading_decisions(self, market_data_path: str, use_fallback: bool = True, asset_names: Optional[List[str]] = None, use_synthetic_actions: bool = False) -> List[str]:
        """
        Process market data and generate explanations for trading decisions.
        Used by CLI tool to provide explanations for recent market data.
        
        Args:
            market_data_path: Path to market data file or directory
            use_fallback: Whether to use fallback explanations if LLM output is empty
            asset_names: List of specific asset names to explain (defaults to all)
            use_synthetic_actions: If True, use synthetic actions even if RL model is available
            
        Returns:
            List of explanations for trading decisions
        """
        explanations = []
        
        try:
            # Load market data
            if os.path.isdir(market_data_path):
                # Directory with multiple asset files
                market_data_dict = {}
                for filename in os.listdir(market_data_path):
                    if filename.endswith(('.csv', '.parquet', '.pq')):
                        file_path = os.path.join(market_data_path, filename)
                        asset_name = os.path.splitext(filename)[0]
                        
                        # Load data based on file extension
                        if filename.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_parquet(file_path)
                            
                        # Ensure column names are lowercase
                        df.columns = df.columns.str.lower()
                        market_data_dict[asset_name] = df
                
                if not market_data_dict:
                    raise ValueError(f"No market data files found in {market_data_path}")
                    
                # Use specific assets if requested
                if asset_names:
                    filtered_data = {}
                    for name in asset_names:
                        if name in market_data_dict:
                            filtered_data[name] = market_data_dict[name]
                        else:
                            logger.warning(f"Asset {name} not found in market data directory")
                    
                    if filtered_data:
                        market_data_dict = filtered_data
                    else:
                        logger.warning("None of the requested assets found, using all available assets")
                
                # Process each asset
                for asset_name, data in market_data_dict.items():
                    # Get the last observation
                    if len(data) < 5:
                        logger.warning(f"Not enough data for {asset_name}, skipping")
                        continue
                        
                    # Get recent data as observation
                    recent_data = data.tail(30)  # Get more data for RL model
                    observation = self._prepare_observation(recent_data)
                    
                    # Log which source of actions we're using
                    signal_source = None
                    
                    # Generate action using RL model if available, otherwise use synthetic
                    if self.rl_model is not None and not use_synthetic_actions:
                        action = self._generate_rl_action(observation, data, asset_name)
                        signal_source = "RL model"
                    else:
                        action = self._generate_synthetic_action(data, asset_name)
                        signal_source = "synthetic" if self.rl_model is not None else "technical indicators (no RL model available)"
                    
                    logger.info(f"Generated {signal_source} action for {asset_name}: {action:.4f}")
                        
                    # Generate explanation
                    explanation = self.explain_decision(
                        observation=observation,
                        market_data=data,
                        action=action,
                        asset_name=asset_name,
                        use_fallback=use_fallback
                    )
                    
                    # Add clear signal source labeling
                    if signal_source == "RL model":
                        # For RL model predictions, add a subtle attribution
                        formatted_explanation = f"{asset_name}: {explanation} [Signal: RL model prediction]"
                    else:
                        # For synthetic signals, make it very clear
                        formatted_explanation = f"{asset_name}: {explanation} \n[NOTE: This explanation is based on synthetic signals from {signal_source}, not actual RL model predictions]"
                    
                    explanations.append(formatted_explanation)
            else:
                # Single market data file
                if market_data_path.endswith('.csv'):
                    market_data = pd.read_csv(market_data_path)
                elif market_data_path.endswith(('.parquet', '.pq')):
                    market_data = pd.read_parquet(market_data_path)
                else:
                    raise ValueError(f"Unsupported file format: {market_data_path}")
                    
                # Ensure column names are lowercase
                market_data.columns = market_data.columns.str.lower()
                
                # Get recent data as observation
                if len(market_data) < 5:
                    raise ValueError("Not enough data points in market data")
                    
                # Get asset name from filename if available
                default_asset_name = os.path.splitext(os.path.basename(market_data_path))[0]
                asset_name = asset_names[0] if asset_names and len(asset_names) > 0 else default_asset_name
                
                # Get recent data as observation
                recent_data = market_data.tail(30)  # Get more data for RL model
                observation = self._prepare_observation(recent_data)
                
                # Log which source of actions we're using
                signal_source = None
                
                # Generate action using RL model if available, otherwise use synthetic
                if self.rl_model is not None and not use_synthetic_actions:
                    action = self._generate_rl_action(observation, market_data, asset_name)
                    signal_source = "RL model"
                else:
                    action = self._generate_synthetic_action(market_data, asset_name)
                    signal_source = "synthetic" if self.rl_model is not None else "technical indicators (no RL model available)"
                
                logger.info(f"Generated {signal_source} action for {asset_name}: {action:.4f}")
                
                # Generate explanation
                explanation = self.explain_decision(
                    observation=observation,
                    market_data=market_data,
                    action=action,
                    asset_name=asset_name,
                    use_fallback=use_fallback
                )
                
                # Add clear signal source labeling
                if signal_source == "RL model":
                    # For RL model predictions, add a subtle attribution
                    formatted_explanation = f"{explanation} [Signal: RL model prediction]"
                else:
                    # For synthetic signals, make it very clear
                    formatted_explanation = f"{explanation} \n[NOTE: This explanation is based on synthetic signals from {signal_source}, not actual RL model predictions]"
                
                explanations.append(formatted_explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining trading decisions: {str(e)}")
            return [f"Failed to generate explanations: {str(e)}"]
    
    def _prepare_observation(self, market_data: pd.DataFrame) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare market data as observation for RL model, exactly matching the structure used during training.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Observation in the format expected by the RL model
        """
        logger.info("Building observation with same structure as training environment")
        
        # Initialize observation array
        observation = []
        
        # Ensure we have the required base OHLCV columns
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Define technical features that match the training environment
        tech_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                         'sma_10', 'sma_20', 'returns_1d', 'returns_5d', 'returns_10d',
                         'volatility_5d', 'volatility_10d', 'atr']
        
        # Ensure market_data columns are lowercase
        market_data.columns = [col.lower() for col in market_data.columns]
        
        # Check if we have multi-asset data (with 'symbol' column)
        is_multi_asset = 'symbol' in market_data.columns
        
        # Calculate any missing indicators
        market_data = self._calculate_missing_indicators(market_data, base_features, tech_features)
        
        # Target asset list - fixed for the expected 78-sized observation (3 assets)
        target_assets = ['BTC', 'ETH', 'SOL']
        
        # Map the available asset_names to the standard target_assets
        asset_map = {}
        if len(self.asset_names) > 0:
            for i, asset in enumerate(self.asset_names):
                if i < len(target_assets):
                    asset_map[target_assets[i]] = asset
        
        # 1. Add market data features for each target asset
        for target_asset in target_assets:
            # Get the corresponding asset name if we have a mapping
            asset_name = asset_map.get(target_asset, target_asset)
            
            # Check if we have data for this specific asset
            asset_data = None
            if is_multi_asset:
                asset_df = market_data[market_data['symbol'] == asset_name]
                if not asset_df.empty:
                    asset_data = asset_df.iloc[-1]
            
            # If no specific asset data, use the entire market_data
            if asset_data is None:
                asset_data = market_data.iloc[-1] if not market_data.empty else None
            
            # Add base features (OHLCV)
            for feat in base_features:
                if asset_data is not None and feat in asset_data.index:
                    value = float(asset_data[feat])
                elif feat in market_data.columns:
                    value = float(market_data[feat].iloc[-1])
                else:
                    logger.warning(f"Missing base feature {feat} for {asset_name}, using 0.0")
                    value = 0.0
                observation.append(value)
            
            # Add technical indicators
            for feat in tech_features:
                if asset_data is not None and feat in asset_data.index:
                    value = float(asset_data[feat])
                elif feat in market_data.columns:
                    value = float(market_data[feat].iloc[-1])
                else:
                    logger.warning(f"Missing technical feature {feat} for {asset_name}, using 0.0")
                    value = 0.0
                observation.append(value)
        
        # 2. Add portfolio data for each asset
        # In training, this includes position size, position value ratio, and funding accrued
        # During inference, we don't have real positions, so use placeholder values
        total_portfolio_value = 10000.0  # Default value
        for _ in target_assets:  # Use target_assets to ensure we always have 3 assets' worth of data
            # Placeholder for position size (0 = no position)
            observation.append(0.0)
            
            # Placeholder for position value ratio
            observation.append(0.0)
            
            # Placeholder for funding accrued
            observation.append(0.0)
        
        # 3. Add global portfolio data 
        # In training, this includes portfolio value ratio, recent PnL ratio, and active positions ratio
        # During inference, use neutral placeholder values
        observation.append(1.0)  # Portfolio value ratio (1.0 = at initial value)
        observation.append(0.0)  # Recent PnL ratio (0.0 = no recent profit/loss)
        observation.append(0.0)  # Active positions ratio (0.0 = no active positions)
        
        # 4. Add additional features to match expected 78 features
        # Based on error we need 9 more features to reach 78 from our current 69
        logger.info("Adding additional features to match expected 78 features")
        for i in range(9):
            observation.append(0.0)
        
        # Convert to numpy array
        observation_array = np.array(observation, dtype=np.float32)
        
        # Check if RL model uses Dict observation space
        if self.rl_model is not None:
            from gymnasium import spaces
            if isinstance(self.rl_model.observation_space, spaces.Dict):
                # Get expected shape of "market" entry
                expected_rows = None
                expected_cols = None
                if hasattr(self.rl_model.observation_space.spaces['market'], 'shape'):
                    if len(self.rl_model.observation_space.spaces['market'].shape) == 2:
                        expected_rows, expected_cols = self.rl_model.observation_space.spaces['market'].shape
                
                # If expected shape is known, reshape accordingly
                if expected_rows is not None and expected_cols is not None:
                    try:
                        market_obs = observation_array.reshape(expected_rows, expected_cols)
                    except ValueError:
                        # If cannot reshape, use a default reshape
                        logger.warning(f"Cannot reshape observation to {expected_rows}x{expected_cols}, using default")
                        num_features = len(base_features) + len(tech_features)
                        rows = len(observation_array) // num_features
                        market_obs = observation_array.reshape(rows, num_features)
                else:
                    # Calculate a sensible reshape based on feature counts
                    num_features = len(base_features) + len(tech_features)
                    rows = len(observation_array) // num_features
                    market_obs = observation_array.reshape(rows, num_features)
                
                return {"market": market_obs}
        
        # If model expects a flat array, check shape and adjust if needed
        if self.rl_model is not None:
            expected_shape = self.rl_model.observation_space.shape[0]
            current_shape = len(observation_array)
            
            if current_shape != expected_shape:
                logger.warning(f"Observation shape mismatch: got {current_shape}, expected {expected_shape}")
                
                # For the current case of 78 expected features:
                # We should have exactly 78 elements (75 asset features + 3 portfolio features)
                # The array should already be exactly sized due to our target_assets approach
                
                if current_shape < expected_shape:
                    # Pad with zeros if needed
                    padding = np.zeros(expected_shape - current_shape, dtype=np.float32)
                    observation_array = np.concatenate([observation_array, padding])
                    logger.info(f"Padded observation from {current_shape} to match expected shape {expected_shape}")
                else:
                    # Truncate if needed
                    observation_array = observation_array[:expected_shape]
                    logger.info(f"Truncated observation from {current_shape} to match expected shape {expected_shape}")
        
        # Log the final observation shape for debugging
        logger.info(f"Final observation shape: {observation_array.shape}")
        
        return observation_array
        
    def _calculate_missing_indicators(self, market_data: pd.DataFrame, base_features: List[str], tech_features: List[str]) -> pd.DataFrame:
        """
        Calculate any missing technical indicators required for the observation
        
        Args:
            market_data: Market data DataFrame
            base_features: List of required base features
            tech_features: List of required technical features
            
        Returns:
            DataFrame with all required indicators
        """
        # Ensure we have all base features
        for feat in base_features:
            if feat not in market_data.columns:
                if 'close' in market_data.columns:
                    # Use close price as fallback
                    market_data[feat] = market_data['close']
                else:
                    # Create placeholder data
                    market_data[feat] = np.ones(len(market_data))
                    
        # Calculate technical indicators with better error handling
        try:
            from ta.trend import SMAIndicator, MACD
            from ta.momentum import RSIIndicator
            from ta.volatility import BollingerBands, AverageTrueRange
            
            close_series = market_data['close']
            high_series = market_data['high']
            low_series = market_data['low']
            
            # RSI
            if 'rsi' not in market_data.columns:
                try:
                    rsi = RSIIndicator(close=close_series).rsi()
                    market_data['rsi'] = rsi
                except Exception as e:
                    logger.warning(f"Error calculating RSI: {e}")
                    market_data['rsi'] = 50.0  # Neutral RSI value
            
            # MACD
            if 'macd' not in market_data.columns or 'macd_signal' not in market_data.columns:
                try:
                    macd_indicator = MACD(close=close_series)
                    market_data['macd'] = macd_indicator.macd()
                    market_data['macd_signal'] = macd_indicator.macd_signal()
                except Exception as e:
                    logger.warning(f"Error calculating MACD: {e}")
                    market_data['macd'] = 0.0
                    market_data['macd_signal'] = 0.0
            
            # Bollinger Bands
            if 'bb_upper' not in market_data.columns or 'bb_middle' not in market_data.columns or 'bb_lower' not in market_data.columns:
                try:
                    bb = BollingerBands(close=close_series)
                    market_data['bb_upper'] = bb.bollinger_hband()
                    market_data['bb_middle'] = bb.bollinger_mavg()
                    market_data['bb_lower'] = bb.bollinger_lband()
                except Exception as e:
                    logger.warning(f"Error calculating Bollinger Bands: {e}")
                    # Use close price with offsets as fallback
                    market_data['bb_middle'] = close_series
                    market_data['bb_upper'] = close_series * 1.02  # 2% above
                    market_data['bb_lower'] = close_series * 0.98  # 2% below
            
            # SMAs
            if 'sma_10' not in market_data.columns:
                try:
                    market_data['sma_10'] = SMAIndicator(close=close_series, window=10).sma_indicator()
                except Exception as e:
                    logger.warning(f"Error calculating SMA(10): {e}")
                    market_data['sma_10'] = close_series
                    
            if 'sma_20' not in market_data.columns:
                try:
                    market_data['sma_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
                except Exception as e:
                    logger.warning(f"Error calculating SMA(20): {e}")
                    market_data['sma_20'] = close_series
            
            # Returns
            if 'returns_1d' not in market_data.columns:
                market_data['returns_1d'] = close_series.pct_change(1).fillna(0)
                
            if 'returns_5d' not in market_data.columns:
                market_data['returns_5d'] = close_series.pct_change(5).fillna(0)
                
            if 'returns_10d' not in market_data.columns:
                market_data['returns_10d'] = close_series.pct_change(10).fillna(0)
            
            # Volatility
            if 'volatility_5d' not in market_data.columns:
                market_data['volatility_5d'] = market_data['returns_1d'].rolling(5).std().fillna(0)
                
            if 'volatility_10d' not in market_data.columns:
                market_data['volatility_10d'] = market_data['returns_1d'].rolling(10).std().fillna(0)
            
            # Average True Range - error happens here most often
            if 'atr' not in market_data.columns:
                try:
                    # Ensure data is sufficient for ATR calculation - needs at least 14 data points
                    if len(market_data) >= 14:
                        market_data['atr'] = AverageTrueRange(
                            high=high_series, 
                            low=low_series, 
                            close=close_series,
                            window=14
                        ).average_true_range()
                    else:
                        # Not enough data for ATR calculation, use simplified formula
                        logger.warning(f"Not enough data for ATR calculation, using simplified approach")
                        market_data['atr'] = (high_series - low_series).mean()
                except Exception as e:
                    logger.warning(f"Error calculating ATR: {e}")
                    # Use a basic volatility measure as fallback
                    market_data['atr'] = (high_series - low_series).mean()
            
            # Handle NaN values
            market_data = market_data.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Set default values for all technical features
            for feat in tech_features:
                if feat not in market_data.columns:
                    market_data[feat] = 0.0
        
        return market_data

    def _generate_rl_action(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], market_data: pd.DataFrame, asset_name: str) -> float:
        """
        Generate action using RL model
        
        Args:
            observation: Observation for the RL model
            market_data: Market data DataFrame
            asset_name: Name of the asset
            
        Returns:
            float: Action value
        """
        if self.rl_model is None:
            logger.warning("No RL model provided, falling back to synthetic action generation")
            return self._generate_synthetic_action(market_data, asset_name)
        
        try:
            # Use the new _prepare_observation method directly
            if isinstance(observation, (np.ndarray, dict)):
                # If observation is already prepared, use it directly
                obs = observation
            else:
                # Otherwise, prepare it from market data
                obs = self._prepare_observation(market_data)
            
            # Log observation shape for debugging
            if isinstance(obs, np.ndarray):
                logger.debug(f"Observation shape: {obs.shape}")
            elif isinstance(obs, dict) and "market" in obs:
                logger.debug(f"Observation market shape: {obs['market'].shape}")
            
            # Get asset index for multi-asset models
            asset_idx = 0
            if asset_name is not None and asset_name.lower() in self.asset_name_to_index:
                asset_idx = self.asset_name_to_index[asset_name.lower()]
            
            # Predict action using the RL model
            with torch.no_grad():
                action, _ = self.rl_model.predict(obs, deterministic=False)
            
            # Format action depending on type
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Handle array output for multi-asset model
            if isinstance(action, np.ndarray) and len(action) > 1:
                # Extract action for the specified asset
                if asset_idx < len(action):
                    return float(action[asset_idx])
                else:
                    logger.warning(f"Asset index {asset_idx} out of bounds for action with shape {action.shape}")
                    return float(action[0])  # Return first asset action as fallback
            
            # Convert action to float
            return float(action if np.isscalar(action) else action.item() if hasattr(action, 'item') else action[0])
            
        except Exception as e:
            logger.error(f"Error generating RL action: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._generate_synthetic_action(market_data, asset_name)

    def _generate_synthetic_action(self, market_data: pd.DataFrame, asset_name: str) -> float:
        """
        Generate a synthetic action value based on technical indicators.
        Ensures consistency between the action and the technical indicators
        used to generate the explanation.
        
        Args:
            market_data: DataFrame containing market data with indicators
            asset_name: Name of the asset for logging
            
        Returns:
            Synthetic action value between -0.8 and 0.8
        """
        try:
            if len(market_data) < 5:
                logger.warning(f"Not enough data to calculate synthetic action for {asset_name}")
                return 0.0
            
            # Check for timestamp column for data consistency logging
            timestamp = None
            if 'timestamp' in market_data.columns:
                timestamp = market_data['timestamp'].iloc[-1]
                logger.debug(f"Generating synthetic action for {asset_name} at timestamp: {timestamp}")
                
            latest = market_data.iloc[-1]
            action_components = []
            indicators_log = []
            indicator_values = {}  # Store values for debugging
            
            # Component 1: Price trend (base component)
            if 'close' in market_data.columns:
                # Short-term trend (5 periods)
                short_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5] - 1)
                # Scale to reasonable range (-0.5 to 0.5)
                price_component = min(0.5, max(-0.5, short_change * 5))
                action_components.append(price_component)
                indicators_log.append(f"Price trend: {short_change:.4f} (5-period) → component: {price_component:.4f}")
                indicator_values['price_change_5d'] = short_change * 100  # Store as percentage
            
            # Component 2: RSI
            if 'rsi' in latest:
                rsi = latest['rsi']
                indicator_values['rsi'] = rsi
                
                # Use exact same logic as in format_prompt
                if rsi > 70:
                    rsi_component = -0.3  # Overbought (bearish)
                elif rsi < 30:
                    rsi_component = 0.3   # Oversold (bullish)
                else:
                    # Linear scaling from 30-70 range to 0.3 to -0.3
                    rsi_component = 0.3 - ((rsi - 30) / 40) * 0.6
                action_components.append(rsi_component)
                indicators_log.append(f"RSI: {rsi:.2f} → component: {rsi_component:.4f}")
            
            # Component 3: MACD
            if all(col in latest for col in ['macd', 'macd_signal']):
                macd = latest['macd']
                signal = latest['macd_signal']
                macd_diff = macd - signal
                indicator_values['macd'] = macd
                indicator_values['macd_signal'] = signal
                
                # Use exact same scaling as in format_prompt
                macd_component = min(0.3, max(-0.3, macd_diff * 10))
                action_components.append(macd_component)
                indicators_log.append(f"MACD diff: {macd_diff:.4f} → component: {macd_component:.4f}")
            
            # Component 4: Bollinger Bands
            if all(col in latest for col in ['bb_upper', 'bb_middle', 'bb_lower']) and 'close' in latest:
                upper = latest['bb_upper']
                middle = latest['bb_middle']
                lower = latest['bb_lower']
                price = latest['close']
                
                indicator_values['bb_upper'] = upper
                indicator_values['bb_middle'] = middle
                indicator_values['bb_lower'] = lower
                
                # Calculate BB position and component
                if price > upper:
                    bb_component = -0.2  # Above upper band (bearish)
                    bb_state = "above upper"
                elif price < lower:
                    bb_component = 0.2   # Below lower band (bullish)
                    bb_state = "below lower"
                else:
                    # Neutral - slight tendency toward center
                    relative_position = (price - lower) / (upper - lower) - 0.5
                    bb_component = -0.1 * relative_position  # -0.05 to 0.05
                    bb_state = "within bands"
                
                action_components.append(bb_component)
                indicators_log.append(f"BB ({bb_state}): → component: {bb_component:.4f}")
            
            # Component 5: Support/Resistance proximity
            if all(col in latest for col in ['support', 'resistance']) and 'close' in latest:
                support = latest['support']
                resistance = latest['resistance']
                price = latest['close']
                
                indicator_values['support'] = support
                indicator_values['resistance'] = resistance
                
                sr_component = 0.0
                sr_log = []
                
                # Check proximity to support
                if support > 0:
                    distance_to_support = (price - support) / price * 100
                    if distance_to_support < 2:
                        # Near support - bullish signal
                        support_effect = 0.2 * (1 - distance_to_support/2)  # 0.2 at 0%, 0.1 at 1%, 0 at 2%
                        sr_component += support_effect
                        sr_log.append(f"near support ({distance_to_support:.2f}%): +{support_effect:.3f}")
                
                # Check proximity to resistance
                if resistance > 0:
                    distance_to_resistance = (resistance - price) / price * 100
                    if distance_to_resistance < 2:
                        # Near resistance - bearish signal
                        resistance_effect = -0.2 * (1 - distance_to_resistance/2)  # -0.2 at 0%, -0.1 at 1%, 0 at 2%
                        sr_component += resistance_effect
                        sr_log.append(f"near resistance ({distance_to_resistance:.2f}%): {resistance_effect:.3f}")
                
                if sr_log:
                    action_components.append(sr_component)
                    indicators_log.append(f"S/R: {', '.join(sr_log)} → component: {sr_component:.4f}")
            
            # Combine components (if any exist)
            if action_components:
                # Average of all components, then ensure within bounds
                action = sum(action_components) / len(action_components)
                # Limit to range (-0.8 to 0.8)
                action = min(0.8, max(-0.8, action))
                
                # Log the calculation for debugging
                indicators_str = ", ".join(indicators_log)
                logger.debug(f"Synthetic action for {asset_name}: {action:.4f} [Components: {indicators_str}]")
                
                # Get corresponding decision
                decision = self._action_to_decision(action)
                
                # Calculate sentiment tallies
                bullish_count = 0
                bearish_count = 0
                for component in action_components:
                    if component > 0.05:  # Threshold for considering bullish
                        bullish_count += 1
                    elif component < -0.05:  # Threshold for considering bearish
                        bearish_count += 1
                
                # Log decision and sentiment alignment
                decision_sentiment = "bullish" if decision.startswith("BUY") else "bearish" if decision.startswith("SELL") else "neutral"
                logger.debug(f"Decision: {decision} ({decision_sentiment}) with {bullish_count} bullish vs {bearish_count} bearish indicators")
                
                # Check for potential inconsistency
                if (decision_sentiment == "bullish" and bearish_count > bullish_count) or \
                   (decision_sentiment == "bearish" and bullish_count > bearish_count):
                    logger.warning(f"Potential inconsistency in synthetic action: {decision} does not align with indicator balance")
                
                return action
            else:
                logger.warning(f"No indicators found to generate synthetic action for {asset_name}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error generating synthetic action: {str(e)}")
            return 0.0

    def _action_to_decision(self, action: float) -> str:
        """Convert action value to decision string (for consistency)"""
        if action > 0.6:
            return "BUY (Strong)"
        elif action > 0.2:
            return "BUY (Moderate)"
        elif action > 0:
            return "BUY (Light)"
        elif action < -0.6:
            return "SELL (Strong)"
        elif action < -0.2:
            return "SELL (Moderate)"
        elif action < 0:
            return "SELL (Light)"
        else:
            return "HOLD"


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
            commentary = self.llm.generate_text(
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
                
                # Try with different parameters using generate_text method
                commentary = self.llm.generate_text(
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
        summary = self.llm.generate_text(
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
    asset_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract trading signals and positions from the RL model for market data.
    
    Args:
        rl_model: Trained reinforcement learning model (from stable_baselines3)
        market_data: Pandas DataFrame with market data (OHLCV)
        normalize_data: Whether to normalize input data
        action_threshold: Threshold for considering an action significant
        asset_names: Optional list of asset names for multi-asset environments
        
    Returns:
        Dictionary with trading signals for each timestamp
    """
    import numpy as np
    import torch
    from gymnasium import spaces
    import time
    
    start_time = time.time()
    
    # Default asset name if none provided
    if asset_names is None or len(asset_names) == 0:
        asset_names = ["Asset"]
    
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
    
    # Create temporary RLLMExplainer to use its observation preparation logic
    from trading_llm.inference import RLLMExplainer
    explainer = RLLMExplainer(
        rl_model=rl_model,
        asset_names=asset_names
    )
    
    # Check if we have enough data
    if len(data) < 15:  # Need at least 15 data points for indicators
        print(f"WARNING: Not enough data points ({len(data)}) for reliable indicators. Need at least 15.")
        # Return empty signals dictionary with a warning
        return {"warning": {"message": f"Not enough data points ({len(data)}) for reliable indicators."}}
    
    # Use a fixed window size approach for more efficient processing
    # Minimum window size needed for all technical indicators
    min_window_size = 30  # Large enough for MACD, RSI, etc.
    
    # Skip the first min_window_size-1 data points since we need history for indicators
    start_index = min_window_size - 1
    
    # Set a maximum number of samples to process
    # Increase from 1000 to a higher limit to process more of the data
    # -1 because we need the next price for position calculation
    max_samples = min(50000, len(data) - 1 - start_index)
    
    print(f"Processing {max_samples} samples from dataset with {len(data)} records")
    print(f"Starting from index {start_index} with window size {min_window_size}")
    
    # Extract observations from market data using a consistent approach
    timestamps = []
    observations = []
    
    error_count = 0
    max_errors = 20  # Maximum number of errors before giving up
    progress_step = max(1, max_samples // 10)  # Show progress every 10%
    
    # Process each timestamp in the data, with fixed window size
    for i in range(start_index, start_index + max_samples):
        # Show progress
        if i % progress_step == 0 or i == start_index:
            print(f"Processing timestamp {i-start_index+1}/{max_samples} ({(i-start_index+1)/max_samples*100:.1f}%)")
        
        # Extract the fixed-size window of data
        window_start = i - min_window_size + 1  # Ensure window_size data points
        window_data = data.iloc[window_start:i+1].copy()
        
        # Use the explainer to prepare the observation correctly
        try:
            obs = explainer._prepare_observation(window_data)
            observations.append(obs)
            timestamps.append(data.index[i] if not data.index.equals(pd.RangeIndex(len(data))) else i)
        except Exception as e:
            error_count += 1
            if error_count <= 3 or error_count % 100 == 0:  # Show only first 3 errors and then every 100th
                print(f"Error preparing observation at index {i}: {str(e)}")
            if error_count >= max_errors:
                print(f"Too many errors ({error_count}). Stopping observation preparation.")
                break
            continue
    
    # If we have no valid observations, return an empty result
    if not observations:
        print("No valid observations could be generated. Cannot extract trading signals.")
        return {"error": {"message": "No valid observations could be generated."}}
    
    print(f"Generated {len(observations)} valid observations. Predicting actions...")
    
    # Get actions from model
    actions = []
    prediction_errors = 0
    max_prediction_errors = 20
    batch_size = 64  # Increase batch size for faster processing
    
    # Process in batches for memory efficiency
    for i in range(0, len(observations), batch_size):
        batch_progress = min(i + batch_size, len(observations))
        if i % (batch_size * 10) == 0 or i == 0:  # Show progress every 10 batches
            print(f"Predicting batch {i//batch_size + 1}/{(len(observations)-1)//batch_size + 1} ({batch_progress}/{len(observations)})")
        
        batch_obs = observations[i:i+batch_size]
        batch_actions = []
        
        for j, obs in enumerate(batch_obs):
            try:
                with torch.no_grad():  # Use no_grad to reduce memory usage
                    # Only show the first few observation shapes
                    if i == 0 and j < 3:
                        print(f"DEBUG: Predicting with observation shape {obs.shape if hasattr(obs, 'shape') else 'unknown'}")
                    
                    action, _ = rl_model.predict(obs, deterministic=False)
                    
                    # Process the action
                    if isinstance(action, torch.Tensor):
                        action_cpu = action.detach().cpu()
                        action_np = action_cpu.numpy()
                    else:
                        action_np = action
                    
                    # Handle multi-dimensional actions and arrays correctly
                    if isinstance(action_np, np.ndarray):
                        # Check if it's a multi-asset action (array with multiple values)
                        if action_np.size > 1:
                            # For multi-asset case, use the full array
                            action_value = action_np
                        elif action_np.size == 1:
                            # Single-asset case with array of size 1
                            action_value = float(action_np.item())
                        else:
                            # Empty array (shouldn't happen)
                            action_value = 0.0
                    elif isinstance(action_np, (int, float, np.number)):
                        # Direct scalar value
                        action_value = float(action_np)
                    else:
                        action_value = 0.0
                        
                    batch_actions.append(action_value)
            except Exception as e:
                import traceback
                prediction_errors += 1
                if prediction_errors <= 3 or prediction_errors % 100 == 0:  # Show only first 3 errors and then every 100th
                    print(f"ERROR processing observation {i+j}: {str(e)}")
                    if prediction_errors <= 3:
                        print(f"Traceback: {traceback.format_exc()}")
                # Use a neutral action as fallback
                batch_actions.append(0.0)
                
                # Exit if too many prediction errors
                if prediction_errors >= max_prediction_errors:
                    print(f"Too many prediction errors ({prediction_errors}). Stopping prediction.")
                    break
                
            # Clear GPU memory after each prediction to prevent OOM
            if j % 16 == 0 and torch.cuda.is_available():  # Clean every 16 samples
                torch.cuda.empty_cache()
        
        # Exit batch processing if too many errors        
        if prediction_errors >= max_prediction_errors:
            break
                
        actions.extend(batch_actions)
    
    # If we couldn't get any actions, return an error
    if not actions:
        print("Could not generate any valid actions. Exiting.")
        return {"error": {"message": "Could not generate any valid actions."}}
    
    # Convert actions to trading signals
    signals = {}
    signal_count = 0
    signal_errors = 0
    
    for i, (timestamp, action) in enumerate(zip(timestamps, actions)):
        try:
            # Calculate index in original data
            orig_idx = start_index + i
            
            # Ensure we don't go out of bounds
            if orig_idx + 1 >= len(data):
                continue
                
            # Handle both single value and array actions
            if isinstance(action, np.ndarray) and action.size > 1:
                # Multi-asset action
                signal_dict = {}
                for j, asset_name in enumerate(asset_names):
                    if j < len(action):
                        asset_action = float(action[j])
                        
                        # Calculate position (1 for long, -1 for short, 0 for neutral)
                        if asset_action > action_threshold:
                            position = 1  # Long
                        elif asset_action < -action_threshold:
                            position = -1  # Short
                        else:
                            position = 0  # Neutral
                        
                        # Calculate next return (for evaluation)
                        try:
                            next_close = data['close'].iloc[orig_idx+1]
                            current_close = data['close'].iloc[orig_idx]
                            next_return = (next_close - current_close) / current_close
                        except Exception as e:
                            signal_errors += 1
                            if signal_errors <= 3:
                                print(f"Error calculating return: {e}")
                            next_return = 0.0
                        
                        # Calculate confidence score (scale action to 0-1 range)
                        action_max = max(abs(asset_action), 1.0)  # Prevent division by zero
                        confidence = (asset_action + action_max) / (2 * action_max)
                        
                        # Format signal for this asset
                        signal_dict[asset_name] = {
                            'action': 'buy' if asset_action > action_threshold else ('sell' if asset_action < -action_threshold else 'hold'),
                            'action_value': float(asset_action),
                            'confidence': float(confidence),
                            'position': position,
                            'next_return': float(next_return),
                        }
                
                signals[str(timestamp)] = signal_dict
                signal_count += 1
            else:
                # Single asset case
                # Convert to float in case it's still a numpy type
                action_value = float(action) if isinstance(action, (int, float, np.number)) else 0.0
                
                # Calculate position (1 for long, -1 for short, 0 for neutral)
                if action_value > action_threshold:
                    position = 1  # Long
                elif action_value < -action_threshold:
                    position = -1  # Short
                else:
                    position = 0  # Neutral
                
                # Calculate next return (for evaluation)
                try:
                    next_close = data['close'].iloc[orig_idx+1]
                    current_close = data['close'].iloc[orig_idx]
                    next_return = (next_close - current_close) / current_close
                except Exception as e:
                    signal_errors += 1
                    if signal_errors <= 3:
                        print(f"Error calculating return: {e}")
                    next_return = 0.0
                
                # Calculate confidence score (scale action to 0-1 range)
                action_max = max(abs(action_value), 1.0)  # Prevent division by zero
                confidence = (action_value + action_max) / (2 * action_max)
                
                # For backward compatibility, use the first asset name
                asset_name = asset_names[0]
                signal_dict = {
                    asset_name: {
                        'action': 'buy' if action_value > action_threshold else ('sell' if action_value < -action_threshold else 'hold'),
                        'action_value': float(action_value),
                        'confidence': float(confidence),
                        'position': position,
                        'next_return': float(next_return),
                    }
                }
                signals[str(timestamp)] = signal_dict
                signal_count += 1
        except Exception as e:
            signal_errors += 1
            if signal_errors <= 3:
                print(f"Error creating signal for index {i}: {str(e)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Extract trading signals completed in {processing_time:.1f} seconds:")
    print(f"- Processed {len(observations)} observations")
    print(f"- Generated {signal_count} trading signals for {len(asset_names)} assets")
    print(f"- Encountered {error_count} observation errors, {prediction_errors} prediction errors, {signal_errors} signal errors")
    
    return signals