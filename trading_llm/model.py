"""
Trading LLM Model - Integrates Mistral-7B with RL trading model
"""

import os
import torch
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)

logger = logging.getLogger(__name__)

class TradingLLM:
    """
    Trading Language Model for generating explanations and commentary about trading decisions.
    
    This class handles the language model used for explaining RL trading decisions
    and providing market commentary based on technical indicators.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attn: bool = True,
        use_gradient_checkpointing: bool = True,
        cache_dir: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        """
        Initialize the Trading LLM model
        
        Args:
            model_name: Base model to load
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout rate for LoRA layers
            device_map: Device mapping strategy
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
            use_flash_attn: Whether to use flash attention
            use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
            local_path: Optional local path for models
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cache_dir = cache_dir
        self.is_lora_applied = False
        
        # Will be initialized in load_model
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Load model and tokenizer
        self._load_model()
        
    def generate_explanation(
        self,
        observation,
        market_data,
        action,
        decision=None,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=None,
        asset_name=None
    ):
        """
        Generate a natural language explanation for a trading decision.
        
        Args:
            observation: Observation vector from the environment
            market_data: Market data dataframe with technical indicators
            action: Action value from policy
            decision: Trading decision label (optional)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation sampling
            top_p: Top-p probability for nucleus sampling
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling instead of greedy decoding
            asset_name: Name of the asset being analyzed
            
        Returns:
            Generated explanation string
        """
        try:
            # Create generation config
            gen_config = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Apply user-provided generation parameters or use defaults
            if temperature is not None:
                gen_config["temperature"] = temperature
            else:
                gen_config["temperature"] = 0.85  # Default
                
            if top_p is not None:
                gen_config["top_p"] = top_p
            else:
                gen_config["top_p"] = 0.92  # Default
                
            if repetition_penalty is not None:
                gen_config["repetition_penalty"] = repetition_penalty
            else:
                gen_config["repetition_penalty"] = 1.1  # Default
                
            if do_sample is not None:
                gen_config["do_sample"] = do_sample
            else:
                gen_config["do_sample"] = True  # Default
                
            # Format prompt for explanation
            prompt = self.format_prompt(
                observation=observation,
                market_data=market_data,
                action=action,
                decision=decision,
                asset_name=asset_name
            )
            
            # Print formatted prompt for debugging
            logger.debug(f"Formatted prompt: {prompt}")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]
            logger.debug(f"Input length: {input_len} tokens")
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
                
            # Extract only the generated tokens (skip input tokens)
            generated_tokens = outputs[0, input_len:]
            logger.debug(f"Generated token count: {len(generated_tokens)}")
            
            # Log raw token IDs and their decoded values (don't skip special tokens)
            logger.debug(f"Raw token IDs: {generated_tokens.tolist()}")
            raw_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            logger.debug(f"Raw decoded (with special tokens): '{raw_decoded}'")
            
            # If only 1-2 tokens generated (likely just EOS), retry with more aggressive settings
            if len(generated_tokens) <= 2:
                logger.warning("Only 1-2 tokens generated, retrying with more aggressive settings")
                
                # More aggressive configuration
                aggressive_config = {
                    "temperature": 1.0,  # Higher temperature for more variability
                    "top_p": 0.95,  # Broader selection of tokens
                    "repetition_penalty": 1.05,  # Discourage repetition
                    "do_sample": True,  # Force sampling
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens
                }
                
                # Nudge the prompt to encourage explanation
                nudged_prompt = prompt + "\nPlease provide a detailed explanation for this trading decision:\n"
                
                # Tokenize input with nudged prompt
                nudged_inputs = self.tokenizer(nudged_prompt, return_tensors="pt").to(self.device)
                nudged_input_len = nudged_inputs.input_ids.shape[1]
                
                # Generate with aggressive settings
                with torch.no_grad():
                    outputs = self.model.generate(
                        **nudged_inputs,
                        **aggressive_config
                    )
                
                # Extract only the generated tokens
                generated_tokens = outputs[0, nudged_input_len:]
                logger.debug(f"Retry generated token count: {len(generated_tokens)}")
                logger.debug(f"Retry raw token IDs: {generated_tokens.tolist()}")
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.debug(f"Generated text length: {len(generated_text)} characters")
            
            if not generated_text.strip():
                return "Analysis not available at this time."
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Analysis not available at this time."

    def format_prompt(
        self,
        observation,
        market_data,
        action,
        decision=None,
        asset_name=None
    ):
        """
        Format a prompt for the language model to explain a trading decision.
        
        Args:
            observation: Observation vector from the environment
            market_data: Market data with technical indicators
            action: Action value from policy
            decision: Trading decision label (optional)
            asset_name: Name of the asset being analyzed
            
        Returns:
            Formatted prompt string
        """
        try:
            # Record timestamp for consistency check
            timestamp = None
            if isinstance(market_data, pd.DataFrame) and 'timestamp' in market_data.columns:
                timestamp = market_data['timestamp'].iloc[-1]
                logger.debug(f"Using market data from timestamp: {timestamp}")
            
            # Format action and decision info
            original_decision = decision
            if decision is None:
                # Use standardized mapping logic for consistency with RLLMExplainer
                if action > 0.6:
                    decision = "BUY (Strong)"
                elif action > 0.2:
                    decision = "BUY (Moderate)"
                elif action > 0:
                    decision = "BUY (Light)"
                elif action < -0.6:
                    decision = "SELL (Strong)"
                elif action < -0.2:
                    decision = "SELL (Moderate)"
                elif action < 0:
                    decision = "SELL (Light)"
                else:
                    decision = "HOLD"
                logger.debug(f"Mapped action {action:.4f} to decision: {decision}")
            else:
                # Log if decision was provided externally
                logger.debug(f"Using provided decision '{decision}' for action {action:.4f}")
                
            # Verify action and decision alignment
            expected_decision = None
            if action > 0.6:
                expected_decision = "BUY (Strong)"
            elif action > 0.2:
                expected_decision = "BUY (Moderate)"
            elif action > 0:
                expected_decision = "BUY (Light)"
            elif action < -0.6:
                expected_decision = "SELL (Strong)"
            elif action < -0.2:
                expected_decision = "SELL (Moderate)"
            elif action < 0:
                expected_decision = "SELL (Light)"
            else:
                expected_decision = "HOLD"
                
            if decision != expected_decision:
                logger.warning(f"Decision-action mismatch: decision '{decision}' doesn't match expected '{expected_decision}' for action {action:.4f}")
                    
            # Use asset name if provided, otherwise generic "the asset"
            if asset_name is None:
                asset_name = "the asset"
                
            # Format technical indicators information
            indicators = ""
            indicators_states = []
            indicator_values = {}  # Store indicator values for debugging
            
            # RSI
            if 'rsi' in market_data.columns:
                rsi_value = market_data['rsi'].iloc[-1]
                rsi_info = f"RSI is {rsi_value:.2f}"
                indicator_values['rsi'] = rsi_value
                
                # Track the indicator state for consistency check
                if rsi_value > 70:
                    rsi_info += " (overbought)"
                    indicators_states.append("bearish")
                elif rsi_value < 30:
                    rsi_info += " (oversold)"
                    indicators_states.append("bullish")
                
                indicators += rsi_info + ". "
                
            # MACD
            if all(col in market_data.columns for col in ['macd', 'macd_signal']):
                macd = market_data['macd'].iloc[-1]
                signal = market_data['macd_signal'].iloc[-1]
                macd_cross = "above" if macd > signal else "below"
                indicator_values['macd'] = macd
                indicator_values['macd_signal'] = signal
                
                # Track the indicator state for consistency check
                if macd > signal:
                    indicators_states.append("bullish")
                else:
                    indicators_states.append("bearish")
                    
                indicators += f"MACD is {macd:.4f} which is {macd_cross} signal line ({signal:.4f}). "
                
            # Bollinger Bands
            if all(col in market_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                upper = market_data['bb_upper'].iloc[-1]
                middle = market_data['bb_middle'].iloc[-1]
                lower = market_data['bb_lower'].iloc[-1]
                price = market_data['close'].iloc[-1] if 'close' in market_data.columns else None
                
                if price is not None:
                    indicator_values['bb_upper'] = upper
                    indicator_values['bb_middle'] = middle
                    indicator_values['bb_lower'] = lower
                    
                    if price > upper:
                        bb_info = f"Price is above upper Bollinger Band ({upper:.2f})"
                        indicators_states.append("bearish")  # Potential overbought
                    elif price < lower:
                        bb_info = f"Price is below lower Bollinger Band ({lower:.2f})"
                        indicators_states.append("bullish")  # Potential oversold
                    else:
                        bb_info = f"Price is within Bollinger Bands (middle: {middle:.2f})"
                    indicators += bb_info + ". "
                    
            # Trend
            if 'trend' in market_data.columns:
                trend = market_data['trend'].iloc[-1]
                indicator_values['trend'] = trend
                
                if trend > 0.5:
                    trend_desc = "strong upward"
                    indicators_states.append("bullish")
                elif trend > 0:
                    trend_desc = "moderate upward"
                    indicators_states.append("bullish")
                elif trend < -0.5:
                    trend_desc = "strong downward"
                    indicators_states.append("bearish")
                elif trend < 0:
                    trend_desc = "moderate downward"
                    indicators_states.append("bearish")
                else:
                    trend_desc = "neutral"
                indicators += f"Current trend is {trend_desc}. "
                
            # Support/Resistance
            if 'support' in market_data.columns and 'resistance' in market_data.columns:
                support = market_data['support'].iloc[-1]
                resistance = market_data['resistance'].iloc[-1]
                price = market_data['close'].iloc[-1] if 'close' in market_data.columns else None
                
                if price is not None:
                    indicator_values['support'] = support
                    indicator_values['resistance'] = resistance
                    
                    distance_to_support = (price - support) / price * 100 if support > 0 else float('inf')
                    distance_to_resistance = (resistance - price) / price * 100 if resistance > 0 else float('inf')
                    
                    if distance_to_support < 2:
                        indicators += f"Price is near support level ({support:.2f}). "
                        indicators_states.append("bullish")  # Near support is potential buy
                    if distance_to_resistance < 2:
                        indicators += f"Price is near resistance level ({resistance:.2f}). "
                        indicators_states.append("bearish")  # Near resistance is potential sell
                        
            # Format market context
            context = ""
            if 'close' in market_data.columns:
                price = market_data['close'].iloc[-1]
                indicator_values['price'] = price
                context += f"Current price of {asset_name} is {price:.2f}. "
                
                # Recent price changes
                if len(market_data) > 1:
                    day_change = (price - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2] * 100
                    indicator_values['price_change_1d'] = day_change
                    context += f"Price changed {day_change:.2f}% from previous period. "
                    
                    # Track recent price change for consistency check
                    if day_change > 1:
                        indicators_states.append("bullish")
                    elif day_change < -1:
                        indicators_states.append("bearish")
                    
                # Longer trends if available
                if len(market_data) >= 5:
                    week_change = (price - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5] * 100
                    indicator_values['price_change_5d'] = week_change
                    context += f"5-period change is {week_change:.2f}%. "
                    
                    # Track longer price change for consistency check
                    if week_change > 3:
                        indicators_states.append("bullish")
                    elif week_change < -3:
                        indicators_states.append("bearish")
                    
            # Add volume info if available
            if 'volume' in market_data.columns:
                volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].mean() if len(market_data) > 1 else volume
                vol_ratio = volume / avg_volume
                indicator_values['volume_ratio'] = vol_ratio
                
                if vol_ratio > 1.5:
                    vol_desc = "significantly higher than average"
                elif vol_ratio > 1.1:
                    vol_desc = "higher than average"
                elif vol_ratio < 0.5:
                    vol_desc = "significantly lower than average"
                elif vol_ratio < 0.9:
                    vol_desc = "lower than average"
                else:
                    vol_desc = "around average"
                    
                context += f"Trading volume is {vol_desc}. "
            
            # Check for potential inconsistency between decision and indicators
            if indicators_states:
                bullish_count = indicators_states.count("bullish")
                bearish_count = indicators_states.count("bearish")
                decision_sentiment = "bullish" if decision.startswith("BUY") else "bearish" if decision.startswith("SELL") else "neutral"
                
                # Compute synthetic action from indicators to check alignment
                synthetic_components = []
                
                # Add components based on the same indicators used above
                if 'rsi' in indicator_values:
                    rsi = indicator_values['rsi']
                    if rsi > 70:
                        synthetic_components.append(-0.3)  # Bearish
                    elif rsi < 30:
                        synthetic_components.append(0.3)   # Bullish
                    else:
                        synthetic_components.append(0.3 - ((rsi - 30) / 40) * 0.6)
                
                if 'macd' in indicator_values and 'macd_signal' in indicator_values:
                    macd_diff = indicator_values['macd'] - indicator_values['macd_signal']
                    synthetic_components.append(min(0.3, max(-0.3, macd_diff * 10)))
                
                if 'price_change_5d' in indicator_values:
                    price_change = indicator_values['price_change_5d']
                    synthetic_components.append(min(0.5, max(-0.5, price_change * 0.1)))
                
                # Calculate synthetic action
                if synthetic_components:
                    synthetic_action = sum(synthetic_components) / len(synthetic_components)
                    synthetic_action = min(0.8, max(-0.8, synthetic_action))
                    
                    # Expected decision based on synthetic action
                    synthetic_decision = None
                    if synthetic_action > 0.6:
                        synthetic_decision = "BUY (Strong)"
                    elif synthetic_action > 0.2:
                        synthetic_decision = "BUY (Moderate)"
                    elif synthetic_action > 0:
                        synthetic_decision = "BUY (Light)"
                    elif synthetic_action < -0.6:
                        synthetic_decision = "SELL (Strong)"
                    elif synthetic_action < -0.2:
                        synthetic_decision = "SELL (Moderate)"
                    elif synthetic_action < 0:
                        synthetic_decision = "SELL (Light)"
                    else:
                        synthetic_decision = "HOLD"
                    
                    # Log the alignment between actual action and what indicators suggest
                    if decision != synthetic_decision:
                        logger.warning(f"Decision-indicator mismatch: decision '{decision}' differs from indicator-based '{synthetic_decision}' (action: {action:.4f}, synthetic: {synthetic_action:.4f})")
                    
                if (decision_sentiment == "bullish" and bearish_count > bullish_count) or \
                   (decision_sentiment == "bearish" and bullish_count > bearish_count):
                    logger.warning(f"Potential inconsistency: {decision} decision with {bullish_count} bullish vs {bearish_count} bearish indicators")
                    context += f"Note: This decision may seem counter-trend based on some indicators. "
                
            # Construct full prompt
            prompt = f"""
You are a professional trading advisor. 
Based on the following market data for {asset_name}, explain the {decision} trading decision with detailed reasoning.

Market Context:
{context}

Technical Indicators:
{indicators}

Your thorough analysis of why this {decision} decision makes sense for {asset_name}:
"""
            
            # Debug indicator values along with the prompt for better troubleshooting
            logger.debug(f"Indicator values for {asset_name}: {indicator_values}")
            logger.debug(f"Format prompt: {prompt}")
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return f"Explain why the trading decision for {asset_name} is {decision}."
    
    def _load_model(self) -> None:
        """Load the base model and tokenizer with appropriate quantization"""
        try:
            logger.info(f"Loading base model: {self.model_name}")
            
            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                use_fast=True,
                token=True  # Enable token-based authentication for gated models
            )
            
            # Make sure we have pad token for proper batching
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Load model with quantization
            model_kwargs = {
                "device_map": self.device_map,
                "quantization_config": quantization_config,
                "torch_dtype": torch.bfloat16,
                "cache_dir": self.cache_dir,
                "token": True,  # Enable token-based authentication for gated models
            }
            
            
            if self.use_flash_attn:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2 for faster training")
                except Exception as e:
                    logger.warning(f"Flash Attention 2 not available: {e}. Using default attention.")
                
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Store the model's device for reference
            self._device = None  # Will be determined on-demand via property
            
            # Enable gradient checkpointing for memory efficiency if requested
            if self.use_gradient_checkpointing:
                # Set use_cache=False when using gradient checkpointing
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                    logger.info("Disabled model cache for gradient checkpointing compatibility")
                
                # Enable gradient checkpointing with safer approach for compatibility
                try:
                    # Newer transformers versions support use_reentrant
                    self.model.gradient_checkpointing_enable(use_reentrant=False)
                except TypeError:
                    # Fall back to older version without use_reentrant parameter
                    self.model.gradient_checkpointing_enable()
                    
                logger.info("Gradient checkpointing enabled for memory efficiency")
            
            # Configure generation parameters optimized for Llama-3
            self.generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512
            )
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @property
    def device(self) -> torch.device:
        """
        Get the device where the model is located.
        Determines the device on first call and caches the result.
        
        Returns:
            The torch device (e.g., 'cuda:0', 'cpu') where the model is loaded
        """
        if self._device is None and self.model is not None:
            # For model with multiple device assignments, get the first parameter's device
            self._device = next(self.model.parameters()).device
            logger.debug(f"Model is on device: {self._device}")
        return self._device if self._device is not None else torch.device('cpu')
    
    def apply_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """
        Apply LoRA adapters to the model for efficient fine-tuning
        
        Args:
            target_modules: List of module names to apply LoRA to
        """
        if self.is_lora_applied:
            logger.warning("LoRA has already been applied to the model")
            return
            
        try:
            logger.info("Applying LoRA adapters to model")
            
            # Default target modules for Llama-3
            if target_modules is None:
                if "llama" in self.model_name.lower():
                    target_modules = [
                        "q_proj", 
                        "k_proj", 
                        "v_proj", 
                        "o_proj",
                        "gate_proj", 
                        "up_proj", 
                        "down_proj"
                    ]
                else:
                    # Default for other models like Mistral
                    target_modules = [
                        "q_proj", 
                        "k_proj", 
                        "v_proj", 
                        "o_proj",
                        "gate_proj", 
                        "up_proj", 
                        "down_proj"
                    ]
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Prepare model for training if using quantization
            if self.load_in_8bit or self.load_in_4bit:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.is_lora_applied = True
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("LoRA adapters applied successfully")
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        asset_name: Optional[str] = None,
    ) -> str:
        """
        Generate general text completion based on given prompt.
        Used by chatbot for interactive conversations.
        
        Args:
            prompt: Input prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            asset_name: Name of the asset being discussed (for context awareness)
            
        Returns:
            Generated text completion
        """
        try:
            # Create generation config
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens
            )
            
            # Add asset name to prompt if provided and not already in prompt
            if asset_name and asset_name not in prompt:
                # Include asset name in a non-disruptive way if not already present
                if "[context]" in prompt.lower() or "current information" in prompt.lower():
                    # Already has context section - no need to modify
                    pass
                else:
                    # Add a subtle reference to make model aware of the asset
                    prompt = f"Regarding {asset_name}: {prompt}"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Extract only the generated tokens (excluding input tokens)
            generated_tokens = output[0][input_length:]
            
            # Decode only the generated part
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up special tokens that might remain
            response = response.replace("</s>", "").strip()
            
            return response
        except Exception as e:
            import traceback
            logger.error(f"Error generating text: {str(e)}")
            logger.debug(traceback.format_exc())
            return "I'm sorry, I encountered an error while trying to respond."
    
    def save_model(self, save_dir: str) -> None:
        """
        Save the model and tokenizer
        
        Args:
            save_dir: Directory to save the model
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            logger.info(f"Saving model to {save_dir}")
            
            # Save model
            if self.is_lora_applied:
                self.model.save_pretrained(save_dir)
            else:
                logger.warning("Saving full model (not LoRA adapters) - this may be very large")
                self.model.save_pretrained(save_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_dir)
            
            # Save generation config
            if self.generation_config:
                self.generation_config.save_pretrained(save_dir)
                
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_from_pretrained(
        cls,
        model_dir: str,
        base_model: Optional[str] = None,
        device_map: str = "auto",
        **kwargs
    ) -> "TradingLLM":
        """
        Load a fine-tuned model from directory
        
        Args:
            model_dir: Directory containing the saved model
            base_model: Optional base model name if not loading full model
            device_map: Device mapping strategy
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded TradingLLM instance
        """
        try:
            logger.info(f"Loading model from {model_dir}")
            
            # Determine if loading LoRA or full model
            is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
            
            if is_lora and base_model is None:
                raise ValueError("Base model must be provided when loading LoRA adapters")
            
            # Initialize with base model
            model_name = base_model if is_lora else model_dir
            instance = cls(model_name=model_name, device_map=device_map, **kwargs)
            
            # Load adapters if this is a LoRA model
            if is_lora:
                logger.info("Loading LoRA adapters")
                instance.model = PeftModel.from_pretrained(
                    instance.model,
                    model_dir,
                    device_map=device_map
                )
                instance.is_lora_applied = True
            
            # Load generation config if available
            gen_config_path = os.path.join(model_dir, "generation_config.json")
            if os.path.exists(gen_config_path):
                instance.generation_config = GenerationConfig.from_pretrained(model_dir)
                
            logger.info("Model loaded successfully")
            return instance
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {str(e)}")
            raise 