"""
Trading LLM Model - Integrates Mistral-7B with RL trading model
"""

import os
import torch
import logging
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
    Explanation model for RL trading decisions using Mistral-7B with LoRA fine-tuning
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
            cache_dir: Directory to cache models
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
            
            # Enable gradient checkpointing for memory efficiency if requested
            if self.use_gradient_checkpointing:
                # Set use_cache=False when using gradient checkpointing
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                    logger.info("Disabled model cache for gradient checkpointing compatibility")
                
                # Enable gradient checkpointing with explicit use_reentrant parameter
                self.model.gradient_checkpointing_enable(use_reentrant=False)
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
    
    def generate_explanation(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Generate trading explanation based on given prompt
        
        Args:
            prompt: Input prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            
        Returns:
            Generated explanation text
        """
        try:
            # Create custom generation config if parameters provided
            generation_config = self.generation_config
            if any(param is not None for param in [temperature, top_p, repetition_penalty, do_sample]):
                generation_config = GenerationConfig(
                    temperature=temperature if temperature is not None else self.generation_config.temperature,
                    top_p=top_p if top_p is not None else self.generation_config.top_p,
                    repetition_penalty=repetition_penalty if repetition_penalty is not None else self.generation_config.repetition_penalty,
                    do_sample=do_sample if do_sample is not None else self.generation_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens
                )
            
            # Tokenize input with proper handling of token type IDs
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode the response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part after the prompt
            prompt_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            explanation = response[prompt_length:].strip()
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
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
            
    def format_prompt(
        self, 
        market_context: Dict[str, Any], 
        action: float,
        indicators: Dict[str, Any] = None,
        timeframe: str = "current"
    ) -> str:
        """
        Format a prompt for the LLM based on market context and action
        
        Args:
            market_context: Dictionary containing market data
            action: Action value from RL model
            indicators: Dictionary of technical indicators
            timeframe: Time context for the analysis
            
        Returns:
            Formatted prompt string
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
            
        # Format indicators if provided
        indicators_text = ""
        if indicators:
            indicators_text = "Technical indicators:\n"
            for name, value in indicators.items():
                if isinstance(value, float):
                    indicators_text += f"- {name}: {value:.2f}\n"
                else:
                    indicators_text += f"- {name}: {value}\n"
        
        # Create the prompt
        prompt = f"""<s>[INST] I'm analyzing the market for trading opportunities.

Here's the {timeframe} market situation:
- Price: {market_context.get('price', 'N/A')}
- Recent price change: {market_context.get('price_change', 'N/A')}
- Volume: {market_context.get('volume', 'N/A')}
- Market volatility: {market_context.get('volatility', 'N/A')}
{indicators_text}

The trading algorithm decided to {action_desc}.

Can you explain why this might be a good decision given the current market conditions? Give a concise explanation based on technical analysis and price action. [/INST]
"""
        return prompt 