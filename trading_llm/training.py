"""
Training module for Trading LLM with efficient LoRA fine-tuning
"""

import os
import time
import logging
import torch
import wandb
import random
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainerState,
    TrainerControl
)

from training.loggable_trainer import LoggableTrainer
from trading_llm.model import TradingLLM

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TradingTrainer:
    """Trainer for Trading LLM model"""
    
    def __init__(
        self,
        model: Union[TradingLLM, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        output_dir: str = "./outputs",
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        early_stopping_patience: Optional[int] = 3,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        seed: int = 42,
        fp16: bool = False,
        bf16: bool = False
    ):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            tokenizer: Tokenizer to use
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_ratio: Ratio of steps for LR warmup
            weight_decay: Weight decay
            gradient_accumulation_steps: Number of steps for gradient accumulation
            max_grad_norm: Maximum gradient norm
            eval_steps: Steps between evaluations
            save_steps: Steps between model saving
            logging_steps: Steps between logging
            early_stopping_patience: Patience for early stopping
            use_wandb: Whether to use wandb
            wandb_project: wandb project name
            wandb_entity: wandb entity name
            seed: Random seed
            fp16: Whether to use FP16 training
            bf16: Whether to use BF16 training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.early_stopping_patience = early_stopping_patience
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.seed = seed
        self.fp16 = fp16
        self.bf16 = bf16
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        set_seed(seed)
        
        # Initialize wandb if needed
        if use_wandb:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize wandb for tracking"""
        try:
            wandb.init(
                project=self.wandb_project or "trading-llm",
                entity=self.wandb_entity,
                config={
                    "model_name": self.model.model_name if hasattr(self.model, "model_name") else "custom",
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "warmup_ratio": self.warmup_ratio,
                    "weight_decay": self.weight_decay,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "seed": self.seed,
                    "fp16": self.fp16,
                    "bf16": self.bf16
                }
            )
            logger.info("Initialized wandb tracking")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {str(e)}")
            self.use_wandb = False
    
    def train(self) -> Dict[str, float]:
        """
        Train the model
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting training")
        
        # Apply LoRA if using TradingLLM
        if isinstance(self.model, TradingLLM) and not self.model.is_lora_applied:
            self.model.apply_lora()
        
        # Ensure model is properly loaded on GPU
        if hasattr(self.model, "model"):
            model_to_train = self.model.model  # For TradingLLM wrapper
        else:
            model_to_train = self.model
        
        # Get total number of training steps
        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model_to_train.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup training arguments for Trainer
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_dataloader.batch_size,
            per_device_eval_batch_size=self.val_dataloader.batch_size if self.val_dataloader else self.train_dataloader.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_steps=self.eval_steps if self.val_dataloader else None,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            report_to="wandb" if self.use_wandb else "none",
            remove_unused_columns=False,  # Important for custom datasets
            label_names=["labels"],
            push_to_hub=False,
            save_total_limit=3,  # Keep only the 3 most recent checkpoints
            load_best_model_at_end=True if self.val_dataloader else False,
            metric_for_best_model="eval_loss" if self.val_dataloader else None,
            greater_is_better=False if self.val_dataloader else None,
            logging_first_step=True,
            # Use eval_strategy instead of evaluation_strategy (deprecated)
            eval_strategy="steps" if self.val_dataloader else "no",
            save_strategy="steps",
        )
        
        # Create trainer
        callbacks = []
        if self.early_stopping_patience and self.val_dataloader:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience
            ))
            
        # Use LoggableTrainer to get detailed logs
        trainer = LoggableTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.val_dataloader.dataset if self.val_dataloader else None,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, lr_scheduler),
            callbacks=callbacks
        )
        
        # Start training
        start_time = time.time()
        train_result = trainer.train()
        end_time = time.time()
        
        # Log training results
        metrics = train_result.metrics
        metrics["training_time"] = end_time - start_time
        
        logger.info(f"Training completed in {metrics['training_time']:.2f} seconds")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")
        
        # Save the final model
        if isinstance(self.model, TradingLLM):
            self.model.save_model(os.path.join(self.output_dir, "final_model"))
        else:
            trainer.save_model(os.path.join(self.output_dir, "final_model"))
            
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.val_dataloader:
            logger.warning("No validation dataloader provided for evaluation")
            return {}
        
        logger.info("Starting evaluation")
        
        # Ensure model is in evaluation mode
        if hasattr(self.model, "model"):
            model_to_eval = self.model.model  # For TradingLLM wrapper
        else:
            model_to_eval = self.model
        
        model_to_eval.eval()
        
        # Setup metrics
        eval_loss = 0
        num_eval_steps = 0
        
        # Evaluate
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(model_to_eval.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model_to_eval(**batch)
                
                # Update metrics
                eval_loss += outputs.loss.item()
                num_eval_steps += 1
        
        # Calculate final metrics
        metrics = {
            "eval_loss": eval_loss / num_eval_steps,
            "perplexity": torch.exp(torch.tensor(eval_loss / num_eval_steps)).item()
        }
        
        # Log metrics
        logger.info("Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics)
        
        return metrics

    @classmethod
    def load_and_train(
        cls,
        model_name: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        output_dir: str = "./outputs",
        **kwargs
    ) -> "TradingTrainer":
        """
        Load a model and train it
        
        Args:
            model_name: Name or path of base model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout parameter
            output_dir: Directory to save model
            **kwargs: Additional arguments for trainer
            
        Returns:
            Trained TradingTrainer instance
        """
        # Initialize model
        model = TradingLLM(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Create trainer
        trainer = cls(
            model=model,
            tokenizer=model.tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=output_dir,
            **kwargs
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        if val_dataloader:
            trainer.evaluate()
        
        return trainer


class EvaluationMetrics:
    """Evaluate explanation quality metrics"""
    
    @staticmethod
    def relevance_score(explanation: str, indicators: Dict[str, Any]) -> float:
        """
        Calculate relevance score based on how well the explanation uses technical indicators
        
        Args:
            explanation: Generated explanation text
            indicators: Dictionary of technical indicators
            
        Returns:
            Relevance score between 0 and 1
        """
        # List of key indicator terms to check for
        key_terms = [
            "RSI", "MACD", "moving average", "MA", "support", "resistance",
            "oversold", "overbought", "bullish", "bearish", "trend",
            "crossover", "divergence", "momentum", "volume", "volatility"
        ]
        
        # Add specific indicator values if they exist in explanation
        for indicator, value in indicators.items():
            if isinstance(value, (float, int)) and str(round(value, 1)) in explanation:
                key_terms.append(indicator)
        
        # Count how many key terms appear in the explanation
        count = sum(1 for term in key_terms if term.lower() in explanation.lower())
        
        # Calculate score (max out at 5 terms for a perfect score)
        relevance = min(count / 5, 1.0)
        
        return relevance
    
    @staticmethod
    def consistency_score(explanation: str, action: float) -> float:
        """
        Calculate consistency between explanation and action
        
        Args:
            explanation: Generated explanation text
            action: RL model action
            
        Returns:
            Consistency score between 0 and 1
        """
        # Identify if explanation is bullish or bearish
        bullish_terms = ["long", "buy", "bullish", "uptrend", "upward", "rise", "increase", "higher"]
        bearish_terms = ["short", "sell", "bearish", "downtrend", "downward", "fall", "decrease", "lower"]
        
        bullish_count = sum(1 for term in bullish_terms if term.lower() in explanation.lower())
        bearish_count = sum(1 for term in bearish_terms if term.lower() in explanation.lower())
        
        explanation_sentiment = bullish_count - bearish_count
        
        # Check consistency with action
        if (action > 0 and explanation_sentiment > 0) or (action < 0 and explanation_sentiment < 0):
            return 1.0
        elif explanation_sentiment == 0:
            return 0.5  # Neutral explanation
        else:
            return 0.0  # Inconsistent
    
    @staticmethod
    def diversity_score(explanations: List[str]) -> float:
        """
        Calculate diversity across multiple explanations
        
        Args:
            explanations: List of explanation texts
            
        Returns:
            Diversity score between 0 and 1
        """
        if len(explanations) <= 1:
            return 1.0  # Can't measure diversity with just one explanation
        
        # Calculate simple n-gram based diversity
        unique_bigrams = set()
        total_bigrams = 0
        
        for explanation in explanations:
            # Create bigrams from words
            words = explanation.lower().split()
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            
            # Update counts
            unique_bigrams.update(bigrams)
            total_bigrams += len(bigrams)
        
        # Calculate diversity as ratio of unique bigrams to total
        if total_bigrams == 0:
            return 0.0
        
        return len(unique_bigrams) / total_bigrams
    
    @staticmethod
    def evaluate_explanation(
        explanation: str,
        action: float,
        indicators: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate a single explanation
        
        Args:
            explanation: Generated explanation text
            action: RL model action
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        relevance = EvaluationMetrics.relevance_score(explanation, indicators)
        consistency = EvaluationMetrics.consistency_score(explanation, action)
        
        # Overall quality score
        quality = (relevance + consistency) / 2
        
        return {
            "relevance": relevance,
            "consistency": consistency,
            "quality": quality
        }
    
    @classmethod
    def evaluate_batch(
        cls,
        explanations: List[str],
        actions: List[float],
        indicators_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of explanations
        
        Args:
            explanations: List of generated explanations
            actions: List of RL model actions
            indicators_list: List of technical indicator dictionaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Individual metrics
        metrics_list = [
            cls.evaluate_explanation(e, a, i)
            for e, a, i in zip(explanations, actions, indicators_list)
        ]
        
        # Average metrics
        avg_metrics = {
            "relevance": sum(m["relevance"] for m in metrics_list) / len(metrics_list),
            "consistency": sum(m["consistency"] for m in metrics_list) / len(metrics_list),
            "quality": sum(m["quality"] for m in metrics_list) / len(metrics_list),
        }
        
        # Add diversity
        avg_metrics["diversity"] = cls.diversity_score(explanations)
        
        return avg_metrics 