"""
Enhanced Trainer implementation with detailed logging capabilities
"""

import os
import torch
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class GradientLoggerCallback(TrainerCallback):
    """Callback that logs gradient statistics during training"""
    
    def __init__(self, log_every_n_steps: int = 100):
        """
        Initialize the gradient logger
        
        Args:
            log_every_n_steps: Frequency of logging
        """
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log gradients at the end of each step"""
        if state.global_step % self.log_every_n_steps == 0 and model is not None:
            # Unwrap model if needed
            if hasattr(model, "module"):
                model = model.module
            
            # Compute gradient stats
            grad_norm = 0.0
            grad_max = 0.0
            grad_mean = 0.0
            grad_median = 0.0
            param_count = 0
            
            # Collect gradients from all parameters
            all_grads = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach()
                    all_grads.append(grad.flatten())
                    
                    # Update running stats
                    grad_norm += torch.norm(grad).item() ** 2
                    grad_max = max(grad_max, torch.max(torch.abs(grad)).item())
                    grad_mean += torch.mean(torch.abs(grad)).item()
                    param_count += 1
            
            # Compute final stats
            if param_count > 0:
                grad_norm = np.sqrt(grad_norm)
                grad_mean /= param_count
                
                # Compute median by concatenating all gradients
                if all_grads:
                    all_grads_tensor = torch.cat(all_grads)
                    grad_median = torch.median(torch.abs(all_grads_tensor)).item()
                
                # Log stats
                logger.info(f"Gradient stats at step {state.global_step}:")
                logger.info(f"  - Norm: {grad_norm:.4f}")
                logger.info(f"  - Max: {grad_max:.4f}")
                logger.info(f"  - Mean: {grad_mean:.4f}")
                logger.info(f"  - Median: {grad_median:.4f}")
                
                # Log to tensorboard if available
                if hasattr(args, "report_to") and "tensorboard" in args.report_to:
                    try:
                        tb_writer = kwargs.get("tb_writer", None)
                        if tb_writer:
                            tb_writer.add_scalar("gradients/norm", grad_norm, state.global_step)
                            tb_writer.add_scalar("gradients/max", grad_max, state.global_step)
                            tb_writer.add_scalar("gradients/mean", grad_mean, state.global_step)
                            tb_writer.add_scalar("gradients/median", grad_median, state.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log gradients to tensorboard: {e}")
                
                # Log to wandb if available
                if hasattr(args, "report_to") and "wandb" in args.report_to:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "gradients/norm": grad_norm,
                                "gradients/max": grad_max,
                                "gradients/mean": grad_mean,
                                "gradients/median": grad_median,
                                "global_step": state.global_step
                            })
                    except Exception as e:
                        logger.warning(f"Failed to log gradients to wandb: {e}")


class LoggableTrainer(Trainer):
    """Enhanced Trainer with better logging capabilities"""
    
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        log_gradients: bool = True,
        log_every_n_steps: int = 100
    ):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            args: Training arguments
            data_collator: Data collator
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            model_init: Model initialization function
            compute_metrics: Metrics computation function
            callbacks: List of callbacks
            optimizers: Tuple of (optimizer, scheduler)
            log_gradients: Whether to log gradients
            log_every_n_steps: Frequency of gradient logging
        """
        # Initialize parent class
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers
        )
        
        # Add gradient logging callback if requested
        self.log_gradients = log_gradients
        if log_gradients:
            grad_logger = GradientLoggerCallback(log_every_n_steps=log_every_n_steps)
            self.add_callback(grad_logger)
        
        # Track training performance
        self.train_start_time = None
        self.last_log_time = None
        self.log_every_n_steps = log_every_n_steps
        self.steps_since_last_log = 0
        self.loss_since_last_log = 0.0
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to add more detailed logging"""
        # Call parent implementation
        output = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        # Track loss and performance
        if self.control.should_log:
            # Calculate throughput
            if self.train_start_time is None:
                self.train_start_time = time.time()
                self.last_log_time = self.train_start_time
            
            # Calculate time elapsed since last log
            current_time = time.time()
            time_elapsed = current_time - self.last_log_time
            total_time = current_time - self.train_start_time
            
            # Calculate training speed
            if time_elapsed > 0:
                steps_per_second = self.steps_since_last_log / time_elapsed
                examples_per_second = (
                    self.steps_since_last_log 
                    * self.args.per_device_train_batch_size 
                    * self.args.gradient_accumulation_steps 
                    * (self.args.world_size if hasattr(self.args, "world_size") else 1)
                ) / time_elapsed
                
                # Log performance metrics
                logger.info(f"Performance at step {self.state.global_step}:")
                logger.info(f"  - Steps/second: {steps_per_second:.2f}")
                logger.info(f"  - Examples/second: {examples_per_second:.2f}")
                logger.info(f"  - Time elapsed: {time_elapsed:.2f}s (total: {total_time:.2f}s)")
                
                # Log to tensorboard if available
                if hasattr(self.args, "report_to") and "tensorboard" in self.args.report_to:
                    try:
                        self.get_train_dataloader()
                        self.tb_writer.add_scalar("performance/steps_per_second", steps_per_second, self.state.global_step)
                        self.tb_writer.add_scalar("performance/examples_per_second", examples_per_second, self.state.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log performance to tensorboard: {e}")
                
                # Log to wandb if available
                if hasattr(self.args, "report_to") and "wandb" in self.args.report_to:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "performance/steps_per_second": steps_per_second,
                                "performance/examples_per_second": examples_per_second,
                                "performance/time_elapsed": time_elapsed,
                                "performance/total_time": total_time,
                                "global_step": self.state.global_step
                            })
                    except Exception as e:
                        logger.warning(f"Failed to log performance to wandb: {e}")
            
            # Reset counters
            self.last_log_time = current_time
            self.steps_since_last_log = 0
            self.loss_since_last_log = 0.0
            
        return output
    
    def training_step(self, model, inputs):
        """Override training step to track performance metrics"""
        # Call parent implementation
        loss = super().training_step(model, inputs)
        
        # Update counters
        self.steps_since_last_log += 1
        self.loss_since_last_log += loss.item()
        
        return loss
    
    def log_metrics(self, split, metrics):
        """Override to add more detailed metric logging"""
        # Call parent implementation
        super().log_metrics(split, metrics)
        
        # Log each metric in detail
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value}")
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override to add more detailed saving information"""
        # Log save operation
        if output_dir is None:
            output_dir = self.args.output_dir
        
        logger.info(f"Saving model to {output_dir}")
        
        # Call parent implementation
        output = super().save_model(output_dir, _internal_call)
        
        # Log the size of the saved model
        try:
            model_size = sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            ) / (1024 * 1024)  # Convert to MB
            
            logger.info(f"Model saved successfully. Size: {model_size:.2f} MB")
            
            # Log to wandb if available
            if hasattr(self.args, "report_to") and "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "model/size_mb": model_size,
                            "global_step": self.state.global_step
                        })
                except Exception as e:
                    logger.warning(f"Failed to log model size to wandb: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
        
        return output
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to add more detailed evaluation logging"""
        # Log evaluation start
        logger.info(f"Starting evaluation at step {self.state.global_step}")
        eval_start_time = time.time()
        
        # Call parent implementation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Log evaluation time
        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Add evaluation time to metrics
        if output is not None:
            output["eval_time"] = eval_time
            
            # Log to wandb if available
            if hasattr(self.args, "report_to") and "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            f"{metric_key_prefix}/time": eval_time,
                            "global_step": self.state.global_step
                        })
                except Exception as e:
                    logger.warning(f"Failed to log evaluation time to wandb: {e}")
        
        return output 