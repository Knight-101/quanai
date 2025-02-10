import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WandBCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        """Log training metrics to wandb on each step"""
        try:
            # Log episode rewards
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                wandb.log({
                    "train/episode_reward": np.mean([ep['r'] for ep in self.model.ep_info_buffer]),
                    "train/episode_length": np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                }, step=self.num_timesteps)
            return True
        except Exception as e:
            logger.warning(f"Failed to log to wandb in callback: {str(e)}")
            return True

class WandBLogger:
    """Weights & Biases logger for training metrics"""
    def __init__(self, config: Dict[str, Any], project: str = None, entity: str = None):
        self.config = config
        self.project = project
        self.entity = entity
        
        try:
            wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=f"training_run_{wandb.util.generate_id()}",
                reinit=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {str(e)}")
            
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to wandb"""
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {str(e)}")
            
    def log_model_gradients(self, model):
        """Log model gradients"""
        try:
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())
            wandb.log(gradients)
        except Exception as e:
            logger.warning(f"Failed to log gradients to wandb: {str(e)}")
            
    def log_training_step(self, metrics: Dict[str, float], step: int):
        """Log training step metrics"""
        try:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log training step to wandb: {str(e)}")
            
    def create_callback(self) -> WandBCallback:
        """Create a wandb callback for training"""
        return WandBCallback()
            
    def finish(self):
        """Finish logging and close wandb run"""
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {str(e)}")
