import wandb
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WandBLogger:
    def __init__(self, config: Dict[str, Any], project: str, entity: str = None):
        self.enabled = True
        try:
            wandb.init(
                project=project,
                entity=entity,
                config=config,
                resume=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {str(e)}")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        if not self.enabled:
            return
            
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {str(e)}")

    def log_model_gradients(self, model):
        if not self.enabled:
            return
            
        try:
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())
            self.log_metrics(gradients)
        except Exception as e:
            logger.warning(f"Failed to log gradients: {str(e)}")

    def log_training_step(self, metrics: Dict[str, float], step: int):
        if not self.enabled:
            return
            
        training_metrics = {
            f"training/{k}": v for k, v in metrics.items()
        }
        self.log_metrics(training_metrics, step)

    def log_evaluation(self, metrics: Dict[str, float], step: int):
        if not self.enabled:
            return
            
        eval_metrics = {
            f"eval/{k}": v for k, v in metrics.items()
        }
        self.log_metrics(eval_metrics, step)

    def finish(self):
        if self.enabled:
            wandb.finish()
