"""
Logging utilities for Trading LLM module
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Union
import datetime
import json

# Configure default logger
logger = logging.getLogger("trading_llm")


def setup_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: Optional[str] = None,
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        console: Whether to log to console
        log_format: Log format string
        log_date_format: Date format string
        
    Returns:
        Configured logger
    """
    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Set default format if not provided
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=log_date_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure trading_llm logger
    trading_llm_logger = logging.getLogger("trading_llm")
    trading_llm_logger.setLevel(log_level)
    
    # Disable propagation to avoid duplicate logs
    trading_llm_logger.propagate = False
    
    return trading_llm_logger


class TrainingLogger:
    """Logger for training process"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: wandb project name
            wandb_entity: wandb entity name
            config: Configuration dictionary
        """
        # Set experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"trading_llm_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.config = config or {}
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        if config:
            config_path = os.path.join(self.log_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        
        # Initialize TensorBoard if requested
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                logging.warning("TensorBoard not available. Install with 'pip install tensorboard'")
                use_tensorboard = False
        
        # Initialize wandb if requested
        if use_wandb:
            try:
                import wandb
                if not wandb.api.api_key:
                    logging.warning("wandb API key not found. Set it with 'wandb login'")
                    use_wandb = False
                else:
                    wandb.init(
                        project=wandb_project or "trading-llm",
                        entity=wandb_entity,
                        name=experiment_name,
                        config=config,
                        dir=self.log_dir
                    )
            except ImportError:
                logging.warning("wandb not available. Install with 'pip install wandb'")
                use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to all enabled backends
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except (ImportError, Exception) as e:
                logging.warning(f"Error logging to wandb: {str(e)}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        with open(metrics_file, "a") as f:
            metrics_with_step = {"step": step, **metrics}
            f.write(json.dumps(metrics_with_step) + "\n")
    
    def log_model_info(self, model_type: str, model_config: Dict[str, Any]) -> None:
        """
        Log model information
        
        Args:
            model_type: Type of model
            model_config: Model configuration
        """
        # Save model info
        model_info = {
            "model_type": model_type,
            "config": model_config
        }
        info_path = os.path.join(self.log_dir, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.config.update({"model_type": model_type, **model_config})
            except (ImportError, Exception) as e:
                logging.warning(f"Error logging model info to wandb: {str(e)}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Save hyperparameters
        hp_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hp_path, "w") as f:
            json.dump(hyperparams, f, indent=2)
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.config.update(hyperparams)
            except (ImportError, Exception) as e:
                logging.warning(f"Error logging hyperparameters to wandb: {str(e)}")
    
    def close(self) -> None:
        """Close logger and all backends"""
        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        # Close wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except (ImportError, Exception) as e:
                logging.warning(f"Error closing wandb: {str(e)}") 