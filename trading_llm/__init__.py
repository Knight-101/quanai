"""
Trading LLM - Generating explanations for RL trading using language models
"""

__version__ = "0.1.0"

# Import main classes
from trading_llm.model import TradingLLM
from trading_llm.dataset import (
    TradingDatasetGenerator, 
    TradingTextDataset, 
    TechnicalIndicatorProcessor,
    create_dataloaders
)
from trading_llm.training import TradingTrainer, EvaluationMetrics
from trading_llm.inference import RLStateExtractor, RLLMExplainer, MarketCommentaryGenerator
from trading_llm.utils.logging import setup_logging, TrainingLogger

# For convenience, import main CLI entry point
from trading_llm.train_llm import main

# Export public API
__all__ = [
    # Core model
    'TradingLLM',
    
    # Dataset utilities
    'TradingDatasetGenerator',
    'TradingTextDataset',
    'TechnicalIndicatorProcessor',
    'create_dataloaders',
    
    # Training utilities
    'TradingTrainer',
    'EvaluationMetrics',
    
    # Inference classes
    'RLStateExtractor',
    'RLLMExplainer',
    'MarketCommentaryGenerator',
    
    # Logging utilities
    'setup_logging',
    'TrainingLogger',
    
    # CLI entry point
    'main',
] 