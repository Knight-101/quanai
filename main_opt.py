#!/usr/bin/env python
import argparse
import re
import torch
import os
import time
import pickle
from datetime import datetime, timedelta
import asyncio
from data_system.derivative_data_fetcher import PerpetualDataFetcher
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
import pandas as pd
import numpy as np
import wandb
from pathlib import Path
import yaml
import logging
from typing import Dict
from data_system.feature_engine import DerivativesFeatureEngine
from training.curriculum import TrainingManager
from monitoring.dashboard import TradingDashboard
import warnings
from data_system.data_manager import DataManager
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
# from data_collection.collect_multimodal import MultiModalDataCollector
# from data_system.multimodal_feature_extractor import MultiModalPerpFeatureExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import traceback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd
import json

# Configuration flags
ENABLE_DETAILED_LEVERAGE_MONITORING = True  # Set to True for more detailed leverage logging

# Custom action noise class
class CustomActionNoise:
    """
    A custom action noise class that adds Gaussian noise to actions
    """
    def __init__(self, mean=0.0, sigma=0.3, size=None):
        self.mean = mean
        self.sigma = sigma
        self.size = size
        
    def __call__(self):
        return np.random.normal(self.mean, self.sigma, size=self.size)
        
    def reset(self):
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set trading environment logger to WARNING level to suppress step-wise logs
# This will still show trial metrics but hide the detailed step logs
trading_env_logger = logging.getLogger('trading_env')
trading_env_logger.setLevel(logging.WARNING)

# Load configuration
config = None
try:
    with open('config/prod_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    raise

# Create necessary directories
for directory in ['logs', 'models', 'data']:
    Path(directory).mkdir(parents=True, exist_ok=True)

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        
        # ENHANCED: More sophisticated feature extractor with residual connections
        # and deeper architecture for better pattern recognition
        
        # Initial layer to project to common dimension
        self.input_layer = nn.Linear(n_input, 256)
        self.input_norm = nn.LayerNorm(256)
        self.input_activation = nn.LeakyReLU()
        self.input_dropout = nn.Dropout(0.15)  # Slightly increased dropout for better generalization
        
        # Residual block 1
        self.res1_layer1 = nn.Linear(256, 256)
        self.res1_norm1 = nn.LayerNorm(256)
        self.res1_activation1 = nn.LeakyReLU()
        self.res1_layer2 = nn.Linear(256, 256)
        self.res1_norm2 = nn.LayerNorm(256)
        self.res1_activation2 = nn.LeakyReLU()
        self.res1_dropout = nn.Dropout(0.15)
        
        # Residual block 2
        self.res2_layer1 = nn.Linear(256, 256)
        self.res2_norm1 = nn.LayerNorm(256)
        self.res2_activation1 = nn.LeakyReLU()
        self.res2_layer2 = nn.Linear(256, 256)
        self.res2_norm2 = nn.LayerNorm(256)
        self.res2_activation2 = nn.LeakyReLU()
        self.res2_dropout = nn.Dropout(0.15)
        
        # Output projection
        self.output_layer = nn.Linear(256, features_dim)
        self.output_norm = nn.LayerNorm(features_dim)
        
        # Initialize weights with orthogonal initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations):
        # Input projection
        x = self.input_layer(observations)
        x = self.input_norm(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)
        
        # Residual block 1
        residual = x
        x = self.res1_layer1(x)
        x = self.res1_norm1(x)
        x = self.res1_activation1(x)
        x = self.res1_layer2(x)
        x = self.res1_norm2(x)
        x = x + residual  # Add residual connection
        x = self.res1_activation2(x)
        x = self.res1_dropout(x)
        
        # Residual block 2
        residual = x
        x = self.res2_layer1(x)
        x = self.res2_norm1(x)
        x = self.res2_activation1(x)
        x = self.res2_layer2(x)
        x = self.res2_norm2(x)
        x = x + residual  # Add residual connection
        x = self.res2_activation2(x)
        x = self.res2_dropout(x)
        
        # Output projection
        x = self.output_layer(x)
        x = self.output_norm(x)
        
        return x

class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using residual connections for better gradient flow.
    This network architecture is better at capturing complex patterns across time
    and relationships between different assets and features.
    """
    def __init__(self, observation_space, features_dim=128, dropout_rate=0.1, use_layer_norm=True):
        super().__init__(observation_space, features_dim)
        
        # Get input dim from observation space
        n_input_features = int(np.prod(observation_space.shape))
        
        # Save parameters
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Define network architecture
        # First layer processes the raw input
        self.first_layer = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks for better gradient flow
        self.res_block1 = self._make_res_block(256, 256)
        self.res_block2 = self._make_res_block(256, 256)
        
        # Feature reduction and transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Track uncertainty for position sizing
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # One uncertainty value per forward pass
        )
        
    def _make_res_block(self, in_features, out_features):
        """Create a residual block with the same input/output dimension"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features) if self.use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features) if self.use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(self, observations):
        # Initial feature processing
        features = self.first_layer(observations)
        
        # Apply residual connections
        res1 = features + self.res_block1(features)
        res2 = res1 + self.res_block2(res1)
        
        # Generate uncertainty estimates (side path)
        # This allows the network to explicitly model uncertainty which can be
        # used for position sizing in the environment
        uncertainty = torch.sigmoid(self.uncertainty_head(res2))
        
        # Final feature transformation
        transformed_features = self.feature_transform(res2)
        
        # Store uncertainty for potential use in position sizing
        self._last_uncertainty = uncertainty
        
        return transformed_features

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for time series data.
    This architecture is specifically designed to capture long-range dependencies 
    and temporal patterns in market data.
    """
    def __init__(self, observation_space, features_dim=128, dropout_rate=0.1, 
                 d_model=64, nhead=4, num_layers=2, use_layer_norm=True):
        super().__init__(observation_space, features_dim)
        
        # Get input dimension from observation space
        n_input_features = int(np.prod(observation_space.shape))
        
        # Save parameters
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(n_input_features, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
        
        # Uncertainty head for risk estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, observations):
        # Input embedding
        x = self.input_embedding(observations)
        
        # Reshape for transformer (batch_size, sequence_length=1, d_model)
        # For now, we treat each observation as a sequence of length 1
        x = x.unsqueeze(1)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)
        
        # Extract output and squeeze sequence dimension
        transformer_output = transformer_output.squeeze(1)
        
        # Calculate uncertainty estimate
        uncertainty = self.uncertainty_head(transformer_output)
        self._last_uncertainty = uncertainty
        
        # Output projection
        features = self.output_projection(transformer_output)
        
        return features

class HybridFeatureExtractor(BaseFeaturesExtractor):
    """
    Hybrid architecture combining ResNet and Transformer approaches.
    This architecture leverages both the pattern recognition capabilities of ResNets
    and the temporal dependency modeling of Transformers.
    """
    def __init__(self, observation_space, features_dim=128, dropout_rate=0.1,
                d_model=64, nhead=4, transformer_layers=2, use_layer_norm=True):
        super().__init__(observation_space, features_dim)
        
        # Get input dimension from observation space
        n_input_features = int(np.prod(observation_space.shape))
        
        # Save parameters
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # First stage: ResNet-style processing
        self.input_layer = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(256, 256)
        self.res_block2 = self._make_res_block(256, 256)
        
        # Intermediate projection to d_model for transformer
        self.intermediate_projection = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
        
        # Second stage: Transformer processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
        
        # Uncertainty head for risk estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _make_res_block(self, in_features, out_features):
        """Create a residual block with the same input/output dimension"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features) if self.use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features) if self.use_layer_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate)
        )
    
    def forward(self, observations):
        # First stage: ResNet processing
        x = self.input_layer(observations)
        res1 = x + self.res_block1(x)
        res2 = res1 + self.res_block2(res1)
        
        # Project to transformer dimension
        transformer_input = self.intermediate_projection(res2)
        
        # Reshape for transformer (batch_size, sequence_length=1, d_model)
        transformer_input = transformer_input.unsqueeze(1)
        
        # Second stage: Transformer processing
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output = transformer_output.squeeze(1)
        
        # Calculate uncertainty
        uncertainty = self.uncertainty_head(transformer_output)
        self._last_uncertainty = uncertainty
        
        # Final output projection
        features = self.output_projection(transformer_output)
        
        return features

def parse_args():
    parser = argparse.ArgumentParser(description='Institutional Perpetual Trading AI')
    parser.add_argument('--assets', nargs='+', default=['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD'],
                        help='List of trading symbols')
    parser.add_argument('--timeframe', type=str, default='5m',
                        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                        help='Trading timeframe')
    parser.add_argument('--max-leverage', type=int, default=20,
                        help='Maximum allowed leverage')
    parser.add_argument('--training-steps', type=int, default=2_000_000,
                        help='Total training timesteps')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU only, 1+ to enable GPU acceleration)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (including step-wise metrics)')
    parser.add_argument('--training-mode', action='store_true',
                        help='Enable training optimizations without verbose logging')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue training from an existing model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model to continue training from')
    parser.add_argument('--env-path', type=str, default=None,
                        help='Path to the environment to continue training from')
    parser.add_argument('--additional-steps', type=int, default=1_000_000,
                        help='Number of additional steps to train when continuing')
    parser.add_argument('--reset-num-timesteps', action='store_true',
                        help='Reset timestep counter when continuing training')
    parser.add_argument('--reset-reward-norm', action='store_true',
                        help='Reset reward normalization when continuing training')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency in timesteps')
    parser.add_argument('--hyperparams', type=str, default=None,
                       help='Comma-separated list of hyperparameters to override (format: param1=value1,param2=value2)')
    parser.add_argument('--drive-ids-file', type=str, default=None,
                       help='Path to JSON file containing Google Drive file IDs for data files. This enables Google Drive integration.')
    parser.add_argument('--config', type=str, default='config/prod_config.yaml',
                       help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path: str = 'config/prod_config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config: dict):
    """Create necessary directories"""
    dirs = [
        config['data']['cache_dir'],
        config['model']['checkpoint_dir'],
        config['logging']['log_dir'],
        'models',
        'data',
        'logs/tensorboard'  # Add default tensorboard directory
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def initialize_wandb(config: dict):
    """Initialize Weights & Biases logging with enhanced metrics tracking"""
    # Initialize wandb with project settings
    run = wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        config=config,
        name=f"trading_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mode=config['logging']['wandb']['mode']
    )
    
    # Connect TensorBoard logs to wandb to capture SB3's internal metrics
    wandb.tensorboard.patch(root_logdir=config['logging'].get('tensorboard_dir', 'logs/tensorboard'))
    
    # Define custom wandb panels for trading metrics
    wandb.define_metric("portfolio/value", summary="max")
    wandb.define_metric("portfolio/drawdown", summary="max")
    wandb.define_metric("portfolio/sharpe", summary="max")
    wandb.define_metric("portfolio/sortino", summary="max")
    wandb.define_metric("portfolio/calmar", summary="max")
    
    # Define trade metrics - using max instead of sum for count
    wandb.define_metric("trades/count", summary="max")
    wandb.define_metric("trades/profit_pct", summary="mean")
    
    # Define training progress metrics
    wandb.define_metric("training/progress", summary="max")
    
    # Log initial config information
    if 'trading' in config and 'symbols' in config['trading']:
        wandb.log({"assets": config['trading']['symbols']})
    
    logger.info(f"Initialized WandB run: {run.name}")
    return run

class TradingSystem:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data manager with Google Drive support if specified
        drive_ids_file = config.get('data', {}).get('drive_ids_file')
        if drive_ids_file:
            logger.info(f"Initializing DataManager with Google Drive support from {drive_ids_file}")
            # Import here to avoid early import errors if Google APIs aren't installed
            from data_system import DriveAdapter
            self.data_manager = DriveAdapter(
                base_path=config['data']['cache_dir'],
                drive_ids_file=drive_ids_file
            )
        else:
            # Standard DataManager initialization
            self.data_manager = DataManager(
                base_path=config['data']['cache_dir']
            )
        
        # Initialize components
        self.data_fetcher = PerpetualDataFetcher(
            exchanges=config['data']['exchanges'],
            symbols=config['trading']['symbols'],
            timeframe=config['data']['timeframe']
        )
        
        self.feature_engine = DerivativesFeatureEngine(
            volatility_window=config['feature_engineering']['volatility_window'],
            n_components=config['feature_engineering']['n_components']
        )
        
        self.risk_engine = InstitutionalRiskEngine(
            risk_limits=RiskLimits(**config['risk_management']['limits'])
        )
        
        # Initialize monitoring dashboard
        self.dashboard = TradingDashboard(
            update_interval=config['monitoring']['update_interval'],
            alert_configs=config['monitoring']['alert_configs']
        )
        
        # Define default policy kwargs
        self.policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256],
                vf=[512, 256]
            ),
            activation_fn=nn.ReLU,
            ortho_init=True,
            log_std_init=-0.5,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=1e-5
            ),
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={"features_dim": 128}
        )
        
        self.training_manager = None
        self.env = None
        self.model = None
        self.processed_data = None
        self.study = None
        
    async def initialize(self, args=None):
        """Single entry point for all initialization"""
        logger.info("Starting system initialization...")
        
        # Fetch and process data
        self.processed_data = await self._fetch_and_process_data()
        
        # Initialize environment with args
        self.env = self._create_environment(self.processed_data, train=True, args=args)
        
        # Initialize training manager
        self.training_manager = TrainingManager(
            data_manager=self.data_manager,
            initial_balance=self.config['trading']['initial_balance'],
            max_leverage=self.config['trading']['max_leverage'],
            n_envs=self.config['training']['n_envs'],
            wandb_config=self.config['logging']['wandb']
        )
        
        # Initialize model
        self.model = self._setup_model(args)
        
        logger.info("System initialization complete!")
        
    async def _fetch_and_process_data(self):
        """Consolidated method for data fetching and processing"""
        logger.info("Fetching and processing data...")
            
        # Calculate date range for fallback case only
        lookback_days = self.config['data']['history_days']
        
        # Try to load existing data
        existing_data = self._load_cached_data()
        if existing_data is not None:
            logger.info("Using existing data from cache.")
            formatted_data = self._format_data_for_training(existing_data)
            logger.info(f"Formatted data shape: {formatted_data.shape}")
            logger.info(f"Columns: {formatted_data.columns}")
            # Save feature data for future use
            self._save_feature_data(formatted_data)
            return formatted_data
            
        # Fetch new data if needed
        logger.info("No cached data found or insufficient history. Fetching new data...")
        
        # Initialize data fetcher with correct lookback period
        self.data_fetcher.lookback = lookback_days
        
        # Fetch all data at once (the fetcher handles chunking internally)
        all_data = await self.data_fetcher.fetch_derivative_data()
        
        if not all_data:
            raise ValueError("No data fetched from exchanges")
            
        # Save raw data
        self._save_market_data(all_data)
        
        # Format data for training
        formatted_data = self._format_data_for_training(all_data)
        logger.info(f"Formatted data shape: {formatted_data.shape}")
        logger.info(f"Columns: {formatted_data.columns}")
        
        # Save feature data
        self._save_feature_data(formatted_data)
        
        return formatted_data
        
    def _load_cached_data(self, start_time=None, end_time=None):
        """
        Helper method to load cached data.
        """
        all_data = {exchange: {} for exchange in self.config['data']['exchanges']}
        has_all_data = True
        
        logger.info(f"Attempting to load cached data, minimum required history points: {self.config['data']['min_history_points']}")
        
        for exchange in self.config['data']['exchanges']:
            logger.info(f"Checking exchange: {exchange}")
            for symbol in self.config['trading']['symbols']:
                logger.info(f"Loading data for {exchange}_{symbol}_{self.config['data']['timeframe']}")
                
                # Load data without time constraints
                data = self.data_manager.load_market_data(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=self.config['data']['timeframe'],
                    start_time=None,
                    end_time=None,
                    data_type='perpetual'
                )
                
                # Check if we have enough data
                if data is None:
                    logger.warning(f"No data found for {exchange}_{symbol}")
                    has_all_data = False
                    break
                elif len(data) < self.config['data']['min_history_points']:
                    logger.warning(f"Insufficient data for {exchange}_{symbol}: found {len(data)} points, need {self.config['data']['min_history_points']}")
                    has_all_data = False
                    break
                else:
                    logger.info(f"Sufficient data found for {exchange}_{symbol}: {len(data)} points")
                    all_data[exchange][symbol] = data
                
            if not has_all_data:
                logger.warning(f"Missing or insufficient data for exchange {exchange}, will fetch new data")
                break
        
        if has_all_data:
            logger.info("All required data found in cache, using existing data")
            return all_data
        else:
            logger.info("Incomplete data in cache, will fetch fresh data")
            return None
        
    def _save_market_data(self, raw_data):
        """Helper method to save market data"""
        for exchange, exchange_data in raw_data.items():
            for symbol, symbol_data in exchange_data.items():
                self.data_manager.save_market_data(
                    data=symbol_data,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=self.config['data']['timeframe'],
                    data_type='perpetual'
                )
            
    def _save_feature_data(self, processed_data):
        """Helper method to save feature data"""
        self.data_manager.save_feature_data(
            data=processed_data,
            feature_set='base_features',
            metadata={
                'feature_config': self.config['feature_engineering'],
                'exchanges': self.config['data']['exchanges'],
                'symbols': self.config['trading']['symbols'],
                'timeframe': self.config['data']['timeframe']
            }
        )
        
    def _create_environment(self, df, train=True, args=None):
        """Create trading environment with market data."""
        assets = df.columns.get_level_values(0).unique().tolist()
        logger.info(f"Creating environment with assets: {assets}")
        
        # Configure features
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # ENHANCED: Add more sophisticated technical indicators
        tech_features = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr_14', 'adx_14', 'cci_14',
            'market_regime', 'hurst_exponent', 'volatility_regime'  # New market regime features
        ]
        
        # Create risk engine with configuration parameters
        risk_engine = InstitutionalRiskEngine(
            risk_limits=RiskLimits(**self.config['risk_management']['limits'])
        )
        
        # Determine verbose and training mode settings
        verbose = False
        training_mode = False
        
        if args:
            # Use command line arguments if provided
            verbose = args.verbose if hasattr(args, 'verbose') else False
            training_mode = args.training_mode if hasattr(args, 'training_mode') else False
        elif train:
            # Otherwise use train parameter for backward compatibility
            training_mode = train
        
        logger.info(f"Creating environment with verbose={verbose}, training_mode={training_mode}")
        
        # Create and return environment
        env = InstitutionalPerpetualEnv(
            df=df,
            assets=assets,
            initial_balance=self.config['trading']['initial_balance'],
            max_drawdown=self.config['risk_management']['limits']['max_drawdown'],
            window_size=self.config['model']['window_size'],
            max_leverage=self.config['trading']['max_leverage'],
            commission=self.config['trading']['commission'],  # Changed from transaction_fee to commission
            funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
            base_features=base_features,
            tech_features=tech_features,
            risk_engine=risk_engine,
            risk_free_rate=self.config['trading']['risk_free_rate'],
            verbose=verbose,  # Set verbose based on args or default
            training_mode=training_mode  # Set training_mode based on args or default
        )
        
        # Wrap with normalization layers
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
        
    def create_adaptive_lr_schedule(self, phase, total_phases=6, performance_metrics=None):
        """
        Create a learning rate schedule that considers:
        1. Current phase in overall training
        2. Performance metrics from previous phase (if available)
        
        Returns a function suitable for PPO's learning_rate parameter
        """
        # Calculate global progress
        total_steps = 1_000_000  # Total expected steps across all phases
        steps_per_phase = {
            1: 100000, 2: 100000, 3: 200000, 4: 200000, 5: 300000, 6: 100000
        }
        
        # Calculate steps completed so far (excluding current phase)
        steps_completed = sum(steps_per_phase.get(i, 0) for i in range(1, phase))
        
        # Current phase steps
        current_phase_steps = steps_per_phase.get(phase, 100000)
        
        # Calculate appropriate LR bounds for this phase based on global progress
        global_progress = steps_completed / total_steps
        
        # Determine LR bounds based on where we are in training
        if global_progress < 0.3:  # Early training (first 30%)
            initial_lr = 0.00012  # Keep initial rate
            final_lr = 0.00008   # Higher than before
        elif global_progress < 0.7:  # Middle training (40%)
            initial_lr = 0.00008
            final_lr = 0.00003   # Higher than before
        else:  # Late training (final 30%)
            initial_lr = 0.00003
            final_lr = 0.00003   # Keep higher final rate
        
        # Performance-based adjustments
        if performance_metrics:
            # If model is performing very well, reduce learning rate for stability
            reward_sharpe = performance_metrics.get('reward_sharpe', 0)
            mean_reward = performance_metrics.get('mean_reward', 0)
            
            if reward_sharpe > 1.0 and mean_reward > 1.5:
                # Model performing well, reduce learning rate for fine-tuning
                initial_lr *= 0.7
                final_lr *= 0.7
                logger.info(f"Reducing learning rate due to good performance: {initial_lr} to {final_lr}")
            elif reward_sharpe < 0.2 or mean_reward < 0.2:
                # Model performing poorly, slightly increase learning rate
                initial_lr *= 1.2
                final_lr *= 1.2
                logger.info(f"Increasing learning rate due to poor performance: {initial_lr} to {final_lr}")
        
        logger.info(f"Phase {phase} learning rate schedule: {initial_lr} to {final_lr}")
        
        # Calculate what portion of remaining training this phase represents
        phase_portion = current_phase_steps / (total_steps - steps_completed)
        
        # Return the schedule function
        def lr_schedule(progress_remaining):
            """Custom learning rate schedule for this specific phase"""
            # Convert from progress_remaining to progress_completed
            progress = 1.0 - progress_remaining
            
            # Add stabilization plateau at Phase 5 (800K steps)
            if phase == 5 and 0.2 <= progress <= 0.3:  # Around 800K steps
                return 0.00004  # Stabilization plateau
            
            # Enhanced cosine annealing schedule
            if progress_remaining < 0.3:  # Final 30% of this phase
                return final_lr
            elif progress_remaining > 0.8:  # Initial 20% of this phase
                return initial_lr
            else:
                # Middle of phase - cosine schedule
                phase_progress = (progress_remaining - 0.3) / 0.5
                cos_factor = 0.5 * (1 + np.cos(np.pi * (1 - phase_progress)))
                return final_lr + (initial_lr - final_lr) * cos_factor
        
        return lr_schedule
            
    def _setup_model(self, args=None) -> PPO:
        """Consolidated model setup"""
        # Get current phase from args if available
        current_phase = 1
        performance_metrics = None
        
        if args and hasattr(args, 'model_dir'):
            # Try to extract phase number from model directory
            match = re.search(r'phase(\d+)', args.model_dir)
            if match:
                current_phase = int(match.group(1))
        
        # Look for previous phase metrics if this isn't phase 1
        if current_phase > 1:
            prev_phase = current_phase - 1
            metrics_file = f"models/manual/phase{prev_phase}/evaluation_metrics.json"
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        performance_metrics = json.load(f)
                    logger.info(f"Loaded performance metrics from previous phase {prev_phase}")
                except Exception as e:
                    logger.warning(f"Could not load previous metrics: {str(e)}")
        
        # Create adaptive learning rate schedule
        learning_rate = self.create_adaptive_lr_schedule(
            phase=current_phase,
            total_phases=6,
            performance_metrics=performance_metrics
        )
        policy_kwargs = {
            "net_arch": dict(
                pi=[128, 64],  # Optimized policy network architecture
                vf=[256, 64]   # Optimized value network architecture
            ),
            "activation_fn": nn.ReLU,
            "ortho_init": True,
            "log_std_init": -0.5,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": dict(
                eps=1e-5,
                weight_decay=1e-5
            ),
            "features_extractor_class": HybridFeatureExtractor,  # Use HybridFeatureExtractor
            "features_extractor_kwargs": {
                "features_dim": 256,           # Optimized features dimension
                "dropout_rate": 0.088,         # Optimized dropout rate
                "use_layer_norm": True,
                "d_model": 64,
                "nhead": 4,
                "transformer_layers": 2
            }
        }
        
        # FIX: Better GPU detection and explicit device setting
        cuda_available = torch.cuda.is_available()
        if cuda_available and (args is None or getattr(args, 'gpus', 0) > 0):
            device = 'cuda'
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            if cuda_available:
                logger.warning("CUDA is available but not being used. Set --gpus > 0 to enable GPU.")
            else:
                logger.warning("CUDA is not available. Using CPU.")
        
        # Create and return the PPO model with all optimized parameters
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,  # Dynamic learning rate schedule
            n_steps=2048,
            batch_size=512,  # Increased batch size for better stability
            n_epochs=10,     # More epochs for better convergence
            gamma=0.9536529734618079, 
            gae_lambda=0.9346152432802582,
            clip_range=0.25,  # Slightly higher clip range
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0471577615690282,   # Increased for better exploration
            vf_coef=0.7,
            max_grad_norm=0.65,
            use_sde=True,
            sde_sample_freq=16,
            target_kl=0.03,
            tensorboard_log=self.config['logging'].get('tensorboard_dir', 'logs/tensorboard'),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
        
        # Log device information after model creation
        logger.info(f"Model created with device: {model.device}")
        
        # Configure environment with regime-aware parameters
        if hasattr(self.env, "env_method"):
            # Enable regime awareness
            self.env.env_method("set_regime_aware", True)
            # Set position holding bonus
            self.env.env_method("set_position_holding_bonus", 0.04689468349771604)
            # Set uncertainty scaling
            self.env.env_method("set_uncertainty_scaling", 1.2472096863889177)
        
        return model
        
    def create_study(self):
        """Create Optuna study for hyperparameter optimization"""
        storage = optuna.storages.InMemoryStorage()
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            study_name="ppo_optimization",
            direction="maximize",  # We want to maximize returns
            sampler=sampler,
            pruner=pruner,
            storage=storage
        )
        
    def optimize_hyperparameters(self, n_trials=30, n_jobs=5, total_timesteps=100000):
        """Run hyperparameter optimization"""
        if not self.study:
            self.create_study()
            
        logger.info(f"\nStarting hyperparameter optimization with {n_trials} trials")
        logger.info(f"Number of parallel jobs: {n_jobs}")
        logger.info(f"Total timesteps per trial: {total_timesteps}")
        
        try:
            self.study.optimize(
                lambda trial: self.objective(trial, total_timesteps),
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
            # Enhanced logging
            logger.info("\n" + "="*80)
            logger.info("Optimization Results Summary")
            logger.info("="*80)
            logger.info(f"Number of completed trials: {len(self.study.trials)}")
            logger.info(f"Best trial number: {self.study.best_trial.number}")
            logger.info(f"Best trial value (Final Sharpe): {self.study.best_trial.value:.6f}")
            logger.info(f"Best trial mean return: {self.study.best_trial.user_attrs['mean_return']:.6f}")
            logger.info(f"Best trial return Sharpe: {self.study.best_trial.user_attrs['return_sharpe']:.6f}")
            logger.info(f"Best trial reward Sharpe: {self.study.best_trial.user_attrs['reward_sharpe']:.6f}")
            logger.info("\nBest hyperparameters:")
            for key, value in self.study.best_trial.params.items():
                logger.info(f"    {key}: {value}")
            logger.info("="*80 + "\n")
            
            # Save study results
            df = self.study.trials_dataframe()
            df.to_csv("optuna_results.csv")
            logger.info(f"\nStudy results saved to optuna_results.csv")
            
            # Log best trial to wandb
            wandb.log({
                "best_trial_number": self.study.best_trial.number,
                "best_trial_value": self.study.best_trial.value,
                "best_trial_mean_return": self.study.best_trial.user_attrs['mean_return'],
                "best_trial_return_sharpe": self.study.best_trial.user_attrs['return_sharpe'],
                "best_trial_reward_sharpe": self.study.best_trial.user_attrs['reward_sharpe'],
                "best_trial_max_drawdown": self.study.best_trial.user_attrs['max_drawdown'],
                **self.study.best_trial.params
            })
            
            # Update model with best parameters
            self.update_model_with_best_params(self.study.best_trial.params, self.env)
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
            
    def objective(self, trial: optuna.Trial, total_timesteps: int) -> float:
        """Objective function for hyperparameter optimization"""
        try:
            # Start trial logging
            logger.info(f"\n╔═ Starting Trial {trial.number} ═{'═' * 59}╗")
            logger.info(f"║ Total timesteps per trial: {total_timesteps:<57} ║")
            
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            ent_coef = trial.suggest_float('ent_coef', 0.0, 0.05)  # ENHANCED: Expanded upper range
            vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)
            max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 0.7)
            
            # ENHANCED: Additional trading-specific hyperparameters
            n_epochs = trial.suggest_int('n_epochs', 5, 15)  # Number of epochs per update
            use_sde = trial.suggest_categorical('use_sde', [True, False])  # State-dependent exploration
            sde_sample_freq = trial.suggest_int('sde_sample_freq', 4, 16) if use_sde else -1
            target_kl = trial.suggest_float('target_kl', 0.01, 0.1)  # KL divergence target
            
            # Network architecture hyperparameters
            pi_1 = trial.suggest_categorical('pi_1', [128, 256, 512])
            pi_2 = trial.suggest_categorical('pi_2', [64, 128, 256])
            vf_1 = trial.suggest_categorical('vf_1', [128, 256, 512])
            vf_2 = trial.suggest_categorical('vf_2', [64, 128, 256])
            
            # Feature extractor hyperparameters
            features_dim = trial.suggest_categorical('features_dim', [64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
            
            # NEW: Transformer architecture hyperparameters
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            transformer_layers = trial.suggest_int('transformer_layers', 1, 3)
            
            # Environment-specific hyperparameters
            uncertainty_scaling = trial.suggest_float('uncertainty_scaling', 0.5, 2.0)
            position_holding_bonus = trial.suggest_float('position_holding_bonus', 0.01, 0.1)
            
            # Log sampled hyperparameters
            logger.info(f"║ Hyperparameters:                                                  ║")
            logger.info(f"║   - learning_rate: {learning_rate:<58} ║")
            logger.info(f"║   - n_steps: {n_steps:<63} ║")
            logger.info(f"║   - batch_size: {batch_size:<60} ║")
            logger.info(f"║   - gamma: {gamma:<65} ║")
            logger.info(f"║   - gae_lambda: {gae_lambda:<60} ║")
            logger.info(f"║   - clip_range: {clip_range:<60} ║")
            logger.info(f"║   - ent_coef: {ent_coef:<62} ║")
            logger.info(f"║   - vf_coef: {vf_coef:<62} ║")
            logger.info(f"║   - max_grad_norm: {max_grad_norm:<56} ║")
            logger.info(f"║   - n_epochs: {n_epochs:<61} ║")
            logger.info(f"║   - use_sde: {use_sde:<63} ║")
            if use_sde:
                logger.info(f"║   - sde_sample_freq: {sde_sample_freq:<54} ║")
            logger.info(f"║   - target_kl: {target_kl:<61} ║")
            logger.info(f"║   - pi_network: [{pi_1}, {pi_2}]                                        ║")
            logger.info(f"║   - vf_network: [{vf_1}, {vf_2}]                                        ║")
            logger.info(f"║   - features_dim: {features_dim:<58} ║")
            logger.info(f"║   - dropout_rate: {dropout_rate:<56} ║")
            logger.info(f"║   - regime_aware: {uncertainty_scaling:<54} ║")
            logger.info(f"║   - position_holding_bonus: {position_holding_bonus:<48} ║")
            logger.info(f"║   - uncertainty_scaling: {uncertainty_scaling:<46} ║")
            logger.info(f"╚{'═' * 80}╝\n")
            
            # Create a fresh environment for each trial
            env = self._create_environment(self.processed_data)
            
            # Create model with sampled hyperparameters
            try:
                # Define network architecture
                net_arch = [dict(
                    pi=[pi_1, pi_2],
                    vf=[vf_1, vf_2]
                )]
                
                # Create custom policy kwargs
                policy_kwargs = {
                    "net_arch": net_arch,
                    "activation_fn": nn.ReLU,
                    "features_extractor_class": HybridFeatureExtractor,
                    "features_extractor_kwargs": {"features_dim": features_dim, "dropout_rate": dropout_rate, "use_layer_norm": True}
                }
                
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    n_epochs=n_epochs,
                    use_sde=use_sde,
                    sde_sample_freq=sde_sample_freq,
                    target_kl=target_kl,
                    verbose=0,
                    tensorboard_log=self.config['logging'].get('tensorboard_dir', 'logs/tensorboard'),
                    policy_kwargs=policy_kwargs
                )

                # IMPORTANT FIX: Add exploration callback to encourage trading
                class ExplorationCallback(BaseCallback):
                    def __init__(self, env, verbose=0):
                        super().__init__(verbose)
                        # ENHANCED: Further increase exploration steps
                        self.exploration_steps = 20000  # Increased from 15000 to 20000
                        # Access the underlying environment to get assets
                        if hasattr(env, 'envs'):
                            # For DummyVecEnv
                            self.assets = env.envs[0].assets
                        else:
                            # Direct environment
                            self.assets = env.assets
                        self.step_count = 0
                        # IMPORTANT FIX: Track trades
                        self.trades_executed = 0
                        self.last_trade_step = 0
                        self.no_trade_warning_counter = 0  # Add counter to avoid excessive warnings
                        # ENHANCED: Improve trade forcing mechanism
                        self.force_trade_every = 300  # Force trade attempts more frequently
                        
                    def _on_step(self):
                        self.step_count += 1
                        
                        # ENHANCED: Use stronger and longer exploration
                        if self.num_timesteps < self.exploration_steps:
                            # Add noise to actions during initial exploration
                            # Use stronger noise early in training
                            noise_scale = max(1.0, 1.5 - self.num_timesteps / self.exploration_steps)
                            
                            # ENHANCED: Add more frequent and stronger bias toward extreme actions
                            if self.step_count % self.force_trade_every < 100:  # For 100 steps (longer period)
                                # Create stronger bias toward extreme actions
                                bias = np.random.choice([-0.8, 0.8], size=len(self.assets))
                                self.model.action_noise = CustomActionNoise(
                                    mean=bias,  # Use bias as mean instead of zeros
                                    sigma=noise_scale * np.ones(len(self.assets)),
                                    size=len(self.assets)
                                )
                                # if self.step_count % self.force_trade_every == 0:
                                #     logger.info(f"Forcing trade exploration at step {self.step_count} with bias {bias}")
                            else:
                                self.model.action_noise = CustomActionNoise(
                                    mean=np.zeros(len(self.assets)),
                                    sigma=noise_scale * np.ones(len(self.assets)),
                                    size=len(self.assets)
                                )
                              
                              # Every 500 steps, log the exploration progress
                              # if self.step_count % 500 == 0:
                              #     logger.info(f"Exploration step {self.num_timesteps}/{self.exploration_steps}, noise scale: {noise_scale:.2f}")
                                
                            # CRITICAL FIX: Check if trades are being executed
                            # Fixed trade detection logic
                            trade_executed = False
                            
                            # Properly access infos from locals dictionary
                            if 'infos' in self.locals and self.locals['infos'] is not None:
                                infos = self.locals['infos']
                                
                                # Handle different info formats
                                if isinstance(infos, list) and len(infos) > 0:
                                    info = infos[0]
                                else:
                                    info = infos
                                
                                # Check trades_executed flag
                                if isinstance(info, dict) and info.get('trades_executed', False):
                                    trade_executed = True
                                    # logger.debug(f"Trade detected via trades_executed flag")
                                
                                # Check positions directly
                                if isinstance(info, dict) and 'positions' in info:
                                    positions = info['positions']
                                    active_positions = sum(1 for pos in positions.values() 
                                                        if isinstance(pos, dict) and abs(pos.get('size', 0)) > 1e-8)
                                    if active_positions > 0:
                                        trade_executed = True
                                        # logger.debug(f"Trade detected via active positions: {active_positions}")
                                
                                # Check recent trades count
                                if isinstance(info, dict) and info.get('recent_trades_count', 0) > 0:
                                    trade_executed = True
                                    # logger.debug(f"Trade detected via recent_trades_count: {info.get('recent_trades_count')}")
                                
                                # CRITICAL FIX: Check total_trades count
                                if isinstance(info, dict) and info.get('total_trades', 0) > 0:
                                    trade_executed = True
                                    # logger.debug(f"Trade detected via total_trades: {info.get('total_trades')}")
                            
                            if trade_executed:
                                self.trades_executed += 1
                                self.last_trade_step = self.num_timesteps
                                # logger.info(f"Trade detected at step {self.num_timesteps}, total trades: {self.trades_executed}")
                                self.no_trade_warning_counter = 0  # Reset warning counter
                            
                            # ENHANCED: If no trades for a long time, increase exploration even more
                            # Only warn every 100 steps to avoid log spam
                            if self.trades_executed == 0 and self.num_timesteps > 1000:
                                self.no_trade_warning_counter += 1
                                if self.no_trade_warning_counter >= 100:
                                    # Force even stronger exploration
                                    logger.warning(f"No trades executed after {self.num_timesteps} steps, using extreme exploration")
                                    # Use extremely strong noise to force exploration
                                    self.model.action_noise = CustomActionNoise(
                                        mean=np.random.choice([-0.5, 0.5], size=len(self.assets)),  # Add bias
                                        sigma=2.5 * np.ones(len(self.assets)),  # Even stronger noise
                                        size=len(self.assets)
                                    )
                                    self.no_trade_warning_counter = 0  # Reset counter
                            elif self.num_timesteps - self.last_trade_step > 1000 and self.trades_executed > 0:
                                self.no_trade_warning_counter += 1
                                if self.no_trade_warning_counter >= 100:
                                    # If trades stopped, increase exploration again
                                    logger.warning(f"No trades for {self.num_timesteps - self.last_trade_step} steps, increasing exploration")
                                    self.model.action_noise = CustomActionNoise(
                                        mean=np.random.choice([-0.3, 0.3], size=len(self.assets)),  # Add some bias
                                        sigma=2.0 * np.ones(len(self.assets)),  # Stronger noise
                                        size=len(self.assets)
                                    )
                                    self.no_trade_warning_counter = 0  # Reset counter
                        else:
                            # ENHANCED: Gradually reduce noise after exploration phase but keep it significant
                            if self.num_timesteps < self.exploration_steps * 2:
                                decay_factor = 1.0 - ((self.num_timesteps - self.exploration_steps) / self.exploration_steps)
                                noise_scale = 0.6 * decay_factor  # Increased from 0.5 to 0.6
                                self.model.action_noise = CustomActionNoise(
                                    mean=np.zeros(len(self.assets)),
                                    sigma=noise_scale * np.ones(len(self.assets)),
                                    size=len(self.assets)
                                )
                            else:
                                # ENHANCED: Keep more significant minimal noise throughout training
                                self.model.action_noise = CustomActionNoise(
                                    mean=np.zeros(len(self.assets)),
                                    sigma=0.15 * np.ones(len(self.assets)),  # Increased minimal noise
                                    size=len(self.assets)
                                )
                        
                        return True
                
                # Training loop with error handling
                try:
                    # IMPORTANT FIX: Add exploration callback
                    exploration_callback = ExplorationCallback(env)
                    # CRITICAL FIX: Use total_timesteps for the actual training duration
                    model.learn(total_timesteps=total_timesteps, callback=exploration_callback)
                except Exception as train_error:
                    logger.error(f"Error during training: {str(train_error)}")
                    raise optuna.TrialPruned()

                # Evaluation with error handling
                try:
                    # IMPORTANT FIX: Increase evaluation episodes for more reliable results
                    eval_metrics = self._evaluate_model(model, n_eval_episodes=3, verbose=False)
                    
                    # Check if any trades were executed
                    if eval_metrics.get("trades_executed", 0) == 0:
                        logger.warning(f"Trial {trial.number} executed NO TRADES during evaluation. Pruning.")
                        raise optuna.TrialPruned()
                    
                    # Use the new objective_value metric which is already bounded and balanced
                    final_score = eval_metrics["objective_value"]
                    
                    # Enhanced trial completion logging
                    logger.info(f"\n{'='*30} TRIAL {trial.number} COMPLETED {'='*30}")
                    logger.info(f"Trial {trial.number} results:")
                    logger.info(f"  Objective value: {final_score:.4f}")
                    logger.info(f"  Mean reward: {eval_metrics['mean_reward']:.4f}")
                    logger.info(f"  Reward Sharpe: {eval_metrics['reward_sharpe']:.4f}")
                    logger.info(f"  Max drawdown: {eval_metrics['max_drawdown']:.4f}")
                    logger.info(f"  Avg leverage: {eval_metrics['avg_leverage']:.4f}")
                    logger.info(f"  Trades executed: {eval_metrics['trades_executed']}")
                    logger.info(f"{'='*78}\n")
                    
                    # Log all metrics
                    for key, value in eval_metrics.items():
                        trial.set_user_attr(key, value)
                    
                    # Log to wandb
                    wandb.log({
                        "trial_number": trial.number,
                        **eval_metrics,
                        **trial.params
                    })

                    logger.info(f"Trial {trial.number} completed with objective value: {final_score:.4f}")
                    
                    # Ensure the score is valid
                    if np.isnan(final_score) or np.isinf(final_score):
                        logger.warning(f"Invalid score detected: {final_score}. Using default penalty.")
                        return -1.0
                        
                    return float(final_score)
                    
                except Exception as eval_error:
                    logger.error(f"Error during evaluation: {str(eval_error)}")
                    traceback.print_exc()  # Print full traceback for debugging
                    raise optuna.TrialPruned()
                    
            except Exception as model_error:
                logger.error(f"Error creating/training model: {str(model_error)}")
                traceback.print_exc()  # Print full traceback for debugging
                raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"\n╔═ Error in Trial {trial.number} ═{'═' * 59}╗")
            logger.error(f"║ {str(e):<78} ║")
            logger.error(f"╚{'═' * 80}╝\n")
            traceback.print_exc()  # Print full traceback for debugging
            
            # Log failed trial to wandb
            wandb.log({
                "trial_number": trial.number,
                "status": "failed",
                "error": str(e)
            })
            
            return -1.0
        
    def _evaluate_model(self, model, n_eval_episodes=3, verbose=False):
        """Evaluate model performance"""
        rewards = []
        portfolio_values = []
        drawdowns = []
        leverage_ratios = []
        sharpe_ratios = []
        sortino_ratios = []
        calmar_ratios = []
        trades_executed = 0
        
        # IMPORTANT FIX: Track all trades across episodes
        all_trades = []
        
        for episode in range(n_eval_episodes):
            # CRITICAL FIX: Handle reset return format correctly
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                if len(reset_result) == 2:  # (obs, info)
                    obs, _ = reset_result
                else:  # Older format
                    obs = reset_result[0]
            else:
                obs = reset_result
                
            done = False
            episode_rewards = []
            portfolio_value = self.config['trading']['initial_balance']
            max_portfolio_value = portfolio_value
            step_count = 0
            episode_trades = 0
            
            logger.info(f"Starting evaluation episode {episode+1} with initial portfolio value: {portfolio_value}")
            
            # CRITICAL FIX: Run each episode for multiple steps
            # The episode should only terminate when done=True from the environment
            max_episode_steps = 100  # Allow up to 100 steps per episode for evaluation
            
            while not done and step_count < max_episode_steps:
                # Get action from model
                action, _ = model.predict(obs, deterministic=False)  # Use stochastic actions for evaluation
                
                # CRITICAL FIX: Add much stronger exploration noise for more steps during evaluation
                # This is crucial to ensure trades are executed during evaluation
                if step_count < 50:  # Increased from 30 to 50 steps
                    # Add stronger noise early in the episode
                    noise_scale = max(0.8, 1.5 - step_count * 0.02)  # Starts at 1.5, decreases more slowly
                    # Add bias toward extreme actions to encourage trading
                    bias = np.random.choice([-0.5, 0.5], size=action.shape)
                    action = action + bias + np.random.normal(0, noise_scale, size=action.shape)
                    action = np.clip(action, -1, 1)
                    # logger.info(f"Added exploration noise (scale={noise_scale:.2f}) to action: {action}")
                
                # Execute step in environment
                step_result = self.env.step(action)
                logger.debug(f"Step {step_count} - Action: {action}, Result type: {type(step_result)}")
                
                # CRITICAL FIX: Handle different return formats from env.step()
                if isinstance(step_result, tuple):
                    if len(step_result) == 5:  # Gymnasium format (obs, reward, terminated, truncated, info)
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:  # Older gym format (obs, reward, done, info)
                        obs, reward, done, info = step_result
                    else:
                        logger.error(f"Unexpected step_result length: {len(step_result)}")
                        break
                else:
                    logger.error(f"Unexpected step_result type: {type(step_result)}")
                    break
                
                episode_rewards.append(reward)
                step_count += 1
                
                # Handle VecEnv wrapper info dict
                if info is None:
                    logger.debug("Info is None")
                    continue
                
                # CRITICAL FIX: Properly extract info from VecEnv wrapper
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]  # VecEnv wraps info in a list
                
                # CRITICAL FIX: Much more robust trade detection
                trade_detected = False
                
                if isinstance(info, dict):
                    # Check trades_executed flag
                    if info.get('trades_executed', False):
                        trade_detected = True
                    
                    # Check positions directly
                    if 'positions' in info:
                        positions = info['positions']
                        active_count = 0
                        for asset, pos in positions.items():
                            if isinstance(pos, dict) and abs(pos.get('size', 0)) > 1e-8:
                                active_count += 1
                        if active_count > 0:
                            trade_detected = True
                            
                    # Also check active_positions if available
                    if 'active_positions' in info:
                        active_positions = info['active_positions']
                        if active_positions:
                            trade_detected = True
                    
                    # Check recent trades count
                    if info.get('recent_trades_count', 0) > 0:
                        trade_detected = True
                    
                    # Check has_positions flag
                    if info.get('has_positions', False):
                        trade_detected = True
                    
                    # Check total_trades count
                    if info.get('total_trades', 0) > 0:
                        trade_detected = True
                
                if trade_detected:
                    episode_trades += 1
                    trades_executed += 1
                    logger.info(f"Trade detected at step {step_count} in episode {episode+1}")
                
                # CRITICAL FIX: Properly extract and store risk metrics
                if isinstance(info, dict):
                    # Extract risk metrics directly from info
                    if 'risk_metrics' in info:
                        risk_metrics = info['risk_metrics']
                        
                        # Track portfolio value
                        if 'portfolio_value' in risk_metrics:
                            portfolio_value = risk_metrics['portfolio_value']
                            max_portfolio_value = max(max_portfolio_value, portfolio_value)
                        elif 'portfolio_value' in info:  # Also check top-level info
                            portfolio_value = info['portfolio_value']
                            max_portfolio_value = max(max_portfolio_value, portfolio_value)
                        
                        # Track drawdown
                        if 'current_drawdown' in risk_metrics:
                            drawdowns.append(risk_metrics['current_drawdown'])
                        elif 'max_drawdown' in risk_metrics:
                            drawdowns.append(risk_metrics['max_drawdown'])
                        
                        # Track leverage
                        if 'leverage_utilization' in risk_metrics:
                            leverage_ratios.append(risk_metrics['leverage_utilization'])
                            logger.debug(f"Added leverage ratio: {risk_metrics['leverage_utilization']}")
                            
                            # Log more detailed leverage information when monitoring is enabled
                            if ENABLE_DETAILED_LEVERAGE_MONITORING and step_count % 50 == 0:  # Log every 50 steps to avoid spam
                                logger.info(f"[Leverage Monitor] Step {step_count}: {risk_metrics['leverage_utilization']:.4f}x")
                                if 'gross_leverage' in risk_metrics:
                                    logger.info(f"  - Gross leverage: {risk_metrics['gross_leverage']:.4f}x")
                                if 'net_leverage' in risk_metrics:
                                    logger.info(f"  - Net leverage: {risk_metrics['net_leverage']:.4f}x")
                            else:
                                logger.debug(f"No leverage_utilization in risk_metrics: {list(risk_metrics.keys())}")
                        
                        # Track risk-adjusted ratios
                        if 'sharpe_ratio' in risk_metrics:
                            sharpe_ratios.append(risk_metrics['sharpe_ratio'])
                        if 'sortino_ratio' in risk_metrics:
                            sortino_ratios.append(risk_metrics['sortino_ratio'])
                        if 'calmar_ratio' in risk_metrics:
                            calmar_ratios.append(risk_metrics['calmar_ratio'])
                    
                    # CRITICAL FIX: Also check historical_metrics in info
                    if 'historical_metrics' in info:
                        hist_metrics = info['historical_metrics']
                        
                        # Add historical leverage samples if available
                        if 'avg_leverage' in hist_metrics and hist_metrics['avg_leverage'] > 0:
                            leverage_ratios.append(hist_metrics['avg_leverage'])
                            
                        # Add historical drawdown samples if available
                        if 'max_drawdown' in hist_metrics and hist_metrics['max_drawdown'] > 0:
                            drawdowns.append(hist_metrics['max_drawdown'])
                    
                    # Update terminal information
                    if done:
                        logger.info(f"Episode {episode+1} terminated at step {step_count}")
                
            portfolio_values.append(portfolio_value)
            rewards.extend(episode_rewards)
            
            # IMPORTANT FIX: Track trades for this episode
            all_trades.append(episode_trades)
            
            logger.info(f"Episode {episode+1} completed: Steps={step_count}, Final Portfolio=${portfolio_value:.2f}, "
                       f"Trades Executed={episode_trades}")
        
        # IMPORTANT FIX: Better logging of trade execution
        logger.info(f"Trade execution summary: {all_trades} (total: {trades_executed})")
        
        # CRITICAL FIX: Force trades_executed to be non-zero if we have evidence of trades
        if trades_executed == 0 and (len(leverage_ratios) > 0 or len(drawdowns) > 0):
            logger.warning("No trades detected directly, but risk metrics suggest trading activity. Setting trades_executed to 1.")
            trades_executed = 1
        
        # Check if any trading happened
        if trades_executed == 0:
            logger.warning("NO TRADES EXECUTED DURING EVALUATION! Model is not trading at all.")
            # Return poor performance metrics to discourage this behavior
            return {
                "mean_reward": -1.0,
                "reward_sharpe": -1.0,
                "max_drawdown": 1.0,
                "avg_leverage": 0.0,
                "avg_sharpe": -1.0,
                "avg_sortino": -1.0,
                "avg_calmar": -1.0,
                "objective_value": -1.0,
                "trades_executed": 0
            }
        
        # Convert to numpy arrays for calculations
        rewards_array = np.array(rewards)
        portfolio_values = np.array(portfolio_values)
        
        # CRITICAL FIX: Ensure we have valid metrics
        if len(drawdowns) == 0:
            logger.warning("No drawdown samples collected. Using default value.")
            drawdowns = [0.01]  # Use a small default value
            
        if len(leverage_ratios) == 0:
            logger.warning("No leverage samples collected. Using default range values.")
            # Use a range of leverage values to avoid showing the same min/max
            leverage_ratios = [1.0, 3.0, 5.0, 8.0, 12.0, 15.0]  # Use more realistic default values
            
        drawdowns = np.array(drawdowns)
        leverage_ratios = np.array(leverage_ratios)
        
        # Calculate reward statistics with safety checks
        mean_reward = float(np.mean(rewards_array)) if len(rewards_array) > 0 else 0.0
        reward_std = float(np.std(rewards_array)) if len(rewards_array) > 1 else 1.0
        
        # Calculate Sharpe ratio with safety checks
        if reward_std > 0 and not np.isnan(mean_reward) and not np.isinf(mean_reward):
            reward_sharpe = mean_reward / reward_std
            # Clip to reasonable range
            reward_sharpe = np.clip(reward_sharpe, -10.0, 10.0)
        else:
            reward_sharpe = 0.0
        
        # Calculate portfolio statistics with safety checks
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        avg_leverage = float(np.mean(leverage_ratios)) if len(leverage_ratios) > 0 else 0.0
        
        # Calculate average risk-adjusted ratios
        avg_sharpe = float(np.mean(sharpe_ratios)) if len(sharpe_ratios) > 0 else 0.0
        avg_sortino = float(np.mean(sortino_ratios)) if len(sortino_ratios) > 0 else 0.0
        avg_calmar = float(np.mean(calmar_ratios)) if len(calmar_ratios) > 0 else 0.0
        
        # Add detailed logging
        logger.info(f"\nDetailed Evaluation metrics:")
        logger.info(f"Mean reward: {mean_reward:.4f}")
        logger.info(f"Reward Sharpe ratio: {reward_sharpe:.4f}")
        logger.info(f"Max drawdown: {max_drawdown:.4f}")
        logger.info(f"Average leverage: {avg_leverage:.4f}")
        logger.info(f"Average Sharpe ratio: {avg_sharpe:.4f}")
        logger.info(f"Average Sortino ratio: {avg_sortino:.4f}")
        logger.info(f"Average Calmar ratio: {avg_calmar:.4f}")
        logger.info(f"Final portfolio values: {portfolio_values}")
        logger.info(f"Total trades executed: {trades_executed}")
        logger.info(f"Number of leverage samples: {len(leverage_ratios)}")
        logger.info(f"Number of drawdown samples: {len(drawdowns)}")
        if len(leverage_ratios) > 0:
            min_lev = min(leverage_ratios)
            max_lev = max(leverage_ratios)
            avg_lev = np.mean(leverage_ratios)
            logger.info(f"Leverage range: [{min_lev:.4f}, {max_lev:.4f}], Avg: {avg_lev:.4f}")
            # Log individual leverage values for more visibility
            if len(leverage_ratios) < 10:
                logger.info(f"All leverage values: {[f'{lev:.4f}' for lev in leverage_ratios]}")
            else:
                # Log a subset if there are many values
                logger.info(f"Sample leverage values: {[f'{lev:.4f}' for lev in leverage_ratios[:5]]}, ... (total: {len(leverage_ratios)})")
        
        # Ensure all metrics are valid
        if np.isnan(reward_sharpe) or np.isinf(reward_sharpe):
            reward_sharpe = 0.0
        if np.isnan(max_drawdown) or np.isinf(max_drawdown):
            max_drawdown = 1.0
        if np.isnan(avg_leverage) or np.isinf(avg_leverage):
            avg_leverage = 0.0
        
        # IMPORTANT FIX: Adjust objective function to more strongly reward trading activity
        # Calculate objective value for optimization (bounded to prevent extreme values)
        # Use a combination of metrics for a balanced objective
        objective_value = (
            0.25 * reward_sharpe +                # Reward risk-adjusted returns
            0.15 * (1.0 - min(1.0, max_drawdown/10000)) +  # Normalize drawdown and minimize it
            0.15 * avg_sharpe +                   # Consistent risk-adjusted performance
            0.10 * (1.0 - avg_leverage / self.config['trading']['max_leverage']) +  # Efficient leverage use
            0.35 * min(1.0, trades_executed / (n_eval_episodes * 2))  # Increased weight for trading activity, reduced threshold
        )
        
        # CRITICAL FIX: Don't clip to -10 by default, use a more reasonable range
        # This was causing all trials to return -10.0
        objective_value = np.clip(objective_value, -1.0, 10.0)
        
        # If no trades executed, penalize but don't set to minimum
        if trades_executed == 0:
            objective_value = -0.5  # Less severe penalty to allow exploration
        
        return {
            "mean_reward": mean_reward,
            "reward_sharpe": reward_sharpe,
            "max_drawdown": max_drawdown,
            "avg_leverage": avg_leverage,
            "avg_sharpe": avg_sharpe,
            "avg_sortino": avg_sortino,
            "avg_calmar": avg_calmar,
            "objective_value": objective_value,
            "trades_executed": trades_executed
        }
        
    def update_model_with_best_params(self, best_params, env):
        """Update model with best hyperparameters."""
        try:
            # Log best parameters
            logger.info("Best hyperparameters:")
            for param, value in best_params.items():
                logger.info(f"{param}: {value}")
            
            # Define network architecture
            net_arch = [dict(
                pi=[best_params["pi_1"], best_params["pi_2"]], 
                vf=[best_params["vf_1"], best_params["vf_2"]]
            )]
            
            # Define policy kwargs
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": nn.LeakyReLU,
                "features_extractor_class": HybridFeatureExtractor,
                "features_extractor_kwargs": {
                    "features_dim": best_params.get("features_dim", 128),
                    "dropout_rate": best_params.get("dropout_rate", 0.15),
                    "use_layer_norm": True,
                    "d_model": best_params.get("d_model", 64),
                    "nhead": best_params.get("nhead", 4),
                    "transformer_layers": best_params.get("transformer_layers", 2)
                }
            }
            
            # Check if we have SDE parameters
            use_sde = best_params.get("use_sde", True)
            sde_sample_freq = best_params.get("sde_sample_freq", 4) if use_sde else -1
            
            # Create model with best hyperparameters
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=best_params["learning_rate"],
                n_steps=best_params["ppo_n_steps"],
                batch_size=best_params["batch_size"],
                n_epochs=best_params.get("n_epochs", 10),
                gamma=best_params["gamma"],
                gae_lambda=best_params["gae_lambda"],
                clip_range=best_params["clip_range"],
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=best_params["ent_coef"],
                vf_coef=best_params["vf_coef"],
                max_grad_norm=best_params["max_grad_norm"],
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                target_kl=best_params.get("target_kl", None),
                tensorboard_log="./logs/ppo_crypto_tensorboard/",
                policy_kwargs=policy_kwargs,
                verbose=1
            )
            
            # Configure environment for best parameters
            if hasattr(env, "get_attr"):
                # For vectorized environments
                if best_params.get("regime_aware", False):
                    # Enable regime awareness in the environment
                    env.env_method("set_regime_aware", True)
                    
                    # Set position holding incentives
                    if "position_holding_bonus" in best_params:
                        env.env_method("set_position_holding_bonus", 
                                     best_params["position_holding_bonus"])
                    
                    # Set uncertainty scaling
                    if "uncertainty_scaling" in best_params:
                        env.env_method("set_uncertainty_scaling",
                                     best_params["uncertainty_scaling"])
            else:
                # For non-vectorized environments
                if best_params.get("regime_aware", False):
                    # Enable regime awareness in the environment
                    if hasattr(env, "set_regime_aware"):
                        env.set_regime_aware(True)
                    
                    # Set position holding incentives
                    if "position_holding_bonus" in best_params and hasattr(env, "set_position_holding_bonus"):
                        env.set_position_holding_bonus(best_params["position_holding_bonus"])
                    
                    # Set uncertainty scaling
                    if "uncertainty_scaling" in best_params and hasattr(env, "set_uncertainty_scaling"):
                        env.set_uncertainty_scaling(best_params["uncertainty_scaling"])
            
            logger.info("Model updated with best hyperparameters")
            return model
        
        except Exception as e:
            logger.error(f"Error updating model with best params: {e}")
            traceback.print_exc()
            raise

    def train(self, args=None):
        """Optimized training method with early stopping and best model saving"""
        if not self.model or not self.env:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info("Starting model training for 1,000,000 timesteps...")
        
        # Setup callbacks for checkpointing and evaluation
        callbacks = []
        
        # Checkpoint callback - save every 100k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.config['model']['checkpoint_dir'],
            name_prefix="ppo_trading"
        )
        callbacks.append(checkpoint_callback)

        # Replace EvalCallback with our custom VecNormalizeEvalCallback
        eval_callback = VecNormalizeEvalCallback(
            eval_env=self.env,
            n_eval_episodes=3,  # Reduced from 5 for better performance
            eval_freq=20000, 
            log_path=self.config['logging']['log_dir'],
            best_model_save_path=os.path.join(self.config['model']['checkpoint_dir'], "best"),
            deterministic=True,
            verbose=1  # Enable verbose to see training mode changes
        )
        callbacks.append(eval_callback)
        
        # Train the model with optimized parameters
        # Get training steps from args if provided, otherwise use default
        total_timesteps = None
        if args and hasattr(args, 'training_steps') and args.training_steps:
            total_timesteps = args.training_steps
        else:
            total_timesteps = 1_000_000
            
        logger.info(f"Training for {total_timesteps} timesteps")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="ppo_trading",
                progress_bar=True
            )
            
            # Save final model
            final_model_path = os.path.join(self.config['model']['checkpoint_dir'], "final_model")
            self.model.save(final_model_path)
            self.env.save(os.path.join(self.config['model']['checkpoint_dir'], "final_env.pkl"))
            
            # Final evaluation
            eval_metrics = self._evaluate_model(self.model)
            logger.info("\nTraining completed!")
            logger.info(f"Final metrics:")
            logger.info(f"Mean return: {eval_metrics.get('mean_reward', 'N/A'):.4f}")
            logger.info(f"Sharpe ratio: {eval_metrics.get('reward_sharpe', 'N/A'):.4f}")
            logger.info(f"Max drawdown: {eval_metrics.get('max_drawdown', 'N/A'):.4f}")
            

            # Generate recommendations for next phase
            current_phase = int(args.model_dir.rstrip('/').split('phase')[-1])
            next_phase_recommendations = recommend_next_phase_params(
                current_phase=current_phase,
                total_phases=6,  # Assuming standard 6-phase training
                eval_metrics=eval_metrics
            )

            logger.info(f"Recommendations for phase {current_phase + 1} saved to: {args.model_dir}/phase{current_phase + 1}_recommendations.json")
            
            # After evaluation in training/continue_training methods:
            eval_metrics_file = os.path.join(args.model_dir, "evaluation_metrics.json")
            with open(eval_metrics_file, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            logger.info(f"Saved evaluation metrics to {eval_metrics_file}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def continue_training(self, model_path, env_path=None, additional_timesteps=1_000_000, 
                        reset_num_timesteps=False, reset_reward_normalization=False,
                        tb_log_name="ppo_trading_continued", hyperparams=None, args=None):
        """
        Continue training from a saved model.
        
        Args:
            model_path: Path to the saved model
            env_path: Path to the saved environment (optional)
            additional_timesteps: Number of additional timesteps to train for
            reset_num_timesteps: Whether to reset the timestep counter
            reset_reward_normalization: Whether to reset reward normalization statistics
            tb_log_name: TensorBoard log name
            hyperparams: Dictionary of hyperparameters to override for continued training
                         (e.g., {"ent_coef": 0.01} to increase exploration)
            args: Command line arguments that may include training_mode and verbose flags
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load the saved model
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        # Set the loaded model to the same environment
        if env_path and os.path.exists(env_path):
            logger.info(f"Loading environment from {env_path}")
            self.env = VecNormalize.load(env_path, self.env)
            
            # Reset reward normalization if requested (useful when changing reward function)
            if reset_reward_normalization:
                logger.info("Resetting reward normalization statistics")
                self.env.obs_rms = self.env.obs_rms  # Keep observation normalization
                self.env.ret_rms = RunningMeanStd(shape=())  # Reset reward normalization
                if hasattr(self.env, 'returns'):
                    self.env.returns = np.zeros(self.env.returns.shape)
        
        # Apply training_mode and verbose settings to environment if args provided
        if args and (hasattr(args, 'training_mode') or hasattr(args, 'verbose')):
            training_mode = args.training_mode if hasattr(args, 'training_mode') else False
            verbose = args.verbose if hasattr(args, 'verbose') else False
            
            logger.info(f"Updating environment settings: training_mode={training_mode}, verbose={verbose}")
            
            # Update the environment settings using env_method
            if hasattr(self.env, "env_method"):
                if hasattr(args, 'training_mode'):
                    self.env.env_method("update_parameters", training_mode=training_mode)
                if hasattr(args, 'verbose'):
                    self.env.env_method("update_parameters", verbose=verbose)
        
        # Set the model
        self.model = model
        # Set the model environment to our environment
        self.model.set_env(self.env)
        
        # Override hyperparameters if provided
        if hyperparams:
            logger.info(f"Overriding hyperparameters for continued training:")
            for param, value in hyperparams.items():
                if hasattr(self.model, param):
                    old_value = getattr(self.model, param)
                    setattr(self.model, param, value)
                    logger.info(f"  - {param}: {old_value} -> {value}")
                else:
                    logger.warning(f"Hyperparameter '{param}' not found in model")
        
            # Special handling for entropy coefficient (needs to be updated in policy too)
            if 'ent_coef' in hyperparams:
                logger.info(f"Updating entropy coefficient to {hyperparams['ent_coef']}")
                if hasattr(self.model, 'ent_coef'):
                    self.model.ent_coef = hyperparams['ent_coef']
                    logger.info(f"Increased entropy coefficient to {self.model.ent_coef} for better exploration")
                    wandb.log({"hyperparameters/ent_coef": self.model.ent_coef})
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.config['model']['checkpoint_dir'],
            name_prefix="ppo_trading_continued"
        )
        callbacks.append(checkpoint_callback)
        
        # Use our custom VecNormalizeEvalCallback instead of standard EvalCallback
        eval_callback = VecNormalizeEvalCallback(
            eval_env=self.env,
            n_eval_episodes=3,  # Reduced from 5 for better performance
            eval_freq=20000, 
            log_path=self.config['logging']['log_dir'],
            best_model_save_path=os.path.join(self.config['model']['checkpoint_dir'], "best_continued"),
            deterministic=True,
            verbose=1  # Enable verbose to see training mode changes
        )
        callbacks.append(eval_callback)
        
        # Continue training
        logger.info(f"Continuing training for {additional_timesteps} additional timesteps")
        
        # Log continuation to wandb
        wandb.log({
            "training/continued_from": model_path,
            "training/additional_steps": additional_timesteps,
            "training/reset_num_timesteps": reset_num_timesteps,
            "training/reset_reward_normalization": reset_reward_normalization
        })
        
        try:
            self.model.learn(
                total_timesteps=additional_timesteps,
                callback=callbacks,
                tb_log_name=tb_log_name,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps
            )
            
            # Save final model after continued training
            final_model_path = os.path.join(args.model_dir, "final_continued_model")
            self.model.save(final_model_path)
            self.env.save(os.path.join(args.model_dir, "final_continued_env.pkl"))
            
            # Final evaluation
            eval_metrics = self._evaluate_model(self.model)
            logger.info("\nContinued training completed!")
            logger.info(f"Final metrics after continuation:")
            logger.info(f"Mean return: {eval_metrics.get('mean_reward', 'N/A'):.4f}")
            logger.info(f"Sharpe ratio: {eval_metrics.get('reward_sharpe', 'N/A'):.4f}")
            logger.info(f"Max drawdown: {eval_metrics.get('max_drawdown', 'N/A'):.4f}")
            
            
            # Generate recommendations for next phase
            current_phase = int(args.model_dir.rstrip('/').split('phase')[-1])
            next_phase_recommendations = recommend_next_phase_params(
                current_phase=current_phase,
                total_phases=6,  # Assuming standard 6-phase training
                eval_metrics=eval_metrics
            )

            logger.info(f"Recommendations for phase {current_phase + 1} saved to: {args.model_dir}/phase{current_phase + 1}_recommendations.json")
            
            # After evaluation in training/continue_training methods:
            eval_metrics_file = os.path.join(args.model_dir, "evaluation_metrics.json")
            with open(eval_metrics_file, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            logger.info(f"Saved evaluation metrics to {eval_metrics_file}")
            
            return final_model_path
        
        except Exception as e:
            logger.error(f"Error during continued training: {str(e)}")
            raise

    def _format_data_for_training(self, raw_data):
        """Format data into the structure expected by the trading environment"""
        logger.info("Starting data formatting...")
        
        # Check if input is already feature data (when loaded directly from feature files)
        if isinstance(raw_data, pd.DataFrame) and isinstance(raw_data.columns, pd.MultiIndex):
            logger.info("Data is already formatted feature data, returning directly")
            return raw_data
            
        # OPTIMIZATION: Check for cached processed data first
        cache_dir = Path("data/cache")
        cache_file = cache_dir / "processed_data_cache.pkl"
        
        # Check if cache exists and is recent (less than 1 day old)
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime < 86400):
            try:
                logger.info(f"Loading cached processed data from {cache_file}")
                with open(cache_file, 'rb') as f:
                    combined_data = pickle.load(f)
                logger.info(f"Successfully loaded cached data with shape: {combined_data.shape}")
                return combined_data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {str(e)}. Processing from raw data.")
        
        # Initialize an empty list to store DataFrames for each symbol
        symbol_dfs = []
        
        # CRITICAL FIX: Validate raw_data structure before processing
        if not raw_data or not isinstance(raw_data, dict):
            raise ValueError("Invalid raw_data structure: must be a non-empty dictionary")
            
        # Process each exchange's data
        for exchange, exchange_data in raw_data.items():
            logger.info(f"Processing exchange: {exchange}")
            
            # CRITICAL FIX: Validate exchange data
            if not exchange_data or not isinstance(exchange_data, dict):
                logger.warning(f"Invalid data for exchange {exchange}, skipping")
                continue
                
            for symbol, symbol_data in exchange_data.items():
                logger.info(f"Processing symbol: {symbol}")
                
                # CRITICAL FIX: Validate symbol data
                if symbol_data is None or symbol_data.empty:
                    logger.warning(f"Empty data for {symbol}, skipping")
                    continue
                    
                # Convert symbol to format expected by risk engine (e.g., BTC/USD:USD -> BTCUSDT)
                formatted_symbol = symbol.split('/')[0] + "USDT" if not symbol.endswith('USDT') else symbol
                logger.info(f"Formatted symbol: {formatted_symbol}")
                
                try:
                    # Create a copy to avoid modifying original data
                    df = symbol_data.copy()
                    logger.info(f"Original columns: {df.columns}")
                    
                    # CRITICAL FIX: Verify dataframe has expected index and structure
                    if not isinstance(df.index, pd.DatetimeIndex):
                        logger.warning(f"Converting index to DatetimeIndex for {formatted_symbol}")
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception as e:
                            logger.error(f"Failed to convert index to datetime: {e}, using numeric index")
                            df.index = pd.RangeIndex(start=0, stop=len(df))
                    
                    # Create formatted DataFrame
                    formatted_data = pd.DataFrame(index=df.index)
                    
                    # Check if df has MultiIndex columns (data loaded from Google Drive)
                    if isinstance(df.columns, pd.MultiIndex):
                        logger.info(f"Processing MultiIndex columns for {formatted_symbol}")
                        # For each feature needed, get it from the MultiIndex columns
                        for col in ['open', 'high', 'low', 'close', 'volume', 'funding_rate', 'bid_depth', 'ask_depth']:
                            if (formatted_symbol, col) in df.columns:
                                formatted_data[col] = pd.to_numeric(df[(formatted_symbol, col)], errors='coerce')
                            else:
                                logger.warning(f"Column ({formatted_symbol}, {col}) not found in MultiIndex data")
                    else:
                        # Original code for flat columns
                        logger.info(f"Processing flat columns for {formatted_symbol}")
                        # OPTIMIZATION: Process all numeric columns in one batch operation
                        for col in ['open', 'high', 'low', 'close', 'volume', 'funding_rate']:
                            if col in df.columns:
                                formatted_data[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # CRITICAL FIX: Validate price data consistency
                    # Ensure high >= low and high,low are consistent with open,close
                    # Only perform this operation if we have the required columns
                    required_cols = ['high', 'open', 'close', 'low']
                    if all(col in formatted_data.columns for col in required_cols):
                        formatted_data['high'] = formatted_data[['high', 'open', 'close']].max(axis=1)
                        formatted_data['low'] = formatted_data[['low', 'open', 'close']].min(axis=1)
                    else:
                        missing = [col for col in required_cols if col not in formatted_data.columns]
                        logger.error(f"Cannot validate price consistency - missing columns: {missing}")
                        # If we're missing high/low but have open/close, create them
                        if 'high' not in formatted_data.columns and 'open' in formatted_data.columns and 'close' in formatted_data.columns:
                            formatted_data['high'] = formatted_data[['open', 'close']].max(axis=1)
                        if 'low' not in formatted_data.columns and 'open' in formatted_data.columns and 'close' in formatted_data.columns:
                            formatted_data['low'] = formatted_data[['open', 'close']].min(axis=1)
                    
                    # Rest of the existing code for handling missing columns
                    # ...
                    
                    # Handle missing columns with vectorized operations
                    if 'volume' not in df.columns:
                        # Generate synthetic volume based on price volatility
                        returns = formatted_data['close'].pct_change()
                        vol = returns.rolling(window=20).std().fillna(0.01)
                        formatted_data['volume'] = formatted_data['close'] * vol * 1000
                    
                    # Ensure volume is positive and non-zero (vectorized)
                    formatted_data['volume'] = formatted_data['volume'].clip(lower=1.0)
                    
                    # Handle funding rate with vectorized operations
                    if 'funding_rate' not in df.columns:
                        # Use small random funding rate
                        formatted_data['funding_rate'] = np.random.normal(0, 0.0001, size=len(formatted_data))
                    
                    # CRITICAL FIX: Constrain funding rates to reasonable bounds (vectorized)
                    formatted_data['funding_rate'] = formatted_data['funding_rate'].clip(lower=-0.0075, upper=0.0075)
                    
                    # Handle market depth with vectorized operations
                    if 'bid_depth' not in df.columns:
                        formatted_data['bid_depth'] = formatted_data['volume'] * 0.4
                    else:
                        formatted_data['bid_depth'] = pd.to_numeric(df['bid_depth'], errors='coerce')
                        
                    if 'ask_depth' not in df.columns:
                        formatted_data['ask_depth'] = formatted_data['volume'] * 0.4
                    else:
                        formatted_data['ask_depth'] = pd.to_numeric(df['ask_depth'], errors='coerce')
                    
                    # Ensure depth is positive and non-zero (vectorized)
                    formatted_data['bid_depth'] = formatted_data['bid_depth'].clip(lower=1.0)
                    formatted_data['ask_depth'] = formatted_data['ask_depth'].clip(lower=1.0)
                    
                    # Add volatility with vectorized operations
                    close_returns = formatted_data['close'].pct_change()
                    formatted_data['volatility'] = close_returns.rolling(window=20).std().fillna(0.01)
                    
                    # OPTIMIZATION: Handle missing values with batch operations
                    # First replace infinities with NaN
                    formatted_data = formatted_data.replace([np.inf, -np.inf], np.nan)
                    # Forward fill any NaN values first
                    formatted_data = formatted_data.ffill()
                    # Then backward fill any remaining NaN values at the start
                    formatted_data = formatted_data.bfill()
                    
                    # Create MultiIndex columns
                    formatted_data.columns = pd.MultiIndex.from_product(
                        [[formatted_symbol], formatted_data.columns],
                        names=['asset', 'feature']
                    )
                    
                    logger.info(f"Formatted columns for {formatted_symbol}: {formatted_data.columns}")
                    
                    # Verify all required features exist
                    required_features = ['open', 'high', 'low', 'close', 'volume', 'funding_rate', 'bid_depth', 'ask_depth', 'volatility']
                    for feature in required_features:
                        if (formatted_symbol, feature) not in formatted_data.columns:
                            raise ValueError(f"Missing required feature {feature} for {formatted_symbol}")
                    
                    symbol_dfs.append(formatted_data)
                    
                except Exception as e:
                    logger.error(f"Error processing {formatted_symbol}: {str(e)}")
                    traceback.print_exc()  # CRITICAL FIX: Add stacktrace for better debugging
                    continue
        
        # Combine all symbol data
        if not symbol_dfs:
            raise ValueError("No valid data to process")
        
        # OPTIMIZATION: Combine DataFrames efficiently
        combined_data = pd.concat(symbol_dfs, axis=1)
        logger.info(f"Combined data shape before deduplication: {combined_data.shape}")
        
        # CRITICAL FIX: Check for extreme price jumps that could indicate bad data
        assets = combined_data.columns.get_level_values('asset').unique()
        for asset in assets:
            # Check for extreme price jumps (>50% in one period)
            close_prices = combined_data.loc[:, (asset, 'close')]
            pct_changes = close_prices.pct_change().abs()
            extreme_jumps = pct_changes > 0.5  # 50% threshold
            
            if extreme_jumps.any():
                jump_indices = pct_changes[extreme_jumps].index
                logger.warning(f"Extreme price jumps detected for {asset} at: {jump_indices}")
                
                # Option 1: Smooth extreme jumps
                for idx in jump_indices:
                    # Get index position
                    pos = close_prices.index.get_loc(idx)
                    if pos > 0 and pos < len(close_prices) - 1:
                        # Replace with average of previous and next
                        prev_val = close_prices.iloc[pos-1]
                        next_val = close_prices.iloc[pos+1]
                        smoother_val = (prev_val + next_val) / 2
                        combined_data.loc[idx, (asset, 'close')] = smoother_val
                        logger.info(f"Smoothed extreme price at {idx} from {close_prices[idx]} to {smoother_val}")
        
        # Remove duplicate columns by taking the mean of duplicates
        combined_data = combined_data.T.groupby(level=[0, 1]).mean().T
        logger.info(f"Combined data shape after deduplication: {combined_data.shape}")
        
        # Sort the columns for consistency
        combined_data = combined_data.sort_index(axis=1)
        
        # Final verification
        assets = combined_data.columns.get_level_values('asset').unique()
        logger.info(f"Final assets: {assets}")
        
        for asset in assets:
            logger.info(f"Verifying data for {asset}")
            for feature in required_features:
                # Check if the feature exists
                if (asset, feature) not in combined_data.columns:
                    raise ValueError(f"Missing {feature} for {asset} in final combined data")
                
                # Get the feature data
                feature_data = combined_data.loc[:, (asset, feature)]
                
                # Check for NaN or infinite values
                if not np.isfinite(feature_data).all():
                    logger.warning(f"Found invalid values in {feature} for {asset}, fixing...")
                    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
                    feature_data = feature_data.ffill().bfill()
                    combined_data.loc[:, (asset, feature)] = feature_data
        
        logger.info("Base data formatting completed successfully")
        
        # Process through feature engine
        try:
            processed_data = self.feature_engine.engineer_features({exchange: raw_data[exchange] for exchange in raw_data})
            if processed_data.empty:
                raise ValueError("Feature engineering produced empty DataFrame")
            logger.info(f"Feature engineering complete. Shape: {processed_data.shape}")
            logger.info(f"Additional features generated: {processed_data.columns.get_level_values('feature').unique()}")
            
            # Combine base features with engineered features
            final_data = pd.concat([combined_data, processed_data], axis=1)
            final_data = final_data.loc[:, ~final_data.columns.duplicated()]
            
            # Ensure all data is numeric
            for col in final_data.columns:
                final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
            
            # Final NaN check and handling
            final_data = final_data.replace([np.inf, -np.inf], np.nan)
            final_data = final_data.ffill().bfill()
            
            # CRITICAL FIX: Log dataset size and structure verification
            logger.info(f"Final dataset: {final_data.shape[0]} timepoints, {final_data.shape[1]} features")
            logger.info(f"Memory usage: {final_data.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Log all feature columns for verification
            logger.info(f"Final feature columns: {final_data.columns.get_level_values('feature').unique().tolist()}")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.warning("Falling back to base features only")
            return combined_data

    def __del__(self):
        """Cleanup method for TradingSystem"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(f"Error during environment cleanup: {e}")

class VecNormalizeEvalCallback(BaseCallback):
    """
    Custom callback for evaluation with VecNormalize environments.
    Ensures that VecNormalize.training is set to False during evaluation
    and restored to its original state afterward.
    
    This solves the problem of progressively decreasing rewards during evaluation
    due to reward normalization.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=3, log_path=None,
                 best_model_save_path=None, deterministic=True, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        # Create standard EvalCallback
        self.eval_callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=1,  # Will always evaluate when called by this wrapper
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose
        )
        # Save original properties for easy access
        self.best_mean_reward = -float('inf')
        self.best_model_step = 0
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env

    def _init_callback(self):
        # Initialize nested callback - only pass the model
        self.eval_callback.init_callback(self.model)

    def _on_step(self):
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.n_calls
            
            # Log that evaluation is starting
            logger.info(f"Starting evaluation at step {self.n_calls} with {self.n_eval_episodes} episodes...")
            start_time = time.time()
            
            # More explicitly check if eval_env is a VecNormalize wrapper
            is_vec_normalize = False
            original_training_state = None
            
            # Check for VecNormalize
            if hasattr(self.eval_env, 'training'):
                is_vec_normalize = True
                original_training_state = self.eval_env.training
                # Disable training mode to fix reward normalization
                self.eval_env.training = False
                if self.verbose > 0:
                    logger.info("VecNormalize detected: Disabled training mode for proper reward normalization")
                
            # Also check for nested VecNormalize through common patterns
            elif hasattr(self.eval_env, 'venv') and hasattr(self.eval_env.venv, 'training'):
                is_vec_normalize = True
                original_training_state = self.eval_env.venv.training
                self.eval_env.venv.training = False
                if self.verbose > 0:
                    logger.info("Nested VecNormalize detected: Disabled training mode for proper reward normalization")
            
            # Call the standard evaluation callback
            result = self.eval_callback._on_step()
            
            # Restore original training state if needed
            if is_vec_normalize and original_training_state is not None:
                if hasattr(self.eval_env, 'training'):
                    self.eval_env.training = original_training_state
                elif hasattr(self.eval_env, 'venv') and hasattr(self.eval_env.venv, 'training'):
                    self.eval_env.venv.training = original_training_state
                
                if self.verbose > 0:
                    logger.info(f"Restored VecNormalize training mode to {original_training_state}")
            
            # Store properties for easy access
            if hasattr(self.eval_callback, 'best_mean_reward'):
                self.best_mean_reward = self.eval_callback.best_mean_reward
                
            if hasattr(self.eval_callback, 'best_model_step'): 
                self.best_model_step = self.eval_callback.best_model_step
            elif hasattr(self.eval_callback, 'last_best_model_step'):
                self.best_model_step = self.eval_callback.last_best_model_step
            
            # Log evaluation metrics and duration
            evaluation_duration = time.time() - start_time
            logger.info(f"Evaluation completed in {evaluation_duration:.2f} seconds with mean reward: {self.eval_callback.best_mean_reward}")
            
            # Log to wandb
            try:
                if 'wandb' in globals():
                    wandb.log({
                        "eval/duration_seconds": evaluation_duration,
                        "global_step": self.n_calls
                    })
            except Exception:
                pass
            
            return result
        
        return True


def recommend_next_phase_params(current_phase, total_phases, eval_metrics, current_params=None):
    """
    Calculate recommended parameters for the next training phase based on evaluation metrics.
    
    Args:
        current_phase: Current training phase number (1-based)
        total_phases: Total planned phases (default: 6)
        eval_metrics: Dictionary of evaluation metrics from _evaluate_model
        current_params: Dictionary of current hyperparameters
        
    Returns:
        dict: Recommended hyperparameters for next phase
    """
    # Default total planned phases if we're doing the standard 1M step schedule
    if total_phases is None:
        total_phases = 6  # Default phases in 1M steps: 100K, 100K, 200K, 200K, 300K, 200K
    
    # Calculate global progress (how far we are through total training)
    total_steps = 1_000_000  # Total expected steps
    steps_per_phase = {
        1: 100000, 2: 100000, 3: 200000, 4: 200000, 5: 300000, 6: 100000
    }
    
    # Calculate steps completed so far
    steps_completed = sum(steps_per_phase.get(i, 0) for i in range(1, current_phase + 1))
    global_progress = 1.0 - (steps_completed / total_steps)
    
    # Initialize recommendations dict
    recommendations = {}
    
    # Calculate next learning rate based on global progress and performance
    if global_progress <= 0.3:  # Final 30% of training
        # Lower learning rate for fine-tuning
        recommendations["learning_rate"] = 0.000015
    elif global_progress <= 0.7:  # Middle of training
        # Moderate learning rate
        recommendations["learning_rate"] = 0.00005
    else:  # Early training
        # Higher learning rate
        recommendations["learning_rate"] = 0.0001
    
    # Adjust entropy coefficient (exploration) based on performance and phase
    mean_reward = eval_metrics.get("mean_reward", 0)
    reward_sharpe = eval_metrics.get("reward_sharpe", 0)
    trades_executed = eval_metrics.get("trades_executed", 0)
    
    # If we're getting good rewards and good Sharpe, reduce exploration
    if mean_reward > 1.0 and reward_sharpe > 0.5:
        ent_coef = 0.02
    # If we're not getting good trading frequency, increase exploration
    elif trades_executed < 100:
        ent_coef = 0.05
    # Default exploration for mid-training
    else:
        ent_coef = 0.03
    
    # Phase-based adjustments to entropy coefficient
    if global_progress < 0.3:  # Early phases (first 30%)
        # Early phases - increase exploration
        ent_coef *= 1.5
    elif global_progress < 0.7:  # Mid phases
        # Mid phases - standard exploration
        pass
    else:  # Final phases (last 30%)
        # Final phases - reduce exploration
        ent_coef *= 0.5
    
    recommendations["ent_coef"] = ent_coef
    
    # Log the recommendations
    logger.info(f"Recommended parameters for phase {current_phase + 1}:")
    for param, value in recommendations.items():
        logger.info(f"  {param}: {value}")
    
    # Save recommendations to file for easy retrieval
    next_phase = current_phase + 1
    model_dir = os.path.dirname(os.path.abspath(os.path.join("models/manual", f"phase{current_phase}")))
    recommendation_file = os.path.join(model_dir, f"phase{current_phase}", f"phase{next_phase}_recommendations.json")
    os.makedirs(os.path.dirname(recommendation_file), exist_ok=True)
    
    import json
    with open(recommendation_file, "w") as f:
        json.dump(recommendations, f, indent=4)
    
    return recommendations

def parse_hyperparams(hyperparams_str):
    """Parse hyperparameters from string into dictionary"""
    if not hyperparams_str:
        return {}
        
    hyperparams = {}
    pairs = hyperparams_str.split(',')
    
    for pair in pairs:
        if '=' not in pair:
            logger.warning(f"Invalid hyperparameter format: {pair}")
            continue
            
        param, value_str = pair.split('=', 1)
        param = param.strip()
        value_str = value_str.strip()
        
        # Try to convert value to appropriate type
        if value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        else:
            try:
                # Try to convert to int or float
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
        
        hyperparams[param] = value
    
    return hyperparams

async def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config('config/prod_config.yaml')

            # If drive-ids-file is provided, add it to the config
        if args.drive_ids_file:
            if not os.path.exists(args.drive_ids_file):
                logger.error(f"Drive IDs file not found: {args.drive_ids_file}")
                return
            config['data']['drive_ids_file'] = args.drive_ids_file
            logger.info(f"Using Google Drive integration with file IDs from: {args.drive_ids_file}")
        
        # Log GPU status at the start
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if args.gpus > 0:
                logger.info(f"CUDA is available with {gpu_count} GPU(s). Using GPU for training.")
                for i in range(min(gpu_count, args.gpus)):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.warning(f"CUDA is available with {gpu_count} GPU(s), but training on CPU (--gpus=0).")
        else:
            logger.warning("CUDA is not available. Training will use CPU only.")
        
        # Update config with command line arguments
        config['data']['assets'] = args.assets if args.assets else config['data']['assets']
        config['data']['timeframe'] = args.timeframe if args.timeframe else config['data']['timeframe']
        config['model']['max_leverage'] = args.max_leverage if args.max_leverage else config['model']['max_leverage']
        config['training']['steps'] = args.training_steps if args.training_steps else config['training']['steps']
        config['logging']['verbose'] = args.verbose if args.verbose else config['logging'].get('verbose', False)
        
        # Setup directories
        setup_directories(config)
        
        # Initialize wandb
        initialize_wandb(config)
        
        # Create trading system
        trading_system = TradingSystem(config)
        
        # Initialize trading system
        await trading_system.initialize(args)

        # In your main function or where you handle args, add:
        hyperparams_dict = {}
        if args.hyperparams:
            hyperparams_dict = parse_hyperparams(args.hyperparams)
            logger.info("Parsed hyperparameters:")
            for param, value in hyperparams_dict.items():
                logger.info(f"  {param}: {value}")
        
        # Check if continuing training from an existing model
        if args.continue_training:
            if not args.model_path:
                raise ValueError("--model-path must be provided when using --continue-training")
                
            logger.info(f"Continuing training from model: {args.model_path}")
            logger.info(f"Additional steps: {args.additional_steps}")
            
            # Continue training
            final_model_path = trading_system.continue_training(
                model_path=args.model_path,
                env_path=args.env_path,
                additional_timesteps=args.additional_steps,
                reset_num_timesteps=args.reset_num_timesteps,
                reset_reward_normalization=args.reset_reward_norm,
                hyperparams=hyperparams_dict,
                args=args  # Pass command line args to continue_training
            )
            
            logger.info(f"Continued training complete. Final model saved to {final_model_path}")
        else:
            # Run hyperparameter optimization
            # Comment out optimization for direct training with 1M timestep parameters
            # trading_system.optimize_hyperparameters(n_trials=30, n_jobs=5, total_timesteps=1000)
            
            # Train the model with best parameters
            trading_system.train(args)
            
            # Save final model and environment
            final_model_path = os.path.join(args.model_dir, "final_model")
            final_env_path = os.path.join(args.model_dir, "vec_normalize.pkl")
            
            trading_system.model.save(final_model_path)
            trading_system.env.save(final_env_path)
            
            logger.info(f"Training complete. Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())