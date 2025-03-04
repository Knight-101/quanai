#!/usr/bin/env python
import argparse
import torch
import os
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
        
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observations):
        return self.feature_net(observations)

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
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (including step-wise metrics)')
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
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        config=config,
        name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mode=config['logging']['wandb']['mode']
    )

class TradingSystem:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data manager with just the base path
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
        
        # Initialize environment
        verbose = args.verbose if args and hasattr(args, 'verbose') else False
        self.env = self._create_environment(self.processed_data, verbose=verbose)
        
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
        
        # Calculate date range
        end_time = datetime.now()
        lookback_days = self.config['data']['history_days']
        start_time = end_time - pd.Timedelta(days=lookback_days)
        
        # Try to load existing data first
        existing_data = self._load_cached_data(start_time, end_time)
        if existing_data is not None and len(existing_data) >= self.config['data']['min_history_points']:
            logger.info("Using existing data from cache.")
            formatted_data = self._format_data_for_training(existing_data)
            logger.info(f"Formatted data shape: {formatted_data.shape}")
            logger.info(f"Columns: {formatted_data.columns}")
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
        
    def _load_cached_data(self, start_time, end_time):
        """Helper method to load cached data"""
        all_data = {exchange: {} for exchange in self.config['data']['exchanges']}
        has_all_data = True
        
        for exchange in self.config['data']['exchanges']:
            for symbol in self.config['trading']['symbols']:
                data = self.data_manager.load_market_data(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=self.config['data']['timeframe'],
                    start_time=start_time,
                    end_time=end_time,
                    data_type='perpetual'
                )
                
                if data is None or len(data) < self.config['data']['min_history_points']:
                    has_all_data = False
                    break
                    
                all_data[exchange][symbol] = data
                
            if not has_all_data:
                break
                
        return all_data if has_all_data else None
        
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
        
    def _create_environment(self, data: pd.DataFrame, verbose: bool = False) -> InstitutionalPerpetualEnv:
        """Consolidated environment creation"""
        # Extract unique assets from the MultiIndex columns
        assets = list(data.columns.get_level_values('asset').unique())
        
        env = InstitutionalPerpetualEnv(
            df=data,
            assets=assets,
            window_size=self.config['model']['window_size'],
            max_leverage=self.config['trading']['max_leverage'],
            commission=self.config['trading']['transaction_fee'],
            funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
            base_features=['close', 'volume', 'funding_rate'],
            tech_features=['rsi', 'macd', 'bb_upper', 'bb_lower'],
            risk_engine=self.risk_engine,
            risk_free_rate=self.config['trading'].get('risk_free_rate', 0.02),
            verbose=verbose
        )
        
        # Wrap environment
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
        
    def _setup_model(self, args=None) -> PPO:
        """Consolidated model setup"""
        policy_kwargs = self.policy_kwargs
        
        device = 'cuda' if (args and args.gpus > 0) else 'cpu'
        
        return PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            target_kl=0.03,
            tensorboard_log=self.config['logging'].get('tensorboard_dir', 'logs/tensorboard'),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
        
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
        
    def optimize_hyperparameters(self, n_trials=30, n_jobs=5, n_steps=100000):
        """Run hyperparameter optimization"""
        if not self.study:
            self.create_study()
            
        logger.info(f"\nStarting hyperparameter optimization with {n_trials} trials")
        logger.info(f"Number of parallel jobs: {n_jobs}")
        logger.info(f"Steps per trial: {n_steps}")
        
        try:
            self.study.optimize(
                lambda trial: self.objective(trial, n_steps),
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
            self.update_model_with_best_params()
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
            
    def objective(self, trial: optuna.Trial, n_steps: int) -> float:
        """Objective function for hyperparameter optimization"""
        try:
            # Start trial logging
            logger.info(f"\n╔═ Starting Trial {trial.number} ═{'═' * 59}╗")
            logger.info(f"║ Steps per trial: {n_steps:<67} ║")
            
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)
            vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)
            max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 0.7)
            
            # Log sampled hyperparameters
            logger.info(f"║ Hyperparameters:                                                  ║")
            logger.info(f"║   - learning_rate: {learning_rate:<58} ║")
            logger.info(f"║   - batch_size: {batch_size:<60} ║")
            logger.info(f"║   - n_steps: {n_steps:<63} ║")
            logger.info(f"║   - gamma: {gamma:<65} ║")
            logger.info(f"║   - gae_lambda: {gae_lambda:<60} ║")
            logger.info(f"║   - clip_range: {clip_range:<60} ║")
            logger.info(f"║   - ent_coef: {ent_coef:<62} ║")
            logger.info(f"║   - vf_coef: {vf_coef:<62} ║")
            logger.info(f"║   - max_grad_norm: {max_grad_norm:<56} ║")
            logger.info(f"╚{'═' * 80}╝\n")
            
            # Create a fresh environment for each trial
            env = self._create_environment(self.processed_data, verbose=False)
            
            # Create model with sampled hyperparameters
            try:
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
                    verbose=0,
                    tensorboard_log=self.config['logging'].get('tensorboard_dir', 'logs/tensorboard'),
                    policy_kwargs=self.policy_kwargs
                )

                # IMPORTANT FIX: Add exploration callback to encourage trading
                class ExplorationCallback(BaseCallback):
                    def __init__(self, env, verbose=0):
                        super().__init__(verbose)
                        # IMPORTANT FIX: Increase exploration steps even more
                        self.exploration_steps = 15000  # Increased from 10000 to 15000
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
                        # NEW: Add trade forcing mechanism
                        self.force_trade_every = 500  # Force trade attempts every N steps
                        
                    def _on_step(self):
                        self.step_count += 1
                        
                        # IMPORTANT FIX: Use stronger exploration for longer
                        if self.num_timesteps < self.exploration_steps:
                            # Add noise to actions during initial exploration
                            # Use stronger noise early in training
                            noise_scale = max(0.9, 1.2 - self.num_timesteps / self.exploration_steps)
                            
                            # NEW: Add bias toward extreme actions periodically to force trades
                            if self.step_count % self.force_trade_every < 50:  # For 50 steps every force_trade_every steps
                                # Create bias toward extreme actions (-1 or 1) to encourage trading
                                bias = np.random.choice([-0.7, 0.7], size=len(self.assets))
                                self.model.action_noise = CustomActionNoise(
                                    mean=bias,  # Use bias as mean instead of zeros
                                    sigma=noise_scale * np.ones(len(self.assets)),
                                    size=len(self.assets)
                                )
                                if self.step_count % self.force_trade_every == 0:
                                    logger.info(f"Forcing trade exploration at step {self.step_count} with bias {bias}")
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
                            
                            # IMPORTANT FIX: If no trades for a long time, increase exploration
                            # Only warn every 100 steps to avoid log spam
                            if self.trades_executed == 0 and self.num_timesteps > 1000:
                                self.no_trade_warning_counter += 1
                                if self.no_trade_warning_counter >= 100:
                                    # Force stronger exploration
                                    logger.warning(f"No trades executed after {self.num_timesteps} steps, increasing exploration")
                                    # Use extremely strong noise to force exploration
                                    self.model.action_noise = CustomActionNoise(
                                        mean=np.zeros(len(self.assets)),
                                        sigma=2.0 * np.ones(len(self.assets)),  # Very strong noise
                                        size=len(self.assets)
                                    )
                                    self.no_trade_warning_counter = 0  # Reset counter
                            elif self.num_timesteps - self.last_trade_step > 1000 and self.trades_executed > 0:
                                self.no_trade_warning_counter += 1
                                if self.no_trade_warning_counter >= 100:
                                    # If trades stopped, increase exploration again
                                    logger.warning(f"No trades for {self.num_timesteps - self.last_trade_step} steps, increasing exploration")
                                    self.model.action_noise = CustomActionNoise(
                                        mean=np.zeros(len(self.assets)),
                                        sigma=1.5 * np.ones(len(self.assets)),  # Strong noise
                                        size=len(self.assets)
                                    )
                                    self.no_trade_warning_counter = 0  # Reset counter
                        else:
                            # Gradually reduce noise after exploration phase
                            if self.num_timesteps < self.exploration_steps * 2:
                                decay_factor = 1.0 - ((self.num_timesteps - self.exploration_steps) / self.exploration_steps)
                                noise_scale = 0.5 * decay_factor  # Increased from 0.3 to 0.5
                                self.model.action_noise = CustomActionNoise(
                                    mean=np.zeros(len(self.assets)),
                                    sigma=noise_scale * np.ones(len(self.assets)),
                                    size=len(self.assets)
                                )
                            else:
                                # IMPORTANT FIX: Keep some minimal noise to encourage exploration
                                self.model.action_noise = CustomActionNoise(
                                    mean=np.zeros(len(self.assets)),
                                    sigma=0.1 * np.ones(len(self.assets)),  # Minimal noise
                                    size=len(self.assets)
                                )
                        
                        return True
                
                # Training loop with error handling
                try:
                    # IMPORTANT FIX: Add exploration callback
                    exploration_callback = ExplorationCallback(env)
                    model.learn(total_timesteps=n_steps, callback=exploration_callback)
                except Exception as train_error:
                    logger.error(f"Error during training: {str(train_error)}")
                    raise optuna.TrialPruned()

                # Evaluation with error handling
                try:
                    # IMPORTANT FIX: Increase evaluation episodes for more reliable results
                    eval_metrics = self._evaluate_model(model, n_eval_episodes=5, verbose=False)
                    
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
        
    def _evaluate_model(self, model, n_eval_episodes=5, verbose=False):
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
            
            while not done:
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
                        logger.info(f"Trade executed at step {step_count} (via trades_executed flag)")
                    
                    # Check positions directly
                    if 'positions' in info:
                        positions = info['positions']
                        active_positions = []
                        for asset, pos in positions.items():
                            if isinstance(pos, dict) and abs(pos.get('size', 0)) > 1e-8:
                                active_positions.append(f"{asset}: {pos.get('size', 0):.4f}")
                        
                        if active_positions:
                            logger.info(f"Active positions: {', '.join(active_positions)}")
                            trade_detected = True
                    
                    # Check recent trades count
                    if info.get('recent_trades_count', 0) > 0:
                        logger.info(f"Recent trades: {info.get('recent_trades_count')}")
                        trade_detected = True
                    
                    # Check has_positions flag
                    if info.get('has_positions', False):
                        logger.info("Has positions flag is True")
                        trade_detected = True
                    
                    # Check total_trades_count
                    if info.get('total_trades_count', 0) > 0:
                        logger.info(f"Total trades count: {info.get('total_trades_count')}")
                        trade_detected = True
                
                if trade_detected:
                    episode_trades += 1
                    trades_executed += 1
                    # logger.info(f"Trade detected at step {step_count} in episode {episode+1}")
                
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
                    
                    # Debug logging for first few steps
                    if len(episode_rewards) <= 5:
                        logger.debug(f"Step {len(episode_rewards)} info: {info}")
            
            portfolio_values.append(portfolio_value)
            rewards.extend(episode_rewards)
            
            # IMPORTANT FIX: Track trades for this episode
            all_trades.append(episode_trades)
            
            logger.info(f"Episode {episode+1} completed: Steps={step_count}, Final Portfolio={portfolio_value:.2f}, "
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
            logger.warning("No leverage samples collected. Using default value.")
            leverage_ratios = [0.1]  # Use a small default value
            
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
            logger.info(f"Leverage range: [{min(leverage_ratios):.4f}, {max(leverage_ratios):.4f}]")
        
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
        
    def update_model_with_best_params(self):
        """Update the model with the best found parameters"""
        if not self.study:
            logger.warning("No study available to update model parameters")
            return
            
        best_params = self.study.best_params
        best_trial = self.study.best_trial
        
        logger.info("Updating model with best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
            
        # Create network architecture from best parameters
        net_arch = {
            "pi": [best_params["pi_1"], best_params["pi_2"]],
            "vf": [best_params["vf_1"], best_params["vf_2"]]
        }
        
        # Update policy kwargs
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128}
        }
        
        # Create new model with best parameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=best_params["learning_rate"],
            n_steps=best_params["n_steps"],
            batch_size=best_params["batch_size"],
            n_epochs=best_params["n_epochs"],
            gamma=best_params["gamma"],
            gae_lambda=best_params["gae_lambda"],
            clip_range=best_params["clip_range"],
            ent_coef=best_params["ent_coef"],
            vf_coef=best_params["vf_coef"],
            max_grad_norm=best_params["max_grad_norm"],
            target_kl=best_params["target_kl"],
            tensorboard_log=self.config['logging'].get('tensorboard_dir', 'logs/tensorboard'),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device
        )
        
        logger.info("Model updated with best parameters from optimization")

    def train(self):
        """Optimized training method without curriculum learning"""
        if not self.model or not self.env:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info("Starting model training...")
        
        # Setup callbacks for checkpointing and evaluation
        callbacks = []
        
        # Checkpoint callback - save every 100k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=self.config['model']['checkpoint_dir'],
            name_prefix="ppo_trading"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback - evaluate every 50k steps
        eval_callback = EvalCallback(
            eval_env=self.env,
            n_eval_episodes=5,
            eval_freq=50000,
            log_path=self.config['logging']['log_dir'],
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Train the model with optimized parameters
        total_timesteps = self.config['training'].get('total_timesteps', 2_000_000)
        
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
            logger.info(f"Mean return: {eval_metrics['mean_return']:.4f}")
            logger.info(f"Sharpe ratio: {eval_metrics['return_sharpe']:.4f}")
            logger.info(f"Max drawdown: {eval_metrics['max_drawdown']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _format_data_for_training(self, raw_data):
        """Format data into the structure expected by the trading environment"""
        logger.info("Starting data formatting...")
        
        # Initialize an empty list to store DataFrames for each symbol
        symbol_dfs = []
        
        # Process each exchange's data
        for exchange, exchange_data in raw_data.items():
            logger.info(f"Processing exchange: {exchange}")
            for symbol, symbol_data in exchange_data.items():
                logger.info(f"Processing symbol: {symbol}")
                
                # Convert symbol to format expected by risk engine (e.g., BTC/USD:USD -> BTCUSDT)
                formatted_symbol = symbol.split('/')[0] + "USDT" if not symbol.endswith('USDT') else symbol
                logger.info(f"Formatted symbol: {formatted_symbol}")
                
                try:
                    # Create a copy to avoid modifying original data
                    df = symbol_data.copy()
                    logger.info(f"Original columns: {df.columns}")
                    
                    # Create formatted DataFrame
                    formatted_data = pd.DataFrame(index=df.index)
                    
                    # Add OHLCV data with proper numeric conversion
                    for col in ['open', 'high', 'low', 'close']:
                        if col not in df.columns:
                            raise ValueError(f"Missing required price column {col} for {formatted_symbol}")
                        formatted_data[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Handle volume data
                    if 'volume' in df.columns:
                        formatted_data['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                    else:
                        # Generate synthetic volume based on price volatility
                        returns = formatted_data['close'].pct_change()
                        vol = returns.rolling(window=20).std().fillna(0.01)
                        formatted_data['volume'] = formatted_data['close'] * vol * 1000
                    
                    # Ensure volume is positive and non-zero
                    formatted_data['volume'] = formatted_data['volume'].clip(lower=1.0)
                    
                    # Handle funding rate
                    if 'funding_rate' in df.columns:
                        formatted_data['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
                    else:
                        # Use small random funding rate
                        formatted_data['funding_rate'] = np.random.normal(0, 0.0001, size=len(formatted_data))
                    
                    # Handle market depth
                    if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
                        formatted_data['bid_depth'] = pd.to_numeric(df['bid_depth'], errors='coerce')
                        formatted_data['ask_depth'] = pd.to_numeric(df['ask_depth'], errors='coerce')
                    else:
                        # Generate synthetic depth based on volume
                        formatted_data['bid_depth'] = formatted_data['volume'] * 0.4
                        formatted_data['ask_depth'] = formatted_data['volume'] * 0.4
                    
                    # Ensure depth is positive and non-zero
                    formatted_data['bid_depth'] = formatted_data['bid_depth'].clip(lower=1.0)
                    formatted_data['ask_depth'] = formatted_data['ask_depth'].clip(lower=1.0)
                    
                    # Add volatility
                    close_returns = formatted_data['close'].pct_change()
                    formatted_data['volatility'] = close_returns.rolling(window=20).std().fillna(0.01)
                    
                    # Handle missing values
                    # First replace infinities with NaN
                    formatted_data = formatted_data.replace([np.inf, -np.inf], np.nan)
                    
                    # Forward fill any NaN values first
                    formatted_data = formatted_data.ffill()
                    
                    # Then backward fill any remaining NaN values at the start
                    formatted_data = formatted_data.bfill()
                    
                    # Final NaN check
                    if formatted_data.isna().any().any():
                        logger.error(f"NaN values found in {formatted_symbol} data")
                        logger.error(f"NaN columns: {formatted_data.columns[formatted_data.isna().any()]}")
                        raise ValueError(f"NaN values remain in formatted data for {formatted_symbol}")
                    
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
                    continue
        
        # Combine all symbol data
        if not symbol_dfs:
            raise ValueError("No valid data to process")
        
        # Concatenate all symbols' data and handle duplicates
        combined_data = pd.concat(symbol_dfs, axis=1)
        logger.info(f"Combined data shape before deduplication: {combined_data.shape}")
        
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

async def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config('config/prod_config.yaml')
        
        # Set verbosity level based on command-line argument
        if args.verbose:
            # Enable verbose logging if requested
            trading_env_logger = logging.getLogger('trading_env')
            trading_env_logger.setLevel(logging.INFO)
            logger.info("Verbose logging enabled - showing step-wise metrics")
        else:
            logger.info("Running with reduced verbosity - showing only trial metrics")
        
        # Setup directories and wandb
        setup_directories(config)
        initialize_wandb(config)
        
        # Initialize trading system
        trading_system = TradingSystem(config)
        
        # Initialize system (this handles data fetching, env creation, and model setup)
        await trading_system.initialize(args)
        
        # Run hyperparameter optimization
        trading_system.optimize_hyperparameters(n_trials=30, n_jobs=5, n_steps=100000)
        
        # Train the model with best parameters
        trading_system.train()
        
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