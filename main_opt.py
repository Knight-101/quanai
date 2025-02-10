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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
# from data_collection.collect_multimodal import MultiModalDataCollector
# from data_system.multimodal_feature_extractor import MultiModalPerpFeatureExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        'data'
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
        self.env = self._create_environment(self.processed_data)
        
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
        
    def _create_environment(self, data: pd.DataFrame) -> InstitutionalPerpetualEnv:
        """Consolidated environment creation"""
        env = InstitutionalPerpetualEnv(
            df=data,
            initial_balance=self.config['trading']['initial_balance'],
            max_leverage=self.config['trading']['max_leverage'],
            transaction_fee=self.config['trading']['transaction_fee'],
            funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
            risk_free_rate=self.config['trading']['risk_free_rate'],
            max_drawdown=self.config['risk_management']['limits']['max_drawdown'],
            window_size=self.config['model']['window_size']
        )
        
        # Wrap environment
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env
        
    def _setup_model(self, args=None) -> PPO:
        """Consolidated model setup"""
        policy_kwargs = {
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128]
            },
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128}
        }
        
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
            tensorboard_log=self.config['logging']['log_dir'],
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
            
            # Log best parameters
            logger.info("\nOptimization finished!")
            logger.info("Best trial:")
            trial = self.study.best_trial
            logger.info(f"  Value (Sharpe Ratio): {trial.value:.4f}")
            logger.info("  Params:")
            for key, value in trial.params.items():
                logger.info(f"    {key}: {value}")
                
            # Save study results
            df = self.study.trials_dataframe()
            df.to_csv("optuna_results.csv")
            logger.info(f"\nStudy results saved to optuna_results.csv")
            
            # Update model with best parameters
            self.update_model_with_best_params()
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
            
    def objective(self, trial: optuna.Trial, n_steps: int) -> float:
        """Optuna objective function for hyperparameter optimization"""
        # Print trial start
        logger.info(f"\nStarting trial {trial.number}")
        
        # Define hyperparameter search space
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
            "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
            "n_epochs": trial.suggest_int("n_epochs", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
            "target_kl": trial.suggest_float("target_kl", 0.01, 0.05)
        }
        
        # Log trial parameters
        logger.info("\nTraining with parameters:")
        for key, value in params.items():
            logger.info(f"    {key}: {value}")
        
        # Neural network architecture
        net_arch = {
            "pi": [
                trial.suggest_int("pi_1", 64, 256, step=64),
                trial.suggest_int("pi_2", 32, 128, step=32)
            ],
            "vf": [
                trial.suggest_int("vf_1", 64, 256, step=64),
                trial.suggest_int("vf_2", 32, 128, step=32)
            ]
        }
        
        # Log network architecture
        logger.info("\nNetwork architecture:")
        logger.info(f"    Policy network: {net_arch['pi']}")
        logger.info(f"    Value network: {net_arch['vf']}")
        
        # Update policy kwargs
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128}
        }
        
        # Create model with trial parameters
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            target_kl=params["target_kl"],
            tensorboard_log=self.config['logging']['log_dir'],
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=self.device
        )
        
        try:
            # Train for specified number of steps
            logger.info(f"\nTraining model for {n_steps} steps...")
            model.learn(total_timesteps=n_steps)
            
            # Evaluate model
            logger.info("\nEvaluating model performance...")
            eval_metrics = self._evaluate_model(model)
            
            # Log metrics
            trial.set_user_attr("sharpe_ratio", eval_metrics["sharpe_ratio"])
            trial.set_user_attr("max_drawdown", eval_metrics["max_drawdown"])
            trial.set_user_attr("mean_return", eval_metrics["mean_return"])
            
            # Log evaluation results
            logger.info("\nTrial results:")
            logger.info(f"    Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}")
            logger.info(f"    Mean Return: {eval_metrics['mean_return']:.4f}")
            logger.info(f"    Max Drawdown: {eval_metrics['max_drawdown']:.4f}")
            
            # Compare with best trial so far
            if len(self.study.trials) > 1:
                best_trial = self.study.best_trial
                logger.info("\nComparison with best trial:")
                logger.info(f"    Current trial Sharpe: {eval_metrics['sharpe_ratio']:.4f}")
                logger.info(f"    Best trial Sharpe: {best_trial.value:.4f}")
                if eval_metrics['sharpe_ratio'] > best_trial.value:
                    logger.info("    New best trial found!")
            
            # Save trial model
            model.save(f"models/trial_{trial.number}_model")
            self.env.save(f"models/trial_{trial.number}_env.pkl")
            logger.info(f"\nTrial {trial.number} model saved")
            
            return eval_metrics["sharpe_ratio"]
            
        except Exception as e:
            logger.error(f"\nError in trial {trial.number}: {str(e)}")
            raise optuna.TrialPruned()
            
    def _evaluate_model(self, model, n_eval_episodes=5):
        """Evaluate model performance"""
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for _ in range(n_eval_episodes):
            obs = self.env.reset()
            done = False
            episode_return = 0
            returns_list = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_return += reward
                returns_list.append(reward)
                
            returns.append(episode_return)
            
            # Calculate Sharpe ratio
            returns_array = np.array(returns_list)
            if len(returns_array) > 1:
                sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-6) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns_list)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            max_drawdowns.append(max_drawdown)
        
        return {
            "mean_return": np.mean(returns),
            "sharpe_ratio": np.mean(sharpe_ratios),
            "max_drawdown": np.mean(max_drawdowns)
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
            tensorboard_log=self.config['logging']['log_dir'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device
        )
        
        logger.info("Model updated with best parameters from optimization")

    def train(self):
        """Consolidated training method"""
        if not self.model or not self.env:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        logger.info("Starting model training...")
        
        # Training loop with curriculum
        for curriculum_stage in self.config['training']['curriculum_stages']:
            logger.info(f"Starting curriculum stage: {curriculum_stage['name']}")
            
            # Update environment parameters for this stage
            self.env.envs[0].update_parameters(**curriculum_stage['env_params'])
            
            # Setup callbacks for this stage
            callbacks = []
            if 'callbacks' in self.config['training']:
                for callback_config in self.config['training']['callbacks']:
                    if callback_config['type'] == 'checkpoint':
                        callbacks.append(CheckpointCallback(
                            save_freq=callback_config['save_freq'],
                            save_path=self.config['model']['checkpoint_dir'],
                            name_prefix=f"stage_{curriculum_stage['name']}"
                        ))
                    elif callback_config['type'] == 'eval':
                        callbacks.append(EvalCallback(
                            eval_env=self.env,
                            n_eval_episodes=callback_config['n_eval_episodes'],
                            eval_freq=callback_config['eval_freq'],
                            log_path=self.config['logging']['log_dir'],
                            deterministic=True
                        ))
                    # Add more callback types as needed
            
            # Train for this stage
            self.model.learn(
                total_timesteps=curriculum_stage['timesteps'],
                callback=callbacks if callbacks else None
            )
            
            # Evaluate stage performance
            self._evaluate_stage(curriculum_stage['name'])
            
            # Save stage checkpoint
            self._save_checkpoint(f"stage_{curriculum_stage['name']}")
            
    def _evaluate_stage(self, stage: str):
        """Helper method for stage evaluation"""
        logger.info(f"Evaluating model performance for stage: {stage}")
        
        eval_episodes = self.config['training']['eval_episodes']
        metrics = self._run_evaluation_episodes(eval_episodes)
        
        # Log metrics
        wandb.log({
            f'{stage}/mean_return': metrics['mean_return'],
            f'{stage}/sharpe_ratio': metrics['sharpe_ratio'],
            f'{stage}/max_drawdown': metrics['max_drawdown']
        })
        
    def _run_evaluation_episodes(self, n_episodes):
        """Helper method to run evaluation episodes"""
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for _ in range(n_episodes):
            episode_metrics = self._run_single_episode()
            returns.append(episode_metrics['return'])
            sharpe_ratios.append(episode_metrics['sharpe'])
            max_drawdowns.append(episode_metrics['max_drawdown'])
            
        return {
            'mean_return': np.mean(returns),
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': np.mean(max_drawdowns)
        }
        
    def _run_single_episode(self):
        """Helper method to run a single evaluation episode"""
        obs = self.env.reset()
        done = False
        episode_return = 0
        info = {}
        
        while not done:
            action = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)
            episode_return += reward
            self.dashboard.update_metrics(info)
        
        return {
            'return': episode_return,
            'sharpe': info['risk_metrics']['sharpe'],
            'max_drawdown': info['risk_metrics']['max_drawdown']
        }
        
    def _save_checkpoint(self, name: str):
        """Helper method to save model checkpoint"""
        save_path = os.path.join(self.config['model']['checkpoint_dir'], f"{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        self.model.save(save_path)
        logger.info(f"Model checkpoint saved to {save_path}")

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
                
                # Create a copy to avoid modifying original data
                df = symbol_data.copy()
                logger.info(f"Original columns: {df.columns}")
                
                # Create formatted DataFrame
                formatted_data = pd.DataFrame(index=df.index)
                
                # Add OHLCV data
                for col in ['open', 'high', 'low', 'close']:
                    if col not in df.columns:
                        raise ValueError(f"Missing required price column {col} for {formatted_symbol}")
                    formatted_data[col] = df[col].astype(float)
                
                # Handle volume data
                if 'volume' in df.columns:
                    formatted_data['volume'] = df['volume'].astype(float)
                else:
                    # Generate synthetic volume based on price volatility
                    returns = formatted_data['close'].pct_change()
                    vol = returns.rolling(window=20).std().fillna(0.01)
                    formatted_data['volume'] = formatted_data['close'] * vol * 1000
                
                # Ensure volume is positive and non-zero
                formatted_data['volume'] = formatted_data['volume'].clip(lower=1.0)
                
                # Handle funding rate
                if 'funding_rate' in df.columns:
                    formatted_data['funding_rate'] = df['funding_rate'].astype(float)
                else:
                    # Use small random funding rate
                    formatted_data['funding_rate'] = np.random.normal(0, 0.0001, size=len(formatted_data))
                
                # Handle market depth
                if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
                    formatted_data['bid_depth'] = df['bid_depth'].astype(float)
                    formatted_data['ask_depth'] = df['ask_depth'].astype(float)
                else:
                    # Generate synthetic depth based on volume
                    formatted_data['bid_depth'] = formatted_data['volume'] * 0.4
                    formatted_data['ask_depth'] = formatted_data['volume'] * 0.4
                
                # Ensure depth is positive and non-zero
                formatted_data['bid_depth'] = formatted_data['bid_depth'].clip(lower=1.0)
                formatted_data['ask_depth'] = formatted_data['ask_depth'].clip(lower=1.0)
                
                # Add volatility
                returns = formatted_data['close'].pct_change()
                formatted_data['volatility'] = returns.rolling(window=20).std().fillna(0.01)
                
                # Forward fill any NaN values first
                formatted_data = formatted_data.ffill()
                
                # Then backward fill any remaining NaN values at the start
                formatted_data = formatted_data.bfill()
                
                # Ensure no NaN values remain
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
        
        # Combine all symbol data
        if not symbol_dfs:
            raise ValueError("No valid data to process")
        
        # Concatenate all symbols' data and handle duplicates
        combined_data = pd.concat(symbol_dfs, axis=1)
        logger.info(f"Combined data shape before deduplication: {combined_data.shape}")
        
        # Remove duplicate columns by taking the mean of duplicates
        combined_data = combined_data.groupby(level=[0, 1], axis=1).mean()
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
                    combined_data.loc[:, (asset, feature)] = feature_data.replace(
                        [np.inf, -np.inf], np.nan
                    ).fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Data formatting completed successfully")
        return combined_data

async def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config('config/prod_config.yaml')
        
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