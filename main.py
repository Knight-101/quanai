#!/usr/bin/env python
import argparse
import torch
import os
from datetime import datetime
import asyncio
from data_system.derivative_data_fetcher import PerpetualDataFetcher
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from training.hierarchical_ppo import HierarchicalPPO
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
        
        # The training manager will be initialized later after data is fetched and processed
        self.training_manager = None
        
    async def initialize_data(self):
        """Initialize data by fetching and processing if not already present"""
        logger.info("Initializing data...")
        
        # Check if data exists
        start_time = datetime.now() - pd.Timedelta(days=self.config['data']['history_days'])
        end_time = datetime.now()
        
        # Try to load existing data for any symbol/exchange combination
        has_data = False
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
                if data is not None:
                    has_data = True
                    break
            if has_data:
                break
        
        if not has_data:
            logger.info("No existing data found. Fetching initial data...")
            # Fetch new data
            raw_data = await self.data_fetcher.fetch_derivative_data()
            
            # Save market data
            logger.info("Saving market data...")
            for exchange, exchange_data in raw_data.items():
                for symbol, symbol_data in exchange_data.items():
                    self.data_manager.save_market_data(
                        data=symbol_data,
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=self.config['data']['timeframe'],
                        data_type='perpetual'
                    )
            
            # Process and save features
            logger.info("Engineering and saving features...")
            processed_data = self.feature_engine.engineer_features(raw_data)
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
            logger.info("Data initialization complete!")
        else:
            logger.info("Using existing data from cache.")
        
    async def fetch_and_process_data(self):
        """Fetch and process market data"""
        logger.info("Checking for existing data...")
        
        # Try to load existing data first
        start_time = datetime.now() - pd.Timedelta(days=self.config['data']['history_days'])
        end_time = datetime.now()
        
        all_data = {}
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
                
                if data is None:
                    logger.info(f"No cached data found for {exchange}:{symbol}, fetching from exchange...")
                    # Fetch new data
                    raw_data = await self.data_fetcher.fetch_derivative_data()
                    
                    # Save the raw data
                    for exch, exchange_data in raw_data.items():
                        for sym, symbol_data in exchange_data.items():
                            self.data_manager.save_market_data(
                                data=symbol_data,
                                exchange=exch,
                                symbol=sym,
                                timeframe=self.config['data']['timeframe'],
                                data_type='perpetual'
                            )
                    
                    # Update our data variable
                    if exchange in raw_data and symbol in raw_data[exchange]:
                        data = raw_data[exchange][symbol]
                
                if data is not None:
                    if exchange not in all_data:
                        all_data[exchange] = {}
                    all_data[exchange][symbol] = data
        
        logger.info("Engineering features...")
        processed_data = self.feature_engine.engineer_features(all_data)
        
        # Save engineered features
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
        
        return processed_data
        
    def create_training_env(self, data: pd.DataFrame) -> InstitutionalPerpetualEnv:
        """Create training environment"""
        return InstitutionalPerpetualEnv(
            df=data,
            initial_balance=self.config['trading']['initial_balance'],
            max_leverage=self.config['trading']['max_leverage'],
            transaction_fee=self.config['trading']['transaction_fee'],
            funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
            risk_free_rate=self.config['trading']['risk_free_rate'],
            max_drawdown=self.config['risk_management']['limits']['max_drawdown'],
            window_size=self.config['model']['window_size']
        )
        
    def train_model(self, env: InstitutionalPerpetualEnv) -> PPO:
        """Train the model"""
        logger.info("Initializing model training...")
        
        # Wrap environment in VecEnv
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        policy_kwargs = {
            "net_arch": {
                "pi": [128, 128],  # Actor network
                "vf": [128, 128]   # Critic network
            },
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128}
        }
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            target_kl=0.015,
            tensorboard_log=self.config['logging']['log_dir'],
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        # Training loop with curriculum
        for curriculum_stage in self.config['training']['curriculum_stages']:
            logger.info(f"Starting curriculum stage: {curriculum_stage['name']}")
            
            # Update environment parameters for this stage
            vec_env.envs[0].update_parameters(**curriculum_stage['env_params'])
            
            # Train for this stage
            model.learn(
                total_timesteps=curriculum_stage['timesteps'],
                callback_configs=self.config['training']['callbacks']
            )
            
            # Evaluate stage performance
            self.evaluate_model(model, env, curriculum_stage['name'])
            
            # Save stage checkpoint
            self.save_model(
                model,
                f"stage_{curriculum_stage['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
            )
            
        return model
        
    def evaluate_model(self, model: PPO, env: InstitutionalPerpetualEnv, stage: str):
        """Evaluate model performance"""
        logger.info(f"Evaluating model performance for stage: {stage}")
        
        eval_episodes = self.config['training']['eval_episodes']
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for episode in range(eval_episodes):
            obs = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                action = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_return += reward
                
                # Update dashboard
                self.dashboard.update_metrics(info)
                
            returns.append(episode_return)
            sharpe_ratios.append(info['risk_metrics']['sharpe'])
            max_drawdowns.append(info['risk_metrics']['max_drawdown'])
            
        # Log metrics
        metrics = {
            f'{stage}/mean_return': np.mean(returns),
            f'{stage}/sharpe_ratio': np.mean(sharpe_ratios),
            f'{stage}/max_drawdown': np.mean(max_drawdowns)
        }
        wandb.log(metrics)
        
    def save_model(self, model: PPO, name: str):
        """Save trained model"""
        save_path = os.path.join(self.config['model']['checkpoint_dir'], name)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
    def load_model(self, name: str) -> PPO:
        """Load trained model"""
        load_path = os.path.join(self.config['model']['checkpoint_dir'], name)
        model = PPO.load(load_path)
        logger.info(f"Model loaded from {load_path}")
        return model
        
    async def run_production(self, model: PPO):
        """Run the model in production"""
        logger.info("Starting production trading...")
        
        while True:
            try:
                # Fetch latest data
                data = await self.fetch_and_process_data()
                
                # Get model prediction
                obs = self._prepare_observation(data)
                action = model.predict(obs, deterministic=True)
                
                # Pre-trade risk check
                if not self._check_risk_limits(action, data):
                    logger.warning("Trade rejected due to risk limits")
                    continue
                    
                # Execute trade
                self._execute_trade(action)
                
                # Update monitoring
                self._update_monitoring()
                
            except Exception as e:
                logger.error(f"Error in production loop: {str(e)}")
                if self._should_emergency_stop():
                    logger.critical("Emergency stop triggered")
                    break
                    
    def _check_risk_limits(self, action: np.ndarray, data: pd.DataFrame) -> bool:
        """Check if action violates risk limits"""
        risk_metrics = self.risk_engine.calculate_portfolio_risk(
            positions=self._get_current_positions(),
            market_data=data,
            portfolio_value=self._get_portfolio_value()
        )
        return self.risk_engine.check_risk_limits(risk_metrics)[0]
        
    def _should_emergency_stop(self) -> bool:
        """Check if emergency stop is needed"""
        return any([
            self._get_drawdown() > self.config['emergency']['max_drawdown'],
            self._get_daily_loss() > self.config['emergency']['max_daily_loss'],
            self._get_var() > self.config['emergency']['max_var']
        ])

    async def initialize_training_manager(self):
        """Initialize training manager after data is fetched and processed"""
        # Fetch and process data if needed
        data = await self.fetch_and_process_data()
        
        # Now initialize the training manager
        self.training_manager = TrainingManager(
            data_manager=self.data_manager,
            initial_balance=self.config['trading']['initial_balance'],
            max_leverage=self.config['trading']['max_leverage'],
            n_envs=self.config['training'].get('n_envs', 8)
        )
        return data

async def make_env(args):
    """Create the trading environment"""
    # Initialize data fetcher and get data
    data_fetcher = PerpetualDataFetcher(
        exchanges=config['data']['exchanges'],
        symbols=args.assets,
        timeframe=args.timeframe
    )
    
    # Fetch data using await instead of asyncio.run()
    data = await data_fetcher.fetch_derivative_data()
    if not data:
        raise ValueError("No data fetched from exchanges")
    
    # Combine data from all exchanges and create MultiIndex columns
    combined_data = pd.DataFrame()
    for exchange_data in data.values():
        for symbol, symbol_data in exchange_data.items():
            symbol_cols = pd.MultiIndex.from_product(
                [[symbol], symbol_data.columns],
                names=['asset', 'feature']
            )
            symbol_data.columns = symbol_cols
            
            if combined_data.empty:
                combined_data = symbol_data
            else:
                combined_data = pd.concat([combined_data, symbol_data], axis=1)
    
    combined_data = combined_data.sort_index(axis=1)
    
    # Create environment
    env = InstitutionalPerpetualEnv(
        df=combined_data,
        initial_balance=config['trading']['initial_balance'],
        max_leverage=args.max_leverage,
        transaction_fee=config['trading']['transaction_fee'],
        funding_fee_multiplier=config['trading']['funding_fee_multiplier'],
        risk_free_rate=config['trading']['risk_free_rate'],
        max_drawdown=config['risk_management']['limits']['max_drawdown'],
        window_size=config['model']['window_size']
    )
    
    return env

def setup_model(env, args):
    """Setup the PPO model"""
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128],  # Actor network
            "vf": [128, 128]   # Critic network
        },
        "activation_fn": torch.nn.ReLU,
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128}
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_sde=config['training']['use_sde'],
        sde_sample_freq=config['training']['sde_sample_freq'],
        target_kl=config['training']['target_kl'],
        tensorboard_log=args.log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cuda' if args.gpus > 0 else 'cpu'
    )
    return model

def setup_wandb_config(config):
    return {
        'project': config['logging']['wandb']['project'],
        'entity': config['logging']['wandb']['entity'],
        'tags': config['logging']['wandb']['tags'],
        'mode': config['logging']['wandb']['mode']
    }

async def main():
    try:
        args = parse_args()
        
        # Load configuration
        with open('config/prod_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize trading system
        trading_system = TradingSystem(config)
        
        # Initialize data
        await trading_system.initialize_data()
        
        # Initialize environment
        env = await make_env(args)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Setup wandb configuration
        wandb_config = setup_wandb_config(config)
        
        # Initialize training
        training_manager = TrainingManager(
            data_manager=trading_system.data_manager,
            initial_balance=config['trading']['initial_balance'],
            max_leverage=config['trading']['max_leverage'],
            n_envs=config['training'].get('n_envs', 8),
            wandb_config=wandb_config
        )

        # Create and initialize the model
        model = setup_model(env, args)
        
        # Train the model
        training_manager.train(model)
        
        # Save model and environment statistics
        model.save(os.path.join(args.model_dir, "final_model"))
        env.save(os.path.join(args.model_dir, "vec_normalize.pkl"))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())