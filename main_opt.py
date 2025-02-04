import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from data_collection.collect_multimodal import MultiModalDataCollector
from trading_env.institutional_perp_env import InstitutionalPerpEnv
from training.hierarchical_ppo import HierarchicalPPO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_env(data_path: str, **kwargs) -> InstitutionalPerpEnv:
    """Create trading environment with given parameters"""
    return InstitutionalPerpEnv(
        data_path=data_path,
        initial_balance=kwargs.get('initial_balance', 10000),
        leverage=kwargs.get('leverage', 5),
        trading_fee=kwargs.get('trading_fee', 0.0004),
        window_size=kwargs.get('window_size', 100),
        reward_scaling=kwargs.get('reward_scaling', 1.0)
    )

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Environment hyperparameters
    env_params = {
        'initial_balance': trial.suggest_float('initial_balance', 5000, 20000),
        'leverage': trial.suggest_int('leverage', 1, 10),
        'trading_fee': trial.suggest_float('trading_fee', 0.0001, 0.001, log=True),
        'window_size': trial.suggest_int('window_size', 50, 200),
        'reward_scaling': trial.suggest_float('reward_scaling', 0.1, 10.0, log=True)
    }
    
    # Model hyperparameters
    model_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_int('n_steps', 1024, 4096),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'n_epochs': trial.suggest_int('n_epochs', 5, 20),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
    }
    
    # Neural network architecture
    architecture_params = {
        'policy_net_arch': [
            trial.suggest_int('policy_layer1', 64, 512),
            trial.suggest_int('policy_layer2', 32, 256)
        ],
        'value_net_arch': [
            trial.suggest_int('value_layer1', 64, 512),
            trial.suggest_int('value_layer2', 32, 256)
        ]
    }
    
    try:
        # Create and prepare environment
        env = create_env('data/train_data.parquet', **env_params)
        
        # Initialize model with trial parameters
        model = HierarchicalPPO(
            env=env,
            learning_rate=model_params['learning_rate'],
            n_steps=model_params['n_steps'],
            batch_size=model_params['batch_size'],
            n_epochs=model_params['n_epochs'],
            gamma=model_params['gamma'],
            gae_lambda=model_params['gae_lambda'],
            clip_range=model_params['clip_range'],
            ent_coef=model_params['ent_coef'],
            vf_coef=model_params['vf_coef'],
            max_grad_norm=model_params['max_grad_norm'],
            policy_kwargs={'net_arch': architecture_params['policy_net_arch']},
            value_kwargs={'net_arch': architecture_params['value_net_arch']}
        )
        
        # Train for a fixed number of timesteps (reduced to 100k)
        total_timesteps = 100_000
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate on validation set
        val_env = create_env('data/val_data.parquet', **env_params)
        mean_reward = evaluate_model(model, val_env)
        
        return mean_reward
        
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        raise optuna.TrialPruned()

def evaluate_model(model: HierarchicalPPO, env: InstitutionalPerpEnv, n_episodes: int = 10) -> float:
    """Evaluate model performance over multiple episodes"""
    rewards = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def collect_training_data():
    """Collect and prepare training data"""
    collector = MultiModalDataCollector()
    
    # Collect training data (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    collector.collect_and_save_data(
        start_date=start_date,
        end_date=end_date - timedelta(days=30),  # Leave last 30 days for validation
        output_path='data/train_data.parquet'
    )
    
    # Collect validation data (last 30 days)
    collector.collect_and_save_data(
        start_date=end_date - timedelta(days=30),
        end_date=end_date,
        output_path='data/val_data.parquet'
    )

def main():
    """Main optimization function"""
    # Collect data if needed
    collect_training_data()
    
    # Create study
    storage = optuna.storages.RDBStorage(
        url='sqlite:///optimization.db',
        heartbeat_interval=1
    )
    
    study = optuna.create_study(
        study_name='trading_optimization',
        direction='maximize',
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization with parallel jobs
    n_trials = 30  # Reduced number of trials
    n_jobs = 5    # Number of parallel jobs
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Log best parameters
    logger.info("\nBest trial:")
    trial = study.best_trial
    
    logger.info(f"Value: {trial.value}")
    logger.info("\nBest parameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Save optimization results
    study.trials_dataframe().to_csv('optimization_results.csv', index=False)

if __name__ == "__main__":
    main() 