#!/usr/bin/env python
# Script to continue training the model with bias correction

import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import pickle
import logging
import random
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unwrap_env(env):
    """
    Safely unwrap environment to get base environment without recursion.
    """
    # Maximum depth to prevent infinite recursion
    max_depth = 10
    current_depth = 0
    
    # Keep track of visited environments to prevent circular references
    visited = set()
    visited.add(id(env))
    
    while current_depth < max_depth:
        current_depth += 1
        
        # Try to get venv (VecNormalize or VecFrameStack)
        if hasattr(env, 'venv') and id(env.venv) not in visited:
            env = env.venv
            visited.add(id(env))
            continue
            
        # Try to get first env from envs list (VecEnv)
        if hasattr(env, 'envs') and len(env.envs) > 0 and id(env.envs[0]) not in visited:
            env = env.envs[0]
            visited.add(id(env))
            continue
            
        # Try to get env (general wrapper)
        if hasattr(env, 'env') and id(env.env) not in visited:
            env = env.env
            visited.add(id(env))
            continue
            
        # No more wrappers found
        break
        
    logger.info(f"Unwrapped environment to: {env.__class__.__name__}")
    return env

def load_environment(env_path: str) -> gym.Env:
    """
    Load environment from pickle file
    """
    logger.info(f"Loading environment from {env_path}")
    try:
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
            logger.info(f"Loaded environment: {type(env).__name__}")
            return env
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        raise

def apply_data_augmentation(env):
    """
    Apply data augmentation to trading environment to reduce bias.
    
    This function:
    1. Reshuffles portions of the dataset
    2. Applies symmetric transformations to assets
    3. Adds slight noise to features
    """
    logger.info("Applying data augmentation to reduce asset bias")
    
    # Get the base environment
    base_env = unwrap_env(env)
    
    # Check if the environment has a dataframe attribute
    if not hasattr(base_env, 'df'):
        logger.warning("Environment doesn't have a dataframe, skipping data augmentation")
        return env
    
    # Get the original dataframe
    orig_df = base_env.df
    
    # Check if we have multi-index columns (asset, feature)
    if not isinstance(orig_df.columns, pd.MultiIndex):
        logger.warning("DataFrame doesn't have MultiIndex columns, skipping data augmentation")
        return env
    
    # Get the list of assets
    assets = orig_df.columns.get_level_values(0).unique()
    logger.info(f"Found assets: {list(assets)}")
    
    # Create a copy of the dataframe to modify
    df = orig_df.copy()
    
    # 1. Apply mild feature noise (0.5% standard deviation)
    logger.info("Adding feature noise to reduce overfitting")
    noise_scale = 0.005  # 0.5% noise
    for asset in assets:
        # Add noise to numeric features only
        for feature in df[asset].select_dtypes(include=[np.number]).columns:
            original_values = df[(asset, feature)].values
            # Calculate asset-specific noise scale based on feature volatility
            feature_std = np.std(original_values)
            if feature_std > 0:
                asset_noise = noise_scale * feature_std
                noise = np.random.normal(0, asset_noise, size=len(df))
                # Only apply noise to non-NaN values
                mask = ~np.isnan(original_values)
                df.loc[mask, (asset, feature)] += noise[mask]
    
    # 2. Create temporary price fluctuations to reduce directional bias
    logger.info("Creating temporary price fluctuations to reduce directional bias")
    # Split the dataframe into segments and apply different scaled adjustments
    # to create more varied price patterns
    n_segments = 10
    segment_size = len(df) // n_segments
    
    for asset in assets:
        if ('close' in df[asset].columns):
            close_col = df[(asset, 'close')]
            # Get segmenting indices
            segment_indices = [(i * segment_size, min((i + 1) * segment_size, len(df))) 
                              for i in range(n_segments)]
            
            # Apply random scaling to segments (between 0.9 and 1.1)
            for start, end in segment_indices:
                # Random scaling factor between 0.97 and 1.03 (3% variation)
                scale_factor = np.random.uniform(0.97, 1.03)
                # Apply scaling to price data
                for price_col in ['open', 'high', 'low', 'close']:
                    if price_col in df[asset].columns:
                        df.loc[start:end, (asset, price_col)] *= scale_factor
    
    # 3. Apply time segment shuffling (breaks perfect time continuity to reduce memorization)
    logger.info("Applying time segment shuffling")
    # Only shuffle segments that aren't at the start or end of the dataset
    shuffle_segments = list(range(1, n_segments-1))
    random.shuffle(shuffle_segments)
    
    # Create a new dataframe with shuffled segments
    shuffled_df = df.copy()
    for i, shuffled_idx in enumerate(shuffle_segments):
        original_idx = i + 1  # Skip the first segment
        original_start = original_idx * segment_size
        original_end = min((original_idx + 1) * segment_size, len(df))
        
        shuffled_start = shuffled_idx * segment_size
        shuffled_end = min((shuffled_idx + 1) * segment_size, len(df))
        
        # Copy data from shuffled segment to original segment
        segment_length = min(original_end - original_start, shuffled_end - shuffled_start)
        shuffled_df.iloc[original_start:original_start+segment_length] = df.iloc[shuffled_start:shuffled_start+segment_length].values
    
    # Update the environment's dataframe
    base_env.df = shuffled_df
    logger.info("Data augmentation completed successfully")
    
    return env

def recalibrate_value_function(model, env, num_steps=10000):
    """
    Recalibrate the value function of the model to the augmented environment
    """
    logger.info(f"Recalibrating value function over {num_steps} steps")
    
    # Create a separate PPO model with the same policy network but new value network
    recalibration_model = PPO(
        policy=model.policy.__class__,
        env=env,
        learning_rate=1e-5,  # Small learning rate just for value function
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        clip_range=0.1,
        normalize_advantage=True,
        ent_coef=0.0,  # No entropy to keep policy stable
        vf_coef=1.0,   # Emphasize value function learning
        max_grad_norm=0.5,
        use_sde=model.use_sde,
        sde_sample_freq=model.sde_sample_freq,
        policy_kwargs=model.policy_kwargs,
        verbose=1
    )
    
    # Copy policy parameters from the original model
    recalibration_model.policy.load_state_dict(model.policy.state_dict())
    
    # Freeze actor parameters and only train the critic
    for param in recalibration_model.policy.actor.parameters():
        param.requires_grad = False
    
    # Train for specified steps (just enough to recalibrate the value function)
    recalibration_model.learn(total_timesteps=num_steps, 
                              log_interval=100, 
                              reset_num_timesteps=False,
                              progress_bar=True)
    
    # Copy the recalibrated value function back to the original model
    model.policy.value_net.load_state_dict(recalibration_model.policy.value_net.state_dict())
    
    logger.info("Value function recalibration complete")
    return model
    
def continue_training(model_path: str, env_path: str, output_dir: str, 
                     total_timesteps: int, learning_rate: float, batch_size: int, 
                     ent_coef: float, clip_range: float, n_epochs: int,
                     apply_augmentation: bool = True, recalibrate_value: bool = True):
    """
    Continue training from a saved model with improved parameters
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment
    env = load_environment(env_path)
    
    # Apply data augmentation to reduce bias if requested
    if apply_augmentation:
        env = apply_data_augmentation(env)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Update training parameters
    model.learning_rate = learning_rate
    model.batch_size = batch_size
    model.ent_coef = ent_coef
    model.clip_range = clip_range
    model.n_epochs = n_epochs
    
    # Recalibrate value function if requested
    if recalibrate_value:
        model = recalibrate_value_function(model, env)
    
    # Setup callbacks
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="ppo_trading",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Continue training
    logger.info(f"Continuing training for {total_timesteps} steps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.zip")
    final_env_path = os.path.join(output_dir, "final_env.pkl")
    
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save the environment
    with open(final_env_path, 'wb') as f:
        pickle.dump(env, f)
    logger.info(f"Saved environment to {final_env_path}")
    
    return final_model_path, final_env_path

def parse_args():
    parser = argparse.ArgumentParser(description="Continue training a RL trading model with bias reduction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--env-path", type=str, required=True, help="Path to the saved environment")
    parser.add_argument("--output-dir", type=str, default="./improved_model", help="Directory to save the improved model")
    parser.add_argument("--timesteps", type=int, default=5000000, help="Number of additional training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs for each update")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-recalibration", action="store_true", help="Disable value function recalibration")
    return parser.parse_args()

def main():
    args = parse_args()
    
    final_model_path, final_env_path = continue_training(
        model_path=args.model_path,
        env_path=args.env_path,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        apply_augmentation=not args.no_augmentation,
        recalibrate_value=not args.no_recalibration
    )
    
    logger.info("Training completed successfully")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Final environment saved to: {final_env_path}")

if __name__ == "__main__":
    main() 