import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, List, Optional, Tuple, Any
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_market_data(assets=["BTC", "ETH", "SOL"]):
    """
    Load market data from parquet files
    """
    logger.info("Loading market data...")
    dfs = []
    
    for asset in assets:
        try:
            file_path = f"data/{asset}_USDT_100days_5m.parquet"
            if not os.path.exists(file_path):
                logger.error(f"Market data file not found: {file_path}")
                continue
                
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Ensure we have required columns
            required_cols = ['close', 'volume', 'funding_rate']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {asset}: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    df[col] = 0.0
            
            # Add asset identifier
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error loading data for {asset}: {e}")
            continue
    
    if not dfs:
        logger.error("No market data could be loaded")
        return None
        
    # Combine all dataframes
    combined_df = pd.concat(dfs, axis=1)
    logger.info(f"Loaded market data with shape: {combined_df.shape}")
    
    return combined_df

def recreate_environment(assets=["BTC", "ETH", "SOL"], window_size=100):
    """
    Recreate a fresh environment with the same parameters
    """
    logger.info("Recreating environment with default parameters")
    try:
        # Load market data
        df = load_market_data(assets)
        if df is None:
            logger.error("Failed to load market data")
            return None
        
        # Create base environment
        env = InstitutionalPerpetualEnv(
            df=df,
            assets=assets,
            window_size=window_size,
            max_leverage=20.0,
            commission=0.0004,
            initial_balance=10000.0,
            max_steps=10000,
            verbose=False
        )
        
        # Wrap in DummyVecEnv
        env = DummyVecEnv([lambda: env])
        
        # Add normalization wrapper
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-08
        )
        
        return env
        
    except Exception as e:
        logger.error(f"Failed to recreate environment: {e}")
        return None

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

def fix_observation_space(env):
    """
    Fix observation space issues in the environment while preserving normalization stats
    """
    try:
        # Check if we have a VecNormalize wrapper with valid stats
        if isinstance(env, VecNormalize):
            logger.info("Found VecNormalize wrapper with normalization stats")
            
            # Store normalization parameters
            obs_rms = env.obs_rms
            ret_rms = env.ret_rms
            norm_obs = env.norm_obs
            norm_reward = env.norm_reward
            clip_obs = env.clip_obs
            clip_reward = env.clip_reward
            gamma = env.gamma
            epsilon = env.epsilon
            
            # Get the base environment
            base_env = None
            if hasattr(env, 'venv') and isinstance(env.venv, DummyVecEnv):
                if len(env.venv.envs) > 0:
                    base_env = env.venv.envs[0]
            
            if base_env is None or not hasattr(base_env, 'observation_space'):
                logger.warning("Base environment is invalid, attempting to fix observation space only")
                # Create a minimal environment with correct spaces
                base_env = gym.Env()
                base_env.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_rms.mean.shape[0],),
                    dtype=np.float32
                )
                base_env.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(3,),  # 3 assets
                    dtype=np.float32
                )
            
            # Wrap in DummyVecEnv
            vec_env = DummyVecEnv([lambda: base_env])
            
            # Create new VecNormalize with preserved stats
            new_env = VecNormalize(
                vec_env,
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                gamma=gamma,
                epsilon=epsilon
            )
            
            # Restore normalization statistics
            new_env.obs_rms = obs_rms
            new_env.ret_rms = ret_rms
            
            logger.info("Successfully preserved normalization statistics")
            return new_env, True
            
        logger.warning("Environment is not VecNormalize wrapped")
        return env, False
    
    except Exception as e:
        logger.error(f"Error fixing observation space: {e}")
        return env, False

def fix_env_file(env_path: str, output_path: str = None) -> bool:
    """
    Load an environment pickle file, fix observation space issues, and save it back
    
    Args:
        env_path: Path to the environment pickle file
        output_path: Path to save the fixed environment (default: overwrite original)
        
    Returns:
        bool: True if fix was successful, False otherwise
    """
    if output_path is None:
        output_path = env_path
        
    try:
        # Load the environment
        logger.info(f"Loading environment from {env_path}")
        try:
            with open(env_path, 'rb') as f:
                env = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load environment, recreating: {e}")
            env = recreate_environment()
            
        # Fix observation space issues
        fixed_env, success = fix_observation_space(env)
        
        if not success:
            logger.warning("Could not fix observation space issues")
            return False
            
        # Save the fixed environment
        logger.info(f"Saving fixed environment to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(fixed_env, f)
            
        logger.info("Environment fixed and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing environment file: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Fix observation space errors in saved environments")
    parser.add_argument("--env-path", type=str, required=True, help="Path to the saved environment")
    parser.add_argument("--output-path", type=str, help="Path to save the fixed environment (default: overwrite original)")
    parser.add_argument("--create-backup", action="store_true", help="Create a backup of the original environment")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create backup if requested
    if args.create_backup:
        import shutil
        backup_path = args.env_path + ".backup"
        logger.info(f"Creating backup at {backup_path}")
        shutil.copy2(args.env_path, backup_path)
    
    # Fix the environment file
    output_path = args.output_path if args.output_path else args.env_path
    success = fix_env_file(args.env_path, output_path)
    
    if success:
        logger.info("✅ Environment fixed successfully!")
    else:
        logger.error("❌ Failed to fix environment")
        
if __name__ == "__main__":
    main() 