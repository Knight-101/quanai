import pickle
import logging
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import json
import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_env(path):
    logger.info(f"Attempting to load environment from {path}")
    try:
        with open(path, 'rb') as f:
            env = pickle.load(f)
            logger.info(f"Successfully loaded environment of type: {type(env)}")
            
            # Check if it's a vectorized environment
            if isinstance(env, VecNormalize):
                logger.info("Environment is VecNormalize wrapped")
                logger.info(f"Observation normalization: {env.norm_obs}")
                logger.info(f"Reward normalization: {env.norm_reward}")
                
                # Detailed normalization statistics
                if hasattr(env, 'obs_rms') and env.obs_rms is not None:
                    logger.info(f"Observation stats:")
                    logger.info(f"  Shape: {env.obs_rms.mean.shape}")
                    logger.info(f"  Mean range: [{env.obs_rms.mean.min():.4f}, {env.obs_rms.mean.max():.4f}]")
                    logger.info(f"  Variance range: [{env.obs_rms.var.min():.4f}, {env.obs_rms.var.max():.4f}]")
                    logger.info(f"  Count: {env.obs_rms.count}")
                
                if hasattr(env, 'ret_rms') and env.ret_rms is not None:
                    logger.info(f"Return stats:")
                    logger.info(f"  Mean: {env.ret_rms.mean:.4f}")
                    logger.info(f"  Variance: {env.ret_rms.var:.4f}")
                    logger.info(f"  Count: {env.ret_rms.count}")
                    
                # Check training parameters
                logger.info(f"Training parameters:")
                logger.info(f"  Gamma: {env.gamma}")
                logger.info(f"  Epsilon: {env.epsilon}")
                logger.info(f"  Clip obs: {env.clip_obs}")
                logger.info(f"  Clip reward: {env.clip_reward}")
                    
                # Check if base env exists
                if hasattr(env, 'venv'):
                    logger.info(f"Base environment type: {type(env.venv)}")
                    
                    # Check DummyVecEnv
                    if isinstance(env.venv, DummyVecEnv):
                        if len(env.venv.envs) > 0:
                            base_env = env.venv.envs[0]
                            logger.info(f"Actual environment type: {type(base_env)}")
                            
                            # Check for critical attributes
                            if hasattr(base_env, 'df'):
                                logger.info(f"Has market data with shape: {base_env.df.shape}")
                            if hasattr(base_env, 'market_conditions'):
                                logger.info(f"Market conditions: {json.dumps(base_env.market_conditions, indent=2)}")
                            if hasattr(base_env, 'risk_engine'):
                                logger.info("Has risk engine configuration")
                                if hasattr(base_env.risk_engine, 'risk_limits'):
                                    logger.info(f"Risk limits: {base_env.risk_engine.risk_limits.__dict__}")
            
            # Check for observation space
            if hasattr(env, 'observation_space'):
                logger.info(f"Observation space: {env.observation_space}")
                if isinstance(env.observation_space, gym.spaces.Box):
                    logger.info(f"  Shape: {env.observation_space.shape}")
                    logger.info(f"  Low: [{env.observation_space.low.min()}, {env.observation_space.low.max()}]")
                    logger.info(f"  High: [{env.observation_space.high.min()}, {env.observation_space.high.max()}]")
            else:
                logger.warning("No observation_space found")
                
            # Check for action space
            if hasattr(env, 'action_space'):
                logger.info(f"Action space: {env.action_space}")
                if isinstance(env.action_space, gym.spaces.Box):
                    logger.info(f"  Shape: {env.action_space.shape}")
                    logger.info(f"  Low: {env.action_space.low}")
                    logger.info(f"  High: {env.action_space.high}")
            else:
                logger.warning("No action_space found")
                
            return env
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        return None

if __name__ == "__main__":
    env_path = "./models/manual/phase6/final_env.pkl"
    env = inspect_env(env_path) 