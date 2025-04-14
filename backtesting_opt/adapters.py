"""
Adapter module for handling observation space compatibility issues.

This module provides classes and functions to adapt observation spaces between
different formats, particularly when there's a mismatch between a trained model's
expected observations and the test/backtesting environment's observations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
import pickle

# Configure logging
logger = logging.getLogger(__name__)

class ObservationAdapter:
    """
    Adapter for handling observation space conversions between
    training and testing environments.
    
    This class is particularly useful when a model was trained on one
    observation space shape/format but needs to be used with a slightly
    different observation space during backtesting.
    """
    
    def __init__(
        self, 
        model_observation_shape: Tuple[int, ...],
        env_observation_shape: Tuple[int, ...],
        padding_value: float = 0.0,
        padding_mode: str = "zeros",
        truncation_mode: str = "end"
    ):
        """
        Initialize the ObservationAdapter.
        
        Args:
            model_observation_shape: Shape expected by the model
            env_observation_shape: Shape provided by the environment
            padding_value: Value to use for padding when needed
            padding_mode: Mode for padding ('zeros', 'mean', 'repeat')
            truncation_mode: Mode for truncation ('start', 'end', 'middle')
        """
        self.model_shape = model_observation_shape
        self.env_shape = env_observation_shape
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.truncation_mode = truncation_mode
        
        # Flatten the shapes for easier comparison
        self.model_size = np.prod(model_observation_shape)
        self.env_size = np.prod(env_observation_shape)
        
        logger.info(f"ObservationAdapter created for model shape {model_observation_shape} "
                   f"and environment shape {env_observation_shape}")
                   
        # Pre-determine operation type
        if self.model_size > self.env_size:
            self.operation = "pad"
            logger.info(f"Will pad observations from {self.env_size} to {self.model_size}")
        elif self.model_size < self.env_size:
            self.operation = "truncate"
            logger.info(f"Will truncate observations from {self.env_size} to {self.model_size}")
        else:
            # Check if they have the same shape or just same size
            if model_observation_shape == env_observation_shape:
                self.operation = "none"
                logger.info("Observation shapes match exactly, no adaptation needed")
            else:
                self.operation = "reshape"
                logger.info(f"Will reshape observations from {env_observation_shape} to {model_observation_shape}")
    
    def adapt(self, observation: np.ndarray) -> np.ndarray:
        """
        Adapt the observation to match the model's expected format.
        
        Args:
            observation: Observation from the environment
            
        Returns:
            Adapted observation compatible with the model
        """
        try:
            # Ensure observation is a numpy array
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
                
            # Flatten the observation
            flat_obs = observation.flatten()
            
            # Apply appropriate operation
            if self.operation == "none":
                return observation
            elif self.operation == "reshape":
                return observation.reshape(self.model_shape)
            elif self.operation == "pad":
                return self._pad_observation(flat_obs)
            elif self.operation == "truncate":
                return self._truncate_observation(flat_obs)
            else:
                logger.warning(f"Unknown operation: {self.operation}, returning original observation")
                return observation
                
        except Exception as e:
            logger.error(f"Error adapting observation: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.model_shape, dtype=np.float32)
            
    def _pad_observation(self, flat_obs: np.ndarray) -> np.ndarray:
        """Pad the observation to match the model's expected size"""
        padding_size = self.model_size - len(flat_obs)
        
        if self.padding_mode == "zeros":
            padded = np.pad(flat_obs, (0, padding_size), 'constant', constant_values=self.padding_value)
        elif self.padding_mode == "mean":
            padded = np.pad(flat_obs, (0, padding_size), 'constant', constant_values=flat_obs.mean())
        elif self.padding_mode == "repeat":
            # Repeat the last elements
            if len(flat_obs) > 0:
                padding = np.repeat(flat_obs[-1], padding_size)
                padded = np.concatenate([flat_obs, padding])
            else:
                padded = np.zeros(self.model_size, dtype=np.float32)
        else:
            logger.warning(f"Unknown padding mode: {self.padding_mode}, using zeros instead")
            padded = np.pad(flat_obs, (0, padding_size), 'constant', constant_values=0)
            
        return padded.reshape(self.model_shape)
        
    def _truncate_observation(self, flat_obs: np.ndarray) -> np.ndarray:
        """Truncate the observation to match the model's expected size"""
        if self.truncation_mode == "start":
            truncated = flat_obs[-self.model_size:]
        elif self.truncation_mode == "end":
            truncated = flat_obs[:self.model_size]
        elif self.truncation_mode == "middle":
            # Keep from both ends
            start_size = self.model_size // 2
            end_size = self.model_size - start_size
            truncated = np.concatenate([flat_obs[:start_size], flat_obs[-end_size:]])
        else:
            logger.warning(f"Unknown truncation mode: {self.truncation_mode}, truncating from end")
            truncated = flat_obs[:self.model_size]
            
        return truncated.reshape(self.model_shape)

class EnvObservationAdapter(gym.Wrapper):
    """
    Gym environment wrapper that adapts observations to match a model's expected format.
    
    This wrapper can be used to make an environment with one observation space compatible
    with a model trained on a different observation space.
    """
    
    def __init__(self, env: gym.Env, model_observation_shape: Tuple[int, ...] = None):
        """
        Initialize the EnvObservationAdapter.
        
        Args:
            env: The environment to wrap
            model_observation_shape: Expected observation shape for the model
        """
        super().__init__(env)
        
        # Store original observation space shape
        self.original_shape = env.observation_space.shape
        
        # Use provided model shape or infer from environment
        if model_observation_shape is None:
            self.model_shape = self.original_shape
            logger.info(f"No model shape provided, using environment shape: {self.original_shape}")
        else:
            self.model_shape = model_observation_shape
            logger.info(f"Using provided model shape: {self.model_shape}")
            
        # Create adapter
        self.adapter = ObservationAdapter(
            model_observation_shape=self.model_shape,
            env_observation_shape=self.original_shape
        )
        
        # Override observation space
        if isinstance(env.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=self.model_shape, dtype=np.float32
            )
            logger.info(f"Created adapted observation space with shape: {self.observation_space.shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
    
    def reset(self, **kwargs):
        """Reset the environment and adapt the observation"""
        if 'return_info' in kwargs and kwargs['return_info']:
            obs, info = self.env.reset(**kwargs)
            return self.adapter.adapt(obs), info
        else:
            obs = self.env.reset(**kwargs)
            return self.adapter.adapt(obs)
            
    def step(self, action):
        """Take a step in the environment and adapt the observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.adapter.adapt(obs), reward, terminated, truncated, info

class ModelWrapper:
    """
    Wrapper for ML models to handle observation space adaptation.
    
    This wrapper can be used to make a model compatible with an environment
    that has a different observation space than what the model was trained on.
    """
    
    def __init__(self, model: Any, env_observation_shape: Tuple[int, ...]):
        """
        Initialize the ModelWrapper.
        
        Args:
            model: The ML model to wrap
            env_observation_shape: Shape of observations from the environment
        """
        self.model = model
        
        # Determine model's expected observation shape
        if hasattr(model, 'policy') and hasattr(model.policy, 'observation_space'):
            self.model_observation_shape = model.policy.observation_space.shape
            logger.info(f"Model expects observation shape: {self.model_observation_shape}")
        else:
            self.model_observation_shape = env_observation_shape
            logger.warning(f"Could not determine model observation shape, using environment shape: {env_observation_shape}")
            
        # Create adapter
        self.adapter = ObservationAdapter(
            model_observation_shape=self.model_observation_shape,
            env_observation_shape=env_observation_shape
        )
        
    def predict(self, observation: np.ndarray, **kwargs):
        """
        Make a prediction with the model, adapting the observation first.
        
        Args:
            observation: Observation from the environment
            **kwargs: Additional arguments to pass to the model's predict method
            
        Returns:
            Model's prediction
        """
        adapted_obs = self.adapter.adapt(observation)
        return self.model.predict(adapted_obs, **kwargs)
        
def create_compatible_vec_normalize(
    vec_env: gym.Env,
    original_vec_normalize_path: str,
    output_path: Optional[str] = None,
    use_adapted_stats: bool = False
) -> Any:
    """
    Create a VecNormalize wrapper that's compatible with a different environment.
    
    This function loads normalization statistics from a saved VecNormalize wrapper
    and creates a new one that's compatible with the provided environment, even
    if the observation spaces are different.
    
    Args:
        vec_env: The vectorized environment to wrap
        original_vec_normalize_path: Path to the saved VecNormalize pickle
        output_path: Path to save the created VecNormalize wrapper (optional)
        use_adapted_stats: Whether to adapt the normalization statistics to match the new env
        
    Returns:
        A new VecNormalize wrapper for the provided environment
    """
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        log = logging.getLogger(__name__)
        
        # Create a new VecNormalize wrapper for the provided environment
        new_vec_normalize = VecNormalize(vec_env, training=False, norm_obs=True, norm_reward=False)
        
        # Try to load the original VecNormalize wrapper
        with open(original_vec_normalize_path, "rb") as f:
            original_vec_normalize = pickle.load(f)
            log.info("Loaded original VecNormalize wrapper")
            
        # Check observation space dimensions
        if hasattr(original_vec_normalize, 'observation_space'):
            log.info(f"Original VecNormalize observation space: {original_vec_normalize.observation_space.shape}")
            
        # Check if observation statistics are available
        if hasattr(original_vec_normalize, 'obs_rms'):
            original_stats_shape = original_vec_normalize.obs_rms.mean.shape
            new_stats_shape = new_vec_normalize.obs_rms.mean.shape
            
            log.info(f"Original stats shape: {original_stats_shape}, New stats shape: {new_stats_shape}")
            
            if original_stats_shape == new_stats_shape:
                # Copy normalization statistics directly
                log.info("Shapes match, copying normalization statistics directly")
                new_vec_normalize.obs_rms.mean = original_vec_normalize.obs_rms.mean.copy()
                new_vec_normalize.obs_rms.var = original_vec_normalize.obs_rms.var.copy()
                new_vec_normalize.obs_rms.count = original_vec_normalize.obs_rms.count
            elif use_adapted_stats and len(original_stats_shape) == 1 and len(new_stats_shape) == 1:
                # Try to adapt the statistics
                log.info("Adapting normalization statistics to match new environment")
                
                adapter = ObservationAdapter(
                    model_observation_shape=new_stats_shape,
                    env_observation_shape=original_stats_shape
                )
                
                # Adapt mean and variance
                adapted_mean = adapter.adapt(original_vec_normalize.obs_rms.mean)
                adapted_var = adapter.adapt(original_vec_normalize.obs_rms.var)
                
                # Copy adapted statistics
                new_vec_normalize.obs_rms.mean = adapted_mean
                new_vec_normalize.obs_rms.var = adapted_var
                new_vec_normalize.obs_rms.count = original_vec_normalize.obs_rms.count
            else:
                log.warning("Shapes don't match and adaptation not enabled, keeping default statistics")
                
        # Get reward statistics if available
        if hasattr(original_vec_normalize, 'ret_rms'):
            new_vec_normalize.ret_rms.mean = original_vec_normalize.ret_rms.mean
            new_vec_normalize.ret_rms.var = original_vec_normalize.ret_rms.var
            new_vec_normalize.ret_rms.count = original_vec_normalize.ret_rms.count
            
        # Copy other attributes
        if hasattr(original_vec_normalize, 'gamma'):
            new_vec_normalize.gamma = original_vec_normalize.gamma
            
        if hasattr(original_vec_normalize, 'epsilon'):
            new_vec_normalize.epsilon = original_vec_normalize.epsilon
            
        log.info(f"New VecNormalize observation space: {new_vec_normalize.observation_space.shape}")
        
        # Save the new VecNormalize wrapper if requested
        if output_path:
            with open(output_path, "wb") as f:
                pickle.dump(new_vec_normalize, f)
                log.info(f"Saved compatible VecNormalize wrapper to {output_path}")
                
        return new_vec_normalize
        
    except Exception as e:
        logger.error(f"Error creating compatible VecNormalize: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 