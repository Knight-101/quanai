import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ObservationAdapter(gym.Wrapper):
    """
    A wrapper that adapts observations from the environment to match the shape expected by the model.
    This handles the mismatch between training and backtesting environments.
    """
    def __init__(self, env, expected_shape, padding_value=0.0, verbose=False):
        """
        Initialize the ObservationAdapter
        
        Args:
            env: The environment to wrap
            expected_shape: The expected shape of observations (model's observation shape)
            padding_value: Value to use for padding (default: 0.0)
            verbose: Whether to log detailed information
        """
        super().__init__(env)
        self.expected_shape = expected_shape
        self.padding_value = padding_value
        self.verbose = verbose
        
        # Create new observation space with expected shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=expected_shape,
            dtype=np.float32
        )
        
        # Get original shape for logging
        original_shape = env.observation_space.shape
        logger.info(f"Created ObservationAdapter to transform observations from {original_shape} to {expected_shape}")
    
    def reset(self, **kwargs):
        """Reset the environment and adapt the observation"""
        obs, info = self.env.reset(**kwargs)
        adapted_obs = self._adapt_observation(obs)
        
        if self.verbose:
            logger.debug(f"Reset observation adapted from {obs.shape} to {adapted_obs.shape}")
            
        return adapted_obs, info
    
    def step(self, action):
        """Step the environment and adapt the observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        adapted_obs = self._adapt_observation(obs)
        
        if self.verbose and hasattr(self, 'step_count'):
            self.step_count += 1
            if self.step_count % 1000 == 0:
                logger.debug(f"Step {self.step_count} observation adapted from {obs.shape} to {adapted_obs.shape}")
        
        return adapted_obs, reward, terminated, truncated, info
    
    def _adapt_observation(self, obs):
        """
        Adapt observation to match expected shape
        
        This handles several cases:
        1. 1D observations -> 1D expected shape
        2. 1D observations -> 2D expected shape
        3. 2D observations -> 1D expected shape
        4. 2D observations -> 2D expected shape
        """
        # If obs is not a numpy array, return as is (unexpected)
        if not isinstance(obs, np.ndarray):
            logger.warning(f"Unexpected observation type: {type(obs)}")
            return obs
        
        # Create a zero-filled array of the expected shape
        adapted_obs = np.full(self.expected_shape, self.padding_value, dtype=np.float32)
        
        try:
            # Case 1: 1D observations -> 1D expected shape
            if len(obs.shape) == 1 and len(self.expected_shape) == 1:
                min_dim = min(obs.shape[0], self.expected_shape[0])
                adapted_obs[:min_dim] = obs[:min_dim]
                
            # Case 2: 1D observations -> 2D expected shape
            elif len(obs.shape) == 1 and len(self.expected_shape) == 2:
                min_dim = min(obs.shape[0], self.expected_shape[1])
                adapted_obs[0, :min_dim] = obs[:min_dim]
                
            # Case 3: 2D observations -> 1D expected shape
            elif len(obs.shape) == 2 and len(self.expected_shape) == 1:
                # Take only the first row if batch dimension is 1
                if obs.shape[0] == 1:
                    min_dim = min(obs.shape[1], self.expected_shape[0])
                    adapted_obs[:min_dim] = obs[0, :min_dim]
                else:
                    # If batch dimension > 1, flatten and take as much as will fit
                    flattened = obs.flatten()
                    min_dim = min(len(flattened), self.expected_shape[0])
                    adapted_obs[:min_dim] = flattened[:min_dim]
                    
            # Case 4: 2D observations -> 2D expected shape
            elif len(obs.shape) == 2 and len(self.expected_shape) == 2:
                min_dim0 = min(obs.shape[0], self.expected_shape[0])
                min_dim1 = min(obs.shape[1], self.expected_shape[1])
                adapted_obs[:min_dim0, :min_dim1] = obs[:min_dim0, :min_dim1]
                
            else:
                logger.warning(f"Complex shape adaptation: {obs.shape} -> {self.expected_shape}")
                # For other cases, just do our best
                adapted_obs = np.zeros(self.expected_shape, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error adapting observation: {str(e)}")
            # Return zeros on error
            adapted_obs = np.zeros(self.expected_shape, dtype=np.float32)
            
        return adapted_obs 