import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from collections import defaultdict
import pickle
from gymnasium import spaces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unwrap_env(env):
    """
    Safely unwrap environment to get base environment without recursion.
    """
    if env is None:
        return None
        
    # If it's already a VecNormalize, return it
    if isinstance(env, VecNormalize):
        return env
        
    # Maximum depth to prevent infinite recursion
    max_depth = 10
    current_depth = 0
    current_env = env
    
    while current_depth < max_depth:
        if current_env is None:
            break
            
        # If we found a VecNormalize, return it
        if isinstance(current_env, VecNormalize):
            return current_env
            
        # Try different unwrapping attributes
        if hasattr(current_env, 'venv'):
            current_env = current_env.venv
        elif hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'envs') and len(current_env.envs) > 0:
            current_env = current_env.envs[0]
        else:
            # No more wrappers found
            break
            
        current_depth += 1
    
    # Return whatever environment we ended up with
    return current_env

def load_environment(env_path: str) -> Optional[gym.Env]:
    """
    Load environment from path with error handling.
    """
    try:
        # Try loading as pickle first
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
            logger.info("Loaded environment from pickle")
            
            # If it's already a VecNormalize, we're good
            if isinstance(env, VecNormalize):
                logger.info("Environment is VecNormalize")
                return env
                
            # Try to unwrap to find VecNormalize
            unwrapped = unwrap_env(env)
            if unwrapped is not None:
                if isinstance(unwrapped, VecNormalize):
                    logger.info("Found VecNormalize wrapper")
                    return unwrapped
                else:
                    logger.info(f"Unwrapped to {unwrapped.__class__.__name__}")
                    return env
            else:
                logger.warning("Could not unwrap environment, using as is")
                return env
                
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        return None

def get_env_assets(env: gym.Env) -> List[str]:
    """Safely extract asset list from environment"""
    try:
        # First unwrap the environment safely to get the base environment
        base_env = unwrap_env(env)
        
        # Check for InstitutionalPerpetualEnv environment (from trading_env)
        if hasattr(base_env, 'assets'):
            logger.info(f"Found assets directly on base environment: {base_env.assets}")
            return base_env.assets
        
        # Attempt to infer assets from environment action space
        if hasattr(base_env, 'action_space'):
            action_shape = base_env.action_space.shape
            if action_shape and len(action_shape) > 0:
                num_assets = action_shape[0]
                # Check if asset names are available in action_names
                if hasattr(base_env, 'action_names'):
                    return base_env.action_names
                # Check if we can get asset names from df
                elif hasattr(base_env, 'df') and hasattr(base_env.df, 'columns') and isinstance(base_env.df.columns, pd.MultiIndex):
                    asset_names = base_env.df.columns.get_level_values(0).unique().tolist()
                    if len(asset_names) == num_assets:
                        return asset_names
                # Default to generic names based on action_space dimension
                logger.info(f"Inferred {num_assets} assets from action space")
                return [f"asset_{i}" for i in range(num_assets)]
        
        # For any other case, try to infer from action_space shape
        if hasattr(env, 'action_space'):
            action_shape = env.action_space.shape
            num_assets = action_shape[0] if action_shape and len(action_shape) > 0 else 1
            logger.info(f"Fallback: inferred {num_assets} assets from action space")
            return [f"asset_{i}" for i in range(num_assets)]
            
        # If we still can't determine, use the most common crypto assets
        logger.warning("Could not determine assets from environment, using default crypto assets")
        return ["BTC", "ETH", "SOL"]
        
    except Exception as e:
        logger.warning(f"Error getting assets from environment: {e}")
        return ["BTC", "ETH", "SOL"]  # Default to common crypto assets

def load_model(model_path: str, env: gym.Env) -> Optional[BaseAlgorithm]:
    """
    Load model from path with error handling.
    """
    try:
        # Try loading as PPO first
        model = PPO.load(model_path, env=env)
        logger.info("Loaded model as PPO")
        return model
    except Exception as e:
        logger.debug(f"Failed to load model as PPO: {e}")
        
    try:
        # Try loading as A2C
        model = A2C.load(model_path, env=env)
        logger.info("Loaded model as A2C")
        return model
    except Exception as e:
        logger.debug(f"Failed to load model as A2C: {e}")
        
    logger.error("Failed to load model using any known method")
    return None

def collect_actions(model: BaseAlgorithm, env: gym.Env, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Collect model predictions using environment's normalization statistics
    """
    logger.info(f"Collecting {num_samples} action samples...")
    
    # Get action space to determine number of assets
    action_space = env.action_space
    action_shape = action_space.shape[0] if hasattr(action_space, 'shape') else 1
    assets = ["BTC", "ETH", "SOL"][:action_shape]
    logger.info(f"Using assets: {assets}")
    
    # Initialize collections
    actions = []
    observations = []
    
    try:
        # Generate observations using environment stats if available
        if isinstance(env, VecNormalize) and hasattr(env, 'obs_rms'):
            obs_mean = env.obs_rms.mean
            obs_var = env.obs_rms.var
            obs_shape = obs_mean.shape[0]
            
            logger.info(f"Using VecNormalize stats for observations (shape: {obs_shape})")
            
            for i in range(num_samples):
                try:
                    # Generate normalized observation
                    obs = np.random.normal(obs_mean, np.sqrt(obs_var + 1e-8))
                    if env.norm_obs:
                        obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + env.epsilon),
                                    -env.clip_obs, env.clip_obs)
                    
                    # Ensure observation is the right shape
                    obs = obs.astype(np.float32)
                    if len(obs.shape) == 1:
                        obs = obs[np.newaxis, :]
                    
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=False)
                    
                    # Store only if action is valid
                    if action is not None and action.shape[-1] == len(assets):
                        actions.append(action[0])
                        observations.append(obs[0])
                        
                    if (i + 1) % 100 == 0:
                        logger.info(f"Collected {i + 1}/{num_samples} samples")
                        
                except Exception as e:
                    logger.warning(f"Error collecting sample {i}: {e}")
                    continue
        else:
            logger.error("Environment does not have required normalization statistics")
            return {
                'actions': np.array([]),
                'observations': np.array([]),
                'assets': assets
            }
        
        # Convert to numpy arrays
        if actions and observations:
            actions_np = np.vstack(actions)
            observations_np = np.vstack(observations)
            
            # Log action statistics for debugging
            logger.info(f"\nAction statistics:")
            for i, asset in enumerate(assets):
                asset_actions = actions_np[:, i]
                logger.info(f"{asset}:")
                logger.info(f"  Mean: {np.mean(asset_actions):.4f}")
                logger.info(f"  Std: {np.std(asset_actions):.4f}")
                logger.info(f"  Range: [{np.min(asset_actions):.4f}, {np.max(asset_actions):.4f}]")
            
            return {
                'actions': actions_np,
                'observations': observations_np,
                'assets': assets
            }
        else:
            logger.error("No valid samples collected")
            return {
                'actions': np.array([]),
                'observations': np.array([]),
                'assets': assets
            }
            
    except Exception as e:
        logger.error(f"Error in collect_actions: {e}")
        logger.debug("Traceback:", exc_info=True)
        return {
            'actions': np.array([]),
            'observations': np.array([]),
            'assets': assets
        }

def generate_synthetic_observations(env: gym.Env) -> np.ndarray:
    """
    Generate synthetic observations for testing when environment reset fails.
    """
    logger.warning("Generating synthetic observations for testing")
    
    # Try to infer observation shape from environment
    if hasattr(env, 'observation_space'):
        obs_shape = env.observation_space.shape
        # Generate random observation within bounds
        low = env.observation_space.low
        high = env.observation_space.high
        # Handle infinite bounds
        low = np.where(np.isinf(low), -1.0, low)
        high = np.where(np.isinf(high), 1.0, high)
        
        return np.random.uniform(low=low, high=high, size=obs_shape)
    
    # Default to a reasonably-sized observation if we can't infer
    return np.random.uniform(-1, 1, size=(1, 100))

def perturb_observation(obs: np.ndarray) -> np.ndarray:
    """
    Add small random perturbation to observation for next sample.
    """
    if isinstance(obs, np.ndarray):
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.01, size=obs.shape)
        return obs + noise
    else:
        return obs

def analyze_actions(actions: np.ndarray, assets: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Analyze actions for each asset and compute statistics.
    """
    stats = {}
    for i, asset in enumerate(assets):
        asset_actions = actions[:, i]
        
        # Basic statistics
        stats[asset] = {
            'mean': float(np.mean(asset_actions)),
            'median': float(np.median(asset_actions)),
            'std': float(np.std(asset_actions)),
            'min': float(np.min(asset_actions)),
            'max': float(np.max(asset_actions)),
            'abs_mean': float(np.mean(np.abs(asset_actions))),
            'pct_positive': float(np.mean(asset_actions > 0)),
            'pct_negative': float(np.mean(asset_actions < 0)),
            'pct_zero': float(np.mean(asset_actions == 0)),
            'pct_extreme_positive': float(np.mean(asset_actions > 0.8)),
            'pct_extreme_negative': float(np.mean(asset_actions < -0.8))
        }
        
        # Calculate bias level
        mean_abs = stats[asset]['abs_mean']
        mean = stats[asset]['mean']
        if mean_abs > 0:
            bias_level = mean / mean_abs
        else:
            bias_level = 0
        stats[asset]['bias_level'] = float(bias_level)
        
    return stats

def apply_bias_correction(actions: np.ndarray, stats: Dict[str, Dict[str, float]], 
                         assets: List[str]) -> np.ndarray:
    """
    Apply bias correction to actions based on computed statistics.
    """
    corrected_actions = actions.copy()
    
    for i, asset in enumerate(assets):
        asset_stats = stats[asset]
        asset_actions = actions[:, i]
        
        # Skip correction if no significant bias
        if abs(asset_stats['bias_level']) < 0.1:
            continue
            
        # Calculate correction factors
        mean = asset_stats['mean']
        std = asset_stats['std']
        
        # Center the actions around zero
        centered_actions = asset_actions - mean
        
        # Scale extreme values
        if asset_stats['pct_extreme_positive'] > 0.1 or asset_stats['pct_extreme_negative'] > 0.1:
            scale_factor = 0.8 / max(abs(asset_stats['max']), abs(asset_stats['min']))
            centered_actions *= scale_factor
            
        # Apply smoothing for high variance
        if std > 0.5:
            smoothing_factor = 0.5 / std
            centered_actions *= smoothing_factor
            
        corrected_actions[:, i] = centered_actions
        
    return corrected_actions

def calculate_bias_reduction(original_stats: Dict[str, Dict[str, float]], 
                           corrected_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the reduction in bias after correction.
    """
    reduction_stats = {}
    
    for asset in original_stats.keys():
        orig_bias = abs(original_stats[asset]['bias_level'])
        corr_bias = abs(corrected_stats[asset]['bias_level'])
        
        if orig_bias > 0:
            reduction = (orig_bias - corr_bias) / orig_bias * 100
        else:
            reduction = 0
            
        reduction_stats[asset] = {
            'original_bias': orig_bias,
            'corrected_bias': corr_bias,
            'reduction_percent': reduction
        }
        
    return reduction_stats

def plot_action_distributions(actions: np.ndarray, corrected_actions: np.ndarray, 
                             assets: List[str], output_dir: str):
    """
    Plot original and corrected action distributions.
    """
    plt.style.use('default')  # Use default style instead of seaborn
    num_assets = len(assets)
    
    # Create subplots for each asset
    fig, axes = plt.subplots(num_assets, 2, figsize=(15, 5*num_assets))
    if num_assets == 1:
        axes = axes.reshape(1, -1)
    
    for i, asset in enumerate(assets):
        # Histogram plot
        axes[i, 0].hist(actions[:, i], bins=50, alpha=0.5, label='Original', color='blue')
        axes[i, 0].hist(corrected_actions[:, i], bins=50, alpha=0.5, label='Corrected', color='green')
        axes[i, 0].set_title(f'{asset} Action Distribution')
        axes[i, 0].set_xlabel('Action Value')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Scatter plot
        axes[i, 1].scatter(actions[:, i], corrected_actions[:, i], alpha=0.5, s=20)
        axes[i, 1].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)  # Identity line
        axes[i, 1].set_title(f'{asset} Original vs Corrected Actions')
        axes[i, 1].set_xlabel('Original Action')
        axes[i, 1].set_ylabel('Corrected Action')
        axes[i, 1].grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distributions.png'))
    plt.close()

def plot_bias_metrics(stats: Dict[str, Dict[str, float]], 
                     reduction_stats: Dict[str, float], 
                     output_dir: str):
    """
    Plot bias metrics and reduction statistics.
    """
    plt.style.use('default')  # Use default style instead of seaborn
    assets = list(stats.keys())
    num_assets = len(assets)
    
    # Create figure for bias metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot bias levels
    bias_levels = [stats[asset]['bias_level'] for asset in assets]
    abs_means = [stats[asset]['abs_mean'] for asset in assets]
    
    x = np.arange(num_assets)
    width = 0.35
    
    ax1.bar(x - width/2, bias_levels, width, label='Bias Level')
    ax1.bar(x + width/2, abs_means, width, label='Absolute Mean')
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets)
    ax1.set_title('Bias Metrics by Asset')
    ax1.legend()
    ax1.grid(True)
    
    # Plot bias reduction
    reductions = [reduction_stats[asset]['reduction_percent'] for asset in assets]
    ax2.bar(assets, reductions, color='green', alpha=0.7)
    ax2.set_title('Bias Reduction Percentage')
    ax2.set_ylabel('Reduction (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_metrics.png'))
    plt.close()

def save_analysis_results(stats: Dict[str, Dict[str, float]], 
                         reduction_stats: Dict[str, float],
                         output_dir: str):
    """
    Save analysis results to CSV files.
    """
    # Save action statistics
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(output_dir, 'action_statistics.csv'))
    
    # Save bias reduction statistics
    reduction_df = pd.DataFrame.from_dict(reduction_stats, orient='index')
    reduction_df.to_csv(os.path.join(output_dir, 'bias_reduction.csv'))

def save_action_data(actions: np.ndarray, corrected_actions: np.ndarray, stats: Dict, assets: List[str], output_dir: str):
    """
    Save action data and statistics to CSV files.
    """
    # Save raw actions
    actions_df = pd.DataFrame(actions, columns=assets)
    actions_df.to_csv(os.path.join(output_dir, 'raw_actions.csv'), index=False)
    
    # Save corrected actions
    corrected_df = pd.DataFrame(corrected_actions, columns=assets)
    corrected_df.to_csv(os.path.join(output_dir, 'corrected_actions.csv'), index=False)
    
    # Save statistics
    stats_data = []
    for asset in assets:
        asset_stats = stats[asset]
        stats_data.append({
            'asset': asset,
            'mean': asset_stats['mean'],
            'median': asset_stats['median'],
            'std': asset_stats['std'],
            'min': asset_stats['min'],
            'max': asset_stats['max'],
            'bias_level': asset_stats['bias_level'],
            'pct_positive': asset_stats['pct_positive'],
            'pct_negative': asset_stats['pct_negative'],
            'pct_extreme_positive': asset_stats['pct_extreme_positive'],
            'pct_extreme_negative': asset_stats['pct_extreme_negative']
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(output_dir, 'action_statistics.csv'), index=False)
    
    logger.info(f"Saved action data and statistics to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model bias for any asset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--env-path", type=str, required=True, help="Path to the environment")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of action samples to collect")
    parser.add_argument("--output-dir", type=str, default="./analysis", help="Directory to save analysis results")
    parser.add_argument("--correction-method", type=str, default="nonlinear", 
                        choices=["mean_centering", "distribution_scaling", "nonlinear"],
                        help="Bias correction method to apply")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load environment and model
        env = load_environment(args.env_path)
        if env is None:
            logger.error("Failed to load environment")
            return
            
        # Validate environment
        if not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
            logger.error("Invalid environment: missing observation_space or action_space")
            return
            
        # Load model
        try:
            model = load_model(args.model_path, env)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
            
        # Collect actions
        results = collect_actions(model, env, num_samples=args.num_samples)
        
        # Validate results
        if len(results['actions']) == 0:
            logger.error("No valid actions collected")
            return
            
        actions = results['actions']
        assets = results['assets']
        
        # Validate shapes
        if len(actions.shape) != 2:
            logger.error(f"Invalid action shape: {actions.shape}")
            return
            
        if actions.shape[1] != len(assets):
            logger.warning(f"Mismatch between number of actions ({actions.shape[1]}) and assets ({len(assets)})")
            # Try to fix by taking min
            num_dims = min(actions.shape[1], len(assets))
            actions = actions[:, :num_dims]
            assets = assets[:num_dims]
        
        # Analyze actions
        try:
            stats = analyze_actions(actions, assets)
        except Exception as e:
            logger.error(f"Error analyzing actions: {e}")
            return
        
        # Print summary statistics
        logger.info("\n=== Action Statistics Summary ===")
        for asset, asset_stats in stats.items():
            bias_level = asset_stats['bias_level']
            bias_category = "HIGH" if bias_level > 0.2 else "MEDIUM" if bias_level > 0.1 else "LOW"
            
            logger.info(f"\nAsset: {asset} - Bias Level: {bias_level:.4f} ({bias_category})")
            logger.info(f"  Mean: {asset_stats['mean']:.4f}, Median: {asset_stats['median']:.4f}, Std: {asset_stats['std']:.4f}")
            logger.info(f"  Range: [{asset_stats['min']:.4f}, {asset_stats['max']:.4f}]")
            logger.info(f"  Positive: {asset_stats['pct_positive']*100:.1f}%, Negative: {asset_stats['pct_negative']*100:.1f}%")
            logger.info(f"  Extreme: +{asset_stats['pct_extreme_positive']*100:.1f}%, -{asset_stats['pct_extreme_negative']*100:.1f}%")
        
        # Apply bias correction
        try:
            corrected_actions = apply_bias_correction(actions, stats, assets)
        except Exception as e:
            logger.error(f"Error applying bias correction: {e}")
            return
        
        # Create output directory
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            return
        
        # Plot distributions
        try:
            plot_action_distributions(actions, corrected_actions, assets, args.output_dir)
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            logger.debug("Traceback:", exc_info=True)
            # Continue anyway - plotting failure shouldn't stop everything
        
        # Save data
        try:
            save_action_data(actions, corrected_actions, stats, assets, args.output_dir)
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return
        
        # Print final summary
        logger.info("\n=== Bias Analysis Summary ===")
        logger.info(f"Total samples analyzed: {len(actions)}")
        logger.info(f"Number of assets: {len(assets)}")
        
        high_bias_assets = [asset for asset, stats in stats.items() if stats['bias_level'] > 0.2]
        if high_bias_assets:
            logger.warning(f"Assets with high bias: {', '.join(high_bias_assets)}")
        
        # Calculate overall bias reduction
        original_mean_bias = np.mean([stats[asset]['bias_level'] for asset in assets])
        corrected_mean_bias = np.mean([abs(np.mean(corrected_actions[:, i])) for i in range(len(assets))])
        
        logger.info(f"\nOverall mean bias: {original_mean_bias:.4f} -> {corrected_mean_bias:.4f}")
        logger.info(f"Bias reduction: {((original_mean_bias - corrected_mean_bias) / original_mean_bias * 100):.1f}%")
        
        # Calculate bias reduction statistics
        reduction_stats = calculate_bias_reduction(stats, stats)
        
        # Plot bias metrics
        try:
            plot_bias_metrics(stats, reduction_stats, args.output_dir)
        except Exception as e:
            logger.error(f"Error creating bias metrics plot: {e}")
            logger.debug("Traceback:", exc_info=True)
        
        # Save analysis results
        try:
            save_analysis_results(stats, reduction_stats, args.output_dir)
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
        
        logger.info(f"\nResults saved to: {args.output_dir}")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.debug("Traceback:", exc_info=True)
        
if __name__ == "__main__":
    main() 