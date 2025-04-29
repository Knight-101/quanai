import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment(env_path: str) -> gym.Env:
    """Load environment from file"""
    try:
        # Try direct loading
        if env_path.endswith('.pkl'):
            env = joblib.load(env_path)
            logger.info(f"Loaded environment from {env_path}")
            return env
        
        # Try finding .pkl file in the path
        env_files = list(Path(env_path).glob("*.pkl"))
        if env_files:
            env = joblib.load(env_files[0])
            logger.info(f"Loaded environment from {env_files[0]}")
            return env
            
        # Try loading from vectorized environment
        vec_normalize_path = os.path.join(env_path, "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            env = joblib.load(vec_normalize_path)
            logger.info(f"Loaded vectorized environment from {vec_normalize_path}")
            return env
            
        raise FileNotFoundError(f"No environment file found in {env_path}")
    except Exception as e:
        logger.error(f"Error loading environment: {e}")
        raise

def load_model(model_path: str, env: gym.Env) -> PPO:
    """Load model from file"""
    try:
        model = PPO.load(model_path, env=env)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def collect_actions(model: PPO, env: gym.Env, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Collect model predictions on random observations
    
    Returns:
        Dictionary with collected actions and stats
    """
    logger.info(f"Collecting {num_samples} action samples...")
    
    # Get list of assets
    if hasattr(env, 'assets'):
        assets = env.assets
    elif hasattr(env, 'get_attr') and hasattr(env.envs[0], 'assets'):
        assets = env.envs[0].assets
    else:
        # If we can't determine assets, use generic names
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        num_assets = action_shape[0] if len(action_shape) > 0 else 1
        assets = [f"asset_{i}" for i in range(num_assets)]
    
    logger.info(f"Detected assets: {assets}")
    
    # Initialize collections
    actions = []
    observations = []
    
    # Reset environment
    obs, _ = env.reset()
    
    # Collect samples
    for i in range(num_samples):
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Store
        actions.append(action)
        observations.append(obs)
        
        # Step environment
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Reset if needed
        if terminated or truncated:
            obs, _ = env.reset()
            
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Collected {i + 1}/{num_samples} samples")
    
    # Convert to numpy arrays
    actions_np = np.vstack(actions)
    observations_np = np.vstack(observations)
    
    # Analysis results
    results = {
        'actions': actions_np,
        'observations': observations_np,
        'assets': assets
    }
    
    return results

def analyze_actions(actions: np.ndarray, assets: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Analyze collected actions for biases
    
    Returns:
        Dictionary of statistics by asset
    """
    logger.info("Analyzing action distributions...")
    
    stats = {}
    num_assets = min(actions.shape[1], len(assets))
    
    for i in range(num_assets):
        asset_name = assets[i]
        asset_actions = actions[:, i]
        
        # Calculate statistics
        mean = float(np.mean(asset_actions))
        median = float(np.median(asset_actions))
        std = float(np.std(asset_actions))
        min_val = float(np.min(asset_actions))
        max_val = float(np.max(asset_actions))
        
        # Calculate percentages
        pct_positive = float(np.mean(asset_actions > 0))
        pct_negative = float(np.mean(asset_actions < 0))
        pct_zero = float(np.mean(np.abs(asset_actions) < 1e-6))
        
        # Extreme values
        pct_extreme_positive = float(np.mean(asset_actions > 0.9))
        pct_extreme_negative = float(np.mean(asset_actions < -0.9))
        
        # Store statistics
        stats[asset_name] = {
            'mean': mean,
            'median': median,
            'std': std,
            'min': min_val,
            'max': max_val,
            'pct_positive': pct_positive,
            'pct_negative': pct_negative,
            'pct_zero': pct_zero,
            'pct_extreme_positive': pct_extreme_positive,
            'pct_extreme_negative': pct_extreme_negative,
            'bias_level': abs(mean)
        }
    
    return stats

def correct_actions(actions: np.ndarray, stats: Dict[str, Dict[str, float]], 
                    assets: List[str], method: str = 'mean_centering') -> np.ndarray:
    """
    Apply bias correction to actions
    
    Args:
        actions: Action array
        stats: Action statistics by asset
        assets: List of asset names
        method: Correction method ('mean_centering', 'distribution_scaling', 'nonlinear')
        
    Returns:
        Corrected actions
    """
    logger.info(f"Applying {method} bias correction...")
    
    corrected_actions = actions.copy()
    num_assets = min(actions.shape[1], len(assets))
    
    for i in range(num_assets):
        asset_name = assets[i]
        asset_stats = stats[asset_name]
        
        if method == 'mean_centering':
            # Correct mean bias (shift distribution)
            corrected_actions[:, i] = actions[:, i] - (asset_stats['mean'] * 0.7)
            
        elif method == 'distribution_scaling':
            # Scale the distribution to correct skew and extreme values
            mean = asset_stats['mean']
            std = asset_stats['std']
            
            # Center and normalize
            centered = actions[:, i] - mean
            if std > 0:
                normalized = centered / std
                # Scale back with target std (slightly reduced to avoid extremes)
                corrected_actions[:, i] = normalized * (std * 0.9)
            else:
                corrected_actions[:, i] = centered
                
        elif method == 'nonlinear':
            # Non-linear correction using tanh to handle extreme values
            # while preserving direction and relative magnitude
            mean = asset_stats['mean']
            
            # Remove mean bias
            centered = actions[:, i] - (mean * 0.7)
            
            # Apply non-linear transformation to extreme values
            extreme_mask = np.abs(centered) > 0.7
            if np.any(extreme_mask):
                # Apply tanh scaling to extreme values
                extreme_values = centered[extreme_mask]
                signs = np.sign(extreme_values)
                magnitudes = np.abs(extreme_values)
                
                # Scaled magnitudes (tanh keeps values in [-1,1] range)
                scaled_magnitudes = np.tanh(magnitudes * 1.2) * 0.85
                
                # Replace extreme values
                centered[extreme_mask] = signs * scaled_magnitudes
                
            corrected_actions[:, i] = centered
    
    # Ensure all actions are within valid range
    corrected_actions = np.clip(corrected_actions, -1, 1)
    
    return corrected_actions

def plot_action_distributions(actions: np.ndarray, corrected_actions: np.ndarray, 
                             assets: List[str], stats: Dict[str, Dict[str, float]],
                             output_dir: str):
    """
    Plot action distributions before and after correction
    """
    logger.info("Plotting action distributions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Plot histograms for each asset
    num_assets = min(actions.shape[1], len(assets))
    
    for i in range(num_assets):
        asset_name = assets[i]
        asset_stats = stats[asset_name]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Action Distribution for {asset_name}", fontsize=16)
        
        # Original histogram
        sns.histplot(actions[:, i], bins=50, kde=True, ax=axs[0, 0], color='blue')
        axs[0, 0].set_title(f"Original Distribution\nMean: {asset_stats['mean']:.4f}, Std: {asset_stats['std']:.4f}")
        axs[0, 0].axvline(asset_stats['mean'], color='red', linestyle='--', label=f"Mean: {asset_stats['mean']:.4f}")
        axs[0, 0].axvline(0, color='black', linestyle='-', alpha=0.3)
        axs[0, 0].legend()
        
        # Corrected histogram
        corrected_mean = np.mean(corrected_actions[:, i])
        corrected_std = np.std(corrected_actions[:, i])
        sns.histplot(corrected_actions[:, i], bins=50, kde=True, ax=axs[0, 1], color='green')
        axs[0, 1].set_title(f"Corrected Distribution\nMean: {corrected_mean:.4f}, Std: {corrected_std:.4f}")
        axs[0, 1].axvline(corrected_mean, color='red', linestyle='--', label=f"Mean: {corrected_mean:.4f}")
        axs[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        axs[0, 1].legend()
        
        # Original vs corrected scatter
        axs[1, 0].scatter(actions[:, i], corrected_actions[:, i], alpha=0.5, s=10)
        axs[1, 0].set_title("Original vs Corrected Actions")
        axs[1, 0].set_xlabel("Original Action")
        axs[1, 0].set_ylabel("Corrected Action")
        axs[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.3)  # Diagonal line
        
        # Direction change analysis
        direction_original = np.sign(actions[:, i])
        direction_corrected = np.sign(corrected_actions[:, i])
        direction_changed = direction_original != direction_corrected
        pct_direction_changed = np.mean(direction_changed) * 100
        
        categories = ['Positive→Negative', 'Negative→Positive', 'Zero→Nonzero', 'Nonzero→Zero', 'No Change']
        counts = [0, 0, 0, 0, 0]
        
        # Count specific changes
        for j in range(len(direction_original)):
            if not direction_changed[j]:
                counts[4] += 1  # No change
            elif direction_original[j] > 0 and direction_corrected[j] < 0:
                counts[0] += 1  # Positive to Negative
            elif direction_original[j] < 0 and direction_corrected[j] > 0:
                counts[1] += 1  # Negative to Positive
            elif abs(direction_original[j]) < 1e-6 and abs(direction_corrected[j]) > 1e-6:
                counts[2] += 1  # Zero to Nonzero
            elif abs(direction_original[j]) > 1e-6 and abs(direction_corrected[j]) < 1e-6:
                counts[3] += 1  # Nonzero to Zero
        
        # Convert to percentages
        counts = [c / len(direction_original) * 100 for c in counts]
        
        # Plot direction changes
        axs[1, 1].bar(categories, counts, color=['red', 'green', 'orange', 'blue', 'gray'])
        axs[1, 1].set_title(f"Direction Changes: {pct_direction_changed:.1f}% of actions")
        axs[1, 1].set_xticklabels(categories, rotation=45, ha='right')
        axs[1, 1].set_ylabel("Percentage of Actions")
        for i, count in enumerate(counts):
            if count > 0.5:
                axs[1, 1].text(i, count + 0.5, f"{count:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{asset_name}_action_distribution.png"), dpi=300)
        plt.close(fig)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Compare bias before and after correction
    asset_names = list(stats.keys())
    original_bias = [stats[asset]['bias_level'] for asset in asset_names]
    
    # Calculate corrected bias
    corrected_bias = []
    for i, asset in enumerate(asset_names):
        if i < actions.shape[1]:
            corrected_actions_i = corrected_actions[:, i]
            corrected_bias.append(abs(np.mean(corrected_actions_i)))
        else:
            corrected_bias.append(0)
    
    # Create bar chart
    bar_width = 0.35
    x = np.arange(len(asset_names))
    
    plt.bar(x - bar_width/2, original_bias, bar_width, label='Original Bias')
    plt.bar(x + bar_width/2, corrected_bias, bar_width, label='Corrected Bias')
    
    plt.xlabel('Assets')
    plt.ylabel('Absolute Bias Level')
    plt.title('Bias Comparison Before vs After Correction')
    plt.xticks(x, asset_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "bias_comparison_summary.png"), dpi=300)
    plt.close()
    
    logger.info(f"Saved plots to {output_dir}")

def save_action_data(actions: np.ndarray, corrected_actions: np.ndarray, 
                    stats: Dict[str, Dict[str, float]], assets: List[str],
                    output_dir: str):
    """
    Save action data to CSV files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics to CSV
    stats_rows = []
    for asset, asset_stats in stats.items():
        row = {'asset': asset}
        row.update(asset_stats)
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(output_dir, "action_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved statistics to {stats_path}")
    
    # Save sample of raw actions to CSV
    max_samples = min(1000, actions.shape[0])
    action_data = {}
    corrected_data = {}
    
    for i, asset in enumerate(assets):
        if i < actions.shape[1]:
            action_data[f"{asset}_action"] = actions[:max_samples, i]
            corrected_data[f"{asset}_corrected"] = corrected_actions[:max_samples, i]
    
    actions_df = pd.DataFrame(action_data)
    corrected_df = pd.DataFrame(corrected_data)
    
    actions_path = os.path.join(output_dir, "sample_actions.csv")
    corrected_path = os.path.join(output_dir, "sample_corrected_actions.csv")
    
    actions_df.to_csv(actions_path, index=False)
    corrected_df.to_csv(corrected_path, index=False)
    
    logger.info(f"Saved action samples to {actions_path} and {corrected_path}")

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
        model = load_model(args.model_path, env)
        
        # Collect actions
        results = collect_actions(model, env, num_samples=args.num_samples)
        actions = results['actions']
        assets = results['assets']
        
        # Analyze actions
        stats = analyze_actions(actions, assets)
        
        # Print summary statistics
        logger.info("=== Action Statistics Summary ===")
        for asset, asset_stats in stats.items():
            bias_level = asset_stats['bias_level']
            bias_category = "HIGH" if bias_level > 0.2 else "MEDIUM" if bias_level > 0.1 else "LOW"
            
            logger.info(f"Asset: {asset} - Bias Level: {bias_level:.4f} ({bias_category})")
            logger.info(f"  Mean: {asset_stats['mean']:.4f}, Median: {asset_stats['median']:.4f}, Std: {asset_stats['std']:.4f}")
            logger.info(f"  Range: [{asset_stats['min']:.4f}, {asset_stats['max']:.4f}]")
            logger.info(f"  Positive: {asset_stats['pct_positive']*100:.1f}%, Negative: {asset_stats['pct_negative']*100:.1f}%")
            logger.info(f"  Extreme: +{asset_stats['pct_extreme_positive']*100:.1f}%, -{asset_stats['pct_extreme_negative']*100:.1f}%")
            logger.info("")
        
        # Apply bias correction
        corrected_actions = correct_actions(actions, stats, assets, method=args.correction_method)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot distributions
        plot_action_distributions(actions, corrected_actions, assets, stats, args.output_dir)
        
        # Save data
        save_action_data(actions, corrected_actions, stats, assets, args.output_dir)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 