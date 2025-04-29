#!/usr/bin/env python
# Script to test model action outputs for bias detection and correction

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Test Model Action Biases')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the saved model')
    parser.add_argument('--env-path', type=str, required=True,
                        help='Path to the saved environment')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to collect (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy for action prediction')
    parser.add_argument('--apply-correction', action='store_true',
                        help='Apply action bias correction')
    return parser.parse_args()

def analyze_actions(actions, asset_names):
    """Analyze action statistics"""
    actions_df = pd.DataFrame(actions, columns=asset_names)
    
    # Calculate basic statistics
    stats = {}
    for asset in asset_names:
        asset_actions = actions_df[asset]
        stats[asset] = {
            'mean': asset_actions.mean(),
            'std': asset_actions.std(),
            'median': asset_actions.median(),
            'min': asset_actions.min(),
            'max': asset_actions.max(),
            'abs_mean': asset_actions.abs().mean(),
            'positive_pct': (asset_actions > 0).mean() * 100,
            'negative_pct': (asset_actions < 0).mean() * 100,
            'zero_pct': (asset_actions == 0).mean() * 100,
            'extreme_pct': (asset_actions.abs() > 0.9).mean() * 100
        }
    
    # Overall statistics
    stats['overall'] = {
        'mean': actions_df.values.mean(),
        'std': actions_df.values.std(),
        'abs_mean': np.abs(actions_df.values).mean(),
        'extreme_pct': (np.abs(actions_df.values) > 0.9).mean() * 100
    }
    
    return stats, actions_df

def correct_actions(actions, sensitivity=0.7):
    """Apply bias correction to actions"""
    corrected = actions.copy()
    
    # Calculate bias for each column
    for col in corrected.columns:
        col_mean = corrected[col].mean()
        col_abs_mean = corrected[col].abs().mean()
        
        # Detect bias
        if abs(col_mean) > 0.1:  # Significant directional bias
            # Apply partial bias correction (remove some of the bias)
            corrected[col] = corrected[col] - (col_mean * sensitivity)
            
        # If values are mostly extreme, soften them
        if col_abs_mean > 0.7:
            # Apply sigmoid-like scaling to compress extreme values
            corrected[col] = corrected[col].apply(
                lambda x: np.sign(x) * np.tanh(abs(x) * 1.2) * 0.85
            )
    
    return corrected

def plot_action_distributions(orig_actions, corrected_actions, asset_names, output_dir):
    """Plot original and corrected action distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms for each asset
    fig, axes = plt.subplots(len(asset_names), 2, figsize=(14, 4 * len(asset_names)))
    
    for i, asset in enumerate(asset_names):
        # Histogram
        axes[i, 0].hist(orig_actions[asset], bins=30, alpha=0.5, label='Original')
        axes[i, 0].hist(corrected_actions[asset], bins=30, alpha=0.5, label='Corrected')
        axes[i, 0].set_title(f'{asset} Action Distribution')
        axes[i, 0].set_xlabel('Action Value')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].legend()
        axes[i, 0].axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # Scatter plot (original vs corrected)
        axes[i, 1].scatter(orig_actions[asset], corrected_actions[asset], alpha=0.5)
        axes[i, 1].set_title(f'{asset} Original vs Corrected')
        axes[i, 1].set_xlabel('Original Action')
        axes[i, 1].set_ylabel('Corrected Action')
        # Add diagonal line
        min_val = min(orig_actions[asset].min(), corrected_actions[asset].min())
        max_val = max(orig_actions[asset].max(), corrected_actions[asset].max())
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distributions.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    logger.info(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)
    
    # Load the environment
    logger.info(f"Loading environment from {args.env_path}")
    env = VecNormalize.load(args.env_path, None)
    
    # Get asset names from environment
    if hasattr(env, 'get_attr'):
        # For vectorized environments
        asset_names = env.get_attr('assets')[0]
    else:
        # Fallback
        asset_names = ['BTC', 'ETH', 'SOL']  # Default asset names
    
    # Update the model with the environment
    model.set_env(env)
    
    # Collect samples
    logger.info(f"Collecting {args.num_samples} action samples...")
    actions = []
    
    # Reset environment to start collecting samples
    obs = env.reset()
    
    # Collect samples
    for i in range(args.num_samples):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        actions.append(action[0])  # Only take the first env's action if vectorized
        
        # Step environment to get next observation
        obs, _, dones, _ = env.step(action)
        if dones.any():
            obs = env.reset()
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Collected {i + 1}/{args.num_samples} samples")
    
    # Convert to array
    actions = np.array(actions)
    
    # Analyze actions
    logger.info("Analyzing action distributions...")
    stats, actions_df = analyze_actions(actions, asset_names)
    
    # Print statistics
    print("\nAction Statistics:")
    print("=================")
    
    for asset in asset_names:
        print(f"\n{asset}:")
        for metric, value in stats[asset].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nOverall:")
    for metric, value in stats['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Apply correction if requested
    if args.apply_correction:
        logger.info("Applying action bias correction...")
        corrected_df = correct_actions(actions_df)
        
        # Analyze corrected actions
        corrected_stats, _ = analyze_actions(corrected_df.values, asset_names)
        
        # Print corrected statistics
        print("\nCorrected Action Statistics:")
        print("===========================")
        
        for asset in asset_names:
            print(f"\n{asset}:")
            for metric, value in corrected_stats[asset].items():
                print(f"  {metric}: {value:.4f}")
        
        print("\nOverall:")
        for metric, value in corrected_stats['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Save before/after stats to CSV
        comparison = pd.DataFrame()
        for asset in asset_names:
            for metric in ['mean', 'std', 'abs_mean', 'extreme_pct']:
                comparison.loc[f"{asset}_{metric}", 'Original'] = stats[asset][metric]
                comparison.loc[f"{asset}_{metric}", 'Corrected'] = corrected_stats[asset][metric]
        
        comparison.to_csv(os.path.join(args.output_dir, 'action_comparison.csv'))
        
        # Plot action distributions
        plot_action_distributions(actions_df, corrected_df, asset_names, args.output_dir)

if __name__ == "__main__":
    main() 