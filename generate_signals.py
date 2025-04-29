#!/usr/bin/env python
# Script to generate trading signals with bias correction

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
    parser = argparse.ArgumentParser(description='Generate Trading Signals with Bias Correction')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the saved model')
    parser.add_argument('--env-path', type=str, required=True,
                        help='Path to the saved environment')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Number of steps to generate signals for (default: 100)')
    parser.add_argument('--output-dir', type=str, default='signals',
                        help='Directory to save signals')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (default: False)')
    parser.add_argument('--use-correction', action='store_true',
                        help='Apply bias correction to the signals (default: False)')
    parser.add_argument('--signal-threshold', type=float, default=0.25,
                        help='Threshold for signal significance (default: 0.25)')
    return parser.parse_args()

def correct_bias(action, asset_bias=None):
    """Apply bias correction to an action"""
    if asset_bias is None:
        # Default correction for common biases
        # For BTC and ETH which tend to get shorted heavily
        asset_bias = {
            'BTC': 0.3,    # BTC has -0.3 bias (gets shorted too much)
            'ETH': 0.25,   # ETH has -0.25 bias (gets shorted too much)
            'SOL': -0.15   # SOL has +0.15 bias (gets longed too much)
        }
    
    corrected = action.copy()
    
    # Apply individual asset corrections
    for i, asset in enumerate(asset_bias.keys()):
        if i < len(corrected):
            # Add the bias correction (opposite of the bias)
            corrected[i] += asset_bias[asset]
    
    # Apply non-linear scaling to extreme values
    for i in range(len(corrected)):
        if abs(corrected[i]) > 0.7:
            # Use tanh to compress extreme values while preserving direction
            sign = np.sign(corrected[i])
            magnitude = abs(corrected[i])
            scaled_magnitude = np.tanh(magnitude * 1.2) * 0.8
            corrected[i] = sign * scaled_magnitude
    
    # Final clipping to ensure values are in [-1, 1]
    corrected = np.clip(corrected, -1.0, 1.0)
    
    return corrected

def action_to_signal(action, threshold=0.25):
    """Convert action to trading signal with confidence level"""
    signals = []
    
    for i, a in enumerate(action):
        # Skip weak signals
        if abs(a) < threshold:
            direction = "HOLD"
            leverage = 0.0
        else:
            # Determine direction
            direction = "LONG" if a > 0 else "SHORT"
            
            # Calculate leverage based on signal strength
            # Scale from threshold to 1.0 into leverage range 1.0-20.0
            signal_strength = (min(abs(a), 1.0) - threshold) / (1.0 - threshold)
            # Adjust the curve to be more conservative at higher leverage
            leverage = 1.0 + (20.0 - 1.0) * (signal_strength ** 1.5)
            leverage = min(leverage, 20.0)  # Cap at 20x
        
        signals.append({
            'direction': direction,
            'leverage': round(leverage, 2),
            'raw_signal': a
        })
    
    return signals

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
    
    # Generate signals
    logger.info(f"Generating signals for {args.num_steps} steps...")
    
    # Initialize DataFrame to store signals
    signals_df = pd.DataFrame()
    
    # Reset environment
    obs = env.reset()
    
    # Generate signals
    for step in range(args.num_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=args.deterministic)
        action = action[0]  # Get action for first environment if vectorized
        
        # Apply bias correction if requested
        if args.use_correction:
            corrected_action = correct_bias(action)
        else:
            corrected_action = action.copy()
        
        # Convert to trading signals
        signals = action_to_signal(corrected_action, threshold=args.signal_threshold)
        
        # Step the environment
        obs, _, dones, _ = env.step(action)
        if dones.any():
            obs = env.reset()
        
        # Create record for this step
        step_data = {'step': step}
        
        # Add raw and corrected actions
        for i, asset in enumerate(asset_names):
            if i < len(action):
                step_data[f'{asset}_raw'] = action[i]
                step_data[f'{asset}_corrected'] = corrected_action[i]
                step_data[f'{asset}_direction'] = signals[i]['direction']
                step_data[f'{asset}_leverage'] = signals[i]['leverage']
        
        # Add to DataFrame
        signals_df = pd.concat([signals_df, pd.DataFrame([step_data])], ignore_index=True)
        
        # Log progress
        if (step + 1) % 10 == 0 or step == args.num_steps - 1:
            logger.info(f"Generated signals for step {step + 1}/{args.num_steps}")
    
    # Save signals to CSV
    signals_file = os.path.join(args.output_dir, 'trading_signals.csv')
    signals_df.to_csv(signals_file, index=False)
    logger.info(f"Saved signals to {signals_file}")
    
    # Create summary DataFrame
    summary = pd.DataFrame()
    for asset in asset_names:
        # Calculate statistics for raw and corrected signals
        raw_col = f'{asset}_raw'
        corrected_col = f'{asset}_corrected'
        
        if raw_col in signals_df.columns and corrected_col in signals_df.columns:
            # Signal statistics
            summary.loc[asset, 'raw_mean'] = signals_df[raw_col].mean()
            summary.loc[asset, 'corrected_mean'] = signals_df[corrected_col].mean()
            summary.loc[asset, 'raw_std'] = signals_df[raw_col].std()
            summary.loc[asset, 'corrected_std'] = signals_df[corrected_col].std()
            
            # Direction stats
            long_pct = (signals_df[f'{asset}_direction'] == 'LONG').mean() * 100
            short_pct = (signals_df[f'{asset}_direction'] == 'SHORT').mean() * 100
            hold_pct = (signals_df[f'{asset}_direction'] == 'HOLD').mean() * 100
            
            summary.loc[asset, 'long_pct'] = long_pct
            summary.loc[asset, 'short_pct'] = short_pct
            summary.loc[asset, 'hold_pct'] = hold_pct
            
            # Average leverage when trading
            trading_mask = signals_df[f'{asset}_direction'] != 'HOLD'
            if trading_mask.sum() > 0:
                avg_leverage = signals_df.loc[trading_mask, f'{asset}_leverage'].mean()
                summary.loc[asset, 'avg_leverage'] = avg_leverage
            else:
                summary.loc[asset, 'avg_leverage'] = 0
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'signal_summary.csv')
    summary.to_csv(summary_file)
    logger.info(f"Saved signal summary to {summary_file}")
    
    # Print summary
    print("\nSignal Summary:")
    print("==============")
    print(summary)
    
    # Plot signal distributions
    fig, axes = plt.subplots(len(asset_names), 2, figsize=(14, 4 * len(asset_names)))
    
    for i, asset in enumerate(asset_names):
        # Raw signals histogram
        axes[i, 0].hist(signals_df[f'{asset}_raw'], bins=20, alpha=0.7, color='blue')
        axes[i, 0].set_title(f'{asset} Raw Signals')
        axes[i, 0].set_xlabel('Signal Value')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Corrected signals histogram
        axes[i, 1].hist(signals_df[f'{asset}_corrected'], bins=20, alpha=0.7, color='green')
        axes[i, 1].set_title(f'{asset} Corrected Signals')
        axes[i, 1].set_xlabel('Signal Value')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add threshold lines
        axes[i, 1].axvline(args.signal_threshold, color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axvline(-args.signal_threshold, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'signal_distributions.png'), dpi=300)
    plt.close()
    
    logger.info("Signal generation complete!")

if __name__ == "__main__":
    main() 