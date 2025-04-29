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
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiasCorrector:
    """
    Adaptive bias correction for model actions
    Works with any assets by learning bias patterns dynamically
    """
    def __init__(self, window_size: int = 100, correction_strength: float = 0.7,
                 min_samples: int = 50, method: str = 'nonlinear'):
        self.window_size = window_size
        self.correction_strength = correction_strength
        self.min_samples = min_samples
        self.method = method
        self.action_history = {}
        self.bias_stats = {}
        self.is_ready = False
    
    def update(self, actions: np.ndarray, assets: List[str]):
        """Update bias statistics with new actions"""
        # Initialize histories if needed
        for i, asset in enumerate(assets):
            if asset not in self.action_history:
                self.action_history[asset] = []
            
            # Add action to history
            if i < len(actions):
                self.action_history[asset].append(float(actions[i]))
                
                # Limit history size
                if len(self.action_history[asset]) > self.window_size:
                    self.action_history[asset] = self.action_history[asset][-self.window_size:]
        
        # Update statistics for each asset
        for asset in assets:
            if len(self.action_history[asset]) >= self.min_samples:
                actions_array = np.array(self.action_history[asset])
                
                # Calculate statistics
                mean = float(np.mean(actions_array))
                std = float(np.std(actions_array))
                median = float(np.median(actions_array))
                abs_mean = float(np.mean(np.abs(actions_array)))
                pct_positive = float(np.mean(actions_array > 0))
                pct_negative = float(np.mean(actions_array < 0))
                
                # Store statistics
                self.bias_stats[asset] = {
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'abs_mean': abs_mean,
                    'pct_positive': pct_positive,
                    'pct_negative': pct_negative,
                    'bias_level': abs(mean),
                    'samples': len(self.action_history[asset])
                }
        
        # Mark as ready if we have stats for all assets
        self.is_ready = all(asset in self.bias_stats for asset in assets)
        
        return self.is_ready
    
    def correct(self, actions: np.ndarray, assets: List[str]) -> np.ndarray:
        """Apply adaptive bias correction to actions"""
        # If not ready, return original actions
        if not self.is_ready:
            return actions
        
        # Apply corrections based on method
        corrected = actions.copy()
        
        for i, asset in enumerate(assets):
            if i >= len(actions) or asset not in self.bias_stats:
                continue
                
            stats = self.bias_stats[asset]
            
            if self.method == 'mean_centering':
                # Simple mean centering
                corrected[i] = actions[i] - (stats['mean'] * self.correction_strength)
                
            elif self.method == 'nonlinear':
                # First apply mean correction
                centered = actions[i] - (stats['mean'] * self.correction_strength)
                
                # Then apply nonlinear scaling to extreme values
                if abs(centered) > 0.7:
                    # Apply tanh transformation to compress extreme values
                    sign = np.sign(centered)
                    magnitude = abs(centered)
                    scaled = np.tanh(magnitude * 1.2) * 0.85
                    corrected[i] = sign * scaled
                else:
                    corrected[i] = centered
                    
            elif self.method == 'distribution_scaling':
                # Center and standardize
                if stats['std'] > 0:
                    centered = (actions[i] - stats['mean'])
                    standardized = centered / stats['std']
                    # Scale back with target mean=0, std slightly reduced
                    corrected[i] = standardized * (stats['std'] * 0.9)
                else:
                    corrected[i] = actions[i] - stats['mean']
        
        # Clip to valid range
        corrected = np.clip(corrected, -1, 1)
        
        return corrected

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

def action_to_signal(action: float, threshold: float = 0.25, max_leverage: float = 20.0) -> Dict[str, Any]:
    """
    Convert action value to trading signal
    
    Args:
        action: Model action value (-1 to 1)
        threshold: Signal threshold (0 to 1)
        max_leverage: Maximum leverage allowed
        
    Returns:
        Dictionary with signal details
    """
    # Default signal (HOLD)
    signal = {
        'direction': 'HOLD',
        'strength': 0.0,
        'leverage': 0.0,
        'confidence': 0.0,
        'raw_action': float(action)
    }
    
    # Check if action exceeds threshold
    if abs(action) <= threshold:
        return signal
    
    # Determine direction
    direction = 'LONG' if action > 0 else 'SHORT'
    
    # Calculate normalized strength (0 to 1)
    # Scale from threshold to 1.0
    raw_strength = (abs(action) - threshold) / (1.0 - threshold)
    strength = min(1.0, max(0.0, raw_strength))
    
    # Calculate confidence based on action magnitude
    confidence = strength
    
    # Calculate leverage using non-linear scaling
    # This gives more precise control in the lower ranges
    # and reaches max leverage only for very strong signals
    leverage = 1.0 + (max_leverage - 1.0) * (strength ** 1.5)
    leverage = min(max_leverage, max(1.0, leverage))
    
    # Build signal dictionary
    signal = {
        'direction': direction,
        'strength': float(strength),
        'leverage': float(leverage),
        'confidence': float(confidence),
        'raw_action': float(action)
    }
    
    return signal

def generate_signals(model: PPO, env: gym.Env, num_steps: int, 
                    use_correction: bool = True, signal_threshold: float = 0.25,
                    deterministic: bool = False) -> Dict[str, Any]:
    """
    Generate trading signals using the model
    
    Args:
        model: Trained PPO model
        env: Environment
        num_steps: Number of steps to generate signals for
        use_correction: Whether to apply bias correction
        signal_threshold: Threshold for signal generation
        deterministic: Whether to use deterministic policy
        
    Returns:
        Dictionary with generated signals and metadata
    """
    logger.info(f"Generating signals for {num_steps} steps (correction: {use_correction})")
    
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
    
    logger.info(f"Generating signals for assets: {assets}")
    
    # Initialize bias corrector if needed
    bias_corrector = BiasCorrector(window_size=min(100, num_steps), 
                                 correction_strength=0.7,
                                 method='nonlinear') if use_correction else None
    
    # Initialize collections
    signals_by_asset = {asset: [] for asset in assets}
    raw_actions = []
    corrected_actions = []
    observations = []
    timestamps = []
    
    # Reset environment
    obs, _ = env.reset()
    
    # Generate signals
    for step in range(num_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Store observation
        observations.append(obs.copy())
        
        # Apply bias correction if enabled
        if use_correction and bias_corrector is not None:
            # Update bias statistics
            bias_corrector.update(action, assets)
            
            # Apply correction
            corrected_action = bias_corrector.correct(action, assets)
            
            # Store both raw and corrected actions
            raw_actions.append(action.copy())
            corrected_actions.append(corrected_action.copy())
            
            # Use corrected action for signal generation
            signal_action = corrected_action
        else:
            # Store raw action
            raw_actions.append(action.copy())
            corrected_actions.append(action.copy())  # Same as raw for consistency
            
            # Use raw action for signal generation
            signal_action = action
        
        # Generate signals for each asset
        for i, asset in enumerate(assets):
            if i < len(signal_action):
                # Convert action to signal
                signal = action_to_signal(
                    signal_action[i], 
                    threshold=signal_threshold
                )
                
                # Add metadata
                signal['asset'] = asset
                signal['step'] = step
                signal['timestamp'] = datetime.now().isoformat()
                
                # Store signal
                signals_by_asset[asset].append(signal)
        
        # Store timestamp
        timestamps.append(datetime.now().isoformat())
        
        # Step environment
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Reset if needed
        if terminated or truncated:
            obs, _ = env.reset()
            
        # Log progress
        if (step + 1) % 10 == 0 or step == num_steps - 1:
            logger.info(f"Generated signals for {step + 1}/{num_steps} steps")
    
    # Convert to numpy arrays
    raw_actions_np = np.vstack(raw_actions)
    corrected_actions_np = np.vstack(corrected_actions)
    observations_np = np.vstack(observations)
    
    # Compile results
    results = {
        'signals_by_asset': signals_by_asset,
        'raw_actions': raw_actions_np,
        'corrected_actions': corrected_actions_np,
        'observations': observations_np,
        'timestamps': timestamps,
        'assets': assets,
        'metadata': {
            'model_path': str(model),
            'num_steps': num_steps,
            'use_correction': use_correction,
            'signal_threshold': signal_threshold,
            'deterministic': deterministic,
            'generation_time': datetime.now().isoformat()
        }
    }
    
    # Add bias statistics if available
    if use_correction and bias_corrector is not None and bias_corrector.bias_stats:
        results['bias_stats'] = bias_corrector.bias_stats
    
    return results

def save_signals(signals: Dict[str, Any], output_dir: str):
    """
    Save generated signals to CSV files
    
    Args:
        signals: Dictionary with signal data
        output_dir: Output directory
    """
    logger.info(f"Saving signals to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save signals by asset
    for asset, asset_signals in signals['signals_by_asset'].items():
        if asset_signals:
            # Convert to DataFrame
            df = pd.DataFrame(asset_signals)
            
            # Save to CSV
            filepath = os.path.join(output_dir, f"{asset}_signals.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(asset_signals)} signals for {asset} to {filepath}")
    
    # Save summary
    summary_rows = []
    for asset, asset_signals in signals['signals_by_asset'].items():
        if not asset_signals:
            continue
            
        # Count signals by direction
        long_count = sum(1 for s in asset_signals if s['direction'] == 'LONG')
        short_count = sum(1 for s in asset_signals if s['direction'] == 'SHORT')
        hold_count = sum(1 for s in asset_signals if s['direction'] == 'HOLD')
        
        # Calculate average leverage and confidence
        avg_leverage = np.mean([s['leverage'] for s in asset_signals if s['direction'] != 'HOLD']) if long_count + short_count > 0 else 0
        avg_confidence = np.mean([s['confidence'] for s in asset_signals if s['direction'] != 'HOLD']) if long_count + short_count > 0 else 0
        
        # Add to summary
        summary_rows.append({
            'asset': asset,
            'total_signals': len(asset_signals),
            'long_signals': long_count,
            'short_signals': short_count,
            'hold_signals': hold_count,
            'long_pct': long_count / len(asset_signals) * 100 if asset_signals else 0,
            'short_pct': short_count / len(asset_signals) * 100 if asset_signals else 0,
            'hold_pct': hold_count / len(asset_signals) * 100 if asset_signals else 0,
            'avg_leverage': avg_leverage,
            'avg_confidence': avg_confidence
        })
    
    # Save summary to CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "signals_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved signals summary to {summary_path}")
    
    # Save bias statistics if available
    if 'bias_stats' in signals:
        bias_stats_path = os.path.join(output_dir, "bias_statistics.json")
        with open(bias_stats_path, 'w') as f:
            json.dump(signals['bias_stats'], f, indent=2)
        logger.info(f"Saved bias statistics to {bias_stats_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        # Convert numpy arrays in metadata to lists for JSON serialization
        metadata = signals['metadata'].copy()
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
                
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

def plot_signals(signals: Dict[str, Any], output_dir: str):
    """
    Create visualizations of the generated signals
    
    Args:
        signals: Dictionary with signal data
        output_dir: Output directory
    """
    logger.info("Creating signal visualizations...")
    
    # Create output directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # 1. Plot signal distributions for each asset
    for asset, asset_signals in signals['signals_by_asset'].items():
        if not asset_signals:
            continue
            
        # Extract data
        raw_actions = [s['raw_action'] for s in asset_signals]
        directions = [s['direction'] for s in asset_signals]
        leverages = [s['leverage'] for s in asset_signals]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Trading Signals for {asset}", fontsize=16)
        
        # Plot raw action distribution
        sns.histplot(raw_actions, bins=50, kde=True, ax=axs[0, 0])
        axs[0, 0].set_title("Raw Action Distribution")
        axs[0, 0].set_xlabel("Action Value")
        axs[0, 0].set_ylabel("Frequency")
        axs[0, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Plot signal directions
        direction_counts = pd.Series(directions).value_counts()
        axs[0, 1].pie(
            direction_counts, 
            labels=direction_counts.index,
            autopct='%1.1f%%',
            colors=['green', 'red', 'gray'] if 'LONG' in direction_counts.index and 'SHORT' in direction_counts.index else None,
            explode=[0.05 if d != 'HOLD' else 0 for d in direction_counts.index]
        )
        axs[0, 1].set_title("Signal Directions")
        
        # Plot leverage distribution (excluding HOLDs)
        non_hold_leverages = [leverages[i] for i, d in enumerate(directions) if d != 'HOLD']
        non_hold_directions = [directions[i] for i, d in enumerate(directions) if d != 'HOLD']
        
        if non_hold_leverages:
            # Create colors based on direction
            colors = ['green' if d == 'LONG' else 'red' for d in non_hold_directions]
            
            # Scatter plot of leverages
            for i, (lev, dir) in enumerate(zip(non_hold_leverages, non_hold_directions)):
                axs[1, 0].scatter(i, lev, color='green' if dir == 'LONG' else 'red', alpha=0.7)
            
            axs[1, 0].set_title("Leverage by Signal")
            axs[1, 0].set_xlabel("Signal Index")
            axs[1, 0].set_ylabel("Leverage")
            axs[1, 0].set_ylim(0, max(non_hold_leverages) * 1.1)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='LONG'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='SHORT')
            ]
            axs[1, 0].legend(handles=legend_elements)
            
            # Histogram of leverages
            sns.histplot(non_hold_leverages, bins=20, kde=True, ax=axs[1, 1])
            axs[1, 1].set_title("Leverage Distribution")
            axs[1, 1].set_xlabel("Leverage")
            axs[1, 1].set_ylabel("Frequency")
        else:
            axs[1, 0].text(0.5, 0.5, "No active signals", horizontalalignment='center', verticalalignment='center')
            axs[1, 0].set_title("Leverage by Signal")
            
            axs[1, 1].text(0.5, 0.5, "No active signals", horizontalalignment='center', verticalalignment='center')
            axs[1, 1].set_title("Leverage Distribution")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{asset}_signals.png"), dpi=300)
        plt.close(fig)
    
    # 2. Plot bias correction effect
    if 'raw_actions' in signals and 'corrected_actions' in signals:
        raw_actions = signals['raw_actions']
        corrected_actions = signals['corrected_actions']
        assets = signals['assets']
        
        # For each asset, plot before vs after correction
        for i, asset in enumerate(assets):
            if i >= raw_actions.shape[1]:
                continue
                
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Bias Correction Effect for {asset}", fontsize=16)
            
            # Raw vs corrected actions
            raw = raw_actions[:, i]
            corrected = corrected_actions[:, i]
            
            # Calculate statistics
            raw_mean = np.mean(raw)
            corrected_mean = np.mean(corrected)
            
            # Histograms
            sns.histplot(raw, bins=30, kde=True, ax=axs[0], color='blue')
            axs[0].axvline(raw_mean, color='red', linestyle='--', label=f'Mean: {raw_mean:.4f}')
            axs[0].axvline(0, color='black', linestyle='-', alpha=0.3)
            axs[0].set_title(f"Raw Actions\nMean: {raw_mean:.4f}")
            axs[0].legend()
            
            sns.histplot(corrected, bins=30, kde=True, ax=axs[1], color='green')
            axs[1].axvline(corrected_mean, color='red', linestyle='--', label=f'Mean: {corrected_mean:.4f}')
            axs[1].axvline(0, color='black', linestyle='-', alpha=0.3)
            axs[1].set_title(f"Corrected Actions\nMean: {corrected_mean:.4f}")
            axs[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{asset}_bias_correction.png"), dpi=300)
            plt.close(fig)
    
    # 3. Create summary comparison across assets
    assets = list(signals['signals_by_asset'].keys())
    if assets:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        asset_names = []
        long_pcts = []
        short_pcts = []
        hold_pcts = []
        
        for asset, asset_signals in signals['signals_by_asset'].items():
            if not asset_signals:
                continue
                
            # Count signals by direction
            long_count = sum(1 for s in asset_signals if s['direction'] == 'LONG')
            short_count = sum(1 for s in asset_signals if s['direction'] == 'SHORT')
            hold_count = sum(1 for s in asset_signals if s['direction'] == 'HOLD')
            total = len(asset_signals)
            
            # Add to lists
            asset_names.append(asset)
            long_pcts.append(long_count / total * 100)
            short_pcts.append(short_count / total * 100)
            hold_pcts.append(hold_count / total * 100)
        
        # Create stacked bar chart
        if asset_names:
            x = range(len(asset_names))
            width = 0.8
            
            plt.bar(x, long_pcts, width, label='LONG', color='green')
            plt.bar(x, short_pcts, width, bottom=long_pcts, label='SHORT', color='red')
            plt.bar(x, hold_pcts, width, bottom=[a+b for a,b in zip(long_pcts, short_pcts)], label='HOLD', color='gray')
            
            plt.xlabel('Assets')
            plt.ylabel('Percentage of Signals')
            plt.title('Signal Direction Distribution by Asset')
            plt.xticks(x, asset_names, rotation=45, ha='right')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "signal_distribution_summary.png"), dpi=300)
            plt.close()
            
            logger.info(f"Saved visualizations to {plots_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate trading signals with bias correction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--env-path", type=str, required=True, help="Path to the environment")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps to generate signals for")
    parser.add_argument("--output-dir", type=str, default="./signals", help="Directory to save signals")
    parser.add_argument("--use-correction", action="store_true", help="Apply bias correction to actions")
    parser.add_argument("--signal-threshold", type=float, default=0.25, help="Threshold for signal generation")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
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
        
        # Generate signals
        signals = generate_signals(
            model, 
            env, 
            num_steps=args.num_steps, 
            use_correction=args.use_correction,
            signal_threshold=args.signal_threshold,
            deterministic=args.deterministic
        )
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"signals_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save signals to CSV
        save_signals(signals, output_dir)
        
        # Create visualizations
        plot_signals(signals, output_dir)
        
        logger.info(f"Signal generation completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 