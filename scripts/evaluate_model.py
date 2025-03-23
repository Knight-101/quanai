#!/usr/bin/env python
"""
Evaluation script for trained trading models

This script loads a trained model and environment, then runs evaluation
episodes to calculate performance metrics including mean reward, Sharpe ratio,
max drawdown, and trading activity.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import logging
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
import pickle
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import unwrap_vec_normalize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained trading model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--env-path', type=str, default=None,
                        help='Path to the saved environment (if not provided, will be inferred from model path)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes to run')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions for evaluation')
    parser.add_argument('--log-wandb', action='store_true',
                        help='Log results to Weights & Biases')
    parser.add_argument('--phase', type=int, default=0,
                        help='Training phase number (for tracking)')
    parser.add_argument('--output-dir', type=str, default='logs/evaluation',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def evaluate_model(model_path, env_path=None, episodes=10, deterministic=True, log_wandb=False, phase=0, output_dir='logs/evaluation'):
    """Evaluate a trained model over multiple episodes and record metrics"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"eval_{os.path.basename(model_path)}_{timestamp}.csv")
        
        # If env_path not provided, infer from model_path
        if env_path is None:
            env_path = model_path.replace('model', 'env') + '.pkl'
            if not os.path.exists(env_path):
                env_path = os.path.dirname(model_path) + "/vec_normalize.pkl"
        
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        logger.info(f"Loading environment from {env_path}")
        
        # Create a DummyVecEnv with a single environment instance
        # First, we need to get the environment parameters from the model
        
        # Load the raw environment data from the pickle file
        with open(env_path, "rb") as file:
            env_data = pickle.load(file)
        
        # Check if it's a wrapped environment or a raw environment
        if hasattr(env_data, 'venv') or hasattr(env_data, 'env'):
            # This is likely a VecNormalize wrapper, try to extract parameters
            try:
                # If it's from an older version where raw env was saved
                if hasattr(env_data, 'venv') and hasattr(env_data.venv, 'envs'):
                    original_env = env_data.venv.envs[0]
                    df = original_env.df
                    assets = original_env.assets
                    parameters = {
                        'df': df,
                        'assets': assets,
                        'window_size': original_env.window_size,
                        'max_leverage': original_env.max_leverage,
                        'commission': original_env.commission,
                        'initial_balance': original_env.initial_balance,
                        'max_drawdown': original_env.max_drawdown
                    }
                    
                    # Create a new base environment
                    base_env = InstitutionalPerpetualEnv(**parameters)
                    # Wrap it in a DummyVecEnv
                    vec_env = DummyVecEnv([lambda: base_env])
                    # Now load the normalized environment with our vec_env
                    env = VecNormalize.load(env_path, vec_env)
                else:
                    logger.error("Cannot extract parameters from saved environment")
                    raise ValueError("Environment format not supported for reconstruction")
            except Exception as e:
                logger.error(f"Error reconstructing environment: {str(e)}")
                raise
        else:
            logger.info("Using direct environment loading...")
            try:
                # For older SB3 versions, try direct loading
                if isinstance(env_path, str) and os.path.isfile(env_path):
                    # Try the simplest approach - load directly
                    env = VecNormalize.load(env_path, DummyVecEnv([lambda: gym.make("CartPole-v1")]))
                    # Replace with our actual environment
                    logger.info("Loaded normalized environment, will use as is")
                else:
                    logger.error("Environment format not supported for direct loading")
                    raise ValueError("Environment format not supported for direct loading")
            except Exception as e:
                logger.error(f"Error with direct loading: {str(e)}")
                logger.info("Falling back to model's get_env method if available")
                try:
                    # Try to use model's env
                    if hasattr(model, "get_env") and callable(model.get_env):
                        env = model.get_env()
                        logger.info("Using environment from model")
                    else:
                        raise ValueError("Cannot load or create environment")
                except Exception as e2:
                    logger.error(f"Failed to get environment from model: {str(e2)}")
                    raise
        
        # Set environment to evaluation mode
        env.training = False  
        env.norm_reward = False  # Don't normalize rewards during evaluation
        
        # Prepare containers for metrics
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        drawdowns = []
        trade_counts = []
        positions_data = []
        gross_leverages = []  # Track gross leverage (always positive)
        net_leverages = []    # Track net leverage (can be negative)
        
        logger.info(f"Starting evaluation over {episodes} episodes...")
        
        # Run evaluation episodes
        for episode in range(episodes):
            logger.info(f"Starting episode {episode+1}/{episodes}")
            
            # Reset environment and get initial observation
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            episode_trades = 0
            episode_portfolio = []
            episode_gross_leverage = []  # Track gross leverage per episode
            episode_net_leverage = []    # Track net leverage per episode
            
            # Track positions and trades for this episode
            episode_positions = []
            
            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Execute action - Gymnasium returns (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Update tracking
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                # Track portfolio value
                if isinstance(info, dict) and 'portfolio_value' in info:
                    episode_portfolio.append(info['portfolio_value'])
                elif isinstance(info, list) and len(info) > 0 and 'portfolio_value' in info[0]:
                    episode_portfolio.append(info[0]['portfolio_value'])
                
                # Track gross and net leverage
                if isinstance(info, dict):
                    if 'gross_leverage' in info:
                        episode_gross_leverage.append(info['gross_leverage'])
                    if 'net_leverage' in info:
                        episode_net_leverage.append(info['net_leverage'])
                elif isinstance(info, list) and len(info) > 0:
                    if 'gross_leverage' in info[0]:
                        episode_gross_leverage.append(info[0]['gross_leverage'])
                    if 'net_leverage' in info[0]:
                        episode_net_leverage.append(info[0]['net_leverage'])
                
                # Track trades
                trades_executed = False
                if isinstance(info, dict) and 'trades_executed' in info:
                    trades_executed = info['trades_executed']
                elif isinstance(info, list) and len(info) > 0 and 'trades_executed' in info[0]:
                    trades_executed = info[0]['trades_executed']
                
                if trades_executed:
                    episode_trades += 1
                
                # Track positions
                positions = None
                if isinstance(info, dict) and 'positions' in info:
                    positions = info['positions']
                elif isinstance(info, list) and len(info) > 0 and 'positions' in info[0]:
                    positions = info[0]['positions']
                
                if positions:
                    pos_snapshot = {
                        'step': step_count,
                        'positions': positions
                    }
                    episode_positions.append(pos_snapshot)
                
                # Update observation
                obs = next_obs
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            trade_counts.append(episode_trades)
            
            # Calculate and record leverage metrics
            if episode_gross_leverage:
                gross_leverages.append(np.mean(episode_gross_leverage))
            else:
                gross_leverages.append(0)
                
            if episode_net_leverage:
                net_leverages.append(np.mean(episode_net_leverage))
            else:
                net_leverages.append(0)
            
            # Calculate portfolio metrics if available
            if episode_portfolio:
                try:
                    portfolio_values.append(max(episode_portfolio))
                    
                    # Calculate max drawdown
                    peak = 0
                    max_dd = 0
                    for value in episode_portfolio:
                        if value > peak:
                            peak = value
                        dd = (peak - value) / peak if peak > 0 else 0
                        max_dd = max(max_dd, dd)
                    
                    drawdowns.append(max_dd)
                except Exception as e:
                    logger.warning(f"Error calculating portfolio metrics: {str(e)}")
                    # Use safe defaults
                    portfolio_values.append(0)
                    drawdowns.append(0)
            
            # Store position data
            positions_data.append(episode_positions)
            
            logger.info(f"Episode {episode+1} completed: Reward={episode_reward:.4f}, Steps={step_count}, Trades={episode_trades}")
        
        # Calculate aggregate metrics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        std_reward = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        mean_trades = np.mean(trade_counts) if trade_counts else 0
        mean_gross_leverage = np.mean(gross_leverages) if gross_leverages else 0
        mean_net_leverage = np.mean(net_leverages) if net_leverages else 0
        
        sharpe_ratio = mean_reward / std_reward if std_reward > 0 else 0
        mean_drawdown = np.mean(drawdowns) if drawdowns else 0
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {model_path}")
        logger.info(f"{'='*50}")
        logger.info(f"Number of episodes: {episodes}")
        logger.info(f"Mean episode reward: {mean_reward:.6f}")
        logger.info(f"Reward standard deviation: {std_reward:.6f}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        logger.info(f"Mean episode length: {mean_length:.1f}")
        logger.info(f"Mean trading frequency: {mean_trades:.2f}")
        logger.info(f"Mean gross leverage: {mean_gross_leverage:.4f}x")
        logger.info(f"Mean net leverage: {mean_net_leverage:.4f}x")
        logger.info(f"Mean max drawdown: {mean_drawdown:.4%}")
        
        # Prepare results dataframe
        results = {
            'episode': list(range(1, episodes+1)),
            'reward': episode_rewards,
            'length': episode_lengths,
            'trades': trade_counts,
            'gross_leverage': gross_leverages,
            'net_leverage': net_leverages
        }
        
        if drawdowns:
            results['max_drawdown'] = drawdowns
        
        if portfolio_values:
            results['max_portfolio_value'] = portfolio_values
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        logger.info(f"Detailed results saved to {results_file}")
        
        # Log to wandb if requested
        if log_wandb:
            try:
                if wandb.run is None:
                    wandb.init(project="trading_evaluation", name=f"eval_{os.path.basename(model_path)}_{timestamp}")
                
                wandb.log({
                    "eval/phase": phase,
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/sharpe": sharpe_ratio,
                    "eval/mean_episode_length": mean_length,
                    "eval/mean_trades": mean_trades,
                    "eval/mean_gross_leverage": mean_gross_leverage,
                    "eval/mean_net_leverage": mean_net_leverage,
                    "eval/mean_max_drawdown": mean_drawdown
                })
                
                # Log reward distribution
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                ax.hist(episode_rewards, bins=min(10, episodes), alpha=0.7)
                ax.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2)
                ax.set_title("Reward Distribution")
                ax.set_xlabel("Episode Reward")
                ax.set_ylabel("Count")
                wandb.log({"eval/reward_distribution": wandb.Image(fig)})
                plt.close(fig)
                
                # Log episode rewards
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                ax.plot(range(1, episodes+1), episode_rewards, 'o-')
                ax.set_title("Episode Rewards")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.grid(True)
                wandb.log({"eval/episode_rewards": wandb.Image(fig)})
                plt.close(fig)
                
                # Log leverage charts
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                ax.plot(range(1, episodes+1), gross_leverages, 'o-', label='Gross Leverage')
                ax.plot(range(1, episodes+1), net_leverages, 'o-', label='Net Leverage')
                ax.set_title("Leverage by Episode")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Leverage")
                ax.grid(True)
                ax.legend()
                wandb.log({"eval/leverage": wandb.Image(fig)})
                plt.close(fig)
                
                # Upload the CSV
                wandb.save(results_file)
                
                logger.info(f"Results logged to Weights & Biases")
            except Exception as e:
                logger.error(f"Error logging to wandb: {str(e)}")
        
        # Return the aggregate metrics
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "sharpe_ratio": sharpe_ratio,
            "mean_episode_length": mean_length,
            "mean_trades": mean_trades,
            "mean_gross_leverage": mean_gross_leverage,
            "mean_net_leverage": mean_net_leverage,
            "mean_max_drawdown": mean_drawdown,
            "results_file": results_file
        }
    
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        env_path=args.env_path,
        episodes=args.episodes,
        deterministic=args.deterministic,
        log_wandb=args.log_wandb,
        phase=args.phase,
        output_dir=args.output_dir
    ) 