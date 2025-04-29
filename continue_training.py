#!/usr/bin/env python
# Script to continue training the model with bias correction

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Continue Training with Bias Correction')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the saved model')
    parser.add_argument('--env-path', type=str, required=True,
                        help='Path to the saved environment')
    parser.add_argument('--additional-steps', type=int, default=5_000_000,
                        help='Number of additional steps to train (default: 5M)')
    parser.add_argument('--output-dir', type=str, default='models/improved',
                        help='Directory to save improved model')
    parser.add_argument('--entropy-coef', type=float, default=0.05,
                        help='Entropy coefficient (default: 0.05)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    return parser.parse_args()

def linear_schedule(initial_value, final_value=0, progress_remaining=1.0):
    """Linear learning rate schedule."""
    return initial_value + (final_value - initial_value) * (1 - progress_remaining)

def recalibrate_value_function(model, env, num_steps=10000):
    """Recalibrate the value function by running the model for num_steps"""
    logger.info(f"Recalibrating value function for {num_steps} steps")
    
    # Set a very low learning rate for the policy network to freeze it
    # while letting the value function adapt
    original_lr = model.learning_rate
    model.learning_rate = 1e-6
    original_ent_coef = model.ent_coef
    model.ent_coef = 0.0
    
    # Adjust parameters to focus on value function
    original_vf_coef = model.vf_coef
    model.vf_coef = 1.0  # Increase value function coefficient
    
    # Train for a short period to recalibrate
    model.learn(total_timesteps=num_steps, reset_num_timesteps=False)
    
    # Restore original parameters
    model.learning_rate = original_lr
    model.ent_coef = original_ent_coef
    model.vf_coef = original_vf_coef
    
    logger.info("Value function recalibration complete")
    return model

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
    
    # Update the model with the environment
    model.set_env(env)
    
    # Get the current parameters
    old_ent_coef = model.ent_coef
    old_learning_rate = model.learning_rate
    
    # Recalibrate the model's value function
    logger.info("Recalibrating value function...")
    model = recalibrate_value_function(model, env)
    
    # Update training parameters
    logger.info("Updating model parameters for continued training")
    logger.info(f"Old entropy coefficient: {old_ent_coef}, New: {args.entropy_coef}")
    logger.info(f"Old learning rate: {old_learning_rate}, New: {args.learning_rate}")
    
    # Create a custom learning rate scheduler
    def custom_lr_schedule(progress_remaining):
        # Start with the specified learning rate and decay to 10% of it
        return linear_schedule(args.learning_rate, args.learning_rate * 0.1, progress_remaining)
    
    # Update model parameters for continued training
    model.ent_coef = args.entropy_coef  # Increase entropy to encourage exploration
    model.learning_rate = custom_lr_schedule  # Use learning rate schedule
    model.batch_size = args.batch_size  # Adjust batch size
    
    # Reset reward normalization to adapt to the new dynamics
    # but keep other stats like observation normalization
    env.norm_reward = False  # Temporarily disable
    env.ret_rms.reset()  # Reset reward normalization
    env.norm_reward = True  # Re-enable
    
    # Configure environment with better parameters
    if hasattr(env, "env_method"):
        # Reset parameters to reduce bias
        env.env_method("set_regime_aware", True)
        env.env_method("set_position_holding_bonus", 0.03)  # Reduced from previous value
        env.env_method("set_uncertainty_scaling", 1.5)  # Increased to be more conservative
    
    # Continue training with the updated parameters
    logger.info(f"Continuing training for {args.additional_steps} steps...")
    model.learn(
        total_timesteps=args.additional_steps,
        reset_num_timesteps=False,  # Continue from current timestep count
        progress_bar=True
    )
    
    # Save the improved model
    improved_model_path = os.path.join(args.output_dir, "improved_model")
    improved_env_path = os.path.join(args.output_dir, "improved_env.pkl")
    
    logger.info(f"Saving improved model to {improved_model_path}")
    model.save(improved_model_path)
    env.save(improved_env_path)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 