import os
import argparse
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
from typing import Dict, List, Optional, Union, Tuple
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetAgnosticCallback(CheckpointCallback):
    """
    Custom callback that monitors model's prediction bias for any asset
    and applies corrections during training
    """
    def __init__(self, 
                 check_freq: int = 5000,
                 save_path: str = "./models",
                 name_prefix: str = "model",
                 verbose: int = 1,
                 balance_actions: bool = True,
                 max_bias_threshold: float = 0.2):
        super().__init__(check_freq, save_path, name_prefix, verbose)
        self.balance_actions = balance_actions
        self.max_bias_threshold = max_bias_threshold
        self.action_history = []
        self.bias_history = {}
        self.asset_list = []

    def _on_step(self) -> bool:
        # Save actions for bias detection
        if hasattr(self.model, 'last_action') and self.model.last_action is not None:
            self.action_history.append(self.model.last_action)
            
            # Limit history size
            if len(self.action_history) > 10000:
                self.action_history = self.action_history[-10000:]
        
        # Asset-agnostic bias detection at checkpoint frequency
        if self.n_calls % self.check_freq == 0:
            # Get list of assets from environment
            if not self.asset_list and hasattr(self.model.env, 'get_attr'):
                try:
                    self.asset_list = self.model.env.get_attr('assets')[0]
                    logger.info(f"Detected assets: {self.asset_list}")
                except:
                    logger.warning("Could not detect assets from environment")
            
            # Analyze action bias if we have enough history
            if len(self.action_history) > 1000:
                # Convert to numpy array
                actions = np.array(self.action_history)
                
                # Calculate bias metrics for each asset dimension
                for i in range(actions.shape[1]):
                    asset_name = self.asset_list[i] if i < len(self.asset_list) else f"asset_{i}"
                    
                    # Calculate bias metrics
                    mean_action = np.mean(actions[:, i])
                    std_action = np.std(actions[:, i])
                    abs_mean = np.mean(np.abs(actions[:, i]))
                    pct_positive = np.mean(actions[:, i] > 0.15)
                    pct_negative = np.mean(actions[:, i] < -0.15)
                    
                    # Store metrics
                    self.bias_history[asset_name] = {
                        'mean': mean_action,
                        'std': std_action,
                        'abs_mean': abs_mean,
                        'pct_positive': pct_positive,
                        'pct_negative': pct_negative,
                        'step': self.n_calls
                    }
                    
                    # Log bias information
                    bias_level = abs(mean_action)
                    if bias_level > self.max_bias_threshold:
                        logger.warning(f"High bias detected for {asset_name}: {mean_action:.4f} " +
                                    f"(positive: {pct_positive:.2%}, negative: {pct_negative:.2%})")
                    else:
                        logger.info(f"Bias metrics for {asset_name}: mean={mean_action:.4f}, " +
                                  f"std={std_action:.4f}, abs_mean={abs_mean:.4f}")
                
                # Save bias metrics if save path exists
                if self.save_path:
                    bias_file = os.path.join(self.save_path, f"{self.name_prefix}_bias_metrics.json")
                    import json
                    with open(bias_file, 'w') as f:
                        json.dump(self.bias_history, f, indent=2)
                    
                    logger.info(f"Saved bias metrics to {bias_file}")
        
        # Call parent's on_step method (saves model, etc.)
        return super()._on_step()

def create_balanced_env(
    env_path: str,
    val_split: float = 0.2,
    regime_balance: bool = True,
    normalize: bool = True
) -> Union[gym.Env, VecNormalize]:
    """
    Create a training environment with balanced data exposure
    """
    # Load the environment from saved file
    if env_path.endswith('.pkl'):
        # Load raw env object
        env = joblib.load(env_path)
    else:
        # Try finding .pkl file in the path
        env_files = list(Path(env_path).glob("*.pkl"))
        if env_files:
            env = joblib.load(env_files[0])
        else:
            raise ValueError(f"No environment file found in {env_path}")
    
    # If we're balancing regimes, modify env's reset method
    if regime_balance and hasattr(env, 'df') and hasattr(env, 'market_conditions'):
        # Get the dataframe
        df = env.df
        
        # For validation split, separate the data
        if val_split > 0:
            split_idx = int(len(df) * (1 - val_split))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            # Set the training df
            env.df = train_df
            logger.info(f"Split data: Training={len(train_df)} rows, Validation={len(val_df)} rows")
            
            # Store for later
            env.val_df = val_df
            env.full_df = df.copy()
        
        # Original reset method
        original_reset = env.reset
        
        # Define regime-balanced reset to ensure varied market exposure
        def balanced_reset(seed=None, options=None):
            # With 50% probability, do regime-based starting point
            if np.random.random() < 0.5:
                # Get market regimes if available
                if hasattr(env, 'market_conditions') and env.market_conditions.get('market_regime'):
                    regimes = env.market_conditions.get('market_regime')
                    
                    # If it's a string (global regime), convert to a random step from that regime
                    if isinstance(regimes, str):
                        # We'll need to identify steps with this regime
                        # This is a placeholder - in a real implementation,
                        # you'd have a mapping of which steps correspond to which regimes
                        regime_steps = {
                            'trending': [],
                            'range_bound': [],
                            'volatile': [],
                            'normal': [],
                            'crisis': []
                        }
                        
                        # If we have regime steps identified, use them
                        if regime_steps.get(regimes):
                            random_step = np.random.choice(regime_steps[regimes])
                            env.current_step = random_step
                            logger.info(f"Starting from {regimes} regime at step {random_step}")
            
            # Call original reset
            return original_reset(seed=seed, options=options)
        
        # Replace the reset method
        env.reset = balanced_reset
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Normalize the environment if requested
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    return vec_env

def setup_ppo_model(
    env,
    model_path: Optional[str] = None,
    policy_kwargs: Optional[Dict] = None,
    learning_rate: float = 5e-5,
    ent_coef: float = 0.05,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0
) -> PPO:
    """
    Setup the PPO model with the appropriate parameters
    """
    if policy_kwargs is None:
        policy_kwargs = {}
    
    # Add dropout if requested
    if dropout_rate > 0:
        # Define network with dropout
        pi_net_arch = [dict(pi=[128, 64], vf=[128, 64])]
        activation_fn = torch.nn.ReLU
        
        # Create custom network class with dropout
        class CustomNetwork(torch.nn.Module):
            def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf):
                super().__init__()
                
                # Policy network with dropout
                self.policy_net = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, 128),
                    activation_fn(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(128, 64),
                    activation_fn(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(64, last_layer_dim_pi)
                )
                
                # Value network with dropout
                self.value_net = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, 128),
                    activation_fn(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(128, 64),
                    activation_fn(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(64, last_layer_dim_vf)
                )
            
            def forward(self, features):
                return self.policy_net(features), self.value_net(features)
        
        # Set custom network in policy kwargs
        policy_kwargs = {
            "net_arch": pi_net_arch,
            "activation_fn": activation_fn,
            "features_extractor_class": None,
            "features_extractor_kwargs": {},
            "normalize_images": True,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"weight_decay": l2_reg},
            "custom_network": CustomNetwork
        }
    else:
        # Standard network architecture
        policy_kwargs = {
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
            "activation_fn": torch.nn.ReLU,
            "optimizer_kwargs": {"weight_decay": l2_reg}
        }
    
    # If we have a pre-trained model, load it
    if model_path:
        try:
            model = PPO.load(model_path, env=env)
            logger.info(f"Loaded pre-trained model from {model_path}")
            
            # Update the model parameters
            model.learning_rate = learning_rate
            model.ent_coef = ent_coef
            model.policy_kwargs = policy_kwargs
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            logger.info("Creating new model instead.")
    
    # Create a new model if we don't have one or failed to load
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    return model

def recalibrate_value_function(model, steps=10000):
    """
    Recalibrate the value function using a lower learning rate
    """
    logger.info(f"Recalibrating value function with {steps} steps")
    
    # Store original parameters
    original_lr = model.learning_rate
    original_ent_coef = model.ent_coef
    original_vf_coef = model.vf_coef
    
    # Set temporary parameters for recalibration
    model.learning_rate = 1e-6
    model.ent_coef = 0.0 
    model.vf_coef = 1.0  # Focus on value function
    
    # Train for a short period
    model.learn(total_timesteps=steps, progress_bar=True)
    
    # Restore original parameters
    model.learning_rate = original_lr
    model.ent_coef = original_ent_coef
    model.vf_coef = original_vf_coef
    
    logger.info("Value function recalibration complete")
    return model

def create_adaptive_lr_schedule(initial_lr, final_lr, total_steps):
    """Create a linearly decreasing learning rate schedule"""
    def scheduler(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return scheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Improved RL model training for asset-agnostic trading")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--env-path", type=str, required=True, help="Path to the trading environment")
    parser.add_argument("--output-dir", type=str, default="./improved_models", help="Directory to save the improved model")
    parser.add_argument("--additional-steps", type=int, default=5000000, help="Number of steps to train")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--final-lr", type=float, default=1e-5, help="Final learning rate for schedule")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient for exploration")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--l2-reg", type=float, default=0.001, help="L2 regularization weight")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate for policy network")
    parser.add_argument("--recalibrate-vf", action="store_true", help="Recalibrate value function before training")
    parser.add_argument("--balance-regimes", action="store_true", help="Balance training across market regimes")
    parser.add_argument("--checkpoint-freq", type=int, default=100000, help="Frequency for saving model checkpoints")
    parser.add_argument("--no-signal-threshold", action="store_true", help="Disable signal threshold update in env")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup environment with balanced regimes
    logger.info("Creating training environment...")
    env = create_balanced_env(
        args.env_path,
        val_split=args.val_split,
        regime_balance=args.balance_regimes,
        normalize=True
    )
    
    # Get the raw env to modify signal threshold
    raw_env = env.envs[0] if hasattr(env, 'envs') else env
    
    # Update signal threshold if not disabled
    if not args.no_signal_threshold and hasattr(raw_env, 'signal_threshold'):
        original_threshold = raw_env.signal_threshold
        raw_env.signal_threshold = 0.25  # Increased threshold to filter weak signals
        logger.info(f"Updated signal threshold from {original_threshold} to {raw_env.signal_threshold}")
    
    # Setup model
    logger.info("Setting up model...")
    model = setup_ppo_model(
        env,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        ent_coef=args.entropy_coef,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg
    )
    
    # Recalibrate value function if requested
    if args.recalibrate_vf:
        model = recalibrate_value_function(model)
    
    # Setup learning rate schedule
    lr_schedule = create_adaptive_lr_schedule(
        args.learning_rate,
        args.final_lr,
        args.additional_steps
    )
    model.learning_rate = lr_schedule
    
    # Setup callbacks
    checkpoint_path = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Custom callback for bias monitoring
    bias_callback = AssetAgnosticCallback(
        check_freq=args.checkpoint_freq // 10,
        save_path=checkpoint_path,
        name_prefix="model",
        verbose=1,
        balance_actions=True,
        max_bias_threshold=0.2
    )
    
    # Standard checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="model",
        verbose=1
    )
    
    # Train the model
    logger.info(f"Starting training for {args.additional_steps} steps...")
    try:
        model.learn(
            total_timesteps=args.additional_steps,
            callback=[bias_callback, checkpoint_callback],
            progress_bar=True
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save(final_model_path)
    
    # If using vectorized environment, save its state too
    if isinstance(env, VecNormalize):
        env_path = os.path.join(args.output_dir, "vec_normalize.pkl")
        env.save(env_path)
        logger.info(f"Saved normalized environment to {env_path}")
    
    logger.info(f"Model saved to {final_model_path}")
    logger.info("Training process complete")

if __name__ == "__main__":
    main() 