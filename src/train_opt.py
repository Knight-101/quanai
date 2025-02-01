import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn as linear_schedule
import optuna  # Import Optuna for hyperparameter optimization

from trading_env import MultiCryptoEnv, CurriculumTradingWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

class ICMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 256),  # Reduced from 512
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(256, 128),  # Reduced from 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()  # Added bounded activation
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(2*features_dim, 128),  # Reduced from 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(128, 64),  # Reduced from 128
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_input)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(features_dim + n_input, 128),  # Reduced from 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(128, 64),  # Reduced from 128
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, observations):
        # Ensure input is on the correct device and has correct dtype
        observations = observations.to(device=device, dtype=torch.float32)
        # Clip input values to prevent extreme values
        observations = torch.clamp(observations, -10.0, 10.0)
        return self.encoder(observations)

    def curiosity_loss(self, states, next_states, actions):
        # Ensure inputs are on the correct device and have correct dtype
        states = states.to(device=device, dtype=torch.float32)
        next_states = next_states.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)
        
        # Clip input values
        states = torch.clamp(states, -10.0, 10.0)
        next_states = torch.clamp(next_states, -10.0, 10.0)
        actions = torch.clamp(actions, -10.0, 10.0)
        
        phi = self.encoder(states)
        phi_next = self.encoder(next_states)
        
        # Inverse loss
        pred_actions = self.inverse_model(torch.cat([phi, phi_next], dim=1))
        inverse_loss = nn.MSELoss()(pred_actions, actions)
        
        # Forward loss
        pred_phi_next = self.forward_model(torch.cat([phi, actions], dim=1))
        forward_loss = nn.MSELoss()(pred_phi_next, phi_next)
        
        # Clip losses to prevent extreme values
        inverse_loss = torch.clamp(inverse_loss, 0.0, 10.0)
        forward_loss = torch.clamp(forward_loss, 0.0, 10.0)
        
        return 0.5*forward_loss + 0.5*inverse_loss  # More balanced loss weights

class CuriosityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.curiosity_losses = []
        self.curiosity_scale = 0.005  # Reduced from 0.01
        self.max_curiosity = 1.0  # Added max curiosity value

    def _on_step(self):
        # Get the current buffer size
        buffer_size = len(self.locals['rollout_buffer'].observations)
        if buffer_size <= 1:
            return True
            
        # Collect data for curiosity calculation
        states = self.locals['rollout_buffer'].observations[:buffer_size-1]
        actions = self.locals['rollout_buffer'].actions[:buffer_size-1]
        next_states = self.locals['rollout_buffer'].observations[1:buffer_size]
        
        # Calculate and apply curiosity reward
        with torch.no_grad():
            try:
                # Convert to float32 and move to device
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                
                # Clip inputs
                states = torch.clamp(states, -10.0, 10.0)
                next_states = torch.clamp(next_states, -10.0, 10.0)
                
                # Get feature representations
                phi = self.model.policy.features_extractor(states)
                phi_next = self.model.policy.features_extractor(next_states)
                
                # Calculate curiosity (L2 norm of feature differences)
                curiosity = torch.norm(phi_next - phi, dim=1, p=2)
                
                # Clip curiosity values
                curiosity = torch.clamp(curiosity, 0.0, self.max_curiosity)
                
                # Convert to numpy and reshape
                curiosity = curiosity.cpu().numpy()
                
                # Normalize curiosity to [0, 1] range
                if len(curiosity) > 0:
                    curiosity = (curiosity - curiosity.min()) / (curiosity.max() - curiosity.min() + 1e-8)
                
                # Pad the last step with zero curiosity
                curiosity = np.append(curiosity, 0)
                curiosity = curiosity.reshape(-1, 1)
                
                # Ensure curiosity matches the buffer size
                if len(curiosity) > buffer_size:
                    curiosity = curiosity[:buffer_size]
                
                # Add scaled curiosity reward
                self.locals['rollout_buffer'].rewards += self.curiosity_scale * curiosity
                
            except Exception as e:
                print(f"Warning: Error in curiosity calculation: {e}")
                # Continue training even if curiosity calculation fails
                pass

        return True

def objective(trial):
    # Dynamically sample hyperparameters using Optuna
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.92, 0.95])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.85, 0.9, 0.95])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7])
    vf_coef = trial.suggest_uniform("vf_coef", 0.3, 0.7)

    # Load data and add error checking
    df = pd.read_parquet('data/multi_crypto.parquet')
    if df.empty:
        raise ValueError("DataFrame is empty. Please ensure data/multi_crypto.parquet contains valid data")
    
    print(f'DataFrame shape: {df.shape}')
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: CurriculumTradingWrapper(MultiCryptoEnv(df))])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,  # Reduced from 10.0
        clip_reward=5.0,  # Reduced from 10.0
        gamma=gamma,
        epsilon=1e-8  # Added for numerical stability
    )
    
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[256, 256], vf=[256, 256])
        ],
        activation_fn=nn.Tanh,
        ortho_init=True,
        log_std_init=0.5,  # Reduced initial exploration
        features_extractor_class=ICMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        tensorboard_log="logs/",
        seed=42,
        device="auto"
    )
    
    # Train with curiosity
    model.learn(
        total_timesteps=1_000_000,
        callback=CuriosityCallback(),
        tb_log_name="multi_crypto_icm"
    )
    
    # # Evaluate the model (replace with your own evaluation metric)
    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    
    # Save both the model and the normalized environment
    model.save("models/multi_crypto_icm")
    env.save("models/vec_normalize.pkl")
    
    # return mean_reward  # Return the mean reward for optimization

def train_model():
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Run 50 trials for hyperparameter optimization

    # Print the best hyperparameters and the corresponding reward
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    train_model()