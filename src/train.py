import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from trading_env import MultiCryptoEnv

class ICMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(2*features_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_input)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(features_dim + n_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations):
        return self.encoder(observations)

    def curiosity_loss(self, states, next_states, actions):
        phi = self.encoder(states)
        phi_next = self.encoder(next_states)
        
        # Inverse loss
        pred_actions = self.inverse_model(torch.cat([phi, phi_next], dim=1))
        inverse_loss = nn.MSELoss()(pred_actions, actions)
        
        # Forward loss
        pred_phi_next = self.forward_model(torch.cat([phi, actions], dim=1))
        forward_loss = nn.MSELoss()(pred_phi_next, phi_next)
        
        return 0.8*forward_loss + 0.2*inverse_loss

class CuriosityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.curiosity_losses = []

    def _on_step(self):
        # Get the current buffer size
        buffer_size = len(self.locals['rollout_buffer'].observations)
        if buffer_size <= 1:
            return True
            
        # Collect data for curiosity calculation
        states = self.locals['rollout_buffer'].observations[:buffer_size-1]  # Current states
        actions = self.locals['rollout_buffer'].actions[:buffer_size-1]  # Align with states
        next_states = self.locals['rollout_buffer'].observations[1:buffer_size]  # Next states
        
        # Calculate and apply curiosity reward
        with torch.no_grad():
            phi = self.model.policy.features_extractor(torch.FloatTensor(states))
            phi_next = self.model.policy.features_extractor(torch.FloatTensor(next_states))
            curiosity = torch.norm(phi_next - phi, dim=1).cpu().numpy()
            
            # Pad the last step with zero curiosity
            curiosity = np.append(curiosity, 0)
            curiosity = curiosity.reshape(-1, 1)  # Reshape to match rewards shape
            
            # Ensure curiosity matches the buffer size
            if len(curiosity) > buffer_size:
                curiosity = curiosity[:buffer_size]
        
        # Add curiosity reward to the existing rewards
        self.locals['rollout_buffer'].rewards += 0.01 * curiosity  # Reduced curiosity weight

        return True

def train_model():
    # Load data and add error checking
    df = pd.read_parquet('data/multi_crypto.parquet')
    if df.empty:
        raise ValueError("DataFrame is empty. Please ensure data/multi_crypto.parquet contains valid data")
    
    print(f'DataFrame shape: {df.shape}')
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: MultiCryptoEnv(df)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    policy_kwargs = dict(
        features_extractor_class=ICMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(
            pi=[256, 128],  # Policy network
            vf=[256, 128]   # Value network
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-5,        # Much lower learning rate
        n_steps=1024,              # Shorter rollout length
        batch_size=32,             # Even smaller batch size
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,           # Much lower entropy
        clip_range=0.05,          # Very conservative clipping
        clip_range_vf=0.05,       # Match policy clipping
        max_grad_norm=0.2,        # Even lower grad norm
        n_epochs=3,               # Fewer epochs
        use_sde=False,            # Disable SDE for now
        vf_coef=1.0,             # Stronger value function training
        target_kl=0.025,          # Even more lenient KL target
        tensorboard_log="logs/",
    )
    
    # Train with curiosity
    model.learn(
        total_timesteps=2_000_000,
        callback=CuriosityCallback(),
        tb_log_name="multi_crypto_icm"
    )
    
    # Save both the model and the normalized environment
    model.save("models/multi_crypto_icm")
    env.save("models/vec_normalize.pkl")
    
if __name__ == "__main__":
    train_model()