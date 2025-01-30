import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from trading_env import CryptoTradingEnv
import pandas as pd

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize with Kaiming Normal
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        nn.init.constant_(self.bias_mu, 0.1)  # Small positive bias
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
        bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        return torch.nn.functional.linear(x, weight, bias)

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.noisy_net = nn.Sequential(
            NoisyLinear(n_input, 64),
            nn.ReLU(),
            NoisyLinear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.noisy_net(observations)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/btc_merged.csv')
    
    # Initialize environment
    env = CryptoTradingEnv(df)
    
    # Policy with noisy networks
    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=128)
    )
    
    # Train PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=1e-5,  # Learning rate annealing
        n_steps=2048,           # Batch size (collect 2048 steps per update)
        batch_size=64,          # Minibatch size (balance speed/accuracy)
        gamma=0.99,             # Discount factor (prioritize long-term rewards)
        ent_coef=0.01,          # Encourage exploration
        max_grad_norm=0.5,      # Prevent exploding gradients
        clip_range=0.2,         # PPO clipping (stability)
        n_epochs=15
    )
    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_btc_base")
    print("Training complete. Model saved to models/ppo_btc_base")