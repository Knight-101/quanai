import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Price data processing
        self.price_net = nn.Sequential(
            nn.Linear(observation_space.shape[1] // 2, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # Multimodal data processing
        self.multimodal_net = nn.Sequential(
            nn.Linear(observation_space.shape[1] // 2, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(features_dim)
        )
        
    def forward(self, observations):
        # Split observation into price and multimodal components
        n_features = observations.shape[1]
        price_data = observations[:, :n_features//2]
        multimodal_data = observations[:, n_features//2:]
        
        # Process each component
        price_features = self.price_net(price_data)
        multimodal_features = self.multimodal_net(multimodal_data)
        
        # Combine features
        combined = torch.cat([price_features, multimodal_features], dim=1)
        return self.combined_net(combined) 