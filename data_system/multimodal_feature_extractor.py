import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class MultiModalPerpFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for multimodal perpetual trading data"""
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        
        # Market data processing
        self.market_net = nn.Sequential(
            nn.Linear(n_input // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # News and sentiment processing
        self.sentiment_net = nn.Sequential(
            nn.Linear(n_input // 4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # On-chain data processing
        self.onchain_net = nn.Sequential(
            nn.Linear(n_input // 4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Combine all features
        self.combine_net = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observations):
        # Split input into different data types
        market_data = observations[:, :observations.shape[1]//2]
        sentiment_data = observations[:, observations.shape[1]//2:3*observations.shape[1]//4]
        onchain_data = observations[:, 3*observations.shape[1]//4:]
        
        # Process each data type
        market_features = self.market_net(market_data)
        sentiment_features = self.sentiment_net(sentiment_data)
        onchain_features = self.onchain_net(onchain_data)
        
        # Combine features
        combined = torch.cat([
            market_features,
            sentiment_features,
            onchain_features
        ], dim=1)
        
        return self.combine_net(combined) 