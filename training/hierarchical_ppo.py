import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Any
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from training.loggers import WandBLogger
from transformers import AutoModel

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return x

class MarketTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        input_dim: int = None  # Add input dimension parameter
    ):
        super().__init__()
        self.d_model = d_model
        if input_dim is None:
            input_dim = d_model
        self.input_projection = nn.Linear(input_dim, d_model)  # Project input to d_model dimensions
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, time_steps, num_assets, features)
        batch_size, time_steps, num_assets, features = x.shape
        
        # Reshape to (batch_size * num_assets, time_steps, features)
        x = x.transpose(1, 2).reshape(-1, time_steps, features)
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Reshape back to (batch_size, num_assets, d_model)
        x = x.reshape(batch_size, num_assets, time_steps, self.d_model)
        
        # Average over time dimension
        x = x.mean(dim=2)  # Now shape is (batch_size, num_assets, d_model)
        
        return x

class RiskLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)

class CustomActorCriticPolicy(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        action_space: Dict,
        net_arch: Dict,
        activation_fn: nn.Module = nn.ReLU,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        
        # Market feature processing
        market_input_dim = observation_space['market'].shape[-1]
        self.market_transformer = MarketTransformer(
            d_model=net_arch['transformer']['d_model'],
            nhead=net_arch['transformer']['nhead'],
            num_layers=net_arch['transformer']['num_layers'],
            dim_feedforward=net_arch['transformer']['dim_feedforward'],
            input_dim=market_input_dim
        )
        
        # News and social media processing
        self.text_encoder = AutoModel.from_pretrained('roberta-base')
        self.text_projection = nn.Linear(768, net_arch['text']['hidden_size'])
        self.text_attention = nn.MultiheadAttention(
            net_arch['text']['hidden_size'], 
            net_arch['text']['num_heads'],
            batch_first=True
        )
        
        # On-chain data processing
        self.onchain_lstm = nn.LSTM(
            input_size=observation_space['onchain'].shape[-1],
            hidden_size=net_arch['onchain']['hidden_size'],
            num_layers=net_arch['onchain']['num_layers'],
            batch_first=True
        )
        
        # Cross-asset attention for market sentiment
        self.cross_asset_attention = nn.MultiheadAttention(
            net_arch['transformer']['d_model'],
            net_arch['cross_asset']['num_heads'],
            batch_first=True
        )
        
        # Risk metrics processing
        self.risk_lstm = RiskLSTM(
            input_size=observation_space['risk'].shape[0],
            hidden_size=net_arch['lstm']['hidden_size'],
            num_layers=net_arch['lstm']['num_layers']
        )
        
        # Portfolio state processing
        self.portfolio_net = nn.Sequential(
            nn.Linear(observation_space['portfolio'].shape[0], net_arch['portfolio']['hidden_size']),
            activation_fn(),
            nn.Linear(net_arch['portfolio']['hidden_size'], net_arch['portfolio']['hidden_size']),
            activation_fn()
        )
        
        # Feature fusion with attention
        fusion_input_size = (
            net_arch['transformer']['d_model'] +  # Market features
            net_arch['text']['hidden_size'] +     # Text features
            net_arch['onchain']['hidden_size'] +  # On-chain features
            net_arch['lstm']['hidden_size'] +     # Risk features
            net_arch['portfolio']['hidden_size']  # Portfolio features
        )
        
        self.feature_fusion = nn.MultiheadAttention(
            fusion_input_size,
            net_arch['fusion']['num_heads'],
            batch_first=True
        )
        
        # Actor heads remain the same
        self.actor_heads = nn.ModuleDict({
            'trade_decision': self._create_actor_head(
                fusion_input_size,
                action_space['trade_decision'].shape[0],
                net_arch['actor']
            ),
            'direction': self._create_actor_head(
                fusion_input_size,
                action_space['direction'].shape[0],
                net_arch['actor']
            ),
            'leverage': self._create_actor_head(
                fusion_input_size,
                action_space['leverage'].shape[0],
                net_arch['actor']
            ),
            'risk_limits': self._create_actor_head(
                fusion_input_size,
                np.prod(action_space['risk_limits'].shape),
                net_arch['actor']
            ),
            'execution_params': self._create_actor_head(
                fusion_input_size,
                np.prod(action_space['execution_params'].shape),
                net_arch['actor']
            )
        })
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(fusion_input_size, net_arch['critic']['hidden_size']),
            activation_fn(),
            nn.Linear(net_arch['critic']['hidden_size'], net_arch['critic']['hidden_size']),
            activation_fn(),
            nn.Linear(net_arch['critic']['hidden_size'], 1)
        )
        
        # Action bounds
        self.register_buffer('action_means', torch.zeros(sum(space.shape[0] for space in action_space.values())))
        self.register_buffer('action_stds', torch.ones(sum(space.shape[0] for space in action_space.values())))
        
    def _create_actor_head(self, input_size: int, output_size: int, arch: Dict) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_size, arch['hidden_size']),
            nn.ReLU(),
            nn.Linear(arch['hidden_size'], arch['hidden_size']),
            nn.ReLU(),
            nn.Linear(arch['hidden_size'], output_size * 2)  # Mean and log_std
        )
        
    def forward(self, obs: Dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Process market data
        market_features = self.market_transformer(obs['market'])
        
        # Process text data (news & social media)
        text_features = self.text_encoder(obs['text'])[0]  # Get last hidden state
        text_features = self.text_projection(text_features)
        text_features, _ = self.text_attention(text_features, text_features, text_features)
        text_features = text_features.mean(dim=1)  # Pool text features
        
        # Process on-chain data
        onchain_features, _ = self.onchain_lstm(obs['onchain'])
        onchain_features = onchain_features[:, -1]  # Take last timestep
        
        # Process cross-asset relationships
        cross_asset_features, _ = self.cross_asset_attention(
            market_features,
            market_features,
            market_features
        )
        
        # Process risk and portfolio features
        risk_features = self.risk_lstm(obs['risk'].unsqueeze(1))
        risk_features = risk_features.squeeze(1)
        portfolio_features = self.portfolio_net(obs['portfolio'])
        
        # Combine all features with attention-based fusion
        combined_features = torch.cat([
            market_features.mean(dim=1),
            text_features,
            onchain_features,
            risk_features,
            portfolio_features
        ], dim=1).unsqueeze(1)
        
        fused_features, _ = self.feature_fusion(
            combined_features,
            combined_features,
            combined_features
        )
        fused_features = fused_features.squeeze(1)
        
        # Get action distributions
        action_dists = {}
        for name, head in self.actor_heads.items():
            out = head(fused_features)
            mean, log_std = torch.chunk(out, 2, dim=1)
            log_std = torch.clamp(log_std, -20, 2)
            action_dists[name] = Normal(mean, log_std.exp())
        
        # Get value estimate
        value = self.critic(fused_features)
        
        return action_dists, value
        
    def evaluate_actions(
        self,
        obs: Dict,
        actions: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        action_dists, values = self.forward(obs)
        
        # Calculate log probs and entropy
        entropy = 0
        log_prob = 0
        for name, dist in action_dists.items():
            log_prob += dist.log_prob(actions[name]).sum(dim=1)
            entropy += dist.entropy().sum(dim=1)
            
        return values, log_prob, entropy

class HierarchicalPPO:
    def __init__(
        self,
        env: VecEnv,
        device: torch.device,
        net_arch: Dict,
        policy_kwargs: dict = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        update_adv: bool = True
    ):
        if policy_kwargs is not None:
            pass
        self.env = env
        self.device = device
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_adv = update_adv
        
        # Initialize policy
        self.policy = CustomActorCriticPolicy(
            env.observation_space,
            env.action_space,
            net_arch,
            device=device
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Add to existing init
        self.logger = None
        if policy_kwargs and policy_kwargs.get('use_wandb'):
            self.logger = WandBLogger(
                config={
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'gamma': gamma,
                    'gae_lambda': gae_lambda,
                    'clip_range': clip_range,
                    'ent_coef': ent_coef,
                    'vf_coef': vf_coef,
                    'architecture': net_arch
                },
                project=policy_kwargs.get('wandb_project'),
                entity=policy_kwargs.get('wandb_entity')
            )
        
    def learn(
        self,
        total_timesteps: int,
        callback_configs: Dict = None,
        log_interval: int = 1
    ):
        """Training loop with curriculum learning support"""
        try:
            try:
                wandb.init(
                    project=callback_configs['wandb_project'],
                    config={
                        'learning_rate': self.learning_rate,
                        'n_steps': self.n_steps,
                        'batch_size': self.batch_size,
                        'n_epochs': self.n_epochs,
                        'gamma': self.gamma,
                        'gae_lambda': self.gae_lambda,
                        'clip_range': self.clip_range,
                        'ent_coef': self.ent_coef,
                        'vf_coef': self.vf_coef
                    }
                )
            except Exception as e:
                    logger.warning(f"Failed to initialize wandb: {str(e)}")
                    callback_configs['use_wandb'] = False
                
            n_updates = total_timesteps // self.n_steps
            logger.info(f"Starting training for {total_timesteps} timesteps ({n_updates} updates)")
            
            for update in range(n_updates):
                try:
                    # Collect rollouts
                    logger.info(f"Collecting rollouts for update {update}/{n_updates}")
                    rollout_data = self._collect_rollouts()
                    
                    # Update policy
                    logger.info(f"Updating policy for update {update}/{n_updates}")
                    metrics = self._update_policy(rollout_data)
                    
                    # Log metrics
                    if update % log_interval == 0:
                        for key, value in metrics.items():
                            logger.info(f"{key}: {value:.4f}")
                            try:
                                wandb.log({key: value}, step=update)
                            except Exception as e:
                                logger.warning(f"Failed to log to wandb: {str(e)}")
                                    
                except Exception as e:
                    logger.error(f"Error during update {update}: {str(e)}")
                    logger.error(f"Rollout data shapes: {[(k, v.shape) for k, v in rollout_data.items() if hasattr(v, 'shape')]}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error in learn: {str(e)}")
            raise
        
    def _collect_rollouts(self) -> Dict:
        """Collect rollouts using current policy"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        # Reset environment and get initial observation
        obs = self.env.reset()  # VecEnv reset returns just the observation
        
        for _ in range(self.n_steps):
            with torch.no_grad():
                # Convert observation to tensor
                obs_tensor = self._to_tensor(obs)
                
                # Get action distributions and value
                action_dists, value = self.policy(obs_tensor)
                
                # Sample actions and convert to numpy arrays
                actions_dict = {}
                log_prob = 0
                for name, dist in action_dists.items():
                    action = dist.sample()
                    # Convert to numpy and ensure correct shape for vectorized env
                    actions_dict[name] = action.detach().cpu().numpy()
                    log_prob += dist.log_prob(action).sum(dim=1)
                
            # Execute action in vectorized environment
            next_obs, reward, done, info = self.env.step(actions_dict)
            
            # Store data
            observations.append(obs)
            actions.append({k: v.copy() for k, v in actions_dict.items()})  # Deep copy to prevent reference issues
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            
            obs = next_obs
            
        # Calculate advantages using GAE
        advantages = self._compute_gae(
            np.array(rewards),
            np.array(values),
            np.array(dones)
        )
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'advantages': advantages
        }
        
    def _update_policy(self, rollout_data: Dict) -> Dict:
        """Update policy using PPO"""
        # Convert data to tensors
        observations = self._to_tensor(rollout_data['observations'])
        old_values = torch.FloatTensor(rollout_data['values']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch updates
        metrics = []
        for epoch in range(self.n_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(self.n_steps)
            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_obs = {
                    key: value[batch_indices]
                    for key, value in observations.items()
                }
                batch_actions = {
                    key: torch.FloatTensor(
                        np.array([rollout_data['actions'][i][key] for i in batch_indices])
                    ).to(self.device)
                    for key in rollout_data['actions'][0].keys()
                }
                
                # Forward pass
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch_obs,
                    batch_actions
                )
                
                # Calculate losses
                ratio = torch.exp(log_probs - old_log_probs[batch_indices])
                
                # Policy loss
                policy_loss1 = -advantages[batch_indices] * ratio
                policy_loss2 = -advantages[batch_indices] * torch.clamp(
                    ratio,
                    1 - self.clip_range,
                    1 + self.clip_range
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values[batch_indices] + torch.clamp(
                        values - old_values[batch_indices],
                        -self.clip_range_vf,
                        self.clip_range_vf
                    )
                value_loss = F.mse_loss(values_pred, returns[batch_indices])
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.vf_coef * value_loss +
                    self.ent_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                metrics.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'total_loss': loss.item(),
                    'approx_kl': (old_log_probs[batch_indices] - log_probs).mean().item()
                })
                
                # Early stopping if KL divergence is too high
                if metrics[-1]['approx_kl'] > 1.5 * self.target_kl:
                    break
                    
        # Average metrics
        metrics = {
            key: np.mean([m[key] for m in metrics])
            for key in metrics[0].keys()
        }
        
        if self.logger:
            self.logger.log_training_step(metrics, self.num_timesteps)
            self.logger.log_model_gradients(self.policy)
        
        return metrics
        
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        return advantages
        
    def _to_tensor(self, data: Dict) -> Dict:
        """Convert numpy arrays to PyTorch tensors"""
        if isinstance(data, dict):
            return {
                key: self._to_tensor(value)
                for key, value in data.items()
            }
        elif isinstance(data, np.ndarray):
            return torch.FloatTensor(data).to(self.device)
        elif isinstance(data, list):
            return [self._to_tensor(item) for item in data]
        return data
        
    def _to_numpy(self, data: Dict) -> Dict:
        """Convert PyTorch tensors to numpy arrays"""
        if isinstance(data, dict):
            return {
                key: self._to_numpy(value)
                for key, value in data.items()
            }
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        return data
        
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def predict(self, obs: Dict, deterministic: bool = False) -> Dict:
        """Predict action for a given observation"""
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = self._to_tensor(obs)
            
            # Get action distributions
            action_dists, _ = self.policy(obs_tensor)
            
            # Sample actions
            actions = {}
            for name, dist in action_dists.items():
                if deterministic:
                    actions[name] = dist.mean.cpu().numpy()
                else:
                    actions[name] = dist.sample().cpu().numpy()
            
            return actions