import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

class MultiCryptoEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, initial_balance=10000.0, risk_free_rate=0.0):
        super().__init__()
        self.df = df
        self.assets = list(df.columns.get_level_values(0).unique())
        self.features = self._get_feature_list()
        self.n_features = len(self.features)
        
        # Action space: [0-1 allocations for each asset + cash]
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.assets)+1,))
        
        # Observation space: All features + current allocations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + len(self.assets)+1,)
        )
        
        # Financial parameters
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.reset()

    def _get_feature_list(self):
        return [f"{asset}/{feat}" for asset in self.assets 
                for feat in self.df[asset].columns if feat != 'timestamp']

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.allocations = {asset: 0.0 for asset in self.assets}
        self.portfolio_history = []
        self.trade_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Market features
        market_data = self.df.iloc[self.current_step].values.astype(np.float32)
        # Portfolio features (allocations + cash)
        portfolio_state = [self.balance/self.initial_balance] + [
            self.allocations[asset] for asset in self.assets
        ]
        return np.concatenate([market_data, portfolio_state])

    def step(self, action):
        # Execute trades with market impact
        self._rebalance_portfolio(action)
        
        # Move to next timestep
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        # Calculate rewards
        reward = self._calculate_reward()
        
        return self._get_obs(), reward, terminated, False, {}

    def _rebalance_portfolio(self, action):
        total_value = self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        
        # Apply slippage model
        slippage = 0.0005 * np.sqrt(self.trade_count + 1)  # Increasing slippage
        
        # Convert actions to target allocations
        target_alloc = action / (action.sum() + 1e-9)
        
        # Execute rebalancing
        for i, asset in enumerate(self.assets):
            target_value = total_value * target_alloc[i+1]
            current_value = self.allocations[asset] * self._get_current_price(asset)
            delta = (target_value - current_value) * (1 - slippage)
            
            if delta > 0:  # Buy
                self.balance -= delta
                self.allocations[asset] += delta / self._get_current_price(asset)
            else:  # Sell
                self.allocations[asset] += delta / self._get_current_price(asset)
                self.balance -= delta * (1 - slippage)
        
        self.trade_count += 1

    def _calculate_reward(self):
        # Current portfolio value
        current_value = self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        
        # Store portfolio value history
        self.portfolio_history.append({'value': current_value})
        
        # Calculate returns-based reward if we have enough history
        if len(self.portfolio_history) > 1:
            returns = np.diff([pv['value'] for pv in self.portfolio_history[-2:]])
            sharpe = returns[0] / (np.std(returns) + 1e-9)
            
            # Risk penalty
            var = self._calculate_var()
            
            return sharpe - 0.3*var
        
        return 0.0  # No reward for first step

    def _calculate_var(self, alpha=0.95):
        # Calculate Value-at-Risk
        returns = np.diff([pv['value'] for pv in self.portfolio_history])
        return np.percentile(returns, 100*(1-alpha))

    def _get_current_price(self, asset):
        return self.df[asset]['close'].iloc[self.current_step]

    def render(self, mode='human'):
        current_value = self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        print(f"Step {self.current_step} | Value: ${current_value:.2f} | Trades: {self.trade_count}")