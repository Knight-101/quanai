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
        # 1. Returns Calculation with Smoothing
        current_value = self._portfolio_value()
        prev_value = self.portfolio_history[-2]['value'] if len(self.portfolio_history)>=2 else current_value
        raw_return = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Apply Huber loss to returns for robustness
        portfolio_return = np.sign(raw_return) * np.sqrt(np.abs(raw_return))  # Compress extreme values
        
        # 2. Adaptive Risk-Adjusted Return Component
        WINDOW = 90  # Align with typical quarterly rebalancing
        
        # Use Conditional Sharpe Ratio (favors positive skew)
        returns_window = np.array(self.portfolio_returns[-WINDOW:])
        positive_returns = returns_window[returns_window > 0]
        downside_returns = returns_window[returns_window <= 0]
        
        # Modified denominator with volatility floor
        denominator = np.std(downside_returns) if len(downside_returns) > 5 else 0.01  # 1% floor
        risk_adjusted_return = portfolio_return / (denominator + 1e-8)
        
        # 3. Penalties with Adaptive Scaling
        # Concentration (Herfindahl Index)
        allocations = np.array([self.allocations[asset] for asset in self.assets])
        herfindahl = np.sum(allocations**2)
        concentration_penalty = 0.2 * herfindahl  # [0-0.2] penalty
        
        # Drawdown (Current Peak-to-Trough)
        peak = np.max([pv['value'] for pv in self.portfolio_history[-WINDOW:]])
        current_drawdown = (peak - current_value) / (peak + 1e-8)
        drawdown_penalty = 0.5 * current_drawdown  # Linear penalty
        
        # Conditional VaR (Beyond 95% quantile)
        var = self._calculate_var(alpha=0.95)
        var_penalty = 0.1 * max(0, var - 0.05)  # Only penalize CVaR >5%
        
        # 4. Composite Reward with Normalization
        reward = (risk_adjusted_return * 0.7  # Primary driver
                  - concentration_penalty 
                  - drawdown_penalty 
                  - var_penalty)
        
        # 5. Dynamic Clipping and Scaling
        reward = np.clip(reward, -2.0, 2.0)  # Prevent exploding gradients
        return float(reward)
    
    def _portfolio_value(self):
        return self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
    )

    def _calculate_var(self, alpha=0.95, window=90):
        """Conditional Value at Risk with lookback window"""
        if len(self.portfolio_history) < window:
            return 0.0
        
        values = np.array([pv['value'] for pv in self.portfolio_history[-window:]])
        returns = np.diff(values) / (values[:-1] + 1e-8)
        if len(returns) == 0:
            return 0.0
        
        # Calculate CVaR
        var = np.percentile(returns, 100*(1-alpha))
        cvar = returns[returns <= var].mean()
        return abs(cvar) if not np.isnan(cvar) else 0.0

    def _get_current_price(self, asset):
        return self.df[asset]['close'].iloc[self.current_step]

    def render(self, mode='human'):
        current_value = self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        print(f"Step {self.current_step} | Value: ${current_value:.2f} | Trades: {self.trade_count}")

class CurriculumTradingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.initial_volatility = 0.1
        self.max_volatility = 1.0
        self.volatility_increment = 0.1
        self.success_threshold = 0.05  # 5% returns threshold
        self.episode_returns = []
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if len(self.episode_returns) > 0:
            # Adjust difficulty based on performance
            avg_return = np.mean(self.episode_returns[-5:])  # Last 5 episodes
            if avg_return > self.success_threshold:
                self.initial_volatility = min(
                    self.initial_volatility + self.volatility_increment,
                    self.max_volatility
                )
            self.episode_returns = []
        return obs
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Apply volatility scaling to market data only
        n_market_features = self.env.n_features
        market_data = obs[:n_market_features]
        portfolio_state = obs[n_market_features:]
        
        # Scale market volatility while preserving relationships
        scaled_market_data = market_data * (1 + self.initial_volatility)
        
        # Combine scaled market data with unchanged portfolio state
        obs = np.concatenate([scaled_market_data, portfolio_state])
        
        if done:
            final_value = self.env.portfolio_history[-1]['value']
            initial_value = self.env.initial_balance
            episode_return = (final_value - initial_value) / initial_value
            self.episode_returns.append(episode_return)
            
        return obs, reward, done, truncated, info