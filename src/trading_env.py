import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

class CryptoTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        self.df = df.dropna().reset_index(drop=True)
        self.features = self.df.columns.difference(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.n_features = len(self.features)
        
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,))
        
        # Initialize with reset() instead of setting values directly
        self.initial_balance = initial_balance
        self.reset()

    def _get_obs(self):
        return self.df.iloc[self.current_step][self.features].values.astype(np.float32)

    def _calculate_reward(self):
        portfolio_value = self.balance + self.btc_held * self.current_price
        daily_return = (portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value + 1e-9)
        self.returns.append(daily_return)
        
        # Sharpe ratio (30-day rolling)
        excess_returns = np.array(self.returns[-30:]) - 0.0  # Risk-free rate = 0
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)
        
        # Penalties
        penalty = 0.0
        # Transaction cost (0.1%)
        if self.last_action != 0:
            penalty += 0.001 * portfolio_value
        # Drawdown penalty (>5% daily loss)
        if daily_return < -0.05:
            penalty += abs(daily_return) * 2
        
        reward = sharpe - penalty
        self.last_portfolio_value = portfolio_value
        return reward

    def step(self, action):
        # Prevent out-of-bounds check first
        if self.current_step >= len(self.df) - 1:
            terminated = True
            return self._get_obs(), 0.0, terminated, False, {}
            
        self.current_step += 1
        self.current_price = self.df.iloc[self.current_step]['close']
        self.last_action = action
        
        # Calculate portfolio value before trade
        prev_portfolio = self.balance + self.btc_held * self.current_price
        
        # Apply slippage (increased to 0.2% for market orders)
        slippage = 0.002
        executed_price = self.current_price * (1 + slippage) if action == 1 else self.current_price * (1 - slippage)
        
        # Execute valid actions with checks
        if action == 1 and self.balance > 0:  # Buy
            self.btc_held = self.balance / executed_price
            self.balance = 0.0
        elif action == 2 and self.btc_held > 0:  # Sell
            self.balance = self.btc_held * executed_price
            self.btc_held = 0.0
        
        # Simplified reward calculation
        new_portfolio = self.balance + self.btc_held * self.current_price
        fee = 0.001 * new_portfolio  # 0.1% transaction fee
        reward = (new_portfolio - prev_portfolio - fee) / prev_portfolio if prev_portfolio != 0 else 0
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.last_action = 0
        self.returns = []
        self.last_portfolio_value = self.initial_balance
        return self._get_obs(), {}

    def render(self, mode='human'):
        portfolio = self.balance + self.btc_held * self.current_price
        print(
            f"Step: {self.current_step}/{len(self.df)}, "
            f"Balance: ${self.balance:.2f}, "
            f"BTC: {self.btc_held:.6f}, "
            f"Value: ${portfolio:.2f}"
        )