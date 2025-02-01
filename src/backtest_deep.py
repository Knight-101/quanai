import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, Optional

class BacktestEngine:
    """Industry-grade backtesting system with advanced metrics and risk management"""
    
    def __init__(
        self,
        model_path: str,
        env_stats_path: str,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.0005,  # 5bps per trade
        slippage: float = 0.0001,    # 10bps slippage
        risk_free_rate: float = 0.0
    ):
        # Financial parameters
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # Load model artifact
        self.model = PPO.load(model_path, device='auto')
        self.assets = list(data.columns.get_level_values(0).unique())
        self.features = self._get_feature_list(data)
        
        # Data preparation
        self.data = self._preprocess_data(data)
        self.n_steps = len(self.data)
        
        # State tracking
        self.current_step = 0
        self.balance = initial_balance
        self.allocations = {asset: 0.0 for asset in self.assets}
        self.trades = []
        self.portfolio_history = []
        self._init_metrics()
        
        # --- Set up dummy environment for VecNormalize ---
        # Instead of calling _get_observation (which uses self.env_stats),
        # compute the observation dimension manually.
        #
        # The raw observation comes from a row of the DataFrame,
        # and then we add the portfolio state dimensions:
        #   - normalized balance (1)
        #   - one allocation per asset (len(self.assets))
        raw_obs_dim = self.data.iloc[self.current_step].values.shape[0]
        portfolio_state_dim = 1 + len(self.assets)
        obs_dim = raw_obs_dim + portfolio_state_dim
        action_dim = len(self.assets) + 1  # one extra for the portfolio state
        
        def create_dummy_env():
            class DummyEnv(gym.Env):
                def __init__(self):
                    super(DummyEnv, self).__init__()
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
                    self.num_envs = 1  # Required attribute for VecNormalize

                def reset(self):
                    return np.zeros((obs_dim,), dtype=np.float32)

                def step(self, action):
                    return np.zeros((obs_dim,), dtype=np.float32), 0.0, True, {}
            return DummyEnv()
        
        dummy_env = DummyVecEnv([lambda: create_dummy_env()])
        
        # Load the environment normalization using the dummy environment.
        self.env_stats = VecNormalize.load(env_stats_path, venv=dummy_env)
        # Set to evaluation mode.
        self.env_stats.training = False
        self.env_stats.norm_reward = False

    def _init_metrics(self):
        """Initialize performance tracking structures"""
        self.metrics = {
            'returns': [],
            'drawdown': [],
            'volatility': [],
            'sharpe': [],
            'sortino': [],
            'max_drawdown': 0.0,
            'calmar': 0.0,
            'value_at_risk': [],
            'conditional_var': [],
            'allocation_history': [],
            'trade_count': 0,
            'win_rate': 0.0
        }

    def _get_feature_list(self, data: pd.DataFrame) -> list:
        """Recreate feature list matching training setup"""
        return [f"{asset}/{feat}" for asset in self.assets 
                for feat in data[asset].columns if feat != 'timestamp']

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data alignment and forward filling"""
        processed = data.copy()
        for asset in self.assets:
            processed[asset] = processed[asset].ffill().bfill()
            processed[asset] = processed[asset].interpolate(method='linear')
        return processed.dropna()

    def _get_observation(self) -> np.ndarray:
        """Get normalized observation matching training conditions"""
        raw_obs = self.data.iloc[self.current_step].values.astype(np.float32)
        portfolio_state = [self.balance / self.initial_balance] + [
            self.allocations[asset] for asset in self.assets
        ]
        full_obs = np.concatenate([raw_obs, portfolio_state])
        
        if self.env_stats is not None:
            norm_obs = self.env_stats.normalize_obs(full_obs.reshape(1, -1))
            return norm_obs.flatten()
        return full_obs

    def _execute_trade(self, action: np.ndarray):
        """Execute trades with market impact and transaction costs"""
        total_value = self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        slippage = self.slippage * np.sqrt(self.metrics['trade_count'] + 1)
        target_alloc = action / (action.sum() + 1e-9)
        
        for i, asset in enumerate(self.assets):
            target_value = total_value * target_alloc[i+1]
            current_value = self.allocations[asset] * self._get_current_price(asset)
            delta = target_value - current_value
            
            if abs(delta) > 1:
                execution_price = self._get_current_price(asset) * (1 + np.sign(delta) * slippage)
                trade_amount = delta / execution_price
                commission_cost = abs(delta) * self.commission
                self.balance -= delta + commission_cost
                self.allocations[asset] += trade_amount
                self.trades.append({
                    'step': self.current_step,
                    'asset': asset,
                    'direction': 'BUY' if delta > 0 else 'SELL',
                    'quantity': abs(trade_amount),
                    'price': execution_price,
                    'commission': commission_cost,
                    'slippage': abs(delta * slippage)
                })
                self.metrics['trade_count'] += 1

    def _update_metrics(self):
        """Calculate real-time performance metrics"""
        current_value = self.portfolio_value
        prev_value = self.portfolio_history[-2] if len(self.portfolio_history) >= 2 else current_value
        ret = (current_value - prev_value) / prev_value
        self.metrics['returns'].append(ret)
        peak = np.max(self.portfolio_history[:self.current_step+1])
        dd = (peak - current_value) / peak
        self.metrics['drawdown'].append(dd)
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], dd)
        if len(self.metrics['returns']) >= 30:
            returns_window = np.array(self.metrics['returns'][-30:])
            vol = np.std(returns_window) * np.sqrt(365*24*2)
            self.metrics['volatility'].append(vol)
            excess_returns = returns_window - self.risk_free_rate/365/24/2
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365*24*2)
            self.metrics['sharpe'].append(sharpe)
            downside_returns = returns_window[returns_window < 0]
            sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(365*24*2)
            self.metrics['sortino'].append(sortino)

    @property
    def portfolio_value(self) -> float:
        return self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )

    def _get_current_price(self, asset: str) -> float:
        return self.data[asset]['close'].iloc[self.current_step]

    def run_backtest(self):
        """Execute full backtest with risk management"""
        print(f"Starting backtest with {self.n_steps} periods...")
        while self.current_step < self.n_steps:
            try:
                obs = self._get_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                self._execute_trade(action)
                current_value = self.portfolio_value
                self.portfolio_history.append(current_value)
                if len(self.portfolio_history) >= 2:
                    self._update_metrics()
                self.current_step += 1
                if current_value < self.initial_balance * 0.5:
                    print(f"Circuit breaker triggered at step {self.current_step}")
                    break
            except Exception as e:
                print(f"Error at step {self.current_step}: {str(e)}")
                break
        self._finalize_metrics()
        print("Backtest completed")

    def _finalize_metrics(self):
        """Calculate final performance metrics"""
        positive_trades = len([t for t in self.trades if t['direction'] == 'BUY' and 
                                 (self._get_current_price(t['asset']) > t['price'])])
        self.metrics['win_rate'] = positive_trades / len(self.trades) if self.trades else 0
        annualized_return = (self.portfolio_history[-1] / self.initial_balance) ** (1/(len(self.portfolio_history)/365/24/2)) - 1
        self.metrics['calmar'] = annualized_return / self.metrics['max_drawdown'] if self.metrics['max_drawdown'] > 0 else 0
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        self.metrics['value_at_risk'] = np.percentile(returns, 5)
        self.metrics['conditional_var'] = returns[returns <= self.metrics['value_at_risk']].mean()

    def generate_report(self):
        """Generate text-based performance report"""
        # Calculate returns
        returns = pd.Series(np.diff(self.portfolio_history) / self.portfolio_history[:-1], 
                          index=self.data.index[1:len(self.portfolio_history)])
        
        # Calculate additional metrics
        total_return = (self.portfolio_history[-1]/self.initial_balance-1)*100
        volatility = np.std(returns) * np.sqrt(365*24*2)  # Annualized volatility
        avg_daily_return = np.mean(returns)
        
        # Print comprehensive text report
        print(f"\n{'='*60}")
        print(f"{'BACKTEST RESULTS':^60}")
        print(f"{'='*60}\n")
        
        print("Portfolio Statistics:")
        print(f"Initial Balance:      ${self.initial_balance:,.2f}")
        print(f"Final Portfolio:      ${self.portfolio_history[-1]:,.2f}")
        print(f"Total Return:         {total_return:,.2f}%")
        print(f"Number of Trades:     {self.metrics['trade_count']}")
        print(f"Win Rate:            {self.metrics['win_rate']*100:.2f}%")
        
        print("\nRisk Metrics:")
        print(f"Max Drawdown:        {self.metrics['max_drawdown']*100:.2f}%")
        print(f"Annualized Vol:      {volatility*100:.2f}%")
        print(f"Sharpe Ratio:        {np.mean(self.metrics['sharpe']):.2f}")
        print(f"Sortino Ratio:       {np.mean(self.metrics['sortino']):.2f}")
        print(f"Calmar Ratio:        {self.metrics['calmar']:.2f}")
        
        print("\nRisk Management:")
        print(f"Value at Risk (95%): {self.metrics['value_at_risk']*100:.2f}%")
        print(f"Conditional VaR:     {self.metrics['conditional_var']*100:.2f}%")
        print(f"Avg Daily Return:    {avg_daily_return*100:.3f}%")
        
        print("\nTrading Period:")
        print(f"Start Date:          {self.data.index[0]}")
        print(f"End Date:            {self.data.index[len(self.portfolio_history)-1]}")
        print(f"Total Days:          {len(self.portfolio_history)}")
        print(f"{'='*60}\n")
        
        # Save basic metrics to file
        try:
            with open('backtest_results.txt', 'w') as f:
                f.write(f"Backtest Results Summary\n")
                f.write(f"Total Return: {total_return:.2f}%\n")
                f.write(f"Sharpe Ratio: {np.mean(self.metrics['sharpe']):.2f}\n")
                f.write(f"Max Drawdown: {self.metrics['max_drawdown']*100:.2f}%\n")
                f.write(f"Win Rate: {self.metrics['win_rate']*100:.2f}%\n")
                f.write(f"Total Trades: {self.metrics['trade_count']}\n")
        except Exception as e:
            print(f"Warning: Could not save results to file: {str(e)}")

if __name__ == "__main__":
    try:
        data = pd.read_parquet('data/multi_crypto.parquet')
        if data.empty:
            raise ValueError("Data file is empty")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        exit()

    backtester = BacktestEngine(
        model_path="models/multi_crypto_icm",
        env_stats_path="models/vec_normalize.pkl",
        data=data,
        initial_balance=10000.0,
        commission=0.0005,
        slippage=0.0001
    )

    backtester.run_backtest()
    backtester.generate_report()
