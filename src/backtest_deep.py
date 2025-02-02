import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

class BacktestEngine:
    """Fixed backtesting system with proper action interpretation"""
    
    def __init__(
        self,
        model_path: str,
        env_stats_path: str,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.0005,
        slippage: float = 0.0001,
        risk_free_rate: float = 0.0,
        rebalance_threshold: float = 0.03,  # 3% allocation difference threshold
        volatility_window: int = 30  # Periods for dynamic threshold adjustment
    ):
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.rebalance_threshold = rebalance_threshold
        self.volatility_window = volatility_window
        
        # Load model with proper device mapping
        self.model = PPO.load(model_path, device='auto')
        self.assets = list(data.columns.get_level_values(0).unique())
        self.features = self._get_feature_list(data)
        
        # Data preparation
        self.data = self._preprocess_data(data)
        self.n_steps = len(self.data)
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.allocations = {asset: 0.0 for asset in self.assets}
        self.trades = []
        self.portfolio_history = [initial_balance]  # Initialize with starting balance
        self._init_metrics()

        # Create proper dummy environment matching training specs
        def create_proper_env():
            class ProperEnv(gym.Env):
                def __init__(self_env):  # Use self_env to avoid confusion with outer self
                    super().__init__()
                    raw_obs_dim = data.iloc[0].values.shape[0]
                    portfolio_dim = 1 + len(self.assets)  # Now self.assets is accessible
                    self_env.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, 
                        shape=(raw_obs_dim + portfolio_dim,),
                        dtype=np.float32
                    )
                    self_env.action_space = spaces.Box(
                        low=0, high=1, shape=(len(self.assets)+1,), dtype=np.float32
                    )
            return ProperEnv()
        
        dummy_env = DummyVecEnv([lambda: create_proper_env()])
        self.env_stats = VecNormalize.load(env_stats_path, venv=dummy_env)
        self.env_stats.training = False
        self.env_stats.norm_reward = False

    def _init_metrics(self):
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
            'win_rate': 0.0,
            'cash_history': [self.initial_balance],
            'rebalance_events': 0
        }

    def _get_feature_list(self, data: pd.DataFrame) -> list:
        return [f"{asset}/{feat}" for asset in self.assets 
                for feat in data[asset].columns if feat != 'timestamp']

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        processed = data.copy()
        for asset in self.assets:
            processed[asset] = processed[asset].ffill().bfill()
            processed[asset] = processed[asset].interpolate(method='linear')
        return processed.dropna()

    def _get_observation(self) -> np.ndarray:
        raw_obs = self.data.iloc[self.current_step].values.astype(np.float32)
        portfolio_state = [self.balance / self.initial_balance] + [
            self.allocations[asset] for asset in self.assets
        ]
        full_obs = np.concatenate([raw_obs, portfolio_state])
        
        if self.env_stats:
            norm_obs = self.env_stats.normalize_obs(full_obs.reshape(1, -1))
            return norm_obs.flatten()
        return full_obs

    def _execute_trade(self, action: np.ndarray):
        """Enhanced rebalancing with dynamic thresholds"""
        # Convert action to valid allocation
        action = np.clip(action, 0, 1)
        target_alloc = np.exp(action) / np.sum(np.exp(action))
        
        total_value = self.portfolio_value
        if total_value <= 0:
            return

        # Calculate current allocations
        current_alloc = np.array([self.balance / total_value] + [
            (self.allocations[asset] * self._get_current_price(asset)) / total_value
            for asset in self.assets
        ])
        
        # Dynamic threshold adjustment based on market volatility
        if len(self.metrics['returns']) >= self.volatility_window:
            recent_vol = np.std(self.metrics['returns'][-self.volatility_window:])
            adj_threshold = self.rebalance_threshold * (1 + recent_vol * 10)
        else:
            adj_threshold = self.rebalance_threshold
            
        # Calculate allocation drift
        max_diff = np.max(np.abs(target_alloc - current_alloc))
        
        # Only rebalance if beyond threshold
        if max_diff < adj_threshold:
            self.metrics['allocation_history'].append(current_alloc)
            return

        self.metrics['rebalance_events'] += 1
        
        # Calculate target values with portfolio value protection
        target_values = total_value * target_alloc[1:]
        current_values = np.array([
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        ])
        
        # Execute trades only for significantly changed allocations
        slippage = self.slippage * np.sqrt(self.metrics['trade_count'] + 1)
        
        print("\n=== Trade Execution ===")
        print(f"Step {self.current_step} - Starting Balance: ${self.balance:,.2f}")
        
        for i, asset in enumerate(self.assets):
            target_val = target_values[i]
            current_val = current_values[i]
            delta = target_val - current_val
            
            # Trade only if allocation difference > threshold
            alloc_diff = abs(target_alloc[i+1] - current_alloc[i+1])
            if alloc_diff < adj_threshold:
                continue

            if abs(delta) > 1:  # Minimum trade amount
                execution_price = self._get_current_price(asset) * (
                    1 + np.sign(delta) * slippage
                )
                commission = abs(delta) * self.commission
                trade_amount = delta / execution_price
                
                # Update balances and allocations
                self.balance -= delta + commission
                self.allocations[asset] += trade_amount
                
                # Print trade details
                print(f"\nTrade executed for {asset}:")
                print(f"  Direction: {'BUY' if delta > 0 else 'SELL'}")
                print(f"  Quantity: {abs(trade_amount):.6f}")
                print(f"  Price: ${execution_price:,.2f}")
                print(f"  Value: ${abs(delta):,.2f}")
                print(f"  Commission: ${commission:,.2f}")
                print(f"  Slippage Cost: ${abs(delta * slippage):,.2f}")
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'asset': asset,
                    'direction': 'BUY' if delta > 0 else 'SELL',
                    'quantity': abs(trade_amount),
                    'price': execution_price,
                    'commission': commission,
                    'slippage': abs(delta * slippage)
                })
                self.metrics['trade_count'] += 1

                # Emergency liquidation check
                if self.balance < 0:
                    liquidate_amount = abs(self.balance) / execution_price
                    self.allocations[asset] -= liquidate_amount
                    self.balance += liquidate_amount * execution_price
                    print(f"\n⚠️ Emergency liquidation at step {self.current_step}")
                    print(f"  Liquidated {liquidate_amount:.6f} {asset}")

        # Update cash allocation after asset rebalancing
        self.balance = total_value * target_alloc[0]
        self.metrics['allocation_history'].append(target_alloc)

        # Print final portfolio state
        print("\n=== Portfolio State ===")
        print(f"Final Cash Balance: ${self.balance:,.2f}")
        print("Current Positions:")
        for asset, qty in self.allocations.items():
            current_price = self._get_current_price(asset)
            position_value = qty * current_price
            print(f"  {asset}: {qty:.6f} (${position_value:,.2f})")
        print(f"Total Portfolio Value: ${self.portfolio_value:,.2f}")
        print("="*30)

    def _update_metrics(self):
        current_value = self.portfolio_value
        prev_value = self.portfolio_history[-2]
        
        ret = (current_value - prev_value) / prev_value
        self.metrics['returns'].append(ret)
        
        peak = np.max(self.portfolio_history)
        dd = (peak - current_value) / peak
        self.metrics['drawdown'].append(dd)
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], dd)
        
        if len(self.portfolio_history) >= 30:
            returns_window = np.array(self.metrics['returns'][-30:])
            vol = np.std(returns_window) * np.sqrt(365*24*2)
            self.metrics['volatility'].append(vol)
            
            excess_returns = returns_window - self.risk_free_rate/365/24/2
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365*24*2)
            self.metrics['sharpe'].append(sharpe)
            
            downside_returns = returns_window[returns_window < 0]
            sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(365*24*2)
            self.metrics['sortino'].append(sortino)
        
        self.metrics['cash_history'].append(self.balance)

    @property
    def portfolio_value(self) -> float:
        return self.balance + sum(
            self.allocations[asset] * self._get_current_price(asset)
            for asset in self.assets
        )

    def _get_current_price(self, asset: str) -> float:
        return self.data[asset]['close'].iloc[self.current_step]

    def run_backtest(self):
        print(f"Starting backtest with {self.n_steps} periods...")
        try:
            while self.current_step < self.n_steps:
                obs = self._get_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                self._execute_trade(action)
                
                current_value = self.portfolio_value
                self.portfolio_history.append(current_value)
                
                if len(self.portfolio_history) >= 2:
                    self._update_metrics()
                
                self.current_step += 1

                # Dynamic risk management
                if len(self.portfolio_history) > 10:
                    recent_drawdown = np.mean(self.metrics['drawdown'][-10:])
                    if recent_drawdown > 0.4:
                        print(f"Volatility circuit breaker triggered at {self.current_step}")
                        break
                    
        except Exception as e:
            print(f"Error at step {self.current_step}: {str(e)}")
        
        self._finalize_metrics()
        print("Backtest completed")

    def _finalize_metrics(self):
        if len(self.trades) == 0:
            return

        # Calculate win rate based on profitable trades
        profitable_trades = []
        for trade in self.trades:
            if trade['direction'] == 'BUY':
                exit_price = self.data[trade['asset']]['close'].iloc[min(
                    self.current_step, 
                    trade['step'] + 10  # Consider 10-period holding
                )]
                profitable = exit_price > trade['price']
            else:
                exit_price = self.data[trade['asset']]['close'].iloc[min(
                    self.current_step, 
                    trade['step'] + 10
                )]
                profitable = exit_price < trade['price']
            profitable_trades.append(profitable)
        
        self.metrics['win_rate'] = np.mean(profitable_trades)
        
        # Calculate annualized return
        period_return = self.portfolio_history[-1] / self.initial_balance
        n_periods = len(self.portfolio_history)
        annualized_return = (period_return ** (365*24*2/n_periods)) - 1
        
        self.metrics['calmar'] = annualized_return / self.metrics['max_drawdown'] \
            if self.metrics['max_drawdown'] > 0 else 0
        
        # Risk metrics
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        self.metrics['value_at_risk'] = np.percentile(returns, 5)
        self.metrics['conditional_var'] = returns[returns <= self.metrics['value_at_risk']].mean()

    def generate_report(self):
        print(f"\n{' Backtest Report ':=^60}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Value:    ${self.portfolio_history[-1]:,.2f}")
        print(f"Total Return:   {(self.portfolio_history[-1]/self.initial_balance-1)*100:.2f}%")
        print(f"Max Drawdown:   {self.metrics['max_drawdown']*100:.2f}%")
        print(f"Win Rate:       {self.metrics['win_rate']*100:.2f}%")
        print(f"Total Trades:   {self.metrics['trade_count']}")
        print(f"Sharpe Ratio:   {np.nanmean(self.metrics['sharpe']):.2f}")
        print(f"Sortino Ratio:  {np.nanmean(self.metrics['sortino']):.2f}")
        print(f"Calmar Ratio:   {self.metrics['calmar']:.2f}")
        print("="*60)

if __name__ == "__main__":
    try:
        data = pd.read_parquet('data/multi_crypto.parquet')
        if data.empty:
            raise ValueError("Data file is empty")
    except Exception as e:
        print(f"Data error: {str(e)}")
        exit()

    backtester = BacktestEngine(
        model_path="models/multi_crypto_icm_1M",
        env_stats_path="models/vec_normalize_1M.pkl",
        data=data,
        initial_balance=10000.0,
        commission=0.0005,
        slippage=0.0001,
        rebalance_threshold=0.03  # 3% allocation difference threshold
    )

    backtester.run_backtest()
    backtester.generate_report()