import os
import time
import signal
import logging
import ccxt
import numpy as np
import pandas as pd
import torch
import ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, Tuple, Optional
from trading_env import MultiCryptoEnv
import gc

# Configuration
CONFIG = {
    "initial_balance": 10000.0,
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "timeframe": "5m",
    "max_drawdown": 0.5,  # 50% max drawdown before stopping
    "position_limit": 0.5,  # Max 50% allocation per asset (more conservative)
    "slippage": 0.0005,  # 5 bps slippage
    "fee": 0.0002,  # 2 bps taker fee
    "data_refresh": 60,  # Seconds between data checks
    "model_path": "models/multi_crypto_icm_1M.zip",  # Add .zip extension
    "vecnormalize_path": "models/vec_normalize_1M.pkl",
    "log_file": "trading_log.csv"
}

class RealTimeTradingAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        self.trading_enabled = True
        self.portfolio = {
            "balance": config["initial_balance"],
            "allocations": {sym.split('/')[0]: 0.0 for sym in config["symbols"]},
            "value_history": [],
            "trade_history": []
        }
        
        # Initialize exchange connection
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Load model and normalization
        self.model, self.vec_normalize = self._load_model()
        self.asset_names = [sym.split('/')[0] for sym in config["symbols"]]
        
        # Set up logging
        self.logger = self._configure_logging()
        signal.signal(signal.SIGINT, self.signal_handler)

    def _load_model(self) -> Tuple[PPO, VecNormalize]:
        """Load trained model and normalization stats"""
        if not os.path.exists(CONFIG["model_path"]):
            raise FileNotFoundError(f"Model file {CONFIG['model_path']} not found")
            
        # Create environment with historical data
        def make_env():
            # Load historical data from parquet
            historical_data = pd.read_parquet('data/multi_crypto.parquet')
            # Take last 1000 periods to initialize environment
            historical_data = historical_data.iloc[-1000:]
            env = MultiCryptoEnv(historical_data)
            return env
        
        vec_env = DummyVecEnv([make_env])
        
        # Load model
        model = PPO.load(CONFIG["model_path"], device="auto")
        
        # Load normalization stats and apply to the environment
        vec_normalize = VecNormalize.load(CONFIG["vecnormalize_path"], vec_env)
        vec_normalize.training = False  # Disable training mode
        vec_normalize.norm_reward = False  # Disable reward normalization for inference
        
        return model, vec_normalize

    def _configure_logging(self) -> logging.Logger:
        """Set up structured logging"""
        logger = logging.getLogger("trading_agent")
        logger.setLevel(logging.WARNING)
        
        # File handler
        fh = logging.FileHandler(CONFIG["log_file"])
        fh.setLevel(logging.WARNING)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def signal_handler(self, signum, frame):
        """Handle SIGINT signals"""
        self.logger.warning("Received shutdown signal!")
        self.running = False
        self.trading_enabled = False

    def _fetch_realtime_data(self) -> bool:
        """Fetch latest candle for all symbols and update parquet file directly"""
        success = True
        data_updated = False
        
        try:
            # Read only the last 1000 rows from parquet for each asset
            full_data = pd.read_parquet('data/multi_crypto.parquet')
            latest_data = {asset: full_data[asset].iloc[-1000:] for asset in self.asset_names}
            
            for sym in self.config["symbols"]:
                try:
                    # Fetch OHLCV
                    ohlcv = self.exchange.fetch_ohlcv(
                        sym, 
                        self.config["timeframe"], 
                        since=int(time.time() * 1000) - 2*self.exchange.parse_timeframe(self.config["timeframe"])*1000,
                        limit=1
                    )
                    
                    # Process new data
                    asset = sym.split('/')[0]
                    new_row = pd.DataFrame([ohlcv[-1]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    new_row['timestamp'] = pd.to_datetime(new_row['timestamp'], unit='ms')
                    new_row.set_index('timestamp', inplace=True)
                    
                    # Check if this is actually new data
                    if new_row.index[0] not in latest_data[asset].index:
                        data_updated = True
                        # Add features to new row
                        new_row_with_features = self._add_features(pd.concat([latest_data[asset], new_row]))
                        latest_data[asset] = new_row_with_features.iloc[-1000:]  # Keep only last 1000 rows
                        
                        # Update full data
                        full_data[asset] = pd.concat([
                            full_data[asset].iloc[:-1000],  # Keep historical data
                            latest_data[asset]  # Update recent data
                        ])
                        
                except Exception as e:
                    self.logger.error(f"Error fetching data for {sym}: {str(e)}")
                    success = False
            
            # If we got new data, update the parquet file
            if data_updated:
                full_data.to_parquet('data/multi_crypto.parquet')
                self.logger.info("Updated historical data file with new records")
                
        except Exception as e:
            self.logger.error(f"Error updating parquet file: {str(e)}")
            success = False
        
        return success, latest_data if success else None

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add TA features matching training data"""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure we have enough data for indicators
            if len(df) < 26:  # 26 is the max window size needed for our indicators
                return df
            
            # Volume indicators
            df['volume_ema'] = ta.volume.volume_weighted_average_price(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume'],
                window=14
            )
            df['obv'] = ta.volume.on_balance_volume(
                close=df['close'], 
                volume=df['volume']
            )
            df['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            )
            
            # Trend indicators
            df['ema_12'] = ta.trend.ema_indicator(
                close=df['close'],
                window=12
            )
            df['ema_26'] = ta.trend.ema_indicator(
                close=df['close'],
                window=26
            )
            df['macd'] = ta.trend.macd(
                close=df['close'],
                window_slow=26,
                window_fast=12
            )
            df['macd_signal'] = ta.trend.macd_signal(
                close=df['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(
                close=df['close'],
                window=14
            )
            df['stoch'] = ta.momentum.stoch(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            df['stoch_signal'] = ta.momentum.stoch_signal(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # Volatility indicators
            df['bb_high'] = ta.volatility.bollinger_hband(
                close=df['close'],
                window=20
            )
            df['bb_low'] = ta.volatility.bollinger_lband(
                close=df['close'],
                window=20
            )
            df['atr'] = ta.volatility.average_true_range(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # Fill any NaN values and return only the latest row
            return df.ffill().bfill()
            
        except Exception as e:
            self.logger.error(f"Feature engineering error: {str(e)}")
            return df

    def _get_current_prices(self, latest_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Get latest closing prices"""
        return {
            asset: latest_data[asset].iloc[-1]['close']
            for asset in self.asset_names
        }

    def _create_observation(self, latest_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create model observation vector using provided data"""
        # Market features
        market_data = []
        for asset in self.asset_names:
            market_data.extend(latest_data[asset].iloc[-1].values)
        
        # Portfolio features
        prices = self._get_current_prices(latest_data)
        portfolio_value = self.portfolio["balance"] + sum(
            self.portfolio["allocations"][asset] * prices[asset]
            for asset in self.asset_names
        )
        portfolio_state = [
            self.portfolio["balance"] / self.config["initial_balance"]
        ] + [
            self.portfolio["allocations"][asset] * prices[asset] / portfolio_value
            for asset in self.asset_names
        ]
        
        obs = np.concatenate([np.array(market_data), np.array(portfolio_state)])
        return self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

    def _execute_trades(self, action: np.ndarray, latest_data: Dict[str, pd.DataFrame]):
        """Execute trades with slippage and fee modeling"""
        prices = self._get_current_prices(latest_data)
        total_value = self.portfolio["balance"] + sum(
            self.portfolio["allocations"][asset] * prices[asset]
            for asset in self.asset_names
        )
        
        # Apply action constraints
        action = np.clip(action, 0, 1)
        action /= action.sum() + 1e-8  # Normalize
        
        # Enforce position limits
        for i, alloc in enumerate(action[1:]):  # Skip cash allocation
            if alloc > self.config["position_limit"]:
                excess = alloc - self.config["position_limit"]
                action[i+1] = self.config["position_limit"]
                # Redistribute excess to cash position
                action[0] += excess
        
        # Renormalize after position limit adjustment
        action /= action.sum() + 1e-8

        # Convert allocations to dollar amounts
        target_alloc = {
            asset: total_value * action[i+1]
            for i, asset in enumerate(self.asset_names)
        }
        
        # Calculate trades
        for asset in self.asset_names:
            current_value = self.portfolio["allocations"][asset] * prices[asset]
            delta = target_alloc[asset] - current_value
            
            if abs(delta) < 1.0:  # Minimum trade size
                continue
                
            # Execute trade with slippage and fees
            if delta > 0:  # Buy
                slippage = delta * self.config["slippage"]
                fee = (delta - slippage) * self.config["fee"]
                self.portfolio["balance"] -= (delta + fee)
                self.portfolio["allocations"][asset] += (delta - slippage) / prices[asset]
            else:  # Sell
                slippage = abs(delta) * self.config["slippage"]
                fee = (abs(delta) - slippage) * self.config["fee"]
                self.portfolio["balance"] += (abs(delta) - slippage - fee)
                self.portfolio["allocations"][asset] -= (abs(delta) + slippage) / prices[asset]
                
            # Log trade
            self.portfolio["trade_history"].append({
                "timestamp": pd.Timestamp.now(),
                "asset": asset,
                "type": "BUY" if delta > 0 else "SELL",
                "amount": abs(delta),
                "price": prices[asset]
            })

    def _safety_checks(self, latest_data: Dict[str, pd.DataFrame]) -> bool:
        """Perform risk management checks"""
        current_value = self.portfolio["balance"] + sum(
            self.portfolio["allocations"][asset] * self._get_current_prices(latest_data)[asset]
            for asset in self.asset_names
        )
        
        # Check max drawdown
        if len(self.portfolio["value_history"]) > 0:
            peak = max(self.portfolio["value_history"])
            drawdown = (peak - current_value) / peak
            if drawdown > self.config["max_drawdown"]:
                self.logger.critical(f"Max drawdown exceeded: {drawdown:.2%}")
                return False
                
        # Check position limits - now just logs warnings but doesn't stop trading
        for asset in self.asset_names:
            allocation = self.portfolio["allocations"][asset] * self._get_current_prices(latest_data)[asset] / current_value
            if allocation > self.config["position_limit"]:
                self.logger.warning(f"Position limit exceeded for {asset}: {allocation:.2%}, will rebalance next cycle")
                
        return True

    def log_model_insights(self, obs: np.ndarray, action: np.ndarray):
        """
        Log intermediate outputs from the policy network to help explain decisions.
        This function assumes that your PPO model exposes a 'policy' attribute with methods to
        extract latent features and compute logits.
        """
        # Convert observation to torch tensor (ensure the right dtype and device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            # Extract latent features from the policy's feature extractor, if available.
            latent = self.model.policy.extract_features(obs_tensor.unsqueeze(0))
            
            # Get the policy logits (the raw output before applying softmax)
            logits = self.model.policy.mlp_extractor.policy_net(latent)
            
            # Compute action probabilities using softmax
            action_probs = torch.softmax(logits, dim=-1)
        
        # Convert tensors to numpy for logging
        latent_np = latent.cpu().numpy().flatten()
        logits_np = logits.cpu().numpy().flatten()
        action_probs_np = action_probs.cpu().numpy().flatten()
        
        self.logger.info("=== Model Insights ===")
        self.logger.info(f"Raw Observation: {obs}")
        self.logger.info(f"Latent Features: {latent_np}")
        self.logger.info(f"Policy Logits: {logits_np}")
        self.logger.info(f"Action Probabilities: {action_probs_np}")
        self.logger.info(f"Action Taken: {action}")
        self.logger.info("======================")

    def run(self):
        """Main trading loop with memory-efficient data handling"""
        self.logger.info("Starting trading session")
        
        while self.running:
            start_time = time.time()
            
            try:
                # Fetch and process data
                success, latest_data = self._fetch_realtime_data()
                if not success:
                    self.logger.warning("Data fetch failed, retrying next iteration")
                    time.sleep(self.config["data_refresh"])
                    continue
                
                # Create observation using latest data
                obs = self._create_observation(latest_data)
                
                if self.trading_enabled:
                    # Get model action
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Execute trades
                    self._execute_trades(action, latest_data)
                    
                    # Perform safety checks
                    if not self._safety_checks(latest_data):
                        self.logger.critical("Safety checks failed, stopping trading")
                        self.trading_enabled = False
                
                # Update portfolio value using latest prices
                current_value = self.portfolio["balance"] + sum(
                    self.portfolio["allocations"][asset] * latest_data[asset].iloc[-1]['close']
                    for asset in self.asset_names
                )
                self.portfolio["value_history"].append(current_value)
                
                # Log status
                self.logger.info(
                    f"Portfolio Value: ${current_value:,.2f} | "
                    f"Balance: ${self.portfolio['balance']:,.2f} | "
                    f"Allocations: {self.portfolio['allocations']}"
                )
                
                # Clean up to free memory
                del latest_data
                gc.collect()
                
                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config["data_refresh"] - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Critical error in main loop: {str(e)}", exc_info=True)
                self.running = False
                
        self.logger.info("Trading session ended")

if __name__ == "__main__":
    agent = RealTimeTradingAgent(CONFIG)
    agent.run()