import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Tuple
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Set logger level to WARNING

class InstitutionalPerpetualEnv(gym.Env):
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 1e6,
                 max_leverage: float = 20,
                 transaction_fee: float = 0.0004,
                 funding_fee_multiplier: float = 0.8,
                 risk_free_rate: float = 0.03,
                 max_drawdown: float = 0.3,
                 window_size: int = 100):
        
        super().__init__()
        
        self.df = df
        # Get unique assets from the first level of MultiIndex columns
        self.assets = list(df.columns.get_level_values('asset').unique())
        
        # Split features by type
        all_features = df.columns.get_level_values('feature').unique()
        
        # Base features that should exist for all assets
        self.base_features = ['open', 'high', 'low', 'close', 'volume', 'funding_rate', 'bid_depth', 'ask_depth', 'volatility']
        
        # Technical features
        self.tech_features = [f for f in all_features if f not in self.base_features]
        
        # Log feature information
        logger.info(f"Initialized with assets: {self.assets}")
        logger.info(f"Base features: {self.base_features}")
        logger.info(f"Technical features: {self.tech_features}")
        
        # Calculate total feature dimension
        n_base_features = len(self.base_features)
        n_tech_features = len(self.tech_features)
        n_portfolio_features = 3  # size, value ratio, funding accrued
        n_global_features = 3     # balance ratio, pnl ratio, active positions ratio
        
        total_features = (n_base_features + n_tech_features) * len(self.assets) + \
                        n_portfolio_features * len(self.assets) + n_global_features
        
        logger.info(f"Total feature dimension: {total_features}")
        
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.transaction_fee = transaction_fee
        self.funding_fee_multiplier = funding_fee_multiplier
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        self.window_size = window_size
        
        # Initialize risk engine
        self.risk_engine = InstitutionalRiskEngine(
            risk_limits=RiskLimits(
                max_drawdown=max_drawdown,
                max_leverage=max_leverage,
                var_limit=0.05,
                position_concentration=0.4,
                correlation_limit=0.7,
                liquidity_ratio=0.1
            )
        )
        
        # Track historical performance
        self.returns_history = deque(maxlen=10000)
        self.positions_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=10000)
        
        n_assets = len(self.assets)
        
        # Action space: [trade_decisions (-1 to 1), position_sizes (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1] * n_assets + [0] * n_assets),
            high=np.array([1] * n_assets + [1] * n_assets),
            dtype=np.float32
        )
        
        # Initialize current step and positions before getting observation
        self.current_step = self.window_size
        self.positions = {asset: {'size': 0, 'entry_price': 0, 'funding_accrued': 0} 
                         for asset in self.assets}
        
        # Set observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # Log the initial state of the DataFrame
        logger.info(f"Initial DataFrame columns: {self.df.columns}")
        logger.info(f"Assets: {self.assets}")
        
        # Log the features being split
        logger.info(f"Base features: {self.base_features}")
        logger.info(f"Technical features: {self.tech_features}")
        
        # Check for missing features
        for asset in self.assets:
            for feature in self.base_features + self.tech_features:
                if (asset, feature) not in self.df.columns:
                    logger.error(f"Missing feature {feature} for asset {asset}")
        
        # Log the initialization completion
        logger.info("InstitutionalPerpetualEnv initialization complete.")
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.positions = {asset: {'size': 0, 'entry_price': 0, 'funding_accrued': 0} 
                         for asset in self.assets}
        self.last_action = None
        self.done = False
        
        return self._get_observation(), {}
        
    def step(self, action):
        """
        Execute one step in the environment
        action: numpy array of shape (n_assets * 2,) containing:
            - First n_assets elements: trade decisions (-1 for sell, 0 for hold, 1 for buy)
            - Last n_assets elements: position sizes (0 to 1)
        """
        try:
            # Split action array into decisions and sizes
            n_assets = len(self.assets)
            trade_decisions = action[:n_assets]
            position_sizes = action[n_assets:]
            
            # Process each asset
            total_pnl = 0
            info = {}
            
            for i, asset in enumerate(self.assets):
                # Convert to float and clip values
                trade_decision = float(np.clip(trade_decisions[i], -1, 1))
                position_size = float(np.clip(position_sizes[i], 0, 1))
                
                # Execute trade
                pnl = self._execute_trade(asset, trade_decision, position_size)
                total_pnl += pnl
                
                # Store trade info
                info[f"{asset}_trade"] = trade_decision
                info[f"{asset}_size"] = position_size
                info[f"{asset}_pnl"] = pnl
            
            # Update positions and get next observation
            self._update_positions()
            self.current_step += 1
            
            # Calculate reward and check done conditions
            reward = self._calculate_reward(total_pnl)
            done = self._is_done()
            
            return self._get_observation(), reward, done, False, info
            
        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            return self._get_observation(), -1, True, False, {"error": str(e)}
        
    def _simulate_trades(self, trades: List[Dict]) -> Dict:
        """Simulate trades to check risk limits"""
        simulated_positions = self.positions.copy()
        
        for trade in trades:
            asset = trade['asset']
            if 'size' in trade:  # Close position
                simulated_positions[asset]['size'] = 0
            else:  # New trade
                target_exposure = trade['direction'] * trade['leverage'] * self.balance
                current_exposure = simulated_positions[asset]['size'] * self._get_mark_price(asset)
                order_size = (target_exposure - current_exposure) * trade['execution_params'][0]
                simulated_positions[asset]['size'] = order_size / self._get_mark_price(asset)
                
        return simulated_positions
        
    def _execute_trades(self, trades: List[Dict]):
        """Smart order execution with transaction cost model"""
        for trade in trades:
            asset = trade['asset']
            direction = trade['direction']
            leverage = trade['leverage']
            risk_limits = trade['risk_limits']
            execution_params = trade['execution_params']
            
            # Target position calculation
            target_exposure = direction * leverage * self.balance
            current_exposure = self.positions[asset]['size'] * self._get_mark_price(asset)
            
            # Smart execution sizing
            order_size = (target_exposure - current_exposure) * execution_params[0]
            if abs(order_size) > 0:
                # Implementation shortfall model
                price_impact = self._estimate_price_impact(asset, order_size)
                execution_price = self._get_mark_price(asset) * (1 + price_impact)
                
                # Apply transaction costs
                total_cost = abs(order_size) * (
                    self.transaction_fee + 
                    price_impact + 
                    self._get_spread_cost(asset)
                )
                
                # Update position and balance
                self.positions[asset]['size'] += order_size / execution_price
                self.balance -= total_cost
                
                if self.positions[asset]['size'] != 0:
                    self.positions[asset]['entry_price'] = execution_price
                    
    def _update_positions(self):
        """Update positions with funding and mark-to-market"""
        total_pnl = 0
        for asset in self.assets:
            try:
                position = self.positions[asset]
                mark_price = self._get_mark_price(asset)
                
                # Get funding rate using MultiIndex
                try:
                    funding_rate = self.df.loc[self.df.index[self.current_step], (asset, 'funding_rate')]
                    if isinstance(funding_rate, (pd.Series, np.ndarray)):
                        funding_rate = funding_rate.iloc[0] if isinstance(funding_rate, pd.Series) else funding_rate[0]
                    funding_rate = float(funding_rate)
                except Exception as e:
                    logger.warning(f"Error getting funding rate for {asset}: {str(e)}, using 0.0")
                    funding_rate = 0.0
                
                # Calculate funding cost
                funding_cost = position['size'] * mark_price * funding_rate * self.funding_fee_multiplier
                position['funding_accrued'] -= funding_cost
                self.balance -= funding_cost
                
                # Mark-to-market PnL
                if position['size'] != 0:
                    unrealized_pnl = position['size'] * (mark_price - position['entry_price'])
                    total_pnl += unrealized_pnl
                    
            except Exception as e:
                logger.error(f"Error updating position for {asset}: {str(e)}")
                continue
                
        # Update peak balance for drawdown calculation
        self.peak_balance = max(self.peak_balance, self.balance + total_pnl)
        current_drawdown = (self.peak_balance - (self.balance + total_pnl)) / self.peak_balance
        self.drawdown_history.append(current_drawdown)
        
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        try:
            # Calculate basic metrics
            total_exposure = 0
            total_pnl = 0
            
            # Get current market data window with proper MultiIndex handling
            market_data = self.df.iloc[self.current_step - self.window_size:self.current_step + 1].copy()
            
            # Ensure market data has unique index
            market_data = market_data.loc[~market_data.index.duplicated(keep='first')]
            
            # Calculate portfolio value and positions
            portfolio_value = self.balance
            positions_dict = {}
            
            for asset, position in self.positions.items():
                try:
                    # Get mark price using proper MultiIndex access
                    mark_price = market_data.loc[market_data.index[-1], (asset, 'close')]
                    if isinstance(mark_price, (pd.Series, pd.DataFrame)):
                        mark_price = mark_price.iloc[0]
                    mark_price = float(mark_price)
                    
                    position_value = position['size'] * mark_price
                    total_exposure += abs(position_value)
                    total_pnl += position['size'] * (mark_price - position['entry_price'])
                    portfolio_value += position_value
                    
                    # Update positions dict for risk engine
                    positions_dict[asset] = {
                        'size': float(position['size']),
                        'value': float(position_value),
                        'entry_price': float(position['entry_price'])
                    }
                except Exception as e:
                    logger.error(f"Error processing position for {asset}: {str(e)}")
                    positions_dict[asset] = {'size': 0, 'value': 0, 'entry_price': 0}
            
            # Calculate risk metrics using risk engine
            try:
                risk_metrics = self.risk_engine.calculate_portfolio_risk(
                    positions=positions_dict,
                    market_data=market_data,
                    portfolio_value=portfolio_value
                )
            except Exception as e:
                logger.error(f"Error in risk engine calculations: {str(e)}")
                # Provide default risk metrics if risk engine fails
                risk_metrics = {
                    'var': 0.0,
                    'expected_shortfall': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0,
                    'leverage_utilization': total_exposure / (portfolio_value + 1e-8),
                    'max_concentration': 0.0,
                    'correlation_risk': 0.0,
                    'max_adv_ratio': 0.0,
                    'total_liquidation_cost': 0.0,
                    'max_liquidation_cost': 0.0
                }
            
            # Ensure all required metrics are present
            required_metrics = [
                'var', 'expected_shortfall', 'volatility', 'max_drawdown',
                'leverage_utilization', 'max_concentration', 'correlation_risk',
                'max_adv_ratio', 'total_liquidation_cost', 'max_liquidation_cost'
            ]
            
            for metric in required_metrics:
                if metric not in risk_metrics:
                    risk_metrics[metric] = 0.0
            
            # Add basic portfolio metrics
            risk_metrics.update({
                'total_exposure': total_exposure,
                'total_pnl': total_pnl,
                'portfolio_value': portfolio_value,
                'balance': self.balance,
                'current_drawdown': (self.peak_balance - portfolio_value) / (self.peak_balance + 1e-8)
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in _calculate_risk_metrics: {str(e)}")
            # Return safe default values if calculation fails
            return {
                'var': 0.0,
                'expected_shortfall': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'leverage_utilization': 0.0,
                'max_concentration': 0.0,
                'correlation_risk': 0.0,
                'max_adv_ratio': 0.0,
                'total_liquidation_cost': 0.0,
                'max_liquidation_cost': 0.0,
                'total_exposure': 0.0,
                'total_pnl': 0.0,
                'portfolio_value': self.balance,
                'balance': self.balance,
                'current_drawdown': 0.0
            }
        
    def _calculate_risk_adjusted_reward(self, risk_metrics: Dict) -> float:
        """Enhanced reward function incorporating risk metrics"""
        try:
            # Extract metrics
            portfolio_return = risk_metrics['total_pnl'] / self.initial_balance
            volatility = max(risk_metrics.get('volatility', 0), 1e-8)
            max_drawdown = max(risk_metrics.get('max_drawdown', 0), 1e-8)
            leverage = risk_metrics.get('leverage_utilization', 0)
            var = risk_metrics.get('var', 0)
            es = risk_metrics.get('expected_shortfall', 0)
            
            # Calculate components
            excess_return = portfolio_return - self.risk_free_rate
            
            # Risk-adjusted ratios
            sharpe = excess_return / volatility
            sortino = excess_return / (abs(var) + 1e-8)
            calmar = excess_return / max_drawdown
            
            # Risk penalties
            risk_penalty = 0
            if leverage > self.max_leverage:
                risk_penalty -= 0.2
            if max_drawdown > self.max_drawdown:
                risk_penalty -= 0.2
            if abs(var) > self.risk_engine.risk_limits.var_limit:
                risk_penalty -= 0.2
            
            # Position diversification bonus
            active_positions = len([p for p in self.positions.values() if p['size'] != 0])
            diversification_score = active_positions / len(self.assets)
            
            # Combine components with weights
            reward = (
                0.3 * sharpe +
                0.2 * sortino +
                0.2 * calmar +
                0.1 * diversification_score +
                0.2 * (1.0 - leverage / self.max_leverage) +  # Penalize high leverage
                risk_penalty
            )
            
            return float(np.clip(reward, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted reward: {str(e)}")
            return -1.0
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation including multimodal features"""
        try:
            # Get market data window
            market_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
            
            # Initialize feature lists
            features = []
            
            # Process each asset
            for asset in self.assets:
                # Add base features
                for feature in self.base_features:
                    try:
                        value = market_data.loc[market_data.index[-1], (asset, feature)]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        features.append(float(value))
                    except Exception as e:
                        logger.debug(f"Could not get base feature {feature} for {asset}: {str(e)}")
                        features.append(0.0)
                
                # Add technical features
                for feature in self.tech_features:
                    try:
                        value = market_data.loc[market_data.index[-1], (asset, feature)]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        features.append(float(value))
                    except Exception as e:
                        logger.debug(f"Could not get tech feature {feature} for {asset}: {str(e)}")
                        features.append(0.0)
            
            # Add portfolio state features
            total_value = self.balance
            for asset in self.assets:
                position = self.positions[asset]
                mark_price = self._get_mark_price(asset)
                position_value = position['size'] * mark_price
                total_value += position_value
                
                features.extend([
                    float(position['size']),
                    float(position_value / total_value if total_value > 0 else 0),
                    float(position['funding_accrued'] / self.initial_balance)
                ])
            
            # Add global portfolio features
            features.extend([
                float(self.balance / self.initial_balance),
                float((total_value - self.initial_balance) / self.initial_balance),
                float(len([p for p in self.positions.values() if p['size'] != 0]) / len(self.assets))
            ])
            
            # Convert to numpy array
            observation = np.array(features, dtype=np.float32)
            
            # Ensure observation is finite and within bounds
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e3, neginf=-1e3)
            observation = np.clip(observation, -1e3, 1e3)
            
            # Verify observation shape
            expected_shape = self.observation_space.shape[0]
            actual_shape = observation.shape[0]
            if expected_shape != actual_shape:
                logger.error(f"Observation shape mismatch: expected {expected_shape}, got {actual_shape}")
                logger.error(f"Features length: {len(features)}")
                # Pad or truncate to match expected shape
                if actual_shape < expected_shape:
                    observation = np.pad(observation, (0, expected_shape - actual_shape))
                else:
                    observation = observation[:expected_shape]
            
            return observation
            
        except Exception as e:
            logger.error(f"Error in _get_observation: {str(e)}")
            # Return zero array with correct shape
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
    def _calculate_reward(self, total_pnl: float) -> float:
        """Calculate reward based on PnL and risk metrics"""
        try:
            # Store return for this step
            current_return = total_pnl / self.initial_balance
            self.returns_history.append(current_return)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Calculate risk-adjusted reward
            reward = self._calculate_risk_adjusted_reward(risk_metrics)
            
            return float(reward)
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return -1.0

    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        try:
            # Check if we've reached the end of data
            if self.current_step >= len(self.df) - 1:
                logger.info("Episode done: Reached end of data")
                return True
            
            # Calculate current risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Check account depletion
            if risk_metrics['portfolio_value'] <= 0:
                logger.info("Episode done: Account depleted")
                return True
            
            # Check risk limits using risk engine
            try:
                is_within_limits, violations = self.risk_engine.check_risk_limits(risk_metrics)
                if not is_within_limits:
                    logger.info(f"Episode done: Risk limits violated - {', '.join(violations)}")
                    return True
            except Exception as e:
                logger.error(f"Error checking risk limits: {str(e)}")
                # Continue if risk check fails
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error in _is_done: {str(e)}")
            # Return True to safely terminate if we can't determine state
            return True

    def update_parameters(self, **kwargs):
        """Update environment parameters for curriculum learning"""
        # Update risk limits
        if 'max_leverage' in kwargs:
            self.max_leverage = kwargs['max_leverage']
            self.risk_engine.risk_limits.max_leverage = kwargs['max_leverage']
            
        if 'max_drawdown' in kwargs:
            self.max_drawdown = kwargs['max_drawdown']
            self.risk_engine.risk_limits.max_drawdown = kwargs['max_drawdown']
            
        if 'var_limit' in kwargs:
            self.risk_engine.risk_limits.var_limit = kwargs['var_limit']
            
        if 'position_concentration' in kwargs:
            self.risk_engine.risk_limits.position_concentration = kwargs['position_concentration']
            
        # Update other parameters
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                
        logger.info(f"Updated environment parameters: {kwargs}")

    def _execute_trade(self, asset: str, trade_decision: float, position_size: float) -> float:
        """Execute a trade for a single asset with enhanced liquidity and slippage modeling"""
        try:
            current_price = self._get_mark_price(asset)
            if current_price <= 0:
                return 0.0
                
            # Calculate target position size
            portfolio_value = self.balance + sum(
                pos['size'] * self._get_mark_price(a) 
                for a, pos in self.positions.items()
            )
            
            # Get volume data for liquidity checks
            volume = self.df.loc[self.df.index[self.current_step], (asset, 'volume')]
            if isinstance(volume, (pd.Series, pd.DataFrame)):
                volume = volume.iloc[0]
            volume = float(volume)
            
            # Calculate ADV (5-day average daily volume)
            adv = self.df.loc[
                self.df.index[max(0, self.current_step-5):self.current_step+1],
                (asset, 'volume')
            ].mean()
            if isinstance(adv, (pd.Series, pd.DataFrame)):
                adv = adv.iloc[0]
            adv = float(adv)
            
            # Limit position size based on ADV
            max_position_value = adv * current_price * 0.1  # Max 10% of ADV
            max_position_size = max_position_value / current_price
            
            # Scale target size by trade decision (-1 to 1) and ADV limit
            raw_target_size = position_size * portfolio_value / current_price * trade_decision
            target_size = np.clip(raw_target_size, -max_position_size, max_position_size)
            
            # Calculate size difference
            size_diff = target_size - self.positions[asset]['size']
            
            if abs(size_diff) > 0:
                # Calculate price impact using square-root law
                price_impact = 0.1 * np.sqrt(abs(size_diff) / (volume + 1e-8))
                effective_price = current_price * (1 + price_impact * np.sign(size_diff))
                
                # Calculate TWAP chunks if size is large
                twap_chunks = 1
                if abs(size_diff) > max_position_size * 0.1:  # If order > 10% of max size
                    twap_chunks = min(5, int(abs(size_diff) / (max_position_size * 0.1)))
                    size_diff = size_diff / twap_chunks
                
                total_cost = 0
                total_pnl = 0
                executed_size = 0
                
                for _ in range(twap_chunks):
                    # Calculate transaction cost including spread and impact
                    transaction_cost = (
                        abs(size_diff * effective_price * self.transaction_fee) +  # Base fee
                        abs(size_diff * effective_price * self._get_spread_cost(asset)) +  # Spread cost
                        abs(size_diff * effective_price * price_impact)  # Impact cost
                    )
                    
                    # Check if we have enough balance
                    if transaction_cost > self.balance:
                        break
                    
                    # Update position
                    old_size = self.positions[asset]['size']
                    self.positions[asset]['size'] += size_diff
                    executed_size += size_diff
                    
                    # Update entry price using VWAP
                    if self.positions[asset]['size'] != 0:
                        self.positions[asset]['entry_price'] = (
                            (old_size * self.positions[asset]['entry_price'] + 
                             size_diff * effective_price) / 
                            self.positions[asset]['size']
                        )
                    
                    # Update balance
                    self.balance -= transaction_cost
                    total_cost += transaction_cost
                    
                    # Calculate PnL if reducing position
                    if (old_size > 0 and size_diff < 0) or (old_size < 0 and size_diff > 0):
                        chunk_pnl = abs(size_diff) * (effective_price - self.positions[asset]['entry_price'])
                        self.balance += chunk_pnl
                        total_pnl += chunk_pnl
                
                # Log execution details
                logger.info(f"Trade executed for {asset}:")
                logger.info(f"  Target size: {target_size:.6f}")
                logger.info(f"  Executed size: {executed_size:.6f}")
                logger.info(f"  TWAP chunks: {twap_chunks}")
                logger.info(f"  Price impact: {price_impact:.6f}")
                logger.info(f"  Total cost: {total_cost:.2f}")
                
                return total_pnl - total_cost
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Error executing trade for {asset}: {str(e)}")
            return 0.0

    def _get_mark_price(self, asset: str) -> float:
        """Get the current mark price for an asset"""
        try:
            # Access close price using MultiIndex
            price = self.df.loc[self.df.index[self.current_step], (asset, 'close')]
            # Convert to float, handling both scalar and array cases
            if isinstance(price, (pd.Series, np.ndarray)):
                price = price.iloc[0] if isinstance(price, pd.Series) else price[0]
            price = float(price)
            if price <= 0:
                logger.warning(f"Invalid price ({price}) for {asset}, using fallback value of 1.0")
                return 1.0
            return price
        except Exception as e:
            logger.error(f"Error getting mark price for {asset}: {str(e)}")
            return 1.0  # Fallback value
        
    def _get_spread_cost(self, asset: str) -> float:
        """Estimate spread cost based on order book"""
        try:
            asset_data = self.df.xs(asset, axis=1, level='asset')
            if 'bid' in asset_data.columns and 'ask' in asset_data.columns:
                spread = (asset_data['ask'].iloc[self.current_step] - 
                         asset_data['bid'].iloc[self.current_step])
                return spread / asset_data['close'].iloc[self.current_step]
            return self.transaction_fee  # Default to base fee if no order book data
        except Exception as e:
            logger.error(f"Error getting spread cost for {asset}: {str(e)}")
            return self.transaction_fee
        
    def _estimate_price_impact(self, asset: str, order_size: float) -> float:
        """Estimate price impact using square-root law"""
        try:
            asset_data = self.df.xs(asset, axis=1, level='asset')
            adv = asset_data['volume'].iloc[self.current_step]
            return 0.1 * np.sqrt(abs(order_size) / adv) * np.sign(order_size)
        except Exception as e:
            logger.error(f"Error estimating price impact for {asset}: {str(e)}")
            return 0.0
        
    def _get_info(self, risk_metrics: Dict) -> Dict:
        """Get additional info for monitoring"""
        return {
            'balance': self.balance,
            'returns': self.returns_history[-1] if len(self.returns_history) > 0 else 0,
            'drawdown': self.drawdown_history[-1] if len(self.drawdown_history) > 0 else 0,
            'positions': self.positions,
            'step': self.current_step,
            'risk_metrics': risk_metrics
        }