import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Tuple, Optional
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
import logging
from utils.time_utils import get_utc_now, timestamp_to_datetime
import traceback
import copy
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Change from WARNING to ERROR for less verbosity

class InstitutionalPerpetualEnv(gym.Env):
    def __init__(self, 
                 df: pd.DataFrame,
                 assets: List[str],
                 window_size: int = 100,
                 max_leverage: float = 20.0,
                 commission: float = 0.0004,
                 funding_fee_multiplier: float = 1.0,
                 base_features: List[str] = None,
                 tech_features: List[str] = None,
                 risk_engine: Optional[InstitutionalRiskEngine] = None,
                 risk_free_rate: float = 0.02,
                 initial_balance: float = 10000.0,  # Make initial balance configurable
                 max_drawdown: float = 0.3,  # Make max drawdown configurable
                 maintenance_margin: float = 0.1,  # Maintenance margin as fraction of initial balance
                 verbose: bool = False):  # Add verbose flag
        
        super().__init__()
        
        # Store verbose flag
        self.verbose = verbose
        
        # Adjust logging level based on verbose flag
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)  # Set to ERROR when not verbose
        
        # Store DataFrame and validate MultiIndex
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex columns (asset, feature)")
        self.df = df
        
        # Basic parameters
        self.initial_balance = initial_balance  # Use the passed parameter
        self.balance = self.initial_balance
        self.commission = commission
        self.max_leverage = max_leverage
        # DEX trading requirements: leverage must be at least 1.0x and at most max_leverage
        # Leverage values are continuous from 1.0 to max_leverage (e.g., 1.5x, 3.2x, etc.)
        self.window_size = window_size
        self.current_step = self.window_size
        self.funding_fee_multiplier = funding_fee_multiplier
        self.max_drawdown = max_drawdown  # Use the passed parameter
        self.risk_free_rate = risk_free_rate
        self.maintenance_margin = maintenance_margin  # Store maintenance margin
        
        # Extract unique assets from MultiIndex
        self.assets = assets
        
        # Initialize last prices dictionary
        self.last_prices = {}
        for asset in self.assets:
            try:
                self.last_prices[asset] = float(df.iloc[0].loc[(asset, 'close')])
            except:
                self.last_prices[asset] = 1000.0  # Default fallback price
        
        # Initialize positions
        self.positions = {asset: {'size': 0, 'entry_price': 0, 'funding_accrued': 0,
                                 'last_price': self._get_mark_price(asset) if not self.df.empty else 1000.0,
                                 'leverage': 0.0} 
                         for asset in self.assets}
        
        # Initialize total costs tracking
        self.total_costs = 0.0
        
        # ENHANCED: Track position duration and profitability
        self.position_duration = {asset: 0 for asset in self.assets}
        self.position_profits = {asset: [] for asset in self.assets}
        self.profitable_holding_bonus = 0.0  # Will accumulate bonuses for holding profitable positions
        
        # Initialize funding rates and accrued funding
        self.funding_rates = {asset: 0.0 for asset in self.assets}
        self.funding_accrued = {asset: 0.0 for asset in self.assets}
        
        # ENHANCED: Track trading frequency to penalize overtrading
        self.trade_counts = deque(maxlen=100)  # Track trades per step for last 100 steps
        self.last_action_vector = np.zeros(len(self.assets))
        self.consecutive_no_trade_steps = 0
        self.optimal_trade_frequency = 0.1  # Target 10% of steps to have trades
        
        # Trading history
        self.trades = []
        self.portfolio_history = [{'step': 0, 'value': self.initial_balance}]
        
        # Track historical performance with deques
        self.returns_history = deque(maxlen=10000)
        self.positions_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=10000)
        self.historical_leverage = deque(maxlen=10000)
        
        # Initialize peak values
        self.peak_value = self.initial_balance
        self.peak_balance = self.initial_balance
        
        # Define action and observation spaces
        n_assets = len(self.assets)
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_assets,), dtype=np.float32)
        
        # Set default features if none provided
        self.base_features = base_features or ['close', 'volume', 'funding_rate']
        self.tech_features = tech_features or ['rsi', 'macd', 'bb_upper', 'bb_lower']
        
        # Calculate observation space size
        n_base_features = len(self.base_features)
        n_tech_features = len(self.tech_features)
        n_portfolio_features = 3  # size, value ratio, funding accrued
        n_global_features = 3     # balance ratio, pnl ratio, active positions ratio
        
        total_features = (n_base_features + n_tech_features) * len(self.assets) + \
                        n_portfolio_features * len(self.assets) + n_global_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        # Initialize risk engine with larger lookback window
        self.risk_engine = risk_engine or InstitutionalRiskEngine(lookback_window=250)
        
        # Initialize liquidation flag
        self.liquidated = False
        
        # Log initialization
        logger.info(f"Initialized with assets: {self.assets}")
        logger.info(f"Base features: {self.base_features}")
        logger.info(f"Technical features: {self.tech_features}")
        logger.info(f"Total feature dimension: {total_features}")
        logger.info(f"Initial DataFrame columns: {self.df.columns}")
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.peak_value = self.initial_balance
        self.positions = {asset: {'size': 0, 'entry_price': 0, 'funding_accrued': 0,
                                 'last_price': self._get_mark_price(asset) if not self.df.empty else 1000.0,
                                 'leverage': 0.0} 
                         for asset in self.assets}
        self.last_action = None
        self.done = False
        self.liquidated = False
        
        # Reset total costs
        self.total_costs = 0.0
        
        # IMPORTANT FIX: Properly initialize all collections
        self.returns_history = deque(maxlen=10000)
        self.positions_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=10000)
        self.historical_leverage = deque(maxlen=10000)
        self.portfolio_history = [{'step': 0, 'value': self.initial_balance}]
        self.trades = []  # Clear trade history
        
        # IMPORTANT FIX: Initialize funding rates and accrued funding
        self.funding_rates = {asset: 0.0 for asset in self.assets}
        self.funding_accrued = {asset: 0.0 for asset in self.assets}
        
        # IMPORTANT FIX: Initialize last prices
        for asset in self.assets:
            try:
                self.last_prices[asset] = float(self.df.iloc[self.current_step].loc[(asset, 'close')])
            except:
                self.last_prices[asset] = 1000.0  # Default fallback price
        
        # IMPORTANT FIX: Add initial positions to history
        initial_positions = {asset: {'size': 0, 'entry_price': 0} for asset in self.assets}
        self.positions_history.append(initial_positions)
        
        # IMPORTANT FIX: Initialize position tracking variables
        self.position_duration = {asset: 0 for asset in self.assets}
        self.position_profits = {asset: [] for asset in self.assets}
        
        # Initialize action vector
        self.last_action_vector = np.zeros(len(self.assets))
        
        # Reset tracking counters
        self.consecutive_no_trade_steps = 0
        
        # Log reset
        logger.info(f"Environment reset: window_size={self.window_size}, initial_balance={self.initial_balance}")
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Execute trading step with the given action"""
        try:
            # CRITICAL FIX: Store initial portfolio value for reward calculation
            initial_portfolio = self._calculate_portfolio_value()
            initial_positions = copy.deepcopy(self.positions)
            
            # Check if we've reached the end of data
            if self.current_step >= len(self.df) - 1:
                # Return final state with done flag
                logger.info(f"Reached end of data at step {self.current_step}")
                return self._get_observation(), 0, True, False, self._get_info({})
            
            # Advance to next step
            self.current_step += 1
            
            # # Log that we're moving to the next step
            # if self.verbose:
            #     logger.info(f"Advancing to step {self.current_step} / {len(self.df)-1}")
            
            # ENHANCED: Process action and detect changes for trade frequency tracking
            action_vector = np.array(action).flatten()
            position_changes = np.abs(action_vector - self.last_action_vector)
            significant_change = np.any(position_changes > 0.1)  # Consider >10% change as significant
            self.last_action_vector = action_vector.copy()
            
            # CRITICAL FIX: Normalize action vector for multi-asset allocation
            # This ensures the total allocation across all assets is properly distributed
            total_allocation = np.sum(np.abs(action_vector))
            if total_allocation > 1e-8:
                normalized_allocation = np.abs(action_vector) / total_allocation
            else:
                normalized_allocation = np.ones_like(action_vector) / len(action_vector)
            
            # ENHANCED: Apply uncertainty-based scaling to actions
            uncertainty_scaled_action = self._apply_uncertainty_scaling(action_vector)
            
            # Execute trades based on uncertainty-scaled action
            trades = []
            trades_executed = False
            
            # Calculate risk metrics before executing trades
            risk_metrics = self._calculate_risk_metrics()
            
            # Process each asset's action
            for i, asset in enumerate(self.assets):
                # Get current position size for this asset
                current_size = self.positions[asset]['size']
                
                # Get the action value for this asset (-1 to 1)
                signal = float(uncertainty_scaled_action[i])
                
                # Get the current price
                price = self._get_mark_price(asset)
                
                # Update the last price in position data
                self.positions[asset]['last_price'] = price
                
                # CRITICAL FIX: Convert signal to target position size with proper allocation
                # Use normalized allocation to distribute leverage across assets
                asset_allocation = normalized_allocation[i]
                # Apply minimum leverage of 1.0x to comply with DEX requirements
                raw_leverage = abs(signal) * self.max_leverage
                # Ensure each asset has at least 1.0x leverage if signal is strong enough
                min_leverage = 1.0 if abs(signal) > 0.1 else 0.0  # Only apply minimum if signal is significant
                target_leverage = max(raw_leverage * asset_allocation, min_leverage)
                
                # Store the target leverage in a way that can be accessed during trade execution
                if abs(signal) > 0.1:  # Only update leverage when we have a significant signal
                    # This will be used in execute_trades to set the actual leverage
                    self.positions[asset]['target_leverage'] = target_leverage
                
                direction = np.sign(signal) if np.abs(signal) > 1e-8 else 0
                portfolio_value = self._calculate_portfolio_value()
                target_value = direction * target_leverage * portfolio_value
                
                # Enforce risk limits for position concentration
                if self.risk_engine:
                    max_position_value = portfolio_value * self.risk_engine.risk_limits.position_concentration
                    if abs(target_value) > max_position_value:
                        target_value = max_position_value * direction
                        # if self.verbose:
                        #     logger.info(f"Position size for {asset} limited by concentration risk: {target_value:.2f}")
                
                # Convert target value to size with sanity checks
                if price > 0:
                    target_size = target_value / price
                    
                    # CRITICAL FIX: Add asset-specific position limits
                    # Calculate max asset value based on portfolio size and leverage
                    max_asset_value = min(portfolio_value * self.max_leverage, 2000000)  # Cap at $2M for safety
                    
                    # Calculate max units for each asset based on its price
                    # More liquid assets can have larger positions
                    max_asset_units = {
                        'BTCUSDT': max_asset_value / price * 0.8,  # 80% of max for BTC (most liquid)
                        'ETHUSDT': max_asset_value / price * 0.7,  # 70% of max for ETH 
                        'SOLUSDT': max_asset_value / price * 0.5,  # 50% of max for SOL (less liquid)
                    }.get(asset, max_asset_value / price * 0.3)  # Default 30% for other assets
                    
                    # Additional hard cap based on standard lot sizes for each asset
                    hard_caps = {
                        'BTCUSDT': 100,      # Hard cap of 100 BTC 
                        'ETHUSDT': 1000,     # Hard cap of 1000 ETH
                        'SOLUSDT': 5000,     # Hard cap of 5000 SOL
                    }.get(asset, 10000)      # Default cap for other assets
                    
                    # Use the smaller of the two limits
                    max_asset_units = min(max_asset_units, hard_caps)
                    
                    if abs(target_size) > max_asset_units:
                        target_size = max_asset_units * direction
                        if self.verbose:
                            logger.debug(f"Capping target {asset} size from {target_size:.2f} to {max_asset_units * direction:.2f} units")
                else:
                    target_size = 0
                
                size_diff = target_size - current_size
                
                # Only create trade if the size difference is significant
                min_trade_size = 1e-6  # Increase minimum trade size threshold
                if abs(size_diff) > min_trade_size:
                    # Add the trade to the list
                    trades.append({
                        'asset': asset,
                        'size_change': size_diff,
                        'direction': np.sign(size_diff),
                        'current_price': price
                    })
                    # if self.verbose:
                    #     logger.info(f"Created trade for {asset}: {size_diff:.6f} units at ${price:.2f}")
            
            # Simulate trades to check risk limits
            simulation_result = self._simulate_trades(trades)
            
            # Execute trades if no risk violations, or execute scaled trades if available
            if not simulation_result.get('risk_limit_exceeded', False):
                # Execute the trades
                self._execute_trades(trades)
                trades_executed = True
                if len(trades) > 0:
                    logger.debug(f"Executed {len(trades)} trades at step {self.current_step}")
            elif 'scaled_trades' in simulation_result and simulation_result['scaled_trades']:
                # Execute scaled trades that comply with risk limits
                logger.info(f"Executing scaled trades to comply with risk limits")
                self._execute_trades(simulation_result['scaled_trades'])
                trades_executed = True
            else:
                logger.debug(f"No trades executed due to risk limits")
                trades_executed = False
            
            # ENHANCED: Update trade counts for frequency tracking
            self.trade_counts.append(1 if trades_executed else 0)
            
            # ENHANCED: Update consecutive no trade steps counter
            if trades_executed:
                self.consecutive_no_trade_steps = 0  # Reset counter when trades are executed
            else:
                self.consecutive_no_trade_steps += 1  # Increment counter when no trades
            
            # CRITICAL FIX: Apply funding costs and update positions before calculating final portfolio value
            self._update_positions()
            
            # CRITICAL FIX: Calculate portfolio value after trades and position updates
            portfolio_value = self._calculate_portfolio_value()
            
            # Log portfolio value after all updates
            logger.debug(f"Step {self.current_step} complete. Portfolio value: ${portfolio_value:.2f} (change: ${portfolio_value - initial_portfolio:.2f})")
            
            # CRITICAL FIX: Calculate actual PnL for this step
            total_pnl = portfolio_value - initial_portfolio
            
            # Update risk metrics after executing trades
            risk_metrics = self._calculate_risk_metrics()
            
            # ENHANCED: Calculate holding time bonus/penalty
            holding_reward = self._calculate_holding_time_reward()
            
            # CRITICAL FIX: Calculate reward using the risk-adjusted reward function
            reward = self._calculate_risk_adjusted_reward(total_pnl, risk_metrics)
            
            # Check if episode is done
            done = self._is_done()
            
            # CRITICAL FIX: Gymnasium requires terminated and truncated flags
            terminated = done
            truncated = False  # We don't use truncation in this environment
            
            # Get info dictionary
            info = self._get_info(risk_metrics)
            
            # CRITICAL FIX: Ensure trades_executed flag is set correctly
            info['trades_executed'] = trades_executed
            
            # CRITICAL FIX: Add more detailed trade information
            info['recent_trades_count'] = len([t for t in self.trades if t['timestamp'] == self.current_step])
            info['total_trades'] = len(self.trades)
            info['portfolio_change'] = portfolio_value - initial_portfolio
            
            # Add more position information
            active_positions = {}
            for asset, pos in self.positions.items():
                if abs(pos['size']) > 1e-8:
                    active_positions[asset] = pos['size']
            
            info['positions'] = self.positions
            info['active_positions'] = active_positions
            
            # CRITICAL FIX: Return 5-tuple format for Gymnasium
            return self._get_observation(), reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            traceback.print_exc()
            # Return a default observation, negative reward, done=True, and error info
            # CRITICAL FIX: Return 5-tuple format for Gymnasium
            return self._get_observation(), -1.0, True, False, {"error": str(e)}

    def _simulate_trades(self, trades: List[Dict]) -> Dict:
        """Simulate trades to check risk limits"""
        try:
            simulated_positions = copy.deepcopy(self.positions)
            
            for trade in trades:
                asset = trade['asset']
                
                # Handle different trade dictionary structures
                if 'size_change' in trade:  
                    size_change = trade['size_change']
                    
                    # Update simulated position
                    if asset in simulated_positions:
                        simulated_positions[asset]['size'] += size_change
                    else:
                        # Initialize if missing
                        mark_price = self._get_mark_price(asset)
                        simulated_positions[asset] = {
                            'size': size_change,
                            'entry_price': mark_price,
                            'last_price': mark_price
                        }
            
            # Calculate simulated risk metrics
            portfolio_value = self.balance  # Start with cash balance
            gross_exposure = 0  # Total absolute exposure (for risk limits)
            net_exposure = 0    # Net directional exposure (for leverage direction)
            asset_values = {}
            
            for asset, position in simulated_positions.items():
                # Skip assets with no position
                if abs(position['size']) < 1e-8:
                    continue
                    
                # Get current price
                price = self._get_mark_price(asset)
                
                # CRITICAL FIX: Calculate unrealized PnL correctly for both long and short positions
                unrealized_pnl = position['size'] * (price - position['entry_price'])
                portfolio_value += unrealized_pnl
                
                # Calculate both gross and net exposure
                position_value = position['size'] * price  # With sign (negative for shorts)
                position_exposure = abs(position_value)    # Absolute value (for risk)
                
                gross_exposure += position_exposure  # Always positive (for risk limits)
                net_exposure += position_value       # Can be negative (for directional leverage)
                asset_values[asset] = position_exposure
            
            # Ensure we don't divide by zero and portfolio value isn't extremely negative
            portfolio_value = max(portfolio_value, self.initial_balance * 0.01)
            
            # Calculate leverage - now with proper sign for direction
            # For risk purposes, we use gross leverage (always positive)
            gross_leverage = gross_exposure / portfolio_value
            
            # For tracking directional exposure, we use net leverage (can be negative)
            net_leverage = net_exposure / portfolio_value
            
            # CRITICAL FIX: Enforce maximum leverage limit on gross leverage
            # This ensures risk limits are enforced regardless of direction
            max_allowed_leverage = self.max_leverage  # Use configured max_leverage without hardcoding
            
            if gross_leverage > max_allowed_leverage:
                # Scale down positions to achieve target leverage
                scale_factor = max_allowed_leverage / gross_leverage
                
                # Create scaled trades
                scaled_trades = []
                for trade in trades:
                    if 'size_change' in trade:
                        scaled_trade = trade.copy()
                        scaled_trade['size_change'] = trade['size_change'] * scale_factor
                        if abs(scaled_trade['size_change']) > 1e-8:  # Only include non-zero trades
                            scaled_trades.append(scaled_trade)
                
                # Recalculate leverage with scaled trades
                gross_leverage = max_allowed_leverage
                
                # Log leverage scaling
                logger.warning(f"Scaling down trades to maintain leverage below {max_allowed_leverage:.2f}x (was {gross_leverage:.2f}x)")
                
                return {
                    'portfolio_value': portfolio_value,
                    'gross_leverage': gross_leverage,
                    'net_leverage': net_leverage * scale_factor,  # Scale net leverage too
                    'max_concentration': max(asset_values.values()) / portfolio_value if asset_values else 0,
                    'risk_limit_exceeded': True,
                    'exceeded_limits': [f"Leverage {gross_leverage:.2f}x > {max_allowed_leverage:.2f}x"],
                    'scaled_trades': scaled_trades
                }
            
            # Calculate position concentration
            max_concentration = 0
            max_asset = ""
            for asset, value in asset_values.items():
                concentration = value / portfolio_value
                if concentration > max_concentration:
                    max_concentration = concentration
                    max_asset = asset
            
            # CRITICAL FIX: Check if we exceed risk limits
            risk_limit_exceeded = False
            exceeded_limits = []
            
            # Check leverage limit
            if gross_leverage > self.max_leverage:
                risk_limit_exceeded = True
                exceeded_limits.append(f"Leverage {gross_leverage:.2f}x > {self.max_leverage:.2f}x")
            
            # Check concentration limit
            concentration_limit = 0.4  # Default, can be overridden by risk engine
            if self.risk_engine:
                concentration_limit = self.risk_engine.risk_limits.position_concentration
                
            if max_concentration > concentration_limit:
                risk_limit_exceeded = True
                exceeded_limits.append(f"Concentration {max_concentration:.2%} > {concentration_limit:.2%} for {max_asset}")
            
            # CRITICAL FIX: Scale down trades if they would exceed risk limits
            scaled_trades = []
            if risk_limit_exceeded and len(trades) > 0:
                # logger.warning(f"Risk limits would be exceeded: {', '.join(exceeded_limits)}")
                
                # Calculate scaling factor
                scale_factor = 0.8  # Default scale down by 20%
                
                if gross_leverage > self.max_leverage:
                    # Scale to get within leverage limit
                    leverage_scale = (self.max_leverage * 0.9) / gross_leverage
                    scale_factor = min(scale_factor, leverage_scale)
                
                if max_concentration > concentration_limit:
                    # Scale to get within concentration limit
                    concentration_scale = (concentration_limit * 0.9) / max_concentration
                    scale_factor = min(scale_factor, concentration_scale)
                
                # Apply scaling to all trades
                for trade in trades:
                    if 'size_change' in trade:
                        scaled_trade = trade.copy()
                        scaled_trade['size_change'] = trade['size_change'] * scale_factor
                        if abs(scaled_trade['size_change']) > 1e-8:  # Only include non-zero trades
                            scaled_trades.append(scaled_trade)
                
                logger.info(f"Scaling trades by factor {scale_factor:.4f} to comply with risk limits")
                return {
                    'portfolio_value': portfolio_value,
                    'gross_leverage': gross_leverage,
                    'net_leverage': net_leverage,
                    'max_concentration': max_concentration,
                    'risk_limit_exceeded': risk_limit_exceeded,
                    'exceeded_limits': exceeded_limits,
                    'scaled_trades': scaled_trades
                }
            
            # Return simulated metrics
            return {
                'portfolio_value': portfolio_value,
                'gross_leverage': gross_leverage,
                'net_leverage': net_leverage,
                'max_concentration': max_concentration,
                'risk_limit_exceeded': risk_limit_exceeded,
                'exceeded_limits': exceeded_limits,
                'scaled_trades': trades  # No scaling needed
            }
            
        except Exception as e:
            logger.error(f"Error in simulating trades: {str(e)}")
            return {
                'portfolio_value': self.balance,
                'gross_leverage': 0,
                'net_leverage': 0,
                'max_concentration': 0,
                'risk_limit_exceeded': True,
                'exceeded_limits': ["Error in simulation"],
                'scaled_trades': []  # Return empty list on error
            }
        
    def _execute_trades(self, trades: List[Dict]):
        """Smart order execution with transaction cost model"""
        try:
            # CRITICAL FIX: Calculate portfolio value at the beginning of trade execution
            portfolio_value = self._calculate_portfolio_value()
            
            for trade in trades:
                asset = trade['asset']
                
                # Safely extract trade parameters with defaults
                direction = trade.get('direction', 1 if trade.get('size_change', 0) > 0 else -1)
                leverage = trade.get('leverage', 1.0)
                risk_limits = trade.get('risk_limits', {'max_slippage': 0.001, 'max_impact': 0.002})
                
                # Get execution parameters (default to full execution)
                execution_params = trade.get('execution_params', [1.0])
                
                # CRITICAL FIX: Capture original position for proper PnL calculation
                original_position = self.positions[asset]['size']
                original_entry_price = self.positions[asset]['entry_price']
                
                # If we have size_change, use that directly
                if 'size_change' in trade and trade['size_change'] != 0:
                    # Get current price
                    mark_price = self._get_mark_price(asset)
                    size_change = trade['size_change']
                    
                    # CRITICAL FIX: Calculate monetary value of the position change
                    position_change_value = size_change * mark_price
                
                    # Apply transaction costs
                    # Apply slippage model - implementation shortfall
                    price_impact = self._estimate_price_impact(asset, position_change_value)
                    execution_price = mark_price * (1 + price_impact * direction)
                    
                    # Apply transaction costs
                    total_cost = abs(position_change_value) * (
                        self.commission + 
                        price_impact + 
                        self._get_spread_cost(asset)
                    )
                
                    # CRITICAL FIX: If closing a position, calculate realized PnL
                    realized_pnl = 0.0
                    if original_position != 0 and ((original_position > 0 and size_change < 0) or 
                                                  (original_position < 0 and size_change > 0)):
                        # We're reducing or closing position, calculate PnL on closed portion
                        if abs(size_change) >= abs(original_position):
                            # Fully closing or flipping
                            if original_position > 0:  # Long position
                                realized_pnl = original_position * (execution_price - original_entry_price)
                            else:  # Short position
                                realized_pnl = original_position * (original_entry_price - execution_price)
                        else:
                            # Partial close
                            closed_size = min(abs(original_position), abs(size_change))
                            if original_position > 0:  # Long position
                                realized_pnl = closed_size * (execution_price - original_entry_price)
                            else:  # Short position
                                realized_pnl = closed_size * (original_entry_price - execution_price)
                            
                    # Adjust realized PnL by commission costs
                    realized_pnl -= abs(size_change) * execution_price * self.commission
                    
                    # CRITICAL FIX: Update balance with realized PnL and costs
                    self.balance += realized_pnl - total_cost
                    
                    # Update position size
                    self.positions[asset]['size'] += size_change
                    
                    # CRITICAL FIX: Update entry price properly using weighted average
                    if self.positions[asset]['size'] != 0:
                        if abs(size_change) > abs(original_position) and np.sign(size_change) != np.sign(original_position):
                            # Direction flipped, use new price
                            self.positions[asset]['entry_price'] = execution_price
                        elif original_position != 0 and np.sign(size_change) == np.sign(original_position):
                            # Adding to position, calculate weighted average
                            total_size = abs(self.positions[asset]['size'])
                            original_value = abs(original_position) * original_entry_price
                            new_value = abs(size_change) * execution_price
                            self.positions[asset]['entry_price'] = (original_value + new_value) / total_size
                        elif original_position == 0:
                            # New position
                            self.positions[asset]['entry_price'] = execution_price
                    
                    self.total_costs += total_cost
                    
                    # INDUSTRY-LEVEL FIX: Properly handle leverage for DEX-style trading
                    position_size = self.positions[asset]['size']  # Current position size after update
                    
                    # Use target leverage when establishing/modifying position
                    if abs(position_size) > 1e-8:  # Only for non-zero positions
                        # Get the target leverage from the stored value during signal processing
                        target_leverage = self.positions[asset].get('target_leverage', 0.0)
                        
                        # For new positions or when adding to position, use the target leverage
                        if original_position == 0 or (np.sign(original_position) == np.sign(size_change)):
                            # When opening or increasing position, use the target leverage
                            self.positions[asset]['leverage'] = max(target_leverage, 1.0)  # Minimum 1.0x for DEX
                        
                        # For existing positions being reduced, keep the existing leverage
                        # This maintains the leverage when taking partial profits
                        
                        # Ensure leverage is capped at max_leverage
                        self.positions[asset]['leverage'] = min(self.positions[asset]['leverage'], self.max_leverage)
                        
                        # Use the position's leverage for reporting
                        actual_leverage = self.positions[asset]['leverage']
                    else:
                        # Zero position has zero leverage
                        self.positions[asset]['leverage'] = 0.0
                        actual_leverage = 0.0
                    
                    # Add trade to history
                    self.trades.append({
                        'timestamp': self.current_step,
                        'asset': asset,
                        'size': size_change,
                        'price': execution_price,
                        'cost': total_cost,
                        'realized_pnl': realized_pnl,  # CRITICAL FIX: Change 'pnl' to 'realized_pnl' for consistency
                        'leverage': actual_leverage  # Use the calculated actual leverage
                    })
                    
                    # Add leverage information to logging
                    if self.verbose:
                        logger.info(f"Trade executed: {asset} {size_change:.6f} @ {execution_price:.2f}, " +
                                   f"Cost: {total_cost:.2f}, PnL: {realized_pnl:.2f}, Leverage: {actual_leverage:.2f}x")
                else:
                    # Here we would handle the old trade format, but this branch is deprecated
                    # and shouldn't be called with our new trading logic
                    logger.warning("Deprecated trade format used without size_change")
        except Exception as e:
            logger.error(f"Error in _execute_trades: {str(e)}")
            traceback.print_exc()
                    
    def _update_positions(self):
        """Update positions with mark-to-market and funding rates"""
        try:
            total_pnl = 0.0
            portfolio_value_before = self._calculate_portfolio_value()
            
            # CRITICAL FIX: Track position metrics
            for asset in self.assets:
                position_size = self.positions[asset]['size']
                
                if abs(position_size) > 1e-8:
                    # Position exists, increment duration
                    self.position_duration[asset] += 1
                    
                    # Calculate profit/loss for this position
                    entry_price = self.positions[asset]['entry_price']
                    current_price = self.positions[asset]['last_price']
                    if entry_price > 0:  # Avoid division by zero
                        pnl_pct = (current_price / entry_price - 1) * np.sign(position_size)
                        self.position_profits[asset].append(pnl_pct)
                else:
                    # No position, reset duration and profits
                    self.position_duration[asset] = 0
                    self.position_profits[asset] = []
            
            # CRITICAL FIX: Calculate total portfolio value before funding
            portfolio_value_before_funding = self._calculate_portfolio_value()
            
            # Update positions with funding rates
            for asset in self.assets:
                try:
                    position = self.positions[asset]
                    position_size = position['size']
                    
                    # Skip updating if no position
                    if abs(position_size) <= 1e-8:
                        continue
                        
                    # Update funding rates based on current data
                    funding_rate = self._get_funding_rate(asset)
                    self.funding_rates[asset] = funding_rate
                    
                    # Update mark price
                    mark_price = self._get_mark_price(asset)
                    
                    # Skip if mark price is zero or not positive
                    if mark_price <= 0:
                        continue
                    
                    # Calculate time-weighted funding (8-hourly rate per step)
                    # Apply funding fee multiplier to control intensity
                    # Realistic funding rates are ~0.01% per 8 hours
                    funding_fee = position_size * mark_price * funding_rate * self.funding_fee_multiplier
                    
                    # Track funding costs over time
                    self.funding_accrued[asset] += funding_fee
                    
                    # Apply funding fee to balance
                    funding_cost = funding_fee
                    
                    # Update position value with funding
                    # Long positions pay funding when rate is positive
                    # Short positions pay funding when rate is negative
                    if (position_size > 0 and funding_rate > 0) or (position_size < 0 and funding_rate < 0):
                        # Position pays funding
                        self.balance -= abs(funding_cost)
                    else:
                        # Position receives funding
                        self.balance += abs(funding_cost)
                    
                    # Update last price
                    position['last_price'] = mark_price
                    
                    # Calculate unrealized PnL for this update
                    unrealized_pnl = position_size * (mark_price - position['entry_price'])
                    total_pnl += unrealized_pnl

                except Exception as e:
                    logger.error(f"Error updating position for {asset}: {str(e)}")
                    continue
            
            # Calculate current portfolio value after updates
            current_portfolio_value = self._calculate_portfolio_value()
            
            # CRITICAL FIX: Check for liquidation condition
            maintenance_threshold = self.initial_balance * self.maintenance_margin
            
            # SAFETY IMPROVEMENT: Also trigger liquidation if portfolio value drops below -50% of initial balance
            # This prevents extreme negative portfolio values
            early_liquidation_threshold = -self.initial_balance * 0.5
            
            if current_portfolio_value < maintenance_threshold or current_portfolio_value < early_liquidation_threshold:
                # Portfolio value below maintenance margin or extremely negative, liquidate all positions
                if current_portfolio_value < maintenance_threshold:
                    logger.warning(f"LIQUIDATION TRIGGERED: Portfolio value (${current_portfolio_value:.2f}) below maintenance margin (${maintenance_threshold:.2f})")
                else:
                    logger.warning(f"EMERGENCY LIQUIDATION TRIGGERED: Portfolio value (${current_portfolio_value:.2f}) extremely negative, closing all positions")
                
                # Close all positions
                self._close_all_positions()
                
                # Set liquidation flag
                self.liquidated = True
                
                # Update portfolio value after liquidation
                current_portfolio_value = self._calculate_portfolio_value()
                
                # Apply liquidation penalty (1% of initial balance)
                liquidation_penalty = self.initial_balance * 0.01
                self.balance -= liquidation_penalty
                logger.warning(f"Applied liquidation penalty: ${liquidation_penalty:.2f}")
            
            # CRITICAL FIX: Add to portfolio history after all updates
            self.portfolio_history.append({
                'step': self.current_step,
                'timestamp': get_utc_now() if 'get_utc_now' in globals() else datetime.now(),
                'balance': self.balance,
                'value': current_portfolio_value,
                'return': 0.0,  # Will be calculated in _update_history
                'leverage': 0.0, # Will be calculated in _update_history
                'drawdown': 0.0, # Will be calculated in _update_history
                'exposure': sum(abs(p['size'] * self._get_mark_price(a)) for a, p in self.positions.items())
            })
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error in _update_positions: {str(e)}")
            return 0.0
            
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value including positions and unrealized PnL"""
        try:
            # Start with cash balance
            total_value = self.balance
            
            # Add value of all positions
            total_position_value = 0.0
            total_unrealized_pnl = 0.0
            
            for asset, position in self.positions.items():
                try:
                    mark_price = self._get_mark_price(asset)
                    
                    # CRITICAL FIX: Calculate position value correctly for perpetual futures
                    position_size = position['size']
                    entry_price = position['entry_price']
                    
                    # Skip extremely small positions
                    if abs(position_size) <= 1e-8:
                        continue
                        
                    # For perpetual futures, the total value is:
                    # Cash balance + Unrealized PnL
                    # The unrealized PnL is: position_size * (mark_price - entry_price)
                    unrealized_pnl = position_size * (mark_price - entry_price)
                    
                    # EXTREME SAFETY CHECK: Limit maximum possible loss per position
                    # No position should lose more than 3x the initial balance
                    max_loss_per_position = self.initial_balance * 3
                    if unrealized_pnl < -max_loss_per_position:
                        logger.warning(f"Extreme loss detected in {asset} position: ${unrealized_pnl:.2f}, capping at ${-max_loss_per_position:.2f}")
                        unrealized_pnl = -max_loss_per_position
                    
                    # The position value is position_size * mark_price (absolute value)
                    position_value = abs(position_size * mark_price)
                    
                    # Add to totals
                    total_position_value += position_value  # This is for tracking only
                    total_unrealized_pnl += unrealized_pnl
                    
                    # Log details for significant positions
                    if abs(position_size) > 0.001:
                        logger.debug(f"Position value for {asset}: size={position_size:.6f} units, "
                                   f"entry=${entry_price:.2f}, mark=${mark_price:.2f}, "
                                   f"position_value=${position_value:.2f}, unrealized_pnl=${unrealized_pnl:.2f}")
                        
                except Exception as e:
                    logger.error(f"Error calculating position value for {asset}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Final portfolio value is cash balance plus unrealized PnL
            final_value = self.balance + total_unrealized_pnl
            
            # EXTREME SAFETY CHECK: Limit the maximum possible portfolio loss
            # Portfolio shouldn't lose more than 5x initial balance
            min_possible_value = -self.initial_balance * 5
            if final_value < min_possible_value:
                logger.warning(f"Extreme portfolio loss detected: ${final_value:.2f}, capping at ${min_possible_value:.2f}")
                final_value = min_possible_value
            
            # Log detailed breakdown
            logger.debug(f"Portfolio value: ${final_value:.2f} = Cash (${self.balance:.2f}) + Unrealized PnL (${total_unrealized_pnl:.2f})")
            logger.debug(f"Total position exposure: ${total_position_value:.2f}")
            
            return final_value
            
        except Exception as e:
            logger.error(f"Error in _calculate_portfolio_value: {str(e)}")
            traceback.print_exc()
            # In case of error, return cash balance as fallback
            return self.balance

    def _get_funding_rate(self, asset: str) -> float:
        """Get current funding rate for an asset"""
        try:
            # Try to get funding rate from market data
            current_data = self.df.iloc[self.current_step]
            funding_rate = current_data.get((asset, 'funding_rate'), 0.0)
            
            if isinstance(funding_rate, (pd.Series, pd.DataFrame)):
                funding_rate = funding_rate.iloc[0]
            
            return float(funding_rate)
        except Exception as e:
            logger.error(f"Error getting funding rate for {asset}: {str(e)}")
            return 0.0001  # Default funding rate (0.01% per 8 hours)
        
    def _calculate_risk_metrics(self, refresh_metrics=False):
        """Calculate risk metrics for the current state"""
        if len(self.positions) == 0 and not refresh_metrics and hasattr(self, 'risk_metrics'):
            return self.risk_metrics
        
        try:
            # Start with current balance
            portfolio_value = self.balance
            gross_exposure = 0
            net_exposure = 0
            positions_data = []  # For metrics that need to process all positions
            asset_values = {}    # For concentration calculation
            
            # Process each position
            for asset, position in self.positions.items():
                # Skip positions with negligible size
                if abs(position['size']) < 1e-8:
                    continue
                
                try:
                    # Get current price
                    mark_price = self._get_mark_price(asset)
                    
                    # Calculate unrealized PnL
                    unrealized_pnl = position['size'] * (mark_price - position['entry_price'])
                    portfolio_value += unrealized_pnl
                    
                    # Calculate both gross and net exposure
                    position_value = position['size'] * mark_price  # With sign (negative for shorts)
                    position_exposure = abs(position_value)         # Absolute value (for risk limits)
                    
                    gross_exposure += position_exposure  # Always positive (for risk limits)
                    net_exposure += position_value       # Can be negative (for directional leverage)
                    
                    asset_values[asset] = position_exposure
                    
                    # Prepare position data for risk engine
                    position_data = {
                        'asset': asset,
                        'size': position['size'],
                        'entry_price': position['entry_price'],
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'position_value': position_value,
                        'position_exposure': position_exposure,
                    }
                    positions_data.append(position_data)
                    
                except Exception as e:
                    logger.error(f"Error processing position {asset}: {e}")
            
            # Calculate portfolio value for leverage (with safety check)
            portfolio_value_for_leverage = max(portfolio_value, self.initial_balance * 0.1)
            
            # Calculate both types of leverage
            gross_leverage = gross_exposure / portfolio_value_for_leverage
            net_leverage = net_exposure / portfolio_value_for_leverage
            
            # Initialize all risk metrics to zero in case we don't have a risk engine
            metrics = {
                'total_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'gross_leverage': gross_leverage,
                'net_leverage': net_leverage,  # Can be negative for short-biased portfolios
                'leverage_utilization': gross_leverage,  # Add this for compatibility with main_opt.py
                'max_drawdown': getattr(self, 'max_drawdown', 0),
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'total_pnl': 0,
                'pnl_volatility': 0,
                'portfolio_value': portfolio_value,
                'balance': self.balance,
                'current_drawdown': max(0, 1 - portfolio_value / self.max_portfolio_value) if hasattr(self, 'max_portfolio_value') else 0,
            }
            
            # Additional metrics calculation through Risk Engine
            if self.risk_engine and refresh_metrics:
                try:
                    risk_metrics = self.risk_engine.calculate_risk_metrics(
                        positions=positions_data,
                        portfolio_value=portfolio_value,
                        balance=self.balance,
                        initial_balance=self.initial_balance,
                        # FIXED: portfolio_history is a list of dictionaries, extract values correctly
                        returns_history=[entry['value'] for entry in self.portfolio_history] if len(self.portfolio_history) > 0 else None,
                    )
                    # Update with risk engine metrics
                    metrics.update(risk_metrics)
                except Exception as e:
                    logger.error(f"Error calculating risk metrics: {e}")
            
            # Update history of leverage and drawdown
            if hasattr(self, 'leverage_history'):
                self.leverage_history.append(gross_leverage)
            else:
                self.leverage_history = [gross_leverage]
                
            if hasattr(self, 'net_leverage_history'):
                self.net_leverage_history.append(net_leverage)
            else:
                self.net_leverage_history = [net_leverage]
                
            # Calculate drawdown
            if not hasattr(self, 'max_portfolio_value'):
                self.max_portfolio_value = portfolio_value
            elif portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = portfolio_value
                
            current_drawdown = max(0, 1 - portfolio_value / self.max_portfolio_value)
            
            if hasattr(self, 'drawdown_history'):
                self.drawdown_history.append(current_drawdown)
            else:
                self.drawdown_history = [current_drawdown]
                
            # Update maximum drawdown
            if current_drawdown > getattr(self, 'max_drawdown', 0):
                self.max_drawdown = current_drawdown
                
            # Risk-adjusted ratios calculation (with safety checks)
            if len(self.portfolio_history) > 1:
                try:
                    # Calculate returns
                    returns = []
                    # FIXED: portfolio_history is a list of dictionaries, not a dictionary
                    # Extract portfolio values from the history
                    values = [entry['value'] for entry in self.portfolio_history]
                    for i in range(1, len(values)):
                        if values[i-1] > 0:
                            returns.append((values[i] - values[i-1]) / values[i-1])
                        else:
                            returns.append(0)
                    
                    # Calculate metrics if we have returns
                    if len(returns) > 0:
                        # Sharpe ratio
                        if np.std(returns) > 0:
                            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                            metrics['sharpe_ratio'] = sharpe
                        
                        # Sortino ratio (downside deviation)
                        negative_returns = [r for r in returns if r < 0]
                        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                            sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
                            metrics['sortino_ratio'] = sortino
                        
                        # Calmar ratio (return / max drawdown)
                        if self.max_drawdown > 0:
                            total_return = (portfolio_value / self.initial_balance) - 1
                            calmar = total_return / self.max_drawdown
                            metrics['calmar_ratio'] = calmar
                except Exception as e:
                    logger.error(f"Error calculating risk-adjusted ratios: {e}")
                    logger.exception("Detailed traceback for risk-adjusted ratios error:")
            
            # Save metrics for future reference
            self.risk_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error in _calculate_risk_metrics: {e}")
            # Return basic metrics in case of error
            return {
                'total_exposure': 0,
                'net_exposure': 0,
                'gross_leverage': 0,
                'net_leverage': 0,
                'portfolio_value': self.balance,
                'balance': self.balance,
            }
        
    def _calculate_risk_adjusted_reward(self, total_pnl: float, risk_metrics: Dict) -> float:
        """Calculate reward based on PnL and risk-adjusted metrics"""
        try:
            # CRITICAL FIX: Improved portfolio return calculation
            portfolio_value = risk_metrics.get('portfolio_value', self.balance)
            previous_value = portfolio_value - total_pnl  # Value before this step
            
            # Ensure previous value is positive to avoid division issues
            previous_value = max(previous_value, self.initial_balance * 0.01)
            
            # Calculate return as percentage of previous portfolio value
            portfolio_return = total_pnl / previous_value
            
            # CRITICAL FIX: Add sanity checks for unrealistic returns
            if abs(portfolio_return) > 0.1:  # >10% return in a single step is suspicious
                if abs(portfolio_return) > 0.5:  # >50% return is extremely unrealistic
                    # Apply very severe penalty for extremely unrealistic returns
                    if portfolio_return > 0:
                        portfolio_return = 0.05  # Cap positive return at 5%
                    else:
                        portfolio_return = -0.05  # Cap negative return at -5%
                else:
                    # Apply moderate penalty for unrealistic returns
                    if portfolio_return > 0:
                        portfolio_return = 0.1  # Cap positive return at 10%
                    else:
                        portfolio_return = -0.1  # Cap negative return at -10%
            
            # IMPROVED: Scale returns differently - increase reward for positive returns
            # and decrease penalty for negative returns to encourage more trading
            if portfolio_return > 0:
                base_reward = portfolio_return * 6.0  # Increased multiplier for positive returns
            else:
                base_reward = portfolio_return * 4.0  # Reduced multiplier for negative returns
            
            # Risk-adjusted components
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            sortino = risk_metrics.get('sortino_ratio', 0)
            calmar = risk_metrics.get('calmar_ratio', 0)
            
            # CRITICAL FIX: Add sanity checks for risk metrics
            sharpe = np.clip(sharpe, -5, 5)
            sortino = np.clip(sortino, -5, 5)
            calmar = np.clip(calmar, -5, 5)
            
            # IMPROVED: Give more weight to risk-adjusted metrics
            risk_reward = (sharpe + sortino + calmar) / 3.0 * 0.7  # Increased weight from 0.5 to 0.7
            
            # CRITICAL FIX: Penalize excessive risk more aggressively
            leverage_penalty = 0.0
            concentration_penalty = 0.0
            
            # Penalize high leverage relative to limits
            max_leverage = self.max_leverage
            current_leverage = risk_metrics.get('leverage_utilization', 0)
            if current_leverage > max_leverage * 0.5:  # Start penalizing at 50% of max
                leverage_ratio = current_leverage / max_leverage
                leverage_penalty = (leverage_ratio - 0.5) * 2.0  # More aggressive penalty
            
            # Penalize high concentration
            max_concentration = self.risk_engine.risk_limits.position_concentration if self.risk_engine else 0.4
            current_concentration = risk_metrics.get('max_concentration', 0)
            if current_concentration > max_concentration * 0.5:  # Start penalizing at 50% of limit
                concentration_ratio = current_concentration / max_concentration
                concentration_penalty = (concentration_ratio - 0.5) * 3.0  # Stronger concentration penalty
            
            # IMPROVED: Reduce drawdown penalty
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            drawdown_penalty = min(max_drawdown * 8.0, 1.5)  # Reduced from 10.0 to 8.0, cap at 1.5 instead of 2.0
            
            # IMPROVED: Strengthen trading activity incentive
            trade_count = len([t for t in self.trades if t['timestamp'] == self.current_step])
            trading_incentive = 0.0
            if trade_count == 0:
                # New penalty for no trading to encourage activity
                trading_incentive = -0.02
            elif 0 < trade_count <= 5:  # Reward moderate trading (1-5 trades per step)
                trading_incentive = 0.03 * trade_count  # Increased from 0.02
            elif trade_count > 5:  # Penalize excessive trading
                trading_incentive = 0.09 - (trade_count - 5) * 0.015  # Starts at 0.09 (increased) and decreases more slowly
            
            # IMPROVED: Balance penalty for negative balance
            balance_penalty = 0.0
            if self.balance < 0:
                balance_penalty = 2.5  # Reduced from 3.0
            elif self.balance < self.initial_balance * 0.5:
                balance_penalty = 0.8  # Reduced from 1.0
                
            # IMPROVED: Add bonus for maintaining balance above initial
            balance_bonus = 0.0
            if portfolio_value > self.initial_balance * 1.05:  # 5% above initial
                # Add increasing bonus for better performance, capped at 0.5
                balance_bonus = min((portfolio_value / self.initial_balance - 1) * 2, 0.5)
                
            # Combine all components
            reward = base_reward + risk_reward + trading_incentive + balance_bonus - leverage_penalty - concentration_penalty - drawdown_penalty - balance_penalty
            
            # Bound the reward to prevent extreme values
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted reward: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return -1.0
        
    def _calculate_reward(self, risk_metrics: Dict, total_pnl: float, holding_reward: float = 0.0) -> float:
        """Calculate the reward signal for the current step"""
        try:
            # Use the risk-adjusted reward as a base
            reward = self._calculate_risk_adjusted_reward(total_pnl, risk_metrics)
            
            # Add holding time bonus/penalty
            reward += holding_reward
            
            # ENHANCED: Encourage exploration by punishing static behavior
            if risk_metrics.get('positions_count', 0) == 0:
                # Penalize having no positions (encourage taking positions)
                reward -= 0.01
            
            # ENHANCED: If we've been inactive too long, increase the penalty
            if self.consecutive_no_trade_steps > 50:
                reward -= 0.1  # Strong penalty for extended inactivity
            
            # Apply risk limits penalties
            for violation, value in risk_metrics.items():
                if violation.endswith('_violation') and value:
                    # Apply stronger penalties for risk violations
                    reward -= 0.5
            
            # Bound the reward to prevent extreme values
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            # Return a default small negative reward on error
            return -0.1

    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        try:
            # Check if we've reached the end of data
            if self.current_step >= len(self.df) - 1:
                logger.info("Episode done: Reached end of data")
                return True
            
            # Check if liquidation has occurred
            if self.liquidated:
                logger.info("Episode done: Account liquidated due to insufficient margin")
                return True
            
            # Calculate current risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Check account depletion - dynamic threshold based on initial balance and max_drawdown
            # Allow for negative values but terminate on severe depletion
            severe_depletion_threshold = -self.initial_balance * self.max_drawdown
            if risk_metrics['portfolio_value'] <= severe_depletion_threshold:
                logger.info(f"Episode done: Account severely depleted (${risk_metrics['portfolio_value']:.2f}, threshold: ${severe_depletion_threshold:.2f})")
                return True
                
            # Check max drawdown exceeded
            if 'current_drawdown' in risk_metrics and risk_metrics['current_drawdown'] > self.max_drawdown:
                logger.info(f"Episode done: Max drawdown exceeded ({risk_metrics['current_drawdown']:.2%} > {self.max_drawdown:.2%})")
                return True
            
            # Check for risk limit violations
            if self.risk_engine:
                # Check VaR limit
                if 'var' in risk_metrics and risk_metrics['var'] > self.risk_engine.risk_limits.var_limit:
                    logger.info(f"Episode done: VaR limit exceeded ({risk_metrics['var']:.2%} > {self.risk_engine.risk_limits.var_limit:.2%})")
                    return True
                
                # Check for extended leverage violation
                if 'leverage_utilization' in risk_metrics and risk_metrics['leverage_utilization'] > self.max_leverage * 1.1:
                    logger.info(f"Episode done: Leverage limit exceeded ({risk_metrics['leverage_utilization']:.2f}x > {self.max_leverage * 1.1:.2f}x)")
                    return True
            
            # Continue the episode
            return False
            
        except Exception as e:
            logger.error(f"Error in _is_done: {str(e)}")
            return True  # Terminate on error

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

    def _execute_trade(self, asset_idx: int, signal: float, price: float) -> float:
        """Execute a trade for a single asset and return the PnL"""
        try:
            # Initialize target_leverage at the beginning to ensure it's always defined
            target_leverage = max(abs(signal) * self.max_leverage, 1.0)
            
            asset = self.assets[asset_idx]
            old_position = self.positions[asset]['size']
            old_entry_price = self.positions[asset]['entry_price']
            
            # Calculate current portfolio value with safety check
            portfolio_value = max(self._calculate_portfolio_value(), self.initial_balance * 0.1)
            
            # CRITICAL FIX: Add realistic position sizing constraints
            # Convert signal [-1, 1] to target leverage
            # Use minimum leverage of 1.0x to comply with DEX requirements
            
            # Determine direction (-1 or 1) from signal
            direction = np.sign(signal)
            if direction == 0 or abs(direction) < 0.001:  # Handle zero or near-zero signal case
                return 0.0  # No trade if no clear direction
            
            # CRITICAL FIX: Apply concentration limits
            if self.risk_engine:
                max_position_value = portfolio_value * self.risk_engine.risk_limits.position_concentration
            else:
                max_position_value = portfolio_value * 0.4  # Default to 40% if no risk engine
                
            # Calculate target position value with concentration limit
            target_value = portfolio_value * target_leverage * direction
            if abs(target_value) > max_position_value:
                target_value = max_position_value * direction
            
            # Convert to target position size
            target_size = target_value / price if price > 0 else 0
            
            # CRITICAL FIX: Add asset-specific position limits
            # Calculate max asset value based on portfolio size and leverage
            max_asset_value = min(portfolio_value * self.max_leverage, 2000000)  # Cap at $2M for safety
            
            # Calculate max units for each asset based on its price
            # More liquid assets can have larger positions
            max_asset_units = {
                'BTCUSDT': max_asset_value / price * 0.8,  # 80% of max for BTC (most liquid)
                'ETHUSDT': max_asset_value / price * 0.7,  # 70% of max for ETH 
                'SOLUSDT': max_asset_value / price * 0.5,  # 50% of max for SOL (less liquid)
            }.get(asset, max_asset_value / price * 0.3)  # Default 30% for other assets
            
            # Additional hard cap based on standard lot sizes for each asset
            # This prevents unrealistically large positions in any asset
            hard_caps = {
                'BTCUSDT': 100,      # Hard cap of 100 BTC 
                'ETHUSDT': 1000,     # Hard cap of 1000 ETH
                'SOLUSDT': 5000,     # Hard cap of 5000 SOL
            }.get(asset, 10000)      # Default cap for other assets
            
            # Use the smaller of the two limits
            max_asset_units = min(max_asset_units, hard_caps)
            
            if abs(target_size) > max_asset_units:
                logger.debug(f"Capping target {asset} size from {target_size:.2f} to {max_asset_units * np.sign(target_size):.2f} units")
                target_size = max_asset_units * np.sign(target_size)
            
            # Calculate size difference
            size_diff = target_size - old_position
            
            # Skip tiny trades
            min_trade_size = portfolio_value * 0.001 / price  # 0.1% of portfolio
            if abs(size_diff) < min_trade_size:
                return 0.0  # No meaningful trade
            
            # CRITICAL FIX: Ensure trade size is realistic and executable
            max_trade_size = portfolio_value / price  # Can't trade more than portfolio value
            if abs(size_diff) > max_trade_size:
                size_diff = max_trade_size * np.sign(size_diff)
            
            # Calculate commission with realistic transaction costs
            base_commission = abs(size_diff) * price * self.commission
            slippage_cost = abs(size_diff) * price * 0.0002  # 0.02% slippage
            total_cost = base_commission + slippage_cost
            
            # Check if balance can cover costs
            if total_cost > self.balance:
                # Scale down trade size if insufficient funds
                scale_factor = self.balance / (total_cost * 1.1)  # 10% safety margin
                size_diff *= scale_factor
                total_cost = abs(size_diff) * price * (self.commission + 0.0002)
            
            # Update balance and execute trade
            self.balance -= total_cost
            self.total_costs += total_cost
            
            # Calculate PnL from old position if closing or reducing
            pnl = 0.0
            if old_position != 0 and ((old_position > 0 and size_diff < 0) or 
                                      (old_position < 0 and size_diff > 0)):
                # Closing or reducing position
                if abs(size_diff) >= abs(old_position):  # Fully closing or flipping
                    if old_position > 0:  # Long position
                        pnl = old_position * (price - old_entry_price)
                    else:  # Short position
                        pnl = old_position * (old_entry_price - price)
                    position_size = size_diff + old_position
                    entry_price = price  # New entry price for remaining/flipped position
                else:  # Partially reducing
                    closed_size = min(abs(old_position), abs(size_diff))
                    if old_position > 0:  # Long position
                        pnl = closed_size * (price - old_entry_price)
                    else:  # Short position
                        pnl = closed_size * (old_entry_price - price)
                    position_size = old_position + size_diff
                    entry_price = old_entry_price  # Keep same entry for remaining
            else:
                # Increasing existing position or opening new
                if old_position == 0:
                    # New position
                    position_size = size_diff
                    entry_price = price
                else:
                    # Adding to existing position - calculate weighted average entry
                    position_size = old_position + size_diff
                    entry_price = (old_position * old_entry_price + size_diff * price) / position_size
            
            # CRITICAL FIX: Final sanity check on position size
            if abs(position_size) > max_asset_units:
                # logger.warning(f"Position size for {asset} exceeds limit after execution: {position_size}. Capping at {max_asset_units * np.sign(position_size)}")
                position_size = max_asset_units * np.sign(position_size)
            
            # Update position
            self.positions[asset]['size'] = position_size
            self.positions[asset]['entry_price'] = entry_price
            
            # Calculate leverage properly based on total position size and portfolio
            # FIXED: Use total position size (after trade) for leverage calculation, not just size_diff
            total_position_value = abs(position_size * price)
            
            # INDUSTRY-LEVEL FIX: Properly handle leverage for DEX-style trading
            if abs(position_size) > 1e-8:  # Only for non-zero positions
                # Get the target leverage from the stored value during signal processing
                target_leverage = self.positions[asset].get('target_leverage', 0.0)
                
                # For new positions or when adding to position, use the target leverage
                if old_position == 0 or (np.sign(old_position) == np.sign(size_diff)):
                    # When opening or increasing position, use the target leverage
                    self.positions[asset]['leverage'] = max(target_leverage, 1.0)  # Minimum 1.0x for DEX
                
                # For existing positions being reduced, keep the existing leverage
                # This maintains the leverage when taking partial profits
                
                # Ensure leverage is capped at max_leverage
                self.positions[asset]['leverage'] = min(self.positions[asset]['leverage'], self.max_leverage)
                
                # Use the position's leverage for reporting
                actual_leverage = self.positions[asset]['leverage']
            else:
                # Zero position has zero leverage
                self.positions[asset]['leverage'] = 0.0
                actual_leverage = 0.0
            
            # Add trade to history
            self.trades.append({
                'timestamp': self.current_step,
                'asset': asset,
                'size': size_diff,
                'price': price,
                'cost': total_cost,
                'realized_pnl': pnl,  # CRITICAL FIX: Change 'pnl' to 'realized_pnl' for consistency
                'leverage': actual_leverage  # Use the calculated actual leverage
            })
            
            # Add leverage information to logging
            if self.verbose:
                logger.info(f"Trade executed: {asset} {size_diff:.6f} @ {price:.2f}, " +
                           f"Cost: {total_cost:.2f}, PnL: {pnl:.2f}, Leverage: {actual_leverage:.2f}x")
            
            return pnl
            
        except Exception as e:
            logger.error(f"Error executing trade for asset idx {asset_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0  # Return no PnL on error

    def _get_mark_price(self, asset: str) -> float:
        """Get current mark price for an asset"""
        try:
            # For MultiIndex DataFrame, we need to use cross-section
            current_data = self.df.iloc[self.current_step]
            # Access using tuple for MultiIndex: (asset, 'close')
            price = current_data.loc[(asset, 'close')]
            
            # Handle if price is a Series or numpy array
            if isinstance(price, (pd.Series, np.ndarray)):
                price = float(price.iloc[0] if isinstance(price, pd.Series) else price[0])
            
            return float(price)
        except Exception as e:
            logger.error(f"Error getting mark price for {asset}: {str(e)}")
            # Return last known price or default
            return self.last_prices.get(asset, 1000.0)

    def _get_spread_cost(self, asset: str) -> float:
        """Estimate spread cost based on order book"""
        try:
            asset_data = self.df.xs(asset, axis=1, level='asset')
            if 'bid' in asset_data.columns and 'ask' in asset_data.columns:
                spread = (asset_data['ask'].iloc[self.current_step] - 
                         asset_data['bid'].iloc[self.current_step])
                return spread / asset_data['close'].iloc[self.current_step]
            return self.commission  # Default to base fee if no order book data
        except Exception as e:
            logger.error(f"Error getting spread cost for {asset}: {str(e)}")
            return self.commission
        
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
        """Return additional information about the environment"""
        info = {
            'step': self.current_step,
            'portfolio_value': self._calculate_portfolio_value(),
            'balance': self.balance,
            'gross_leverage': risk_metrics.get('gross_leverage', 0),  # Add gross leverage (always positive)
            'net_leverage': risk_metrics.get('net_leverage', 0),      # Add net leverage (can be negative for shorts)
            'positions': {asset: {'size': pos['size'], 'entry_price': pos['entry_price']} 
                          for asset, pos in self.positions.items()},
            'total_trades': len(self.trades),
            'risk_metrics': risk_metrics
        }
        
        # ENHANCED: Add position durations to info
        info['position_durations'] = self.position_duration.copy()
        
        # ENHANCED: Add uncertainty metrics to info if available
        if hasattr(self, 'uncertainty_metrics'):
            info['uncertainty'] = {asset: metrics['uncertainty_score'] 
                                  for asset, metrics in self.uncertainty_metrics.items()}
        
        return info

    def _apply_uncertainty_scaling(self, action_vector):
        """
        Scale actions based on market uncertainty and volatility
        to make position sizing more conservative in uncertain conditions
        """
        # Get the scaling factor, use default if not set
        scaling_factor = getattr(self, 'uncertainty_scaling_factor', 1.0)
        
        scaled_action = action_vector.copy()
        
        # Initialize uncertainty metrics if not yet done
        if not hasattr(self, 'uncertainty_metrics'):
            self.uncertainty_metrics = {asset: {
                'volatility_history': deque(maxlen=20),
                'avg_volatility': 0.0,
                'uncertainty_score': 0.5  # Start with middle uncertainty
            } for asset in self.assets}
        
        # Update uncertainty metrics for each asset
        for i, asset in enumerate(self.assets):
            try:
                # Get market data for this asset
                asset_data = self.df.iloc[max(0, self.current_step-30):self.current_step+1].xs(asset, level=0, axis=1)
                
                # Calculate metrics
                if 'close' in asset_data.columns:
                    # Calculate recent volatility
                    if len(asset_data) >= 5:
                        returns = asset_data['close'].pct_change().dropna()
                        current_vol = returns.std()
                        self.uncertainty_metrics[asset]['volatility_history'].append(current_vol)
                    
                    # Calculate average volatility over time
                    vol_history = self.uncertainty_metrics[asset]['volatility_history']
                    if vol_history:
                        self.uncertainty_metrics[asset]['avg_volatility'] = np.mean(vol_history)
                    
                    # Get recent market regime data if available
                    market_regime = 0.5  # Neutral by default
                    if 'market_regime' in asset_data.columns:
                        market_regime = asset_data['market_regime'].iloc[-1]
                    
                    # Determine if volatility is trending up
                    vol_trend = 0.0
                    if len(vol_history) >= 5:
                        recent_mean = np.mean(list(vol_history)[-3:])
                        older_mean = np.mean(list(vol_history)[:-3])
                        if np.isnan(recent_mean) or np.isnan(older_mean) or older_mean == 0:
                            vol_trend = 0.0  # Default to no trend when we have invalid data
                        else:
                            vol_trend = recent_mean / older_mean - 1
                    
                    # Calculate uncertainty score (0 = certain, 1 = uncertain)
                    # Base on:
                    # 1. Current volatility relative to average
                    # 2. Trending or ranging market
                    # 3. Volatility trend
                    # Adding safeguards for NaN and zero division
                    denominator = max(0.0001, self.uncertainty_metrics[asset]['avg_volatility'])
                    if np.isnan(current_vol) or np.isnan(denominator) or denominator == 0:
                        volatility_factor = 1.0  # Default to neutral when we have invalid data
                    else:
                        volatility_factor = min(current_vol / denominator, 3)
                    regime_factor = 0.5 if abs(market_regime - 0.5) < 0.2 else 0.0  # More uncertain in neutral regimes
                    trend_factor = max(0, min(vol_trend * 3, 1))  # More uncertain when volatility increasing
                    
                    uncertainty_score = 0.4 * volatility_factor + 0.3 * regime_factor + 0.3 * trend_factor
                    uncertainty_score = min(max(uncertainty_score, 0.1), 0.9)  # Bound between 10% and 90%
                    
                    # Apply scaling factor to uncertainty (higher scaling = more aggressive reduction)
                    # Only apply if regime awareness is enabled
                    if getattr(self, 'regime_aware', True):
                        uncertainty_score = min(uncertainty_score * scaling_factor, 0.95)
                    
                    # Store the uncertainty score
                    self.uncertainty_metrics[asset]['uncertainty_score'] = uncertainty_score
                    
                    # Scale action based on uncertainty (reduce size when uncertain)
                    if abs(action_vector[i]) > 0.05:  # Only scale meaningful positions
                        confidence = 1.0 - uncertainty_score
                        
                        # Apply non-linear scaling - keep small positions, reduce large ones more aggressively
                        # when uncertainty is high
                        raw_action = action_vector[i]
                        action_magnitude = abs(raw_action)
                        sign = np.sign(raw_action)
                        
                        # Only apply uncertainty scaling if regime awareness is enabled
                        if getattr(self, 'regime_aware', True):
                            # Scale more conservatively for large positions in uncertain conditions
                            if action_magnitude > 0.5 and uncertainty_score > 0.6:
                                scaled_magnitude = action_magnitude * (confidence ** 1.5)
                            else:
                                scaled_magnitude = action_magnitude * (confidence ** 0.8)
                            
                            scaled_action[i] = sign * scaled_magnitude
            
            except Exception as e:
                logger.error(f"Error in uncertainty scaling for {asset}: {str(e)}")
                # Fall back to original action
                continue
        
        return scaled_action

    def _update_history(self):
        """Update portfolio history"""
        try:
            # CRITICAL FIX: Use the actual portfolio value calculation method
            # which correctly handles both long and short positions
            current_value = self._calculate_portfolio_value()
            
            # Calculate total exposure (this is still needed for leverage)
            total_exposure = 0
            for asset, position in self.positions.items():
                mark_price = self._get_mark_price(asset)
                # Use absolute value for exposure calculation
                position_value = abs(position['size'] * mark_price)
                total_exposure += position_value
            
            # Safety check for portfolio value
            if current_value <= 0:
                # logger.warning("Portfolio value is zero or negative, setting metrics to zero")
                current_leverage = 0
                current_drawdown = 1  # Maximum drawdown
            else:
                # Calculate current leverage with bounds
                current_leverage = min(total_exposure / current_value, self.max_leverage)
                
                # Update peak value if current value is higher
                if current_value > self.peak_value:
                    self.peak_value = current_value
                
                # Calculate drawdown
                current_drawdown = 1 - current_value / self.peak_value if self.peak_value > 0 else 0
            
            # Calculate return for this step
            current_return = (current_value / self.initial_balance) - 1
            
            # CRITICAL FIX: Ensure all history collections are initialized
            if not hasattr(self, 'returns_history'):
                self.returns_history = deque(maxlen=10000)
            if not hasattr(self, 'positions_history'):
                self.positions_history = deque(maxlen=10000)
            if not hasattr(self, 'portfolio_history'):
                self.portfolio_history = deque(maxlen=10000)
            if not hasattr(self, 'historical_leverage'):
                self.historical_leverage = deque(maxlen=10000)
            if not hasattr(self, 'drawdown_history'):
                self.drawdown_history = deque(maxlen=10000)
            
            # Update all history collections
            self.returns_history.append(current_return)
            
            # Store position sizes for history
            position_sizes = {asset: pos['size'] for asset, pos in self.positions.items()}
            self.positions_history.append(position_sizes)
            
            # Always store metrics regardless of value
            self.portfolio_history.append({
                'step': self.current_step,
                'timestamp': get_utc_now(),
                'balance': self.balance,
                'value': current_value,
                'return': current_return,
                'leverage': current_leverage,
                'drawdown': current_drawdown,
                'exposure': total_exposure
            })
            
            # Always append metrics to history
            self.historical_leverage.append(current_leverage)
            self.drawdown_history.append(current_drawdown)
            
            # CRITICAL FIX: Calculate active positions for logging
            active_positions = sum(1 for pos in self.positions.values() if abs(pos['size']) > 1e-8)
            
            # Log metrics for debugging only if verbose is enabled or we have active positions
            if self.verbose and active_positions > 0:
                logger.debug(f"Portfolio metrics - Step {self.current_step}:")
                logger.debug(f"  Value: {current_value:.2f}")
                logger.debug(f"  Total Exposure: {total_exposure:.2f}")
                logger.debug(f"  Leverage: {current_leverage:.4f}")
                logger.debug(f"  Drawdown: {current_drawdown:.4f}")
                logger.debug(f"  Return: {current_return:.4f}")
                logger.debug(f"  Active Positions: {active_positions}")
                
                # Log position details if we have active positions
                if active_positions > 0:
                    position_details = []
                    for asset, pos in self.positions.items():
                        if abs(pos['size']) > 1e-8:
                            mark_price = self._get_mark_price(asset)
                            unrealized_pnl = pos['size'] * (mark_price - pos['entry_price'])
                            position_details.append(
                                f"{asset}: size={pos['size']:.4f}, value={abs(pos['size']*mark_price):.2f}, "
                                f"entry={pos['entry_price']:.2f}, current={mark_price:.2f}, "
                                f"pnl={unrealized_pnl:.2f}"
                            )
                    logger.debug(f"  Position Details: {', '.join(position_details)}")
            
        except Exception as e:
            logger.error(f"Error updating history: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            # Initialize collections if they don't exist
            if not hasattr(self, 'historical_leverage'):
                self.historical_leverage = deque(maxlen=10000)
            if not hasattr(self, 'drawdown_history'):
                self.drawdown_history = deque(maxlen=10000)
            if not hasattr(self, 'returns_history'):
                self.returns_history = deque(maxlen=10000)
            if not hasattr(self, 'positions_history'):
                self.positions_history = deque(maxlen=10000)
            if not hasattr(self, 'portfolio_history'):
                self.portfolio_history = deque(maxlen=10000)

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        try:
            # Initialize observation array
            observation = []
            
            # Get the window of data we need
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = min(len(self.df), self.current_step + 1)
            window_data = self.df.iloc[start_idx:end_idx]
            
            if window_data.empty:
                logger.error("Empty window data in _get_observation")
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Get the latest data point
            try:
                current_data = window_data.iloc[-1]
            except IndexError as e:
                logger.error(f"IndexError accessing window data: {str(e)}")
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Add market data for each asset
            for asset in self.assets:
                try:
                    # Get asset data using proper MultiIndex access
                    asset_data = {}
                    
                    # Add base features
                    for feat in self.base_features:
                        try:
                            value = current_data.get((asset, feat), 0.0)
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = value.iloc[0]
                            asset_data[feat] = float(value)
                        except Exception as e:
                            logger.error(f"Error getting base feature {feat} for {asset}: {str(e)}")
                            asset_data[feat] = 0.0
                    
                    # Add technical features
                    for feat in self.tech_features:
                        try:
                            value = current_data.get((asset, feat), 0.0)
                            if isinstance(value, (pd.Series, pd.DataFrame)):
                                value = value.iloc[0]
                            asset_data[feat] = float(value)
                        except Exception as e:
                            logger.error(f"Error getting tech feature {feat} for {asset}: {str(e)}")
                            asset_data[feat] = 0.0
                    
                    # Add features to observation
                    observation.extend([asset_data.get(feat, 0.0) for feat in self.base_features])
                    observation.extend([asset_data.get(feat, 0.0) for feat in self.tech_features])
                    
                except Exception as e:
                    logger.error(f"Error processing asset {asset} in observation: {str(e)}")
                    # Add zeros for all features of this asset
                    observation.extend([0.0] * (len(self.base_features) + len(self.tech_features)))
            
            # Add portfolio data for each asset
            total_portfolio_value = self._calculate_portfolio_value()
            for asset in self.assets:
                try:
                    position = self.positions[asset]
                    mark_price = self._get_mark_price(asset)
                    position_value = position['size'] * mark_price
                    
                    observation.extend([
                        float(position['size']),
                        float(position_value / (total_portfolio_value + 1e-8)),
                        float(self.funding_accrued[asset] / (total_portfolio_value + 1e-8))
                    ])
                except Exception as e:
                    logger.error(f"Error adding portfolio data for {asset}: {str(e)}")
                    observation.extend([0.0, 0.0, 0.0])
            
            # Add global portfolio data
            try:
                # Calculate recent PnL using last 100 trades
                # CRITICAL FIX: Handle both 'pnl' and 'realized_pnl' keys for backwards compatibility
                recent_trades_pnl = 0.0
                for trade in self.trades[-100:]:
                    if 'pnl' in trade:
                        recent_trades_pnl += trade['pnl']
                    elif 'realized_pnl' in trade:
                        recent_trades_pnl += trade['realized_pnl']
                
                active_positions = sum(1 for p in self.positions.values() if abs(p['size']) > 1e-8)
                
                observation.extend([
                    float(total_portfolio_value / (self.initial_balance + 1e-8)),
                    float(recent_trades_pnl / (self.initial_balance + 1e-8)),
                    float(active_positions / len(self.assets))
                ])
            except Exception as e:
                logger.error(f"Error adding global portfolio data: {str(e)}")
                observation.extend([1.0, 0.0, 0.0])
            
            # Ensure observation matches expected shape
            if len(observation) != self.observation_space.shape[0]:
                logger.error(f"Observation shape mismatch: expected {self.observation_space.shape[0]}, got {len(observation)}")
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Clip observation values to prevent extreme values
            observation = np.clip(observation, -1e6, 1e6)
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting observation: {str(e)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _close_position(self, asset: str):
        """Close a single position"""
        try:
            position = self.positions[asset]
            if abs(position['size']) > 1e-8:
                mark_price = self._get_mark_price(asset)
                commission = abs(position['size']) * mark_price * self.commission
                self.balance -= commission
                position['size'] = 0
                position['entry_price'] = 0
                position['last_price'] = mark_price
        except Exception as e:
            logger.error(f"Error closing position for {asset}: {str(e)}")
            
    def _close_all_positions(self):
        """Close all positions"""
        for asset in self.assets:
            self._close_position(asset)

    # Setter methods for external configuration
    def set_regime_aware(self, enabled=True):
        """Enable or disable market regime awareness"""
        self.regime_aware = enabled
        logger.info(f"Market regime awareness {'enabled' if enabled else 'disabled'}")
        return self.regime_aware
    
    def set_position_holding_bonus(self, bonus_factor=0.02):
        """Set the multiplier for position holding time bonuses"""
        self.position_holding_bonus_factor = float(bonus_factor)
        logger.info(f"Position holding bonus factor set to {self.position_holding_bonus_factor}")
        return self.position_holding_bonus_factor
    
    def set_uncertainty_scaling(self, scaling_factor=1.0):
        """Set the scaling factor for uncertainty-based position sizing"""
        self.uncertainty_scaling_factor = float(scaling_factor)
        logger.info(f"Uncertainty scaling factor set to {self.uncertainty_scaling_factor}")
        return self.uncertainty_scaling_factor
    
    def _calculate_holding_time_reward(self) -> float:
        """Calculate reward/penalty based on position holding time and profitability"""
        # Only apply if position holding bonus is enabled
        if not hasattr(self, 'position_holding_bonus_factor'):
            self.position_holding_bonus_factor = 0.02  # Default value
            
        total_bonus = 0.0
        
        for asset in self.assets:
            duration = self.position_duration[asset]
            profits = self.position_profits[asset]
            
            if duration == 0 or not profits:  # No active position or no profit data
                continue
                
            # Get current position size
            position_size = self.positions[asset]['size']
            
            # Skip tiny positions
            if abs(position_size) < 0.001 * self.balance / self.positions[asset]['last_price']:
                continue
            
            # Calculate average profit
            avg_profit = sum(profits) / len(profits)
            
            # Profitable positions: Reward holding longer (diminishing returns)
            if avg_profit > 0:
                # Logarithmic bonus that increases with time but with diminishing returns
                # Scale by position_holding_bonus_factor
                holding_bonus = self.position_holding_bonus_factor * np.log1p(duration) * avg_profit
                total_bonus += holding_bonus
            
            # Unprofitable positions: Small penalty for holding too long
            elif avg_profit < -0.05 and duration > 10:  # >5% loss and held for >10 steps
                # Penalty increases with time and loss magnitude
                holding_penalty = (self.position_holding_bonus_factor / 2) * np.log1p(duration) * abs(avg_profit)
                total_bonus -= holding_penalty
        
        # Trading frequency adjustment
        if len(self.trade_counts) > 10:  # Need some history to calculate
            current_trade_frequency = sum(self.trade_counts) / len(self.trade_counts)
            
            # Penalize deviation from optimal trading frequency
            frequency_deviation = abs(current_trade_frequency - self.optimal_trade_frequency)
            frequency_penalty = 0.05 * frequency_deviation
            
            # Apply penalty
            total_bonus -= frequency_penalty
            
            # Add penalty for excessive no-trade periods (encourages periodic action)
            if self.consecutive_no_trade_steps > 20:
                inactivity_penalty = 0.005 * (self.consecutive_no_trade_steps - 20)
                total_bonus -= min(inactivity_penalty, 0.1)  # Cap penalty
        
        return total_bonus