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
        self.initial_balance = 100000.0
        self.balance = self.initial_balance
        self.commission = commission
        self.max_leverage = max_leverage
        self.window_size = window_size
        self.current_step = self.window_size
        self.funding_fee_multiplier = funding_fee_multiplier
        self.max_drawdown = 0.5
        self.risk_free_rate = risk_free_rate
        
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
        self.positions = {asset: {'size': 0.0, 'entry_price': 0.0, 'last_price': 0.0} 
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
        self.positions = {asset: {'size': 0, 'entry_price': 0, 'funding_accrued': 0, 'last_price': self._get_mark_price(asset)} 
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
        
        # Log reset
        # logger.info(f"Environment reset: window_size={self.window_size}, initial_balance={self.initial_balance}")
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Execute trading step with the given action"""
        try:
            # Store initial portfolio value for reward calculation
            initial_portfolio = self._calculate_portfolio_value()
            initial_positions = copy.deepcopy(self.positions)
            
            # ENHANCED: Track consecutive no-trade steps
            had_trade_this_step = False
            
            # Check if we've reached the end of data
            if self.current_step >= len(self.df) - 1:
                # Return final state with done flag
                return self._get_observation(), 0, True, False, self._get_info({})
            
            # Advance to next step
            self.current_step += 1
            
            # ENHANCED: Process action and detect changes for trade frequency tracking
            action_vector = np.array(action).flatten()
            position_changes = np.abs(action_vector - self.last_action_vector)
            significant_change = np.any(position_changes > 0.1)  # Consider >10% change as significant
            self.last_action_vector = action_vector.copy()
            
            # ENHANCED: Apply uncertainty-based scaling to actions
            uncertainty_scaled_action = self._apply_uncertainty_scaling(action_vector)
            
            # Execute trades based on uncertainty-scaled action
            trades = []
            trades_executed = False
            
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
                
                # Convert signal to target position size (-max_leverage to +max_leverage)
                target_size = signal * self.max_leverage * self.balance / price
                
                # Change from current to target (positive = increase, negative = decrease)
                size_change = target_size - current_size
                
                # Skip tiny trades (less than 0.1% of balance)
                min_trade_size = 0.001 * self.balance / price
                if abs(size_change) < min_trade_size:
                    continue
                
                # Create trade object with size change
                trade = {
                    'asset': asset,
                    'timestamp': self.current_step,
                    'price': price,
                    'target_size': target_size,
                    'size_change': size_change,
                    # Add required keys for _simulate_trades
                    'direction': 1 if size_change > 0 else -1,  # Direction based on size change
                    'leverage': abs(signal * self.max_leverage),  # Leverage calculation
                    'execution_params': [1.0],  # Default execution param
                    'risk_limits': {  # Default risk limits
                        'max_slippage': 0.001,
                        'max_impact': 0.002
                    }
                }
                
                trades.append(trade)
            
            # Simulate trades to check risk limits
            risk_metrics = self._simulate_trades(trades)
            
            # Execute trades if no risk violations
            if not risk_metrics.get('risk_violation', False):
                # Execute the trades
                self._execute_trades(trades)
                trades_executed = len(trades) > 0
                
                # ENHANCED: If we executed trades, reset consecutive no-trade counter
                if trades_executed:
                    had_trade_this_step = True
                    self.consecutive_no_trade_steps = 0
                else:
                    self.consecutive_no_trade_steps += 1
            
            # ENHANCED: Update trade counts for frequency tracking
            self.trade_counts.append(1 if had_trade_this_step else 0)
            
            # Update all positions prices and values
            self._update_positions()
            
            # ENHANCED: Update position durations and track profits
            for asset in self.assets:
                position_size = self.positions[asset]['size']
                
                if position_size != 0:
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
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Add to portfolio history
            self.portfolio_history.append({
                'step': self.current_step,
                'value': portfolio_value
            })
            
            # Calculate PnL for this step
            total_pnl = portfolio_value - initial_portfolio
            
            # Update risk metrics after executing trades
            risk_metrics = self._calculate_risk_metrics()
            
            # ENHANCED: Calculate holding time bonus/penalty
            holding_reward = self._calculate_holding_time_reward()
            
            # Calculate reward
            reward = self._calculate_reward(risk_metrics, total_pnl, holding_reward)
            
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
            
            # Log step results
            # if self.verbose:
            #     logger.info(f"Step {self.current_step}: Reward = {reward:.4f}, Done = {done}, "
            #                f"Trades Executed = {trades_executed}, Portfolio Value = {portfolio_value:.2f}")
            
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
            simulated_positions = self.positions.copy()
            
            for trade in trades:
                asset = trade['asset']
                
                # Handle different trade dictionary structures
                if 'size' in trade:  
                    # Close position
                    simulated_positions[asset]['size'] = 0
                elif 'size_change' in trade and 'direction' in trade and 'leverage' in trade:  
                    # New trade with our enhanced structure
                    size_change = trade['size_change']
                    direction = trade['direction']
                    leverage = trade['leverage']
                    
                    # Calculate target exposure
                    target_exposure = direction * leverage * self.balance
                    current_exposure = simulated_positions[asset]['size'] * self._get_mark_price(asset)
                    
                    # Get execution parameter (default to 1.0 if not present)
                    execution_param = 1.0
                    if 'execution_params' in trade and trade['execution_params']:
                        execution_param = trade['execution_params'][0]
                    
                    # Calculate order size
                    order_size = (target_exposure - current_exposure) * execution_param
                    
                    # Update simulated position
                    new_size = simulated_positions[asset]['size'] + (size_change or 0)
                    simulated_positions[asset]['size'] = new_size
                
            return simulated_positions
            
        except Exception as e:
            logger.error(f"Error in _simulate_trades: {str(e)}")
            traceback.print_exc()
            # Return empty risk metrics with error flag
            return {"risk_violation": True, "error": str(e)}
        
    def _execute_trades(self, trades: List[Dict]):
        """Smart order execution with transaction cost model"""
        try:
            for trade in trades:
                asset = trade['asset']
                
                # Safely extract trade parameters with defaults
                direction = trade.get('direction', 1 if trade.get('size_change', 0) > 0 else -1)
                leverage = trade.get('leverage', 1.0)
                risk_limits = trade.get('risk_limits', {'max_slippage': 0.001, 'max_impact': 0.002})
                
                # Get execution parameters (default to full execution)
                execution_params = trade.get('execution_params', [1.0])
                
                # If we have size_change, use that directly
                if 'size_change' in trade and trade['size_change'] != 0:
                    # Get current price
                    mark_price = self._get_mark_price(asset)
                    order_size = trade['size_change'] * mark_price
                    
                    # Apply transaction costs
                    # Apply slippage model - implementation shortfall
                    price_impact = self._estimate_price_impact(asset, order_size)
                    execution_price = mark_price * (1 + price_impact * direction)
                    
                    # Apply transaction costs
                    total_cost = abs(order_size) * (
                        self.commission + 
                        price_impact + 
                        self._get_spread_cost(asset)
                    )
                    
                    # Update balance and position
                    self.balance -= total_cost
                    self.positions[asset]['size'] += trade['size_change']
                    self.positions[asset]['entry_price'] = execution_price
                    self.total_costs += total_cost
                else:
                    # Target position calculation (original approach)
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
                            self.commission + 
                            price_impact + 
                            self._get_spread_cost(asset)
                        )
                        
                        # Update balance and position
                        self.balance -= total_cost
                        self.positions[asset]['size'] = target_exposure / execution_price
                        self.positions[asset]['entry_price'] = execution_price
                        self.total_costs += total_cost
        except Exception as e:
            logger.error(f"Error in _execute_trades: {str(e)}")
            traceback.print_exc()
        
    def _update_positions(self):
        """Update positions with mark-to-market and funding rates"""
        try:
            total_pnl = 0.0
            portfolio_value_before = self._calculate_portfolio_value()
            
            for asset in self.assets:
                try:
                    position = self.positions[asset]
                    mark_price = self._get_mark_price(asset)
                    
                    if abs(position['size']) > 1e-8:
                        # Store previous price for PnL calculation
                        prev_price = position.get('last_price', mark_price)
                        
                        # Calculate unrealized PnL from price movement
                        price_change_pnl = position['size'] * (mark_price - prev_price)
                        total_pnl += price_change_pnl
                        
                        # Calculate and apply funding rate
                        funding_rate = self._get_funding_rate(asset)
                        
                        # Funding is paid by longs to shorts when positive, and by shorts to longs when negative
                        # For long positions (positive size), subtract positive funding rate (pay)
                        # For short positions (negative size), add positive funding rate (receive)
                        funding_cost = position['size'] * mark_price * funding_rate * self.funding_fee_multiplier
                        
                        # Track funding for metrics
                        self.funding_accrued[asset] += funding_cost
                        
                        # Apply funding to balance
                        if abs(funding_cost) > 0:
                            if funding_cost > self.balance and funding_cost > 0:
                                # Force close position if can't afford funding
                                # logger.warning(f"Forced liquidation of {asset} due to insufficient balance for funding cost: {funding_cost:.2f} > {self.balance:.2f}")
                                self._close_position(asset)
                            else:
                                # Update balance with funding cost
                                self.balance -= funding_cost
                                if self.verbose:
                                    logger.debug(f"Applied funding for {asset}: {funding_cost:.4f} (rate: {funding_rate:.6f})")
                        
                        # Update last price
                        position['last_price'] = mark_price
                        
                except Exception as e:
                    logger.error(f"Error updating position for {asset}: {str(e)}")
                    continue
            
            # Calculate portfolio value after updates
            portfolio_value_after = self._calculate_portfolio_value()
            
            # Update peak value for drawdown calculations
            if portfolio_value_after > self.peak_value:
                self.peak_value = portfolio_value_after
            
            # Calculate current drawdown
            current_drawdown = 0.0
            if self.peak_value > 0:
                current_drawdown = 1.0 - (portfolio_value_after / self.peak_value)
                
                # Add to drawdown history
                if current_drawdown > 0:
                    self.drawdown_history.append(current_drawdown)
            
            # Check for liquidation
            if portfolio_value_after <= 0 or self.balance <= 0:
                # logger.warning("Account liquidated due to insufficient funds")
                self.liquidated = True
                self._close_all_positions()
            
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
            for asset, position in self.positions.items():
                try:
                    mark_price = self._get_mark_price(asset)
                    position_value = position['size'] * mark_price
                    unrealized_pnl = position['size'] * (mark_price - position['entry_price'])
                    
                    # Add position value including unrealized PnL
                    total_value += position_value
                    
                    # Log significant positions
                    if abs(position['size']) > 1e-8 and self.verbose:
                        logger.debug(f"Position value for {asset}: size={position['size']:.4f}, "
                                   f"price={mark_price:.2f}, value={position_value:.2f}, "
                                   f"unrealized_pnl={unrealized_pnl:.2f}")
                        
                except Exception as e:
                    logger.error(f"Error calculating position value for {asset}: {str(e)}")
                    continue
            
            # Log total portfolio value
            if self.verbose:
                logger.debug(f"Total portfolio value: {total_value:.2f} (balance: {self.balance:.2f})")
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
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
                    mark_price = market_data.iloc[-1][(asset, 'close')]
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
            
            # CRITICAL FIX: Ensure historical collections are updated
            # Update leverage history
            if 'leverage_utilization' in risk_metrics:
                leverage = risk_metrics['leverage_utilization']
                if not hasattr(self, 'historical_leverage'):
                    self.historical_leverage = deque(maxlen=10000)
                self.historical_leverage.append(leverage)
            
            # Update drawdown history
            if 'current_drawdown' in risk_metrics:
                drawdown = risk_metrics['current_drawdown']
                if not hasattr(self, 'drawdown_history'):
                    self.drawdown_history = deque(maxlen=10000)
                self.drawdown_history.append(drawdown)
            
            # CRITICAL FIX: Calculate and add risk-adjusted ratios
            # These are needed for proper evaluation
            if not hasattr(self, 'returns_history') or len(self.returns_history) < 2:
                # Not enough history for proper calculation
                risk_metrics['sharpe_ratio'] = 0.0
                risk_metrics['sortino_ratio'] = 0.0
                risk_metrics['calmar_ratio'] = 0.0
            else:
                # Calculate returns
                returns = np.array(list(self.returns_history))
                
                # Calculate Sharpe ratio (with safety checks)
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns)
                    risk_metrics['sharpe_ratio'] = np.clip(sharpe, -3.0, 3.0)
                else:
                    risk_metrics['sharpe_ratio'] = 0.0
                
                # Calculate Sortino ratio (with safety checks)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_dev = np.std(negative_returns)
                    if downside_dev > 0:
                        sortino = np.mean(returns) / downside_dev
                        risk_metrics['sortino_ratio'] = np.clip(sortino, -3.0, 3.0)
                    else:
                        risk_metrics['sortino_ratio'] = 0.0
                else:
                    risk_metrics['sortino_ratio'] = 0.0
                
                # Calculate Calmar ratio (with safety checks)
                if hasattr(self, 'drawdown_history') and len(self.drawdown_history) > 0:
                    max_dd = max(self.drawdown_history)
                    if max_dd > 0:
                        calmar = np.mean(returns) / max_dd
                        risk_metrics['calmar_ratio'] = np.clip(calmar, -3.0, 3.0)
                    else:
                        risk_metrics['calmar_ratio'] = 0.0
                else:
                    risk_metrics['calmar_ratio'] = 0.0
            
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
                'current_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
    def _calculate_risk_adjusted_reward(self, total_pnl: float, risk_metrics: Dict) -> float:
        """Enhanced reward function incorporating risk metrics"""
        try:
            # Calculate portfolio value and return
            portfolio_value = self._calculate_portfolio_value()
            
            # Safety check for portfolio value
            if portfolio_value <= 0:
                # logger.warning("Portfolio value is zero or negative in reward calculation")
                return -1.0
                
            # Calculate portfolio return (capped to prevent extreme values)
            portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
            portfolio_return = np.clip(portfolio_return, -1.0, 1.0)  # Cap return to reasonable range
            
            # Get risk metrics with safety checks
            volatility = max(risk_metrics.get('volatility', 0.0001), 0.0001)  # Avoid division by zero
            var = abs(risk_metrics.get('var', 0.0))
            es = abs(risk_metrics.get('expected_shortfall', 0.0))
            max_drawdown = max(risk_metrics.get('max_drawdown', 0.0), 0.0001)
            current_drawdown = risk_metrics.get('current_drawdown', 0.0)
            leverage = risk_metrics.get('leverage_utilization', 0.0)
            concentration = risk_metrics.get('max_concentration', 0.0)
            
            # Calculate excess return over risk-free rate (daily)
            daily_risk_free = self.risk_free_rate / 252  # Convert annual to daily
            excess_return = portfolio_return - daily_risk_free
            
            # Calculate risk-adjusted ratios with strict bounds
            # Sharpe ratio - limit to reasonable range (-3 to 3 is typical)
            sharpe = excess_return / volatility if volatility > 0 else 0
            sharpe = np.clip(sharpe, -3.0, 3.0)
            
            # Calculate Sortino ratio (using downside deviation)
            if not hasattr(self, 'returns_history') or len(self.returns_history) < 2:
                sortino = 0.0
            else:
                # Calculate downside deviation (only negative returns)
                negative_returns = [r for r in self.returns_history if r < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    downside_deviation = max(downside_deviation, 0.0001)  # Safety check
                    sortino = excess_return / downside_deviation
                else:
                    sortino = 0.0  # No negative returns yet
            
            # Clip Sortino to reasonable range
            sortino = np.clip(sortino, -3.0, 3.0)
            
            # Calculate Calmar ratio (return / max drawdown)
            if max_drawdown > 0.0001:
                calmar = portfolio_return / max_drawdown
                calmar = np.clip(calmar, -3.0, 3.0)  # Clip to reasonable range
            else:
                calmar = 0.0  # No significant drawdown yet
            
            # Calculate diversification score
            active_positions = sum(1 for p in self.positions.values() if abs(p['size']) > 0)
            max_possible_positions = max(len(self.assets), 1)  # Avoid division by zero
            diversification_score = min(active_positions / max_possible_positions, 1.0)
            
            # Calculate risk penalties
            # Leverage penalty (higher leverage = higher penalty)
            leverage_ratio = min(leverage / max(self.max_leverage, 0.0001), 1.0)
            leverage_penalty = 1.0 - leverage_ratio
            
            # Drawdown penalty (higher drawdown = higher penalty)
            drawdown_ratio = min(current_drawdown / max(self.max_drawdown, 0.0001), 1.0)
            drawdown_penalty = 1.0 - drawdown_ratio
            
            # VaR penalty (higher VaR = higher penalty)
            var_ratio = min(var / max(0.05 * portfolio_value, 0.0001), 1.0)
            var_penalty = 1.0 - var_ratio
            
            # Concentration penalty (higher concentration = higher penalty)
            concentration_ratio = min(concentration / 0.5, 1.0)
            concentration_penalty = 1.0 - concentration_ratio
            
            # Trading activity bonus (encourage trading)
            trading_activity_bonus = min(active_positions / max_possible_positions, 1.0)
            
            # Combined risk penalty (ensure it's bounded)
            risk_penalty = np.clip(
                0.3 * leverage_penalty +
                0.3 * drawdown_penalty +
                0.2 * var_penalty +
                0.2 * concentration_penalty,
                0.0, 1.0
            )
            
            # Final reward calculation with weighted components
            reward = (
                0.25 * sharpe +                  # Reward risk-adjusted returns (Sharpe)
                0.15 * sortino +                 # Reward downside risk-adjusted returns (Sortino)
                0.15 * calmar +                  # Reward drawdown-adjusted returns (Calmar)
                0.15 * diversification_score +   # Reward diversification
                0.20 * risk_penalty +            # Penalize excessive risk
                0.10 * trading_activity_bonus    # Encourage trading activity
            )
            
            # Ensure reward is properly bounded
            reward = np.clip(reward, -1.0, 1.0)
            
            # Add risk metrics to the risk_metrics dict for evaluation
            risk_metrics['sharpe_ratio'] = float(sharpe)
            risk_metrics['sortino_ratio'] = float(sortino)
            risk_metrics['calmar_ratio'] = float(calmar)
            
            # Log reward components
            if self.verbose:
                logger.debug(f"Reward components:")
                logger.debug(f"  Sharpe: {sharpe:.4f}")
                logger.debug(f"  Sortino: {sortino:.4f}")
                logger.debug(f"  Calmar: {calmar:.4f}")
                logger.debug(f"  Diversification: {diversification_score:.4f}")
                logger.debug(f"  Leverage penalty: {leverage_penalty:.4f}")
                logger.debug(f"  Drawdown penalty: {drawdown_penalty:.4f}")
                logger.debug(f"  VaR penalty: {var_penalty:.4f}")
                logger.debug(f"  Concentration penalty: {concentration_penalty:.4f}")
                logger.debug(f"  Trading bonus: {trading_activity_bonus:.4f}")
                logger.debug(f"  Risk penalty: {risk_penalty:.4f}")
                logger.debug(f"  Final reward: {reward:.4f}")
            
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
                # logger.info("Episode done: Reached end of data")
                return True
            
            # Calculate current risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Check account depletion
            if risk_metrics['portfolio_value'] <= 0:
                # logger.info("Episode done: Account depleted")
                return True
            
            # Check risk limits using risk engine
            try:
                is_within_limits, violations = self.risk_engine.check_risk_limits(risk_metrics)
                if not is_within_limits:
                    # logger.info(f"Episode done: Risk limits violated - {', '.join(violations)}")
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

    def _execute_trade(self, asset_idx: int, signal: float, price: float) -> float:
        """Execute a trade for a single asset and return the PnL"""
        try:
            asset = self.assets[asset_idx]
            old_position = self.positions[asset]['size']
            old_entry_price = self.positions[asset]['entry_price']
            
            # Calculate current portfolio value with safety check
            portfolio_value = max(self._calculate_portfolio_value(), self.initial_balance * 0.1)
            
            # IMPORTANT FIX: Lower threshold for trade execution to ensure trades happen
            # Convert signal [-1, 1] to target leverage [0, max_leverage]
            # Use a minimum leverage to ensure trades are executed
            target_leverage = max(abs(signal) * self.max_leverage, 0.2)  # Increased minimum leverage from 0.1 to 0.2
            
            # Determine direction (-1 or 1) from signal
            direction = np.sign(signal)
            if direction == 0 or abs(direction) < 0.001:  # Handle zero or near-zero signal case
                direction = 1 if np.random.random() > 0.5 else -1  # Random direction for zero signal
            
            # Calculate target position size
            target_value = portfolio_value * target_leverage * direction
            target_size = target_value / price if price > 0 else 0
            size_diff = target_size - old_position
            
            # IMPORTANT FIX: Ensure minimum trade size relative to portfolio
            # Reduced minimum trade size to make it easier to execute trades
            min_trade_size = portfolio_value * 0.003 / price  # Reduced from 0.005 to 0.003 (0.3% of portfolio)
            if abs(size_diff) < min_trade_size:
                size_diff = min_trade_size * np.sign(size_diff) if size_diff != 0 else min_trade_size * direction
            
            # Calculate commission
            commission = abs(size_diff) * price * self.commission
            
            # IMPORTANT FIX: Always ensure we can execute the trade
            # If insufficient balance, scale down the trade size instead of canceling
            if commission > self.balance:
                if self.verbose:
                    logger.warning(f"Insufficient balance for commission: {commission:.2f} > {self.balance:.2f}")
                # Scale down the trade size but ensure it's still meaningful
                scale_factor = max(0.2, self.balance / (commission * 1.05))  # Increased from 0.1 to 0.2, reduced buffer
                size_diff *= scale_factor
                commission = abs(size_diff) * price * self.commission
                if self.verbose:
                    logger.info(f"Scaled down trade size by factor {scale_factor:.4f}")
            
            # IMPORTANT FIX: Ensure minimum trade size even after scaling
            # Reduced minimum threshold to allow smaller trades
            if abs(size_diff) < min_trade_size * 0.05:  # Reduced from 0.1 to 0.05 (5% of minimum size)
                size_diff = min_trade_size * 0.05 * np.sign(size_diff) if size_diff != 0 else min_trade_size * 0.05 * direction
                commission = abs(size_diff) * price * self.commission
            
            # NEW: Randomly force some trades to succeed (during training only)
            # This helps with exploration by ensuring some trades go through
            if np.random.random() < 0.1:  # 10% chance to force trade
                # Ensure the trade is at least 0.5% of portfolio value
                forced_size = portfolio_value * 0.005 / price * direction
                if abs(forced_size) > abs(size_diff):
                    if self.verbose:
                        logger.info(f"Forcing larger trade: {forced_size:.6f} instead of {size_diff:.6f}")
                    size_diff = forced_size
                    commission = abs(size_diff) * price * self.commission
            
            # Calculate PnL for closing part of the position
            trade_pnl = -commission
            if old_position != 0 and np.sign(old_position) != np.sign(target_size) and abs(old_position) > 0:
                # Closing position (full or partial)
                closing_size = min(abs(old_position), abs(target_size)) * np.sign(old_position)
                closing_pnl = closing_size * (price - old_entry_price)
                trade_pnl += closing_pnl
                if self.verbose:
                    logger.info(f"Closing position PnL: {closing_pnl:.2f}")
            
            # Update balance
            self.balance += trade_pnl
            
            # Update position
            self.positions[asset]['size'] = target_size
            
            # Update entry price with weighted average if increasing position
            if old_position != 0 and np.sign(old_position) == np.sign(target_size) and abs(target_size) > abs(old_position):
                # Increasing existing position
                new_size = abs(target_size) - abs(old_position)
                total_size = abs(target_size)
                self.positions[asset]['entry_price'] = (old_entry_price * abs(old_position) + price * new_size) / total_size
            elif np.sign(old_position) != np.sign(target_size) or old_position == 0:
                # New position or flipped direction
                self.positions[asset]['entry_price'] = price
            
            # IMPORTANT FIX: Always log trades for debugging
            logger.info(f"Trade executed for {asset}: Old={old_position:.4f}, New={target_size:.4f}, "
                       f"Price={price:.2f}, PnL={trade_pnl:.2f}")
            
            # Add trade to history
            self.trades.append({
                'asset': asset,
                'timestamp': self.current_step,
                'size': size_diff,
                'price': price,
                'pnl': trade_pnl
            })
            
            return trade_pnl
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            # IMPORTANT FIX: Return a small negative value instead of 0 to indicate error
            return -0.1

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
                asset_data = self.df.iloc[max(0, self.current_step-30):self.current_step+1]
                asset_data = asset_data.xs(asset, level=0, axis=1)
                
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
                        vol_trend = np.mean(list(vol_history)[-3:]) / np.mean(list(vol_history)[:-3]) - 1
                    
                    # Calculate uncertainty score (0 = certain, 1 = uncertain)
                    # Base on:
                    # 1. Current volatility relative to average
                    # 2. Trending or ranging market
                    # 3. Volatility trend
                    volatility_factor = min(current_vol / max(0.0001, self.uncertainty_metrics[asset]['avg_volatility']), 3)
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
                            
                            # # Log significant scaling
                            # if abs(scaled_action[i] - raw_action) > 0.2:
                            #     logger.info(f"Uncertainty scaling for {asset}: {raw_action:.2f} -> {scaled_action[i]:.2f} " 
                            #               f"(uncertainty: {uncertainty_score:.2f})")
            
            except Exception as e:
                logger.error(f"Error in uncertainty scaling for {asset}: {str(e)}")
                # Fall back to original action
                continue
        
        return scaled_action

    def _update_history(self):
        """Update portfolio history"""
        try:
            current_value = self.balance
            total_exposure = 0
            
            # Calculate current portfolio value and exposure
            for asset, position in self.positions.items():
                mark_price = self._get_mark_price(asset)
                position_value = position['size'] * mark_price
                current_value += position_value
                total_exposure += abs(position_value)  # Use absolute value for exposure
            
            # Safety check for portfolio value
            if current_value <= 0:
                # logger.warning("Portfolio value is zero or negative, setting metrics to zero")
                current_leverage = 0
                current_drawdown = 1  # Maximum drawdown
            else:
                # Calculate current leverage with bounds
                current_leverage = min(total_exposure / current_value, self.max_leverage * 1.1)
                
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
            if self.verbose or active_positions > 0:
                logger.info(f"Portfolio metrics - Step {self.current_step}:")
                logger.info(f"  Value: {current_value:.2f}")
                logger.info(f"  Total Exposure: {total_exposure:.2f}")
                logger.info(f"  Leverage: {current_leverage:.4f}")
                logger.info(f"  Drawdown: {current_drawdown:.4f}")
                logger.info(f"  Return: {current_return:.4f}")
                logger.info(f"  Active Positions: {active_positions}")
                
                # Log position details if we have active positions
                if active_positions > 0:
                    position_details = []
                    for asset, pos in self.positions.items():
                        if abs(pos['size']) > 1e-8:
                            mark_price = self._get_mark_price(asset)
                            position_details.append(
                                f"{asset}: size={pos['size']:.4f}, value={pos['size']*mark_price:.2f}, "
                                f"entry={pos['entry_price']:.2f}, current={mark_price:.2f}"
                            )
                    logger.info(f"  Position Details: {', '.join(position_details)}")
            
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
                recent_trades_pnl = sum(trade['pnl'] for trade in self.trades[-100:])
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
        # logger.info(f"Uncertainty scaling factor set to {self.uncertainty_scaling_factor}")
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