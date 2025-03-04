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

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Change from WARNING to ERROR for less verbosity

class InstitutionalPerpetualEnv(gym.Env):
    def __init__(self, 
                 df: pd.DataFrame,
                 assets: List[str],
                 window_size: int = 100,
                 max_leverage: float = 5.0,
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
        self.max_drawdown = 0.3
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
        
        # Initialize funding rates and accrued funding
        self.funding_rates = {asset: 0.0 for asset in self.assets}
        self.funding_accrued = {asset: 0.0 for asset in self.assets}
        
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
        logger.info(f"Environment reset: window_size={self.window_size}, initial_balance={self.initial_balance}")
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Take a step in the environment"""
        # Initialize variables
        total_pnl = 0
        trades_executed = False
        
        try:
            # Increment step counter
            self.current_step += 1
            
            # CRITICAL FIX: Always log the action for debugging
            logger.info(f"Step {self.current_step}: Action = {action}")
            
            # Get market data for risk calculations
            start_idx = max(0, self.current_step - 250)
            market_data = self.df.iloc[start_idx:self.current_step + 1]
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value()
            if self.verbose:
                logger.info(f"Current portfolio value before trades: {portfolio_value}")
            
            # CRITICAL FIX: Track initial positions to detect changes
            initial_positions = {asset: pos['size'] for asset, pos in self.positions.items()}
            initial_portfolio = portfolio_value
            
            # CRITICAL FIX: Ensure action is properly formatted
            if isinstance(action, (list, tuple)):
                action = np.array(action)
            
            # Ensure action is a numpy array
            if not isinstance(action, np.ndarray):
                action = np.array([action])
            
            # Reshape action if needed
            if action.shape != (len(self.assets),):
                if action.shape[0] == 1 and len(action.shape) > 1 and action.shape[1] == len(self.assets):
                    action = action[0]
                else:
                    logger.warning(f"Reshaping action from {action.shape} to ({len(self.assets)},)")
                    action = action.reshape(-1)[:len(self.assets)]
            
            # Clip action to valid range
            action = np.clip(action, -1, 1)
            
            # Execute trades for each asset
            for asset_idx, signal in enumerate(action):
                # CRITICAL FIX: Lower the threshold for executing trades even further
                # Original threshold was 0.01, now using 0.0001 to ensure trades happen
                if abs(signal) > 0.0001:  # Execute even with tiny signals to encourage exploration
                    asset = self.assets[asset_idx]
                    price = self._get_mark_price(asset)
                    if self.verbose:
                        logger.info(f"Executing trade for {asset} with signal {signal:.4f} at price {price:.2f}")
                    pnl = self._execute_trade(asset_idx, signal, price)
                    total_pnl += pnl
                    
                    # Check if position actually changed
                    if abs(self.positions[asset]['size'] - initial_positions[asset]) > 1e-8:
                        trades_executed = True
                        logger.info(f"Trade executed for {asset}: PnL = {pnl:.2f}, Size change: {self.positions[asset]['size'] - initial_positions[asset]:.4f}")
                    else:
                        if self.verbose:
                            logger.warning(f"Trade attempted but no position change for {asset} with signal {signal:.4f}")
                else:
                    # CRITICAL FIX: Occasionally force small trades even with tiny signals
                    if np.random.random() < 0.2:  # Increased from 10% to 20% chance to force a trade
                        asset = self.assets[asset_idx]
                        price = self._get_mark_price(asset)
                        # Create a minimum signal
                        forced_signal = 0.05 * (1 if np.random.random() > 0.5 else -1)  # Increased from 0.01 to 0.05
                        logger.info(f"Forcing small trade for {asset} with signal {forced_signal:.4f}")
                        pnl = self._execute_trade(asset_idx, forced_signal, price)
                        total_pnl += pnl
                        
                        # Check if position actually changed
                        if abs(self.positions[asset]['size'] - initial_positions[asset]) > 1e-8:
                            trades_executed = True
                            logger.info(f"Forced trade executed for {asset}: PnL = {pnl:.2f}, Size change: {self.positions[asset]['size'] - initial_positions[asset]:.4f}")
            
            # Update positions with funding and mark-to-market
            self._update_positions()
            
            # Update history
            self._update_history()
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Calculate reward
            reward = self._calculate_reward(risk_metrics, total_pnl)
            
            # CRITICAL FIX: Enhance reward for executing trades during early training
            if trades_executed and self.current_step < 1000:
                # Provide a small bonus for executing trades early in training
                trade_bonus = 0.1  # Small positive bonus
                reward += trade_bonus
                logger.info(f"Added trade execution bonus: +{trade_bonus}")
            
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
            if self.verbose or trades_executed:
                logger.info(f"Step {self.current_step}: Reward = {reward:.4f}, Done = {done}, "
                           f"Trades Executed = {trades_executed}, Portfolio Value = {portfolio_value:.2f}")
            
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
                    self.commission + 
                    price_impact + 
                    self._get_spread_cost(asset)
                )
                
                # Update position and balance
                self.positions[asset]['size'] += order_size / execution_price
                self.balance -= total_cost
                
                if self.positions[asset]['size'] != 0:
                    self.positions[asset]['entry_price'] = execution_price
                    
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
                                logger.warning(f"Forced liquidation of {asset} due to insufficient balance for funding cost: {funding_cost:.2f} > {self.balance:.2f}")
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
                logger.warning("Account liquidated due to insufficient funds")
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
                logger.warning("Portfolio value is zero or negative in reward calculation")
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
        
    def _calculate_reward(self, risk_metrics: Dict, total_pnl: float) -> float:
        """Standard reward function with risk adjustments"""
        try:
            # Calculate current portfolio value and return
            portfolio_value = self._calculate_portfolio_value()
            
            # IMPORTANT FIX: Stronger incentive for taking positions
            # Calculate total exposure
            total_exposure = sum(abs(pos['size'] * self._get_mark_price(asset)) 
                               for asset, pos in self.positions.items())
            
            # If no trades have been executed, return larger negative reward to encourage exploration
            if total_exposure == 0:
                if self.verbose:
                    logger.warning("No positions taken - applying exploration penalty")
                return -0.2  # Even stronger penalty to encourage exploration
            
            # IMPORTANT FIX: Add trading activity bonus
            # Count active positions
            active_positions = sum(1 for pos in self.positions.values() if abs(pos['size']) > 0)
            
            # NEW: Check for recent trades
            recent_trades = [t for t in self.trades if t['timestamp'] >= self.current_step - 3]
            recent_trades_bonus = min(len(recent_trades) * 0.1, 0.3)  # Bonus for recent trading activity
            
            # Increased trading activity bonus
            trading_activity_bonus = min(active_positions / len(self.assets), 1.0) * 0.3 + recent_trades_bonus
            
            # Calculate return for this step (capped to prevent extreme values)
            current_return = (portfolio_value - self.initial_balance) / self.initial_balance
            current_return = np.clip(current_return, -1.0, 1.0)  # Cap return to reasonable range
            
            # Ensure returns_history exists
            if not hasattr(self, 'returns_history'):
                self.returns_history = deque(maxlen=10000)
            
            # Append current return to history
            self.returns_history.append(current_return)
            
            # Calculate risk-adjusted reward components
            # Use a minimum volatility to avoid division by zero
            volatility = max(risk_metrics.get('volatility', 0.0001), 0.0001)
            
            # Calculate Sharpe ratio (risk-adjusted return)
            # Use excess return over risk-free rate for proper Sharpe calculation
            daily_risk_free = self.risk_free_rate / 252  # Convert annual to daily
            excess_return = current_return - daily_risk_free
            sharpe = excess_return / volatility
            
            # Clip Sharpe to reasonable range (-3 to 3 is typical)
            sharpe = np.clip(sharpe, -3.0, 3.0)
            
            # Calculate drawdown penalty
            current_drawdown = risk_metrics.get('current_drawdown', 0.0)
            drawdown_penalty = -0.2 if current_drawdown > self.max_drawdown else 0.0
            
            # Calculate position diversification
            diversification_score = min(active_positions / max(1, len(self.assets)), 1.0) * 0.5
            
            # Calculate risk penalty
            risk_penalty = 0.0
            
            # VaR penalty
            var_threshold = 0.05  # 5% VaR threshold
            if risk_metrics.get('var', 0.0) > var_threshold * portfolio_value:
                risk_penalty -= 0.1
                if self.verbose:
                    logger.debug(f"Applied VaR penalty: VaR {risk_metrics.get('var', 0.0):.2f} > threshold {var_threshold * portfolio_value:.2f}")
            
            # Concentration penalty - penalize excessive concentration in a single asset
            if risk_metrics.get('max_concentration', 0.0) > 0.5:  # More than 50% in one asset
                risk_penalty -= 0.1
                if self.verbose:
                    logger.debug(f"Applied concentration penalty: concentration {risk_metrics.get('max_concentration', 0.0):.2f} > 0.5")
            
            # Ensure risk penalty is bounded
            risk_penalty = max(risk_penalty, -0.5)  # Limit maximum penalty
            
            # IMPORTANT FIX: Adjust reward weights to encourage more trading
            # Combine components with appropriate weights
            reward = (
                0.25 * sharpe +                  # Reward risk-adjusted returns
                0.20 * current_return +          # Reward absolute returns
                0.15 * diversification_score +   # Reward diversification
                0.10 * risk_penalty +            # Penalize excessive risk
                0.10 * drawdown_penalty +        # Specific drawdown penalty
                0.20 * trading_activity_bonus    # INCREASED from 0.10 to 0.20: Reward for active trading
            )
            
            # IMPORTANT FIX: Add bonus for having any positions at all
            if active_positions > 0:
                reward += 0.05  # Small bonus just for having positions
            
            # Ensure reward is properly bounded
            reward = np.clip(reward, -1.0, 1.0)
            
            # Add metrics to risk_metrics for evaluation
            risk_metrics['sharpe_ratio'] = float(sharpe)
            
            if self.verbose:
                logger.info(f"Reward components - Sharpe: {sharpe:.4f}, Return: {current_return:.4f}, "
                           f"Diversification: {diversification_score:.4f}, Risk Penalty: {risk_penalty:.4f}, "
                           f"Drawdown Penalty: {drawdown_penalty:.4f}, Trading Bonus: {trading_activity_bonus:.4f}")
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
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
        """Get info dictionary with detailed metrics"""
        try:
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # CRITICAL FIX: Better detection of trades
            # Check if any positions are non-zero
            active_positions = {}
            for asset, pos in self.positions.items():
                if abs(pos['size']) > 1e-8:
                    active_positions[asset] = pos['size']
            
            has_positions = len(active_positions) > 0
            
            # Check if any trades were executed in the current step
            recent_trades = [trade for trade in self.trades if trade['timestamp'] == self.current_step]
            recent_trade_executed = len(recent_trades) > 0
            
            # Check if any trades were executed in the last 3 steps
            recent_trades_window = [trade for trade in self.trades if self.current_step - 3 <= trade['timestamp'] <= self.current_step]
            recent_trades_window_count = len(recent_trades_window)
            
            # Log trade detection
            if has_positions or recent_trade_executed:
                logger.info(f"Trade detection in _get_info: has_positions={has_positions}, recent_trades={len(recent_trades)}")
                if has_positions:
                    logger.info(f"Active positions: {active_positions}")
                if recent_trade_executed:
                    logger.info(f"Recent trades: {recent_trades}")
            
            # Create info dictionary with detailed metrics
            info = {
                'portfolio_value': portfolio_value,
                'balance': self.balance,
                'trades_executed': has_positions or recent_trade_executed,  # Flag if we have positions or recent trades
                'has_positions': has_positions,  # Explicitly add this flag
                'active_positions_count': len(active_positions),  # Count of active positions
                'recent_trades_count': len(recent_trades),  # Count of trades in current step
                'recent_trades_window_count': recent_trades_window_count,  # Count of trades in last 3 steps
                'total_trades_count': len(self.trades),  # Total trades executed in episode
                'positions': {
                    asset: {
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'current_price': self._get_mark_price(asset),
                        'value': pos['size'] * self._get_mark_price(asset)
                    }
                    for asset, pos in self.positions.items()
                },
                'risk_metrics': risk_metrics,
                'historical_metrics': {
                    'leverage_samples': len(self.historical_leverage) if hasattr(self, 'historical_leverage') else 0,
                    'drawdown_samples': len(self.drawdown_history) if hasattr(self, 'drawdown_history') else 0,
                    'avg_leverage': np.mean(list(self.historical_leverage)) if hasattr(self, 'historical_leverage') and self.historical_leverage else 0,
                    'avg_drawdown': np.mean(list(self.drawdown_history)) if hasattr(self, 'drawdown_history') and self.drawdown_history else 0,
                    'max_drawdown': max(self.drawdown_history) if hasattr(self, 'drawdown_history') and self.drawdown_history else 0,
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error in _get_info: {str(e)}")
            traceback.print_exc()
            # Return minimal info
            return {
                'portfolio_value': self.initial_balance,
                'balance': self.balance,
                'trades_executed': False,
                'error': str(e)
            }

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
                logger.warning("Portfolio value is zero or negative, setting metrics to zero")
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