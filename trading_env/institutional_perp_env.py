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
    """
    Reinforcement Learning environment for institutional-grade perpetual futures trading.
    
    This environment simulates trading on perpetual futures markets with institutional-grade
    risk management, adaptive parameters, market regime detection, and comprehensive
    performance analytics.
    
    Key Features:
    -------------
    1. Sophisticated Risk Management:
       - Dynamic stop-loss and take-profit levels that adapt to volatility
       - Position sizing based on risk limits and volatility
       - Drawdown controls and portfolio-level risk constraints
       - Risk-adjusted reward function that incentivizes proper risk management
    
    2. Market Regime Awareness:
       - Detection of trending, range-bound, volatile, and crisis market regimes
       - Adjustment of trading parameters based on detected market conditions
       - Regime-specific performance tracking for strategy refinement
    
    3. Uncertainty Handling:
       - Model uncertainty quantification and action scaling
       - Adaptive exploration based on uncertainty and market regimes
       - Confidence-aware position sizing
    
    4. Adaptive Parameters:
       - Transaction costs that adjust to market volatility and liquidity
       - Risk limits that tighten during volatile periods
       - Trading incentives that adapt to different market regimes
    
    5. Comprehensive Analytics:
       - Performance tracking across different market regimes
       - Detailed trade statistics and risk-adjusted metrics
       - Visualization capabilities for performance analysis
    
    Usage:
    ------
    ```python
    # Import dependencies
    import pandas as pd
    from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
    from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
    
    # Create risk engine with custom limits
    risk_limits = RiskLimits(
        account_max_leverage=5.0,
        position_max_leverage=10.0,
        max_drawdown_pct=0.25
    )
    risk_engine = InstitutionalRiskEngine(risk_limits=risk_limits)
    
    # Create environment with your market data
    env = InstitutionalPerpetualEnv(
        df=market_data_df,            # Multi-level DataFrame with price data
        assets=["BTC", "ETH"],        # Assets to trade
        window_size=100,              # Observation window size
        risk_engine=risk_engine,      # Custom risk engine
        initial_balance=100000.0,     # Starting capital
        verbose=True                  # Enable detailed logging
    )
    
    # Use with your RL agent
    obs = env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        
    # Analyze performance
    summary = env.get_performance_summary()
    print(f"Total Return: {summary['total_return']:.2%}")
    
    # Visualize results
    env.visualize_performance(save_path="results")
    ```
    """
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
                 max_steps: int = 10000,  # Maximum number of steps before episode terminates
                 max_no_trade_steps: int = 1000,  # Maximum number of steps without trading
                 enforce_min_leverage: bool = False,  # Flag to enforce minimum leverage of 1.0
                 verbose: bool = False,
                 training_mode: bool = False):  # Add training mode flag
        """
        Initialize the Trading Environment
        
        Args:
            df: DataFrame with market data, expected to have MultiIndex columns with (asset, feature)
            assets: List of asset symbols to trade
            window_size: Size of the observation window
            max_leverage: Maximum allowed leverage for positions
            commission: Trading commission as a fraction
            funding_fee_multiplier: Multiplier for funding fees
            base_features: List of base features to use
            tech_features: List of technical features to use
            risk_engine: Risk engine instance (created automatically if None)
            risk_free_rate: Annual risk-free rate used for risk-adjusted metrics
            initial_balance: Initial account balance
            max_drawdown: Maximum allowed drawdown before liquidation
            maintenance_margin: Maintenance margin requirement as fraction of position
            max_steps: Maximum number of steps before episode terminates
            max_no_trade_steps: Maximum number of steps without trading
            enforce_min_leverage: Flag to enforce minimum leverage of 1.0
            verbose: Whether to log detailed trading information
            training_mode: Whether environment is in training mode (reduces calculation frequency)
        """
        super().__init__()
        
        # Store inputs
        self.df = df
        self.assets = assets
        self.window_size = window_size
        self.max_leverage = max_leverage
        self.commission = commission
        self.funding_fee_multiplier = funding_fee_multiplier
        self.initial_balance = initial_balance  # Make initial balance configurable
        self.max_drawdown = max_drawdown  # Make max drawdown configurable
        self.maintenance_margin = maintenance_margin  # Maintenance margin as fraction
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose  # Use the passed parameter instead of hardcoding to False
        self.training_mode = training_mode  # Store training mode flag
        
        # Optimization properties for training mode
        self.calc_frequency = 100 if training_mode else 1  # Calculate intensive metrics every 100 steps during training
        self.last_metrics_step = 0  # Track last step when metrics were calculated
        self.last_market_analysis_step = 0  # Track last step when market conditions were analyzed
        self.last_correlation_step = 0  # Track last step when correlations were calculated
        
        # Create a default risk engine if none is provided
        if risk_engine is None:
            # Create default risk limits based on environment settings
            risk_limits = RiskLimits(
                account_max_leverage=max_leverage * 0.8,  # Less conservative (was 0.5)
                position_max_leverage=max_leverage,        # Individual position can use max leverage
                max_drawdown_pct=max_drawdown,            # Use provided max_drawdown
                position_concentration=0.5,                # Up to 50% in one asset (was 30%)
                daily_loss_limit_pct=0.15                  # Up to 15% daily loss (was 10%)
            )
            self.risk_engine = InstitutionalRiskEngine(
                initial_balance=initial_balance,
                risk_limits=risk_limits,
                use_dynamic_limits=True,
                use_vol_scaling=True
            )
            logger.info(f"Created default risk engine with limits: {risk_limits}")
        else:
            self.risk_engine = risk_engine
            logger.info("Using provided risk engine")
        
        # Store verbose flag
        self.verbose = verbose  # Use the passed parameter instead of hardcoding to False
        
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
        self.enforce_min_leverage = enforce_min_leverage  # Store flag for minimum leverage enforcement
        self.slippage = 0.0002  # Default slippage of 0.02%
        
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
                                 'leverage': 0.0,
                                 'direction': 0,   # Direction stored separately: -1=short, 0=none, 1=long
                                 'stop_loss': None,       # Stop loss price level
                                 'take_profit': None,     # Take profit price level
                                 'trailing_stop': None,   # Trailing stop distance (in %)
                                 'stop_loss_pct': None,   # Stop loss as percentage from entry
                                 'take_profit_pct': None  # Take profit as percentage from entry
                                } 
                         for asset in self.assets}
        
        # Initialize total costs tracking
        self.total_costs = 0.0
        
        # ENHANCED: Track position duration and profitability
        self.position_duration = {asset: 0 for asset in self.assets}
        self.position_profits = {asset: [] for asset in self.assets}
        self.profitable_holding_bonus = 0.0  # Will accumulate bonuses for holding profitable positions
        
        # Set signal threshold for trade triggering
        self.signal_threshold = 0.1  # Default signal threshold (action must exceed this to trigger trade)
        
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
        self.trades_history = []  # Record detailed trade information
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
        
        # Store episode termination parameters
        self.max_steps = max_steps
        self.max_no_trade_steps = max_no_trade_steps
        
        # ENHANCEMENT: Add trading cooldown tracker to prevent excessive trading
        self.last_trade_step = {asset: 0 for asset in self.assets}  # Initialize with 0 steps
        self.min_steps_between_trades = 1  # Minimum steps between trades for the same asset
        self.signal_threshold = 0.15  # Minimum signal strength required for trading (increased from 0.1)
        
        # Initialize price history
        self.price_history = {asset: deque(maxlen=1000) for asset in self.assets}
        
        # Initialize historical metrics tracking
        self.historical_metrics = {
            'returns': deque(maxlen=100),
            'volatility': deque(maxlen=100),
            'sharpe': deque(maxlen=100),
            'drawdown': deque(maxlen=100)
        }
        
        # Initialize market conditions dictionary
        self.market_conditions = {
            'volatility_regime': {asset: 0.5 for asset in self.assets},
            'trend_strength': {asset: 0.0 for asset in self.assets},
            'market_regime': 'normal',
            'overall_market_state': 'normal',  # FIXED: Add missing overall_market_state key
            'liquidity_conditions': {}  # FIXED: Add missing liquidity_conditions to avoid errors
        }
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.peak_value = self.initial_balance
        self.positions = {asset: {'size': 0, 
                                 'entry_price': 0, 
                                 'funding_accrued': 0,
                                 'last_price': self._get_mark_price(asset) if not self.df.empty else 1000.0,
                                 'leverage': 0.0,  # This is now ABSOLUTE leverage (always positive)
                                 'direction': 0,   # Direction stored separately: -1=short, 0=none, 1=long
                                 'stop_loss': None,       # Stop loss price level
                                 'take_profit': None,     # Take profit price level
                                 'trailing_stop': None,   # Trailing stop distance (in %)
                                 'stop_loss_pct': None,   # Stop loss as percentage from entry
                                 'take_profit_pct': None  # Take profit as percentage from entry
                                } 
                         for asset in self.assets}
        self.last_action = None
        
        # Use terminated and truncated instead of done
        self.terminated = False
        self.truncated = False
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
        self.trades_history = []  # CRITICAL FIX: Initialize trades_history
        
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
        
        # Reset price history
        self.price_history = {asset: [] for asset in self.assets}
        
        # CRITICAL FIX: Reset performance tracking system - using a comprehensive structure
        # Using only one definition to avoid conflicts
        self.performance_tracking = {
            # Overall performance tracking
            'portfolio_values': [],
            'returns': [],
            'drawdowns': [],
            'leverage': [],
            'sharpe_ratios': [],
            'sortino_ratios': [],
            'calmar_ratios': [],
            
            # Regime-specific performance
            'regime_performance': {
                'trending': {'returns': [], 'trades': [], 'win_rate': 0},
                'range_bound': {'returns': [], 'trades': [], 'win_rate': 0},
                'volatile': {'returns': [], 'trades': [], 'win_rate': 0},
                'crisis': {'returns': [], 'trades': [], 'win_rate': 0},
                'normal': {'returns': [], 'trades': [], 'win_rate': 0}
            },
            
            # Trade statistics
            'trades': {
                'profitable': 0,
                'unprofitable': 0,
                'total_profit': 0,
                'total_loss': 0,
                'avg_profit_per_trade': 0,
                'avg_loss_per_trade': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'avg_trade_duration': 0,
            },
            
            # Stop/take-profit effectiveness
            'risk_management': {
                'stop_loss_executions': 0,
                'take_profit_executions': 0,
                'stop_loss_pnl': 0,
                'take_profit_pnl': 0,
                'avg_stop_loss_pnl': 0,
                'avg_take_profit_pnl': 0
            },
            
            # Adaptive parameters tracking
            'adaptive_parameters': {
                'commission': [],
                'slippage': [],
                'risk_limits': []
            }
        }
        
        # Reset risk engine
        if self.risk_engine:
            self.risk_engine.reset()
            
        # Reset market conditions
        self.market_conditions = {
            'volatility_regime': {asset: 0.5 for asset in self.assets},
            'trend_strength': {asset: 0.0 for asset in self.assets},
            'market_regime': 'normal',
            'overall_market_state': 'normal',  # FIXED: Add missing overall_market_state key
            'liquidity_conditions': {}  # FIXED: Add missing liquidity_conditions to avoid errors
        }
        
        # Reset market regime
        self.market_regime = 0.5  # Neutral by default
        
        # Initialize info dict for tracking
        self.info = {}
        
        # Initialize peak values for proper drawdown calculation
        self.peak_value = self.initial_balance
        
        # Log reset
        logger.info(f"Environment reset: window_size={self.window_size}, initial_balance={self.initial_balance}")
        
        # Return observation and info dictionary
        return self._get_observation(), {}
        
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action from the agent
            
        Returns:
            next_state: Next state
            reward: Reward for the step
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        try:
            # CRITICAL FIX: Add trade execution tracking for this step
            self.position_changes_this_step = 0  # Track actual position changes
            
            # Check for stop-loss and take-profit executions first
            executed_stops, total_stop_pnl = self._check_and_execute_stops()
            
            # Skip if the account has been liquidated
            if self.liquidated:
                logger.warning("Account liquidated - skipping action.")
                return self._get_observation(), -10.0, True, False, {"liquidated": True, "balance": self.balance}
                
            # Store executed stops for info dict
            num_executed_stops = len(executed_stops)
            
            # CRITICAL FIX: If stops were executed, count as position changes
            if num_executed_stops > 0:
                self.position_changes_this_step += num_executed_stops
                
            # Calculate portfolio value at the beginning of the step
            portfolio_value = self._calculate_portfolio_value()
            self.risk_engine.update_portfolio_value(portfolio_value, self.current_step)
            
            # Store previous portfolio value for profit calculation
            prev_portfolio_value = portfolio_value
            
            # Update highest_value for drawdown tracking if needed
            if not hasattr(self, 'highest_value') or portfolio_value > self.highest_value:
                self.highest_value = portfolio_value

            # ENHANCEMENT: Skip action processing entirely every N steps for efficiency
            # This reduces computational overhead and prevents excessive trading
            should_skip_actions = False
            
            # Skip if we recently made trades and are in cooldown period
            if hasattr(self, 'last_trade_step'):
                steps_since_trade = min([self.current_step - step for asset, step in self.last_trade_step.items()])
                if steps_since_trade < self.min_steps_between_trades:
                    # We've traded recently, consider skipping action evaluation
                    should_skip_actions = np.random.random() < 0.7  # 70% chance to skip
                    if should_skip_actions and self.verbose:
                        logger.info(f"Skipping action evaluation due to recent trades ({steps_since_trade} steps ago)")
            
            # CRITICAL FIX: Handle case where we skip action processing
            if should_skip_actions:
                # Just update market and skip action processing
                self._update_market()
                portfolio_value_after = self._calculate_portfolio_value()
                
                # Calculate simple market return (no trading)
                market_pnl = portfolio_value_after - portfolio_value
                
                # Get risk metrics
                risk_metrics = self._calculate_risk_metrics(refresh_metrics=True)
                
                # Calculate reward - just based on market movement
                reward = self._calculate_risk_adjusted_reward(market_pnl, risk_metrics)
                
                # Update info
                info = self._get_info(risk_metrics)
                self.info = info
                
                # Add market conditions data to info
                info['market_conditions'] = {
                    'overall_state': self.market_conditions.get('overall_market_state', 'normal'),
                    'regime': self.market_regime,
                    'skipped_actions': True
                }
                
                # Track performance metrics
                self._track_performance_metrics()
                
                # Increment step and check if done
                self.current_step += 1
                terminated = False
                truncated = self.current_step >= self.max_steps - 1
                
                # Return observation, reward, done flags, and info
                return self._get_observation(), reward, terminated, truncated, info

            # NEW: Check if risk circuit breakers should be triggered
            circuit_breakers_triggered = self._check_risk_circuit_breakers()
            
            # If circuit breakers were triggered, recalculate portfolio value
            if circuit_breakers_triggered:
                # Recalculate portfolio value after position reduction
                portfolio_value = self._calculate_portfolio_value()
                self.risk_engine.update_portfolio_value(portfolio_value, self.current_step)
            
            # ENHANCED: Analyze market conditions for adaptive trading
            self.market_conditions = self._analyze_market_conditions()
            
            # Store the overall market regime for simpler access
            self.market_regime = 0.5  # Default neutral
            
            # Calculate average market regime across assets
            if self.market_conditions and isinstance(self.market_conditions, dict) and 'market_regime' in self.market_conditions:
                regimes = self.market_conditions['market_regime']
                # FIXED: Check if regimes is a dictionary before trying to call values() on it
                if isinstance(regimes, dict) and regimes:
                    self.market_regime = np.mean(list(regimes.values()))
                elif isinstance(regimes, str):
                    # Handle case where market_regime is a string
                    regime_value_map = {'volatile': 0.8, 'range_bound': 0.2, 'normal': 0.5, 'trending': 0.7, 'crisis': 0.9}
                    self.market_regime = regime_value_map.get(regimes, 0.5)
            
            # ENHANCED: Update environment parameters based on market conditions
            self._update_adaptive_parameters()
                
            # Update asset volatilities for dynamic position sizing
            for asset in self.assets:
                # Get recent price history for volatility calculation
                start_idx = max(0, len(self.price_history[asset]) - self.risk_engine.risk_limits.vol_lookback)
                if start_idx > 0 and len(self.price_history[asset][start_idx:]) > 5:
                    self.risk_engine.update_asset_volatility(asset, self.price_history[asset][start_idx:])
            
            # Convert action to numpy array if it isn't already
            action_np = np.array(action)
            
            # DIAGNOSTIC: Always log raw action values to diagnose issues
            action_sum = np.sum(np.abs(action_np))
            action_max = np.max(np.abs(action_np))
            logger.warning(f"DIAGNOSTIC: Raw action values - Sum: {action_sum:.6f}, Max: {action_max:.6f}, Values: {action_np}")
            
            # ENHANCEMENT: Add an adaptive filter based on market conditions
            # In stable or low-volatility market conditions, apply higher threshold
            # FIXED: Safely check if market_conditions is a dictionary and contains 'overall_market_state'
            if hasattr(self, 'market_conditions') and isinstance(self.market_conditions, dict) and 'overall_market_state' in self.market_conditions:
                market_state = self.market_conditions.get('overall_market_state', 'normal')
                
                # Determine action threshold based on market state
                action_threshold = 0.0  # Default - no filtering
                
                if market_state == 'range_bound':
                    # In range-bound markets, require stronger signals (30% increase in threshold)
                    action_threshold = self.signal_threshold * 1.3
                    if self.verbose:
                        logger.info(f"Range-bound market detected - increasing action threshold to {action_threshold:.2f}")
                elif market_state == 'normal':
                    # Normal state - apply standard threshold
                    action_threshold = self.signal_threshold
                
                # Apply adaptive threshold filtering
                if action_threshold > 0:
                    # Create a mask for actions that don't meet the threshold
                    weak_signal_mask = np.abs(action_np) < action_threshold
                    # Zero out actions that don't meet threshold
                    action_np = np.where(weak_signal_mask, 0.0, action_np)
                    
                    if self.verbose and np.any(weak_signal_mask):
                        filtered_count = np.sum(weak_signal_mask)
                        logger.info(f"Filtered {filtered_count} weak signals below threshold {action_threshold:.2f}")
            
            # DIAGNOSTIC: Log the raw action values
            if self.verbose:
                logger.info(f"Raw action values: {action_np}")
            
            # IMPROVED: Apply model uncertainty handling
            # This replaces the simple uncertainty scaling with a more sophisticated approach
            action_np = self._handle_model_uncertainty(action_np)
            
            # DIAGNOSTIC: Log action values after uncertainty handling
            if self.verbose:
                logger.info(f"Action values after uncertainty handling: {action_np}")
            
            # ENHANCED: Loop through assets and apply risk scaling
            for i, asset in enumerate(self.assets):
                if i < len(action_np):
                    # Scale action by risk parameters
                    raw_action = action_np[i]
                    
                    # Apply risk scaling to the action
                    action_np[i] = self.risk_engine.scale_action_by_risk(raw_action, asset)
            
            # Ensure action is within bounds
            action_np = np.clip(action_np, -1.0, 1.0)
            
            # DIAGNOSTIC: Log action values after risk scaling and clipping
            if self.verbose:
                logger.info(f"Final action values after risk scaling: {action_np}")
            
            # Create a dictionary of current prices for risk metrics calculation
            current_prices = {}
            for asset in self.assets:
                current_prices[asset] = self._get_mark_price(asset)
            
            # ENHANCEMENT 2: Calculate correlation-based position limits
            # OPTIMIZATION: In training mode, calculate correlation clusters less frequently
            correlation_adjustments = None
            if hasattr(self, 'df') and len(self.df) > 0:
                # In training mode, only calculate correlations periodically
                should_calculate_correlations = True
                if self.training_mode:
                    # Check if we calculated correlations recently
                    step_diff = self.current_step - self.last_correlation_step
                    should_calculate_correlations = step_diff >= self.calc_frequency
                
                if should_calculate_correlations:
                    # Create correlation data from recent price history
                    correlation_window = 100  # Use last 100 rows for correlation
                    start_idx = max(0, self.current_step - correlation_window)
                    end_idx = self.current_step + 1
                    
                    # Extract relevant market data rows
                    correlation_data_window = self.df.iloc[start_idx:end_idx]
                    
                    # Calculate correlation-based limits
                    correlation_adjustments = self.risk_engine.implement_correlation_based_position_limits(
                        self.positions, 
                        correlation_data_window,
                        self.verbose  # Pass verbose flag
                    )
                    
                    # Track last correlation calculation step
                    self.last_correlation_step = self.current_step
                    
                    # Log correlation clusters if any were found
                    asset_clusters = {}
                    for asset, data in correlation_adjustments.items():
                        cluster = data.get('cluster_membership', 'none')
                        if cluster != 'none':
                            if cluster not in asset_clusters:
                                asset_clusters[cluster] = []
                            asset_clusters[cluster].append(asset)
                    
                    if asset_clusters and self.verbose:
                        logger.info(f"Detected correlation clusters: {asset_clusters}")
                    
                    # Store correlation adjustments for reuse
                    self.correlation_adjustments = correlation_adjustments
                else:
                    # Reuse previously calculated correlation adjustments
                    correlation_adjustments = getattr(self, 'correlation_adjustments', None)
            
            # Track step risk metrics before executing trades
            risk_metrics_before = self.risk_engine.calculate_risk_metrics(
                self.positions, self.balance, current_prices
            )
            
            total_pnl = total_stop_pnl  # Initialize with PnL from stop executions
            trades_executed = (num_executed_stops > 0)  # Stops count as trades
            
            # Process each asset's action and execute trades
            for i, asset in enumerate(self.assets):
                if i < len(action_np):
                    # Get action for this asset
                    asset_action = float(action_np[i])
                    
                    # DIAGNOSTIC: Log the action signal for this asset
                    if self.verbose:
                        logger.info(f"Processing action for {asset}: signal={asset_action:.4f}")
                    
                    # Use risk engine to determine target leverage and execute trade
                    target_leverage = self._get_target_leverage(asset_action)
                    
                    # DIAGNOSTIC: Log the target leverage
                    if self.verbose:
                        logger.info(f"Target leverage for {asset}: {target_leverage:.4f}x")
                    
                    # ENHANCEMENT 3: Get market impact estimate before trading
                    # Extract market data for impact modeling
                    if hasattr(self, 'df') and len(self.df) > 0:
                        # Create market data subset for impact calculation
                        impact_window = 20  # Use last 20 rows for impact calculation
                        start_idx = max(0, self.current_step - impact_window)
                        end_idx = self.current_step + 1
                        
                        # Extract relevant market data rows
                        impact_data_window = self.df.iloc[start_idx:end_idx]
                        
                        # Get current price
                        price = self._get_mark_price(asset)
                        
                        # Estimate target position size for impact calculation
                        # This is just a rough estimate for impact modeling
                        naive_target_size = portfolio_value * target_leverage * np.sign(asset_action) / price
                        
                        # Calculate potential market impact
                        impact_estimate = self.risk_engine.calculate_market_impact(
                            asset, naive_target_size, price, impact_data_window
                        )
                        
                        # Log significant impact estimates
                        if impact_estimate['impact_bps'] > 10:
                            logger.warning(f"High estimated market impact for {asset}: " +
                                         f"{impact_estimate['impact_bps']:.1f} bps " +
                                         f"(${impact_estimate['impact_usd']:.2f})")
                    
                    # Execute trade with enhanced risk management
                    position_size, entry_price, actual_leverage, position_value, trade_cost = self._execute_trade(
                        asset, asset_action, target_leverage, portfolio_value
                    )
                    
                    # CRITICAL FIX: Only count as a trade if position actually changed
                    position_changed = abs(position_size - self.positions[asset]['size']) > 1e-8
                    if position_changed:
                        trades_executed = True
                        self.position_changes_this_step += 1
                    
                    # DIAGNOSTIC: Log the trade execution result
                    if self.verbose:
                        logger.info(f"Trade execution result for {asset}: size={position_size:.6f}, entry_price={entry_price:.2f}, " +
                                  f"leverage={actual_leverage:.2f}x, value=${position_value:.2f}, cost=${trade_cost:.2f}")
                    
                    # Track PnL from this trade
                    if abs(trade_cost) > 0:
                        # Get PnL from this trade - cost is already negative
                        pnl = trade_cost  # Just the cost for new trades
                        total_pnl += pnl
            
            # Update market prices and calculate unrealized PnL
            self._update_market()
            
            # Recalculate portfolio value after trades and market update
            portfolio_value_after = self._calculate_portfolio_value()
            
            # Calculate mark-to-market PnL
            market_pnl = portfolio_value_after - portfolio_value - total_pnl
            total_pnl += market_pnl
            
            # Update price dictionary for risk metrics calculation
            for asset in self.assets:
                current_prices[asset] = self._get_mark_price(asset)
            
            # Update risk metrics after trades
            risk_metrics = self.risk_engine.calculate_risk_metrics(
                self.positions, self.balance, current_prices
            )
            
            # CRITICAL FIX: Update consecutive_no_trade_steps correctly
            # Only count as no-trade if NO position changes happened
            if self.position_changes_this_step > 0:
                self.consecutive_no_trade_steps = 0
            else:
                self.consecutive_no_trade_steps += 1
            
            # Check liquidation
            if portfolio_value_after <= 0 or self.balance <= 0:
                logger.warning(f"Account liquidated: Portfolio value: {portfolio_value_after}, Balance: {self.balance}")
                self.liquidated = True
                reward = -10.0
                terminated = True
                truncated = False
            else:
                # Calculate reward
                reward = self._calculate_risk_adjusted_reward(total_pnl, risk_metrics)
                terminated = False
                truncated = False
                
                # Check if we're done due to reaching max steps
                if self.current_step >= self.max_steps - 1:
                    truncated = True
                    
                # Check if we're done due to reaching stop criteria
                if self.consecutive_no_trade_steps >= self.max_no_trade_steps:
                    logger.info(f"Stopping after {self.consecutive_no_trade_steps} steps without trading")
                    truncated = True
            
            # Get info dict
            info = self._get_info(risk_metrics)
            
            # Store the info dict for performance tracking
            self.info = info
            
            # Add stop loss/take profit execution info
            info['stop_executions'] = num_executed_stops
            if num_executed_stops > 0:
                info['executed_stops'] = num_executed_stops
                info['stop_pnl'] = total_stop_pnl
                
            # Add circuit breaker info if triggered
            if circuit_breakers_triggered:
                info['circuit_breaker_triggered'] = True
                
            # Add market conditions data to info
            info['market_conditions'] = {
                'overall_state': self.market_conditions.get('overall_market_state', 'normal'),
                'regime': self.market_regime,
                'asset_regimes': self.market_conditions.get('market_regime', {}),
                'volatility_regimes': self.market_conditions.get('volatility_regime', {})
            }
            
            # CRITICAL FIX: Add the count of actual position changes to info
            info['position_changes'] = self.position_changes_this_step
            
            # Add abnormal price movement indicators
            abnormal_movements = self.market_conditions.get('abnormal_price_movements', {})
            if any(abnormal_movements.values()):
                info['abnormal_price_movements'] = abnormal_movements
            
            # CRITICAL FIX: Track performance metrics only once per step
            # Only track after all trades are executed and all calculations are done
            self._track_performance_metrics()
            
            # Add summary performance metrics to info on every 10th step or last step
            if terminated or truncated or self.current_step % 10 == 0:
                info['performance_summary'] = self.get_performance_summary()
            
            # CRITICAL FIX: Increment step only once per step 
            self.current_step += 1
            
            # Check if we've reached the maximum allowed steps
            truncated = self.current_step >= len(self.df) - 1 or self.current_step >= self.max_steps
            
            # Add at the end of the step method, before returning the observation, reward, etc.
            # Log portfolio summary at the end of the step
            if self.verbose:
                # Calculate total portfolio statistics
                total_notional_value = 0
                total_margin_used = 0
                total_unrealized_pnl = 0
                active_positions_count = 0
                
                # Calculate current portfolio value
                portfolio_value = self._calculate_portfolio_value()
                
                # Format header as a table
                logger.info(f"\n{'='*70}")
                logger.info(f"|| PORTFOLIO SUMMARY - STEP {self.current_step} {' '*(44-len(str(self.current_step)))}||")
                logger.info(f"||{'-'*68}||")
                logger.info(f"|| Balance: ${self.balance:.2f} | Portfolio Value: ${portfolio_value:.2f} {' '*(21-len(f'{portfolio_value:.2f}'))}||")
                logger.info(f"||{'-'*68}||")
                
                # Table header for positions if any exist
                has_active_positions = any(abs(p['size']) > 1e-8 for p in self.positions.values())
                if has_active_positions:
                    logger.info(f"|| {'ASSET':<8} | {'DIRECTION':<8} | {'SIZE':<10} | {'ENTRY':<8} | {'CURRENT':<8} | {'PNL':<12} ||")
                    logger.info(f"||{'-'*68}||")
                
                # Log each position
                for asset, position in self.positions.items():
                    position_size = position['size']
                    
                    if abs(position_size) > 1e-8:  # Only log active positions
                        active_positions_count += 1
                        current_price = position.get('mark_price', self._get_mark_price(asset))
                        entry_price = position['entry_price']
                        leverage = position.get('leverage', 1.0)
                        
                        # Calculate position metrics
                        notional_value = abs(position_size * current_price)
                        
                        # Get actual position leverage, using a default of max_leverage if not set
                        leverage = min(position.get('leverage', self.max_leverage), self.max_leverage)
                        leverage = max(1.0, leverage)  # Ensure leverage is at least 1.0
                        
                        # Calculate margin used correctly based on leverage
                        margin_used = notional_value / leverage if leverage > 0 else notional_value
                        
                        # Ensure margin used doesn't exceed notional value for leverage < 1
                        margin_used = min(margin_used, notional_value)
                        
                        # Calculate PnL
                        if position_size > 0:  # Long
                            unrealized_pnl = position_size * (current_price - entry_price)
                            direction = "LONG"
                        else:  # Short
                            unrealized_pnl = abs(position_size) * (entry_price - current_price)
                            direction = "SHORT"
                        
                        # Add to totals
                        total_notional_value += notional_value
                        total_margin_used += margin_used
                        total_unrealized_pnl += unrealized_pnl
                        
                        # Calculate PnL percentage
                        pnl_pct = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
                        
                        # Format position row
                        pnl_str = f"${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
                        logger.info(f"|| {asset:<8} | {direction:<8} | {position_size:<10.6f} | ${entry_price:<7.2f} | ${current_price:<7.2f} | {pnl_str:<12} ||")
                
                # Print summary footer
                logger.info(f"||{'-'*68}||")
                logger.info(f"|| ACCOUNT METRICS {' '*51}||")
                logger.info(f"||{'-'*68}||")
                
                # Calculate account metrics
                available_balance = portfolio_value - total_margin_used
                
                # Check if margin used exceeds portfolio value, which shouldn't be possible
                if total_margin_used > portfolio_value:
                    logger.warning(f"Margin used (${total_margin_used:.2f}) exceeds portfolio value (${portfolio_value:.2f}). Capping at portfolio value.")
                    total_margin_used = portfolio_value
                    available_balance = 0
                
                # Calculate margin ratio based on portfolio value, not just balance
                margin_ratio = (total_margin_used / portfolio_value) * 100 if portfolio_value > 0 else 0
                
                # Calculate leverage based on notional value and portfolio value
                total_leverage = total_notional_value / portfolio_value if portfolio_value > 0 else 0
                
                # Check if leverage exceeds max leverage
                if total_leverage > self.max_leverage:
                    logger.warning(f"Current leverage ({total_leverage:.2f}x) exceeds max leverage ({self.max_leverage:.2f}x)")
                
                # Log account summary
                logger.info(f"|| Available Balance:     ${available_balance:<10.2f} {' '*36}||")
                logger.info(f"|| Total Notional Value:  ${total_notional_value:<10.2f} {' '*36}||")
                logger.info(f"|| Total Margin Used:     ${total_margin_used:<10.2f} ({margin_ratio:.2f}% of portfolio) {' '*10}||")
                logger.info(f"|| Total Unrealized PnL:  ${total_unrealized_pnl:<10.2f} {' '*36}||")
                logger.info(f"|| Account Leverage:      {total_leverage:<10.2f}x {' '*36}||")
                logger.info(f"|| Active Positions:      {active_positions_count}/{len(self.assets)} {' '*36}||")
                logger.info(f"{'='*70}\n")
            
            # Return the observation, reward, done flag, and info
            return self._get_observation(), reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            traceback.print_exc()
            # Return a default observation, a large negative reward, and a done flag
            return self._get_observation(), -10.0, True, False, {"error": str(e)}

    def _simulate_trades(self, trades: List[Dict]) -> Dict:
        """Simulate trades to check risk limits"""
        try:
            # PERFORMANCE OPTIMIZATION: Cache all mark prices 
            mark_prices = {}
            
            # First identify all assets we'll need prices for
            assets_to_check = set(self.positions.keys())
            for trade in trades:
                if 'asset' in trade:
                    assets_to_check.add(trade['asset'])
            
            # Cache all prices in one go
            for asset in assets_to_check:
                mark_prices[asset] = self._get_mark_price(asset)
            
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
                        simulated_positions[asset] = {
                            'size': size_change,
                            'entry_price': mark_prices[asset],
                            'last_price': mark_prices[asset]
                        }
            
            # Calculate simulated risk metrics
            portfolio_value = self.balance  # Start with cash balance
            gross_exposure = 0  # Total absolute exposure (for risk limits)
            net_exposure = 0    # Net directional exposure (for leverage direction)
            asset_values = {}
            total_unrealized_pnl = 0
            
            # OPTIMIZATION: Calculate unrealized PnL and exposures in a single loop
            for asset, position in simulated_positions.items():
                # Skip assets with no position
                position_size = position['size']
                if abs(position_size) < 1e-8:
                    continue
                    
                # Use cached price
                price = mark_prices[asset]
                
                # Calculate unrealized PnL for both long and short positions
                entry_price = position['entry_price']
                unrealized_pnl = position_size * (price - entry_price)
                total_unrealized_pnl += unrealized_pnl
                
                # Calculate exposures
                position_value = position_size * price  # Signed value (negative for shorts)
                position_exposure = abs(position_value)  # Absolute value (for risk limits)
                
                gross_exposure += position_exposure
                net_exposure += position_value
                asset_values[asset] = position_exposure
            
            # Update portfolio value with unrealized PnL
            portfolio_value += total_unrealized_pnl
            
            # Ensure portfolio value isn't extremely negative
            portfolio_value = max(portfolio_value, self.initial_balance * 0.01)
            
            # OPTIMIZATION: Calculate margin and leverage in a single loop
            total_margin = 0
            
            for asset, position in simulated_positions.items():
                position_size = position['size']
                if abs(position_size) < 1e-8:
                    continue
                    
                # Get leverage with limits
                leverage = min(position.get('leverage', self.max_leverage), self.max_leverage)
                leverage = max(1.0, leverage)  # Ensure minimum leverage
                
                # Calculate margin using cached price
                notional = abs(position_size * mark_prices[asset])
                position_margin = notional / leverage if leverage > 0 else notional
                total_margin += position_margin
            
            # Calculate leverage ratios
            gross_leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0
            net_leverage = net_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Check for risk limit violations
            risk_limit_exceeded = False
            exceeded_limits = []
            
            # Check margin limit
            if total_margin > portfolio_value:
                risk_limit_exceeded = True
                exceeded_limits.append(f"Margin (${total_margin:.2f}) exceeds portfolio value (${portfolio_value:.2f})")
                
                # Return immediately with no scaled trades
                return {
                    'portfolio_value': portfolio_value,
                    'gross_leverage': gross_leverage,
                    'net_leverage': net_leverage,
                    'max_concentration': max(asset_values.values()) / portfolio_value if asset_values else 0,
                    'risk_limit_exceeded': True,
                    'exceeded_limits': exceeded_limits,
                    'scaled_trades': []  # No trades will be executed
                }
            
            # Check leverage limit
            max_allowed_leverage = self.max_leverage
            if gross_leverage > max_allowed_leverage:
                risk_limit_exceeded = True
                exceeded_limits.append(f"Leverage {gross_leverage:.2f}x > {max_allowed_leverage:.2f}x")
                
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
                
                # Log leverage scaling
                logger.warning(f"Scaling down trades to maintain leverage below {max_allowed_leverage:.2f}x (was {gross_leverage:.2f}x)")
                
                return {
                    'portfolio_value': portfolio_value,
                    'gross_leverage': max_allowed_leverage,  # Use the limit as the new leverage
                    'net_leverage': net_leverage * scale_factor,
                    'max_concentration': max(asset_values.values()) / portfolio_value if asset_values else 0,
                    'risk_limit_exceeded': True,
                    'exceeded_limits': exceeded_limits,
                    'scaled_trades': scaled_trades
                }
            
            # Check concentration limit
            max_concentration = 0
            max_asset = ""
            for asset, value in asset_values.items():
                concentration = value / portfolio_value
                if concentration > max_concentration:
                    max_concentration = concentration
                    max_asset = asset
            
            concentration_limit = 0.4  # Default
            if self.risk_engine:
                concentration_limit = self.risk_engine.risk_limits.position_concentration
                
            if max_concentration > concentration_limit:
                risk_limit_exceeded = True
                exceeded_limits.append(f"Concentration {max_concentration:.2%} > {concentration_limit:.2%} for {max_asset}")
            
            # Scale down trades if any risk limits exceeded
            scaled_trades = []
            if risk_limit_exceeded and len(trades) > 0:
                # Calculate scaling factor
                scale_factor = 0.8  # Default scale down by 20%
                
                if gross_leverage > self.max_leverage:
                    leverage_scale = (self.max_leverage * 0.9) / gross_leverage
                    scale_factor = min(scale_factor, leverage_scale)
                
                if max_concentration > concentration_limit:
                    concentration_scale = (concentration_limit * 0.9) / max_concentration
                    scale_factor = min(scale_factor, concentration_scale)
                
                # Apply scaling to all trades
                for trade in trades:
                    if 'size_change' in trade:
                        scaled_trade = trade.copy()
                        scaled_trade['size_change'] = trade['size_change'] * scale_factor
                        if abs(scaled_trade['size_change']) > 1e-8:
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
        """
        Execute a list of trades and update positions
        
        Args:
            trades: List of trade dictionaries
        """
        try:
            if not trades:
                return
                
            # Get current portfolio value before execution
            current_portfolio_value = self._calculate_portfolio_value()
            
            # PERFORMANCE OPTIMIZATION: Cache all mark prices once
            mark_prices = {asset: self._get_mark_price(asset) for asset in self.positions.keys()}
            # Also cache mark prices for any assets in trades that might not be in positions yet
            for trade in trades:
                if 'asset' in trade and trade['asset'] not in mark_prices:
                    mark_prices[trade['asset']] = self._get_mark_price(trade['asset'])
            
            # First pass: estimate margin for each position after proposed trades
            simulated_positions = copy.deepcopy(self.positions)
            
            # Update simulated positions with the proposed trades
            for trade in trades:
                asset = trade['asset']
                
                if 'size_change' in trade:
                    size_change = trade['size_change']
                    
                    if asset in simulated_positions:
                        # Update existing position
                        simulated_positions[asset]['size'] += size_change
                    else:
                        # Create new position
                        leverage = trade.get('leverage', 1.0)
                        simulated_positions[asset] = {
                            'size': size_change,
                            'entry_price': mark_prices[asset],
                            'mark_price': mark_prices[asset],
                            'leverage': leverage
                        }
            
            # Calculate total margin that would be used after these trades
            estimated_total_margin = 0
            total_notional_after_trade = 0
            
            # PERFORMANCE OPTIMIZATION: Calculate margin and notional in a single loop
            for asset, position in simulated_positions.items():
                position_size = position['size']
                
                # Skip if position would be effectively closed
                if abs(position_size) < 1e-8:
                    continue
                    
                # Get leverage for this position
                leverage = min(position.get('leverage', self.max_leverage), self.max_leverage)
                leverage = max(1.0, leverage)  # Ensure leverage is at least 1.0
                
                # Calculate position metrics using cached price
                notional_value = abs(position_size * mark_prices[asset])
                position_margin = notional_value / leverage if leverage > 0 else notional_value
                
                # Add to totals
                estimated_total_margin += position_margin
                total_notional_after_trade += notional_value
            
            # Check if margin would exceed portfolio value
            if estimated_total_margin > current_portfolio_value:
                logger.warning(f"Trade rejected: Estimated margin (${estimated_total_margin:.2f}) would exceed portfolio value (${current_portfolio_value:.2f})")
                return
            
            # Check if leverage would exceed max account leverage
            estimated_leverage = total_notional_after_trade / current_portfolio_value if current_portfolio_value > 0 else 0
            if estimated_leverage > self.max_leverage:
                logger.warning(f"Trade rejected: Estimated leverage ({estimated_leverage:.2f}x) would exceed max account leverage ({self.max_leverage:.2f}x)")
                return
            
            # Execute each trade
            for trade in trades:
                asset = trade['asset']
                
                # Handle different trade dictionary structures
                if 'size_change' in trade:
                    size_change = trade['size_change']
                    execution_price = trade.get('execution_price', mark_prices[asset])
                    target_leverage = trade.get('leverage', 1.0)  # Default to 1.0x leverage
                    
                    # Skip trades with negligible size
                    if abs(size_change) < 1e-8:
                        continue
                    
                    # Calculate trade value for cost calculation
                    trade_value = abs(size_change * execution_price)
                    
                    # Calculate transaction cost
                    cost = self._calculate_transaction_cost(trade_value)
                    
                    # Log the trade
                    if self.verbose:
                        direction = "buy" if size_change > 0 else "sell"
                        leverage_str = f", leverage={target_leverage:.2f}x" if target_leverage != 1.0 else ""
                        logger.info(f"Executing trade for {asset}: size change {size_change} at ${execution_price:.2f}{leverage_str}")
                    
                    # Update position
                    if asset in self.positions:
                        # Get existing position details
                        current_size = self.positions[asset]['size']
                        current_entry = self.positions[asset]['entry_price']
                        
                        # CRITICAL FIX: Calculate new entry price based on direction
                        # If adding to position, calculate weighted average
                        if (current_size > 0 and size_change > 0) or (current_size < 0 and size_change < 0):
                            # Adding to existing position (same direction)
                            new_size = current_size + size_change
                            if abs(new_size) > 1e-8:  # Avoid division by zero
                                new_entry = (current_size * current_entry + size_change * execution_price) / new_size
                            else:
                                new_entry = current_entry  # Position being closed, entry doesn't matter
                        else:
                            # Reducing or reversing position
                            if abs(current_size) <= abs(size_change):
                                # Position is being closed and possibly reversed
                                if abs(current_size + size_change) < 1e-8:
                                    # Position closed completely
                                    new_entry = 0  # Will be reset if needed
                                    if self.verbose:
                                        logger.info(f"Position closed for {asset}")
                                else:
                                    # Position reversed, use execution price as new entry
                                    new_entry = execution_price
                            else:
                                # Position reduced but not closed or reversed, keep same entry
                                new_entry = current_entry
                        
                        # Update position
                        if abs(current_size + size_change) < 1e-8:
                            # Position effectively closed
                            self.positions[asset]['size'] = 0
                            self.positions[asset]['entry_price'] = 0
                            # Preserve other fields like funding_accrued
                        else:
                            # Update position size and entry price
                            self.positions[asset]['size'] = current_size + size_change
                            self.positions[asset]['entry_price'] = new_entry
                            self.positions[asset]['leverage'] = target_leverage
                        
                        # Add trade to history with updated position details
                        trade_info = {
                            'asset': asset,
                            'timestamp': self.current_step,
                            'size_change': size_change,
                            'price': execution_price,
                            'value': trade_value,
                            'cost': cost,
                            'type': 'add' if (current_size > 0 and size_change > 0) or (current_size < 0 and size_change < 0) else 'reduce'
                        }
                        
                        # ENHANCED TRACKING: Record the reason for the trade and its PnL
                        if 'reason' in trade:
                            trade_info['reason'] = trade['reason']
                        
                        # Add to trades history
                        self.trades_history.append(trade_info)
                        
                    else:
                        # New position
                        self.positions[asset] = {
                            'size': size_change,
                            'entry_price': execution_price,
                            'leverage': target_leverage,
                            'last_size_change': self.current_step,
                            'stop_loss': None,
                            'take_profit': None,
                            'trailing_stop': None,
                            'funding_accrued': 0.0
                        }
                        
                        # Add to trades history
                        trade_info = {
                            'asset': asset,
                            'timestamp': self.current_step,
                            'size_change': size_change,
                            'price': execution_price,
                            'value': trade_value,
                            'cost': cost,
                            'type': 'new'
                        }
                        
                        # ENHANCED TRACKING: Record the reason for the trade
                        if 'reason' in trade:
                            trade_info['reason'] = trade['reason']
                            
                        self.trades_history.append(trade_info)
                    
                    # Update last trade step
                    self.last_trade_step[asset] = self.current_step
                    
                    # Deduct cost from balance
                    self.balance -= cost
                    
                    # Get updated position after trade
                    position = self.positions[asset]
                    position_size = position['size']
                    
                    # Log updated position
                    if self.verbose and abs(position_size) > 1e-8:
                        # Calculate position details for logging
                        notional_value = abs(position_size * execution_price)
                        current_leverage = position.get('leverage', 1.0)
                        margin_used = notional_value / current_leverage if current_leverage > 0 else notional_value
                        
                        # Calculate unrealized PnL and PnL percentage
                        if position_size > 0:  # Long
                            unrealized_pnl = position_size * (execution_price - position['entry_price'])
                            direction_str = "LONG"
                        else:  # Short
                            unrealized_pnl = abs(position_size) * (position['entry_price'] - execution_price)
                            direction_str = "SHORT"
                        
                        pnl_pct = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
                        
                        # Format position details as a table
                        old_size = position_size - size_change
                        if old_size > 0 and position_size > 0:
                            action = "Increased long"
                        elif old_size < 0 and position_size < 0:
                            action = "Increased short"
                        elif old_size > 0 and position_size < 0:
                            action = "Reversed to short"
                        elif old_size < 0 and position_size > 0:
                            action = "Reversed to long"
                        elif old_size > 0 and position_size < old_size:
                            action = "Reduced long"
                        elif old_size < 0 and position_size > old_size:
                            action = "Reduced short"
                        else:
                            action = "Updated"
                            
                        logger.info(f"{action} position: {position_size} {asset} (was: {old_size}), new entry: {position['entry_price']:.2f}, leverage: {current_leverage:.2f}x")
                        
                        # Format position details as a table
                        logger.info(f"\n{'-'*70}")
                        logger.info(f"| UPDATED POSITION: {asset} {direction_str} {' '*(49-len(asset)-len(direction_str))}|")
                        logger.info(f"|{'-'*68}|")
                        logger.info(f"| Asset             | {asset:<20} {' '*30}|")
                        logger.info(f"| Position Size     | {position_size:<20.6f} {' '*30}|")
                        logger.info(f"| Entry Price       | ${position['entry_price']:<19.2f} {' '*30}|")
                        logger.info(f"| Execution Price   | ${execution_price:<19.2f} {' '*30}|")
                        logger.info(f"| Notional Value    | ${notional_value:<19.2f} {' '*30}|")
                        logger.info(f"| Leverage          | {current_leverage:<19.2f}x {' '*30}|")
                        logger.info(f"| Margin Used       | ${margin_used:<19.2f} {' '*30}|")
                        logger.info(f"| Unrealized PnL    | ${unrealized_pnl:<19.2f} ({pnl_pct:+.2f}%) {' '*19}|")
                        
                        # Add risk parameters if available
                        if 'stop_loss' in position and position['stop_loss'] is not None:
                            logger.info(f"| Stop Loss         | ${position['stop_loss']:<19.2f} {' '*30}|")
                        if 'take_profit' in position and position['take_profit'] is not None:
                            logger.info(f"| Take Profit       | ${position['take_profit']:<19.2f} {' '*30}|")
                        if 'funding_accrued' in position:
                            logger.info(f"| Funding Accrued   | ${position['funding_accrued']:<19.2f} {' '*30}|")
                        
                        logger.info(f"{'-'*70}\n")
                        
                        # If position was closed, log it
                        if abs(position_size) < 1e-8:
                            logger.info(f"Position closed for {asset}")
                
                # Handle other trade dictionary structures...
                
                # Track this trade execution
                self.performance_tracking['last_trade_step'] = self.current_step
            
            # Update position status
            self._update_positions()
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            traceback.print_exc()

    def _update_positions(self):
        """Update positions with current market prices and unrealized PnL"""
        portfolio_value = 0.0
        total_unrealized_pnl = 0.0
        
        for asset in self.assets:
            # Get current mark price for the asset
            mark_price = self._get_mark_price(asset)
            
            # Update mark price in position
            self.positions[asset]['mark_price'] = mark_price
            
            # Update last price for reference
            self.last_prices[asset] = mark_price
            
            # Store price in history
            self.price_history[asset].append(mark_price)
            
            # Calculate unrealized PnL if we have a position
            if abs(self.positions[asset]['size']) > 1e-8:
                # Get position data
                size = self.positions[asset]['size']
                entry_price = self.positions[asset]['entry_price']
                
                # Calculate unrealized PnL
                if size > 0:  # Long position
                    unrealized_pnl = size * (mark_price - entry_price)
                else:  # Short position
                    unrealized_pnl = size * (entry_price - mark_price)
                
                # Update position data
                self.positions[asset]['unrealized_pnl'] = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl
                
                # Add to portfolio value (position notional value)
                position_value = size * mark_price
                portfolio_value += position_value
        
        # Update portfolio value including balance and unrealized PnL
        self.portfolio_value = self.balance + total_unrealized_pnl
        
        return total_unrealized_pnl
    
    def _update_market(self):
        """
        Update market prices and calculate unrealized PnL for all positions.
        This method is called during each step to refresh asset prices and position values.
        """
        # Initialize total unrealized PnL
        total_unrealized_pnl = 0.0
        
        # Update prices and PnL for each asset
        for asset in self.assets:
            # Get current mark price
            current_price = self._get_mark_price(asset)
            
            # Store the price in last_prices dictionary
            self.last_prices[asset] = current_price
            
            # Add to price history for volatility calculations
            if asset not in self.price_history:
                self.price_history[asset] = []
            self.price_history[asset].append(current_price)
            
            # Limit price history size to avoid memory issues
            max_history = 1000  # Reasonable history length
            if len(self.price_history[asset]) > max_history:
                self.price_history[asset] = self.price_history[asset][-max_history:]
                
            # Get current position for this asset
            position = self.positions[asset]
            
            # Update mark price in position data
            position['mark_price'] = current_price
            
            # Skip assets with no position
            if abs(position['size']) < 1e-8:
                position['unrealized_pnl'] = 0
                continue
            
            # Calculate unrealized P&L
            entry_price = position['entry_price']
            position_size = position['size']
            
            if position_size > 0:  # Long position
                unrealized_pnl = position_size * (current_price - entry_price)
            else:  # Short position
                unrealized_pnl = position_size * (current_price - entry_price)
                
            # Update position P&L
            position['unrealized_pnl'] = unrealized_pnl
            total_unrealized_pnl += unrealized_pnl
            
            # Calculate and update funding payments if applicable
            if hasattr(self, 'funding_rate_updated') and self.funding_rate_updated:
                funding_rate = self._get_funding_rate(asset)
                funding_payment = -position_size * current_price * funding_rate  # Negative for longs when positive rate
                self.balance += funding_payment
                
                # Log significant funding payments
                if abs(funding_payment) > 1.0 and self.verbose:
                    logger.info(f"Funding payment for {asset}: {funding_payment:.2f} (rate: {funding_rate:.6f})")
                    
            # Check and update trailing stops if needed
            self._update_trailing_stop(asset, current_price)
                
        # Updated current total unrealized PnL
        self.total_unrealized_pnl = total_unrealized_pnl
        
        # Update portfolio value
        self.portfolio_value = self.balance + total_unrealized_pnl
        
        # Reset funding rate flag
        if hasattr(self, 'funding_rate_updated'):
            self.funding_rate_updated = False
            
        return total_unrealized_pnl
        
    def _update_trailing_stop(self, asset: str, current_price: float):
        """
        Update the trailing stop level based on price movement
        
        Args:
            asset: Asset symbol
            current_price: Current market price
        """
        position = self.positions[asset]
        
        # Skip if no trailing stop or no position
        if position['trailing_stop'] is None or abs(position['size']) < 1e-8:
            return
            
        trailing_pct = position['trailing_stop']
        
        # For long positions, trailing stop moves up with price
        if position['size'] > 0:
            # Calculate potential new stop level
            if position['stop_loss'] is None:
                # Initialize stop loss if none exists
                position['stop_loss'] = current_price * (1 - trailing_pct)
            else:
                # Update only if price moved higher and would result in a higher stop
                potential_stop = current_price * (1 - trailing_pct)
                if potential_stop > position['stop_loss']:
                    position['stop_loss'] = potential_stop
                    if self.verbose:
                        logger.info(f"Updated trailing stop for {asset} long: {position['stop_loss']:.2f}")
        
        # For short positions, trailing stop moves down with price
        else:
            # Calculate potential new stop level
            if position['stop_loss'] is None:
                # Initialize stop loss if none exists
                position['stop_loss'] = current_price * (1 + trailing_pct)
            else:
                # Update only if price moved lower and would result in a lower stop
                potential_stop = current_price * (1 + trailing_pct)
                if potential_stop < position['stop_loss']:
                    position['stop_loss'] = potential_stop
                    if self.verbose:
                        logger.info(f"Updated trailing stop for {asset} short: {position['stop_loss']:.2f}")
            
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
            
            # Also set self.portfolio_value for other methods to access
            self.portfolio_value = final_value
            
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
        # CRITICAL CHANGE: Always calculate risk metrics regardless of training mode
        # We'll keep the skip-check but only for cases with no positions
        
        # If we don't have positions and not explicitly refreshing metrics, return cached metrics
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
                    # CRITICAL FIX: Convert positions_data list to dictionary keyed by asset
                    positions_dict = {}
                    for pos in positions_data:
                        if 'asset' in pos:
                            positions_dict[pos['asset']] = pos
                    
                    # Get current prices for each asset
                    prices_dict = {}
                    for asset in self.positions:
                        if asset in positions_dict and 'mark_price' in positions_dict[asset]:
                            prices_dict[asset] = positions_dict[asset]['mark_price']
                    
                    # Call risk engine with properly formatted data
                    risk_metrics = self.risk_engine.calculate_risk_metrics(
                        positions=positions_dict,
                        balance=self.balance,
                        prices=prices_dict
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
                
            # CRITICAL CHANGE: Always calculate risk-adjusted ratios
            # Risk-adjusted ratios calculation (with safety checks)
            if len(self.portfolio_history) > 1:
                try:
                    # Calculate returns
                    returns = []
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
                    
            # Update last metrics calculation step
            self.last_metrics_step = self.current_step
            
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
            
            # IMPROVED: More balanced reward structure with linear scaling and tiered incentives
            if portfolio_return > 0:
                # Linear scaling for moderate returns, with bonus for exceptional returns
                if portfolio_return < 0.005:  # Under 0.5%
                    base_reward = portfolio_return * 15.0
                else:  # 0.5% or higher
                    base_reward = (0.005 * 15.0) + ((portfolio_return - 0.005) * 25.0)
            else:
                # For negative returns: slightly reduced penalty
                base_reward = -np.sqrt(abs(portfolio_return)) * 6.0
            
            # Risk-adjusted components with more emphasis on risk management
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            sortino = risk_metrics.get('sortino_ratio', 0)
            calmar = risk_metrics.get('calmar_ratio', 0)
            
            # CRITICAL FIX: Add sanity checks for risk metrics
            sharpe = np.clip(sharpe, -5, 5)
            sortino = np.clip(sortino, -5, 5)
            calmar = np.clip(calmar, -5, 5)
            
            # IMPROVED: Weighted risk metrics with more emphasis on downside protection
            risk_reward = sharpe * 0.3 + sortino * 0.4 + calmar * 0.3
            
            # ENHANCED: Dynamic risk penalties based on market regime
            # Apply stronger penalties in high volatility regimes
            volatility_scale = 1.0
            if 'volatility' in risk_metrics:
                # Higher vol = stricter penalties
                vol_percentile = risk_metrics.get('volatility_percentile', 0.5)
                volatility_scale = 1.0 + vol_percentile
            
            # CRITICAL FIX: Penalize excessive risk more aggressively, scaled by volatility
            leverage_penalty = 0.0
            concentration_penalty = 0.0
            
            # Calculate account-wide leverage limits from risk engine
            account_max_leverage = self.risk_engine.risk_limits.account_max_leverage if self.risk_engine else 5.0
            position_max_leverage = self.max_leverage
            
            # Penalize high leverage relative to account limits - more strict
            current_leverage = risk_metrics.get('leverage_utilization', 0)
            if current_leverage > account_max_leverage * 0.5:  # Start penalizing at 50% of max
                leverage_ratio = current_leverage / account_max_leverage
                leverage_penalty = (leverage_ratio - 0.5) * 2.0 * volatility_scale
            
            # Penalize high concentration
            max_concentration = self.risk_engine.risk_limits.position_concentration if self.risk_engine else 0.4
            current_concentration = risk_metrics.get('max_concentration', 0)
            if current_concentration > max_concentration * 0.5:  # Start penalizing at 50% of limit
                concentration_ratio = current_concentration / max_concentration
                concentration_penalty = (concentration_ratio - 0.5) * 3.0 * volatility_scale
            
            # IMPROVED: Drawdown penalty with nonlinear scaling
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            # Use a quadratic penalty that increases sharply as drawdown grows
            drawdown_penalty = min(max_drawdown * max_drawdown * 20.0, 2.0)
            
            # IMPROVED: Stronger trading activity incentive with regime awareness
            trade_count = len([t for t in self.trades if t['timestamp'] == self.current_step])
            
            # Trading incentive depends on market regime
            # In trending markets, reward fewer larger trades
            # In range-bound markets, reward more frequent smaller trades
            market_regime = getattr(self, 'market_regime', 0.5)  # Default to middle
            
            # Adjust trading incentive based on regime
            if market_regime > 0.7:  # Strong trend
                # In trending markets: reward fewer, larger trades
                if trade_count == 0:
                    trading_incentive = -0.02
                elif trade_count == 1:
                    trading_incentive = 0.04  # Reward single decisive trade
                else:
                    trading_incentive = 0.04 - (trade_count - 1) * 0.01  # Penalize too many trades
            else:  # Range-bound or weak trend
                # In range-bound markets: reward more frequent trading
                if trade_count == 0:
                    trading_incentive = -0.03  # Stronger penalty for inaction
                elif 1 <= trade_count <= 3:
                    trading_incentive = 0.02 * trade_count  # Reward multiple trades
                else:
                    trading_incentive = 0.06 - (trade_count - 3) * 0.01  # Diminishing returns
            
            # ENHANCED: Dynamic balance penalties based on drawdown
            balance_penalty = 0.0
            portfolio_value_ratio = portfolio_value / self.initial_balance
            
            if portfolio_value_ratio < 0.5:  # Lost more than 50%
                balance_penalty = 1.0 + (0.5 - portfolio_value_ratio) * 3.0  # Stronger penalty
            elif portfolio_value_ratio < 0.7:  # Lost 30-50%
                balance_penalty = 0.5 + (0.7 - portfolio_value_ratio) * 1.0  # Moderate penalty
            elif portfolio_value_ratio < 0.9:  # Lost 10-30%
                balance_penalty = (0.9 - portfolio_value_ratio) * 0.3  # Light penalty
                
            # ENHANCED: Add bonus for maintaining balance above initial
            balance_bonus = 0.0
            if portfolio_value_ratio > 1.05:  # 5% above initial
                # Add increasing bonus for better performance using log scale
                # This rewards early gains more than later gains
                balance_bonus = np.log10(portfolio_value_ratio) * 1.5
            
            # Combine all components with adjusted weights
            # Base formula: reward = pnl_reward + risk_metrics - risk_penalties + incentives - penalties
            reward = (
                base_reward * 0.4 +                    # 40% weight to P&L
                risk_reward * 0.3 +                    # 30% to risk-adjusted metrics
                trading_incentive * 0.1 +              # 10% to trading activity
                balance_bonus * 0.1 -                  # 10% to balance growth bonus
                leverage_penalty * 0.25 -              # 25% to leverage penalty
                concentration_penalty * 0.15 -         # 15% to concentration penalty
                drawdown_penalty * 0.3 -               # 30% to drawdown penalty
                balance_penalty * 0.2                  # 20% to balance penalty
            )
            
            # Bound the reward to prevent extreme values
            reward = np.clip(reward, -15.0, 15.0)
            
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

    def _execute_trade(self, asset: str, signal: float, target_leverage: float, portfolio_value: float) -> tuple:
        """
        Execute a trade on an asset with the given signal and target leverage
        
        Args:
            asset: Asset symbol
            signal: Action signal in range [-1.0, 1.0]
            target_leverage: Target leverage for this position
            portfolio_value: Current portfolio value
            
        Returns:
            tuple: (position_size, entry_price, actual_leverage, position_value, cost)
        """
        try:
            # PERFORMANCE OPTIMIZATION: Cache mark prices for all assets upfront
            current_price = self._get_mark_price(asset)
            
            # Log existing position details before executing a new trade
            current_position = self.positions[asset]
            current_size = current_position['size']
            
            if abs(current_size) > 1e-8:
                # Calculate position details for logging
                notional_value = abs(current_size * current_price)
                current_leverage = current_position.get('leverage', 0)
                # Margin used = notional value / leverage (if leverage is non-zero)
                margin_used = notional_value / current_leverage if current_leverage > 0 else notional_value
                
                # Calculate unrealized PnL
                entry_price = current_position['entry_price']
                if current_size > 0:  # Long position
                    unrealized_pnl = current_size * (current_price - entry_price)
                else:  # Short position
                    unrealized_pnl = abs(current_size) * (entry_price - current_price)
                
                # Calculate PnL percentage based on margin used
                pnl_pct = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0
                
                direction_str = "LONG" if current_size > 0 else "SHORT"
                
                # Format position details as a table
                logger.info(f"\n{'-'*60}")
                logger.info(f"| EXISTING POSITION DETAILS: {asset} {direction_str} {' '*(34-len(asset)-len(direction_str))}|")
                logger.info(f"|{'-'*58}|")
                logger.info(f"| Position Size     | {current_size:<11.6f} {' '*29}|")
                logger.info(f"| Entry Price       | ${entry_price:<10.2f} {' '*30}|")
                logger.info(f"| Current Price     | ${current_price:<10.2f} {' '*30}|")
                logger.info(f"| Notional Value    | ${notional_value:<10.2f} {' '*30}|")
                logger.info(f"| Leverage          | {current_leverage:<10.2f}x {' '*29}|")
                logger.info(f"| Margin Used       | ${margin_used:<10.2f} {' '*30}|")
                logger.info(f"| Unrealized PnL    | ${unrealized_pnl:<10.2f} ({pnl_pct:+.2f}%) {' '*17}|")
                
                # Add risk parameters if available
                if 'stop_loss' in current_position and current_position['stop_loss'] is not None:
                    logger.info(f"| Stop Loss         | ${current_position['stop_loss']:<10.2f} {' '*30}|")
                if 'take_profit' in current_position and current_position['take_profit'] is not None:
                    logger.info(f"| Take Profit       | ${current_position['take_profit']:<10.2f} {' '*30}|")
                if 'funding_accrued' in current_position:
                    logger.info(f"| Funding Accrued   | ${current_position['funding_accrued']:<10.2f} {' '*30}|")
                
                logger.info(f"{'-'*60}\n")
            else:
                logger.info(f"No existing position for {asset}")
            
            # Get current price is already cached above
            price = current_price
            if price <= 0:
                logger.warning(f"Invalid price for {asset}: {price}")
                return 0, 0, 0, 0, 0
                
            # Get current position
            current_size = self.positions[asset]['size']
            current_value = current_size * price
            
            # CRITICAL FIX: Determine direction separately from leverage
            # Target leverage sign indicates direction, but actual leverage is always positive
            direction = np.sign(signal) if abs(signal) > 1e-8 else 0
            abs_target_leverage = abs(target_leverage)  # Always positive
                
            # DIAGNOSTIC: Log the initial trade computation values
            if self.verbose:
                logger.info(f"Trade calculation for {asset}: price=${price:.2f}, current_size={current_size:.6f}, signal={signal:.4f}, direction={direction}")
                
            # ENHANCEMENT: Check trading cooldown for this asset
            steps_since_last_trade = self.current_step - self.last_trade_step[asset]
            if steps_since_last_trade < self.min_steps_between_trades and abs(current_size) > 1e-8:
                # Skip if we already have a position and haven't waited enough steps
                if self.verbose:
                    logger.info(f"Trading cooldown for {asset}: {steps_since_last_trade} steps since last trade (min: {self.min_steps_between_trades})")
                return current_size, self.positions[asset]['entry_price'], self.positions[asset]['leverage'], current_value, 0
                
            # ENHANCEMENT: Check if market volatility justifies trading
            volatility_check_passed = True
            if hasattr(self, 'market_conditions') and isinstance(self.market_conditions, dict) and 'volatility_regime' in self.market_conditions:
                asset_volatility = self.market_conditions['volatility_regime'].get(asset, 0.5)
                # If volatility is very low, require a stronger signal
                if asset_volatility < 0.3 and abs(signal) < self.signal_threshold * 1.5:
                    volatility_check_passed = False
                    if self.verbose:
                        logger.info(f"Low volatility for {asset}: {asset_volatility:.2f}, requires stronger signal")
                
            if not volatility_check_passed:
                return current_size, self.positions[asset]['entry_price'], self.positions[asset]['leverage'], current_value, 0
        
            # Initialize target size
            target_size = 0
            
            # Extract market data for enhanced risk management
            market_data = None
            if hasattr(self, 'df') and len(self.df) > 0:
                # Get last N rows for risk calculations
                start_idx = max(0, self.current_step - 100)  # Use last 100 rows
                end_idx = self.current_step + 1
                market_data = self.df.iloc[start_idx:end_idx]
        
            # Calculate target position value based on portfolio_value, leverage, and signal
            if abs(signal) >= self.signal_threshold:  # Use the configured threshold
                # ENHANCED: Calculate target position value with target leverage 
                # Scale by signal strength for more precise position sizing
                signal_strength = min(1.0, abs(signal) / self.signal_threshold)  # Normalized signal strength
                
                # CRITICAL FIX: Use absolute leverage value with separate direction
                effective_leverage = abs_target_leverage  # Always positive
            
                # Log the leverage being used
                if self.verbose:
                    logger.info(f"Trade signal: {signal:.2f}, target leverage: {abs_target_leverage:.2f}x, "
                            f"effective leverage: {effective_leverage:.2f}x for {asset}")
            
                # CRITICAL FIX: Properly calculate target position value
                # Only enforce minimum leverage of 1.0 if specifically requested
                if effective_leverage > 0 and effective_leverage < 1.0 and self.enforce_min_leverage:
                    effective_leverage = 1.0
                    logger.warning(f"Enforcing minimum leverage of 1.0x for {asset}")
            
                # Calculate target position value based on direction, leverage, and notional amount
                target_value = direction * effective_leverage * portfolio_value * signal_strength
                
                # DIAGNOSTIC: Log target position value
                if self.verbose:
                    logger.info(f"Target position value for {asset}: ${target_value:.2f}")
                
                # Use risk engine to check position sizing limits with enhanced risk controls
                max_size = self.risk_engine.get_max_position_size(
                    asset, 
                    portfolio_value, 
                    price,
                    market_data,  # Pass market data for impact modeling
                    self.positions,  # Pass current positions for correlation analysis
                    self.verbose  # Pass verbose flag
                )
            
                # Convert target value to size
                raw_target_size = target_value / price if price > 0 else 0
                
                # Apply position size limits
                target_size = np.sign(raw_target_size) * min(abs(raw_target_size), max_size)
                
                # DIAGNOSTIC: Log target size calculation
                if self.verbose:
                    logger.info(f"Target size calculation for {asset}: raw={raw_target_size:.6f}, max_size={max_size:.6f}, final={target_size:.6f}")
                
                # Check if this trade would cause margin to exceed portfolio value
                # First, calculate what total margin would be after this trade
                total_margin_after_trade = 0
                
                # PERFORMANCE OPTIMIZATION: Cache all mark prices for positions
                all_mark_prices = {asset: price}  # We already know the current asset price
                for pos_asset in self.positions.keys():
                    if pos_asset != asset:  # Skip the current asset, we already have its price
                        all_mark_prices[pos_asset] = self._get_mark_price(pos_asset)
                
                # Create a copy of positions to simulate the trade
                simulated_positions = copy.deepcopy(self.positions)
                simulated_positions[asset]['size'] = target_size  # Set to target size, not just size_diff
                simulated_positions[asset]['leverage'] = effective_leverage
                
                # Calculate total margin and notional in a single loop with cached prices
                total_notional_after_trade = 0
                
                # Calculate total margin that would be used after this trade
                for pos_asset, position in simulated_positions.items():
                    position_size = position['size']
                    
                    # Skip if position would be effectively closed
                    if abs(position_size) < 1e-8:
                        continue
                        
                    # Get leverage for this position
                    pos_leverage = min(position.get('leverage', self.max_leverage), self.max_leverage)
                    pos_leverage = max(1.0, pos_leverage)  # Ensure leverage is at least 1.0
                    
                    # Calculate position metrics using cached price
                    pos_price = all_mark_prices[pos_asset]
                    notional_value = abs(position_size * pos_price)
                    position_margin = notional_value / pos_leverage if pos_leverage > 0 else notional_value
                    
                    # Add to totals
                    total_margin_after_trade += position_margin
                    total_notional_after_trade += notional_value
                
                # Check if margin would exceed portfolio value
                if total_margin_after_trade > portfolio_value:
                    logger.warning(f"Trade rejected for {asset}: Margin after trade (${total_margin_after_trade:.2f}) would exceed portfolio value (${portfolio_value:.2f})")
                    return current_size, self.positions[asset]['entry_price'], self.positions[asset]['leverage'], current_value, 0
                
                # Check if total leverage would exceed account-wide leverage limit
                total_leverage_after_trade = total_notional_after_trade / portfolio_value if portfolio_value > 0 else 0
                
                if total_leverage_after_trade > self.max_leverage:
                    logger.warning(f"Trade rejected for {asset}: Total leverage after trade ({total_leverage_after_trade:.2f}x) would exceed max account leverage ({self.max_leverage:.2f}x)")
                    return current_size, self.positions[asset]['entry_price'], self.positions[asset]['leverage'], current_value, 0
            
                # Minimum position check - don't take tiny positions
                min_position_value = self.risk_engine.risk_limits.min_trade_size_usd
                if abs(target_size * price) < min_position_value:
                    # Position too small to be worth taking
                    if self.verbose:
                        logger.info(f"Position too small for {asset}: ${abs(target_size * price):.2f} < ${min_position_value:.2f} (min)")
                    target_size = 0
            else:
                # Signal too weak to take a position
                if self.verbose:
                    logger.info(f"Signal too weak for {asset}: {signal:.4f} < 0.01 (min threshold)")
                target_size = 0
            
            # Calculate size change
            size_diff = target_size - current_size
            
            # Only execute if the size change is significant
            # DIAGNOSTIC: Use an extremely small minimum trade size threshold to ensure trades are executed
            min_trade_size = 1e-12  # Was 1e-8, reduced to basically zero for crypto trading
            
            if abs(size_diff) <= min_trade_size:
                # No significant change
                if self.verbose:
                    logger.info(f"No significant size change for {asset}: {size_diff:.8f} (min: {min_trade_size})")
                return current_size, self.positions[asset]['entry_price'], self.positions[asset]['leverage'], current_value, 0
                
            # ENHANCEMENT: Use market impact model for more accurate execution price
            if market_data is not None:
                # Calculate market impact using enhanced model
                impact_data = self.risk_engine.calculate_market_impact(
                    asset, 
                    size_diff, 
                    price, 
                    market_data,
                    self.verbose  # Pass verbose flag
                )
                execution_price = impact_data['impact_price']
                
                # Log impact details
                if self.verbose:
                    logger.info(f"Market impact for {asset}: {impact_data['impact_bps']:.2f} bps " +
                              f"(${impact_data['impact_usd']:.2f}), using model: {impact_data['model_used']}")
            else:
                # Fall back to simple impact model if no market data
                execution_price = self._calculate_execution_price(asset, size_diff, price)
            
            # Calculate cost of the trade
            # Transaction costs include commission, price impact, and spread costs
            trade_value = abs(size_diff * execution_price)
            cost = -self._calculate_transaction_cost(trade_value)
            
            # If closing a position or flipping direction, realize PnL
            if current_size * target_size <= 0 and abs(current_size) > 1e-8:
                # Either closing or flipping direction
                # Calculate PnL from the closed position
                entry_price = self.positions[asset]['entry_price']
                closed_size = current_size if target_size == 0 else current_size  # Close entire position
                
                # Calculate PnL
                if closed_size > 0:
                    # Long position - profit when exit price > entry price
                    realized_pnl = closed_size * (execution_price - entry_price)
                else:
                    # Short position - profit when exit price < entry price
                    realized_pnl = closed_size * (execution_price - entry_price)
                    
                # Update balance with realized PnL
                self.balance += realized_pnl
                
                # Log the trade
                if self.verbose:
                    logger.info(f"Realized P&L: {realized_pnl:.2f} from closing {closed_size:.6f} {asset} at {execution_price:.2f}")
                    
                # Record the trade
                self.trades.append({
                    'timestamp': self.current_step,
                    'asset': asset,
                    'size': -closed_size,  # Negative means closing
                    'price': execution_price,
                    'cost': cost,
                    'realized_pnl': realized_pnl
                })
                
                # Reset position if fully closing
                if target_size == 0 or (current_size * target_size < 0):
                    # Completely closing or flipping direction
                    self.positions[asset]['size'] = 0
                    self.positions[asset]['entry_price'] = 0
                    self.positions[asset]['mark_price'] = price
                    self.positions[asset]['entry_time'] = 0
                    self.positions[asset]['leverage'] = 0
                    self.positions[asset]['unrealized_pnl'] = 0
                    
                    # Reset stop-loss and take-profit levels
                    self.positions[asset]['stop_loss'] = None
                    self.positions[asset]['take_profit'] = None
                    self.positions[asset]['trailing_stop'] = None
                    self.positions[asset]['stop_loss_pct'] = None
                    self.positions[asset]['take_profit_pct'] = None
            
            # Create new position or add to existing
            if abs(target_size) > 1e-8:
                # Opening or modifying a position
                if abs(current_size) < 1e-8:
                    # New position
                    self.positions[asset]['size'] = target_size
                    self.positions[asset]['entry_price'] = execution_price
                    self.positions[asset]['last_price'] = price
                    self.positions[asset]['mark_price'] = price
                    self.positions[asset]['entry_time'] = self.current_step
                    # CRITICAL FIX: Store absolute leverage with direction separately
                    self.positions[asset]['leverage'] = abs(effective_leverage)
                    self.positions[asset]['direction'] = direction
                    
                    # Get dynamic stop-loss and take-profit levels from risk engine
                    sl_tp_levels = None
                    # Get stop loss and take profit from risk engine if available
                    if self.risk_engine:
                        try:
                            # Calculate appropriate stop loss / take profit levels
                            sl_tp_levels = self.risk_engine.get_stop_loss_take_profit_levels(
                                asset, execution_price, target_size, signal, portfolio_value
                            )
                            
                            if sl_tp_levels:
                                if 'stop_loss' in sl_tp_levels:
                                    self.positions[asset]['stop_loss'] = sl_tp_levels['stop_loss']
                                    self.positions[asset]['stop_loss_pct'] = sl_tp_levels.get('stop_loss_pct', None)
                                    
                                if 'take_profit' in sl_tp_levels:
                                    self.positions[asset]['take_profit'] = sl_tp_levels['take_profit']
                                    self.positions[asset]['take_profit_pct'] = sl_tp_levels.get('take_profit_pct', None)
                                    
                                if 'trailing_stop' in sl_tp_levels:
                                    self.positions[asset]['trailing_stop'] = sl_tp_levels['trailing_stop']
                        except Exception as e:
                            logger.error(f"Error setting stop-loss/take-profit: {e}")
                    
                    # Log the new position
                    if self.verbose:
                        sl_str = f", SL: {self.positions[asset]['stop_loss']:.2f}" if self.positions[asset]['stop_loss'] else ""
                        tp_str = f", TP: {self.positions[asset]['take_profit']:.2f}" if self.positions[asset]['take_profit'] else ""
                        trail_str = f", Trail: {self.positions[asset]['trailing_stop']:.2f}%" if self.positions[asset]['trailing_stop'] else ""
                        direction_str = "long" if direction > 0 else "short"
                        logger.info(f"New {direction_str} position: {target_size} {asset} at {execution_price:.2f}, leverage: {abs(effective_leverage):.2f}x{sl_str}{tp_str}{trail_str}")
                else:
                    # Update existing position with new size
                    old_size = self.positions[asset]['size']
                    old_entry = self.positions[asset]['entry_price']
                    
                    # If closing position completely, record realized pnl
                    if abs(target_size) < 1e-8:
                        # Completely closing the position
                        close_pnl = 0.0
                        if old_size > 0:
                            # Long position closing - gain when close price > entry
                            close_pnl = old_size * (execution_price - old_entry)
                        else:
                            # Short position closing - gain when close price < entry
                            close_pnl = old_size * (old_entry - execution_price)
                            
                        # Update balance with realized pnl
                        self.balance += close_pnl
                        
                        # Log
                        if self.verbose:
                            logger.info(f"Closed position {asset}: {old_size:.6f} -> 0 at {execution_price:.2f}, pnl: ${close_pnl:.2f}")
                            
                        # Reset position
                        self.positions[asset]['size'] = 0
                        self.positions[asset]['entry_price'] = 0
                        self.positions[asset]['leverage'] = 0
                        self.positions[asset]['direction'] = 0
                        self.positions[asset]['entry_time'] = 0
                        
                        # Reset stop-loss and take-profit levels
                        self.positions[asset]['stop_loss'] = None
                        self.positions[asset]['take_profit'] = None
                        self.positions[asset]['trailing_stop'] = None
                        self.positions[asset]['stop_loss_pct'] = None
                        self.positions[asset]['take_profit_pct'] = None
                        
                        # Set others for the return values
                        new_size = 0
                        new_price = 0
                        actual_leverage = 0
                        position_value = 0
                    else:
                        # Increment or modify position
                        old_size = self.positions[asset]['size']
                        old_entry = self.positions[asset]['entry_price']
                        new_size = target_size
                        
                        # Calculate new average entry price
                        if old_size * new_size > 0:
                            # Same direction, calculate weighted average entry price
                            total_value = abs(old_size * old_entry) + abs(size_diff * execution_price)
                            total_size = abs(old_size) + abs(size_diff)
                            new_price = total_value / total_size if total_size > 0 else execution_price
                        else:
                            # Different direction, use new execution price
                            new_price = execution_price
                        
                        # Update position
                        self.positions[asset]['size'] = new_size
                        self.positions[asset]['entry_price'] = new_price
                        self.positions[asset]['last_price'] = price
                        self.positions[asset]['mark_price'] = price
                        
                        # Calculate actual leverage based on position value and portfolio
                        position_value = abs(new_size * execution_price)
                        
                        # CRITICAL FIX: Store absolute leverage with direction separately
                        self.positions[asset]['leverage'] = abs(effective_leverage)
                        self.positions[asset]['direction'] = np.sign(new_size)
                        
                        # Log the position adjustment
                        if self.verbose:
                            action_str = "Increased" if abs(new_size) > abs(old_size) else "Reduced"
                            direction_str = "long" if new_size > 0 else "short"
                            logger.info(f"{action_str} {direction_str} position: {new_size:.6f} {asset} (was: {old_size:.6f}), new entry: {new_price:.2f}, leverage: {abs(effective_leverage):.2f}x")
                            
            # Calculate position value and actual leverage after all adjustments
            position_value = abs(self.positions[asset]['size'] * price)
            actual_leverage = self.positions[asset]['leverage']  # Now always positive
        
            # Update balance with cost
            self.balance += cost
            
            # Record the trade
            self.trades.append({
                'timestamp': self.current_step,
                'asset': asset,
                'size': size_diff,
                'price': execution_price,
                'cost': cost,
                'realized_pnl': 0  # No realized PnL for this trade component
            })
            
            # Calculate new position values after trade
            new_size = self.positions[asset]['size']
            new_price = self.positions[asset]['entry_price']
            position_value = abs(new_size * price)
            
            # Calculate actual leverage using total position size
            if abs(new_size) > 1e-8:  # Only calculate leverage for non-zero positions
                # FIXED: Don't force minimum of 1.0x here, this overrides the target leverage
                # Use the target_leverage value from the position data, not recalculate
                actual_leverage = min(self.positions[asset]['leverage'], self.max_leverage)
            else:
                actual_leverage = 0.0
                
            # Update the last trade step for this asset
            self.last_trade_step[asset] = self.current_step
            
            # Log that we're executing a trade
            if self.verbose:
                logger.info(f"Executing trade for {asset}: size change {size_diff:.6f} at ${execution_price:.2f}")
            
            return new_size, new_price, actual_leverage, position_value, cost
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            return 0, 0, 0, 0, 0

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
    
    def _calculate_execution_price(self, asset: str, order_size: float, mark_price: float) -> float:
        """
        Calculate the execution price for a trade, including slippage and price impact
        
        Args:
            asset: Asset symbol
            order_size: Size of the order (positive for buy, negative for sell)
            mark_price: Current mark price
            
        Returns:
            float: Execution price
        """
        # Skip if order size is too small
        if abs(order_size) < 1e-8:
            return mark_price
            
        # Calculate price impact
        price_impact = self._estimate_price_impact(asset, order_size)
        
        # Calculate spread cost
        spread_cost = self._get_spread_cost(asset)
        
        # Direction of the trade affects slippage (buys get higher prices, sells get lower)
        direction = 1 if order_size > 0 else -1
        
        # Apply spread cost based on direction
        spread_adjustment = direction * spread_cost * mark_price
        
        # Apply price impact based on direction and size
        impact_adjustment = direction * price_impact * mark_price
        
        # Calculate execution price with slippage and impact
        execution_price = mark_price + spread_adjustment + impact_adjustment
        
        # Ensure price is positive
        execution_price = max(0.01, execution_price)
        
        return execution_price
        
    def _calculate_transaction_cost(self, trade_value: float) -> float:
        """
        Calculate the total transaction cost for a trade
        
        Args:
            trade_value: Absolute value of the trade (in quote currency)
            
        Returns:
            float: Total transaction cost (negative value)
        """
        # Base commission cost
        commission_cost = trade_value * self.commission
        
        # Additional costs can be added here (exchange fees, etc.)
        additional_costs = 0.0
        
        # Calculate total transaction cost (negative value)
        total_cost = -(commission_cost + additional_costs)
        
        return total_cost

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
        # Calculate portfolio value first to use it later
        portfolio_value = self._calculate_portfolio_value()
        
        # Collect trades from this step
        current_trades = []
        if hasattr(self, 'position_changes_this_step') and self.position_changes_this_step > 0:
            # Extract trades from trades_history that occurred in the current step
            if hasattr(self, 'trades_history'):
                current_trades = [trade for trade in self.trades_history if trade.get('timestamp') == self.current_step]
                # If no trades found but we know position changes happened, create synthetic trades
                if not current_trades and self.position_changes_this_step > 0:
                    for asset, position in self.positions.items():
                        # Skip positions that haven't changed
                        if not hasattr(position, 'last_size_change') or position.get('last_size_change') != self.current_step:
                            continue
                        
                        # Create a synthetic trade record
                        current_trades.append({
                            'asset': asset,
                            'timestamp': self.current_step,
                            'size_change': position.get('size', 0),  # Current size is the change from zero
                            'price': position.get('entry_price', self._get_mark_price(asset)),
                            'type': 'synthetic',
                            'generated': 'backtest'
                        })
        
        info = {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'gross_leverage': risk_metrics.get('gross_leverage', 0),  # Add gross leverage (always positive)
            'net_leverage': risk_metrics.get('net_leverage', 0),      # Add net leverage (can be negative for shorts)
            'positions': {asset: {'size': pos['size'], 'entry_price': pos['entry_price']} 
                          for asset, pos in self.positions.items()},
            'total_trades': len(self.trades),
            'risk_metrics': risk_metrics,
            # Include trades from this step
            'trades': current_trades
        }
        
        # ENHANCED: Add position durations to info
        info['position_durations'] = self.position_duration.copy()
        
        # ENHANCED: Add uncertainty metrics to info if available
        if hasattr(self, 'uncertainty_metrics'):
            # Safely extract uncertainty scores
            uncertainty_dict = {}
            for asset, metrics in self.uncertainty_metrics.items():
                # Check if the key exists before accessing
                if 'uncertainty_score' in metrics:
                    uncertainty_dict[asset] = metrics['uncertainty_score']
                else:
                    # Use a default value if the key doesn't exist
                    uncertainty_dict[asset] = 0.5  # Neutral uncertainty
            
            info['uncertainty'] = uncertainty_dict
            
        # CRITICAL FIX: Add performance metrics if available
        if hasattr(self, 'performance_tracking'):
            if self.performance_tracking['sharpe_ratios'] and len(self.performance_tracking['sharpe_ratios']) > 0:
                info['sharpe_ratio'] = self.performance_tracking['sharpe_ratios'][-1]
            
            if self.performance_tracking['sortino_ratios'] and len(self.performance_tracking['sortino_ratios']) > 0:
                info['sortino_ratio'] = self.performance_tracking['sortino_ratios'][-1]
                
            if self.performance_tracking['calmar_ratios'] and len(self.performance_tracking['calmar_ratios']) > 0:
                info['calmar_ratio'] = self.performance_tracking['calmar_ratios'][-1]
                
        # Add drawdown if we have history
        if hasattr(self, 'portfolio_history') and len(self.portfolio_history) > 0:
            # Get peak portfolio value
            peak_value = max(entry['value'] for entry in self.portfolio_history)
            # Calculate drawdown
            drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            info['drawdown'] = drawdown
        
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
        """Close a position for a specific asset"""
        try:
            if abs(self.positions[asset]['size']) > 1e-8:
                # Get current price
                price = self._get_mark_price(asset)
                
                # Calculate PnL
                size = self.positions[asset]['size']
                entry_price = self.positions[asset]['entry_price']
                
                if size > 0:  # Long position
                    pnl = size * (price - entry_price)
                else:  # Short position
                    pnl = size * (entry_price - price)
                
                # Calculate costs
                cost = abs(size * price) * self.commission
                
                # Update balance
                self.balance += pnl - cost
                
                # Record trade in history
                self.trades.append({
                    'timestamp': self.current_step,
                    'asset': asset,
                    'size': -size,  # Closing position is opposite of current size
                    'price': price,
                    'cost': cost,
                    'realized_pnl': pnl,
                    'type': 'close',
                    'leverage': 0.0  # No leverage after closing
                })
                
                # Reset position
                self.positions[asset]['size'] = 0
                self.positions[asset]['entry_price'] = 0
                self.positions[asset]['leverage'] = 0.0
                
                # IMPROVEMENT: Reset stop-loss and take-profit
                self.positions[asset]['stop_loss'] = None
                self.positions[asset]['take_profit'] = None
                self.positions[asset]['trailing_stop'] = None
                self.positions[asset]['stop_loss_pct'] = None
                self.positions[asset]['take_profit_pct'] = None
                
                # Reset position duration tracking
                self.position_duration[asset] = 0
                
                if self.verbose:
                    logger.info(f"Closed position for {asset} @ {price:.2f}, PnL: {pnl:.2f}, Cost: {cost:.2f}")
                
                return pnl, cost
            return 0.0, 0.0
        
        except Exception as e:
            logger.error(f"Error closing position for {asset}: {str(e)}")
            return 0.0, 0.0
            
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

    def _check_and_execute_stops(self):
        """Check if any stop-loss or take-profit conditions are met and execute orders accordingly"""
        executed_stops = []
        total_pnl = 0.0
        
        for asset in self.assets:
            position = self.positions[asset]
            
            # Skip if no position or tiny position
            if abs(position['size']) < 1e-8:
                continue
                
            # Get current price
            current_price = self._get_mark_price(asset)
            entry_price = position['entry_price']
            position_size = position['size']
            is_long = position_size > 0
            
            # Update trailing stop if applicable
            if position['trailing_stop'] is not None:
                # For long positions: move stop up if price moves up
                if is_long:
                    # Calculate minimum price that would trigger trailing stop
                    trail_distance = entry_price * position['trailing_stop']
                    # If no stop_loss set yet or price moved up enough to move the stop
                    if position['stop_loss'] is None or current_price - trail_distance > position['stop_loss']:
                        position['stop_loss'] = current_price - trail_distance
                        logger.info(f"Updated trailing stop for {asset}: {position['stop_loss']:.2f}")
                # For short positions: move stop down if price moves down
                else:
                    trail_distance = entry_price * position['trailing_stop']
                    if position['stop_loss'] is None or current_price + trail_distance < position['stop_loss']:
                        position['stop_loss'] = current_price + trail_distance
                        logger.info(f"Updated trailing stop for {asset}: {position['stop_loss']:.2f}")
            
            # Stop Loss check
            stop_triggered = False
            if position['stop_loss'] is not None:
                if (is_long and current_price <= position['stop_loss']) or \
                   (not is_long and current_price >= position['stop_loss']):
                    logger.info(f"Stop loss triggered for {asset} at {current_price:.2f} (stop: {position['stop_loss']:.2f})")
                    stop_triggered = True
                    reason = "stop_loss"
            
            # Take Profit check
            if not stop_triggered and position['take_profit'] is not None:
                if (is_long and current_price >= position['take_profit']) or \
                   (not is_long and current_price <= position['take_profit']):
                    logger.info(f"Take profit triggered for {asset} at {current_price:.2f} (target: {position['take_profit']:.2f})")
                    stop_triggered = True
                    reason = "take_profit"
            
            # Execute stop if triggered
            if stop_triggered:
                # Calculate PnL
                if is_long:
                    pnl = position_size * (current_price - entry_price)
                else:
                    pnl = position_size * (entry_price - current_price)
                
                # Close the position
                cost = abs(position_size * current_price) * self.commission
                total_pnl += pnl - cost
                
                # Log the stop execution
                executed_stops.append({
                    'asset': asset,
                    'type': reason,
                    'price': current_price,
                    'size': position_size,
                    'pnl': pnl,
                    'cost': cost
                })
                
                # Actually close the position
                self._close_position(asset)
        
        return executed_stops, total_pnl

    def _get_target_leverage(self, signal: float) -> float:
        """
        Calculate target leverage based on signal strength
        
        Args:
            signal: Action signal in range [-1.0, 1.0]
            
        Returns:
            float: Target leverage for the position
        """
        try:
            # Get the magnitude of the signal (0 to 1)
            signal_strength = abs(signal)
            
            # Check risk engine max leverage
            max_leverage = self.risk_engine.risk_limits.max_leverage if hasattr(self, 'risk_engine') and hasattr(self.risk_engine, 'risk_limits') else self.max_leverage
            
            # CRITICAL FIX: Ensure max_leverage is never negative
            max_leverage = max(0.0, max_leverage)
            
            # For verbose logging
            if self.verbose:
                logger.info(f"Max leverage from risk engine: {max_leverage:.2f}x")
            
            # Base case - no signal means no leverage (close position)
            if signal_strength < self.signal_threshold:  # Increased from 0.1 to the configured threshold
                return 0.0
                
            # Calculate base leverage based on signal strength
            # We use a min leverage of 1x for small signals
            min_leverage = 1.0
            
            # Calculate base leverage using more aggressive scaling for crypto 
            # This gives more precision in the middle range of signals while
            # still providing sufficient leverage for small signals
            signal_strength = max(self.signal_threshold, min(1.0, signal_strength))  # Clamp to valid range
            
            # FIXED: Ensure leverage is properly calculated between min and max
            # Apply quadratic scaling for smoother leverage curve
            base_leverage = min_leverage + (max_leverage - min_leverage) * (signal_strength ** 1.5)
            
            # For verbose logging
            if self.verbose:
                logger.info(f"Base leverage calculation: min={min_leverage:.2f}x, signal_strength={signal_strength:.2f}, base={base_leverage:.2f}x")
            
            # Apply scaling factors based on market conditions
            # 1. Volatility scaling - reduce leverage in high volatility
            vol_scale = 1.0
            
            # 2. Market regime scaling - reduce in trending markets, increase in range-bound
            regime_scale = 1.0
            
            # Calculate final leverage
            final_leverage = base_leverage * vol_scale * regime_scale
            
            # CRITICAL FIX: Ensure minimum leverage is 1.0 for any non-zero position
            if signal_strength >= self.signal_threshold:
                final_leverage = max(1.0, min(final_leverage, max_leverage))
            
            # Apply direction (sign) of the signal
            # CRITICAL FIX: Do NOT use negative leverage - leverage should always be positive
            # Instead, use signal sign to determine direction
            effective_leverage = final_leverage  # Always positive
            direction = np.sign(signal)  # This already captures the direction
            
            # For verbose logging
            if self.verbose:
                logger.info(f"Leverage calculation for signal {signal:.2f}: base={base_leverage:.2f}, vol_scale={vol_scale:.2f}, regime_scale={regime_scale:.2f}, final={effective_leverage:.2f}x")
                if direction < 0:
                    logger.info(f"Short position: direction={direction}, leverage={effective_leverage:.2f}x")
            
            # CRITICAL FIX: Enforce minimum leverage of 1.0x for backtesting consistency
            if effective_leverage > 0 and effective_leverage < 1.0:
                effective_leverage = 1.0
                if self.verbose:
                    logger.info(f"Enforcing minimum leverage of 1.0x for {asset} (was {final_leverage:.2f}x)")
            
            return effective_leverage * direction
            
        except Exception as e:
            logger.error(f"Error calculating target leverage: {str(e)}")
            # Default to no leverage in case of error
            return 0.0

    def _handle_model_uncertainty(self, action_vector, uncertainty_vector=None):
        """
        Handle model uncertainty by adjusting actions based on uncertainty levels
        
        Args:
            action_vector: Original action vector from the model
            uncertainty_vector: Optional vector of uncertainty levels (0-1) for each asset
                               If None, will use uncertainty from feature extractor if available
                               
        Returns:
            np.ndarray: Adjusted action vector
        """
        # IMPORTANT: As requested, completely disable uncertainty handling
        logger.info("Uncertainty handling has been disabled as requested")
        return action_vector

    def _analyze_market_conditions(self):
        """Analyze market conditions and regime detection"""
        try:
            # Skip if we're not due for an update yet
            if self.current_step - self.last_market_analysis_step < self.calc_frequency:
                # FIXED: Return existing market conditions instead of None
                return self.market_conditions
                
            self.last_market_analysis_step = self.current_step
            
            # Detect market regimes for each asset
            regimes = {}
            for asset in self.assets:
                # Get asset price history
                if asset in self.price_history and len(self.price_history[asset]) > 20:
                    prices = self.price_history[asset]
                    
                    # Calculate returns
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Volatility
                    vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
                    
                    # Categorize regime
                    if vol > 0.03:  # High volatility
                        self.market_conditions['volatility_regime'][asset] = 0.8
                        regimes[asset] = 'volatile'
                    elif vol < 0.01:  # Low volatility
                        self.market_conditions['volatility_regime'][asset] = 0.2
                        regimes[asset] = 'range_bound'
                    else:  # Medium volatility
                        self.market_conditions['volatility_regime'][asset] = 0.5
                        regimes[asset] = 'normal'
            
            # Update overall market regime
            # FIXED: Handle the case where market_regime is a string or dictionary
            if isinstance(self.market_conditions['market_regime'], dict):
                # Update the dictionary version
                for asset, regime in regimes.items():
                    self.market_conditions['market_regime'][asset] = regime
                # Calculate average regime value
                regime_values = [0.8 if r == 'volatile' else (0.2 if r == 'range_bound' else 0.5) 
                               for r in regimes.values()]
                self.market_regime = np.mean(regime_values) if regime_values else 0.5
                
                # FIXED: Set the overall_market_state based on average regime value
                if self.market_regime > 0.7:
                    self.market_conditions['overall_market_state'] = 'volatile'
                elif self.market_regime < 0.3:
                    self.market_conditions['overall_market_state'] = 'range_bound'
                else:
                    self.market_conditions['overall_market_state'] = 'normal'
            else:
                # Use the most common regime as the overall market regime
                if regimes:
                    regime_counts = {}
                    for regime in regimes.values():
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1
                    
                    most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
                    self.market_conditions['market_regime'] = most_common_regime
                    
                    # FIXED: Also set the overall_market_state to the most common regime
                    self.market_conditions['overall_market_state'] = most_common_regime
                    
                    # Set numerical market regime value
                    regime_value_map = {'volatile': 0.8, 'range_bound': 0.2, 'normal': 0.5, 'trending': 0.7, 'crisis': 0.9}
                    self.market_regime = regime_value_map.get(most_common_regime, 0.5)
            
            return self.market_conditions
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            self.market_conditions['market_regime'] = 'normal'
            self.market_regime = 0.5
            # FIXED: Set a safe value for overall_market_state
            self.market_conditions['overall_market_state'] = 'normal'
            return self.market_conditions

    def _update_adaptive_parameters(self):
        """
        Update environment parameters based on market conditions
        
        This adjusts:
        1. Transaction costs (higher in volatile markets)
        2. Slippage factors (higher in volatile/illiquid markets)
        3. Risk limits (tighter in volatile markets)
        4. Exploration parameters (higher in range-bound markets)
        """
        try:
            # Only update if we have market conditions data
            if not hasattr(self, 'market_conditions') or not isinstance(self.market_conditions, dict):
                return
                
            # Get key market condition metrics
            overall_state = self.market_conditions.get('overall_market_state', 'normal')
            volatility_regimes = self.market_conditions.get('volatility_regime', {})
            liquidity_conditions = self.market_conditions.get('liquidity_conditions', {})
            
            # Get average metrics across assets
            # FIXED: Ensure these are dictionaries before trying to call values()
            avg_volatility = np.mean(list(volatility_regimes.values())) if isinstance(volatility_regimes, dict) and volatility_regimes else 0.5
            avg_liquidity = np.mean(list(liquidity_conditions.values())) if isinstance(liquidity_conditions, dict) and liquidity_conditions else 0.5
            
            # Store base parameters if not stored yet (only once)
            if not hasattr(self, 'base_parameters'):
                self.base_parameters = {
                    'commission': self.commission,
                    'slippage': getattr(self, 'slippage', 0.0002),  # Default slippage of 0.02%
                    'uncertainty_scaling_factor': getattr(self, 'uncertainty_scaling_factor', 1.0),
                    'risk_limits': {
                        'account_max_leverage': self.risk_engine.risk_limits.account_max_leverage,
                        'position_max_leverage': self.risk_engine.risk_limits.position_max_leverage,
                        'max_drawdown_pct': self.risk_engine.risk_limits.max_drawdown_pct,
                        'position_concentration': self.risk_engine.risk_limits.position_concentration
                    }
                }
            
            # 1. Adjust transaction costs based on volatility and liquidity
            # Higher volatility and lower liquidity = higher costs
            volatility_factor = 1.0 + (avg_volatility * 0.5)  # 1.0 to 1.5x
            liquidity_factor = 1.0 + ((1.0 - avg_liquidity) * 0.5)  # 1.0 to 1.5x
            
            # Combined adjustment for costs
            cost_adjustment = volatility_factor * liquidity_factor
            
            # Update commission rate
            self.commission = self.base_parameters['commission'] * cost_adjustment
            
            # 2. Adjust slippage based on similar factors, but with higher weight
            # Slippage is more affected by market conditions than fixed fees
            slippage_adjustment = volatility_factor * liquidity_factor * 1.5  # Up to 2.25x
            if hasattr(self, 'slippage'):
                self.slippage = self.base_parameters['slippage'] * slippage_adjustment
            else:
                self.slippage = 0.0002 * slippage_adjustment  # Default if not already set
            
            # 3. Adjust risk limits based on market state
            risk_scale = 1.0
            
            if overall_state == 'crisis':
                risk_scale = 0.5  # Reduce risk limits by 50% in crisis
                logger.warning(f"Crisis market state detected - reducing risk limits by 50%")
            elif overall_state == 'volatile':
                risk_scale = 0.7  # Reduce by 30% in volatile markets
                logger.info(f"Volatile market state detected - reducing risk limits by 30%")
            elif overall_state == 'trending':
                risk_scale = 0.9  # Reduce by 10% in trending markets
            
            # Update risk limits if the scale changed
            if risk_scale < 1.0:
                self.risk_engine.risk_limits.account_max_leverage = self.base_parameters['risk_limits']['account_max_leverage'] * risk_scale
                self.risk_engine.risk_limits.position_max_leverage = self.base_parameters['risk_limits']['position_max_leverage'] * risk_scale
                self.risk_engine.risk_limits.position_concentration = self.base_parameters['risk_limits']['position_concentration'] * risk_scale
            else:
                # Reset to base values if not in a special state
                self.risk_engine.risk_limits.account_max_leverage = self.base_parameters['risk_limits']['account_max_leverage']
                self.risk_engine.risk_limits.position_max_leverage = self.base_parameters['risk_limits']['position_max_leverage']
                self.risk_engine.risk_limits.position_concentration = self.base_parameters['risk_limits']['position_concentration']
            
            # 4. Adjust uncertainty scaling based on market state
            if hasattr(self, 'uncertainty_scaling_factor'):
                if overall_state == 'range_bound':
                    # Encourage more exploration in range-bound markets
                    self.uncertainty_scaling_factor = self.base_parameters['uncertainty_scaling_factor'] * 0.8
                elif overall_state == 'trending':
                    # Less exploration in trending markets
                    self.uncertainty_scaling_factor = self.base_parameters['uncertainty_scaling_factor'] * 1.2
                elif overall_state == 'crisis':
                    # Much less exploration in crisis
                    self.uncertainty_scaling_factor = self.base_parameters['uncertainty_scaling_factor'] * 1.5
                else:
                    # Reset to base value
                    self.uncertainty_scaling_factor = self.base_parameters['uncertainty_scaling_factor']
                
            # Log the adaptive parameter updates if they've changed significantly
            if abs(cost_adjustment - 1.0) > 0.1 or abs(slippage_adjustment - 1.0) > 0.2 or abs(risk_scale - 1.0) > 0.1:
                logger.info(
                    f"Adaptive parameters updated for {overall_state} market: "
                    f"Cost adj: {cost_adjustment:.2f}x, Slippage adj: {slippage_adjustment:.2f}x, "
                    f"Risk scale: {risk_scale:.2f}x"
                )
                
        except Exception as e:
            logger.error(f"Error updating adaptive parameters: {e}")
            traceback.print_exc()

    def _track_performance_metrics(self):
        """Track and update performance metrics"""
        # CRITICAL CHANGE: Always update portfolio history and calculate performance metrics 
        # regardless of training mode
        
        try:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Add to history
            self.portfolio_history.append({
                'step': self.current_step,
                'value': portfolio_value,
                'balance': self.balance,
                'unrealized_pnl': portfolio_value - self.balance
            })
            
            # CRITICAL FIX: Ensure we're consistently adding values to performance tracking
            self.performance_tracking['portfolio_values'].append(portfolio_value)
            
            # Calculate return since last step
            if len(self.performance_tracking['portfolio_values']) > 1:
                prev_value = self.performance_tracking['portfolio_values'][-2]
                if prev_value > 0:
                    step_return = (portfolio_value - prev_value) / prev_value
                else:
                    step_return = 0.0
            else:
                step_return = 0.0
                
            # Track return
            self.performance_tracking['returns'].append(step_return)
            
            # Calculate and track drawdown
            peak_value = max(self.performance_tracking['portfolio_values'])
            if peak_value > 0:
                drawdown = (peak_value - portfolio_value) / peak_value
            else:
                drawdown = 0.0
                
            self.performance_tracking['drawdowns'].append(drawdown)
            
            # Calculate Sharpe if we have enough returns
            min_returns_for_calcs = 10
            if len(self.performance_tracking['returns']) >= min_returns_for_calcs:
                returns_array = np.array(self.performance_tracking['returns'])
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                # Sharpe ratio calculation (with error handling)
                if std_return > 0:
                    sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                    self.performance_tracking['sharpe_ratios'].append(sharpe)
                else:
                    # If std is zero, can't calculate Sharpe
                    if len(self.performance_tracking['sharpe_ratios']) > 0:
                        # Use previous value if available
                        self.performance_tracking['sharpe_ratios'].append(
                            self.performance_tracking['sharpe_ratios'][-1]
                        )
                    else:
                        # Default to zero if no previous value
                        self.performance_tracking['sharpe_ratios'].append(0)
                
                # Sortino ratio calculation (with error handling)
                neg_returns = returns_array[returns_array < 0]
                if len(neg_returns) > 0:
                    downside_deviation = np.std(neg_returns)
                    if downside_deviation > 0:
                        sortino = mean_return / downside_deviation * np.sqrt(252)
                        self.performance_tracking['sortino_ratios'].append(sortino)
                    else:
                        # Handle zero downside deviation
                        if len(self.performance_tracking['sortino_ratios']) > 0:
                            self.performance_tracking['sortino_ratios'].append(
                                self.performance_tracking['sortino_ratios'][-1]
                            )
                        else:
                            self.performance_tracking['sortino_ratios'].append(0)
                else:
                    # No negative returns - very good!
                    # Use a high value or previous value
                    if len(self.performance_tracking['sortino_ratios']) > 0:
                        self.performance_tracking['sortino_ratios'].append(
                            max(3.0, self.performance_tracking['sortino_ratios'][-1])
                        )
                    else:
                        self.performance_tracking['sortino_ratios'].append(3.0)  # Good sortino if no losses
                
                # Calmar ratio calculation (with error handling)
                if len(self.performance_tracking['drawdowns']) > 0:
                    max_dd = max(self.performance_tracking['drawdowns'])
                    if max_dd > 0:
                        # Get total return from beginning
                        total_return = (portfolio_value / self.initial_balance) - 1
                        calmar = total_return / max_dd
                        self.performance_tracking['calmar_ratios'].append(calmar)
                    else:
                        # Handle zero max drawdown (unusual but possible)
                        if len(self.performance_tracking['calmar_ratios']) > 0:
                            self.performance_tracking['calmar_ratios'].append(
                                self.performance_tracking['calmar_ratios'][-1]
                            )
                        else:
                            self.performance_tracking['calmar_ratios'].append(1.0)  # Default value
                else:
                    # No drawdown history yet
                    self.performance_tracking['calmar_ratios'].append(1.0)  # Default value
            
            # Calculate win rate with trades
            if hasattr(self, 'trades') and len(self.trades) > 0:
                # Get only closed trades
                closed_trades = [t for t in self.trades if 'pnl' in t or 'realized_pnl' in t]
                if len(closed_trades) > 0:
                    # Count winning trades (with different key names for compatibility)
                    winning_trades = sum(1 for t in closed_trades if 
                                        ('pnl' in t and t['pnl'] > 0) or 
                                        ('realized_pnl' in t and t['realized_pnl'] > 0))
                    win_rate = winning_trades / len(closed_trades)
                    self.performance_tracking['win_rate'] = win_rate
            
            # Calculate current leverage
            total_exposure = sum(abs(pos['size'] * self._get_mark_price(asset)) 
                               for asset, pos in self.positions.items() 
                               if abs(pos['size']) > 1e-8)
            
            current_leverage = min(total_exposure / max(portfolio_value, self.initial_balance * 0.01), 
                                self.max_leverage)
            
            self.historical_leverage.append(current_leverage)
            self.performance_tracking['leverage'].append(current_leverage)
            
            # CRITICAL FIX: Track performance by market regime
            if hasattr(self, 'market_conditions') and isinstance(self.market_conditions, dict) and 'overall_market_state' in self.market_conditions:
                regime = self.market_conditions['overall_market_state']
                if regime in self.performance_tracking['regime_performance']:
                    if len(self.performance_tracking['returns']) > 0:
                        latest_return = self.performance_tracking['returns'][-1]
                        self.performance_tracking['regime_performance'][regime]['returns'].append(latest_return)
            
            # Track the last time we calculated performance metrics
            self.last_perf_metrics_step = self.current_step
            
            # Log occasional performance updates
            if self.current_step % 50 == 0:
                metrics_summary = self.get_performance_summary()
                logger.info(f"Performance at step {self.current_step}: "
                           f"Return: {metrics_summary.get('total_return', 0):.2%}, "
                           f"Sharpe: {metrics_summary.get('sharpe_ratio', 0):.2f}, "
                           f"Max DD: {metrics_summary.get('max_drawdown', 0):.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in tracking performance metrics: {str(e)}")
            traceback.print_exc()
            return False

    def _calculate_current_leverage(self):
        """Calculate current leverage based on portfolio value and risk limits"""
        try:
            # Calculate total exposure
            total_exposure = 0
            for asset, position in self.positions.items():
                mark_price = self._get_mark_price(asset)
                position_value = abs(position['size'] * mark_price)
                total_exposure += position_value
            
            # CRITICAL FIX: Ensure we don't divide by zero or tiny values
            # Use a minimum balance value to avoid division errors
            safe_balance = max(self.balance, self.initial_balance * 0.01)
            
            # Calculate leverage
            leverage = total_exposure / safe_balance
            
            # CRITICAL FIX: Leverage can never be negative
            leverage = max(0.0, leverage)
            
            # Ensure leverage is within risk limits
            if hasattr(self, 'risk_engine') and hasattr(self.risk_engine, 'risk_limits'):
                max_leverage = self.risk_engine.risk_limits.max_leverage
                # Only log warning if we're actually exceeding non-zero max leverage
                if leverage > max_leverage and max_leverage > 0:
                    logger.warning(f"Current leverage {leverage:.2f}x exceeds max leverage {max_leverage:.2f}x")
                    leverage = max_leverage
            
            return leverage
            
        except Exception as e:
            logger.error(f"Error calculating current leverage: {str(e)}")
            return 0.0
            
    def get_performance_summary(self) -> Dict:
        """
        Generate a comprehensive performance summary
        
        Returns:
            Dict: Performance metrics summary
        """
        if not hasattr(self, 'performance_tracking') or not isinstance(self.performance_tracking, dict):
            return {'error': 'No performance tracking data available'}
            
        try:
            # Get the current portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Calculate total return
            total_return = (portfolio_value / self.initial_balance) - 1.0
            
            # Calculate annualized return (assuming 252 trading days per year)
            # and accounting for the actual number of steps we've taken
            trading_days = max(1, self.current_step - self.window_size)  # Avoid division by zero
            annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
                
            # CRITICAL FIX: Get proper metrics from tracking with safety checks
                
            # Get max drawdown
            drawdowns = self.performance_tracking.get('drawdowns', [])
            max_drawdown = max(drawdowns) if drawdowns else 0
            
            # Get most recent risk-adjusted metrics with proper error handling
            sharpe_ratios = self.performance_tracking.get('sharpe_ratios', [])
            if sharpe_ratios:
                sharpe = sharpe_ratios[-1]
                # Average Sharpe - using only recent data for more relevance
                recent_window = min(len(sharpe_ratios), 20)
                avg_sharpe = np.mean(sharpe_ratios[-recent_window:])
            else:
                sharpe = 0
                avg_sharpe = 0
                
            sortino_ratios = self.performance_tracking.get('sortino_ratios', [])
            if sortino_ratios:
                sortino = sortino_ratios[-1]
                # Average Sortino - using only recent data for more relevance
                recent_window = min(len(sortino_ratios), 20)
                avg_sortino = np.mean(sortino_ratios[-recent_window:])
            else:
                sortino = 0
                avg_sortino = 0
                
            calmar_ratios = self.performance_tracking.get('calmar_ratios', [])
            if calmar_ratios:
                calmar = calmar_ratios[-1]
                # Average Calmar - using only recent data for more relevance
                recent_window = min(len(calmar_ratios), 20)
                avg_calmar = np.mean(calmar_ratios[-recent_window:])
            else:
                calmar = 0
                avg_calmar = 0
            
            # Calculate trades metrics if available
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_profit_per_trade = 0
            avg_loss_per_trade = 0
            
            if 'trades' in self.performance_tracking:
                trades = self.performance_tracking['trades']
                if isinstance(trades, dict):
                    profitable_trades = trades.get('profitable', 0)
                    unprofitable_trades = trades.get('unprofitable', 0)
                    total_trades = profitable_trades + unprofitable_trades
                    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                    
                    total_profit = trades.get('total_profit', 0)
                    total_loss = abs(trades.get('total_loss', 0)) if trades.get('total_loss', 0) < 0 else 1
                    profit_factor = total_profit / total_loss if total_loss > 0 else 0
                    
                    avg_profit_per_trade = trades.get('avg_profit_per_trade', 0)
                    avg_loss_per_trade = trades.get('avg_loss_per_trade', 0)
            
            # CRITICAL FIX: Calculate average returns and volatility with safety checks
            returns = self.performance_tracking.get('returns', [])
            avg_return = np.mean(returns) if returns else 0
            volatility = np.std(returns) * np.sqrt(252) if returns else 0
            
            # Log detailed summary occasionally
            if self.current_step % 50 == 0:
                logger.info(f"Performance Summary at step {self.current_step}:")
                logger.info(f"  Total Return: {total_return:.2%}")
                logger.info(f"  Annualized Return: {annualized_return:.2%}")
                logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
                logger.info(f"  Sharpe Ratio: {sharpe:.2f}, Avg: {avg_sharpe:.2f}")
                logger.info(f"  Sortino Ratio: {sortino:.2f}, Avg: {avg_sortino:.2f}")
                logger.info(f"  Calmar Ratio: {calmar:.2f}, Avg: {avg_calmar:.2f}")
                if total_trades > 0:
                    logger.info(f"  Total Trades: {total_trades}, Win Rate: {win_rate:.2%}")
            
            # Build the summary dictionary
            summary = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'average_return': avg_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_sortino_ratio': avg_sortino,
                'avg_calmar_ratio': avg_calmar,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_loss_per_trade': avg_loss_per_trade,
            }
            
            # Add regime performance if data exists
            if 'regime_performance' in self.performance_tracking and isinstance(self.performance_tracking['regime_performance'], dict):
                regime_summary = {}
                for regime, data in self.performance_tracking['regime_performance'].items():
                    if isinstance(data, dict) and data.get('returns', []):
                        returns = data.get('returns', [])
                        if returns:  # Only calculate if returns list is not empty
                            regime_summary[regime] = {
                                'avg_return': np.mean(returns),
                                'win_rate': data.get('win_rate', 0),
                                'num_trades': len(data.get('trades', []))
                            }
                if regime_summary:
                    summary['regime_performance'] = regime_summary
                
            # Add risk management data if available
            if 'risk_management' in self.performance_tracking and isinstance(self.performance_tracking['risk_management'], dict):
                rm_data = self.performance_tracking['risk_management']
                risk_management = {
                    'stop_loss_executions': rm_data.get('stop_loss_executions', 0),
                    'take_profit_executions': rm_data.get('take_profit_executions', 0),
                    'stop_loss_pnl': rm_data.get('stop_loss_pnl', 0),
                    'take_profit_pnl': rm_data.get('take_profit_pnl', 0),
                }
                
                # Calculate effectiveness if data exists
                if 'avg_stop_loss_pnl' in rm_data and avg_loss_per_trade != 0:
                    risk_management['stop_loss_effectiveness'] = abs(rm_data['avg_stop_loss_pnl'] / avg_loss_per_trade)
                    
                summary['risk_management'] = risk_management
            
            # CRITICAL FIX: Include current leverage
            current_leverage = self._calculate_current_leverage()
            average_leverage = np.mean(self.performance_tracking['leverage']) if self.performance_tracking['leverage'] else 0
            summary['current_leverage'] = current_leverage
            summary['average_leverage'] = average_leverage
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def visualize_performance(self, save_path=None, show_plots=True):
        """
        Generate comprehensive performance visualizations
        
        Args:
            save_path: Path to save the plots (None = don't save)
            show_plots: Whether to display the plots interactively
            
        Returns:
            dict: Dictionary of created figures
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.ticker import FuncFormatter
            
            if not hasattr(self, 'performance_tracking'):
                logger.error("No performance data available for visualization")
                return {}
                
            # Create dictionary to hold all figures
            figures = {}
            
            # Set style
            plt.style.use('ggplot')
            
            # Create figure for portfolio performance
            fig1 = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(3, 2)
            
            # Plot 1: Portfolio Value
            ax1 = fig1.add_subplot(gs[0, :])
            
            # Extract data
            steps = [x[0] for x in self.performance_tracking['portfolio_values']]
            values = [x[1] for x in self.performance_tracking['portfolio_values']]
            
            # Plot portfolio value
            ax1.plot(steps, values, 'b-', linewidth=2, label='Portfolio Value')
            
            # Add initial balance reference line
            ax1.axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
            
            # Format y-axis as currency
            def currency_formatter(x, pos):
                return f'${x:,.0f}'
                
            ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
            
            # Add labels and title
            ax1.set_xlabel('Trading Steps')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.set_title('Portfolio Value over Time')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Drawdowns
            ax2 = fig1.add_subplot(gs[1, 0])
            
            # Extract data
            if self.performance_tracking['drawdowns']:
                dd_steps = [x[0] for x in self.performance_tracking['drawdowns']]
                drawdowns = [x[1] * 100 for x in self.performance_tracking['drawdowns']]  # Convert to percentage
                
                # Plot drawdowns
                ax2.fill_between(dd_steps, 0, drawdowns, color='r', alpha=0.3)
                ax2.plot(dd_steps, drawdowns, 'r-', linewidth=1)
                
                # Add max drawdown line
                max_dd = max(drawdowns) if drawdowns else 0
                ax2.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7, 
                           label=f'Max Drawdown: {max_dd:.2f}%')
                
                # Format y-axis as percentage
                def pct_formatter(x, pos):
                    return f'{x:.1f}%'
                    
                ax2.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
                
                # Add labels and title
                ax2.set_xlabel('Trading Steps')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_title('Portfolio Drawdowns')
                ax2.legend()
                ax2.grid(True)
                
                # Invert y-axis for better visualization (0 at top)
                ax2.invert_yaxis()
            
            # Plot 3: Leverage
            ax3 = fig1.add_subplot(gs[1, 1])
            
            # Extract data
            if self.performance_tracking['leverage']:
                lev_steps = [x[0] for x in self.performance_tracking['leverage']]
                leverage = [x[1] for x in self.performance_tracking['leverage']]
                
                # Plot leverage
                ax3.plot(lev_steps, leverage, 'g-', linewidth=1)
                
                # Add max leverage line
                if hasattr(self, 'risk_engine') and hasattr(self.risk_engine, 'risk_limits'):
                    max_lev = self.risk_engine.risk_limits.account_max_leverage
                    ax3.axhline(y=max_lev, color='r', linestyle='--', alpha=0.7, 
                               label=f'Max Leverage: {max_lev:.1f}x')
                
                # Add labels and title
                ax3.set_xlabel('Trading Steps')
                ax3.set_ylabel('Leverage (x)')
                ax3.set_title('Portfolio Leverage')
                ax3.legend()
                ax3.grid(True)
            
            # Plot 4: Risk-adjusted metrics
            ax4 = fig1.add_subplot(gs[2, 0])
            
            # Extract data for Sharpe ratio
            if self.performance_tracking['sharpe_ratios']:
                sharpe_steps = [x[0] for x in self.performance_tracking['sharpe_ratios']]
                sharpe_values = [x[1] for x in self.performance_tracking['sharpe_ratios']]
                
                # Plot Sharpe ratio
                ax4.plot(sharpe_steps, sharpe_values, 'b-', linewidth=1, label='Sharpe Ratio')
            
            # Extract data for Sortino ratio
            if self.performance_tracking['sortino_ratios']:
                sortino_steps = [x[0] for x in self.performance_tracking['sortino_ratios']]
                sortino_values = [x[1] for x in self.performance_tracking['sortino_ratios']]
                
                # Plot Sortino ratio
                ax4.plot(sortino_steps, sortino_values, 'g-', linewidth=1, label='Sortino Ratio')
            
            # Add labels and title
            ax4.set_xlabel('Trading Steps')
            ax4.set_ylabel('Ratio Value')
            ax4.set_title('Risk-Adjusted Performance Metrics')
            ax4.legend()
            ax4.grid(True)
            
            # Plot 5: Trade Distribution
            ax5 = fig1.add_subplot(gs[2, 1])
            
            # Extract trade data
            trades = self.performance_tracking['trades']
            profit_count = trades['profitable']
            loss_count = trades['unprofitable']
            
            # Create bar chart
            if profit_count + loss_count > 0:  # Only plot if we have trades
                categories = ['Profitable', 'Unprofitable']
                counts = [profit_count, loss_count]
                colors = ['green', 'red']
                
                ax5.bar(categories, counts, color=colors)
                
                # Add win rate text
                win_rate = profit_count / (profit_count + loss_count) if (profit_count + loss_count) > 0 else 0
                ax5.text(0.5, 0.9, f'Win Rate: {win_rate:.1%}', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax5.transAxes, fontsize=12)
                
                # Add labels and title
                ax5.set_ylabel('Number of Trades')
                ax5.set_title('Trade Distribution')
                ax5.grid(True, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                fig1.savefig(f"{save_path}/portfolio_performance.png", dpi=300, bbox_inches='tight')
            
            # Add to figures dict
            figures['portfolio_performance'] = fig1
            
            # Create figure for regime analysis
            fig2 = plt.figure(figsize=(15, 8))
            gs2 = gridspec.GridSpec(2, 2)
            
            # Plot 1: Returns by Market Regime
            ax1 = fig2.add_subplot(gs2[0, 0])
            
            # Extract regime data
            regimes = []
            returns = []
            colors = []
            
            for regime, data in self.performance_tracking['regime_performance'].items():
                if data['returns']:
                    avg_return = np.mean(data['returns']) * 100  # Convert to percentage
                    regimes.append(regime)
                    returns.append(avg_return)
                    
                    # Set color based on return
                    if avg_return > 0:
                        colors.append('green')
                    else:
                        colors.append('red')
            
            # Plot regime returns
            if regimes:  # Only plot if we have regime data
                ax1.bar(regimes, returns, color=colors)
                
                # Format y-axis as percentage
                ax1.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
                
                # Add labels and title
                ax1.set_xlabel('Market Regime')
                ax1.set_ylabel('Average Return (%)')
                ax1.set_title('Returns by Market Regime')
                ax1.grid(True, axis='y')
                
                # Rotate x-axis labels for better readability
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 2: Win Rate by Market Regime
            ax2 = fig2.add_subplot(gs2[0, 1])
            
            # Extract win rate data
            regimes = []
            win_rates = []
            trade_counts = []
            
            for regime, data in self.performance_tracking['regime_performance'].items():
                if len(data['trades']) > 0:
                    regimes.append(regime)
                    win_rates.append(data['win_rate'] * 100)  # Convert to percentage
                    trade_counts.append(len(data['trades']))
            
            # Plot win rates
            if regimes:  # Only plot if we have regime data
                # Use scatter size to represent number of trades
                min_size = 50
                max_size = 500
                
                if max(trade_counts) > min(trade_counts):
                    sizes = [min_size + (max_size - min_size) * (count - min(trade_counts)) / 
                            (max(trade_counts) - min(trade_counts)) for count in trade_counts]
                else:
                    sizes = [min_size for _ in trade_counts]
                
                # Create scatter plot
                scatter = ax2.scatter(regimes, win_rates, s=sizes, alpha=0.6, c=win_rates, 
                                     cmap='RdYlGn', vmin=0, vmax=100)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Win Rate (%)')
                
                # Add labels and title
                ax2.set_xlabel('Market Regime')
                ax2.set_ylabel('Win Rate (%)')
                ax2.set_title('Win Rate by Market Regime')
                ax2.grid(True)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                # Add trade count annotations
                for i, regime in enumerate(regimes):
                    ax2.annotate(f"{trade_counts[i]} trades", 
                                (regime, win_rates[i]),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center')
            
            # Plot 3: Adaptive Parameters
            ax3 = fig2.add_subplot(gs2[1, 0])
            
            # Extract parameter data
            if self.performance_tracking['adaptive_parameters']['commission']:
                comm_steps = [x[0] for x in self.performance_tracking['adaptive_parameters']['commission']]
                comm_values = [x[1] * 10000 for x in self.performance_tracking['adaptive_parameters']['commission']]  # Convert to basis points
                
                # Plot commission
                ax3.plot(comm_steps, comm_values, 'b-', linewidth=1, label='Commission (bps)')
            
            if self.performance_tracking['adaptive_parameters']['slippage']:
                slip_steps = [x[0] for x in self.performance_tracking['adaptive_parameters']['slippage']]
                slip_values = [x[1] * 10000 for x in self.performance_tracking['adaptive_parameters']['slippage']]  # Convert to basis points
                
                # Plot slippage
                ax3.plot(slip_steps, slip_values, 'r-', linewidth=1, label='Slippage (bps)')
            
            # Add labels and title
            ax3.set_xlabel('Trading Steps')
            ax3.set_ylabel('Basis Points')
            ax3.set_title('Adaptive Transaction Costs')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Risk Limit Adjustments
            ax4 = fig2.add_subplot(gs2[1, 1])
            
            # Extract risk limit data
            if self.performance_tracking['adaptive_parameters']['risk_limits']:
                risk_steps = [x[0] for x in self.performance_tracking['adaptive_parameters']['risk_limits']]
                acc_lev_values = [x[1]['account_max_leverage'] for x in self.performance_tracking['adaptive_parameters']['risk_limits']]
                pos_lev_values = [x[1]['position_max_leverage'] for x in self.performance_tracking['adaptive_parameters']['risk_limits']]
                
                # Plot risk limits
                ax4.plot(risk_steps, acc_lev_values, 'g-', linewidth=1, label='Account Max Leverage')
                ax4.plot(risk_steps, pos_lev_values, 'b-', linewidth=1, label='Position Max Leverage')
            
            # Add labels and title
            ax4.set_xlabel('Trading Steps')
            ax4.set_ylabel('Leverage (x)')
            ax4.set_title('Adaptive Risk Limits')
            ax4.legend()
            ax4.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                fig2.savefig(f"{save_path}/regime_analysis.png", dpi=300, bbox_inches='tight')
            
            # Add to figures dict
            figures['regime_analysis'] = fig2
            
            # Show plots if requested
            if show_plots:
                plt.show()
            else:
                plt.close('all')
            
            return figures
            
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return {}
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            traceback.print_exc()
            return {}

    def _check_risk_circuit_breakers(self) -> bool:
        """Check if risk circuit breakers should be triggered"""
        # Calculate current drawdown
        current_portfolio = self._calculate_portfolio_value()
        
        if not hasattr(self, 'highest_value'):
            self.highest_value = current_portfolio
            
        drawdown = 0.0
        if self.highest_value > 0:
            drawdown = (self.highest_value - current_portfolio) / self.highest_value
        
        # FIXED: Properly reset highest value when changing episodes
        # This prevents false drawdown from previous episodes
        if drawdown < 0:
            # Portfolio has reached a new peak
            self.highest_value = current_portfolio
            drawdown = 0.0
            
        # CRITICAL FIX: Sanity check on drawdown - ignore extreme values likely due to calculation errors
        if drawdown > 0.5:  # If 50%+ drawdown, likely a calculation error
            logger.warning(f"Extreme drawdown of {drawdown:.2%} detected, ignoring as likely calculation error")
            drawdown = 0.0
            self.highest_value = current_portfolio
        
        # Circuit breaker thresholds
        extreme_threshold = 0.45  # 45% drawdown
        severe_threshold = 0.35   # 35% drawdown
        high_threshold = 0.25     # 25% drawdown
        
        # Check if any circuit breakers triggered
        triggered = False
        
        # Check extreme drawdown - reduce all positions by 50%
        if drawdown > extreme_threshold:
            logger.warning(f"Circuit breaker triggered: Extreme drawdown {drawdown:.2%} > {extreme_threshold:.2%}")
            # Only if real drawdown, not a calculation error
            if drawdown < 0.5:  
                self._reduce_all_positions(0.5)  # Cut all positions by 50%
                triggered = True
        
        # Check severe drawdown - reduce all positions by 25%
        elif drawdown > severe_threshold:
            logger.warning(f"Circuit breaker triggered: Severe drawdown {drawdown:.2%} > {severe_threshold:.2%}")
            self._reduce_all_positions(0.75)  # Cut all positions by 25%
            triggered = True
            
        # Check high drawdown - reduce all positions by 10%
        elif drawdown > high_threshold:
            logger.warning(f"Circuit breaker triggered: High drawdown {drawdown:.2%} > {high_threshold:.2%}")
            self._reduce_all_positions(0.9)  # Cut all positions by 10%
            triggered = True
            
        return triggered

    def _reduce_all_positions(self, scale_factor: float):
        """
        Reduce all positions by a scale factor.
        Used by circuit breakers to reduce risk.
        
        Args:
            scale_factor: Factor to scale positions by (0.5 = reduce by half)
        """
        try:
            for asset in self.assets:
                position_size = self.positions[asset]['size']
                
                # Skip if no position
                if abs(position_size) < 1e-8:
                    continue
                    
                # Calculate the amount to reduce
                reduction = position_size * (1 - scale_factor)
                
                # Execute a trade to reduce the position
                mark_price = self._get_mark_price(asset)
                
                # Add a simulated trade to close part of the position
                self.trades.append({
                    'timestamp': self.current_step,
                    'asset': asset,
                    'size': -reduction,  # Negative = reduce position
                    'price': mark_price,
                    'cost': abs(reduction * mark_price) * self.commission,  # Approximate cost
                    'realized_pnl': 0,  # Will be calculated later
                    'circuit_breaker': True  # Flag this as a circuit breaker action
                })
                
                # Update position size
                self.positions[asset]['size'] *= scale_factor
                
                logger.info(f"Circuit breaker reduced {asset} position by {1-scale_factor:.1%} to {self.positions[asset]['size']:.6f}")
                
        except Exception as e:
            logger.error(f"Error reducing positions: {str(e)}")
            import traceback
            traceback.print_exc()