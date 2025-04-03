import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    # Account-wide limits
    account_max_leverage: float = 5.0  # Maximum account-wide leverage
    max_leverage: float = 20.0        # Maximum leverage for individual positions
    position_max_leverage: float = 20.0  # Alias for max_leverage for compatibility
    
    # Risk thresholds
    max_drawdown: float = 0.3        # Maximum allowed drawdown
    max_drawdown_pct: float = 0.3    # Alias for max_drawdown for compatibility
    var_limit: float = 0.05          # Value at Risk limit
    
    # Position limits
    position_concentration: float = 0.4  # Maximum concentration in a single asset
    account_max_concentration: float = 0.6  # Maximum account-wide concentration
    
    # Correlation and liquidity
    correlation_limit: float = 0.7    # Maximum correlation between positions
    liquidity_ratio: float = 0.1      # Minimum liquidity ratio
    
    # Loss limits
    daily_loss_limit_pct: float = 0.10  # Maximum allowed daily loss (10%)
    
    # Trade limits
    max_trade_size_pct: float = 0.2  # Maximum size of a single trade as % of portfolio
    min_trade_size_usd: float = 100.0  # Minimum trade size in USD
    
    # Volatility scaling
    vol_scaling_enabled: bool = True  # Whether to scale position sizes based on volatility
    vol_target: float = 0.01  # Daily volatility target (1%)
    vol_lookback: int = 21  # Lookback period for volatility calculation
    
    # Stress testing
    stress_multiplier: float = 2.0  # Multiplier for stress testing
    
    # Risk limits for different market regimes
    regime_risk_adjustments: Dict[str, float] = None
    
    # New asset-specific limits
    asset_max_leverage: Dict[str, float] = None  # Asset-specific leverage limits
    
    def __post_init__(self):
        if self.regime_risk_adjustments is None:
            self.regime_risk_adjustments = {
                "low_vol": 1.2,      # Scale up risk limits in low volatility
                "normal": 1.0,       # Normal risk limits
                "high_vol": 0.7,     # Scale down in high volatility
                "crisis": 0.5        # Significantly reduce risk in crisis
            }
        
        # Ensure consistency between aliases
        if self.max_drawdown != self.max_drawdown_pct:
            self.max_drawdown = self.max_drawdown_pct
            
        if self.max_leverage != self.position_max_leverage:
            self.position_max_leverage = self.max_leverage

class InstitutionalRiskEngine:
    def __init__(
        self,
        risk_limits: RiskLimits = RiskLimits(),
        lookback_window: int = 100,
        confidence_level: float = 0.95,
        stress_test_scenarios: List[Dict] = None,
        initial_balance: float = 10000.0,
        use_dynamic_limits: bool = False,
        use_vol_scaling: bool = True
    ):
        self.risk_limits = risk_limits
        self.lookback_window = lookback_window
        self.confidence_level = confidence_level
        self.stress_test_scenarios = stress_test_scenarios or self._default_stress_scenarios()
        self.initial_balance = initial_balance
        self.use_dynamic_limits = use_dynamic_limits
        self.use_vol_scaling = use_vol_scaling
        
        # Historical metrics
        self.historical_var = []
        self.historical_drawdowns = []
        self.historical_leverage = []
        self.historical_net_leverage = []
        self.historical_gross_leverage = []
        self.position_history = []
        
        # Portfolio tracking
        self.portfolio_values = []
        self.current_value = initial_balance
        self.highest_value = initial_balance
        self.current_drawdown = 0.0
        
        # Asset-specific data
        self.asset_volatilities = {}
        self.returns_history = []
        
        # Advanced drawdown protection
        self.drawdown_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        self.drawdown_actions = {}
        self.last_drawdown_alert = 0.0
        self.recovery_high_watermarks = []
        
        logger.info(f"Initialized risk engine with dynamic limits: {use_dynamic_limits}, vol scaling: {use_vol_scaling}")
        
    def reset(self):
        """Reset the risk engine's state variables to initial values"""
        # Reset historical metrics
        self.historical_var = []
        self.historical_drawdowns = []
        self.historical_leverage = []
        self.historical_net_leverage = []
        self.historical_gross_leverage = []
        self.position_history = []
        
        # Reset portfolio tracking
        self.portfolio_values = []
        self.current_value = self.initial_balance
        self.highest_value = self.initial_balance
        self.current_drawdown = 0.0
        
        # Reset asset data
        self.asset_volatilities = {}
        self.returns_history = []
        
        logger.info("Risk engine reset to initial state")
        
    def update_portfolio_value(self, portfolio_value: float, timestamp: int = None):
        """
        Update portfolio value and track metrics
        
        Args:
            portfolio_value: Current portfolio value
            timestamp: Current timestamp (optional)
        """
        # Store previous value for return calculation
        prev_value = self.current_value if self.portfolio_values else self.initial_balance
        self.current_value = portfolio_value
        
        # Use current timestamp or step counter if provided
        if timestamp is None:
            timestamp = len(self.portfolio_values)
            
        # Store portfolio value with timestamp
        self.portfolio_values.append((timestamp, portfolio_value))
        
        # Calculate return if we have previous value
        if prev_value > 0:
            returns = (portfolio_value - prev_value) / prev_value
            self.returns_history.append(returns)
            
            # Keep returns history manageable
            if len(self.returns_history) > 252:  # Roughly 1 year of daily data
                self.returns_history = self.returns_history[-252:]
        
        # Update highest value and drawdown
        if portfolio_value > self.highest_value:
            self.highest_value = portfolio_value
            
        # Calculate current drawdown
        if self.highest_value > 0:
            self.current_drawdown = 1.0 - (portfolio_value / self.highest_value)
            
        # Update volatility calculation
        self._update_volatility()
        
    def _update_volatility(self):
        """Update portfolio volatility estimates based on returns history"""
        if len(self.returns_history) >= 5:
            # Calculate annualized volatility (assuming daily returns)
            vol = np.std(self.returns_history[-21:] if len(self.returns_history) >= 21 else self.returns_history) * np.sqrt(252)
            
            # Store in history
            if not hasattr(self, 'volatility_history'):
                self.volatility_history = []
                
            self.volatility_history.append(vol)
            
            # Keep history manageable
            if len(self.volatility_history) > 63:  # ~3 months
                self.volatility_history = self.volatility_history[-63:]
        
    def update_asset_volatility(self, asset: str, price_history: List[float]):
        """
        Update volatility estimate for a specific asset
        
        Args:
            asset: Asset symbol
            price_history: List of historical prices
        """
        if len(price_history) < 5:
            return
            
        # Calculate returns
        prices = np.array(price_history)
        returns = np.diff(np.log(prices))
        
        # Calculate annualized volatility (assuming daily returns)
        vol = np.std(returns) * np.sqrt(252)
        
        # Store asset volatility
        self.asset_volatilities[asset] = vol
        
    def get_max_leverage_for_asset(self, asset: str) -> float:
        """
        Get the maximum allowed leverage for a specific asset based on risk limits
        
        Args:
            asset: Asset symbol
            
        Returns:
            float: Maximum allowed leverage
        """
        # Start with the account-wide leverage limit
        max_leverage = self.risk_limits.account_max_leverage
        
        # Apply asset-specific adjustment if available
        if hasattr(self.risk_limits, 'asset_max_leverage') and isinstance(self.risk_limits.asset_max_leverage, dict):
            if asset in self.risk_limits.asset_max_leverage:
                max_leverage = self.risk_limits.asset_max_leverage.get(asset, max_leverage)
        
        # Apply volatility-based scaling if we have volatility data
        if asset in self.asset_volatilities:
            vol = self.asset_volatilities[asset]
            # Scale down leverage for higher volatility assets
            vol_scaling = 1.0 / (1.0 + vol)
            max_leverage = max_leverage * min(vol_scaling, 1.0)
        
        return max_leverage
        
    def get_position_size_limits(self, asset: str, portfolio_value: float, verbose: bool = False) -> Dict[str, float]:
        """
        Calculate position sizing limits for a specific asset
        
        Args:
            asset: Asset symbol
            portfolio_value: Current portfolio value
            verbose: Whether to log detailed information
            
        Returns:
            Dict: Position size limits including max_value, min_value
        """
        # Get maximum leverage for this asset
        max_leverage = self.get_max_leverage_for_asset(asset)
        
        # CRYPTO ADJUSTMENT: More permissive position sizing for crypto
        # For leveraged trading, we need to allow larger positions than standard risk models
        max_value_by_leverage = portfolio_value * max_leverage * 1.2  # Increase by 20% for crypto
        max_value_by_concentration = portfolio_value * self.risk_limits.position_concentration * 1.25  # Increase by 25%
        
        # FIXED: For crypto trading, we want to allow more aggressive position sizing
        # Use the more permissive of the two limits for crypto markets
        max_value = max(max_value_by_leverage, max_value_by_concentration)
        
        # DIAGNOSTIC: Log the position size limits calculation
        if verbose:
            logger.debug(f"Position limits for {asset}: max_lev={max_leverage:.2f}x, " +
                       f"by_lev=${max_value_by_leverage:.2f}, by_conc=${max_value_by_concentration:.2f}, " +
                       f"final=${max_value:.2f}")
        
        # Minimum position value - keep at $100 as requested
        min_value = 100.0  # Fixed minimum position size of $100
        
        # Return limits
        return {
            'max_value': max_value,
            'min_value': min_value,
            'max_leverage': max_leverage
        }
    
    def calculate_risk_bands(self, market_data: pd.DataFrame, asset: str) -> Dict[str, float]:
        """
        Calculate risk bands (stop loss and take profit levels) for a given asset
        
        Args:
            market_data: Market data DataFrame
            asset: Asset symbol
            
        Returns:
            Dict: Risk bands including stop_loss_pct and take_profit_pct
        """
        try:
            # Get relevant price data for this asset
            if (asset, 'close') in market_data.columns:
                prices = market_data[(asset, 'close')].values
            else:
                # No price data available, use default values
                return {
                    'stop_loss_pct': 0.10,  # Default 10% stop loss
                    'take_profit_pct': 0.20  # Default 20% take profit
                }
            
            # Calculate volatility using price data
            if len(prices) > 5:
                returns = np.diff(np.log(prices))
                vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
                
                # Store in asset volatilities
                self.asset_volatilities[asset] = vol
                
                # Calculate adaptive risk bands based on volatility
                # Higher volatility = wider bands to avoid getting stopped out too easily
                stop_loss_pct = min(0.25, max(0.05, vol * 2.0))  # Between 5% and 25%
                take_profit_pct = min(0.50, max(0.10, vol * 3.0))  # Between 10% and 50%
            else:
                # Not enough data, use default values
                stop_loss_pct = 0.10
                take_profit_pct = 0.20
                
            return {
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            }
        except Exception as e:
            logger.error(f"Error calculating risk bands for {asset}: {str(e)}")
            return {
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.20
            }
    
    def _default_stress_scenarios(self) -> List[Dict]:
        """Default stress test scenarios"""
        return [
            {
                'name': 'normal',
                'vol_multiplier': 1.0,
                'volume_multiplier': 1.0,
                'correlation_multiplier': 1.0
            },
            {
                'name': 'stress',
                'vol_multiplier': 2.0,
                'volume_multiplier': 0.5,
                'correlation_multiplier': 1.2
            },
            {
                'name': 'crisis',
                'vol_multiplier': 3.0,
                'volume_multiplier': 0.2,
                'correlation_multiplier': 1.5
            }
        ]
        
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Ensure market data has unique index
            market_data = market_data.loc[~market_data.index.duplicated(keep='first')]
            
            # Calculate position metrics
            position_metrics = self._calculate_position_metrics(positions, market_data, portfolio_value)
            
            # Calculate market risk
            market_risk = self._calculate_market_risk(positions, market_data, portfolio_value)
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(positions, market_data)
            
            # Run stress tests
            stress_results = self._run_stress_tests(positions, market_data, portfolio_value)
            
            # Combine all metrics
            risk_metrics = {
                **position_metrics,
                **market_risk,
                **liquidity_risk,
                'stress_test': stress_results
            }
            
            # Update historical metrics
            self._update_historical_metrics(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_portfolio_risk: {str(e)}")
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
                'portfolio_value': portfolio_value,
                'current_drawdown': 0.0
            }
        
    def calculate_risk_metrics(
        self,
        positions: Dict[str, Dict],
        balance: float,
        prices: Dict[str, float]
    ) -> Dict:
        """Alias for calculate_portfolio_risk for backward compatibility"""
        # Create a simple market data DataFrame from the prices
        index = pd.date_range(end=pd.Timestamp.now(), periods=self.lookback_window, freq='5min')
        market_data = pd.DataFrame(index=index)
        
        # Add price data for each asset
        for asset, price in prices.items():
            # Create columns for each asset - support both MultiIndex and flat formats
            # Handle both cases: the environment might pass either format
            if isinstance(price, (int, float)):
                # Single price format - create columns with MultiIndex format
                market_data[(asset, 'close')] = price
                market_data[(asset, 'open')] = price
                market_data[(asset, 'high')] = price
                market_data[(asset, 'low')] = price
                market_data[(asset, 'volume')] = 1000000  # Default volume
            elif isinstance(price, dict):
                # Dict format with multiple price fields
                for field, value in price.items():
                    market_data[(asset, field)] = value
        
        # Ensure the market data has proper MultiIndex columns
        if not isinstance(market_data.columns, pd.MultiIndex):
            # Convert columns to MultiIndex
            tuples = []
            for col in market_data.columns:
                if isinstance(col, tuple):
                    tuples.append(col)
                elif '_' in col:
                    # Split column names like "BTC_close" into ("BTC", "close")
                    asset, field = col.split('_', 1)
                    tuples.append((asset, field))
                else:
                    # Default for columns without underscore
                    tuples.append((col, 'value'))
            
            # Create MultiIndex
            market_data.columns = pd.MultiIndex.from_tuples(tuples, names=['asset', 'field'])
        
        # Calculate comprehensive risk metrics
        return self.calculate_portfolio_risk(positions, market_data, balance)
        
    def _calculate_position_metrics(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """
        Calculate position-based risk metrics, including exposure, concentration, and leverage
        """
        try:
            # Initialize metrics
            total_exposure = 0.0  # Absolute value (for risk limits)
            net_exposure = 0.0    # With sign (for directional bias)
            exposures = {}
            
            # Process each position
            for asset, position in positions.items():
                # Skip positions with zero size
                if position['size'] == 0:
                    continue
                
                try:
                    # Get price data for this asset
                    if 'value' in position:
                        # If value is provided directly (absolute)
                        position_exposure = abs(position['value'])
                        # Try to determine position direction for net exposure
                        position_value = position['value']
                        if 'size' in position:
                            # If size is available, use its sign for direction
                            position_value = position['value'] * (1 if position['size'] > 0 else -1)
                    else:
                        # Calculate from size and price
                        try:
                            # First try using position's mark_price if available
                            if 'mark_price' in position:
                                price = position['mark_price']
                            # Then try the last row of market data - handle both MultiIndex and flat formats
                            elif market_data is not None:
                                # Try MultiIndex format first
                                if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'close') in market_data.columns:
                                    price = market_data.iloc[-1][(asset, 'close')]
                                    if isinstance(price, (pd.Series, pd.DataFrame)):
                                        price = price.iloc[0]
                                # Then try flat format
                                elif f"{asset}_close" in market_data.columns:
                                    price = market_data.iloc[-1][f"{asset}_close"]
                                # Try other variations if needed
                                elif asset in market_data.columns:
                                    price = market_data.iloc[-1][asset]
                                else:
                                    # Default to entry price if no market data found
                                    price = position['entry_price']
                            else:
                                # Default to entry price if no market data
                                price = position['entry_price']
                            
                            # Calculate position value with direction
                            position_value = position['size'] * price
                            # Position exposure is the absolute value
                            position_exposure = abs(position_value)
                            
                        except Exception as e:
                            logger.error(f"Error calculating position value for {asset}: {str(e)}")
                            # Use reasonable defaults
                            position_value = 0
                            position_exposure = 0
                    
                    # Add to total exposure (absolute value for risk limits)
                    total_exposure += position_exposure
                    # Add to net exposure (with sign for directional bias)
                    net_exposure += position_value
                    # Track individual asset exposure
                    exposures[asset] = position_exposure
                    
                except Exception as e:
                    logger.error(f"Error processing position for {asset}: {str(e)}")
                    continue
            
            # Position concentration with safety checks
            max_concentration = max(exposures.values()) / (portfolio_value + 1e-8) if exposures else 0
            
            # Calculate both gross and net leverage
            gross_leverage = total_exposure / (portfolio_value + 1e-8)  # Always positive
            net_leverage = net_exposure / (portfolio_value + 1e-8)     # Can be negative
            
            # Calculate correlation risk using returns data
            correlation_risk = 0.0
            try:
                # Only calculate correlation if we have multiple positions
                if len([p for p in positions.values() if p['size'] != 0]) >= 2:
                    # Create returns DataFrame with proper index
                    returns_data = pd.DataFrame(index=market_data.index)
                    
                    # Get returns for each asset with a position
                    for asset, position in positions.items():
                        if abs(position['size']) <= 1e-8:
                            continue
                            
                        # Get price data - try different column formats
                        try:
                            # Try MultiIndex format
                            if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'close') in market_data.columns:
                                prices = market_data.loc[:, (asset, 'close')]
                                if isinstance(prices, pd.DataFrame):
                                    prices = prices.iloc[:, 0]
                            # Try flat format
                            elif f"{asset}_close" in market_data.columns:
                                prices = market_data.loc[:, f"{asset}_close"]
                            # Try other variations
                            elif asset in market_data.columns:
                                prices = market_data.loc[:, asset]
                            else:
                                # Skip this asset if no price data found
                                logger.warning(f"No price data found for {asset}, skipping correlation calculation")
                                continue
                                
                            # Calculate log returns
                            returns_data[asset] = np.log(prices).diff().fillna(0)
                        except Exception as e:
                            logger.error(f"Error calculating returns for {asset}: {str(e)}")
                            continue
                    
                    # Calculate correlation risk only if we have enough data
                    if len(returns_data.columns) >= 2:
                        # Calculate correlation matrix
                        corr_matrix = returns_data.corr().fillna(0)
                        
                        # Calculate position-weighted correlation
                        weights = []
                        for asset in returns_data.columns:
                            abs_size = abs(positions[asset]['size'])
                            weights.append(abs_size / (sum(abs(p['size']) for p in positions.values()) + 1e-8))
                        
                        weights = np.array(weights)
                        correlation_risk = float(weights.T @ corr_matrix @ weights)
            except Exception as e:
                logger.error(f"Error calculating correlation risk: {str(e)}")
                correlation_risk = 0.0
            
            return {
                'total_exposure': total_exposure,
                'net_exposure': net_exposure,
                'gross_leverage': gross_leverage,
                'net_leverage': net_leverage,
                'leverage_utilization': gross_leverage,  # Keep for backward compatibility
                'max_concentration': max_concentration,
                'correlation_risk': correlation_risk,
                'num_positions': len([p for p in positions.values() if p['size'] != 0])
            }
        except Exception as e:
            logger.error(f"Error in _calculate_position_metrics: {str(e)}")
            return {
                'total_exposure': 0,
                'net_exposure': 0,
                'gross_leverage': 0,
                'net_leverage': 0,
                'leverage_utilization': 0,
                'max_concentration': 0,
                'correlation_risk': 0,
                'num_positions': 0
            }
        
    def _calculate_market_risk(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """Calculate market risk metrics including VaR and Expected Shortfall"""
        try:
            # Calculate portfolio returns using proper MultiIndex handling
            portfolio_returns = np.zeros(len(market_data))
            
            for asset, position in positions.items():
                if position['size'] != 0:
                    try:
                        # Get close prices using flexible column access
                        prices = None
                        
                        # Try different column formats
                        if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'close') in market_data.columns:
                            # MultiIndex format
                            prices = market_data.loc[:, (asset, 'close')]
                            if isinstance(prices, pd.DataFrame):
                                prices = prices.iloc[:, 0]
                        elif f"{asset}_close" in market_data.columns:
                            # Flat format with underscore
                            prices = market_data.loc[:, f"{asset}_close"]
                        elif asset in market_data.columns:
                            # Simple column format
                            prices = market_data.loc[:, asset]
                        else:
                            # Skip if no price data available
                            logger.warning(f"No price data found for {asset}, skipping market risk calculation")
                            continue
                        
                        # Calculate log returns
                        asset_returns = np.log(prices).diff().fillna(0)
                        
                        # Weight returns by position size and value
                        position_value = position['size'] * prices.iloc[-1]
                        weight = position_value / (portfolio_value + 1e-8)
                        portfolio_returns += weight * asset_returns.values
                        
                    except Exception as e:
                        logger.error(f"Error calculating returns for {asset}: {str(e)}")
                        continue
            
            if len(portfolio_returns) < self.lookback_window:
                return {
                    'var': 0.0,
                    'expected_shortfall': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Use recent window for calculations
            recent_returns = portfolio_returns[-self.lookback_window:]
            
            # Value at Risk (VaR)
            var = -np.percentile(recent_returns, (1 - self.confidence_level) * 100)
            
            # Expected Shortfall (CVaR)
            es = -np.mean(recent_returns[recent_returns <= -var]) if any(recent_returns <= -var) else var
            
            # Volatility (annualized)
            volatility = np.std(recent_returns) * np.sqrt(252)
            
            # Maximum Drawdown
            cumulative_returns = np.cumsum(recent_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = -np.min(drawdowns)
            
            return {
                'var': float(var),
                'expected_shortfall': float(es),
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown)
            }
            
        except Exception as e:
            logger.error(f"Error in _calculate_market_risk: {str(e)}")
            return {
                'var': 0.0,
                'expected_shortfall': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0
            }
        
    def _calculate_liquidity_risk(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame
    ) -> Dict:
        """Calculate liquidity risk metrics"""
        try:
            liquidation_costs = {}
            adv_ratios = {}
            
            # Get the latest timestamp
            latest_timestamp = market_data.index[-1]
            
            for asset, position in positions.items():
                if position['size'] != 0:
                    try:
                        # Get volume using flexible column access
                        volume = None
                        close_price = None
                        
                        # Try different column formats for volume
                        if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'volume') in market_data.columns:
                            # MultiIndex format
                            volume = market_data.loc[:, (asset, 'volume')]
                            if isinstance(volume, pd.DataFrame):
                                volume = volume.iloc[:, 0]
                        elif f"{asset}_volume" in market_data.columns:
                            # Flat format with underscore
                            volume = market_data.loc[:, f"{asset}_volume"]
                        else:
                            # Use default value if no volume data
                            volume = pd.Series(1000000, index=market_data.index)
                            logger.warning(f"No volume data for {asset}, using default value")
                            
                        # Calculate average daily volume (5-day moving average)
                        adv = volume.rolling(window=5, min_periods=1).mean().iloc[-1]
                        
                        # Try different column formats for close price
                        if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'close') in market_data.columns:
                            # MultiIndex format 
                            close = market_data.loc[latest_timestamp, (asset, 'close')]
                            if isinstance(close, (pd.Series, pd.DataFrame)):
                                close = float(close.iloc[0])
                            else:
                                close = float(close)
                        elif f"{asset}_close" in market_data.columns:
                            # Flat format with underscore
                            close = float(market_data.loc[latest_timestamp, f"{asset}_close"])
                        elif asset in market_data.columns:
                            # Simple column
                            close = float(market_data.loc[latest_timestamp, asset])
                        elif 'mark_price' in position:
                            # Use position's mark price if available
                            close = position['mark_price']
                        else:
                            # Use entry price as fallback
                            close = position['entry_price']
                        
                        # Calculate position value
                        position_value = abs(position['size'] * close)
                        
                        # Calculate ADV ratio (position value / daily volume value)
                        adv_ratio = position_value / (adv * close + 1e-8)
                        adv_ratios[asset] = adv_ratio
                        
                        # Calculate liquidation cost using square-root law
                        # Higher ADV ratio means higher impact
                        impact = 0.1 * np.sqrt(adv_ratio)  # 10 bps per sqrt of ADV ratio
                        liquidation_costs[asset] = impact * position_value
                        
                    except Exception as e:
                        logger.error(f"Error calculating liquidity metrics for {asset}: {str(e)}")
                        adv_ratios[asset] = 0.0
                        liquidation_costs[asset] = 0.0
            
            return {
                'max_adv_ratio': float(max(adv_ratios.values())) if adv_ratios else 0.0,
                'total_liquidation_cost': float(sum(liquidation_costs.values())),
                'max_liquidation_cost': float(max(liquidation_costs.values())) if liquidation_costs else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in _calculate_liquidity_risk: {str(e)}")
            return {
                'max_adv_ratio': 0.0,
                'total_liquidation_cost': 0.0,
                'max_liquidation_cost': 0.0
            }
        
    def _run_stress_tests(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """Run stress tests under different scenarios"""
        try:
            results = {}
            base_metrics = self._calculate_market_risk(positions, market_data, portfolio_value)
            
            for scenario in self.stress_test_scenarios:
                try:
                    # Create a copy of market data for this scenario
                    stressed_data = market_data.copy()
                    
                    # Apply scenario multipliers to each asset
                    for asset in positions.keys():
                        if positions[asset]['size'] != 0:
                            try:
                                # Create new DataFrames for the stressed values
                                asset_data = pd.DataFrame(index=market_data.index)
                                
                                # Get and process close prices
                                close_prices = market_data.loc[:, (asset, 'close')]
                                if isinstance(close_prices, pd.DataFrame):
                                    close_prices = close_prices.iloc[:, 0]
                                
                                # Calculate stressed close prices
                                returns = np.log(close_prices).diff()
                                stressed_returns = returns * scenario['vol_multiplier']
                                initial_price = close_prices.iloc[0]
                                stressed_prices = np.exp(np.log(initial_price) + stressed_returns.cumsum())
                                asset_data['close'] = stressed_prices
                                
                                # Get and process volume
                                volume = market_data.loc[:, (asset, 'volume')]
                                if isinstance(volume, pd.DataFrame):
                                    volume = volume.iloc[:, 0]
                                asset_data['volume'] = volume * scenario['volume_multiplier']
                                
                                # Update the MultiIndex DataFrame properly
                                for col in ['close', 'volume']:
                                    stressed_data[(asset, col)] = asset_data[col]
                                
                            except Exception as e:
                                logger.error(f"Error applying stress scenario to {asset}: {str(e)}")
                                continue
                    
                    # Calculate stressed metrics
                    stressed_metrics = self._calculate_market_risk(
                        positions, stressed_data, portfolio_value
                    )
                    
                    # Calculate impact ratios with safety checks
                    results[scenario['name']] = {}
                    for metric in stressed_metrics:
                        base_value = base_metrics.get(metric, 1e-8)
                        stressed_value = stressed_metrics.get(metric, base_value)
                        results[scenario['name']][metric] = stressed_value / (base_value + 1e-8)
                    
                except Exception as e:
                    logger.error(f"Error in stress scenario {scenario['name']}: {str(e)}")
                    results[scenario['name']] = {
                        'var': 1.0,
                        'expected_shortfall': 1.0,
                        'volatility': 1.0,
                        'max_drawdown': 1.0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in _run_stress_tests: {str(e)}")
            return {
                'normal': {
                    'var': 1.0,
                    'expected_shortfall': 1.0,
                    'volatility': 1.0,
                    'max_drawdown': 1.0
                }
            }
        
    def _calculate_portfolio_returns(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate historical portfolio returns"""
        try:
            portfolio_returns = np.zeros(len(market_data))
            
            for asset, position in positions.items():
                if position['size'] != 0:
                    try:
                        # Get close prices using proper MultiIndex access
                        prices = market_data.loc[:, (asset, 'close')]
                        if isinstance(prices, pd.DataFrame):
                            prices = prices.iloc[:, 0]
                        
                        # Calculate log returns
                        asset_returns = np.log(prices).diff().fillna(0)
                        
                        # Weight returns by position size
                        portfolio_returns += position['size'] * asset_returns.values
                    except Exception as e:
                        logger.error(f"Error calculating returns for {asset}: {str(e)}")
                        continue
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error in _calculate_portfolio_returns: {str(e)}")
            return np.zeros(len(market_data))
        
    def _update_historical_metrics(self, risk_metrics: Dict):
        """Update historical risk metrics"""
        self.historical_var.append(risk_metrics['var'])
        self.historical_drawdowns.append(risk_metrics['max_drawdown'])
        
        # Update both gross and net leverage history
        if 'gross_leverage' in risk_metrics:
            if not hasattr(self, 'historical_gross_leverage'):
                self.historical_gross_leverage = []
            self.historical_gross_leverage.append(risk_metrics['gross_leverage'])
            
            # Keep backward compatibility
            self.historical_leverage.append(risk_metrics['gross_leverage'])
        else:
            self.historical_leverage.append(risk_metrics['leverage_utilization'])
            
        if 'net_leverage' in risk_metrics:
            if not hasattr(self, 'historical_net_leverage'):
                self.historical_net_leverage = []
            self.historical_net_leverage.append(risk_metrics['net_leverage'])
        
    def check_risk_limits(self, risk_metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if current risk metrics exceed defined limits"""
        violations = []
        
        # Check drawdown
        if risk_metrics['max_drawdown'] > self.risk_limits.max_drawdown:
            violations.append(f"Max drawdown ({risk_metrics['max_drawdown']:.2%}) exceeds limit ({self.risk_limits.max_drawdown:.2%})")
            
        # Check VaR
        if risk_metrics['var'] > self.risk_limits.var_limit:
            violations.append(f"VaR ({risk_metrics['var']:.2%}) exceeds limit ({self.risk_limits.var_limit:.2%})")
            
        # Check leverage using gross leverage (always positive)
        # This ensures risk limits are enforced regardless of short/long direction
        leverage_to_check = risk_metrics.get('gross_leverage', risk_metrics['leverage_utilization'])
        
        # CRITICAL IMPROVEMENT: Separate checks for account leverage vs position leverage
        # Check account-wide leverage (stricter limit)
        if leverage_to_check > self.risk_limits.account_max_leverage:
            violations.append(f"Account gross leverage ({leverage_to_check:.2f}x) exceeds account limit ({self.risk_limits.account_max_leverage:.2f}x)")
            
        # Individual position leverage is checked separately in trade execution
        # But we still check if we're exceeding the absolute maximum leverage
        if leverage_to_check > self.risk_limits.max_leverage:
            violations.append(f"Gross leverage ({leverage_to_check:.2f}x) exceeds absolute maximum ({self.risk_limits.max_leverage:.2f}x)")
            
        # Check for excessive net leverage in either direction
        if 'net_leverage' in risk_metrics:
            # For net leverage, we look at the absolute value to enforce limits in both directions
            net_leverage_abs = abs(risk_metrics['net_leverage'])
            if net_leverage_abs > self.risk_limits.account_max_leverage:
                directions = "short" if risk_metrics['net_leverage'] < 0 else "long"
                violations.append(f"Net {directions} leverage ({risk_metrics['net_leverage']:.2f}x) exceeds limit (±{self.risk_limits.account_max_leverage:.2f}x)")
                
        # Check concentration
        if risk_metrics['max_concentration'] > self.risk_limits.position_concentration:
            violations.append(f"Position concentration ({risk_metrics['max_concentration']:.2%}) exceeds limit ({self.risk_limits.position_concentration:.2%})")
            
        # Check correlation risk
        if risk_metrics['correlation_risk'] > self.risk_limits.correlation_limit:
            violations.append(f"Correlation risk ({risk_metrics['correlation_risk']:.2f}) exceeds limit ({self.risk_limits.correlation_limit:.2f})")
            
        return len(violations) > 0, violations
        
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'historical_metrics': {
                'var': self.historical_var,
                'drawdowns': self.historical_drawdowns,
                'leverage': self.historical_leverage
            },
            'risk_limits': vars(self.risk_limits),
            'current_utilization': {
                'var_utilization': self.historical_var[-1] / self.risk_limits.var_limit if self.historical_var else 0,
                'drawdown_utilization': self.historical_drawdowns[-1] / self.risk_limits.max_drawdown if self.historical_drawdowns else 0,
                'leverage_utilization': self.historical_leverage[-1] / self.risk_limits.max_leverage if self.historical_leverage else 0
            }
        }
        
    def scale_action_by_risk(self, action: float, asset: str) -> float:
        """
        Scale an action by risk parameters
        
        Args:
            action: Original action value in range [-1, 1]
            asset: Asset symbol
            
        Returns:
            float: Scaled action value
        """
        # REVISED: No more boosting small actions, let natural signal flow
        # Base case - if action is close to zero or asset unknown, return as is
        if abs(action) < 1e-8 or asset not in self.asset_volatilities:
            return action
            
        # Get volatility scaling factor - reduce position size for high volatility assets
        vol_scaling = 1.0
        if self.use_vol_scaling and asset in self.asset_volatilities:
            # Get current volatility
            vol = self.asset_volatilities[asset]
            
            # Calculate inverse volatility scaling (lower for higher volatility)
            base_vol = 0.01  # 1% daily volatility as reference
            vol_ratio = base_vol / max(vol, 0.001)  # Avoid division by zero
            
            # Apply square root scaling (mathematical risk approach)
            vol_scaling = min(1.0, max(0.1, vol_ratio))
            
        # Get leverage adjustment based on position concentration
        leverage_scaling = 1.0
        if hasattr(self, 'current_value') and self.current_value > 0:
            # Get current portfolio concentration
            exposure_utilization = self.historical_leverage[-1] / self.risk_limits.max_leverage if self.historical_leverage else 0
            
            # Scale down as we approach max leverage
            if exposure_utilization > 0.8:
                leverage_scaling = 1.0 - (exposure_utilization - 0.8) * 2  # Linear reduction from 80% to 100%
                leverage_scaling = max(0.2, leverage_scaling)  # Don't go below 20%
                
        # Get dynamic risk adjustment based on current drawdown
        drawdown_scaling = 1.0
        if hasattr(self, 'current_drawdown') and self.current_drawdown > 0:
            # Scale down as drawdown increases
            drawdown_threshold = self.risk_limits.max_drawdown * 0.5  # Start scaling at 50% of max drawdown
            if self.current_drawdown > drawdown_threshold:
                # Linear reduction from threshold to max drawdown
                scale_factor = 1.0 - (self.current_drawdown - drawdown_threshold) / (self.risk_limits.max_drawdown - drawdown_threshold)
                drawdown_scaling = max(0.25, scale_factor)  # Don't go below 25%
        
        # Calculate final scaling factor
        risk_scaling = vol_scaling * leverage_scaling * drawdown_scaling
        
        # Apply scaling to action
        scaled_action = action * risk_scaling
        
        return scaled_action
        
    def get_max_position_size(self, asset: str, portfolio_value: float, price: float, 
                             market_data: pd.DataFrame = None, positions: Dict[str, Dict] = None, 
                             verbose: bool = False) -> float:
        """
        Get the maximum allowed position size for an asset with enhanced risk controls
        
        Args:
            asset: Asset symbol
            portfolio_value: Current portfolio value
            price: Current price of the asset
            market_data: Optional market data for impact analysis
            positions: Optional current positions dictionary for correlation analysis
            verbose: Whether to log detailed information
            
        Returns:
            float: Maximum allowed position size in units of the asset
        """
        # Get position size limits in dollar value - either basic or correlation-adjusted
        correlation_data = None
        
        # Debug logging to diagnose issues
        # logger.warning(f"DEBUG: get_max_position_size called for {asset} with portfolio_value=${portfolio_value:.2f}, price=${price:.2f}")
        
        # If we have positions and market data, calculate correlation-based limits
        if positions is not None and market_data is not None:
            try:
                # Calculate correlation adjustments
                correlation_data = self.implement_correlation_based_position_limits(positions, market_data)
                
                # Get adjusted limits using correlation data
                limits = self.get_adjusted_position_size_limits(asset, portfolio_value, correlation_data, verbose)
                
                # Log correlation-adjusted limits
                if 'correlation_adjustment' in limits and verbose:
                    logger.debug(f"Applied correlation adjustment for {asset}: factor={limits['correlation_adjustment']:.2f}")
            except Exception as e:
                logger.error(f"Error applying correlation-based limits: {str(e)}, using default limits")
                limits = self.get_position_size_limits(asset, portfolio_value, verbose)
        else:
            # Use standard limits if we don't have correlation data
            limits = self.get_position_size_limits(asset, portfolio_value, verbose)
        
        # FIXED: Ensure proper conversion from dollar value to asset units
        # For very high-priced assets like BTC, this can result in tiny unit numbers
        if price <= 0:
            logger.warning(f"Invalid price for {asset}: {price}, using 1.0 as fallback")
            price = 1.0
            
        # Convert max dollar value to max position size using the price
        naive_max_size = limits['max_value'] / price
        
        # ENHANCED: Use market impact model to adjust maximum position size
        if market_data is not None:
            try:
                # Calculate max size that keeps impact under threshold
                max_impact_bps = 50.0  # Increased from 30 to 50 bps maximum impact for crypto
                optimal_execution = self.get_optimal_execution_size(
                    asset, naive_max_size, price, max_impact_bps, market_data
                )
                
                # If market impact would be too severe, reduce max size
                if optimal_execution['optimal_size'] < naive_max_size:
                    impact_limited_size = optimal_execution['optimal_size']
                    if verbose:
                        logger.info(f"Market impact limiting position size for {asset}: " +
                                   f"{naive_max_size:.6f} -> {impact_limited_size:.6f} " +
                                   f"(impact: {optimal_execution['expected_impact'].get('impact_bps', 0):.1f} bps)")
                    naive_max_size = impact_limited_size
            except Exception as e:
                logger.error(f"Error applying market impact limits: {str(e)}")
                # Continue with naive max size
        
        # IMPROVED: Add safeguards for extremely large or small position sizes
        # Cap to reasonable maximum sizes for each asset
        asset_caps = {
            'BTCUSDT': 100,     # Max 100 BTC
            'ETHUSDT': 1000,    # Max 1000 ETH
            'SOLUSDT': 10000,   # Max 10000 SOL
        }
        
        # Get asset specific cap or use default
        default_cap = 10000  # Default cap for assets not in the list
        asset_cap = asset_caps.get(asset, default_cap)
        
        # Apply the cap
        max_size = min(naive_max_size, asset_cap)
        
        # IMPROVED: Apply any active drawdown protection
        if hasattr(self, 'drawdown_protection_active') and self.drawdown_protection_active:
            # Get latest drawdown protection settings
            drawdown_protection = getattr(self, 'last_drawdown_protection', None)
            if drawdown_protection and 'position_scale_factor' in drawdown_protection:
                scale_factor = drawdown_protection['position_scale_factor']
                if scale_factor < 1.0:
                    original_max = max_size
                    max_size *= scale_factor
                    if verbose:
                        logger.info(f"Drawdown protection reducing max position for {asset}: " +
                               f"{original_max:.6f} -> {max_size:.6f} (factor: {scale_factor:.2f})")
                    
        # IMPROVED: Log more details about position size calculation
        corr_factor = correlation_data[asset]['corr_adjustment_factor'] if correlation_data and asset in correlation_data else 1.0
        if verbose:
            logger.debug(f"Max position size for {asset}: {max_size:.4f} units (${limits['max_value']:.2f} at ${price:.2f}, " + 
                        f"leverage: {limits['max_leverage']:.2f}x, cap: {asset_cap} units, corr_factor: {corr_factor:.2f})")
        
        return float(max_size)  # Ensure we return a float for consistency
        
    def get_stop_loss_take_profit_levels(self, asset: str, price: float, position_size: float, signal: float, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate dynamic stop loss and take profit levels based on asset volatility and position characteristics
        
        Args:
            asset: Asset symbol
            price: Current price of the asset
            position_size: Size of the position (positive for long, negative for short)
            signal: Trading signal strength (-1.0 to 1.0)
            portfolio_value: Current portfolio value
            
        Returns:
            Dict: Contains stop_loss_pct, take_profit_pct, and trailing_stop_pct
        """
        # Default values
        default_stop_pct = 0.10  # 10% stop loss
        default_take_pct = 0.20  # 20% take profit
        default_trail_pct = 0.05  # 5% trailing stop
        
        # Get asset volatility if available
        if asset in self.asset_volatilities:
            vol = self.asset_volatilities[asset]
            
            # Adjust stop loss based on volatility - higher vol = wider stops
            # Use 2x daily volatility as a baseline for stop loss
            stop_loss_pct = min(0.25, max(0.05, vol * 2.0))  # Between 5% and 25%
            
            # Take profit should be wider than stop loss
            # Use risk:reward ratio of at least 1:2
            take_profit_pct = min(0.50, max(0.10, stop_loss_pct * 2.0))  # Between 10% and 50%
            
            # Trailing stop should be tighter than initial stop
            trailing_stop_pct = min(0.15, max(0.03, vol * 1.5))  # Between 3% and 15%
            
            # Adjust based on signal strength - stronger signals get more room
            signal_modifier = abs(signal)
            stop_loss_pct = stop_loss_pct * (0.8 + 0.4 * signal_modifier)
            take_profit_pct = take_profit_pct * (0.8 + 0.4 * signal_modifier)
            
            # Adjust based on position size relative to portfolio
            # Larger positions (as % of portfolio) get tighter stops
            position_value = abs(position_size * price)
            position_pct = position_value / max(portfolio_value, 1e-8)
            
            if position_pct > 0.2:  # Position is > 20% of portfolio
                # Apply tighter stops for large positions to manage risk
                position_size_factor = 1.0 - min(0.5, (position_pct - 0.2) / 0.6)  # Scaling factor
                stop_loss_pct *= position_size_factor
                take_profit_pct *= position_size_factor
        else:
            # No volatility data available, use defaults
            stop_loss_pct = default_stop_pct
            take_profit_pct = default_take_pct
            trailing_stop_pct = default_trail_pct
        
        # Apply any market regime adjustments if enabled
        if self.use_dynamic_limits and hasattr(self, 'current_market_regime'):
            regime = getattr(self, 'current_market_regime', 'normal')
            if regime in self.risk_limits.regime_risk_adjustments:
                # Adjust stop/target based on market regime
                modifier = self.risk_limits.regime_risk_adjustments[regime]
                
                # In high vol/crisis regimes, tighten stops (lower modifier)
                # In low vol regimes, can give more room (higher modifier)
                stop_loss_pct /= modifier
                take_profit_pct /= modifier
                trailing_stop_pct /= modifier
        
        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct
        }

    def implement_sophisticated_drawdown_protection(self, current_portfolio_value: float) -> Dict[str, Any]:
        """
        Implement sophisticated drawdown protection with multiple thresholds and progressive risk reduction.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Dict: Protection actions to be taken including:
                - position_scale_factor: Factor to scale positions by (0-1)
                - leverage_reduction: Factor to reduce leverage by (0-1)
                - trading_suspension: Whether to temporarily suspend trading
                - hedging_actions: Specific hedging actions to take
                - threshold_breached: The threshold that was breached
                - drawdown: Current drawdown value
        """
        # Initialize result dictionary
        result = {
            'position_scale_factor': 1.0,
            'leverage_reduction': 0.0,
            'trading_suspension': False,
            'hedging_actions': [],
            'threshold_breached': None,
            'drawdown': self.current_drawdown,
            'actions_taken': []
        }
        
        # Early return if no drawdown
        if self.current_drawdown <= 0.01:
            return result
            
        # Calculate time since last update (to avoid too frequent protection triggers)
        current_time = len(self.portfolio_values)
        time_since_last_alert = current_time - getattr(self, 'last_drawdown_alert_time', 0)
        
        # Find the highest threshold breached
        breached_threshold = 0.0
        for threshold in sorted(self.drawdown_thresholds, reverse=True):
            if self.current_drawdown >= threshold:
                breached_threshold = threshold
                break
                
        # If no threshold breached, return default values
        if breached_threshold == 0.0:
            return result
            
        # Check if we've already alerted at this level and cooling off period not passed
        if self.last_drawdown_alert >= breached_threshold and time_since_last_alert < 5:
            return result
            
        # Update last alert threshold and time
        self.last_drawdown_alert = breached_threshold
        self.last_drawdown_alert_time = current_time
        
        # Track new recovery high watermark
        if not self.recovery_high_watermarks or current_portfolio_value > self.recovery_high_watermarks[-1]['value']:
            self.recovery_high_watermarks.append({
                'time': current_time,
                'value': current_portfolio_value,
                'drawdown_from_peak': self.current_drawdown
            })
        
        # Progressive risk reduction based on drawdown thresholds
        if breached_threshold >= 0.30:
            # Severe drawdown: Major intervention needed
            result['position_scale_factor'] = 0.1  # Reduce positions by 90%
            result['leverage_reduction'] = 0.9  # Reduce leverage by 90%
            result['trading_suspension'] = True  # Pause trading
            result['hedging_actions'] = ['full_hedge']  # Add specific hedging instructions
            result['actions_taken'].append("Emergency risk reduction: 90% position reduction")
            
        elif breached_threshold >= 0.25:
            # Critical drawdown: Significant intervention
            result['position_scale_factor'] = 0.2  # Reduce positions by 80%
            result['leverage_reduction'] = 0.8  # Reduce leverage by 80%
            result['hedging_actions'] = ['partial_hedge']
            result['actions_taken'].append("Critical risk reduction: 80% position reduction")
            
        elif breached_threshold >= 0.20:
            # Serious drawdown: Strong intervention
            result['position_scale_factor'] = 0.3  # Reduce positions by 70%
            result['leverage_reduction'] = 0.7  # Reduce leverage by 70%
            result['actions_taken'].append("Strong risk reduction: 70% position reduction")
            
        elif breached_threshold >= 0.15:
            # Moderate drawdown: Meaningful intervention
            result['position_scale_factor'] = 0.5  # Reduce positions by 50%
            result['leverage_reduction'] = 0.5  # Reduce leverage by 50%
            result['actions_taken'].append("Moderate risk reduction: 50% position reduction")
            
        elif breached_threshold >= 0.10:
            # Mild drawdown: Precautionary intervention
            result['position_scale_factor'] = 0.7  # Reduce positions by 30%
            result['leverage_reduction'] = 0.3  # Reduce leverage by 30%
            result['actions_taken'].append("Precautionary risk reduction: 30% position reduction")
            
        elif breached_threshold >= 0.05:
            # Minor drawdown: Light intervention
            result['position_scale_factor'] = 0.85  # Reduce positions by 15%
            result['leverage_reduction'] = 0.15  # Reduce leverage by 15%
            result['actions_taken'].append("Light risk reduction: 15% position reduction")
        
        # Set the threshold that was breached
        result['threshold_breached'] = breached_threshold
        
        # Log the protection actions
        actions_str = ", ".join(result['actions_taken'])
        logger.warning(f"Drawdown protection triggered at {self.current_drawdown:.2%} drawdown. Actions: {actions_str}")
        
        return result
        
    def check_drawdown_recovery(self) -> Dict[str, Any]:
        """
        Check if the portfolio is recovering from a drawdown and adjust risk parameters accordingly.
        
        Returns:
            Dict: Recovery actions including:
                - position_scale_increase: Factor to increase position sizing (>1.0)
                - leverage_increase: Factor to increase leverage by (0-1)
                - recovery_level: Recovery level identified (none, partial, significant, full)
        """
        result = {
            'position_scale_increase': 1.0,
            'leverage_increase': 0.0,
            'recovery_level': 'none',
            'actions_taken': []
        }
        
        # Need at least 2 values to calculate recovery
        if len(self.portfolio_values) < 2 or len(self.recovery_high_watermarks) < 2:
            return result
            
        # Get current value and previous high watermark
        current_value = self.portfolio_values[-1][1]
        
        # Calculate recovery from the most severe drawdown point
        max_drawdown_point = min(self.portfolio_values[-100:], key=lambda x: x[1])[1] if len(self.portfolio_values) > 100 else min(self.portfolio_values, key=lambda x: x[1])[1]
        
        if max_drawdown_point <= 0:
            return result
            
        recovery_from_bottom = (current_value - max_drawdown_point) / max_drawdown_point
        
        # Calculate recovery percentage from previous high watermark
        recovery_from_watermark = 1.0 - self.current_drawdown
        
        # Determine recovery level
        if recovery_from_bottom > 0.25 and self.current_drawdown < 0.10:
            # Significant recovery (>25% from bottom, <10% from peak)
            result['position_scale_increase'] = 1.3  # Allow 30% larger positions
            result['leverage_increase'] = 0.3  # Allow 30% more leverage
            result['recovery_level'] = 'significant'
            result['actions_taken'].append("Significant recovery detected: increasing risk allowance by 30%")
            
        elif recovery_from_bottom > 0.15 and self.current_drawdown < 0.15:
            # Moderate recovery
            result['position_scale_increase'] = 1.15  # Allow 15% larger positions
            result['leverage_increase'] = 0.15  # Allow 15% more leverage
            result['recovery_level'] = 'moderate'
            result['actions_taken'].append("Moderate recovery detected: increasing risk allowance by 15%")
            
        elif recovery_from_bottom > 0.05:
            # Slight recovery
            result['position_scale_increase'] = 1.05  # Allow 5% larger positions
            result['leverage_increase'] = 0.05  # Allow 5% more leverage
            result['recovery_level'] = 'slight'
            result['actions_taken'].append("Slight recovery detected: increasing risk allowance by 5%")
        
        # Log recovery actions if any
        # if result['actions_taken']:
        #     logger.info(f"Drawdown recovery adjustment: {', '.join(result['actions_taken'])}")
        
        return result

    def implement_correlation_based_position_limits(self, positions: Dict[str, Dict], market_data: pd.DataFrame, verbose: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate position limits based on asset correlations to avoid concentration in correlated assets.
        
        Args:
            positions: Dictionary of current positions
            market_data: Market data DataFrame with price history
            verbose: Whether to log detailed information
            
        Returns:
            Dict: Position limits for each asset adjusted for correlation
        """
        # Initialize result dictionary
        correlation_adjustments = {}
        
        try:
            # Need at least 2 assets to calculate correlation
            active_assets = [asset for asset, pos in positions.items() if abs(pos.get('size', 0)) > 1e-8]
            
            if len(active_assets) < 2:
                # Not enough assets to calculate correlation-based limits
                for asset in positions.keys():
                    correlation_adjustments[asset] = {
                        'corr_adjustment_factor': 1.0,
                        'effective_weight': positions[asset].get('effective_weight', 0.0),
                        'raw_correlation': 0.0,
                        'cluster_membership': 'none',
                        'diversification_score': 1.0
                    }
                return correlation_adjustments
                
            # Create returns DataFrame for correlation calculation
            returns_data = pd.DataFrame(index=market_data.index)
            
            # Extract returns for each asset
            for asset in active_assets:
                try:
                    # Get price data with flexible column access
                    prices = None
                    
                    # Try different column formats
                    if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'close') in market_data.columns:
                        # MultiIndex format
                        prices = market_data.loc[:, (asset, 'close')]
                        if isinstance(prices, pd.DataFrame):
                            prices = prices.iloc[:, 0]
                    elif f"{asset}_close" in market_data.columns:
                        # Flat format with underscore
                        prices = market_data.loc[:, f"{asset}_close"]
                    elif asset in market_data.columns:
                        # Simple column name
                        prices = market_data.loc[:, asset]
                    else:
                        # Skip if no price data available
                        if verbose:
                            logger.warning(f"No price data found for {asset}, skipping correlation calculation")
                        continue
                        
                    # Calculate log returns with error handling
                    if not prices.empty:
                        if (prices <= 0).any():
                            # Clean up non-positive values
                            prices = prices.replace(0, np.nan).fillna(method='ffill')
                        
                        # Calculate log returns
                        returns_data[asset] = np.log(prices).diff().fillna(0)
                except Exception as e:
                    logger.error(f"Error calculating returns for {asset}: {str(e)}")
                    continue
                    
            # Ensure we have at least 2 columns with data
            if len(returns_data.columns) < 2:
                if verbose:
                    logger.warning("Not enough assets with valid return data for correlation calculation")
                for asset in positions.keys():
                    correlation_adjustments[asset] = {
                        'corr_adjustment_factor': 1.0,
                        'effective_weight': positions[asset].get('effective_weight', 0.0),
                        'raw_correlation': 0.0,
                        'cluster_membership': 'none',
                        'diversification_score': 1.0
                    }
                return correlation_adjustments
                
            # Calculate correlation matrix
            corr_matrix = returns_data.corr().fillna(0)
            
            # Calculate position weights
            total_exposure = sum(abs(pos.get('size', 0) * pos.get('mark_price', pos.get('entry_price', 0))) 
                                for pos in positions.values() if abs(pos.get('size', 0)) > 1e-8)
            
            weights = {}
            for asset in positions.keys():
                try:
                    if abs(positions[asset].get('size', 0)) <= 1e-8:
                        weights[asset] = 0.0
                        continue
                        
                    # Get position value
                    price = positions[asset].get('mark_price', positions[asset].get('entry_price', 0))
                    pos_value = abs(positions[asset]['size'] * price)
                    
                    # Calculate weight
                    weights[asset] = pos_value / max(total_exposure, 1e-8)
                except Exception as e:
                    logger.error(f"Error calculating weight for {asset}: {str(e)}")
                    weights[asset] = 0.0
            
            # Store original weights
            original_weights = weights.copy()
            
            # Identify highly correlated clusters
            clusters = self._identify_correlation_clusters(corr_matrix, threshold=0.7)
            
            # Calculate effective exposure considering correlations
            for asset in positions.keys():
                if asset not in returns_data.columns:
                    correlation_adjustments[asset] = {
                        'corr_adjustment_factor': 1.0,
                        'effective_weight': weights.get(asset, 0.0),
                        'raw_correlation': 0.0,
                        'cluster_membership': 'none',
                        'diversification_score': 1.0
                    }
                    continue
                    
                # Get asset's cluster
                asset_cluster = 'none'
                for i, cluster in enumerate(clusters):
                    if asset in cluster:
                        asset_cluster = f"cluster_{i+1}"
                        break
                
                # Calculate weighted average correlation with existing positions
                weighted_corr = 0.0
                corr_count = 0
                
                for other_asset in active_assets:
                    if other_asset == asset or other_asset not in corr_matrix.columns:
                        continue
                        
                    # Get correlation between assets
                    correlation = corr_matrix.loc[asset, other_asset]
                    
                    # Only consider meaningful correlations
                    if abs(correlation) > 0.3:
                        weighted_corr += correlation * weights.get(other_asset, 0.0)
                        corr_count += 1
                
                # Normalize by number of correlations
                avg_correlation = weighted_corr if corr_count == 0 else weighted_corr / corr_count
                
                # Calculate correlation adjustment factor
                # - High positive correlation -> reduce position limit (factor < 1)
                # - Low correlation -> no adjustment (factor = 1)
                # - Negative correlation -> increase position limit (factor > 1)
                if avg_correlation > 0.7:
                    # Strong positive correlation - reduce limits but less aggressively for crypto
                    corr_factor = 0.7  # Reduce limit by 30% (was 50%)
                elif avg_correlation > 0.5:
                    # Moderate positive correlation - reduce limits slightly for crypto
                    corr_factor = 0.85  # Reduce limit by 15% (was 30%)
                elif avg_correlation > 0.3:
                    # Mild positive correlation - minimal reduction for crypto
                    corr_factor = 0.95  # Reduce limit by only 5% (was 15%)
                elif avg_correlation < -0.3:
                    # Negative correlation - increase limits (diversification benefit)
                    corr_factor = 1.25  # Increase limit by 25% (was 20%)
                else:
                    # Low correlation - no adjustment
                    corr_factor = 1.0
                
                # Calculate diversification score (1 = well diversified, 0 = poor diversification)
                cluster_size = 1
                for cluster in clusters:
                    if asset in cluster:
                        cluster_size = len(cluster)
                        break
                        
                diversification_score = 1.0 - (cluster_size - 1) / max(len(active_assets) - 1, 1)
                
                # Additional adjustment for large clusters
                if cluster_size > 2:
                    # Reduce position limit by cluster size factor
                    # CRYPTO ADJUSTMENT: Less aggressive reduction for crypto markets
                    cluster_factor = 1.0 - (0.05 * (cluster_size - 2))  # Was 0.1 - cut in half
                    cluster_factor = max(0.8, cluster_factor)  # Cap reduction at 20% (was 40%)
                    corr_factor *= cluster_factor
                
                # Store results
                correlation_adjustments[asset] = {
                    'corr_adjustment_factor': corr_factor,
                    'effective_weight': weights.get(asset, 0.0) * corr_factor,  # Adjusted weight
                    'raw_correlation': avg_correlation,
                    'cluster_membership': asset_cluster,
                    'diversification_score': diversification_score
                }
                
            # Log correlation adjustments
            if verbose:
                logger.info(f"Calculated correlation-based position limits. Clusters: {len(clusters)}")
            
            return correlation_adjustments
            
        except Exception as e:
            logger.error(f"Error in correlation-based position limits: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return default values
            for asset in positions.keys():
                correlation_adjustments[asset] = {
                    'corr_adjustment_factor': 1.0,
                    'effective_weight': positions[asset].get('effective_weight', 0.0),
                    'raw_correlation': 0.0,
                    'cluster_membership': 'none',
                    'diversification_score': 1.0
                }
            return correlation_adjustments

    def get_adjusted_position_size_limits(self, asset: str, portfolio_value: float, 
                                         correlation_data: Dict[str, Dict[str, float]] = None, 
                                         verbose: bool = False) -> Dict[str, float]:
        """
        Get position size limits adjusted by correlation data
        
        Args:
            asset: Asset symbol
            portfolio_value: Current portfolio value
            correlation_data: Optional correlation data dictionary
            verbose: Whether to log detailed information
            
        Returns:
            Dict: Adjusted position size limits
        """
        # Get base position size limits
        base_limits = self.get_position_size_limits(asset, portfolio_value, verbose)
        
        # If no correlation data provided, return base limits
        if correlation_data is None or asset not in correlation_data:
            return base_limits
        
        # Get correlation adjustment factor
        corr_factor = correlation_data[asset].get('corr_adjustment_factor', 1.0)
        
        # CRYPTO ADJUSTMENT: Make adjustments less restrictive for crypto
        # Ensure correlation doesn't reduce limits too much (min 70% of original)
        corr_factor = max(0.7, corr_factor)  # Was not limited before
        
        # Adjust maximum position value based on correlation
        adjusted_max_value = base_limits['max_value'] * corr_factor
        
        # Log the adjustment
        if verbose:
            logger.debug(f"Adjusted position limit for {asset}: ${base_limits['max_value']:.2f} -> " +
                       f"${adjusted_max_value:.2f} (corr_factor: {corr_factor:.2f})")
        
        # Return adjusted limits
        return {
            'max_value': adjusted_max_value,
            'min_value': base_limits['min_value'],
            'max_leverage': base_limits['max_leverage'],
            'correlation_adjustment': corr_factor
        }
        
    def _identify_correlation_clusters(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[List[str]]:
        """
        Identify clusters of highly correlated assets
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold for clustering
            
        Returns:
            List[List[str]]: List of asset clusters
        """
        try:
            # Initialize clusters
            clusters = []
            assets = list(corr_matrix.columns)
            
            # Track assets that have been assigned to clusters
            assigned = set()
            
            # Start with unassigned assets
            for asset in assets:
                if asset in assigned:
                    continue
                    
                # Find correlated assets
                correlated = [other for other in assets 
                             if other != asset and abs(corr_matrix.loc[asset, other]) >= threshold]
                
                # If we have correlated assets, form a cluster
                if correlated:
                    # Create a new cluster with the asset and its correlations
                    cluster = [asset] + correlated
                    
                    # Add to clusters and mark as assigned
                    clusters.append(cluster)
                    assigned.update(cluster)
                else:
                    # Asset not strongly correlated with others
                    clusters.append([asset])
                    assigned.add(asset)
            
            # Remove duplicates from clusters (an asset can be in multiple clusters)
            # We want to merge clusters with overlapping assets
            i = 0
            while i < len(clusters):
                j = i + 1
                merge_occurred = False
                
                while j < len(clusters):
                    # Check for overlap
                    if set(clusters[i]).intersection(set(clusters[j])):
                        # Merge clusters
                        clusters[i] = list(set(clusters[i]).union(set(clusters[j])))
                        clusters.pop(j)
                        merge_occurred = True
                    else:
                        j += 1
                
                if not merge_occurred:
                    i += 1
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error identifying correlation clusters: {str(e)}")
            return []

    def calculate_market_impact(self, asset: str, order_size: float, price: float, 
                                market_data: pd.DataFrame = None, 
                                verbose: bool = False) -> Dict[str, float]:
        """
        Calculate expected market impact for a trade using advanced market impact models.
        
        This method implements a sophisticated market impact model that considers:
        1. Asset liquidity (volume)
        2. Order size relative to average daily volume
        3. Asset volatility
        4. Market conditions (normal vs. stressed)
        5. Order type and direction
        
        Args:
            asset: Asset symbol
            order_size: Size of order in units (signed, positive for buy, negative for sell)
            price: Current price of the asset
            market_data: Optional market data for additional context
            verbose: Whether to log detailed information
            
        Returns:
            Dict: Market impact details including:
                - impact_bps: Market impact in basis points
                - impact_usd: Market impact in USD
                - impact_price: Expected execution price after impact
                - temporary_impact: Temporary price impact (recovers)
                - permanent_impact: Permanent price impact (doesn't recover)
                - adv_ratio: Order size as percentage of average daily volume
                - model_used: The impact model used for calculation
        """
        try:
            # Default result with minimal impact
            default_result = {
                'impact_bps': 1.0,  # 1 basis point
                'impact_usd': abs(order_size * price) * 0.0001,  # 1 bps in USD
                'impact_price': price * (1 + 0.0001 * np.sign(order_size)),
                'temporary_impact': 0.0001,
                'permanent_impact': 0.00005,
                'adv_ratio': 0.0,
                'model_used': 'default_minimal'
            }
            
            # Return default if order size is too small
            if abs(order_size) < 1e-8 or price <= 0:
                return default_result
                
            # Calculate order value
            order_value = abs(order_size * price)
            
            # Get asset volatility if available
            volatility = self.asset_volatilities.get(asset, 0.02)  # Default 2% daily vol
            
            # Get market data for volume analysis
            adv = 0.0  # Average daily volume
            adv_value = 0.0  # ADV in value terms
            
            if market_data is not None:
                try:
                    # Get average trading volume for the asset
                    if isinstance(market_data.columns, pd.MultiIndex) and (asset, 'volume') in market_data.columns:
                        volumes = market_data.loc[:, (asset, 'volume')]
                        if isinstance(volumes, pd.DataFrame):
                            volumes = volumes.iloc[:, 0]
                        adv = volumes.mean()
                    elif f"{asset}_volume" in market_data.columns:
                        adv = market_data[f"{asset}_volume"].mean()
                    elif asset in market_data.columns and 'volume' in market_data.columns:
                        adv = market_data['volume'].mean()
                        
                    # Calculate ADV in value terms
                    adv_value = adv * price
                    
                except Exception as e:
                    logger.error(f"Error calculating ADV for {asset}: {str(e)}")
                    
            # If we couldn't get ADV from market data, use a default based on asset type
            if adv <= 0:
                # Default ADVs based on asset tier (just rough estimates)
                default_advs = {
                    'BTCUSDT': 5000.0,    # 5000 BTC
                    'ETHUSDT': 50000.0,   # 50000 ETH
                    'SOLUSDT': 500000.0,  # 500000 SOL
                    'BNBUSDT': 100000.0   # 100000 BNB
                }
                
                # Get default ADV or use generic value
                adv = default_advs.get(asset, 10000.0)
                adv_value = adv * price
            
            # Calculate the ADV ratio
            adv_ratio = order_value / adv_value if adv_value > 0 else 0.1
            
            # Select impact model based on asset and size
            if adv_ratio > 0.1:
                # Large order relative to ADV - use Almgren-Chriss model
                model = 'almgren_chriss'
            elif volatility > 0.05:
                # High volatility asset - use square-root model with volatility adjustment
                model = 'square_root_vol_adjusted'
            else:
                # Default case - use simple square-root model
                model = 'square_root'
            
            # Calculate market impact based on selected model
            impact_bps = 0.0
            temporary_impact = 0.0
            permanent_impact = 0.0
            
            if model == 'almgren_chriss':
                # Almgren-Chriss model parameters
                # These params should be calibrated per asset class ideally
                sigma = volatility
                gamma = 0.1  # Market impact coefficient
                eta = 1.0    # Market impact decay coefficient
                tau = 1.0    # Execution time horizon (1 day)
                
                # Calculate impact components
                # Temporary impact scales with square root of (order size / ADV)
                temporary_impact = gamma * sigma * np.sqrt(adv_ratio / tau)
                
                # Permanent impact scales linearly with order size / ADV
                permanent_impact = eta * sigma * adv_ratio
                
                # Total impact in basis points
                impact_bps = (temporary_impact + permanent_impact) * 10000  # Convert to bps
                
            elif model == 'square_root_vol_adjusted':
                # Square-root model with volatility adjustment
                # Higher volatility = higher impact
                vol_multiplier = min(3.0, max(1.0, volatility / 0.02))  # Scale by vol relative to 2%
                base_impact = 10.0 * np.sqrt(adv_ratio) * vol_multiplier  # 10 bps per sqrt of ADV ratio
                
                # Adjust impact based on order direction and market state
                # We assume selling in high vol has more impact than buying
                direction_multiplier = 1.1 if order_size < 0 and volatility > 0.03 else 1.0
                
                impact_bps = base_impact * direction_multiplier
                
                # Distribute between temporary and permanent
                temporary_impact = impact_bps * 0.7 / 10000  # 70% temporary
                permanent_impact = impact_bps * 0.3 / 10000  # 30% permanent
                
            else:  # square_root model
                # Basic square-root model
                # CRYPTO ADJUSTMENT: Crypto markets are more liquid and can absorb impact better
                impact_bps = 6.0 * np.sqrt(adv_ratio)  # Reduced from 8.0 bps for crypto
                
                # Distribute between temporary and permanent
                temporary_impact = impact_bps * 0.8 / 10000  # 80% temporary
                permanent_impact = impact_bps * 0.2 / 10000  # 20% permanent
            
            # Ensure minimum impact
            impact_bps = max(1.0, impact_bps)  # At least 1 bps
            
            # Apply urgency multiplier for large orders - less aggressive for crypto
            if adv_ratio > 0.1:  # Raised threshold from 0.05 to 0.1 for crypto
                urgency_multiplier = 1.0 + min(0.5, (adv_ratio - 0.1) * 3)  # Less multiplier (was up to 2x)
                impact_bps *= urgency_multiplier
            
            # Convert impact to USD
            impact_usd = order_value * (impact_bps / 10000)
            
            # Calculate expected execution price
            impact_price = price * (1 + (impact_bps / 10000) * np.sign(order_size))
            
            # Prepare result
            result = {
                'impact_bps': impact_bps,
                'impact_usd': impact_usd,
                'impact_price': impact_price,
                'temporary_impact': temporary_impact,
                'permanent_impact': permanent_impact,
                'adv_ratio': adv_ratio,
                'model_used': model
            }
            
            # Log significant impacts
            # if impact_bps > 10:
            #     logger.warning(f"High market impact for {asset}: {impact_bps:.2f} bps " +
            #                  f"({impact_usd:.2f} USD) for order size {order_size:.6f} " +
            #                  f"({adv_ratio:.2%} of ADV)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating market impact for {asset}: {str(e)}")
            return default_result

    def get_optimal_execution_size(self, asset: str, target_size: float, price: float, 
                                  max_impact_bps: float = 50.0, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Calculate the optimal execution size to minimize market impact.
        For large orders, this may suggest breaking the order into smaller chunks.
        
        Args:
            asset: Asset symbol
            target_size: Target position size (can be positive or negative)
            price: Current price of the asset
            max_impact_bps: Maximum acceptable impact in basis points (default 50.0 for crypto, was 30.0)
            market_data: Optional market data
            
        Returns:
            Dict: Execution recommendation including:
                - optimal_size: Recommended size for immediate execution
                - remaining_size: Size to execute later
                - expected_impact: Expected market impact
                - execution_chunks: Suggested execution chunks if splitting is recommended
        """
        try:
            # Default to executing the full size
            result = {
                'optimal_size': target_size,
                'remaining_size': 0.0,
                'expected_impact': {},
                'execution_chunks': [target_size],
                'delay_seconds': 0
            }
            
            # Skip tiny orders
            if abs(target_size) < 1e-6:
                return result
                
            # Calculate market impact for the full order
            impact = self.calculate_market_impact(asset, target_size, price, market_data)
            
            # If impact is acceptable, execute full size
            if impact['impact_bps'] <= max_impact_bps:
                result['expected_impact'] = impact
                return result
                
            # Impact too high, need to split the order
            # Use binary search to find optimal size
            low = 0.0
            high = abs(target_size)
            optimal_size = 0.0
            optimal_impact = None
            
            for _ in range(10):  # 10 iterations should be enough for convergence
                mid = (low + high) / 2
                test_size = mid * np.sign(target_size)
                
                # Calculate impact for this test size
                test_impact = self.calculate_market_impact(asset, test_size, price, market_data)
                
                if test_impact['impact_bps'] <= max_impact_bps:
                    # This size is acceptable, try larger
                    optimal_size = mid
                    optimal_impact = test_impact
                    low = mid
                else:
                    # Impact too high, try smaller
                    high = mid
            
            # Calculate number of chunks needed
            if optimal_size <= 0:
                # Couldn't find any size with acceptable impact, use minimum size
                optimal_size = 0.01 * abs(target_size) * np.sign(target_size)
                optimal_impact = self.calculate_market_impact(asset, optimal_size, price, market_data)
                
            # Remaining size
            remaining_size = target_size - optimal_size
            
            # Calculate execution chunks
            if abs(remaining_size) > 1e-8:
                # Estimate number of chunks
                num_chunks = max(2, int(np.ceil(abs(target_size) / abs(optimal_size))))
                chunk_size = target_size / num_chunks
                
                # Create chunks with delays
                chunks = []
                for i in range(num_chunks):
                    chunks.append(chunk_size)
                
                # Calculate delay between chunks
                # Base delay on volatility and size
                volatility = self.asset_volatilities.get(asset, 0.02)
                base_delay = 60  # 1 minute base delay
                volatility_factor = volatility / 0.02  # Scale by vol relative to 2%
                size_factor = min(10, np.sqrt(impact['adv_ratio'] * 100))
                
                delay_seconds = int(base_delay * volatility_factor * size_factor)
                delay_seconds = max(30, min(3600, delay_seconds))  # Between 30s and 1 hour
                
                result['delay_seconds'] = delay_seconds
                result['execution_chunks'] = chunks
            
            # Update result
            result['optimal_size'] = optimal_size
            result['remaining_size'] = remaining_size
            result['expected_impact'] = optimal_impact
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating optimal execution size for {asset}: {str(e)}")
            # Return default execution plan
            return {
                'optimal_size': target_size,
                'remaining_size': 0.0,
                'expected_impact': self.calculate_market_impact(asset, target_size, price, market_data),
                'execution_chunks': [target_size],
                'delay_seconds': 0
            }