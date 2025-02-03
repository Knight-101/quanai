import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    max_drawdown: float = 0.2
    var_limit: float = 0.05
    max_leverage: float = 20.0
    position_concentration: float = 0.4
    correlation_limit: float = 0.7
    liquidity_ratio: float = 0.1
    stress_multiplier: float = 2.0

class InstitutionalRiskEngine:
    def __init__(
        self,
        risk_limits: RiskLimits = RiskLimits(),
        lookback_window: int = 100,
        confidence_level: float = 0.95,
        stress_test_scenarios: List[Dict] = None
    ):
        self.risk_limits = risk_limits
        self.lookback_window = lookback_window
        self.confidence_level = confidence_level
        self.stress_test_scenarios = stress_test_scenarios or self._default_stress_scenarios()
        
        # Historical metrics
        self.historical_var = []
        self.historical_drawdowns = []
        self.historical_leverage = []
        self.position_history = []
        
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
        
    def _calculate_position_metrics(
        self,
        positions: Dict[str, Dict],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """Calculate position-based risk metrics"""
        try:
            # Calculate position sizes and exposures
            exposures = {}
            total_exposure = 0
            
            # Get the latest timestamp
            latest_timestamp = market_data.index[-1]
            
            for asset, position in positions.items():
                try:
                    # Get close price using proper MultiIndex access
                    price = market_data.loc[latest_timestamp, (asset, 'close')]
                    if isinstance(price, (pd.Series, pd.DataFrame)):
                        price = float(price.iloc[0])
                    else:
                        price = float(price)
                    
                    # Calculate exposure
                    exposure = abs(position['size'] * price)
                    exposures[asset] = exposure
                    total_exposure += exposure
                except Exception as e:
                    logger.error(f"Error calculating exposure for {asset}: {str(e)}")
                    exposures[asset] = 0
                    continue
            
            # Position concentration with safety checks
            max_concentration = max(exposures.values()) / (portfolio_value + 1e-8) if exposures else 0
            
            # Leverage utilization with safety checks
            current_leverage = total_exposure / (portfolio_value + 1e-8)
            
            # Calculate correlation risk using returns data
            try:
                # Create returns DataFrame with proper index
                returns_data = pd.DataFrame(index=market_data.index)
                
                # Calculate returns for each asset
                for asset in positions.keys():
                    if positions[asset]['size'] != 0:
                        prices = market_data.loc[:, (asset, 'close')]
                        if isinstance(prices, pd.DataFrame):
                            prices = prices.iloc[:, 0]
                        returns_data[asset] = np.log(prices).diff()
                
                # Calculate correlation risk only if we have enough data
                if len(returns_data.columns) >= 2:
                    corr_matrix = returns_data.corr().fillna(0)
                    weights = np.array([
                        abs(positions[asset]['size']) / (sum(abs(p['size']) for p in positions.values()) + 1e-8)
                        for asset in returns_data.columns
                    ])
                    correlation_risk = float(weights.T @ corr_matrix @ weights)
                else:
                    correlation_risk = 0.0
            except Exception as e:
                logger.error(f"Error calculating correlation risk: {str(e)}")
                correlation_risk = 0.0
            
            return {
                'total_exposure': total_exposure,
                'leverage_utilization': current_leverage,
                'max_concentration': max_concentration,
                'correlation_risk': correlation_risk,
                'num_positions': len([p for p in positions.values() if p['size'] != 0])
            }
        except Exception as e:
            logger.error(f"Error in _calculate_position_metrics: {str(e)}")
            return {
                'total_exposure': 0,
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
                        # Get close prices using proper MultiIndex access
                        prices = market_data.loc[:, (asset, 'close')]
                        if isinstance(prices, pd.DataFrame):
                            prices = prices.iloc[:, 0]
                        
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
            es = -np.mean(recent_returns[recent_returns <= -var])
            
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
                        # Get volume using proper MultiIndex access
                        volume = market_data.loc[:, (asset, 'volume')]
                        if isinstance(volume, pd.DataFrame):
                            volume = volume.iloc[:, 0]
                            
                        # Calculate average daily volume (5-day moving average)
                        adv = volume.rolling(window=5, min_periods=1).mean().iloc[-1]
                        
                        # Get close price
                        close = market_data.loc[latest_timestamp, (asset, 'close')]
                        if isinstance(close, (pd.Series, pd.DataFrame)):
                            close = float(close.iloc[0])
                        else:
                            close = float(close)
                        
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
        self.historical_leverage.append(risk_metrics['leverage_utilization'])
        
    def check_risk_limits(self, risk_metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if current risk metrics exceed defined limits"""
        violations = []
        
        # Check drawdown
        if risk_metrics['max_drawdown'] > self.risk_limits.max_drawdown:
            violations.append(f"Max drawdown ({risk_metrics['max_drawdown']:.2%}) exceeds limit ({self.risk_limits.max_drawdown:.2%})")
            
        # Check VaR
        if risk_metrics['var'] > self.risk_limits.var_limit:
            violations.append(f"VaR ({risk_metrics['var']:.2%}) exceeds limit ({self.risk_limits.var_limit:.2%})")
            
        # Check leverage
        if risk_metrics['leverage_utilization'] > self.risk_limits.max_leverage:
            violations.append(f"Leverage ({risk_metrics['leverage_utilization']:.2f}x) exceeds limit ({self.risk_limits.max_leverage:.2f}x)")
            
        # Check concentration
        if risk_metrics['max_concentration'] > self.risk_limits.position_concentration:
            violations.append(f"Position concentration ({risk_metrics['max_concentration']:.2%}) exceeds limit ({self.risk_limits.position_concentration:.2%})")
            
        # Check correlation
        if risk_metrics['correlation_risk'] > self.risk_limits.correlation_limit:
            violations.append(f"Correlation risk ({risk_metrics['correlation_risk']:.2f}) exceeds limit ({self.risk_limits.correlation_limit:.2f})")
            
        # Check liquidity
        if risk_metrics['max_adv_ratio'] > self.risk_limits.liquidity_ratio:
            violations.append(f"ADV ratio ({risk_metrics['max_adv_ratio']:.2%}) exceeds limit ({self.risk_limits.liquidity_ratio:.2%})")
            
        return len(violations) == 0, violations
        
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