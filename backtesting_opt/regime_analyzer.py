import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enum representing different market regimes."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    NORMAL = "normal"

class RegimePeriod:
    """Class representing a period with a specific market regime."""
    
    def __init__(
        self,
        regime: MarketRegime,
        start_date: datetime, 
        end_date: datetime,
        description: str = "",
        metrics: Dict = None
    ):
        self.regime = regime
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        self.metrics = metrics or {}
        
    def __repr__(self):
        return (f"RegimePeriod({self.regime.value}, "
                f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}, "
                f"{self.description})")

class RegimeAnalyzer:
    """
    Market regime analyzer that identifies different market conditions 
    such as trends, range-bound periods, volatile periods, and crises.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        volatility_threshold: float = 2.0,
        trend_window: int = 50,
        trend_threshold: float = 0.15,
        crisis_threshold: float = -0.15,
        recovery_threshold: float = 0.15,
        min_regime_days: int = 5
    ):
        """
        Initialize the RegimeAnalyzer.
        
        Args:
            volatility_window: Window for volatility calculation
            volatility_threshold: Threshold for high volatility detection (multiplier of average)
            trend_window: Window for trend detection
            trend_threshold: Threshold for trend detection (min slope)
            crisis_threshold: Threshold for crisis detection (max drawdown)
            recovery_threshold: Threshold for recovery detection (min return)
            min_regime_days: Minimum days for a regime to be considered valid
        """
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.crisis_threshold = crisis_threshold
        self.recovery_threshold = recovery_threshold
        self.min_regime_days = min_regime_days
        
    def analyze_market_data(
        self, 
        df: pd.DataFrame,
        price_col: str = 'close',
        asset_level: bool = True,
        use_index_as_date: bool = True
    ) -> Dict[str, List[RegimePeriod]]:
        """
        Analyze market data to identify regimes.
        
        Args:
            df: DataFrame with market data
            price_col: Column name for price data
            asset_level: Whether to analyze at asset level (True) or market level (False)
            use_index_as_date: Whether to use index as date
            
        Returns:
            Dictionary mapping asset names to lists of RegimePeriod objects
        """
        logger.info("Starting market regime analysis")
        
        # Prepare results dictionary
        regime_periods: Dict[str, List[RegimePeriod]] = {}
        
        # Extract dates from index if needed
        if use_index_as_date and isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            # Try to find a date column
            date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]
            if date_cols:
                dates = pd.to_datetime(df[date_cols[0]])
            else:
                # Create synthetic dates
                start_date = datetime.now() - timedelta(days=len(df))
                dates = [start_date + timedelta(days=i) for i in range(len(df))]
                dates = pd.DatetimeIndex(dates)
        
        # Identify if this is MultiIndex data
        is_multi_index = isinstance(df.columns, pd.MultiIndex)
        
        # If analyzing at asset level with MultiIndex
        if asset_level and is_multi_index:
            assets = df.columns.get_level_values(0).unique()
            logger.info(f"Analyzing regimes for {len(assets)} assets")
            
            for asset in assets:
                try:
                    # Get price series for this asset
                    price_data = df.xs(asset, axis=1, level=0)
                    
                    # Make sure price column exists
                    if price_col not in price_data.columns:
                        logger.warning(f"Price column '{price_col}' not found for {asset}, skipping")
                        continue
                        
                    # Get price series
                    market_prices = price_data[price_col]
                    
                    # Analyze regimes
                    asset_regimes = self._detect_regimes(market_prices, dates)
                    regime_periods[asset] = asset_regimes
                    
                    logger.info(f"Identified {len(asset_regimes)} regime periods for {asset}")
                except Exception as e:
                    logger.error(f"Error analyzing regimes for {asset}: {str(e)}")
                    regime_periods[asset] = []
        
        # If analyzing at market level or not MultiIndex
        else:
            # For market level, we use a single analysis for all data
            try:
                if is_multi_index:
                    # For MultiIndex, combine all assets to create a market index
                    market_prices = pd.Series(0.0, index=df.index)
                    assets = df.columns.get_level_values(0).unique()
                    
                    # Sum up price changes across assets
                    for asset in assets:
                        try:
                            # Get price series
                            price_series = df.xs(asset, axis=1, level=0)[price_col]
                            # Normalize by starting price
                            normalized = price_series / price_series.iloc[0]
                            market_prices += normalized
                        except:
                            pass
                            
                    # Normalize by number of assets
                    market_prices /= len(assets)
                else:
                    # For regular index, just use the price column
                    if price_col in df.columns:
                        market_prices = df[price_col]
                    else:
                        # Try to find a suitable price column
                        price_columns = [col for col in df.columns if 'close' in str(col).lower() or 'price' in str(col).lower()]
                        if price_columns:
                            market_prices = df[price_columns[0]]
                        else:
                            raise ValueError("Could not find suitable price column")
                
                # Analyze regimes
                market_regimes = self._detect_regimes(market_prices, dates)
                regime_periods['MARKET'] = market_regimes
                
                logger.info(f"Identified {len(market_regimes)} market-level regime periods")
            except Exception as e:
                logger.error(f"Error analyzing market regimes: {str(e)}")
                regime_periods['MARKET'] = []
                
        return regime_periods
    
    def _detect_regimes(self, prices: pd.Series, dates: pd.DatetimeIndex) -> List[RegimePeriod]:
        """
        Detect regimes in a price series.
        
        Args:
            prices: Series of prices
            dates: DatetimeIndex of dates
            
        Returns:
            List of RegimePeriod objects
        """
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Calculate volatility
        volatility = returns.rolling(window=self.volatility_window).std().fillna(method='bfill')
        avg_volatility = volatility.mean()
        high_volatility = volatility > (avg_volatility * self.volatility_threshold)
        
        # Calculate trends
        # Use simple moving averages for trend detection
        sma_short = market_prices.rolling(window=self.trend_window//5).mean().fillna(method='bfill')
        sma_long = market_prices.rolling(window=self.trend_window).mean().fillna(method='bfill')
        
        # Calculate slope of SMA
        sma_long_slope = (sma_long - sma_long.shift(self.trend_window//2)) / (self.trend_window//2)
        slope_normalized = sma_long_slope / prices
        
        # Detect trend regimes
        uptrend = slope_normalized > self.trend_threshold
        downtrend = slope_normalized < -self.trend_threshold
        
        # Calculate drawdowns for crisis detection
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        
        # Identify crisis periods (severe drawdowns)
        crisis = drawdown < self.crisis_threshold
        
        # Calculate recovery periods
        recovery_returns = returns.rolling(window=self.trend_window//2).sum()
        recovery = recovery_returns > self.recovery_threshold
        
        # Combine signals into regime states
        regime_states = pd.Series('normal', index=prices.index)
        regime_states[uptrend & ~high_volatility] = 'trending_up'
        regime_states[downtrend & ~high_volatility] = 'trending_down'
        regime_states[~uptrend & ~downtrend & ~high_volatility] = 'range_bound'
        regime_states[high_volatility & ~crisis] = 'volatile'
        regime_states[crisis] = 'crisis'
        regime_states[recovery & ~crisis] = 'recovery'
        
        # Convert to market regimes enum
        regime_enum = regime_states.apply(lambda x: getattr(MarketRegime, x.upper()))
        
        # Find regime transitions
        transitions = (regime_states != regime_states.shift(1)).astype(int)
        transition_indices = transitions[transitions == 1].index.tolist()
        
        # Add start and end indices
        if len(transition_indices) == 0 or transition_indices[0] != prices.index[0]:
            transition_indices.insert(0, prices.index[0])
        if transition_indices[-1] != prices.index[-1]:
            transition_indices.append(prices.index[-1])
            
        # Create regime periods
        regime_periods = []
        
        for i in range(len(transition_indices) - 1):
            start_idx = transition_indices[i]
            end_idx = transition_indices[i+1]
            
            if isinstance(start_idx, pd.Timestamp):
                start_date = start_idx
            else:
                start_date = dates[start_idx]
                
            if isinstance(end_idx, pd.Timestamp):
                end_date = end_idx
            else:
                end_date = dates[end_idx]
                
            # Skip very short regimes
            days_in_regime = (end_date - start_date).days
            if days_in_regime < self.min_regime_days:
                continue
                
            # Get the regime for this period
            regime = regime_enum[start_idx]
            
            # Calculate metrics for this regime
            window_prices = prices.loc[start_idx:end_idx]
            window_returns = returns.loc[start_idx:end_idx]
            
            metrics = {
                'duration_days': days_in_regime,
                'price_change_pct': (window_prices.iloc[-1] / window_prices.iloc[0] - 1) * 100,
                'volatility': window_returns.std() * np.sqrt(252),  # Annualized
                'max_drawdown': window_returns.cumsum().min() * 100 if len(window_returns) > 0 else 0,
                'sharpe': (window_returns.mean() / window_returns.std()) * np.sqrt(252) if window_returns.std() > 0 else 0
            }
            
            # Create description
            description = f"{regime.value.replace('_', ' ').title()} regime lasting {days_in_regime} days"
            
            # Create regime period
            period = RegimePeriod(
                regime=regime,
                start_date=start_date,
                end_date=end_date,
                description=description,
                metrics=metrics
            )
            
            regime_periods.append(period)
            
        return regime_periods
    
    def get_current_regime(self, prices: pd.Series, window: int = 30) -> MarketRegime:
        """
        Determine the current market regime based on recent price data.
        
        Args:
            prices: Series of recent prices
            window: Window size for analysis
            
        Returns:
            MarketRegime enum indicating current regime
        """
        # Ensure we have enough data
        if len(prices) < max(self.volatility_window, self.trend_window):
            return MarketRegime.NORMAL
            
        # Get most recent data
        recent_prices = prices.iloc[-window:]
        
        # Calculate returns
        returns = recent_prices.pct_change().fillna(0)
        
        # Calculate volatility
        volatility = returns.rolling(window=min(self.volatility_window, len(returns))).std().fillna(0)
        recent_volatility = volatility.iloc[-1]
        avg_volatility = volatility.mean()
        
        # Detect high volatility
        high_vol = recent_volatility > (avg_volatility * self.volatility_threshold)
        
        # Calculate trend
        if len(recent_prices) >= self.trend_window:
            sma_short = recent_prices.rolling(window=self.trend_window//5).mean().fillna(method='ffill')
            sma_long = recent_prices.rolling(window=self.trend_window).mean().fillna(method='ffill')
            
            # Calculate slope
            recent_slope = (sma_long.iloc[-1] - sma_long.iloc[-self.trend_window//2]) / (self.trend_window//2)
            slope_normalized = recent_slope / recent_prices.iloc[-1]
            
            # Detect trend
            uptrend = slope_normalized > self.trend_threshold
            downtrend = slope_normalized < -self.trend_threshold
        else:
            # Not enough data for proper trend detection
            uptrend = False
            downtrend = False
            
        # Calculate drawdown
        peak = recent_prices.max()
        current_drawdown = (recent_prices.iloc[-1] - peak) / peak
        
        # Detect crisis
        crisis = current_drawdown < self.crisis_threshold
        
        # Calculate recovery
        if len(returns) >= self.trend_window//2:
            recovery_return = returns.iloc[-self.trend_window//2:].sum()
            recovery = recovery_return > self.recovery_threshold
        else:
            recovery = False
            
        # Determine regime
        if crisis:
            return MarketRegime.CRISIS
        elif recovery:
            return MarketRegime.RECOVERY
        elif uptrend and not high_vol:
            return MarketRegime.TRENDING_UP
        elif downtrend and not high_vol:
            return MarketRegime.TRENDING_DOWN
        elif high_vol:
            return MarketRegime.VOLATILE
        elif not uptrend and not downtrend and not high_vol:
            return MarketRegime.RANGE_BOUND
        else:
            return MarketRegime.NORMAL 