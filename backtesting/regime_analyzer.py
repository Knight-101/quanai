"""
Market Regime Analyzer Module

This module provides tools to analyze and classify different market regimes
in financial time series data, which is crucial for robust backtesting.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enum defining different market regimes"""
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CRISIS = "crisis"


@dataclass
class RegimePeriod:
    """Class representing a period of a specific market regime"""
    regime: MarketRegime
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    description: str = ""
    metrics: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class RegimeAnalyzer:
    """
    Analyzes market data to identify different market regimes.
    
    This class provides tools to classify market periods into different regimes
    such as bull markets, bear markets, sideways markets, and periods of high volatility.
    These classifications can be used for targeted backtesting analysis.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        high_vol_threshold: float = 0.8,
        low_vol_threshold: float = 0.2,
        trend_threshold: float = 0.6,
        crisis_threshold: float = 0.95,
        lookback_window: int = 100
    ):
        """
        Initialize the RegimeAnalyzer.
        
        Args:
            volatility_window: Window size for volatility calculations
            trend_window: Window size for trend calculations
            high_vol_threshold: Percentile threshold for high volatility regime
            low_vol_threshold: Percentile threshold for low volatility regime
            trend_threshold: Threshold for determining trending markets
            crisis_threshold: Percentile threshold for crisis regime
            lookback_window: Window for rolling calculations
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.trend_threshold = trend_threshold
        self.crisis_threshold = crisis_threshold
        self.lookback_window = lookback_window
        
        # Storage for identified regimes
        self.regimes: List[RegimePeriod] = []
        
    def analyze_market_data(
        self, 
        df: pd.DataFrame,
        price_col: str = 'close',
        asset_level: bool = True
    ) -> Dict[str, List[RegimePeriod]]:
        """
        Analyze market data to identify different market regimes.
        
        Args:
            df: DataFrame with market data, with MultiIndex columns (asset, feature)
            price_col: Column name for price data
            asset_level: Whether to analyze regimes per asset
            
        Returns:
            Dictionary mapping asset names to lists of regime periods
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot analyze regimes")
            return {}
            
        regimes_by_asset = {}
        
        # If using MultiIndex columns with (asset, feature) structure
        if isinstance(df.columns, pd.MultiIndex):
            assets = df.columns.get_level_values(0).unique()
            
            for asset in assets:
                if asset_level:
                    # Extract price series for this asset
                    try:
                        price_series = df[(asset, price_col)]
                        asset_regimes = self._identify_regimes(price_series, asset)
                        regimes_by_asset[asset] = asset_regimes
                    except Exception as e:
                        logger.error(f"Error analyzing regimes for {asset}: {str(e)}")
                else:
                    # Convert all asset prices to a single DataFrame for overall market analysis
                    try:
                        price_df = pd.DataFrame()
                        for asset in assets:
                            price_df[asset] = df[(asset, price_col)]
                        
                        # Analyze the combined data
                        overall_regimes = self._identify_regimes_overall(price_df)
                        regimes_by_asset['MARKET'] = overall_regimes
                        return regimes_by_asset  # Early return since we're only doing market level
                    except Exception as e:
                        logger.error(f"Error analyzing market-level regimes: {str(e)}")
        else:
            # Single asset case
            try:
                price_series = df[price_col]
                asset_regimes = self._identify_regimes(price_series, 'ASSET')
                regimes_by_asset['ASSET'] = asset_regimes
            except Exception as e:
                logger.error(f"Error analyzing single-asset regimes: {str(e)}")
                
        return regimes_by_asset
        
    def _identify_regimes(self, price_series: pd.Series, asset_name: str) -> List[RegimePeriod]:
        """
        Identify different regimes in a single price series.
        
        Args:
            price_series: Time series of prices
            asset_name: Name of the asset
            
        Returns:
            List of regime periods
        """
        if len(price_series) < self.lookback_window:
            logger.warning(f"Price series too short for {asset_name}, needs {self.lookback_window} points")
            return []
            
        # Calculate returns
        returns = price_series.pct_change().fillna(0)
        
        # Calculate volatility
        volatility = returns.rolling(window=self.volatility_window).std().fillna(0)
        
        # Calculate trend indicators
        sma_short = price_series.rolling(window=self.trend_window//5).mean().fillna(method='bfill')
        sma_long = price_series.rolling(window=self.trend_window).mean().fillna(method='bfill')
        
        # Identify trend direction
        trend_direction = np.where(sma_short > sma_long, 1, np.where(sma_short < sma_long, -1, 0))
        
        # Calculate drawdowns
        rolling_max = price_series.rolling(window=self.lookback_window).max().fillna(price_series.iloc[0])
        drawdown = (price_series - rolling_max) / rolling_max
        
        # Calculate regime indicators
        vol_quantile = volatility.rank(pct=True)
        
        # Determine regimes for each point
        high_vol_mask = vol_quantile > self.high_vol_threshold
        low_vol_mask = vol_quantile < self.low_vol_threshold
        crisis_mask = drawdown < -0.2  # More than 20% drawdown is crisis
        bull_mask = (trend_direction == 1) & ~high_vol_mask & ~crisis_mask
        bear_mask = (trend_direction == -1) & ~crisis_mask
        sideways_mask = (trend_direction == 0) & ~high_vol_mask & ~low_vol_mask & ~crisis_mask
        
        # Combine into regime series
        regime_series = pd.Series(index=price_series.index, dtype='object')
        regime_series[bull_mask] = MarketRegime.BULL
        regime_series[bear_mask] = MarketRegime.BEAR
        regime_series[sideways_mask] = MarketRegime.SIDEWAYS
        regime_series[high_vol_mask & ~crisis_mask] = MarketRegime.HIGH_VOL
        regime_series[low_vol_mask] = MarketRegime.LOW_VOL
        regime_series[crisis_mask] = MarketRegime.CRISIS
        regime_series[(trend_direction == 1) & high_vol_mask & ~crisis_mask] = MarketRegime.TRENDING_UP
        regime_series[(trend_direction == -1) & high_vol_mask & ~crisis_mask] = MarketRegime.TRENDING_DOWN
        
        # Fill any remaining NaN values
        regime_series.fillna(MarketRegime.SIDEWAYS, inplace=True)
        
        # Convert to periods
        regime_periods = self._convert_to_periods(regime_series, asset_name)
        
        return regime_periods
    
    def _identify_regimes_overall(self, price_df: pd.DataFrame) -> List[RegimePeriod]:
        """
        Identify market regimes based on multiple assets.
        
        Args:
            price_df: DataFrame with price series for multiple assets
            
        Returns:
            List of regime periods
        """
        # Calculate returns for all assets
        returns_df = price_df.pct_change().fillna(0)
        
        # Calculate cross-asset correlation
        correlation = returns_df.rolling(window=self.volatility_window).corr()
        
        # Extract mean pairwise correlation by day
        if len(returns_df.columns) >= 2:
            daily_corr = []
            for date in returns_df.index[self.volatility_window:]:
                date_corr = correlation.loc[date]
                # Get lower triangle of correlation matrix excluding diagonal
                mask = np.tril(np.ones(date_corr.shape), k=-1).astype(bool)
                date_corr_values = date_corr.where(mask).stack().values
                daily_corr.append(np.mean(date_corr_values))
            
            mean_corr = pd.Series(daily_corr, index=returns_df.index[self.volatility_window:])
            # Backfill for the initial window
            mean_corr = pd.Series(mean_corr.iloc[0], index=returns_df.index[:self.volatility_window]).append(mean_corr)
        else:
            # For single asset, use autocorrelation
            mean_corr = returns_df.iloc[:,0].rolling(window=self.volatility_window).apply(
                lambda x: x.autocorr(lag=1)).fillna(0)
        
        # Calculate cross-asset volatility (market volatility)
        market_returns = returns_df.mean(axis=1)
        market_vol = market_returns.rolling(window=self.volatility_window).std().fillna(0)
        
        # Calculate market trend
        market_prices = price_df.mean(axis=1)
        sma_short = market_prices.rolling(window=self.trend_window//5).mean().fillna(method='bfill')
        sma_long = market_prices.rolling(window=self.trend_window).mean().fillna(method='bfill')
        
        # Identify trend direction
        trend_direction = np.where(sma_short > sma_long, 1, np.where(sma_short < sma_long, -1, 0))
        
        # Calculate market drawdowns
        rolling_max = market_prices.rolling(window=self.lookback_window).max().fillna(market_prices.iloc[0])
        drawdown = (market_prices - rolling_max) / rolling_max
        
        # Determine regimes using more complex logic combining correlation and volatility
        vol_quantile = market_vol.rank(pct=True)
        corr_quantile = mean_corr.rank(pct=True)
        
        # Define regime masks
        crisis_mask = (drawdown < -0.15) | ((vol_quantile > 0.9) & (corr_quantile > 0.9))
        high_vol_mask = (vol_quantile > self.high_vol_threshold) & ~crisis_mask
        low_vol_mask = vol_quantile < self.low_vol_threshold
        bull_mask = (trend_direction == 1) & ~high_vol_mask & ~crisis_mask
        bear_mask = (trend_direction == -1) & ~crisis_mask
        sideways_mask = (trend_direction == 0) & ~high_vol_mask & ~low_vol_mask & ~crisis_mask
        
        # Combine into regime series
        regime_series = pd.Series(index=market_prices.index, dtype='object')
        regime_series[bull_mask] = MarketRegime.BULL
        regime_series[bear_mask] = MarketRegime.BEAR
        regime_series[sideways_mask] = MarketRegime.SIDEWAYS
        regime_series[high_vol_mask] = MarketRegime.HIGH_VOL
        regime_series[low_vol_mask] = MarketRegime.LOW_VOL
        regime_series[crisis_mask] = MarketRegime.CRISIS
        regime_series[(trend_direction == 1) & high_vol_mask] = MarketRegime.TRENDING_UP
        regime_series[(trend_direction == -1) & high_vol_mask] = MarketRegime.TRENDING_DOWN
        
        # Fill any remaining NaN values
        regime_series.fillna(MarketRegime.SIDEWAYS, inplace=True)
        
        # Convert to periods
        regime_periods = self._convert_to_periods(regime_series, "MARKET")
        
        return regime_periods
    
    def _convert_to_periods(self, regime_series: pd.Series, asset_name: str) -> List[RegimePeriod]:
        """
        Convert a series of point-in-time regimes to continuous periods.
        
        Args:
            regime_series: Series with regime for each timestamp
            asset_name: Name of the asset
            
        Returns:
            List of regime periods
        """
        if regime_series.empty:
            return []
            
        periods = []
        current_regime = regime_series.iloc[0]
        start_date = regime_series.index[0]
        
        for date, regime in regime_series.items():
            # If regime changes, end the previous period and start a new one
            if regime != current_regime:
                # Create period for the regime that just ended
                period = RegimePeriod(
                    regime=current_regime,
                    start_date=start_date,
                    end_date=date,
                    description=f"{asset_name} {current_regime.value} from {start_date.date()} to {date.date()}"
                )
                periods.append(period)
                
                # Start new period
                current_regime = regime
                start_date = date
        
        # Add the final period
        end_date = regime_series.index[-1]
        period = RegimePeriod(
            regime=current_regime,
            start_date=start_date,
            end_date=end_date,
            description=f"{asset_name} {current_regime.value} from {start_date.date()} to {end_date.date()}"
        )
        periods.append(period)
        
        return periods
        
    def get_regime_at_time(self, timestamp: pd.Timestamp, asset: str = "MARKET") -> Optional[MarketRegime]:
        """
        Get the market regime at a specific time.
        
        Args:
            timestamp: The timestamp to check
            asset: The asset to check regimes for
            
        Returns:
            The market regime at the specified time, or None if not found
        """
        for period in self.regimes:
            if period.start_date <= timestamp <= period.end_date:
                return period.regime
        return None
        
    def filter_data_by_regime(
        self, 
        data: pd.DataFrame, 
        regime: MarketRegime,
        asset: str = "MARKET"
    ) -> pd.DataFrame:
        """
        Filter data to only include periods of a specific regime.
        
        Args:
            data: DataFrame with market data
            regime: The regime to filter for
            asset: The asset to use for regime filtering
            
        Returns:
            Filtered DataFrame containing only the specified regime
        """
        if data.empty:
            return data
            
        # Find all periods of the specified regime
        matching_periods = [p for p in self.regimes if p.regime == regime]
        
        if not matching_periods:
            logger.warning(f"No periods found for regime {regime}")
            return pd.DataFrame()
            
        # Create a mask for all matching periods
        mask = pd.Series(False, index=data.index)
        for period in matching_periods:
            period_mask = (data.index >= period.start_date) & (data.index <= period.end_date)
            mask = mask | period_mask
            
        # Apply the mask to the data
        filtered_data = data.loc[mask]
        logger.info(f"Filtered data for {regime}: {len(filtered_data)} rows from {len(data)} original rows")
        
        return filtered_data 