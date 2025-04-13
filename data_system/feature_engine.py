import pandas as pd
import numpy as np
from scipy.stats import norm
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
from ta.trend import (
    EMAIndicator, MACD, ADXIndicator
)
from ta.momentum import (
    RSIIndicator, ROCIndicator, WilliamsRIndicator,
    StochasticOscillator
)
from ta.volatility import (
    BollingerBands, AverageTrueRange
)
from ta.volume import (
    OnBalanceVolumeIndicator, AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator
)
from hmmlearn import hmm
import traceback

logger = logging.getLogger(__name__)

class DerivativesFeatureEngine:
    def __init__(self, 
                 volatility_window=10080, 
                 n_components=5,
                 feature_selection_threshold=0.01):
        self.volatility_window = volatility_window
        self.n_components = n_components
        self.feature_selection_threshold = feature_selection_threshold
        self.pca = PCA(n_components=n_components)
        self.std_scaler = StandardScaler()
        # Add parameters for market regime detection
        self.adx_period = 14
        self.hurst_period = 100
        self.regime_smoothing = 10
        
        self.scaler = StandardScaler()
        self._setup_neural_features()
        self.selected_features = None
        
    def _setup_neural_features(self):
        """Setup neural network for feature extraction"""
        self.input_size = 14  # Number of base features
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16)
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering pipeline"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided to transform")
                return pd.DataFrame()
            
            # Special handling for pipe-separated columns (format "BTCUSDT|close")
            if not isinstance(df.columns, pd.MultiIndex) and any('|' in col for col in df.columns):
                logger.info("Detected pipe-separated columns, converting to MultiIndex")
                # Create a new DataFrame with proper MultiIndex columns
                new_df = pd.DataFrame(index=df.index)
                for col in df.columns:
                    if '|' in col:
                        asset, feature = col.split('|', 1)
                        new_df[(asset, feature)] = df[col]
                    else:
                        # Handle any columns without pipe separator
                        new_df[('unknown', col)] = df[col]
                
                # Convert to MultiIndex
                new_df.columns = pd.MultiIndex.from_tuples(new_df.columns, names=['asset', 'feature'])
                df = new_df
                logger.info(f"Converted to MultiIndex with shape {df.shape}")
            
            # Ensure MultiIndex columns with proper names
            if not isinstance(df.columns, pd.MultiIndex):
                logger.warning("Input DataFrame columns are not MultiIndex, which may cause issues")
                logger.info(f"Columns: {list(df.columns)}")
            else:
                logger.info(f"Column structure: {df.columns.names}")
                logger.info(f"Available assets: {df.columns.get_level_values(0).unique()}")
            
            # Get the level name for the asset index (first level)
            asset_level = 0
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.names[0] is not None:
                    asset_level = df.columns.names[0]
            
            features = {}
            
            try:
                # Get unique assets for processing
                if isinstance(df.columns, pd.MultiIndex):
                    assets = df.columns.get_level_values(0).unique()
                    logger.info(f"Processing features for assets: {list(assets)}")
                else:
                    # For non-MultiIndex, assume single asset
                    logger.warning("Single asset DataFrame detected with flat columns")
                    assets = ["asset"]
                    temp_df = df.copy()
                    temp_df.columns = pd.MultiIndex.from_product([["asset"], temp_df.columns], names=['asset', 'feature'])
                    df = temp_df
                    
                for asset in assets:
                    try:
                        # Get data for this asset
                        asset_df = df.xs(asset, axis=1, level=0) if isinstance(df.columns, pd.MultiIndex) else df
                        
                        # Fix: Check for NaN values in the DataFrame itself, not in the columns
                        # Ensure we're working with actual values
                        if asset_df.values.size > 0:
                            nan_check = pd.isna(asset_df.values).any()
                            if nan_check:
                                logger.warning(f"NaN values detected in {asset} data, applying fillna")
                                asset_df = asset_df.ffill().fillna(0)
                        
                        # Process technical indicators
                        tech_indicators = self._compute_technical_indicators(asset_df)
                        features[asset] = tech_indicators
                        
                        # NEW LOG: Log indicators for this asset
                        logger.info(f"Technical indicators for {asset}: {list(tech_indicators.keys())}")
                        
                        # Add volume/price features if basic features are present
                        if isinstance(features[asset], dict) and 'close' in features[asset]:
                            # Add volatility surface features
                            vol_features = self._add_vol_surface_features(asset_df)
                            features[asset].update(vol_features)
                            
                            # Add flow features
                            flow_features = self._add_flow_features(asset_df)
                            features[asset].update(flow_features)
                            
                            # Add sentiment features
                            sentiment_features = self._add_market_sentiment(asset_df)
                            features[asset].update(sentiment_features)
                    except Exception as e:
                        logger.error(f"Error processing asset {asset}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                    
                # NEW LOG: Log all features before combining
                logger.info(f"Features before combining: {[{k: list(v.keys()) for k, v in features.items()}]}")
                
                # Handle cross-sectional features if multiple assets
                if len(assets) > 1:
                    for asset in assets:
                        try:
                            # Skip if asset failed in earlier processing
                            if asset not in features:
                                continue
                                
                            # Add cross-asset features
                            cross_features = self._add_cross_sectional_features(df, asset)
                            if cross_features:
                                features[asset].update(cross_features)
                                
                            # Add intermarket correlations
                            corr_features = self._add_intermarket_correlations(df, asset)
                            if corr_features:
                                features[asset].update(corr_features)
                        except Exception as e:
                            logger.error(f"Error computing cross-sectional features for {asset}: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"Error in feature processing: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Combine all features
            result = self._combine_features(features)
            
            # NEW LOG: Log the structure of the combined result
            if isinstance(result, pd.DataFrame):
                logger.info(f"Combined result columns: {list(result.columns)}")
            else:
                logger.info(f"Combined result type: {type(result)}")
            
            logger.info(f"Transformed data shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def _compute_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Compute various technical indicators for each asset"""
        try:
            # Log available columns for debugging
            logger.info(f"DataFrame columns type: {type(df.columns)}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            result = {'asset': df.iloc[:, 0].name if hasattr(df.iloc[:, 0], 'name') else 'unknown'}
            
            # Make sure we have required columns
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                logger.warning(f"Missing required columns for TA calculations. Available: {df.columns}")
                return result
            
            # Ensure all data is numeric and fill NaN values to avoid calculation issues
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].ffill().bfill().fillna(0)
            
            # Avoid zero values in close price to prevent division by zero
            df['close'] = df['close'].replace(0, np.nan).ffill().fillna(1e-8)
            
            # Calculate returns
            result['returns_1d'] = df['close'].pct_change(1).fillna(0)
            result['returns_5d'] = df['close'].pct_change(5).fillna(0)
            result['returns_10d'] = df['close'].pct_change(10).fillna(0)
            
            # Calculate volatility with proper handling of NaN/zero values
            returns = df['close'].pct_change().fillna(0)
            result['volatility_5d'] = returns.rolling(5, min_periods=2).std().fillna(0)
            result['volatility_10d'] = returns.rolling(10, min_periods=3).std().fillna(0)
            result['volatility_20d'] = returns.rolling(20, min_periods=5).std().fillna(0)
            
            # Calculate RSI
            try:
                rsi = RSIIndicator(df['close'], window=14)
                result['rsi_14'] = rsi.rsi().fillna(50)  # Neutral value for NaN
            except Exception as e:
                logger.warning(f"Error calculating RSI: {e}")
                result['rsi_14'] = pd.Series(50, index=df.index)  # Default to neutral
            
            # Calculate MACD
            try:
                macd_indicator = MACD(df['close'])
                result['macd'] = macd_indicator.macd().fillna(0)
                result['macd_signal'] = macd_indicator.macd_signal().fillna(0)
                result['macd_diff'] = macd_indicator.macd_diff().fillna(0)
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                # Create default values
                result['macd'] = pd.Series(0, index=df.index)
                result['macd_signal'] = pd.Series(0, index=df.index)
                result['macd_diff'] = pd.Series(0, index=df.index)
            
            # Calculate Bollinger Bands
            try:
                bb = BollingerBands(df['close'])
                result['bb_upper'] = bb.bollinger_hband().fillna(df['close'] * 1.02)  # Default: 2% above price
                result['bb_middle'] = bb.bollinger_mavg().fillna(df['close'])         # Default: current price
                result['bb_lower'] = bb.bollinger_lband().fillna(df['close'] * 0.98)  # Default: 2% below price
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {e}")
                # Create default bands
                result['bb_upper'] = df['close'] * 1.02
                result['bb_middle'] = df['close']
                result['bb_lower'] = df['close'] * 0.98
            
            # Calculate ATR
            try:
                atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
                result['atr_14'] = atr.average_true_range().ffill().fillna(df['close'] * 0.02)
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                result['atr_14'] = df['close'] * 0.02  # Default: 2% of price
            
            # Calculate ADX
            try:
                adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
                result['adx_14'] = adx.adx().fillna(20)  # Default: moderate trend strength
            except Exception as e:
                logger.warning(f"Error calculating ADX: {e}")
                result['adx_14'] = pd.Series(20, index=df.index)
            
            # Calculate CCI (simple implementation rather than using talib to avoid dependency)
            try:
                tp = (df['high'] + df['low'] + df['close']) / 3
                ma20 = tp.rolling(window=20, min_periods=5).mean()
                # Avoid division by zero with small constant
                mean_dev = tp.rolling(window=20, min_periods=5).apply(
                    lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
                ).fillna(0.0001)
                # Where mean_dev is too small, use a minimum value
                mean_dev = mean_dev.replace(0, 0.0001)
                result['cci_14'] = ((tp - ma20) / (0.015 * mean_dev)).fillna(0)
            except Exception as e:
                logger.warning(f"Error calculating CCI: {e}")
                result['cci_14'] = pd.Series(0, index=df.index)
            
            # Simplified Hurst exponent to avoid numerical issues
            try:
                # We'll calculate a simplified proxy for trend strength instead of actual Hurst
                # Use ratio of long-term to short-term volatility as a proxy
                short_vol = returns.rolling(10, min_periods=3).std().fillna(0.0001)
                long_vol = returns.rolling(40, min_periods=10).std().fillna(0.0001)
                # Avoid division by zero
                short_vol = short_vol.replace(0, 0.0001)
                # Values > 1 suggest trend (like high Hurst), < 1 suggest mean reversion (low Hurst)
                vol_ratio = (long_vol / short_vol).fillna(1.0)
                # Map to 0-1 range like Hurst
                result['hurst_exponent'] = (0.5 + 0.5 * np.tanh(vol_ratio - 1)).fillna(0.5)
            except Exception as e:
                logger.warning(f"Error calculating Hurst proxy: {e}")
                result['hurst_exponent'] = pd.Series(0.5, index=df.index)  # Default: random walk
            
            # Market regime (simplified)
            try:
                # Base regime on ADX and direction
                adx_series = result['adx_14'] 
                
                # 0 = ranging, 1 = uptrend, 2 = downtrend
                result['market_regime'] = pd.Series(0, index=df.index)  # Default to ranging
                
                # Uptrend: ADX > 25 and price rising
                uptrend = (adx_series > 25) & (result['returns_5d'] > 0)
                # Downtrend: ADX > 25 and price falling
                downtrend = (adx_series > 25) & (result['returns_5d'] < 0)
                
                result['market_regime'] = np.where(uptrend, 1, 
                                          np.where(downtrend, 2, 0))
            except Exception as e:
                logger.warning(f"Error calculating market regime: {e}")
                result['market_regime'] = pd.Series(0, index=df.index)  # Default to ranging
            
            # Volatility regime (simplified)
            try:
                vol_series = result['volatility_20d']
                # Ensure we have some non-zero values to work with
                if vol_series.max() > 0:
                    vol_med = vol_series.replace(0, np.nan).median()
                    if np.isnan(vol_med) or vol_med == 0:
                        vol_med = 0.01  # Fallback if median is NaN or zero
                    
                    # 0 = low, 1 = medium, 2 = high volatility
                    result['volatility_regime'] = np.where(
                        vol_series > vol_med * 1.5, 2,
                        np.where(vol_series < vol_med * 0.5, 0, 1)
                    )
                else:
                    result['volatility_regime'] = pd.Series(1, index=df.index)  # Default to medium
            except Exception as e:
                logger.warning(f"Error calculating volatility regime: {e}")
                result['volatility_regime'] = pd.Series(1, index=df.index)  # Default to medium
            
            # Ensure all calculated indicators are properly aligned Series with the same index
            for key, value in list(result.items()):
                if isinstance(value, pd.Series):
                    # Ensure the series has the right index
                    if not value.index.equals(df.index):
                        result[key] = value.reindex(df.index, fill_value=0)
                    # Replace any remaining NaN/inf values
                    result[key] = result[key].replace([np.inf, -np.inf], np.nan).fillna(0)
                else:
                    # If not a Series, remove it (except 'asset' key)
                    if key != 'asset':
                        del result[key]
            
            logger.info(f"Technical indicators calculated: {list(result.keys())}")
            return result
        
        except Exception as e:
            logger.error(f"Error in _compute_technical_indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return {'asset': 'error'}
        
    def _calculate_indicators(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate comprehensive technical indicators from price and volume data"""
        try:
            features = {}
            
            # Helper function for safe indicator calculation
            def safe_indicator(func, default_value=None):
                try:
                    result = func()
                    # Check for NaN values
                    if isinstance(result, pd.Series) and result.isna().any():
                        result = result.ffill().fillna(0)
                    return result
                except Exception as e:
                    logger.debug(f"Error calculating indicator: {str(e)}")
                    # Return appropriate default
                    if default_value is not None:
                        return default_value
                    else:
                        # Create a Series of zeros with the same index as close
                        return pd.Series(0, index=data['close'].index)
            
            # Check if we have all OHLC data
            has_ohlc = all(k in data for k in ['open', 'high', 'low', 'close'])
            # Get close prices (required for all indicators)
            close = data['close'].copy()
            
            # Basic price features
            features['close'] = close
            if 'open' in data:
                features['open'] = data['open']
            if 'high' in data:
                features['high'] = data['high']
            if 'low' in data:
                features['low'] = data['low']
            
            # Volume features
            if 'volume' in data:
                features['volume'] = data['volume']
                
                # Calculate more sophisticated volume indicators
                try:
                    if has_ohlc:
                        obv = OnBalanceVolumeIndicator(close=close, volume=data['volume'])
                        features['obv'] = safe_indicator(obv.on_balance_volume)
                        
                        cmf = ChaikinMoneyFlowIndicator(
                            high=data['high'], 
                            low=data['low'], 
                            close=close, 
                            volume=data['volume']
                        )
                        features['cmf'] = safe_indicator(cmf.chaikin_money_flow)
                        
                        accum_dist = AccDistIndexIndicator(
                            high=data['high'], 
                            low=data['low'], 
                            close=close, 
                            volume=data['volume']
                        )
                        features['accum_dist'] = safe_indicator(accum_dist.acc_dist_index)
                except Exception as e:
                    logger.debug(f"Error calculating volume indicators: {str(e)}")
            
            # =================== TREND INDICATORS ===================
            
            # EMA indicators
            try:
                for period in [5, 10, 20, 50, 100, 200]:
                    try:
                        ema = EMAIndicator(close=close, window=period)
                        features[f'ema_{period}'] = safe_indicator(ema.ema_indicator)
                    except Exception as e:
                        logger.debug(f"Error calculating EMA-{period}: {str(e)}")
            except Exception as e:
                logger.debug(f"Error calculating EMA group: {str(e)}")
            
            # MACD
            try:
                macd_ind = MACD(close=close)
                features['macd'] = safe_indicator(macd_ind.macd)
                features['macd_signal'] = safe_indicator(macd_ind.macd_signal)
                features['macd_diff'] = safe_indicator(macd_ind.macd_diff)
            except Exception as e:
                logger.debug(f"Error calculating MACD: {str(e)}")
            
            # ADX (trend strength)
            if has_ohlc:
                try:
                    adx = ADXIndicator(high=data['high'], low=data['low'], close=close)
                    features['adx'] = safe_indicator(adx.adx)
                    features['adx_pos'] = safe_indicator(adx.adx_pos)
                    features['adx_neg'] = safe_indicator(adx.adx_neg)
                except Exception as e:
                    logger.debug(f"Error calculating ADX: {str(e)}")
            
            # =================== MOMENTUM INDICATORS ===================
            
            # RSI
            try:
                rsi = RSIIndicator(close=close)
                features['rsi'] = safe_indicator(rsi.rsi)
            except Exception as e:
                logger.debug(f"Error calculating RSI: {str(e)}")
            
            # Other momentum indicators
            if has_ohlc:
                try:
                    roc = ROCIndicator(close=close, window=10)
                    features['roc'] = safe_indicator(roc.roc)
                    
                    williams = WilliamsRIndicator(high=data['high'], low=data['low'], close=close)
                    features['willr'] = safe_indicator(williams.williams_r)
                    
                    stoch = StochasticOscillator(high=data['high'], low=data['low'], close=close)
                    features['stoch_k'] = safe_indicator(stoch.stoch)
                    features['stoch_d'] = safe_indicator(stoch.stoch_signal)
                except Exception as e:
                    logger.debug(f"Error calculating momentum indicators: {str(e)}")
            
            # =================== VOLATILITY INDICATORS ===================
            
            # Bollinger Bands
            try:
                bb = BollingerBands(close=close)
                features['bbands_upper'] = safe_indicator(bb.bollinger_hband)
                features['bbands_middle'] = safe_indicator(bb.bollinger_mavg)
                features['bbands_lower'] = safe_indicator(bb.bollinger_lband)
            except Exception as e:
                logger.debug(f"Error calculating Bollinger Bands: {str(e)}")
            
            # ATR
            if has_ohlc:
                try:
                    atr = AverageTrueRange(high=data['high'], low=data['low'], close=close)
                    features['atr'] = safe_indicator(atr.average_true_range)
                except Exception as e:
                    logger.debug(f"Error calculating ATR: {str(e)}")
            
            # =================== DERIVED FEATURES ===================
            
            # Volatility calculation
            try:
                # Calculate returns
                returns = close.pct_change().fillna(0)
                # Calculate rolling volatility
                for window in [5, 10, 20]:
                    features[f'volatility_{window}d'] = returns.rolling(window=window).std().fillna(0) * np.sqrt(252)
            except Exception as e:
                logger.debug(f"Error calculating volatility features: {str(e)}")
            
            # Calculate returns
            try:
                # Calculate price changes
                for period in [1, 5, 10]:
                    features[f'returns_{period}d'] = close.pct_change(periods=period).fillna(0)
            except Exception as e:
                logger.debug(f"Error calculating returns features: {str(e)}")
            
            # Final check for NaN values in all features
            for key, value in features.items():
                if isinstance(value, pd.Series) and value.isna().any():
                    features[key] = value.fillna(0)
            
            return features
        except Exception as e:
            logger.error(f"Error in _calculate_indicators: {str(e)}")
            logger.error(traceback.format_exc())
            # Return at least the raw price data
            return {'close': data.get('close', pd.Series())}
        
    def _add_vol_surface_features(self, df: pd.DataFrame) -> Dict:
        """Advanced volatility modeling and regime detection"""
        try:
            features = {}
            close_col = [col for col in df.columns if 'close' in col.lower()][0]
            
            # Ensure close prices are positive and handle missing values
            close_prices = pd.to_numeric(df[close_col], errors='coerce')
            close_prices = close_prices.replace([0, np.inf, -np.inf], np.nan)
            close_prices = close_prices.ffill().bfill()
            close_prices = close_prices.clip(lower=1e-8)
            
            # Calculate returns safely
            returns = pd.Series(
                np.log(close_prices / close_prices.shift(1)),
                index=close_prices.index
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale returns for ARCH model
            scaled_returns = returns * 100  # Scale up by 100 for better numerical stability
            
            # Multi-horizon volatility forecasting
            for horizon in [1, 5, 22]:
                try:
                    if len(scaled_returns.dropna()) > 100:  # Only fit if enough data
                        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                        res = model.fit(disp='off', show_warning=False)
                        forecast = res.forecast(horizon=horizon)
                        vol = np.sqrt(forecast.variance.iloc[-1]) / 100
                        
                        # FIXED: Create a proper Series with the correct index
                        features[f'vol_{horizon}d'] = pd.Series(
                            np.repeat(vol, len(returns)),  # Repeat the value for all indices
                            index=returns.index
                        ).fillna(0)
                    else:
                        features[f'vol_{horizon}d'] = returns.rolling(horizon*10).std().fillna(0)
                except Exception as e:
                    logger.warning(f"Error in GARCH modeling for horizon {horizon}: {str(e)}")
                    features[f'vol_{horizon}d'] = returns.rolling(horizon*10).std().fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in volatility features: {str(e)}")
            return {}
        
    def _add_flow_features(self, df: pd.DataFrame) -> Dict:
        """Extract and predict market flow metrics"""
        try:
            result = {}
            
            # Volume and liquidity features
            if 'volume' in df.columns:
                # Calculate volume moving average and momentum
                volume = df['volume']
                result['vol_ma'] = volume.rolling(5).mean()
                # Replace deprecated fillna(method='pad') with ffill()
                result['vol_ratio'] = (volume / volume.rolling(10).mean()).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
                
                # Volume trend
                result['vol_trend'] = volume.diff(5)
            
            # Order flow imbalance if available
            if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
                bid_depth = df['bid_depth']
                ask_depth = df['ask_depth']
                
                # Order book imbalance
                total_depth = bid_depth + ask_depth
                result['ob_imbalance'] = (bid_depth - ask_depth) / total_depth.replace(0, np.nan)
                
                # Fill missing values
                result['ob_imbalance'] = result['ob_imbalance'].ffill().fillna(0)
                
                # Smoothed imbalance
                result['ob_imbalance_ma'] = result['ob_imbalance'].rolling(5).mean()
            
            # Funding rate features
            if 'funding_rate' in df.columns:
                funding = df['funding_rate']
                result['funding_ma'] = funding.rolling(8).mean()
                # Normalized funding vs historical
                funding_std = funding.rolling(24).std().replace(0, np.nan)
                result['funding_z'] = (funding - result['funding_ma']) / funding_std
                result['funding_z'] = result['funding_z'].fillna(0)
            
            return result
        
        except Exception as e:
            logger.error(f"Error adding flow features: {str(e)}")
            return {}
        
    def _add_market_sentiment(self, df: pd.DataFrame) -> Dict:
        """Extract market sentiment indicators"""
        try:
            result = {}
            
            if 'close' in df.columns:
                close = df['close']
                
                # Price momentum at different timeframes
                # Replace deprecated default fill_method
                result['returns_1d'] = close.pct_change(1, fill_method=None)
                result['returns_5d'] = close.pct_change(5, fill_method=None)
                result['returns_20d'] = close.pct_change(20, fill_method=None)
                
                # Fill NaN values with 0
                for col in ['returns_1d', 'returns_5d', 'returns_20d']:
                    result[col] = result[col].fillna(0)
                
                # Momentum: Smoothed rate of change
                momentum = (close / close.shift(10) - 1) * 100
                result['momentum'] = momentum.rolling(5).mean()
            
            return result
        
        except Exception as e:
            logger.error(f"Error in market sentiment: {str(e)}")
            return {}
            
    def _add_intermarket_correlations(self, full_df: pd.DataFrame, current_asset: str) -> Dict:
        """Calculate inter-market correlation features"""
        try:
            features = {}
            window = 14  # Correlation window
            
            # Get returns for all assets
            returns_dict = {}
            for asset in full_df.columns.get_level_values('asset').unique():
                try:
                    # Get close prices properly handling MultiIndex
                    try:
                        close_prices = full_df.loc[:, (asset, 'close')]
                        if isinstance(close_prices, pd.DataFrame):
                            close_prices = close_prices.iloc[:, 0]
                    except Exception as e:
                        logger.warning(f"Error accessing close prices for {asset} in intermarket correlation: {str(e)}")
                        continue
                    
                    # Handle zeros and missing values before log
                    close_prices = pd.to_numeric(close_prices, errors='coerce')
                    # Check for NaN/inf on values, not on MultiIndex
                    if pd.isna(close_prices.values).any() or np.isinf(close_prices.values).any():
                        close_prices = close_prices.replace([0, np.inf, -np.inf], np.nan)
                        close_prices = close_prices.ffill().bfill()
                        close_prices = close_prices.clip(lower=1e-8)
                    
                    # Calculate returns safely
                    returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
                    returns = returns.replace([np.inf, -np.inf], 0)
                    returns_dict[asset] = returns
                    
                except Exception as e:
                    logger.warning(f"Could not calculate returns for {asset} in intermarket correlation: {str(e)}")
                    continue
            
            if not returns_dict:
                return {}
            
            returns_df = pd.DataFrame(returns_dict).fillna(0)
            
            # Ensure the current asset exists in the returns data
            if current_asset not in returns_df.columns:
                logger.warning(f"Current asset {current_asset} not found in returns data")
                return {}
            
            current_returns = returns_df[current_asset]
            
            # Calculate correlations safely
            for asset in returns_df.columns:
                if asset != current_asset:
                    try:
                        other_returns = returns_df[asset]
                        # Only calculate correlation if both series have variation
                        if current_returns.std() > 0 and other_returns.std() > 0:
                            corr = returns_df[asset].rolling(window).corr(current_returns)
                            features[f'corr_{asset}'] = corr.fillna(0).clip(-1, 1)
                        else:
                            features[f'corr_{asset}'] = pd.Series(0, index=returns_df.index)
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation between {current_asset} and {asset}: {str(e)}")
                        features[f'corr_{asset}'] = pd.Series(0, index=returns_df.index)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in intermarket correlations for {current_asset}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
        
    def _add_cross_sectional_features(self, full_df: pd.DataFrame, current_asset: str) -> Dict:
        """Cross-sectional and correlation-based features"""
        try:
            features = {}
            
            # Get returns for all assets
            returns_dict = {}
            for asset in full_df.columns.get_level_values('asset').unique():
                try:
                    # Get close prices properly handling MultiIndex
                    try:
                        close_prices = full_df.loc[:, (asset, 'close')]
                        if isinstance(close_prices, pd.DataFrame):
                            close_prices = close_prices.iloc[:, 0]
                    except Exception as e:
                        logger.warning(f"Error accessing close prices for {asset}: {str(e)}")
                        continue
                    
                    # Handle zeros and missing values before log
                    close_prices = pd.to_numeric(close_prices, errors='coerce')
                    # Use pandas method to check for NaN or infinity, not directly on MultiIndex
                    mask = pd.isna(close_prices.values) | np.isinf(close_prices.values)
                    if mask.any():
                        close_prices = close_prices.replace([0, np.inf, -np.inf], np.nan)
                        close_prices = close_prices.ffill().bfill()
                        close_prices = close_prices.clip(lower=1e-8)
                    
                    # Calculate returns safely
                    returns = pd.Series(
                        np.log(close_prices / close_prices.shift(1)),
                        index=close_prices.index
                    )
                    # Replace inf with NaN then fill NaN with 0
                    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    returns_dict[asset] = returns
                    
                except Exception as e:
                    logger.warning(f"Could not calculate returns for {asset}: {str(e)}")
                    continue
            
            if not returns_dict:
                return {}
            
            returns_df = pd.DataFrame(returns_dict)
            
            # Determine number of components based on available data
            n_samples, n_features = returns_df.shape
            n_components = min(self.n_components, min(n_samples, n_features) - 1)

            if n_components > 0:
                try:
                    # Modify before PCA application:
                    # Add volatility-normalized features
                    volatility = returns_df.rolling(100).std().ffill().fillna(0.01)
                    # Avoid division by zero
                    volatility = volatility.replace(0, 0.01)
                    
                    # Create normalized returns for PCA
                    normalized_returns = returns_df.div(volatility)
                    
                    # Apply PCA safely
                    if len(normalized_returns) > n_components + 10:  # Ensure enough data
                        pca = PCA(n_components=n_components)
                        pca.fit(normalized_returns.fillna(0))
                        
                        # Factor loadings
                        if current_asset in normalized_returns.columns:
                            idx = list(normalized_returns.columns).index(current_asset)
                            loadings = pca.components_[:, idx]
                            for i, loading in enumerate(loadings):
                                features[f'factor_{i+1}_loading'] = pd.Series(loading, index=returns_df.index)
                    else:
                        # Not enough data for PCA
                        for i in range(n_components):
                            features[f'factor_{i+1}_loading'] = pd.Series(0, index=returns_df.index)
                except Exception as e:
                    logger.warning(f"Error in PCA calculation: {str(e)}")
                    # Fill with zeros if PCA fails
                    for i in range(n_components):
                        features[f'factor_{i+1}_loading'] = pd.Series(0, index=returns_df.index)
            
            # Cross-sectional momentum
            try:
                # Calculate rolling means for each asset
                rolling_means = returns_df.rolling(5).mean().fillna(0)
                
                # Rank assets at each timestamp
                ranks = rolling_means.rank(axis=1)
                
                # Get the rank for the current asset
                if current_asset in ranks.columns:
                    features['xs_momentum'] = ranks[current_asset]
                else:
                    features['xs_momentum'] = pd.Series(0, index=returns_df.index)
                
            except Exception as e:
                logger.warning(f"Error calculating cross-sectional momentum: {str(e)}")
                features['xs_momentum'] = pd.Series(0, index=returns_df.index)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in cross-sectional features: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
        
    def _detect_regime(self, returns: pd.Series) -> pd.Series:
        """Detect market regime using Hidden Markov Model"""
        from hmmlearn import hmm
        
        # Prepare data
        data = returns.values.reshape(-1, 1)
        
        # Determine number of states based on data
        n_states = min(3, len(data) // 50)  # At least 50 samples per state
        if n_states < 2:
            return pd.Series(0, index=returns.index)
        
        # Fit HMM with dynamic states
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            tol=0.01,
            random_state=42
        )
        
        try:
            model.fit(data)
            regime = model.predict(data)
            return pd.Series(regime, index=returns.index)
        except Exception as e:
            logger.warning(f"HMM fitting failed: {str(e)}")
            return pd.Series(0, index=returns.index)
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values more intelligently"""
        try:
            # Forward fill first (for time series consistency)
            df = df.ffill()
            
            # For any remaining NaNs, use rolling median
            window_size = min(24, len(df) // 2)  # Use smaller of 24 periods or half the data
            rolling_median = df.rolling(window=window_size, min_periods=1).median()
            df = df.fillna(rolling_median)
            
            # If still any NaNs, fill with 0
            df = df.fillna(0)
            
            # Clip extreme values
            df = df.clip(lower=-1e8, upper=1e8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df.fillna(0)

    def _select_features(self, df: pd.DataFrame, target_col: str = 'close') -> List[str]:
        """Process features and optionally remove highly correlated ones"""
        try:
            # By default, keep all features
            features = list(df.columns)
            
            # Optionally, we can remove highly correlated features (correlation > 0.95)
            # to reduce multicollinearity while preserving information
            if self.feature_selection_threshold > 0:  # Only if threshold is set
                correlation_matrix = df.corr().abs()
                upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                
                if to_drop:
                    logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
                    features = [f for f in features if f not in to_drop]
            
            logger.info(f"Using {len(features)} features out of {len(df.columns)} total")
            return features
            
        except Exception as e:
            logger.error(f"Error in feature processing: {str(e)}")
            return list(df.columns)  # Return all features if there's an error

    def _combine_features(self, features: Dict) -> pd.DataFrame:
        """Combine all features into a single DataFrame with proper normalization"""
        try:
            # ADD THIS: Log what's coming into the combine method
            logger.info(f"Combining features for assets: {list(features.keys())}")
            
            combined = pd.DataFrame()
            all_features = set()
            
            # First pass: collect all feature names and find common index
            common_index = None
            for asset, asset_features in features.items():
                for feat_name, feat_values in asset_features.items():
                    if isinstance(feat_values, (pd.Series, np.ndarray)):
                        all_features.add(feat_name)
                        if isinstance(feat_values, pd.Series):
                            if common_index is None:
                                common_index = feat_values.index
                            else:
                                common_index = common_index.union(feat_values.index)
            
            if common_index is None:
                raise ValueError("No valid time series data found")
                
            logger.info(f"Total features to process: {len(all_features)}")
            
            # Second pass: ensure all assets have all features with aligned index
            for asset, asset_features in features.items():
                try:
                    # Convert all features to DataFrame with proper types
                    asset_df = pd.DataFrame(index=common_index)
                    
                    # Process each feature
                    for feat_name in all_features:
                        if feat_name in asset_features:
                            value = asset_features[feat_name]
                            if isinstance(value, (pd.Series, np.ndarray)):
                                if isinstance(value, pd.Series):
                                    # Reindex to common index
                                    value = value.reindex(common_index)
                                else:
                                    # Convert numpy array to series with common index
                                    value = pd.Series(value, index=common_index[:len(value)])
                                # Use pd.to_numeric to convert, and handle any errors gracefully
                                try:
                                    asset_df[feat_name] = pd.to_numeric(value, errors='coerce')
                                except Exception as e:
                                    logger.warning(f"Error converting {feat_name} to numeric: {str(e)}")
                                    asset_df[feat_name] = 0.0
                        else:
                            # If feature doesn't exist for this asset, fill with 0
                            asset_df[feat_name] = 0.0
                            logger.debug(f"Adding missing feature {feat_name} for {asset}")
                    
                    if len(asset_df) > 0:
                        # Check for NaN values directly on the values array, not the columns
                        if pd.isna(asset_df.values).any():
                            logger.debug(f"Asset {asset} has NaN values, filling them")
                            asset_df = self._handle_missing_values(asset_df)
                        
                        # Keep all features by default
                        if self.selected_features is None:
                            self.selected_features = self._select_features(asset_df)
                        
                        # Create proper MultiIndex columns
                        asset_df.columns = pd.MultiIndex.from_product(
                            [[asset], asset_df.columns],
                            names=['asset', 'feature']
                        )
                        
                        # Combine with main DataFrame
                        if combined.empty:
                            combined = asset_df
                        else:
                            combined = pd.concat([combined, asset_df], axis=1)
                    
                except Exception as e:
                    logger.error(f"Error combining features for {asset}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Sort columns for consistency
            try:
                combined = combined.sort_index(axis=1)
            except Exception as e:
                logger.warning(f"Error sorting columns: {str(e)}")
            
            # Final verification
            logger.info(f"Combined features shape: {combined.shape}")
            logger.info(f"Features per asset: {len(self.selected_features) if self.selected_features else 0}")
            logger.info(f"Total assets: {len(features)}")
            
            # ADD THIS: Log the combined features before returning
            logger.info(f"Combined features columns: {list(combined.columns) if isinstance(combined, pd.DataFrame) else 'Not a DataFrame'}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in combine_features: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def engineer_features(self, data_dict):
        try:
            processed_data = {}
            
            # Handle nested dictionary structure (exchange -> symbols -> dataframe)
            for exchange, exchange_data in data_dict.items():
                logger.info(f"Processing features for exchange: {exchange}")
                
                # Case A: exchange_data is already a DataFrame
                if isinstance(exchange_data, pd.DataFrame):
                    df = exchange_data
                    if df.empty:
                        logger.warning(f"Empty DataFrame for {exchange}, skipping")
                        continue
                        
                    # Ensure MultiIndex columns with proper names
                    if not isinstance(df.columns, pd.MultiIndex):
                        logger.warning(f"Converting {exchange} columns to MultiIndex")
                        df.columns = pd.MultiIndex.from_product([[exchange], df.columns], names=['asset', 'feature'])
                    else:
                        # Ensure the multi-index has proper names
                        # Don't try to check isna on the MultiIndex
                        df.columns.names = ['asset', 'feature']
                    
                    # Process features with error handling
                    try:
                        processed = self.transform(df)
                        if isinstance(processed, pd.DataFrame) and not processed.empty:
                            processed_data[exchange] = processed
                        else:
                            logger.warning(f"Transform returned empty DataFrame for {exchange}")
                    except Exception as e:
                        logger.error(f"Error transforming data for {exchange}: {str(e)}")
                        logger.error(traceback.format_exc())
                
                # Case B: exchange_data is a nested dictionary (symbol -> dataframe)
                elif isinstance(exchange_data, dict):
                    # Combine all symbol dataframes for this exchange
                    symbol_dfs = []
                    for symbol, symbol_data in exchange_data.items():
                        if not isinstance(symbol_data, pd.DataFrame):
                            logger.warning(f"Data for {exchange}/{symbol} is not a DataFrame (type: {type(symbol_data)}), skipping")
                            continue
                            
                        if symbol_data.empty:
                            logger.warning(f"Empty DataFrame for {exchange}/{symbol}, skipping")
                            continue
                            
                        # Create multi-level columns for the symbol with proper names
                        try:
                            symbol_data.columns = pd.MultiIndex.from_product([[symbol], symbol_data.columns], names=['asset', 'feature'])
                            symbol_dfs.append(symbol_data)
                        except Exception as e:
                            logger.error(f"Error creating MultiIndex for {symbol}: {str(e)}")
                            continue
                    
                    if not symbol_dfs:
                        logger.warning(f"No valid DataFrames for exchange {exchange}, skipping")
                        continue
                        
                    # Combine all symbols into one DataFrame for this exchange
                    try:
                        combined_df = pd.concat(symbol_dfs, axis=1)
                        
                        # Process features with error handling
                        processed = self.transform(combined_df)
                        if isinstance(processed, pd.DataFrame) and not processed.empty:
                            processed_data[exchange] = processed
                        else:
                            logger.warning(f"Transform returned empty DataFrame for {exchange} combined data")
                    except Exception as e:
                        logger.error(f"Error processing combined data for {exchange}: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"Data for {exchange} is not a DataFrame or dict (type: {type(exchange_data)}), skipping")
                    continue
                
            if not processed_data:
                logger.warning("No data processed successfully, returning empty DataFrame")
                return pd.DataFrame()
            
            # Combine all exchanges
            try:
                combined = pd.concat(processed_data.values(), axis=1)
                
                # Final validation
                if combined.empty:
                    logger.warning("Combined DataFrame is empty")
                    return pd.DataFrame()
                
                return combined
            except Exception as e:
                logger.error(f"Error combining processed data: {str(e)}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _calculate_rolling_hurst(self, series, window):
        """Calculate Hurst exponent in a rolling window to identify trend strength"""
        series = series.ffill()
        result = pd.Series(index=series.index, data=np.nan)
        
        # Need at least 100 points for a reasonable Hurst calculation
        min_window = min(100, window)
        if len(series) < min_window:
            return result
            
        # Calculate for each window
        for i in range(min_window, len(series)):
            start_idx = max(0, i - window)
            time_series = series.iloc[start_idx:i].values
            result.iloc[i] = self._hurst_exponent(time_series)
            
        # Forward fill initial NaN values - replace deprecated fillna(method='ffill')
        result = result.ffill().fillna(0)
        return result
        
    def _hurst_exponent(self, time_series, max_lag=20):
        """Calculate Hurst exponent for a time series"""
        # Convert to numpy array if needed
        time_series = np.array(time_series)
        
        # Create log returns
        returns = np.diff(np.log(time_series))
        
        # Zero mean returns
        zero_mean = returns - np.mean(returns)
        
        # Calculate variance of difference for each lag
        tau = np.arange(1, min(max_lag, len(returns) // 4))
        var = np.zeros(len(tau))
        
        for i, lag in enumerate(tau):
            # Calculate variance of difference
            var[i] = np.std(zero_mean[lag:] - zero_mean[:-lag])
            
        # Avoid log(0) errors
        var = var[var > 0]
        tau = tau[:len(var)]
        
        if len(var) <= 1:
            return 0.5  # Default to random walk
            
        # Fit power law: var = c * tau^(2*H)
        log_tau = np.log(tau)
        log_var = np.log(var)
        
        # Linear regression to find Hurst exponent
        hurst = np.polyfit(log_tau, log_var, 1)[0] / 2.0
        
        return min(max(hurst, 0), 1)  # Bound between 0 and 1