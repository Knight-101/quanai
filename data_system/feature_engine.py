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
                raise ValueError("Empty DataFrame provided")
                
            features = {}
            logger.info(f"Processing features for assets: {list(df.columns.get_level_values('asset').unique())}")
            
            for asset in df.columns.get_level_values('asset').unique():
                try:
                    # Get data for this asset
                    asset_df = df.xs(asset, axis=1, level='asset')
                    
                    # Technical indicators
                    features[asset] = self._compute_technical_indicators(asset_df)
                    if 'close' not in features[asset]:
                        features[asset]['close'] = pd.to_numeric(asset_df['close'], errors='coerce')
                    
                    # Volatility surface and regime detection
                    vol_features = self._add_vol_surface_features(asset_df)
                    features[asset].update(vol_features)
                    
                    # Order flow and liquidity metrics
                    flow_features = self._add_flow_features(asset_df)
                    features[asset].update(flow_features)
                    
                    # Market sentiment features
                    sentiment_features = self._add_market_sentiment(asset_df)
                    features[asset].update(sentiment_features)
                    
                    # Inter-market correlation features
                    correlation_features = self._add_intermarket_correlations(df, asset)
                    features[asset].update(correlation_features)
                    
                    # Cross-sectional features
                    xs_features = self._add_cross_sectional_features(df, asset)
                    features[asset].update(xs_features)
                    
                except Exception as e:
                    logger.error(f"Error processing features for {asset}: {str(e)}")
                    continue
            
            if not features:
                raise ValueError("No features generated")
            
            # Combine all features
            return self._combine_features(features)
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            return pd.DataFrame()
        
    def _compute_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Compute various technical indicators for each asset"""
        try:
            result = {}
            for asset in df.columns.get_level_values('asset').unique():
                asset_data = {}
                # Get OHLCV data for the current asset
                try:
                    ohlcv = df[asset]
                except:
                    continue
                
                if 'close' not in ohlcv.columns:
                    continue
                    
                # Work with asset prices
                closes = ohlcv['close']
                
                if 'open' in ohlcv.columns:
                    opens = ohlcv['open']
                if 'high' in ohlcv.columns:
                    highs = ohlcv['high']
                if 'low' in ohlcv.columns:
                    lows = ohlcv['low']
                if 'volume' in ohlcv.columns:
                    volumes = ohlcv['volume']
                
                # ENHANCED: Market Regime Detection
                # 1. ADX to identify trending markets
                if all(x in ohlcv.columns for x in ['high', 'low', 'close']):
                    try:
                        adx = ADXIndicator(highs, lows, closes, self.adx_period)
                        asset_data['adx'] = adx.adx()
                        asset_data['adx_pos'] = adx.adx_pos()  # Positive directional indicator
                        asset_data['adx_neg'] = adx.adx_neg()  # Negative directional indicator
                        
                        # ADX-based regime classification
                        # ADX > 25 typically indicates trending market
                        asset_data['is_trending'] = (asset_data['adx'] > 25).astype(float)
                        
                        # Direction strength indicator (combination of +DI and -DI)
                        asset_data['trend_strength'] = asset_data['adx_pos'] - asset_data['adx_neg']
                    except Exception as e:
                        logger.warning(f"Error calculating ADX: {e}")
                
                # 2. Hurst Exponent for long-term trend/mean-reversion detection
                try:
                    # Calculate Hurst exponent in a rolling window
                    asset_data['hurst_exponent'] = self._calculate_rolling_hurst(closes, self.hurst_period)
                    
                    # Hurst > 0.5 suggests trending, < 0.5 suggests mean-reverting
                    asset_data['is_mean_reverting'] = (asset_data['hurst_exponent'] < 0.45).astype(float)
                    asset_data['is_random_walk'] = ((asset_data['hurst_exponent'] >= 0.45) & 
                                                   (asset_data['hurst_exponent'] <= 0.55)).astype(float)
                    asset_data['is_persistent'] = (asset_data['hurst_exponent'] > 0.55).astype(float)
                except Exception as e:
                    logger.warning(f"Error calculating Hurst exponent: {e}")
                
                # 3. Volatility regime
                try:
                    returns = closes.pct_change().fillna(0)
                    # Rolling volatility
                    rolling_vol = returns.rolling(30).std()
                    # Volatility of volatility - meta-volatility
                    vol_of_vol = rolling_vol.rolling(30).std()
                    asset_data['volatility'] = rolling_vol
                    asset_data['vol_of_vol'] = vol_of_vol
                    
                    # High vol-of-vol indicates regime shifts
                    asset_data['vol_regime_shift'] = (vol_of_vol > vol_of_vol.rolling(100).mean() * 1.5).astype(float)
                    
                    # Classify volatility regime
                    vol_quantiles = rolling_vol.rolling(252).quantile(0.75)
                    asset_data['high_vol_regime'] = (rolling_vol > vol_quantiles).astype(float)
                except Exception as e:
                    logger.warning(f"Error calculating volatility regime: {e}")
                
                # 4. Combined market regime score (1 = strong trend, 0 = strong range)
                try:
                    if 'is_trending' in asset_data and 'is_persistent' in asset_data:
                        # Combine ADX trend and Hurst persistence
                        raw_regime = (asset_data['is_trending'] + asset_data['is_persistent']) / 2
                        # Smooth the regime indicator
                        asset_data['market_regime'] = raw_regime.rolling(self.regime_smoothing).mean().fillna(0.5)
                except Exception as e:
                    logger.warning(f"Error calculating market regime: {e}")
                
                # Continue with existing indicators
                def safe_indicator(func):
                    try:
                        return func()
                    except:
                        return pd.Series(index=closes.index, data=np.nan)
                
                # Trend indicators
                ema_short = EMAIndicator(close=closes, window=12)
                ema_medium = EMAIndicator(close=closes, window=26)
                ema_long = EMAIndicator(close=closes, window=50)
                
                asset_data['ema_short'] = safe_indicator(ema_short.ema_indicator)
                asset_data['ema_medium'] = safe_indicator(ema_medium.ema_indicator)
                asset_data['ema_long'] = safe_indicator(ema_long.ema_indicator)
                
                # MACD
                macd = MACD(close=closes)
                asset_data['macd'] = safe_indicator(macd.macd)
                asset_data['macd_signal'] = safe_indicator(macd.macd_signal)
                
                # Enhanced momentum indicators
                rsi = RSIIndicator(close=closes)
                asset_data['rsi'] = safe_indicator(rsi.rsi)
                
                roc = ROCIndicator(close=closes, window=10)
                asset_data['roc'] = safe_indicator(roc.roc)
                
                williams = WilliamsRIndicator(high=highs, low=lows, close=closes)
                asset_data['willr'] = safe_indicator(williams.williams_r)
                
                stoch = StochasticOscillator(high=highs, low=lows, close=closes)
                asset_data['stoch_k'] = safe_indicator(stoch.stoch)
                asset_data['stoch_d'] = safe_indicator(stoch.stoch_signal)
                
                # Advanced trend indicators
                bb = BollingerBands(close=closes)
                asset_data['bbands_upper'] = safe_indicator(bb.bollinger_hband)
                asset_data['bbands_middle'] = safe_indicator(bb.bollinger_mavg)
                asset_data['bbands_lower'] = safe_indicator(bb.bollinger_lband)
                
                atr = AverageTrueRange(high=highs, low=lows, close=closes)
                asset_data['atr'] = safe_indicator(atr.average_true_range)
                
                # Volume and momentum indicators
                obv = OnBalanceVolumeIndicator(close=closes, volume=volumes)
                asset_data['obv'] = safe_indicator(obv.on_balance_volume)
                
                adi = AccDistIndexIndicator(high=highs, low=lows, close=closes, volume=volumes)
                asset_data['ad'] = safe_indicator(adi.acc_dist_index)
                
                cmf = ChaikinMoneyFlowIndicator(high=highs, low=lows, close=closes, volume=volumes)
                asset_data['cmf'] = safe_indicator(cmf.chaikin_money_flow)
                
                result[asset] = asset_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return {}
        
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
        """Order flow and market microstructure features"""
        try:
            features = {}
            
            # Ensure numeric values and handle missing data
            close = pd.to_numeric(df['close'], errors='coerce').fillna(method='ffill')
            volume = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            
            # Volume profile with safety checks
            volume_sum = volume.rolling(24).sum()
            volume_sum = volume_sum.replace(0, np.nan)  # Avoid division by zero
            vwap = (close * volume).rolling(24).sum() / volume_sum
            vwap = vwap.fillna(close)  # Use close price if VWAP calculation fails
            
            # Calculate divergence safely
            features['vwap_divergence'] = ((close - vwap) / vwap.clip(lower=1e-8)).fillna(0).clip(-10, 10)
            
            # Order book pressure with safety checks
            if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
                bid_depth = pd.to_numeric(df['bid_depth'], errors='coerce').fillna(0).clip(lower=1e-8)
                ask_depth = pd.to_numeric(df['ask_depth'], errors='coerce').fillna(0).clip(lower=1e-8)
                sum_depth = bid_depth + ask_depth
                features['book_pressure'] = np.where(
                    sum_depth > 0,
                    (bid_depth - ask_depth) / sum_depth,
                    0
                )
            
            # Funding rate dynamics with safety checks
            if 'funding_rate' in df.columns:
                funding = pd.to_numeric(df['funding_rate'], errors='coerce').fillna(0)
                funding_std = funding.rolling(24).std()
                features['funding_zscore'] = np.where(
                    funding_std > 0,
                    (funding - funding.rolling(24).mean()) / funding_std,
                    0
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error in flow features: {str(e)}")
            return {}
        
    def _add_market_sentiment(self, df: pd.DataFrame) -> Dict:
        """Calculate market sentiment indicators"""
        try:
            features = {}
            
            # Price momentum sentiment
            returns = df['close'].pct_change()
            features['sentiment_ma'] = returns.rolling(14).mean()
            features['sentiment_std'] = returns.rolling(14).std()
            
            # Volume sentiment
            volume_ma = df['volume'].rolling(14).mean()
            features['volume_sentiment'] = (df['volume'] - volume_ma) / volume_ma
            
            # Price trend strength
            features['trend_strength'] = abs(
                df['close'].rolling(14).mean() - df['close'].rolling(28).mean()
            ) / df['close'].rolling(28).std()
            
            # Volatility regime
            features['volatility_regime'] = returns.rolling(14).std() / returns.rolling(28).std()
            
            # Market efficiency ratio
            price_change = abs(df['close'] - df['close'].shift(14))
            path_length = (abs(df['close'].diff())).rolling(14).sum()
            features['market_efficiency'] = price_change / path_length
            
            # Funding rate sentiment (if available)
            if 'funding_rate' in df.columns:
                features['funding_sentiment'] = (
                    df['funding_rate'] - df['funding_rate'].rolling(24).mean()
                ) / df['funding_rate'].rolling(24).std()
            
            return features
            
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
                    close_prices = full_df.loc[:, (asset, 'close')]
                    if isinstance(close_prices, pd.DataFrame):
                        close_prices = close_prices.iloc[:, 0]
                    
                    # Handle zeros and missing values before log
                    close_prices = pd.to_numeric(close_prices, errors='coerce')
                    close_prices = close_prices.replace([0, np.inf, -np.inf], np.nan)
                    close_prices = close_prices.ffill().bfill()
                    close_prices = close_prices.clip(lower=1e-8)
                    
                    # Calculate returns safely
                    returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
                    returns = returns.replace([np.inf, -np.inf], 0)
                    returns_dict[asset] = returns
                    
                except Exception as e:
                    logger.warning(f"Could not calculate returns for {asset}: {str(e)}")
                    continue
            
            if not returns_dict:
                return {}
                
            returns_df = pd.DataFrame(returns_dict).fillna(0)
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
                    close_prices = full_df.loc[:, (asset, 'close')]
                    if isinstance(close_prices, pd.DataFrame):
                        close_prices = close_prices.iloc[:, 0]
                    
                    # Handle zeros and missing values before log
                    close_prices = pd.to_numeric(close_prices, errors='coerce')
                    close_prices = close_prices.replace([0, np.inf, -np.inf], np.nan)
                    close_prices = close_prices.ffill().bfill()
                    close_prices = close_prices.clip(lower=1e-8)
                    
                    # Calculate returns safely
                    returns = pd.Series(
                        np.log(close_prices / close_prices.shift(1)),
                        index=close_prices.index
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                    
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
                    # Fill any remaining NaN values with 0 before PCA
                    returns_df_filled = returns_df.fillna(0)
                    
                    # PCA decomposition
                    pca = PCA(n_components=n_components)
                    pca_features = pca.fit_transform(self.scaler.fit_transform(returns_df_filled))
                    
                    # Factor loadings
                    loadings = pca.components_[:, list(returns_dict.keys()).index(current_asset)]
                    for i, loading in enumerate(loadings):
                        features[f'factor_{i+1}_loading'] = pd.Series(loading, index=returns_df.index)
                except Exception as e:
                    logger.warning(f"Error in PCA calculation: {str(e)}")
            
            # Cross-sectional momentum
            try:
                # Calculate rolling means for each asset
                rolling_means = returns_df.rolling(5).mean()
                
                # Rank assets at each timestamp
                ranks = rolling_means.rank(axis=1)
                
                # Get the rank for the current asset
                features['xs_momentum'] = ranks[current_asset]
                
            except Exception as e:
                logger.warning(f"Error calculating cross-sectional momentum: {str(e)}")
                features['xs_momentum'] = pd.Series(0, index=returns_df.index)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in cross-sectional features: {str(e)}")
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
                                asset_df[feat_name] = pd.to_numeric(value, errors='coerce')
                        else:
                            # If feature doesn't exist for this asset, fill with 0
                            asset_df[feat_name] = 0.0
                            logger.debug(f"Adding missing feature {feat_name} for {asset}")
                    
                    if len(asset_df) > 0:
                        # Handle missing values properly
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
                    continue
            
            # Sort columns for consistency
            combined = combined.sort_index(axis=1)
            
            # Final verification
            logger.info(f"Combined features shape: {combined.shape}")
            logger.info(f"Features per asset: {len(self.selected_features) if self.selected_features else 0}")
            logger.info(f"Total assets: {len(features)}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in combine_features: {str(e)}")
            return pd.DataFrame()

    def engineer_features(self, data_dict):
        try:
            processed_data = {}
            for exchange, df in data_dict.items():
                # Check if df is a DataFrame and if it's empty
                if isinstance(df, pd.DataFrame) and df.empty:
                    logger.warning(f"Empty DataFrame for {exchange}, skipping")
                    continue
                # Check if df is a dict (handle this case properly)
                elif isinstance(df, dict) and not df:
                    logger.warning(f"Empty dict for {exchange}, skipping")
                    continue
                    
                # Ensure MultiIndex columns
                if not isinstance(df.columns, pd.MultiIndex):
                    logger.warning(f"Converting {exchange} columns to MultiIndex")
                    df.columns = pd.MultiIndex.from_product([[exchange], df.columns])
                
                # Process features
                processed = self.transform(df)
                if isinstance(processed, pd.DataFrame) and not processed.empty:
                    processed_data[exchange] = processed
                
            if not processed_data:
                raise ValueError("No data processed successfully")
            
            # Combine all exchanges
            combined = pd.concat(processed_data.values(), axis=1)
            
            # Final validation
            if combined.empty:
                raise ValueError("Combined DataFrame is empty")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            return pd.DataFrame()

    def _calculate_rolling_hurst(self, series, window):
        """Calculate Hurst exponent in a rolling window to identify trend strength"""
        series = series.fillna(method='ffill')
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
            
        # Forward fill initial NaN values
        result = result.fillna(method='ffill')
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