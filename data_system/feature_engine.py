import pandas as pd
import numpy as np
from scipy.stats import norm
from arch import arch_model
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DerivativesFeatureEngine:
    def __init__(self, 
                 volatility_window=10080, 
                 n_components=5,
                 feature_selection_threshold=0.01):
        self.volatility_window = volatility_window
        self.n_components = n_components
        self.feature_selection_threshold = feature_selection_threshold
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
        """Compute comprehensive technical indicators"""
        try:
            features = {}
            
            # Get price and volume data
            close = df['close']
            features['close'] = close
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Trend indicators
            features['ema_short'] = talib.EMA(close, timeperiod=12)
            features['ema_medium'] = talib.EMA(close, timeperiod=26)
            features['ema_long'] = talib.EMA(close, timeperiod=50)
            features['macd'], features['macd_signal'], _ = talib.MACD(close)
            
            # Enhanced momentum indicators
            features['rsi'] = talib.RSI(close)
            features['mom'] = talib.MOM(close, timeperiod=10)
            features['willr'] = talib.WILLR(high, low, close)
            features['roc'] = talib.ROC(close, timeperiod=10)
            features['ppo'] = talib.PPO(close)
            features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # Advanced trend indicators
            features['adx'] = talib.ADX(high, low, close, timeperiod=14)
            features['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Volatility indicators
            features['atr'] = talib.ATR(high, low, close)
            features['natr'] = talib.NATR(high, low, close)
            features['bbands_upper'], features['bbands_middle'], features['bbands_lower'] = \
                talib.BBANDS(close, timeperiod=20)
            
            # Volume and momentum indicators
            features['obv'] = talib.OBV(close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['ad'] = talib.AD(high, low, close, volume)
            
            # Cycle indicators
            features['ht_dcperiod'] = talib.HT_DCPERIOD(close)
            features['ht_dcphase'] = talib.HT_DCPHASE(close)
            features['ht_trendmode'] = talib.HT_TRENDMODE(close)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return {}
        
    def _add_vol_surface_features(self, df: pd.DataFrame) -> Dict:
        """Advanced volatility modeling and regime detection"""
        try:
            features = {}
            close_col = [col for col in df.columns if 'close' in col.lower()][0]
            returns = np.log(df[close_col]).diff().dropna()
            
            # Scale returns for ARCH model
            scaled_returns = returns * 100  # Scale up by 100 for better numerical stability
            
            # Multi-horizon volatility forecasting
            for horizon in [1, 5, 22]:
                try:
                    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    forecast = res.forecast(horizon=horizon)
                    features[f'vol_{horizon}d'] = np.sqrt(forecast.variance.iloc[-1]) / 100  # Scale back down
                except Exception as e:
                    logger.warning(f"Error in GARCH modeling for horizon {horizon}: {str(e)}")
                    features[f'vol_{horizon}d'] = returns.std()
            
            # Volatility regime detection using HMM
            try:
                vol_regime = self._detect_regime(returns)
                features['vol_regime'] = vol_regime
            except Exception as e:
                logger.warning(f"Error in regime detection: {str(e)}")
                features['vol_regime'] = 0
            
            # Term structure features
            if all(k in features for k in ['vol_1d', 'vol_5d', 'vol_22d']):
                features['vol_term_struct'] = features['vol_22d'] / features['vol_1d']
                features['vol_curvature'] = (features['vol_5d'] - features['vol_1d']) / \
                                          (features['vol_22d'] - features['vol_5d'])
            
            return features
            
        except Exception as e:
            logger.error(f"Error in volatility features: {str(e)}")
            return {}
        
    def _add_flow_features(self, df: pd.DataFrame) -> Dict:
        """Order flow and market microstructure features"""
        features = {}
        
        # Volume profile
        vwap = (df['close'] * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum()
        features['vwap_divergence'] = (df['close'] - vwap) / vwap
        
        # Order book pressure
        if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
            features['book_pressure'] = (df['bid_depth'] - df['ask_depth']) / \
                                      (df['bid_depth'] + df['ask_depth'])
            
        # Funding rate dynamics
        if 'funding_rate' in df.columns:
            features['funding_zscore'] = (
                df['funding_rate'] - df['funding_rate'].rolling(24).mean()
            ) / df['funding_rate'].rolling(24).std()
            
        # Liquidation risk metrics
        if 'liquidations' in df.columns:
            features['liq_intensity'] = df['liquidations'].rolling(12).sum() / \
                                      df['liquidations'].rolling(48).sum()
            
        return features
        
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
                    close_prices = full_df.xs(asset, level='asset')['close']
                    returns_dict[asset] = np.log(close_prices).diff()
                except Exception as e:
                    logger.warning(f"Could not calculate returns for {asset}: {str(e)}")
                    continue
            
            if not returns_dict:
                logger.warning("No valid returns data for correlation calculation")
                return {}
                
            returns_df = pd.DataFrame(returns_dict)
            current_returns = returns_df[current_asset]
            
            # Rolling correlations with other assets
            for asset in returns_df.columns:
                if asset != current_asset:
                    try:
                        corr = returns_df[asset].rolling(window).corr(current_returns)
                        features[f'corr_{asset}'] = corr
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation between {current_asset} and {asset}: {str(e)}")
                        features[f'corr_{asset}'] = pd.Series(0, index=returns_df.index)
            
            # Average correlation
            correlations = [features[f'corr_{asset}'] for asset in returns_df.columns if asset != current_asset]
            if correlations:
                features['avg_correlation'] = pd.concat(correlations, axis=1).mean(axis=1)
            else:
                features['avg_correlation'] = pd.Series(0, index=returns_df.index)
            
            # Correlation regime
            features['correlation_regime'] = (
                features['avg_correlation'] - 
                pd.Series(features['avg_correlation']).rolling(window*2).mean()
            ).fillna(0)
            
            # Beta to market (using average returns as market proxy)
            market_returns = returns_df.mean(axis=1)
            try:
                features['market_beta'] = (
                    returns_df[current_asset].rolling(window).cov(market_returns) /
                    market_returns.rolling(window).var()
                ).fillna(0)
            except Exception as e:
                logger.warning(f"Could not calculate market beta for {current_asset}: {str(e)}")
                features['market_beta'] = pd.Series(0, index=returns_df.index)
            
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
                close_col = [col for col in full_df[asset].columns if 'close' in col.lower()][0]
                returns_dict[asset] = np.log(full_df[asset][close_col]).diff()
            
            returns_df = pd.DataFrame(returns_dict)
            
            # Determine number of components based on available data
            n_samples, n_features = returns_df.shape
            n_components = min(self.n_components, min(n_samples, n_features) - 1)
            
            if n_components > 0:
                # PCA decomposition
                pca = PCA(n_components=n_components)
                pca_features = pca.fit_transform(
                    self.scaler.fit_transform(returns_df.fillna(0))
                )
                
                # Factor loadings
                loadings = pca.components_[:, list(returns_dict.keys()).index(current_asset)]
                for i, loading in enumerate(loadings):
                    features[f'factor_{i+1}_loading'] = loading
            
            # Cross-sectional momentum
            try:
                # Calculate rolling means for each asset
                rolling_means = returns_df.rolling(5).mean()
                
                # Rank assets at each timestamp
                ranks = rolling_means.rank(axis=1)
                
                # Get the rank for the current asset
                features['xs_momentum'] = ranks[current_asset].iloc[-1]
                
            except Exception as e:
                logger.warning(f"Error calculating cross-sectional momentum: {str(e)}")
                features['xs_momentum'] = 0
            
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
        
    def _select_features(self, df: pd.DataFrame, target_col: str = 'close') -> List[str]:
        """Select most important features using mutual information"""
        try:
            # Prepare data
            X = df.copy()
            y = X[target_col].pct_change().shift(-1)  # Future returns as target
            X = X.iloc[:-1]  # Remove last row as we don't have target for it
            y = y.iloc[:-1]
            
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X.fillna(0), y)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(X.columns, mi_scores))
            
            # Select features above threshold
            selected = [feat for feat, score in feature_importance.items() 
                       if score > self.feature_selection_threshold]
            
            logger.info(f"Selected {len(selected)} features out of {len(X.columns)}")
            return selected
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return list(df.columns)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values more intelligently"""
        try:
            # Forward fill first (for time series consistency)
            df = df.fillna(method='ffill')
            
            # For any remaining NaNs, use rolling median
            window_size = min(24, len(df) // 2)  # Use smaller of 24 periods or half the data
            df = df.fillna(df.rolling(window=window_size, min_periods=1).median())
            
            # If still any NaNs, fill with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df.fillna(0)

    def _combine_features(self, features: Dict) -> pd.DataFrame:
        """Combine all features into a single DataFrame with proper normalization"""
        try:
            combined = pd.DataFrame()
            all_features = set()
            
            # First pass: collect all feature names
            for asset, asset_features in features.items():
                for feat_name, feat_values in asset_features.items():
                    if isinstance(feat_values, (pd.Series, np.ndarray)):
                        all_features.add(feat_name)
            
            logger.info(f"Total features to process: {len(all_features)}")
            
            # Second pass: ensure all assets have all features
            for asset, asset_features in features.items():
                try:
                    # Convert all features to DataFrame with proper types
                    asset_df = pd.DataFrame()
                    
                    # Process each feature
                    for feat_name in all_features:
                        if feat_name in asset_features:
                            value = asset_features[feat_name]
                            if isinstance(value, (pd.Series, np.ndarray)):
                                asset_df[feat_name] = pd.to_numeric(value, errors='coerce')
                        else:
                            # If feature doesn't exist for this asset, fill with 0
                            asset_df[feat_name] = 0.0
                            logger.debug(f"Adding missing feature {feat_name} for {asset}")
                    
                    if len(asset_df) > 0:
                        # Handle missing values properly
                        asset_df = self._handle_missing_values(asset_df)
                        
                        # Select important features if not already selected
                        if self.selected_features is None:
                            self.selected_features = self._select_features(asset_df)
                        
                        # Keep only selected features
                        asset_df = asset_df[self.selected_features]
                        
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
            
            # Verify all assets have the same features
            assets = combined.columns.get_level_values('asset').unique()
            features = combined.columns.get_level_values('feature').unique()
            
            for asset in assets:
                asset_features = combined.xs(asset, axis=1, level='asset').columns
                missing_features = set(features) - set(asset_features)
                if missing_features:
                    logger.warning(f"Asset {asset} is missing features: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        combined[(asset, feature)] = 0.0
            
            # Sort columns for consistency
            combined = combined.sort_index(axis=1)
            
            # Final verification
            logger.info(f"Combined features shape: {combined.shape}")
            logger.info(f"Features per asset: {len(features)}")
            logger.info(f"Total assets: {len(assets)}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in combine_features: {str(e)}")
            return pd.DataFrame()

    def engineer_features(self, df: Dict) -> pd.DataFrame:
        """Main feature engineering method that processes nested exchange data structure"""
        try:
            processed_data = {}
            
            for exchange, symbols in df.items():
                for symbol, data in symbols.items():
                    if not isinstance(data, pd.DataFrame):
                        logger.warning(f"Invalid data type for {exchange}:{symbol}")
                        continue
                        
                    # Ensure all required columns exist
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    if not all(col in data.columns for col in required_columns):
                        logger.warning(f"Missing required columns in {exchange}:{symbol}")
                        continue
                        
                    # Process each symbol's data
                    try:
                        symbol_data = data.copy()
                        for col in symbol_data.columns:
                            symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')

                        key = symbol  # use just the symbol as key
                        if key in processed_data:
                            logger.info(f"Duplicate symbol {symbol} found from {exchange}, skipping.")
                            continue
                        processed_data[key] = symbol_data
                    except Exception as e:
                        logger.error(f"Error processing {exchange}:{symbol}: {str(e)}")
                        continue
            
            if not processed_data:
                raise ValueError("No valid data found in input")
            
            # Combine data for each symbol
            combined_data = {}
            for symbol, symbol_data in processed_data.items():
                try:
                    # Group by timestamp and aggregate
                    agg_dict = {
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'funding_rate': 'mean',
                        'bid_depth': 'mean',
                        'ask_depth': 'mean'
                    }
                    
                    # Remove any columns not in agg_dict
                    valid_columns = [col for col in symbol_data.columns if col in agg_dict]
                    symbol_data = symbol_data[valid_columns]
                    
                    # Aggregate data
                    aggregated = symbol_data.groupby(symbol_data.index).agg(
                        {col: agg_dict[col] for col in valid_columns}
                    )
                    
                    # Create MultiIndex columns
                    aggregated.columns = pd.MultiIndex.from_product(
                        [[symbol], aggregated.columns],
                        names=['asset', 'feature']
                    )
                    
                    combined_data[symbol] = aggregated
                    
                except Exception as e:
                    logger.error(f"Error combining data for {symbol}: {str(e)}")
                    continue
            
            if not combined_data:
                raise ValueError("No data after combining")
            
            # Combine all symbols into final DataFrame
            final_df = pd.concat(combined_data.values(), axis=1)
            
            # Fill any NaN values
            final_df = final_df.fillna(0)
            
            logger.info(f"Successfully processed data for assets: {list(combined_data.keys())}")
            
            # Apply feature transformation
            transformed_df = self.transform(final_df)
            
            # Ensure all data is numeric
            for col in transformed_df.columns:
                if not np.issubdtype(transformed_df[col].dtype, np.number):
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
            
            # Fill any remaining NaN values
            transformed_df = transformed_df.fillna(0)
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(f"Data structure: {df.keys() if isinstance(df, dict) else 'Not a dict'}")
            return pd.DataFrame()