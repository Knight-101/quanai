import pandas as pd
import numpy as np
from scipy.stats import norm
from arch import arch_model
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DerivativesFeatureEngine:
    def __init__(self, volatility_window=10080, n_components=5):
        self.volatility_window = volatility_window
        self.n_components = n_components
        self.scaler = StandardScaler()
        self._setup_neural_features()
        
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
            
            # Momentum indicators
            features['rsi'] = talib.RSI(close)
            features['mom'] = talib.MOM(close, timeperiod=10)
            features['willr'] = talib.WILLR(high, low, close)
            
            # Volatility indicators
            features['atr'] = talib.ATR(high, low, close)
            features['bbands_upper'], features['bbands_middle'], features['bbands_lower'] = \
                talib.BBANDS(close, timeperiod=20)
                
            # Volume indicators
            features['obv'] = talib.OBV(close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            
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
        
    def _combine_features(self, features: Dict) -> pd.DataFrame:
        """Combine all features into a single DataFrame with proper normalization"""
        try:
            combined = pd.DataFrame()
            
            for asset, asset_features in features.items():
                try:
                    # Convert all features to DataFrame with proper types
                    asset_df = pd.DataFrame()
                    for feat_name, feat_values in asset_features.items():
                        # Convert to numeric, replacing non-numeric values with NaN
                        if isinstance(feat_values, (pd.Series, np.ndarray)):
                            asset_df[feat_name] = pd.to_numeric(feat_values, errors='coerce')
                        else:
                            # Skip non-numeric features
                            logger.warning(f"Skipping non-numeric feature {feat_name} for {asset}")
                            continue
                    
                    if len(asset_df) > 0:
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
            
            # Fill any remaining NaN values with 0
            combined = combined.fillna(0)
            
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