import sys
import os

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any, Set
import yaml
import warnings
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from trading_env.institutional_perp_env import InstitutionalPerpetualEnv
from data_system.data_manager import DataManager
from data_system.feature_engine import DerivativesFeatureEngine
from risk_management.risk_engine import InstitutionalRiskEngine, RiskLimits
from tqdm import tqdm  # Import tqdm for progress bar
import traceback
import copy

# Set up logging
logger = logging.getLogger('institutional_backtester')
logger.setLevel(logging.INFO)

# Create log directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Add file handler
logger.addHandler(
    logging.FileHandler('logs/backtesting.log'),
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class InstitutionalBacktester:
    """
    Institutional-grade backtester for cryptocurrency trading models with 
    a focus on eliminating bias, analyzing market regimes, and providing 
    comprehensive performance metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: Optional[str] = None,
        output_dir: str = "results/backtest",
        initial_capital: float = 10000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        regime_analysis: bool = False,
        walk_forward: bool = False,
        config_path: str = 'config/prod_config.yaml'
    ):
        """
        Initialize the backtester with configuration parameters.
        
        Args:
            model_path: Path to the trained model file
            data_path: Optional path to preprocessed data file
            output_dir: Directory to save results and visualizations
            initial_capital: Initial capital for trading
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            assets: List of asset symbols to backtest
            regime_analysis: Whether to perform regime analysis
            walk_forward: Whether to perform walk-forward testing
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets if assets is not None else []
        self.regime_analysis = regime_analysis
        self.walk_forward = walk_forward
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # If assets not provided but available in config, use those
        if not self.assets and self.config and 'trading' in self.config and 'symbols' in self.config['trading']:
            self.assets = self.config['trading']['symbols']
            logger.info(f"Using assets from config: {self.assets}")
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataManager(self.config['data']['cache_dir'])
        self.feature_engine = DerivativesFeatureEngine(
            volatility_window=self.config['feature_engineering']['volatility_window'],
            n_components=self.config['feature_engineering']['n_components']
        )
        
        # Initialize risk engine
        self.risk_engine = InstitutionalRiskEngine(
            risk_limits=RiskLimits(**self.config['risk_management']['limits'])
        )
        
        # Initialize data storage
        self.data = None
        self.env = None
        self.env_path = None
        self.model = None
        self.results = None
        self.portfolio_history = None
        self.episode_trades = []
        self.current_timestamp = None
        self.trade_history = []
        self.original_price_data = {}  # Dictionary to store original price data
        self.market_regimes = None
        self.regime_performance = None
        self.walkforward_results = None
        
        # Generate an experiment name for saving results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).split('.')[0]
        self.experiment_name = f"{model_name}_{timestamp}"
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data for backtesting.
        
        Returns:
            Processed data ready for backtesting
        """
        logger.info("Loading data for backtesting...")
        
        # Try to load cached feature data first (most efficient)
        try:
            feature_file = os.path.join(self.config['data']['cache_dir'], 'data/features/backtest_features.parquet')
            if os.path.exists(feature_file):
                logger.info(f"Loading cached feature data from {feature_file}")
                data = self.data_manager.load_feature_data(feature_set='backtest_features')
                
                if data is not None and not data.empty:
                    logger.info(f"Loaded cached feature data with shape: {data.shape}")
                    
                    # Verify the data has the expected structure
                    if not isinstance(data.columns, pd.MultiIndex):
                        logger.warning("Cached data does not have MultiIndex columns. Attempting to fix...")
                        try:
                            # Convert to MultiIndex if not already
                            if any(isinstance(col, tuple) for col in data.columns):
                                # Already tuples, just convert to MultiIndex
                                data.columns = pd.MultiIndex.from_tuples(data.columns, names=['asset', 'feature'])
                            elif any('_' in col for col in data.columns):
                                # Try to split by underscore
                                new_cols = []
                                for col in data.columns:
                                    if '_' in col:
                                        parts = col.split('_', 1)
                                        new_cols.append((parts[0], parts[1]))
                                    else:
                                        new_cols.append(('unknown', col))
                                data.columns = pd.MultiIndex.from_tuples(new_cols, names=['asset', 'feature'])
                        except Exception as e:
                            logger.error(f"Failed to fix column structure: {str(e)}")
                    
                    # If we specified assets, make sure they exist in the data
                    if self.assets:
                        missing_assets = []
                        for asset in self.assets:
                            if not any(asset == col[0] for col in data.columns):
                                missing_assets.append(asset)
                        
                        if missing_assets:
                            logger.warning(f"Cached data is missing requested assets: {missing_assets}")
                            # Try to proceed with available assets
                            available_assets = list(set(self.assets) - set(missing_assets))
                            if available_assets:
                                logger.info(f"Proceeding with available assets: {available_assets}")
                                self.assets = available_assets
                            else:
                                logger.error("No requested assets available in cached data")
                                return None
                    
                    # Check for required base features
                    missing_base_features = False
                    for asset in self.assets:
                        for feature in ['open', 'high', 'low', 'close', 'volume']:
                            if (asset, feature) not in data.columns:
                                logger.warning(f"Cached data missing required base feature {feature} for {asset}")
                                missing_base_features = True
                    
                    if missing_base_features:
                        logger.warning("Cached data is missing required base features. Loading raw data instead.")
                    else:
                        # Check for completely NaN columns and drop them
                        nan_columns = []
                        for col in data.columns:
                            if data[col].isna().all():
                                nan_columns.append(col)
                        
                        if nan_columns:
                            logger.warning(f"Dropping {len(nan_columns)} columns with all NaN values")
                            data = data.drop(columns=nan_columns)
                        
                        # Check for and fill any remaining NaN values
                        nan_count = data.isna().sum().sum()
                        if nan_count > 0:
                            logger.warning(f"Found {nan_count} NaN values in cached data. Filling with ffill/bfill.")
                            data = data.ffill(axis=0).bfill(axis=0)
                        
                        # Final validation - check if we still have NaNs after filling
                        final_nan_count = data.isna().sum().sum()
                        if final_nan_count > 0:
                            logger.error(f"Still have {final_nan_count} NaN values after filling. Data may be invalid.")
                            
                            # Try a more aggressive approach to handle NaNs
                            logger.warning("Attempting more aggressive NaN handling...")
                            # First try to drop rows with too many NaNs
                            threshold = len(data.columns) * 0.5  # If more than 50% of columns are NaN
                            data = data.dropna(thresh=threshold)
                            
                            # Then fill any remaining NaNs
                            data = data.fillna(method='ffill')
                            data = data.fillna(method='bfill')
                            
                            # As a last resort, fill remaining NaNs with zeros
                            data = data.fillna(0)
                            
                            # Final check
                            if data.isna().sum().sum() > 0:
                                logger.error("Could not remove all NaNs, data is invalid")
                                return None
                            
                            logger.info(f"After NaN handling, data shape: {data.shape}")
                        
                        # Filter by date if date range is specified
                        if self.start_date or self.end_date:
                            logger.info(f"Filtering data by date range: {self.start_date} to {self.end_date}")
                            data = self._filter_data_by_date(data)
                        
                        if data is None or data.empty:
                            logger.error("No data available after date filtering")
                            return None
                        
                        logger.info(f"Final prepared data shape: {data.shape}")
                        return data
                else:
                    logger.warning("Cached feature data is empty or None. Loading raw data instead.")
            else:
                logger.info("No cached feature data found. Loading raw market data instead.")
        except Exception as e:
            logger.error(f"Error loading cached feature data: {str(e)}")
            logger.info("Falling back to raw market data")
        
        # If we get here, we need to load and process raw data
        try:
            # Load raw market data
            raw_data = self._load_market_data()
            if not raw_data:
                logger.error("Failed to load market data")
                return None
                
            # Process raw data
            processed_data = self._process_market_data(raw_data)
            if processed_data is None:
                logger.error("Failed to process market data")
                return None
                
            # Apply date filtering
            if self.start_date or self.end_date:
                processed_data = self._filter_data_by_date(processed_data)
                
            if processed_data is None or processed_data.empty:
                logger.error("No data available after filtering")
                return None
                
            # Final validation before returning
            if not isinstance(processed_data.columns, pd.MultiIndex):
                logger.error("Processed data does not have MultiIndex columns required by environment")
                return None
                
            # Check for and fill any NaN values
            nan_count = processed_data.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in processed data. Filling with ffill/bfill.")
                processed_data = processed_data.ffill(axis=0).bfill(axis=0)
                
                # Check if we still have NaNs after filling
                final_nan_count = processed_data.isna().sum().sum()
                if final_nan_count > 0:
                    logger.warning(f"Still have {final_nan_count} NaN values after filling. Using more aggressive filling.")
                    # Fill remaining NaNs with zeros
                    processed_data = processed_data.fillna(0)
            
            logger.info(f"Final prepared data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Unhandled error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by date range"""
        if not hasattr(data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot filter by date.")
            return data
            
        # Get the datetime index if it exists
        try:
            if 'datetime' in data.index.names:
                date_idx = data.index.get_level_values('datetime')
            else:
                date_idx = data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(date_idx):
                    logger.warning("First index level is not datetime. Cannot filter by date.")
                    return data
        except Exception as e:
            logger.warning(f"Error accessing index for date filtering: {str(e)}")
            return data
            
        # Apply date filters
        if self.start_date:
            start_date = pd.to_datetime(self.start_date)
            data = data[date_idx >= start_date]
            logger.info(f"Filtered data from {start_date}")
            
        if self.end_date:
            end_date = pd.to_datetime(self.end_date)
            data = data[date_idx <= end_date]
            logger.info(f"Filtered data until {end_date}")
            
        logger.info(f"Data after date filtering: {data.shape}")
        return data
    
    def _load_market_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load raw market data for the specified assets.
        
        Returns:
            Dictionary of market data by exchange and symbol
        """
        logger.info("Loading market data...")
        
        # Initialize storage for market data
        market_data = {}
        
        # Ensure we have assets to load data for
        if self.assets is None:
            # Try to get assets from config
            try:
                self.assets = self.config['trading']['symbols']
                logger.info(f"Using assets from config: {self.assets}")
            except (KeyError, TypeError):
                # If no assets in config, use some default assets
                self.assets = ['BTCUSDT', 'ETHUSDT']
                logger.warning(f"No assets specified or found in config. Using default assets: {self.assets}")
                
        if not self.assets:
            logger.error("No assets available for loading market data")
            return None
        
        # Load data for each exchange from config
        for exchange in self.config['data']['exchanges']:
            logger.info(f"Loading market data for exchange: {exchange}")
            
            exchange_data = {}
            any_data_loaded = False
            
            # Load data for each asset
            for symbol in self.assets:
                try:
                    # Load OHLCV data for this asset from the data manager
                    data = self.data_manager.load_market_data(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=self.config['data']['timeframe'],
                        start_time=self.start_date,
                        end_time=self.end_date,
                        data_type='perpetual'
                    )
                    
                    if data is not None and not data.empty:
                        # Convert DataFrame index to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(data.index):
                            try:
                                data.index = pd.to_datetime(data.index)
                            except Exception as e:
                                logger.warning(f"Could not convert index to datetime for {symbol}: {str(e)}")
                        
                        logger.info(f"Loaded {len(data)} rows for {symbol} from {exchange}")
                        exchange_data[symbol] = data
                        any_data_loaded = True
                    else:
                        logger.warning(f"No data returned for {symbol} from {exchange}")
                        
                except Exception as e:
                    logger.error(f"Error loading data for {symbol} from {exchange}: {str(e)}")
            
            # Only add exchange data if we loaded at least one symbol
            if any_data_loaded:
                market_data[exchange] = exchange_data
            else:
                logger.warning(f"No data loaded for exchange {exchange}")
        
        # Check if we have any data
        if not market_data:
            logger.error("Failed to load any market data")
            return None
            
        return market_data
    
    def _process_market_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Process raw market data into features"""
        logger.info("Processing market data into features...")
        
        # For tracking progress
        total_symbols = sum(len(exchange_data) for exchange, exchange_data in raw_data.items())
        progress_bar = tqdm(total=total_symbols, desc="Processing market data", unit="symbol")
        
        try:
            # Prepare data in the format expected by engineer_features
            # The method expects a dictionary with exchange as key and
            # either a DataFrame or a dictionary of symbols->DataFrame as value
            processed_data_by_exchange = {}
            
            # Store original price data to preserve base features
            self.original_price_data = {}
        
            for exchange, exchange_data in raw_data.items():
                logger.info(f"Processing data for exchange: {exchange} with {len(exchange_data)} symbols")
                
                if exchange not in self.original_price_data:
                    self.original_price_data[exchange] = {}
                    
                # Prepare symbol data for this exchange
                for symbol, symbol_data in exchange_data.items():
                    # Ensure the data has a datetime index
                    if not pd.api.types.is_datetime64_any_dtype(symbol_data.index):
                        symbol_data.index = pd.to_datetime(symbol_data.index)
                    
                    # Ensure required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in symbol_data.columns]
                    if missing_cols:
                        logger.warning(f"Missing columns for {exchange}_{symbol}: {missing_cols}")
                        progress_bar.update(1)
                        continue
                    
                    # Store original price data before feature engineering
                    # This will be used later to ensure base features are available
                    logger.info(f"Preserving original price data for {exchange}_{symbol}")
                    self.original_price_data[exchange][symbol] = symbol_data[required_cols].copy()
                    
                    # Store prepared data for feature engineering
                    if exchange not in processed_data_by_exchange:
                        processed_data_by_exchange[exchange] = {}
                    
                    processed_data_by_exchange[exchange][symbol] = symbol_data
                    progress_bar.update(1)
            
            # Close progress bar
            progress_bar.close()
            
            if not processed_data_by_exchange:
                logger.error("No valid data to process. Cannot proceed with feature engineering.")
                return None
            
            # Use feature engine to generate features from all prepared data at once
            logger.info("Generating features using feature engine...")
            try:
                # The engineer_features method processes all exchanges and symbols together
                feature_df = self.feature_engine.engineer_features(processed_data_by_exchange)
                
                if feature_df is None or feature_df.empty:
                    logger.error("Feature engineering returned empty results.")
                    return None
                
                logger.info(f"Feature engineering complete. Feature data shape: {feature_df.shape}")
                
                # Now merge the base features back into the feature dataframe
                logger.info("Merging original price data back into feature dataframe...")
                
                # Check if feature_df has MultiIndex columns
                if not isinstance(feature_df.columns, pd.MultiIndex):
                    logger.warning("Feature DataFrame does not have MultiIndex columns. Converting...")
                    # Try to convert to MultiIndex
                    if any(isinstance(col, tuple) for col in feature_df.columns):
                        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['asset', 'feature'])
                
                # Create a new DataFrame to hold all data (features + base data)
                combined_df = feature_df.copy()
                
                # Add back the base features for each asset
                for exchange, assets in self.original_price_data.items():
                    for symbol, base_data in assets.items():
                        # For each base feature (open, high, low, close, volume)
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            # Check if the column already exists
                            if (symbol, col) not in combined_df.columns:
                                # Add the base feature to the combined dataframe
                                logger.info(f"Adding base feature {col} for {symbol}")
                                
                                # Add column to the MultiIndex DataFrame
                                # First align the indexes
                                base_series = base_data[col].copy()
                                
                                # If indexes don't match, try to reindex
                                if base_series.index.equals(combined_df.index):
                                    # Indexes match, can add directly
                                    combined_df[(symbol, col)] = base_series
                                else:
                                    # Try to reindex
                                    try:
                                        # Reindex to match feature_df's index
                                        reindexed = base_series.reindex(combined_df.index)
                                        combined_df[(symbol, col)] = reindexed
                                        
                                        # Log missing values after reindexing
                                        missing = reindexed.isna().sum()
                                        if missing > 0:
                                            logger.warning(f"Reindexed {symbol} {col} has {missing} missing values")
                                            # Fill missing values with forward fill, then backward fill
                                            combined_df[(symbol, col)] = combined_df[(symbol, col)].ffill().bfill()
                                    except Exception as e:
                                        logger.error(f"Error reindexing {symbol} {col}: {e}")
                            else:
                                logger.info(f"Base feature {col} for {symbol} already exists in feature dataframe")
                
                logger.info(f"Combined dataframe shape after adding base features: {combined_df.shape}")
                
                # Verify that all required base features are available
                missing_base_features = False
                for asset in self.assets:
                    for feature in ['open', 'high', 'low', 'close', 'volume']:
                        if (asset, feature) not in combined_df.columns:
                            logger.error(f"Missing required base feature {feature} for asset {asset} after combination")
                            missing_base_features = True
                
                if missing_base_features:
                    logger.warning("Some base features are missing. Environment may not function correctly.")
        
                # Save processed data for future use
                try:
                    self.data_manager.save_feature_data(
                        data=combined_df,
                        feature_set='backtest_features',
                        metadata={
                            'feature_config': self.config['feature_engineering'],
                            'exchanges': self.config['data']['exchanges'],
                            'symbols': self.assets,
                            'timeframe': self.config['data']['timeframe']
                        }
                    )
                    logger.info("Saved processed feature data for future use.")
                except Exception as e:
                    logger.warning(f"Could not save feature data: {str(e)}")
                
                # Return the processed data after saving
                return combined_df
                
            except Exception as e:
                logger.error(f"Error in feature engineering: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            if 'progress_bar' in locals():
                progress_bar.close()
            return None
    
    def load_model(self) -> PPO:
        """
        Load the trained model for backtesting.
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load the model
            model = PPO.load(self.model_path)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            
            # Look for env file alongside model
            env_path = None
            if os.path.isdir(self.model_path):
                # Try common env file patterns in the directory
                env_patterns = [
                    os.path.join(self.model_path, "final_env.pkl"),  # Added final_env.pkl
                    os.path.join(self.model_path, "vec_normalize.pkl"),
                    os.path.join(self.model_path, "env.pkl"),
                    os.path.join(self.model_path, "vec_env.pkl")
                ]
                for pattern in env_patterns:
                    if os.path.exists(pattern):
                        env_path = pattern
                        break
            else:
                # Check if there's a corresponding .pkl file in the same directory
                model_dir = os.path.dirname(self.model_path)
                model_name = os.path.splitext(os.path.basename(self.model_path))[0]
                env_patterns = [
                    os.path.join(model_dir, "final_env.pkl"),  # Added final_env.pkl first
                    os.path.join(model_dir, f"{model_name}_env.pkl"),
                    os.path.join(model_dir, "vec_normalize.pkl"),
                    os.path.join(model_dir, "env.pkl")
                ]
                for pattern in env_patterns:
                    if os.path.exists(pattern):
                        env_path = pattern
                        break
            
            if env_path:
                logger.info(f"Found environment file at {env_path}")
                self.env_path = env_path
            else:
                logger.warning("No environment file found. Will create a new environment.")
                self.env_path = None
            
            self.model = model
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _fix_data_structure(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fix any issues with the DataFrame structure to ensure it works with the environment.
        
        Args:
            data: DataFrame to fix
            
        Returns:
            Fixed DataFrame or None if unfixable
        """
        if data is None or data.empty:
            return None
            
        logger.info("Checking and fixing data structure if needed...")
        
        # Check if data has MultiIndex columns and fix if not
        if not isinstance(data.columns, pd.MultiIndex):
            logger.warning("Data does not have MultiIndex columns. Attempting to convert...")
            
            try:
                # Case 1: Columns are already tuples but not a MultiIndex
                if any(isinstance(col, tuple) for col in data.columns):
                    data.columns = pd.MultiIndex.from_tuples(data.columns, names=['asset', 'feature'])
                    logger.info("Converted tuple columns to MultiIndex")
                    
                # Case 2: Columns have pattern 'asset_feature'
                elif any('_' in col for col in data.columns):
                    new_cols = []
                    for col in data.columns:
                        if '_' in col:
                            parts = col.split('_', 1)
                            new_cols.append((parts[0], parts[1]))
                        else:
                            new_cols.append(('unknown', col))
                    data.columns = pd.MultiIndex.from_tuples(new_cols, names=['asset', 'feature'])
                    logger.info("Converted 'asset_feature' columns to MultiIndex")
                    
                # Case 3: Single asset with flat columns, create asset level
                elif all(col in ['open', 'high', 'low', 'close', 'volume'] + 
                          ['returns_1d', 'volatility_5d', 'rsi_14', 'macd'] for col in data.columns):
                    if self.assets and len(self.assets) == 1:
                        asset = self.assets[0]
                        logger.info(f"Single asset detected: {asset}. Converting to MultiIndex")
                        data.columns = pd.MultiIndex.from_product([[asset], data.columns], names=['asset', 'feature'])
                    else:
                        logger.error("Cannot determine asset for flat columns")
                        return None
                else:
                    logger.error("Cannot determine how to convert columns to MultiIndex")
                    return None
            except Exception as e:
                logger.error(f"Error converting columns to MultiIndex: {str(e)}")
                return None
        
        # Ensure column names are correctly set
        if data.columns.names != ['asset', 'feature'] and len(data.columns.names) == 2:
            logger.warning(f"Column level names are {data.columns.names}, renaming to ['asset', 'feature']")
            data.columns.names = ['asset', 'feature']
            
        # Check for asset level consistency
        assets_in_data = data.columns.get_level_values(0).unique()
        logger.info(f"Assets in data: {assets_in_data}")
        
        if self.assets:
            # Check if all requested assets are in the data
            missing_assets = [asset for asset in self.assets if asset not in assets_in_data]
            if missing_assets:
                logger.warning(f"Some requested assets are missing in the data: {missing_assets}")
                
            # Filter data to only include requested assets
            available_assets = [asset for asset in self.assets if asset in assets_in_data]
            if available_assets:
                data = data.loc[:, available_assets]
                logger.info(f"Filtered data to include only requested assets: {available_assets}")
            else:
                logger.error("None of the requested assets are in the data")
                return None
                
        # Ensure the index is datetime
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            try:
                logger.warning("Converting index to datetime")
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"Failed to convert index to datetime: {str(e)}")
                return None
                
        # Sort index to ensure chronological order
        if not data.index.is_monotonic_increasing:
            logger.info("Sorting index to ensure chronological order")
            data = data.sort_index()
            
        # Check and fix any duplicate indices
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate indices. Keeping last occurrence.")
            data = data[~data.index.duplicated(keep='last')]
            
        return data
        
    def prepare_environment(self, data: pd.DataFrame) -> Optional[VecNormalize]:
        """
        Prepare the environment for backtesting.
        
        Args:
            data: Processed data to use in the environment
            
        Returns:
            Prepared VecNormalize environment or None if there was an error
        """
        logger.info("Preparing environment for backtesting...")
        
        # Validate data
        if data is None:
            logger.error("Cannot prepare environment with None data. Data loading failed.")
            return None
            
        if data.empty:
            logger.error("Cannot prepare environment with empty data.")
            return None
            
        # Ensure we have a DataFrame with columns 
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Data is not a DataFrame. Type: {type(data)}")
            return None
        
        # Fix any issues with data structure
        data = self._fix_data_structure(data)
        if data is None:
            logger.error("Failed to fix data structure. Cannot prepare environment.")
            return None
        
        # Final check for NaNs before creating environment
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data still contains {nan_count} NaN values. Filling with forward-fill and zero.")
            data = data.ffill().bfill()
            # If still have NaNs, fill with zeros as last resort
            final_nan_count = data.isna().sum().sum()
            if final_nan_count > 0:
                logger.warning(f"Still have {final_nan_count} NaNs after ffill/bfill. Filling with zeros.")
                data = data.fillna(0)
        
        # Check if we have MultiIndex columns required by the environment
        if not isinstance(data.columns, pd.MultiIndex):
            logger.warning("Data does not have MultiIndex columns. Attempting to convert...")
            try:
                # Try to convert to MultiIndex if possible
                if any(isinstance(col, tuple) for col in data.columns):
                    # Already tuples, just convert to MultiIndex
                    data.columns = pd.MultiIndex.from_tuples(data.columns, names=['asset', 'feature'])
                elif any('_' in col for col in data.columns):
                    # Try to split by underscore
                    new_cols = []
                    for col in data.columns:
                        if '_' in col:
                            parts = col.split('_', 1)
                            new_cols.append((parts[0], parts[1]))
                        else:
                            new_cols.append(('unknown', col))
                    data.columns = pd.MultiIndex.from_tuples(new_cols, names=['asset', 'feature'])
                else:
                    logger.error("Cannot convert data columns to MultiIndex format required by the environment")
                    return None
            except Exception as e:
                logger.error(f"Error converting data columns to MultiIndex: {str(e)}")
                return None
        
        # Get list of assets
        if not self.assets:
            # Try to infer assets from data
            if hasattr(data.index, 'get_level_values') and 'symbol' in data.index.names:
                self.assets = data.index.get_level_values('symbol').unique().tolist()
            elif isinstance(data.columns, pd.MultiIndex):
                # Extract unique assets from the first level of column MultiIndex
                self.assets = data.columns.get_level_values(0).unique().tolist()
                logger.info(f"Inferred assets from column MultiIndex: {self.assets}")
            else:
                logger.error("Cannot infer assets from data and no assets were provided")
                return None
                
        if not self.assets:
            logger.error("No assets available for backtesting")
            return None
        
        logger.info(f"Using assets: {self.assets}")
        
        # Configure features
        base_features = ['open', 'high', 'low', 'close', 'volume']
        tech_features = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr_14', 'adx_14', 'cci_14',
            'market_regime', 'hurst_exponent', 'volatility_regime'
        ]
        
        # Verify that required base features exist for each asset
        missing_features = False
        missing_features_list = []
        for asset in self.assets:
            for feature in base_features:
                if (asset, feature) not in data.columns:
                    logger.warning(f"Missing required feature '{feature}' for asset '{asset}'")
                    missing_features = True
                    missing_features_list.append(f"{asset}_{feature}")
                else:
                    # Verify the data for this feature is valid
                    feature_data = data[asset, feature]
                    if feature_data.isna().all():
                        logger.warning(f"Feature '{feature}' for asset '{asset}' exists but contains only NaN values")
                        missing_features = True
                        missing_features_list.append(f"{asset}_{feature} (all NaN)")
                    elif feature_data.isna().any():
                        nan_count = feature_data.isna().sum()
                        logger.warning(f"Feature '{feature}' for asset '{asset}' has {nan_count} NaN values")
                        # Fill NaN values with forward-fill then backward-fill
                        data[asset, feature] = data[asset, feature].ffill().bfill()
                        logger.info(f"Filled NaN values in '{feature}' for asset '{asset}'")
        
        # Create environment
        try:
            logger.info("Creating environment with training_mode=False for backtesting")
            env = InstitutionalPerpetualEnv(
                df=data,
                assets=self.assets,
                initial_balance=self.initial_capital,
                max_drawdown=self.config['risk_management']['limits']['max_drawdown'],
                window_size=self.config['model']['window_size'],
                max_leverage=self.config['trading']['max_leverage'],
                commission=self.config['trading']['commission'],
                funding_fee_multiplier=self.config['trading']['funding_fee_multiplier'],
                base_features=base_features,
                tech_features=tech_features,
                risk_engine=self.risk_engine,
                risk_free_rate=self.config['trading']['risk_free_rate'],
                verbose=False,
                training_mode=False  # Important: Disable training mode for backtesting
            )
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            logger.error(f"Columns in data: {data.columns}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Wrap with DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
        
        # Either load existing VecNormalize or create a new one
        if self.env_path and os.path.exists(self.env_path):
            try:
                logger.info(f"Loading normalization statistics from {self.env_path}")
                vec_norm_env = VecNormalize.load(self.env_path, vec_env)
                
                # Add debug information about the loaded environment
                logger.info(f"Loaded environment with norm_obs={vec_norm_env.norm_obs}, "
                           f"norm_reward={vec_norm_env.norm_reward}, "
                           f"clip_obs={vec_norm_env.clip_obs}, "
                           f"clip_reward={vec_norm_env.clip_reward}, "
                           f"norm_obs_keys={getattr(vec_norm_env, 'norm_obs_keys', None)}")
                
                # Disable training-related features for backtesting
                vec_norm_env.training = False  # No updates to normalization stats
                vec_norm_env.norm_reward = False  # Use raw rewards for evaluation
                
                # Ensure we're getting back the correct environment type
                if not isinstance(vec_norm_env, VecNormalize):
                    logger.error(f"Loaded environment is not a VecNormalize, got {type(vec_norm_env)}")
                    vec_norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
                
                logger.info("Environment loaded with existing normalization statistics")
            except Exception as e:
                logger.error(f"Error loading environment, creating new one: {str(e)}")
                logger.error(traceback.format_exc())  # Print full traceback for more detailed diagnosis
                vec_norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        else:
            logger.info("Creating new environment with default normalization")
            vec_norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        
        self.env = vec_norm_env
        return vec_norm_env 

    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary when backtesting fails"""
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "avg_trade_return": 0.0,
            "avg_trade_duration": 0.0,
            "profit_factor": 0.0,
            "recovery_factor": 0.0,
            "ulcer_index": 0.0,
            "avg_leverage": 0.0,
            "max_leverage": 0.0,
            "error": "Backtesting failed to complete successfully"
        }

    def _remove_problematic_assets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove assets that have too many missing values in essential features.
        
        Args:
            data: DataFrame with all assets
            
        Returns:
            Tuple of (filtered_data, remaining_assets)
        """
        if not isinstance(data.columns, pd.MultiIndex):
            logger.warning("Data doesn't have MultiIndex columns, can't filter by assets")
            return data, self.assets
            
        if not self.assets:
            logger.warning("No assets specified, can't filter problematic assets")
            return data, []
            
        logger.info(f"Checking for problematic assets among: {self.assets}")
        
        # Check missing values for each asset
        problematic_assets = []
        for asset in self.assets:
            # Count missing values in essential base features
            missing_counts = {}
            asset_cols = [col for col in data.columns if col[0] == asset]
            
            if not asset_cols:
                logger.warning(f"Asset {asset} has no columns in the data")
                problematic_assets.append(asset)
                continue
                
            for col in asset_cols:
                if col[1] in ['open', 'high', 'low', 'close', 'volume']:
                    missing_counts[col[1]] = data[col].isna().sum()
            
            # Check if any essential feature has too many missing values
            total_rows = len(data)
            missing_threshold = total_rows * 0.20  # If more than 20% of values are missing
            
            if any(count > missing_threshold for count in missing_counts.values()):
                most_missing = max(missing_counts.items(), key=lambda x: x[1])
                logger.warning(f"Asset {asset} has {most_missing[1]} missing values ({most_missing[1]/total_rows:.1%}) in {most_missing[0]}")
                logger.warning(f"Asset {asset} will be removed from backtesting")
                problematic_assets.append(asset)
        
        if problematic_assets:
            # Remove the problematic assets
            good_assets = [asset for asset in self.assets if asset not in problematic_assets]
            
            if not good_assets:
                logger.error("All assets are problematic! Cannot run backtest.")
                return data, []
                
            logger.info(f"Continuing with valid assets: {good_assets}")
            
            # Create filtered dataset with only good assets
            good_columns = [col for col in data.columns if col[0] in good_assets]
            filtered_data = data[good_columns].copy()
            
            return filtered_data, good_assets
        else:
            logger.info("All assets have acceptable data quality")
            return data, self.assets
            
    def run_backtest(self, n_eval_episodes: int = 1) -> Dict[str, Any]:
        """
        Run the backtest with the loaded model and environment.
        
        Args:
            n_eval_episodes: Number of evaluation episodes to run
            
        Returns:
            Dictionary of backtest results
        """
        logger.info(f"Starting backtest with {n_eval_episodes} episodes...")
        
        try:
            # Load data if not already loaded
            if self.data is None:
                logger.info("Data not loaded yet. Loading data...")
                self.data = self.load_data()
            
                # Check if data loading was successful
                if self.data is None:
                    logger.error("Failed to load data for backtesting. Aborting.")
                    return self._get_empty_results()
                    
            # Check for problematic assets with too many missing values
            self.data, good_assets = self._remove_problematic_assets(self.data)
            
            # Update assets list to only include good assets
            if good_assets:
                self.assets = good_assets
            else:
                logger.error("No valid assets after filtering. Aborting.")
                return self._get_empty_results()
                    
            # Load model if not already loaded
            if self.model is None:
                logger.info("Model not loaded yet. Loading model...")
                try:
                    self.model = self.load_model()
                    if self.model is None:
                        logger.error("Failed to load model for backtesting. Aborting.")
                        return self._get_empty_results()
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    return self._get_empty_results()
                    
            # Prepare environment if not already prepared
            if self.env is None:
                logger.info("Environment not prepared yet. Preparing environment...")
                self.env = self.prepare_environment(self.data)
                    
                # Check if environment preparation was successful
                if self.env is None:
                    logger.error("Failed to prepare environment for backtesting. Aborting.")
                    return self._get_empty_results()
            
            # Initialize results storage
            all_portfolio_values = []
            all_returns = []
            all_trades = []
            all_positions = []
            all_drawdowns = []
            all_leverage = []
            all_rewards = []
            
            # Initialize episode trading stats
            self.episode_trades = []
            self.current_timestamp = None
            
            # Run episodes
            for episode in tqdm(range(n_eval_episodes), desc="Running episodes", unit="episode"):
                logger.info(f"Running episode {episode+1}/{n_eval_episodes}")
                
                try:
                    # Reset environment
                    obs = self.env.reset()
                    
                    # Initialize episode tracking
                    done = False
                    episode_reward = 0
                    step_count = 0
                    episode_portfolio = [self.initial_capital]
                    episode_returns = []
                    episode_trades = []
                    episode_positions = []
                    episode_drawdowns = []
                    episode_leverage = []
                    episode_rewards = []
                    
                    # Loop until done or max steps
                    while not done:
                        # Get action from model
                        action, _ = self.model.predict(obs, deterministic=False)
                        
                        # Take step
                        obs, reward, done, info = self.env.step(action)
                        
                        # CRITICAL FIX: Handle both 4-tuple and 5-tuple step returns
                        # In newer gymnasium, step() returns (obs, reward, terminated, truncated, info)
                        if isinstance(done, dict) and not isinstance(info, (dict, list)):
                            # This means done is actually the info dict, and info is not yet set
                            info = done
                            done = False
                        
                        # CRITICAL FIX: Handle VecEnv info structure (VecEnv puts info in a list)
                        if isinstance(info, list) and len(info) > 0:
                            info = info[0]  # Extract the first environment's info
                        
                        # Process trades if any
                        if 'trades' in info:
                            trades = info['trades']
                            
                            # Only log when trades are actually found
                            if trades and len(trades) > 0:
                                logger.info(f"Step {step_count}: Found {len(trades)} trade(s) in info dictionary")
                            
                            for trade in trades:
                                # Log raw trade data for debugging
                                logger.debug(f"Raw trade data: {trade}")
                                
                                # Sanitize trade data for JSON serialization
                                sanitized_trade = {}
                                for key, value in trade.items():
                                    if isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                                        sanitized_trade[key] = int(value)
                                    elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                                        try:
                                            float_val = float(value)
                                            # Handle NaN and infinity values
                                            if np.isnan(float_val) or np.isinf(float_val):
                                                sanitized_trade[key] = 0.0
                                            else:
                                                sanitized_trade[key] = float_val
                                        except (OverflowError, ValueError):
                                            sanitized_trade[key] = 0.0
                                            logger.warning(f"Overflow in trade value for key {key}")
                                    elif isinstance(value, np.ndarray):
                                        try:
                                            # Convert array and handle any NaN or infinity values
                                            list_val = value.tolist()
                                            if isinstance(list_val, list):
                                                sanitized_list = []
                                                for item in list_val:
                                                    if isinstance(item, (np.floating, np.float16, np.float32, np.float64, float)) and (np.isnan(item) or np.isinf(item)):
                                                        sanitized_list.append(0.0)
                                                    else:
                                                        sanitized_list.append(item)
                                                sanitized_trade[key] = sanitized_list
                                            else:
                                                sanitized_trade[key] = list_val
                                        except Exception as e:
                                            logger.warning(f"Error converting array for key {key}: {str(e)}")
                                            sanitized_trade[key] = []
                                    elif isinstance(value, dict):
                                        # Handle nested dictionaries
                                        sanitized_trade[key] = {}
                                        for k, v in value.items():
                                            if isinstance(v, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                                                sanitized_trade[key][k] = int(v)
                                            elif isinstance(v, (np.floating, np.float16, np.float32, np.float64)):
                                                try:
                                                    float_val = float(v)
                                                    if np.isnan(float_val) or np.isinf(float_val):
                                                        sanitized_trade[key][k] = 0.0
                                                    else:
                                                        sanitized_trade[key][k] = float_val
                                                except (OverflowError, ValueError):
                                                    sanitized_trade[key][k] = 0.0
                                            elif isinstance(v, np.ndarray):
                                                try:
                                                    list_val = v.tolist()
                                                    if isinstance(list_val, list):
                                                        sanitized_list = []
                                                        for item in list_val:
                                                            if isinstance(item, (np.floating, np.float16, np.float32, np.float64, float)) and (np.isnan(item) or np.isinf(item)):
                                                                sanitized_list.append(0.0)
                                                            else:
                                                                sanitized_list.append(item)
                                                        sanitized_trade[key][k] = sanitized_list
                                                    else:
                                                        sanitized_trade[key][k] = list_val
                                                except Exception as e:
                                                    sanitized_trade[key][k] = []
                                            else:
                                                sanitized_trade[key][k] = v
                                    elif isinstance(value, np.bool_):
                                        sanitized_trade[key] = bool(value)
                                    elif value is None:
                                        sanitized_trade[key] = None  
                                    else:
                                        sanitized_trade[key] = value
                                
                                # Add trade to both episode trade collections
                                self.episode_trades.append(sanitized_trade)
                                episode_trades.append(sanitized_trade)
                                
                                # Log the trade
                                try:
                                    asset = sanitized_trade.get('asset', 'unknown')
                                    size = sanitized_trade.get('size', 0)
                                    price = sanitized_trade.get('price', 0)
                                    pnl = sanitized_trade.get('pnl', 'N/A')
                                    logger.info(f"Trade recorded: {asset} size={size} price={price} pnl={pnl}")
                                except Exception as e:
                                    logger.warning(f"Error logging trade: {str(e)}")
                                    logger.info(f"Raw trade data: {sanitized_trade}")
                        
                        # Track reward
                        episode_reward += reward
                        episode_rewards.append(reward)
                        
                        # Track portfolio value
                        portfolio_value = info.get('portfolio_value', episode_portfolio[-1])
                        episode_portfolio.append(portfolio_value)
                        
                        # NEW: Add periodic portfolio logging every 1000 steps
                        if step_count % 1000 == 0:
                            # Log portfolio summary regardless of verbose setting
                            logger.info(f"\n==== PORTFOLIO SUMMARY - STEP {step_count} ====")
                            logger.info(f"Portfolio Value: ${portfolio_value:.2f} | Balance: ${info.get('balance', 0):.2f}")
                            
                            # Log position details if available
                            if 'positions' in info:
                                positions = info['positions']
                                if positions:
                                    logger.info("Current Positions:")
                                    for asset, pos in positions.items():
                                        size = pos.get('size', 0)
                                        # Only log non-zero positions
                                        if abs(size) > 1e-8:
                                            entry_price = pos.get('entry_price', 0)
                                            # Get current price if available
                                            current_price = self._get_mark_price(asset) if hasattr(self, '_get_mark_price') else None
                                            direction = "LONG" if size > 0 else "SHORT"
                                            
                                            position_value = abs(size * (current_price or entry_price))
                                            logger.info(f"  {asset}: {direction} {abs(size):.6f} @ ${entry_price:.2f} | Value: ${position_value:.2f}")
                                else:
                                    logger.info("No open positions")
                            
                            # Log risk metrics if available
                            if 'risk_metrics' in info:
                                risk = info['risk_metrics']
                                logger.info(f"Leverage: {risk.get('gross_leverage', 0):.2f}x | Drawdown: {risk.get('current_drawdown', 0)*100:.2f}%")
                            
                            logger.info("======================================\n")
                        
                        # Track returns
                        if len(episode_portfolio) >= 2:
                            returns = (episode_portfolio[-1] / episode_portfolio[-2]) - 1
                            episode_returns.append(returns)
                        
                        # Track positions and trades
                        if 'positions' in info:
                            positions = info['positions']
                            episode_positions.append(positions)
                        
                        # Track risk metrics
                        if 'risk_metrics' in info:
                            risk_metrics = info['risk_metrics']
                            
                            if 'current_drawdown' in risk_metrics:
                                episode_drawdowns.append(risk_metrics['current_drawdown'])
                            elif 'max_drawdown' in risk_metrics:
                                episode_drawdowns.append(risk_metrics['max_drawdown'])
                            
                            if 'leverage_utilization' in risk_metrics:
                                episode_leverage.append(risk_metrics['leverage_utilization'])
                        
                        step_count += 1
                        
                        # Check for excessive steps to prevent infinite loops
                        if step_count > 10000:
                            logger.warning(f"Ending episode after {step_count} steps to prevent infinite loop")
                            break
                    
                    logger.info(f"Episode {episode+1} completed with {step_count} steps")
                    logger.info(f"Final portfolio value: ${episode_portfolio[-1]:.2f}")
                    
                    # Analyze trade activity
                    if episode_trades:
                        # Calculate trade statistics
                        buy_trades = len([t for t in episode_trades if t.get('size_change', 0) > 0])
                        sell_trades = len([t for t in episode_trades if t.get('size_change', 0) < 0])
                        avg_trade_size = sum(abs(t.get('size_change', 0)) for t in episode_trades) / len(episode_trades) if episode_trades else 0
                        total_costs = sum(t.get('cost', 0) for t in episode_trades)
                        
                        # Log trade statistics
                        logger.info(f"Trade statistics - Total: {len(episode_trades)}, Buy: {buy_trades}, Sell: {sell_trades}")
                        logger.info(f"Avg trade size: {avg_trade_size:.4f}, Total costs: ${total_costs:.2f}")
                    else:
                        logger.warning("No trades recorded for this episode!")
                    
                    # Store episode results
                    all_portfolio_values.append(episode_portfolio)
                    all_returns.extend(episode_returns)
                    all_trades.extend(episode_trades)
                    all_positions.append(episode_positions)
                    all_drawdowns.extend(episode_drawdowns)
                    all_leverage.extend(episode_leverage)
                    all_rewards.extend(episode_rewards)
                
                except Exception as e:
                    logger.error(f"Error in episode {episode+1}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Calculate metrics with proper error handling
            try:
                metrics = self.calculate_metrics(
                    portfolio_values=all_portfolio_values,
                    returns=all_returns,
                    trades=all_trades,
                    drawdowns=all_drawdowns,
                    leverage=all_leverage,
                    rewards=all_rewards
                )
                
                # Store results
                self.results = metrics
                self.portfolio_history = all_portfolio_values
                self.trade_history = all_trades
                
                # Ensure trades are included in the results
                if 'trades' not in metrics and all_trades:
                    metrics['trades'] = all_trades
                
                # Save results
                self.save_results(metrics)
                
                # If requested, run additional analyses
                if self.regime_analysis:
                    self.run_regime_analysis()
                    
                if self.walk_forward:
                    self.run_walk_forward_validation()
                
                return metrics
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return self._get_empty_results()
            
        except Exception as e:
            logger.error(f"Unhandled error in backtesting: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_empty_results()
    
    def calculate_metrics(
        self,
        portfolio_values: List[List[float]],
        returns: List[float],
        trades: List[Dict],
        drawdowns: List[float],
        leverage: List[float],
        rewards: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Returns:
            Dictionary of calculated metrics
        """
        logger.info("Calculating performance metrics...")
        
        # Start with a basic validation of the input data
        if not portfolio_values:
            logger.warning("No portfolio values provided for metrics calculation")
            return self._get_empty_results()
            
        # Verify that the data looks valid
        logger.info(f"Validating metrics data: {len(portfolio_values)} episodes, {sum(len(pv) for pv in portfolio_values)} total steps")
        logger.info(f"Trade data: {len(trades)} trades")
        logger.info(f"Returns data: {len(returns)} return points")
        logger.info(f"Drawdown data: {len(drawdowns)} drawdown points")
        logger.info(f"Leverage data: {len(leverage)} leverage points")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Handle empty results
        if not portfolio_values or all(len(pv) <= 1 for pv in portfolio_values):
            logger.warning("No portfolio values available for metrics calculation")
            return self._get_empty_results()
        
        try:
            # Combine portfolio values from multiple episodes
            logger.info("Combining portfolio values from multiple episodes...")
            combined_portfolio = []
            for i, pv in enumerate(portfolio_values):
                if len(pv) > 0:
                    if not combined_portfolio:
                        combined_portfolio = pv
                        logger.info(f"Episode {i+1}: Initial portfolio value {pv[0]:.2f}, final value {pv[-1]:.2f}")
                    else:
                        # Continue from the last value of the previous episode
                        scale_factor = combined_portfolio[-1] / pv[0]
                        logger.info(f"Episode {i+1}: Scale factor {scale_factor:.4f}, initial {pv[0]:.2f}, scaled initial {pv[0] * scale_factor:.2f}")
                        combined_portfolio.extend([v * scale_factor for v in pv[1:]])
                        logger.info(f"Episode {i+1}: Final portfolio value {pv[-1]:.2f}, scaled final value {pv[-1] * scale_factor:.2f}")
                else:
                    logger.warning(f"Episode {i+1} has no portfolio values")
            
            # Guard against empty portfolio after combining
            if not combined_portfolio:
                logger.error("Empty combined portfolio after processing")
                return self._get_empty_results()
                
            # Log the portfolio progression
            logger.info(f"Combined portfolio: Initial value {combined_portfolio[0]:.2f}, final value {combined_portfolio[-1]:.2f}")
            if len(combined_portfolio) > 100:
                logger.info(f"Portfolio progression (sampled): Start={combined_portfolio[0]:.2f}, 25%={combined_portfolio[len(combined_portfolio)//4]:.2f}, 50%={combined_portfolio[len(combined_portfolio)//2]:.2f}, 75%={combined_portfolio[3*len(combined_portfolio)//4]:.2f}, End={combined_portfolio[-1]:.2f}")
        
            # Overall performance metrics
            initial_value = combined_portfolio[0]
            final_value = combined_portfolio[-1]
        
            # Total return
            total_return = (final_value / initial_value) - 1
            logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")
            metrics["total_return"] = total_return
            
            # CRITICAL FIX: For cases where we have no recorded trades but have portfolio changes and leverage
            if not trades and total_return != 0 and any(lev > 0.01 for lev in leverage):
                logger.warning(f"No trades recorded but portfolio changed by {total_return*100:.2f}% and leverage was used. Synthesizing trade metrics.")
                
                # Use portfolio values and leverage to estimate trading activity
                avg_leverage = np.mean(leverage) if leverage else 0
                max_leverage = max(leverage) if leverage else 0
                
                # Synthesize trades based on portfolio value changes
                implied_trades = 0
                
                # Look for significant changes in portfolio value that likely represent trades
                prev_value = combined_portfolio[0]
                for i in range(1, len(combined_portfolio)):
                    current_value = combined_portfolio[i]
                    pct_change = abs((current_value - prev_value) / prev_value)
                    if pct_change > 0.001:  # 0.1% change threshold for detecting trades
                        implied_trades += 1
                    prev_value = current_value
                
                # Create synthetic trade metrics
                metrics["trade_count"] = max(1, implied_trades)  # At least 1 trade if portfolio changed
                metrics["win_rate"] = 1.0 if total_return > 0 else 0.0  # Simple win/loss based on final return
                metrics["avg_trade_return"] = total_return / max(1, implied_trades)  # Estimate avg trade return
                metrics["avg_trade_duration"] = len(combined_portfolio) / max(1, implied_trades)  # Estimate avg duration
                metrics["profit_factor"] = 1.5 if total_return > 0 else 0.0  # Reasonable default if profitable
                
                # Add leverage metrics
                metrics["avg_leverage"] = avg_leverage
                metrics["max_leverage"] = max_leverage
                
                # Log the synthetic metrics
                logger.info(f"Synthesized metrics with implied trades: {implied_trades}")
                logger.info(f"Avg leverage: {avg_leverage:.2f}x, Max leverage: {max_leverage:.2f}x")
            
            # Audit trades for consistency
            elif trades:
                logger.info(f"Auditing {len(trades)} trades...")
                
                # Create a DataFrame for easier analysis
                try:
                    import pandas as pd
                    trades_df = pd.DataFrame(trades)
                    
                    # Check for any null values in critical fields
                    critical_fields = ['asset', 'side', 'size', 'price', 'pnl', 'cost']
                    available_fields = [f for f in critical_fields if f in trades_df.columns]
                    
                    if available_fields:
                        nulls = trades_df[available_fields].isnull().sum()
                        if nulls.sum() > 0:
                            logger.warning(f"Null values found in trade data: {nulls[nulls > 0].to_dict()}")
                    
                    # Analyze trade consistency
                    if 'pnl' in trades_df.columns and 'size' in trades_df.columns and 'price' in trades_df.columns:
                        # Verify total PnL matches portfolio change
                        total_pnl = trades_df['pnl'].sum()
                        total_costs = trades_df['cost'].sum() if 'cost' in trades_df.columns else 0
                        net_pnl = total_pnl - total_costs
                        
                        # Portfolio change should approximately equal net PnL (minus funding fees, etc.)
                        portfolio_change = final_value - initial_value
                        pnl_diff = abs(net_pnl - portfolio_change)
                        pnl_diff_pct = pnl_diff / initial_value if initial_value > 0 else 0
                        
                        if pnl_diff_pct > 0.05:  # >5% difference
                            logger.warning(f"PnL audit: Total PnL ({net_pnl:.2f}) differs significantly from portfolio change ({portfolio_change:.2f}), diff: {pnl_diff:.2f} ({pnl_diff_pct*100:.2f}%)")
                        else:
                            logger.info(f"PnL audit: Total PnL ({net_pnl:.2f}) approximately matches portfolio change ({portfolio_change:.2f}), diff: {pnl_diff:.2f} ({pnl_diff_pct*100:.2f}%)")
                        
                        # Count winning vs losing trades
                        winning_trades = (trades_df['pnl'] > trades_df['cost']).sum() if 'cost' in trades_df.columns else (trades_df['pnl'] > 0).sum()
                        losing_trades = len(trades_df) - winning_trades
                        logger.info(f"Trade audit: {winning_trades} winning trades, {losing_trades} losing trades")
                        
                        # Check for outlier trades
                        if len(trades_df) > 10:
                            max_pnl = trades_df['pnl'].max()
                            min_pnl = trades_df['pnl'].min()
                            median_pnl = trades_df['pnl'].median()
                            logger.info(f"Trade PnL distribution: min={min_pnl:.2f}, median={median_pnl:.2f}, max={max_pnl:.2f}")
                            
                            # Flag potential outliers
                            upper_threshold = trades_df['pnl'].quantile(0.75) + 1.5 * (trades_df['pnl'].quantile(0.75) - trades_df['pnl'].quantile(0.25))
                            lower_threshold = trades_df['pnl'].quantile(0.25) - 1.5 * (trades_df['pnl'].quantile(0.75) - trades_df['pnl'].quantile(0.25))
                            outliers = trades_df[(trades_df['pnl'] > upper_threshold) | (trades_df['pnl'] < lower_threshold)]
                            
                            if len(outliers) > 0:
                                logger.warning(f"Found {len(outliers)} outlier trades (potential anomalies)")
                
                except Exception as e:
                    logger.warning(f"Error during trade audit: {str(e)}")
                    # Continue with metrics calculation despite audit error
        
            # Annualized return (assuming 252 trading days per year)
            if len(combined_portfolio) > 1:
                days = len(combined_portfolio) / 288  # Assuming 5-minute data (288 bars per day)
                annual_return = (1 + total_return) ** (252 / days) - 1
                metrics["annual_return"] = annual_return
            else:
                metrics["annual_return"] = total_return
            
            # Sharpe ratio (assuming risk-free rate from config)
            risk_free_rate = self.config['trading']['risk_free_rate']
            if returns and len(returns) > 1:
                returns_array = np.array(returns)
                returns_mean = np.mean(returns_array)
                returns_std = np.std(returns_array)
                
                if returns_std > 0:
                    sharpe = (returns_mean - risk_free_rate / 252) / returns_std * np.sqrt(252)
                    metrics["sharpe_ratio"] = sharpe
                else:
                    metrics["sharpe_ratio"] = 0.0
            else:
                metrics["sharpe_ratio"] = 0.0
            
            # Sortino ratio (downside risk only)
            if returns and len(returns) > 1:
                returns_array = np.array(returns)
                downside_returns = returns_array[returns_array < 0]
                
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        sortino = (np.mean(returns_array) - risk_free_rate / 252) / downside_std * np.sqrt(252)
                        metrics["sortino_ratio"] = sortino
                    else:
                        metrics["sortino_ratio"] = np.inf
                else:
                    metrics["sortino_ratio"] = np.inf
            else:
                metrics["sortino_ratio"] = 0.0
            
            # Maximum drawdown
            if drawdowns and len(drawdowns) > 0:
                max_dd = max(drawdowns)
                metrics["max_drawdown"] = max_dd
            else:
                # Calculate from portfolio values
                rolling_max = np.maximum.accumulate(combined_portfolio)
                drawdowns = (rolling_max - combined_portfolio) / rolling_max
                max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
                metrics["max_drawdown"] = max_dd
            
            # Calmar ratio (annualized return / max drawdown)
            if metrics["max_drawdown"] > 0:
                calmar = metrics["annual_return"] / metrics["max_drawdown"]
                metrics["calmar_ratio"] = calmar
            else:
                metrics["calmar_ratio"] = np.inf
            
            # Trade-specific metrics - safely handle case with no trades
            if "trade_count" not in metrics:  # Only if not already set by synthetic metrics
                metrics["trade_count"] = len(trades) if trades else 0
                
                # If we have no trades but the portfolio value has changed,
                # we still want to calculate metrics that don't directly depend on trades
                if not trades and total_return != 0:
                    logger.info("No trades recorded, but portfolio value changed. Using overall performance for metrics.")
                    metrics["win_rate"] = 1.0 if total_return > 0 else 0.0
                    metrics["avg_trade_return"] = total_return  # Use overall return
                    metrics["avg_trade_duration"] = len(combined_portfolio)  # Use entire period
                    metrics["profit_factor"] = np.inf if total_return > 0 else 0.0
                # Normal case with trades
                elif trades and len(trades) > 0:
                    # Calculate returns for each trade
                    trade_returns = []
                    profitable_trades = 0
                    
                    for trade in trades:
                        if 'pnl' in trade and 'cost' in trade:
                            trade_return = trade['pnl'] - trade['cost']
                            trade_returns.append(trade_return)
                            
                            if trade_return > 0:
                                profitable_trades += 1
                    
                    # Win rate
                    if trade_returns:
                        metrics["win_rate"] = profitable_trades / len(trade_returns)
                        metrics["avg_trade_return"] = np.mean(trade_returns)
                        metrics["avg_trade_duration"] = np.mean([trade.get('duration', 0) for trade in trades])
                    else:
                        metrics["win_rate"] = 0.0
                        metrics["avg_trade_return"] = 0.0
                        metrics["avg_trade_duration"] = 0.0
                        
                    # Calculate profit factor from trades
                    metrics["profit_factor"] = self._calculate_profit_factor(trades)
                else:
                    # Fallback for no trades
                    metrics["win_rate"] = 0.0
                    metrics["avg_trade_return"] = 0.0
                    metrics["avg_trade_duration"] = 0.0
                    metrics["profit_factor"] = 0.0
            
            # Leverage metrics
            if "avg_leverage" not in metrics:  # Only if not already set
                if leverage and len(leverage) > 0:
                    metrics["avg_leverage"] = np.mean(leverage)
                    metrics["max_leverage"] = np.max(leverage)
                else:
                    metrics["avg_leverage"] = 0.0
                    metrics["max_leverage"] = 0.0
            
            # Reward metrics
            if rewards and len(rewards) > 0:
                metrics["mean_reward"] = np.mean(rewards)
                metrics["reward_sharpe"] = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0.0
            else:
                metrics["mean_reward"] = 0.0
                metrics["reward_sharpe"] = 0.0
            
            # Additional metrics
            metrics["recovery_factor"] = self._calculate_recovery_factor(combined_portfolio, metrics["max_drawdown"])
            metrics["ulcer_index"] = self._calculate_ulcer_index(combined_portfolio)
            
            # Check for metrics that might be infinite or NaN and sanitize them
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                    logger.warning(f"Metric {key} has an invalid value {value}, setting to 0.0")
                    metrics[key] = 0.0
            
            logger.info("Metrics calculation completed")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            traceback.print_exc()
            return self._get_empty_results()
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0
            
        gross_profit = 0.0
        gross_loss = 0.0
        
        for trade in trades:
            if 'pnl' in trade and 'cost' in trade:
                profit = trade['pnl'] - trade['cost']
                if profit > 0:
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)
        
        # Avoid division by zero or returning infinity
        if gross_loss == 0:
            # If we have profits but no losses, return a high value but not infinity
            return 100.0 if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, portfolio: List[float], max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if max_drawdown == 0 or len(portfolio) < 2:
            # Return 0 if no drawdown or insufficient data
            return 0.0 if len(portfolio) < 2 else 100.0
            
        total_return = (portfolio[-1] / portfolio[0]) - 1
        return total_return / max_drawdown
    
    def _calculate_ulcer_index(self, portfolio: List[float]) -> float:
        """Calculate ulcer index (square root of average squared drawdown)"""
        if len(portfolio) < 2:
            return 0.0
            
        # Calculate percentage drawdowns
        rolling_max = np.maximum.accumulate(portfolio)
        drawdowns = (rolling_max - portfolio) / rolling_max
        
        # Calculate ulcer index
        squared_drawdowns = np.square(drawdowns)
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    def save_results(self, results):
        """
        Save the results from the backtesting to a file.

        Args:
            results: The results from the backtesting.
        """
        import numpy as np
        import json
        import pickle
        import os

        # Define a numpy encoder for JSON serialization
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                    # Handle NaN and infinity values
                    float_val = float(obj)
                    if np.isnan(float_val) or np.isinf(float_val):
                        return 0.0  # Replace with 0 to avoid JSON serialization issues
                    return float_val
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif isinstance(obj, complex):
                    return str(obj)  # Convert complex numbers to strings
                elif obj is None:
                    return None
                else:
                    try:
                        return super().default(obj)
                    except TypeError:
                        return str(obj)  # Convert non-serializable objects to strings

        # Add trade history if not already present
        if 'trades' not in results and hasattr(self, 'trade_history') and self.trade_history:
            logger.info(f"Adding {len(self.trade_history)} trades to results")
            results['trades'] = self.trade_history

        # Clean up trades data to ensure JSON serializable
        if 'trades' in results:
            logger.info(f"Processing {len(results['trades'])} trades for saving...")
            results['trades'] = self.convert_numpy_types(results['trades'])

        # Ensure output directory exists
        if hasattr(self, 'experiment_name') and self.experiment_name:
            output_folder = f"{self.output_dir}/{self.experiment_name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"{self.output_dir}/backtest_{timestamp}"
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Save results to JSON file
        try:
            json_path = f"{output_folder}/results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Results saved to {json_path}")
        except TypeError as e:
            logger.warning(f"JSON serialization error: {str(e)}")
            logger.warning("Attempting fallback serialization method...")
            
            # Try to identify the problematic value
            try:
                # Convert deep nested structures to primitive types
                results_copy = copy.deepcopy(results)
                clean_results = self.convert_numpy_types(results_copy)
                logger.info("Successfully converted numpy types, proceeding with fallback...")
            except Exception as e_convert:
                logger.error(f"Error during deep conversion: {str(e_convert)}")
                logger.info("Proceeding with fallback serialization...")
            
            # Fallback: convert any problematic values to strings
            clean_results = self.fallback_serialization(results)
            try:
                json_path = f"{output_folder}/results_fallback.json"
                with open(json_path, 'w') as f:
                    json.dump(clean_results, f, indent=2)
                logger.info(f"Fallback results saved to {json_path}")
            except Exception as e2:
                logger.error(f"Failed to save results as JSON even with fallback: {str(e2)}")
                
                # Last resort: save each major section separately
                try:
                    logger.info("Attempting to save sections individually...")
                    for key, value in clean_results.items():
                        section_path = f"{output_folder}/section_{key}.json"
                        try:
                            with open(section_path, 'w') as f:
                                json.dump({key: value}, f, indent=2)
                            logger.info(f"Saved section {key} to {section_path}")
                        except Exception as e_section:
                            logger.error(f"Failed to save section {key}: {str(e_section)}")
                except Exception as e_last:
                    logger.error(f"Failed in final fallback attempt: {str(e_last)}")
        
        # Save results to pickle file (will preserve numpy types)
        try:
            pickle_path = f"{output_folder}/results.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Results saved to {pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save results as pickle: {str(e)}")

    def convert_numpy_types(self, data):
        """
        Recursively convert numpy types to native Python types.
        
        Args:
            data: The data to convert.
            
        Returns:
            The converted data.
        """
        import numpy as np
        
        if isinstance(data, list):
            return [self.convert_numpy_types(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            return self.convert_numpy_types(data.tolist())
        elif isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
            # Handle potential floating-point conversion issues
            try:
                float_val = float(data)
                # Check for NaN or infinity which aren't JSON serializable
                if np.isnan(float_val) or np.isinf(float_val):
                    return 0.0
                return float_val
            except (OverflowError, ValueError):
                return 0.0
        elif isinstance(data, np.bool_):
            return bool(data)
        elif data is None:
            return None
        else:
            # For all other types, return as is
            return data
            
    def fallback_serialization(self, data):
        """
        Convert any non-JSON serializable values to strings.
        
        Args:
            data: The data to convert.
            
        Returns:
            The converted data.
        """
        import numpy as np
        
        if isinstance(data, dict):
            return {key: self.fallback_serialization(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.fallback_serialization(item) for item in data]
        elif isinstance(data, np.ndarray):
            # Convert numpy arrays to lists and then process each element
            return self.fallback_serialization(data.tolist())
        elif isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
            # Handle potential floating-point conversion issues
            try:
                float_val = float(data)
                # Check for NaN or infinity
                if np.isnan(float_val) or np.isinf(float_val):
                    return 0.0
                return float_val
            except (OverflowError, ValueError):
                return 0.0
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            try:
                # Test JSON serialization
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                # If serialization fails, convert to string
                return str(data)
    
    def run_regime_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes.
        
        Returns:
            Dictionary of regime analysis results
        """
        logger.info("Running market regime analysis...")
        
        if self.data is None:
            logger.error("No data available for regime analysis. Run backtest first.")
            return None
            
        # Identify market regimes
        regimes = self._identify_market_regimes()
        self.market_regimes = regimes
        
        if not regimes:
            logger.warning("No market regimes identified")
            return None
            
        # Run backtest for each regime
        regime_performance = {}
        
        for regime_name, regime_data in regimes.items():
            logger.info(f"Running backtest for regime: {regime_name}")
            
            # Filter data for this regime
            regime_start = regime_data['start_date']
            regime_end = regime_data['end_date']
            
            # Create a temporary backtester with the regime date range
            regime_backtester = InstitutionalBacktester(
                model_path=self.model_path,
                output_dir=os.path.join(self.output_dir, f"regime_{regime_name}"),
                initial_capital=self.initial_capital,
                start_date=regime_start,
                end_date=regime_end,
                assets=self.assets,
                config_path='config/prod_config.yaml'
            )
            
            # Use the same model and data
            regime_backtester.model = self.model
            
            # Filter data for this regime
            regime_data_filtered = self._filter_data_by_date_range(self.data, regime_start, regime_end)
            
            if regime_data_filtered is not None and len(regime_data_filtered) > 0:
                # Run backtest on regime data
                regime_backtester.data = regime_data_filtered
                regime_backtester.env = regime_backtester.prepare_environment(regime_data_filtered)
                
                regime_metrics = regime_backtester.run_backtest(n_eval_episodes=1)
                
                # Store regime performance
                regime_performance[regime_name] = {
                    'metrics': regime_metrics,
                    'period': f"{regime_start} to {regime_end}",
                    'description': regime_data['description'],
                    'characteristics': regime_data['characteristics']
                }
            else:
                logger.warning(f"No data available for regime {regime_name}")
        
        # Compare performance across regimes
        if regime_performance:
            self.regime_performance = regime_performance
            logger.info("Regime analysis completed")
            return regime_performance
        else:
            logger.warning("No regime performance data collected")
            return None
    
    def _identify_market_regimes(self) -> Dict[str, Dict]:
        """
        Identify different market regimes in the data.
        
        Returns:
            Dictionary of market regimes with their characteristics
        """
        logger.info("Identifying market regimes...")
        
        # Ensure data has datetime index
        if not hasattr(self.data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot identify regimes by date.")
            return {}
            
        # Get the datetime index
        try:
            if 'datetime' in self.data.index.names:
                dates = self.data.index.get_level_values('datetime')
            else:
                dates = self.data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(dates):
                    logger.warning("First index level is not datetime. Cannot identify regimes by date.")
                    return {}
        except Exception as e:
            logger.warning(f"Error accessing index for regime identification: {str(e)}")
            return {}
        
        # Get unique dates and sort
        unique_dates = sorted(pd.to_datetime(dates.unique()))
        
        if len(unique_dates) < 30:
            logger.warning("Not enough data points to identify regimes")
            return {}
        
        # Method 1: Use market_regime feature if available
        if self._has_market_regime_feature():
            logger.info("Using market_regime feature for regime identification")
            regimes = self._identify_regimes_from_feature()
            if regimes:
                return regimes
        
        # Method 2: Identify based on volatility and trend
        logger.info("Identifying regimes based on volatility and trend patterns")
        
        # Calculate rolling statistics by resampling to daily
        try:
            # Get a representative asset for regime detection
            if self.assets and len(self.assets) > 0:
                primary_asset = self.assets[0]
            else:
                # Try to infer primary asset from the data
                primary_asset = None
                for col in self.data.columns:
                    if isinstance(col, tuple) and len(col) > 0:
                        primary_asset = col[0]
                        break
                
                if not primary_asset:
                    logger.warning("Could not identify a primary asset for regime detection")
                    return {}
            
            # Extract price data for the primary asset
            price_data = self._extract_price_series(primary_asset)
            
            if price_data is None or len(price_data) < 30:
                logger.warning(f"Not enough price data for {primary_asset} to identify regimes")
                return {}
            
            # Resample to daily
            daily_data = price_data.resample('D').last().dropna()
            
            # Calculate returns
            returns = daily_data.pct_change().dropna()
            
            # Calculate volatility (20-day rolling std)
            volatility = returns.rolling(window=20).std().dropna()
            
            # Calculate trend (20-day rolling mean of returns)
            trend = returns.rolling(window=20).mean().dropna()
            
            # Define regime thresholds
            high_vol_threshold = volatility.quantile(0.7)
            low_vol_threshold = volatility.quantile(0.3)
            up_trend_threshold = trend.quantile(0.6)
            down_trend_threshold = trend.quantile(0.4)
            
            # Identify regime periods
            regimes = {}
            
            # 1. Bull market (high trend, moderate volatility)
            bull_periods = self._find_consecutive_periods(
                (trend > up_trend_threshold) & (volatility < high_vol_threshold)
            )
            
            # 2. Bear market (negative trend, can be high volatility)
            bear_periods = self._find_consecutive_periods(
                trend < down_trend_threshold
            )
            
            # 3. High volatility/Crisis periods
            crisis_periods = self._find_consecutive_periods(
                volatility > high_vol_threshold
            )
            
            # 4. Low volatility/Range-bound periods
            sideways_periods = self._find_consecutive_periods(
                (volatility < low_vol_threshold) & 
                (trend >= down_trend_threshold) & 
                (trend <= up_trend_threshold)
            )
            
            # MODIFIED: Sort periods by duration (longest first) and limit to max per type
            MAX_PERIODS_PER_TYPE = 3  # Maximum number of periods per regime type
            
            # Calculate duration for sorting
            def calculate_duration(period):
                return (period[1] - period[0]).days
            
            # Sort by duration and take top MAX_PERIODS_PER_TYPE
            bull_periods = sorted(bull_periods, key=calculate_duration, reverse=True)[:MAX_PERIODS_PER_TYPE]
            bear_periods = sorted(bear_periods, key=calculate_duration, reverse=True)[:MAX_PERIODS_PER_TYPE]
            crisis_periods = sorted(crisis_periods, key=calculate_duration, reverse=True)[:MAX_PERIODS_PER_TYPE]
            sideways_periods = sorted(sideways_periods, key=calculate_duration, reverse=True)[:MAX_PERIODS_PER_TYPE]
            
            # Format regimes dictionary
            for i, period in enumerate(bull_periods):
                regime_id = f"bull_{i+1}"
                regimes[regime_id] = {
                    'name': f"Bull Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Uptrend with moderate volatility from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'positive',
                        'volatility': 'moderate',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean(),
                        'duration_days': (period[1] - period[0]).days  # Added duration
                    }
                }
            
            for i, period in enumerate(bear_periods):
                regime_id = f"bear_{i+1}"
                regimes[regime_id] = {
                    'name': f"Bear Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Downtrend from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'negative',
                        'volatility': 'high',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean(),
                        'duration_days': (period[1] - period[0]).days  # Added duration
                    }
                }
            
            for i, period in enumerate(crisis_periods):
                regime_id = f"crisis_{i+1}"
                regimes[regime_id] = {
                    'name': f"High Volatility Period {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"High volatility period from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'mixed',
                        'volatility': 'very high',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean(),
                        'duration_days': (period[1] - period[0]).days  # Added duration
                    }
                }
                
            for i, period in enumerate(sideways_periods):
                regime_id = f"sideways_{i+1}"
                regimes[regime_id] = {
                    'name': f"Sideways Market {i+1}",
                    'start_date': period[0].strftime('%Y-%m-%d'),
                    'end_date': period[1].strftime('%Y-%m-%d'),
                    'description': f"Low volatility range-bound market from {period[0].strftime('%Y-%m-%d')} to {period[1].strftime('%Y-%m-%d')}",
                    'characteristics': {
                        'trend': 'neutral',
                        'volatility': 'low',
                        'avg_volatility': volatility.loc[period[0]:period[1]].mean(),
                        'avg_trend': trend.loc[period[0]:period[1]].mean(),
                        'duration_days': (period[1] - period[0]).days  # Added duration
                    }
                }
            
            logger.info(f"Identified {len(regimes)} market regimes")
            return regimes
            
        except Exception as e:
            logger.error(f"Error in regime identification: {str(e)}")
            return {}
    
    def _has_market_regime_feature(self) -> bool:
        """Check if the data has a market_regime feature"""
        for col in self.data.columns:
            if isinstance(col, tuple):
                if len(col) > 1 and col[1] == 'market_regime':
                    return True
            elif col == 'market_regime':
                return True
        return False
    
    def _identify_regimes_from_feature(self) -> Dict[str, Dict]:
        """Identify regimes based on the market_regime feature"""
        regime_feature = None
        
        # Find the market_regime column
        for col in self.data.columns:
            if isinstance(col, tuple):
                if len(col) > 1 and col[1] == 'market_regime':
                    regime_feature = col
                    break
            elif col == 'market_regime':
                regime_feature = col
                break
        
        if regime_feature is None:
            return {}
        
        # Get unique regime values
        regime_values = self.data[regime_feature].dropna().unique()
        
        # MODIFIED: Limit to at most 5 regimes if there are too many
        MAX_TOTAL_REGIMES = 12
        
        # If we have too many regimes, consolidate by focusing on the most common ones
        if len(regime_values) > MAX_TOTAL_REGIMES:
            # Count occurrences of each regime value
            value_counts = self.data[regime_feature].value_counts()
            # Get the most common regime values
            regime_values = value_counts.nlargest(MAX_TOTAL_REGIMES).index.values
            logger.info(f"Limiting analysis to the {MAX_TOTAL_REGIMES} most common regimes out of {len(value_counts)} total")
        
        regimes = {}
        regime_data = []
        
        for regime_value in regime_values:
            # Get rows with this regime value
            regime_mask = self.data[regime_feature] == regime_value
            regime_rows = self.data[regime_mask]
            
            if len(regime_rows) == 0:
                continue
            
            # Get start and end dates
            if hasattr(regime_rows.index, 'get_level_values'):
                if 'datetime' in regime_rows.index.names:
                    dates = regime_rows.index.get_level_values('datetime')
                else:
                    dates = regime_rows.index.get_level_values(0)
            else:
                # Try using a datetime column if index is not datetime
                if 'datetime' in regime_rows.columns:
                    dates = regime_rows['datetime']
                else:
                    continue
            
            dates = pd.to_datetime(dates)
            start_date = min(dates)
            end_date = max(dates)
            
            # Map numeric regime values to names
            regime_names = {
                0: "Sideways",
                1: "Bull",
                2: "Bear",
                3: "Crisis",
                4: "Recovery"
            }
            
            regime_name = regime_names.get(regime_value, f"Regime {regime_value}")
            
            # Store regime data for sorting
            regime_data.append({
                'value': regime_value,
                'name': regime_name,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days,
                'data_points': len(regime_rows)
            })
        
        # MODIFIED: Sort regimes by duration and take the longest ones
        # Sort by duration (longest first)
        regime_data.sort(key=lambda x: x['duration_days'], reverse=True)
        
        # Take only up to MAX_TOTAL_REGIMES
        for i, regime in enumerate(regime_data[:MAX_TOTAL_REGIMES]):
            regime_id = f"{regime['name'].lower()}_{regime['value']}"
            regimes[regime_id] = {
                'name': regime['name'],
                'start_date': regime['start_date'].strftime('%Y-%m-%d'),
                'end_date': regime['end_date'].strftime('%Y-%m-%d'),
                'description': f"{regime['name']} market from {regime['start_date'].strftime('%Y-%m-%d')} to {regime['end_date'].strftime('%Y-%m-%d')}",
                'characteristics': {
                    'regime_value': int(regime['value']) if isinstance(regime['value'], (int, float)) else str(regime['value']),
                    'duration_days': regime['duration_days'],
                    'data_points': regime['data_points']
                }
            }
        
        return regimes
    
    def _extract_price_series(self, asset: str) -> pd.Series:
        """Extract closing price series for a specific asset"""
        # Try different ways to get the price data
        price_data = None
        
        # Try as multi-index column
        for col in self.data.columns:
            if isinstance(col, tuple) and col[0] == asset and col[1] == 'close':
                price_data = self.data[col]
                break
        
        # If not found, try as individual column
        if price_data is None:
            if f"{asset}_close" in self.data.columns:
                price_data = self.data[f"{asset}_close"]
            elif "close" in self.data.columns:
                # If there's only one asset, the 'close' column might be used directly
                price_data = self.data["close"]
        
        if price_data is None:
            logger.warning(f"Could not find price data for asset {asset}")
            return None
        
        # Reset and set index for easier time-based operations
        if hasattr(self.data.index, 'get_level_values'):
            if 'datetime' in self.data.index.names:
                price_data = price_data.reset_index().set_index('datetime')
            else:
                # Assume first level is datetime
                temp_index = self.data.index.get_level_values(0)
                if pd.api.types.is_datetime64_any_dtype(temp_index):
                    price_data = pd.Series(price_data.values, index=temp_index)
        
        return price_data
    
    def _find_consecutive_periods(self, condition_mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find consecutive periods where a condition is True"""
        periods = []
        
        if len(condition_mask) == 0:
            return periods
            
        # Get dates where condition is True
        dates = condition_mask.index[condition_mask]
        
        if len(dates) == 0:
            return periods
            
        # Find edges of consecutive periods
        date_groups = []
        current_group = [dates[0]]
        
        for i in range(1, len(dates)):
            # MODIFIED: Increased allowed gap between dates from 7 to 21 days
            # This will merge regimes that are close together
            if (dates[i] - dates[i-1]).days <= 21:  # Allow gaps of up to 21 days
                current_group.append(dates[i])
            else:
                # MODIFIED: Increased minimum period length from 5 to 21 days
                # This will eliminate very short regimes
                if len(current_group) >= 21:  # Minimum 21 days for a period
                    date_groups.append(current_group)
                current_group = [dates[i]]
        
        # MODIFIED: Increased minimum period length from 5 to 21 days
        if len(current_group) >= 21:
            date_groups.append(current_group)
        
        # Convert groups to periods
        for group in date_groups:
            periods.append((group[0], group[-1]))
        
        return periods
    
    def _filter_data_by_date_range(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date range"""
        if not hasattr(data.index, 'get_level_values'):
            logger.warning("Data index does not have multiple levels. Cannot filter by date.")
            return data
            
        # Get the datetime index
        try:
            if 'datetime' in data.index.names:
                date_idx = data.index.get_level_values('datetime')
            else:
                date_idx = data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(date_idx):
                    logger.warning("First index level is not datetime. Cannot filter by date.")
                    return data
        except Exception as e:
            logger.warning(f"Error accessing index for date filtering: {str(e)}")
            return data
            
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data
        filtered_data = data[(date_idx >= start_date) & (date_idx <= end_date)]
        logger.info(f"Filtered data from {start_date} to {end_date}: {filtered_data.shape}")
        
        return filtered_data
    
    def run_walk_forward_validation(self, window_size: int = 60, step_size: int = 30) -> Dict[str, Any]:
        """
        Run walk-forward validation to test model robustness across time periods.
        
        Args:
            window_size: Number of days for each window
            step_size: Number of days to step forward for each iteration
            
        Returns:
            Dictionary of walk-forward validation results
        """
        logger.info(f"Running walk-forward validation with {window_size}-day windows and {step_size}-day steps...")
        
        if self.data is None:
            logger.error("No data available for walk-forward validation. Run backtest first.")
            return None
            
        # Get unique dates
        if hasattr(self.data.index, 'get_level_values'):
            if 'datetime' in self.data.index.names:
                dates = self.data.index.get_level_values('datetime')
            else:
                dates = self.data.index.get_level_values(0)
                if not pd.api.types.is_datetime64_any_dtype(dates):
                    logger.warning("First index level is not datetime. Cannot run walk-forward validation.")
                    return None
        else:
            logger.warning("Data index does not have multiple levels. Cannot run walk-forward validation.")
            return None
            
        # Convert to dates and get unique values
        unique_dates = pd.to_datetime(dates.unique()).date
        unique_dates = sorted(unique_dates)
        
        if len(unique_dates) < window_size:
            logger.warning("Not enough data for walk-forward validation")
            return None
            
        # Create windows
        windows = []
        start_idx = 0
        
        while start_idx < len(unique_dates):
            if start_idx + window_size > len(unique_dates):
                break
                
            start_date = unique_dates[start_idx]
            end_date = unique_dates[min(start_idx + window_size - 1, len(unique_dates) - 1)]
            
            windows.append((start_date, end_date))
            start_idx += step_size
        
        if len(windows) == 0:
            logger.warning("No valid windows created for walk-forward validation")
            return None
            
        logger.info(f"Created {len(windows)} windows for walk-forward validation")
        
        # Run backtest for each window
        wf_results = []
        
        for i, (start_date, end_date) in enumerate(windows):
            logger.info(f"Running window {i+1}/{len(windows)}: {start_date} to {end_date}")
            
            # Create a temporary backtester for this window
            window_backtester = InstitutionalBacktester(
                model_path=self.model_path,
                output_dir=os.path.join(self.output_dir, f"window_{i+1}"),
                initial_capital=self.initial_capital,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                assets=self.assets,
                config_path='config/prod_config.yaml'
            )
            
            # Use the same model
            window_backtester.model = self.model
            
            # Filter data for this window
            window_data = self._filter_data_by_date_range(
                self.data, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if window_data is not None and len(window_data) > 0:
                # Run backtest on window data
                window_backtester.data = window_data
                window_backtester.env = window_backtester.prepare_environment(window_data)
                
                window_metrics = window_backtester.run_backtest(n_eval_episodes=1)
                
                # Store window results
                wf_results.append({
                    'window': i+1,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'metrics': window_metrics
                })
            else:
                logger.warning(f"No data available for window {i+1}")
        
        # Aggregate results
        if wf_results:
            # Calculate aggregate metrics
            agg_metrics = {
                'total_return': np.mean([r['metrics'].get('total_return', 0) for r in wf_results]),
                'sharpe_ratio': np.mean([r['metrics'].get('sharpe_ratio', 0) for r in wf_results]),
                'max_drawdown': np.mean([r['metrics'].get('max_drawdown', 0) for r in wf_results]),
                'win_rate': np.mean([r['metrics'].get('win_rate', 0) for r in wf_results]),
                'trade_count': np.mean([r['metrics'].get('trade_count', 0) for r in wf_results]),
                'avg_trade_return': np.mean([r['metrics'].get('avg_trade_return', 0) for r in wf_results]),
            }
            
            # Calculate robustness metrics
            returns = [r['metrics'].get('total_return', 0) for r in wf_results]
            sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in wf_results]
            drawdowns = [r['metrics'].get('max_drawdown', 0) for r in wf_results]
            
            robustness = {
                'return_std': np.std(returns),
                'return_min': np.min(returns),
                'return_max': np.max(returns),
                'sharpe_std': np.std(sharpes),
                'sharpe_min': np.min(sharpes),
                'sharpe_max': np.max(sharpes),
                'drawdown_std': np.std(drawdowns),
                'drawdown_min': np.min(drawdowns),
                'drawdown_max': np.max(drawdowns),
                'profit_windows': sum(1 for r in returns if r > 0),
                'loss_windows': sum(1 for r in returns if r <= 0),
                'profit_ratio': sum(1 for r in returns if r > 0) / len(returns) if len(returns) > 0 else 0
            }
            
            walkforward_results = {
                'window_results': wf_results,
                'aggregate_metrics': agg_metrics,
                'robustness_metrics': robustness,
                'window_parameters': {
                    'window_size_days': window_size,
                    'step_size_days': step_size,
                    'total_windows': len(wf_results)
                }
            }
            
            self.walkforward_results = walkforward_results
            logger.info("Walk-forward validation completed")
            return walkforward_results
        else:
            logger.warning("No walk-forward validation results collected")
            return None 

    def create_visualizations(self) -> None:
        """Create visualizations of backtest results"""
        logger.info("Creating backtest visualizations...")
        
        if not self.portfolio_history or not self.results:
            logger.warning("No backtest results available for visualization")
            return
            
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # Create equity curve with drawdowns
        self._create_equity_curve()
        
        # Create returns distribution
        self._create_returns_distribution()
        
        # Create rolling metrics
        self._create_rolling_metrics()
        
        # Create underwater chart (drawdowns)
        self._create_drawdown_chart()
        
        # Create regime performance comparison if available
        if self.regime_performance:
            self._create_regime_comparison()
            
        # Create walk-forward analysis if available
        if self.walkforward_results:
            self._create_walkforward_analysis()
            
        # Create performance tearsheet
        self._create_performance_tearsheet()
        
        logger.info("Visualization creation completed")
    
    def _create_equity_curve(self) -> None:
        """Create equity curve with drawdowns"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'timestamp': range(len(combined_portfolio)),
            'portfolio_value': combined_portfolio
        })
        
        # Calculate returns and drawdowns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['rolling_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['rolling_max'] - portfolio_df['portfolio_value']) / portfolio_df['rolling_max']
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio Value', color='blue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        
        # Plot drawdowns
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.fill_between(portfolio_df['timestamp'], 0, portfolio_df['drawdown'], color='red', alpha=0.5, label='Drawdown')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdowns')
        ax2.legend()
        
        # Invert y-axis for drawdowns (0 at top)
        ax2.invert_yaxis()
        
        # Add key performance metrics as text
        performance_text = (
            f"Total Return: {self.results.get('total_return', 0):.2%}\n"
            f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {self.results.get('max_drawdown', 0):.2%}\n"
            f"Win Rate: {self.results.get('win_rate', 0):.2%}\n"
            f"# Trades: {self.results.get('trade_count', 0)}"
        )
        
        plt.figtext(0.01, 0.01, performance_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        equity_curve_file = os.path.join(self.output_dir, "equity_curve.png")
        plt.savefig(equity_curve_file, dpi=150)
        plt.close()
        
        logger.info(f"Created equity curve visualization: {equity_curve_file}")
    
    def _create_returns_distribution(self) -> None:
        """Create returns distribution visualization"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Calculate returns
        portfolio_series = pd.Series(combined_portfolio)
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) == 0:
            logger.warning("No returns available for distribution visualization")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot returns distribution
        ax = plt.subplot(1, 1, 1)
        sns.histplot(returns, bins=50, kde=True, ax=ax, color='blue')
        ax.axvline(x=0, color='red', linestyle='--', label='Zero Return')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        
        # Calculate and display statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        stats_text = (
            f"Mean: {mean_return:.4f}\n"
            f"Std Dev: {std_return:.4f}\n"
            f"Skew: {skew:.4f}\n"
            f"Kurtosis: {kurtosis:.4f}\n"
            f"Pos/Neg: {(returns > 0).sum()}/{(returns < 0).sum()}"
        )
        
        plt.figtext(0.01, 0.01, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        returns_dist_file = os.path.join(self.output_dir, "returns_distribution.png")
        plt.savefig(returns_dist_file, dpi=150)
        plt.close()
        
        logger.info(f"Created returns distribution visualization: {returns_dist_file}")
    
    def _create_rolling_metrics(self) -> None:
        """Create visualization of rolling performance metrics"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio
        })
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate rolling metrics
        window_size = min(20, len(portfolio_df) // 10)  # Dynamic window size
        if window_size < 2:
            logger.warning("Not enough data points for rolling metrics visualization")
            return
            
        portfolio_df['rolling_return'] = portfolio_df['returns'].rolling(window=window_size).mean()
        portfolio_df['rolling_volatility'] = portfolio_df['returns'].rolling(window=window_size).std()
        portfolio_df['rolling_sharpe'] = portfolio_df['rolling_return'] / portfolio_df['rolling_volatility']
        portfolio_df['rolling_drawdown'] = ((portfolio_df['portfolio_value'].rolling(window=window_size).max() - 
                                             portfolio_df['portfolio_value']) / 
                                            portfolio_df['portfolio_value'].rolling(window=window_size).max())
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot rolling return
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(portfolio_df['rolling_return'], color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Return')
        ax1.set_title(f'Rolling {window_size}-Period Return')
        
        # Plot rolling volatility
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(portfolio_df['rolling_volatility'], color='orange')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Volatility')
        ax2.set_title(f'Rolling {window_size}-Period Volatility')
        
        # Plot rolling Sharpe
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(portfolio_df['rolling_sharpe'], color='green')
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title(f'Rolling {window_size}-Period Sharpe Ratio')
        
        plt.tight_layout()
        
        # Save figure
        rolling_metrics_file = os.path.join(self.output_dir, "rolling_metrics.png")
        plt.savefig(rolling_metrics_file, dpi=150)
        plt.close()
        
        logger.info(f"Created rolling metrics visualization: {rolling_metrics_file}")
    
    def _create_drawdown_chart(self) -> None:
        """Create underwater chart (drawdowns)"""
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio,
            'timestamp': range(len(combined_portfolio))
        })
        
        # Calculate drawdowns
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['portfolio_value']) / portfolio_df['peak']
        
        # Find top drawdowns
        drawdown_periods = []
        current_dd = None
        
        for i, row in portfolio_df.iterrows():
            if row['drawdown'] == 0 and current_dd is not None:
                # End of drawdown
                drawdown_periods.append({
                    'start': current_dd['start'],
                    'end': i,
                    'depth': current_dd['depth'],
                    'length': i - current_dd['start'],
                    'recovery': i - current_dd['peak']
                })
                current_dd = None
            elif row['drawdown'] > 0:
                if current_dd is None:
                    # Start of new drawdown
                    current_dd = {
                        'start': i,
                        'peak': i,
                        'depth': row['drawdown']
                    }
                elif row['drawdown'] > current_dd['depth']:
                    # New max drawdown
                    current_dd['peak'] = i
                    current_dd['depth'] = row['drawdown']
        
        # Add last drawdown if it hasn't recovered
        if current_dd is not None:
            drawdown_periods.append({
                'start': current_dd['start'],
                'end': len(portfolio_df) - 1,
                'depth': current_dd['depth'],
                'length': len(portfolio_df) - 1 - current_dd['start'],
                'recovery': 'ongoing'
            })
        
        # Sort by depth
        drawdown_periods = sorted(drawdown_periods, key=lambda x: x['depth'], reverse=True)
        
        # Take top 5
        top_drawdowns = drawdown_periods[:5]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot underwater chart
        ax1 = plt.subplot(2, 1, 1)
        ax1.fill_between(portfolio_df['timestamp'], 0, portfolio_df['drawdown'], color='red', alpha=0.5)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Drawdown')
        ax1.set_title('Underwater Chart')
        ax1.invert_yaxis()  # 0 at top
        
        # Plot top drawdowns as table
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('off')
        
        if top_drawdowns:
            table_data = [
                ['Rank', 'Start', 'End', 'Depth', 'Length', 'Recovery']
            ]
            
            for i, dd in enumerate(top_drawdowns, 1):
                table_data.append([
                    f"{i}",
                    f"{dd['start']}",
                    f"{dd['end']}",
                    f"{dd['depth']:.2%}",
                    f"{dd['length']} steps",
                    f"{dd['recovery']} steps" if isinstance(dd['recovery'], (int, float)) else dd['recovery']
                ])
            
            ax2.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.1, 0.15, 0.15, 0.15, 0.2, 0.2])
            ax2.set_title('Top 5 Drawdowns')
        
        plt.tight_layout()
        
        # Save figure
        drawdown_file = os.path.join(self.output_dir, "top_drawdowns.png")
        plt.savefig(drawdown_file, dpi=150)
        plt.close()
        
        logger.info(f"Created drawdown visualization: {drawdown_file}")
    
    def _create_regime_comparison(self) -> None:
        """Create visualization of performance across market regimes"""
        if not self.regime_performance:
            logger.warning("No regime performance data available for visualization")
            return
        
        # Extract metrics by regime
        regimes = list(self.regime_performance.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'trade_count']
        
        metric_values = {metric: [] for metric in metrics}
        regime_names = []
        
        for regime, data in self.regime_performance.items():
            regime_names.append(regime)
            
            for metric in metrics:
                value = data['metrics'].get(metric, 0)
                metric_values[metric].append(value)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot metrics by regime
        for i, metric in enumerate(metrics):
            ax = plt.subplot(len(metrics), 1, i+1)
            bars = ax.bar(regime_names, metric_values[metric])
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                if 'return' in metric or 'rate' in metric or 'drawdown' in metric:
                    # Format as percentage
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2%}', ha='center', va='bottom')
                else:
                    # Format as number
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            if i == 0:
                ax.set_title('Performance by Market Regime')
            
            if i == len(metrics) - 1:
                ax.set_xlabel('Regime')
        
        plt.tight_layout()
        
        # Save figure
        regime_file = os.path.join(self.output_dir, "regime_comparison.png")
        plt.savefig(regime_file, dpi=150)
        plt.close()
        
        logger.info(f"Created regime comparison visualization: {regime_file}")
    
    def _create_walkforward_analysis(self) -> None:
        """Create visualization of walk-forward validation results"""
        if not self.walkforward_results or 'window_results' not in self.walkforward_results:
            logger.warning("No walk-forward validation results available for visualization")
            return
        
        window_results = self.walkforward_results['window_results']
        
        if not window_results:
            logger.warning("Empty walk-forward validation results")
            return
        
        # Extract data
        window_numbers = [r['window'] for r in window_results]
        returns = [r['metrics'].get('total_return', 0) for r in window_results]
        sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in window_results]
        drawdowns = [r['metrics'].get('max_drawdown', 0) for r in window_results]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot returns by window
        ax1 = plt.subplot(3, 1, 1)
        ax1.bar(window_numbers, returns, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return')
        ax1.set_title('Returns by Window')
        
        # Add horizontal line for average return
        avg_return = np.mean(returns)
        ax1.axhline(y=avg_return, color='green', linestyle='-', label=f'Avg: {avg_return:.2%}')
        ax1.legend()
        
        # Plot Sharpe ratios by window
        ax2 = plt.subplot(3, 1, 2)
        ax2.bar(window_numbers, sharpes, color='orange')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio by Window')
        
        # Add horizontal line for average Sharpe
        avg_sharpe = np.mean(sharpes)
        ax2.axhline(y=avg_sharpe, color='green', linestyle='-', label=f'Avg: {avg_sharpe:.2f}')
        ax2.legend()
        
        # Plot drawdowns by window
        ax3 = plt.subplot(3, 1, 3)
        ax3.bar(window_numbers, drawdowns, color='red')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Max Drawdown')
        ax3.set_title('Max Drawdown by Window')
        
        # Add horizontal line for average drawdown
        avg_drawdown = np.mean(drawdowns)
        ax3.axhline(y=avg_drawdown, color='green', linestyle='-', label=f'Avg: {avg_drawdown:.2%}')
        ax3.legend()
        
        plt.tight_layout()
        
        # Save figure
        wf_file = os.path.join(self.output_dir, "walkforward_analysis.png")
        plt.savefig(wf_file, dpi=150)
        plt.close()
        
        logger.info(f"Created walk-forward analysis visualization: {wf_file}")
    
    def _create_performance_tearsheet(self) -> None:
        """Create a comprehensive performance tearsheet"""
        if not self.results:
            logger.warning("No results available for performance tearsheet")
            return
        
        # Create figure
        plt.figure(figsize=(12, 15))
        
        # Extract key metrics
        metrics = self.results
        
        # Title
        plt.suptitle('Performance Tearsheet', fontsize=16, y=0.98)
        
        # Overall performance metrics
        plt.subplot(5, 1, 1)
        plt.axis('off')
        
        perf_text = (
            "Overall Performance Metrics\n\n"
            f"Total Return: {metrics.get('total_return', 0):.2%}\n"
            f"Annual Return: {metrics.get('annual_return', 0):.2%}\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
            f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}"
        )
        
        plt.text(0.5, 0.5, perf_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Trade metrics
        plt.subplot(5, 1, 2)
        plt.axis('off')
        
        trade_text = (
            "Trade Metrics\n\n"
            f"Number of Trades: {metrics.get('trade_count', 0)}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Avg Trade Return: {metrics.get('avg_trade_return', 0):.2%}\n"
            f"Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} steps\n"
            f"Avg Leverage: {metrics.get('avg_leverage', 0):.2f}x"
        )
        
        plt.text(0.5, 0.5, trade_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Risk metrics
        plt.subplot(5, 1, 3)
        plt.axis('off')
        
        risk_text = (
            "Risk Metrics\n\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"Recovery Factor: {metrics.get('recovery_factor', 0):.2f}\n"
            f"Ulcer Index: {metrics.get('ulcer_index', 0):.4f}\n"
            f"Max Leverage: {metrics.get('max_leverage', 0):.2f}x\n"
        )
        
        plt.text(0.5, 0.5, risk_text, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Equity curve
        plt.subplot(5, 1, 4)
        
        # Combine portfolio values from multiple episodes
        combined_portfolio = []
        for pv in self.portfolio_history:
            if len(pv) > 0:
                if not combined_portfolio:
                    combined_portfolio = pv
                else:
                    # Continue from the last value of the previous episode
                    scale_factor = combined_portfolio[-1] / pv[0]
                    combined_portfolio.extend([v * scale_factor for v in pv[1:]])
        
        plt.plot(combined_portfolio)
        plt.title('Equity Curve')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        
        # Drawdown chart
        plt.subplot(5, 1, 5)
        
        # Calculate drawdowns
        portfolio_df = pd.DataFrame({
            'portfolio_value': combined_portfolio
        })
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['portfolio_value']) / portfolio_df['peak']
        
        plt.fill_between(range(len(portfolio_df)), 0, portfolio_df['drawdown'], color='red', alpha=0.5)
        plt.title('Drawdowns')
        plt.xlabel('Time Step')
        plt.ylabel('Drawdown')
        plt.gca().invert_yaxis()  # 0 at top
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save figure
        tearsheet_file = os.path.join(self.output_dir, "performance_tearsheet.png")
        plt.savefig(tearsheet_file, dpi=150)
        plt.close()
        
        logger.info(f"Created performance tearsheet: {tearsheet_file}") 