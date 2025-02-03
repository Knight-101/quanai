import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class DataManager:
    """Manages market data storage and retrieval using local Parquet files"""
    
    def __init__(self, base_path: str = 'data'):
        """Initialize DataManager with base path for data storage"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.market_data_path = self.base_path / 'market_data'
        self.feature_data_path = self.base_path / 'features'
        self.market_data_path.mkdir(exist_ok=True)
        self.feature_data_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized DataManager with base path: {base_path}")
    
    def _get_market_data_filename(self, exchange: str, symbol: str, timeframe: str) -> str:
        """Generate filename for market data"""
        return f"{exchange}_{symbol.replace('/', '_')}_{timeframe}.parquet"
    
    def _get_feature_data_filename(self, feature_set: str) -> str:
        """Generate filename for feature data"""
        return f"{feature_set}.parquet"
    
    def save_market_data(
        self,
        data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_type: str = 'perpetual'
    ) -> bool:
        """Save market data to Parquet file"""
        try:
            filename = self._get_market_data_filename(exchange, symbol, timeframe)
            filepath = self.market_data_path / filename
            
            # Add metadata as columns
            data = data.copy()
            data['_exchange'] = exchange
            data['_symbol'] = symbol
            data['_timeframe'] = timeframe
            data['_data_type'] = data_type
            data['_saved_at'] = datetime.now()
            
            # Save to Parquet
            data.to_parquet(filepath)
            logger.info(f"Saved market data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
            return False
    
    def load_market_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        data_type: str = 'perpetual'
    ) -> Optional[pd.DataFrame]:
        """Load market data from Parquet file with optional time filtering"""
        try:
            filename = self._get_market_data_filename(exchange, symbol, timeframe)
            filepath = self.market_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"No data file found at {filepath}")
                return None
            
            # Load data
            data = pd.read_parquet(filepath)
            
            # Filter by time range if specified
            if start_time is not None:
                data = data[data.index >= start_time]
            if end_time is not None:
                data = data[data.index <= end_time]
            
            # Remove metadata columns
            for col in ['_exchange', '_symbol', '_timeframe', '_data_type', '_saved_at']:
                if col in data.columns:
                    data = data.drop(columns=[col])
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return None
    
    def save_feature_data(
        self,
        data: pd.DataFrame,
        feature_set: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Save feature data to Parquet file"""
        try:
            filename = self._get_feature_data_filename(feature_set)
            filepath = self.feature_data_path / filename
            
            # Save metadata separately
            metadata_file = filepath.with_suffix('.meta.parquet')
            pd.DataFrame([metadata]).to_parquet(metadata_file)
            
            # Create a copy to avoid modifying the original data
            data_to_save = data.copy()
            
            # Convert MultiIndex columns to string format for saving
            if isinstance(data_to_save.columns, pd.MultiIndex):
                # Convert None values to 'unknown' and ensure string type
                new_columns = []
                for col in data_to_save.columns:
                    asset, feature = col
                    asset = str(asset) if asset is not None else 'unknown'
                    feature = str(feature) if feature is not None else 'unknown'
                    new_columns.append(f"{asset}|{feature}")
                data_to_save.columns = new_columns
            
            # Ensure all columns are numeric and handle non-numeric data
            for col in data_to_save.columns:
                try:
                    if not np.issubdtype(data_to_save[col].dtype, np.number):
                        # Try to convert to numeric, replace non-convertible values with NaN
                        data_to_save[col] = pd.to_numeric(data_to_save[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                    # If conversion fails, drop the problematic column
                    data_to_save = data_to_save.drop(columns=[col])
            
            # Fill any NaN values with 0
            data_to_save = data_to_save.fillna(0)
            
            # Save feature data
            data_to_save.to_parquet(filepath)
            logger.info(f"Saved feature data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feature data: {str(e)}")
            return False
    
    def load_feature_data(
        self,
        feature_set: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Load feature data from Parquet file with optional time filtering"""
        try:
            filename = self._get_feature_data_filename(feature_set)
            filepath = self.feature_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"No feature data file found at {filepath}")
                return None
            
            # Load data
            data = pd.read_parquet(filepath)
            
            # Convert string columns back to MultiIndex
            if all('|' in col for col in data.columns):
                data.columns = pd.MultiIndex.from_tuples(
                    [tuple(col.split('|')) for col in data.columns],
                    names=['asset', 'feature']
                )
            
            # Filter by time range if specified
            if start_time is not None:
                data = data[data.index >= start_time]
            if end_time is not None:
                data = data[data.index <= end_time]
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading feature data: {str(e)}")
            return None
    
    def get_available_data_ranges(self) -> Dict[str, Dict[str, datetime]]:
        """Get available data ranges for all stored data"""
        ranges = {}
        
        # Scan market data
        for file in self.market_data_path.glob('*.parquet'):
            if file.suffix == '.parquet':
                try:
                    data = pd.read_parquet(file)
                    name = file.stem
                    ranges[name] = {
                        'start': data.index.min(),
                        'end': data.index.max(),
                        'rows': len(data)
                    }
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
        
        return ranges 