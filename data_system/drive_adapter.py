import os
import json
import pandas as pd
import requests
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Import the original DataManager to inherit from it
from data_system.data_manager import DataManager

logger = logging.getLogger(__name__)

class DriveAdapter(DataManager):
    """
    Adapter that augments DataManager to load from Google Drive when local files don't exist.
    This maintains full compatibility with existing code while adding Google Drive functionality.
    """
    
    def __init__(self, base_path: str = 'data', drive_ids_file: str = None):
        """
        Initialize the adapter with the original DataManager functionality plus Google Drive.
        
        Args:
            base_path: Local path for data - same as DataManager
            drive_ids_file: Path to JSON file containing drive file IDs, or dict of file IDs
        """
        # Initialize the parent DataManager
        super().__init__(base_path)
        
        # Map of file paths to Google Drive file IDs
        self.drive_file_ids = {}
        
        # Load drive file IDs if provided
        if drive_ids_file:
            self._load_drive_ids(drive_ids_file)
    
    def _load_drive_ids(self, drive_ids_source):
        """Load file IDs from file or dict"""
        try:
            if isinstance(drive_ids_source, dict):
                self.drive_file_ids = drive_ids_source
                logger.info(f"Loaded {len(drive_ids_source)} Google Drive file IDs from dict")
            elif os.path.exists(drive_ids_source):
                with open(drive_ids_source, 'r') as f:
                    self.drive_file_ids = json.load(f)
                logger.info(f"Loaded {len(self.drive_file_ids)} Google Drive file IDs from {drive_ids_source}")
                logger.debug(f"Drive file IDs: {self.drive_file_ids}")
            else:
                logger.warning(f"Drive IDs file not found: {drive_ids_source}")
        except Exception as e:
            logger.error(f"Error loading drive file IDs: {e}")
    
    def _get_drive_key(self, file_path: str) -> str:
        """Convert a file path to a drive file ID key"""
        # Handle paths with or without the base_path prefix
        rel_path = str(file_path)
        if rel_path.startswith(str(self.base_path)):
            rel_path = os.path.relpath(rel_path, str(self.base_path))
        
        # Remove leading slash if present
        if rel_path.startswith('/') or rel_path.startswith('\\'):
            rel_path = rel_path[1:]
            
        drive_key = rel_path.replace('\\', '/')
        logger.debug(f"Converted local path '{file_path}' to drive key '{drive_key}'")
        return drive_key
    
    def _download_from_drive(self, file_path: str) -> bool:
        """
        Download a file from Google Drive to the specified local path
        
        Args:
            file_path: Local path where the file should be saved
            
        Returns:
            bool: True if download successful, False otherwise
        """
        drive_key = self._get_drive_key(file_path)
        file_id = self.drive_file_ids.get(drive_key)
        
        if not file_id:
            logger.info(f"No drive file ID found for path: '{drive_key}'")
            logger.debug(f"Available drive keys: {list(self.drive_file_ids.keys())}")
            return False
        
        try:
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            logger.info(f"Downloading file from Google Drive: '{drive_key}' with ID: {file_id}")
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to download file from Drive. Status code: {response.status_code}")
                logger.error(f"Response content: {response.content[:500]}")
                return False
            
            # Make sure the directory exists
            local_path = Path(file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Successfully downloaded from Drive to: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from Google Drive: {str(e)}")
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
        """
        Enhanced load_market_data that tries Google Drive if local file not found
        """
        # First try to load data using the parent DataManager
        data = super().load_market_data(exchange, symbol, timeframe, start_time, end_time, data_type)
        
        if data is not None:
            logger.info(f"Found local market data for {exchange}_{symbol}_{timeframe} with {len(data)} data points")
            # Ensure data has proper MultiIndex columns
            data = self._ensure_multiindex_structure(data, symbol)
            return data
        
        # If data not found locally, try to get it from Google Drive
        try:
            logger.info(f"Local market data not found for {exchange}_{symbol}_{timeframe}, trying Google Drive")
            filename = self._get_market_data_filename(exchange, symbol, timeframe)
            filepath = self.market_data_path / filename
            
            # Try to download from Google Drive
            drive_key = f"market_data/{filename}"
            logger.info(f"Looking for Drive key: '{drive_key}'")
            if drive_key in self.drive_file_ids:
                if self._download_from_drive(filepath):
                    # Now try loading again
                    logger.info(f"Attempting to load downloaded market data for {exchange}_{symbol}_{timeframe}")
                    loaded_data = super().load_market_data(exchange, symbol, timeframe, start_time, end_time, data_type)
                    if loaded_data is not None:
                        logger.info(f"Successfully loaded {len(loaded_data)} data points from downloaded file")
                        # Ensure data has proper MultiIndex columns
                        loaded_data = self._ensure_multiindex_structure(loaded_data, symbol)
                    else:
                        logger.warning(f"Downloaded file loaded but returned None")
                    return loaded_data
                else:
                    logger.warning(f"Failed to download market data from Drive for {exchange}_{symbol}_{timeframe}")
            else:
                logger.info(f"No Drive ID found for market data: '{drive_key}'")
            
            # Try perpetual data location as well
            if data_type == 'perpetual':
                perp_key = f"perpetual/{exchange}/{timeframe}/{symbol.replace('/', '_')}.parquet"
                perp_path = self.base_path / perp_key
                logger.info(f"Looking for Drive key (perpetual): '{perp_key}'")
                
                if perp_key in self.drive_file_ids:
                    if self._download_from_drive(perp_path):
                        # Load the perpetual data directly
                        logger.info(f"Loading downloaded perpetual data for {exchange}_{symbol}_{timeframe}")
                        data = pd.read_parquet(perp_path)
                        
                        # Filter by time range
                        if start_time is not None:
                            data = data[data.index >= start_time]
                        if end_time is not None:
                            data = data[data.index <= end_time]
                        
                        logger.info(f"Successfully loaded {len(data)} data points from downloaded perpetual file")
                        # Ensure data has proper MultiIndex columns
                        data = self._ensure_multiindex_structure(data, symbol)
                        return data
                    else:
                        logger.warning(f"Failed to download perpetual data from Drive for {exchange}_{symbol}_{timeframe}")
                else:
                    logger.info(f"No Drive ID found for perpetual data: '{perp_key}'")
            
            logger.warning(f"Could not find market data locally or on Google Drive for {exchange}_{symbol}_{timeframe}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data from Google Drive: {str(e)}", exc_info=True)
            return None
    
    def _ensure_multiindex_structure(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Ensure the DataFrame has the proper MultiIndex column structure expected by the feature engine.
        This converts flat column structure to MultiIndex with (asset, feature) levels.
        """
        # Skip if already has MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            logger.info(f"Data already has MultiIndex columns - no conversion needed")
            return df
        
        logger.info(f"Converting flat columns to MultiIndex structure for {symbol}")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create MultiIndex columns with (asset, feature) structure
        result_df.columns = pd.MultiIndex.from_product(
            [[symbol], result_df.columns],
            names=['asset', 'feature']
        )
        
        logger.info(f"Converted to MultiIndex columns: {result_df.columns.names}")
        return result_df
    
    def load_feature_data(
        self,
        feature_set: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Enhanced load_feature_data that tries Google Drive if local file not found
        """
        # First try to load using parent method
        logger.info(f"Attempting to load feature data for '{feature_set}' from local files...")
        data = super().load_feature_data(feature_set, start_time, end_time)
        
        if data is not None:
            logger.info(f"✅ Found local feature data for '{feature_set}' with {len(data)} rows")
            # Features should already have MultiIndex from load_feature_data
            if not isinstance(data.columns, pd.MultiIndex):
                logger.warning(f"Feature data doesn't have MultiIndex columns, attempting to fix")
                # Try to convert if using the pipe format
                if any("|" in col for col in data.columns):
                    try:
                        data.columns = pd.MultiIndex.from_tuples(
                            [tuple(col.split('|')) for col in data.columns],
                            names=['asset', 'feature']
                        )
                        logger.info(f"Successfully converted feature data to MultiIndex")
                    except Exception as e:
                        logger.error(f"Failed to convert feature data to MultiIndex: {str(e)}")
            return data
        
        # If not found locally, try Google Drive
        try:
            logger.info(f"⚠️ Local feature data not found for '{feature_set}', checking Google Drive...")
            filename = self._get_feature_data_filename(feature_set)
            filepath = self.feature_data_path / filename
            
            # Try to download from Google Drive
            drive_key = f"features/{filename}"
            logger.info(f"Looking for Google Drive key: '{drive_key}'")
            if drive_key in self.drive_file_ids:
                logger.info(f"🔍 Found matching Drive ID for '{drive_key}': {self.drive_file_ids[drive_key]}")
                if self._download_from_drive(filepath):
                    # Also try to download metadata file
                    meta_file = filepath.with_suffix('.meta.json')
                    meta_key = f"features/{feature_set}.meta.json"
                    logger.info(f"Looking for feature metadata: '{meta_key}'")
                    if meta_key in self.drive_file_ids:
                        self._download_from_drive(meta_file)
                    
                    # Now try loading again
                    logger.info(f"Attempting to load downloaded feature data for '{feature_set}'")
                    loaded_data = super().load_feature_data(feature_set, start_time, end_time)
                    
                    if loaded_data is not None:
                        logger.info(f"✅ Successfully loaded feature data from Google Drive with {len(loaded_data)} rows")
                    else:
                        logger.warning(f"❌ Downloaded feature file but couldn't load data from it")
                    
                    # Fix MultiIndex if needed
                    if loaded_data is not None and not isinstance(loaded_data.columns, pd.MultiIndex):
                        logger.warning(f"Downloaded feature data doesn't have MultiIndex columns, attempting to fix")
                        # Try to convert if using the pipe format
                        if any("|" in col for col in loaded_data.columns):
                            try:
                                loaded_data.columns = pd.MultiIndex.from_tuples(
                                    [tuple(col.split('|')) for col in loaded_data.columns],
                                    names=['asset', 'feature']
                                )
                                logger.info(f"Successfully converted feature data to MultiIndex")
                            except Exception as e:
                                logger.error(f"Failed to convert feature data to MultiIndex: {str(e)}")
                    
                    return loaded_data
                else:
                    logger.warning(f"❌ Failed to download feature data from Drive for '{feature_set}'")
            else:
                logger.info(f"❌ No Drive ID found for feature data: '{drive_key}'")
            
            logger.warning(f"❌ Could not find feature data locally or on Google Drive for '{feature_set}'")
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature data from Google Drive: {str(e)}", exc_info=True)
            return None 