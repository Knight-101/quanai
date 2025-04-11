#!/usr/bin/env python3
"""
Backtest Data Fetcher

This module provides data fetching functionality for backtesting
with compatibility with the training data system.
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Import local modules
from data_system.derivative_data_fetcher import PerpetualDataFetcher
from data_system.data_manager import DataManager
from data_system.drive_adapter import DriveAdapter
from data_system.feature_engine import DerivativesFeatureEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestDataFetcher")

class BacktestDataFetcher:
    """
    Data fetcher for backtesting that mirrors the training data logic.
    This ensures feature extraction compatibility with trained models.
    """
    
    def __init__(
        self,
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframe="5m",
        start_date=None,
        end_date=None,
        lookback_days=365,
        exchange="binance",
        data_dir="data",
        drive_ids_file="drive_file_ids.json"
    ):
        """
        Initialize the backtesting data fetcher
        
        Args:
            symbols: List of trading symbols to fetch
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            lookback_days: Number of days to look back if start_date is not provided
            exchange: Trading exchange to fetch from
            data_dir: Directory for data storage
            drive_ids_file: Path to JSON file with Google Drive file IDs
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.exchange = exchange
        self.lookback_days = lookback_days
        
        # Process dates
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        if start_date:
            self.start_date = pd.Timestamp(start_date)
        else:
            self.start_date = self.end_date - timedelta(days=lookback_days)
            
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        # Map symbols for fetching and processing
        self.symbol_mappings = {
            "BTC/USDT": "BTC",
            "ETH/USDT": "ETH", 
            "SOL/USDT": "SOL"
        }
        
        # Initialize data manager with Drive adapter for Google Drive support
        if os.path.exists(drive_ids_file):
            logger.info(f"Using Google Drive adapter with file IDs from {drive_ids_file}")
            self.data_manager = DriveAdapter(base_path=data_dir, drive_ids_file=drive_ids_file)
        else:
            logger.info(f"Using regular data manager (no Google Drive)")
            self.data_manager = DataManager(base_path=data_dir)
            
        # Initialize data fetcher
        self.data_fetcher = PerpetualDataFetcher(
            symbols=symbols,
            timeframe=timeframe,
            exchanges=[exchange]
        )
        self.data_fetcher.lookback = lookback_days
        
        # Initialize feature engine
        self.feature_engine = DerivativesFeatureEngine()
        
    async def run(self, output_path=None):
        """
        Run the complete data fetching process
        
        Args:
            output_path: Path to save the output data
            
        Returns:
            Processed DataFrame ready for backtesting
        """
        logger.info(f"Starting data fetching process for symbols: {self.symbols}")
        
        # Try to load cached data first (same as training logic)
        logger.info("Attempting to load cached data...")
        all_data = self._load_cached_data()
        
        if all_data is None:
            logger.info("No cached data found or insufficient history. Fetching new data...")
            
            # Fetch new data from exchange
            data = await self.data_fetcher.fetch_derivative_data()
            
            if not data:
                logger.error("Failed to fetch data from exchanges")
                return None
                
            # Save the raw market data
            logger.info("Saving raw market data...")
            self._save_market_data(data)
            
            all_data = data
        
        # Format data for backtesting (same as training)
        logger.info("Processing data for backtesting...")
        processed_data = self._format_data_for_training(all_data)
        
        # Save processed data if output path provided
        if output_path:
            logger.info(f"Saving processed data to {output_path}")
            processed_data.to_parquet(output_path)
            
        logger.info(f"Data ready for backtesting with shape: {processed_data.shape}")
        return processed_data
        
    def _load_cached_data(self):
        """
        Helper method to load cached data.
        Follows the same logic as in training to ensure compatibility.
        """
        all_data = {self.exchange: {}}
        has_all_data = True
        
        logger.info(f"Checking for cached data for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            logger.info(f"Loading data for {self.exchange}_{symbol}_{self.timeframe}")
            
            # Load data with time constraints
            data = self.data_manager.load_market_data(
                exchange=self.exchange,
                symbol=symbol,
                timeframe=self.timeframe,
                start_time=self.start_date,
                end_time=self.end_date,
                data_type='perpetual'
            )
            
            # Check if we have enough data
            min_required_points = 100  # Minimum data points needed for backtesting
            if data is None:
                logger.warning(f"No data found for {self.exchange}_{symbol}")
                has_all_data = False
                break
            elif len(data) < min_required_points:
                logger.warning(f"Insufficient data for {self.exchange}_{symbol}: found {len(data)} points, need {min_required_points}")
                has_all_data = False
                break
            else:
                logger.info(f"Sufficient data found for {self.exchange}_{symbol}: {len(data)} points")
                all_data[self.exchange][symbol] = data
            
        if has_all_data:
            logger.info("All required data found in cache, using existing data")
            return all_data
        else:
            logger.info("Incomplete data in cache, will fetch fresh data")
            return None
            
    def _save_market_data(self, raw_data):
        """
        Helper method to save market data.
        Ensures compatibility with training data system.
        """
        for exchange, exchange_data in raw_data.items():
            for symbol, symbol_data in exchange_data.items():
                self.data_manager.save_market_data(
                    data=symbol_data,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    data_type='perpetual'
                )
                
    def _format_data_for_training(self, raw_data):
        """
        Format raw data into the structure expected by the backtester.
        This ensures feature compatibility with trained models.
        """
        # Aggregate data from all exchanges
        all_dataframes = []
        
        # Process each exchange's data
        for exchange, exchange_data in raw_data.items():
            for symbol, data in exchange_data.items():
                # Create a copy with MultiIndex columns
                processed_df = data.copy()
                
                # Map to short symbol name (e.g. BTC/USDT -> BTC)
                short_symbol = self.symbol_mappings.get(symbol, symbol)
                
                # Create MultiIndex columns with (asset, feature) format
                processed_df.columns = pd.MultiIndex.from_product(
                    [[short_symbol], processed_df.columns],
                    names=['asset', 'feature']
                )
                
                all_dataframes.append(processed_df)
        
        # Combine all dataframes
        if not all_dataframes:
            logger.error("No data to process")
            return None
            
        combined_df = pd.concat(all_dataframes, axis=1)
        combined_df = combined_df.fillna(method='ffill').dropna()
        
        # Generate technical features using the same engine as training
        logger.info("Generating technical features...")
        feature_config = {
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_advanced": True,
            "window_sizes": [14, 20, 50]
        }
        
        assets = [self.symbol_mappings.get(s, s) for s in self.symbols]
        with_features = self.feature_engine.transform(combined_df)
        
        logger.info(f"Final data shape: {with_features.shape}")
        return with_features

class CustomBacktestDataFetcher(BacktestDataFetcher):
    """
    Custom implementation of BacktestDataFetcher that is compatible with
    existing file naming conventions where symbols don't have underscores between base/quote.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with parent class constructor"""
        super().__init__(*args, **kwargs)
        
        # Create compatible symbol format mappings for file access
        self.file_symbol_mappings = {}
        for symbol in self.symbols:
            # Convert "BTC/USDT" to "BTCUSDT" for file access
            self.file_symbol_mappings[symbol] = symbol.replace('/', '')
    
    def _load_cached_data(self):
        """
        Override to use compatible symbol format for file access
        """
        all_data = {self.exchange: {}}
        has_all_data = True
        
        logger.info(f"Checking for cached data for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            # Use compatible symbol format for file access
            file_symbol = self.file_symbol_mappings.get(symbol, symbol)
            logger.info(f"Loading data for {self.exchange}_{file_symbol}_{self.timeframe}")
            
            # Load data with time constraints
            data = self.data_manager.load_market_data(
                exchange=self.exchange,
                symbol=file_symbol,  # Use compatible format
                timeframe=self.timeframe,
                start_time=self.start_date,
                end_time=self.end_date,
                data_type='perpetual'
            )
            
            # Check if we have enough data
            min_required_points = 100  # Minimum data points needed for backtesting
            if data is None:
                logger.warning(f"No data found for {self.exchange}_{symbol}")
                has_all_data = False
                break
            elif len(data) < min_required_points:
                logger.warning(f"Insufficient data for {self.exchange}_{symbol}: found {len(data)} points, need {min_required_points}")
                has_all_data = False
                break
            else:
                logger.info(f"Sufficient data found for {self.exchange}_{symbol}: {len(data)} points")
                all_data[self.exchange][symbol] = data
            
        if has_all_data:
            logger.info("All required data found in cache, using existing data")
            return all_data
        else:
            logger.info("Incomplete data in cache, will fetch fresh data")
            return None
    
    def _save_market_data(self, raw_data):
        """
        Override to use compatible symbol format for file saving
        """
        for exchange, exchange_data in raw_data.items():
            for symbol, symbol_data in exchange_data.items():
                # Use compatible symbol format for file access
                file_symbol = self.file_symbol_mappings.get(symbol, symbol)
                
                self.data_manager.save_market_data(
                    data=symbol_data,
                    exchange=exchange,
                    symbol=file_symbol,  # Use compatible format
                    timeframe=self.timeframe,
                    data_type='perpetual'
                )
    
    def _format_data_for_training(self, raw_data):
        """
        Override the parent method to handle data that might already have MultiIndex columns.
        """
        # Aggregate data from all exchanges
        all_dataframes = []
        
        # Process each exchange's data
        for exchange, exchange_data in raw_data.items():
            for symbol, data in exchange_data.items():
                # Create a copy
                processed_df = data.copy()
                
                # Check if columns are already MultiIndex
                if not isinstance(processed_df.columns, pd.MultiIndex):
                    # Map to short symbol name (e.g. BTC/USDT -> BTC)
                    short_symbol = self.symbol_mappings.get(symbol, symbol)
                    
                    # Create MultiIndex columns with (asset, feature) format
                    logger.info(f"Converting columns to MultiIndex for {short_symbol}")
                    processed_df.columns = pd.MultiIndex.from_product(
                        [[short_symbol], processed_df.columns],
                        names=['asset', 'feature']
                    )
                else:
                    # Already MultiIndex, make sure the asset level matches our mapping
                    logger.info(f"Columns already MultiIndex for {symbol}, ensuring correct asset level")
                    current_assets = processed_df.columns.get_level_values(0).unique()
                    short_symbol = self.symbol_mappings.get(symbol, symbol)
                    
                    # If there's only one asset in the index and it doesn't match what we expect,
                    # we need to rename it
                    if len(current_assets) == 1 and current_assets[0] != short_symbol:
                        logger.info(f"Renaming asset from {current_assets[0]} to {short_symbol}")
                        # Create a mapping dictionary
                        level_map = {current_assets[0]: short_symbol}
                        # Create new columns with renamed level 0
                        new_cols = processed_df.columns.set_levels(
                            [pd.Index([level_map.get(x, x) for x in lev]) if i == 0 else lev 
                             for i, lev in enumerate(processed_df.columns.levels)]
                        )
                        processed_df.columns = new_cols
                
                all_dataframes.append(processed_df)
        
        # Combine all dataframes
        if not all_dataframes:
            logger.error("No data to process")
            return None
            
        combined_df = pd.concat(all_dataframes, axis=1)
        combined_df = combined_df.fillna(method='ffill').dropna()
        
        # Generate technical features using the same engine as training
        logger.info("Generating technical features...")
        feature_config = {
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_advanced": True,
            "window_sizes": [14, 20, 50]
        }
        
        assets = [self.symbol_mappings.get(s, s) for s in self.symbols]
        with_features = self.feature_engine.transform(combined_df)
        
        logger.info(f"Final data shape: {with_features.shape}")
        return with_features

# For testing
async def main():
    """Test the data fetcher"""
    fetcher = BacktestDataFetcher()
    data = await fetcher.run()
    print(f"Fetched data with shape: {data.shape}")
    print(f"Assets: {data.columns.get_level_values(0).unique().tolist()}")
    print(f"Features: {data.columns.get_level_values(1).unique().tolist()[:10]}")

if __name__ == "__main__":
    asyncio.run(main()) 