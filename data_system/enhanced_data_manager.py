from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from .data_manager import DataManager
from data_collection.collect_multimodal import MultiModalDataCollector
from .feature_engine import DerivativesFeatureEngine
import logging

logger = logging.getLogger(__name__)

class EnhancedDataManager:
    def __init__(self, base_path: str, config: Dict):
        self.base_path = base_path
        self.config = config
        self.data_manager = DataManager(base_path)
        self.multimodal_collector = MultiModalDataCollector()
        self.feature_engine = DerivativesFeatureEngine(
            volatility_window=config['feature_engineering']['volatility_window'],
            n_components=config['feature_engineering']['n_components']
        )
        
    async def fetch_and_process_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and process both price and multimodal data"""
        logger.info("Fetching multimodal data...")
        
        # Collect multimodal data
        self.multimodal_collector.collect_and_save_data(
            start_date=start_date,
            end_date=end_date,
            output_path=f"{self.base_path}/multimodal_data.parquet"
        )
        
        # Load the saved multimodal data
        multimodal_data = pd.read_parquet(f"{self.base_path}/multimodal_data.parquet")
        
        # Fetch derivative data using existing system
        logger.info("Fetching derivative data...")
        derivative_data = await self.data_manager.load_market_data(
            exchange="binance",
            symbol="BTCUSDT",
            timeframe=self.config['data']['timeframe'],
            start_time=start_date,
            end_time=end_date,
            data_type='perpetual'
        )
        
        # Combine the data sources
        combined_data = self._combine_data_sources(derivative_data, multimodal_data)
        
        # Engineer features
        processed_data = self.feature_engine.engineer_features(combined_data)
        
        return processed_data
    
    def _combine_data_sources(self, derivative_data: pd.DataFrame, multimodal_data: pd.DataFrame) -> pd.DataFrame:
        """Combine derivative and multimodal data with proper alignment"""
        # Ensure both dataframes have timestamp as index
        derivative_data = derivative_data.set_index('timestamp')
        multimodal_data = multimodal_data.set_index('timestamp')
        
        # Resample both to common timeframe
        timeframe = self.config['data']['timeframe']
        derivative_data = derivative_data.resample(timeframe).last()
        multimodal_data = multimodal_data.resample(timeframe).last()
        
        # Merge the dataframes
        combined = pd.merge(
            derivative_data,
            multimodal_data,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Forward fill missing values then backward fill
        combined = combined.ffill().bfill()
        
        return combined 