import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from collect_multimodal import MultiModalDataCollector
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import json
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.required_columns = {
            'price': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'tweets': ['timestamp', 'text', 'likes', 'retweets'],
            'news': ['timestamp', 'title', 'description'],
            'onchain': ['timestamp', 'active_addresses', 'transaction_count', 'transaction_volume']
        }
    
    def validate_data_completeness(self, data: pd.DataFrame) -> Dict:
        """Check for missing values and data completeness"""
        missing_stats = {
            'missing_percentage': data.isnull().mean() * 100,
            'complete_rows': len(data.dropna()),
            'total_rows': len(data),
            'completeness_ratio': len(data.dropna()) / len(data)
        }
        return missing_stats
    
    def validate_time_continuity(self, data: pd.DataFrame) -> Dict:
        """Check for gaps in time series data"""
        data = data.sort_values('timestamp')
        time_diffs = data['timestamp'].diff()
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
        
        return {
            'total_gaps': len(gaps),
            'max_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta(0),
            'gap_locations': gaps.index.tolist()
        }
    
    def plot_data_availability(self, data: pd.DataFrame, output_path: str):
        """Plot data availability across different sources"""
        plt.figure(figsize=(15, 8))
        
        # Create availability matrix
        availability = pd.DataFrame({
            'Price Data': ~data[self.required_columns['price']].isnull().any(axis=1),
            'Tweet Data': ~data[['text', 'likes', 'retweets']].isnull().any(axis=1),
            'News Data': ~data[['title', 'description']].isnull().any(axis=1),
            'On-chain Data': ~data[['active_addresses', 'transaction_count']].isnull().any(axis=1)
        })
        
        sns.heatmap(availability.T, cmap='RdYlGn', cbar_kws={'label': 'Data Available'})
        plt.title('Data Availability Across Sources')
        plt.xlabel('Time')
        plt.savefig(output_path)
        plt.close()

def plot_whale_activity(df: pd.DataFrame, output_path: str):
    """Plot whale activity scores over time for each asset"""
    plt.figure(figsize=(15, 8))
    for asset in df['asset'].unique():
        asset_data = df[df['asset'] == asset]
        plt.plot(asset_data['timestamp'], asset_data['whale_activity_score'], label=asset)
    
    plt.title('Whale Activity Score Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Whale Activity Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def test_collection():
    """Run a test collection for the last 24 hours"""
    logger.info("Starting test collection...")
    
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Initialize collector
    collector = MultiModalDataCollector()
    
    # Set time range for last 24 hours
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=24)
    
    try:
        # Collect data
        collector.collect_and_save_data(
            start_date=start_date,
            end_date=end_date,
            output_path='test_results/test_multi_modal.parquet'
        )
        
        # Load and analyze the collected data
        df = pd.read_parquet('test_results/test_multi_modal.parquet')
        
        # Generate test report
        report = {
            'collection_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'data_stats': {
                'total_rows': len(df),
                'assets_covered': df['asset'].unique().tolist(),
                'metrics_collected': df.columns.tolist(),
                'missing_data_ratio': df.isnull().mean().to_dict()
            }
        }
        
        # Add whale activity stats if available
        if 'whale_activity_score' in df.columns:
            report['whale_activity'] = {
                asset: {
                    'avg_score': float(df[df['asset'] == asset]['whale_activity_score'].mean()),
                    'max_score': float(df[df['asset'] == asset]['whale_activity_score'].max()),
                    'significant_events': int(df[(df['asset'] == asset) & 
                                              (df['whale_activity_score'] > 0.7)].shape[0])
                }
                for asset in df['asset'].unique()
            }
        else:
            report['whale_activity'] = "Whale activity data not available"
        
        # Save report
        with open('test_results/test_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Plot data availability
        plt.figure(figsize=(15, 8))
        
        # Plot price data
        plt.subplot(2, 1, 1)
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset]
            plt.plot(asset_data['timestamp'], asset_data['close'], label=f"{asset} Price")
        plt.title('Asset Prices Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Plot whale activity if available
        if 'whale_activity_score' in df.columns:
            plt.subplot(2, 1, 2)
            for asset in df['asset'].unique():
                asset_data = df[df['asset'] == asset]
                plt.plot(asset_data['timestamp'], asset_data['whale_activity_score'], label=f"{asset} Whale Activity")
            plt.title('Whale Activity Score Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Activity Score')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('test_results/data_visualization.png')
        plt.close()
        
        logger.info("Test collection completed successfully!")
        logger.info(f"Results saved in test_results/")
        
        # Print summary
        print("\nTest Collection Summary:")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total rows collected: {report['data_stats']['total_rows']}")
        print("\nMetrics collected:")
        for metric in report['data_stats']['metrics_collected']:
            missing_ratio = report['data_stats']['missing_data_ratio'].get(metric, 0)
            print(f"  {metric}: {(1 - missing_ratio) * 100:.1f}% complete")
        
        if isinstance(report['whale_activity'], dict):
            print("\nWhale Activity Summary:")
            for asset, stats in report['whale_activity'].items():
                print(f"\n{asset}:")
                print(f"  Average Score: {stats['avg_score']:.3f}")
                print(f"  Maximum Score: {stats['max_score']:.3f}")
                print(f"  Significant Events: {stats['significant_events']}")
        else:
            print("\nWhale Activity: Not available")
        
    except Exception as e:
        logger.error(f"Error during test collection: {str(e)}")
        raise

if __name__ == "__main__":
    test_collection() 