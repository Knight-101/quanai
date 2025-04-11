#!/usr/bin/env python
"""
Command-line script for running institutional-grade backtests.

This script provides a convenient interface for running backtests
using the InstitutionalBacktester class.
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any
import json
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backtest_runner")

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the backtester
from backtesting.institutional_backtester import run_institutional_backtest
from data_system.feature_engine import DerivativesFeatureEngine
from data_system.data_manager import DataManager
from data_system.drive_adapter import DriveAdapter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run institutional-grade backtests for trading models")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    
    # Data source (one of these required)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-path", type=str,
                         help="Path to market data file (parquet, csv, pickle)")
    data_group.add_argument("--data-dir", type=str,
                         help="Directory containing market data files (will use most recent)")
    
    # Optional arguments
    parser.add_argument("--assets", type=str, nargs="+",
                        help="List of assets to trade (if not specified, will be detected from data)")
    parser.add_argument("--initial-capital", type=float, default=10000.0,
                        help="Initial capital for backtesting (default: 10000.0)")
    parser.add_argument("--start-date", type=str,
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                        help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--risk-free-rate", type=float, default=0.02,
                        help="Annual risk-free rate (default: 0.02)")
    parser.add_argument("--commission", type=float, default=0.0004,
                        help="Trading commission (default: 0.0004)")
    parser.add_argument("--slippage", type=float, default=0.0002,
                        help="Trading slippage (default: 0.0002)")
    parser.add_argument("--max-leverage", type=float, default=10.0,
                        help="Maximum allowed leverage (default: 10.0)")
    parser.add_argument("--output-dir", type=str, default="backtest_results",
                        help="Directory for output files (default: backtest_results)")
    parser.add_argument("--benchmark", type=str,
                        help="Symbol for benchmark comparison")
    parser.add_argument("--no-regime-analysis", action="store_true",
                        help="Disable market regime analysis")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--walk-forward-window", type=int, default=60,
                        help="Window size for walk-forward validation in days (default: 60)")
    parser.add_argument("--walk-forward-step", type=int, default=30,
                        help="Step size for walk-forward validation in days (default: 30)")
    parser.add_argument("--use-gdrive", action="store_true",
                        help="Enable Google Drive support for data loading")
    parser.add_argument("--drive-ids-file", type=str, default="drive_file_ids.json",
                        help="Path to the Google Drive file IDs JSON (default: drive_file_ids.json)")
    parser.add_argument("--generate-features", action="store_true",
                        help="Generate features even if data already has features")
    parser.add_argument("--feature-config", type=str, default=None,
                        help="JSON string with feature generation configuration")
    
    return parser.parse_args()


def find_latest_data_file(data_dir: str) -> str:
    """Find the most recent data file in the given directory"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    # Look for data files
    data_files = []
    for ext in ['.parquet', '.pkl', '.pickle', '.csv']:
        data_files.extend(list(Path(data_dir).glob(f"*{ext}")))
        
    if not data_files:
        raise ValueError(f"No data files found in directory: {data_dir}")
        
    # Sort by modification time (newest first)
    data_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return the most recent file
    latest_file = str(data_files[0])
    logger.info(f"Using most recent data file: {latest_file}")
    
    return latest_file


def load_and_prepare_data(
    data_path: str, 
    assets: List[str] = None, 
    start_date: str = None, 
    end_date: str = None,
    generate_features: bool = False,
    feature_config: Dict = None
) -> pd.DataFrame:
    """
    Load and prepare data for backtesting, ensuring it has proper format and features.
    
    Args:
        data_path: Path to the data file
        assets: List of assets to include
        start_date: Start date for filtering
        end_date: End date for filtering
        generate_features: Whether to generate features even if data has them
        feature_config: Configuration for feature generation
        
    Returns:
        Prepared DataFrame with proper MultiIndex columns
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load data based on file extension
    file_ext = os.path.splitext(data_path)[1].lower()
    try:
        if file_ext == '.parquet':
            data = pd.read_parquet(data_path)
        elif file_ext == '.csv':
            data = pd.read_csv(data_path, parse_dates=True, index_col=0)
        elif file_ext in ['.pkl', '.pickle']:
            with open(data_path, 'rb') as f:
                import pickle
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Apply date filters if provided
    if start_date or end_date:
        logger.info(f"Filtering data from {start_date} to {end_date}")
        if isinstance(data.index, pd.DatetimeIndex):
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask = mask & (data.index >= pd.Timestamp(start_date))
            if end_date:
                mask = mask & (data.index <= pd.Timestamp(end_date))
            data = data.loc[mask]
        else:
            logger.warning("Data index is not DatetimeIndex, cannot filter by date")
    
    # Check if data has proper format
    needs_feature_extraction = False
    
    # If it's not a MultiIndex, we need to create features
    if not isinstance(data.columns, pd.MultiIndex):
        logger.info("Data doesn't have MultiIndex columns, will perform feature extraction")
        needs_feature_extraction = True
    elif generate_features:
        logger.info("Forcing feature generation even though data has MultiIndex columns")
        needs_feature_extraction = True
    else:
        # Check if we have the required technical features
        tech_features = [
            'rsi', 'volatility', 'macd', 'returns_1d', 'returns_5d', 'returns_10d'
        ]
        # Check for at least some of these features
        if not any(feat in data.columns.get_level_values(1) for feat in tech_features):
            logger.info("Data has MultiIndex but lacks technical features, performing feature extraction")
            needs_feature_extraction = True
        
    # Extract or filter assets if provided
    if assets:
        logger.info(f"Filtering for assets: {assets}")
        
        if isinstance(data.columns, pd.MultiIndex):
            # For MultiIndex, we need to filter level 0
            available_assets = data.columns.get_level_values(0).unique()
            valid_assets = [a for a in assets if a in available_assets]
            
            if not valid_assets:
                raise ValueError(f"None of the requested assets {assets} found in data. Available: {available_assets}")
            
            # Create subset with requested assets
            data = data.loc[:, data.columns.get_level_values(0).isin(valid_assets)]
        else:
            # If not MultiIndex, just warn - we'll create proper structure below
            logger.warning(f"Cannot filter non-MultiIndex data for assets {assets} before feature extraction")
    elif isinstance(data.columns, pd.MultiIndex):
        # If assets not specified, use all available in the data
        assets = list(data.columns.get_level_values(0).unique())
        logger.info(f"Using all assets from data: {assets}")
    
    # Perform feature extraction if needed
    if needs_feature_extraction:
        logger.info("Generating technical features from raw data")
        
        # Create feature engine
        engine = DerivativesFeatureEngine()
        
        # Default feature config
        default_config = {
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_advanced": True,
            "window_sizes": [14, 20, 50]
        }
        
        # Use provided config or default
        config = feature_config or default_config
        logger.info(f"Using feature config: {config}")
        
        # Generate features
        data = engine.calculate_features(data, assets=assets, **config)
    
    logger.info(f"Final data shape: {data.shape}")
    return data


def main():
    """Main entry point"""
    args = parse_args()
    
    # Determine data path
    data_path = None
    if args.data_path:
        data_path = args.data_path
    elif args.data_dir:
        data_path = find_latest_data_file(args.data_dir)
    
    # Check that model and data exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        return 1
        
    if data_path and not os.path.exists(data_path):
        # Try Google Drive if enabled
        if args.use_gdrive and os.path.exists(args.drive_ids_file):
            logger.info(f"Data file not found locally, will try Google Drive: {data_path}")
            
            # Initialize Drive adapter
            drive_adapter = DriveAdapter(drive_ids_file=args.drive_ids_file)
            
            # Try to download the file
            if not drive_adapter._download_from_drive(data_path):
                logger.error(f"Data file could not be downloaded from Google Drive: {data_path}")
                return 1
        else:
            logger.error(f"Data file not found: {data_path}")
            return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse feature config if provided
    feature_config = None
    if args.feature_config:
        try:
            feature_config = json.loads(args.feature_config)
        except json.JSONDecodeError:
            logger.error(f"Invalid feature config JSON: {args.feature_config}")
            return 1
    
    # Load and prepare data
    try:
        data = load_and_prepare_data(
            data_path=data_path,
            assets=args.assets,
            start_date=args.start_date,
            end_date=args.end_date,
            generate_features=args.generate_features,
            feature_config=feature_config
        )
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save run configuration
    config = {
        "model_path": args.model_path,
        "data_path": data_path,
        "assets": args.assets if args.assets else list(data.columns.get_level_values(0).unique()),
        "initial_capital": args.initial_capital,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "risk_free_rate": args.risk_free_rate,
        "commission": args.commission,
        "slippage": args.slippage,
        "max_leverage": args.max_leverage,
        "benchmark": args.benchmark,
        "regime_analysis": not args.no_regime_analysis,
        "walk_forward": args.walk_forward,
        "walk_forward_window": args.walk_forward_window,
        "walk_forward_step": args.walk_forward_step,
        "data_shape": data.shape
    }
    
    config_path = os.path.join(args.output_dir, "backtest_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Starting backtest with configuration saved to {config_path}")
    
    try:
        # Run the backtest with data
        results = run_institutional_backtest(
            model_path=args.model_path,
            data_df=data,  # Pass DataFrame directly
            assets=args.assets,
            initial_capital=args.initial_capital,
            output_dir=args.output_dir,
            regime_analysis=not args.no_regime_analysis,
            walk_forward=args.walk_forward
        )
        
        logger.info("Backtest completed successfully!")
        
        # Display key metrics
        metrics = results.get('metrics', {})
        if metrics:
            print("\n=== Backtest Results ===")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print("\nFull results saved to:", args.output_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 