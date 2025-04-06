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
        logger.error(f"Data file not found: {data_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run configuration
    config = {
        "model_path": args.model_path,
        "data_path": data_path,
        "assets": args.assets,
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
        "walk_forward_step": args.walk_forward_step
    }
    
    config_path = os.path.join(args.output_dir, "backtest_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Starting backtest with configuration saved to {config_path}")
    
    try:
        # Run the backtest
        results = run_institutional_backtest(
            model_path=args.model_path,
            data_path=data_path,
            assets=args.assets,
            initial_capital=args.initial_capital,
            start_date=args.start_date,
            end_date=args.end_date,
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