#!/usr/bin/env python3
"""
Backtest With Fetched Data

This script demonstrates how to use the data fetcher to get fresh market data
and then run a backtest using the institutional backtester.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestRunner")

# Import our custom data fetcher
from data_fetcher_backtest import CustomBacktestDataFetcher

# Import the institutional backtester
from backtesting.institutional_backtester import run_institutional_backtest

async def run_backtest(
    model_path: str,
    symbols: list = ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframe: str = "5m",
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 365, # Default to 1 year
    initial_capital: float = 100000.0,
    output_dir: str = "results/backtest",
    exchange: str = "binance",
    regime_analysis: bool = True,
    walk_forward: bool = False,
    use_existing_features: bool = True,
    drive_ids_file: str = "drive_file_ids.json"
):
    """
    Run a complete backtest with freshly fetched data or use existing data following
    the same logic as in the training module.
    
    Args:
        model_path: Path to the trained model
        symbols: List of trading symbols to fetch
        timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        lookback_days: Number of days to look back if start_date is not provided
        initial_capital: Initial capital for backtesting
        output_dir: Directory for output files
        exchange: Trading exchange to fetch from
        regime_analysis: Whether to perform market regime analysis
        walk_forward: Whether to perform walk-forward validation
        use_existing_features: Whether to use existing feature data files
        drive_ids_file: Path to the Google Drive file IDs JSON
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Set up the data fetcher with compatibility for both local and Google Drive data
        logger.info(f"Setting up data fetcher for {symbols} from {exchange}...")
        
        data_fetcher = CustomBacktestDataFetcher(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            exchange=exchange,
            data_dir="data",  # Use the same data directory as training
            drive_ids_file=drive_ids_file
        )
        
        # Generate a descriptive filename for the data
        data_filename = f"market_data_for_backtest.parquet"
        data_path = os.path.join(output_dir, data_filename)
        
        # Fetch and save the data
        market_data = await data_fetcher.run(data_path)
        
        if market_data is None or market_data.empty:
            logger.error("Failed to fetch market data")
            return
            
        logger.info(f"Market data processed and saved to {data_path}")
        
        # Step 2: Run the backtest
        logger.info(f"Running institutional backtest with model {model_path}...")
        
        # Extract asset names from the symbols
        assets = [data_fetcher.symbol_mappings.get(symbol, symbol) for symbol in symbols]
        
        # Run the institutional backtest
        backtest_results = run_institutional_backtest(
            model_path=model_path,
            data_path=data_path,  # Use the fetched data
            assets=assets,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            regime_analysis=regime_analysis,
            walk_forward=walk_forward
        )
        
        # Step 3: Display results summary
        logger.info("Backtest completed!")
        
        if backtest_results and 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            
            print("\n====== BACKTEST RESULTS SUMMARY ======")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print("=======================================")
            
            print(f"\nDetailed results saved to: {output_dir}")
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Parse command line arguments and run the backtest"""
    parser = argparse.ArgumentParser(description="Run a complete backtest with data fetching that matches training logic")
    
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to the trained model")
    
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                      help="Symbols to fetch (default: BTC/USDT ETH/USDT SOL/USDT)")
                      
    parser.add_argument("--timeframe", type=str, default="5m",
                     help="Timeframe (default: 5m)")
                     
    parser.add_argument("--start-date", type=str,
                     help="Start date (YYYY-MM-DD)")
                     
    parser.add_argument("--end-date", type=str,
                     help="End date (YYYY-MM-DD)")
                     
    parser.add_argument("--lookback-days", type=int, default=365,
                     help="Number of days to look back (default: 365)")
                     
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                     help="Initial capital for backtesting (default: 100000.0)")
                     
    parser.add_argument("--output-dir", type=str, default="results/backtest",
                     help="Directory for output files (default: results/backtest)")
                     
    parser.add_argument("--exchange", type=str, default="binance",
                     help="Exchange to fetch from (default: binance)")
                     
    parser.add_argument("--regime-analysis", action="store_true", default=True,
                     help="Perform market regime analysis (default: True)")
                     
    parser.add_argument("--walk-forward", action="store_true",
                     help="Perform walk-forward validation")
                     
    parser.add_argument("--use-existing-features", action="store_true", default=True,
                     help="Use existing feature data files if available (default: True)")
                     
    parser.add_argument("--drive-ids-file", type=str, default="drive_file_ids.json",
                     help="Path to Google Drive file IDs JSON (default: drive_file_ids.json)")
    
    args = parser.parse_args()
    
    # Run the backtest
    await run_backtest(
        model_path=args.model_path,
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
        exchange=args.exchange,
        regime_analysis=args.regime_analysis,
        walk_forward=args.walk_forward,
        use_existing_features=args.use_existing_features,
        drive_ids_file=args.drive_ids_file
    )

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 