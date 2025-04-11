#!/usr/bin/env python3
"""
Run Complete Backtest

This script provides a simple interface to run a complete backtest
with the same data loading/feature extraction logic as used in training,
ensuring compatibility with trained models.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestRunner")

# Add the current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our data fetcher and backtester
from data_fetcher_backtest import CustomBacktestDataFetcher
from backtesting.institutional_backtester import run_institutional_backtest

async def run_complete_backtest(
    model_path: str,
    symbols: list = ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframe: str = "5m",
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 365,
    initial_capital: float = 100000.0,
    output_dir: str = "results/backtest",
    drive_ids_file: str = "drive_file_ids.json",
    regime_analysis: bool = True,
    walk_forward: bool = False
):
    """
    Run a complete backtest with the same data loading/feature logic as training.
    
    Args:
        model_path: Path to the trained model
        symbols: List of trading symbols
        timeframe: Data timeframe (e.g., '5m', '1h')
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        lookback_days: Number of days to look back if start_date not provided
        initial_capital: Initial capital for backtesting
        output_dir: Directory for output files
        drive_ids_file: Path to Google Drive file IDs JSON
        regime_analysis: Whether to perform market regime analysis
        walk_forward: Whether to perform walk-forward validation
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Fetch and process data
        logger.info(f"Fetching data for backtest: {symbols}")
        data_fetcher = CustomBacktestDataFetcher(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            drive_ids_file=drive_ids_file
        )
        
        # Save data to output directory
        data_path = os.path.join(output_dir, "backtest_data.parquet")
        market_data = await data_fetcher.run(data_path)
        
        if market_data is None or market_data.empty:
            logger.error("Failed to fetch or process market data")
            return None
            
        logger.info(f"Data processed successfully with shape: {market_data.shape}")
        
        # Step 2: Run backtest
        logger.info(f"Running backtest with model: {model_path}")
        
        # Extract asset names for backtester
        assets = [data_fetcher.symbol_mappings.get(symbol, symbol) for symbol in symbols]
        
        results = run_institutional_backtest(
            model_path=model_path,
            data_df=market_data,  # Pass DataFrame directly
            assets=assets,
            initial_capital=initial_capital,
            output_dir=output_dir,
            regime_analysis=regime_analysis,
            walk_forward=walk_forward
        )
        
        # Display results
        if results and 'metrics' in results:
            metrics = results['metrics']
            
            print("\n====== BACKTEST RESULTS SUMMARY ======")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print("=======================================")
            
            # Show results path
            print(f"\nDetailed backtest results saved to: {output_dir}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest process: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run complete backtest with training-compatible data processing")
    
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
                     
    parser.add_argument("--drive-ids-file", type=str, default="drive_file_ids.json",
                     help="Path to Google Drive file IDs JSON (default: drive_file_ids.json)")
                     
    parser.add_argument("--regime-analysis", action="store_true", default=True,
                     help="Perform market regime analysis (default: True)")
                     
    parser.add_argument("--walk-forward", action="store_true",
                     help="Perform walk-forward validation")
                     
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    await run_complete_backtest(
        model_path=args.model_path,
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
        drive_ids_file=args.drive_ids_file,
        regime_analysis=args.regime_analysis,
        walk_forward=args.walk_forward
    )

if __name__ == "__main__":
    asyncio.run(main()) 