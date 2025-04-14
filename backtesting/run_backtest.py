#!/usr/bin/env python3
"""
Institutional-grade backtesting script for cryptocurrency trading models.

This script provides a command-line interface to run backtests using the
InstitutionalBacktester class. It supports various options for data selection,
analysis modes, and output configuration.
"""

import sys
import os

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from institutional_backtester import InstitutionalBacktester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting/backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run institutional-grade backtests for trading models')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model to backtest')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data file (if not specified, will use data from data directory)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for backtesting (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtesting (format: YYYY-MM-DD)')
    parser.add_argument('--assets', nargs='+', default=None,
                        help='List of assets to backtest (e.g., BTCUSDT ETHUSDT)')
    
    # Backtest configuration
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital for backtesting')
    parser.add_argument('--n-episodes', type=int, default=1,
                        help='Number of backtest episodes to run')
    parser.add_argument('--regime-analysis', action='store_true',
                        help='Perform market regime analysis')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Perform walk-forward validation')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size in days for walk-forward validation')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size in days for walk-forward validation')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results/backtest',
                        help='Directory to save backtest results')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Disable creation of visualizations')
    parser.add_argument('--config-path', type=str, default='config/prod_config.yaml',
                        help='Path to configuration file')
    
    return parser.parse_args()

def main():
    """Main function to run backtest"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log args for debugging
    logger.info(f"Starting backtest with arguments: {args}")
    
    # Create metadata for the backtest
    metadata = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'assets': args.assets,
        'initial_capital': args.initial_capital,
        'n_episodes': args.n_episodes,
        'regime_analysis': args.regime_analysis,
        'walk_forward': args.walk_forward,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata
    with open(os.path.join(args.output_dir, 'backtest_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Initialize backtester
    backtester = InstitutionalBacktester(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        assets=args.assets,
        regime_analysis=args.regime_analysis,
        walk_forward=args.walk_forward,
        config_path=args.config_path
    )
    
    try:
        # Run backtest
        results = backtester.run_backtest(n_eval_episodes=args.n_episodes)
        
        # Create visualizations if enabled
        if not args.no_visualizations:
            backtester.create_visualizations()
        
        # Run additional analyses if requested
        if args.walk_forward:
            backtester.run_walk_forward_validation(
                window_size=args.window_size, 
                step_size=args.step_size
            )
            if not args.no_visualizations:
                backtester._create_walkforward_analysis()
        
        if args.regime_analysis:
            backtester.run_regime_analysis()
            if not args.no_visualizations:
                backtester._create_regime_comparison()
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Number of Trades: {results.get('trade_count', 0)}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Avg Leverage: {results.get('avg_leverage', 0):.2f}x")
        print("="*50)
        print(f"Results saved to: {args.output_dir}")
        print("="*50)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}", exc_info=True)
        print(f"Error during backtesting: {str(e)}")
        return None

if __name__ == "__main__":
    main() 