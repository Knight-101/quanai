#!/usr/bin/env python3
"""
Example script demonstrating how to use the InstitutionalBacktester programmatically.
"""

import os
import json
from institutional_backtester import InstitutionalBacktester
from datetime import datetime

def simple_backtest():
    """Run a simple backtest with default settings"""
    print("\n===== RUNNING SIMPLE BACKTEST =====")
    
    # Create backtest output directory
    output_dir = f"results/backtest_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model path - replace with your actual model path
    model_path = "models/best_model"
    
    # Initialize backtester
    backtester = InstitutionalBacktester(
        model_path=model_path,
        output_dir=output_dir,
        initial_capital=10000.0
    )
    
    # Run the backtest
    results = backtester.run_backtest(n_eval_episodes=1)
    
    # Create visualizations
    backtester.create_visualizations()
    
    # Print summary
    print("\nBACKTEST RESULTS SUMMARY")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Number of Trades: {results.get('trade_count', 0)}")
    print(f"Results saved to: {output_dir}")
    
    return results

def advanced_backtest():
    """Run a more advanced backtest with regime analysis and walk-forward validation"""
    print("\n===== RUNNING ADVANCED BACKTEST =====")
    
    # Create backtest output directory
    output_dir = f"results/advanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model path - replace with your actual model path
    model_path = "models/best_model"
    
    # Define asset list
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Initialize backtester with additional options
    backtester = InstitutionalBacktester(
        model_path=model_path,
        output_dir=output_dir,
        initial_capital=10000.0,
        assets=assets,
        start_date="2022-01-01",
        end_date="2022-12-31",
        regime_analysis=True,
        walk_forward=True
    )
    
    # Run the backtest
    results = backtester.run_backtest()
    
    # Create visualizations
    backtester.create_visualizations()
    
    # Print summary
    print("\nADVANCED BACKTEST RESULTS SUMMARY")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Number of Trades: {results.get('trade_count', 0)}")
    
    # Check if regime analysis was performed
    if backtester.regime_performance:
        print("\nPERFORMANCE BY MARKET REGIME")
        for regime, data in backtester.regime_performance.items():
            print(f"- {regime}: Return={data['metrics'].get('total_return', 0):.2%}, Sharpe={data['metrics'].get('sharpe_ratio', 0):.2f}")
    
    # Check if walk-forward validation was performed
    if backtester.walkforward_results:
        wf = backtester.walkforward_results
        print("\nWALK-FORWARD VALIDATION SUMMARY")
        print(f"Windows: {len(wf['window_results'])}")
        print(f"Avg Return: {wf['aggregate_metrics'].get('total_return', 0):.2%}")
        print(f"Avg Sharpe: {wf['aggregate_metrics'].get('sharpe_ratio', 0):.2f}")
        print(f"Profit Ratio: {wf['robustness_metrics'].get('profit_ratio', 0):.2f}")
    
    print(f"Results saved to: {output_dir}")
    
    return results

def custom_data_backtest(data_path):
    """Run a backtest with custom data"""
    print("\n===== RUNNING BACKTEST WITH CUSTOM DATA =====")
    
    # Create backtest output directory
    output_dir = f"results/custom_data_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model path - replace with your actual model path
    model_path = "models/best_model"
    
    # Initialize backtester with custom data
    backtester = InstitutionalBacktester(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        initial_capital=10000.0
    )
    
    # Run the backtest
    results = backtester.run_backtest()
    
    # Create visualizations
    backtester.create_visualizations()
    
    # Print summary
    print("\nCUSTOM DATA BACKTEST RESULTS SUMMARY")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Number of Trades: {results.get('trade_count', 0)}")
    print(f"Results saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    # Run the simple backtest example
    simple_results = simple_backtest()
    
    # Uncomment to run more advanced examples
    # advanced_results = advanced_backtest()
    # custom_results = custom_data_backtest("data/features/base_features.parquet")
    
    print("\nAll examples completed successfully!") 