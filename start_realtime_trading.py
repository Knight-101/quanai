#!/usr/bin/env python3
"""
Start Real-time Paper Trading

A simple script to start the real-time paper trading system with recommended settings.
This script serves as a convenient entry point to launch the trading module.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Real-time Paper Trading')
    
    parser.add_argument('--model', type=str, 
                       default='models/best_model',
                       help='Path to the trained RL model')
    
    parser.add_argument('--env', type=str,
                       default='models/best_model_env.pkl',
                       help='Path to the saved normalized environment')
    
    parser.add_argument('--config', type=str,
                       default='config/prod_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--balance', type=float,
                       default=10000.0,
                       help='Initial balance for paper trading')
    
    parser.add_argument('--historical-data', type=str,
                       default=None,
                       help='Path to historical data file (optional)')
    
    parser.add_argument('--max-leverage', type=float,
                       default=10.0,
                       help='Maximum allowed leverage (default: 10.0)')
    
    parser.add_argument('--websocket-port', type=int,
                       default=8765,
                       help='Websocket server port for UI connections')
    
    parser.add_argument('--save-path', type=str,
                       default='data/trades',
                       help='Directory to save trade logs')
    
    parser.add_argument('--backfill-days', type=int,
                       default=5,
                       help='Number of days to backfill data if no historical data provided')
    
    return parser.parse_args()

async def main():
    """Main function to start the real-time paper trading system."""
    args = parse_arguments()
    
    # Import realtime_trading module
    try:
        from realtime_trading import RealTimeTrader
    except ImportError:
        print("Error: Could not import RealTimeTrader. Make sure realtime_trading.py is in the same directory.")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first or specify a correct model path with --model")
        sys.exit(1)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Starting real-time paper trading with model: {args.model}")
    print(f"Initial balance: ${args.balance}")
    print(f"Max leverage: {args.max_leverage}x")
    print(f"Websocket server will be available on port: {args.websocket_port}")
    print(f"Trade logs will be saved to: {args.save_path}")
    print("Press Ctrl+C to stop trading and close all positions")
    
    # Initialize trader
    trader = RealTimeTrader(
        model_path=args.model,
        env_path=args.env,
        config_path=args.config,
        initial_balance=args.balance,
        historical_data_path=args.historical_data,
        max_leverage=args.max_leverage,
        websocket_port=args.websocket_port,
        save_trades_path=args.save_path,
        backfill_days=args.backfill_days
    )
    
    try:
        # Set up trader
        print("Setting up trading system...")
        await trader.setup()
        
        # Run trading loop
        print("Starting trading loop. Monitoring market data and executing trades...")
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        print("\nTrading interrupted. Closing positions and shutting down...")
    except Exception as e:
        print(f"Error during trading: {str(e)}")
    finally:
        # Ensure proper shutdown
        if trader:
            print("Shutting down trading system...")
            await trader.shutdown()
            print("Trading system shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main()) 