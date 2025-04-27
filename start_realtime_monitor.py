#!/usr/bin/env python3
"""
Start Real-time Trading Monitor

This script starts the real-time trading monitor with the proper configuration
for trading BTC, ETH, and SOL on 5-minute timeframes.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Real-time Trading Monitor')
    
    parser.add_argument('--rl-model', type=str, 
                       default='models/rl_model',
                       help='Path to the trained RL model')
    
    parser.add_argument('--llm-model', type=str,
                       default='models/llm_model',
                       help='Path to the LLM model for commentary')
    
    parser.add_argument('--llm-base-model', type=str,
                       default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Base model for LLM if using LoRA')
    
    parser.add_argument('--config', type=str,
                       default='config/realtime_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--log-dir', type=str,
                       default='logs',
                       help='Directory for logs')
    
    parser.add_argument('--timeframe', type=str,
                       default='5m',
                       help='Timeframe for price data')
    
    parser.add_argument('--symbols', type=str,
                       default='BTCUSDT,ETHUSDT,SOLUSDT',
                       help='Comma-separated list of symbols to trade')
    
    parser.add_argument('--balance', type=float,
                       default=10000.0,
                       help='Initial account balance')
    
    parser.add_argument('--max-leverage', type=float,
                       default=20.0,
                       help='Maximum allowed leverage')
    
    return parser.parse_args()

async def main():
    """Main function to start the real-time trading monitor."""
    args = parse_arguments()
    
    # Check if RL model exists
    if not os.path.exists(args.rl_model):
        print(f"Error: RL model not found at {args.rl_model}")
        print("Please train a model first or specify a correct model path with --rl-model")
        sys.exit(1)
    
    # Check if LLM model exists
    if not os.path.exists(args.llm_model):
        print(f"Error: LLM model not found at {args.llm_model}")
        print("Please provide a valid LLM model path with --llm-model")
        sys.exit(1)
    
    # Create logs directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print startup information
    print(f"Starting real-time trading monitor (Run ID: {run_id})")
    print(f"Trading symbols: {args.symbols}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Initial balance: ${args.balance}")
    print(f"Max leverage: {args.max_leverage}x")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Using RL model: {args.rl_model}")
    print(f"Using LLM model: {args.llm_model}")
    print("Press Ctrl+C to stop trading and close all positions")
    
    # Import and run the monitor
    try:
        # We need to import here to make sure all initialization is done correctly
        from realtime_trading_monitor import RealTimeTradeMonitor
        
        # Initialize monitor
        monitor = RealTimeTradeMonitor(
            rl_model_path=args.rl_model,
            llm_model_path=args.llm_model,
            llm_base_model=args.llm_base_model,
            config_path=args.config,
            log_dir=f"{args.log_dir}/{run_id}",
            timeframe=args.timeframe,
            symbols=args.symbols.split(','),
            initial_balance=args.balance,
            max_leverage=args.max_leverage
        )
        
        # Setup monitor
        print("Setting up trading monitor...")
        await monitor.setup()
        
        # Run trading loop
        print("Starting trading loop. Monitoring market data and executing trades...")
        await monitor.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\nTrading interrupted. Closing positions and shutting down...")
    except Exception as e:
        print(f"Error starting trading monitor: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper shutdown if monitor was initialized
        if 'monitor' in locals():
            print("Shutting down trading monitor...")
            await monitor.shutdown()
            print("Trading monitor shutdown complete.")
        
if __name__ == "__main__":
    asyncio.run(main()) 