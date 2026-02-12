#!/usr/bin/env python3
"""
Start Real-time Paper Trading

Convenience entry point to launch the real-time (or replay) paper trading system.
"""

import os
import sys
import argparse
import asyncio

# Work around some DNS resolver issues in async HTTP stacks
os.environ.setdefault("AIOHTTP_DISABLE_AIODNS", "1")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Real-time Paper Trading')

    parser.add_argument(
        '--model',
        type=str,
        default='models/manual/phase6/phase6_model.zip',
        help='Path to the trained RL model'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='models/manual/phase6/vec_normalize.pkl',
        help='Path to the saved normalized environment'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/prod_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=None,
        help='Initial balance for paper trading (overrides config)'
    )

    parser.add_argument(
        '--historical-data',
        type=str,
        default=None,
        help='Path to historical data file (optional)'
    )

    parser.add_argument(
        '--max-leverage',
        type=float,
        default=None,
        help='Maximum allowed leverage (overrides config)'
    )

    parser.add_argument(
        '--websocket-port',
        type=int,
        default=8765,
        help='Websocket server port for UI connections'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='data/trades',
        help='Directory to save trade logs'
    )

    parser.add_argument(
        '--backfill-days',
        type=int,
        default=30,
        help='Number of days of history to use for replay or backfill'
    )

    parser.add_argument(
        '--force-timeframe',
        type=str,
        default=None,
        help='Force a specific timeframe for API calls (e.g., "5m")'
    )

    parser.add_argument(
        '--disable-resampling',
        action='store_true',
        help='Disable automatic resampling of 1m data to match training timeframe'
    )

    parser.add_argument(
        '--disable-reports',
        action='store_true',
        help='Disable detailed daily trading reports'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated list of symbols to trade (e.g., "BTCUSDT,ETHUSDT,SOLUSDT")'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='live',
        choices=['live', 'replay'],
        help='Data source for trading: live (exchange) or replay (local data)'
    )

    parser.add_argument(
        '--replay-speed',
        type=float,
        default=1.0,
        help='Seconds to wait between replay bars'
    )

    parser.add_argument(
        '--replay-start',
        type=str,
        default=None,
        help='Replay start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--replay-end',
        type=str,
        default=None,
        help='Replay end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--model-mode',
        type=str,
        default='sb3',
        choices=['sb3', 'stub'],
        help='Model mode: sb3 (load PPO model) or stub (random actions)'
    )

    parser.add_argument(
        '--stochastic',
        action='store_true',
        default=True,
        help='Use stochastic policy sampling (non-deterministic actions)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Polling interval in seconds for live data (default: 10)'
    )

    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=30,
        help='Rolling metrics log interval in minutes (default: 30)'
    )

    return parser.parse_args()


async def main() -> None:
    """Main function to start the real-time paper trading system."""
    args = parse_arguments()

    if args.model_mode == 'sb3' and not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first or specify a correct model path with --model")
        sys.exit(1)

    os.makedirs(args.save_path, exist_ok=True)

    symbols_override = None
    if args.symbols:
        symbols_override = [sym.strip() for sym in args.symbols.split(',') if sym.strip()]

    print(f"Starting paper trading with model mode: {args.model_mode}")
    print(f"Data source: {args.data_source}")
    print(f"Model path: {args.model}")
    balance_display = f"${args.balance}" if args.balance is not None else "from config"
    leverage_display = f"{args.max_leverage}x" if args.max_leverage is not None else "from config"
    print(f"Initial balance: {balance_display}")
    print(f"Max leverage: {leverage_display}")
    print(f"Websocket server port: {args.websocket_port}")
    print(f"Trade logs directory: {args.save_path}")
    print("Press Ctrl+C to stop trading and close all positions")

    try:
        from realtime_trading import RealTimeTrader
    except ImportError:
        print("Error: Could not import RealTimeTrader. Make sure realtime_trading.py is present.")
        sys.exit(1)

    trader = RealTimeTrader(
        model_path=args.model,
        env_path=args.env,
        config_path=args.config,
        initial_balance=args.balance,
        historical_data_path=args.historical_data,
        max_leverage=args.max_leverage,
        websocket_port=args.websocket_port,
        save_trades_path=args.save_path,
        backfill_days=args.backfill_days,
        force_timeframe=args.force_timeframe,
        resample_data=not args.disable_resampling,
        enable_reports=not args.disable_reports,
        symbols_override=symbols_override,
        data_source=args.data_source,
        replay_speed=args.replay_speed,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        model_mode=args.model_mode,
        poll_interval=args.poll_interval,
        metrics_interval_minutes=args.metrics_interval,
        deterministic_policy=not args.stochastic
    )

    try:
        print("Setting up trading system...")
        await trader.setup()

        print("Starting trading loop...")
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        print("\nTrading interrupted. Closing positions and shutting down...")
    except Exception as exc:
        print(f"Error during trading: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if trader:
            print("Shutting down trading system...")
            await trader.shutdown()
            print("Trading system shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
