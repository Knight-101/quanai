#!/usr/bin/env python3
"""
Convenience script to run a backtest with fresh data

This script serves as an entry point to run backtests with fresh market data
without having to deal with Python import path issues.
"""

import os
import sys
import asyncio
from backtesting.backtest_with_fetched_data import run_backtest, main

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 