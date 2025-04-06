# Institutional Backtesting Module

This module provides institutional-grade backtesting capabilities for cryptocurrency trading models with a focus on eliminating bias, analyzing market regimes, and providing comprehensive metrics.

## Features

- **Robust Backtesting**: Evaluate trading models with proper observation and reward normalization
- **Market Regime Analysis**: Identify and analyze performance across different market regimes
- **Walk-Forward Validation**: Test model robustness across multiple time windows
- **Comprehensive Metrics**: Calculate performance metrics, risk metrics, and trade statistics
- **Visualization Tools**: Create visualizations of performance, drawdowns, returns distribution, and more
- **Bias Reduction**: Techniques to minimize look-ahead bias and data leakage
- **Live Data Fetching**: Option to fetch fresh market data rather than using stored data

## Installation

1. Install the required dependencies:

```bash
pip install -r backtesting/requirements.txt
```

2. Make sure the module is in your Python path or include the parent directory in your path:

```python
import sys
sys.path.append('/path/to/parent/directory')
```

## Quick Start

### Command Line Usage

#### Using Stored Data

The simplest way to run a backtest with stored data:

```bash
python backtesting/run_backtest.py --model-path models/best_model --data-path data/market_data.parquet --output-dir results/backtest
```

#### Using Live Data Fetching

To fetch fresh market data each time you run a backtest:

```bash
python backtesting/backtest_with_fetched_data.py --model-path models/best_model --lookback-days 365 --output-dir results/live_backtest
```

This approach ensures you always test with the most current data available from the source.

#### Advanced Backtest with Data Fetching

For a comprehensive backtest with regime analysis, walk-forward validation, and live data:

```bash
python backtesting/backtest_with_fetched_data.py --model-path models/best_model --symbols BTC/USDT ETH/USDT SOL/USDT --timeframe 1h --start-date 2022-01-01 --initial-capital 100000 --walk-forward --output-dir results/comprehensive_backtest
```

For a complete list of options:

```bash
python backtesting/backtest_with_fetched_data.py --help
```

### Python API Usage

```python
from backtesting.institutional_backtester import run_institutional_backtest

# Run a standard backtest
results = run_institutional_backtest(
    model_path="models/best_model",
    data_path="data/market_data.parquet",
    initial_capital=10000.0,
    output_dir="results/backtest"
)

# Run a more advanced backtest with regime analysis and walk-forward validation
results = run_institutional_backtest(
    model_path="models/best_model",
    data_path="data/market_data.parquet",
    assets=["BTC", "ETH"],
    initial_capital=10000.0,
    start_date="2022-01-01",
    end_date="2022-12-31",
    output_dir="results/backtest_advanced",
    regime_analysis=True,
    walk_forward=True
)
```

## Output

The backtester generates the following outputs in the specified output directory:

- `backtest_metrics.json`: Performance metrics for the full backtest
- `backtest_trades.csv`: Details of all executed trades
- `backtest_portfolio.csv`: Daily portfolio values and returns
- `market_regimes.json`: Identified market regimes and their details
- `regime_comparison.json`: Performance comparison across market regimes
- `walkforward_results.json`: Results from walk-forward validation
- `fetched_data.parquet`: When using data fetching, the raw market data that was retrieved
- Visualization files:
  - `equity_curve.png`: Portfolio equity curve with drawdowns
  - `returns_distribution.png`: Histogram of returns
  - `rolling_metrics.png`: Rolling performance metrics
  - `top_drawdowns.png`: Analysis of major drawdowns
  - `performance_tearsheet.png`: Summary of performance metrics

## Advanced Usage

### Market Regime Analysis

The backtester can identify different market regimes (bull, bear, sideways, high volatility, crisis, etc.) and analyze performance in each regime:

```python
# Run backtest with regime analysis
from backtesting.institutional_backtester import InstitutionalBacktester

backtester = InstitutionalBacktester(
    model_path="models/best_model",
    data_path="data/market_data.parquet",
    output_dir="results/regime_analysis",
    regime_analysis=True
)

# Run main backtest
results = backtester.run_backtest()

# Run regime-specific backtests
regime_results = backtester.run_regime_backtest()
```

### Walk-Forward Validation

To test robustness across time periods:

```python
backtester = InstitutionalBacktester(
    model_path="models/best_model",
    data_path="data/market_data.parquet",
    output_dir="results/walk_forward"
)

# Run walk-forward validation
wf_results = backtester.run_walk_forward_validation(
    window_size=60,  # 60-day windows
    step_size=30     # 30-day steps
)
```

### Data Fetching for Backtesting

You can fetch fresh market data with technical indicators using the data fetcher utilities:

```python
import asyncio
from backtesting.data_fetchers.data_fetcher_backtest import BacktestDataFetcher

async def fetch_data():
    # Initialize the data fetcher
    fetcher = BacktestDataFetcher(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframe="5m",
        lookback_days=365  # 1 year of data
    )

    # Fetch and save data
    data = await fetcher.run("data/fresh_market_data.parquet")
    print(f"Fetched data with shape: {data.shape}")

    return data

# Run the async function
market_data = asyncio.run(fetch_data())
```

Or use the integrated script to fetch data and run the backtest in one go:

```python
import asyncio
from backtesting.backtest_with_fetched_data import run_backtest

async def run_full_backtest():
    results = await run_backtest(
        model_path="models/best_model",
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframe="1h",
        lookback_days=365,
        walk_forward=True,
        output_dir="results/fresh_backtest"
    )
    return results

# Run the async function
backtest_results = asyncio.run(run_full_backtest())
```

#### Technical Indicators

The data fetcher automatically calculates the following technical indicators:

- Moving Averages (SMA and EMA) for various windows
- Bollinger Bands
- MACD
- RSI
- Stochastic Oscillator
- ATR
- ADX
- Volume indicators
- Volatility metrics
- MA crossovers

These match the indicators used during model training to ensure consistent behavior.

## Data Fetching Options

When using the integrated backtest script, you can customize the data retrieval with these options:

- `--symbols`: List of trading symbols to fetch (e.g., "BTC/USDT ETH/USDT SOL/USDT")
- `--timeframe`: Data timeframe interval (default: "5m", options: "1m", "5m", "15m", "1h", "4h", "1d")
- `--start-date`: Starting date for data fetching (YYYY-MM-DD)
- `--end-date`: Ending date for data fetching (YYYY-MM-DD)
- `--lookback-days`: Number of days to look back if start-date is not provided (default: 365)
- `--exchange`: Data source (default: "binance", options: "binance", "coinbase", "bybit", etc.)

## Bias Reduction

The backtester implements several techniques to reduce bias:

1. **Proper Train/Test Separation**: Uses the model as-is without retraining on test data
2. **Observation Normalization**: Preserves normalization statistics from training
3. **Reward Normalization**: Disables reward normalization during evaluation
4. **Walk-Forward Testing**: Evaluates performance across multiple time windows
5. **Regime Analysis**: Analyzes performance across different market conditions
6. **Cold-Start Handling**: Allows for a warm-up period before tracking metrics
7. **Fresh Data**: Option to fetch current market data rather than relying on stored data

## Contributing

Contributions to the backtesting module are welcome! Please submit pull requests or open issues to improve the functionality.
