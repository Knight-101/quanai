# Institutional Backtesting Module

This module provides institutional-grade backtesting capabilities for cryptocurrency trading models with a focus on eliminating bias, analyzing market regimes, and providing comprehensive metrics.

## Features

- **Robust Backtesting**: Evaluate trading models with proper observation and reward normalization
- **Market Regime Analysis**: Identify and analyze performance across different market regimes
- **Walk-Forward Validation**: Test model robustness across multiple time windows
- **Comprehensive Metrics**: Calculate performance metrics, risk metrics, and trade statistics
- **Visualization Tools**: Create visualizations of performance, drawdowns, returns distribution, and more
- **Bias Reduction**: Techniques to minimize look-ahead bias and data leakage

## Installation

1. Install the required dependencies:

```bash
pip install -r backtesting/requirements.txt
```

## Quick Start

### Command Line Usage

The simplest way to run a backtest with stored data:

```bash
python backtesting/run_backtest.py --model-path models/best_model --output-dir results/backtest
```

For a comprehensive backtest with regime analysis and walk-forward validation:

```bash
python backtesting/run_backtest.py --model-path models/best_model --assets BTCUSDT ETHUSDT SOLUSDT --start-date 2022-01-01 --end-date 2022-12-31 --initial-capital 100000 --regime-analysis --walk-forward --output-dir results/comprehensive_backtest
```

For a complete list of options:

```bash
python backtesting/run_backtest.py --help
```

### Python API Usage

```python
from backtesting.institutional_backtester import InstitutionalBacktester

# Run a standard backtest
backtester = InstitutionalBacktester(
    model_path="models/best_model",
    output_dir="results/backtest",
    initial_capital=10000.0
)
results = backtester.run_backtest()
backtester.create_visualizations()

# Run a more advanced backtest with regime analysis and walk-forward validation
advanced_backtester = InstitutionalBacktester(
    model_path="models/best_model",
    output_dir="results/advanced_backtest",
    assets=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    initial_capital=10000.0,
    start_date="2022-01-01",
    end_date="2022-12-31",
    regime_analysis=True,
    walk_forward=True
)
results = advanced_backtester.run_backtest()
advanced_backtester.create_visualizations()
```

For a complete example, see the `example_usage.py` script.

## Output

The backtester generates the following outputs in the specified output directory:

- `backtest_metrics.json`: Performance metrics for the full backtest
- `backtest_trades.csv`: Details of all executed trades
- `backtest_portfolio.csv`: Daily portfolio values and returns
- `market_regimes.json`: Identified market regimes and their details
- `regime_comparison.json`: Performance comparison across market regimes
- `walkforward_results.json`: Results from walk-forward validation
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
backtester = InstitutionalBacktester(
    model_path="models/best_model",
    output_dir="results/regime_analysis",
    regime_analysis=True
)

# Run main backtest
results = backtester.run_backtest()

# Examine regime-specific performance
regime_results = backtester.regime_performance
```

### Walk-Forward Validation

To test robustness across time periods:

```python
backtester = InstitutionalBacktester(
    model_path="models/best_model",
    output_dir="results/walk_forward",
    walk_forward=True
)

# Run backtest with default walk-forward parameters
results = backtester.run_backtest()

# Or customize walk-forward parameters
wf_results = backtester.run_walk_forward_validation(
    window_size=60,  # 60-day windows
    step_size=30     # 30-day steps
)
```

## Bias Reduction

The backtester implements several techniques to reduce bias:

1. **Proper Train/Test Separation**: Uses the model as-is without retraining on test data
2. **Observation Normalization**: Preserves normalization statistics from training
3. **Reward Normalization**: Disables reward normalization during evaluation
4. **Walk-Forward Testing**: Evaluates performance across multiple time windows
5. **Regime Analysis**: Analyzes performance across different market conditions
6. **Cold-Start Handling**: Allows for a warm-up period before tracking metrics

## Metrics Calculation

The backtester calculates a comprehensive set of performance metrics:

- **Total Return**: Overall portfolio return
- **Annual Return**: Annualized return (assuming 252 trading days)
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Return adjusted for downside risk
- **Max Drawdown**: Maximum percentage drop from peak
- **Calmar Ratio**: Annual return divided by max drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Ulcer Index**: Measure of drawdown severity
- **Recovery Factor**: Return divided by max drawdown
- **Average Leverage**: Mean leverage used during trading

## Contributing

Contributions to the backtesting module are welcome! Please submit pull requests or open issues to improve the functionality.
