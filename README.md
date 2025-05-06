# Quantum AI Trading System

<!-- ./scripts/manual_incremental.sh init 500000 --training-mode --drive-ids-file drive_file_ids.json  -->

A sophisticated AI-powered perpetual futures trading system that utilizes multiple data sources and advanced machine learning techniques to make trading decisions across multiple cryptocurrencies. The system employs a hierarchical reinforcement learning approach with attention mechanisms to process various types of data and generate trading signals.

## Trading Strategies

### 1. Multi-Modal Market Analysis

The system combines multiple data sources to form a comprehensive market view:

#### Technical Analysis

- **Trend Analysis**: Uses transformers to process price action and identify trends across multiple timeframes
- **Momentum Indicators**: RSI, MACD, and Bollinger Bands with adaptive thresholds
- **Volume Analysis**: Volume-weighted metrics for trade confirmation
- **Cross-Asset Correlations**: Analyzes relationships between different cryptocurrencies

#### Sentiment Analysis (Planned)

- **News Impact**: Processes news articles using RoBERTa for sentiment scoring
- **Social Media**: Analyzes Twitter and other social media platforms for market sentiment
- **Market Psychology**: Measures fear/greed through various indicators

#### On-Chain Analysis (Planned)

- **Whale Tracking**: Monitors large wallet movements
- **Network Health**: Analyzes transaction volumes, active addresses, and network growth
- **Smart Contract Activity**: Monitors DeFi and other protocol usage

### 2. Risk Management

- **Dynamic Position Sizing**: Adjusts position sizes based on:
  - Market volatility
  - Account equity
  - Cross-asset correlation risk
  - Current market regime
- **Stop Loss Strategies**:
  - Volatility-adjusted stops
  - Time-based stops
  - Profit protection mechanisms

### 3. Execution Strategy

- **Smart Order Routing**:
  - Minimizes market impact
  - Considers liquidity pools
  - Adapts to market microstructure
- **Entry/Exit Optimization**:
  - Uses TWAP/VWAP for large orders
  - Implements iceberg orders when needed
  - Considers funding rates for perpetual futures

## System Architecture

### 1. Data Collection (`data_collection/`)

#### MultiModalDataCollector (`collect_multimodal.py`)

- **Purpose**: Collects and processes multiple types of data sources for trading
- **Data Types**:
  - Price/Market Data (OHLCV)
  - Technical Indicators
  - News Sentiment (currently commented out)
  - Social Media Sentiment (currently commented out)
  - On-chain Metrics (currently commented out)
- **Key Features**:
  - Rate limiting for API calls
  - Robust error handling
  - Data normalization and preprocessing
  - [Technical Analysis Library (ta)](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) for indicators

### 2. Trading Environment (`trading_env/`)

#### InstitutionalPerpetualEnv (`institutional_perp_env.py`)

- **Purpose**: Custom OpenAI Gym environment for perpetual futures trading
- **Features**:
  - Multi-asset support
  - Realistic trading mechanics:
    - Funding rates
    - Transaction fees
    - Price impact modeling
    - Spread costs
  - Risk management integration
  - Position tracking
  - **Learn More**:
    - [What are Perpetual Futures? (Binance Academy)](https://academy.binance.com/en/articles/what-are-perpetual-futures-contracts)
    - [Perpetual Futures Trading Guide (YouTube)](https://www.youtube.com/watch?v=_uXHGf_LWyM)
    - [Understanding Funding Rates (YouTube)](https://www.youtube.com/watch?v=Yl7DZQJ_v2s)

### 3. Training System (`training/`)

#### HierarchicalPPO (`hierarchical_ppo.py`)

- **Purpose**: Implementation of Hierarchical Proximal Policy Optimization for training the trading agent
- **Architecture**:
  - Market Transformer: Processes market data using attention mechanisms
  - Text Encoder: Processes news and social media (using RoBERTa)
  - Risk LSTM: Processes risk metrics
  - Cross-asset attention mechanism
  - Feature fusion layer
- **Components**:
  - Custom Actor-Critic Policy
  - Multiple action heads for different aspects of trading
  - GAE (Generalized Advantage Estimation)
- **Learn More**:
  - [Reinforcement Learning in Trading (YouTube)](https://www.youtube.com/watch?v=9P-p8li4boE)
  - [Introduction to PPO (YouTube)](https://www.youtube.com/watch?v=5P7I-xPq8u8)
  - [Transformers Explained (YouTube)](https://www.youtube.com/watch?v=4Bdc55j80l8)
  - [Actor-Critic Methods Explained](https://huggingface.co/blog/deep-rl-a2c)

### 4. Risk Management System (`risk_management/`)

#### InstitutionalRiskEngine (`risk_engine.py`)

- **Purpose**: Comprehensive risk management and monitoring system
- **Features**:
  - Position Risk Management:
    - Value at Risk (VaR) calculations
    - Expected Shortfall (ES) metrics
    - Stress testing scenarios
  - Portfolio Risk Controls:
    - Dynamic position sizing
    - Correlation-based exposure limits
    - Drawdown management
  - Market Risk Monitoring:
    - Volatility regime detection
    - Liquidity risk assessment
    - Counterparty risk tracking
  - Risk Reporting:
    - Real-time risk metrics
    - Historical risk analysis
    - Risk limit breach alerts
- **Learn More**:
  - Risk Management Fundamentals:
    - [Introduction to Risk Management (YouTube)](https://www.youtube.com/watch?v=Ql8KUUUOHNc)
    - [Value at Risk (VaR) Explained Simply (YouTube)](https://www.youtube.com/watch?v=92WaNz9mPeY)
    - [Position Sizing and Risk Management Guide](https://www.tradingwithrayner.com/position-sizing/)
  - Advanced Topics:
    - [Portfolio Risk Management (Investopedia)](https://www.investopedia.com/terms/p/portfolio-management.asp)
    - [Understanding Expected Shortfall](https://www.investopedia.com/terms/e/expected-shortfall.asp)
    - [Stress Testing Explained](https://www.investopedia.com/terms/s/stresstesting.asp)

## Key Technologies and Libraries

### Machine Learning

- PyTorch: Deep learning framework
- Transformers: For NLP tasks
- Stable-Baselines3: RL algorithms base
- Gymnasium: Environment framework

### Data Processing

- Pandas: Data manipulation
- NumPy: Numerical computations
- ta: Technical analysis

### APIs and Services

- CCXT: Cryptocurrency exchange interface
- News API (commented)
- Twitter API (commented)
- On-chain data APIs (commented)

## Trading Strategy Components

### 1. Feature Processing

- **Market Data**:
  ```python
  class MarketTransformer(nn.Module):
      # Processes market data using transformer architecture
      # Captures temporal dependencies and cross-asset relationships
  ```

### 2. Risk Management

- **Risk Metrics**:
  ```python
  class RiskLSTM(nn.Module):
      # Processes risk-related features
      # Uses LSTM with attention for temporal risk patterns
  ```

### 3. Decision Making

- **Actor-Critic Architecture**:
  ```python
  class CustomActorCriticPolicy(nn.Module):
      # Multiple action heads for:
      # - Trade decisions
      # - Position sizing
      # - Risk limits
      # - Execution parameters
  ```

## Configuration and Hyperparameters

### Environment Parameters

- Initial Balance: 1M USDC
- Max Leverage: 20x
- Transaction Fee: 0.04%
- Funding Fee Multiplier: 0.8
- Risk-free Rate: 3%
- Max Drawdown: 30%

### Training Parameters

- Learning Rate: 3e-4
- Batch Size: 64
- Training Steps: 2048
- Epochs: 10
- Gamma (discount factor): 0.99
- GAE Lambda: 0.95
- Clip Range: 0.2

## Usage

### 1. Data Collection

```python
collector = MultiModalDataCollector()
collector.collect_and_save_data(
    start_date=start_date,
    end_date=end_date,
    output_path='data/train_data.parquet'
)
```

### 2. Training

```python
env = InstitutionalPerpetualEnv(
    df=data,
    initial_balance=1e6,
    max_leverage=20
)

model = HierarchicalPPO(
    env=env,
    learning_rate=3e-4,
    n_steps=2048
)

model.learn(total_timesteps=100_000)
```

## Future Enhancements

1. Integration of news and social media sentiment analysis
2. On-chain data integration
3. Enhanced risk management features
4. Multi-exchange support
5. Advanced order execution strategies

## Additional Resources

### Financial Concepts

- [Complete Guide to Crypto Trading (YouTube)](https://www.youtube.com/watch?v=1prz3utkGAY)
- [Understanding Market Making (YouTube)](https://www.youtube.com/watch?v=Kl4-VJ2K8Ik)
- [Crypto Market Structure (YouTube)](https://www.youtube.com/watch?v=QC1h1K6YwBs)

### Machine Learning in Trading

- [Introduction to ML in Trading (YouTube)](https://www.youtube.com/watch?v=OhUkMY_DHPg)
- [Deep Learning in Trading (YouTube)](https://www.youtube.com/watch?v=u6bOLx4xB7E)
- [Neural Networks for Trading (Article)](https://blog.quantinsti.com/neural-network-trading/)
- [Transformers in Financial Markets](https://blog.quantinsti.com/transformers-in-finance/)

### Risk Management

- [Complete Guide to Risk Management (YouTube)](https://www.youtube.com/watch?v=5M8kXVwgkAg)
- [Position Sizing Calculator & Strategies](https://www.babypips.com/learn/forex/position-sizing)
- [Risk Management Strategies (Article)](https://www.babypips.com/learn/forex/how-much-money-can-i-make-from-forex-trading)

### Algorithmic Trading

- [Building a Trading Bot (YouTube Series)](https://www.youtube.com/watch?v=xfzGZB4HhEE)
- [Algorithmic Trading Strategies](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)
- [Backtesting Best Practices](https://blog.quantinsti.com/backtesting/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/quantum-trading.git
cd quantum-trading
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

## Institutional Backtesting Module

A new institutional-grade backtesting module has been added to the repository. This module provides comprehensive backtesting capabilities with a focus on eliminating bias, analyzing market regimes, and providing detailed performance metrics.

### Features

- **Robust Backtesting**: Evaluate trading models with proper observation and reward normalization
- **Market Regime Analysis**: Identify and analyze performance across different market regimes
- **Walk-Forward Validation**: Test model robustness across multiple time windows
- **Comprehensive Metrics**: Calculate performance metrics, risk metrics, and trade statistics
- **Visualization Tools**: Create visualizations of performance, drawdowns, returns distribution, and more
- **Bias Reduction**: Techniques to minimize look-ahead bias and data leakage

### Usage

The simplest way to run a backtest is using the command-line interface:

```bash
# Basic backtest
python backtesting/run_backtest.py --model-path models/best_model --data-path data/market_data.parquet --output-dir results/backtest

# Comprehensive backtest with regime analysis and walk-forward validation
python backtesting/run_backtest.py --model-path models/best_model --data-path data/market_data.parquet --output-dir results/full_backtest --walk-forward

# Advanced parameters
python backtesting/run_backtest.py --model-path models/best_model --data-path data/market_data.parquet \
  --initial-capital 100000 --risk-free-rate 0.02 --commission 0.0004 --slippage 0.0002 \
  --max-leverage 10.0 --benchmark "BTC-USD" --output-dir results/custom_backtest
```

For more details and advanced usage, see the [Backtesting README](backtesting/README.md).

## Extended Training (10M Steps)

The system now supports extended training for up to 10 million steps with an intelligent phase-based approach:

- **9-Phase Training Structure**: Divided into Foundation Phase (0-1M steps) and Mastery Phase (1M-10M steps)
- **Automatic Phase Transitions**: The system can automatically move between training phases with appropriate hyperparameter adjustments
- **Performance-Adaptive Recommendations**: Hyperparameters are recommended based on model performance in the previous phase

### Using Extended Training

1. **Start normal training for Phase 1**:

   ```bash
   python main_opt.py --train --model_dir models/manual/phase1
   ```

2. **Continue to next phase with automatic recommendations**:

   ```bash
   python main_opt.py --continue_training --model_path models/manual/phase1/final_model --model_dir models/manual/phase1 --use_recommendations
   ```

3. **Check progress and recommendations**:
   ```bash
   cat models/manual/phase1/phase2_recommendations.json
   ```

For complete details on the extended training approach, see [training_schedule.md](training_schedule.md).

## Trading LLM Chatbot

The system includes an interactive chatbot for discussing market conditions, trading decisions, and technical analysis.

### Using the Chatbot

You can interact with the trading chatbot via a simple command-line interface:

```bash
python -m trading_llm.chatbot --model_path /path/to/your/model --base_model meta-llama/Meta-Llama-3-8B-Instruct --market_data /path/to/market_data.csv
```

Arguments:

- `--model_path`: Path to the trained LLM model or LoRA adapter (required)
- `--base_model`: Base model path (required when using LoRA adapters)
- `--max_history`: Maximum number of message pairs to keep in conversation (default: 5)
- `--device`: Device to run the model on ('cpu', 'cuda', 'auto')
- `--market_data`: Optional path to CSV or Parquet market data file
- `--trading_signals`: Optional path to JSON file with trading signals

### Programmatic Usage

You can also use the chatbot programmatically in your code:

```python
from trading_llm.chatbot import load_market_chatbot
import pandas as pd

# Initialize the chatbot
chatbot = load_market_chatbot(
    model_path="/path/to/your/model",
    base_model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_history=5
)

# Optionally provide market data
market_data = pd.read_csv("market_data.csv")
chatbot.update_market_data(market_data)

# Chat with the bot
response = chatbot.chat("What do you think about the current market conditions?")
print(response)

# Reset conversation if needed
chatbot.reset_conversation()
```

The chatbot can provide answers about:

- Current market conditions
- Technical analysis explanations
- Trading strategy insights
- Recent trading signals
- Historical performance

# RL Trading Model Improvement Guide

This repository contains tools and scripts to improve your reinforcement learning trading model by addressing bias and common errors.

## Overview of Issues

1. **Model Bias**: Your model has developed biased signals for specific assets (BTC and ETH short, SOL long)
2. **Observation Space Errors**: Various tools (analyze_bias, realtime_trading) encounter observation space errors
3. **Training Configuration**: Need to continue training with optimal parameters

## Steps to Improve Your Model

### 1. Fix Environment Observation Space Issues

First, fix any observation space errors in your saved environment:

```bash
python fix_observation_space.py --env-path ./models/manual/phase6/final_env.pkl --create-backup
```

This will:

- Create a backup of your environment
- Fix the observation space dimensions
- Save the fixed environment back to the original file

### 2. Analyze Current Model Bias

Use the analyze_model_bias.py script to identify existing biases:

```bash
python analyze_model_bias.py --model-path ./models/manual/phase6/final_model.zip --env-path ./models/manual/phase6/final_env.pkl --num-samples 1000 --correction-method nonlinear --output-dir ./analysis
```

Review the generated charts and CSV files in the `./analysis` directory to understand:

- The extent of bias for each asset
- The distribution of actions
- Statistics about extreme values

### 3. Continue Training with Bias Mitigation

Now continue training your model with data augmentation to reduce bias:

```bash
python continue_training.py --model-path ./models/manual/phase6/final_model.zip --env-path ./models/manual/phase6/final_env.pkl --timesteps 5000000 --learning-rate 1e-4 --batch-size 128 --ent-coef 0.01 --output-dir ./improved_model
```

This script:

- Applies data augmentation to reduce asset bias
- Recalibrates the value function
- Continues training for 5 million steps with improved parameters

## Comparison with main_opt.py

The `continue_training.py` script is focused specifically on continuing training while reducing bias. It differs from `main_opt.py` in these key ways:

1. **Data Augmentation**: `continue_training.py` includes specific techniques to reduce asset bias:

   - Segment shuffling to break perfect time continuity
   - Feature noise to prevent overfitting
   - Price scaling to reduce directional bias

2. **Model Recalibration**: Instead of starting from scratch, it recalibrates the value function but keeps the policy network intact.

3. **Parameter Focus**: It uses optimal parameters specifically for continuing training on existing models.

You can directly use `main_opt.py` if you prefer, but you will need to manually implement the bias reduction techniques.

## Adding or Replacing Assets

If you want to add or replace assets (e.g., replacing SOL with SUI), you have two options:

1. **Retraining from Scratch**: For best results when changing the asset set significantly.

2. **Transfer Learning**: Continue from existing model with new assets:
   - The model will need time to adapt to the new asset data patterns
   - Performance may initially be suboptimal for new assets
   - Old biases may transfer to new assets with similar patterns

For limited changes (e.g., replacing 1 out of 3 assets), transfer learning is usually sufficient. For more substantial changes, retraining from scratch is recommended.

## Troubleshooting Observation Space Errors

The root cause of observation space errors is typically a mismatch between:

1. The observation space defined in the environment (`env.observation_space`)
2. The actual observation vectors returned by the environment (`env._get_observation()`)

This happens because:

- The environment gets pickled and the observation space is fixed at that point
- When unpickled and used again, conditions may have changed (e.g., different number of assets)
- InstitutionalPerpetualEnv dynamically calculates features which might affect the shape

The provided scripts automatically fix these issues by:

1. Unwrapping the environment to access the base env
2. Checking for dimension mismatches
3. Updating the observation space to match what `_get_observation()` actually returns

## Best Practices for Future Training

1. **Regular Data Augmentation**: Implement data augmentation techniques during training
2. **Balanced Asset Sampling**: Ensure balanced representation of all assets
3. **Exploration Emphasis**: Use higher entropy coefficients (0.01-0.05) to reduce bias formation
4. **Regular Checkpoints**: Save checkpoints to prevent catastrophic learning degradation
5. **Periodic Bias Testing**: Use `analyze_model_bias.py` to check for developing biases

By following these practices, you can develop a more robust and unbiased trading model.
