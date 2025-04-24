# Trading LLM Command Cheatsheet

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/trading_llm.git
cd trading_llm

# Install dependencies
pip install -e .
```

## Dataset Generation

```bash
# Basic dataset generation
python -m trading_llm.train_llm generate \
  --rl-model /path/to/rl_model.zip \
  --market-data /path/to/market_data.parquet \
  --output-dir ./data/trading_dataset \
  --num-samples 1000

# With custom train/test split
python -m trading_llm.train_llm generate \
  --rl-model /path/to/rl_model.zip \
  --market-data /path/to/market_data.parquet \
  --output-dir ./data/trading_dataset \
  --num-samples 1000 \
  --split-ratio 0.9
```

## Model Training

```bash
# Basic training
python -m trading_llm.train_llm train \
  --base-model meta-llama/Meta-Llama-3-8B-Instructt \
  --train-data ./data/trading_dataset/train_data.json \
  --output-dir ./models/trading_llm

# Full training with all parameters
python -m trading_llm.train_llm train \
  --base-model meta-llama/Meta-Llama-3-8B-Instructt \
  --train-data ./data/trading_dataset/train_data.json \
  --eval-data ./data/trading_dataset/eval_data.json \
  --output-dir ./models/trading_llm \
  --num-epochs 5 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05
```

## Inference

```bash
# Generate trading explanations
python -m trading_llm.train_llm infer \
  --rl-model /path/to/rl_model.zip \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet

# Save explanations to file
python -m trading_llm.train_llm infer \
  --rl-model /path/to/rl_model.zip \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet \
  --output-file ./explanations.txt
```

## Market Commentary

```bash
# Generate market commentary
python -m trading_llm.train_llm commentary \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet

# Save commentary to file
python -m trading_llm.train_llm commentary \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet \
  --output-file ./commentary.txt
```

## Interactive Chatbot

```bash
# Basic chatbot with just LLM
python -m trading_llm.train_llm chat \
  --llm-model ./models/trading_llm

# Full chatbot with market data and RL model
python -m trading_llm.train_llm chat \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet \
  --rl-model /path/to/rl_model.zip

# With custom history length and system prompt
python -m trading_llm.train_llm chat \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet \
  --rl-model /path/to/rl_model.zip \
  --max-history 10 \
  --system-prompt "You are an expert financial advisor specialized in crypto markets."
```

## Programmatic Usage

```python
# Load and use the TradingLLM model
from trading_llm.model import TradingLLM

model = TradingLLM.load("./models/trading_llm")
explanation = model.generate_text(
    "Explain why the market might be trending upward today.",
    max_new_tokens=512,
    temperature=0.7
)

# Use the Market Chatbot
from trading_llm.chatbot import load_market_chatbot
import pandas as pd

# Load market data
market_data = pd.read_parquet("./data/recent_market.parquet")

# Initialize chatbot
chatbot = load_market_chatbot(
    model_path="./models/trading_llm",
    max_history=5
)

# Update market context
chatbot.update_market_data(market_data)

# Chat interaction
response = chatbot.chat("What do you think about the current market conditions?")
print(response)
```
