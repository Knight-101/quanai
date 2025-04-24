# Trading LLM

A Python module for generating natural language explanations for reinforcement learning-based trading decisions.

## Overview

Trading LLM combines reinforcement learning (RL) models for trading with Large Language Models (LLMs) to provide:

1. Natural language explanations for trading decisions
2. Technical analysis summaries from market data
3. Market trend insights based on price action

The module is designed to work alongside RL trading models to enhance interpretability and provide human-readable insights.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-llm.git
cd trading-llm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PEFT >= 0.5.0
- Stable-Baselines3 >= 2.0.0
- TA (Technical Analysis) >= 0.10.0
- Other dependencies in requirements.txt

## Quickstart

### 1. Generate Training Dataset

First, generate a dataset from your RL model for training the LLM:

```bash
python -m trading_llm.train_llm generate \
    --rl-model /path/to/your/rl_model.zip \
    --market-data /path/to/market_data.parquet \
    --output-dir ./data/trading_llm_dataset \
    --num-samples 1000
```

### 2. Train the LLM

Fine-tune a language model to explain trading decisions:

```bash
python -m trading_llm.train_llm train \
    --base-model meta-llama/Meta-Llama-3-8B-Instructt \
    --train-data ./data/trading_llm_dataset/train_data.json \
    --val-data ./data/trading_llm_dataset/val_data.json \
    --output-dir ./models/trading_llm \
    --learning-rate 2e-5 \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --bf16
```

### 3. Generate Explanations for Trading Decisions

Use the trained model to explain RL trading decisions:

```bash
python -m trading_llm.train_llm infer \
    --rl-model /path/to/your/rl_model.zip \
    --llm-model ./models/trading_llm \
    --base-model meta-llama/Meta-Llama-3-8B-Instructt \
    --market-data /path/to/latest_market_data.parquet
```

### 4. Generate Market Commentary

Generate a comprehensive market analysis:

```bash
python -m trading_llm.train_llm commentary \
    --llm-model ./models/trading_llm \
    --base-model meta-llama/Meta-Llama-3-8B-Instructt \
    --market-data /path/to/market_data.parquet \
    --symbol "BTC/USD" \
    --output-file ./reports/btc_analysis.txt
```

## Programmatic Usage

You can also use the Trading LLM module programmatically in your Python code:

```python
from trading_llm import RLLMExplainer

# Initialize the explainer
explainer = RLLMExplainer(
    rl_model_path="/path/to/your/rl_model.zip",
    llm_model_path="./models/trading_llm",
    llm_base_model="meta-llama/Meta-Llama-3-8B-Instructt"
)

# Example observation from your trading environment
observation = {
    "market": market_data  # Your market data in the format expected by the RL model
}

# Get explanation for a trading decision
result = explainer.explain_decision(
    observation=observation,
    raw_ohlcv=ohlcv_data  # Optional DataFrame with OHLCV data for better explanations
)

# Print the explanation
print(f"Action: {result['action']}")
print(f"Explanation: {result['explanation']}")
```

## Architecture

The module consists of several key components:

1. **TradingLLM**: Wrapper around a language model with LoRA fine-tuning capabilities
2. **RLStateExtractor**: Extracts state information and predictions from the RL model
3. **TradingDatasetGenerator**: Creates training data for fine-tuning the LLM
4. **RLLMExplainer**: Combines RL model and LLM for generating explanations
5. **MarketCommentaryGenerator**: Generates market analyses and summaries

## Advanced Configuration

### Customizing the LLM

You can adjust various parameters when initializing the TradingLLM:

```python
from trading_llm import TradingLLM

model = TradingLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instructt",  # Base model
    lora_r=16,                             # LoRA rank parameter
    lora_alpha=32,                         # LoRA alpha parameter
    lora_dropout=0.05,                     # LoRA dropout rate
    load_in_8bit=False,                    # Whether to load in 8-bit precision
    load_in_4bit=True,                     # Whether to load in 4-bit precision
    use_flash_attn=True                    # Whether to use flash attention
)
```

### Custom Training Loop

For more control over the training process:

```python
from trading_llm import TradingLLM, create_dataloaders, TradingTrainer

# Initialize model
model = TradingLLM(model_name="meta-llama/Meta-Llama-3-8B-Instructt")
model.apply_lora()  # Apply LoRA adapters

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    train_data_path="./data/train.json",
    val_data_path="./data/val.json",
    tokenizer=model.tokenizer,
    batch_size=4
)

# Initialize trainer
trainer = TradingTrainer(
    model=model,
    tokenizer=model.tokenizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    output_dir="./models/custom_training",
    num_epochs=3,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    use_wandb=True,
    wandb_project="trading-llm"
)

# Train and evaluate
trainer.train()
trainer.evaluate()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research or project, please cite:

```
@software{trading_llm,
  author = {Your Name},
  title = {Trading LLM: Explanations for RL Trading Decisions},
  year = {2023},
  url = {https://github.com/yourusername/trading-llm}
}
```
