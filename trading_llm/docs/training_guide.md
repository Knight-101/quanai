# Training Guide for Trading LLM

This guide provides step-by-step instructions for training the Trading LLM components and monitoring key metrics to ensure training quality.

## Prerequisites

Before starting, ensure you have:

1. A trained RL trading model (PPO) for generating explanations
2. Market data with OHLCV values (ideally in Parquet format)
3. Sufficient GPU resources (at least 16GB VRAM recommended)
4. All dependencies installed via `pip install -e .` from the project root

## Training Pipeline Overview

The complete training pipeline consists of the following steps:

1. Generate a training dataset from RL model decisions
2. Fine-tune the LLM on this dataset using LoRA
3. Optional: Evaluate the trained model
4. Deploy for inference (explanations and chatbot)

## Step 1: Generate Training Dataset

First, generate a dataset of trading decisions and corresponding explanations:

```bash
python -m trading_llm.train_llm generate \
  --rl-model /path/to/your/rl_model.zip \
  --market-data /path/to/market_data.parquet \
  --output-dir ./data/trading_dataset \
  --num-samples 10000 \
  --split-ratio 0.9
```

### Key Parameters:

- `--rl-model`: Path to your trained RL model
- `--market-data`: Path to your market data file
- `--output-dir`: Where to save the generated dataset
- `--num-samples`: Number of samples to generate (10000 recommended)
- `--split-ratio`: Train/test split ratio (0.9 recommended)

### Expected Output Files:

- `data/trading_dataset/train_data.json`: Training dataset
- `data/trading_dataset/eval_data.json`: Evaluation dataset

### Quality Checks:

After generation, inspect a few samples from the dataset to ensure quality:

```bash
head -n 20 ./data/trading_dataset/train_data.json
```

Ensure that:

1. Each sample has a clear action (buy/sell/hold)
2. Technical indicators are present and correctly formatted
3. The dataset has a good balance of different actions (not all buys or sells)

## Step 2: Fine-tune the LLM

Next, fine-tune the Mistral-7B model using LoRA:

```bash
python -m trading_llm.train_llm train \
  --base-model mistralai/Mistral-7B-v0.1 \
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

### Key Parameters:

- `--base-model`: Base model to fine-tune (Mistral-7B-v0.1 recommended)
- `--train-data`: Path to training data
- `--eval-data`: Path to evaluation data
- `--output-dir`: Where to save the fine-tuned model
- `--num-epochs`: Number of training epochs (5 recommended)
- `--batch-size`: Training batch size (adjust based on your GPU)
- `--gradient-accumulation-steps`: Increases effective batch size
- `--learning-rate`: Learning rate (2e-4 recommended)
- `--lora-r`: LoRA rank (32 recommended for financial explanations)
- `--lora-alpha`: LoRA alpha parameter (64 recommended)

### Training Resources:

- For a dataset of 10000 samples, training typically takes 3-10 hours on a good GPU
- With 4-bit quantization, you can train on GPUs with as little as 8GB VRAM
- For better quality, use 16GB+ VRAM and increase batch size if possible

### Key Metrics to Monitor:

During training, carefully monitor these metrics:

1. **Training Loss**: Should steadily decrease throughout training

   - Expected range: Starting ~1.5-2.5, decreasing to ~0.5-1.0
   - Warning sign: If loss stalls or increases for multiple steps

2. **Validation Loss**: Should decrease, though not as smoothly as training loss

   - Expected range: Similar to training loss but slightly higher
   - Warning sign: If validation loss diverges significantly from training loss (overfitting)

3. **Perplexity**: Exponential of the loss, should decrease

   - Expected range: Starting ~7-12, decreasing to ~1.5-3
   - Warning sign: If perplexity remains high (>5) after several epochs

4. **Learning Rate**: If using a scheduler, should follow expected pattern

   - Expected behavior: Warm up, then gradual decay

5. **GPU Memory Usage**: Monitor to avoid OOM errors

   - Adjust batch size or gradient accumulation if close to memory limit

6. **Training Speed**: Tokens/second processed
   - Expected range: 500-2000 tokens/sec depending on hardware

## Step 3: Evaluate the Model

After training, evaluate the model's ability to generate meaningful explanations:

```bash
python -m trading_llm.train_llm infer \
  --rl-model /path/to/your/rl_model.zip \
  --llm-model ./models/trading_llm \
  --market-data /path/to/test_market_data.parquet \
  --output-file ./evaluation_results.txt
```

### Evaluation Criteria:

When reviewing explanations, assess:

1. **Factual Accuracy**: Does the explanation align with the action?
2. **Technical Relevance**: Does it reference appropriate technical indicators?
3. **Coherence**: Is the explanation logically structured?
4. **Specificity**: Does it provide specific insights rather than generic statements?
5. **Language Quality**: Is the text fluent and professional?

Score a random sample of 10-20 explanations on a scale of 1-5 for each criterion, with a target average above 4.0.

## Step 4: Generate Market Commentary

Test the model's ability to generate broader market commentary:

```bash
python -m trading_llm.train_llm commentary \
  --llm-model ./models/trading_llm \
  --market-data /path/to/market_data.parquet \
  --output-file ./market_commentary.txt
```

### Quality Assessment:

1. **Market Awareness**: Does it capture overall market trends?
2. **Multi-factor Analysis**: Does it consider multiple factors?
3. **Temporal Awareness**: Does it correctly reference timeframes?
4. **Balanced Perspective**: Does it present balanced views rather than extreme predictions?

## Step 5: Test the Chatbot

Finally, test the interactive chatbot functionality:

```bash
python -m trading_llm.train_llm chat \
  --llm-model ./models/trading_llm \
  --market-data /path/to/recent_market_data.parquet \
  --rl-model /path/to/your/rl_model.zip
```

### Chatbot Evaluation Protocol:

Test the chatbot with these question categories:

1. **Technical Analysis**: "What technical indicators suggest a bullish trend?"
2. **Signal Explanation**: "Why did the model recommend a sell signal yesterday?"
3. **Market Context**: "How does the current market compare to last month?"
4. **Strategy Questions**: "How would this strategy perform in a bear market?"
5. **Follow-up Questions**: Test if the chatbot maintains context in conversation

For each category, rate responses on:

- Relevance (1-5)
- Accuracy (1-5)
- Helpfulness (1-5)
- Conversational Quality (1-5)

Target an average score of 4+ across all categories.

## Common Training Issues and Solutions

### 1. Loss Not Decreasing

- **Symptom**: Training loss plateaus early
- **Solution**: Increase learning rate or check dataset quality

### 2. Overfitting

- **Symptom**: Training loss decreases but validation loss increases
- **Solution**: Add more training data, increase dropout, or reduce training epochs

### 3. Low-Quality Explanations

- **Symptom**: Generated explanations are too generic
- **Solution**: Improve dataset quality, ensure technical indicators are included, fine-tune longer

### 4. Out of Memory Errors

- **Symptom**: Training crashes with OOM error
- **Solution**: Reduce batch size, increase gradient accumulation steps, or use stronger quantization

### 5. Slow Inference

- **Symptom**: Generating explanations takes too long
- **Solution**: Use 4-bit or 8-bit quantization for inference, optimize prompt length

## Advanced Training Techniques

For even better results, consider:

1. **Iterative Dataset Refinement**: Generate initial explanations, manually improve a subset, then retrain
2. **Instruction Tuning**: Add diverse instructions to make the model more flexible
3. **Multi-Task Training**: Train on both explanations and market commentary
4. **Reinforcement Learning from Human Feedback (RLHF)**: For production-quality results

## Deployment Considerations

When deploying the trained model:

1. Use 4-bit or 8-bit quantization for efficient inference
2. Implement caching for common queries
3. Set appropriate temperature (0.7 recommended for balance between creativity and precision)
4. Implement input validation to prevent prompt injection
5. Log and review the quality of explanations to identify areas for improvement

## Conclusion

By following this training guide and monitoring the specified metrics, you should be able to train a high-quality Trading LLM that can generate insightful trading explanations, market commentary, and engage in interactive financial discussions.

Regular retraining with new market data and trading decisions is recommended to keep the model current with evolving market conditions.
