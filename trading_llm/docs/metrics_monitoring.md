# Metrics Monitoring Guide for Trading LLM

This guide provides detailed information on how to monitor key metrics during the training process to ensure your Trading LLM model is learning effectively.

## Training Metrics Overview

When training your LLM model, several key metrics should be monitored to ensure proper learning:

| Metric          | Description                             | Target Values                                 | Warning Signs                           |
| --------------- | --------------------------------------- | --------------------------------------------- | --------------------------------------- |
| Training Loss   | Measures how well the model is learning | Declining from ~2.0 to ~0.7                   | Plateauing early, increasing suddenly   |
| Validation Loss | Measures generalization ability         | Declining, slightly higher than training loss | Diverging from training loss            |
| Perplexity      | Exponential of loss, ease of prediction | Declining from ~7-12 to ~2-3                  | Remaining above 5 after multiple epochs |
| Learning Rate   | Rate of parameter updates               | Following scheduler pattern                   | -                                       |
| GPU Memory      | Resource utilization                    | Stable, below 90% of available                | Near 100%, OOM errors                   |
| Tokens/second   | Processing speed                        | 500-2000 depending on hardware                | Dropping significantly during training  |

## How to Monitor Metrics

### Using Built-in Logging

The training process automatically logs metrics to stdout and to TensorBoard if enabled. You can monitor these logs to track progress.

```bash
# Run training with output visible
python -m trading_llm.train_llm train \
  --base-model meta-llama/Meta-Llama-3-8B-Instructt \
  --train-data ./data/trading_dataset/train_data.json \
  --output-dir ./models/trading_llm
```

### Using TensorBoard

For a more visual inspection of training metrics, use TensorBoard:

```bash
# First, ensure you have tensorboard installed
pip install tensorboard

# Start TensorBoard
tensorboard --logdir ./models/trading_llm/runs

# Access TensorBoard in your browser at http://localhost:6006
```

### Using Weights & Biases (Optional)

For advanced monitoring, you can use Weights & Biases:

```bash
# Install Weights & Biases
pip install wandb

# Login to your account
wandb login

# The training script will automatically log metrics if wandb is installed
```

## Expected Metric Patterns

### 1. Training Loss

![Training Loss Pattern](https://example.com/training_loss.png)

**Expected pattern:**

- Initial rapid decrease in the first 10-20% of training
- Followed by steady, slower decrease
- Eventually gradual flattening as model converges

**Specific values to expect:**

- Starting value: ~1.5-2.5 depending on dataset complexity
- After 1 epoch: Should drop by at least 30-40%
- Final value after 3 epochs: ~0.5-0.9

**Problematic patterns:**

- Loss plateaus very early (e.g., in first 20% of training)
- Loss increases for multiple consecutive steps
- Extreme fluctuations between batches

**Solutions to problems:**

- Early plateau: Increase learning rate or check data quality
- Increasing loss: Decrease learning rate or check for data anomalies
- Fluctuations: Increase batch size or gradient accumulation steps

### 2. Validation Loss

**Expected pattern:**

- Should follow training loss but with more fluctuation
- Gap between training and validation loss should be small
- May occasionally increase slightly before decreasing again

**Specific values to expect:**

- Typically 5-15% higher than training loss
- Final value: ~0.6-1.0 after 3 epochs

**Problematic patterns:**

- Validation loss steadily increases while training loss decreases
- Gap between training and validation loss grows consistently
- Validation loss shows wild fluctuations

**Solutions to problems:**

- Increasing validation loss: Introduce dropout, reduce training time
- Growing gap: Add more training data, implement early stopping
- Fluctuations: Ensure validation set is representative and sufficiently large

### 3. Perplexity

**Expected pattern:**

- Exponential decrease relative to loss
- Sharper initial decrease, then leveling off

**Specific values to expect:**

- Starting: ~7-12
- After 1 epoch: ~3-5
- Final after 3 epochs: ~1.5-3

**Problematic patterns:**

- Perplexity remains high (>5) after several epochs
- Perplexity starts very high (>20)

**Solutions to problems:**

- High perplexity: Check data quality, increase training time
- Very high initial perplexity: Check tokenization and preprocessing

### 4. Learning Rate

**Expected pattern:**

- If using linear warmup: Increase to peak, then linear decay
- If using cosine schedule: Increase to peak, then cosine decay

**Specific values to expect:**

- Peak value: As configured (typically 2e-4)
- Warmup phase: First 10% of steps
- Decay phase: Remaining 90% of steps

### 5. GPU Memory Utilization

**Expected pattern:**

- Sharp increase during initialization
- Stable during training
- Periodic small spikes during evaluation

**Specific values to expect:**

- Should stabilize below 90% of available GPU memory
- Memory usage typically higher during the first batch

**Problematic patterns:**

- Memory usage continuously increasing
- Memory approaching 100% of available

**Solutions to problems:**

- Reduce batch size
- Increase gradient accumulation steps
- Use stronger quantization (8-bit or 4-bit)

### 6. Tokens Per Second

**Expected pattern:**

- Lower during the first few batches
- Stabilizes for the remainder of training
- May decrease slightly as training progresses

**Specific values to expect:**

- A100 GPU: ~1500-2000 tokens/sec
- V100 GPU: ~1000-1500 tokens/sec
- RTX 3090: ~700-1200 tokens/sec
- RTX 2080 Ti: ~400-800 tokens/sec

**Problematic patterns:**

- Significant drop in processing speed during training
- Extremely low tokens/sec relative to hardware capabilities

**Solutions to problems:**

- Check for CPU bottlenecks in data loading
- Ensure GPU is properly utilized
- Check for background processes consuming resources

## Sample Metrics at Different Training Stages

### Beginning of Training (First 10% of steps)

```
Step 100/1000:
- Training Loss: 2.1453
- Learning Rate: 0.0001
- Perplexity: 8.5432
- Tokens/second: 1250
- GPU Memory: 14.2 GB / 16 GB (88.75%)
```

### Middle of Training (Around 50% of steps)

```
Step 500/1000:
- Training Loss: 1.1342
- Validation Loss: 1.2531
- Learning Rate: 0.00015
- Perplexity: 3.1089
- Tokens/second: 1320
- GPU Memory: 14.3 GB / 16 GB (89.38%)
```

### End of Training (Last 10% of steps)

```
Step 900/1000:
- Training Loss: 0.7231
- Validation Loss: 0.8124
- Learning Rate: 0.00002
- Perplexity: 2.0608
- Tokens/second: 1270
- GPU Memory: 14.3 GB / 16 GB (89.38%)
```

## Text Generation Quality Assessment

Beyond numerical metrics, regularly evaluate the quality of generated text:

### Early Training (After ~20% of steps)

- Expect coherent but generic explanations
- May see repetition or hallucination
- Technical indicator references may be incorrect

### Mid Training (After ~50% of steps)

- More specific explanations
- Correct references to technical indicators
- Better reasoning structure
- Some lingering inaccuracies

### End Training (After completion)

- Specific, accurate explanations
- Correct technical analysis
- Logical reasoning flow
- Balanced perspective
- Trading-specific terminology used appropriately

## Conclusion

Monitor these metrics throughout the training process to ensure your model is learning effectively. Regular evaluation of both numerical metrics and text quality will help identify issues early and guide adjustments to hyperparameters if needed.

For best results, save checkpoints regularly and evaluate text generation quality on a small validation set using the evaluation criteria outlined in the training guide.

Remember that metrics are tools to help understand model training, but the ultimate measure of success is the quality and accuracy of the explanations generated for trading decisions.
