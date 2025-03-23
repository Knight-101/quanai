# Incremental Training for Trading Bot

This directory contains scripts for incremental training of the trading bot model. The approach allows you to train the model in phases, increasing the number of steps gradually while evaluating performance after each phase.

## Benefits of Incremental Training

- **Checkpoint Safety**: Prevents loss of progress by saving models after each phase
- **Progressive Evaluation**: Evaluates model performance throughout training
- **Adaptive Step Sizes**: Starts with smaller step increments, increasing as training progresses
- **Detailed Logging**: Maintains logs of training progress and performance metrics
- **Automatic Recovery**: Can continue from the last successful phase if a phase fails

## Scripts Overview

1. `run_incremental_training.py` - Main Python script that manages the incremental training process
2. `evaluate_model.py` - Evaluates model performance after each training phase
3. `start_training.sh` - Simple bash script to start the incremental training process

## Quick Start

To start training with the default parameters (1M total steps, starting with 100K):

```bash
./scripts/start_training.sh
```

To enable Weights & Biases logging:

```bash
./scripts/start_training.sh --use-wandb trading_project_name
```

## Advanced Usage

### Running with Custom Parameters

You can run the Python script directly with custom parameters:

```bash
python scripts/run_incremental_training.py \
  --total-steps 500000 \
  --initial-steps 50000 \
  --model-dir models/custom_run \
  --log-dir logs/custom_run \
  --eval-episodes 5 \
  --use-wandb \
  --wandb-project custom_project_name
```

### Available Parameters

- `--total-steps`: Total number of training steps across all phases (default: 1,000,000)
- `--initial-steps`: Number of steps for first training phase (default: 100,000)
- `--model-dir`: Directory to save model checkpoints (default: models/incremental)
- `--log-dir`: Directory to save logs (default: logs/incremental)
- `--eval-episodes`: Number of episodes for evaluation after each phase (default: 10)
- `--use-wandb`: Flag to enable Weights & Biases logging
- `--wandb-project`: WandB project name (default: trading_incremental)
- `--wandb-entity`: WandB entity name (optional)

### Running Evaluation Only

To evaluate a previously trained model:

```bash
python scripts/evaluate_model.py \
  --model-path models/incremental/phase5_model \
  --episodes 20 \
  --log-wandb
```

## Phase Progression

The training follows this progression of phases:

1. Phase 1: Initial steps (default: 100K)
2. Phase 2: +100K steps (total: 200K)
3. Phase 3: +100K steps (total: 300K)
4. Phase 4: +200K steps (total: 500K)
5. Phase 5: +300K steps (total: 800K)
6. Phase 6: +200K steps (total: 1M)

Each phase builds upon the previous phase's model, continuing training from where it left off.

## Monitoring Progress

- View real-time progress in the console
- Check detailed logs in `logs/incremental_training.log`
- Review phase details in `logs/incremental/phaseX_info.json` files
- Examine evaluation metrics in `logs/incremental/eval_phaseX/` directories
- Monitor training with the Weights & Biases dashboard if enabled

## What to Expect

1. The initial phase focuses on learning basic patterns.
2. Middle phases (2-4) often show the most improvement.
3. Later phases (5+) tend to show more incremental improvements or refinements.
4. If performance plateaus in later phases, consider adjusting hyperparameters or model architecture.

## Troubleshooting

- If a training phase fails, the script will log the error but attempt to continue with the next phase.
- Check the log file for specific error messages.
- The most common issues are related to memory limitations or environment errors.
- You can manually restart from any phase by specifying the appropriate model path from a previous phase.

## Manual Training

You can use the `scripts/manual_incremental.sh` script to manually run the incremental training process. Here's how to use it:

1. **Initial training phase:**

   ```bash
   ./scripts/manual_incremental.sh init 100000
   ```

   This runs the first 100K steps and saves as "phase1_model"

2. **Continue training for subsequent phases:**

   ```bash
   ./scripts/manual_incremental.sh continue 2 1 100000
   ```

   This continues from phase 1 to phase 2, adding 100K more steps

3. **Evaluate any phase:**
   ```bash
   ./scripts/manual_incremental.sh evaluate 2
   ```
   This evaluates the model from phase 2

The script will automatically follow the 1M total steps progression with the right increments for each phase, and it will prompt you to evaluate models after each training phase.
