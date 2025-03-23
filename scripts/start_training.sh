#!/bin/bash
# Start the incremental training process with appropriate parameters

# Ensure we're in the correct directory
cd "$(dirname "$0")/.." || exit 1

# Create necessary directories
mkdir -p models/incremental logs/incremental

# Check if we want to use wandb for logging
USE_WANDB=""
if [ "$1" == "--use-wandb" ]; then
  USE_WANDB="--use-wandb"
  shift
fi

# Get the project name for wandb if provided
WANDB_PROJECT="trading_incremental"
if [ -n "$1" ]; then
  WANDB_PROJECT="$1"
fi

echo "====================================================="
echo "Starting Incremental Training Process"
echo "====================================================="
echo "Total Steps: 1,000,000"
echo "Initial Phase: 100,000 steps"
echo "Using WandB: ${USE_WANDB:-No}"
if [ -n "$USE_WANDB" ]; then
  echo "WandB Project: $WANDB_PROJECT"
fi
echo "====================================================="
echo "Starting in 5 seconds... Press Ctrl+C to cancel"
echo "====================================================="
sleep 5

# Start the training process
python scripts/run_incremental_training.py \
  --total-steps 1000000 \
  --initial-steps 100000 \
  --model-dir models/incremental \
  --log-dir logs/incremental \
  --eval-episodes 10 \
  $USE_WANDB \
  --wandb-project "$WANDB_PROJECT"

echo "====================================================="
echo "Incremental Training Process Completed"
echo "====================================================="
echo "Check logs/incremental_training.log for details"
echo "=====================================================" 