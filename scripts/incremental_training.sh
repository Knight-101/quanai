#!/bin/bash
# Incremental Training Script for Trading Bot
# Runs training in manageable chunks to prevent data loss

set -e  # Exit immediately if a command fails

TOTAL_TARGET_STEPS=1000000  # Total desired training steps (changed from 2M to 1M)
INITIAL_STEPS=100000        # First phase steps
CHECKPOINT_DIR="models/incremental"
LOG_DIR="logs/incremental"
EVAL_EPISODES=10            # Number of episodes for evaluation after each phase

# Create necessary directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Initialize log file
LOG_FILE="${LOG_DIR}/training_progress.log"
echo "=== Incremental Training Started at $(date) ===" > $LOG_FILE

# Function to log messages
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local phase=$2
    
    log_message "Evaluating model from phase $phase..."
    
    # Run our standalone evaluation script
    python scripts/evaluate_model.py \
        --model-path "${model_path}" \
        --episodes ${EVAL_EPISODES} \
        --phase ${phase} \
        --output-dir "${LOG_DIR}/eval_phase${phase}"
    
    local eval_status=$?
    if [ $eval_status -ne 0 ]; then
        log_message "WARNING: Evaluation failed with status $eval_status"
    fi
}

# Phase 1: Initial training (from scratch)
log_message "Starting Phase 1: Initial training with $INITIAL_STEPS steps"
python main_opt.py --training-steps $INITIAL_STEPS --model-dir $CHECKPOINT_DIR

if [ $? -ne 0 ]; then
    log_message "ERROR: Initial training failed. Exiting."
    exit 1
fi

# Save the first phase model with special naming
mkdir -p $CHECKPOINT_DIR/phase1
if [ -f "$CHECKPOINT_DIR/final_model" ]; then
    cp $CHECKPOINT_DIR/final_model $CHECKPOINT_DIR/phase1/phase1_model
elif [ -f "$CHECKPOINT_DIR/model" ]; then
    cp $CHECKPOINT_DIR/model $CHECKPOINT_DIR/phase1/phase1_model
elif [ -f "$CHECKPOINT_DIR/final_model.zip" ]; then
    cp $CHECKPOINT_DIR/final_model.zip $CHECKPOINT_DIR/phase1/phase1_model.zip
elif [ -f "$CHECKPOINT_DIR/model.zip" ]; then
    cp $CHECKPOINT_DIR/model.zip $CHECKPOINT_DIR/phase1/phase1_model.zip
fi

# Copy environment file
if [ -f "$CHECKPOINT_DIR/vec_normalize.pkl" ]; then
    cp $CHECKPOINT_DIR/vec_normalize.pkl $CHECKPOINT_DIR/phase1/phase1_env.pkl
elif [ -f "$CHECKPOINT_DIR/env.pkl" ]; then
    cp $CHECKPOINT_DIR/env.pkl $CHECKPOINT_DIR/phase1/phase1_env.pkl
fi

# Determine model path for evaluation
PHASE1_MODEL_PATH=""
if [ -f "$CHECKPOINT_DIR/phase1/phase1_model" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/phase1_model"
elif [ -f "$CHECKPOINT_DIR/phase1/phase1_model.zip" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/phase1_model.zip"
elif [ -f "$CHECKPOINT_DIR/phase1/final_model" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/final_model"
elif [ -f "$CHECKPOINT_DIR/phase1/model" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/model"
elif [ -f "$CHECKPOINT_DIR/phase1/final_model.zip" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/final_model.zip"
elif [ -f "$CHECKPOINT_DIR/phase1/model.zip" ]; then
    PHASE1_MODEL_PATH="$CHECKPOINT_DIR/phase1/model.zip"
fi

# Run evaluation on phase 1
run_evaluation "$PHASE1_MODEL_PATH" 1

# Subsequent phases with increasing steps
phase=2
current_steps=$INITIAL_STEPS

while [ $current_steps -lt $TOTAL_TARGET_STEPS ]; then
    # Calculate next increment (with increasing size)
    if [ $phase -eq 2 ]; then
        next_increment=100000  # Same as initial
    elif [ $phase -eq 3 ]; then
        next_increment=200000  # Same as previous for phase 3
    elif [ $phase -eq 4 ]; then
        next_increment=200000  # Double for phase 4
    elif [ $phase -eq 5 ]; then
        next_increment=300000  # Triple for phase 5
    else
        next_increment=200000  # Last phase gets remainder up to 1M
    fi
    
    # Ensure we don't exceed total target
    remaining_steps=$((TOTAL_TARGET_STEPS - current_steps))
    if [ $next_increment -gt $remaining_steps ]; then
        next_increment=$remaining_steps
    fi
    
    log_message "Starting Phase $phase: Continuing training for $next_increment additional steps"
    log_message "Total steps after this phase will be $((current_steps + next_increment))"
    
    # Create phase directory
    mkdir -p $CHECKPOINT_DIR/phase$phase
    
    # Continue training from previous phase
    prev_phase=$((phase - 1))
    
    # Find previous phase model
    PREV_MODEL_PATH=""
    if [ -f "$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_model" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_model"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_model.zip" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_model.zip"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/final_model" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/final_model"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/model" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/model"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/final_model.zip" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/final_model.zip"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/model.zip" ]; then
        PREV_MODEL_PATH="$CHECKPOINT_DIR/phase${prev_phase}/model.zip"
    else
        log_message "ERROR: No model file found from previous phase ${prev_phase}. Exiting."
        exit 1
    fi
    
    # Find previous phase environment
    PREV_ENV_PATH=""
    if [ -f "$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_env.pkl" ]; then
        PREV_ENV_PATH="$CHECKPOINT_DIR/phase${prev_phase}/phase${prev_phase}_env.pkl"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/final_env.pkl" ]; then
        PREV_ENV_PATH="$CHECKPOINT_DIR/phase${prev_phase}/final_env.pkl"
    elif [ -f "$CHECKPOINT_DIR/phase${prev_phase}/vec_normalize.pkl" ]; then
        PREV_ENV_PATH="$CHECKPOINT_DIR/phase${prev_phase}/vec_normalize.pkl"
    else
        log_message "WARNING: No environment file found from previous phase ${prev_phase}."
    fi
    
    # Run the training command
    if [ -n "$PREV_ENV_PATH" ]; then
        python main_opt.py --continue-training \
                        --model-path $PREV_MODEL_PATH \
                        --env-path $PREV_ENV_PATH \
                        --additional-steps $next_increment \
                        --model-dir $CHECKPOINT_DIR/phase$phase
    else
        python main_opt.py --continue-training \
                        --model-path $PREV_MODEL_PATH \
                        --additional-steps $next_increment \
                        --model-dir $CHECKPOINT_DIR/phase$phase
    fi
    
    if [ $? -ne 0 ]; then
        log_message "ERROR: Phase $phase training failed. Exiting."
        exit 1
    fi
    
    # Determine file extension based on input model
    if [[ "$PREV_MODEL_PATH" == *.zip ]]; then
        EXT=".zip"
    else
        EXT=""
    fi
    
    # Save this phase with special naming
    if [ -f "$CHECKPOINT_DIR/phase$phase/final_continued_model" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_continued_model $CHECKPOINT_DIR/phase$phase/phase${phase}_model
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_model" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_model $CHECKPOINT_DIR/phase$phase/phase${phase}_model
    elif [ -f "$CHECKPOINT_DIR/phase$phase/model" ]; then
        cp $CHECKPOINT_DIR/phase$phase/model $CHECKPOINT_DIR/phase$phase/phase${phase}_model
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_continued_model.zip" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_continued_model.zip $CHECKPOINT_DIR/phase$phase/phase${phase}_model.zip
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_model.zip" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_model.zip $CHECKPOINT_DIR/phase$phase/phase${phase}_model.zip
    elif [ -f "$CHECKPOINT_DIR/phase$phase/model.zip" ]; then
        cp $CHECKPOINT_DIR/phase$phase/model.zip $CHECKPOINT_DIR/phase$phase/phase${phase}_model.zip
    fi
    
    # Save environment file
    if [ -f "$CHECKPOINT_DIR/phase$phase/final_continued_env.pkl" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_continued_env.pkl $CHECKPOINT_DIR/phase$phase/phase${phase}_env.pkl
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_env.pkl" ]; then
        cp $CHECKPOINT_DIR/phase$phase/final_env.pkl $CHECKPOINT_DIR/phase$phase/phase${phase}_env.pkl
    elif [ -f "$CHECKPOINT_DIR/phase$phase/vec_normalize.pkl" ]; then
        cp $CHECKPOINT_DIR/phase$phase/vec_normalize.pkl $CHECKPOINT_DIR/phase$phase/phase${phase}_env.pkl
    fi
    
    # Find current phase model for evaluation
    CURRENT_MODEL_PATH=""
    if [ -f "$CHECKPOINT_DIR/phase$phase/phase${phase}_model" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/phase${phase}_model"
    elif [ -f "$CHECKPOINT_DIR/phase$phase/phase${phase}_model.zip" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/phase${phase}_model.zip"
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_model" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/final_model"
    elif [ -f "$CHECKPOINT_DIR/phase$phase/model" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/model"
    elif [ -f "$CHECKPOINT_DIR/phase$phase/final_model.zip" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/final_model.zip"
    elif [ -f "$CHECKPOINT_DIR/phase$phase/model.zip" ]; then
        CURRENT_MODEL_PATH="$CHECKPOINT_DIR/phase$phase/model.zip"
    fi
    
    # Run evaluation for this phase
    run_evaluation "$CURRENT_MODEL_PATH" $phase
    
    # Update counters
    current_steps=$((current_steps + next_increment))
    phase=$((phase + 1))
    
    log_message "Completed $current_steps/$TOTAL_TARGET_STEPS total steps"
done

# Find final model path for symbolic link
FINAL_PHASE=$((phase-1))
FINAL_MODEL_PATH=""
if [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/phase${FINAL_PHASE}_model" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/phase${FINAL_PHASE}_model"
elif [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/phase${FINAL_PHASE}_model.zip" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/phase${FINAL_PHASE}_model.zip"
elif [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/final_model" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/final_model"
elif [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/model" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/model"
elif [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/final_model.zip" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/final_model.zip"
elif [ -f "$CHECKPOINT_DIR/phase$FINAL_PHASE/model.zip" ]; then
    FINAL_MODEL_PATH="$CHECKPOINT_DIR/phase$FINAL_PHASE/model.zip"
fi

log_message "Incremental training complete! Final model is at $FINAL_MODEL_PATH"

# Create a symbolic link to the final model for convenience
ln -sf $FINAL_MODEL_PATH $CHECKPOINT_DIR/final_best_model

log_message "=== Training finished at $(date) ===" 