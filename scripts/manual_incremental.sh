#!/bin/bash
# Manual Incremental Training Helper Script
# This script helps you manually train models in phases

set -e  # Exit on errors

# Configure this as needed
MODEL_BASE_DIR="models/manual"
LOG_BASE_DIR="logs/manual"
ENABLE_LOGGING=false  # Set to true to redirect output to log files

# Create directories
mkdir -p "$MODEL_BASE_DIR"
mkdir -p "$LOG_BASE_DIR"

# Parse global options
if [ "$1" == "--log" ]; then
    ENABLE_LOGGING=true
    shift  # Remove the --log option from the arguments
fi

# Function to display help message
show_help() {
    echo "Manual Incremental Training Helper"
    echo ""
    echo "USAGE:"
    echo "  $0 [--log] init 100000                # Initial training for 100K steps"
    echo "  $0 [--log] continue 2 1 100000        # Continue from phase 1 to phase 2 with 100K steps"
    echo "  $0 [--log] evaluate 2                 # Evaluate model from phase 2"
    echo ""
    echo "OPTIONS:"
    echo "  --log                          - Redirect output to log files (may hide progress bars)"
    echo ""
    echo "COMMANDS:"
    echo "  init STEPS                     - Start initial training phase"
    echo "  continue NEW_PHASE PREV_PHASE STEPS - Continue training from previous phase"
    echo "  evaluate PHASE                 - Evaluate a model from a specific phase"
    echo "  help                           - Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  # Complete 1M steps in phases:"
    echo "  $0 init 100000                 # Phase 1: 100K"
    echo "  $0 continue 2 1 100000         # Phase 2: +100K (total: 200K)"
    echo "  $0 continue 3 2 100000         # Phase 3: +100K (total: 300K)"
    echo "  $0 continue 4 3 200000         # Phase 4: +200K (total: 500K)"
    echo "  $0 continue 5 4 300000         # Phase 5: +300K (total: 800K)"
    echo "  $0 continue 6 5 200000         # Phase 6: +200K (total: 1M)"
}

# Function to run initial training
run_initial_training() {
    local steps=$1
    
    echo "===================================================================="
    echo "  STARTING INITIAL TRAINING (PHASE 1) WITH $steps STEPS"
    echo "===================================================================="
    echo ""
    
    # Create phase directories
    PHASE_DIR="$MODEL_BASE_DIR/phase1"
    mkdir -p "$PHASE_DIR"
    
    # Check if model already exists
    if [ -f "$PHASE_DIR/model.zip" ] || [ -f "$PHASE_DIR/final_model" ]; then
        read -p "Model for phase 1 already exists. Overwrite? (y/n): " confirm
        if [[ $confirm != "y" ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    
    # Run initial training
    echo "Running initial training with $steps steps..."
    echo "Saving to directory $PHASE_DIR"
    
    # Create log directory for this phase
    mkdir -p "$LOG_BASE_DIR/phase1"
    
    # Actual training command
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "Output will be logged to $LOG_BASE_DIR/phase1/training.log"
        python main_opt.py --training-steps $steps --model-dir "$PHASE_DIR" 2>&1 | tee "$LOG_BASE_DIR/phase1/training.log"
    else
        echo "Showing live progress (not logging to file)..."
        python main_opt.py --training-steps $steps --model-dir "$PHASE_DIR"
    fi
    
    # Copy final model to phase1_model for consistency with other phases
    if [ -f "$PHASE_DIR/final_model" ]; then
        cp "$PHASE_DIR/final_model" "$PHASE_DIR/phase1_model"
    elif [ -f "$PHASE_DIR/model" ]; then
        cp "$PHASE_DIR/model" "$PHASE_DIR/phase1_model"
    elif [ -f "$PHASE_DIR/model.zip" ]; then
        cp "$PHASE_DIR/model.zip" "$PHASE_DIR/phase1_model.zip"
    elif [ -f "$PHASE_DIR/final_model.zip" ]; then
        cp "$PHASE_DIR/final_model.zip" "$PHASE_DIR/phase1_model.zip"
    fi
    
    # Copy environment file if exists
    if [ -f "$PHASE_DIR/final_env.pkl" ]; then
        cp "$PHASE_DIR/final_env.pkl" "$PHASE_DIR/phase1_env.pkl"
    elif [ -f "$PHASE_DIR/vec_normalize.pkl" ]; then
        cp "$PHASE_DIR/vec_normalize.pkl" "$PHASE_DIR/phase1_env.pkl"
    fi
    
    echo ""
    echo "✅ Initial training complete!"
    echo "Model saved to $PHASE_DIR"
    echo ""
    
    # Offer to evaluate
    read -p "Do you want to evaluate this model now? (y/n): " do_eval
    if [[ $do_eval == "y" ]]; then
        evaluate_model 1
    fi
}

# Function to continue training
continue_training() {
    local new_phase=$1
    local prev_phase=$2
    local steps=$3
    
    echo "===================================================================="
    echo "  CONTINUING TRAINING FROM PHASE $prev_phase TO PHASE $new_phase"
    echo "  ADDING $steps ADDITIONAL STEPS"
    echo "===================================================================="
    echo ""
    
    # Create phase directories
    PREV_PHASE_DIR="$MODEL_BASE_DIR/phase$prev_phase"
    NEW_PHASE_DIR="$MODEL_BASE_DIR/phase$new_phase"
    
    mkdir -p "$PREV_PHASE_DIR"
    mkdir -p "$NEW_PHASE_DIR"
    
    # Check if previous model exists
    PREV_MODEL_PATH=""
    # First check for phase-specific models with various extensions
    if [ -f "$PREV_PHASE_DIR/phase${prev_phase}_model" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/phase${prev_phase}_model"
    elif [ -f "$PREV_PHASE_DIR/phase${prev_phase}_model.zip" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/phase${prev_phase}_model.zip"
    # Then check for standard model filenames
    elif [ -f "$PREV_PHASE_DIR/final_model" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/final_model"
    elif [ -f "$PREV_PHASE_DIR/model" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/model"
    elif [ -f "$PREV_PHASE_DIR/final_model.zip" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/final_model.zip"
    elif [ -f "$PREV_PHASE_DIR/model.zip" ]; then
        PREV_MODEL_PATH="$PREV_PHASE_DIR/model.zip"
    else
        echo "❌ ERROR: No model files found in $PREV_PHASE_DIR"
        echo "Please ensure you've completed phase $prev_phase first."
        exit 1
    fi
    
    # Check for environment file
    PREV_ENV_PATH=""
    if [ -f "$PREV_PHASE_DIR/phase${prev_phase}_env.pkl" ]; then
        PREV_ENV_PATH="$PREV_PHASE_DIR/phase${prev_phase}_env.pkl"
    elif [ -f "$PREV_PHASE_DIR/final_env.pkl" ]; then
        PREV_ENV_PATH="$PREV_PHASE_DIR/final_env.pkl"
    elif [ -f "$PREV_PHASE_DIR/vec_normalize.pkl" ]; then
        PREV_ENV_PATH="$PREV_PHASE_DIR/vec_normalize.pkl"
    else
        echo "⚠️ Warning: No environment file found in $PREV_PHASE_DIR"
    fi
    
    # Create log directory for this phase
    mkdir -p "$LOG_BASE_DIR/phase$new_phase"
    
    # Run continued training
    echo "Continuing training from phase $prev_phase with $steps additional steps..."
    echo "Loading model from: $PREV_MODEL_PATH"
    if [ -n "$PREV_ENV_PATH" ]; then
        echo "Loading environment from: $PREV_ENV_PATH"
    fi
    echo "Saving new model to: $NEW_PHASE_DIR"
    
    # Actual training command
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "Output will be logged to $LOG_BASE_DIR/phase$new_phase/training.log"
        if [ -n "$PREV_ENV_PATH" ]; then
            python main_opt.py --continue-training \
                --model-path "$PREV_MODEL_PATH" \
                --env-path "$PREV_ENV_PATH" \
                --additional-steps $steps \
                --model-dir "$NEW_PHASE_DIR" 2>&1 | tee "$LOG_BASE_DIR/phase$new_phase/training.log"
        else
            python main_opt.py --continue-training \
                --model-path "$PREV_MODEL_PATH" \
                --additional-steps $steps \
                --model-dir "$NEW_PHASE_DIR" 2>&1 | tee "$LOG_BASE_DIR/phase$new_phase/training.log"
        fi
    else
        echo "Showing live progress (not logging to file)..."
        if [ -n "$PREV_ENV_PATH" ]; then
            python main_opt.py --continue-training \
                --model-path "$PREV_MODEL_PATH" \
                --env-path "$PREV_ENV_PATH" \
                --additional-steps $steps \
                --model-dir "$NEW_PHASE_DIR"
        else
            python main_opt.py --continue-training \
                --model-path "$PREV_MODEL_PATH" \
                --additional-steps $steps \
                --model-dir "$NEW_PHASE_DIR"
        fi
    fi
    
    # Determine file extension based on input model
    if [[ "$PREV_MODEL_PATH" == *.zip ]]; then
        EXT=".zip"
    else
        EXT=""
    fi
    
    # Copy final model to phase-specific name for consistency
    if [ -f "$NEW_PHASE_DIR/final_continued_model" ]; then
        cp "$NEW_PHASE_DIR/final_continued_model" "$NEW_PHASE_DIR/phase${new_phase}_model"
    elif [ -f "$NEW_PHASE_DIR/final_model" ]; then
        cp "$NEW_PHASE_DIR/final_model" "$NEW_PHASE_DIR/phase${new_phase}_model"
    elif [ -f "$NEW_PHASE_DIR/model" ]; then
        cp "$NEW_PHASE_DIR/model" "$NEW_PHASE_DIR/phase${new_phase}_model"
    elif [ -f "$NEW_PHASE_DIR/final_continued_model.zip" ]; then
        cp "$NEW_PHASE_DIR/final_continued_model.zip" "$NEW_PHASE_DIR/phase${new_phase}_model.zip"
    elif [ -f "$NEW_PHASE_DIR/final_model.zip" ]; then
        cp "$NEW_PHASE_DIR/final_model.zip" "$NEW_PHASE_DIR/phase${new_phase}_model.zip"
    elif [ -f "$NEW_PHASE_DIR/model.zip" ]; then
        cp "$NEW_PHASE_DIR/model.zip" "$NEW_PHASE_DIR/phase${new_phase}_model.zip"
    fi
    
    # Copy environment file to phase-specific name
    if [ -f "$NEW_PHASE_DIR/final_continued_env.pkl" ]; then
        cp "$NEW_PHASE_DIR/final_continued_env.pkl" "$NEW_PHASE_DIR/phase${new_phase}_env.pkl"
    elif [ -f "$NEW_PHASE_DIR/final_env.pkl" ]; then
        cp "$NEW_PHASE_DIR/final_env.pkl" "$NEW_PHASE_DIR/phase${new_phase}_env.pkl"
    elif [ -f "$NEW_PHASE_DIR/vec_normalize.pkl" ]; then
        cp "$NEW_PHASE_DIR/vec_normalize.pkl" "$NEW_PHASE_DIR/phase${new_phase}_env.pkl"
    fi
    
    echo ""
    echo "✅ Phase $new_phase training complete!"
    echo "Model saved to $NEW_PHASE_DIR/phase${new_phase}_model$EXT"
    echo ""
    
    # Offer to evaluate
    read -p "Do you want to evaluate this model now? (y/n): " do_eval
    if [[ $do_eval == "y" ]]; then
        evaluate_model $new_phase
    fi
    
    # Show next suggested phase
    next_phase=$((new_phase + 1))
    next_steps=100000
    if [ $new_phase -eq 2 ]; then
        next_steps=100000
    elif [ $new_phase -eq 3 ]; then
        next_steps=200000
    elif [ $new_phase -eq 4 ]; then
        next_steps=300000
    elif [ $new_phase -eq 5 ]; then
        next_steps=200000
    fi
    
    echo ""
    echo "📋 NEXT SUGGESTED COMMAND:"
    echo "./scripts/manual_incremental.sh continue $next_phase $new_phase $next_steps"
    echo ""
}

# Function to evaluate a model
evaluate_model() {
    local phase=$1
    
    echo "===================================================================="
    echo "  EVALUATING MODEL FROM PHASE $phase"
    echo "===================================================================="
    echo ""
    
    # Phase directory
    PHASE_DIR="$MODEL_BASE_DIR/phase$phase"
    
    # Determine model path - check phase-specific model first
    if [ -f "$PHASE_DIR/phase${phase}_model" ]; then
        MODEL_PATH="$PHASE_DIR/phase${phase}_model"
    elif [ -f "$PHASE_DIR/phase${phase}_model.zip" ]; then
        MODEL_PATH="$PHASE_DIR/phase${phase}_model.zip"
    elif [ -f "$PHASE_DIR/final_model" ]; then
        MODEL_PATH="$PHASE_DIR/final_model"
    elif [ -f "$PHASE_DIR/model" ]; then
        MODEL_PATH="$PHASE_DIR/model"
    elif [ -f "$PHASE_DIR/final_model.zip" ]; then
        MODEL_PATH="$PHASE_DIR/final_model.zip"
    elif [ -f "$PHASE_DIR/model.zip" ]; then
        MODEL_PATH="$PHASE_DIR/model.zip"
    else
        echo "❌ ERROR: Model from phase $phase does not exist at $PHASE_DIR"
        exit 1
    fi
    
    # Determine environment path - check phase-specific env first
    ENV_PATH=""
    if [ -f "$PHASE_DIR/phase${phase}_env.pkl" ]; then
        ENV_PATH="$PHASE_DIR/phase${phase}_env.pkl"
    elif [ -f "$PHASE_DIR/final_env.pkl" ]; then
        ENV_PATH="$PHASE_DIR/final_env.pkl"
    elif [ -f "$PHASE_DIR/vec_normalize.pkl" ]; then
        ENV_PATH="$PHASE_DIR/vec_normalize.pkl"
    fi
    
    # Create evaluation directory
    mkdir -p "$LOG_BASE_DIR/eval_phase$phase"
    
    # Run evaluation
    echo "Running evaluation for phase $phase model..."
    echo "Using model: $MODEL_PATH"
    if [ -n "$ENV_PATH" ]; then
        echo "Using environment: $ENV_PATH"
    fi
    
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "Output will be logged to $LOG_BASE_DIR/eval_phase$phase/eval.log"
        if [ -n "$ENV_PATH" ]; then
            python scripts/evaluate_model.py \
                --model-path "$MODEL_PATH" \
                --env-path "$ENV_PATH" \
                --episodes 10 \
                --phase $phase \
                --output-dir "$LOG_BASE_DIR/eval_phase$phase" 2>&1 | tee "$LOG_BASE_DIR/eval_phase$phase/eval.log"
        else
            python scripts/evaluate_model.py \
                --model-path "$MODEL_PATH" \
                --episodes 10 \
                --phase $phase \
                --output-dir "$LOG_BASE_DIR/eval_phase$phase" 2>&1 | tee "$LOG_BASE_DIR/eval_phase$phase/eval.log"
        fi
    else
        echo "Showing live progress (not logging to file)..."
        if [ -n "$ENV_PATH" ]; then
            python scripts/evaluate_model.py \
                --model-path "$MODEL_PATH" \
                --env-path "$ENV_PATH" \
                --episodes 10 \
                --phase $phase \
                --output-dir "$LOG_BASE_DIR/eval_phase$phase"
        else
            python scripts/evaluate_model.py \
                --model-path "$MODEL_PATH" \
                --episodes 10 \
                --phase $phase \
                --output-dir "$LOG_BASE_DIR/eval_phase$phase"
        fi
    fi
    
    echo ""
    echo "✅ Evaluation complete!"
    echo "Results saved to $LOG_BASE_DIR/eval_phase$phase"
    echo ""
}

# Main command processing
case "$1" in
    "init")
        if [ -z "$2" ]; then
            echo "❌ ERROR: Please specify the number of steps for initial training"
            show_help
            exit 1
        fi
        run_initial_training $2
        ;;
    
    "continue")
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
            echo "❌ ERROR: Missing arguments for continue command"
            show_help
            exit 1
        fi
        continue_training $2 $3 $4
        ;;
    
    "evaluate")
        if [ -z "$2" ]; then
            echo "❌ ERROR: Please specify the phase to evaluate"
            show_help
            exit 1
        fi
        evaluate_model $2
        ;;
    
    "help" | "-h" | "--help")
        show_help
        ;;
    
    *)
        echo "❌ ERROR: Unknown command '$1'"
        show_help
        exit 1
        ;;
esac 