#!/bin/bash
# Manual Incremental Training Helper Script
# This script helps you manually train models in phases

set -e  # Exit on errors

# Configure this as needed
MODEL_BASE_DIR="models/manual"
LOG_BASE_DIR="logs/manual"
ENABLE_LOGGING=false  # Set to true to redirect output to log files
HYPERPARAMS=""  # Initialize empty hyperparams string
DRIVE_IDS_FILE=""  # Initialize empty drive IDs file parameter
USE_RECOMMENDATIONS=false # Initialize use_recommendations flag

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
    echo "  $0 [--log] init 500000                # Initial training for 500K steps"
    echo "  $0 [--log] init 500000 --training-mode # Initial training with optimizations"
    echo "  $0 [--log] continue 2 1 500000        # Continue from phase 1 to phase 2 with 500K steps"
    echo "  $0 [--log] evaluate 2                 # Evaluate model from phase 2"
    echo ""
    echo "OPTIONS:"
    echo "  --log                          - Redirect output to log files (may hide progress bars)"
    echo "  --verbose                      - Enable verbose output (more detailed logging)"
    echo "  --training-mode                - Enable training optimizations for faster speed"
    echo "  --hyperparams 'param1=value1,param2=value2' - Override hyperparameters for this training phase"
    echo "  --drive-ids-file FILE_PATH     - Path to JSON file with Google Drive file IDs for data"
    echo "  --use-recommendations          - Automatically use hyperparameter recommendations from previous phase"
    echo ""
    echo "COMMANDS:"
    echo "  init STEPS [--verbose] [--training-mode] - Start initial training phase"
    echo "  continue NEW_PHASE PREV_PHASE STEPS [--verbose] [--training-mode] - Continue training from previous phase"
    echo "  evaluate PHASE                 - Evaluate a model from a specific phase"
    echo "  help                           - Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  # Complete 10M steps in phases:"
    echo "  $0 init 500000 --training-mode # Phase 1: 500K with performance optimizations"
    echo "  $0 continue 2 1 500000 --training-mode --use-recommendations # Phase 2: +500K (total: 1M) with recommendations"
    echo "  $0 continue 3 2 1000000 --training-mode --use-recommendations # Phase 3: +1M (total: 2M) with recommendations"
    echo "  $0 continue 4 3 1000000 --training-mode --use-recommendations # Phase 4: +1M (total: 3M) with recommendations"
    echo "  $0 continue 5 4 1000000 --training-mode --use-recommendations # Phase 5: +1M (total: 4M) with recommendations"
    echo "  $0 continue 6 5 1000000 --training-mode --use-recommendations # Phase 6: +1M (total: 5M) with recommendations"
    echo "  $0 continue 7 6 2000000 --training-mode --use-recommendations # Phase 7: +2M (total: 7M) with recommendations"
    echo "  $0 continue 8 7 2000000 --training-mode --use-recommendations # Phase 8: +2M (total: 9M) with recommendations"
    echo "  $0 continue 9 8 1000000 --training-mode --use-recommendations # Phase 9: +1M (total: 10M) with recommendations"
    echo ""
    echo "  # Using Google Drive data:"
    echo "  $0 init 500000 --drive-ids-file drive_file_ids.json # Train using data from Google Drive"
}

# Process global arguments that apply to all commands
for arg in "$@"; do
    if [[ $arg == --hyperparams=* ]]; then
        HYPERPARAMS="${arg#*=}"
    elif [[ $arg == --hyperparams ]]; then
        # Get the next argument
        for ((i=1; i<=$#; i++)); do
            if [[ "${!i}" == "--hyperparams" ]]; then
                next=$((i+1))
                if [[ $next -le $# ]]; then
                    HYPERPARAMS="${!next}"
                    break
                fi
            fi
        done
    elif [[ $arg == --drive-ids-file=* ]]; then
        DRIVE_IDS_FILE="${arg#*=}"
    elif [[ $arg == --drive-ids-file ]]; then
        # Get the next argument
        for ((i=1; i<=$#; i++)); do
            if [[ "${!i}" == "--drive-ids-file" ]]; then
                next=$((i+1))
                if [[ $next -le $# ]]; then
                    DRIVE_IDS_FILE="${!next}"
                    break
                fi
            fi
        done
    elif [[ $arg == --use-recommendations ]]; then
        USE_RECOMMENDATIONS=true
    fi
done

# If a drive IDs file is specified, check that it exists
if [ -n "$DRIVE_IDS_FILE" ]; then
    if [ ! -f "$DRIVE_IDS_FILE" ]; then
        echo "❌ ERROR: Drive IDs file not found: $DRIVE_IDS_FILE"
        exit 1
    fi
    echo "Using Google Drive integration with file IDs from: $DRIVE_IDS_FILE"
fi

# Function to run initial training
run_initial_training() {
    local steps=$1
    local verbose=$2
    local training_mode=$3
    
    echo "===================================================================="
    echo "  STARTING INITIAL TRAINING (PHASE 1) WITH $steps STEPS"
    if [ "$training_mode" = true ]; then
        echo "  WITH TRAINING OPTIMIZATIONS ENABLED"
    fi
    if [ "$verbose" = true ]; then
        echo "  WITH VERBOSE LOGGING ENABLED"
    fi
    if [ -n "$DRIVE_IDS_FILE" ]; then
        echo "  USING GOOGLE DRIVE DATA FROM: $DRIVE_IDS_FILE"
    fi
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
    
    # Build the command
    COMMAND="python main_opt.py --training-steps $steps --model-dir \"$PHASE_DIR\""
    
    # Add verbose flag if requested
    if [ "$verbose" = true ]; then
        COMMAND="$COMMAND --verbose"
    fi
    
    # Add training mode flag if requested
    if [ "$training_mode" = true ]; then
        COMMAND="$COMMAND --training-mode"
    fi
    
    # Add hyperparams flag if requested
    if [ -n "$HYPERPARAMS" ]; then
        COMMAND="$COMMAND --hyperparams \"$HYPERPARAMS\""
    fi
    
    # Add Google Drive integration if specified
    if [ -n "$DRIVE_IDS_FILE" ]; then
        COMMAND="$COMMAND --drive-ids-file \"$DRIVE_IDS_FILE\""
    fi
    
    # Actual training command
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "Output will be logged to $LOG_BASE_DIR/phase1/training.log"
        eval "$COMMAND 2>&1 | tee \"$LOG_BASE_DIR/phase1/training.log\""
    else
        echo "Showing live progress (not logging to file)..."
        eval "$COMMAND"
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
}

# Function to continue training
continue_training() {
    local new_phase=$1
    local prev_phase=$2
    local steps=$3
    local verbose=$4
    local training_mode=$5
    
    echo "===================================================================="
    echo "  CONTINUING TRAINING FROM PHASE $prev_phase TO PHASE $new_phase"
    echo "  ADDING $steps ADDITIONAL STEPS"
    if [ "$training_mode" = true ]; then
        echo "  WITH TRAINING OPTIMIZATIONS ENABLED"
    fi
    if [ "$verbose" = true ]; then
        echo "  WITH VERBOSE LOGGING ENABLED"
    fi
    if [ "$USE_RECOMMENDATIONS" = true ]; then
        echo "  USING HYPERPARAMETER RECOMMENDATIONS FROM PREVIOUS PHASE"
    fi
    if [ -n "$DRIVE_IDS_FILE" ]; then
        echo "  USING GOOGLE DRIVE DATA FROM: $DRIVE_IDS_FILE"
    fi
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
    
    # Calculate global training progress for 10M steps
    total_planned_steps=10000000  # Total expected training across all phases (10M)
    
    # Check if recommendations exist from previous phase
    RECOMMENDATIONS_FILE="$PREV_PHASE_DIR/phase${new_phase}_recommendations.json"
    if [ -f "$RECOMMENDATIONS_FILE" ] && [ -z "$HYPERPARAMS" ] && [ "$USE_RECOMMENDATIONS" = false ]; then
        echo "Found parameter recommendations from previous phase"
        # Extract recommendations using jq (make sure it's installed)
        if command -v jq &> /dev/null; then
            RECOMMENDED_LR=$(jq -r '.learning_rate' "$RECOMMENDATIONS_FILE")
            RECOMMENDED_ENT=$(jq -r '.ent_coef' "$RECOMMENDATIONS_FILE")
            
            echo "Recommended learning_rate: $RECOMMENDED_LR"
            echo "Recommended ent_coef: $RECOMMENDED_ENT"
            
            # Ask if user wants to use recommendations
            read -p "Use recommended parameters? (y/n): " use_recommendations
            if [[ $use_recommendations == "y" ]]; then
                HYPERPARAMS="learning_rate=$RECOMMENDED_LR,ent_coef=$RECOMMENDED_ENT"
                echo "Using recommended parameters: $HYPERPARAMS"
            fi
        else
            echo "jq not found. Install jq for automatic parameter recommendations."
        fi
    fi
    
    # Run continued training
    echo "Continuing training from phase $prev_phase with $steps additional steps..."
    echo "Loading model from: $PREV_MODEL_PATH"
    if [ -n "$PREV_ENV_PATH" ]; then
        echo "Loading environment from: $PREV_ENV_PATH"
    fi
    echo "Saving new model to: $NEW_PHASE_DIR"
    
    # Build the base command
    if [ -n "$PREV_ENV_PATH" ]; then
        COMMAND="python main_opt.py --continue-training --model-path \"$PREV_MODEL_PATH\" --env-path \"$PREV_ENV_PATH\" --additional-steps $steps --model-dir \"$NEW_PHASE_DIR\""
    else
        COMMAND="python main_opt.py --continue-training --model-path \"$PREV_MODEL_PATH\" --additional-steps $steps --model-dir \"$NEW_PHASE_DIR\""
    fi
    
    # Add verbose flag if requested
    if [ "$verbose" = true ]; then
        COMMAND="$COMMAND --verbose"
    fi
    
    # Add training mode flag if requested
    if [ "$training_mode" = true ]; then
        COMMAND="$COMMAND --training-mode"
    fi
    
    # Add hyperparams flag if requested
    if [ -n "$HYPERPARAMS" ]; then
        COMMAND="$COMMAND --hyperparams \"$HYPERPARAMS\""
    fi
    
    # Add use-recommendations flag if requested
    if [ "$USE_RECOMMENDATIONS" = true ]; then
        COMMAND="$COMMAND --use-recommendations"
    fi
    
    # Add Google Drive integration if specified
    if [ -n "$DRIVE_IDS_FILE" ]; then
        COMMAND="$COMMAND --drive-ids-file \"$DRIVE_IDS_FILE\""
    fi
    
    # Actual training command
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "Output will be logged to $LOG_BASE_DIR/phase$new_phase/training.log"
        eval "$COMMAND 2>&1 | tee \"$LOG_BASE_DIR/phase$new_phase/training.log\""
    else
        echo "Showing live progress (not logging to file)..."
        eval "$COMMAND"
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
    
    # Show next suggested phase based on 10M steps training schedule
    next_phase=$((new_phase + 1))
    
    # Updated for 10M step training with 9 phases
    if [ $new_phase -eq 1 ]; then
        next_steps=500000    # Phase 2: 500K
    elif [ $new_phase -eq 2 ]; then
        next_steps=1000000   # Phase 3: 1M
    elif [ $new_phase -eq 3 ]; then
        next_steps=1000000   # Phase 4: 1M
    elif [ $new_phase -eq 4 ]; then
        next_steps=1000000   # Phase 5: 1M
    elif [ $new_phase -eq 5 ]; then
        next_steps=1000000   # Phase 6: 1M
    elif [ $new_phase -eq 6 ]; then
        next_steps=2000000   # Phase 7: 2M
    elif [ $new_phase -eq 7 ]; then
        next_steps=2000000   # Phase 8: 2M
    elif [ $new_phase -eq 8 ]; then
        next_steps=1000000   # Phase 9: 1M
    else
        next_steps=500000    # Default if somehow beyond phase 9
    fi
    
    echo ""
    echo "📋 NEXT SUGGESTED COMMAND:"
    suggestion_cmd="./scripts/manual_incremental.sh continue $next_phase $new_phase $next_steps"
    
    if [ "$training_mode" = true ]; then
        suggestion_cmd="$suggestion_cmd --training-mode"
    fi
    
    if [ "$USE_RECOMMENDATIONS" = true ]; then
        suggestion_cmd="$suggestion_cmd --use-recommendations"
    fi
    
    if [ -n "$DRIVE_IDS_FILE" ]; then
        suggestion_cmd="$suggestion_cmd --drive-ids-file \"$DRIVE_IDS_FILE\""
    fi
    
    echo "$suggestion_cmd"
    
    if [ "$verbose" = true ]; then
        echo "To enable verbose logging, add the --verbose flag."
    fi
    echo ""
}

# Main command processing
case "$1" in
    "init")
        VERBOSE=false
        TRAINING_MODE=false
        STEPS=""
        
        if [ -z "$2" ]; then
            echo "❌ ERROR: Please specify the number of steps for initial training"
            show_help
            exit 1
        fi
        
        STEPS="$2"
        
        # Check for flags
        for arg in "$@"; do
            if [ "$arg" = "--verbose" ]; then
                VERBOSE=true
            fi
            if [ "$arg" = "--training-mode" ]; then
                TRAINING_MODE=true
            fi
        done
        
        run_initial_training $STEPS $VERBOSE $TRAINING_MODE
        ;;
    
    "continue")
        VERBOSE=false
        TRAINING_MODE=false
        
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
            echo "❌ ERROR: Missing arguments for continue command"
            show_help
            exit 1
        fi
        
        # Check for flags
        for arg in "$@"; do
            if [ "$arg" = "--verbose" ]; then
                VERBOSE=true
            fi
            if [ "$arg" = "--training-mode" ]; then
                TRAINING_MODE=true
            fi
        done
        
        continue_training $2 $3 $4 $VERBOSE $TRAINING_MODE
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