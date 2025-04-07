#!/usr/bin/env python
"""
Incremental Training Manager for Trading Model

This script automates the process of incrementally training a trading model through
multiple phases, starting with smaller step increments and gradually increasing.
It manages checkpoints, evaluations, and progress tracking.
"""

import argparse
import subprocess
import os
import time
import json
import logging
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/incremental_training.log')
    ]
)
logger = logging.getLogger('incremental_training')

def parse_args():
    parser = argparse.ArgumentParser(description='Manage incremental training of trading model')
    parser.add_argument('--total-steps', type=int, default=1000000,
                        help='Total number of training steps across all phases')
    parser.add_argument('--initial-steps', type=int, default=100000,
                        help='Number of steps for first training phase')
    parser.add_argument('--model-dir', type=str, default='models/incremental',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/incremental',
                        help='Directory to save logs')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation after each phase')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='trading_incremental',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity name')
    parser.add_argument('--drive-ids-file', type=str, default=None,
                        help='Path to JSON file with Google Drive file IDs for market data')
    return parser.parse_args()

def ensure_dirs(dirs):
    """Ensure all directories in the list exist"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

def run_command(cmd, description):
    """Run a shell command and log the output"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                logger.info(f"[{description}] {line}")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"{description} failed with return code {return_code}")
            return False
        
        logger.info(f"{description} completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error executing {description}: {str(e)}")
        return False

def calculate_next_increment(phase, current_steps):
    """Calculate the step increment for the next phase"""
    if phase <= 2:
        return 100000
    elif phase <= 3:
        return 200000
    elif phase <= 4:
        return 300000
    else:
        return 400000

def run_training_phase(phase, total_steps_so_far, next_increment, args, is_first_phase=False):
    """Run a single training phase"""
    phase_start_time = time.time()
    
    # Determine model paths for this phase
    if is_first_phase:
        model_dir = os.path.join(args.model_dir, f"phase{phase}")
        os.makedirs(model_dir, exist_ok=True)
        cmd = [
            "python", "main_opt.py",
            "--training-steps", str(next_increment),
            "--model-dir", model_dir
        ]
    else:
        prev_model_dir = os.path.join(args.model_dir, f"phase{phase-1}")
        model_dir = os.path.join(args.model_dir, f"phase{phase}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if we have a final_model file
        prev_model_path = os.path.join(prev_model_dir, "final_model")
        if not os.path.exists(prev_model_path):
            prev_model_path = os.path.join(prev_model_dir, "model")
        
        cmd = [
            "python", "main_opt.py",
            "--continue-training",
            "--model-path", prev_model_path,
            "--additional-steps", str(next_increment),
            "--model-dir", model_dir
        ]
    
    # Add WandB parameters if enabled
    if args.use_wandb:
        cmd.extend([
            "--wandb",
            "--wandb-project", args.wandb_project
        ])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
    
    # Add Google Drive integration if specified
    if args.drive_ids_file:
        if os.path.exists(args.drive_ids_file):
            cmd.extend(["--drive-ids-file", args.drive_ids_file])
            logger.info(f"Using Google Drive data from: {args.drive_ids_file}")
        else:
            logger.warning(f"Drive IDs file not found: {args.drive_ids_file}")
    
    # Run the training phase
    success = run_command(cmd, f"Training Phase {phase}")
    
    # Calculate duration
    phase_duration = time.time() - phase_start_time
    hours, remainder = divmod(phase_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Log results
    if success:
        logger.info(f"Phase {phase} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Total steps so far: {total_steps_so_far + next_increment}")
        
        # Record phase information
        phase_info = {
            "phase": phase,
            "steps_this_phase": next_increment,
            "total_steps": total_steps_so_far + next_increment,
            "duration_seconds": phase_duration,
            "model_dir": model_dir,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save phase info to JSON
        phase_info_path = os.path.join(args.log_dir, f"phase{phase}_info.json")
        with open(phase_info_path, 'w') as f:
            json.dump(phase_info, f, indent=2)
        
        # Run evaluation
        eval_success = run_evaluation(phase, model_dir, args)
        if not eval_success:
            logger.warning(f"Evaluation after phase {phase} failed, but continuing with training")
        
        return True, total_steps_so_far + next_increment
    else:
        logger.error(f"Phase {phase} failed after {int(hours)}h {int(minutes)}m {int(seconds)}s")
        return False, total_steps_so_far

def run_evaluation(phase, model_dir, args):
    """Run evaluation after a training phase"""
    # Find model path - either 'final_model' or 'model'
    model_path = os.path.join(model_dir, "final_model")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model")
    
    # Find environment path
    env_path = os.path.join(model_dir, "vec_normalize.pkl")
    if not os.path.exists(env_path):
        env_path = os.path.join(model_dir, "env.pkl")
    
    eval_output_dir = os.path.join(args.log_dir, f"eval_phase{phase}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    cmd = [
        "python", "scripts/evaluate_model.py",
        "--model-path", model_path,
        "--env-path", env_path,
        "--episodes", str(args.eval_episodes),
        "--phase", str(phase),
        "--output-dir", eval_output_dir
    ]
    
    if args.use_wandb:
        cmd.append("--log-wandb")
    
    return run_command(cmd, f"Evaluation After Phase {phase}")

def main():
    """Main function to run the incremental training process"""
    args = parse_args()
    
    # Create necessary directories
    ensure_dirs([args.model_dir, args.log_dir, 'logs'])
    
    # Initialize tracking variables
    current_phase = 1
    total_steps_so_far = 0
    training_start_time = time.time()
    
    # Log the training configuration
    logger.info("=" * 50)
    logger.info("STARTING INCREMENTAL TRAINING")
    logger.info("=" * 50)
    logger.info(f"Total target steps: {args.total_steps}")
    logger.info(f"Initial phase steps: {args.initial_steps}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info(f"Using WandB: {args.use_wandb}")
    if args.use_wandb:
        logger.info(f"WandB project: {args.wandb_project}")
        if args.wandb_entity:
            logger.info(f"WandB entity: {args.wandb_entity}")
    logger.info("=" * 50)
    
    # First phase
    next_increment = args.initial_steps
    logger.info(f"Starting Phase 1 with {next_increment} steps")
    success, total_steps_so_far = run_training_phase(
        current_phase, total_steps_so_far, next_increment, args, is_first_phase=True
    )
    
    if not success:
        logger.error("Initial training phase failed. Exiting.")
        return
    
    current_phase += 1
    
    # Subsequent phases
    while total_steps_so_far < args.total_steps:
        # Calculate next increment
        next_increment = calculate_next_increment(current_phase, total_steps_so_far)
        
        # Make sure we don't exceed the total
        if total_steps_so_far + next_increment > args.total_steps:
            next_increment = args.total_steps - total_steps_so_far
        
        logger.info(f"Starting Phase {current_phase} with {next_increment} steps")
        logger.info(f"Total steps so far: {total_steps_so_far}/{args.total_steps}")
        
        # Run this phase
        success, total_steps_so_far = run_training_phase(
            current_phase, total_steps_so_far, next_increment, args
        )
        
        if not success:
            logger.error(f"Phase {current_phase} failed. Attempting to continue from last successful phase.")
            # Could implement recovery logic here if needed
        
        current_phase += 1
        
        # Break if we've reached the target
        if total_steps_so_far >= args.total_steps:
            break
    
    # Calculate total training time
    total_duration = time.time() - training_start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("=" * 50)
    logger.info("INCREMENTAL TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total steps trained: {total_steps_so_far}")
    logger.info(f"Total phases completed: {current_phase - 1}")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Final model path: {os.path.join(args.model_dir, f'phase{current_phase-1}')}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 