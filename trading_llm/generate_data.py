#!/usr/bin/env python3
"""
Generate training data for the trading LLM with improved action processing
"""

import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from trading_llm.dataset import TradingDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate trading LLM training data')
    parser.add_argument('--rl-model', type=str, required=True,
                        help='Path to trained RL model')
    parser.add_argument('--market-data', type=str, required=True,
                        help='Path to market data file (parquet or csv)')
    parser.add_argument('--output-dir', type=str, default='data/trading_dataset',
                        help='Output directory for generated data')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to generate per symbol')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Lookback window size for observations')
    parser.add_argument('--min-action-threshold', type=float, default=0.1,
                        help='Minimum action strength threshold to consider')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--env-config', type=str, default=None,
                        help='Optional path to environment config')
    parser.add_argument('--template-path', type=str, default=None,
                        help='Optional path to explanation templates')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Initializing dataset generator...")
    
    # Initialize dataset generator
    generator = TradingDatasetGenerator(
        rl_model_path=args.rl_model,
        market_data_path=args.market_data,
        output_dir=args.output_dir,
        lookback_window=args.window_size,
        samples_per_symbol=args.num_samples,
        template_path=args.template_path
    )
    
    logger.info(f"Generating dataset with {args.num_samples} samples per symbol...")
    
    # Generate the dataset
    train_df, val_df = generator.generate_dataset(
        policy_model_path=args.rl_model,
        env_config_path=args.env_config,
        window_size=args.window_size,
        num_samples=args.num_samples,
        val_split=args.val_split,
        min_action_threshold=args.min_action_threshold
    )
    
    # Print dataset statistics
    logger.info(f"Generated dataset statistics:")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Count actions by type
    if len(train_df) > 0 and 'action' in train_df.columns:
        action_types = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for action in train_df['action']:
            # Extract direction from the action list [direction, raw_value, leverage]
            if isinstance(action, list) and len(action) >= 3:
                direction = action[0]
                if direction > 0:
                    action_types['buy'] += 1
                elif direction < 0:
                    action_types['sell'] += 1
                else:
                    action_types['hold'] += 1
        
        logger.info(f"Action distribution:")
        for action_type, count in action_types.items():
            logger.info(f"  - {action_type.capitalize()}: {count} ({count/len(train_df)*100:.1f}%)")
    
    logger.info(f"Data saved to {args.output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 