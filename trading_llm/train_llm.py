#!/usr/bin/env python3
"""
Training and inference utilities for the Trading LLM system.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trading_llm.dataset import TradingDatasetGenerator, create_dataloaders
from trading_llm.training import TradingTrainer
from trading_llm.inference import RLLMExplainer, extract_trading_signals, MarketCommentaryGenerator
from trading_llm.model import TradingLLM
from trading_llm.chatbot import MarketChatbot
from trading_llm.utils import setup_logging

logger = logging.getLogger(__name__)

def load_multi_asset_data(for_training=False):
    """
    Load and combine market data from BTC, ETH, and SOL into a single multi-asset dataset.
    If original files are not found, generate synthetic data for testing.
    
    Args:
        for_training: If True, load 15-year data; otherwise load 100-day data
        
    Returns:
        Path to the parquet file with combined market data for all assets
    """
    logger.info("Loading and combining multi-asset data...")
    
    if for_training:
        # Use 15-year data for training
        logger.info("Using 15-year data for training")
        btc_path = "data/market_data/binance_BTCUSDT_5m.parquet"
        eth_path = "data/market_data/binance_ETHUSDT_5m.parquet"
        sol_path = "data/market_data/binance_SOLUSDT_5m.parquet"
    else:
        # Use 100-day data for inference/other tasks
        logger.info("Using 100-day data for inference")
        btc_path = "data/BTC_USDT_100days_5m.parquet"
        eth_path = "data/ETH_USDT_100days_5m.parquet"
        sol_path = "data/SOL_USDT_100days_5m.parquet"
    
    # Check if files exist, generate synthetic data if not
    missing_files = []
    for path, symbol in [(btc_path, 'BTC'), (eth_path, 'ETH'), (sol_path, 'SOL')]:
        if not os.path.exists(path):
            missing_files.append((path, symbol))
    
    if missing_files:
        logger.warning(f"Market data files not found. Generating synthetic data for testing.")
        # Generate synthetic data for missing files
        for path, symbol in missing_files:
            _generate_synthetic_data(path, symbol)
    
    try:
        # Load individual files
        btc = pd.read_parquet(btc_path)
        btc['symbol'] = 'BTC'
        eth = pd.read_parquet(eth_path)
        eth['symbol'] = 'ETH'
        sol = pd.read_parquet(sol_path)
        sol['symbol'] = 'SOL'
        
        # Ensure all have the same structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for df, asset in [(btc, 'BTC'), (eth, 'ETH'), (sol, 'SOL')]:
            # Convert all column names to lowercase
            df.columns = df.columns.str.lower()
            # Check for required columns
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {asset} data: {missing}")
        
        # Determine common columns (excluding 'symbol' which we just added)
        common_cols = list(set(btc.columns) & set(eth.columns) & set(sol.columns))
        if 'symbol' in common_cols:
            common_cols.remove('symbol')  # Remove symbol since we'll add it back
        common_cols = ['symbol'] + common_cols  # Add symbol as the first column
        
        # Combine datasets using common columns
        combined = pd.concat([
            btc[common_cols], 
            eth[common_cols], 
            sol[common_cols]
        ])
        
        # Sort by timestamp if available
        if 'timestamp' in combined.columns:
            combined.sort_values('timestamp', inplace=True)
        
        # Save combined dataset to a temporary file
        combined_path = 'data/market_data/multi_asset_data.parquet'
        combined.to_parquet(combined_path)
        
        logger.info(f"Created combined dataset with {len(combined)} rows from BTC, ETH, and SOL data")
        logger.info(f"Saved to {combined_path}")
        
        return combined_path
    
    except Exception as e:
        logger.error(f"Error combining market data: {str(e)}")
        raise

def _generate_synthetic_data(path: str, symbol: str):
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        path: Path to save the synthetic data
        symbol: Symbol name (BTC, ETH, SOL)
    """
    logger.info(f"Generating synthetic data for {symbol} at {path}")
    
    # Starting price based on symbol
    base_prices = {
        'BTC': 40000,
        'ETH': 2000,
        'SOL': 100
    }
    base_price = base_prices.get(symbol, 1000)
    
    # Generate timestamps - last 30 days, 5 minute intervals
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=30)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5T')
    
    # Initialize price series with some randomness
    np.random.seed(42 + sum(ord(c) for c in symbol))  # Different seed for each symbol
    price_changes = np.random.normal(0, 0.002, len(timestamps))
    price_series = np.cumprod(1 + price_changes) * base_price
    
    # Generate OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps):
        # Reference price for this candle
        ref_price = price_series[i]
        
        # Generate candle data with some randomness
        high_pct = np.random.uniform(0, 0.005)
        low_pct = np.random.uniform(0, 0.005)
        open_pct = np.random.uniform(-0.003, 0.003)
        
        high = ref_price * (1 + high_pct)
        low = ref_price * (1 - low_pct)
        open_price = ref_price * (1 + open_pct)
        close = ref_price
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume with some randomness
        volume = np.random.lognormal(10, 1) * (base_price / 1000)
        
        # Add row
        data.append({
            'timestamp': timestamp.value // 10**9,  # Convert to Unix timestamp
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    
    logger.info(f"Generated synthetic data for {symbol} with {len(df)} rows")

def parse_args():
    """Parse command line arguments."""
    DEFAULT_LLM_MODEL_PATH = "models/llm/llm_model.pt"
    DEFAULT_BASE_MODEL = "models/llm/base_model.pt"
    parser = argparse.ArgumentParser(description="Trading LLM Training and Inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Common arguments
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    
    # Dataset generation command
    generate_parser = subparsers.add_parser("generate", help="Generate training dataset")
    generate_parser.add_argument("--rl-model", required=True, help="Path to trained RL model")
    # Use combined dataset by default
    generate_parser.add_argument("--market-data", default="auto", help="Path to market data file or 'auto' to use combined multi-asset data")
    generate_parser.add_argument("--output-dir", required=True, help="Output directory for dataset")
    generate_parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    generate_parser.add_argument("--split-ratio", type=float, default=0.9, help="Train/eval split ratio")
    generate_parser.add_argument("--window-size", type=int, default=100, help="Window size for market data")
    generate_parser.add_argument("--templates", help="Path to custom templates file")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train LLM model")
    train_parser.add_argument("--base-model", required=True, help="Base model to fine-tune")
    train_parser.add_argument("--train-data", required=True, help="Path to training data")
    train_parser.add_argument("--eval-data", help="Path to evaluation data")
    train_parser.add_argument("--output-dir", required=True, help="Output directory for trained model")
    train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--max-steps", type=int, help="Maximum number of training steps")
    train_parser.add_argument("--lora-r", type=int, default=8, help="LoRA r parameter")
    train_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter")
    train_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    train_parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    train_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    train_parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    train_parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    train_parser.add_argument("--wandb-project", default="trading_llm", help="W&B project name")
    train_parser.add_argument("--wandb-entity", help="W&B entity")
    train_parser.add_argument("--early-stopping", type=int, help="Early stopping patience")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--rl-model", required=True, help="Path to RL model")
    infer_parser.add_argument("--llm-model", required=True, help="Path to LLM model")
    infer_parser.add_argument("--base-model", help="Base model for LLM (if needed)")
    infer_parser.add_argument("--market-data", default="auto", help="Path to market data file or 'auto' to use combined multi-asset data")
    infer_parser.add_argument("--input-file", help="Input file with observations")
    infer_parser.add_argument("--output-file", help="Output file for explanations")
    infer_parser.add_argument("--assets", "--asset-names", nargs="+", help="Specific asset names to explain")
    
    # Market commentary command
    commentary_parser = subparsers.add_parser("commentary", help="Generate market commentary")
    commentary_parser.add_argument("--llm-model", required=True, help="Path to LLM model")
    commentary_parser.add_argument("--base-model", help="Base model for LLM (if needed)")
    commentary_parser.add_argument("--market-data", default="auto", help="Path to market data file or 'auto' to use combined multi-asset data")
    commentary_parser.add_argument("--symbol", help="Symbol to generate commentary for")
    commentary_parser.add_argument("--symbols-file", help="File with list of symbols")
    commentary_parser.add_argument("--lookback-days", type=int, default=30, help="Number of days to look back")
    commentary_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    commentary_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    commentary_parser.add_argument("--output-file", help="Output file for commentary")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chatbot")
    chat_parser.add_argument("--llm-model", required=True, help="Path to LLM model")
    chat_parser.add_argument("--base-model", help="Base model for LLM (if needed)")
    chat_parser.add_argument("--rl-model", help="Path to RL model (optional)")
    chat_parser.add_argument("--market-data", default="auto", help="Path to market data file or 'auto' to use combined multi-asset data")
    chat_parser.add_argument("--max-history", type=int, default=5, help="Maximum conversation history")
    chat_parser.add_argument("--system-prompt", help="System prompt for the chatbot")
    chat_parser.add_argument("--asset-name", help="Specific asset name to focus on")
    
    # Explain subcommand
    explain_parser = subparsers.add_parser('explain', help='Generate natural language explanations for trading decisions')
    explain_parser.add_argument('--market-data', '-m', required=True, help='Path to market data file or directory (accepts .csv, .parquet)')
    explain_parser.add_argument('--llm-model', '-l', default=DEFAULT_LLM_MODEL_PATH, help='Path to LLM model or model name')
    explain_parser.add_argument('--base-model', '-b', default=DEFAULT_BASE_MODEL, help='Base model name (if loading LoRA adapter)')
    explain_parser.add_argument('--output-file', '-o', help='Output file for generated explanations')
    explain_parser.add_argument('--use-fallback', action='store_true', help='Use fallback explanations if LLM fails')
    explain_parser.add_argument('--no-fallback', dest='use_fallback', action='store_false', help='Do not use fallback explanations')
    explain_parser.add_argument('--asset-names', nargs='+', help='Specific asset names to explain')
    explain_parser.set_defaults(use_fallback=True)
    
    return parser.parse_args()

def load_market_data(path: str, for_training=False):
    """Load market data from CSV or Parquet file."""
    # If auto mode is specified, generate combined data
    if path == "auto":
        path = load_multi_asset_data(for_training=for_training)
    
    extension = os.path.splitext(path)[1].lower()
    if extension == '.csv':
        data = pd.read_csv(path)
    elif extension in ['.parquet', '.pq']:
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported market data format: {extension}")
    
    # Convert column names to lowercase
    data.columns = data.columns.str.lower()
    return data

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Set specific modules to DEBUG level for more detailed output
    logging.getLogger("trading_llm.dataset").setLevel(logging.DEBUG)
    
    if args.command == "generate":
        logger.info("Generating training dataset")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data(for_training=True)
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        generator = TradingDatasetGenerator(
            rl_model_path=args.rl_model,
            market_data_path=market_data_path,
            output_dir=args.output_dir,
            template_path=args.templates
        )
        
        generator.generate_dataset(
            num_samples=args.num_samples, 
            window_size=args.window_size,
            val_split=1.0 - args.split_ratio,
            min_action_threshold=0.1  # Default threshold for filtering neutral actions
        )
        
    elif args.command == "train":
        logger.info("Training LLM model")
        
        # Load the training and evaluation datasets
        train_path = args.train_data
        eval_path = args.eval_data
        
        if not os.path.exists(train_path):
            logger.error(f"Training data file not found: {train_path}")
            sys.exit(1)
            
        if eval_path and not os.path.exists(eval_path):
            logger.warning(f"Evaluation data file not found: {eval_path}")
            eval_path = None
        
        # Initialize the model
        model = TradingLLM(
            model_name=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            load_in_4bit=args.bf16,  # Use BF16 for 4-bit quantization
            load_in_8bit=not args.bf16 and args.fp16,  # Use 8-bit only if not using BF16
        )
        
        # Create data loaders
        train_dataloader, val_dataloader = create_dataloaders(
            train_data_path=train_path,
            val_data_path=eval_path,
            tokenizer=model.tokenizer,
            batch_size=args.batch_size,
            max_length=512  # Default max sequence length
        )
        
        # Initialize trainer
        trainer = TradingTrainer(
            model=model,
            tokenizer=model.tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16
        )
        
        # Start training
        metrics = trainer.train()
        
        # Log results
        logger.info("Training completed successfully")
        logger.info(f"Model saved to {args.output_dir}")
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value}")
        
        # Save the final model
        model.save_model(args.output_dir)
        
    elif args.command == "infer":
        logger.info("Running inference")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data(for_training=False)
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        # Load the RL model if path is provided
        rl_model = None
        if hasattr(args, 'rl_model') and args.rl_model:
            try:
                from stable_baselines3 import PPO
                logger.info(f"Loading RL model from {args.rl_model}")
                rl_model = PPO.load(args.rl_model)
                logger.info("RL model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RL model from {args.rl_model}: {str(e)}")
                logger.warning("Falling back to synthetic actions for trading signals")
        
        # Load the LLM model using the appropriate method
        llm_model = None
        try:
            from trading_llm.model import TradingLLM
            logger.info(f"Loading LLM model from {args.llm_model}")
            
            if args.base_model:
                # Use load_from_pretrained for LoRA adapter
                llm_model = TradingLLM.load_from_pretrained(
                    model_dir=args.llm_model,
                    base_model=args.base_model
                )
            else:
                # Use direct initialization for complete model
                llm_model = TradingLLM(model_name=args.llm_model)
                
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            logger.warning("Using base LLM model without fine-tuning")
            llm_model = TradingLLM(model_name=args.base_model)
        
        explainer = RLLMExplainer(
            model=llm_model,
            asset_names=args.assets,
            rl_model=rl_model
        )
        
        # Load market data if provided
        raw_ohlcv = None
        if market_data_path != "auto":
            raw_ohlcv = load_market_data(market_data_path)
        
        # Load observations from file if provided
        if args.input_file:
            with open(args.input_file, 'r') as f:
                observations = json.load(f)
                
            # Run batch inference
            results = explainer.batch_explain(
                observations=observations,
                raw_ohlcvs=[raw_ohlcv] * len(observations) if raw_ohlcv is not None else None
            )
            
            # Save results if output file provided
            if args.output_file:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(args.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                # Print first result
                print(f"Action: {results[0]['action']}")
                print(f"Explanation: {results[0]['explanation']}")
        else:
            # Generate explanations from market data
            explanations = explainer.explain_trading_decisions(
                market_data_path,
                use_fallback=args.use_fallback if hasattr(args, 'use_fallback') else True,
                asset_names=args.assets if hasattr(args, 'assets') else None
            )
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    for expl in explanations:
                        f.write(f"{expl}\n\n")
            else:
                for expl in explanations:
                    print(expl)
                    print("---")
                
    elif args.command == "commentary":
        logger.info("Generating market commentary")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data(for_training=False)
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        # Initialize commentary generator
        generator = MarketCommentaryGenerator(
            llm_model_path=args.llm_model,
            llm_base_model=args.base_model
        )
        
        # Load market data
        market_data = {}
        extension = os.path.splitext(market_data_path)[1].lower()
        if extension == '.csv':
            # Single market data file
            market_data = {args.symbol or "default": pd.read_csv(market_data_path).iloc[-args.lookback_days:]}
        elif extension in ['.parquet', '.pq']:
            # Single market data file
            market_data = {args.symbol or "default": pd.read_parquet(market_data_path).iloc[-args.lookback_days:]}
        else:
            logger.error(f"Unsupported market data format: {extension}")
            sys.exit(1)
        
        # Convert column names to lowercase
        for symbol, data in market_data.items():
            market_data[symbol].columns = data.columns.str.lower()
        
        # Generate commentary
        if args.symbol or len(market_data) == 1:
            # Single market commentary
            symbol = args.symbol or list(market_data.keys())[0]
            commentary = generator.generate_daily_commentary(
                symbol=symbol,
                ohlcv_data=market_data[symbol],
                lookback_days=args.lookback_days,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print commentary
            print(f"Commentary for {symbol}:")
            print(commentary)
            
            # Save commentary if output file provided
            if args.output_file:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(args.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                
                with open(args.output_file, 'w') as f:
                    f.write(f"Commentary for {symbol}:\n\n")
                    f.write(commentary)
        elif args.symbols_file:
            # Multi-market summary
            with open(args.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            # Generate summary
            summary = generator.generate_market_summary(
                symbols=symbols,
                ohlcv_dict=market_data,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print summary
            print("Market Summary:")
            print(summary)
            
            # Save summary if output file provided
            if args.output_file:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(args.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                
                with open(args.output_file, 'w') as f:
                    f.write("Market Summary:\n\n")
                    f.write(summary)
        else:
            logger.error("Either --symbol or --symbols-file must be provided")
            sys.exit(1)
            
    elif args.command == "chat":
        logger.info("Starting interactive market chatbot")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data(for_training=False)
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        # Determine which asset names to use
        default_asset_names = ["BTC", "ETH", "SOL"]
        used_asset_names = None
        
        # Get asset names from command line argument if provided
        if hasattr(args, 'asset_name') and args.asset_name:
            if isinstance(args.asset_name, list):
                used_asset_names = args.asset_name
            else:
                used_asset_names = [args.asset_name]
            logger.info(f"Using asset names from command line: {used_asset_names}")
        elif hasattr(args, 'assets') and args.assets:
            used_asset_names = args.assets
            logger.info(f"Using asset names from command line: {used_asset_names}")
        else:
            # Fall back to default asset names
            used_asset_names = default_asset_names
            logger.info(f"No asset names provided, using defaults: {used_asset_names}")
        
        # Load the LLM model using the appropriate method
        llm_model = None
        try:
            from trading_llm.model import TradingLLM
            logger.info(f"Loading LLM model from {args.llm_model}")
            
            if args.base_model:
                # Use load_from_pretrained for LoRA adapter
                llm_model = TradingLLM.load_from_pretrained(
                    model_dir=args.llm_model,
                    base_model=args.base_model
                )
            else:
                # Use direct initialization for complete model
                llm_model = TradingLLM(model_name=args.llm_model)
                
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            logger.warning("Using base LLM model without fine-tuning")
            from trading_llm.model import TradingLLM
            llm_model = TradingLLM(model_name=args.base_model or "meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Load the chatbot with the determined asset names
        chatbot = MarketChatbot(
            llm_model=llm_model,
            max_history=args.max_history,
            system_prompt=args.system_prompt,
            asset_names=used_asset_names
        )
        
        # Load market data if provided
        if market_data_path != "auto":
            market_data = load_market_data(market_data_path)
            chatbot.update_market_data(market_data)
            print(f"Loaded market data with {len(market_data)} records")
        
        # Load RL model and get trading signals if provided
        if args.rl_model and market_data_path != "auto":
            try:
                from stable_baselines3 import PPO
                from gymnasium import spaces
                
                logger.info(f"Loading RL model from {args.rl_model}")
                model = PPO.load(args.rl_model)
                
                # Log RL model observation space details for debugging
                if hasattr(model, 'observation_space'):
                    if isinstance(model.observation_space, spaces.Dict):
                        logger.info(f"RL model uses Dict observation space: {model.observation_space}")
                    else:
                        logger.info(f"RL model observation space shape: {model.observation_space.shape}")
                        logger.info(f"RL model observation space: {model.observation_space}")
                
                # Log market data info for debugging
                logger.info(f"Market data shape: {market_data.shape}")
                logger.info(f"Market data columns: {market_data.columns.tolist()}")
                
                # Verify required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in market_data.columns.str.lower()]
                if missing_cols:
                    logger.error(f"Missing required columns: {missing_cols}")
                    # Convert columns to lowercase if needed
                    market_data.columns = market_data.columns.str.lower()
                    
                # Extract trading signals with better error handling
                # Pass the asset_names to ensure proper multi-asset handling
                signals = extract_trading_signals(model, market_data, asset_names=used_asset_names)
                chatbot.update_trading_signals(signals)
                print("Loaded trading signals from RL model")
                
                # Calculate basic performance metrics
                returns = np.diff(market_data['close'].values) / market_data['close'].values[:-1]
                
                # Create positions array from signals (handling the new nested structure)
                # First get list of all timestamp keys
                timestamps = list(signals.keys())
                if timestamps:
                    latest_timestamp = timestamps[-1]  # Use last timestamp for performance metrics
                    positions_dict = {}
                    
                    # Extract positions for each asset
                    for asset_name in used_asset_names:
                        if asset_name in signals[latest_timestamp]:
                            positions_dict[asset_name] = signals[latest_timestamp][asset_name]['position']
                        else:
                            positions_dict[asset_name] = 0  # Default if no signal

                    # Calculate performance for each asset
                    portfolio_performance = {}
                    primary_asset = used_asset_names[0] if used_asset_names else "Asset"
                    
                    # Use primary asset's position for basic metrics
                    positions = np.array([signals[ts][primary_asset]['position'] 
                                       for ts in timestamps 
                                       if primary_asset in signals[ts]])
                    
                    # Ensure arrays have matching lengths before calculations
                    if len(returns) != len(positions[:-1]):
                        logger.warning(f"Length mismatch: returns ({len(returns)}) vs positions ({len(positions[:-1])})")
                        
                        # Adjust the shorter array to match the longer one's length
                        if len(returns) > len(positions[:-1]):
                            # Truncate returns to match positions length
                            logger.info(f"Truncating returns array from {len(returns)} to {len(positions[:-1])}")
                            returns = returns[-len(positions[:-1]):]
                        else:
                            # Truncate positions to match returns length
                            logger.info(f"Truncating positions array from {len(positions[:-1])} to {len(returns)}")
                            positions = positions[:len(returns)+1]
                    
                    # Calculate overall portfolio performance
                    portfolio_performance = {
                        "total_return": f"{np.sum(returns * positions[:-1]):.4f}",
                        "sharpe_ratio": f"{np.mean(returns) / np.std(returns):.4f}",
                        "win_rate": f"{np.mean([signals[ts][primary_asset]['confidence'] > 0.5 for ts in timestamps if primary_asset in signals[ts]]):.4f}"
                    }
                    
                    chatbot.update_portfolio_performance(portfolio_performance)
                else:
                    logger.warning("No signals available for performance calculation")
                
            except Exception as e:
                logger.error(f"Error loading RL model: {str(e)}")
                logger.error(traceback.format_exc())  # Print full traceback for debugging
                print(f"Failed to load RL model: {str(e)}")
        
        # Set initial asset focus if specified
        if hasattr(args, 'asset_name') and args.asset_name and isinstance(args.asset_name, str):
            if chatbot.set_current_asset(args.asset_name):
                print(f"Initially focusing on asset: {args.asset_name}")
        
        print("\nTrading Assistant Chatbot")
        print("Type 'exit', 'quit', or 'q' to end the conversation")
        print("Type 'reset' to reset the conversation history")
        print("Available assets: " + ", ".join(used_asset_names))
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ('exit', 'quit', 'q'):
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == 'reset':
                    chatbot.reset_conversation()
                    print("Conversation history has been reset.")
                    continue
                
                response = chatbot.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                traceback.print_exc()  # Print full traceback for debugging
    
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 