#!/usr/bin/env python3
"""
Main CLI entry point for the Trading LLM system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from typing import Optional
import json
import numpy as np
from stable_baselines3 import PPO
import traceback

from trading_llm.dataset import TradingDatasetGenerator, create_dataloaders
from trading_llm.training import train_llm_model, TrainingArgs
from trading_llm.inference import RLLMExplainer, generate_market_commentary, MarketCommentaryGenerator, extract_trading_signals
from trading_llm.model import TradingLLM
from trading_llm.chatbot import load_market_chatbot
from trading_llm.utils import setup_logging

logger = logging.getLogger(__name__)

def load_multi_asset_data():
    """
    Load and combine market data from BTC, ETH, and SOL into a single multi-asset dataset.
    
    Returns:
        DataFrame with combined market data for all assets
    """
    logger.info("Loading and combining multi-asset data...")
    
    # Hardcoded paths for the three assets
    btc_path = "market_data/binance_BTCUSDT_5m.parquet"
    eth_path = "market_data/binance_ETHUSDT_5m.parquet"
    sol_path = "market_data/binance_SOLUSDT_5m.parquet"
    
    # Ensure all paths exist
    for path in [btc_path, eth_path, sol_path]:
        if not os.path.exists(path):
            logger.error(f"Market data file not found: {path}")
            raise FileNotFoundError(f"Market data file not found: {path}")
    
    try:
        # Load individual files
        btc = pd.read_parquet(btc_path)
        btc['asset'] = 'BTC'
        eth = pd.read_parquet(eth_path)
        eth['asset'] = 'ETH'
        sol = pd.read_parquet(sol_path)
        sol['asset'] = 'SOL'
        
        # Ensure all have the same structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for df, asset in [(btc, 'BTC'), (eth, 'ETH'), (sol, 'SOL')]:
            # Convert all column names to lowercase
            df.columns = df.columns.str.lower()
            # Check for required columns
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {asset} data: {missing}")
        
        # Determine common columns (excluding 'asset' which we just added)
        common_cols = list(set(btc.columns) & set(eth.columns) & set(sol.columns))
        common_cols.remove('asset')  # Remove asset since we'll add it back
        common_cols = ['asset'] + common_cols  # Add asset as the first column
        
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
        os.makedirs('combined_data', exist_ok=True)
        combined_path = 'combined_data/multi_asset_data.parquet'
        combined.to_parquet(combined_path)
        
        logger.info(f"Created combined dataset with {len(combined)} rows from BTC, ETH, and SOL data")
        logger.info(f"Saved to {combined_path}")
        
        return combined_path
    
    except Exception as e:
        logger.error(f"Error combining market data: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
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
    
    return parser.parse_args()

def load_market_data(path: str):
    """Load market data from CSV or Parquet file."""
    # If auto mode is specified, generate combined data
    if path == "auto":
        path = load_multi_asset_data()
    
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
    
    if args.command == "generate":
        logger.info("Generating training dataset")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data()
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
        training_args = TrainingArgs(
            base_model=args.base_model,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            early_stopping_patience=args.early_stopping,
            seed=args.seed
        )
        train_llm_model(training_args)
        
    elif args.command == "infer":
        logger.info("Running inference")
        
        # Handle auto market data option
        market_data_path = args.market_data
        if market_data_path == "auto":
            market_data_path = load_multi_asset_data()
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        explainer = RLLMExplainer(
            rl_model_path=args.rl_model,
            llm_model_path=args.llm_model,
            llm_base_model=args.base_model
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
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                # Print first result
                print(f"Action: {results[0]['action']}")
                print(f"Explanation: {results[0]['explanation']}")
        else:
            # Generate explanations from market data
            explanations = explainer.explain_trading_decisions(market_data_path)
            
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
            market_data_path = load_multi_asset_data()
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
            market_data_path = load_multi_asset_data()
            logger.info(f"Using auto-generated combined market data: {market_data_path}")
        
        # Load the chatbot
        chatbot = load_market_chatbot(
            model_path=args.llm_model,
            max_history=args.max_history,
            system_prompt=args.system_prompt,
            device="auto"
        )
        
        # Load market data if provided
        if market_data_path != "auto":
            market_data = load_market_data(market_data_path)
            chatbot.update_market_data(market_data)
            print(f"Loaded market data with {len(market_data)} records")
        
        # Load RL model and get trading signals if provided
        if args.rl_model and market_data_path != "auto":
            model = PPO.load(args.rl_model)
            signals = extract_trading_signals(model, market_data)
            chatbot.update_trading_signals(signals)
            print("Loaded trading signals from RL model")
            
            # Calculate basic performance metrics
            returns = np.diff(market_data['close'].values) / market_data['close'].values[:-1]
            portfolio_performance = {
                "total_return": f"{np.sum(returns * np.array([s['position'] for s in signals.values()][:-1])):.4f}",
                "sharpe_ratio": f"{np.mean(returns) / np.std(returns):.4f}",
                "win_rate": f"{np.mean([s['confidence'] > 0.5 for s in signals.values()]):.4f}"
            }
            chatbot.update_portfolio_performance(portfolio_performance)
            print("Calculated portfolio performance metrics")
        
        print("\nTrading Assistant Chatbot")
        print("Type 'exit', 'quit', or 'q' to end the conversation")
        print("Type 'reset' to reset the conversation history")
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