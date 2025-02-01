import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import MultiCryptoEnv, CurriculumTradingWrapper

# Configure logging for robust error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(data_path='data/multi_crypto.parquet'):
    if not os.path.exists(data_path):
        logging.error(f"Data file {data_path} not found.")
        sys.exit(1)
    df = pd.read_parquet(data_path)
    if df.empty:
        logging.error("Loaded DataFrame is empty. Check your data source.")
        sys.exit(1)
    logging.info(f"Loaded data with shape: {df.shape}")
    return df

def create_env(df, initial_balance=10000.0):
    """
    Create and return a vectorized trading environment.
    Wrap the base MultiCryptoEnv with the CurriculumTradingWrapper.
    """
    try:
        def make_env():
            base_env = MultiCryptoEnv(df, initial_balance=initial_balance)
            wrapped_env = CurriculumTradingWrapper(base_env)
            return wrapped_env

        env = DummyVecEnv([make_env])
        logging.info("Vectorized trading environment created successfully.")
        return env
    except Exception as e:
        logging.exception("Error creating trading environment.")
        sys.exit(1)

def load_normalized_env(env, vec_norm_path="models/vec_normalize.pkl"):
    """
    Load the VecNormalize wrapper from disk and apply it to the vectorized environment.
    """
    if not os.path.exists(vec_norm_path):
        logging.error(f"VecNormalize file {vec_norm_path} not found.")
        sys.exit(1)
    try:
        env = VecNormalize.load(vec_norm_path, env)
        # Set evaluation mode so stats are not updated during backtesting.
        env.training = False
        env.norm_reward = False  # Optionally disable reward normalization for evaluation.
        logging.info("VecNormalize wrapper loaded and applied successfully.")
        return env
    except Exception as e:
        logging.exception("Error loading VecNormalize wrapper.")
        sys.exit(1)

def load_model(model_path="models/multi_crypto_icm"):
    try:
        model = PPO.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.exception("Error loading the model.")
        sys.exit(1)

def run_backtest(model, env, output_csv='backtest_results.csv', trade_log_csv='trade_log.csv'):
    """
    Run the backtesting loop.
    Logs overall portfolio metrics and detailed trade events.
    Uses env.envs[0].unwrapped to obtain the underlying base environment.
    Handles both 4-tuple and 5-tuple returns from env.step().
    """
    obs = env.reset()
    portfolio_history = []
    trade_log = []  # To record detailed trade events.
    step_counter = 0  # External counter for steps

    try:
        while True:
            # Predict action deterministically.
            action, _ = model.predict(obs, deterministic=True)
            # Unwrap the single environment instance to get the base environment.
            single_env = env.envs[0].unwrapped

            # Record pre-trade info.
            prev_balance = single_env.balance
            pre_allocations = single_env.allocations.copy()
            pre_prices = {asset: single_env._get_current_price(asset) for asset in single_env.assets}

            # Take a step.
            result = env.step(action)
            # Handle both possible tuple lengths.
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            elif len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                raise ValueError("env.step() returned an unexpected number of values.")

            current_value = single_env._portfolio_value()

            # Record portfolio metrics.
            portfolio_history.append({
                "step": step_counter,
                "balance": single_env.balance,
                "portfolio_value": current_value,
                "reward": reward[0],
                "trade_count": single_env.trade_count
            })

            # Log trade details by comparing pre- and post-trade allocations.
            post_allocations = single_env.allocations
            for asset in pre_allocations.keys():
                pre_qty = pre_allocations.get(asset, 0)
                post_qty = post_allocations.get(asset, 0)
                if abs(post_qty - pre_qty) > 1e-8:
                    trade_log.append({
                        "step": step_counter,
                        "asset": asset,
                        "pre_qty": pre_qty,
                        "post_qty": post_qty,
                        "delta_qty": post_qty - pre_qty,
                        "execution_price": pre_prices.get(asset, np.nan),
                        "balance_change": single_env.balance - prev_balance
                    })

            step_counter += 1
            if step_counter % 100 == 0:
                logging.info(f"Step: {step_counter}, Portfolio Value: ${current_value:.2f}")

            if done[0] or truncated:
                break

    except Exception as e:
        logging.exception("Error during backtesting loop.")
    finally:
        logging.info("Backtesting completed.")
        results_df = pd.DataFrame(portfolio_history)
        trade_log_df = pd.DataFrame(trade_log)
        try:
            results_df.to_csv(output_csv, index=False)
            logging.info(f"Backtest results saved to {output_csv}")
        except Exception as e:
            logging.exception("Failed to save backtest results to CSV.")
        try:
            trade_log_df.to_csv(trade_log_csv, index=False)
            logging.info(f"Trade log saved to {trade_log_csv}")
        except Exception as e:
            logging.exception("Failed to save trade log to CSV.")
        return results_df, trade_log_df

def plot_results(results_df):
    if results_df.empty or "step" not in results_df.columns:
        logging.warning("No portfolio history data to plot.")
        return
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(results_df["step"], results_df["portfolio_value"], label="Portfolio Value", color='blue')
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value")
        plt.title("Backtest: Portfolio Value Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.exception("Error while plotting backtest results.")

if __name__ == "__main__":
    # Define file paths.
    data_path = "data/multi_crypto.parquet"
    model_path = "models/multi_crypto_icm"
    vec_norm_path = "models/vec_normalize.pkl"
    output_csv = "backtest_results.csv"
    trade_log_csv = "trade_log.csv"

    # Load historical data.
    df = load_data(data_path)

    # Create a vectorized trading environment.
    env = create_env(df, initial_balance=10000.0)

    # Load the VecNormalize wrapper and apply it.
    env = load_normalized_env(env, vec_norm_path)

    # Load the pre-trained model.
    model = load_model(model_path)

    # Run the backtest simulation.
    results_df, trade_log_df = run_backtest(model, env, output_csv, trade_log_csv)

    # Plot the portfolio performance.
    plot_results(results_df)
