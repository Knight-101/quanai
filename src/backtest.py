from stable_baselines3 import PPO
from trading_env import CryptoTradingEnv
import pandas as pd

def backtest(model_path: str = "models/ppo_btc_base"):
    df = pd.read_csv('data/btc_merged.csv')
    env = CryptoTradingEnv(df)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    total_trades = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if action in [1, 2]:  # Count buy/sell actions
            total_trades += 1
        env.render()
    
    final_value = env.balance + env.btc_held * env.current_price
    print(f"\nBacktest Complete")
    print(f"Initial: $10,000 | Final: ${final_value:.2f}")
    print(f"Return: {(final_value / 10000 - 1) * 100:.2f}%")
    print(f"Total Trades: {total_trades}")

if __name__ == "__main__":
    backtest()