import numpy as np
import torch
from stable_baselines3 import PPO
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Local imports
from trading_env import MultiCryptoEnv

def analyze_model_behavior(model, env, n_episodes=100):
    """Analyze trained model's behavior patterns"""
    action_patterns = []
    feature_importance = {}
    market_conditions = {
        'bull': [],  # Strong uptrend
        'bear': [],  # Strong downtrend
        'volatile': [],  # High volatility
        'stable': []   # Low volatility
    }
    
    obs = env.reset()
    
    for episode in range(n_episodes):
        done = False
        episode_actions = []
        
        while not done:
            # Get model's action and value prediction
            action, _states = model.predict(obs, deterministic=True)
            
            # Extract feature importances using policy network gradients
            with torch.no_grad():
                features = model.policy.features_extractor(
                    torch.FloatTensor(obs).to(model.device)
                )
                gradients = torch.autograd.grad(
                    features.sum(), 
                    model.policy.features_extractor.parameters(),
                    retain_graph=True
                )
                
                # Analyze which features had strongest gradients
                for grad in gradients:
                    feature_importance[grad.mean().item()] = grad.shape
            
            # Classify market condition
            market_features = obs[0][:env.n_features]
            volatility = np.std(market_features)
            returns = np.diff(market_features)
            
            if volatility > np.percentile(volatility, 75):
                market_conditions['volatile'].append(action)
            elif volatility < np.percentile(volatility, 25):
                market_conditions['stable'].append(action)
                
            if np.mean(returns) > np.percentile(returns, 75):
                market_conditions['bull'].append(action)
            elif np.mean(returns) < np.percentile(returns, 25):
                market_conditions['bear'].append(action)
            
            episode_actions.append(action)
            obs, reward, done, info = env.step(action)
            
        action_patterns.append(episode_actions)
    
    return {
        'action_patterns': action_patterns,
        'feature_importance': feature_importance,
        'market_conditions': market_conditions
    }

def print_model_insights(model_path, env):
    model = PPO.load(model_path)
    insights = analyze_model_behavior(model, env)
    
    print("\n=== Model Behavior Analysis ===")
    
    # 1. Market Condition Responses
    print("\nMarket Condition Responses:")
    for condition, actions in insights['market_conditions'].items():
        if actions:
            avg_allocation = np.mean(actions, axis=0)
            print(f"\n{condition.title()} Markets:")
            print(f"- Cash: {avg_allocation[0]:.2%}")
            print(f"- Asset Allocations: {avg_allocation[1:]}")
    
    # 2. Feature Importance
    print("\nMost Important Features:")
    sorted_features = sorted(
        insights['feature_importance'].items(), 
        key=lambda x: abs(x[0]), 
        reverse=True
    )[:5]
    for importance, shape in sorted_features:
        print(f"- Importance: {importance:.4f}, Shape: {shape}")
    
    # 3. Risk Management Patterns
    print("\nRisk Management Patterns:")
    volatile_actions = insights['market_conditions']['volatile']
    stable_actions = insights['market_conditions']['stable']
    print(f"- Avg Cash in Volatile Markets: {np.mean([a[0] for a in volatile_actions]):.2%}")
    print(f"- Avg Cash in Stable Markets: {np.mean([a[0] for a in stable_actions]):.2%}")

def plot_market_responses(market_conditions: Dict):
    """Visualize how the model responds to different market conditions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    conditions = ['bull', 'bear', 'volatile', 'stable']
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx//2, idx%2]
        actions = market_conditions[condition]
        if actions:
            actions_array = np.array(actions)
            sns.boxplot(data=actions_array, ax=ax)
            ax.set_title(f'{condition.title()} Market Allocations')
            ax.set_ylabel('Allocation')
            ax.set_xticklabels(['Cash'] + [f'Asset {i}' for i in range(actions_array.shape[1]-1)])
    
    plt.tight_layout()
    plt.savefig('market_responses.png')
    plt.close()

def analyze_trading_patterns(action_patterns: List):
    """Analyze common trading sequences and patterns"""
    pattern_counts = defaultdict(int)
    for episode_actions in action_patterns:
        for i in range(len(episode_actions)-2):
            # Convert actions to string representation for pattern matching
            pattern = str(episode_actions[i:i+3])  # Look at 3-step patterns
            pattern_counts[pattern] += 1
    
    return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10])

def main():
    try:
        # Load model and environment
        model_path = "models/multi_crypto_icm"
        
        # Use load_data function from backtest.py
        data = pd.read_parquet('data/multi_crypto.parquet')
        if data.empty:
            raise ValueError("DataFrame is empty. Please ensure data/multi_crypto.parquet contains valid data")
            
        env = MultiCryptoEnv(data)  # Create environment directly
        
        # Get insights
        print_model_insights(model_path, env)
        
        print("\nAnalysis completed successfully!")
        print("Check 'market_responses.png' for visualizations.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

