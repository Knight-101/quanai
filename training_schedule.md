# Crypto Trading Bot Training Schedule

<CURRENT_CURSOR_POSITION>

## Phase-Based Training Strategy

### Initial 1M Steps (Foundation Phase)

| Phase | Steps Range | Description                               |
| ----- | ----------- | ----------------------------------------- |
| 1     | 0-500K      | Exploration and initial pattern discovery |
| 2     | 500K-1M     | Core strategy development                 |

### Extended 9M Steps (Mastery Phase)

| Phase | Steps Range | Description                  |
| ----- | ----------- | ---------------------------- |
| 3     | 1M-2M       | Strategy refinement          |
| 4     | 2M-3M       | Strategy consolidation       |
| 5     | 3M-4M       | Stabilization                |
| 6     | 4M-5M       | Regime specialization        |
| 7     | 5M-7M       | Multi-timeframe integration  |
| 8     | 7M-9M       | Advanced pattern recognition |
| 9     | 9M-10M      | Strategy perfection          |

## Hyperparameter Schedule

### Learning Rate Schedule

| Phase | Steps Range | Learning Rate      | Notes                      |
| ----- | ----------- | ------------------ | -------------------------- |
| 1     | 0-500K      | 0.00012 → 0.00008  | Initial high learning rate |
| 2     | 500K-1M     | 0.00008 → 0.00005  | Gradual decrease           |
| 3     | 1M-2M       | 0.00005 → 0.00004  | Further refinement         |
| 4     | 2M-3M       | 0.00004 → 0.00003  | Consolidation              |
| 5     | 3M-4M       | 0.00003 → 0.00003  | Stabilization              |
| 6     | 4M-5M       | 0.00003 → 0.00002  | Regime specialization      |
| 7     | 5M-7M       | 0.00002 → 0.000015 | Multi-timeframe focus      |
| 8     | 7M-9M       | 0.000015 → 0.00001 | Advanced patterns          |
| 9     | 9M-10M      | 0.00001 → 0.00001  | Final refinement           |

### Entropy Coefficient Schedule

| Phase | Steps Range | Entropy Coefficient | Notes                 |
| ----- | ----------- | ------------------- | --------------------- |
| 1     | 0-500K      | 0.05 → 0.04         | High exploration      |
| 2     | 500K-1M     | 0.04 → 0.035        | Moderate exploration  |
| 3     | 1M-2M       | 0.035 → 0.025       | Balanced exploration  |
| 4     | 2M-3M       | 0.025 → 0.018       | Reduced exploration   |
| 5     | 3M-4M       | 0.018 → 0.015       | Stabilization         |
| 6     | 4M-5M       | 0.015 → 0.012       | Regime specialization |
| 7     | 5M-7M       | 0.012 → 0.008       | Multi-timeframe focus |
| 8     | 7M-9M       | 0.008 → 0.005       | Advanced patterns     |
| 9     | 9M-10M      | 0.005 → 0.005       | Final refinement      |

### Other Hyperparameters

| Parameter          | Phase 1-2 | Phase 3-5 | Phase 6-7 | Phase 8-9 |
| ------------------ | --------- | --------- | --------- | --------- |
| Batch Size         | 512       | 768       | 1024      | 1536      |
| GAE Lambda         | 0.935     | 0.95      | 0.96      | 0.97      |
| Clip Range         | 0.25      | 0.2       | 0.15      | 0.1       |
| N_epochs           | 10        | 8         | 5         | 5         |
| VF Coefficient     | 1.0       | 0.8-0.6   | 0.6-0.4   | 0.4       |
| Dropout Rate       | 0.088     | 0.066     | 0.044     | 0.022     |
| Transformer Heads  | 4         | 4         | 8         | 8         |
| Transformer Layers | 2         | 2         | 3         | 3         |

## Feature Extractor Evolution

### Dropout Rate Progression

| Phase | Steps Range | Dropout Rate |
| ----- | ----------- | ------------ |
| 1-3   | 0-2M        | 0.088        |
| 4-5   | 2M-4M       | 0.066        |
| 6-7   | 4M-7M       | 0.044        |
| 8-9   | 7M-10M      | 0.022        |

### Transformer Configuration

| Phase | Steps Range | Heads | Layers |
| ----- | ----------- | ----- | ------ |
| 1-5   | 0-4M        | 4     | 2      |
| 6-9   | 4M-10M      | 8     | 3      |

## Performance Metrics to Monitor

1. **Primary Metrics**

   - Sharpe Ratio
   - Win Rate
   - Max Drawdown
   - Profit Factor

2. **Secondary Metrics**

   - Average Trade Duration
   - Position Concentration
   - Leverage Usage
   - Regime-Specific Performance

3. **Risk Metrics**
   - Value at Risk (VaR)
   - Expected Shortfall
   - Volatility
   - Correlation with Market

## Phase Transition Guidelines

1. **When to Move to Next Phase**

   - Performance metrics have plateaued for 100K steps
   - Current phase's target metrics are achieved
   - No significant improvement in last 10% of the phase's steps

2. **What to Check Before Transition**

   - Model stability across different market regimes
   - Risk-adjusted returns consistency
   - Position sizing behavior
   - Leverage usage patterns

3. **Transition Steps**
   - Save comprehensive checkpoint
   - Update hyperparameters according to schedule
   - Consider resetting optimizer state if performance has plateaued
   - Begin new phase with the recommended hyperparameters

## Using the Auto-Phase System

This codebase includes automatic phase transitions with recommended hyperparameters. To take advantage of this:

1. **Starting from Phase 1**:

   ```bash
   python main_opt.py --train --model_dir models/manual/phase1
   ```

2. **Continuing to Next Phase**:

   ```bash
   python main_opt.py --continue_training --model_path models/manual/phase1/final_model --model_dir models/manual/phase1 --use_recommendations
   ```

   This will:

   - Automatically detect the current phase
   - Load recommended hyperparameters for the next phase
   - Train for the appropriate number of steps
   - Save recommendations for the subsequent phase

3. **Continuing with Custom Hyperparameters**:
   ```bash
   python main_opt.py --continue_training --model_path models/manual/phase1/final_model --model_dir models/manual/phase1 --hyperparams "ent_coef=0.06,learning_rate=0.0001"
   ```

## Notes

- All hyperparameter changes are automatic when using the `--use_recommendations` flag
- Override recommendations with explicit `--hyperparams` when needed
- The system automatically adjusts hyperparameters based on performance
- Keep detailed logs of hyperparameter changes and their effects

## Parameter Decision Guide

### Interpreting Evaluation Metrics

#### Key Metrics and Their Implications

1. **Reward and Sharpe Ratio**

   - Negative mean_reward and reward_sharpe indicate poor strategy performance
   - When both are negative:
     - Increase exploration (higher entropy coefficient)
     - Reduce learning rate to prevent overshooting
     - Consider increasing batch size for more stable updates

2. **Drawdown and Risk Metrics**

   - High max_drawdown (>15%) suggests excessive risk-taking
   - When drawdown is high:
     - Increase clip_range to limit policy changes
     - Reduce max_leverage in environment
     - Increase GAE lambda for more stable value estimates

3. **Leverage Usage**

   - High avg_leverage (>3) indicates aggressive positioning
   - When leverage is high:
     - Increase risk penalties in reward function
     - Reduce position limits in environment
     - Consider increasing value function coefficient

4. **Trade Statistics**
   - Low trades_executed suggests insufficient exploration
   - When trades are low:
     - Increase entropy coefficient
     - Reduce learning rate to encourage exploration
     - Consider adjusting action space bounds

### Parameter Adjustment Rules

#### Based on Performance Metrics

1. **Poor Performance (Negative Sharpe, High Drawdown)**

   ```python
   # Example adjustment for poor performance
   {
       'learning_rate': current_lr * 0.8,  # Reduce by 20%
       'ent_coef': current_ent_coef * 1.3,  # Increase by 30%
       'clip_range': min(current_clip * 1.2, 0.3),  # Increase up to 0.3
       'batch_size': current_batch * 1.2,  # Increase by 20%
       'gae_lambda': min(current_gae * 1.1, 0.99)  # Increase up to 0.99
   }
   ```

2. **Stable but Suboptimal (Low Sharpe, Low Drawdown)**

   ```python
   # Example adjustment for stable but suboptimal
   {
       'learning_rate': current_lr * 0.9,  # Reduce by 10%
       'ent_coef': current_ent_coef * 1.1,  # Increase by 10%
       'clip_range': current_clip,  # Keep stable
       'batch_size': current_batch,  # Keep stable
       'n_epochs': current_epochs + 1  # Increase epochs
   }
   ```

3. **Good Performance (Positive Sharpe, Low Drawdown)**
   ```python
   # Example adjustment for good performance
   {
       'learning_rate': current_lr * 1.1,  # Increase by 10%
       'ent_coef': current_ent_coef * 0.9,  # Reduce by 10%
       'clip_range': current_clip * 0.9,  # Reduce by 10%
       'batch_size': current_batch,  # Keep stable
       'n_epochs': current_epochs  # Keep stable
   }
   ```

### Addressing Specific Metric Issues

#### Highly Negative Explained Variance

Explained variance measures how well your value function predicts future rewards. When it's highly negative:

**What it signifies:**

- The value function is worse than a constant baseline predictor
- The model is making systematically incorrect reward predictions
- There's likely fundamental misalignment between policy and value networks
- Model may be "anti-learning" - learning the opposite of optimal behavior

**Recommended adjustments:**

1. **Immediate fixes:**

   ```python
   {
       'learning_rate': current_lr * 0.5,  # Drastic reduction
       'vf_coef': current_vf_coef * 2.0,  # Double value function emphasis
       'n_epochs': max(current_epochs - 2, 3),  # Reduce to prevent overtraining
       'max_grad_norm': 0.3  # Add or reduce gradient clipping
   }
   ```

2. **Structural changes:**

   - Consider separate networks for policy and value functions
   - Simplify value network architecture
   - Reset optimizer momentum completely
   - Review feature normalization for value inputs

3. **Training approach:**
   - Train value function for several epochs before resuming full training
   - Temporarily increase value function batch size
   - Consider warm-starting with supervised learning on historical data
   - Check for extreme rewards or inappropriate reward scaling

### Value Function Coefficient Guidelines

The value function coefficient (vf_coef) balances the importance of value function learning relative to policy learning. Here's a detailed guide:

#### Ideal Range by Phase

| Phase | Steps Range | vf_coef Range | Notes                                                        |
| ----- | ----------- | ------------- | ------------------------------------------------------------ |
| 1-2   | 0-1M        | 0.8 - 1.2     | Higher emphasis on value learning during initial exploration |
| 3-4   | 1M-3M       | 0.6 - 0.8     | Balance between policy and value learning                    |
| 5-6   | 3M-5M       | 0.4 - 0.6     | More emphasis on policy refinement                           |
| 7-9   | 5M-10M      | 0.3 - 0.5     | Focus on policy optimization                                 |

#### Adjustment Rules

1. **When to Increase vf_coef:**

   - Negative explained variance
   - High value loss
   - Unstable returns
   - Large prediction errors
   - During regime transitions

2. **When to Decrease vf_coef:**

   - Low policy entropy
   - Overfitting to value predictions
   - Stable but suboptimal performance
   - During stable market conditions

3. **Special Cases:**
   - For highly negative explained variance: Increase up to 2.0
   - For very stable markets: Can go as low as 0.2
   - During high volatility: Keep between 0.6-0.8

#### Monitoring Value Function Performance

1. **Good Indicators:**

   - Explained variance between 0.3 and 0.8
   - Value loss decreasing steadily
   - Value predictions tracking actual returns
   - Stable value function gradients

2. **Warning Signs:**
   - Explained variance < 0.1 or > 0.9
   - Value loss increasing
   - Large prediction errors
   - Oscillating value estimates
