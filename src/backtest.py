import backtrader as bt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

class RLStrategy(bt.Strategy):
    params = (
        ('model_path', 'models/multi_crypto_icm'),
        ('initial_balance', 10000),
    )

    def __init__(self):
        self.model = PPO.load(self.params.model_path)
        self.assets = [d._name for d in self.datas]
        self.current_step = 0
        
        # Setup logging
        self.trade_log = []
        
        # Add technical indicators
        for d in self.datas:
            d.rsi = bt.indicators.RSI(d.close, period=14)
            d.macd = bt.indicators.MACD(d.close)
            d.obv = bt.indicators.OnBalanceVolume(d)

    def next(self):
        # Skip if any indicator is not ready
        if not all(d.rsi.ready and d.macd.ready and d.obv.ready for d in self.datas):
            return
            
        # Prepare observation
        obs = self._prepare_observation()
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)  # Use deterministic=True for testing
        
        # Execute trades
        self._execute_trades(action)
        
    def _prepare_observation(self):
        # Get current market state
        market_data = []
        for d in self.datas:
            market_data.extend([
                d.open[0], d.high[0], d.low[0], d.close[0], d.volume[0],
                d.rsi[0], d.macd.macd[0], d.obv[0]
            ])
        
        # Get current portfolio state
        total = self.broker.getvalue()
        allocations = [self.getposition(d).size * d.close[0] / total if total > 0 else 0.0 for d in self.datas]
        cash = self.broker.getcash() / total if total > 0 else 1.0
        
        obs = np.array(market_data + [cash] + allocations)
        return obs.reshape(1, -1)  # Add batch dimension

    def _execute_trades(self, action):
        total = self.broker.getvalue()
        if total <= 0:
            return
            
        cash_alloc = max(0.0, min(1.0, action[0]))  # Clip between 0 and 1
        asset_allocs = action[1:]
        
        # Normalize asset allocations to sum to (1 - cash_alloc)
        asset_allocs = np.clip(asset_allocs, 0, 1)
        if sum(asset_allocs) > 0:
            asset_allocs = asset_allocs * (1 - cash_alloc) / sum(asset_allocs)
        
        # Log current portfolio state
        current_time = self.datas[0].datetime.datetime()
        log_entry = {
            'datetime': current_time,
            'portfolio_value': total,
            'cash_allocation': cash_alloc,
            'trades': []
        }
        
        # Execute trades
        for i, d in enumerate(self.datas):
            target_value = total * asset_allocs[i]
            current_value = self.getposition(d).size * d.close[0]
            
            trade_info = {
                'asset': d._name,
                'current_value': current_value,
                'target_value': target_value,
                'price': d.close[0]
            }
            
            if target_value > current_value:
                self.order_target_value(d, target_value)
                trade_info['action'] = 'BUY'
            elif target_value < current_value:
                self.close(d)  # Close position first
                trade_info['action'] = 'SELL'
                if target_value > 0:
                    self.order_target_value(d, target_value)  # Then open new position if needed
                    trade_info['action'] = 'REBALANCE'
            else:
                trade_info['action'] = 'HOLD'
            
            log_entry['trades'].append(trade_info)
        
        self.trade_log.append(log_entry)

def run_backtest():
    cerebro = bt.Cerebro()
    
    # Add data feeds
    multi_data = pd.read_parquet('data/multi_crypto.parquet')
    
    # Convert index to datetime if it's not already
    if not isinstance(multi_data.index, pd.DatetimeIndex):
        multi_data.index = pd.to_datetime(multi_data.index, unit='s')
    
    for asset in multi_data.columns.get_level_values(0).unique():
        df = multi_data[asset].reset_index()
        df.columns = ['datetime'] + list(df.columns[1:])  # Rename index column to datetime
        df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format
        
        data = bt.feeds.PandasData(
            dataname=df,
            datetime='datetime',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,  # -1 means not used
            timeframe=bt.TimeFrame.Minutes
        )
        cerebro.adddata(data, name=asset)
    
    # Add strategy
    cerebro.addstrategy(RLStrategy)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001)
    
    # Set starting cash
    cerebro.broker.setcash(10000.0)
    
    # Run backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]  # Get the strategy instance after running
    
    # Print trade log
    print("\nTrade Log:")
    for entry in strat.trade_log:
        print(f"\nDate: {entry['datetime']}")
        print(f"Portfolio Value: ${entry['portfolio_value']:.2f}")
        print(f"Cash Allocation: {entry['cash_allocation']:.2%}")
        print("Trades:")
        for trade in entry['trades']:
            print(f"  {trade['asset']}: {trade['action']} - Current: ${trade['current_value']:.2f}, "
                  f"Target: ${trade['target_value']:.2f}, Price: ${trade['price']:.2f}")
    
    print('\nFinal Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Plot if running in a notebook or GUI environment
    try:
        cerebro.plot()
    except:
        print("Could not generate plot - might be running in a non-GUI environment")

if __name__ == "__main__":
    run_backtest()