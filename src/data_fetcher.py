import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

class CryptoDataFetcher:
    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], timeframe='30m'):
        self.exchange = ccxt.binance()
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_days = 365 * 10  # 10 years data

    def _fetch_single(self, symbol):
        try:
            since = self.exchange.parse8601((datetime.now() - timedelta(days=self.lookback_days)).isoformat())
            print(f"Fetching data for {symbol} since {datetime.fromtimestamp(since/1000)}")
            
            all_ohlcv = []
            while True:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=1000)
                    print(f"Fetched {len(ohlcv)} candles for {symbol}")
                    if not ohlcv:
                        break
                    all_ohlcv += ohlcv
                    since = ohlcv[-1][0] + 1
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    break
            
            if not all_ohlcv:
                print(f"No data retrieved for {symbol}")
                return None
                
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            # Verify timeframe consistency
            time_diff = pd.Series(df.index).diff().dt.total_seconds().dropna()
            expected_seconds = pd.Timedelta(self.timeframe).total_seconds()
            irregular_intervals = time_diff != expected_seconds
            if irregular_intervals.any():
                print(f"Warning: {symbol} has {irregular_intervals.sum()} irregular intervals")
            
            print(f"Successfully created DataFrame for {symbol} with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Critical error processing {symbol}: {str(e)}")
            return None

    def _add_features(self, df):
        try:
            # Volume indicators
            df['volume_ema'] = ta.volume.volume_weighted_average_price(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            df['obv'] = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])
            df['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            
            # Trend indicators
            df['ema_12'] = ta.trend.ema_indicator(close=df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(close=df['close'], window=26)
            df['macd'] = ta.trend.macd(close=df['close'])
            df['macd_signal'] = ta.trend.macd_signal(close=df['close'])
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(close=df['close'])
            df['stoch'] = ta.momentum.stoch(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_signal'] = ta.momentum.stoch_signal(high=df['high'], low=df['low'], close=df['close'])
            
            # Volatility indicators
            df['bb_high'] = ta.volatility.bollinger_hband(close=df['close'])
            df['bb_low'] = ta.volatility.bollinger_lband(close=df['close'])
            df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'])
            
            # Forward fill and then backward fill any remaining NaN values
            df = df.ffill().bfill()
            
            print(f"Added technical features. New shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error adding features: {str(e)}")
            return None

    def get_multi_data(self):
        full_data = {}
        for sym in self.symbols:
            print(f"Processing {sym}...")
            df = self._fetch_single(sym)
            if df is not None:
                df = self._add_features(df)
                if df is not None:
                    full_data[sym.split('/')[0]] = df
                else:
                    print(f"Failed to add features for {sym}")
            else:
                print(f"Skipping {sym} due to data fetch failure")
        
        if not full_data:
            raise ValueError("No data was successfully fetched for any symbol")
        
        # Create common index
        start_dates = [df.index.min() for df in full_data.values()]
        end_dates = [df.index.max() for df in full_data.values()]
        common_index = pd.date_range(
            start=max(start_dates),
            end=min(end_dates),
            freq=self.timeframe.replace('m', 'min')  # Convert 'm' to 'min' for minutes
        )
        
        # Align all dataframes
        aligned_data = {}
        for asset, df in full_data.items():
            # Resample to exact timeframe using 'min' for minutes
            df = df.resample(self.timeframe.replace('m', 'min')).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'volume_ema': 'last',
                'obv': 'last',
                'mfi': 'last',
                'ema_12': 'last',
                'ema_26': 'last',
                'macd': 'last',
                'macd_signal': 'last',
                'rsi': 'last',
                'stoch': 'last',
                'stoch_signal': 'last',
                'bb_high': 'last',
                'bb_low': 'last',
                'atr': 'last'
            })
            
            # Reindex to common index
            aligned_df = df.reindex(common_index)
            aligned_data[asset] = aligned_df
        
        # Merge with aligned indices
        dfs = []
        for asset, df in aligned_data.items():
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            dfs.append(df)
        
        # Concatenate and clean
        full_df = pd.concat(dfs, axis=1)
        full_df = full_df.dropna()  # Remove rows with any NaN values
        
        print(f"Final merged shape: {full_df.shape}")
        return full_df

if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    multi_data = fetcher.get_multi_data()
    multi_data.to_parquet('data/multi_crypto.parquet')
    print(f"Saved multi-asset dataset with shape: {multi_data.shape}")