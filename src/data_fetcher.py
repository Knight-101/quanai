import ccxt
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from datetime import datetime, timedelta

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
            print(f"Successfully created DataFrame for {symbol} with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Critical error processing {symbol}: {str(e)}")
            return None

    def _add_features(self, df):
        try:
            # First set the index to timestamp
            df = df.set_index('timestamp')
            
            # Add technical indicators
            df = add_all_ta_features(
                df, 
                open="open", 
                high="high", 
                low="low", 
                close="close", 
                volume="volume",
                fillna=True
            )
            
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
        
        # Merge all dataframes
        dfs = []
        for asset, df in full_data.items():
            # Create multi-index columns
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            dfs.append(df)
        
        # Concatenate all dataframes
        full_df = pd.concat(dfs, axis=1)
        
        # Ensure all data is aligned
        full_df = full_df.ffill().bfill()
        
        print(f"Final merged shape before saving: {full_df.shape}")
        return full_df

if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    multi_data = fetcher.get_multi_data()
    multi_data.to_parquet('data/multi_crypto.parquet')
    print(f"Saved multi-asset dataset with shape: {multi_data.shape}")