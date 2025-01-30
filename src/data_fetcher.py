# src/data_fetcher.py
import ccxt
import pandas as pd
from ta import add_all_ta_features  # Update imports

def fetch_btc_ohlcv():
    """Fetch BTC/USDT OHLCV and compute TA indicators."""
    binance = ccxt.binance()
    since = binance.parse8601('2010-01-01T00:00:00Z')  # Start fetching from 2020
    all_ohlcv = []
    
    while True:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', '30m', since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Move to the next timestamp
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate all TA indicators
    df = add_all_ta_features(
        df,
        open="open", high="high", low="low", close="close", volume="volume"
    )
    
    # Selected key indicators for Bitcoin analysis
    ta_cols = [
        # Volume Indicators
        'volume_mfi',  # Money Flow Index - shows buying/selling pressure
        'volume_vwap',  # Volume Weighted Average Price
        'volume_obv',   # On Balance Volume - shows volume flow
        
        # Momentum Indicators
        'momentum_rsi',  # Relative Strength Index - overbought/oversold
        'momentum_stoch_rsi',  # Stochastic RSI - more sensitive RSI
        'momentum_tsi',  # True Strength Index
        
        # Trend Indicators
        'trend_macd',  # Moving Average Convergence Divergence
        'trend_ema_fast',  # 12-period EMA
        'trend_ema_slow',  # 26-period EMA
        'trend_adx',  # Average Directional Index - trend strength
        
        # Volatility Indicators
        'volatility_bbm',  # Bollinger Bands Middle
        'volatility_bbh',  # Bollinger Bands High
        'volatility_bbl',  # Bollinger Bands Low
        'volatility_atr',  # Average True Range - volatility measure
        
        # Additional Important Indicators
        'trend_ichimoku_a',  # Ichimoku Cloud A
        'trend_ichimoku_b'   # Ichimoku Cloud B
    ]
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume'] + ta_cols]
    return df

if __name__ == "__main__":
    # Fetch data and save
    df = fetch_btc_ohlcv()
    df.to_csv('data/btc_merged.csv', index=False)
    print("BTC OHLCV + TA data saved to data/btc_merged.csv")