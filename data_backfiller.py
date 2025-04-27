#!/usr/bin/env python
import os
import sys
import json
import logging
import asyncio
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_backfiller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_backfiller')

class MarketDataBackfiller:
    """
    Downloads historical market data for specified symbols and timeframes,
    calculates technical indicators, and saves the data to files.
    """
    def __init__(self, symbols, timeframes, output_dir, periods=1500, exchange='binance'):
        self.symbols = symbols.split(',') if isinstance(symbols, str) else symbols
        self.timeframes = timeframes.split(',') if isinstance(timeframes, str) else timeframes
        self.output_dir = output_dir
        self.periods = periods
        self.exchange_id = exchange
        self.exchange = None

    async def fetch_data(self):
        """
        Downloads historical market data for each symbol and timeframe,
        calculates technical indicators, and saves the data.
        """
        try:
            # Create exchange instance
            logger.info(f"Initializing {self.exchange_id} exchange")
            self.exchange = getattr(ccxt, self.exchange_id)({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures market for perpetual contracts
                }
            })
            
            # Load markets
            logger.info("Loading markets")
            await self.exchange.load_markets()
            
            # Process each timeframe
            for timeframe in self.timeframes:
                logger.info(f"Processing timeframe: {timeframe}")
                
                # Create directory for this timeframe
                timeframe_dir = os.path.join(self.output_dir, timeframe)
                os.makedirs(timeframe_dir, exist_ok=True)
                
                # Download data for each symbol
                all_dfs = {}
                for symbol in self.symbols:
                    try:
                        logger.info(f"Fetching data for {symbol}, timeframe {timeframe}")
                        
                        # Fetch historical OHLCV data
                        ohlcv = await self.exchange.fetch_ohlcv(
                            symbol=symbol, 
                            timeframe=timeframe, 
                            limit=self.periods
                        )
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Handle any duplicate timestamps
                        df = df[~df.index.duplicated(keep='first')]
                        
                        logger.info(f"Fetched {len(df)} candles for {symbol}")
                        
                        # Calculate technical indicators
                        df = self._calculate_indicators(df)
                        
                        # Save individual symbol data
                        symbol_filename = symbol.replace('/', '') + '.parquet'
                        symbol_path = os.path.join(timeframe_dir, symbol_filename)
                        df.to_parquet(symbol_path)
                        logger.info(f"Saved data to {symbol_path}")
                        
                        # Also save as CSV for easier inspection
                        csv_path = os.path.join(timeframe_dir, symbol.replace('/', '') + '.csv')
                        df.to_csv(csv_path)
                        
                        # Store for combined file
                        all_dfs[symbol] = df
                        
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
                # Create a combined file with all symbols for this timeframe
                if all_dfs:
                    await self._create_combined_file(all_dfs, timeframe_dir, timeframe)
            
            logger.info("Data download completed")
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
        finally:
            # Close exchange
            if self.exchange:
                await self.exchange.close()
    
    async def _create_combined_file(self, all_dfs, timeframe_dir, timeframe):
        """Creates a combined file with all symbols for a timeframe."""
        try:
            logger.info(f"Creating combined file for timeframe {timeframe}")
            
            # Create MultiIndex columns with symbol as the first level
            combined_data = pd.DataFrame()
            
            for symbol, df in all_dfs.items():
                # Create MultiIndex columns
                symbol_id = symbol.replace('/', '')
                df.columns = pd.MultiIndex.from_product(
                    [[symbol_id], df.columns],
                    names=['asset', 'feature']
                )
                
                # Add to combined DataFrame
                if combined_data.empty:
                    combined_data = df
                else:
                    # Merge on index with outer join to include all timestamps
                    combined_data = combined_data.join(df, how='outer')
            
            # Sort by timestamp
            combined_data.sort_index(inplace=True)
            
            # Forward fill missing values
            combined_data = combined_data.ffill()
            
            # Save combined file
            combined_path = os.path.join(timeframe_dir, 'combined_data.parquet')
            combined_data.to_parquet(combined_path)
            logger.info(f"Saved combined data to {combined_path}")
            
            # Also save as CSV for easier inspection
            csv_path = os.path.join(timeframe_dir, 'combined_data.csv')
            combined_data.to_csv(csv_path)
            
        except Exception as e:
            logger.error(f"Error creating combined file: {str(e)}")
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators for a DataFrame."""
        try:
            # Make sure we have numeric data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # ---------------- Simple Moving Averages ----------------
            # Short-term
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Medium-term
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_100'] = df['close'].rolling(window=100).mean()
            
            # Long-term
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # ---------------- Exponential Moving Averages ----------------
            # Short-term
            df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Medium-term
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
            
            # Long-term
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            # ---------------- MACD ----------------
            # MACD Line: (12-day EMA - 26-day EMA)
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            
            # Signal Line: 9-day EMA of MACD Line
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # MACD Histogram: MACD Line - Signal Line
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # ---------------- RSI ----------------
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Calculate RSI
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # ---------------- Bollinger Bands ----------------
            # Middle band = 20-day SMA
            df['bb_middle'] = df['sma_20']
            
            # Calculate standard deviation
            df['20d_std'] = df['close'].rolling(window=20).std()
            
            # Upper band = Middle band + (20-day standard deviation * 2)
            df['bb_upper'] = df['bb_middle'] + (df['20d_std'] * 2)
            
            # Lower band = Middle band - (20-day standard deviation * 2)
            df['bb_lower'] = df['bb_middle'] - (df['20d_std'] * 2)
            
            # ---------------- Average True Range (ATR) ----------------
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
            
            # ---------------- Stochastic Oscillator ----------------
            # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            
            # %D = 3-day SMA of %K
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # ---------------- Rate of Change (ROC) ----------------
            # Formula: ((Current Close - Close n periods ago) / Close n periods ago) * 100
            df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # ---------------- Volatility Metrics ----------------
            # Daily Returns
            df['returns'] = df['close'].pct_change()
            
            # 10-day standard deviation of returns (annualized)
            df['volatility_10d'] = df['returns'].rolling(window=10).std() * np.sqrt(252)
            
            # 30-day standard deviation of returns (annualized)
            df['volatility_30d'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
            
            # ---------------- Indicators for market regime detection ----------------
            # ADX (Average Directional Index) - Used to determine trend strength
            df['tr'] = true_range  # True Range calculated above
            df['dm_plus'] = df['high'].diff().clip(lower=0)
            df['dm_minus'] = df['low'].diff().clip(upper=0).abs()
            
            df['tr_14'] = df['tr'].rolling(window=14).sum()
            df['dm_plus_14'] = df['dm_plus'].rolling(window=14).sum()
            df['dm_minus_14'] = df['dm_minus'].rolling(window=14).sum()
            
            df['di_plus_14'] = (df['dm_plus_14'] / df['tr_14']) * 100
            df['di_minus_14'] = (df['dm_minus_14'] / df['tr_14']) * 100
            
            df['dx'] = (abs(df['di_plus_14'] - df['di_minus_14']) / (df['di_plus_14'] + df['di_minus_14'])) * 100
            df['adx_14'] = df['dx'].rolling(window=14).mean()
            
            # Label market regimes (Trending, Ranging, Volatile)
            # A simple way to determine market regime:
            # ADX > 25 = Trending, ADX < 20 = Ranging
            df['trending'] = (df['adx_14'] > 25).astype(int)
            df['ranging'] = (df['adx_14'] < 20).astype(int)
            
            # Clean up intermediate columns used for calculations
            columns_to_drop = [
                'tr', 'dm_plus', 'dm_minus', 'tr_14', 'dm_plus_14', 
                'dm_minus_14', 'di_plus_14', 'di_minus_14', 'dx', '20d_std'
            ]
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Fill NaN values
            df = df.fillna(method='bfill')
            df = df.fillna(0)  # Fill any remaining NaNs with 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            # Return original dataframe if there's an error
            return df

async def main():
    parser = argparse.ArgumentParser(description='Download market data and calculate technical indicators')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT',
                        help='Comma-separated list of symbols to fetch data for')
    parser.add_argument('--timeframes', type=str, default='5m',
                        help='Comma-separated list of timeframes to fetch data for')
    parser.add_argument('--output-dir', type=str, default='data/market_data',
                        help='Directory to save the downloaded data')
    parser.add_argument('--periods', type=int, default=5000,
                        help='Number of candlesticks to fetch per symbol')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to fetch data from')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize backfiller
    backfiller = MarketDataBackfiller(
        symbols=args.symbols,
        timeframes=args.timeframes,
        output_dir=args.output_dir,
        periods=args.periods,
        exchange=args.exchange
    )
    
    # Fetch data
    await backfiller.fetch_data()

if __name__ == '__main__':
    asyncio.run(main()) 