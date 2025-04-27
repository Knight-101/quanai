#!/usr/bin/env python
import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import ccxt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backfiller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_backfiller')

# Try to import async version of ccxt for compatibility
try:
    import ccxt.async_support as ccxt_async
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False
    logger.warning("Could not import ccxt.async_support. Using synchronous version.")

class MarketDataBackfiller:
    """
    Downloads historical market data for specified symbols and timeframes,
    calculates technical indicators, and saves the data to files.
    """
    def __init__(self, symbols=None, timeframes=None, output_dir=None, periods=1000, exchange='binance'):
        """
        Initialize the market data backfiller.
        
        Args:
            symbols: List of trading pairs (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., '5m', '15m', '1h')
            output_dir: Directory to save the data files
            periods: Number of candles to fetch for each symbol and timeframe
            exchange: Exchange to fetch data from (e.g., 'binance')
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.timeframes = timeframes or ['5m']
        self.output_dir = output_dir or 'data/market_data'
        self.periods = periods
        
        # Initialize exchange client
        try:
            exchange_class = getattr(ccxt, exchange)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            logger.info(f"Initialized exchange client for {exchange}")
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            self.exchange = None
    
    def fetch_data(self):
        """
        Fetch historical market data for all symbols and timeframes.
        """
        if not self.exchange:
            logger.error("Exchange client not initialized. Cannot fetch data.")
            return False
        
        try:
            for timeframe in self.timeframes:
                logger.info(f"Processing timeframe: {timeframe}")
                
                # Create directory for this timeframe
                timeframe_dir = os.path.join(self.output_dir, timeframe)
                os.makedirs(timeframe_dir, exist_ok=True)
                
                # Dictionary to store DataFrames for all symbols
                all_dfs = {}
                
                # Process each symbol
                for symbol in self.symbols:
                    logger.info(f"Fetching data for {symbol} on {timeframe} timeframe")
                    
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=self.periods
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Add symbol column (important for training compatibility)
                    df['symbol'] = symbol
                    
                    # Calculate technical indicators
                    df = self._calculate_indicators(df)
                    
                    # Store in dictionary
                    all_dfs[symbol] = df
                    
                    # Save individual symbol data
                    symbol_file = os.path.join(timeframe_dir, f"{symbol}.parquet")
                    df.to_parquet(symbol_file)
                    
                    # Also save as CSV for easier inspection
                    csv_file = os.path.join(timeframe_dir, f"{symbol}.csv")
                    df.to_csv(csv_file)
                    
                    logger.info(f"Saved {len(df)} candles for {symbol} to {symbol_file}")
                    
                    # Rate limiting
                    time.sleep(1)
                
                # Create combined data file - THIS IS THE CRITICAL PART
                self._create_combined_data(all_dfs, timeframe_dir)
            
            return True
        
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return False
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Make a copy to avoid warnings
            df = df.copy()
            
            # Simple Moving Averages
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_25'] = df['close'].rolling(window=25).mean()
            df['sma_99'] = df['close'].rolling(window=99).mean()
            
            # Exponential Moving Averages
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI (14 periods)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Price momentum
            df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
            
            # Rate of Change
            df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Percentage price oscillator
            df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100
            
            # Add some price action indicators
            df['candle_body'] = df['close'] - df['open']
            df['candle_size'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])
            df['lower_shadow'] = df['open'].where(df['close'] >= df['open'], df['close']) - df['low']
            
            # Add day of week and hour of day
            df['day_of_week'] = df.index.dayofweek
            df['hour_of_day'] = df.index.hour
            
            # Clean up NaN values
            df = df.fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def _create_combined_data(self, all_dfs, timeframe_dir):
        """
        Create a combined data file with all symbols.
        This is critical for compatibility with the training code.
        
        Args:
            all_dfs: Dictionary of DataFrames, one for each symbol
            timeframe_dir: Directory to save combined data file
        """
        try:
            # Method 1: Create MultiIndex format (as used in training code)
            combined_data = pd.DataFrame()
            
            for symbol, df in all_dfs.items():
                # Create MultiIndex columns with symbol as first level
                df_copy = df.copy()
                df_copy.columns = pd.MultiIndex.from_product(
                    [[symbol], df_copy.columns],
                    names=['asset', 'feature']
                )
                
                # Add to combined DataFrame
                if combined_data.empty:
                    combined_data = df_copy
                else:
                    combined_data = combined_data.join(df_copy, how='outer')
            
            # Make sure timestamps are sorted and handle missing values
            combined_data.sort_index(inplace=True)
            combined_data = combined_data.ffill()
            
            # Save combined data in MultiIndex format
            combined_file = os.path.join(timeframe_dir, 'combined_data.parquet')
            combined_data.to_parquet(combined_file)
            
            # Also save as CSV for easier inspection
            csv_file = os.path.join(timeframe_dir, 'combined_data.csv')
            combined_data.to_csv(csv_file)
            
            logger.info(f"Saved combined data with {len(combined_data)} rows to {combined_file}")
            
            # Method 2: Also create a flattened version for compatibility with some systems
            flat_combined = pd.DataFrame()
            
            for symbol, df in all_dfs.items():
                # Rename columns to include symbol
                df_flat = df.copy()
                df_flat.columns = [f"{symbol}_{col}" for col in df_flat.columns]
                
                # Add to combined DataFrame
                if flat_combined.empty:
                    flat_combined = df_flat
                else:
                    flat_combined = flat_combined.join(df_flat, how='outer')
            
            # Save flattened combined data
            flat_file = os.path.join(timeframe_dir, 'flat_combined_data.parquet')
            flat_combined.to_parquet(flat_file)
            
            # Also save as CSV
            flat_csv = os.path.join(timeframe_dir, 'flat_combined_data.csv')
            flat_combined.to_csv(flat_csv)
            
            logger.info(f"Saved flattened combined data with {len(flat_combined)} rows to {flat_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating combined data: {str(e)}")
            return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Market Data Backfiller')
    
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT',
                       help='Comma-separated list of trading pairs')
    
    parser.add_argument('--timeframes', type=str, default='5m',
                       help='Comma-separated list of timeframes')
    
    parser.add_argument('--output-dir', type=str, default='data/market_data',
                       help='Directory to save data files')
    
    parser.add_argument('--periods', type=int, default=1000,
                       help='Number of candles to fetch')
    
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange to fetch data from')
    
    args = parser.parse_args()
    
    # Parse arguments
    symbols = args.symbols.split(',')
    timeframes = args.timeframes.split(',')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run backfiller
    backfiller = MarketDataBackfiller(
        symbols=symbols,
        timeframes=timeframes,
        output_dir=args.output_dir,
        periods=args.periods,
        exchange=args.exchange
    )
    
    success = backfiller.fetch_data()
    
    if success:
        logger.info("Market data backfill completed successfully")
    else:
        logger.error("Market data backfill failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 