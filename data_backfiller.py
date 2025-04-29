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
import traceback
import asyncio
import ta

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
            self.exchange = exchange
            self.exchange_instance = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            logger.info(f"Initialized exchange client for {exchange}")
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            self.exchange_instance = None
    
    async def _fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """
        Fetch historical OHLCV data for a symbol with pagination.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            since: Unix timestamp in milliseconds for start time
            limit: Maximum number of candles to fetch
            
        Returns:
            List of OHLCV candles
        """
        try:
            since_date = datetime.fromtimestamp(since/1000) if since else 'beginning'
            logger.info(f"Fetching historical data for {symbol} since {since_date}")
            
            # If the exchange is not initialized, return empty data
            if self.exchange_instance is None:
                logger.error(f"Exchange is not initialized")
                return []
            
            # Use limit if provided, otherwise use self.periods
            if limit is None:
                limit = self.periods
            
            # If using synchronous CCXT (not async version)
            if not HAS_ASYNC:
                all_ohlcv = []
                current_since = since
                
                while True:
                    # Fetch a batch of candles
                    ohlcv = self.exchange_instance.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
                    
                    # If no data or empty response, break
                    if not ohlcv or len(ohlcv) == 0:
                        break
                        
                    # Add to collected data
                    all_ohlcv.extend(ohlcv)
                    
                    # If we got less than the limit, we've reached the end
                    if len(ohlcv) < limit:
                        break
                        
                    # Update since parameter for next batch (add 1ms to avoid duplicates)
                    current_since = ohlcv[-1][0] + 1
                    
                    # Respect rate limits
                    time.sleep(self.exchange_instance.rateLimit / 1000)
                
                logger.info(f"Retrieved {len(all_ohlcv)} candles for {symbol}")
                return all_ohlcv
            
            # Using async CCXT
            else:
                async_exchange = None
                try:
                    # Initialize async exchange
                    exchange_class = getattr(ccxt_async, self.exchange)
                    async_exchange = exchange_class({
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future',
                        }
                    })
                    
                    all_ohlcv = []
                    current_since = since
                    
                    while True:
                        # Fetch a batch of candles
                        ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
                        
                        # If no data or empty response, break
                        if not ohlcv or len(ohlcv) == 0:
                            break
                            
                        # Add to collected data
                        all_ohlcv.extend(ohlcv)
                        
                        # If we got less than the limit, we've reached the end
                        if len(ohlcv) < limit:
                            break
                            
                        # Update since parameter for next batch (add 1ms to avoid duplicates)
                        current_since = ohlcv[-1][0] + 1
                        
                        # Rate limit is automatically handled by ccxt.async_support
                    
                    logger.info(f"Retrieved {len(all_ohlcv)} candles for {symbol}")
                    return all_ohlcv
                    
                finally:
                    # Close async exchange
                    if async_exchange:
                        await async_exchange.close()
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def fetch_data(self):
        """
        Fetch historical data for specified symbols and timeframes.
        Calculate technical indicators and save the data to files.
        """
        try:
            logger.info(f"Initializing exchange {self.exchange}")
            exchange_class = getattr(ccxt, self.exchange)
            exchange = exchange_class()
            
            if not hasattr(exchange, 'fetchOHLCV'):
                raise AttributeError(f"Exchange {self.exchange} does not support fetchOHLCV")
            
            # Process each timeframe
            for timeframe in self.timeframes:
                logger.info(f"Processing timeframe: {timeframe}")
                
                # Create a directory for this timeframe
                timeframe_dir = os.path.join(self.output_dir, timeframe)
                os.makedirs(timeframe_dir, exist_ok=True)
                
                # Dictionary to store DataFrames for all symbols
                all_symbol_data = {}
                
                # Fetch data for each symbol
                for symbol in self.symbols:
                    try:
                        logger.info(f"Fetching data for {symbol} with timeframe {timeframe}")
                        
                        # Fetch OHLCV data
                        ohlcv = exchange.fetchOHLCV(symbol, timeframe, limit=self.periods)
                        
                        if not ohlcv:
                            logger.warning(f"No data returned for {symbol}")
                            continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Calculate technical indicators
                        df = self._calculate_indicators(df)
                        
                        # Store the DataFrame
                        all_symbol_data[symbol] = df
                        
                        # Save individual symbol data
                        symbol_file = os.path.join(timeframe_dir, f"{symbol.replace('/', '')}.parquet")
                        df.to_parquet(symbol_file)
                        logger.info(f"Saved data for {symbol} to {symbol_file}")
                        
                        # Also save as CSV for easy viewing
                        csv_file = os.path.join(timeframe_dir, f"{symbol.replace('/', '')}.csv")
                        df.to_csv(csv_file)
                        
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
                        logger.error(traceback.format_exc())
                
                # Create combined file for all symbols
                if all_symbol_data:
                    # Create combined data with a 'symbol' column
                    combined_df = self._create_combined_data(all_symbol_data, timeframe_dir)
                    
                    if not combined_df.empty:
                        # Save combined data
                        combined_file = os.path.join(timeframe_dir, "combined_data.parquet")
                        combined_df.to_parquet(combined_file)
                        logger.info(f"Saved combined data to {combined_file}")
                        
                        # Also save as CSV
                        csv_file = os.path.join(timeframe_dir, "combined_data.csv")
                        combined_df.to_csv(csv_file)
                    
                    # Create MultiIndex version for the trading environment
                    multi_index_df = self._create_multi_index_dataframe(all_symbol_data)
                    
                    if not multi_index_df.empty:
                        # Save MultiIndex version
                        multi_file = os.path.join(timeframe_dir, "multi_index_data.parquet")
                        multi_index_df.to_parquet(multi_file)
                        logger.info(f"Saved MultiIndex data to {multi_file}")
                        
                        # Log MultiIndex structure for debugging
                        logger.info(f"MultiIndex columns: {multi_index_df.columns.names}")
                        logger.info(f"MultiIndex levels: {[list(multi_index_df.columns.levels[i])[:3] for i in range(len(multi_index_df.columns.levels))]}")
                else:
                    logger.warning(f"No data fetched for timeframe {timeframe}")
            
            logger.info("Data fetching completed successfully")
            
        except Exception as e:
            logger.error(f"Error in fetch_data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with price and volume data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid warnings
        result = df.copy()
        
        try:
            # Simple Moving Averages
            result['sma_7'] = ta.trend.sma_indicator(result['close'], window=7)
            result['sma_25'] = ta.trend.sma_indicator(result['close'], window=25)
            result['sma_99'] = ta.trend.sma_indicator(result['close'], window=99)
            
            # Exponential Moving Averages
            result['ema_9'] = ta.trend.ema_indicator(result['close'], window=9)
            result['ema_21'] = ta.trend.ema_indicator(result['close'], window=21)
            
            # MACD
            macd = ta.trend.MACD(result['close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_histogram'] = macd.macd_diff()
            
            # RSI
            result['rsi_14'] = ta.momentum.RSIIndicator(result['close'], window=14).rsi()
            
            # Bollinger Bands
            bbands = ta.volatility.BollingerBands(result['close'], window=20, window_dev=2.0)
            result['bb_upper'] = bbands.bollinger_hband()
            result['bb_middle'] = bbands.bollinger_mavg()
            result['bb_lower'] = bbands.bollinger_lband()
            
            # ATR - Average True Range
            result['atr'] = ta.volatility.AverageTrueRange(result['high'], result['low'], result['close'], window=14).average_true_range()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(result['high'], result['low'], result['close'])
            result['stoch_k'] = stoch.stoch()
            result['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            result['volume_sma_20'] = ta.trend.sma_indicator(result['volume'], window=20)
            
            # Additional momentum indicators
            result['adx'] = ta.trend.ADXIndicator(result['high'], result['low'], result['close'], window=14).adx()
            
            # Volatility indicators
            result['volatility'] = result['close'].pct_change().rolling(window=20).std() * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Fill NaN values that may be created during calculations
        result = result.ffill().bfill()
        
        return result
    
    def _create_combined_data(self, all_symbol_data, timeframe_dir):
        """
        Create a combined DataFrame from individual symbol DataFrames.
        
        Args:
            all_symbol_data (dict): Dictionary of DataFrames with symbol as key
            timeframe_dir (str): Directory path for the timeframe
            
        Returns:
            pandas.DataFrame: Combined DataFrame with a 'symbol' column
        """
        combined_data = []
        
        for symbol, df in all_symbol_data.items():
            # Create a copy to avoid modifying the original
            df_copy = df.copy()
            df_copy['symbol'] = symbol  # Add symbol column
            combined_data.append(df_copy.reset_index())  # Reset index to add timestamp as column
        
        # Combine all DataFrames
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Created combined DataFrame with {len(combined_df)} rows")
            return combined_df
        else:
            logger.warning("No data to combine")
            return pd.DataFrame()

    def _create_multi_index_dataframe(self, all_symbol_data):
        """
        Create a MultiIndex DataFrame from individual symbol DataFrames.
        This format is required for the trading environment.
        
        Args:
            all_symbol_data (dict): Dictionary of DataFrames with symbol as key
            
        Returns:
            pandas.DataFrame: DataFrame with MultiIndex columns (symbol, field)
        """
        if not all_symbol_data:
            logger.warning("No data to create MultiIndex DataFrame")
            return pd.DataFrame()
        
        # Ensure all DataFrames have the same index
        dfs = []
        common_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Get all column names from the first DataFrame to ensure we include all indicators
        first_df = next(iter(all_symbol_data.values()))
        all_columns = list(first_df.columns)
        
        # Process each symbol
        for symbol, df in all_symbol_data.items():
            clean_symbol = symbol.replace('/', '')  # Clean symbol name
            
            # Create a new DataFrame with renamed columns
            renamed_df = pd.DataFrame(index=df.index)
            
            # Add all columns with MultiIndex structure
            for col in all_columns:
                if col in df.columns:  # Make sure the column exists
                    renamed_df[(clean_symbol, col)] = df[col]
            
            dfs.append(renamed_df)
        
        # Merge all DataFrames on index
        try:
            multi_df = pd.concat(dfs, axis=1)
            
            # Make sure the index is a DateTimeIndex and is sorted
            if not isinstance(multi_df.index, pd.DatetimeIndex):
                multi_df.index = pd.to_datetime(multi_df.index)
            multi_df = multi_df.sort_index()
            
            # Create proper MultiIndex columns
            symbol_list = [s.replace('/', '') for s in self.symbols]
            fields = all_columns
            multi_df.columns = pd.MultiIndex.from_product([symbol_list, fields], 
                                                         names=['symbol', 'field'])
            
            logger.info(f"Created MultiIndex DataFrame with shape {multi_df.shape}")
            return multi_df
            
        except Exception as e:
            logger.error(f"Error creating MultiIndex DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    def _get_timeframe_seconds(self, timeframe):
        """
        Convert a timeframe string to seconds.
        
        Args:
            timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            int: Timeframe in seconds
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

def main():
    """
    Main function to run the backfiller from command line arguments.
    """
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='Backfill historical market data')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'], 
                        help='List of trading symbols (default: BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT)')
    parser.add_argument('--timeframes', nargs='+', default=['5m'], 
                        help='List of timeframes to fetch (default: 5m)')
    parser.add_argument('--output-dir', default='data/market_data', 
                        help='Output directory for saved data (default: data/market_data)')
    parser.add_argument('--exchange', default='binance', 
                        help='Exchange to fetch data from (default: binance)')
    parser.add_argument('--periods', type=int, default=1000,
                        help='Number of periods to fetch (default: 1000)')
    
    args = parser.parse_args()
    
    # Initialize the backfiller
    backfiller = MarketDataBackfiller(
        symbols=args.symbols,
        timeframes=args.timeframes,
        output_dir=args.output_dir,
        periods=args.periods,
        exchange=args.exchange
    )
    
    # Run the backfiller using asyncio
    try:
        asyncio.run(backfiller.fetch_data())
        logger.info("Data backfill completed successfully")
    except Exception as e:
        logger.error(f"Data backfill failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main() 