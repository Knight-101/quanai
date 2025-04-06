#!/usr/bin/env python3
"""
Data Fetcher for Backtesting

This script fetches OHLCV data for BTC, ETH, and SOL, adds technical indicators,
and combines them into a format suitable for the institutional backtesting system.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path
import sys

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add TA-Lib based indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using pandas-ta for indicators")

# Add pandas-ta as a fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas-ta not available, will use basic indicators only")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestDataFetcher")

class BacktestDataFetcher:
    """Fetches and processes data for backtesting purposes"""
    
    def __init__(self, 
                 symbols: List[str] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                 timeframe: str = '5m',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 lookback_days: int = 1825,  # 5 years by default
                 exchange: str = 'binance'):
        """
        Initialize the backtesting data fetcher.
        
        Args:
            symbols: List of trading pairs to fetch
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            lookback_days: Number of days to look back if start_date is not provided
            exchange: Trading exchange to fetch from
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.exchange_id = exchange
        self.output_dir = 'data'
        
        # Process date range
        self.end_date = datetime.now()
        if end_date:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = self.end_date - timedelta(days=lookback_days)
            
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        # Simple symbol mappings for exchanges
        self.symbol_mappings = {
            'BTC/USDT': 'BTC',
            'ETH/USDT': 'ETH',
            'SOL/USDT': 'SOL'
        }
        
        # Initialize exchange
        self.init_exchange()
    
    def init_exchange(self):
        """Initialize the exchange API client"""
        try:
            # Initialize both sync and async versions
            self.exchange = getattr(ccxt, self.exchange_id)({
                'enableRateLimit': True,
            })
            
            self.async_exchange = getattr(ccxt_async, self.exchange_id)({
                'enableRateLimit': True,
            })
            
            logger.info(f"Successfully initialized {self.exchange_id} exchange")
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise
    
    async def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols"""
        data_dict = {}
        
        try:
            await self.async_exchange.load_markets()
            
            # Fetch data for each symbol
            tasks = []
            for symbol in self.symbols:
                tasks.append(self.fetch_ohlcv(symbol))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {self.symbols[i]}: {str(result)}")
                    continue
                
                symbol, df = result
                if df is not None and not df.empty:
                    data_dict[symbol] = df
                    logger.info(f"Fetched {len(df)} rows for {symbol}")
                else:
                    logger.warning(f"No data fetched for {symbol}")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return {}
        finally:
            await self.async_exchange.close()
    
    async def fetch_ohlcv(self, symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Fetch OHLCV data for a specific symbol"""
        try:
            # Convert dates to timestamps
            since = int(self.start_date.timestamp() * 1000)
            until = int(self.end_date.timestamp() * 1000)
            
            # Fetch data with pagination
            all_candles = []
            while since < until:
                candles = await self.async_exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=self.timeframe, 
                    since=since,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since for next iteration
                since = candles[-1][0] + 1
                
                # Respect rate limits
                await asyncio.sleep(self.async_exchange.rateLimit / 1000)
            
            if not all_candles:
                logger.warning(f"No data returned for {symbol}")
                return symbol, None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add basic returns
            df['returns'] = df['close'].pct_change()
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Clean up any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Map to simple symbol name
            simple_symbol = self.symbol_mappings.get(symbol, symbol)
            
            return simple_symbol, df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return symbol, None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame"""
        # Extract price data
        open_price = df['open']
        high_price = df['high']
        low_price = df['low']
        close_price = df['close']
        volume = df['volume']
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = close_price.rolling(window=window).mean()
            df[f'ema_{window}'] = close_price.ewm(span=window, adjust=False).mean()
        
        # Add MA crossovers for trend identification
        df['ma_crossover_20_50'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
        df['ma_crossover_50_200'] = np.where(df['ma_50'] > df['ma_200'], 1, -1)
        
        # Bollinger Bands
        df['bb_middle'] = df['ma_20']
        bb_std = close_price.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # MACD
        df['ema_12'] = close_price.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close_price.ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        if TALIB_AVAILABLE:
            try:
                df['rsi'] = talib.RSI(close_price.values, timeperiod=14)
            except:
                delta = close_price.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
        elif PANDAS_TA_AVAILABLE:
            df['rsi'] = ta.rsi(close_price, length=14)
        else:
            # Calculate RSI without libraries
            delta = close_price.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            try:
                df['stoch_k'], df['stoch_d'] = talib.STOCH(
                    high_price.values, 
                    low_price.values, 
                    close_price.values,
                    fastk_period=14,
                    slowk_period=3,
                    slowd_period=3
                )
            except:
                # Fallback calculation
                n = 14
                df['stoch_k'] = 100 * ((close_price - low_price.rolling(n).min()) / 
                                     (high_price.rolling(n).max() - low_price.rolling(n).min()))
                df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        elif PANDAS_TA_AVAILABLE:
            stoch = ta.stoch(high_price, low_price, close_price, k=14, d=3, smooth_k=3)
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        else:
            # Basic stochastic calculation
            n = 14
            df['stoch_k'] = 100 * ((close_price - low_price.rolling(n).min()) / 
                                 (high_price.rolling(n).max() - low_price.rolling(n).min()))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ATR for volatility
        if TALIB_AVAILABLE:
            try:
                df['atr'] = talib.ATR(high_price.values, low_price.values, close_price.values, timeperiod=14)
            except:
                tr1 = high_price - low_price
                tr2 = abs(high_price - close_price.shift())
                tr3 = abs(low_price - close_price.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(14).mean()
        elif PANDAS_TA_AVAILABLE:
            df['atr'] = ta.atr(high_price, low_price, close_price, length=14)
        else:
            # Basic ATR calculation
            tr1 = high_price - low_price
            tr2 = abs(high_price - close_price.shift())
            tr3 = abs(low_price - close_price.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        
        # ADX for trend strength
        if TALIB_AVAILABLE:
            try:
                df['adx'] = talib.ADX(high_price.values, low_price.values, close_price.values, timeperiod=14)
            except:
                # ADX is complex - we'll skip in the fallback
                df['adx'] = np.nan
        elif PANDAS_TA_AVAILABLE:
            adx = ta.adx(high_price, low_price, close_price, length=14)
            df['adx'] = adx['ADX_14']
        else:
            # Set to NaN if we can't calculate
            df['adx'] = np.nan
        
        # Volume indicators
        df['volume_ma'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_ma']
        
        # Volatility indicators
        df['returns_volatility'] = df['returns'].rolling(window=20).std()
        
        # Clean up NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all data into a single MultiIndex DataFrame"""
        if not data_dict:
            raise ValueError("No data to combine")
        
        # Make sure all DataFrames have the same index
        common_index = None
        for symbol, df in data_dict.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        logger.info(f"Common index size: {len(common_index)}")
        
        # Reindex and fill missing values
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_data[symbol] = df.reindex(common_index).fillna(method='ffill').fillna(method='bfill').fillna(0)
            logger.info(f"Aligned {symbol} data: {aligned_data[symbol].shape}")
        
        # Create multi-index columns
        combined_data = {}
        for symbol, df in aligned_data.items():
            for col in df.columns:
                combined_data[(symbol, col)] = df[col]
        
        # Convert to DataFrame with MultiIndex
        multi_df = pd.DataFrame(combined_data)
        
        # Make sure the MultiIndex is properly formatted
        multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns, names=['asset', 'feature'])
        
        logger.info(f"Final combined shape: {multi_df.shape}")
        return multi_df
    
    async def fetch_and_combine(self) -> pd.DataFrame:
        """Fetch data for all symbols and combine into a single DataFrame"""
        # Fetch data for each symbol
        data_dict = await self.fetch_all_data()
        
        # Check if we have data
        if not data_dict:
            raise ValueError("No data was fetched")
        
        # Combine into a single DataFrame
        combined_df = self.combine_data(data_dict)
        
        return combined_df
    
    async def run(self, output_path: str = None) -> pd.DataFrame:
        """Run the data fetcher and save the result"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Default output path
            if output_path is None:
                start_str = self.start_date.strftime('%Y%m%d')
                end_str = self.end_date.strftime('%Y%m%d')
                symbols_str = '_'.join(self.symbol_mappings.values())
                output_path = os.path.join(self.output_dir, f"market_data_{symbols_str}_{start_str}_{end_str}.parquet")
            
            # Fetch and combine data
            logger.info("Fetching and combining data...")
            combined_df = await self.fetch_and_combine()
            
            # Save to parquet
            combined_df.to_parquet(output_path)
            logger.info(f"Saved combined data to {output_path}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error running data fetcher: {str(e)}")
            raise

async def main():
    """Main function to run the data fetcher"""
    parser = argparse.ArgumentParser(description="Fetch and combine data for backtesting")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                        help="Symbols to fetch (default: BTC/USDT ETH/USDT SOL/USDT)")
    parser.add_argument("--timeframe", type=str, default="5m",
                       help="Timeframe (default: 5m)")
    parser.add_argument("--start-date", type=str,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--lookback-days", type=int, default=1825,
                       help="Number of days to look back (default: 1825 - 5 years)")
    parser.add_argument("--exchange", type=str, default="binance",
                       help="Exchange to fetch from (default: binance)")
    parser.add_argument("--output", type=str,
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize data fetcher
    data_fetcher = BacktestDataFetcher(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        exchange=args.exchange
    )
    
    # Run the fetcher
    await data_fetcher.run(args.output)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 