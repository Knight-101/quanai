import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetDataFetcher:
    def __init__(self, 
                 symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], 
                 timeframe='5m',
                 days=7):
        """
        Initialize the data fetcher
        
        Args:
            symbols: List of trading pairs to fetch
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, etc.)
            days: Number of days of historical data to fetch
        """
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # Use futures market
                'defaultMarket': 'linear',
                'defaultMarginMode': 'cross'
            },
            'enableRateLimit': True
        })
            
        self.symbols = symbols
        self.timeframe = timeframe
        self.days = days
        
        # Mapping for symbol format if needed
        self.symbol_mappings = {
            'BTC/USDT': 'BTCUSDT',
            'ETH/USDT': 'ETHUSDT',
            'SOL/USDT': 'SOLUSDT'
        }
        
        # Ensure output directory exists
        os.makedirs('data', exist_ok=True)
        
    def fetch_asset_data(self):
        """Fetch data for each symbol and save to parquet files"""
        
        logger.info(f"Fetching {self.days} days of data for: {', '.join(self.symbols)}")
        
        for sym in self.symbols:
            try:
                # Get correct symbol format for exchange
                exchange_symbol = self.symbol_mappings.get(sym, sym)
                
                # Calculate timestamp for start date
                since = int((datetime.now() - timedelta(days=self.days)).timestamp() * 1000)
                
                # Fetch OHLCV data
                logger.info(f"Fetching data for {sym}...")
                all_ohlcv = []
                
                while True:
                    ohlcv = self.exchange.fetch_ohlcv(exchange_symbol, self.timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    if len(ohlcv) < 1000:  # Less than limit means we've reached the end
                        break
                    since = ohlcv[-1][0] + 1  # Next timestamp after the last received
                    self.exchange.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
                
                if not all_ohlcv:
                    logger.warning(f"No data found for {sym}")
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Fetch funding rate history with pagination
                try:
                    all_funding = []
                    funding_since = int((datetime.now() - timedelta(days=self.days)).timestamp() * 1000)
                    
                    while True:
                        funding = self.exchange.fetch_funding_rate_history(exchange_symbol, since=funding_since, limit=1000)
                        if not funding:
                            break
                        all_funding.extend(funding)
                        if len(funding) < 1000:
                            break
                        funding_since = funding[-1]['timestamp'] + 1
                        self.exchange.sleep(self.exchange.rateLimit / 1000)
                    
                    if all_funding:
                        funding_df = pd.DataFrame(all_funding)
                        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                        funding_df.set_index('timestamp', inplace=True)
                        df['funding_rate'] = funding_df['fundingRate']
                        
                        # Forward fill funding rates since they update less frequently
                        df['funding_rate'] = df['funding_rate'].ffill()
                    else:
                        df['funding_rate'] = 0
                except Exception as e:
                    logger.warning(f"Could not fetch funding data for {exchange_symbol}: {str(e)}")
                    df['funding_rate'] = 0
                
                # Try to fetch order book for most recent data
                try:
                    ob = self.exchange.fetch_order_book(exchange_symbol)
                    if ob and ob.get('bids') and ob.get('asks'):
                        df['bid_depth'] = ob['bids'][0][1]
                        df['ask_depth'] = ob['asks'][0][1]
                    else:
                        df['bid_depth'] = 0
                        df['ask_depth'] = 0
                except Exception as e:
                    logger.warning(f"Could not fetch orderbook data for {exchange_symbol}: {str(e)}")
                    df['bid_depth'] = 0
                    df['ask_depth'] = 0
                
                # Add placeholder columns to match format
                df['oi'] = 0  # Open Interest placeholder
                df['liquidations'] = 0  # Liquidations placeholder
                
                # Remove duplicates and ensure sorted index
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
                
                # Save to parquet file
                clean_symbol = sym.replace('/', '_').replace(':', '_')
                filename = f"data/{clean_symbol}_{self.days}days_{self.timeframe}.parquet"
                df.to_parquet(filename)
                logger.info(f"Saved {len(df)} rows of data to {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {sym}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Fetch historical data for crypto assets')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                        help='List of symbols to fetch (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe for candles (1m, 5m, 15m, 1h, etc.)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days of historical data to fetch')
    
    args = parser.parse_args()
    
    fetcher = AssetDataFetcher(
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days
    )
    
    fetcher.fetch_asset_data()
    
if __name__ == "__main__":
    main() 