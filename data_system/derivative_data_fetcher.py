import ccxt.async_support as ccxt_async  # Use async version of CCXT
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerpetualDataFetcher:
    def __init__(self, 
                 symbols=['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD'], 
                 timeframe='5m',
                 exchanges=['binance']):  # Default to only Binance
        # Initialize Binance exchange
        self.exchange = ccxt_async.binance({
            'options': {
                'defaultType': 'future',
                'defaultMarket': 'linear',  # Use linear contracts by default
                'defaultMarginMode': 'cross'
            },
            'enableRateLimit': True
        })
            
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = 365*5  # 7 years
        
        
        # Binance-specific symbol mappings
        self.symbol_mappings = {
            'BTC/USD:USD': 'BTCUSDT',
            'ETH/USD:USD': 'ETHUSDT',
            'SOL/USD:USD': 'SOLUSDT'
        }
        
    async def fetch_derivative_data(self):
        """Fetch data from Binance with improved error handling"""
        try:
            # Load markets first
            await self.exchange.load_markets()
            
            data = await self._fetch_exchange_data()
            if data:
                return {'binance': data}
                
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            if 'symbol' in str(e).lower():
                logger.error("Symbol error - check if the trading pair exists on Binance Futures")
        finally:
            await self.exchange.close()  # Properly close exchange connection
        
        return {}
    
    async def _fetch_exchange_data(self):
        """Fetch data from Binance"""
        data = {}
        for sym in self.symbols:
            try:
                # Get Binance-specific symbol format
                exchange_symbol = self.symbol_mappings.get(sym, sym)
                
                # Set market type to futures
                self.exchange.options['defaultType'] = 'future'
                
                # Fetch OHLCV data with pagination
                since = int((datetime.now() - timedelta(days=self.lookback)).timestamp() * 1000)
                all_ohlcv = []
                
                while True:
                    ohlcv = await self.exchange.fetch_ohlcv(exchange_symbol, self.timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    if len(ohlcv) < 1000:  # Less than limit means we've reached the end
                        break
                    since = ohlcv[-1][0] + 1  # Next timestamp after the last received
                    await asyncio.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
                
                if not all_ohlcv:
                    continue
                    
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                try:
                    # Fetch funding rate history with pagination
                    all_funding = []
                    funding_since = since
                    
                    while True:
                        funding = await self.exchange.fetch_funding_rate_history(exchange_symbol, since=funding_since, limit=1000)
                        if not funding:
                            break
                        all_funding.extend(funding)
                        if len(funding) < 1000:
                            break
                        funding_since = funding[-1]['timestamp'] + 1
                        await asyncio.sleep(self.exchange.rateLimit / 1000)
                    
                    if all_funding:
                        funding_df = pd.DataFrame(all_funding)
                        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                        funding_df.set_index('timestamp', inplace=True)
                        df['funding_rate'] = funding_df['fundingRate']  # Binance uses 'fundingRate'
                        
                        # Forward fill funding rates since they update less frequently
                        df['funding_rate'] = df['funding_rate'].ffill()
                        
                except Exception as e:
                    logger.warning(f"Could not fetch funding data for {exchange_symbol}: {str(e)}")
                    df['funding_rate'] = 0
                
                try:
                    # Try to fetch order book
                    ob = await self.exchange.fetch_order_book(exchange_symbol)
                    if ob and ob.get('bids') and ob.get('asks'):
                        df['bid_depth'] = ob['bids'][0][1]
                        df['ask_depth'] = ob['asks'][0][1]
                except Exception as e:
                    logger.warning(f"Could not fetch orderbook data for {exchange_symbol}: {str(e)}")
                    df['bid_depth'] = 0
                    df['ask_depth'] = 0
                
                # Add default values for other columns
                df['oi'] = 0  # Open Interest
                df['liquidations'] = 0  # Liquidations
                
                # Remove duplicates and ensure sorted index
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
                
                data[sym] = df
                
                # Add delay to respect rate limits
                await asyncio.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching {sym}: {str(e)}")
                continue
                
        return data

    async def _fetch_ohlcv(self, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=self.lookback)).timestamp() * 1000)
            return await self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=1000)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return None

    async def _fetch_funding(self, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=self.lookback)).timestamp() * 1000)
            return await self.exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {str(e)}")
            return None

    async def _fetch_liquidations(self, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=self.lookback)).timestamp() * 1000)
            return await self.exchange.fetch_liquidations(symbol, since=since)
        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol}: {str(e)}")
            return None

    async def _fetch_orderbook(self, symbol, session):
        try:
            return await self.exchange.fetch_order_book(symbol, limit=20)
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {str(e)}")
            return None

    def _process_liquidations(self, liquidations, index):
        """Enhanced liquidation processing with side-specific metrics"""
        liq_df = pd.DataFrame(liquidations)
        if len(liq_df) == 0:
            return pd.Series(0, index=index)
            
        liq_df['timestamp'] = pd.to_datetime(liq_df['timestamp'], unit='ms')
        liq_df = liq_df.set_index('timestamp').resample('5T').agg({
            'price': 'mean',
            'side': lambda x: (x == 'buy').sum() - (x == 'sell').sum(),
            'amount': 'sum'
        })
        
        # Add liquidation imbalance metric
        liq_df['liq_imbalance'] = liq_df['side'] / liq_df['amount'].clip(lower=1)
        
        return liq_df.reindex(index, method='ffill')