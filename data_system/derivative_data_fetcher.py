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
                 exchanges=['binance', 'bybit', 'okx']):
        self.exchanges = {}
        for ex in exchanges:
            # Use the async version of CCXT
            exchange_class = getattr(ccxt_async, ex)
            exchange_instance = exchange_class({
                'options': {
                    'defaultType': 'future',
                    'defaultMarket': 'linear',  # Use linear contracts by default
                    'defaultMarginMode': 'cross'
                },
                'enableRateLimit': True
            })
            self.exchanges[ex] = exchange_instance
            
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = 365 * 3  # 3 years
        
        # Exchange-specific symbol mappings
        self.symbol_mappings = {
            'bybit': {
                'BTC/USD:USD': 'BTCUSDT',
                'ETH/USD:USD': 'ETHUSDT',
                'SOL/USD:USD': 'SOLUSDT'
            },
            'binance': {
                'BTC/USD:USD': 'BTCUSDT',
                'ETH/USD:USD': 'ETHUSDT',
                'SOL/USD:USD': 'SOLUSDT'
            },
            'okx': {
                'BTC/USD:USD': 'BTC-USDT-SWAP',
                'ETH/USD:USD': 'ETH-USDT-SWAP',
                'SOL/USD:USD': 'SOL-USDT-SWAP'
            }
        }
        
    async def fetch_derivative_data(self):
        """Fetch data from multiple exchanges for cross-validation"""
        combined_data = {}
        
        for ex_name, exchange in self.exchanges.items():
            try:
                # Load markets first
                await exchange.load_markets()
                
                ex_data = await self._fetch_exchange_data(exchange)
                if ex_data:
                    combined_data[ex_name] = ex_data
                    
            except Exception as e:
                logger.error(f"Error fetching data from {ex_name}: {str(e)}")
                continue
            finally:
                await exchange.close()  # Properly close exchange connection
        
        return combined_data
    
    async def _fetch_exchange_data(self, exchange):
        """Fetch data from a single exchange"""
        data = {}
        for sym in self.symbols:
            try:
                # Get exchange-specific symbol format
                exchange_symbol = self.symbol_mappings.get(exchange.id, {}).get(sym, sym)
                
                # Set appropriate market type for the exchange
                if exchange.id == 'bybit':
                    exchange.options['defaultType'] = 'linear'
                elif exchange.id == 'binance':
                    exchange.options['defaultType'] = 'future'
                elif exchange.id == 'okx':
                    exchange.options['defaultType'] = 'swap'
                
                # Fetch OHLCV data first
                since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
                ohlcv = await exchange.fetch_ohlcv(exchange_symbol, self.timeframe, since=since, limit=1000)
                if not ohlcv:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                try:
                    # Try to fetch funding rate history with exchange-specific handling
                    funding = await exchange.fetch_funding_rate_history(exchange_symbol, since=since, limit=1000)
                    if funding:
                        funding_df = pd.DataFrame(funding)
                        
                        # Handle different funding rate data structures
                        if 'fundingRate' in funding_df.columns:
                            rate_col = 'fundingRate'
                        elif 'rate' in funding_df.columns:
                            rate_col = 'rate'
                        else:
                            raise ValueError(f"Unknown funding rate column format: {funding_df.columns}")
                        
                        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                        funding_df.set_index('timestamp', inplace=True)
                        df['funding_rate'] = funding_df[rate_col]
                        
                        # Forward fill funding rates since they update less frequently
                        df['funding_rate'] = df['funding_rate'].ffill()
                        
                except Exception as e:
                    logger.warning(f"Could not fetch funding data for {exchange_symbol} on {exchange.id}: {str(e)}")
                    df['funding_rate'] = 0
                
                try:
                    # Try to fetch order book
                    ob = await exchange.fetch_order_book(exchange_symbol)
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
                
                data[sym] = df
                
                # Add delay to respect rate limits
                await asyncio.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching {sym} from {exchange.id}: {str(e)}")
                continue
                
        return data

    def _reconcile_exchange_data(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Smart reconciliation of data from different exchanges"""
        weights = [df['volume'].values for df in dfs]
        total_weight = np.sum(weights, axis=0)
        
        # VWAP-weighted price consolidation
        consolidated = pd.DataFrame()
        for col in ['open', 'high', 'low', 'close']:
            values = np.sum([df[col].values * w for df, w in zip(dfs, weights)], axis=0)
            consolidated[col] = values / total_weight
        
        # Sum volumes and OI
        consolidated['volume'] = np.sum([df['volume'].values for df in dfs], axis=0)
        consolidated['oi'] = np.sum([df['oi'].values for df in dfs], axis=0)
        
        # Average funding rates (volume-weighted)
        f_rates = np.sum([df['funding_rate'].values * w for df, w in zip(dfs, weights)], axis=0)
        consolidated['funding_rate'] = f_rates / total_weight
        
        # Combine liquidations
        consolidated['liquidations'] = np.sum([df['liquidations'].values for df in dfs], axis=0)
        
        # Add market depth
        consolidated['bid_depth'] = np.mean([df['bid_depth'].values for df in dfs], axis=0)
        consolidated['ask_depth'] = np.mean([df['ask_depth'].values for df in dfs], axis=0)
        
        return consolidated

    async def _fetch_ohlcv(self, exchange, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            return await exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=1000)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return None

    async def _fetch_funding(self, exchange, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            return await exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {str(e)}")
            return None

    async def _fetch_liquidations(self, exchange, symbol, session):
        try:
            since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            return await exchange.fetch_liquidations(symbol, since=since)
        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol}: {str(e)}")
            return None

    async def _fetch_orderbook(self, exchange, symbol, session):
        try:
            return await exchange.fetch_order_book(symbol, limit=20)
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