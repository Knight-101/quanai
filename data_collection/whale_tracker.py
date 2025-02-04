import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List
import logging
from dotenv import load_dotenv
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
logger = logging.getLogger(__name__)

class WhaleTracker:
    def __init__(self):
        self.whale_alert_key = os.getenv('WHALE_ALERT_KEY')
        self.nansen_key = os.getenv('NANSEN_API_KEY')
        self.arkham_key = os.getenv('ARKHAM_API_KEY')
        
        # Thresholds for whale detection (in USD)
        self.thresholds = {
            'BTC': {
                'min_value': 5_000_000,  # $5M minimum for BTC
                'large_transfer': 10_000_000,  # $10M for large transfers
                'whale_balance': 100_000_000  # $100M for whale status
            },
            'ETH': {
                'min_value': 2_000_000,  # $2M minimum for ETH
                'large_transfer': 5_000_000,  # $5M for large transfers
                'whale_balance': 50_000_000  # $50M for whale status
            },
            'SOL': {
                'min_value': 500_000,  # $500K minimum for SOL
                'large_transfer': 1_000_000,  # $1M for large transfers
                'whale_balance': 10_000_000  # $10M for whale status
            }
        }
        
        # Metric weights for scoring
        self.weights = {
            'transaction_volume': 0.3,    # Volume of transfers
            'wallet_balance': 0.2,        # Total holdings
            'transfer_frequency': 0.15,   # How often they move funds
            'smart_money_flow': 0.2,      # Nansen's smart money tracking
            'market_impact': 0.15         # Historical price impact
        }
        
        # Known whale addresses (top holders, exchanges, etc.)
        self.whale_addresses = {
            'BTC': [
                '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',  # Binance
                '3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r',  # Bitfinex
                '1LQoWist8KkaUXSPKZHNvEyfrEkPHzSsCd',  # Huobi
                '3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B',  # Bitfinex Cold Wallet
                '385cR5DM96n1HvBDMzLHPYcw89fZAXULJP',  # Binance Cold Wallet
                '1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s'   # Binance Hot Wallet
            ],
            'ETH': [
                '0x28c6c06298d514db089934071355e5743bf21d60',  # Binance
                '0x21a31ee1afc51d94c2efccaa2092ad1028285549',  # Binance 2
                '0xdfd5293d8e347dfe59e90efd55b2956a1343963d',  # Wintermute
                '0x9bf4001d307dfd62b26a2f1307ee0c0307632d59',  # Binance Cold
                '0x4976a4a02f38326660d17bf34b431dc6e2eb2327',  # Kraken
                '0xdc24316b9ae028f1497c275eb9192a3ea0f67022'   # Jump Trading
            ],
            'SOL': [
                'HN4bujwwwGgBmBqhxe6B8qpJGwYkzRPfaF3RuSPjNbQJ',  # Alameda
                '9LzCMqDgTKYz9Drzqnpgee3SGa89up3a247ypMj2xrqM',  # Genesis
                'DQyrAcCrDXQ7NeoqGgDCZwBvWDcYmFCjSb9JtteuvPpz',  # Binance
                'HxhWkVpk5NS4Ltg5nij2G671CKXFRKM8Vg7NmNVCaYJP',  # FTX Hot
                'EzYB1JhP7XEvHou2PePBUX3TeXwgyR3eP1GqjrVwESXj'   # Kraken
            ]
        }
        
        # Initialize session with retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
    
    def get_whale_transactions(
        self,
        start_date: datetime,
        end_date: datetime,
        asset: str = None
    ) -> pd.DataFrame:
        """Get large transactions from Whale Alert API with asset-specific thresholds"""
        transactions = []
        
        try:
            min_value = self.thresholds[asset]['min_value'] if asset else 1_000_000
            
            response = self.session.get(
                'https://api.whale-alert.io/v1/transactions',
                params={
                    'api_key': self.whale_alert_key,
                    'start': int(start_date.timestamp()),
                    'end': int(end_date.timestamp()),
                    'min_value': min_value,
                    'cursor': ''
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                for tx in data['transactions']:
                    # Calculate transaction impact score
                    impact_score = self._calculate_impact_score(
                        tx['amount_usd'],
                        tx['symbol'],
                        tx['from'].get('owner_type'),
                        tx['to'].get('owner_type')
                    )
                    
                    transactions.append({
                        'timestamp': pd.to_datetime(tx['timestamp'], unit='s'),
                        'asset': tx['symbol'],
                        'amount': tx['amount'],
                        'amount_usd': tx['amount_usd'],
                        'from_address': tx['from']['address'],
                        'to_address': tx['to']['address'],
                        'from_owner_type': tx['from'].get('owner_type'),
                        'to_owner_type': tx['to'].get('owner_type'),
                        'impact_score': impact_score,
                        'is_large_transfer': tx['amount_usd'] >= self.thresholds.get(
                            tx['symbol'], {'large_transfer': float('inf')}
                        )['large_transfer']
                    })
        
        except Exception as e:
            logger.error(f"Error fetching whale transactions: {str(e)}")
        
        return pd.DataFrame(transactions)
    
    def _calculate_impact_score(
        self,
        amount_usd: float,
        asset: str,
        from_type: str,
        to_type: str
    ) -> float:
        """Calculate the potential market impact score of a transaction"""
        # Base score from transaction size
        if asset not in self.thresholds:
            return 0.0
            
        thresholds = self.thresholds[asset]
        base_score = min(1.0, amount_usd / thresholds['whale_balance'])
        
        # Adjust based on sender/receiver type
        type_multiplier = 1.0
        if from_type == 'unknown' and to_type == 'unknown':
            type_multiplier = 1.2  # Unknown addresses are more significant
        elif 'exchange' in (from_type, to_type):
            type_multiplier = 0.8  # Exchange transfers less significant
            
        return base_score * type_multiplier
    
    def get_nansen_whale_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get whale analytics from Nansen with smart money flow analysis"""
        whale_data = {}
        
        try:
            for asset in ['ETH', 'BTC', 'SOL']:
                response = self.session.post(
                    'https://api.nansen.ai/v1/smart-money',
                    headers={'X-API-KEY': self.nansen_key},
                    json={
                        'blockchain': asset.lower(),
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'interval': '1h',
                        'min_value': self.thresholds[asset]['min_value']
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Calculate smart money score
                    df['smart_money_score'] = df.apply(
                        lambda row: self._calculate_smart_money_score(row, asset),
                        axis=1
                    )
                    
                    whale_data[asset] = df
        
        except Exception as e:
            logger.error(f"Error fetching Nansen data: {str(e)}")
        
        return whale_data
    
    def _calculate_smart_money_score(self, row: pd.Series, asset: str) -> float:
        """Calculate smart money influence score"""
        # Weights for different components
        weights = {
            'volume': 0.4,
            'unique_addresses': 0.3,
            'profit_ratio': 0.3
        }
        
        # Normalize metrics
        volume_score = min(1.0, row['volume'] / self.thresholds[asset]['large_transfer'])
        address_score = min(1.0, row['unique_addresses'] / 100)  # Assume 100 is max
        profit_score = min(1.0, max(0.0, row['profit_ratio']))
        
        return (
            volume_score * weights['volume'] +
            address_score * weights['unique_addresses'] +
            profit_score * weights['profit_ratio']
        )
    
    def get_arkham_whale_flows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get detailed whale flow data from Arkham Intelligence"""
        whale_flows = {}
        
        try:
            for asset, addresses in self.whale_addresses.items():
                all_flows = []
                
                for address in addresses:
                    response = self.session.get(
                        'https://api.arkhamintelligence.com/v1/flows',
                        headers={'Authorization': f'Bearer {self.arkham_key}'},
                        params={
                            'address': address,
                            'start_time': start_date.isoformat(),
                            'end_time': end_date.isoformat(),
                            'interval': '1h'
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        flows = pd.DataFrame(data['flows'])
                        flows['whale_address'] = address
                        all_flows.append(flows)
                
                if all_flows:
                    whale_flows[asset] = pd.concat(all_flows)
                    whale_flows[asset]['timestamp'] = pd.to_datetime(whale_flows[asset]['timestamp'])
        
        except Exception as e:
            logger.error(f"Error fetching Arkham data: {str(e)}")
        
        return whale_flows
    
    def aggregate_whale_metrics(
        self,
        transactions_df: pd.DataFrame,
        nansen_data: Dict[str, pd.DataFrame],
        arkham_flows: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate whale activity metrics with weighted scoring"""
        all_metrics = []
        
        for asset in ['BTC', 'ETH', 'SOL']:
            asset_txs = transactions_df[transactions_df['asset'] == asset]
            
            if not asset_txs.empty:
                # Calculate hourly metrics
                hourly_metrics = asset_txs.set_index('timestamp').resample('1H').agg({
                    'amount': 'sum',
                    'amount_usd': 'sum',
                    'from_address': 'count',
                    'impact_score': 'mean'
                }).reset_index()
                
                hourly_metrics['asset'] = asset
                
                # Add volume score
                max_volume = self.thresholds[asset]['large_transfer']
                hourly_metrics['volume_score'] = hourly_metrics['amount_usd'].apply(
                    lambda x: min(1.0, x / max_volume)
                ) * self.weights['transaction_volume']
                
                # Add frequency score
                max_freq = 20  # Assume 20 transactions per hour is maximum
                hourly_metrics['frequency_score'] = hourly_metrics['from_address'].apply(
                    lambda x: min(1.0, x / max_freq)
                ) * self.weights['transfer_frequency']
                
                # Add Nansen metrics if available
                if asset in nansen_data:
                    nansen_metrics = nansen_data[asset]
                    hourly_metrics = pd.merge_asof(
                        hourly_metrics,
                        nansen_metrics[['timestamp', 'smart_money_score']],
                        on='timestamp',
                        direction='backward'
                    )
                    hourly_metrics['smart_money_score'] = hourly_metrics['smart_money_score'].fillna(0) * self.weights['smart_money_flow']
                
                # Add market impact score
                hourly_metrics['market_impact_score'] = hourly_metrics['impact_score'] * self.weights['market_impact']
                
                # Calculate final whale activity score
                hourly_metrics['whale_activity_score'] = (
                    hourly_metrics['volume_score'] +
                    hourly_metrics['frequency_score'] +
                    hourly_metrics.get('smart_money_score', 0) +
                    hourly_metrics['market_impact_score']
                )
                
                all_metrics.append(hourly_metrics)
        
        if all_metrics:
            return pd.concat(all_metrics)
        return pd.DataFrame()
    
    def get_whale_activity(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Main method to collect all whale activity data"""
        all_transactions = pd.DataFrame()
        nansen_data = {}
        arkham_flows = {}
        
        # Check if we have any API keys configured
        if not any([self.whale_alert_key, self.nansen_key, self.arkham_key]):
            logger.warning("No whale tracking API keys configured. Skipping whale activity collection.")
            return pd.DataFrame()
        
        # Collect whale transactions if API key is available
        if self.whale_alert_key:
            try:
                logger.info("Collecting whale transactions...")
                all_transactions = self.get_whale_transactions(start_date, end_date)
                if not all_transactions.empty:
                    logger.info(f"Collected {len(all_transactions)} whale transactions")
            except Exception as e:
                logger.error(f"Error fetching whale transactions: {str(e)}")
        else:
            logger.warning("Skipping whale transactions - Whale Alert API key not configured")
        
        # Collect Nansen data if API key is available
        if self.nansen_key:
            try:
                logger.info("Collecting Nansen analytics...")
                nansen_data = self.get_nansen_whale_data(start_date, end_date)
                if nansen_data:
                    logger.info(f"Collected Nansen data for {list(nansen_data.keys())} assets")
            except Exception as e:
                logger.error(f"Error fetching Nansen data: {str(e)}")
        else:
            logger.warning("Skipping Nansen analytics - Nansen API key not configured")
        
        # Collect Arkham data if API key is available
        if self.arkham_key:
            try:
                logger.info("Collecting Arkham flow data...")
                arkham_flows = self.get_arkham_whale_flows(start_date, end_date)
                if arkham_flows:
                    logger.info(f"Collected Arkham flows for {list(arkham_flows.keys())} assets")
            except Exception as e:
                logger.error(f"Error fetching Arkham data: {str(e)}")
        else:
            logger.warning("Skipping Arkham flows - Arkham API key not configured")
        
        # Aggregate available metrics
        try:
            logger.info("Aggregating whale metrics...")
            if all_transactions.empty and not nansen_data and not arkham_flows:
                logger.warning("No whale activity data available to aggregate")
                return pd.DataFrame()
            
            return self.aggregate_whale_metrics(
                all_transactions,
                nansen_data,
                arkham_flows
            )
        except Exception as e:
            logger.error(f"Error aggregating whale metrics: {str(e)}")
            return pd.DataFrame() 