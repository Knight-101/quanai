import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# import tweepy  # Commented out as not needed
import requests
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from dotenv import load_dotenv
import ccxt
import ta
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
# from transformers import pipeline  # Commented out as not needed

# Load environment variables
load_dotenv()

# Comment out unused API keys
# TWITTER_BEARER_TOKEN="..."
# NEWS_API_KEY="..."
# ETHERSCAN_API_KEY="..."
# GLASSNODE_API_KEY="..."
# SANTIMENT_API_KEY="..."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
        
    def wait(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < 1.0 / self.calls_per_second:
            time.sleep(1.0 / self.calls_per_second - time_since_last_call)
        self.last_call_time = time.time()

class MultiModalDataCollector:
    def __init__(self):
        # Initialize API clients with retry mechanism
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Comment out unused API clients and keys
        # self.news_api_key = NEWS_API_KEY
        # self.etherscan_api_key = ETHERSCAN_API_KEY
        # self.glassnode_api_key = GLASSNODE_API_KEY
        # self.santiment_api_key = SANTIMENT_API_KEY
        
        # Comment out sentiment analyzer
        # self.sentiment_analyzer = pipeline(
        #     "sentiment-analysis",
        #     model="finiteautomata/bertweet-base-sentiment-analysis"
        # )
        
        # Initialize rate limiters
        # self.news_rate_limiter = RateLimiter(0.5)  # 2 seconds between calls
        # self.onchain_rate_limiter = RateLimiter(0.2)  # 5 seconds between calls
        
        # Define supported assets
        self.assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
    def collect_price_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect OHLCV data with technical indicators"""
        all_price_data = []
        
        for asset in self.assets:
            try:
                symbol = f"{asset}/USDT"
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe='1h',
                    since=int(start_date.timestamp() * 1000),
                    limit=None
                )
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['asset'] = asset
                
                # Add technical indicators
                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                df['macd'] = ta.trend.MACD(df['close']).macd()
                df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
                df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
                
                all_price_data.append(df)
                
            except Exception as e:
                logger.error(f"Error collecting price data for {asset}: {str(e)}")
                
        return pd.concat(all_price_data, ignore_index=True)
    
    def collect_historical_tweets(
        self,
        start_date: datetime,
        end_date: datetime,
        min_engagement: int = 10
    ) -> pd.DataFrame:
        """Collect historical tweets with sentiment analysis"""
        all_tweets = []
        
        for asset in self.assets:
            query = f"#{asset} OR {asset} crypto -is:retweet lang:en"
            
            try:
                tweets = self.twitter_client.search_all_tweets(
                    query=query,
                    start_time=start_date,
                    end_time=end_date,
                    tweet_fields=['created_at', 'public_metrics', 'author_id'],
                    max_results=100
                )
                
                for tweet in tweets.data:
                    # Filter by minimum engagement
                    if tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count'] < min_engagement:
                        continue
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer(tweet.text)[0]
                    
                    all_tweets.append({
                        'asset': asset,
                        'timestamp': tweet.created_at,
                        'text': tweet.text,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'replies': tweet.public_metrics['reply_count'],
                        'sentiment_label': sentiment['label'],
                        'sentiment_score': sentiment['score']
                    })
                    
            except Exception as e:
                logger.error(f"Error collecting tweets for {asset}: {str(e)}")
        
        return pd.DataFrame(all_tweets)
    
    def collect_news(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect crypto news articles with sentiment analysis"""
        all_news = []
        
        for asset in self.assets:
            try:
                self.news_rate_limiter.wait()
                
                response = self.session.get(
                    'https://newsapi.org/v2/everything',
                    params={
                        'q': f'{asset} cryptocurrency',
                        'from': start_date.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d'),
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'apiKey': self.news_api_key
                    }
                )
                
                if response.status_code == 200:
                    articles = response.json()['articles']
                    for article in articles:
                        # Analyze sentiment of title and description
                        title_sentiment = self.sentiment_analyzer(article['title'])[0]
                        desc_sentiment = self.sentiment_analyzer(article['description'])[0]
                        
                        all_news.append({
                            'asset': asset,
                            'timestamp': article['publishedAt'],
                            'title': article['title'],
                            'description': article['description'],
                            'source': article['source']['name'],
                            'title_sentiment': title_sentiment['label'],
                            'title_sentiment_score': title_sentiment['score'],
                            'desc_sentiment': desc_sentiment['label'],
                            'desc_sentiment_score': desc_sentiment['score']
                        })
                        
            except Exception as e:
                logger.error(f"Error collecting news for {asset}: {str(e)}")
        
        return pd.DataFrame(all_news)
    
    def collect_onchain_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive on-chain metrics"""
        onchain_data = {}
        
        # Commenting out whale activity data collection
        # try:
        #     whale_data = self.whale_tracker.get_whale_activity(start_date, end_date)
        #     if not whale_data.empty:
        #         for asset in self.assets:
        #             asset_whale_data = whale_data[whale_data['asset'] == asset].copy()
        #             if not asset_whale_data.empty:
        #                 whale_metrics = asset_whale_data.set_index('timestamp').resample('1H').agg({
        #                     'amount': 'sum',
        #                     'amount_usd': 'sum',
        #                     'from_address': 'count',
        #                     'whale_address': 'nunique'
        #                 }).reset_index()
        #                 whale_metrics['metric'] = 'whale_activity'
        #                 onchain_data[asset] = whale_metrics
        # except Exception as e:
        #     logger.error(f"Error collecting whale activity data: {str(e)}")
        
        # Get Ethereum on-chain data from Etherscan
        try:
            self.onchain_rate_limiter.wait()
            
            # Get multiple metrics
            metrics = ['dailynetutilization', 'dailynewaddress', 'dailytxn']
            eth_data = []
            
            for metric in metrics:
                response = self.session.get(
                    'https://api.etherscan.io/api',
                    params={
                        'module': 'stats',
                        'action': metric,
                        'startdate': start_date.strftime('%Y-%m-%d'),
                        'enddate': end_date.strftime('%Y-%m-%d'),
                        'apikey': self.etherscan_api_key
                    }
                )
                
                if response.status_code == 200:
                    metric_data = pd.DataFrame(response.json()['result'])
                    metric_data['metric'] = metric
                    eth_data.append(metric_data)
            
            if eth_data:
                eth_df = pd.concat(eth_data)
                eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='s')
                onchain_data['ETH'] = eth_df
                
        except Exception as e:
            logger.error(f"Error collecting Ethereum on-chain data: {str(e)}")
        
        # Get Bitcoin network stats from blockchain.info
        try:
            self.onchain_rate_limiter.wait()
            
            btc_metrics = {
                'n_transactions': 'n-transactions',
                'mempool_size': 'mempool-size',
                'hash_rate': 'hash-rate',
                'difficulty': 'difficulty'
            }
            
            btc_data = []
            for metric_name, endpoint in btc_metrics.items():
                response = self.session.get(
                    f'https://api.blockchain.info/charts/{endpoint}',
                    params={
                        'timespan': f'{(end_date - start_date).days}days',
                        'format': 'json'
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data['values'])
                    df['timestamp'] = pd.to_datetime(df['x'], unit='s')
                    df['metric'] = metric_name
                    df = df.rename(columns={'y': 'value'})
                    btc_data.append(df[['timestamp', 'metric', 'value']])
            
            if btc_data:
                btc_df = pd.concat(btc_data)
                onchain_data['BTC'] = btc_df
                
        except Exception as e:
            logger.error(f"Error collecting Bitcoin on-chain data: {str(e)}")
        
        return onchain_data
    
    def collect_market_sentiment(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect market sentiment data from Santiment"""
        try:
            response = self.session.get(
                'https://api.santiment.net/graphql',
                headers={'Authorization': f'Bearer {self.santiment_api_key}'},
                json={
                    'query': '''
                    {
                        getMetric(metric: "sentiment_volume_consumed_total") {
                            timeseriesData(
                                slug: "bitcoin"
                                from: "${start_date}"
                                to: "${end_date}"
                                interval: "1h"
                            ) {
                                datetime
                                value
                            }
                        }
                    }
                    '''.replace('${start_date}', start_date.isoformat())
                       .replace('${end_date}', end_date.isoformat())
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data']['getMetric']['timeseriesData'])
                df['timestamp'] = pd.to_datetime(df['datetime'])
                return df
                
        except Exception as e:
            logger.error(f"Error collecting market sentiment: {str(e)}")
            return pd.DataFrame()
    
    def align_and_merge_data(
        self,
        price_data: pd.DataFrame,
        tweet_data: pd.DataFrame,
        news_data: pd.DataFrame,
        onchain_data: Dict[str, pd.DataFrame],
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align and merge all data sources with improved handling"""
        # Start with price data as the base and sort by timestamp
        merged_data = price_data.copy()
        merged_data = merged_data.sort_values('timestamp')
        
        # Process tweets if available
        if not tweet_data.empty and 'timestamp' in tweet_data.columns:
            tweet_features = (
                tweet_data.set_index('timestamp')
                .groupby(['asset', pd.Grouper(freq='1h')])
                .agg({
                    'likes': 'sum',
                    'retweets': 'sum',
                    'replies': 'sum',
                    'text': lambda x: ' '.join(x),
                    'sentiment_score': 'mean'
                })
                .reset_index()
                .sort_values('timestamp')  # Sort tweet features
            )
            
            for asset in self.assets:
                asset_tweets = tweet_features[tweet_features['asset'] == asset].copy()
                if not asset_tweets.empty:
                    merged_data = pd.merge_asof(
                        merged_data,
                        asset_tweets,
                        on='timestamp',
                        by='asset',
                        direction='backward'
                    )
        
        # Process news if available and has required columns
        required_news_columns = ['timestamp', 'title', 'description', 'title_sentiment_score', 'desc_sentiment_score']
        if not news_data.empty and all(col in news_data.columns for col in required_news_columns):
            news_features = (
                news_data.set_index('timestamp')
                .groupby(['asset', pd.Grouper(freq='1h')])
                .agg({
                    'title': lambda x: ' '.join(x),
                    'description': lambda x: ' '.join(x),
                    'title_sentiment_score': 'mean',
                    'desc_sentiment_score': 'mean'
                })
                .reset_index()
                .sort_values('timestamp')  # Sort news features
            )
            
            for asset in self.assets:
                asset_news = news_features[news_features['asset'] == asset].copy()
                if not asset_news.empty:
                    merged_data = pd.merge_asof(
                        merged_data,
                        asset_news,
                        on='timestamp',
                        by='asset',
                        direction='backward'
                    )
        else:
            logger.warning("News data is empty or missing required columns. Skipping news data merge.")
        
        # Process on-chain data
        for asset in self.assets:
            if asset in onchain_data and not onchain_data[asset].empty:
                onchain_df = onchain_data[asset]
                if 'metric' in onchain_df.columns and 'value' in onchain_df.columns:
                    onchain_df = onchain_df.pivot(
                        index='timestamp',
                        columns='metric',
                        values='value'
                    ).reset_index()
                    onchain_df = onchain_df.sort_values('timestamp')  # Sort onchain data
                    
                    merged_data = pd.merge_asof(
                        merged_data,
                        onchain_df,
                        on='timestamp',
                        direction='backward'
                    )
        
        # Merge market sentiment if available
        if not sentiment_data.empty and 'timestamp' in sentiment_data.columns and 'value' in sentiment_data.columns:
            sentiment_df = sentiment_data[['timestamp', 'value']].rename(columns={'value': 'market_sentiment'})
            sentiment_df = sentiment_df.sort_values('timestamp')  # Sort sentiment data
            merged_data = pd.merge_asof(
                merged_data,
                sentiment_df,
                on='timestamp',
                direction='backward'
            )
        
        # Fill missing values with forward fill then backward fill
        merged_data = merged_data.ffill().bfill()
        
        return merged_data
    
    def collect_and_save_data(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ):
        """Main function to collect and save all data"""
        logger.info("Starting data collection...")
        
        # Initialize empty DataFrames - comment out unused ones
        price_data = pd.DataFrame()
        # tweets = pd.DataFrame()
        # news = pd.DataFrame()
        # onchain = {}
        # sentiment = pd.DataFrame()
        
        # Collect price data (most reliable source)
        try:
            logger.info("Collecting price data...")
            price_data = self.collect_price_data(start_date, end_date)
            if price_data.empty:
                logger.error("Failed to collect price data. Aborting collection.")
                return
        except Exception as e:
            logger.error(f"Error collecting price data: {str(e)}")
            return
        
        # Save directly without merging since we're only using OHLCV data
        logger.info(f"Saving data to {output_path}")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            price_data.to_parquet(output_path, index=False)
            
            logger.info("Data collection completed!")
            
            # Log collection statistics
            logger.info("\nCollection Statistics:")
            logger.info(f"Price data rows: {len(price_data)}")
            
        except Exception as e:
            logger.error(f"Error in final data processing: {str(e)}")
            raise

if __name__ == "__main__":
    collector = MultiModalDataCollector()
    
    # Collect last 7 days of data as a test
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    collector.collect_and_save_data(
        start_date=start_date,
        end_date=end_date,
        output_path='data/test_multi_modal.parquet'
    ) 