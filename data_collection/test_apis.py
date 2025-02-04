import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

TWITTER_BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAAPbuygEAAAAA8NW8nERCa%2BB0Y4xXOk90TRCToLU%3DaQbZA2e3BJtHegRNJpNcbuvYLZqICFSwXDutMNFLFRepp5LhQM"
NEWS_API_KEY="737d6bff71c1497ea2032a34e607d5cc"
ETHERSCAN_API_KEY="5176HU6CYDABAIPNMEM5PIXRP93RMMZ4GE"
GLASSNODE_API_KEY="your_glassnode_key"
SANTIMENT_API_KEY="4pify7t2xec2hv2c_untfy2a6akayl4mi"

def test_news_api():
    """Test NewsAPI connectivity"""
    print("\nTesting NewsAPI...")
    api_key = NEWS_API_KEY
    print(f"API Key: {api_key}")
    
    if not api_key:
        print("Error: NEWS_API_KEY not found in .env file")
        return
    
    try:
        # Add from_date parameter to avoid hitting rate limits
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': 'cryptocurrency OR bitcoin OR ethereum OR solana',
                'searchIn': 'title,description',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'from': from_date,
                'apiKey': api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {len(data['articles'])} articles")
            if data['articles']:
                print("\nSample article:")
                article = data['articles'][0]
                print(f"Title: {article['title']}")
                print(f"Source: {article['source']['name']}")
                print(f"Published: {article['publishedAt']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_etherscan_api():
    """Test Etherscan API connectivity"""
    print("\nTesting Etherscan API...")
    api_key = ETHERSCAN_API_KEY
    print(f"API Key: {api_key}")
    
    if not api_key:
        print("Error: ETHERSCAN_API_KEY not found in .env file")
        return
    
    try:
        # Test ethprice endpoint as it's simpler and doesn't require date parameters
        response = requests.get(
            'https://api.etherscan.io/api',
            params={
                'module': 'stats',
                'action': 'ethprice',
                'apikey': api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                print("Success! Current ETH price data:")
                print(json.dumps(data['result'], indent=2))
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
        # Test a second endpoint - ethsupply2
        response = requests.get(
            'https://api.etherscan.io/api',
            params={
                'module': 'stats',
                'action': 'ethsupply2',
                'apikey': api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                print("\nSuccess! ETH supply data:")
                print(json.dumps(data['result'], indent=2))
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_santiment_api():
    """Test Santiment API connectivity"""
    print("\nTesting Santiment API...")
    api_key = SANTIMENT_API_KEY
    print(f"API Key: {api_key}")
    
    if not api_key:
        print("Error: SANTIMENT_API_KEY not found in .env file")
        return
        
    try:
        # Use basic API key authentication for data fetching
        params = {
            'apikey': api_key,
            'slug': 'ethereum'
        }
        
        # Use a basic metrics endpoint instead of GraphQL
        url = 'https://api.santiment.net/api/v3/projects/ethereum'
        
        response = requests.get(url, params=params)
        
        print(f"Response Status: {response.status_code}")
        print("Response Text:")
        print(json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_blockchain_info():
    """Test Blockchain.info API connectivity"""
    print("\nTesting Blockchain.info API...")
    
    try:
        # Updated endpoint and parameters
        response = requests.get(
            'https://api.blockchain.info/charts/transactions-per-second',
            params={
                'timespan': '1days',
                'rollingAverage': '8hours',
                'format': 'json'
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {len(data['values'])} data points")
            if data['values']:
                print("\nSample data:")
                print(json.dumps(data['values'][0], indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Testing API endpoints...")
    print("=" * 50)
    
    test_news_api()
    print("\n" + "=" * 50)
    
    test_etherscan_api()
    print("\n" + "=" * 50)
    
    test_santiment_api()
    print("\n" + "=" * 50)
    
    test_blockchain_info()
    print("\n" + "=" * 50)
    
    print("\nAPI testing completed!") 