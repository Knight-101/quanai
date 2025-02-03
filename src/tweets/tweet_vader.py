import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Twitter API credentials
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFvNygEAAAAAppPTTPQCSNywpW%2BIGxg4XxisRus%3D1Ze8QR0ekY6EUdgZs9rrV7yrt83S72Cb6XPFQSpHl2XoQPYgzh"  # Replace with your Twitter API Bearer Token

# Initialize Tweepy Client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to fetch tweets using Twitter API v2
def fetch_tweets(query, max_results=10):
    tweet_data = []
    
    # Search tweets using Tweepy
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["created_at", "text"],
        user_fields=["username"],
        expansions=["author_id"]
    )
    
    # Extract tweet data
    for tweet in tweets.data:
        text = tweet.text
        sentiment = analyze_sentiment_vader(text)
        tweet_data.append({
            "source": "twitter",
            "text": text,
            "sentiment": sentiment,
            "timestamp": tweet.created_at,
            "username": tweet.author_id  # You can map this to usernames using tweets.includes
        })
    
    return tweet_data

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = vader_analyzer.polarity_scores(text)
    return sentiment["compound"]  # Compound score (-1 to +1)

# Main function to fetch and process data
def fetch_and_process_data(max_results=10):
    query = "(BTC OR ETH OR SOL)"
    tweets = fetch_tweets(query, max_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    return df


# Example usage

data = fetch_and_process_data(max_results=10)
print(data.head())

# Save data to CSV
data.to_csv("twitter_sentiment_data_vader.csv", index=False)