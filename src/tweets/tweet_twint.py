import twint
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nest_asyncio

# Apply nest_asyncio to avoid asyncio errors
nest_asyncio.apply()

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to scrape tweets using Twint
def fetch_tweets(query, max_results=100):
    # Configure Twint
    c = twint.Config()
    c.Search = query  # Search query
    c.Limit = max_results  # Maximum number of tweets to fetch
    c.Lang = "en"  # Language filter (English)
    c.Pandas = True  # Store results in a Pandas DataFrame

    # Run Twint search
    twint.run.Search(c)

    # Fetch tweets into a DataFrame
    tweets_df = twint.storage.panda.Tweets_df

    # Extract relevant columns
    if not tweets_df.empty:
        tweets_df = tweets_df[["date", "tweet"]]
        tweets_df.columns = ["timestamp", "text"]
        return tweets_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no tweets found

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = vader_analyzer.polarity_scores(text)
    return sentiment["compound"]  # Compound score (-1 to +1)

# Main function to fetch and process data
def fetch_and_process_data(max_results=10):
    query = "(BTC OR Bitcoin OR Ethereum OR ETH OR SP500 OR SPX OR Nasdaq OR Dow OR stocks OR markets OR inflation OR recession OR interest rates OR CPI OR FED OR 'stock market' OR crypto OR 'market crash' OR 'bull market' OR 'bear market' OR '$BTC' OR '$ETH' OR '$SPY' OR '$QQQ' OR '$AAPL')"

    # Fetch tweets
    tweets_df = fetch_tweets(query, max_results)

    if not tweets_df.empty:
        # Analyze sentiment for each tweet
        tweets_df["sentiment"] = tweets_df["text"].apply(analyze_sentiment_vader)

        # Convert timestamp to datetime
        tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])

        # Sort by timestamp
        tweets_df = tweets_df.sort_values(by="timestamp").reset_index(drop=True)

    return tweets_df

# Example usage
# if __name__ == "__main__":
data = fetch_and_process_data(max_results=10)
print(data.head())

# Save data to CSV
if not data.empty:
    data.to_csv("twitter_sentiment_data_twint.csv", index=False)
else:
    print("No tweets found.")