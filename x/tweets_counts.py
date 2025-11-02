import requests
import os
import json
from datetime import datetime, timedelta
import pandas as pd

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")

search_url = "https://api.twitter.com/2/tweets/counts/recent"

# Calculate start_time and end_time for last week
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

# Optional params: start_time,end_time,since_id,until_id,next_token,granularity
query_params = {
    'query': 'from:twitterdev',
    'granularity': 'hour',  # Changed to hourly granularity
    'start_time': start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    'end_time': end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentTweetCountsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def analyze_tweet_counts(json_response):
    """
    Analyze tweet counts using pandas
    """
    # Convert JSON data to DataFrame
    df = pd.DataFrame(json_response['data'])
    
    # Convert start time to datetime
    df['start'] = pd.to_datetime(df['start'])
    
    # Set start time as index
    df.set_index('start', inplace=True)
    
    # Add some basic analysis
    print("\nTweet Count Analysis:")
    print("-------------------")
    print(f"Total tweets: {df['tweet_count'].sum()}")
    print(f"\nHourly Statistics:")
    print(df['tweet_count'].describe())
    
    # Group by hour of day to see patterns
    df['hour'] = df.index.hour
    hourly_stats = df.groupby('hour')['tweet_count'].agg(['mean', 'sum', 'count'])
    print("\nTweet counts by hour of day:")
    print(hourly_stats.sort_values('sum', ascending=False))
    
    # Optional: Save to CSV
    df.to_csv('tweet_counts_analysis.csv')
    
    return df

def main():
    print(f"Fetching tweet counts from {start_time} to {end_time}")
    json_response = connect_to_endpoint(search_url, query_params)
    
    # Save raw response
    with open('tweet_counts_raw.json', 'w') as f:
        json.dump(json_response, indent=4, sort_keys=True, fp=f)
    
    # Analyze the data
    df = analyze_tweet_counts(json_response)
    
    # Create a simple visualization
    try:
        import matplotlib.pyplot as plt
        
        # Plot tweet counts over time
        plt.figure(figsize=(12, 6))
        df['tweet_count'].plot(kind='line', title='Tweet Counts Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('tweet_counts_timeline.png')
        
        # Plot average tweets by hour
        plt.figure(figsize=(10, 6))
        df.groupby('hour')['tweet_count'].mean().plot(
            kind='bar',
            title='Average Tweet Count by Hour'
        )
        plt.xlabel('Hour of Day (UTC)')
        plt.ylabel('Average Tweet Count')
        plt.tight_layout()
        plt.savefig('tweet_counts_by_hour.png')
        
    except ImportError:
        print("matplotlib not installed. Skipping visualizations.")

if __name__ == "__main__":
    main()