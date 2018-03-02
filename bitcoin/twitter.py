import tweepy
import re
import numpy as np
import pandas as pd
from textblob import TextBlob

consumer_key = 'ZF9flFkwQH3vpGZstOteQDe3n'
consumer_secret = 'Z8xk8yrVVHK8EkOKN7NpcJ9c8XJVlMyLQRuAw8aZ5W4B7ovNsT'
access_token = '110371689-Q2B5wa5dmGZCiuVUlD6NLVkJgR1nYkbDYkOq7oxq'
access_token_secret = 'skrgjgRXhXng2QmIMJPMmw5JG4mGDSIFtZIpziEIjVJjo'


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def get_sentiment():
    """
    Returns the sentiment

    :return:
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitter_api = tweepy.API(auth)

    tweets = twitter_api.search(q=['bitcoin, price, crypto, blockchain'], count=100)
    df = pd.DataFrame(columns=['followers', 'polarity'])
    for tweet in tweets:
        pol = TextBlob(clean_tweet(tweet.text)).sentiment.polarity
        df = df.append({
            'followers': tweet.user.followers_count,
            'polarity': pol
        }, ignore_index=True)

    average = np.average(df['polarity'].values, weights=df['followers'].values)
    followers_sum = df['followers'].sum()
    return [average, followers_sum]
