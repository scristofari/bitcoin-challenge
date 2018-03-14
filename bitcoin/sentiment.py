import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import tweepy
import re
from .log import logger

consumer_key = 'ZF9flFkwQH3vpGZstOteQDe3n'
consumer_secret = 'Z8xk8yrVVHK8EkOKN7NpcJ9c8XJVlMyLQRuAw8aZ5W4B7ovNsT'
access_token = '110371689-Q2B5wa5dmGZCiuVUlD6NLVkJgR1nYkbDYkOq7oxq'
access_token_secret = 'skrgjgRXhXng2QmIMJPMmw5JG4mGDSIFtZIpziEIjVJjo'


class Sentiment:
    from_gnews = None
    from_reddit = None
    from_twitter = []

    def build(self):
        self.build_from_gnews()
        self.build_from_reddit()
        self.build_from_twitter()

        logger.info('Build sentiment analysis => [%s, %s, %s]' % (self.from_gnews, self.from_reddit, self.from_twitter))

        return self

    def build_from_gnews(self):
        r = requests.get('https://finance.google.com/finance/company_news?q=currency:btc&output=json')
        resp = r.json()
        df = pd.DataFrame(columns=['polarity'])
        for p in resp['clusters']:
            if 'a' in p:
                text = '%s %s' % (p['a'][0]['t'], p['a'][0]['sp'])
                pol = TextBlob(text).sentiment.polarity
                df = df.append({
                    'polarity': pol
                }, ignore_index=True)

        self.from_gnews = np.mean(df.values)

    def build_from_reddit(self):
        r = requests.get('https://www.reddit.com/r/bitcoin/hot.json', headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        })
        list = r.json()
        posts = list['data']['children']
        df = pd.DataFrame(columns=['polarity'])
        for p in posts:
            pol = TextBlob(p['data']['title']).sentiment.polarity
            df = df.append({
                'polarity': pol
            }, ignore_index=True)

        self.from_reddit = np.mean(df.values)

    def build_from_twitter(self):
        def clean_tweet(tweet):
            return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

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
        self.from_twitter = [average, followers_sum]
