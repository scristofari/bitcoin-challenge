import csv
from . import twitter, rates, reddit, gnews


def generate_spot_data():
    reddit_sentiment = reddit.get_sentiment()
    twitter_sentiment = twitter.get_sentiment()
    gnews_sentiment = gnews.get_sentiment()
    currencies = ['BTC-USD', 'BTC-EUR']
    for c in currencies:
        rate = rates.last_rate(c)
        # rate / twitter / reddit / gnews
        with open('%s.csv' % c, newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rate + twitter_sentiment + reddit_sentiment + gnews_sentiment)
