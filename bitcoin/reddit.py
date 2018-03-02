import requests
from textblob import TextBlob
import pandas as pd
import numpy as np


def get_sentiment():
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

    return np.mean(df.values)
