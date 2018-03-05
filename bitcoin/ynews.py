import requests
import pandas as pd
import numpy as np
from textblob import TextBlob

def get_sentiment():
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

    print([np.mean(df.values)])
