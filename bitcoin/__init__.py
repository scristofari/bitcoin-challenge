# from coinbase.wallet.client import Client
# client = Client(api_key='o1IFREMtdXCXVXBM', api_secret='qqLBoRG0kpIZT4TeTp2lv0aL3AQvZf5f', api_version='2017-12-31')
# price = client.get_spot_price(currency_pair='BTC-USD')
import requests
import json
import sqlite3
import time
import csv


def get_current_spot():
    headers = {'CB-VERSION': '2017-12-31'}
    r = requests.get('https://api.coinbase.com/v2/prices/%s/spot' % 'BTC-USD', headers=headers)
    if r.status_code != 200:
        print('ERR: Failed to get current spot')
        exit(1)

    price = json.loads(r.text)
    print("Currency : %s => %s %s" % (price['data']['base'], price['data']['amount'], price['data']['currency']))
    return price['data']


def history(granularity=300):
    import gdax
    import pandas as pd
    from datetime import datetime
    public_client = gdax.PublicClient()
    rates = public_client.get_product_historic_rates(product_id='BTC-USD', granularity=granularity)
    rates.reverse()

    df = pd.DataFrame(rates)
    df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']

    def human(timestamp):
        date = datetime.fromtimestamp(timestamp)
        return date.strftime('%Y-%m-%d %H:%M:%S')

    # df['date'] = df.apply(lambda row: human(row['time']), axis=1)

    def percent(op, close):
        return ((float(close) - op) / op) * 100

    df['percent'] = df.apply(lambda row: percent(row['open'], row['close']), axis=1)

    # df.to_csv('prices.csv', encoding='utf-8', mode='w+',
    #          header=('time', 'low', 'high', 'open', 'close', 'volume', 'percent'))

    return df


def insert_amount(price):
    conn = sqlite3.connect('btc_challenge.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS btc_challenge (pid INTEGER PRIMARY KEY, amount REAL, timestamp INTEGER)''')
    c.execute("INSERT INTO btc_challenge (amount, timestamp) VALUES (?, ?)", (float(price['amount']), int(time.time())))

    conn.commit()

    conn.close()


def create_dataset():
    conn = sqlite3.connect('btc_challenge.db')
    c = conn.cursor()
    c.execute("SELECT * FROM btc_challenge ORDER BY pid DESC LIMIT 180")
    rows = c.fetchall()
    conn.close()

    with open('prices.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(('pid', 'amount', 'timestamp'))
        writer.writerows(rows)


def predict():
    pass


def train():
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    prices = pd.read_csv('prices.csv')
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices['close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree_reg = DecisionTreeRegressor(max_depth=3)
    tree_reg.fit(X_train, y_train)

    # save the model to disk
    pickle.dump(tree_reg, open("model.sav", 'wb+'))
