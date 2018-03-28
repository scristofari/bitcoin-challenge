import sqlite3


def insert_data(data):
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('INSERT INTO btceur VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)
    conn.commit()
    conn.close()


def get_last_predicted_price():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT predicted_price FROM btceur ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


def get_last_volume():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT volume FROM btceur ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


def get_last_price_and_volume():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT close, volume FROM btceur ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0], result[1]


def get_n2_price():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT close FROM btceur ORDER BY time DESC LIMIT 1 OFFSET 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


def insert_next_buy(price):
    import time

    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('INSERT INTO buy_history VALUES (?,?)', (int(time.time()), price))
    conn.commit()
    conn.close()


def get_last_buy_price():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT buy FROM buy_history ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


def get_all_data():
    import pandas as pd
    from .log import logger
    import numpy as np

    logger.info('Load data from SQL.')

    conn = sqlite3.connect("bitcoin.db")
    df = pd.read_sql_query("SELECT * from btceur ORDER BY time ASC", conn)
    conn.close()

    # @todo Hack, replace zeros by NaN and fill forward.
    def log(x):
        if isinstance(x, bytes):
            return 0.0
        return x

    df['google_sentiment'] = df['google_sentiment'].apply(log)
    df['google_sentiment'] = df['google_sentiment'].replace(0.0, np.NaN)
    df['google_sentiment'] = df['google_sentiment'].fillna(method='ffill')

    df.dropna(how='any', inplace=True)

    df['up'] = df['open'] < df['close']
    df['up'] = df['up'].replace(False, 0)
    df['up'] = df['up'].replace(True, 1)
    return df


def get_last_real_gnews_sentiment():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT google_sentiment FROM btceur WHERE google_sentiment > 0 ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


def get_all_data_from_pas(pas=5):
    import pandas as pd

    df = get_all_data()

    df_60 = pd.DataFrame(columns=['open', 'tw_sentiment', 'reddit_sentiment', 'google_sentiment'])
    for index, row in df.iterrows():
        if index % pas == 0:
            try:
                close = df['open'][index + (pas - 1)]
            except KeyError:
                close = row['close']
            df_60 = df_60.append({
                'time': row['time'],
                'open': row['open'],
                'tw_sentiment': row['tw_sentiment'],
                'reddit_sentiment': row['reddit_sentiment'],
                'google_sentiment': row['google_sentiment'],
                'close': close,
            }, ignore_index=True)

    diff = df_60['close'] - df_60['open']
    df_60['percent'] = diff / df_60['open'] * 100
    df_60['up'] = df_60['open'] < df_60['close']
    df_60['up'] = df_60['up'].replace(False, 0)
    df_60['up'] = df_60['up'].replace(True, 1)
    return df_60
