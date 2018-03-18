import sqlite3


def insert_data(data):
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('INSERT INTO btceur VALUES (?,?,?,?,?,?,?,?,?,?,?)', data)
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


def insert_next_buy(time, price):
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('INSERT INTO buy_history VALUES (?,?)', (time, price))
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]


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
    df['google_sentiment'] = df['google_sentiment'].replace(0.0, np.NaN)
    df['google_sentiment'] = df['google_sentiment'].fillna(method='ffill')

    df.dropna(how='any', inplace=True)
    return df


def get_last_real_gnews_sentiment():
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('SELECT google_sentiment FROM btceur WHERE google_sentiment > 0 ORDER BY time DESC LIMIT 1')
    result = c.fetchone()
    conn.commit()
    conn.close()

    return result[0]
