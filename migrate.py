import sqlite3
import bitcoin.db as db
import pandas as pd


def create_tables():
    print('Create DB')
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE btceur
                 (time INTEGER,low REAL, high REAL, open REAL, close REAL,
                  volume REAL, tw_sentiment REAL, tw_followers REAL,
                  reddit_sentiment REAL, google_sentiment REAL, predicted_price REAL)''')
    c.execute('''CREATE TABLE buy_history (time INTEGER, buy REAL)''')
    c.execute('''CREATE INDEX btceur_time ON btceur (time);''')
    c.execute('''CREATE INDEX buy_history_time ON buy_history (time);''')
    conn.commit()
    conn.close()


def migrate_0001():
    create_tables()
    print('Import to BTC EUR')
    df = pd.read_csv('BTC-EUR.csv',
                     names=['time', 'low', 'high', 'open', 'close', 'volume', 'tw_sentiment', 'tw_followers',
                            'reddit_sentiment', 'google_sentiment']
                     )
    df.dropna(how='any', inplace=True)
    df['predicted_price'] = 0.0
    for index, row in df.iterrows():
        db.insert_data(row)


def migrate_0002():
    print('Clean DB')
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute("DELETE FROM btceur WHERE open == 6767.54")
    conn.commit()
    conn.close()

def migrate_0003():
    print('Add columns DB')
    conn = sqlite3.connect('bitcoin.db')
    c = conn.cursor()
    c.execute("DELETE FROM btceur WHERE open == 6767.54")
    conn.commit()
    conn.close()