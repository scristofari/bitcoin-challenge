import sqlite3
import bitcoin.db as db
import csv


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


create_tables()
print('Import to BTC EUR')
with open('BTC-EUR.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        db.insert_data(row + ['0'])
