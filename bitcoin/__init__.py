import sqlite3
import time
import csv


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
