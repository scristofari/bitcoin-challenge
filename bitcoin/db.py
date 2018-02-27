import sqlite3


def create_db():
    conn = sqlite3.connect('btc_challenge.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE btc_challenge (pid INTEGER PRIMARY KEY, time INTEGER, low REAL, high REAL, open REAL, close REAL, volume REAL, sentiment REAL)''')
    conn.commit()
    conn.close()


def insert_dataframe(df):
    conn = sqlite3.connect('btc_challenge.db')
    df.to_sql("btc_challenge", conn, if_exists="replace")
    conn.close()
