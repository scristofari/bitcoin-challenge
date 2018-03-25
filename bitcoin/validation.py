from .db import get_all_data
from .log import logger
from .predition import Prediction
from .train import train, train_scaler, train_anomaly
from datetime import datetime
import pandas as pd


def test_computed(columns):
    df = get_all_data()
    df_test = df[-int(.3 * len(df)):]
    df_test = df_test.reset_index()
    logger.info('Test set of %d items !' % len(df_test))
    p = Prediction()
    df_computed = pd.DataFrame(columns=['real', 'predicted', 'diff'])
    for index, row in df_test.iterrows():
        close = row['close']

        X = []
        for c in columns:
            X.append(row[c])

        y_predict = p.predict(X, load_model=(index == 0))

        df_computed = df_computed.append({
            'real': close,
            'predicted': y_predict,
            'diff': close - y_predict
        }, ignore_index=True)

    regul = df_computed['diff'].median()
    df_computed['predicted_regul'] = df_computed['predicted'] + regul
    df_computed['diff_regul'] = df_computed['real'] - (df_computed['predicted'] + regul)

    logger.info('Regul %f !' % regul)
    logger.info('Done !')
    return df_computed, regul


def test_model():
    df = get_all_data()
    train_scaler(df=df)
    #train_anomaly(df=df)

    y = df[['close']].values.reshape(-1, 1)
    columns = ['open', 'tw_sentiment', 'tw_followers']
    #history = train(df[columns].values, y)
    #(df, regul), history = test_computed(columns), history
    test_money(columns, regul=0.0)


def test_money(columns, regul=0.0):
    from sklearn.externals import joblib
    import numpy as np

    df = get_all_data()
    df_test = df[-int(.3 * len(df)):].reset_index()
    p = Prediction()
    last_cash = cash = 1000
    bitcoins = last_bitcoin = last_price = last_volume = 0
    model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
    anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 99)))

    for index, row in df_test.iterrows():
        open = row['open']
        close = last_bitcoin = row['close']
        X = []
        for c in columns:
            X.append(row[c])

        if last_volume == 0:
            last_volume = row['volume']
            last_price = row['open']
        y_predict = p.predict(X, regul=regul, load_model=(index == 0))

        anomaly = np.exp(model_anomaly.score(last_volume))
        # if bitcoins > 0 and anomaly < anomaly_limit and last_price > open < close:
        #    logger.info('ANOMALY -> SELL')
        #    cash = (open + 0.01) * bitcoins
        #    bitcoins = 0
        if open >= y_predict > close and cash > 0:
            logger.info('BUY')
            last_cash = cash
            bitcoins = cash / y_predict
            cash = 0
        elif open <= y_predict < close and last_cash < bitcoins * y_predict and bitcoins > 0:
            logger.info('SELL')
            cash = bitcoins * y_predict
            bitcoins = 0

        last_volume = row['volume']
        last_price = row['open']

    if cash == 0:
        cash = bitcoins * last_bitcoin

    from_date = datetime.fromtimestamp(df_test[0:1]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    to_date = datetime.fromtimestamp(df_test[-1:]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    logger.info("TEST From %s to %s" % (from_date, to_date))

    logger.info("With prediction %.2f euros" % (cash))

    bitcoin_first = 1000 / df_test[0:1]['open'].values
    cash_last = bitcoin_first * float(df_test[-1:]['open'].values)
    logger.info("Without prediction %.2f euros" % cash_last)
