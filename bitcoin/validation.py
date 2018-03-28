from .db import get_all_data, get_all_data_from_pas
from .log import logger
from .predition import Prediction
from .train import train, train_scaler, train_anomaly, train_classification
from datetime import datetime
import pandas as pd


def test_computed(df, columns):
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


def test_money(columns, df, regul=0.0):
    df_test = df[-int(.3 * len(df)):].reset_index()
    p = Prediction()
    last_cash = cash = 1000
    bitcoins = last_bitcoin = 0

    for index, row in df_test.iterrows():
        open = row['open']
        close = last_bitcoin = row['close']
        X = []
        for c in columns:
            X.append(row[c])

        y_predict = p.predict(X, regul=regul, load_model=(index == 0))

        if open >= y_predict > close and cash > 0:
            logger.info('BUY')
            bitcoins = cash / y_predict
            last_cash = cash
            cash = 0
        elif open >= y_predict < close and bitcoins > 0 and last_cash < bitcoins * y_predict:
            logger.info('SELL')
            cash = bitcoins * y_predict
            bitcoins = 0

    if cash == 0:
        cash = bitcoins * last_bitcoin

    from_date = datetime.fromtimestamp(df_test[0:1]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    to_date = datetime.fromtimestamp(df_test[-1:]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    logger.info("TEST From %s to %s" % (from_date, to_date))

    logger.info("With prediction %.2f euros" % (cash))

    bitcoin_first = 1000 / df_test[0:1]['open'].values
    cash_last = bitcoin_first * float(df_test[-1:]['open'].values)
    logger.info("Without prediction %.2f euros" % cash_last)


def test_money_fee(columns, df, regul=0.0):
    df_test = df[-int(.3 * len(df)):].reset_index()
    p = Prediction()
    last_cash = cash = 1000
    bitcoins = last_bitcoin = 0

    for index, row in df_test.iterrows():
        open = row['open']
        last_bitcoin = row['close']
        X = []
        for c in columns:
            X.append(row[c])

        y_predict = p.predict(X, regul=regul, load_model=(index == 0))

        if open <= y_predict and cash > 0:
            logger.info('BUY')
            cash = cash - (.25 * cash / 100)
            bitcoins = cash / open
            last_cash = cash
            cash = 0
        elif open > y_predict and bitcoins > 0:
            cash_test = bitcoins * open
            cash_test = cash_test - (.25 * cash / 100)
            logger.info('SELL')
            bitcoins = 0
            cash = cash_test

    if cash == 0:
        cash = bitcoins * last_bitcoin

    from_date = datetime.fromtimestamp(df_test[0:1]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    to_date = datetime.fromtimestamp(df_test[-1:]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    logger.info("TEST From %s to %s" % (from_date, to_date))

    logger.info("With prediction %.2f euros" % (cash))

    bitcoin_first = 1000 / df_test[0:1]['open'].values
    cash_last = bitcoin_first * float(df_test[-1:]['open'].values)
    logger.info("Without prediction %.2f euros" % cash_last)

    
def test_money_step(df, num=1):
    df_test = df[-int(.3 * len(df)):].reset_index()
    p = Prediction()
    last_cash = cash = 1000
    bitcoins = last_bitcoin = 0
    nu = nd = i = 0
    for index, row in df_test.iterrows():
        if index == i:
            side = row['close'] - row['open']
            if side < 0:
                i = i + 1
                continue
        open = row['open']
        last_bitcoin = row['close']
        if side > 0 and cash > 0:
            nd = 0
            nu = nu + 1
            if nu == num:
                logger.info('BUY')
                cash = cash - (.25 * cash / 100)
                bitcoins = cash / open
                last_cash = cash
                cash = 0
                side = -1
                nu = 0
        elif side < 0 and bitcoins > 0:
            nu = 0
            nd = nd + 1
            if nd == num:
                cash_test = bitcoins * open
                cash_test = cash_test - (.25 * cash / 100)
                logger.info('SELL')
                bitcoins = 0
                cash = cash_test
                side = 1
                nd = 0

    if cash == 0:
        cash = bitcoins * last_bitcoin

    from_date = datetime.fromtimestamp(df_test[0:1]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    to_date = datetime.fromtimestamp(df_test[-1:]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
    logger.info("TEST From %s to %s" % (from_date, to_date))

    logger.info("With prediction %.2f euros" % (cash))

    bitcoin_first = 1000 / df_test[0:1]['open'].values
    cash_last = bitcoin_first * float(df_test[-1:]['open'].values)
    logger.info("Without prediction %.2f euros" % cash_last)


def how_many_up_and_down(df=None):
    if df is None:
        return
    result = pd.DataFrame(columns=['n', 'side', 'percent'])
    n = 0
    last_side = None
    last_price = None
    for _, row in df.iterrows():
        if last_side is None:
            last_side = row['up']
            last_price = row['open']
        side = row['up']
        if last_side == side:
            n = n + 1
        else:
            percent = 100 * (row['open'] - last_price) / last_price
            result = result.append({
                'n': n,
                'side': last_side,
                'percent': percent
            }, ignore_index=True)
            n = 1
            last_price = row['open']
        last_side = side

    print("n => median => %d" % int(result['n'].median()))
    return result
