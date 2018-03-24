from .db import get_all_data
from .log import logger
from .predition import Prediction
from .train import train, train_scaler
import pandas as pd

REGUL = 0.569999999999709


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

    logger.info('Done !')
    return df_computed, regul


def test_model():
    df = get_all_data()
    train_scaler(df=df)
    y = df[['close']].values.reshape(-1, 1)
    columns = ['open']
    history = train(df[columns].values, y)

    return test_computed(columns), history


def test_money():
    df = get_all_data()
    df_test = df[-int(.3 * len(df)):].reset_index()
    p = Prediction()
    cash = 1000
    bitcoins = last_bitcoin =  0
    for index, row in df_test.iterrows():
        open = row['open']
        close = last_bitcoin = row['close']
        y_predict = p.predict(open, regul=REGUL, load_model=(index == 0))
        if open < y_predict < close and bitcoins > 0:
            bitcoins = cash / y_predict
            cash = 0
        elif open >= y_predict > close and cash > 0:
            cash = bitcoins * y_predict

    if cash == 0:
        cash = bitcoins * last_bitcoin

    print(cash)
