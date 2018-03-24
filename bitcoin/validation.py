from .db import get_all_data
from .log import logger
from .predition import Prediction
from .train import train, train_scaler
import pandas as pd


def test_computed(columns):
    df = get_all_data()
    # df = df[df['order_book_bids_price'] > 0].reset_index()

    count = df['open'].count()
    n_tests = int(0.3 * count)

    df_test = df[-n_tests:].reset_index()
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
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

    logger.info('Done !')
    return df_computed


def test_model():
    df = get_all_data()
    # df = df[df['order_book_bids_price'] > 0].reset_index()
    train_scaler(df=df)
    y = df[['close']].values.reshape(-1, 1)
    columns = ['open']
    #train(df[columns].values, y)
    return test_computed(columns)
