from .db import get_all_data
from .log import logger
from .predition import Prediction
from .train import train, train_scaler


def test_get_error_percent(columns):
    df = get_all_data()
    df = df[df['order_book_bids_price'] > 0].reset_index()

    n_error = 0
    count = df['open'].count()
    n_tests = int(0.3 * count)

    df_test = df[-n_tests:].reset_index()
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    p = Prediction()
    for index, row in df_test.iterrows():
        open = row['open']
        close = row['close']

        X = []
        for c in columns:
            X.append(row[c])

        y_predict = p.predict(X, load_model=(index == 0))

        predict_order = Prediction.DOWN
        if y_predict > open:
            predict_order = Prediction.UP

        real_order = Prediction.DOWN
        if close > open:
            real_order = Prediction.UP

        if predict_order != real_order:
            n_error = n_error + 1

    percent = (n_error / count_test) * 100
    logger.info("Error Order percentage: %0.2f%%" % percent)

    return percent


def test_model():
    df = get_all_data()
    # df = df[df['order_book_bids_price'] > 0].reset_index()
    train_scaler(df=df)
    y = df[['close']].values.reshape(-1, 1)
    columns = ['open']
    # train(df[columns].values, y)
    test_get_error_percent(columns)
