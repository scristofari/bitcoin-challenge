from .db import get_all_data
from .log import logger
from .predition import Prediction, predict
from .train import train, train_scaler


def test_get_error_percent(columns):
    df = get_all_data()
    df_test = df[df['order_book_bids_price'] > 0].reset_index()

    n_error = 0
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    for index, row in df_test.iterrows():
        open = row['open']
        close = row['close']

        X = []
        for c in columns:
            X.append(row[c])

        logger.info("Predict %d / %d" % (index, count_test))
        y_predict = predict(X)

        predict_order = Prediction.DOWN
        if y_predict > close:
            predict_order = Prediction.UP
        elif y_predict == open:
            predict_order = Prediction.STAY

        real_order = Prediction.DOWN
        if close > open:
            real_order = Prediction.UP
        elif close == open:
            real_order = Prediction.STAY

        if predict_order != real_order:
            n_error = n_error + 1

    percent = (n_error / count_test) * 100
    logger.info("Error Order percentage: %0.2f%%" % percent)


def test_model():
    df = get_all_data()
    train_scaler(df=df)

    y = df[['close']].values.reshape(-1, 1)

    columns = [
        ['open'],
        ['open', 'tw_sentiment']
    ]

    Xs = []
    for c in columns:
        Xs.append(df[c])
    for k, X in Xs:
        #train(X, y)
        test_get_error_percent(columns[k])
