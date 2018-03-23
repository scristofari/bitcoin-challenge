from .db import get_all_data
from .log import logger
from .predition import Prediction, predict
from .train import train, train_scaler


def test_get_error_percent(columns):
    df = get_all_data()
    df = df[df['order_book_bids_price'] > 0].reset_index()

    n_error = 0
    count = df['open'].count()
    n_tests = int(0.3 * count)

    df_test = df[-n_tests:]
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    for index, row in df_test.iterrows():
        open = row['open']
        close = row['close']

        X = []
        for c in columns:
            X.append(row[c])

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

    return percent


def test_model():
    df = get_all_data()
    df = df[df['order_book_bids_price'] > 0].reset_index()

    train_scaler(df=df)

    y = df[['close']].values.reshape(-1, 1)

    columns = [
        #['open'],
        #['open', 'tw_sentiment'],
        #['open', 'tw_sentiment', 'tw_followers'],
        #['open', 'tw_sentiment', 'tw_followers', 'reddit_sentiment'],
        #['open', 'tw_sentiment', 'tw_followers', 'reddit_sentiment', 'google_sentiment'],
        ['open', 'tw_sentiment', 'tw_followers', 'reddit_sentiment', 'google_sentiment', 'order_book_bids_price', 'order_book_bids_price'],
    ]

    percents = Xs = []
    for c in columns:
        Xs.append(df[c].values)

    for (i, X) in enumerate(Xs):
        train(X, y)
        percent = test_get_error_percent(columns[i])
        percents.append(percent)

    print(percents)