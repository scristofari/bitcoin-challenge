from .db import get_all_data
from .log import logger
from .predition import Prediction, predict


def test_get_error_percent():
    df = get_all_data()
    df_test = df[df['order_book_bids_price'] > 0].reset_index()

    n_error = 0
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    for index, row in df_test.iterrows():
        open = row['open']
        close = row['close']

        y_predict = predict(open)

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
