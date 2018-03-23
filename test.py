from bitcoin.db import get_all_data
from bitcoin.log import logger
from bitcoin.predition import Prediction
import numpy as np


def test_get_error_percent():
    from keras.models import load_model
    from sklearn.externals import joblib

    df = get_all_data()
    df_test = df[df['order_book_bids_price'] > 0].reset_index()

    scaler_x = joblib.load('model-scaler-x-BTC-EUR.pkl')
    scaler_y = joblib.load('model-scaler-y-BTC-EUR.pkl')
    model = load_model('./model-BTC-EUR.h5')

    n_error = 0
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    for index, row in df_test.iterrows():
        open = row['open']
        close = row['close']

        x_predict = np.array([open]).reshape(1, -1)
        x_predict = scaler_x.transform(x_predict)
        x_predict = np.reshape(x_predict, (1, 1, x_predict.shape[1]))
        y_predict = model.predict(x_predict)
        y_predict = scaler_y.inverse_transform(y_predict)
        y_predict = float("%.2f" % y_predict)

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

test_get_error_percent()