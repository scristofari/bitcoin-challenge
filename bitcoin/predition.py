from enum import Enum


class Prediction(Enum):
    STAY = 1
    DOWN = 2
    UP = 3


def predict(x):
    """
    Predict the new output.

    :param x: A matrix
    :return: the predicted float
    """
    import numpy as np
    from keras.models import load_model
    from sklearn.externals import joblib

    scaler_x = joblib.load('model-scaler-x-BTC-EUR.pkl')
    scaler_y = joblib.load('model-scaler-y-BTC-EUR.pkl')
    model = load_model('./model-BTC-EUR.h5')

    x_predict = np.array(x).reshape(1, -1)
    x_predict = scaler_x.transform(x_predict)
    x_predict = np.reshape(x_predict, (1, 1, x_predict.shape[1]))
    y_predict = model.predict(x_predict)
    y_predict = scaler_y.inverse_transform(y_predict)
    y_predict = float("%.2f" % y_predict)

    return y_predict
