from enum import Enum
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib

scaler_x = joblib.load('model-scaler-x-BTC-EUR.pkl')
scaler_y = joblib.load('model-scaler-y-BTC-EUR.pkl')
model = load_model('./model-BTC-EUR.h5')

class Prediction(Enum):
    STAY = 1
    DOWN = 2
    UP = 3


def predict(X):
    """
    Predict the new output.

    :param x: A matrix
    :return: the predicted float
    """
    x_predict = np.array(X).reshape(1, -1)
    x_predict = scaler_x.transform(x_predict)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], 1, x_predict.shape[1]))
    y_predict = model.predict(x_predict)
    y_predict = scaler_y.inverse_transform(y_predict)
    y_predict = float("%.2f" % y_predict)

    return y_predict
