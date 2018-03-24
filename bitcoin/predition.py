import numpy as np


class Prediction:
    STAY = 1
    DOWN = 2
    UP = 3

    scaler_x = scaler_y = model = None

    def predict(self, X, regul=0.0, load_model=True):
        """

        :param X:
        :param regul:
        :param load_model:
        :return:
        """
        if load_model is True:
            from keras.models import load_model
            from sklearn.externals import joblib

            self.scaler_x = joblib.load('model-scaler-x-BTC-EUR.pkl')
            self.scaler_y = joblib.load('model-scaler-y-BTC-EUR.pkl')
            self.model = load_model('./model-BTC-EUR.h5')

        x_predict = np.array(X).reshape(1, -1)
        x_predict = self.scaler_x.transform(x_predict)
        x_predict = np.reshape(x_predict, (x_predict.shape[0], 1, x_predict.shape[1]))
        y_predict = self.model.predict(x_predict)
        y_predict = self.scaler_y.inverse_transform(y_predict)
        y_predict = float("%.2f" % y_predict)

        return y_predict + regul
