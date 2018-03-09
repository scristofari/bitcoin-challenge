import csv
from enum import Enum
import numpy as np
import pandas as pd
from . import gdax_client, sentiment

TEST_SIZE = 0.2


class Order(Enum):
    STAY = 1
    DOWN = 2
    UP = 3


class Core:
    gdax_client = gdax_client.GdaxClient()
    product_id = None

    def __init__(self, product_id='BTC-EUR'):
        self.product_id = product_id

    def generate_spot_data(self):
        state = sentiment.Sentiment()
        state.build()

        currencies = ['BTC-USD', 'BTC-EUR']
        for c in currencies:
            rate = self.gdax_client.last_rate(c)
            with open('%s.csv' % c, newline='', encoding='utf-8', mode='a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    rate + state.from_twitter + [state.from_reddit] + [state.from_gnews])

    def load_data(self):
        import pandas as pd
        return pd.read_csv('%s.csv' % self.product_id,
                           names=['time', 'low', 'high', 'open', 'close', 'volume', 'tw_sentiment', 'tw_followers',
                                  'reddit_sentiment', 'google_sentiment']
                           )

    def predict_order(self, state):
        from keras.models import load_model

        history = pd.read_csv('order_history_%s.csv' % self.product_id,
                              names=['price', 'predict_price', 'predict_order'])
        last_history = history[-1:]
        last_predict_price = float(last_history['predict_price'].values)

        df = self.load_data()
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = Core.prepare_inputs_outputs(df)

        price = self.gdax_client.current_price(self.product_id)
        model = load_model('./model-%s.h5' % self.product_id)

        x_predict = np.array(
            [price, state.from_reddit, state.from_twitter[0], state.from_twitter[1],
             state.from_gnews]).reshape(1, -1)
        x_predict = scaler_x.transform(x_predict)
        x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
        y_predict_r = model.predict(x_predict_reshaped)

        predict_price = float("%.2f" % scaler_y.inverse_transform(y_predict_r))

        predict_order = Order.DOWN
        if last_predict_price < predict_price:
            predict_order = Order.UP
        elif last_predict_price == predict_price:
            predict_order = Order.STAY

        with open('order_history_%s.csv' % self.product_id, newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([price, predict_price, predict_order])

    @staticmethod
    def prepare_inputs_outputs(df):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split

        # delete row or mean
        # df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
        df.dropna(how='any', inplace=True)

        X = df[['open', 'reddit_sentiment', 'tw_sentiment', 'tw_followers', 'google_sentiment']]
        y = df['close'].values.reshape(-1, 1)

        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_scale = scaler_x.fit_transform(X)
        y_scale = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=TEST_SIZE, shuffle=False)

        return X_train, X_test, y_train, y_test, scaler_x, scaler_y

    def train(self):
        from keras import regularizers
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Activation
        from keras.layers import Dropout

        df = self.load_data()

        np.random.seed(42)
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = Core.prepare_inputs_outputs(df)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                       recurrent_regularizer=regularizers.l1()))
        model.add(Dropout(0.2))
        model.add(LSTM(200, return_sequences=True,
                       recurrent_regularizer=regularizers.l1()))
        model.add(Dropout(0.2))
        model.add(LSTM(200, return_sequences=False,
                       recurrent_regularizer=regularizers.l1()))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.summary()

        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])
        history = model.fit(X_train, y_train, batch_size=X_train.shape[0],
                            epochs=100, validation_data=(X_test, y_test), shuffle=False, verbose=False)

        model.save('model-%s.h5' % self.product_id)

        return history

    def train_anomaly(self):
        from sklearn.neighbors import KernelDensity
        from sklearn.externals import joblib
        from sklearn.model_selection import GridSearchCV

        df = self.load_data()

        X = df['volume'].values.reshape(-1, 1)
        params = {'bandwidth': np.logspace(0, df['volume'].max())}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(X)

        joblib.dump(grid.best_estimator_, 'model-anomaly-%s.pkl' % self.product_id)

        return grid.best_estimator_

    def test_order_percent(self):
        from keras.models import load_model
        from sklearn.externals import joblib

        df = self.load_data()
        _, _, _, _, scaler_x, scaler_y = Core.prepare_inputs_outputs(df)
        model = load_model('./model-%s.h5' % self.product_id)
        model_anomaly = joblib.load('./model-anomaly-%s.pkl' % self.product_id)
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 95)))

        buy = False
        cash = 1000
        previous_cash = bitcoin = n_error = 0
        last_real_order = y_predict_last = y_last = None

        count = df['open'].count()
        n_test = int(TEST_SIZE * count)
        df_test = df[-n_test:].reset_index()
        count_test = df_test['open'].count()
        for index, row in df_test.iterrows():

            if y_predict_last is None:
                y_predict_last = y_last = row['open']

            x_predict = np.array([row['open'], row['reddit_sentiment'], row['tw_sentiment'], row['tw_followers'],
                                  row['google_sentiment']]).reshape(1, -1)
            x_predict = scaler_x.transform(x_predict)
            x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
            y_predict_r = model.predict(x_predict_reshaped)
            y_predict_r_rescaled = scaler_y.inverse_transform(y_predict_r)
            y_predict_r_rescaled = float("%.2f" % y_predict_r_rescaled)

            predict_order = Order.DOWN
            if y_predict_last < y_predict_r_rescaled:
                predict_order = Order.UP
            elif y_predict_last == y_predict_r_rescaled:
                predict_order = Order.STAY

            real_order = Order.DOWN
            if y_last < row['open']:
                real_order = Order.UP
            elif y_last == row['open']:
                real_order = Order.STAY

            y_predict_last = y_predict_r_rescaled

            if real_order != predict_order:
                n_error = n_error + 1

            if last_real_order is None:
                last_real_order = real_order
                continue

            if predict_order == Order.UP and last_real_order == Order.UP:
                if cash > 0 and buy is False:
                    previous_cash = cash
                    buy = True
                    bitcoin = cash / row['open']
                    cash = 0

            elif predict_order == Order.DOWN and last_real_order == Order.DOWN:
                if cash == 0 and buy is True:
                    anomaly = np.exp(model_anomaly.score(row['volume']))
                    if previous_cash < bitcoin * row['open'] and anomaly > anomaly_limit:
                        buy = False
                        cash = bitcoin * row['open']
                        bitcoin = 0

                    if anomaly < anomaly_limit:
                        buy = False
                        cash = bitcoin * row['open']
                        bitcoin = 0

            y_last = row['open']
            last_real_order = real_order

        percent = (n_error / count_test) * 100
        print("Error Order percentage: %0.2f%%" % percent)

        if cash == 0:
            cash = (bitcoin * y_last)

        print("With prediction %.2f euros" % cash)

        bitcoin_first = 1000 / df[0:1]['open'].values
        cash_last = bitcoin_first * float(df[-1:]['open'].values)
        print("Without prediction %.2f euros" % cash_last)
