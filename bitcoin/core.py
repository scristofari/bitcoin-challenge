import csv
import time
import numpy as np
import pandas as pd
from .gdax_client import GdaxClient
from .sentiment import Sentiment
from bitcoin.order import Order
from .predition import Prediction
from datetime import datetime
from bitcoin.log import logger

TEST_SIZE = 0.1
CASH_FIRST = 1000


class Core:
    gdax_client = GdaxClient()
    product_id = None
    env = None

    def __init__(self, product_id='BTC-EUR', env='test'):
        logger.info('ENV => %s' % env)
        self.product_id = product_id
        self.env = env

    def generate_spot_data(self):
        t0 = time.time()
        state = Sentiment().build()
        rate = self.gdax_client.last_rate(self.product_id)

        with open('%s.csv' % self.product_id, newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rate + state.from_twitter + [state.from_reddit] + [state.from_gnews])

        t1 = time.time()
        print("SPOT execution time : %d s" % (t1 - t0))
        self.predict_order(state)
        t2 = time.time()
        print("PREDICT execution time : %d s" % (t2 - t1))

    def load_data(self):
        logger.info('Load data from CSV.')
        import pandas as pd
        return pd.read_csv('%s.csv' % self.product_id,
                           names=['time', 'low', 'high', 'open', 'close', 'volume', 'tw_sentiment', 'tw_followers',
                                  'reddit_sentiment', 'google_sentiment']
                           )

    def predict_order(self, state):
        from keras.models import load_model
        from sklearn.externals import joblib

        df = self.load_data()

        scaler_x = joblib.load('model-scaler-x-%s.pkl' % self.product_id)
        scaler_y = joblib.load('model-scaler-y-%s.pkl' % self.product_id)

        data = self.gdax_client.get_product_ticker(self.product_id)
        price = float(data['price'])

        try:
            history = pd.read_csv('order_history_%s.csv' % self.product_id,
                                  names=['price', 'predict_price', 'predict_order'])

            last_predict_price = float(history[-1:]['predict_price'])
        except FileNotFoundError:
            last_predict_price = price

        model = load_model('./model-%s.h5' % self.product_id)

        x_predict = np.array(
            [price, state.from_reddit, state.from_twitter[0], state.from_twitter[1],
             state.from_gnews]).reshape(1, -1)
        x_predict = scaler_x.transform(x_predict)
        x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
        y_predict_r = model.predict(x_predict_reshaped)

        predict_price = float("%.2f" % scaler_y.inverse_transform(y_predict_r))

        predict_order = Prediction.DOWN
        if last_predict_price < predict_price:
            predict_order = Prediction.UP
        elif last_predict_price == predict_price:
            predict_order = Prediction.STAY

        with open('order_history_%s.csv' % self.product_id, newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([price, predict_price, predict_order])

        Order(env=self.env).action_limit(df, predict_order)

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
        logger.info('Train Model')

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
                            epochs=100, validation_data=(X_test, y_test), shuffle=False, verbose=True)

        model.save('model-%s.h5' % self.product_id)

        return history

    def train_scaler(self):
        logger.info('Train Scaler Model')

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.externals import joblib

        df = self.load_data()

        # delete row
        df.dropna(how='any', inplace=True)

        X = df[['open', 'reddit_sentiment', 'tw_sentiment', 'tw_followers', 'google_sentiment']]
        y = df['close'].values.reshape(-1, 1)

        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_x.fit(X)
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        scaler_y.fit(y)

        joblib.dump(scaler_x, 'model-scaler-x-%s.pkl' % self.product_id)
        joblib.dump(scaler_y, 'model-scaler-y-%s.pkl' % self.product_id)

        return scaler_x, scaler_y

    def train_anomaly(self):

        logger.info('Train Anomaly Model')

        from sklearn.neighbors import KernelDensity
        from sklearn.externals import joblib
        from sklearn.model_selection import GridSearchCV

        df = self.load_data()

        X = df['volume'].values.reshape(-1, 1)
        params = {'bandwidth': np.logspace(0, df['volume'].max())}
        grid = GridSearchCV(KernelDensity(), params, verbose=True)
        grid.fit(X)

        joblib.dump(grid.best_estimator_, 'model-anomaly-%s.pkl' % self.product_id)

        return grid.best_estimator_

    def test_order_percent(self):
        from keras.models import load_model
        from sklearn.externals import joblib

        try:
            df = self.load_data()
        except FileNotFoundError:
            raise NameError('No data')


        scaler_x = joblib.load('model-scaler-x-%s.pkl' % self.product_id)
        scaler_y = joblib.load('model-scaler-y-%s.pkl' % self.product_id)

        model = load_model('./model-%s.h5' % self.product_id)
        model_anomaly = joblib.load('./model-anomaly-%s.pkl' % self.product_id)
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 60)))

        buy = False
        cash = CASH_FIRST
        previous_cash = bitcoin = n_error = n_anomalies = n_api_call = 0
        last_volume = last_real_order = y_predict_last = y_last = None

        count = df['open'].count()
        n_test = int(TEST_SIZE * count)
        df_test = df[-n_test:].reset_index()
        count_test = df_test['open'].count()
        for index, row in df_test.iterrows():

            if y_predict_last is None:
                y_predict_last = y_last = row['open']
            if last_volume is None:
                last_volume = row['volume']

            x_predict = np.array([row['open'], row['reddit_sentiment'], row['tw_sentiment'], row['tw_followers'],
                                  row['google_sentiment']]).reshape(1, -1)
            x_predict = scaler_x.transform(x_predict)
            x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
            y_predict_r = model.predict(x_predict_reshaped)
            y_predict_r_rescaled = scaler_y.inverse_transform(y_predict_r)
            y_predict_r_rescaled = float("%.2f" % y_predict_r_rescaled)

            predict_order = Prediction.DOWN
            if y_predict_last < y_predict_r_rescaled:
                predict_order = Prediction.UP
            elif y_predict_last == y_predict_r_rescaled:
                predict_order = Prediction.STAY

            real_order = Prediction.DOWN
            if y_last < row['open']:
                real_order = Prediction.UP
            elif y_last == row['open']:
                real_order = Prediction.STAY

            y_predict_last = y_predict_r_rescaled

            if real_order != predict_order:
                n_error = n_error + 1

            if last_real_order is None:
                last_real_order = real_order
                continue

            anomaly = np.exp(model_anomaly.score(last_volume))
            if cash == 0 and buy is True and anomaly < anomaly_limit and real_order == Prediction.DOWN:
                n_anomalies = n_anomalies + 1
                buy = False
                cash = bitcoin * row['open']
                bitcoin = 0
                n_api_call = n_api_call + 1
                y_last = row['open']
                last_real_order = real_order
                continue

            elif predict_order == Prediction.UP:  # and last_real_order == Prediction.UP:
                if cash > 0 and buy is False:
                    previous_cash = cash
                    buy = True
                    bitcoin = cash / row['open']
                    cash = 0
                    n_api_call = n_api_call + 1

            elif predict_order == Prediction.DOWN:  # and last_real_order == Prediction.DOWN:
                if cash == 0 and buy is True:
                    if previous_cash < bitcoin * row['open']:
                        buy = False
                        cash = bitcoin * row['open']
                        bitcoin = 0
                        n_api_call = n_api_call + 1

            y_last = row['open']
            last_volume = row['volume']
            last_real_order = real_order

        percent = (n_error / count_test) * 100
        logger.info("Error Order percentage: %0.2f%%" % percent)

        if cash == 0:
            cash = (bitcoin * y_last)

        from_date = datetime.fromtimestamp(df_test[0:1]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
        to_date = datetime.fromtimestamp(df_test[-1:]['time'].values).strftime('%Y-%m-%d %H:%M:%S')
        logger.info("TEST From %s to %s" % (from_date, to_date))

        percent_predict_win_loss = (cash - CASH_FIRST) / CASH_FIRST * 100
        n_days = int(count_test / 1440)
        logger.info("Number of api calls: %.2f / min" % float(n_api_call / count_test))
        logger.info("Number of anomalies: %d" % n_anomalies)
        logger.info("With prediction %.2f euros => %.2f%% => %.2f%% / day" % (
            cash, percent_predict_win_loss, float(percent_predict_win_loss / n_days)))

        bitcoin_first = CASH_FIRST / df_test[0:1]['open'].values
        cash_last = bitcoin_first * float(df_test[-1:]['open'].values)
        percent_win_loss = (cash_last - CASH_FIRST) / CASH_FIRST * 100
        logger.info("Without prediction %.2f euros => %.2f%% => %.2f%% / day" % (
            cash_last, percent_win_loss, float(percent_win_loss / n_days)))
