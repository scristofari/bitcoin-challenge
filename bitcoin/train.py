from .log import logger
from .db import get_all_data
import numpy as np

TEST_SIZE = 0.3


def train(X, y):
    logger.info('Train Model')

    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split

    from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Activation
    from keras.layers import Dropout

    np.random.seed(42)

    scaler_x = joblib.load('model-scaler-x-BTC-EUR.pkl')
    scaler_y = joblib.load('model-scaler-y-BTC-EUR.pkl')
    X_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=TEST_SIZE, shuffle=False)

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

    model.save('model-BTC-EUR.h5')

    return history


def train_scaler(df=None):
    logger.info('Train Scaler Model')

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.externals import joblib

    if df is None:
        df = get_all_data()

    # delete row
    df.dropna(how='any', inplace=True)

    X = df[['open']]
    y = df['close'].values.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(X)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y)

    joblib.dump(scaler_x, 'model-scaler-x-BTC-EUR.pkl')
    joblib.dump(scaler_y, 'model-scaler-y-BTC-EUR.pkl')

    return scaler_x, scaler_y


def train_anomaly(df=None):
    logger.info('Train Anomaly Model')

    from sklearn.neighbors import KernelDensity
    from sklearn.externals import joblib
    from sklearn.model_selection import GridSearchCV

    if df is None:
        df = get_all_data()

    X = df['volume'].values.reshape(-1, 1)
    params = {'bandwidth': np.logspace(0, df['volume'].max())}
    grid = GridSearchCV(KernelDensity(), params, verbose=True, n_jobs=10)
    grid.fit(X)

    joblib.dump(grid.best_estimator_, 'model-anomaly-BTC-EUR.pkl')

    return grid.best_estimator_
