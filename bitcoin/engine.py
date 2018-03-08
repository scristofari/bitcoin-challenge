from enum import Enum


class Order(Enum):
    STAY = 1
    DOWN = 2
    UP = 3


def generate_spot_data():
    import csv
    from . import twitter, rates, reddit, gnews

    reddit_sentiment = reddit.get_sentiment()
    twitter_sentiment = twitter.get_sentiment()
    gnews_sentiment = gnews.get_sentiment()
    currencies = ['BTC-USD', 'BTC-EUR']
    for c in currencies:
        rate = rates.last_rate(c)
        # rate / twitter / reddit / gnews
        with open('%s.csv' % c, newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rate + twitter_sentiment + [reddit_sentiment] + [gnews_sentiment])


def load_data(product_id):
    import pandas as pd
    df = pd.read_csv('%s.csv' % product_id,
                     names=['time', 'low', 'high', 'open', 'close', 'volume', 'tw_sentiment', 'tw_followers',
                            'reddit_sentiment', 'google_sentiment']
                     )
    return df


def prepare(df):
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

    X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


def train(product_id, X_train, X_test, y_train, y_test):
    import numpy as np
    from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Activation
    from keras.layers import Dropout

    np.random.seed(42)

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

    model.save('model-%s.h5' % product_id)

    return model, history


def test_order_percent(df, model, scalerX, scalerY):
    import numpy as np

    n_error = 0
    y_predict_last = y_last = None

    count = df['open'].count()
    n_test = int(0.2 * count)
    df_test = df[-n_test:].reset_index()
    count_test = df_test['open'].count()
    for index, row in df_test.iterrows():
        if y_predict_last is None:
            y_predict_last = y_last = row['open']

        x_predict = np.array([row['open'], row['reddit_sentiment'], row['tw_sentiment'], row['tw_followers'],
                              row['google_sentiment']]).reshape(1, -1)
        x_predict = scalerX.transform(x_predict)
        x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
        y_predict_r = model.predict(x_predict_reshaped)
        y_predict_r_rescaled = scalerY.inverse_transform(y_predict_r)
        y_predict_r_rescaled = float("%.2f" % y_predict_r_rescaled)
        predict_order = real_order = Order.DOWN
        if y_predict_last < y_predict_r_rescaled:
            predict_order = Order.UP
        elif y_predict_last == y_predict_r_rescaled:
            predict_order = Order.STAY

        if y_last < row['open']:
            real_order = Order.UP
        elif y_last == row['open']:
            real_order = Order.STAY

        y_predict_last = y_predict_r_rescaled
        y_last = row['open']

        if real_order != predict_order:
            n_error = n_error + 1
            # if real_order != Order.STAY:
            print('%d / %d ---> predicted %s - real %s' % (index, count_test, predict_order, real_order))

    percent = (n_error / count_test) * 100
    print("Error Order percentage: %0.2f%%" % percent)


def predict_order(product_id):
    import csv
    import numpy as np
    from keras.models import load_model
    from . import twitter, reddit, gnews, rates

    df = load_data(product_id)
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare(df)

    last_predict_price = buy_price = 0

    price = rates.current_price(product_id)
    reddit_sentiment = reddit.get_sentiment()
    twitter_sentiment = twitter.get_sentiment()
    gnews_sentiment = gnews.get_sentiment()

    model = load_model('./model-%s.h5' % product_id)

    x_predict = np.array(
        [price, reddit_sentiment, twitter_sentiment[0], twitter_sentiment[1], gnews_sentiment]).reshape(1, -1)
    x_predict = scaler_x.transform(x_predict)
    x_predict_reshaped = np.reshape(x_predict, (1, 1, 5))
    y_predict_r = model.predict(x_predict_reshaped)

    predict_price = float("%.2f" % scaler_y.inverse_transform(y_predict_r))

    predict_order = Order.DOWN
    if price < predict_price:
        predict_order = Order.UP
    elif price == predict_price:
        predict_order = Order.STAY

    with open('order_history_%s.csv' % product_id, newline='', encoding='utf-8', mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([product_id, price, buy_price, last_predict_price, predict_price, predict_order])
