import csv
import pandas as pd
import numpy as np
import gdax
from .predition import Prediction
from .gdax_client import GdaxClient


class Order:

    @staticmethod
    def action(df, order_prediction):
        from sklearn.externals import joblib

        last_history = df[-1:]
        last_volume = float(last_history['volume'].values)
        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 75)))
        last_volume_anomaly = np.exp(model_anomaly.score(last_volume))

        euros, bitcoins = GdaxClient().get_accounts_balance()
        order_book = gdax.PublicClient().get_product_order_book(product_id='BTC-EUR', level=1)

        print('[Balance] euros: %f bitcoins %f' % (euros, bitcoins))
        if order_prediction == Prediction.UP and euros > 10.0:
            price = float(order_book['asks'][0][0])
            size = float(order_book['asks'][0][1])
            size_buy = float(euros / price)
            print('size to buy: %f' % size_buy)
            if size_buy < size:
                print('buy at %f with %f size' % (price, size))
                # GdaxClient.buy(price=euros, size=size_buy)
                with open('order_buy_history_BTC-EUR.csv', newline='', encoding='utf-8', mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([euros])
                return price

        elif order_prediction == Prediction.DOWN:
            try:
                buy_history = pd.read_csv('order_buy_history_BTC-EUR.csv', names=['price'])
            except FileNotFoundError:
                return 0

            last_euros = float(buy_history[-1:]['price'].values)

            price = float(order_book['bids'][0][0])
            size = float(order_book['bids'][0][1])
            size_sell = float(euros / price)
            print('sell at %f with %f size' % (price, size))
            if size_sell < size and last_euros < size_sell * price and last_volume_anomaly > anomaly_limit:
                print('sell at %f with %f size' % (price, size))
                # GdaxClient.sell(price=euros, size=size_sell)

            elif last_volume_anomaly < anomaly_limit:
                print('[ANOMALY] sell at %f with %f size' % (price, size))
                # GdaxClient.sell(price=euros, size=size_sell)

        else:
            print('Do nothing')

        return 0
