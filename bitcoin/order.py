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
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(last_volume, 75)))
        last_volume_anomaly = np.exp(model_anomaly.score(last_volume))

        euros, bitcoins = GdaxClient().get_accounts_balance()
        order_book = gdax.PublicClient().get_product_order_book(product_id='BTC-EUR', level=1)

        euros = 100
        order_prediction = Prediction.DOWN
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
                    writer.writerow([price])
                return price

        elif order_prediction == Prediction.DOWN:
            try:
                buy_history = pd.read_csv('order_buy_history_BTC-EUR.csv', names=['price'])
            except FileNotFoundError:
                return 0

            last_buy = float(buy_history[-1:]['price'].values)

            price = float(order_book['bids'][0][0])
            size = float(order_book['bids'][0][1])
            size_sell = float(euros / price)
            print('size to sell: %f' % size_sell)
            if size_sell < size and last_buy < bitcoins * price and last_volume_anomaly > anomaly_limit:
                print('sell at %f with %f size' % (price, size))
                # GdaxClient.sell(price=euros, size=size_sell)

            if last_volume_anomaly < anomaly_limit:
                print('[ANOMALY] sell at %f with %f size' % (price, size))
                # GdaxClient.sell(price=euros, size=size_sell)

        else:
            print('Do nothing')

        return 0
