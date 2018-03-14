import csv
import pandas as pd
import numpy as np
from .predition import Prediction
from .gdax_client import GdaxClient


class Order:
    def __init__(self):
        self.gdax_client = GdaxClient()

    def action(self, df, order_prediction):
        from sklearn.externals import joblib

        last_volume = float(df[-1:]['volume'])
        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 75)))
        last_volume_anomaly = np.exp(model_anomaly.score(last_volume))

        euros, bitcoins = GdaxClient().get_accounts_balance()

        print('[Balance] euros: %f bitcoins %f' % (euros, bitcoins))

        data = GdaxClient().get_product_ticker('BTC-EUR')
        price = float(data['price'])
        if order_prediction == Prediction.UP and euros > 10.0:

            # GdaxClient().buy(product_id='BTC-EUR', type='market', funds=euros)

            with open('order_buy_history_BTC-EUR.csv', newline='', encoding='utf-8', mode='a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([price])

        elif order_prediction == Prediction.DOWN:
            try:
                buy_history = pd.read_csv('order_buy_history_BTC-EUR.csv', names=['price'])
            except FileNotFoundError:
                return 0

            last_price = float(buy_history[-1:]['price'])
            data = GdaxClient().get_product_ticker('BTC-EUR')
            price = float(data['price'])

            if last_price < price:
                print('sell at %f with %f size' % (price, bitcoins))
                # GdaxClient.sell(product_id='BTC-EUR', type='market', size=bitcoins)

            elif last_volume_anomaly < anomaly_limit:
                print('[ANOMALY] sell at %f with %f size' % (price, bitcoins))
                # GdaxClient.sell(product_id='BTC-EUR', type='market', size=bitcoins)

        else:
            print('Do nothing')

        return 0

    def action_limit(self, df, order_prediction):
        from sklearn.externals import joblib

        last_volume = float(df[-1:]['volume'])
        last_price = float(df[-1:]['price'])
        last_price_n2 = float(df[-2:-1]['volume'])
        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 75)))
        last_volume_anomaly = np.exp(model_anomaly.score(last_volume))

        if order_prediction == Prediction.UP or order_prediction == Prediction.DOWN:
            self.gdax_client.cancel_all(product_id='BTC-EUR')

        euros, bitcoins = GdaxClient().get_accounts_balance()
        print('[Balance] euros: %f bitcoins %f' % (euros, bitcoins))
        order_book = self.gdax_client.get_product_order_book(product_id='BTC-EUR', level=1)

        if last_volume_anomaly < anomaly_limit and last_price < last_price_n2:
            price = float(order_book['bids'][0][0])
            print('[ANOMALY] sell at %f with %f size' % (price, bitcoins))
            # GdaxClient.sell(product_id='BTC-EUR', type='limit', price=price, size=bitcoins)

        elif order_prediction == Prediction.UP and euros > 10.0:
            price = float(order_book['asks'][0][0])
            size = float(order_book['asks'][0][1])
            size_buy = float(euros / price)
            if size_buy < size:
                pass
                # self.gdax_client.buy(product_id='BTC-EUR', type='limit', price=price, size=size_buy)

                with open('order_buy_history_BTC-EUR.csv', newline='', encoding='utf-8', mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([price])

        elif order_prediction == Prediction.DOWN:
            try:
                buy_history = pd.read_csv('order_buy_history_BTC-EUR.csv', names=['price'])
            except FileNotFoundError:
                return 0

            last_price = float(buy_history[-1:]['price'])
            price = float(order_book['bids'][0][0])
            size = float(order_book['bids'][0][1])
            if last_price < price and bitcoins < size:
                print('sell at %f with %f size' % (price, bitcoins))
                # GdaxClient.sell(product_id='BTC-EUR', type='limit', price=price, size=bitcoins)

        else:
            print('Do nothing')

        return 0
