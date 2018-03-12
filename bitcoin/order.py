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

        if order_prediction == Prediction.UP and euros > 10.0:
            price = order_book['asks']['price']
            size = order_book['asks']['size']
            size_buy = float(euros / price)
            if size_buy < size:
                GdaxClient.buy(price=euros, size=size_buy)
                with open('order_buy_history_BTC-EUR.csv', newline='', encoding='utf-8', mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([price])
                return price

        elif order_prediction == Prediction.DOWN:
            buy_history = pd.read_csv('order_buy_history_BTC-EUR.csv', names=['price'])
            last_buy = float(buy_history[-1:]['price'].values)

            price = order_book['bids']['price']
            size = order_book['bids']['size']
            size_sell = float(euros / price)
            if size_sell > size:
                size_sell = size

            if last_buy < bitcoins * price and last_volume_anomaly > anomaly_limit:
                GdaxClient.sell(price=euros, size=size_sell)

            if last_volume_anomaly < anomaly_limit:
                GdaxClient.sell(price=euros, size=size_sell)

        return 0
