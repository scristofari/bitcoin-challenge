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


        elif order_prediction == Prediction.DOWN:
            price = order_book['bids']['price']
            size = order_book['bids']['size']
            size_sell = float(euros / price)
            if size_sell > size:
                size_sell = size

            if 0 < bitcoins * price and last_volume_anomaly > anomaly_limit:
                GdaxClient.sell(price=euros, size=size_sell)

            if last_volume_anomaly < anomaly_limit:
                GdaxClient.sell(price=euros, size=size_sell)