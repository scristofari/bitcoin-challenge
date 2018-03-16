import time
from sklearn.externals import joblib
import numpy as np
from .predition import Prediction
from .gdax_client import GdaxClient
from .log import logger
from .db import get_last_price_and_volume, get_n2_price, insert_next_buy, get_last_buy_price


class Order:
    env = None

    def __init__(self, env='test'):
        self.gdax_client = GdaxClient()
        self.env = env

    def action_limit(self, rate, order_prediction):

        last_price, last_volume = get_last_price_and_volume()
        last_price_n2 = get_n2_price()
        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(rate[5], 75)))
        last_volume_anomaly = np.exp(model_anomaly.score(last_volume))

        if order_prediction == Prediction.UP or order_prediction == Prediction.DOWN:
            self.gdax_client.cancel_all(product_id='BTC-EUR')

        euros, bitcoins = GdaxClient().get_accounts_balance()

        order_book = self.gdax_client.get_product_order_book(product_id='BTC-EUR', level=1)
        logger.info('order book => %s' % order_book)

        if last_volume_anomaly < anomaly_limit and last_price < last_price_n2 and bitcoins > 0:
            price = float(order_book['bids'][0][0])
            logger.info('[ANOMALY] sell at %.2f with %.2f size' % (price, bitcoins))

            if self.env == 'prod':
                self.gdax_client.sell(product_id='BTC-EUR', type='limit', price=price, size=bitcoins)

        elif order_prediction == Prediction.UP and euros > 10:
            price = float(order_book['asks'][0][0])
            size = float(order_book['asks'][0][1])
            size_buy = float(euros / price)
            if size_buy < size:
                logger.info('buy at %.2f with %.2f euros' % (price, euros))

                if self.env == 'prod':
                    self.gdax_client.buy(product_id='BTC-EUR', type='limit', price=price, size=size_buy)
                    insert_next_buy(time.time(), price)

        elif order_prediction == Prediction.DOWN and bitcoins > 0:
            last_price = get_last_buy_price()
            price = float(order_book['bids'][0][0])
            size = float(order_book['bids'][0][1])
            if last_price < price and bitcoins < size:
                logger.info('sell at %.2f with %.2f size' % (price, bitcoins))

                if self.env == 'prod':
                    self.gdax_client.sell(product_id='BTC-EUR', type='limit', price=price, size=bitcoins)

        else:
            logger.info('Do nothing')

        return 0
