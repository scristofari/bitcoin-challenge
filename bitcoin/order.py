from sklearn.externals import joblib
import numpy as np
from .predition import Prediction
from .gdax_client import GdaxClient
from .log import logger
from .db import get_last_price_and_volume, get_n2_price, insert_next_buy, get_last_buy_price

GAP = 0.01


def floor3(x):
    import math
    return math.floor(x * 1000.0) / 1000.0


def floor2(x):
    import math
    return math.floor(x * 100.0) / 100.0


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
            r = self.gdax_client.cancel_all(product_id='BTC-EUR')
            logger.info('cancel all => %s' % r)

        euros, bitcoins = GdaxClient().get_accounts_balance()

        order_book = self.gdax_client.get_product_order_book(product_id='BTC-EUR', level=1)
        logger.info('prediction %s' % (order_prediction))
        logger.info('order book => %s' % order_book)

        if last_volume_anomaly < anomaly_limit and last_price < last_price_n2 and bitcoins > 0:
            price = float(order_book['bids'][0][0])
            logger.info('[ANOMALY] sell at %.2f with %.2f size' % (price, bitcoins))

            if self.env == 'prod':
                r = self.gdax_client.sell(product_id='BTC-EUR', type='limit', price=floor2(price + GAP), size=bitcoins)
                logger.info('anomaly sell => %s' % r)

        elif order_prediction == Prediction.UP and euros > 10:
            price = float(order_book['asks'][0][0])
            size = float(order_book['asks'][0][1])
            size_buy = float(euros / price)
            if size_buy < size:
                logger.info('buy at %.2f with %.2f euros and size %.2f' % (price, euros, size_buy))

                if self.env == 'prod':
                    r = self.gdax_client.buy(product_id='BTC-EUR', type='limit', price=floor2(price - GAP),
                                             size=floor3(size_buy))
                    logger.info('buy => %s' % r)

        elif order_prediction == Prediction.DOWN and bitcoins > 0:
            price = float(order_book['bids'][0][0])
            size = float(order_book['bids'][0][1])
            last_price = self.gdax_client.get_last_buy_filled()
            logger.info('last buy price => %.2f' % last_price)

            if (last_price + GAP) < price and bitcoins < size:
                logger.info('sell at %.2f with %.2f size' % (price, bitcoins))

                if self.env == 'prod':
                    r = self.gdax_client.sell(product_id='BTC-EUR', type='limit', price=floor2(price + GAP),
                                              size=bitcoins)
                    logger.info('sell => %s' % r)
        else:
            logger.info('Do nothing')

        return order_book
