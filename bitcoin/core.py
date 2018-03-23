import time
from .gdax_client import GdaxClient
from .sentiment import Sentiment
from bitcoin.log import logger
import bitcoin.db as db


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
        order_book = self.gdax_client.get_product_order_book(product_id=self.product_id, level=1)
        t1 = time.time()
        print("SPOT execution time : %d s" % (t1 - t0))

        predicted_price = 0.0
        t2 = time.time()
        print("PREDICT execution time : %d s" % (t2 - t1))

        db.insert_data(rate + state.from_twitter + [state.from_reddit] + [state.from_gnews] + [predicted_price] +
                       order_book['bids'][0] + order_book['asks'][0])
