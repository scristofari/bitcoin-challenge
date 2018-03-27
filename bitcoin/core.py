from .gdax_client import GdaxClient
from .sentiment import Sentiment
from bitcoin.log import logger
import bitcoin.db as db

REGUL = 0.569999999999709


class Core:
    gdax_client = GdaxClient()
    product_id = None
    env = None

    def __init__(self, product_id='BTC-EUR', env='test'):
        logger.info('ENV => %s' % env)
        self.product_id = product_id
        self.env = env

    def generate_spot_data(self):
        state = Sentiment().build()
        rate = self.gdax_client.last_rate(self.product_id)
        order_book = self.gdax_client.get_product_order_book(product_id=self.product_id, level=1)
        db.insert_data(rate + state.from_twitter + [state.from_reddit] + [state.from_gnews] + [0.0] +
                       order_book['bids'][0] + order_book['asks'][0])
