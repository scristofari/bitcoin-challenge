from gdax import AuthenticatedClient
from .log import logger

gdax_key = ''
gdax_secret = ''
gdax_passphrase = ''


class GdaxClient(AuthenticatedClient):
    def __init__(self):
        super(GdaxClient, self).__init__(gdax_key, gdax_secret, gdax_passphrase)

    def last_rate(self, product_id, granularity=60):
        rates = self.get_product_historic_rates(product_id=product_id, granularity=granularity)
        logger.info("Get the last rates => %s", rates[0])
        return rates[0]

    def get_accounts_balance(self):
        accounts = self.get_accounts()
        euros = bitcoins = None
        for a in accounts:
            if a['currency'] == 'BTC':
                bitcoins = float(a['available'])
            if a['currency'] == 'EUR':
                euros = float(a['available'])

        logger.info('[Balance] euros: %f bitcoins %f' % (euros, bitcoins))
        return euros, bitcoins

    def get_last_buy_filled(self):
        fills = self.get_fills(product_id='BTC-EUR', limit=10)
        for fill in fills[0]:
            if fill['side'] == 'buy':
                return float(fill['price'])
