from gdax import AuthenticatedClient
from .log import logger

gdax_key = '4823e9f74b40090a2995b96cd7a22cb6'
gdax_secret = '95nGkm6XklHwCfGgJ+krvnwBTXk1NO02QZWbjJ4Aasn6EKGaWMYTkyjAIbhrvhiWJ/ZJY/MsCsWRI/wyZV5r7Q=='
gdax_passphrase = '34hpnoe3zhp'


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
                return fill['price']
