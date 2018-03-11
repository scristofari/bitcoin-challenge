from gdax import gdax
import time

gdax_key = '4823e9f74b40090a2995b96cd7a22cb6'
gdax_secret = '95nGkm6XklHwCfGgJ+krvnwBTXk1NO02QZWbjJ4Aasn6EKGaWMYTkyjAIbhrvhiWJ/ZJY/MsCsWRI/wyZV5r7Q=='
gdax_passphrase = '34hpnoe3zhp'


class GdaxClient:
    @staticmethod
    def last_rate(product_id, granularity=60):
        public_client = gdax.PublicClient()
        rates = public_client.get_product_historic_rates(product_id=product_id, granularity=granularity)
        return rates[0]

    @staticmethod
    def current_ticker(product_id):
        public_client = gdax.PublicClient()
        data = public_client.get_product_ticker(product_id=product_id)
        return data

    def get_accounts_balance(self):
        gdax_client = gdax.AuthenticatedClient(gdax_key, gdax_secret, gdax_passphrase)
        accounts = gdax_client.get_accounts()
        euros = bitcoins = None
        for a in accounts:
            # print("Currency %s => Balance %f => Available => %f" % (a['currency'], float(a['balance']), float(a['available'])))
            if a['currency'] == 'BTC':
                bitcoins = a['available']
            if a['currency'] == 'EUR':
                euros = a['available']

        return euros, bitcoins

    @staticmethod
    def get_order_book():
        order_book = gdax.OrderBook(product_id='BTC-USD')
        order_book.start()
        time.sleep(10)
        order_book.close()
