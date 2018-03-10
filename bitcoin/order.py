import gdax
import numpy as np
from .predition import Prediction
from .core import Core

gdax_key = '4823e9f74b40090a2995b96cd7a22cb6'
gdax_secret = '95nGkm6XklHwCfGgJ+krvnwBTXk1NO02QZWbjJ4Aasn6EKGaWMYTkyjAIbhrvhiWJ/ZJY/MsCsWRI/wyZV5r7Q=='
gdax_passphrase = '34hpnoe3zhp'


# Use the sandbox API (requires a different set of API access credentials)
# auth_client = gdax.AuthenticatedClient(key, b64secret, passphrase, api_url="https://api-public.sandbox.gdax.com")

class Order:
    gdax_client = None

    def __init__(self):
        self.gdax_client = gdax.AuthenticatedClient(gdax_key, gdax_secret, gdax_passphrase)
        #gdax.AuthenticatedClient(gdax_key, gdax_secret, gdax_passphrase, api_url="https://api-public.sandbox.gdax.com")

    def action(self, df, price, order_prediction):
        from sklearn.externals import joblib

        last_history = df[-1:]
        last_volume = float(last_history['volume'].values)
        euros, bitcoins = self.get_accounts_balance()
        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 75)))


        if order_prediction == Prediction.UP and euros > 0:
            pass
        elif order_prediction == Prediction.DOWN:
            anomaly = np.exp(model_anomaly.score(last_volume))
            if 0 < bitcoins * price and anomaly > anomaly_limit:
            pass

    def get_accounts_balance(self):
        accounts = self.gdax_client.get_accounts()
        euros = bitcoins = None
        for a in accounts:
            # print("Currency %s => Balance %f => Available => %f" % (a['currency'], float(a['balance']), float(a['available'])))
            if a['currency'] == 'BTC':
                bitcoins = a['available']
            if a['currency'] == 'EUR':
                euros = a['available']

        return euros, bitcoins
