import numpy as np
from .predition import Prediction
from .gdax_client import GdaxClient


class Order:

    def action(self, df, order_prediction):
        from sklearn.externals import joblib

        last_history = df[-1:]
        last_volume = float(last_history['volume'].values)

        gc = GdaxClient()
        ticker = gc.current_ticker('BTC-EUR')
        euros, bitcoins = GdaxClient().get_accounts_balance()

        model_anomaly = joblib.load('./model-anomaly-BTC-EUR.pkl')
        anomaly_limit = np.exp(model_anomaly.score(np.percentile(df['volume'].values, 75)))

        if order_prediction == Prediction.UP and euros > 10:
            # BUY
            pass
        elif order_prediction == Prediction.DOWN:
            # SELL
            anomaly = np.exp(model_anomaly.score(last_volume))
            if 0 < bitcoins * price and anomaly > anomaly_limit:
                pass
