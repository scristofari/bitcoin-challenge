import gdax
import pandas as pd


def last_rate(granularity=60):
    """
    Return the last rate

    :param granularity: in seconds
    :return:
    """
    # from datetime import datetime
    public_client = gdax.PublicClient()
    rates = public_client.get_product_historic_rates(product_id='BTC-USD', granularity=granularity)
    rates.reverse()

    df = pd.DataFrame(rates)
    df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']

    return df.values[0,:]
