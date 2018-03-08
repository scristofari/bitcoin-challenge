import gdax


def last_rate(product_id, granularity=60):
    """

    :param product_id:
    :param granularity:
    :return:
    """
    # from datetime import datetime
    public_client = gdax.PublicClient()
    rates = public_client.get_product_historic_rates(product_id=product_id, granularity=granularity)
    return rates[0]


def current_price(product_id):
    public_client = gdax.PublicClient()
    data = public_client.get_product_ticker(product_id=product_id)

    return float(data['price'])
