import cmd
import sys
from bitcoin import core
from bitcoin.gdax_client import GdaxClient


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        import time
        t0 = time.time()
        if arg == '':
            arg = 'test'
        c = core.Core(env=arg)
        c.generate_spot_data()
        t1 = time.time()
        print("Execution time : %d s" % (t1 - t0))

    def do_train(self, arg):
        c = core.Core()
        c.train()
        c.train_anomaly()
        c.train_scaler()

    def do_test(self, arg):
        c = core.Core()
        c.test_order_percent()

    def do_tick(self, arg):
        print(GdaxClient().get_product_ticker('BTC-EUR'))

    def do_account(self, arg):
        print(GdaxClient().get_accounts_balance())

    def do_order_book(self, arg):
        order_book = GdaxClient().get_product_order_book(product_id='BTC-EUR', level=1)
        print("Asks %s" % order_book['asks'][0][0])


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
