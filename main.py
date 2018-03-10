import cmd
import sys
from bitcoin import core
from bitcoin.gdax_client import GdaxClient
from bitcoin.order import Order


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        import time
        t0 = time.time()
        c = core.Core()
        c.generate_spot_data()
        t1 = time.time()
        print("Execution time : %d s" % (t1 - t0))

    def do_train(self, arg):
        c = core.Core()
        c.train()
        c.train_anomaly()

    def do_test(self, arg):
        c = core.Core()
        c.test_order_percent()

    def do_tick(self, arg):
        print(GdaxClient().current_price('BTC-EUR'))

    def do_account(self, arg):
        Order().get_accounts_balance()


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
