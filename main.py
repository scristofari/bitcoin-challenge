import cmd
import sys
from bitcoin import core


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        c = core.Core()
        c.generate_spot_data()

    def do_train(self, arg):
        c = core.Core()
        c.train()
        c.train_anomaly()

    def do_test(self, arg):
        c = core.Core()
        c.test_order_percent()


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
