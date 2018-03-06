import cmd
import sys
from bitcoin import engine


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        engine.generate_spot_data()

    def do_test(self, arg):
        data = engine.load_data('BTC-EUR')
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = engine.prepare(data)
        model, history = engine.train(X_train, X_test, y_train, y_test)
        engine.test_order_percent(data, model, scaler_x, scaler_y)

if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
