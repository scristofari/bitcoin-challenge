import cmd
import sys
from bitcoin import engine, core


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        c = core.Core()
        c.generate_spot_data()

    def do_train(self, arg):
        df = engine.load_data('BTC-EUR')
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = engine.prepare(df)
        engine.train('BTC-EUR', X_train, X_test, y_train, y_test)
        engine.train_anomaly('BTC-EUR', df)

    def do_test(self, arg):
        from keras.models import load_model
        data = engine.load_data('BTC-EUR')
        model = load_model('model-BTC-EUR.h5')
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = engine.prepare(data)
        engine.test_order_percent(data, model, scaler_x, scaler_y)


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
