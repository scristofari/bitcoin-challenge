import cmd
import sys
import bitcoin


class BitCoinChallenge(cmd.Cmd):
    def do_history(self, arg):
        price = bitcoin.get_current_spot()
        bitcoin.insert_amount(price)

    def do_create_dataset(self, arg):
        bitcoin.create_dataset()

    def do_predict(self, arg):
        prediction = bitcoin.predict()

    def do_train(self, arg):
        bitcoin.train()

    def do_history2(self, arg):
        bitcoin.history()

if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
