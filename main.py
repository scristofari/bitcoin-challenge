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


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
