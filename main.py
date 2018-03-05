import cmd
import sys
from bitcoin import engine


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        engine.generate_spot_data()

if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
