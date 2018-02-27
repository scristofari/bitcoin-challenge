import cmd
import sys
from bitcoin import twitter, rates, db


class BitCoinChallenge(cmd.Cmd):
    def do_init(self, arg):
        db.create_db()

    def do_spot(self, arg):
        rate = rates.last_rate()
        twitter_average = twitter.twitter_sentiment()


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
