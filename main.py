import cmd
import sys
from bitcoin import twitter, rates


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):

        rate = rates.last_rate()
        twitter_average = twitter.twitter_sentiment()


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
