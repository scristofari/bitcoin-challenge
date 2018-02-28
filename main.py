import cmd
import sys
import csv
from bitcoin import twitter, rates, db


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        rate = rates.last_rate()
        sentiment = twitter.twitter_sentiment()
        # rate / sentiment.
        with open('data.csv', newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rate + sentiment)


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
