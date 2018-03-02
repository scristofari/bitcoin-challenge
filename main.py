import cmd
import sys
import csv
from bitcoin import twitter, rates, db, reddit


class BitCoinChallenge(cmd.Cmd):
    def do_spot(self, arg):
        reddit_sentiment = reddit.get_sentiment()
        rate = rates.last_rate()
        twitter_sentiment = twitter.get_sentiment()
        # rate / twitter / reddit.
        with open('data.csv', newline='', encoding='utf-8', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rate + twitter_sentiment + reddit_sentiment)


if __name__ == '__main__':
    BitCoinChallenge().onecmd(' '.join(sys.argv[1:]))
