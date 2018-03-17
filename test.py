from bitcoin.sentiment import Sentiment

s = Sentiment()
s.build_from_gnews()

print(s.from_gnews)