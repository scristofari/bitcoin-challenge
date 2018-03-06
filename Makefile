build:
	docker build -t bitcoin-challenge .

run:
	docker run --rm  -v $(PWD):/src bitcoin-challenge

datacp:
	scp root@159.65.82.141:/root/python/bitcoin-challenge/*.csv /Users/sylvaincristofari/python/bitcoin-challenge

test:
	docker run --rm  -v $(PWD):/src bitcoin-challenge python main.py test