build:
	docker build -t bitcoin-challenge .

run:
	docker run --rm  -v $(PWD):/src bitcoin-challenge

datacp:
	scp root@159.65.82.141:/root/python/bitcoin-challenge/*.csv /Users/sylvaincristofari/python/bitcoin-challenge

test:
	docker run --rm  -v $(PWD):/src bitcoin-challenge:latest python main.py test

train:
	docker run --rm  -v $(PWD):/src bitcoin-challenge:latest python main.py train

ticker:
	docker run --rm  -v $(PWD):/src bitcoin-challenge:latest python main.py tick

account:
	docker run --rm  -v $(PWD):/src bitcoin-challenge:latest python main.py account

order_book:
	docker run --rm  -v $(PWD):/src bitcoin-challenge:latest python main.py order_book

bash:
	docker run -it --rm  -v $(PWD):/src bitcoin-challenge:latest bash

update_all:
	pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U