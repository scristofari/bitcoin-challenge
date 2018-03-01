build:
	docker build -t bitcoin-challenge .

run:
	docker run -it --rm  -v "$PWD":/src bitcoin-challenge bash