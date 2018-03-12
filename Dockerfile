FROM python:3

RUN apt-get update -y && apt-get upgrade -y

WORKDIR /src
COPY . /src

RUN pip install -r requirements.txt
RUN python -m textblob.download_corpora

RUN pip install gdax

CMD python main.py spot