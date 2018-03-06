FROM python:3-slim-stretch

RUN apt-get update -y && apt-get upgrade -y

WORKDIR /src
COPY . /src

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN pip install -r requirements.txt
RUN python -m textblob.download_corpora

CMD python main.py spot