FROM python:3-alpine

RUN apk --no-cache update && apk add bash curl gcc g++ freetype-dev libpng-dev

WORKDIR /src
COPY . /src

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip install -r requirements.txt
RUN python -m textblob.download_corpora

CMD python main.py spot