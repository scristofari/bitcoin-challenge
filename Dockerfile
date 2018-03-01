FROM python:3-alpine

RUN apk --no-cache update && apk add bash curl gcc g++ freetype-dev libpng-dev

WORKDIR /src
COPY . /src

RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt --upgrade
CMD python main.py spot