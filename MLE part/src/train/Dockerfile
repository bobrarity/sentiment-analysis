FROM python:3.9-slim

WORKDIR /usr/src/app

COPY ./src/train/ .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4

RUN mkdir -p /usr/src/app/data/raw
RUN mkdir -p /usr/src/app/data/preprocessed_test

RUN mkdir -p /usr/src/app/outputs/models
RUN mkdir -p /usr/src/app/outputs/predictions

COPY data/raw/ /usr/src/app/data/raw/

CMD ["python", "./train.py"]
