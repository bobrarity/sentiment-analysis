FROM python:3.9-slim

WORKDIR /usr/src/app

COPY ./src/inference/ .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/src/app/data/preprocessed_test
RUN mkdir -p /usr/src/app/outputs/predictions

CMD ["python", "./run_inference.py"]
