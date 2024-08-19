FROM python:3.9

COPY mlflow-policy.json mlflow-policy.json

RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/local/bin/mc

RUN mkdir -p /mlflow/
RUN chmod -R 777 /mlflow/
RUN mkdir -p /data/mlflow
RUN chmod -R 777 /data/mlflow

RUN pip install mlflow boto3

EXPOSE 5000
