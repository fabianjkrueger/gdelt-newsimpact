FROM python:3.12-slim

# install required packages
RUN pip install --no-cache-dir \
    mlflow==3.2.0 \
    psycopg2-binary \
    setuptools

EXPOSE 5000

# mlflow server command
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}