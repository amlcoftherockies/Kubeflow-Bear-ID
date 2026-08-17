FROM python:3.10-slim

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    scikit-learn \
    mlflow

COPY app /app
WORKDIR /app
