FROM python:3.10-slim

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mlflow \
    python-multipart

COPY app /app
WORKDIR /app
