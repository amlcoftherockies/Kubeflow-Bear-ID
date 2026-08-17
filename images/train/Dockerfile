FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir torchvision mlflow

COPY app /app
WORKDIR /app
