FROM python:3.10-slim

RUN pip install --no-cache-dir requests

COPY app /app
WORKDIR /app

ENTRYPOINT ["python", "download_dataset.py"]
