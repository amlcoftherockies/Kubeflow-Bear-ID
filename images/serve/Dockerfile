FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dlib requires cmake and build-essential
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    dlib \
    opencv-python-headless \
    joblib \
    skops \
    python-multipart \
    pillow

# Download frozen models natively for serve so they are baked in
RUN mkdir -p /opt/bearid-models
WORKDIR /opt/bearid-models
RUN wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_detector_mmod.dat \
    && wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_landmark_68_model.dat \
    && wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_embed_network.dat

COPY app /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
