FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    libx11-dev \
    libjpeg-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and compile bearid-dlib
RUN git clone https://github.com/hypraptive/bearid-dlib.git /opt/bearid-dlib
WORKDIR /opt/bearid-dlib
RUN mkdir build && cd build && cmake .. && cmake --build . --config Release

# Install PyTorch CPU and Torchvision
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app /app
WORKDIR /app
