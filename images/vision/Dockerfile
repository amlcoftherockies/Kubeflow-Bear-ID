FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    libx11-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and compile bearid-dlib
RUN git clone https://github.com/hypraptive/bearid-dlib.git /opt/bearid-dlib
WORKDIR /opt/bearid-dlib
RUN mkdir build && cd build && cmake .. && cmake --build . --config Release

# Download frozen models
RUN mkdir -p /opt/bearid-models
WORKDIR /opt/bearid-models
RUN wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_detector_mmod.dat \
    && wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_landmark_68_model.dat \
    && wget -q https://raw.githubusercontent.com/hypraptive/bearid-models/master/bear_embed_network.dat

COPY app /app
RUN chmod +x /app/process_vision.sh

ENTRYPOINT ["/app/process_vision.sh"]
