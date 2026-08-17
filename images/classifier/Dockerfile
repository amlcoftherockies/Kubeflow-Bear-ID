FROM python:3.10-slim

RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    joblib \
    huggingface_hub \
    skops \
    kubeflow-katib \
    kubernetes \
    pyyaml \
    GitPython

COPY app /app
WORKDIR /app

# No specific entrypoint since this image is used for multiple components
